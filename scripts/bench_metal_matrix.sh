#!/usr/bin/env bash
# Metal benchmark matrix — one Mac in, one JSON out.
#
#     scripts/bench_metal_matrix.sh                 # the MoE models, the default round
#     scripts/bench_metal_matrix.sh --models dense  # the dense models
#     scripts/bench_metal_matrix.sh --models full   # both, the complete round
#     scripts/bench_metal_matrix.sh --models qwen3.6-27b,gemma-4-31b-it   # any from MODELS_ALL
#     scripts/bench_metal_matrix.sh --models all
#     scripts/bench_metal_matrix.sh --scenarios cold,warm   # no sudo needed
#
# A RUNNER, not a measurement tool: every number comes from a `scr bench`
# subcommand, except BUILD TIME and BINARY SIZE per model. scratchy compiles one
# model into the binary, so size is a per-model fact and the build is the cost
# of doing that work ahead of time; every startup number excludes it.
#
#   build seconds, binary MiB          cargo build, stat
#   startup (launch -> ready): frozen / cold, then the first request on its own
#                                      scr bench startup --exec (cache ladder)
#   warm: send -> first token, median over unique prompts to a resident server
#   peak RSS, major faults             scr bench startup --exec (per child, wait4)
#   TTFT/TPOT/ITL p50+p99, tok/s       scr bench serve (warm, conc 1)
#   concurrency curve                  scr bench serve (input 512, output 128)
#   input x output grid (heat maps)    scr bench serve (conc 8)
#
# Rung definitions and fairness rules: docs/BENCHMARKING.md.
#
# Needs: passwordless `sudo purge` for the frozen rung (a sudoers rule for
# /usr/sbin/purge; see the Metal bench skill), or pass --scenarios cold,warm.
# Weights download on first use unless --offline.
#
# Scaling (--no-scaling skips it; --scale-axes picks from conc,input,output,grid):
# one seed per (model, axis, rung) — unique so the prefix cache cannot serve a
# later cell, shared across engines so all see the same prompts. Concurrency is
# offered (--max-concurrency), not the effective decode batch.
#
# With mlx-lm, a blocking parity gate runs first: `scr chat` vs mlx_lm.generate,
# greedy, CLI mode (the ladder is server mode, so the gate is its own call). A
# failure skips every timing stage (all engines) for that model and is recorded as
# "parity_mlx_lm": false.
# Comparison engines, each on when installed; --engines names the ones to run:
#   mlx-lm      python with `import mlx_lm` ($VIRTUAL_ENV, python3, or --mlx-python)
#   ollama      `ollama serve` on --port, its MLX engine (the -mlx tags); pulls
#               on first use, ignores ignore_eos
#   llama.cpp   `llama-server`, plain Q4_K_M GGUF
#   vllm-metal  `vllm serve`, the same MLX checkpoint as scratchy
#   oMLX        `omlx serve` over the downloaded MLX checkpoint, twice:
#               as installed (FP16 KV), and with its TurboQuant KV turned on
#               (4-bit), so scratchy's TurboQuant default meets a like-for-like
#               rival as well as oMLX's own default
#   mistral.rs  `mistralrs serve`, the same Q4_K_M GGUF as llama.cpp; a Rust
#               control: is scratchy fast because of its compiler, or Rust?
# Every engine but mlx-lm is restarted per axis, sized for that axis (concurrent
# requests x context), since most of them reserve KV for that up front.
#
# Pins: an agreed run names every engine's version and scratchy's commit, and
# the runner refuses to start if anything installed differs:
#   --pin scratchy=f378699ec --pin mlx-lm=0.31.3 --pin ollama=0.35.1 ...
# (llama.cpp pins its build number, b1234). --unpinned runs anyway and says so.
#
# --serve-args "..." adds extra `scr serve` flags (recorded in the JSON). None
# by default: `scr serve` sizes --max-num-batched-tokens to the largest resident
# prefill bucket, the same as any user gets.
# --cell-timeout-s bounds each scaling cell; a timed-out cell skips the rest of
# that server's cells, since a hung server would hang them all.
# --fail-fast stops at the first scratchy build or parity-gate failure instead
# of going on to the next model. Either way, a model with no scratchy numbers
# fails the run (exit 1).
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "${HERE}/.." &>/dev/null && pwd)"

# Every id verified against the HuggingFace API with its download size.
MODELS_ALL=(
  "qwen3.6-27b=mlx-community/Qwen3.6-27B-4bit:mlx-affine-b4-g64-qembed"          # 16.1 GB, dense, GDN hybrid
  "qwen3.6-35b-a3b=mlx-community/Qwen3.6-35B-A3B-4bit:mlx-affine-b4-g64-qembed"  # 20.4 GB, MoE, GDN hybrid
  "gemma-4-26b-a4b-it=mlx-community/gemma-4-26b-a4b-it-4bit:mlx-affine-b4-g64"   # 15.4 GB, MoE, sliding window
  "gemma-4-31b-it=mlx-community/gemma-4-31b-it-4bit:mlx-affine-b4-g64"           # 18.4 GB, dense, sliding window
  "glm-4.5-air=mlx-community/GLM-4.5-Air-3bit:mlx-affine-b3-g64-qembed"          # 46.8 GB, MoE (12B active), 3-bit
)
# The default round is the MoE models, the ones that make most sense on a
# laptop: 3B, 4B and 12B active (GLM-4.5-Air only where it fits). --models
# dense runs the dense ones, --models full both.
MODELS_MOE="qwen3.6-35b-a3b,gemma-4-26b-a4b-it,glm-4.5-air"
MODELS_DENSE="qwen3.6-27b,gemma-4-31b-it"
MODELS_FULL="${MODELS_MOE},${MODELS_DENSE}"
MODELS_DEFAULT="${MODELS_MOE}"

# What each engine serves per stem; no entry, no column for that engine.
# ollama runs its MLX engine (the -mlx tags), as scratchy and mlx-lm do.
ollama_tag() {
    case "$1" in
        qwen3.6-27b)        echo "qwen3.6:27b-mlx" ;;
        qwen3.6-35b-a3b)    echo "qwen3.6:35b-mlx" ;;
        gemma-4-26b-a4b-it) echo "gemma4:26b-mlx" ;;
        gemma-4-31b-it)     echo "gemma4:31b-mlx" ;;
    esac
}
# llama.cpp and mistral.rs: plain Q4_K_M from one publisher for every model, as
# repo:file (unsloth ships mixed "UD" quants for two of the three).
gguf_ref() {
    case "$1" in
        qwen3.6-27b)        echo "bartowski/Qwen_Qwen3.6-27B-GGUF:Qwen_Qwen3.6-27B-Q4_K_M.gguf" ;;
        qwen3.6-35b-a3b)    echo "bartowski/Qwen_Qwen3.6-35B-A3B-GGUF:Qwen_Qwen3.6-35B-A3B-Q4_K_M.gguf" ;;
        gemma-4-26b-a4b-it) echo "bartowski/google_gemma-4-26B-A4B-it-GGUF:google_gemma-4-26B-A4B-it-Q4_K_M.gguf" ;;
        gemma-4-31b-it)     echo "bartowski/google_gemma-4-31B-it-GGUF:google_gemma-4-31B-it-Q4_K_M.gguf" ;;
    esac
}
# The model each GGUF was converted from. mistral.rs reads its config and
# tokenizer from there: these GGUF repos ship a vision projector too, and for
# a multimodal GGUF it will not start without the original config.json.
orig_ref() {
    case "$1" in
        qwen3.6-27b)        echo "Qwen/Qwen3.6-27B" ;;
        qwen3.6-35b-a3b)    echo "Qwen/Qwen3.6-35B-A3B" ;;
        gemma-4-26b-a4b-it) echo "google/gemma-4-26B-A4B-it" ;;
        gemma-4-31b-it)     echo "google/gemma-4-31B-it" ;;
    esac
}
# glm-4.5-air has neither: ollama has no GLM-4.5-Air, and the nearest GGUF
# (Q3_K_M, about 55 GB in two files) is a different quant and larger than the
# 3-bit MLX checkpoint, so only the MLX engines run it.

# The scratchy preset a stem builds (model/<preset>), when it is not the stem.
# qwen3.6-27b's own preset is the vision-language arch, which `scr chat` and
# `scr serve` cannot load ("Unsupported arch Qwen3_5ForConditionalGeneration");
# the text-only preset reads the same checkpoint and passes the parity gate.
model_preset() {
    case "$1" in
        qwen3.6-27b) echo "qwen3.6-27b-text-only" ;;
        *)           echo "$1" ;;
    esac
}

# What a model needs beyond the defaults (0: no limit). GLM-4.5-Air's 47 GB of
# weights fit only a 64 GB Mac, and its fp16 KV (about 190 KB a token) leaves
# room there for 4 users at the longest prompts, not 8 or 16.
model_min_gb()   { case "$1" in glm-4.5-air) echo 64 ;; *) echo 0 ;; esac; }
model_max_conc() { case "$1" in glm-4.5-air) echo 4 ;;  *) echo 0 ;; esac; }

# Extra --chat-template-config for the parity gate's mlx-lm side, per stem;
# empty = mlx-lm's defaults already render the checkpoint's chat template the
# way `scr chat` does. mlx_lm.generate forces `enable_thinking` on for any
# vocab with think tokens, but gemma-4's own template defaults it OFF (it
# prefills an empty thought channel), so without this pin the gate compares
# two different prompts and its 12/40-token budgets die inside mlx-lm's
# thought before any answer exists to agree on. Qwen3.5's template defaults
# thinking on, matching mlx-lm's injection, so it needs nothing here; so do
# Qwen3.6 (27B and 35B-A3B share one template) and GLM-4.5-Air, whose template
# thinks unless enable_thinking is set false.
# The single quotes ride along into --parity-cmd, whose shell-words split
# would otherwise strip the JSON's double quotes and crash the child on
# json.loads — with stderr nulled, that masquerades as a parity mismatch.
mlx_parity_config() {
    case "$1" in
        gemma-4-*) echo "'{\"enable_thinking\":false}'" ;;
    esac
}

MODELS=()
SCENARIOS="frozen,cold,warm"
REPS=3
# Discarded launches before cold, so it measures a settled cache: scratchy's
# Metal aligned-weights cache takes about three launches to settle. The harness
# does one on its own; more come from a throwaway cold call, so 2 rounds up to 3.
PRIME=3
PORT=8751
NUM_PROMPTS=20
INPUT_LEN=64
OUTPUT_LEN=32
WARM_REQUESTS=20
OFFLINE=0
SKIP_BUILD=0
FAIL_FAST=0
OUT_DIR=""
KV_CACHE_DTYPE=""
EVICT=""                 # --exec picks purge on macOS by itself
EVICT_PATH=()
SETTLE_S=""
# --exec's 600 s default timed out gemma-4-31b-it (18.4 GB) on this class of machine.
READY_TIMEOUT_S=1800
CELL_TIMEOUT_S=3600
SERVE_ARGS=""
SEED=""
MLX_PYTHON=""
MLX_AUTO=1
OLLAMA_AUTO=1
ENGINES=""               # --engines: every installed engine when empty
PINS=()
UNPINNED=0
SCALING=1
SCALE_AXES="conc,grid"
SCALE_CONC="1,4,16"
SCALE_INPUT="128,512,2048,8192"
SCALE_OUTPUT="16,64,256,1024"
SCALE_GRID_INPUT="128,1024,4096"
SCALE_GRID_OUTPUT="16,128,512"
SCALE_BASE_INPUT=512
SCALE_BASE_OUTPUT=128
SCALE_BASE_CONC=8
SCALE_NUM_PROMPTS=12
SCALE_WARMUPS=2
SCALE_SEED_BASE=20260927

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)            IFS=',' read -r -a MODELS <<<"$2"; shift 2 ;;
        --scenarios)         SCENARIOS="$2"; shift 2 ;;
        --reps)              REPS="$2"; shift 2 ;;
        --prime)             PRIME="$2"; shift 2 ;;
        --port)              PORT="$2"; shift 2 ;;
        --num-prompts)       NUM_PROMPTS="$2"; shift 2 ;;
        --input-len)         INPUT_LEN="$2"; shift 2 ;;
        --output-len)        OUTPUT_LEN="$2"; shift 2 ;;
        --warm-requests)     WARM_REQUESTS="$2"; shift 2 ;;
        --kv-cache-dtype)    KV_CACHE_DTYPE="$2"; shift 2 ;;
        --evict)             EVICT="$2"; shift 2 ;;
        --evict-path)        EVICT_PATH+=("$2"); shift 2 ;;
        --settle-s)          SETTLE_S="$2"; shift 2 ;;
        --ready-timeout-s)   READY_TIMEOUT_S="$2"; shift 2 ;;
        --cell-timeout-s)    CELL_TIMEOUT_S="$2"; shift 2 ;;
        --serve-args)        SERVE_ARGS="$2"; shift 2 ;;
        --seed)              SEED="$2"; shift 2 ;;
        --mlx-python)        MLX_PYTHON="$2"; shift 2 ;;
        --no-mlx)            MLX_AUTO=0; MLX_PYTHON=""; shift ;;
        --no-ollama)         OLLAMA_AUTO=0; shift ;;
        --scaling)           SCALING=1; shift ;;
        --no-scaling)        SCALING=0; shift ;;
        --scale-axes)        SCALE_AXES="$2"; shift 2 ;;
        --scale-conc)        SCALE_CONC="$2"; shift 2 ;;
        --scale-input)       SCALE_INPUT="$2"; shift 2 ;;
        --scale-output)      SCALE_OUTPUT="$2"; shift 2 ;;
        --scale-grid-input)  SCALE_GRID_INPUT="$2"; shift 2 ;;
        --scale-grid-output) SCALE_GRID_OUTPUT="$2"; shift 2 ;;
        --scale-num-prompts) SCALE_NUM_PROMPTS="$2"; shift 2 ;;
        --scale-warmups)     SCALE_WARMUPS="$2"; shift 2 ;;
        --scale-base-input)  SCALE_BASE_INPUT="$2"; shift 2 ;;
        --scale-base-output) SCALE_BASE_OUTPUT="$2"; shift 2 ;;
        --scale-base-conc)   SCALE_BASE_CONC="$2"; shift 2 ;;
        --engines)           ENGINES="$2"; shift 2 ;;
        --pin)               PINS+=("$2"); shift 2 ;;
        --unpinned)          UNPINNED=1; shift ;;
        --offline)           OFFLINE=1; shift ;;
        --skip-build)        SKIP_BUILD=1; shift ;;
        --fail-fast)         FAIL_FAST=1; shift ;;
        --out-dir)           OUT_DIR="$2"; shift 2 ;;
        -h|--help)           awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/,""); print}' "$0"; exit 0 ;;
        *)                   echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ ${#MODELS[@]} -eq 0 ]] && IFS=',' read -r -a MODELS <<<"${MODELS_DEFAULT}"
case "${MODELS[*]}" in
    all)   MODELS=("${MODELS_ALL[@]}") ;;
    moe)   IFS=',' read -r -a MODELS <<<"${MODELS_MOE}" ;;
    dense) IFS=',' read -r -a MODELS <<<"${MODELS_DENSE}" ;;
    full)  IFS=',' read -r -a MODELS <<<"${MODELS_FULL}" ;;
esac
# A bare stem resolves against MODELS_ALL; a full stem=id[:quant] is used as is.
for i in "${!MODELS[@]}"; do
    [[ "${MODELS[i]}" == *=* ]] && continue
    hit=""
    for e in "${MODELS_ALL[@]}"; do [[ "${e%%=*}" == "${MODELS[i]}" ]] && hit="${e}"; done
    [[ -n "${hit}" ]] || { echo "unknown model: ${MODELS[i]} (see MODELS_ALL)" >&2; exit 2; }
    MODELS[i]="${hit}"
done
(( OFFLINE )) && export HF_HUB_OFFLINE=1

if (( MLX_AUTO )) && [[ -z "${MLX_PYTHON}" ]]; then
    for py in ${VIRTUAL_ENV:+"${VIRTUAL_ENV}/bin/python"} python3; do
        if "${py}" -c 'import mlx_lm' 2>/dev/null; then MLX_PYTHON="$(command -v "${py}")"; break; fi
    done
fi
OLLAMA_BIN=""
(( OLLAMA_AUTO )) && OLLAMA_BIN="$(command -v ollama || true)"
LLAMA_BIN="$(command -v llama-server || true)"
VLLM_BIN="$(command -v vllm || true)"
OMLX_BIN="$(command -v omlx || true)"
[[ -z "${OMLX_BIN}" && -x "${HOME}/.omlx/bin/omlx" ]] && OMLX_BIN="${HOME}/.omlx/bin/omlx"
MISTRALRS_BIN="$(command -v mistralrs || true)"

# key|label|variable holding its binary: one row per comparison engine, in the
# order the page shows them. key names its JSON fields (cache_ladder_<key>, ...).
ENGINE_ROWS=(
    "mlx_lm|mlx-lm|MLX_PYTHON"
    "ollama|ollama|OLLAMA_BIN"
    "llama_cpp|llama.cpp|LLAMA_BIN"
    "vllm_metal|vllm-metal|VLLM_BIN"
    "omlx|oMLX|OMLX_BIN"
    "omlx_tq|oMLX TurboQuant|OMLX_BIN"
    "mistralrs|mistral.rs|MISTRALRS_BIN"
)
# --engines: run exactly these (by label), and refuse if one is not installed.
lower() { echo "$1" | tr '[:upper:]' '[:lower:]'; }
if [[ -n "${ENGINES}" ]]; then
    for row in "${ENGINE_ROWS[@]}"; do
        IFS='|' read -r key label binvar <<<"${row}"
        if [[ ",$(lower "${ENGINES}")," == *",$(lower "${label}"),"* || ",$(lower "${ENGINES}")," == *",${key},"* ]]; then
            [[ -n "${!binvar}" ]] || { echo "--engines names ${label}, which is not installed" >&2; exit 2; }
        else
            printf -v "${binvar}" '%s' ""
        fi
    done
fi

BIN="${ROOT}/target/release/scr"
chip="$(sysctl -n machdep.cpu.brand_string)"
slug="$(echo "${chip}" | tr '[:upper:] ' '[:lower:]-' | sed 's/[^a-z0-9-]//g')"
: "${OUT_DIR:="${ROOT}/bench_results/metal_matrix"}"
RAW="${OUT_DIR}/${slug}"
# Named <machine>-<start time>-<sha8>.json, the name site/src/metal/ requires,
# so a finished run drops straight into site/data/metal/ and a second run of
# one commit on one day never overwrites the first. One timestamp feeds
# both the name and generated_utc, so the two can never disagree.
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
stamp="${STARTED_UTC//:/}"
JSON="${OUT_DIR}/${slug}-${stamp}-$(git -C "${ROOT}" rev-parse HEAD | cut -c1-8).json"
mkdir -p "${RAW}"
MEM_GB=$(( $(sysctl -n hw.memsize) / 1073741824 ))

die() { echo "error: $*" >&2; exit 1; }
command -v cargo >/dev/null || die "cargo not on PATH"
[[ "$(uname -s)" == "Darwin" ]] || die "this runner is for Apple Silicon; use the cuda/spyre runner elsewhere"

# ---- versions and pins -------------------------------------------------------
# Each Mac installs the comparison engines on its own schedule, so a run records
# which releases it measured, and an agreed run pins them: a newer engine on one
# Mac must not read as faster hardware.
engine_version() { # key
    case "$1" in
        scratchy)   git -C "${ROOT}" rev-parse HEAD ;;
        mlx_lm)     "${MLX_PYTHON}" -c 'import mlx_lm; print(mlx_lm.__version__)' ;;
        ollama)     "${OLLAMA_BIN}" --version | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' | tail -1 ;;
        llama_cpp)  # source builds print version: 11429 (...), Homebrew version: 0.6.0 (build 11429, ...)
                    "${LLAMA_BIN}" --version 2>&1 | grep -oE 'build [0-9]+|version: [0-9]+( |$)' | head -1 \
                        | grep -oE '[0-9]+' | sed 's/^/b/' ;;
        vllm_metal) # the Homebrew formula's version, else the plugin in vllm's own python
                    local v; v="$(brew list --versions vllm-metal 2>/dev/null | awk '{print $2}')"
                    [[ -n "${v}" ]] && echo "${v}" \
                        || "$(head -1 "${VLLM_BIN}" | sed 's/^#!//')" -c 'import importlib.metadata as m; print(m.version("vllm-metal"))' ;;
        omlx)       "${OMLX_BIN}" --version | grep -oE '[0-9]+(\.[0-9]+)+' | head -1 ;;
        mistralrs)  "${MISTRALRS_BIN}" --version | grep -oE '[0-9]+(\.[0-9]+)+' | head -1 ;;
    esac 2>/dev/null || true
}
pin_for() { # label -> its pinned version, or nothing; always succeeds (set -e). Any case.
    local p
    for p in ${PINS[@]+"${PINS[@]}"}; do
        if [[ "$(lower "${p%%=*}")" == "$(lower "$1")" ]]; then echo "${p#*=}"; return 0; fi
    done
    return 0
}
V_scratchy="$(engine_version scratchy)"
pin_problems=()
check_pin() { # label installed
    local want; want="$(pin_for "$1")"
    if [[ -z "${want}" ]]; then
        pin_problems+=("$1 is not pinned (installed: ${2:-unknown})")
    elif [[ "$1" == scratchy ]]; then
        [[ "$2" == "${want}"* ]] || pin_problems+=("scratchy is at ${2:0:12}, pinned ${want}")
    elif [[ "${2#b}" != "${want#b}" ]]; then
        pin_problems+=("$1 is ${2:-unknown}, pinned ${want}")
    fi
}
check_pin scratchy "${V_scratchy}"
known="scratchy"
for row in "${ENGINE_ROWS[@]}"; do
    IFS='|' read -r key label binvar <<<"${row}"
    known+=",${label}"
    [[ -n "${!binvar}" ]] || continue
    # Both oMLX rows run the one installed oMLX: one version, one pin.
    if [[ "${key}" == omlx_tq ]]; then printf -v V_omlx_tq '%s' "$(engine_version omlx)"; continue; fi
    printf -v "V_${key}" '%s' "$(engine_version "${key}")"
    v="V_${key}"; check_pin "${label}" "${!v}"
done
for p in ${PINS[@]+"${PINS[@]}"}; do
    [[ ",$(lower "${known}")," == *",$(lower "${p%%=*}"),"* ]] || pin_problems+=("--pin ${p}: no engine called ${p%%=*} (known: ${known})")
done
if (( ${#pin_problems[@]} )); then
    printf '  %s\n' "${pin_problems[@]}" >&2
    (( UNPINNED )) || die "pins do not match what is installed; fix them, or pass --unpinned to run anyway (recorded as unpinned)"
    echo "running unpinned (recorded in the JSON)" >&2
fi

echo "machine : ${chip}"
echo "models  : ${#MODELS[@]}"
echo "rungs   : ${SCENARIOS}"
echo "scratchy: ${V_scratchy:0:12}"
for row in "${ENGINE_ROWS[@]}"; do
    IFS='|' read -r key label binvar <<<"${row}"
    v="V_${key}"
    printf '%-10s: %s\n' "${label}" "${!binvar:+${!binvar} (${!v:-version unknown})}"
    [[ -n "${!binvar}" ]] || printf '%-10s  skipped (not installed, or left out by --engines)\n' ""
done
(( SCALING )) && echo "scaling : ${SCALE_AXES}"
echo "output  : ${JSON}"

# ---- machine block ----------------------------------------------------------
# Launches actually discarded before COLD: the harness always primes once, and
# run_ladder's throwaway call adds the rest (so 2 rounds up to 3).
PRIME_LAUNCHES=$(( PRIME > 2 ? PRIME : (PRIME > 1 ? 3 : 1) ))
# One line per engine for the JSON: key|label|installed|version|pin.
ENGINE_INFO="scratchy|scratchy|1|${V_scratchy}|$(pin_for scratchy)"
for row in "${ENGINE_ROWS[@]}"; do
    IFS='|' read -r key label binvar <<<"${row}"
    v="V_${key}"
    pl="${label}"; [[ "${key}" == omlx_tq ]] && pl=oMLX
    ENGINE_INFO+=$'\n'"${key}|${label}|$([[ -n "${!binvar}" ]] && echo 1 || echo 0)|${!v:-}|$(pin_for "${pl}")"
done
PIN_PROBLEMS="$(printf '%s\n' ${pin_problems[@]+"${pin_problems[@]}"})"
export ENGINE_INFO PIN_PROBLEMS UNPINNED
export STARTED_UTC PRIME_LAUNCHES INPUT_LEN OUTPUT_LEN WARM_REQUESTS SCENARIOS SCALING SCALE_AXES SCALE_CONC SCALE_INPUT SCALE_OUTPUT SCALE_GRID_INPUT \
       SCALE_GRID_OUTPUT SCALE_BASE_INPUT SCALE_BASE_OUTPUT SCALE_BASE_CONC SCALE_NUM_PROMPTS \
       MLX_PYTHON OLLAMA_BIN KV_CACHE_DTYPE SERVE_ARGS CELL_TIMEOUT_S MODELS_FULL
python3 - "${JSON}" "${chip}" <<'PY'
import json, os, subprocess, sys
out, chip = sys.argv[1:3]
e = os.environ
ints = lambda k: [int(x) for x in e[k].split(",") if x]
sh = lambda *c: subprocess.run(c, capture_output=True, text=True).stdout.strip()
sysctl = lambda k: sh("sysctl", "-n", k)
batt = sh("pmset", "-g", "batt")
json.dump({
  "schema": 2, "issue": 91, "epic": 3,
  "generated_utc": e["STARTED_UTC"],
  "generator": "scripts/bench_metal_matrix.sh",
  "measured_by": {"footprint": "this runner (cargo build, stat)",
                  "cache_ladder": "scr bench startup --exec",
                  "warm_serving": "scr bench serve", "scaling": "scr bench serve"},
  "machine": {
    "hw_model": sysctl("hw.model"), "chip": chip,
    "cores_total": int(sysctl("hw.ncpu") or 0),
    "cores_performance": int(sysctl("hw.perflevel0.physicalcpu") or 0),
    "cores_efficiency": int(sysctl("hw.perflevel1.physicalcpu") or 0),
    "memory_gb": round(int(sysctl("hw.memsize") or 0) / 1024**3),
    "gpu_wired_limit_mb": int(sysctl("iogpu.wired_limit_mb") or 0),   # 0: the macOS default
    "macos": sh("sw_vers", "-productVersion") + " (" + sh("sw_vers", "-buildVersion") + ")",
    "thermal_at_start": sh("pmset", "-g", "therm").replace("\n", " "),
    "power": batt.splitlines()[0] if batt else "",
  },
  "repo": {"sha": sh("git", "rev-parse", "HEAD"),
           "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
           "dirty": bool(sh("git", "status", "--porcelain"))},
  "config": {"scenarios": e["SCENARIOS"].split(","),
             # Every model this version of the runner measures (--models full),
             # so the page retires a dropped one while a subset (the MoE-only
             # default round, or any --models list) hides none.
             "current_models": e["MODELS_FULL"].split(","),
             "cold_priming_launches": int(e["PRIME_LAUNCHES"]),
             "startup_request": {"input_len": int(e["INPUT_LEN"]), "output_len": int(e["OUTPUT_LEN"]),
                                 "warm_requests": int(e["WARM_REQUESTS"])},
             "kv_cache_dtype": e["KV_CACHE_DTYPE"] or "default (TurboQuant)",
             "scratchy_serve_args": e["SERVE_ARGS"] or None,
             "cell_timeout_s": int(e["CELL_TIMEOUT_S"]),
             "scaling": None if e["SCALING"] != "1" else {
                 "axes": e["SCALE_AXES"].split(","), "conc": ints("SCALE_CONC"),
                 "input": ints("SCALE_INPUT"), "output": ints("SCALE_OUTPUT"),
                 "grid_input": ints("SCALE_GRID_INPUT"), "grid_output": ints("SCALE_GRID_OUTPUT"),
                 "base": {"input": int(e["SCALE_BASE_INPUT"]), "output": int(e["SCALE_BASE_OUTPUT"]),
                          "conc": int(e["SCALE_BASE_CONC"])},
                 "num_prompts_per_cell": int(e["SCALE_NUM_PROMPTS"]),
                 "concurrency_is": "offered (--max-concurrency), not the effective decode batch"},
             "comparison": {row.split("|")[0]: row.split("|")[2] == "1"
                            for row in e["ENGINE_INFO"].splitlines()[1:]},
             "engines": [dict(zip(("key", "label", "installed", "version", "pin"), row.split("|")))
                         | {"installed": row.split("|")[2] == "1"}
                         for row in e["ENGINE_INFO"].splitlines()],
             "engine_versions": {row.split("|")[0]: row.split("|")[3] or None
                                 for row in e["ENGINE_INFO"].splitlines()[1:]},
             "pinned": not [x for x in e["PIN_PROBLEMS"].splitlines() if x],
             "pin_problems": [x for x in e["PIN_PROBLEMS"].splitlines() if x]},
  "methodology": "docs/BENCHMARKING.md — rung definitions, fairness rules and disclosed asymmetries live there, not here",
  "models": [],
}, open(out, "w"), indent=2)
PY

# ---- servers ----------------------------------------------------------------
# One server at a time on ${PORT}; the exit trap stops it, and anything else
# this run started (a `scr bench` and the server it launched), on failure,
# Ctrl-C or kill. A kill takes effect once the current step's command returns.
srv=""
serve_up() { # log ready_path cmd...
    local log="$1" path="$2"; shift 2
    "$@" >>"${log}" 2>&1 &
    srv=$!
    local deadline=$(( $(date +%s) + READY_TIMEOUT_S ))
    while (( $(date +%s) < deadline )); do
        curl -fsS -m 2 "http://127.0.0.1:${PORT}${path}" >/dev/null 2>&1 && return 0
        kill -0 "${srv}" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}
serve_down() {
    [[ -n "${srv}" ]] || return 0
    # stderr off for the whole block: bash's "Terminated" job notice is not a failure.
    {
        kill -TERM "${srv}" || true
        for _ in $(seq 1 30); do kill -0 "${srv}" || break; sleep 1; done
        kill -KILL "${srv}" || true
        wait "${srv}" || true
    } 2>/dev/null
    srv=""
}
kill_tree() { local c; for c in $(pgrep -P "$1"); do kill_tree "${c}"; done; kill -TERM "$1" 2>/dev/null || true; }
cleanup() {
    serve_down
    local c; for c in $(pgrep -P $$); do kill_tree "${c}"; done
    # A killed subshell orphans its children, so also stop whatever is on this run's port.
    pkill -TERM -f "bench (startup|serve) .*(--port ${PORT}|127\.0\.0\.1:${PORT})" 2>/dev/null || true
    lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN -t 2>/dev/null | xargs kill -TERM 2>/dev/null || true
}
trap cleanup EXIT
trap 'exit 130' INT TERM HUP
ollama_env() { echo env OLLAMA_HOST="127.0.0.1:${PORT}" OLLAMA_KEEP_ALIVE=-1 OLLAMA_MAX_LOADED_MODELS=1 "$@"; }
ollama_up() { serve_up "${RAW}/serve-ollama-${stem}.log" /api/version $(ollama_env "$@") "${OLLAMA_BIN}" serve; }

max_of() { local m=0 x; IFS=',' read -r -a _xs <<<"$1"; for x in "${_xs[@]}"; do (( x > m )) && m=${x}; done; echo "${m}"; }

# ---- comparison engines -------------------------------------------------------
# Per engine key: <key>_model STEM ID prints what it serves for that model ("" =
# nothing, so no column); <key>_cmd MODEL PAR CTX prints its server command for
# PAR concurrent requests of up to CTX tokens (0 0 = its defaults); <key>_ready
# is the path that answers 200 once it can serve.
mlx_lm_model() { echo "$2"; }
mlx_lm_cmd()   { echo "${MLX_PYTHON} -m mlx_lm.server --model $1 --port ${PORT}"; }
mlx_lm_ready() { echo /v1/models; }

ollama_model() { ollama_tag "$1"; }
ollama_cmd() {
    local env="OLLAMA_HOST=127.0.0.1:${PORT} OLLAMA_KEEP_ALIVE=-1 OLLAMA_MAX_LOADED_MODELS=1"
    (( $2 )) && env+=" OLLAMA_NUM_PARALLEL=$2"
    (( $3 )) && env+=" OLLAMA_CONTEXT_LENGTH=$3"
    echo "env ${env} ${OLLAMA_BIN} serve"
}
ollama_ready() { echo /api/version; }

llama_cpp_model() { gguf_ref "$1"; }
llama_cpp_cmd() { # -c is the total across slots, so it is PAR x CTX
    local c=""; (( $2 && $3 )) && c=" -c $(( $2 * $3 )) -np $2"
    echo "${LLAMA_BIN} --host 127.0.0.1 --port ${PORT} -hf ${1%%:*} -hff ${1#*:}${c}"
}
llama_cpp_ready() { echo /health; }

vllm_metal_model() { echo "$2"; }
vllm_metal_cmd() {
    local x=""; (( $2 )) && x+=" --max-num-seqs $2"; (( $3 )) && x+=" --max-model-len $3"
    echo "${VLLM_BIN} serve $1 --host 127.0.0.1 --port ${PORT}${x}"
}
vllm_metal_ready() { echo /v1/models; }

# oMLX serves every model under --model-dir, so it gets a directory holding only
# this one, linked to the snapshot the other MLX engines already downloaded.
# Its caching stays on, as oMLX ships: --no-cache would drop it to plain mlx-lm
# batching (and may take TurboQuant with it). The SSD cache lives in the
# variant's own --base-path, so nothing outlives the run.
# Each variant keeps its data in its own --base-path under the run's output, so
# the reader's ~/.omlx is never read or written, and its per-model settings are
# exactly what the variant set. TurboQuant is one of those per-model settings
# (model_settings.json), not a flag, so omlx_tq writes it before it starts.
omlx_model()    { echo "$2"; }
omlx_tq_model() { echo "$2"; }
omlx_serve() { # base-dir PAR
    local x=""; (( $2 )) && x=" --max-concurrent-requests $2"
    echo "${OMLX_BIN} serve --base-path $1 --model-dir ${RAW}/omlx-models --host 127.0.0.1 --port ${PORT} --paged-ssd-cache-dir $1/ssd-cache${x}"
}
omlx_cmd()    { omlx_serve "${RAW}/omlx-home" "$2"; }
omlx_tq_cmd() { omlx_serve "${RAW}/omlx-home-tq" "$2"; }
omlx_ready()    { echo /v1/models; }
omlx_tq_ready() { echo /v1/models; }
omlx_turboquant() { # model name as oMLX lists it
    mkdir -p "${RAW}/omlx-home-tq"
    python3 - "${RAW}/omlx-home-tq/model_settings.json" "$1" <<'PY'
import json, sys
json.dump({"version": 1, "models": {sys.argv[2]: {"turboquant_kv_enabled": True, "turboquant_kv_bits": 4}}},
          open(sys.argv[1], "w"), indent=2)
PY
}

mistralrs_model() { gguf_ref "$1"; }
mistralrs_cmd() { # --max-seq-len defaults to 4096, too short for long prompts
    local x="" o; (( $2 )) && x+=" --max-seqs $2"; (( $3 )) && x+=" --max-seq-len $3"
    o=$(orig_ref "${stem}"); [[ -n "${o}" ]] && x+=" --tok-model-id ${o}"
    echo "${MISTRALRS_BIN} serve -m ${1%%:*} -f ${1#*:} -p ${PORT}${x}"
}
mistralrs_ready() { echo /v1/models; }

# Concurrent requests and context one axis needs: "PAR CTX" (+256 tokenizer slack).
axis_size() { # axis
    local par=${BASE_CONC} ctx
    case "$1" in
        conc)   par=$(max_of "${CONC_LIST}"); ctx=$(( SCALE_BASE_INPUT + SCALE_BASE_OUTPUT )) ;;
        input)  ctx=$(( $(max_of "${SCALE_INPUT}") + SCALE_BASE_OUTPUT )) ;;
        output) ctx=$(( SCALE_BASE_INPUT + $(max_of "${SCALE_OUTPUT}") )) ;;
        grid)   ctx=$(( $(max_of "${SCALE_GRID_INPUT}") + $(max_of "${SCALE_GRID_OUTPUT}") )) ;;
        *)      return 1 ;;
    esac
    # Never past the model's own context: some engines refuse to start there.
    local most; most=$(model_max_ctx "${id}")
    (( most && ctx + 256 > most )) && ctx=$(( most - 256 ))
    echo "${par} $(( ctx + 256 ))"
}

# The model's longest context, from its downloaded config.json (0 = unknown), so
# the input axis skips prompts the model cannot take instead of failing them.
model_max_ctx() { # hf id
    python3 - "$1" <<'PY' 2>/dev/null || echo 0
import glob, json, os, sys
hub = os.path.join(os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
for c in glob.glob(os.path.join(hub, "models--" + sys.argv[1].replace("/", "--"), "snapshots", "*", "config.json")):
    d = json.load(open(c)); t = d.get("text_config", d)
    print(t.get("max_position_embeddings") or d.get("max_position_embeddings") or 0); break
else:
    print(0)
PY
}

# The server's memory footprint, summed over its process tree (ollama runs each
# model in a child), sampled once a second into OUT (MiB) while PID lives.
# `footprint` gives the physical footprint in bytes, which counts Metal buffers
# (plain RSS does not); `top` reports the same figure but in whole GB past 10 GB,
# too coarse to tell two engines, or oMLX and oMLX TurboQuant, apart.
tree_pids() { local c; echo "$1"; for c in $(pgrep -P "$1"); do tree_pids "${c}"; done; }
footprint_bytes() { # pid -> bytes, or nothing if it has gone
    footprint -f bytes -p "$1" 2>/dev/null \
        | awk '{ for (i = 1; i < NF; i++) if ($i == "Footprint:") { print $(i + 1); exit } }' || true
}
mem_sampler() { # pid out
    local p b s
    while kill -0 "$1" 2>/dev/null; do
        s=0
        for p in $(tree_pids "$1"); do b=$(footprint_bytes "${p}"); s=$(( s + ${b:-0} )); done
        (( s )) && echo $(( s / 1048576 )) >>"$2"
        sleep 1
    done
}

# ---- scaling ----------------------------------------------------------------
# One `scr bench serve` per cell against the running server, written to
# "<prefix>.<axis>-<rung>.json". cell_model / cell_tok override the model name
# sent and the tokenizer that sizes prompts (ollama serves a tag).
numbers() { # bench-serve json -> one human line
    python3 - "$1" <<'PY' 2>/dev/null || echo "no result"
import json, sys
j = json.load(open(sys.argv[1]))
f = lambda v, spec: "-".rjust(int(spec.split(".")[0])) if v is None else format(v, spec)
un = j.get("unstreamed_requests") or 0
print(f"{j['output_throughput']:7.1f} tok/s · TTFT p50 {f(j['median_ttft_ms'], '6.0f')} ms"
      f" · TPOT p50 {f(j['median_tpot_ms'], '5.1f')} ms · {j['completed']} ok"
      + (f" · {un} unstreamed (untimed)" if un else ""))
PY
}
run_cell() { # prefix axis rung input output conc
    (( cells_hung )) && return 0
    local out_json="$1.$2-$3.json" seed rc=0
    seed=$(printf '%s' "${SCALE_SEED_BASE}:${stem}:$2:$3" | cksum | cut -d' ' -f1)
    printf '    %-16s in %-5s out %-5s conc %-3s ' "$2=$3" "$4" "$5" "$6"
    local mem="${out_json%.json}.mem" sampler=""
    rm -f "${mem}"
    [[ -n "${srv}" ]] && { mem_sampler "${srv}" "${mem}" & sampler=$!; }
    # perl's alarm survives exec: SIGALRM ends the client after CELL_TIMEOUT_S.
    # The client's output goes to a log; stderr off hides bash's job notice.
    { perl -e 'alarm shift; exec @ARGV' "${CELL_TIMEOUT_S}" \
        "${BIN}" bench serve --base-url "http://127.0.0.1:${PORT}" --model "${cell_model:-${id}}" \
        ${cell_tok:+--tokenizer "${cell_tok}"} \
        --num-prompts "${SCALE_NUM_PROMPTS}" --input-len "$4" --output-len "$5" \
        --max-concurrency "$6" --temperature 0 --seed "${seed}" --num-warmups "${SCALE_WARMUPS}" \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
        --output-json "${out_json}" --disable-tqdm >>"${RAW}/cells-${stem}.log" 2>&1; } 2>/dev/null || rc=$?
    [[ -n "${sampler}" ]] && { kill "${sampler}" 2>/dev/null; wait "${sampler}" 2>/dev/null; } || true
    [[ -s "${mem}" && -f "${out_json}" ]] && python3 - "${out_json}" "${mem}" <<'PY' || true
import json, statistics, sys
j, mib = json.load(open(sys.argv[1])), [float(x) for x in open(sys.argv[2]) if x.strip()]
j["server_mem_peak_mib"], j["server_mem_median_mib"] = max(mib), statistics.median(mib)
json.dump(j, open(sys.argv[1], "w"), indent=2)
PY
    if (( rc == 142 )); then
        cells_hung=1
        echo "timed out after ${CELL_TIMEOUT_S}s; skipping this server's remaining cells"
    elif (( rc )); then
        echo "failed (see ${RAW}/cells-${stem}.log)"
    else
        numbers "${out_json}"
    fi
}
run_scale_cells() { # prefix [axes]
    local prefix="$1" axes=",${2:-${SCALE_AXES}}," x o
    cells_hung=0
    local -a xs os
    if [[ "${axes}" == *",conc,"* ]]; then
        IFS=',' read -r -a xs <<<"${CONC_LIST}"
        for x in "${xs[@]}"; do run_cell "${prefix}" conc "${x}" "${SCALE_BASE_INPUT}" "${SCALE_BASE_OUTPUT}" "${x}"; done
    fi
    if [[ "${axes}" == *",input,"* ]]; then
        local max_ctx; max_ctx=$(model_max_ctx "${id}")   # here, not once: scratchy's run downloads the model
        IFS=',' read -r -a xs <<<"${SCALE_INPUT}"
        for x in "${xs[@]}"; do
            if (( max_ctx && x + SCALE_BASE_OUTPUT + 256 > max_ctx )); then
                printf '    %-16s skipped: past the model'"'"'s %s-token context\n' "input=${x}" "${max_ctx}"
                continue
            fi
            run_cell "${prefix}" input "${x}" "${x}" "${SCALE_BASE_OUTPUT}" "${BASE_CONC}"
        done
    fi
    if [[ "${axes}" == *",output,"* ]]; then
        IFS=',' read -r -a xs <<<"${SCALE_OUTPUT}"
        for x in "${xs[@]}"; do run_cell "${prefix}" output "${x}" "${SCALE_BASE_INPUT}" "${x}" "${BASE_CONC}"; done
    fi
    if [[ "${axes}" == *",grid,"* ]]; then
        IFS=',' read -r -a xs <<<"${SCALE_GRID_INPUT}"
        IFS=',' read -r -a os <<<"${SCALE_GRID_OUTPUT}"
        for x in "${xs[@]}"; do for o in "${os[@]}"; do
            run_cell "${prefix}" grid "${x}x${o}" "${x}" "${o}" "${BASE_CONC}"
        done; done
    fi
}

# One comparison engine on one model: a warm-up start (so a download never lands
# inside a measurement, and to learn the name it serves the model under), the
# startup ladder, then one server per scaling axis, sized for that axis.
served_name() { # the first model the running server lists
    curl -fsS -m 10 "http://127.0.0.1:${PORT}/v1/models" \
        | python3 -c 'import json,sys; d=json.load(sys.stdin)["data"]; print(d[0]["id"] if d else "")' 2>/dev/null
}
engine_prepare() { # key label model -> prints the name requests carry; fails if it cannot serve
    local key="$1" label="$2" model="$3" log="${RAW}/serve-$1-${stem}.log" name="" snap
    if [[ "${key}" == omlx || "${key}" == omlx_tq ]]; then
        rm -rf "${RAW}/omlx-home-tq"     # learn the name with TurboQuant off
        snap="${HF_HOME:-${HOME}/.cache/huggingface}/hub/models--${model//\//--}"
        [[ -f "${snap}/refs/main" ]] || return 1
        rm -rf "${RAW}/omlx-models" && mkdir -p "${RAW}/omlx-models/${model%%/*}"
        ln -s "${snap}/snapshots/$(cat "${snap}/refs/main")" "${RAW}/omlx-models/${model}"
    fi
    serve_up "${log}" "$("${key}_ready")" bash -c "exec $("${key}_cmd" "${model}" 0 0)" || { serve_down; return 1; }
    case "${key}" in
        mlx_lm|vllm_metal) name="${model}" ;;
        ollama)
            show() { curl -fsS -m 10 "http://127.0.0.1:${PORT}/api/show" -d "{\"model\":\"${model}\"}"; }
            if ! show >/dev/null 2>&1; then
                (( OFFLINE )) && { serve_down; return 1; }
                echo "    pulling ${model} (first use)" >&2
                OLLAMA_HOST="127.0.0.1:${PORT}" "${OLLAMA_BIN}" pull "${model}" >>"${log}" 2>&1 || { serve_down; return 1; }
            fi
            # Its quantization and digest, one per line, in a file: this runs in
            # the subshell that prints the served name, so a variable is lost.
            { show 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["details"]["quantization_level"])' || echo
              curl -fsS -m 10 "http://127.0.0.1:${PORT}/api/tags" | python3 -c '
import json,sys; t=sys.argv[1]; t=t if ":" in t else t+":latest"
print(next((m["digest"] for m in json.load(sys.stdin)["models"] if m["name"]==t), ""))' "${model}" 2>/dev/null || echo
            } >"${RAW}/served-ollama-${stem}"
            name="${model}" ;;
        *)  name="$(served_name)" ;;
    esac
    serve_down
    [[ "${key}" == omlx_tq && -n "${name}" ]] && omlx_turboquant "${name}"
    [[ -n "${name}" ]] && echo "${name}"
}
run_engine() { # key label
    local key="$1" label="$2" model name ax par ctx
    model="$("${key}_model" "${stem}" "${id}")"
    [[ -n "${model}" ]] || { echo "--- ${label}: nothing to serve for ${stem}, skipped"; return 0; }
    echo "--- ${label}: ${model}"
    if ! name="$(engine_prepare "${key}" "${label}" "${model}")"; then
        echo "    ${label} cannot serve ${model} (offline, a failed download, or it would not start) — skipped;" \
             "see ${RAW}/serve-${key}-${stem}.log" >&2
        return 0
    fi
    printf -v "M_${key}" '%s' "${model}"; export "M_${key}"
    if [[ -f "${RAW}/served-${key}-${stem}" ]]; then
        { read -r "Q_${key}" || true; read -r "R_${key}" || true; } <"${RAW}/served-${key}-${stem}"
        export "Q_${key}" "R_${key}"
    fi
    local evict=()
    [[ "${key}" == ollama && ",${SCENARIOS}," == *",frozen,"* ]] && evict=(--evict-path "${HOME}/.ollama/models/blobs")
    run_ladder "${label}" "${name}" "$("${key}_cmd" "${model}" 0 0)" "$([[ ${key} == mlx_lm ]] && echo mlx-lm || echo vllm)" \
        "${RAW}/exec-${key}-${stem}.json" ${evict[@]+"${evict[@]}"}
    (( SCALING )) || return 0
    cell_model="${name}"; cell_tok="${id}"
    IFS=',' read -r -a axes <<<"${SCALE_AXES}"
    for ax in "${axes[@]}"; do
        read -r par ctx <<<"$(axis_size "${ax}")" || continue
        echo "--- scaling sweep, ${label} ${ax} (${par} at once, context ${ctx})"
        if serve_up "${RAW}/serve-${key}-${stem}.log" "$("${key}_ready")" bash -c "exec $("${key}_cmd" "${model}" "${par}" "${ctx}")"; then
            run_scale_cells "${RAW}/scale-${key}-${stem}" "${ax}"
        else
            echo "    ${label} server never became ready — ${RAW}/serve-${key}-${stem}.log" >&2
        fi
        serve_down
    done
    cell_model=""; cell_tok=""
}

# ---- summary: one model's rows, or everything ----------------------------
summary() { # [stem]
python3 - "${JSON}" "${1:-}" <<'PY'
import json, statistics, sys
d, only = json.load(open(sys.argv[1])), sys.argv[2]
m = d["machine"]
if not only:
    print(f"{m['chip']} · {m['cores_total']} cores ({m['cores_performance']}P+{m['cores_efficiency']}E) "
          f"· {m['memory_gb']} GB · macOS {m['macos']}")
print(f"{'model':26}{'build':>7}{'MiB':>6}{'frozen':>9}{'cold':>8}{'1st req':>9}{'warm':>8}"
      f"{'TTFT':>8}{'TPOT':>8}{'tok/s':>8}")
def med(reps, scenario, field):
    vals = [r[field] for r in (reps or []) if r.get("scenario") == scenario and r.get(field) is not None]
    return statistics.median(vals) if vals else None
def ms(s):
    return None if s is None else s * 1000
def fmt(v, nd=0):
    return "-" if v is None else f"{v:.{nd}f}"
def cells(e, key, axis):
    return [c for c in (e.get(key) or []) if c["axis"] == axis]
for e in [x for x in d["models"] if x["stem"] == only or not only]:
    f, ladder, w = e["footprint"], e.get("cache_ladder"), e.get("warm_serving") or {}
    mib = round(f["binary_bytes"] / 1048576) if f["binary_bytes"] else None
    print(f"{e['stem']:26}{fmt(f['build_seconds']):>7}{fmt(mib):>6}"
          f"{fmt(med(ladder,'frozen','t_ready_s'), 2):>9}{fmt(med(ladder,'cold','t_ready_s'), 2):>8}"
          f"{fmt(med(ladder,'cold','ttft_from_send_s'), 2):>9}"
          f"{fmt(ms(med(ladder,'warm','ttft_from_send_s'))):>8}"
          f"{fmt(w.get('median_ttft_ms')):>8}{fmt(w.get('median_tpot_ms'), 1):>8}"
          f"{fmt(w.get('output_throughput'), 1):>8}")
    if not e["built"]:
        print("    BUILD FAILED — see the build log in this machine's raw/ directory")
    elif e.get("parity_mlx_lm") is False:
        print(f"    PARITY FAILED vs mlx-lm — not timed; see parity-{e['stem']}.log in raw/")
    elif ladder is None and not w and not e.get("scaling"):
        print("    NO SCRATCHY NUMBERS — the server never served; see the exec and serve logs in raw/")
    elif ladder is None:
        print("    cache ladder unavailable")
    # Every comparison engine the run recorded, in the runner's order.
    rivals = [(x["label"], f"scaling_{x['key']}") for x in d["config"].get("engines", [])
              if x["key"] != "scratchy"] or [("mlx-lm", "scaling_mlx_lm"), ("ollama", "scaling_ollama")]
    conc = sorted(cells(e, "scaling", "conc"), key=lambda c: c["rung"])
    if conc:
        # TPOT at the top rung vs conc<=2: the per-request cost of batching (~1.0 is good).
        top = conc[-1]
        base_tp = next((c["median_tpot_ms"] for c in conc if c["rung"] <= 2), None)
        line = f"    scaling: conc {top['rung']} -> {fmt(top.get('output_throughput'), 1)} tok/s"
        if top.get("median_tpot_ms") and base_tp:
            line += f" · TPOT x{top['median_tpot_ms'] / base_tp:.2f} vs conc<=2"
        print(line)
        for name, key in rivals:
            t = [c for c in cells(e, key, "conc") if c["rung"] == top["rung"]]
            if t and t[0].get("output_throughput"):
                print(f"    scaling {name}: conc {top['rung']} -> {fmt(t[0]['output_throughput'], 1)} tok/s")
    # Text heat maps: scratchy / engine output tok/s per grid cell (>1.00 = scratchy faster).
    grid = {(c["input_len"], c["output_len"]): c for c in cells(e, "scaling", "grid")}
    ins = sorted({k[0] for k in grid}); outs = sorted({k[1] for k in grid})
    for name, key in rivals:
        them = {(c["input_len"], c["output_len"]): c for c in cells(e, key, "grid")}
        if not (grid and them):
            continue
        print(f"    grid, scratchy / {name} output tok/s (rows input, cols output):")
        print("      " + "in/out".rjust(8) + "".join(f"{o:>8}" for o in outs))
        for i in ins:
            row = ""
            for o in outs:
                a = grid.get((i, o), {}).get("output_throughput")
                b = them.get((i, o), {}).get("output_throughput")
                row += f"{a / b:>8.2f}" if (a and b) else f"{'-':>8}"
            print("      " + f"{i:>8}" + row)
if not only:
    print("\nfrozen/cold: startup s, launch -> ready (no request). 1st req: s, the cold launch's first"
          "\nrequest, send -> first token. warm: ms, send -> first token, median over unique prompts to a"
          "\nresident server. TTFT/TPOT ms and tok/s: `scr bench serve`, conc 1.")
    print(f"json -> {sys.argv[1]}")
PY
}

# ---- per model --------------------------------------------------------------
for entry in "${MODELS[@]}"; do
    stem="${entry%%=*}"; rest="${entry#*=}"
    id="${rest%%:*}"; quant="${rest#*:}"; [[ "${quant}" == "${rest}" ]] && quant=""
    echo; echo "################ ${stem} ################"
    need=$(model_min_gb "${stem}")
    if (( need && MEM_GB < need )); then
        echo "    skipped: needs ${need} GB, this Mac has ${MEM_GB} GB"
        python3 - "${JSON}" "${stem}" "${id}" "${need}" <<'PY'
import json, sys
js, stem, mid, need = sys.argv[1:5]
d = json.load(open(js))
d.setdefault("skipped_models", []).append(
    {"stem": stem, "model_id": mid, "reason": f"needs {need} GB of memory"})
json.dump(d, open(js, "w"), indent=2)
PY
        continue
    fi
    # Fewer users at once for a model whose KV cache would not fit more.
    BASE_CONC=${SCALE_BASE_CONC}; CONC_LIST=${SCALE_CONC}; cap=$(model_max_conc "${stem}")
    if (( cap )); then
        (( BASE_CONC > cap )) && BASE_CONC=${cap}
        CONC_LIST=$(tr ',' '\n' <<<"${SCALE_CONC}" | awk -v c="${cap}" '$1 <= c' | paste -sd, -)
        [[ -n "${CONC_LIST}" ]] || CONC_LIST=${cap}
        echo "    at most ${cap} users at once: users ${CONC_LIST}, ${BASE_CONC} on the other sweeps"
    fi

    feats="metal,serve,bench,model/$(model_preset "${stem}")${quant:+,quant/${quant}}"
    build_secs=""; bytes=""; built=1
    if (( ! SKIP_BUILD )); then
        echo "--- build -F ${feats}"
        t0=$(date +%s)
        if cargo build --release -p scratchy-cli --features "${feats}" >"${RAW}/build-${stem}.log" 2>&1; then
            build_secs=$(( $(date +%s) - t0 ))
            bytes=$(stat -f%z "${BIN}")
            echo "    ${build_secs}s · $((bytes/1024/1024)) MiB"
        else
            built=0
            echo "    BUILD FAILED — ${RAW}/build-${stem}.log" >&2
            tail -3 "${RAW}/build-${stem}.log" | sed 's/^/    /' >&2
            (( FAIL_FAST )) && die "--fail-fast: stopping at ${stem}'s build failure"
        fi
    fi
    # The next build overwrites ${BIN}, so each model keeps its own copy.
    model_bin="${RAW}/scr-${stem}"
    (( built )) && cp "${BIN}" "${model_bin}"

    rm -f "${RAW}"/exec*-"${stem}".json "${RAW}"/scale*-"${stem}".*.json "${RAW}/serve-${stem}.json" "${RAW}"/served-*-"${stem}"
    cell_model=""; cell_tok=""
    for row in "${ENGINE_ROWS[@]}"; do IFS='|' read -r key _ _ <<<"${row}"; unset "M_${key}" "Q_${key}" "R_${key}"; done

    # The comparison engines only need a working `scr` client, not this model's build.
    if ! { [[ -x "${BIN}" ]] && "${BIN}" bench startup --help 2>/dev/null | grep -q -- '--exec'; }; then
        echo "    no scr with \`bench startup --exec\` — nothing to measure with" >&2
        continue
    fi
    ladder=(--exec --mode server --port "${PORT}"
            --input-len "${INPUT_LEN}" --output-len "${OUTPUT_LEN}" --warm-requests "${WARM_REQUESTS}"
            --ready-timeout-s "${READY_TIMEOUT_S}"
            --remove-path "${HOME}/.cache/scratchy/metal-aligned-weights")
    [[ -n "${EVICT}" ]]    && ladder+=(--evict "${EVICT}")
    [[ -n "${SETTLE_S}" ]] && ladder+=(--settle-s "${SETTLE_S}")
    [[ -n "${SEED}" ]]     && ladder+=(--seed "${SEED}")
    for ep in ${EVICT_PATH[@]+"${EVICT_PATH[@]}"}; do ladder+=(--evict-path "${ep}"); done
    # FROZEN deletes the caches COLD needs settled, and the harness primes only
    # once and never after FROZEN, so the ladder runs as up to three calls:
    # FROZEN alone, a throwaway COLD call to settle the caches, then the rest.
    # The kept calls' runs are joined into one file.
    run_ladder() { # label model child-cmd backend out_json [extra...]
        echo "--- scr bench startup --exec, $1 (${SCENARIOS})"
        local label="$1" model="$2" cmd="$3" backend="$4" out="$5"; shift 5
        local frozen="" rest="" sc parts=()
        for sc in ${SCENARIOS//,/ }; do
            if [[ "${sc}" == frozen ]]; then frozen=frozen; else rest+="${rest:+,}${sc}"; fi
        done
        rm -f "${out}" "${out%.json}".{frozen,rest,prime}.json
        startup() { # scenarios reps out_json [extra...]
            local scs="$1" reps="$2" o="$3"; shift 3
            "${BIN}" bench startup --model "${model}" "${ladder[@]}" --scenarios "${scs}" --reps "${reps}" "$@" \
                --child-cmd "${cmd}" --backend "${backend}" --output-json "${o}"
        }
        show_report() {
            awk '/^ *scenario +mode/{n=0} {l[n++]=$0} END{for(i=(n>6&&!h?n-6:0);i<n;i++)print l[i]} /^ *scenario +mode/{h=1}' \
                | sed 's/^/    /'
        }
        if [[ -n "${frozen}" ]]; then
            startup frozen "${REPS}" "${out%.json}.frozen.json" "$@" 2>&1 | show_report \
                || echo "    ${label} --exec frozen returned non-zero (validity gate, or a real failure); see above" >&2
            parts+=("${out%.json}.frozen.json")
        fi
        if [[ -n "${rest}" ]]; then
            if (( PRIME > 1 )); then
                echo "    priming: $(( PRIME > 2 ? PRIME : 3 )) discarded launches"
                startup cold $(( PRIME > 2 ? PRIME - 2 : 1 )) "${out%.json}.prime.json" "$@" \
                    >>"${RAW}/prime-${label}.log" 2>&1 \
                    || echo "    ${label} priming returned non-zero; see ${RAW}/prime-${label}.log" >&2
            fi
            startup "${rest}" "${REPS}" "${out%.json}.rest.json" "$@" 2>&1 | show_report \
                || echo "    ${label} --exec returned non-zero (validity gate, or a real failure); see above" >&2
            parts+=("${out%.json}.rest.json")
        fi
        python3 - "${out}" ${parts[@]+"${parts[@]}"} <<'PY'
import json, os, statistics, sys
runs = [r for p in sys.argv[2:] if os.path.exists(p) for r in json.load(open(p))]
if runs:
    json.dump(runs, open(sys.argv[1], "w"), indent=2)
# FROZEN and COLD ran as separate harness calls, so the harness's own
# frozen-vs-cold checks never saw both; repeat them here, same rules.
def med(sc, k):
    v = [r[k] for r in runs if r.get("scenario") == sc and r.get(k) is not None]
    return statistics.median(v) if v else None
ff, fc = med("frozen", "major_faults"), med("cold", "major_faults")
tf, tc = med("frozen", "ttft_exec_s"), med("cold", "ttft_exec_s")
if ff is not None and fc is not None:
    print("    frozen vs cold")
    print(f"    - {'PASS' if ff > fc else '**FAIL**'} — FROZEN major faults {ff:.0f} vs COLD {fc:.0f} "
          f"({'eviction took effect' if ff > fc else 'eviction did NOT take effect'})")
    if tf is not None and tc is not None:
        print(f"    - {'PASS' if tf > tc else '**FAIL**'} — FROZEN ttft_exec {tf:.3f}s vs COLD {tc:.3f}s"
              f"{'' if tf > tc else '  (a frozen start should never be faster)'}")
PY
    }

    # ---- parity gate, blocking: a broken dequant path can be fast and wrong,
    # so scratchy and mlx-lm must agree on the same greedy prompts before
    # anything is timed. The gate needs `{prompt}` in both commands, so it is
    # its own CLI-mode call, not part of the server-mode ladder; only its exit
    # code matters. A failure skips every engine for this model: with no
    # scratchy numbers there is nothing to compare against.
    parity_ok=1
    if (( built )) && [[ -n "${MLX_PYTHON}" ]]; then
        echo "--- parity gate vs mlx-lm (blocking)"
        # The gate only means "same load path" if both sides render the SAME
        # prompt; see mlx_parity_config for where mlx-lm's defaults diverge
        # from the checkpoint template `scr chat` renders.
        pconfig="$(mlx_parity_config "${stem}")"
        if "${BIN}" bench startup --model "${id}" --exec --mode cli --scenarios warm --port "${PORT}" \
                --child-cmd "${model_bin} chat -m ${id} --device metal -q {prompt} --max-tokens {output_len} --temperature 0" \
                --backend scratchy \
                --parity-cmd "${MLX_PYTHON} -m mlx_lm.generate --model ${id} --prompt {prompt} --max-tokens {output_len} --temp 0${pconfig:+ --chat-template-config ${pconfig}}" \
                --parity-backend mlx-lm >"${RAW}/parity-${stem}.log" 2>&1; then
            # One line per prompt (name, OK, agree n/m chars), whatever the prompts are.
            grep -E "^ +[^ ]+ +OK +agree |PARITY OK" "${RAW}/parity-${stem}.log" | sed 's/^/    /'
        else
            parity_ok=0
            echo "    PARITY FAILED — not timing ${stem}; ${RAW}/parity-${stem}.log" >&2
            grep -E " (OK|FAIL) +agree | a: | b: |crashed|Error" "${RAW}/parity-${stem}.log" | head -12 | sed 's/^/    /' >&2
            (( FAIL_FAST )) && die "--fail-fast: stopping at ${stem}'s parity-gate failure"
        fi
    fi

    # ---- scratchy
    if (( built && parity_ok )); then
        run_ladder scratchy "${id}" \
            "${model_bin} serve ${id} --port ${PORT}${KV_CACHE_DTYPE:+ --kv-cache-dtype ${KV_CACHE_DTYPE}}${SERVE_ARGS:+ ${SERVE_ARGS}}" \
            scratchy "${RAW}/exec-${stem}.json"
        echo "--- scr bench serve (warm steady state, greedy, conc 1)"
        if serve_up "${RAW}/serve-${stem}.log" /v1/models "${model_bin}" serve "${id}" --port "${PORT}" \
                ${KV_CACHE_DTYPE:+--kv-cache-dtype "${KV_CACHE_DTYPE}"} ${SERVE_ARGS}; then
            "${BIN}" bench serve --base-url "http://127.0.0.1:${PORT}" --model "${id}" \
                --num-prompts "${NUM_PROMPTS}" --input-len "${INPUT_LEN}" --output-len "${OUTPUT_LEN}" \
                --max-concurrency 1 --temperature 0 \
                --seed "$(printf '%s' "${SCALE_SEED_BASE}:${stem}:warm" | cksum | cut -d' ' -f1)" \
                --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
                --output-json "${RAW}/serve-${stem}.json" --disable-tqdm >>"${RAW}/cells-${stem}.log" 2>&1 \
                && { printf '    '; numbers "${RAW}/serve-${stem}.json"; } \
                || echo "    bench serve failed (see ${RAW}/cells-${stem}.log)" >&2
            if (( SCALING )); then
                echo "--- scaling sweep, scratchy"
                run_scale_cells "${RAW}/scale-${stem}"
            fi
        else
            echo "    server never became ready — ${RAW}/serve-${stem}.log" >&2
        fi
        serve_down
    fi

    # ---- comparison engines, in page order; none when the parity gate failed
    if (( parity_ok )); then
        for row in "${ENGINE_ROWS[@]}"; do
            IFS='|' read -r key label binvar <<<"${row}"
            [[ -n "${!binvar}" ]] && run_engine "${key}" "${label}"
        done
    fi

    ENGINE_KEYS="$(for row in "${ENGINE_ROWS[@]}"; do echo "${row%%|*}"; done)"; export ENGINE_KEYS
    OFFLINE="${OFFLINE}" BASE_CONC="${BASE_CONC}" CONC_LIST="${CONC_LIST}" \
    python3 - "${JSON}" "${RAW}" "${stem}" "${id}" "${quant}" "${feats}" "${built}" \
              "${build_secs}" "${bytes}" "${parity_ok}" "${MLX_PYTHON}" <<'PY'
import glob, json, os, re, sys
js, raw, stem, mid, quant, feats, built, secs, size, parity_ok, mlx_python = sys.argv[1:12]
env = os.environ
KEEP = ["median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms", "median_itl_ms",
        "p99_itl_ms", "median_e2el_ms", "output_throughput", "request_throughput",
        "completed", "total_output_tokens", "duration", "unstreamed_requests",
        "server_mem_peak_mib", "server_mem_median_mib"]
CELL = re.compile(r"\.(conc|input|output|grid)-(\d+)(?:x(\d+))?\.json$")
def load(name):
    p = os.path.join(raw, name)
    return json.load(open(p)) if os.path.exists(p) else None
def metrics(j):
    return {k: j[k] for k in KEEP if k in j} if j else None
def scaling(engine):
    cells = []
    for p in sorted(glob.glob(os.path.join(raw, f"scale{engine}-{stem}.*.json"))):
        m = CELL.search(p)
        if not m or not (j := load(os.path.basename(p))):
            continue
        axis, a, b = m.groups()
        where = ({"rung": f"{a}x{b}", "input_len": int(a), "output_len": int(b)}
                 if axis == "grid" else {"rung": int(a)})
        cells.append({"axis": axis, **where, **metrics(j)})
    return cells or None
d = json.load(open(js))
keys = env["ENGINE_KEYS"].split()
engines = {f"cache_ladder_{k}": load(f"exec-{k}-{stem}.json") for k in keys}
engines |= {f"scaling_{k}": scaling(f"-{k}") for k in keys}
def revision(repo):
    """Which version of a Hugging Face repo's files this run used: the snapshot
    already downloaded, else (llama.cpp keeps its own cache) the repo's main at
    run time, which is what it fetches. None offline or when neither answers."""
    hub = os.path.join(env.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
    ref = os.path.join(hub, "models--" + repo.replace("/", "--"), "refs", "main")
    if os.path.exists(ref):
        return {"revision": open(ref).read().strip(), "from": "downloaded snapshot"}
    if env.get("OFFLINE") == "1":
        return None
    try:
        import urllib.request
        with urllib.request.urlopen(f"https://huggingface.co/api/models/{repo}/revision/main", timeout=10) as r:
            return {"revision": json.load(r)["sha"], "from": "hub main at run time"}
    except Exception:
        return None
def files(k, model):
    if env.get(f"R_{k}"):                       # ollama: the tag's manifest digest
        return {"revision": env[f"R_{k}"], "from": "ollama digest"}
    return revision(model.split(":", 1)[0])     # GGUF engines name repo:file
served = {k: {"model": env[f"M_{k}"], "quantization": env.get(f"Q_{k}") or None,
              "files": files(k, env[f"M_{k}"])}
          for k in keys if env.get(f"M_{k}")}
d["models"].append({
    "stem": stem, "model_id": mid, "model_files": revision(mid), "quant": quant or None, "features": feats,
    "built": built == "1",
    # null = gate not run (no mlx-lm, or no build); false = gate failed, nothing timed.
    "parity_mlx_lm": (parity_ok == "1") if (mlx_python and built == "1") else None,
    "footprint": {"build_seconds": int(secs) if secs else None,
                  "binary_bytes": int(size) if size else None,
                  "container_image_bytes": None,
                  "note": "no container image: Metal is not available in Linux containers"},
    "cache_ladder": load(f"exec-{stem}.json"),
    "warm_serving": metrics(load(f"serve-{stem}.json")),
    "scaling": scaling(""),
    "engine_models": served,
    # The shape this model ran at, when a limit made it differ from config.scaling.
    "scaling_limits": None if (env["BASE_CONC"] == env["SCALE_BASE_CONC"] and env["CONC_LIST"] == env["SCALE_CONC"])
                      else {"base_conc": int(env["BASE_CONC"]),
                            "conc": [int(x) for x in env["CONC_LIST"].split(",")]},
    **engines,
})
json.dump(d, open(js, "w"), indent=2)
PY
    echo "    recorded"
    summary "${stem}"
done

echo; echo "================================================================"
summary

# A model scratchy never served must fail the run, not sit in the table as a
# row of dashes: a preset that matches no checkpoint looks exactly like that.
python3 - "${JSON}" <<'PY'
import json, sys
d = json.load(open(sys.argv[1]))
bad = [e["stem"] for e in d["models"] if not e["built"]
       or (not e.get("cache_ladder") and not e.get("warm_serving") and not e.get("scaling"))]
if bad:
    why = {e["stem"]: " (parity gate failed)" for e in d["models"] if e.get("parity_mlx_lm") is False}
    print(f"\nFAILED: scratchy produced no numbers for {', '.join(b + why.get(b, '') for b in bad)} "
          "(results still in the json)", file=sys.stderr)
    sys.exit(1)
PY
