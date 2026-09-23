#!/usr/bin/env bash
# Startup-latency comparison: scratchy (`scr`) vs mlx-lm, across a cache
# ladder of three scenarios. Companion to `bench_serve_compare.sh`, which
# already covers STEADY-STATE serving; this script covers the axis nothing
# measured: how long from `exec` until the user sees a word.
#
# WHY A NEW HARNESS INSTEAD OF `scr bench startup`:
#   `bench startup` rebuilds an `LLM` in-process and calls it "cold" -- its
#   own comment (crates/benches/src/startup.rs:80) admits the HF cache is
#   never wiped. Being in-process it cannot see exec, dyld, or per-process
#   Metal pipeline compilation at all. And `chat --bench` reports a TTFT
#   measured *after* startup, so its two numbers never add up to what a user
#   feels. Every primary number here is taken by ONE external stopwatch
#   (scripts/startup_probe.py), the same code for both backends.
#
# THE CACHE LADDER (this is the whole point):
#   surface                              FROZEN    COLD     WARM
#   OS page cache (weights/binary/dylib) purged    warm     warm
#   ~/.cache/scratchy/metal-aligned-*    REMOVED   present  present
#   mlx-lm __pycache__                   REMOVED   present  present
#   HF snapshot on disk                  present   present  present
#   process                              exec      exec     resident, >=1 req
#
#   Both FROZEN and COLD are reported because scratchy can keep an
#   aligned-weights sidecar that MLX has no equivalent of, and we must not
#   charge it per-launch for a one-time cost nor hide that cost entirely.
#
#   MEASURED, and it corrects the obvious assumption: for
#   Llama-3.2-3B-Instruct-4bit that sidecar is NEITHER needed NOR rebuilt.
#   scratchy writes it only from the realign-COPY path, and this checkpoint
#   loads 648/648 tensors zero-copy directly from the HF mmap. So for THIS
#   model FROZEN and COLD differ by page-cache state alone, for both
#   backends, and the sidecar rung is inert. It is kept because it is not
#   inert for checkpoints that do need realignment -- the code cites
#   Qwen3.5-35B (18.99 GiB), where a warm relaunch pays ~3.5s of residency
#   wiring instead of an ~8s copy. Re-check this per model rather than
#   assuming either way.
#
# FAIRNESS RULES (1-3 inherited from bench_serve_compare.sh):
#   1. UNIQUE PROMPT PER WARM REQUEST. A shared prompt lets scratchy's prefix
#      cache (and mlx-lm's prompt cache) serve the repeat and TTFT collapses
#      to ~0. Prompts are seeded: identical across backends, unique per rep.
#   2. mlx-lm IGNORES `ignore_eos` and under-generates. Compare TTFT/TPOT
#      directly; totals only via E2E* = ttft_exec + (out-1)*tpot. The summary
#      prints generated tokens per request so under-generation stays visible.
#   3. ONE MODEL RESIDENT AT A TIME (32 GiB box). Backends run serially and
#      the probe refuses to start against an already-serving port.
#   4. SAME WEIGHTS. Both read the same mlx-community 4-bit checkpoint;
#      scratchy must be built with the matching `quant/mlx-affine-b4-g64`
#      (verified against the checkpoint's {"group_size":64,"bits":4}).
#   5. PARITY GATE, BLOCKING. Both backends must produce identical text on
#      short high-confidence prompts (a broken dequant can be fast, so a
#      timing run on a wrong load path is worse than no run). Divergence deep
#      inside long open-ended generations is recorded but NOT gated: greedy
#      argmax legitimately splits at near-ties between two int4 kernel stacks.
#   6. ABBA INTERLEAVING across reps, so thermal drift does not accrue to
#      whichever backend always runs second.
#   7. HF_HUB_OFFLINE=1 for both. Left online, mlx-lm makes a hub round trip
#      on every launch ("Fetching 6 files") and network jitter lands inside
#      the measurement.
#   8. `sudo purge` is symmetric: it evicts CPython and the MLX dylibs just as
#      it evicts the scratchy binary. That is the honest FROZEN cost.
#   9. PIN THE SAMPLING PARAMS EXPLICITLY. Every request is greedy
#      (temperature 0) on both backends and from both clients. Leaving
#      temperature unset is the trap: `scr bench serve` then omits the field
#      and each SERVER applies its own default, so the backends are timed on
#      different sampling paths. Measured on scratchy, greedy vs sampled is
#      12.96 vs 15.81 ms/token and 44.3 vs 71.2 ms TTFT -- larger than most
#      differences anyone would report. Greedy also matches the parity gate,
#      so what is timed is what was verified.
#
# NOT MEASURED, DISCLOSED: weight *download* (out of scope -- the snapshot is
# present in all three scenarios) and scratchy's build-time `#[forward]`
# expansion. The latter is recorded as a footnote because scratchy moves
# model->code work to build time, and a startup benchmark that silently
# benefits from that without saying so is not one worth defending.
#
# Usage:
#   scripts/bench_startup_compare.sh --mlx-python /tmp/mlxbench/bin/python
#   scripts/bench_startup_compare.sh --scenarios cold --modes server --reps 1
#
# Requires a build with `serve` + `bench` + the matching quant preset:
#   cargo build --release -p scratchy-cli \
#     --features metal,serve,bench,model/llama-3.2-3b,quant/mlx-affine-b4-g64

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." &>/dev/null && pwd)"

MODEL="mlx-community/Llama-3.2-3B-Instruct-4bit"
SCR_BIN="${REPO_DIR}/target/release/scr"
MLX_PYTHON="${MLX_PYTHON:-}"
BACKENDS="scratchy,mlx-lm"
SCENARIOS="frozen,cold,warm"
MODES="cli,server"
REPS_FROZEN=3
REPS_COLD=5
WARM_REQUESTS=20
BENCH_NUM_PROMPTS=20
BENCH_CONCURRENCY=1
INPUT_LEN=64
LONG_INPUT_LEN=2048
OUTPUT_LEN=32
PORT=8731
SEED_BASE=1000
SETTLE_S=8
DEVICE=metal
KV_CACHE_DTYPE=""
OUT_DIR=""
LABEL=""
DO_PURGE=1
DO_LONG=1
BUILD_SECS=""   # disclosed as a footnote; see rule "NOT MEASURED, DISCLOSED"
SIDECAR="${XDG_CACHE_HOME:-${HOME}/.cache}/scratchy/metal-aligned-weights"
PRIMED=0

usage() { sed -n '1,80p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL="$2"; shift 2;;
        --scr-bin) SCR_BIN="$2"; shift 2;;
        --mlx-python) MLX_PYTHON="$2"; shift 2;;
        --backends) BACKENDS="$2"; shift 2;;
        --scenarios) SCENARIOS="$2"; shift 2;;
        --modes) MODES="$2"; shift 2;;
        --reps) REPS_FROZEN="$2"; REPS_COLD="$2"; shift 2;;
        --reps-frozen) REPS_FROZEN="$2"; shift 2;;
        --reps-cold) REPS_COLD="$2"; shift 2;;
        --warm-requests) WARM_REQUESTS="$2"; shift 2;;
        --bench-num-prompts) BENCH_NUM_PROMPTS="$2"; shift 2;;
        --bench-concurrency) BENCH_CONCURRENCY="$2"; shift 2;;
        --input-len) INPUT_LEN="$2"; shift 2;;
        --long-input-len) LONG_INPUT_LEN="$2"; shift 2;;
        --output-len) OUTPUT_LEN="$2"; shift 2;;
        --port) PORT="$2"; shift 2;;
        --settle-s) SETTLE_S="$2"; shift 2;;
        --device) DEVICE="$2"; shift 2;;
        --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2;;
        --out-dir) OUT_DIR="$2"; shift 2;;
        --label) LABEL="$2"; shift 2;;
        --build-secs) BUILD_SECS="$2"; shift 2;;
        --no-purge) DO_PURGE=0; shift;;
        --no-long) DO_LONG=0; shift;;
        -h|--help) usage;;
        *) echo "unknown flag: $1" >&2; exit 2;;
    esac
done

PROBE="${SCRIPT_DIR}/startup_probe.py"
[[ -x "${SCR_BIN}" ]] || { echo "no scratchy binary at ${SCR_BIN}" >&2; exit 1; }
[[ -f "${PROBE}" ]] || { echo "missing ${PROBE}" >&2; exit 1; }

if [[ -z "${MLX_PYTHON}" ]]; then
    for cand in /tmp/mlxbench/bin/python "$(command -v python3 || true)"; do
        [[ -x "${cand}" ]] && "${cand}" -c "import mlx_lm" 2>/dev/null && {
            MLX_PYTHON="${cand}"; break; }
    done
fi
if [[ ",${BACKENDS}," == *",mlx-lm,"* ]]; then
    [[ -n "${MLX_PYTHON}" ]] || { echo "no python with mlx_lm; pass --mlx-python" >&2; exit 1; }
    "${MLX_PYTHON}" -c "import mlx_lm" || exit 1
fi

TS="$(date +%Y%m%d_%H%M%S)"
[[ -n "${OUT_DIR}" ]] || OUT_DIR="${REPO_DIR}/bench_results/startup_compare/${TS}${LABEL:+_${LABEL}}"
mkdir -p "${OUT_DIR}"
echo "results -> ${OUT_DIR}"

# ---- provenance: a number without its machine state is not a result --------
{
    echo "model: ${MODEL}"
    echo "host: $(sysctl -n hw.model) / $(sysctl -n machdep.cpu.brand_string) / $(( $(sysctl -n hw.memsize) / 1073741824 )) GiB"
    echo "macos: $(sw_vers -productVersion)"
    echo "git: $(git -C "${REPO_DIR}" rev-parse --short HEAD 2>/dev/null || echo n/a) $(git -C "${REPO_DIR}" diff --quiet 2>/dev/null && echo clean || echo DIRTY)"
    echo "scr: ${SCR_BIN}"
    echo "mlx: $("${MLX_PYTHON}" -c 'import mlx.core as m,mlx_lm;print("mlx",m.__version__,"mlx_lm",mlx_lm.__version__)' 2>/dev/null || echo n/a)"
    echo "thermal: $(pmset -g therm 2>/dev/null | tr '\n' ' ' || echo n/a)"
    echo "power: $(pmset -g ps 2>/dev/null | head -1)"
    echo "scenarios: ${SCENARIOS} | modes: ${MODES} | backends: ${BACKENDS}"
    echo "input_len: ${INPUT_LEN} (long ${LONG_INPUT_LEN}) | output_len: ${OUTPUT_LEN}"
    echo "kv_cache_dtype: ${KV_CACHE_DTYPE:-default(TurboQuant 3-bit on metal)}"
    echo "purge: ${DO_PURGE}"
    if [[ -n "${BUILD_SECS}" ]]; then echo "build_secs: ${BUILD_SECS}"; fi
} > "${OUT_DIR}/run_meta.txt"
cat "${OUT_DIR}/run_meta.txt"

# Warn loudly on thermal pressure rather than silently benching a hot box.
if pmset -g therm 2>/dev/null | grep -qE "CPU_Speed_Limit *= *[0-9]{1,2}$"; then
    echo "WARNING: CPU speed limit below 100 -- box is thermally throttled" >&2
fi

if [[ ",${SCENARIOS}," == *",frozen,"* && "${DO_PURGE}" -eq 1 ]]; then
    echo "frozen scenario needs 'sudo purge'; authorizing once up front..."
    sudo -v || { echo "sudo unavailable; rerun with --no-purge" >&2; exit 1; }
fi

# ---- cache-state control ---------------------------------------------------
mlx_pycache_clear() {
    local sp
    sp="$("${MLX_PYTHON}" -c 'import mlx_lm,os;print(os.path.dirname(os.path.dirname(mlx_lm.__file__)))' 2>/dev/null || true)"
    [[ -n "${sp}" && -d "${sp}" ]] || return 0
    find "${sp}/mlx_lm" "${sp}/mlx" -name __pycache__ -type d -prune -exec rm -rf {} + 2>/dev/null || true
}

prepare_state() {
    # $1 = scenario, $2 = backend
    case "$1" in
        frozen)
            # scratchy's derived on-disk cache: the whole point of this rung.
            rm -rf "${SIDECAR}"
            mlx_pycache_clear
            # Drop the unified buffer cache last, so the removals above do not
            # repopulate it. purge is synchronous.
            # NOTE: written as `if`, not `[[ ]] && cmd`. As the last command in
            # this function a false test would return 1 and `set -e` would
            # abort the whole run -- which it did, on the first --no-purge rep.
            if [[ "${DO_PURGE}" -eq 1 ]]; then
                sudo purge
            fi
            ;;
        cold|warm)
            # Both caches present and page cache warm. If a frozen rep just
            # ran, the sidecar is gone, so prime it with a throwaway launch --
            # COLD must be a genuine second launch, not a disguised frozen.
            #
            # MEASURED CAVEAT: scratchy writes that sidecar only from the
            # realign-COPY path (metal_allocator.rs:1372). A checkpoint whose
            # tensors already satisfy the bind alignment loads fully zero-copy
            # from the HF mmap (648/648 for Llama-3.2-3B-4bit) and therefore
            # NEVER recreates it. So prime at most once per run and carry on
            # if it stays absent -- otherwise the old `! -d` test fires a
            # wasted launch before every rep, forever.
            if [[ "$2" == "scratchy" && ! -d "${SIDECAR}" && "${PRIMED}" -eq 0 ]]; then
                echo "  (priming scratchy sidecar cache for a true COLD state)"
                HF_HUB_OFFLINE=1 RUST_LOG=error "${SCR_BIN}" chat -m "${MODEL}" \
                    --device "${DEVICE}" -q "hi" --max-tokens 1 --temperature 0 \
                    >/dev/null 2>&1 || true
                PRIMED=1
                if [[ ! -d "${SIDECAR}" ]]; then
                    echo "  note: no sidecar was written -- this checkpoint"
                    echo "  loads zero-copy without one, so FROZEN and COLD"
                    echo "  differ by page-cache state only for scratchy."
                fi
            fi
            ;;
    esac
}

# ---- parity gate (rule 5, blocking) ---------------------------------------
# Delegated to startup_parity.py, which enforces exact agreement on short
# high-confidence prompts and merely records drift on open-ended ones. See
# that file for why those are different questions.
parity_gate() {
    echo
    echo "=== parity gate: same model, greedy, high-confidence prompts ==="
    python3 "${SCRIPT_DIR}/startup_parity.py" \
        --model "${MODEL}" --scr-bin "${SCR_BIN}" \
        --mlx-python "${MLX_PYTHON}" --device "${DEVICE}" \
        --out-dir "${OUT_DIR}/parity"
}

# ---- one rep --------------------------------------------------------------
run_rep() {
    local backend="$1" scenario="$2" mode="$3" rep="$4" ilen="$5" tag="$6"
    local stem="${backend}.${scenario}.${mode}${tag}.${rep}"
    # bash 3.2 (macOS default) errors on "${empty[@]}" under `set -u`, hence
    # the ${arr[@]+...} guard at the call site below.
    local extra=()
    if [[ -n "${KV_CACHE_DTYPE}" ]]; then
        extra+=(--kv-cache-dtype "${KV_CACHE_DTYPE}")
    fi
    if [[ "${scenario}" == "warm" ]]; then
        extra+=(--warm-requests "${WARM_REQUESTS}" --settle-s "${SETTLE_S}")
        # WARM steady state is measured by `scr bench serve` -- the repo's own
        # backend-agnostic HTTP client, the same one bench_serve_compare.sh
        # points at both backends. scratchy's binary is used purely as a load
        # generator here, including when the server under test is mlx-lm.
        extra+=(--bench-serve-bin "${SCR_BIN}"
                --bench-num-prompts "${BENCH_NUM_PROMPTS}"
                --bench-concurrency "${BENCH_CONCURRENCY}")
    fi
    echo
    echo "--- ${stem} (input_len=${ilen}) ---"
    prepare_state "${scenario}" "${backend}"
    python3 "${PROBE}" \
        --backend "${backend}" --mode "${mode}" --model "${MODEL}" \
        --scr-bin "${SCR_BIN}" --mlx-python "${MLX_PYTHON}" \
        --device "${DEVICE}" --port "${PORT}" \
        --input-len "${ilen}" --output-len "${OUTPUT_LEN}" \
        --seed "$(( SEED_BASE + rep * 17 + ilen ))" \
        --scenario "${scenario}" --rep "${rep}" \
        --out "${OUT_DIR}/${stem}.json" --log "${OUT_DIR}/${stem}.log" \
        ${extra[@]+"${extra[@]}"} || echo "  rep failed; continuing" >&2
}

# ---- run ------------------------------------------------------------------
IFS=',' read -r -a backend_arr <<<"${BACKENDS}"
IFS=',' read -r -a scenario_arr <<<"${SCENARIOS}"
IFS=',' read -r -a mode_arr <<<"${MODES}"

parity_gate

for scenario in "${scenario_arr[@]}"; do
    case "${scenario}" in
        frozen) reps="${REPS_FROZEN}";;
        cold) reps="${REPS_COLD}";;
        warm) reps=1;;
        *) echo "unknown scenario ${scenario}" >&2; continue;;
    esac
    for mode in "${mode_arr[@]}"; do
        # WARM means "process already resident", which only the server mode
        # can express -- a one-shot CLI exits after every request.
        [[ "${scenario}" == "warm" && "${mode}" == "cli" ]] && continue
        for (( rep=0; rep<reps; rep++ )); do
            # Rule 6: ABBA. Reverse backend order on odd reps.
            order=("${backend_arr[@]}")
            if (( rep % 2 == 1 )); then
                order=()
                for (( i=${#backend_arr[@]}-1; i>=0; i-- )); do
                    order+=("${backend_arr[i]}")
                done
            fi
            for backend in "${order[@]}"; do
                run_rep "${backend}" "${scenario}" "${mode}" "${rep}" "${INPUT_LEN}" ""
            done
        done
        # One long-prompt cell per scenario/mode: shows prefill scaling from a
        # cold start, where weight load and prefill compete.
        if [[ "${DO_LONG}" -eq 1 && "${scenario}" != "warm" ]]; then
            for backend in "${backend_arr[@]}"; do
                run_rep "${backend}" "${scenario}" "${mode}" 0 "${LONG_INPUT_LEN}" ".long"
            done
        fi
    done
done

echo
echo "=== summarizing ==="
python3 "${SCRIPT_DIR}/startup_summary.py" "${OUT_DIR}" > "${OUT_DIR}/summary.md"
echo "summary -> ${OUT_DIR}/summary.md"
cat "${OUT_DIR}/summary.md"
