#!/usr/bin/env bash
# Metal benchmark matrix — one Mac in, one JSON out.
#
#     scripts/bench_metal_matrix.sh                 # every model, every rung
#     scripts/bench_metal_matrix.sh --models llama-3.2-3b
#     scripts/bench_metal_matrix.sh --scenarios cold,warm   # no sudo needed
#
# It is a RUNNER, not a measurement tool. Every number comes from a `scr bench`
# subcommand; this script only decides what to run, in what order, and merges the
# results into one file. Two consequences worth knowing:
#
#   * It never starts or stops a server. `scr bench startup --exec` spawns and
#     reaps its own children, so this script cannot leak one.
#   * It adds exactly one measurement of its own: BUILD TIME and BINARY SIZE per model. scratchy
#     compiles one model into the binary, so size is a per-model fact, and the
#     build is the cost side of doing that work ahead of time. Every startup
#     number below excludes it, which is precisely why it has to be reported.
#
# WHAT YOU GET, AND WHICH COMMAND PRODUCES IT
#
#   build seconds, binary MiB          this script (`cargo build`, `stat`)
#   ttft_exec: frozen / cold / warm    scr bench startup --exec
#   t_ready, ttft_from_send, tpot      scr bench startup --exec
#   peak RSS, major faults             scr bench startup --exec  (per child, wait4)
#   TTFT/TPOT/ITL p50+p99, tok/s       scr bench serve
#
# The cache ladder, the eviction proof, the priming launch, per-child resource
# attribution and the validity gates all live in `--exec`. See
# docs/BENCHMARKING.md for what each rung means.
#
# REQUIREMENTS
#   * `--scenarios frozen,...` needs `sudo` (macOS `purge`). Run `sudo -v` first,
#     or pass `--scenarios cold,warm` and leave the frozen cells blank.
#   * Weights are downloaded on first use unless you pass --offline.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "${HERE}/.." &>/dev/null && pwd)"

# Every id verified against the HuggingFace API with its download size
MODELS_DEFAULT=(
  "granite-3.3-2b-instruct=mlx-community/granite-3.3-2b-instruct-4bit:mlx-affine-b4-g64"  #  1.4 GB
  "llama-3.2-3b=mlx-community/Llama-3.2-3B-Instruct-4bit:mlx-affine-b4-g64"               #  1.8 GB
  "granite-3.3-8b-instruct=mlx-community/granite-3.3-8b-instruct-4bit:mlx-affine-b4-g64"  #  4.6 GB
  "qwen2.5-7b=mlx-community/Qwen2.5-7B-Instruct-4bit:mlx-affine-b4-g64"                   #  4.3 GB
  "gemma-4-26b-a4b-it=mlx-community/gemma-4-26b-a4b-it-4bit:mlx-affine-b4-g64"            # 15.4 GB
  "gemma-4-31b-it=mlx-community/gemma-4-31b-it-4bit:mlx-affine-b4-g64"                    # 18.4 GB
  "qwen3.5-35b-a3b=mlx-community/Qwen3.5-35B-A3B-4bit:mlx-affine-b4-g64"                  # 20.4 GB
  # Frontier-ish: Moonlight is a DeepSeek-V3-architecture MoE (16B total, ~3B
  # active), so it exercises the MLA + MoE path at a size a 36 GB machine can
  # host. Listed twice on purpose — same model, two quantizations — which is the
  # only quantization axis in this matrix, since the gemma4 and qwen3.5 families
  # declare mlx-affine-b4-g64 and nothing else.
  "moonlight-16b-a3b-instruct=mlx-community/Moonlight-16B-A3B-Instruct-4-bit:mlx-affine-b4-g64"        #  9.0 GB
  "moonlight-16b-a3b-instruct-fp8-block=starpit/moonlight-16b-a3b-instruct-fp8-block:fp8-block-128x128" # 16.7 GB
  # DeepSeek-V2-Lite has no public 4-bit MLX build, so it would be bf16 at
  # 31.4 GB and exhaust a 36 GB machine once the KV cache is allocated. The
  # DeepSeek-V2 arch is instead reachable via
  # mlx-community/DeepSeek-Coder-V2-Lite-Instruct-4bit-mlx (8.8 GB) if wanted.
)

MODELS=()
SCENARIOS="frozen,cold,warm"
REPS=3
PORT=8751
NUM_PROMPTS=20
INPUT_LEN=64
OUTPUT_LEN=32
WARM_REQUESTS=20
OFFLINE=0
SKIP_BUILD=0
OUT_DIR=""
KV_CACHE_DTYPE=""
# `--exec` picks purge on macOS and fadvise on Linux by itself
EVICT=""
EVICT_PATH=()
SETTLE_S=""
# `--exec` defaults to 600 s, too tight for a ~20 GB checkpoint on this class of
# machine: gemma-4-31b-it (18.4 GB) timed out at 600 s.
READY_TIMEOUT_S="1800"
SEED=""
# Setting --mlx-python turns on the comparison column: the same ladder is run a
# second time with --backend mlx-lm.
MLX_PYTHON=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --models)         IFS=',' read -r -a MODELS <<<"$2"; shift 2 ;;
        --scenarios)      SCENARIOS="$2"; shift 2 ;;
        --reps)           REPS="$2"; shift 2 ;;
        --port)           PORT="$2"; shift 2 ;;
        --num-prompts)    NUM_PROMPTS="$2"; shift 2 ;;
        --input-len)      INPUT_LEN="$2"; shift 2 ;;
        --output-len)     OUTPUT_LEN="$2"; shift 2 ;;
        --warm-requests)  WARM_REQUESTS="$2"; shift 2 ;;
        --kv-cache-dtype) KV_CACHE_DTYPE="$2"; shift 2 ;;
        --evict)          EVICT="$2"; shift 2 ;;
        --evict-path)     EVICT_PATH+=("$2"); shift 2 ;;
        --settle-s)       SETTLE_S="$2"; shift 2 ;;
        --ready-timeout-s) READY_TIMEOUT_S="$2"; shift 2 ;;
        --seed)           SEED="$2"; shift 2 ;;
        --mlx-python)     MLX_PYTHON="$2"; shift 2 ;;
        --offline)        OFFLINE=1; shift ;;
        --skip-build)     SKIP_BUILD=1; shift ;;
        --out-dir)        OUT_DIR="$2"; shift 2 ;;
        -h|--help)        grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)                echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ ${#MODELS[@]} -eq 0 ]] && MODELS=("${MODELS_DEFAULT[@]}")
(( OFFLINE )) && export HF_HUB_OFFLINE=1

BIN="${ROOT}/target/release/scr"
chip="$(sysctl -n machdep.cpu.brand_string)"
slug="$(echo "${chip}" | tr '[:upper:] ' '[:lower:]-' | sed 's/[^a-z0-9-]//g')"
: "${OUT_DIR:="${ROOT}/bench_results/metal_matrix"}"
RAW="${OUT_DIR}/${slug}"
mkdir -p "${RAW}"
JSON="${OUT_DIR}/${slug}.json"

# ---- preflight: fail now, with a fix, rather than three models in ------------
die() { echo "error: $*" >&2; exit 1; }

command -v cargo >/dev/null || die "cargo not on PATH"
[[ "$(uname -s)" == "Darwin" ]] || die "this runner is the metal side of #3; use the cuda/spyre runner elsewhere"

# `--exec` is what measures the cache ladder.
EXEC_OK=0
if [[ -x "${BIN}" ]] && "${BIN}" bench startup --help 2>/dev/null | grep -q -- '--exec'; then
    EXEC_OK=1
fi
if [[ ",${SCENARIOS}," == *",frozen,"* ]]; then
    sudo -n true 2>/dev/null || die "frozen needs sudo for \`purge\`: run \`sudo -v\` first, or pass --scenarios cold,warm"
fi

echo "machine : ${chip}"
echo "models  : ${#MODELS[@]}"
echo "rungs   : ${SCENARIOS}"
echo "output  : ${JSON}"
(( EXEC_OK )) || cat <<EOF

EOF

# ---- machine block ----------------------------------------------------------
python3 - "${JSON}" "${chip}" "${SCENARIOS}" "${EXEC_OK}" <<'PY'
import json, subprocess, sys, time
out, chip, scenarios, exec_ok = sys.argv[1:5]
sh = lambda *c: subprocess.run(c, capture_output=True, text=True).stdout.strip()
sysctl = lambda k: sh("sysctl", "-n", k)
batt = sh("pmset", "-g", "batt")
json.dump({
  "schema": 2, "issue": 91, "epic": 3,
  "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
  "generator": "scripts/bench_metal_matrix.sh",
  "measured_by": {
    "footprint": "this runner (cargo build, stat)",
    "cache_ladder": "scr bench startup --exec" if exec_ok == "1" else "UNAVAILABLE",
    "warm_serving": "scr bench serve",
  },
  "machine": {
    "hw_model": sysctl("hw.model"), "chip": chip,
    "cores_total": int(sysctl("hw.ncpu") or 0),
    "cores_performance": int(sysctl("hw.perflevel0.physicalcpu") or 0),
    "cores_efficiency": int(sysctl("hw.perflevel1.physicalcpu") or 0),
    "memory_gb": round(int(sysctl("hw.memsize") or 0) / 1024**3),
    "macos": sh("sw_vers", "-productVersion") + " (" + sh("sw_vers", "-buildVersion") + ")",
    "thermal_at_start": sh("pmset", "-g", "therm").replace("\n", " "),
    "power": batt.splitlines()[0] if batt else "",
  },
  "repo": {"sha": sh("git", "rev-parse", "HEAD"),
           "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
           "dirty": bool(sh("git", "status", "--porcelain"))},
  "config": {"scenarios": scenarios.split(",")},
  "methodology": "docs/BENCHMARKING.md — rung definitions, fairness rules and disclosed asymmetries live there, not here",
  "models": [],
}, open(out, "w"), indent=2)
PY

# ---- per model --------------------------------------------------------------
for entry in "${MODELS[@]}"; do
    stem="${entry%%=*}"; rest="${entry#*=}"
    id="${rest%%:*}"; quant="${rest#*:}"; [[ "${quant}" == "${rest}" ]] && quant=""
    echo; echo "################ ${stem} ################"

    feats="metal,serve,bench,model/${stem}"
    [[ -n "${quant}" ]] && feats="${feats},quant/${quant}"
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
        fi
    fi

    # Each model needs its own binary: the next build overwrites this one.
    model_bin="${RAW}/scr-${stem}"
    (( built )) && cp "${BIN}" "${model_bin}"

    exec_json="${RAW}/exec-${stem}.json"
    exec_mlx_json="${RAW}/exec-mlx-${stem}.json"
    serve_json="${RAW}/serve-${stem}.json"
    rm -f "${exec_json}" "${exec_mlx_json}" "${serve_json}"

    if (( built && EXEC_OK )); then
        serve_cmd="${model_bin} serve ${id} --port ${PORT}"
        [[ -n "${KV_CACHE_DTYPE}" ]] && serve_cmd="${serve_cmd} --kv-cache-dtype ${KV_CACHE_DTYPE}"
        mlx_cmd=""
        [[ -n "${MLX_PYTHON}" ]] && mlx_cmd="${MLX_PYTHON} -m mlx_lm.server --model ${id} --port ${PORT}"

        # Flags shared by every --exec invocation for this model.
        common=(--model "${id}" --exec
                --mode server
                --scenarios "${SCENARIOS}"
                --reps "${REPS}"
                --port "${PORT}"
                --input-len "${INPUT_LEN}"
                --output-len "${OUTPUT_LEN}"
                --warm-requests "${WARM_REQUESTS}"
                --remove-path "${HOME}/.cache/scratchy/metal-aligned-weights")
        [[ -n "${EVICT}" ]]    && common+=(--evict "${EVICT}")
        [[ -n "${SETTLE_S}" ]] && common+=(--settle-s "${SETTLE_S}")
        [[ -n "${READY_TIMEOUT_S}" ]] && common+=(--ready-timeout-s "${READY_TIMEOUT_S}")
        [[ -n "${SEED}" ]]     && common+=(--seed "${SEED}")
        for ep in "${EVICT_PATH[@]:-}"; do [[ -n "${ep}" ]] && common+=(--evict-path "${ep}"); done

        echo "--- scr bench startup --exec, backend scratchy (${SCENARIOS})"
        # --remove-path clears the derived aligned-weights sidecar, which a FROZEN
        # rung must not find warm. --exec spawns and reaps its own children.
        parity=()
        if [[ -n "${mlx_cmd}" ]]; then
            # Blocking: a broken dequant path can be fast and wrong, so the two
            # engines must agree on the same greedy prompts before either is timed.
            parity=(--parity-cmd "${mlx_cmd}" --parity-backend mlx-lm)
        fi
        "${BIN}" bench startup "${common[@]}" \
            --child-cmd "${serve_cmd}" \
            --backend scratchy \
            ${parity[@]+"${parity[@]}"} \
            --output-json "${exec_json}" \
            2>&1 | tail -8 | sed 's/^/    /' \
            || echo "    --exec returned non-zero (validity gate, or a real failure); see above" >&2

        if [[ -n "${mlx_cmd}" ]]; then
            echo "--- scr bench startup --exec, backend mlx-lm (comparison column)"
            "${BIN}" bench startup "${common[@]}" \
                --child-cmd "${mlx_cmd}" \
                --backend mlx-lm \
                --output-json "${exec_mlx_json}" \
                2>&1 | tail -6 | sed 's/^/    /' \
                || echo "    mlx-lm --exec returned non-zero; see above" >&2
        fi
    fi

    if (( built )); then
        echo "--- scr bench serve (warm steady state, greedy, conc 1)"
        # One resident server for this stage only; --exec owns the ladder above.
        "${model_bin}" serve "${id}" --port "${PORT}" >"${RAW}/serve-${stem}.log" 2>&1 &
        sp=$!
        ready=0
        deadline=$(( $(date +%s) + 1800 ))
        while (( $(date +%s) < deadline )); do
            curl -fsS -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && { ready=1; break; }
            kill -0 "${sp}" 2>/dev/null || break
            sleep 1
        done
        if (( ready )); then
            "${BIN}" bench serve --base-url "http://127.0.0.1:${PORT}" --model "${id}" \
                --num-prompts "${NUM_PROMPTS}" --input-len "${INPUT_LEN}" --output-len "${OUTPUT_LEN}" \
                --max-concurrency 1 --temperature 0 --seed "${RANDOM}${RANDOM}" \
                --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
                --output-json "${serve_json}" --disable-tqdm 2>&1 | tail -4 | sed 's/^/    /' \
                || echo "    bench serve failed" >&2
        else
            echo "    server never became ready — ${RAW}/serve-${stem}.log" >&2
        fi
        # Kill the whole job, not just the pid, so nothing survives this loop.
        kill -INT "${sp}" 2>/dev/null || true
        for _ in $(seq 1 30); do kill -0 "${sp}" 2>/dev/null || break; sleep 1; done
        kill -KILL "${sp}" 2>/dev/null || true
        wait "${sp}" 2>/dev/null || true
        pkill -f "${model_bin} serve" 2>/dev/null || true
    fi

    python3 - "${JSON}" "${stem}" "${id}" "${quant}" "${feats}" "${built}" \
              "${build_secs}" "${bytes}" "${exec_json}" "${serve_json}" "${exec_mlx_json}" <<'PY'
import json, os, sys
js, stem, mid, quant, feats, built, secs, size, exec_json, serve_json, exec_mlx_json = sys.argv[1:12]
def num(x):
    for cast in (int, float):
        try: return cast(x)
        except Exception: pass
    return None
def load(p):
    return json.load(open(p)) if p and os.path.exists(p) else None
d = json.load(open(js))
warm = None
sj = load(serve_json)
if sj:
    keep = ["median_ttft_ms","p99_ttft_ms","median_tpot_ms","p99_tpot_ms","median_itl_ms",
            "p99_itl_ms","median_e2el_ms","output_throughput","request_throughput",
            "completed","total_output_tokens","duration"]
    warm = {k: sj[k] for k in keep if k in sj}
d["models"].append({
    "stem": stem, "model_id": mid, "quant": quant or None, "features": feats,
    "built": built == "1",
    "footprint": {"build_seconds": num(secs), "binary_bytes": num(size),
                  "container_image_bytes": None,
                  "note": "no container image: Metal is not available in Linux containers"},
    "cache_ladder": load(exec_json),
    "cache_ladder_mlx_lm": load(exec_mlx_json),
    "warm_serving": warm,
})
json.dump(d, open(js, "w"), indent=2)
PY
    echo "    recorded"
done

# ---- summary ----------------------------------------------------------------
echo; echo "================================================================"
python3 - "${JSON}" <<'PY'
import json, statistics, sys
d = json.load(open(sys.argv[1]))
m = d["machine"]
print(f"{m['chip']} · {m['cores_total']} cores ({m['cores_performance']}P+{m['cores_efficiency']}E) "
      f"· {m['memory_gb']} GB · macOS {m['macos']}")
print(f"{'model':26}{'build':>7}{'MiB':>6}{'frozen':>9}{'cold':>8}{'TTFT':>8}{'TPOT':>8}{'tok/s':>8}")
def med(reps, scenario, field):
    vals = [r[field] for r in (reps or []) if r.get("scenario") == scenario and r.get(field) is not None]
    return statistics.median(vals) if vals else None
def fmt(v, nd=0):
    return "-" if v is None else f"{v:.{nd}f}"
for e in d["models"]:
    f, ladder, w = e["footprint"], e.get("cache_ladder"), e.get("warm_serving") or {}
    mib = round(f["binary_bytes"] / 1048576) if f["binary_bytes"] else None
    print(f"{e['stem']:26}{fmt(f['build_seconds']):>7}{fmt(mib):>6}"
          f"{fmt(med(ladder,'frozen','ttft_exec_s'), 2):>9}{fmt(med(ladder,'cold','ttft_exec_s'), 2):>8}"
          f"{fmt(w.get('median_ttft_ms')):>8}{fmt(w.get('median_tpot_ms'), 1):>8}"
          f"{fmt(w.get('output_throughput'), 1):>8}")
    if not e["built"]:
        print("    BUILD FAILED — see the build log in this machine's raw/ directory")
    elif ladder is None:
        print("    cache ladder unavailable")
print(f"\nfrozen/cold are ttft_exec seconds (exec -> first token); TTFT/TPOT are warm ms.")
print(f"json -> {sys.argv[1]}")
print("paste this file into issue #91; fill the table in #95 from it")
PY
