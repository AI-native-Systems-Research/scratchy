#!/usr/bin/env bash
# Metal benchmark matrix — one Mac in, one JSON out.
#
#     scripts/bench_metal_matrix.sh                 # the three default models
#     scripts/bench_metal_matrix.sh --models llama-3.2-3b,qwen2.5-7b   # any from MODELS_ALL
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
# Needs: `sudo -v` first for the frozen rung (macOS `purge`), or pass
# --scenarios cold,warm. Weights download on first use unless --offline.
#
# Scaling (--no-scaling skips it; --scale-axes picks from conc,input,output,grid):
# one seed per (model, axis, rung) — unique so the prefix cache cannot serve a
# later cell, shared across engines so all see the same prompts. Concurrency is
# offered (--max-concurrency), not the effective decode batch.
#
# Comparison engines, each on when installed (--no-mlx / --no-ollama):
#   mlx-lm  python with `import mlx_lm` ($VIRTUAL_ENV, python3, or --mlx-python)
#   ollama  its own `ollama serve` on --port (a desktop one on 11434 is left
#           alone); pulls on first use. GGUF Q4_K_M weights, ignores ignore_eos,
#           RSS misses the runner grandchild; restarted per axis with
#           OLLAMA_NUM_PARALLEL / OLLAMA_CONTEXT_LENGTH sized for that axis.
# With mlx-lm, a blocking parity gate runs first: `scr chat` vs mlx_lm.generate,
# greedy, CLI mode (the ladder is server mode, so the gate is its own call). A
# failure skips every timing stage (all engines) for that model and is recorded as
# "parity_mlx_lm": false.
#
# --serve-args "..." adds extra `scr serve` flags (recorded in the JSON). None
# by default: `scr serve` sizes --max-num-batched-tokens to the largest resident
# prefill bucket, the same as any user gets.
# --cell-timeout-s bounds each scaling cell; a timed-out cell skips the rest of
# that server's cells, since a hung server would hang them all.
set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
ROOT="$(cd -- "${HERE}/.." &>/dev/null && pwd)"

# Every id verified against the HuggingFace API with its download size
MODELS_ALL=(
  "granite-3.3-2b-instruct=mlx-community/granite-3.3-2b-instruct-4bit:mlx-affine-b4-g64"  #  1.4 GB
  "llama-3.2-3b=mlx-community/Llama-3.2-3B-Instruct-4bit:mlx-affine-b4-g64"               #  1.8 GB
  "granite-3.3-8b-instruct=mlx-community/granite-3.3-8b-instruct-4bit:mlx-affine-b4-g64"  #  4.6 GB
  "qwen2.5-7b=mlx-community/Qwen2.5-7B-Instruct-4bit:mlx-affine-b4-g64-qembed"            #  4.3 GB
  "gemma-4-26b-a4b-it=mlx-community/gemma-4-26b-a4b-it-4bit:mlx-affine-b4-g64"            # 15.4 GB
  "gemma-4-31b-it=mlx-community/gemma-4-31b-it-4bit:mlx-affine-b4-g64"                    # 18.4 GB
  "qwen3.5-35b-a3b=mlx-community/Qwen3.5-35B-A3B-4bit:mlx-affine-b4-g64"                  # 20.4 GB
)
# Small dense, mid dense, large MoE: one of each kind keeps a run overnight.
MODELS_DEFAULT="granite-3.3-2b-instruct,qwen2.5-7b,gemma-4-26b-a4b-it"

# ollama library tag per stem; no tag, no ollama column.
ollama_tag() {
    case "$1" in
        granite-3.3-2b-instruct) echo "granite3.3:2b" ;;
        llama-3.2-3b)            echo "llama3.2:3b" ;;
        granite-3.3-8b-instruct) echo "granite3.3:8b" ;;
        qwen2.5-7b)              echo "qwen2.5:7b" ;;
        gemma-4-26b-a4b-it)      echo "gemma4:26b" ;;
        gemma-4-31b-it)          echo "gemma4:31b" ;;
        qwen3.5-35b-a3b)         echo "qwen3.5:35b-a3b" ;;
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
        --offline)           OFFLINE=1; shift ;;
        --skip-build)        SKIP_BUILD=1; shift ;;
        --out-dir)           OUT_DIR="$2"; shift 2 ;;
        -h|--help)           awk 'NR>1 && !/^#/{exit} NR>1{sub(/^# ?/,""); print}' "$0"; exit 0 ;;
        *)                   echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done
[[ ${#MODELS[@]} -eq 0 ]] && IFS=',' read -r -a MODELS <<<"${MODELS_DEFAULT}"
[[ "${MODELS[*]}" == "all" ]] && MODELS=("${MODELS_ALL[@]}")
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

BIN="${ROOT}/target/release/scr"
chip="$(sysctl -n machdep.cpu.brand_string)"
slug="$(echo "${chip}" | tr '[:upper:] ' '[:lower:]-' | sed 's/[^a-z0-9-]//g')"
: "${OUT_DIR:="${ROOT}/bench_results/metal_matrix"}"
RAW="${OUT_DIR}/${slug}"
# Named <machine>-<date>-<sha8>.json, the name site/build_metal.py requires, so
# a finished run drops straight into site/data/metal/. One timestamp feeds both
# the name and generated_utc, so the two can never disagree.
STARTED_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
JSON="${OUT_DIR}/${slug}-${STARTED_UTC:0:10}-$(git -C "${ROOT}" rev-parse HEAD | cut -c1-8).json"
mkdir -p "${RAW}"

die() { echo "error: $*" >&2; exit 1; }
command -v cargo >/dev/null || die "cargo not on PATH"
[[ "$(uname -s)" == "Darwin" ]] || die "this runner is for Apple Silicon; use the cuda/spyre runner elsewhere"
if [[ ",${SCENARIOS}," == *",frozen,"* ]]; then
    sudo -n true 2>/dev/null || die "frozen needs sudo for \`purge\`: run \`sudo -v\` first, or pass --scenarios cold,warm"
fi

echo "machine : ${chip}"
echo "models  : ${#MODELS[@]}"
echo "rungs   : ${SCENARIOS}"
echo "mlx-lm  : ${MLX_PYTHON:-skipped (pip install mlx-lm, or pass --mlx-python)}"
echo "ollama  : ${OLLAMA_BIN:-skipped (brew install ollama)}"
(( SCALING )) && echo "scaling : ${SCALE_AXES}"
echo "output  : ${JSON}"

# ---- machine block ----------------------------------------------------------
# Launches actually discarded before COLD: the harness always primes once, and
# run_ladder's throwaway call adds the rest (so 2 rounds up to 3).
PRIME_LAUNCHES=$(( PRIME > 2 ? PRIME : (PRIME > 1 ? 3 : 1) ))
export STARTED_UTC PRIME_LAUNCHES INPUT_LEN OUTPUT_LEN WARM_REQUESTS SCENARIOS SCALING SCALE_AXES SCALE_CONC SCALE_INPUT SCALE_OUTPUT SCALE_GRID_INPUT \
       SCALE_GRID_OUTPUT SCALE_BASE_INPUT SCALE_BASE_OUTPUT SCALE_BASE_CONC SCALE_NUM_PROMPTS \
       MLX_PYTHON OLLAMA_BIN KV_CACHE_DTYPE SERVE_ARGS CELL_TIMEOUT_S
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
    "macos": sh("sw_vers", "-productVersion") + " (" + sh("sw_vers", "-buildVersion") + ")",
    "thermal_at_start": sh("pmset", "-g", "therm").replace("\n", " "),
    "power": batt.splitlines()[0] if batt else "",
  },
  "repo": {"sha": sh("git", "rev-parse", "HEAD"),
           "branch": sh("git", "rev-parse", "--abbrev-ref", "HEAD"),
           "dirty": bool(sh("git", "status", "--porcelain"))},
  "config": {"scenarios": e["SCENARIOS"].split(","),
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
             "comparison": {"mlx_lm": bool(e["MLX_PYTHON"]), "ollama": bool(e["OLLAMA_BIN"])}},
  "methodology": "docs/BENCHMARKING.md — rung definitions, fairness rules and disclosed asymmetries live there, not here",
  "models": [],
}, open(out, "w"), indent=2)
PY

# ---- servers ----------------------------------------------------------------
# One server at a time on ${PORT}; the exit trap stops it on failure or Ctrl-C.
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
trap serve_down EXIT
ollama_env() { echo env OLLAMA_HOST="127.0.0.1:${PORT}" OLLAMA_KEEP_ALIVE=-1 OLLAMA_MAX_LOADED_MODELS=1 "$@"; }
ollama_up() { serve_up "${RAW}/serve-ollama-${stem}.log" /api/version $(ollama_env "$@") "${OLLAMA_BIN}" serve; }

max_of() { local m=0 x; IFS=',' read -r -a _xs <<<"$1"; for x in "${_xs[@]}"; do (( x > m )) && m=${x}; done; echo "${m}"; }

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
    # perl's alarm survives exec: SIGALRM ends the client after CELL_TIMEOUT_S.
    # The client's output goes to a log; stderr off hides bash's job notice.
    { perl -e 'alarm shift; exec @ARGV' "${CELL_TIMEOUT_S}" \
        "${BIN}" bench serve --base-url "http://127.0.0.1:${PORT}" --model "${cell_model:-${id}}" \
        ${cell_tok:+--tokenizer "${cell_tok}"} \
        --num-prompts "${SCALE_NUM_PROMPTS}" --input-len "$4" --output-len "$5" \
        --max-concurrency "$6" --temperature 0 --seed "${seed}" --num-warmups "${SCALE_WARMUPS}" \
        --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 50,99 \
        --output-json "${out_json}" --disable-tqdm >>"${RAW}/cells-${stem}.log" 2>&1; } 2>/dev/null || rc=$?
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
        IFS=',' read -r -a xs <<<"${SCALE_CONC}"
        for x in "${xs[@]}"; do run_cell "${prefix}" conc "${x}" "${SCALE_BASE_INPUT}" "${SCALE_BASE_OUTPUT}" "${x}"; done
    fi
    if [[ "${axes}" == *",input,"* ]]; then
        IFS=',' read -r -a xs <<<"${SCALE_INPUT}"
        for x in "${xs[@]}"; do run_cell "${prefix}" input "${x}" "${x}" "${SCALE_BASE_OUTPUT}" "${SCALE_BASE_CONC}"; done
    fi
    if [[ "${axes}" == *",output,"* ]]; then
        IFS=',' read -r -a xs <<<"${SCALE_OUTPUT}"
        for x in "${xs[@]}"; do run_cell "${prefix}" output "${x}" "${SCALE_BASE_INPUT}" "${x}" "${SCALE_BASE_CONC}"; done
    fi
    if [[ "${axes}" == *",grid,"* ]]; then
        IFS=',' read -r -a xs <<<"${SCALE_GRID_INPUT}"
        IFS=',' read -r -a os <<<"${SCALE_GRID_OUTPUT}"
        for x in "${xs[@]}"; do for o in "${os[@]}"; do
            run_cell "${prefix}" grid "${x}x${o}" "${x}" "${o}" "${SCALE_BASE_CONC}"
        done; done
    fi
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
    conc = sorted(cells(e, "scaling", "conc"), key=lambda c: c["rung"])
    if conc:
        # TPOT at the top rung vs conc<=2: the per-request cost of batching (~1.0 is good).
        top = conc[-1]
        base_tp = next((c["median_tpot_ms"] for c in conc if c["rung"] <= 2), None)
        line = f"    scaling: conc {top['rung']} -> {fmt(top.get('output_throughput'), 1)} tok/s"
        if top.get("median_tpot_ms") and base_tp:
            line += f" · TPOT x{top['median_tpot_ms'] / base_tp:.2f} vs conc<=2"
        print(line)
        for name, key in (("mlx-lm", "scaling_mlx_lm"), ("ollama", "scaling_ollama")):
            t = [c for c in cells(e, key, "conc") if c["rung"] == top["rung"]]
            if t and t[0].get("output_throughput"):
                print(f"    scaling {name}: conc {top['rung']} -> {fmt(t[0]['output_throughput'], 1)} tok/s")
    # Text heat maps: scratchy / engine output tok/s per grid cell (>1.00 = scratchy faster).
    grid = {(c["input_len"], c["output_len"]): c for c in cells(e, "scaling", "grid")}
    ins = sorted({k[0] for k in grid}); outs = sorted({k[1] for k in grid})
    for name, key in (("mlx-lm", "scaling_mlx_lm"), ("ollama", "scaling_ollama")):
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

    feats="metal,serve,bench,model/${stem}${quant:+,quant/${quant}}"
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
    # The next build overwrites ${BIN}, so each model keeps its own copy.
    model_bin="${RAW}/scr-${stem}"
    (( built )) && cp "${BIN}" "${model_bin}"

    rm -f "${RAW}"/{exec,exec-mlx,exec-ollama,serve}-"${stem}".json "${RAW}"/scale{,-mlx,-ollama}-"${stem}".*.json
    cell_model=""; cell_tok=""; oquant=""
    otag=""; [[ -n "${OLLAMA_BIN}" ]] && otag="$(ollama_tag "${stem}")"

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
        if "${BIN}" bench startup --model "${id}" --exec --mode cli --scenarios warm --port "${PORT}" \
                --child-cmd "${model_bin} chat -m ${id} --device metal -q {prompt} --max-tokens {output_len} --temperature 0" \
                --backend scratchy \
                --parity-cmd "${MLX_PYTHON} -m mlx_lm.generate --model ${id} --prompt {prompt} --max-tokens {output_len} --temp 0" \
                --parity-backend mlx-lm \
                --output-json "${RAW}/parity-${stem}.json" >"${RAW}/parity-${stem}.log" 2>&1; then
            grep -E "^ +(capital|arith|count) |PARITY OK" "${RAW}/parity-${stem}.log" | sed 's/^/    /'
        else
            parity_ok=0
            echo "    PARITY FAILED — not timing ${stem}; ${RAW}/parity-${stem}.log" >&2
            grep -E " (OK|FAIL) | a: | b: |Error" "${RAW}/parity-${stem}.log" | head -12 | sed 's/^/    /' >&2
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
                --max-concurrency 1 --temperature 0 --seed "${RANDOM}${RANDOM}" \
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

    # ---- mlx-lm
    if (( parity_ok )) && [[ -n "${MLX_PYTHON}" ]]; then
        run_ladder mlx-lm "${id}" "${MLX_PYTHON} -m mlx_lm.server --model ${id} --port ${PORT}" \
            mlx-lm "${RAW}/exec-mlx-${stem}.json"
        if (( SCALING )); then
            echo "--- scaling sweep, mlx-lm"
            if serve_up "${RAW}/serve-mlx-${stem}.log" /v1/models \
                    "${MLX_PYTHON}" -m mlx_lm.server --model "${id}" --port "${PORT}"; then
                run_scale_cells "${RAW}/scale-mlx-${stem}"
            else
                echo "    mlx-lm server never became ready — ${RAW}/serve-mlx-${stem}.log" >&2
            fi
            serve_down
        fi
    fi

    # ---- ollama
    if (( parity_ok )) && [[ -n "${otag}" ]]; then
        echo "--- ollama ${otag}"
        show() { curl -fsS -m 10 "http://127.0.0.1:${PORT}/api/show" -d "{\"model\":\"${otag}\"}"; }
        have=0
        if ollama_up; then
            if show >/dev/null 2>&1; then
                have=1
            elif (( ! OFFLINE )); then
                echo "    pulling ${otag} (first use)"
                OLLAMA_HOST="127.0.0.1:${PORT}" "${OLLAMA_BIN}" pull "${otag}" \
                    >>"${RAW}/serve-ollama-${stem}.log" 2>&1 && have=1
            fi
            (( have )) && oquant="$(show 2>/dev/null | python3 -c 'import json,sys; print(json.load(sys.stdin)["details"]["quantization_level"])' || true)"
        fi
        serve_down
        if (( ! have )); then
            echo "    ${otag} not available (offline, or the pull failed) — skipped" >&2
        else
            oevict=()
            [[ ",${SCENARIOS}," == *",frozen,"* ]] && oevict=(--evict-path "${HOME}/.ollama/models/blobs")
            run_ladder ollama "${otag}" "$(ollama_env) ${OLLAMA_BIN} serve" vllm \
                "${RAW}/exec-ollama-${stem}.json" ${oevict[@]+"${oevict[@]}"}
            if (( SCALING )); then
                # ollama reserves NUM_PARALLEL x CONTEXT_LENGTH of KV up front, so
                # each axis gets a server sized for just that axis.
                cell_model="${otag}"; cell_tok="${id}"
                IFS=',' read -r -a axes <<<"${SCALE_AXES}"
                for ax in "${axes[@]}"; do
                    par=${SCALE_BASE_CONC}
                    case "${ax}" in
                        conc)   par=$(max_of "${SCALE_CONC}"); ctx=$(( SCALE_BASE_INPUT + SCALE_BASE_OUTPUT )) ;;
                        input)  ctx=$(( $(max_of "${SCALE_INPUT}") + SCALE_BASE_OUTPUT )) ;;
                        output) ctx=$(( SCALE_BASE_INPUT + $(max_of "${SCALE_OUTPUT}") )) ;;
                        grid)   ctx=$(( $(max_of "${SCALE_GRID_INPUT}") + $(max_of "${SCALE_GRID_OUTPUT}") )) ;;
                        *)      continue ;;
                    esac
                    ctx=$(( ctx + 256 ))   # tokenizer slack
                    echo "--- scaling sweep, ollama ${ax} (num_parallel ${par}, context ${ctx})"
                    if ollama_up OLLAMA_NUM_PARALLEL="${par}" OLLAMA_CONTEXT_LENGTH="${ctx}"; then
                        run_scale_cells "${RAW}/scale-ollama-${stem}" "${ax}"
                    else
                        echo "    ollama serve did not start — ${RAW}/serve-ollama-${stem}.log" >&2
                    fi
                    serve_down
                done
                cell_model=""; cell_tok=""
            fi
        fi
    fi

    python3 - "${JSON}" "${RAW}" "${stem}" "${id}" "${quant}" "${feats}" "${built}" \
              "${build_secs}" "${bytes}" "${otag}" "${oquant}" "${parity_ok}" "${MLX_PYTHON}" <<'PY'
import glob, json, os, re, sys
js, raw, stem, mid, quant, feats, built, secs, size, otag, oquant, parity_ok, mlx_python = sys.argv[1:14]
KEEP = ["median_ttft_ms", "p99_ttft_ms", "median_tpot_ms", "p99_tpot_ms", "median_itl_ms",
        "p99_itl_ms", "median_e2el_ms", "output_throughput", "request_throughput",
        "completed", "total_output_tokens", "duration", "unstreamed_requests"]
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
d["models"].append({
    "stem": stem, "model_id": mid, "quant": quant or None, "features": feats,
    "built": built == "1",
    # null = gate not run (no mlx-lm, or no build); false = gate failed, nothing timed.
    "parity_mlx_lm": (parity_ok == "1") if (mlx_python and built == "1") else None,
    "footprint": {"build_seconds": int(secs) if secs else None,
                  "binary_bytes": int(size) if size else None,
                  "container_image_bytes": None,
                  "note": "no container image: Metal is not available in Linux containers"},
    "cache_ladder": load(f"exec-{stem}.json"),
    "cache_ladder_mlx_lm": load(f"exec-mlx-{stem}.json"),
    "warm_serving": metrics(load(f"serve-{stem}.json")),
    "scaling": scaling(""),
    "scaling_mlx_lm": scaling("-mlx"),
    "ollama": None if not otag else {
        "tag": otag, "quantization": oquant or None,
        "disclosed": ["GGUF weights, not the MLX checkpoint: no parity gate",
                      "ignores ignore_eos, so it can under-generate",
                      "peak RSS is `ollama serve` only; the runner grandchild is not counted",
                      "scaling server sized per axis: OLLAMA_NUM_PARALLEL = axis top concurrency, "
                      "OLLAMA_CONTEXT_LENGTH = longest input+output (+256)"]},
    "cache_ladder_ollama": load(f"exec-ollama-{stem}.json"),
    "scaling_ollama": scaling("-ollama"),
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
