#!/usr/bin/env bash
# One-shot A/B for the GDN ssm-state dtype flip (PR "store the GDN ssm state
# as f16" on metal): builds both arms at pinned commits and measures, per arm:
#
#   perf    — `scr bench serve` cells (TTFT/TPOT/ITL p50+p99) over the m=1
#             buckets: short chat, long prefill, decode-heavy, batch-8.
#             ROUNDS balanced ABBA-style rounds (odd rounds base-first, even
#             rounds pr-first), a fresh server per leg so no cache crosses legs.
#   quality — greedy 512-token continuations of the same natural-text corpus
#             from three context depths (2k/5k/8k tokens), prefix caching OFF
#             so every depth is a full cold prefill: the pure state evolution.
#             Reports where the two arms' texts first diverge and how far each
#             continues the corpus's true text, plus a determinism rerun.
#
# Both arms build in ONE throwaway worktree (its own target dir, warm across
# the two checkouts): the merge-base binary first, then the PR head — so the
# head build is a warm increment. Everything each arm measured is printed with
# its commit.
#
# Usage (from any worktree of the repo, clean or not — arms build from the
# pinned SHAs, not the working tree):
#   scripts/bench_gdn_ssm_state_ab.sh [--rounds 3] [--port 8211]
#         [--skip-build] [--skip-perf] [--skip-quality]
#
# Results land in bench_results/gdn_ssm_state_ab/<ts>/: scr-base/scr-pr
# binaries, per-leg bench JSONs, quality texts, summary.md.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

MODEL="mlx-community/Qwen3.6-35B-A3B-4bit"
FEATURES="metal,serve,bench,model/qwen3.6-35b-a3b,quant/mlx-affine-b4-g64-qembed"
ROUNDS=3
PORT=8211
SKIP_BUILD=0
SKIP_PERF=0
SKIP_QUALITY=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)        MODEL="$2"; shift 2 ;;
        --rounds)       ROUNDS="$2"; shift 2 ;;
        --port)         PORT="$2"; shift 2 ;;
        --skip-build)   SKIP_BUILD=1; shift ;;
        --skip-perf)    SKIP_PERF=1; shift ;;
        --skip-quality) SKIP_QUALITY=1; shift ;;
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

BASE_SHA="$(git -C "${REPO_ROOT}" merge-base HEAD origin/main)"
HEAD_SHA="$(git -C "${REPO_ROOT}" rev-parse HEAD)"
TS="$(date -u +%Y%m%dT%H%M%SZ)"
OUT_DIR="${REPO_ROOT}/bench_results/gdn_ssm_state_ab/${TS}"
WT="${REPO_ROOT}/.claude/worktrees/gdnf16-ab"
mkdir -p "${OUT_DIR}"

# The quality corpus: fixed repo files, materialized ONCE from the PR-head
# tree so both arms score byte-identical text.
CORPUS_FILES=(README.md docs/BUILD.md docs/MODELS.md docs/BENCHMARKING.md)

echo "machine:   $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown) ($(uname -m))"
echo "base arm:  ${BASE_SHA} (merge-base with origin/main)"
echo "head arm:  ${HEAD_SHA}"
echo "model:     ${MODEL}"
echo "results:   ${OUT_DIR}"
{
    echo "machine: $(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo unknown) $(uname -m)"
    echo "base: ${BASE_SHA}"
    echo "head: ${HEAD_SHA}"
    echo "model: ${MODEL}"
    echo "features: ${FEATURES}"
    echo "rounds: ${ROUNDS}"
} > "${OUT_DIR}/run_config.txt"

# ---- builds ------------------------------------------------------------------
if [[ "${SKIP_BUILD}" -ne 1 ]]; then
    if [[ ! -d "${WT}" ]]; then
        git -C "${REPO_ROOT}" worktree add --detach "${WT}" "${BASE_SHA}"
    fi
    for spec in "base:${BASE_SHA}" "pr:${HEAD_SHA}"; do
        arm="${spec%%:*}"
        sha="${spec##*:}"
        git -C "${WT}" checkout --detach "${sha}" >/dev/null 2>&1
        echo
        echo "======== building ${arm} @ ${sha} ========"
        (cd "${WT}" && cargo build --release -p scratchy-cli --features "${FEATURES}")
        cp "${WT}/target/release/scr" "${OUT_DIR}/scr-${arm}"
    done
    # The corpus comes from the head tree, pinned to the SHA just built.
    for f in "${CORPUS_FILES[@]}"; do
        cat "${WT}/${f}" >> "${OUT_DIR}/corpus.txt"
        printf '\n\n' >> "${OUT_DIR}/corpus.txt"
    done
fi
BASE_BIN="${OUT_DIR}/scr-base"
PR_BIN="${OUT_DIR}/scr-pr"
CORPUS="${OUT_DIR}/corpus.txt"
for b in "${BASE_BIN}" "${PR_BIN}"; do
    [[ -x "${b}" ]] || { echo "missing ${b} (run without --skip-build first)" >&2; exit 1; }
done
[[ -s "${CORPUS}" ]] || { echo "missing ${CORPUS}" >&2; exit 1; }

# ---- server lifecycle ---------------------------------------------------------
SERVER_PID=""

wait_for_server() {
    local deadline=$((SECONDS + 600))
    while (( SECONDS < deadline )); do
        curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1 && return 0
        if [[ -n "${SERVER_PID}" ]] && ! kill -0 "${SERVER_PID}" 2>/dev/null; then
            echo "server pid ${SERVER_PID} died; see ${1:-server.log}" >&2
            return 1
        fi
        sleep 2
    done
    echo "server not ready within 600s" >&2
    return 1
}

stop_server() {
    if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
        kill -TERM "${SERVER_PID}" 2>/dev/null || true
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

start_server() {
    local bin="$1" log="$2"; shift 2
    if curl -fsS "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
        echo "port ${PORT} already serving — refusing to run against a co-resident server" >&2
        return 1
    fi
    "${bin}" serve -m "${MODEL}" --device metal --num-speculative-tokens 0 \
        --port "${PORT}" "$@" >"${log}" 2>&1 &
    SERVER_PID=$!
    wait_for_server "${log}"
}

# ---- perf legs ----------------------------------------------------------------
CELLS=("128x128x1" "2048x128x1" "128x512x1" "512x128x8")

run_perf_cell() {
    local arm="$1" round="$2" cell="$3" idx="$4"
    local il ol conc
    IFS='x' read -r il ol conc <<<"${cell}"
    local np=$(( conc * 2 ))
    (( np < 6 )) && np=6
    local json="${OUT_DIR}/perf.${arm}.r${round}.${cell}.json"
    echo "--- perf ${arm} round ${round} cell ${cell} ---"
    "${PR_BIN}" bench serve \
        --base-url "http://127.0.0.1:${PORT}" \
        --model "${MODEL}" \
        --num-prompts "${np}" \
        --input-len "${il}" \
        --output-len "${ol}" \
        --max-concurrency "${conc}" \
        --seed "$(( 7000 + idx ))" \
        --num-warmups 1 \
        --percentile-metrics ttft,tpot,itl,e2el \
        --metric-percentiles 50,99 \
        --output-json "${json}" \
        --disable-tqdm >/dev/null
}

if [[ "${SKIP_PERF}" -ne 1 ]]; then
    round=1
    while (( round <= ROUNDS )); do
        # Odd rounds run base first, even rounds pr first: balanced ordering.
        if (( round % 2 == 1 )); then order=(base pr); else order=(pr base); fi
        for arm in "${order[@]}"; do
            arm_bin="${OUT_DIR}/scr-${arm}"
            start_server "${arm_bin}" "${OUT_DIR}/perf.${arm}.r${round}.serve.log" || exit 1
            idx=0
            for cell in "${CELLS[@]}"; do
                run_perf_cell "${arm}" "${round}" "${cell}" "${idx}" || exit 1
                idx=$(( idx + 1 ))
            done
            stop_server
        done
        round=$(( round + 1 ))
    done
fi

# ---- quality legs ---------------------------------------------------------------
# Greedy 512-token continuations of the corpus from three depths, prefix
# caching OFF: each depth is one full cold prefill through the real state
# evolution, no snapshot-resume in the path (that path is pinned bit-exact by
# the kernel tests instead).
QUALITY_DEPTHS=(2048 5120 8192)
CHARS_PER_TOKEN=4.2

python3 - "${CORPUS}" "${OUT_DIR}" "${CHARS_PER_TOKEN}" "${QUALITY_DEPTHS[@]}" <<'PY'
import sys
corpus_path, out_dir, cpt = sys.argv[1], sys.argv[2], float(sys.argv[3])
depths = [int(d) for d in sys.argv[4:]]
text = open(corpus_path).read()
for d in depths:
    target = int(d * cpt)
    # Snap forward to a paragraph break so successive arms tokenize the same
    # bytes (the exact token count is reported by the server per arm).
    cut = text.find("\n\n", target)
    cut = len(text) if cut < 0 else cut
    open(f"{out_dir}/quality.slice.{d}.txt", "w").write(text[:cut])
    open(f"{out_dir}/quality.true.{d}.txt", "w").write(text[cut:])
PY

run_quality_depth() {
    local arm="$1" depth="$2" tag="${3:-}"
    local slice="${OUT_DIR}/quality.slice.${depth}.txt"
    local json="${OUT_DIR}/quality.${arm}.${depth}.${tag}json"
    echo "--- quality ${arm} depth ${depth}${tag:+ (${tag%.})} ---"
    python3 - "${json}" "${slice}" "${PORT}" "${MODEL}" <<'PY'
import json, sys, urllib.request
out, slice_path, port, model = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
prompt = open(slice_path).read()
req = urllib.request.Request(
    f"http://127.0.0.1:{port}/v1/completions",
    data=json.dumps({
        "model": model, "prompt": prompt,
        "max_tokens": 512, "temperature": 0.0,
    }).encode(),
    headers={"Content-Type": "application/json"},
)
with urllib.request.urlopen(req, timeout=1200) as r:
    d = json.load(r)
json.dump(d, open(out, "w"), indent=1)
c = d["choices"][0]
print(f"    prompt_tokens={d['usage']['prompt_tokens']} "
      f"finish={c.get('finish_reason')} chars={len(c.get('text') or '')}")
PY
}

if [[ "${SKIP_QUALITY}" -ne 1 ]]; then
    deepest="${QUALITY_DEPTHS[$(( ${#QUALITY_DEPTHS[@]} - 1 ))]}"
    for arm in base pr; do
        start_server "${OUT_DIR}/scr-${arm}" "${OUT_DIR}/quality.${arm}.serve.log" \
            --no-prefix-caching || exit 1
        for depth in "${QUALITY_DEPTHS[@]}"; do
            run_quality_depth "${arm}" "${depth}" || exit 1
        done
        # Determinism: rerun the deepest depth, expect byte-identical text.
        run_quality_depth "${arm}" "${deepest}" "rerun." || exit 1
        stop_server
    done
fi

# ---- summary --------------------------------------------------------------------
python3 - "${OUT_DIR}" "${BASE_SHA}" "${HEAD_SHA}" "${ROUNDS}" "${QUALITY_DEPTHS[@]}" <<'PY' > "${OUT_DIR}/summary.md"
import json, os, statistics, sys

out_dir, base_sha, head_sha = sys.argv[1], sys.argv[2], sys.argv[3]
rounds, depths = int(sys.argv[4]), [int(d) for d in sys.argv[5:]]

print(f"# GDN ssm-state dtype A/B — {base_sha[:9]} (base) vs {head_sha[:9]} (head)")
print()
print(f"rounds: {rounds} (balanced, order alternates); cells: 128x128x1,"
      " 2048x128x1, 128x512x1, 512x128x8; m=1, MTP off, temperature 0.")
print()

def leg(arm, r, cell):
    p = os.path.join(out_dir, f"perf.{arm}.r{r}.{cell}.json")
    return json.load(open(p)) if os.path.exists(p) else None

def med(vals):
    vals = [v for v in vals if v is not None]
    return statistics.median(vals) if vals else None

cells = ["128x128x1", "2048x128x1", "128x512x1", "512x128x8"]
print("## Perf (median across rounds, ms)")
print()
print("| cell (in×out×conc) | metric | base | head | head/base |")
print("|---|---|---|---|---|")
for cell in cells:
    for m in ("ttft", "tpot", "itl"):
        row = {}
        for arm in ("base", "pr"):
            vals = []
            for r in range(1, rounds + 1):
                d = leg(arm, r, cell)
                vals.append(d.get(f"median_{m}_ms") if d else None)
            row[arm] = med(vals)
        if row["base"] is None or row["pr"] is None:
            continue
        ratio = f"{row['pr'] / row['base']:.3f}x"
        print(f"| {cell} | {m} p50 | {row['base']:.1f} | {row['pr']:.1f} | {ratio} |")

print()
print("## Quality — greedy 512-token continuations, prefix caching off")
print()
print("Same corpus slice to both arms; `first divergence` is the character")
print("position where the arms' texts split; `true-text run` is how far each")
print("arm continues the corpus's actual text before drifting. Determinism =")
print("the deepest depth rerun byte-identical.")
print()
print("| depth (target tok) | prompt tok (base/head) | first divergence | true-text run base | true-text run head | deterministic |")
print("|---|---|---|---|---|---|")

def q(arm, d, rerun=False):
    name = f"quality.{arm}.{d}.{'rerun.' if rerun else ''}json"
    p = os.path.join(out_dir, name)
    return json.load(open(p)) if os.path.exists(p) else None

def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i

for d in depths:
    db, dp = q("base", d), q("pr", d)
    if not (db and dp):
        continue
    tb, tp = db["choices"][0].get("text") or "", dp["choices"][0].get("text") or ""
    true = open(os.path.join(out_dir, f"quality.true.{d}.txt")).read().lstrip()
    rb, rp = q("base", d, rerun=True), q("pr", d, rerun=True)
    det = "—"
    if rb and rp:
        det = "yes" if (rb["choices"][0]["text"] == tb and
                        rp["choices"][0]["text"] == tp) else "NO"
    print(f"| ~{d} | {db['usage']['prompt_tokens']}/{dp['usage']['prompt_tokens']} "
          f"| char {lcp(tb, tp)} | {lcp(tb, true)} | {lcp(tp, true)} | {det} |")
print()
print("Continuation texts: `quality.<arm>.<depth>.json` in this directory.")
PY

echo
echo "summary -> ${OUT_DIR}/summary.md"
cat "${OUT_DIR}/summary.md"
