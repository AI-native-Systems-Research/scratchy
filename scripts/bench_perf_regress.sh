#!/usr/bin/env bash
# Perf regression gate — lock tight per-arch perf ranges on this Mac.
#
# One `scr bench latency` per (model, cell) against a binary built from THIS
# worktree, compared with the recorded baseline for THIS chip. Fails (exit 1)
# when any cell lands outside its band, in either direction: a number that
# moved must be acknowledged — a `perf:` PR re-records the baseline in the
# same commit and carries the table in its message; anything else moving is a
# regression.
#
# Cells (per model), all with prefix caching OFF so every iteration pays its
# full prefill (with it on, iterations 2..N hit the cache and prefill2k would
# time nothing but a cache lookup):
#   decode1   in 64 / out 128 / bs 1   — single-stream decode (the canonical gate)
#   fold4     in 64 / out 128 / bs 4   — the m>=2 decode buckets (the fold)
#   prefill2k in 2048 / out 8 / bs 1   — the prefill ladder
#
# Metric: the p50 of per-iteration total latency (engine-internal, greedy,
# fixed token-id prompts, detokenization excluded) — the tightest repeatable
# number the harness can produce: no server, no HTTP, no sampling noise.
#
# Baselines live in scripts/perf-regress/baselines/<chip>.json, one file per
# chip, (re-)recorded with --update. A model whose weights are not in the HF
# cache is reported as SKIPPED — a chip's locked coverage is exactly what it
# ran, and the manifest below still names every arch, so a machine with the
# weights picks the row up.
#
# Usage:
#   scripts/bench_perf_regress.sh                 # compare vs baseline (exit 1 outside band)
#   scripts/bench_perf_regress.sh --update        # (re-)record this chip's baseline
#   scripts/bench_perf_regress.sh --markdown      # also print a paste-ready table
#   scripts/bench_perf_regress.sh --quick         # fewer iters, small cached models only
#   scripts/bench_perf_regress.sh --models a,b    # subset of stems
#   scripts/bench_perf_regress.sh --skip-build    # reuse target/release/scr as-is
#
# Run from a worktree of the commit under test, on AC power, with no other
# scr process alive. `perf:` commits carry this table; see
# docs/BENCHMARKING.md ("Perf regression gate").

set -euo pipefail

HERE="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &&>/dev/null; pwd)"
ROOT="$(cd -- "${HERE}/.." &&>/dev/null; pwd)"
BASELINE_DIR="${HERE}/perf-regress/baselines"

# ── the manifest: one canonical representative per supported arch ────────────
# arch|stem|HuggingFace id|quant preset feature ('-' = dense/bf16)
#
# Every preset below is declared in that arch's configs/<arch>/quantizations.json
# and must fingerprint-match the checkpoint at load — the loader rejects a
# mismatch loudly, so a wrong row fails at record time, never silently.
#
# Not listed, deliberately:
#   commandr, deepseek-v2, deepseek-v3, deepseek-v3-flat — ggml/GGUF-only
#     presets (no affine preset declared); add a row when one gets an
#     affine-quant canonical model.
#   qwen2-vl, qwen2-5-vl, qwen3-5-vl, gemma3-mm, locateanything — vision
#     arches; they pull the multimodal decode stack and have no text-only
#     perf-canonical cell yet.
#   modernbert — an encoder; it generates nothing to time.
#   gemma4 dense — no 4bit repack cached on the reference Mac; gemma4-moe
#     below covers the family's kernels there.
MANIFEST=(
  "gemma2|gemma2-2b|google/gemma-2-2b-it|-"
  "gemma3|gemma-3-1b-it|mlx-community/gemma-3-1b-it-4bit|mlx-affine-b4-g64"
  "gemma4-moe|gemma-4-26b-a4b-it|mlx-community/gemma-4-26b-a4b-it-4bit|mlx-affine-b4-g64"
  "glm4-moe|glm-4.5-air|mlx-community/GLM-4.5-Air-3bit|mlx-affine-b3-g64-qembed"
  "gpt-oss|gpt-oss-20b|jesusoctavioas/gpt-oss-20b-mlx-4Bit|mlx-affine-b4-g64-qembed"
  "granite|granite-3.3-2b-instruct|mlx-community/granite-3.3-2b-instruct-4bit|mlx-affine-b4-g64"
  "llama|llama-3.2-1b|mlx-community/Llama-3.2-1B-Instruct-4bit|mlx-affine-b4-g64"
  "mistral|mistral-7b-instruct-v0.3|mlx-community/Mistral-7B-Instruct-v0.3-4bit|mlx-affine-b4-g64"
  "mixtral|mixtral-8x7b-instruct-v0.1|mlx-community/Mixtral-8x7B-Instruct-v0.1-4bit|mlx-affine-b4-g64"
  "phi3|phi-3-mini-4k-instruct|mlx-community/Phi-3-mini-4k-instruct-4bit|mlx-affine-b4-g64"
  "qwen2|qwen2.5-3b|mlx-community/Qwen2.5-3B-Instruct-4bit|mlx-affine-b4-g64-qembed"
  "qwen3|qwen3-0.6b|mlx-community/Qwen3-0.6B-4bit|mlx-affine-b4-g64-qembed"
  "qwen3-5|qwen3.5-0.8b|mlx-community/Qwen3.5-0.8B-4bit|mlx-affine-b4-g64-qembed"
  "qwen3-5-moe|qwen3.6-35b-a3b|mlx-community/Qwen3.6-35B-A3B-4bit|mlx-affine-b4-g64-qembed"
  "qwen3-moe|qwen3-30b-a3b-instruct-2507|mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit|mlx-affine-b4-g64-qembed"
)

# Quick tier: the small cached models a Mac runs in minutes.
QUICK_STEMS="granite-3.3-2b-instruct,llama-3.2-1b,qwen3-0.6b,qwen3.5-0.8b,gemma-3-1b-it"

# Per-cell bands, two-sided, measured: decode/fold repeat to ±3.5%
# back-to-back on the same chip, but prefill2k drifts up to ±13% across
# processes (within a run p99≈p50 — the shift is between processes: memory
# layout, thermal/power state), so prefill gets a wider band or the gate
# cries wolf. `--band` overrides both.
BAND_DECODE=0.05
BAND_PREFILL=0.15
ITERS=30
WARMUP=10
OUT_JSON=""
UPDATE=0
MARKDOWN=0
QUICK=0
SKIP_BUILD=0
ONLY=""
ALLOW_BATTERY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --update)         UPDATE=1 ;;
    --markdown)       MARKDOWN=1 ;;
    --quick)          QUICK=1; ITERS=10; WARMUP=3 ;;
    --models)         ONLY="$2"; shift ;;
    --skip-build)     SKIP_BUILD=1 ;;
    --band)           BAND_DECODE="$2"; BAND_PREFILL="$2"; shift ;;
    --allow-battery)  ALLOW_BATTERY=1 ;;
    --out-json)       OUT_JSON="$2"; shift ;;
    -h|--help)        sed -n '2,38p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown flag: $1 (see --help)" >&2; exit 2 ;;
  esac
  shift
done
case "${BAND_DECODE}" in ''|*[!0-9.]*) echo "--band must be a fraction like 0.05" >&2; exit 2 ;; esac
case "${BAND_PREFILL}" in ''|*[!0-9.]*) echo "--band must be a fraction like 0.05" >&2; exit 2 ;; esac

# ── pre-flight ───────────────────────────────────────────────────────────────
chip="$(sysctl -n machdep.cpu.brand_string)"
slug="$(echo "${chip}" | tr '[:upper:] ' '[:lower:]-' | sed 's/[^a-z0-9-]//g')"
baseline_file="${BASELINE_DIR}/${slug}.json"
hf_hub="${HF_HOME:-$HOME/.cache/huggingface}/hub"

power="$(pmset -g batt 2>/dev/null | head -1 || true)"
if [[ "${ALLOW_BATTERY}" != 1 && "${power}" != *"AC Power"* ]]; then
  echo "REFUSING: ${power:-unknown power state} — battery runs are throttled; plug in or pass --allow-battery" >&2
  exit 2
fi
if leftovers="$(pgrep -fl 'scr serve|scr bench|scr chat' 2>/dev/null)"; then
  echo "REFUSING: another scr process is alive — it would take the GPU:" >&2
  echo "${leftovers}" >&2
  exit 2
fi

cd "${ROOT}"
sha="$(git rev-parse HEAD)"
branch="$(git branch --show-current)"
[[ -n "${branch}" ]] || branch="detached"
dirty="$(git status --porcelain | head -1 | wc -l | tr -d ' ')"
echo "perf-regress  ${chip}  ${sha:0:9} (${branch}, dirty=${dirty})"
echo "band ±${BAND_DECODE} decode/fold, ±${BAND_PREFILL} prefill   iters ${ITERS}+${WARMUP}w   baseline ${baseline_file}"

# ── the runnable set: manifest rows whose weights are cached here ───────────
declare -a ROWS=()
for entry in "${MANIFEST[@]}"; do
  IFS='|' read -r arch stem id quant <<<"${entry}"
  if [[ -n "${ONLY}" && ",${ONLY}," != *",${stem},"* ]]; then continue; fi
  if (( QUICK )) && [[ ",${QUICK_STEMS}," != *",${stem},"* ]]; then continue; fi
  hub_dir="${hf_hub}/models--${id//\//--}"
  if [[ ! -d "${hub_dir}" ]]; then
    echo "  SKIP ${stem} — weights not cached (${id})"
    continue
  fi
  ROWS+=("${entry}")
done
if [[ ${#ROWS[@]} -eq 0 ]]; then
  echo "no runnable models (nothing cached matches the manifest under this filter)" >&2
  exit 2
fi

# ── build one binary naming every runnable stem + its preset ────────────────
stems="$(printf '%s\n' "${ROWS[@]}" | cut -d'|' -f2 | sort -u | paste -sd, -)"
quants="$(printf '%s\n' "${ROWS[@]}" | cut -d'|' -f4 | { grep -vx -- - || true; } | sort -u | paste -sd, -)"
BIN="${ROOT}/target/release/scr"
if (( ! SKIP_BUILD )); then
  feats="metal,bench"
  feats+=",model/${stems//,/,model/}"
  [[ -n "${quants}" ]] && feats+=",quant/${quants//,/,quant/}"
  echo "--- build -F ${feats}"
  if ! cargo build --release -p scratchy-cli --features "${feats}" >/tmp/perf-regress-build.log 2>&1; then
    echo "BUILD FAILED (tail below; full log /tmp/perf-regress-build.log)" >&2
    tail -5 /tmp/perf-regress-build.log >&2
    exit 2
  fi
fi

# ── run every (model, cell) ──────────────────────────────────────────────────
RAW="$(mktemp -d /tmp/perf-regress.XXXXXX)"
keep_raw="${PERF_REGRESS_KEEP_RAW:-}"
trap 'if [[ -z "${keep_raw}" ]]; then rm -rf "${RAW}"; else echo "raw results kept: ${RAW}"; fi' EXIT
FAILURES=0
run_cell() { # stem id cell in out bs -> writes $RAW/<stem>-<cell>.json
  local stem="$1" id="$2" cell="$3" in="$4" out="$5" bs="$6"
  printf '  %-28s %-10s (in %s out %s bs %s) ' "${stem}" "${cell}" "${in}" "${out}" "${bs}"
  if ! "${BIN}" bench latency "${id}" --device metal \
      --input-len "${in}" --output-len "${out}" --batch-size "${bs}" \
      --num-iters "${ITERS}" --num-iters-warmup "${WARMUP}" \
      --temperature 0 --no-prefix-caching --disable-detokenize \
      --percentile 50,99 \
      --output-json "${RAW}/${stem}-${cell}.json" >/dev/null 2>"${RAW}/${stem}-${cell}.log"; then
    echo "FAILED (see ${RAW}/${stem}-${cell}.log)"
    return 1
  fi
  python3 - "${RAW}/${stem}-${cell}.json" <<'PY'
import json, sys
j = json.load(open(sys.argv[1]))
p50 = j["percentiles"]["50"] * 1000
p99 = j["percentiles"]["99"] * 1000
print(f"p50 {p50:9.2f} ms   (p99 {p99:9.2f})")
PY
}

for entry in "${ROWS[@]}"; do
  IFS='|' read -r arch stem id quant <<<"${entry}"
  echo "--- ${arch}: ${stem}"
  run_cell "${stem}" "${id}" decode1   64 128 1 || FAILURES=$((FAILURES+1))
  run_cell "${stem}" "${id}" fold4     64 128 4 || FAILURES=$((FAILURES+1))
  run_cell "${stem}" "${id}" prefill2k 2048 8 1 || FAILURES=$((FAILURES+1))
done

# ── compare / record ─────────────────────────────────────────────────────────
gate_rc=0
python3 - "$RAW" "$baseline_file" "$UPDATE" "$MARKDOWN" "$BAND_DECODE" "$BAND_PREFILL" "$chip" "$sha" "$branch" "$dirty" "$OUT_JSON" <<'PY' || gate_rc=$?
import json, os, sys
(raw, baseline_file, update, markdown, band_decode, band_prefill, chip, sha, branch, dirty, out_json) = sys.argv[1:12]
update, markdown = int(update), int(markdown)

cells = {
    "decode1": float(band_decode),
    "fold4": float(band_decode),
    "prefill2k": float(band_prefill),
}
runs = {}   # stem -> cell -> p50 ms
for f in sorted(os.listdir(raw)):
    if not f.endswith(".json"):
        continue
    stem, cell = f[:-5].rsplit("-", 1)
    if cell not in cells:
        continue
    j = json.load(open(os.path.join(raw, f)))
    runs.setdefault(stem, {})[cell] = j["percentiles"]["50"] * 1000

baseline = {"chip": chip, "bands": cells, "models": {}}
if os.path.exists(baseline_file):
    baseline = json.load(open(baseline_file))

rows, fails = [], 0
for stem in runs:
    for cell in cells:
        if cell not in runs[stem]:
            continue
        now = runs[stem][cell]
        band = cells[cell]
        base = baseline["models"].get(stem, {}).get(cell)
        if base is None:
            verdict, delta = "NEW" if not update else "RECORDED", None
        else:
            delta = (now - base) / base
            verdict = "ok" if abs(delta) <= band else "OUT OF BAND"
            if abs(delta) > band and not update:
                fails += 1
        rows.append((stem, cell, base, now, delta, verdict))

if update:
    baseline["chip"] = chip
    baseline["bands"] = cells
    baseline["repo"] = {"sha": sha, "branch": branch, "dirty": dirty != "0"}
    for stem in runs:
        baseline["models"].setdefault(stem, {}).update(runs[stem])
    os.makedirs(os.path.dirname(baseline_file), exist_ok=True)
    with open(baseline_file, "w") as f:
        json.dump(baseline, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nbaseline recorded -> {baseline_file}")

hdr = f"perf-regress  {chip}  vs baseline"
if baseline.get("repo"):
    hdr += f" @ {baseline['repo']['sha'][:9]}"
print("\n" + hdr)
print(f"{'model':<28} {'cell':<10} {'baseline':>12} {'now':>12} {'delta':>8}  verdict")
for stem, cell, base, now, delta, verdict in rows:
    b = f"{base:10.2f} ms" if base is not None else "—"
    d = f"{100*delta:+7.2f}%" if delta is not None else "—"
    print(f"{stem:<28} {cell:<10} {b:>12} {now:9.2f} ms {d:>8}  {verdict}")

if markdown:
    print("\n<!-- paste into the perf: commit message / PR body -->")
    md = f"#### perf-regress — {chip}, band ±{100*cells['decode1']:.0f}% decode/fold · ±{100*cells['prefill2k']:.0f}% prefill"
    if baseline.get("repo"):
        md += f", baseline @ `{baseline['repo']['sha'][:9]}`"
    print(md + "\n")
    print("| model | cell | baseline | now | Δ |")
    print("|---|---|---|---|---|")
    for stem, cell, base, now, delta, verdict in rows:
        b = f"{base:.2f} ms" if base is not None else "—"
        d = f"{100*delta:+.2f}%" if delta is not None else "—"
        print(f"| {stem} | {cell} | {b} | {now:.2f} ms | {d} |")

if out_json:
    with open(out_json, "w") as f:
        json.dump({"chip": chip, "bands": cells,
                   "repo": {"sha": sha, "branch": branch, "dirty": dirty != "0"},
                   "models": runs}, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nresults -> {out_json}")

skipped = [s for s in baseline["models"] if s not in runs]
if skipped:
    print(f"\nnote: {len(skipped)} baseline model(s) not run this pass: {', '.join(sorted(skipped))}")

sys.exit(1 if fails else 0)
PY

if (( FAILURES > 0 )); then
  echo "FAIL: ${FAILURES} cell run(s) errored — logs kept at ${RAW}" >&2
  keep_raw=1
  exit 1
fi
exit "${gate_rc}"
