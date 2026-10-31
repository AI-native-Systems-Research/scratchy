#!/usr/bin/env bash
# DFlash2's benchmark cells, for the draft PR's experiments: each `label=path/to/scr[:extra serve args]` serves
# mlx-community/Qwen3.6-35B-A3B-4bit at its defaults, one fresh server a run, and `scr bench serve` sends 32
# random prompts (seed 1000) per cell. Runs go in bracketed order — every label forward, then reversed — so a
# drift over the sitting shows as a gap between a label's two runs. Prints a summary table to paste back.
#
#   scripts/bench_dflash2_cells.sh <cells> <out dir> <label>=<scr>[:serve args] ...
#
# <cells>: `833` (the five cells of issue #833), `mid` (512x1024 at 1, 2 and 4 at once), or a space-separated
# list like "512x1024@2 8192x128@1". Example:
#
#   scripts/bench_dflash2_cells.sh 833 out/e1 main=../main/target/release/scr pr=target/release/scr
#   scripts/bench_dflash2_cells.sh mid out/e2 main=../main/target/release/scr pr=target/release/scr \
#       "plain=target/release/scr:--num-speculative-tokens 0"
set -euo pipefail
case "$1" in
  833) cells="512x128@1 512x1024@1 8192x128@1 512x128@8 1024x64@1" ;;
  mid) cells="512x1024@1 512x1024@2 512x1024@4" ;;
  *) cells="$1" ;;
esac
out=$2
shift 2
mkdir -p "$out"
labels=("$@")
reversed=()
for ((i = ${#labels[@]} - 1; i >= 0; i--)); do reversed+=("${labels[i]}"); done
model=mlx-community/Qwen3.6-35B-A3B-4bit
port=18190
for cell in $cells; do
  shape=${cell%@*}; conc=${cell#*@}; in=${shape%x*}; outlen=${shape#*x}
  n=0
  for spec in "${labels[@]}" "${reversed[@]}"; do
    n=$((n + 1))
    label=${spec%%=*}; rest=${spec#*=}; scr=${rest%%:*}; extra=()
    [ "$rest" != "$scr" ] && read -r -a extra <<< "${rest#*:}"
    "$scr" serve "$model" --port "$port" ${extra[@]+"${extra[@]}"} > "$out/serve-$cell-$n-$label.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 600); do
      curl -s -m 2 "127.0.0.1:$port/v1/models" > /dev/null 2>&1 && break
      sleep 1
    done
    "$scr" bench serve --base-url "http://127.0.0.1:$port" --model "$model" --num-prompts 32 \
      --input-len "$in" --output-len "$outlen" --max-concurrency "$conc" --seed 1000 --disable-tqdm \
      --output-json "$out/$cell-$n-$label.json" > "$out/bench-$cell-$n-$label.log" 2>&1
    kill -INT "$pid"
    wait "$pid" || true
  done
done
python3 - "$out" "${labels[@]%%=*}" <<'PY'
import glob, json, re, sys
out, labels = sys.argv[1], sys.argv[2:]
runs = {}
for path in glob.glob(f"{out}/*.json"):
    cell, n, label = re.match(r".*/(.+)-(\d+)-([^-]+)\.json$", path).groups()
    j = json.load(open(path))
    runs.setdefault(cell, {}).setdefault(label, []).append(
        (int(n), j["output_throughput"], j.get("mean_ttft_ms") or 0, j.get("mean_tpot_ms") or 0))
head = "| cell | " + " | ".join(f"{l} tok/s (runs) | {l} TTFT ms | {l} TPOT ms" for l in labels) + " |"
print(head)
print("|---" * (1 + 3 * len(labels)) + "|")
for cell in sorted(runs):
    row = [cell]
    for l in labels:
        rs = sorted(runs[cell].get(l, []))
        if not rs:
            row += ["-", "-", "-"]
            continue
        mean = lambda i: sum(r[i] for r in rs) / len(rs)
        row += [f"{mean(1):.1f} ({' / '.join(f'{r[1]:.1f}' for r in rs)})", f"{mean(2):.0f}", f"{mean(3):.1f}"]
    print("| " + " | ".join(row) + " |")
for log in sorted(glob.glob(f"{out}/serve-*-1-*.log")):
    for line in open(log, errors="replace"):
        if "max_num_seqs =" in line or "kv_budget=" in line:
            print(f"{log.rsplit('/', 1)[1]}: {line.strip()[line.find('max_num_seqs') if 'max_num_seqs' in line else line.find('total='):]}")
PY
