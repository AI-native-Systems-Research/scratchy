#!/usr/bin/env bash
# P-EAGLE on gpt-oss-20b and gpt-oss-120b: PR #__N__ against its base on upstream main (their
# merge-base, so main's commits since — #597 changed gpt-oss's MoE prefill rounding — fork neither
# the texts nor the timings), on the P-EAGLE model card's benchmarks (MT-Bench, HumanEval, GSM-8K).
# Builds the base, the PR (two drafts a step, its default) and the PR at four drafts (its
# arch.json's `spec_drafts` 2 -> 4), each in its own
# worktree and target dir under .pr-ab-__N__/, a binary per model: gpt-oss-20b at MLX 4-bit
# (`jesusoctavioas/gpt-oss-20b-mlx-4Bit`, quant/mlx-affine-b4-g64-qembed) and gpt-oss-120b's 3-bit
# (`scratchery/gpt-oss-120b-3bit`, which carries its own recipe: no quant feature). Converts AWS's
# two drafters to the mixed MLX layout the PR compiles (MLPs at 3 bits, attention and fc at 4,
# embedding at 8, group 64). Then, for each model, it serves four legs: base alone, PR alone, PR
# drafting 2 (k2), PR drafting 4 (k4). The order reverses every round (3 rounds), and the wired GPU
# set settles back to its baseline before each leg. Each leg sends 5 prompts of each benchmark,
# greedy, up to 512 tokens, one at a time, and prints tok/s (wall, prefill included) and acceptance
# length (1 + accepted / drafts, from the server's spec-decode counters), then three ~2.5k-token
# prompts' TTFT. Last, whether the PR alone's text matches the base's. MODELS="20b" or "120b" runs
# one. It prints each tree's commit and drafts a step first.
# Run it from a scratchy checkout with nothing else on the GPU.
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
N=__N__
W=.pr-ab-$N
PY=$HOME/git/mlx-lm/.venv/bin/python
PORT=8797
MODELS=${MODELS:-20b 120b}
mkdir -p "$W/bench"
git fetch -q https://github.com/AI-native-Systems-Research/scratchy main:refs/pr-ab-$N/main
git fetch -q https://github.com/AI-native-Systems-Research/scratchy pull/$N/head:refs/pr-ab-$N/pr
git update-ref refs/pr-ab-$N/base "$(git merge-base refs/pr-ab-$N/main refs/pr-ab-$N/pr)"
for t in base pr k4; do
  ref=refs/pr-ab-$N/$t
  [ $t = k4 ] && ref=refs/pr-ab-$N/pr
  [ -d "$W/$t" ] || git worktree add -q --detach "$W/$t" $ref
  git -C "$W/$t" checkout -q -f --detach $ref
done
sed -i '' 's/"spec_drafts": 2,/"spec_drafts": 4,/' "$W/k4/crates/models/arch/configs/p-eagle/arch.json"
grep -q '"spec_drafts": 4,' "$W/k4/crates/models/arch/configs/p-eagle/arch.json"
for t in base pr k4; do
  drafts=$(grep -o '"spec_drafts": [0-9]*' "$W/$t/crates/models/arch/configs/p-eagle/arch.json" \
    2> /dev/null || echo "no P-EAGLE")
  echo "$t: $(git -C "$W/$t" log -1 --format='%h %s') ($drafts)"
done

# Per model: its target repo, its build features, AWS's drafter.
target() { case $1 in 20b) echo jesusoctavioas/gpt-oss-20b-mlx-4Bit ;; 120b) echo scratchery/gpt-oss-120b-3bit ;; esac; }
features() { case $1 in 20b) echo model/gpt-oss-20b,quant/mlx-affine-b4-g64-qembed ;; 120b) echo model/gpt-oss-120b-3bit ;; esac; }
drafter() { case $1 in 20b) echo amazon/GPT-OSS-20B-P-EAGLE ;; 120b) echo amazon/gpt-oss-120b-p-eagle ;; esac; }
build() { # tree model
  local spec=""
  [ $1 = base ] || spec=,spec/mtp
  (cd "$W/$1" && CARGO_TARGET_DIR="../target-$1" cargo build --release -q -p scratchy-cli \
    --features "metal,serve,metrics,$(features $2)$spec")
  cp "$W/target-$1/release/scr" "$W/scr-$1-$2"
}
for m in $MODELS; do
  for t in base pr k4; do build $t $m; done
done

# The benchmarks' prompts: MT-Bench's first turns, HumanEval's and GSM-8K's test rows.
rows=https://datasets-server.huggingface.co/rows
[ -f "$W/bench/mtbench.jsonl" ] || curl -sfL -o "$W/bench/mtbench.jsonl" \
  https://huggingface.co/datasets/HuggingFaceH4/mt_bench_prompts/resolve/main/raw/question.jsonl
[ -f "$W/bench/humaneval.json" ] || curl -sf -o "$W/bench/humaneval.json" \
  "$rows?dataset=openai/openai_humaneval&config=openai_humaneval&split=test&offset=0&length=20"
[ -f "$W/bench/gsm8k.json" ] || curl -sf -o "$W/bench/gsm8k.json" \
  "$rows?dataset=openai/gsm8k&config=main&split=test&offset=0&length=20"

convert() { # drafter repo, out dir
  [ -f "$2/model.safetensors" ] && return
  local src
  src=$("$PY" -c "from huggingface_hub import snapshot_download as d; print(d('$1'))")
  "$PY" - "$src" "$2" <<'PY'
import json, os, sys
import mlx.core as mx
src, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
W = mx.load(os.path.join(src, "model.safetensors"))
def bits_of(base):
    if base == "embed_tokens":
        return 8
    if ".mlp." in base:
        return 3
    return 4
q, modules = {}, {}
for name, w in W.items():
    if name in ("lm_head.weight", "d2t", "t2d"):
        continue
    if name.endswith(".weight") and w.ndim == 2 and "norm" not in name:
        base = name[: -len(".weight")]
        wq, scales, biases = mx.quantize(w, group_size=64, bits=bits_of(base))
        q[base + ".weight"] = wq
        q[base + ".scales"] = scales.astype(mx.bfloat16)
        q[base + ".biases"] = biases.astype(mx.bfloat16)
        modules[base] = {"group_size": 64, "bits": bits_of(base)}
    else:
        q[name] = w
modules["lm_head"] = {"group_size": 64, "bits": 4}
mx.save_safetensors(os.path.join(out, "model.safetensors"), q, metadata={"format": "mlx"})
cfg = json.load(open(os.path.join(src, "config.json")))
cfg["quantization"] = cfg["quantization_config"] = {"group_size": 64, "bits": 4, **dict(sorted(modules.items()))}
json.dump(cfg, open(os.path.join(out, "config.json"), "w"), indent=2)
print(f"drafter: {sum(v.nbytes for v in q.values()) / 1e9:.2f} GB -> {out}")
PY
}
for m in $MODELS; do convert "$(drafter $m)" "$W/gpt-oss-$m-p-eagle-mixed"; done

wired() { vm_stat | awk '/Pages wired down/ { gsub("\\.", "", $4); print $4 * 16384 }'; }
BASE=$(wired)
settle() {
  for _ in $(seq 1 300); do
    [ $(($(wired) - BASE)) -lt $((2 << 30)) ] && return
    timeout 2 tail -f /dev/null || true
  done
  echo "wired GPU memory never settled"
}
client() { # model, out-dir
  "$PY" -I - "$PORT" "$1" "$W/bench" "$2" <<'PY'
import json, re, sys, time, urllib.request
port, model, bench, out = sys.argv[1:5]
def counters():
    try:
        text = urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics").read().decode()
    except Exception:
        return 0.0, 0.0
    def get(name):
        m = re.search(rf"^\S*{name}(?:_total)?(?:{{[^}}]*}})?\s+([0-9.e+]+)$", text, re.M)
        return float(m.group(1)) if m else 0.0
    return get("spec_decode_num_drafts"), get("spec_decode_num_accepted_tokens")
def chat(prompt, max_tokens):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"content-type": "application/json"})
    reply = json.load(urllib.request.urlopen(req))
    return reply["choices"][0]["message"]["content"] or "", reply["usage"]["completion_tokens"]
sets = {
    "mtbench": [json.loads(l)["prompt"][0] for l in open(f"{bench}/mtbench.jsonl")],
    "humaneval": [r["row"]["prompt"] for r in json.load(open(f"{bench}/humaneval.json"))["rows"]],
    "gsm8k": [r["row"]["question"] for r in json.load(open(f"{bench}/gsm8k.json"))["rows"]],
}
row = []
for name, prompts in sets.items():
    d0, a0 = counters()
    toks, t0, texts = 0, time.time(), []
    for p in prompts[:5]:
        text, n = chat(p, 512)
        toks += n
        texts.append(text)
    wall = time.time() - t0
    d1, a1 = counters()
    al = f"{1 + (a1 - a0) / (d1 - d0):.2f}" if d1 > d0 else "-"
    open(f"{out}/{name}.txt", "w").write("\n\x1e\n".join(texts))
    row.append(f"{name} {toks / wall:5.1f} tok/s AL {al}")
base = ("The river carried the old mill's stones downstream every spring, and the town counted the "
        "boats that drifted past the willows while the miller kept his ledgers. ")
ttft = []
for i in range(3):
    body = json.dumps({"model": model, "temperature": 0, "max_tokens": 8, "stream": True,
                       "messages": [{"role": "user", "content": f"Variant {i}: " + base * 70}]}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", body,
                                 {"content-type": "application/json"})
    t0, first = time.time(), None
    with urllib.request.urlopen(req) as r:
        for line in r:
            if first is None and line.startswith(b"data:"):
                first = time.time()
    ttft.append(f"{1000 * (first - t0):.0f}")
print(" | ".join(row) + " | ttft(2.5k) " + "/".join(ttft) + " ms", flush=True)
PY
}
leg() { # model leg round
  local m=$1 bin=$2 extra=()
  case $2 in
    k2) bin=pr extra=(--speculative-model "$W/gpt-oss-$m-p-eagle-mixed") ;;
    k4) extra=(--speculative-model "$W/gpt-oss-$m-p-eagle-mixed") ;;
  esac
  local out="$W/out-$m-$2-$3"
  mkdir -p "$out"
  settle
  "$W/scr-$bin-$m" serve "$(target $m)" --port $PORT --enable-metrics ${extra[@]+"${extra[@]}"} \
    > "$out/serve.log" 2>&1 &
  local pid=$!
  until curl -sf "http://127.0.0.1:$PORT/v1/models" > /dev/null; do
    kill -0 $pid 2> /dev/null || { echo "$m $2: server died, see $out/serve.log"; return 1; }
    timeout 2 tail -f /dev/null || true
  done
  printf '%-5s %-5s ' "$m" "$2"
  client "$(target $m)" "$out"
  kill -INT $pid
  wait $pid || true
}
for m in $MODELS; do
  for round in 1 2 3; do
    echo "== gpt-oss-$m round $round"
    if [ $((round % 2)) = 1 ]; then order=(base pr k2 k4); else order=(k4 k2 pr base); fi
    for l in "${order[@]}"; do leg $m "$l" $round; done
  done
  # A drafted leg's greedy text may fork from the base's at a near-tie (#420: a verify step's rows
  # are not computed as a decode step's row is); the PR alone must match the base.
  for c in mtbench humaneval gsm8k; do
    if cmp -s "$W/out-$m-base-1/$c.txt" "$W/out-$m-pr-1/$c.txt"; then s=same; else s=differs; fi
    echo "gpt-oss-$m pr $c text: $s as the base"
  done
done
echo "Clean up after: for t in base pr k4; do git worktree remove --force $W/\$t; done; rm -rf $W"
