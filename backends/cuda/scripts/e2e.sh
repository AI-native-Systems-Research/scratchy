#!/bin/bash
# One FP8 model end to end, SHA A against SHA B on ONE GPU (pod-side; run through drive.sh or
# chain.sh, README.md):
#   MODEL=27b|35b|gemma REPO_URL=… SHA_A=… SHA_B=… ./drive.sh scratchy-cuda-e2e-27b e2e.sh pins/qwen3.8-27b.txt
#   1. the checkpoint at its pinned revision, its files checked against the pin's sha256s;
#   2. both SHAs' CLI built in two worktrees of one clone, each its own target dir;
#   3. per run (A1, B1, A2, B2) a fresh server: `bench serve` 128x128 and 2048x128 (2 warm-ups,
#      then 10 requests), and at the first run of each SHA the 5 coherence prompts (coh.py scr);
#   4. vLLM in the same pod: its 5 coherence prompts with top-2 logprobs (the D2 reference);
#   5. report.py: /results/ab.md (TTFT and TPOT per SHA and run, the means, the delta, the drift;
#      D2 per SHA against vLLM, and each SHA's first differing token against the baseline's, when
#      drive.sh was given BASELINE_COHERENCE); then the gate that A's and B's tokens are identical.
# Every tool by absolute path or on PATH for the subprocesses; GPU hygiene before every server start
# and after every kill; `stop` on every failure; MODEL_DONE at the end.
set -o pipefail
mkdir -p /logs /work /results /ck
[ -f /work/env.sh ] && . /work/env.sh
stop() { echo "STOP: $1" | tee -a /logs/exits.txt; cp /logs/gpu-hygiene.txt /results/ 2>/dev/null; echo MODEL_DONE >> /logs/exits.txt; exit 1; }
[ -n "$REPO_URL" ] && [ -n "$SHA_A" ] && [ -n "$SHA_B" ] || stop "REPO_URL, SHA_A and SHA_B must be set"
case "$MODEL" in
  27b)   REPO=axteam/Qwen3.8-27B-FP8-dynamic; REV=6349d56f935f2ecc3e3627606b6a77f5b5a47a40; STEM=qwen3.8-27b-text-only ;;
  35b)   REPO=RedHatAI/Qwen3.6-35B-A3B-FP8-dynamic; REV=6496d54f9127add4e6a62b7c5e3c5579ddb04ac7; STEM=qwen3.6-35b-a3b ;;
  gemma) REPO=RedHatAI/gemma-4-26B-A4B-it-FP8-dynamic; REV=ed35d7abe5d940da41b4ff06eb482feb0be8cb44; STEM=gemma-4-26b-a4b-it ;;
  *) stop "MODEL must be 27b, 35b or gemma" ;;
esac
TAG=$MODEL; CK=/ck/$TAG; VLLM_VERSION=${VLLM_VERSION:-0.31.0}
export SHA_A SHA_B MODEL
export PATH=/usr/local/cuda/bin:/root/.cargo/bin:/opt/hf/bin:$PATH CUDA_COMPUTE_CAP=90 SCRATCHY_GPU=h100
run() { local name=$1 t0 rc; shift; t0=$(date +%s); "$@" >/logs/$name.log 2>&1; rc=$?; echo "EXIT $name $rc WALL=$(( $(date +%s)-t0 ))s" | tee -a /logs/exits.txt; return $rc; }
gpu_clean() {
  local procs used
  procs=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "$(date +%T) $1 used_mib=$used procs=[$procs]" >> /logs/gpu-hygiene.txt
  [ -z "$procs" ] && [ "${used:-99999}" -lt 1024 ]
}
gpu_wait_clean() { local i; for i in $(seq 1 12); do gpu_clean "$1 (try $i)" && return 0; sleep 5; done; stop "GPU busy: $1"; }
waitready() { local i; for i in $(seq 1 $1); do curl -s -o /dev/null http://127.0.0.1:8000/v1/models && return 0; kill -0 $2 2>/dev/null || return 1; sleep 5; done; return 1; }
kill_server() { kill $1 2>/dev/null; wait $1 2>/dev/null; gpu_wait_clean "after killing $2"; }

# ── setup ───────────────────────────────────────────────────────────
run apt sh -c 'apt-get update -qq && apt-get install -y -qq git curl jq pkg-config libssl-dev ca-certificates build-essential python3 python3-venv python3-pip binutils' || stop apt
run rustup sh -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal' || stop rustup
run env sh -c 'nvidia-smi; nvcc --version; rustc --version'
nvidia-smi --query-gpu=pci.bus_id,name,uuid,clocks.max.sm --format=csv,noheader > /results/gpu-bus.txt
cat /results/gpu-bus.txt | tee -a /logs/exits.txt
gpu_clean "pod start" || stop "GPU busy at pod start"
python3 -m venv /opt/vllm
# vLLM installs in the background while scratchy builds and runs (pip only downloads).
( t0=$(date +%s); /opt/vllm/bin/pip install vllm==$VLLM_VERSION >/logs/vllm-pip.log 2>&1; rc=$?; echo "EXIT vllm-pip $rc WALL=$(( $(date +%s)-t0 ))s" >> /logs/exits.txt ) &
VPIP=$!
python3 -m venv /opt/hf
run hfpip /opt/hf/bin/pip install huggingface_hub || stop "huggingface_hub"
run ckpt timeout 3000 /opt/hf/bin/hf download $REPO --revision $REV --local-dir $CK || stop "checkpoint download"
cat > /work/chk.py <<'PY'
import sys,glob,subprocess
d,pin=sys.argv[1:3]
want=sorted(l.strip() for l in open(pin) if l.strip())
got=sorted(subprocess.check_output(['sha256sum',f]).split()[0].decode() for f in glob.glob(d+'/*.safetensors'))
ok=got==want;print('files',len(got),'match' if ok else 'MISMATCH');sys.exit(0 if ok else 1)
PY
run shackpt python3 -I /work/chk.py $CK /work/pin.txt || stop "checkpoint sha mismatch"

# ── 1. build both SHAs, one clone, two worktrees (each its own target dir) ─
run clone sh -c "git clone $REPO_URL /root/pr && cd /root/pr && git fetch origin ${FETCH_REFS:-} && for s in $SHA_A $SHA_B; do git cat-file -e \$s^{commit} 2>/dev/null || git fetch origin \$s; done && git worktree add --detach /root/a $SHA_A && git worktree add --detach /root/b $SHA_B && git -C /root/a submodule update --init --recursive && git -C /root/b submodule update --init --recursive" || stop clone
[ "$(git -C /root/a rev-parse HEAD)" = $SHA_A ] && [ "$(git -C /root/b rev-parse HEAD)" = $SHA_B ] || stop "worktree SHAs"
for p in crates/targets/metal/e2e/testdata/prompts.txt crates/e2e/testdata/prompts.txt; do
  [ -f /root/b/$p ] && { cp /root/b/$p /work/prompts.txt; break; }
done
[ -s /work/prompts.txt ] || stop "no e2e prompts file at B"
# Each side's CLI by its layout: `cuda` from backends/cuda, or `scr --device cuda` from the root
# workspace before backends/cuda existed.
for side in a b; do
  if [ -f /root/$side/backends/cuda/Cargo.toml ]; then
    cd /root/$side/backends/cuda
    run build-$side cargo build --release --features nvcc,serve,bench,model/$STEM,quant/fp8-dynamic-per-channel || stop "cuda build ($side)"
    eval "BIN_$side=/root/$side/backends/cuda/target/release/cuda; DEV_$side="
  else
    cd /root/$side
    run build-$side cargo build --release -p scratchy-cli --features nvcc,serve,bench,model/$STEM,quant/fp8-dynamic-per-channel || stop "scr build ($side)"
    eval "BIN_$side=/root/$side/target/release/scr; DEV_$side='--device cuda'"
  fi
  bin=BIN_$side
  ldd ${!bin} > /results/ldd-$side.txt 2>&1
  run ldd-$side sh -c "cat /results/ldd-$side.txt; ! grep -i -E 'cublas|cudnn' /results/ldd-$side.txt" || stop "the $side binary links cuBLAS or cuDNN"
done
cd /work
SFLAGS="--port 8000 --max-num-seqs 1 --dtype bfloat16 --kv-cache-dtype fp8"

# ── 2. the A/B: A1, B1, A2, B2, each a fresh server on this GPU ─────
CELLS="128:128 2048:128"
for rs in a:1 b:1 a:2 b:2; do
  side=${rs%%:*}; rn=${rs##*:}; tag=$side-r$rn; bin=BIN_$side; dev=DEV_$side; SCR=${!bin}
  gpu_clean "before serve $tag" || gpu_wait_clean "before serve $tag"
  $SCR serve $CK ${!dev} $SFLAGS >/logs/serve-$tag.log 2>&1 &
  SP=$!
  waitready 360 $SP || { tail -40 /logs/serve-$tag.log | tee -a /logs/exits.txt; kill $SP 2>/dev/null; stop "serve $tag did not come up"; }
  echo "$(date +%T) serve $tag up" | tee -a /logs/exits.txt
  for c in $CELLS; do
    cin=${c%%:*}; cout=${c##*:}
    A="bench serve --base-url http://127.0.0.1:8000 --tokenizer $CK --dataset-name random --ignore-eos --max-concurrency 1 --request-rate inf --seed 0 --temperature 0 --input-len $cin --output-len $cout"
    run warm-$tag-$cin-$cout $SCR $A --num-prompts 2 || { kill_server $SP $tag; stop "warm-up $tag $cin x $cout"; }
    run cell-$tag-$cin-$cout $SCR $A --num-prompts 10 --save-result --result-dir /results --result-filename ab-$tag-$cin-$cout.json || { kill_server $SP $tag; stop "cell $tag $cin x $cout"; }
    [ -s /results/ab-$tag-$cin-$cout.json ] || { kill_server $SP $tag; stop "cell $tag $cin x $cout wrote no result"; }
    nvidia-smi --query-gpu=memory.used,clocks.sm,temperature.gpu --format=csv,noheader >> /results/gpu-during-ab.txt
  done
  if [ $rn = 1 ]; then
    run coh-$side python3 -I /work/coh.py scr $CK /results/coherence-$side.json || { kill_server $SP $tag; stop "coherence $side"; }
    if [ $MODEL = gemma ]; then
      # Report-only (never a stop): the prompt ids as this server encodes them, and the
      # chat-templated coherence (the template adds Gemma's <bos>).
      run pids-scr-$side python3 -I /work/pids.py scr $side /results/prompt-ids-scr-$side.json
      grep -h "^PROMPT_IDS" /logs/pids-scr-$side.log | tee -a /logs/exits.txt
      run coh-chat-scr-$side python3 -I /work/coh.py scr-chat $CK /results/coherence-chat-scr-$side.json
    fi
  fi
  kill_server $SP $tag
done

# ── 3. vLLM in the same pod: the D2 reference ───────────────────────
wait $VPIP
grep -q "EXIT vllm-pip 0" /logs/exits.txt || stop "vllm pip install"
/opt/vllm/bin/pip list 2>/dev/null | grep -i -E '^(vllm|torch|transformers) ' > /results/vllm-versions.txt
gpu_clean "before vllm" || gpu_wait_clean "before vllm"
PATH=/opt/vllm/bin:$PATH /opt/vllm/bin/vllm serve $CK $SFLAGS --max-model-len 8192 >/logs/vllm-serve.log 2>&1 &
VP=$!
waitready 480 $VP || { tail -40 /logs/vllm-serve.log | tee -a /logs/exits.txt; kill $VP 2>/dev/null; stop "vllm did not come up"; }
echo "vllm up" | tee -a /logs/exits.txt
run coh-vllm /opt/vllm/bin/python -I /work/coh.py vllm $CK /results/coherence-vllm.json || { kill_server $VP vllm; stop "vllm coherence"; }
if [ $MODEL = gemma ]; then
  # Report-only, as scratchy's: vLLM's prompt ids, and its chat-templated coherence (the reference).
  run pids-vllm /opt/vllm/bin/python -I /work/pids.py vllm ref /results/prompt-ids-vllm-ref.json
  grep -h "^PROMPT_IDS" /logs/pids-vllm.log | tee -a /logs/exits.txt
  run coh-chat-vllm /opt/vllm/bin/python -I /work/coh.py vllm-chat $CK /results/coherence-chat-vllm.json
fi
kill_server $VP vllm

if [ $MODEL = gemma ]; then
  # The chat-templated coherence per side against vLLM's (report-only).
  for side in a b; do
    mkdir -p /work/chat-$side
    cp /results/coherence-chat-scr-$side.json /work/chat-$side/coherence-scr-chat.json 2>/dev/null
    cp /results/coherence-chat-vllm.json /work/chat-$side/coherence-vllm-chat.json 2>/dev/null
    run cmp-chat-$side python3 -I /work/cmp.py --chat $TAG-$side /work/chat-$side /results/coherence-chat-$side.md
    n=$(grep -h -o "Coherence (chat): [0-9]*/[0-9]*" /results/coherence-chat-$side.md 2>/dev/null | grep -o "[0-9]*/[0-9]*")
    echo "COH_CHAT $side ${n:-missing}" | tee -a /logs/exits.txt
  done
fi

# ── 4. the report ───────────────────────────────────────────────────
run report python3 -I /work/report.py /results /work/baseline-coherence.json /results/ab.md
# The gate for a change that must not move a bit: every coherence prompt's tokens match between A
# and B. Printed whatever the change; a change that moves bits on purpose reads it as a report.
python3 -I - <<'PY' 2>&1 | tee /results/ab-tokens.txt | tee -a /logs/exits.txt
import json
a = json.load(open("/results/coherence-a.json"))["records"]; b = json.load(open("/results/coherence-b.json"))["records"]
bad = 0
for ra, rb in zip(a, b):
    ta, tb = ra.get("token_ids") or [], rb.get("token_ids") or []
    d = next((i for i, (x, y) in enumerate(zip(ta, tb)) if x != y), None if len(ta) == len(tb) else min(len(ta), len(tb)))
    bad += d is not None
    print(f"AB_TOKENS prompt {ra['idx']}: {'identical' if d is None else f'first difference at {d}'} ({len(ta)} vs {len(tb)} tokens)")
print(f"GATE A vs B tokens identical: {len(a) - bad}/{len(a)} prompts {'PASS' if bad == 0 and len(a) == len(b) and a else 'FAIL'}")
PY
cp /logs/gpu-hygiene.txt /results/
grep -h -E "delta |drift|Coherence, |Problems|^- missing|coherence" /results/ab.md | tee -a /logs/exits.txt
echo MODEL_DONE >> /logs/exits.txt
