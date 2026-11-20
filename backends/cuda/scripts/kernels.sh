#!/bin/bash
# The GPU suite at one SHA, in an H100 pod (pod-side; run through drive.sh, README.md):
#   REPO_URL=… SHA=… [BASE=…] ./drive.sh scratchy-cuda-k kernels.sh -
# 1. the build with `nvcc`, the ThunderKittens source check, and ptxas over every kernel with
#    build.rs's flags and its generated constants header (every compile clean, 0 spills required);
# 2. every CUDA test binary built with `nvcc`, at SHA and, when BASE is set, at BASE, the two logs
#    compared by ab.py (bit and ratio lines; timings dropped);
# 3. the three FP8 models' tape tests (no GPU);
# 4. the CLI built against the kernels, linking no cuBLAS or cuDNN.
# SANITIZE (optional): "<test binary>:<test name> ..." to run memcheck, synccheck and racecheck on.
# Either SHA's layout builds: backends/cuda (this workspace), or the CUDA crates under the root
# workspace (crates/targets/cuda, before backends/cuda existed).
set -o pipefail
mkdir -p /logs /work
[ -f /work/env.sh ] && . /work/env.sh
export PATH=/usr/local/cuda/bin:/root/.cargo/bin:$PATH
export CUDA_COMPUTE_CAP=90 SCRATCHY_GPU=h100
run() { n=$1; shift; s=$(date +%s); "$@" >/logs/$n.log 2>&1; rc=$?; e=$(date +%s); echo "EXIT $n $rc WALL=$((e-s))s" | tee -a /logs/exits.txt; return $rc; }
stop() { echo "STOP: $1" | tee -a /logs/exits.txt; echo K_DONE >> /logs/exits.txt; exit 1; }
[ -n "$REPO_URL" ] && [ -n "$SHA" ] || stop "REPO_URL and SHA must be set"
# A test binary from cargo's own "Executable" line in the build logs (newer cargo puts it under build/, not deps/).
binof(){ echo "$1/$(grep -h -o "Executable tests/$2\.rs ([^)]*)" /logs/*build*.log 2>/dev/null | tail -1 | sed 's/.*(\(.*\))/\1/')"; }
gpu_clean() {
  local procs used
  procs=$(nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader)
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
  echo "$(date +%T) $1 used_mib=$used procs=[$procs]" >> /logs/gpu-hygiene.txt
  [ -z "$procs" ] && [ "${used:-99999}" -lt 1024 ]
}
# The CUDA workspace of a checkout, and its kernels: backends/cuda, or the root before it existed.
ws(){ [ -f $1/backends/cuda/Cargo.toml ] && echo $1/backends/cuda || echo $1; }
kernels_of(){ [ -f $1/backends/cuda/Cargo.toml ] && echo $1/backends/cuda/kernels || echo $1/crates/targets/cuda/kernels; }
# Every CUDA test binary built with `nvcc`: the target crate's, and the worker's (which, in
# backends/cuda, links the models crate, so it is built at one model through the CLI's aliases).
suite(){
  local w; w=$(ws $1); cd $w
  if [ $w != $1 ]; then
    cargo test --release -p scratchy-target-cuda --features nvcc -- --nocapture --test-threads=1 &&
    cargo test --release -p scratchy-target-cuda-worker -p scratchy-target-cuda-cli \
      --features nvcc,model/qwen3.8-27b-text-only,quant/fp8-dynamic-per-channel -- --nocapture --test-threads=1
  else
    cargo test --release -p scratchy-target-cuda -p scratchy-target-cuda-worker --features nvcc -- --nocapture --test-threads=1
  fi
}

run apt    sh -c 'apt-get update -qq && apt-get install -y -qq git curl jq pkg-config libssl-dev ca-certificates build-essential python3 binutils' || stop apt
run rustup sh -c 'curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -c clippy,rustfmt' || stop rustup
run env    sh -c 'rustc --version; nvidia-smi; nvidia-smi --query-gpu=pci.bus_id,clocks.max.sm --format=csv; nvcc --version; compute-sanitizer --version'
gpu_clean "pod start" || stop "GPU busy at pod start"
run clone  sh -c "git clone $REPO_URL /root/pr && cd /root/pr && git fetch origin ${FETCH_REFS:-} && for s in $SHA $BASE; do git cat-file -e \$s^{commit} 2>/dev/null || git fetch origin \$s; done && git checkout --detach $SHA && git submodule update --init --recursive" || stop clone
if [ -n "$BASE" ]; then
  run base sh -c "cd /root/pr && git worktree add --detach /root/base $BASE && git -C /root/base submodule update --init --recursive" || stop base
  [ "$(git -C /root/base rev-parse HEAD)" = $BASE ] || stop "the base worktree is not at BASE"
fi
[ "$(git -C /root/pr rev-parse HEAD)" = $SHA ] || stop "the checkout is not at SHA"
W=$(ws /root/pr); cd $W
run build cargo build --release -p scratchy-target-cuda --features nvcc || stop "build"
run tkcheck cargo test -p scratchy-target-cuda --test kernels_use_thunderkittens -- --nocapture

# 1. ptxas -v over every .cu with build.rs's flags, in its order, and its `-include` of the
#    generated constants header (`scratchy_consts.cuh`, which the build above wrote into its
#    OUT_DIR); `-Xptxas -v` is the one flag added. A SHA from before the header has none and none of
#    its kernels reads one. Every compile clean and 0 spills required.
nvccflags="-std=c++20 -O3 --use_fast_math --expt-extended-lambda --expt-relaxed-constexpr -forward-unknown-to-host-compiler -Xcompiler=-Wno-psabi -Xcompiler=-fno-strict-aliasing -Xcompiler=-fPIC -DNDEBUG -lineinfo -ftemplate-backtrace-limit=0 -DKITTENS_SM90 -gencode arch=compute_90a,code=sm_90a"
KD=$(kernels_of /root/pr); mkdir -p /tmp/ptx
# cargo writes build-script output under build/<crate>-<hash>/out (older) or build/<crate>/<hash>/out (newer).
HDR=$(find $W/target/release/build -path "*scratchy-target-cuda*" -name scratchy_consts.cuh -print 2>/dev/null | xargs -r ls -t 2>/dev/null | head -1)
[ -z "$HDR" ] && [ -f $W/crates/cuda/src/kernel_consts.rs ] && stop "the build wrote no scratchy_consts.cuh"
include=${HDR:+-include $HDR}
echo "PTXAS header=${HDR:-none}" | tee -a /logs/exits.txt
for f in $KD/*.cu; do
  k=$(basename $f .cu)
  nvcc $nvccflags -I $KD/thunderkittens/include -I $KD/thunderkittens/prototype -I $KD $include -c $f -o /tmp/ptx/$k.o -Xptxas -v >/logs/ptxas-$k.log 2>&1
  echo "EXIT ptxas-$k $?" >> /logs/exits.txt
done
cat /logs/ptxas-*.log > /logs/ptxas-all.log
# Fails on a compile that failed, on a spill, and on a report with no `stack frame` line at all
# (ptxas printed nothing, so nothing was checked).
run spills bash -c 'bad=$(grep "^EXIT ptxas-" /logs/exits.txt | grep -vc " 0$"); frames=$(grep -c "stack frame" /logs/ptxas-all.log); spilled=$(grep "stack frame" /logs/ptxas-all.log | grep -v "0 bytes spill stores, 0 bytes spill loads"); echo "ptxas_nonzero_exit=$bad stack_frames=$frames"; [ -n "$spilled" ] && echo "$spilled"; [ "$bad" = 0 ] && [ "$frames" -gt 0 ] && [ -z "$spilled" ]'

# 2. The GPU suite, at SHA and at BASE.
run suite-build sh -c "cd $W && cargo test --release -p scratchy-target-cuda --features nvcc --no-run" || stop "suite build"
run suite-pr suite /root/pr
grep -h -E "^test result|FAILED$" /logs/suite-pr.log | sort | uniq -c | tee -a /logs/exits.txt
if [ -n "$BASE" ]; then
  run suite-base suite /root/base
  run ab python3 -I /work/ab.py /logs/suite-base.log /logs/suite-pr.log
fi
cd $W
for st in $SANITIZE; do
  b=${st%%:*}; t=${st#*:}; BIN=$(binof $W $b)
  [ -x "$BIN" ] || { echo "SANITIZE: no $b test binary in the build logs" | tee -a /logs/exits.txt; continue; }
  for tool in memcheck synccheck racecheck; do
    extra=""; [ $tool = synccheck ] && extra="--num-cuda-barriers 65536"
    run san-$tool-$t timeout 3600 compute-sanitizer --tool $tool $extra --error-exitcode 1 $BIN --exact $t --test-threads=1
  done
done
grep -h "ERROR SUMMARY" /logs/san-*.log 2>/dev/null | sort | uniq -c | sed 's/^/SAN /' | tee -a /logs/exits.txt

# 3. The FP8 models' tape tests (no GPU).
for t in "27b_fp8 qwen3.8-27b-text-only" "35b_fp8 qwen3.6-35b-a3b" "gemma4_fp8 gemma-4-26b-a4b-it"; do
  n=${t%% *}; s=${t#* }
  if [ $W != /root/pr ]; then
    run tapes-$n cargo test -p scratchy-target-cuda-models --features scratchy-models/$s,scratchy-quantizations/fp8-dynamic-per-channel --test cuda_tapes_$n
  else
    run tapes-$n cargo test -p scratchy-models --features cuda,$s,scratchy-quantizations/fp8-dynamic-per-channel --test cuda_tapes_$n
  fi
  echo "TAPES $n failed=[$(grep -h -E "^test .* FAILED$" /logs/tapes-$n.log | sed 's/^test //; s/ \.\.\. FAILED$//' | sort | tr '\n' ' ')] $(grep -h "^test result" /logs/tapes-$n.log)" | tee -a /logs/exits.txt
done

# 4. The CLI builds against the kernels and links no cuBLAS.
if [ $W != /root/pr ]; then
  run cli cargo build --release --features nvcc,serve,bench,model/gemma-4-26b-a4b-it,quant/fp8-dynamic-per-channel; CLI=$W/target/release/cuda
else
  run cli cargo build --release -p scratchy-cli --features nvcc,serve,bench,model/gemma-4-26b-a4b-it,quant/fp8-dynamic-per-channel; CLI=$W/target/release/scr
fi
run ldd sh -c "ldd $CLI; ! ldd $CLI | grep -i -E 'cublas|cudnn'"
cp /logs/gpu-hygiene.txt /logs/gpu-hygiene-final.txt 2>/dev/null
echo K_DONE >> /logs/exits.txt
