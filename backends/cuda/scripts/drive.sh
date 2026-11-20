#!/bin/bash
# usage: drive.sh POD SCRIPT PIN|- [script args...]
# Runs SCRIPT (kernels.sh or e2e.sh) in a fresh H100 pod named POD, waits for its last line, copies
# its logs and results to $WORKDIR/logs/POD, and deletes the pod, always. One pod per call.
#
# Environment (README.md): NAMESPACE (required), KUBE_CONTEXT (default: kubectl's current one),
# WORKDIR (default: the current directory), the pod template's fields (IMAGE, NODE_SELECTOR_KEY,
# NODE_SELECTOR_VALUE, TOLERATION_KEY, TOLERATION_VALUE, HF_SECRET), and what the run script reads
# (REPO_URL, SHA, BASE, SHA_A, SHA_B, MODEL, FETCH_REFS), handed to the pod as /work/env.sh.
# BASELINE_COHERENCE, if set, is a local coherence-scr.json e2e.sh compares its tokens against.
set -o pipefail
SD=$(cd "$(dirname "$0")" && pwd)
: "${NAMESPACE:?set NAMESPACE}"
WORKDIR=${WORKDIR:-$PWD}
export IMAGE=${IMAGE:-nvidia/cuda:12.9.1-devel-ubuntu22.04}
export NODE_SELECTOR_KEY=${NODE_SELECTOR_KEY:-cloud.google.com/gke-accelerator}
export NODE_SELECTOR_VALUE=${NODE_SELECTOR_VALUE:-nvidia-h100-80gb}
export TOLERATION_KEY=${TOLERATION_KEY:-nvidia.com/gpu}
export TOLERATION_VALUE=${TOLERATION_VALUE:-present}
export HF_SECRET=${HF_SECRET:-hf-token}
k(){ kubectl ${KUBE_CONTEXT:+--context "$KUBE_CONTEXT"} -n "$NAMESPACE" "$@"; }
POD=$1; SCR=$2; PIN=$3; shift 3
[ -n "$POD" ] && [ -f "$SCR" ] && [ "$PIN" = - -o -f "$PIN" ] || { echo "usage: drive.sh POD SCRIPT PIN|- [args...]"; exit 2; }
abs(){ echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; }
SCR=$(abs "$SCR"); [ "$PIN" = - ] || PIN=$(abs "$PIN")
[ -z "$BASELINE_COHERENCE" ] || BASELINE_COHERENCE=$(abs "$BASELINE_COHERENCE")
cd "$WORKDIR" || exit 1
mkdir -p logs/$POD
render=logs/$POD/pod.yml
sed "s|\${POD}|$POD|; s|\${NAMESPACE}|$NAMESPACE|; s|\${IMAGE}|$IMAGE|; s|\${NODE_SELECTOR_KEY}|$NODE_SELECTOR_KEY|; s|\${NODE_SELECTOR_VALUE}|$NODE_SELECTOR_VALUE|; s|\${TOLERATION_KEY}|$TOLERATION_KEY|; s|\${TOLERATION_VALUE}|$TOLERATION_VALUE|; s|\${HF_SECRET}|$HF_SECRET|" \
  "$SD/pod-template.yml" > $render || exit 1
grep -q "^  name: $POD$" $render && ! grep -q '\${' $render || { echo "pod template did not render: $render"; exit 1; }
env_sh=logs/$POD/env.sh
for v in REPO_URL SHA BASE SHA_A SHA_B MODEL FETCH_REFS; do
  [ -n "${!v}" ] && printf 'export %s=%q\n' "$v" "${!v}"
done > $env_sh
alive(){ for i in 1 2 3 4 5; do k get pod $POD >/dev/null 2>&1 && return 0; /bin/sleep 60; done; return 1; }
! k get pod $POD >/dev/null 2>&1 || { echo "NOT RUN: $POD already exists"; exit 1; }
# From here the pod is this run's: delete it on every exit.
trap 'k delete pod $POD --ignore-not-found' EXIT
k apply -f $render && k wait --for=condition=Ready pod/$POD --timeout=900s || exit 1
k exec $POD -- mkdir -p /work
[ "$PIN" != - ] && k cp "$PIN" $POD:/work/pin.txt
[ -n "$BASELINE_COHERENCE" ] && k cp "$BASELINE_COHERENCE" $POD:/work/baseline-coherence.json
k cp $env_sh $POD:/work/env.sh
k cp "$SCR" $POD:/work/run.sh
for f in ab.py coh.py cmp.py report.py pids.py; do k cp "$SD/$f" $POD:/work/$f; done
k exec $POD -- sh -c "nohup bash /work/run.sh $* >/work/run.out 2>&1 &"
until k exec $POD -- grep -q -E "MODEL_DONE|K_DONE|STOP" /logs/exits.txt 2>/dev/null; do alive || { echo "POD GONE: $POD"; k get events --field-selector involvedObject.name=$POD 2>&1 | tail -5; exit 1; }; /bin/sleep 45; done
k exec $POD -- sh -c 'cat /logs/exits.txt' > logs/$POD/exits.txt
for f in $(k exec $POD -- sh -c 'ls /logs'); do k exec $POD -- cat /logs/$f > logs/$POD/$f 2>/dev/null; done
mkdir -p logs/$POD/results; k exec $POD -- tar cf - -C / results 2>/dev/null | tar xf - -C logs/$POD/results 2>/dev/null
k delete pod $POD; k get pods
echo POD_DONE $POD
