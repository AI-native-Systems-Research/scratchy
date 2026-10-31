#!/usr/bin/env bash
# ABBA A/B for spyre startup time (#1016). Two `scr` binaries — typically a main
# build and a PR build of the same model scope — run the identical `scr chat`
# bench command in ABBA order (A B B A per round), because a single back-to-back
# run of one side can ride a warmed device or a settled VFIO state the other
# side then pays for.
#
# Runs on a Spyre card pod (senlib under /opt/ibm/spyre). Not a Mac script: the
# binaries need the device.
#
# usage: bench_spyre_startup.sh <scrA> <scrB> [rounds=3]
#   scrA  path to the first binary (labelled A, e.g. main)
#   scrB  path to the second binary (labelled B, e.g. the PR)
#
# Each run: the recorded settle (8 s, avoids /dev/vfio busy between card runs),
# then one `scr chat` process; `startup` and `TTFT` are pulled from --bench.
set -euo pipefail

A=${1:?usage: bench_spyre_startup.sh <scrA> <scrB> [rounds]}
B=${2:?usage: bench_spyre_startup.sh <scrA> <scrB> [rounds]}
R=${3:-3}

export HF_HOME=${HF_HOME:-/work/.cache/huggingface}
export LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-/opt/ibm/spyre/senlib/lib:/opt/ibm/spyre/runtime/lib:/opt/ibm/spyre/deeptools/lib}
export DEEPTOOLS_PATH=${DEEPTOOLS_PATH:-/opt/ibm/spyre/deeptools/share}

MODEL=RedHatAI/granite-3.1-2b-instruct-FP8-dynamic

run_one() {
    local bin=$1
    # The recorded settle between card runs: avoids /dev/vfio busy.
    sleep 8
    "$bin" chat "$MODEL" --system-prompt "" \
        -q "write about the history of france" --bench --max-tokens 64 \
        --max-model-len 69632 2>&1 \
        | grep -E '^startup|^TTFT' | awk '{print $2, $1}'
}

echo "A=$A"
echo "B=$B"
echo "rounds=$R (ABBA per round)"
for round in $(seq 1 "$R"); do
    for side in A B B A; do
        bin=$([ "$side" = A ] && echo "$A" || echo "$B")
        run_one "$bin" | while read -r ms what; do
            echo "round$round $side $what ${ms%,}"
        done
    done
done
