#!/bin/bash
# The end-to-end A/B of every FP8 model, one pod each, one after another (drive.sh; README.md):
#   NAMESPACE=… REPO_URL=… SHA_A=… SHA_B=… ./chain.sh
SD=$(cd "$(dirname "$0")" && pwd)
for m in "27b qwen3.8-27b" "35b qwen3.6-35b-a3b" "gemma gemma-4-26b-a4b-it"; do
  tag=${m%% *}; pin=${m#* }
  MODEL=$tag "$SD/drive.sh" "${POD_PREFIX:-scratchy-cuda-e2e}-$tag" "$SD/e2e.sh" "$SD/pins/$pin.txt" 2>&1 | tail -3
done
