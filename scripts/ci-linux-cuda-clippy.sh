#!/usr/bin/env bash
# CLAUDE.md's x86_64 linux-cuda clippy gate, in a container.
set -euo pipefail
cd "$(dirname "$0")/.."
docker run --rm --platform linux/amd64 -v "$PWD":/work -w /work \
  -e SCRATCHY_GPU=h100 -e CUDA_COMPUTE_CAP=90 -e SCRATCHY_SKIP_CUDA_KERNELS=1 \
  nvidia/cuda:12.9.1-devel-ubuntu22.04 bash -c '
    apt-get update -qq && apt-get install -y -qq curl git pkg-config libssl-dev ca-certificates >/dev/null 2>&1
    curl -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal -c clippy >/dev/null 2>&1
    export PATH=/root/.cargo/bin:/usr/local/cuda/bin:$PATH
    cargo clippy --workspace --features cuda,scratchy-models/all \
      --exclude scratchy-target-metal \
      --exclude scratchy-target-metal-compiler -- -D warnings'
