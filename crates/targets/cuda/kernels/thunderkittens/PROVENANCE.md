# `crates/targets/cuda/kernels/thunderkittens/` — ThunderKittens, vendored

## What is here

ThunderKittens is HazyResearch's header-only CUDA C++20 tile library: register and shared-memory
tile types, warp and warpgroup operations, the Hopper tensor-core (WGMMA) and copy-engine (TMA)
wrappers, and the `lcf` producer/consumer kernel template. The CUDA kernels in `../*.cu` include it;
`build.rs` passes `include/` and `prototype/` to nvcc as `-I` paths. Nothing in this directory is
compiled on its own.

| path | what it is |
|---|---|
| `include/` | `kittens.cuh` and everything it includes (115 files) |
| `prototype/` | the kernel templates, `lcf` among them (11 files) |
| `LICENSE` | upstream's MIT license |

## Where it came from

- Upstream: <https://github.com/HazyResearch/ThunderKittens>
- Commit: `67845f5fa48d05343dbbc1ba1403f11061f08d2d` (2026-09-12)
- License: MIT, "Copyright (c) 2024-2026 HazyResearch" (`LICENSE`). MIT is compatible with this
  repo's Apache-2.0.
- Copied: upstream's `include/`, `prototype/` and `LICENSE`, as they are at that commit.
  Upstream's `demos/`, `kernels/`, `tests/` and `assets/` are left out (unused).
- **No local edits.** The copy is byte-identical to upstream, so a re-pin is a clean diff. Nothing
  inside `include/` is trimmed.

Check it:

```bash
git clone https://github.com/HazyResearch/ThunderKittens /tmp/tk
git -C /tmp/tk checkout 67845f5fa48d05343dbbc1ba1403f11061f08d2d
diff -r /tmp/tk/include   crates/targets/cuda/kernels/thunderkittens/include
diff -r /tmp/tk/prototype crates/targets/cuda/kernels/thunderkittens/prototype
cmp     /tmp/tk/LICENSE   crates/targets/cuda/kernels/thunderkittens/LICENSE
```

All three print nothing.

## Why vendored

- Not fetched at build time: that needs the network during `cargo build`, so builds stop being
  hermetic and fail offline, and the pin would live in code instead of in reviewed files.
- Not a git submodule: the repo has none, and one would need extra steps in every checkout,
  worktree, Docker context and GKE pod.
- Metal does the same with MLX's headers (`crates/targets/metal/shaders/mlx_steel_gemm/` and
  friends).

## Re-pinning

1. `git clone https://github.com/HazyResearch/ThunderKittens /tmp/tk` and
   `git -C /tmp/tk checkout <new SHA>`.
2. Replace `include/` and `prototype/` here with upstream's, and `LICENSE` with upstream's
   (`rm -r include prototype && cp -R /tmp/tk/include /tmp/tk/prototype /tmp/tk/LICENSE .`).
3. Update the SHA and the date above, and the short SHA in `build.rs`'s "too old" message.
4. If upstream's `kernels/gemm/bf16_h100/bf16_h100_gemm.cu` or `kernels/common.mk` changed, carry
   the change into `../gemm_bf16.cu` and `build.rs`'s flags, and update the SHA in
   `gemm_bf16.cu`'s header comment.
5. Rerun the GKE H100 plan (issue AI-native-Systems-Research/scratchy#761): the `nvcc` build, then
   `cargo test -p scratchy-target-cuda --features nvcc --release --test gemm_bf16`. CI never
   builds the `nvcc` feature, so GKE is the only gate.
