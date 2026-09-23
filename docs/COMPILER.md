# scratchy's whole-forward DSL compiler

## The idea

Most inference engines hand-write a Python/CUDA module per model architecture,
then rely on a runtime graph executor to pick kernels. This project takes the
opposite approach: **you write the math once, declaratively, and the compiler
emits the whole forward pass ahead of time.**

A model architecture is a single `#[forward(...)]` function whose body reads
like the math itself. Here is the *entire* definition of LLaMA
(`crates/models/arch/dsl/llama.rs.in`):

```rust
#[forward]
fn llama() {
    hidden_states = embed(input_ids, embed_tokens);
    for layer in 0..num_hidden_layers {
        normed = rmsnorm(hidden_states, input_layernorm[layer]);
        q = gemm(normed, self_attn.q_proj[layer]);
        k = gemm(normed, self_attn.k_proj[layer]);
        v = gemm(normed, self_attn.v_proj[layer]);
        (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer]);
        attn = attention(q, k, v, kv_cache[layer], block_table);
        oproj = gemm(attn, self_attn.o_proj[layer]);
        hidden_states = add(oproj, hidden_states);

        normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer]);
        gate = silu(gemm(normed2, mlp.gate_proj[layer]));
        up = gemm(normed2, mlp.up_proj[layer]);
        down = gemm(gate * up, mlp.down_proj[layer]);
        hidden_states = add(down, hidden_states);
    }
    normed = rmsnorm(hidden_states, norm);
    logits = gemm(normed, lm_head);
}
```

That's the whole architecture. No weight wiring, no kernel selection, no
scratch-buffer management, no per-shape dispatch tables — the compiler derives
all of it.

## "Whole-forward"

The compiler reasons about the **entire** forward pass — embedding through
`lm_head` — as one unit, rather than one kernel at a time. Because it sees
global dataflow, it can fuse across operation boundaries (e.g. folding a
residual `add` into the preceding GEMM, or fusing add + RMSNorm + the next
GEMM), reuse scratch slots across the whole tape, and schedule shared memory and
streams with full-graph knowledge. Kernel selection is *solved*, not guessed at
runtime.

## What it compiles to

For every model the macro discovers, it emits a **canonical**: a concrete
`Weights` struct, a weight loader, and a static forward dispatch table — one per
`(model size, quantization preset, tensor-parallel world size)` tuple. A single
architecture crate typically expands into dozens of canonicals.

The compiler runs these passes at macro-expansion time:

1. **Parse** the DSL body into an AST.
2. **Shape-infer** every tensor from config bounds (`hidden_size`,
   `num_attention_heads`, `num_hidden_layers`, `vocab_size`, …), reading the
   verbatim HuggingFace `config.json` files in the crate's `configs/` directory.
3. **Classify** operations into atoms (GEMM, RoPE, Attention, RMSNorm, …).
4. **Build** the dataflow graph and **unroll** layer loops.
5. **Solve** for the best kernel variant (CUTLASS / FlashInfer / …) at each
   discrete workload point — this is the fusion-synthesis ("FUF") pass.
6. **Schedule** registers, shared memory, and scratch-tile arenas.
7. **Emit** monomorphic Rust dispatch code plus static instruction tables, and
   for CUDA/Metal emit the kernel sources to be built into the binary.

## Workload-aware bucket dispatch

Instead of selecting kernels at runtime, the compiler precomputes optimal kernel
sequences for discrete **buckets** of the workload space — token count `m` and
KV-cache span `sk`. Both ladders default (the standard `sk` ladder is
`[128, 512, 2048, 8192]`); an arch overrides either via `#[forward(workloads =
[...], sk_buckets = [...])]` when its dispatch wants different breakpoints. Each canonical
gets a static `FORWARD_TABLE` mapping `(m, sk)` ranges to an instruction
sequence and its per-bucket scratch-tile layout. At load time, buckets are
pruned to fit the target device's memory — small GPUs simply drop the
large-batch buckets. (The tiny `num_tokens` buckets in the LLaMA comment exist
to serve speculative-decoding draft chains efficiently.)

## Backends and hardware

- **CUDA** — eager driver-side kernel dispatch via `cudarc` (no `nvcc` at
  runtime). Target profiles with empirical cost tables for L4/L40S (sm_89) and
  H100 (sm_90), including **Hopper-native FP8** GEMM (`cutlass_scaled_mm`).
  Tensor parallelism via NCCL (feature-gated, world sizes 1/2/4/8) with
  piecewise CUDA-graph capture across collective boundaries.
- **Metal** — synthesized Metal Shading Language kernels with a command-buffer
  worker pool, and target profiles for Apple M1–M4. The CUDA and Metal backends
  are mutually exclusive build-time features.

## Quantization

Quantization presets are JSON fragments that deep-merge onto a dense base config
and cross-multiply with every model size. Supported presets live in
`crates/models/quantization/presets/` and include AWQ, GPTQ (sym / desc_act),
BitsAndBytes NF4, and FP8 (block / dynamic / static per-tensor), plus
Metal-side affine 4-bit and NVFP4.

## Multimodal

Vision-language models use a sibling `#[vision_forward]` macro for the encoder,
with projected patch embeddings spliced into the language token sequence and
2D/3D MRoPE position handling (Qwen2/2.5-VL).

## Adding a model architecture

All arches share the one `scratchy-models` crate (`crates/models/arch/`):

1. Write the math as one `#[forward(...)]` (or `#[vision_forward(...)]`) carrier
   in `crates/models/arch/dsl/<arch>.rs.in`. The carrier is a bare
   `fn <arch>()` and its body is the math — nothing else goes in this file.
2. Drop the verbatim HuggingFace `config.json` for each model size into
   `crates/models/arch/configs/<arch>/`.
3. Add a `weights.json` only for weight shapes dataflow can't infer, an
   optional `arch.json` for anything about the arch the configs don't already
   say (safetensors layout, gain dtype, bound defaults, the vision `params`
   schema — see [`MODELS.md`](MODELS.md)), and an optional
   `quantizations.json` to opt into presets.
4. Add the `arch-<name>` feature (and list it under `all-arches`) plus one
   `<stem> = ["arch-<name>"]` feature per config (no `model-` text prefix —
   `crates/cli/scr/Cargo.toml` reaches it as `model/<stem>` via the `model`
   dependency alias; add it to a new bare-`<arch>` list and to `all` too) in
   `crates/models/arch/Cargo.toml`, plus a gated `pub mod <arch>` in
   `crates/models/arch/src/lib.rs`.

See [`MODELS.md`](MODELS.md) for the full recipe.
