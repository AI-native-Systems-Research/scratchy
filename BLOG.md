# What Does "Reuse" Mean in the Age of AI?

For fifty years, software reuse has meant depending on someone else's code. Each
era changed the packaging, and each one let the toolchain specialize a little
more:

| Era | Unit of reuse | Who specializes |
| --- | --- | --- |
| Shared libraries | `.so` + header | nobody — the linker just resolves symbols |
| Source modules | a package | nobody; you carry the whole tree |
| Bundlers | a module graph | the bundler, syntactically |
| Rust / Go | a crate | the compiler, semantically |
| **AI** | **an idea** | **you, structurally** |

A Go binary has no shared-library dependencies — not because it's statically
linked, but because every capability arrived as *source* and left as machine
code specialized for that one program. That's as far as fifty years of compiler
engineering got us, and it stops at a wall nobody named: **the code itself is
sacred.** Compilers may delete and specialize. They may never restructure. So
the generality of your dependencies is your generality, whether you wanted it or
not.

AI breaks that, by making faithful transcription cheap. Once you can re-express
someone's *algorithm* inside your own structure in hours rather than quarters,
the thing you're reusing is no longer the module. It's the idea — and every
project becomes bespoke.

## Scratchy

[Scratchy](https://github.com/AI-native-Systems-Research/scratchy) is our
existence proof: a compiler that takes a model architecture, a HuggingFace
`config.json`, and a quantization preset, and emits an inference server that
exists only for that triple.

Here is the *entire* definition of LLaMA — the whole forward pass:

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

Twenty-two lines. No weight wiring, no kernel selection, no buffer management,
no dispatch tables — Rust's procedural macros derive all of it at compile time.
**25 architectures fit in 1,580 lines** this way. The payoff: a 30 MiB binary,
and 300 ms warm startup *independent of model size*, because there's no graph to
build at load time. The graph is a constant.

The serving algorithms — paged KV cache, continuous batching, prefix caching —
are [vLLM's](https://github.com/vllm-project/vllm), transcribed into Rust and
credited by file and line at each site. Seventy citations to a repository that
isn't in our build graph. That's what a dependency looks like when the unit of
reuse is an idea.

## Why this matters for Spyre

Scratchy's first target isn't CUDA. It's the **IBM Spyre AIU** — and that's the
real test, because exotic hardware is where the library era has nothing to
offer.

Each Spyre core has a 2 MiB scratchpad, of which 1,677,721 bytes are yours.
Every tile of every operation must fit. Overflow it and you don't get an error
message — you get `DtException 1535` on the card, minutes later, about a tile
you can no longer inspect. The dominant cost of novel silicon isn't writing
kernels; it's the feedback loop from a nameless on-card fault back to the line
of math that caused it.

Because scratchy knows every tile size at compile time, that fault becomes a
`cargo build` error on your laptop. A general-purpose runtime structurally
can't do this: it doesn't know the shapes until it's already running, on the
card, where the only channel back to you is an integer.

Spyre support is ~129,000 lines of Rust, twice the size of our CUDA backend, and
it cost **zero lines of model code**. Same 1,580 lines. Same 22-line LLaMA. An
8B model boots in 12 seconds from a 330 MiB image.

Which is the part worth tweeting. Reuse-by-code quietly puts a *population
threshold* on what hardware is allowed to exist — "support" means a vendor
maintaining a general backend inside someone else's general framework until the
market justifies the headcount. Reuse-by-idea drops that threshold to one team
with a compiler.

The payoff isn't faster inference. It's that novel silicon becomes viable at a
scale where it wasn't.

---

*Scratchy is Apache-2.0. Serving algorithms derive from vLLM; the Spyre hardware
model from IBM's `torch-spyre` and KTIR. Both Apache-2.0, credited at each
derivation site.*
