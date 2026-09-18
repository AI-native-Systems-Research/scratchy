# What Does "Reuse" Mean in the Age of AI?

Every generation of programmers has been told the same thing: don't write it,
reuse it. What quietly changes, generation to generation, is how much of
somebody else's decisions you have to carry along with the part you wanted.

It used to be all of them. A shared library arrived as a compiled `.so` and a
header file — an opaque box whose insides your compiler couldn't see, let alone
improve. Source-level languages like Python and JavaScript opened the box, but
you still hauled the whole thing around; a `node_modules` directory is a
monument to that. Bundlers were the first tools allowed to throw some of it
away, though tree-shaking only deletes code nobody mentions by name. Then Rust
and Go pushed specialization down into the semantics, and you get the fact that
makes the era legible: a Go binary has no shared-library dependencies. Not
because it's statically linked — because every capability arrived as *source*
and left as machine code specialized for that one program.

Fifty years of compiler engineering, and it all stops at a wall nobody thought
to name. The code itself is sacred. A compiler may delete your dependency's
unused parts and specialize its generic ones, but it may never restructure it.
So the generality of the libraries you depend on is your generality too, whether
or not you ever wanted it.

That's the wall AI knocks down — not by making compilers smarter, but by making
faithful transcription cheap. When you can re-express someone's *algorithm*
inside your own structure in an afternoon instead of over two quarters of pull
requests, the thing you're reusing stops being the module. It's the idea. And
once ideas are the unit, every project gets to be bespoke.

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
