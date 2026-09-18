# What "Reuse" Means in the Age of AI

Every generation of programmers has inherited the same directive: don't write
it, reuse it. What has shifted, generation to generation, is how much of
someone else's architectural decisions must be carried along with the component
you actually needed.

For decades, that cost was total. A shared library arrived as a compiled `.so`
and a header file — an opaque artifact whose internals your compiler could
neither inspect nor optimize. Source-level languages like Python and JavaScript
opened the box, but the entire contents still had to be hauled along; a
`node_modules` directory stands as a monument to that reality. Bundlers
introduced selective elimination through tree-shaking, though only for code that
went entirely unmentioned by name. Rust and Go pushed specialization further into
the language semantics, yielding a telling characteristic of the modern era:
a Go binary carries no shared-library dependencies — not because it is statically
linked in the traditional sense, but because every capability arrived as source
and left as machine code specialized for that single program.

Fifty years of compiler engineering, and it all meets the same wall. The code
itself is treated as sacred. A compiler may delete unused portions of a
dependency and specialize its generic abstractions, but it may never restructure
them. The generality of the libraries a project depends on becomes the project's
generality too, whether that generality was ever wanted or not.

That is the wall AI is knocking down — not by making compilers smarter, but by
making faithful transcription cheap. When an algorithm can be re-expressed
inside a team's own structure in an afternoon rather than across two quarters of
pull requests, what is being reused is no longer the module. It is the idea. And
once ideas become the unit of reuse, every project can be bespoke.

## Introducing Scratchy

[Scratchy](https://github.com/AI-native-Systems-Research/scratchy) is the
existence proof of this thesis: a compiler that takes a model architecture, a
HuggingFace `config.json`, and a quantization preset, and emits an inference
server that exists only for that specific combination.

In Scratchy, model architectures are expressed as simple, readable descriptions
of the tensor operations, and Rust's procedural macro system does the rest —
deriving weight wiring, kernel selection, buffer management, and dispatch
tables entirely at compile time. The following is the complete definition of
LLaMA — the entire forward pass:

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
Using this approach, 25 model architectures fit within 1,580 lines of total
code. The result is a 30 MiB binary with a 300 ms warm startup time that is
independent of model size, because there is no graph to construct at load time.
The graph is a compile-time constant.

The serving algorithms — paged KV cache, continuous batching, prefix caching —
are drawn from [vLLM](https://github.com/vllm-project/vllm), transcribed into
Rust and credited by file and line at each use site. Seventy citations to a
repository that does not appear in the build graph. That is what a dependency
looks like when the unit of reuse is an idea.

## Why This Matters for IBM Spyre

Scratchy's first production target is not CUDA. It is the IBM Spyre AIU — and
that is where the design is put to its most meaningful test, because exotic
hardware is precisely where the library era has nothing to offer.

Each Spyre core provides a 2 MiB scratchpad, of which 1,677,721 bytes are
available to user workloads. Every tile of every operation must fit within that
budget. Overflow it and the result is not a helpful error message — it is a
`DtException 1535` on the card, surfacing minutes later, about a tile that can
no longer be inspected. The dominant cost of novel silicon is not writing
kernels; it is the feedback loop from a nameless on-card fault back to the line
of arithmetic that caused it.

Because Scratchy knows every tile size at compile time, that fault becomes a
`cargo build` error on a developer's laptop. A general-purpose runtime is
structurally incapable of providing this guarantee: it does not know the shapes
until it is already running, on the card, where the only channel back to the
developer is an integer.

Supporting silicon with no ecosystem behind it took about as much Rust as
supporting CUDA, and it required zero changes to model code. The same 1,580
lines. The same 22-line LLaMA definition. An 8B model boots in 12 seconds from a
330 MiB image.

## A New Threshold for Novel Silicon

Reuse-by-code quietly imposes a population threshold on what hardware is
permitted to exist. "Support" under that model means a vendor maintaining a
general backend inside someone else's general framework until the market
justifies the headcount. Reuse-by-idea drops that threshold to a single team
with a compiler.

The payoff is not faster inference on existing hardware. It is that novel
silicon — hardware that would never clear the bar of a traditional software
ecosystem — becomes viable at scales where it was not before. Scratchy is an
early demonstration of what that shift makes possible, and a first look at what
inference infrastructure might look like when ideas, not modules, are the unit
of reuse.

---

*Scratchy is Apache-2.0. Serving algorithms derive from vLLM; the Spyre hardware
model from IBM's `torch-spyre` and KTIR. Both Apache-2.0, credited at each
derivation site.*
