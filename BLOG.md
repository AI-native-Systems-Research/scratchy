# What Does "Reuse" Mean in the Age of AI?

For fifty years, software reuse has meant *depending on someone else's code*.
Every era changed the packaging — the object file, the module, the crate — but
never the premise. You found a library that did approximately what you needed,
you took all of it, and you lived with the parts you didn't want.

That premise just stopped being load-bearing. Here is the history, and then
what replaced it.

## The ladder

Each era of reuse can be read as a single question: *how much specialization is
the toolchain permitted?*

| Era | Unit of reuse | What you carry | Who specializes |
| --- | --- | --- | --- |
| Shared libraries | `.so` + header | the whole library, opaque | nobody — the dynamic linker just resolves symbols |
| Source modules | a package | the whole tree (`node_modules`, site-packages) | nobody |
| Bundlers | a module graph | whatever survives tree-shaking | the bundler, syntactically |
| Rust / Go | a crate, a package | a specialized binary | the compiler, semantically |
| AI | **an idea** | only the part you needed | **you**, structurally |

**Shared libraries** were reuse at its most opaque. You got a symbol table and
a header file. The library's internal structure was not merely unknown to your
compiler; it was unknowable in principle, because it had already been compiled
by someone else, on another machine, under assumptions you could not inspect.
Specialization was zero. If `libpng` had a general-purpose path and you only
ever fed it 8-bit RGBA, you paid for the generality on every call, forever.

**Source-level modules** — Python, Ruby, early Node — opened the box. You could
read the code, patch it, vendor it. But you still carried all of it. A
`node_modules` directory is a monument to this: thousands of files, of which
your program touches a rounding error. Reuse was total or nothing, and the
runtime resolved everything, every time.

**Bundlers** were the first tools permitted to throw some of it away. Webpack,
Rollup, esbuild: trace the import graph from your entry point, keep what's
reachable, drop the rest. This was real progress and a real limit. Tree-shaking
is *syntactic*. It deletes bindings nobody names. It cannot notice that of the
forty branches inside a function you do call, thirty-nine are dead for your
inputs.

**Rust and Go** pushed specialization into the semantics. Monomorphization,
cross-crate inlining, devirtualization, LTO, constant propagation across
library boundaries. The result is the fact that makes the era legible: *a Go
binary has no shared-library dependencies* — and not because it is statically
linked. Static linking is the old era's trick of pasting `.o` files together.
Go's binaries are dependency-free because every capability arrived as **source**
and left as **machine code specialized for this program**. The library boundary
survives in the source tree and evaporates in the artifact.

This is as far as fifty years of compiler engineering got us. It is a long way.
And it stops at a wall that no one in any of those eras thought to name.

## The invariant nobody broke: the code is sacred

Inlining, constant folding, devirtualization, tree-shaking, LTO — every one of
these transformations *preserves the author's structure and semantics*. The
compiler is allowed to **delete** and to **specialize**. It is never allowed to
**restructure**.

That sounds like a technicality. It is the actual ceiling on reuse, and it is
worth saying precisely what it costs you:

> **The generality of your dependencies is your generality, whether or not you
> wanted it.** You inherit the abstraction boundary the original author happened
> to choose, because a compiler may not move it.

Take a concrete case: a modern LLM inference server. vLLM must serve every model
architecture, on every GPU generation, at every batch size, under every
quantization scheme. So vLLM has a runtime graph executor, a kernel-dispatch
layer that picks an implementation per call, and a hand-written Python module
per architecture. Every one of those is the *correct* design for vLLM's
problem.

None of them is correct for *your* problem, if your problem is serving one
model, at one quantization, on one device. You know your shapes. You have known
them since before the process started. The dispatch layer is a machine for
answering a question you already have the answer to — and no compiler in any
era above can remove it, because removing it means *reorganizing the program*.

Your options, historically, were two. Take the library and pay. Or write it
yourself, which meant reimplementing paged attention, continuous batching, and
prefix caching — each a research contribution, each with a decade of subtle
bugs already found and fixed in someone else's repo.

There was, in principle, a third option: contribute upstream. File an issue,
argue for its importance, send a PR, spend six weeks negotiating nuance with
maintainers who are correctly protecting a general-purpose system from your
special case, and wait two quarters for a release. The reason this rarely
happens is not that maintainers are unreasonable. It is that *you are asking a
general system to become less general*, which is exactly the thing it must
refuse.

## What changed

AI broke the invariant. Not by making compilers smarter — by making
**transcription cheap**.

Once you can read a foreign implementation and faithfully re-express its
*algorithm* inside your own structure, at a cost measured in hours rather than
quarters, then the thing you are reusing is no longer the module. It is the
idea.

The `PagedAttention` class is packaging. The paged-attention *algorithm* — KV
cache as a block table, logical-to-physical indirection, copy-on-write across
forked sequences — is the asset. It fits in a paper. What was expensive was
never understanding it; what was expensive was *carrying it*: the class
hierarchy it lived in, the dispatch it assumed, the runtime it needed, the
Python.

So the unit of reuse becomes an idea, and the consequence is that **every
project becomes bespoke**. Not bespoke in the sense of from-scratch — bespoke in
the sense that the specialization boundary is now drawn by *you*, at the point
where it serves your program, rather than inherited from whoever published
first.

Two things worth being honest about, because they are the actual price:

- **You lose upstream's future.** Their bug fixes, their test suite, their
  security patches, their blame log — none of it flows to you anymore. You
  forked an idea, and ideas do not receive updates.
- **Faithful transcription is a discipline, not a vibe.** The failure mode of
  this whole approach is an AI that *reinvents* the algorithm instead of
  *porting* it, produces something that typechecks and runs and is subtly
  wrong, and buries the wrongness under a green test suite. Everything below
  depends on refusing that.

## Scratchy: the existence proof

[Scratchy](https://github.com/AI-native-Systems-Research/scratchy) is an inference-stack compiler
built on exactly this premise. It is worth looking at because the thesis
predicts a specific *shape* of program, and scratchy has that shape.

**The idea is small. The specializer is large. The ratio is the claim.**

Here is the entire definition of the LLaMA architecture in scratchy — the whole
forward pass, embedding through `lm_head`:

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

Twenty-two lines. No weight wiring, no kernel selection, no scratch-buffer
management, no per-shape dispatch table. That is the *idea* of LLaMA, and it is
all that a new architecture costs.

Across 25 architectures — LLaMA, Qwen 2/3, Gemma 2/3/4, DeepSeek V2/V3, Mixtral,
Phi-3, ModernBERT, the VL models — the DSL totals **1,580 lines**. Mistral is 22
lines, Mixtral 20, Granite 23. The machine that specializes them is about
**465,000 lines** of Rust, shared by all 25.

That ratio *is* the thesis. In the module era, adding an architecture meant
adding an architecture's worth of machinery. Here the architecture is the
irreducible idea and nothing else, because the specializer absorbed everything
that was ever generic about it.

**The specialization is structural, which is the part the old eras forbade.**

Scratchy takes a triple as input: the DSL for an architecture, the verbatim
HuggingFace `config.json` for one instance of it, and a JSON quantization
preset. From that triple it emits an inference server that exists only for that
triple.

The mechanism is Rust's procedural macros, and the governing rule is that
**everything is a constant**. `#[forward]` runs the entire compilation pipeline
at macro-expansion time — op tape, layer classes, scratch-slot coloring,
lifetimes, barriers, arena sizes, kernel selection — and emits static
instruction tapes. There is no runtime lowering and no runtime analysis, because
there is nothing left to analyze: the shapes were known before `main`.

Notice what this is. The same mathematical description is *reorganized* per
(model, quantization, device, workload bucket) — not merely pruned, not merely
inlined. Kernel selection is **solved**, not guessed at each call. That is the
transformation the sacred-code invariant made unavailable, and it is available
now only because the input was an idea rather than a library.

The measurable consequences are the ones you'd predict from a program with no
generality left in it: a 30 MiB binary on Metal, a 330 MiB container image for
Spyre, and **300 ms warm startup on Apple Silicon independent of model size** —
there is no graph to build at load time, because the graph is a `static`.

**The provenance: ideas cited, not code linked.**

Scratchy's serving layer — the scheduler, the KV-cache manager, the block pool,
continuous batching, prefix caching, speculative decoding, the OpenAI-compatible
API surface — is vLLM's algorithms, transcribed into Rust. This is not
incidental and it is not hidden. There are **70 source citations to vLLM's
Python** in scratchy's Rust, across roughly 40 distinct upstream files, and they
look like this:

```rust
// Mirror Python vLLM (vllm/config/vllm.py:709-721): force ...
// (matches Python vLLM, where a request is "in prefill" while ...
// Match vLLM: finish such a request as FinishedIgnored ...
// here — matches Python vLLM's `ColumnParallelLinear` weight ...
```

That is what a dependency looks like when the unit of reuse is an idea: a
**citation**, down to the file and line, to a repository that is not in the
build graph. No `Cargo.toml` entry. No vendored tree. No Python. Both projects
are Apache-2.0, and the derivation is credited where it happens rather than in a
NOTICE file nobody reads. vLLM's contributors did the hard part — the
algorithms are theirs, and the comments say so at each site.

The same pattern appears a second time, from a second upstream, and the second
one is where this gets interesting.

And note the complement, which is the sharpest detail in the whole system:
scratchy reads **verbatim, unmodified HuggingFace `config.json` files**. It
rejects the ecosystem's *code* entirely while consuming the ecosystem's *data*
unchanged. Data formats are still worth depending on. Code, increasingly, is
worth reading.

## The hard case: hardware nobody is going to support for you

Scratchy's initial design point is not CUDA. It is the **IBM Spyre AIU** — and
that choice is the sharpest test of the whole thesis, because Spyre is exotic in
the one sense that matters: *the library era has nothing to offer you*.

Start with the machine's own arithmetic. Spyre's HBM layout granularity is a
128-byte **stick**:

```rust
/// HBM stick size in bytes (the Spyre layout granularity).
pub const STICK_BYTES: i64 = 128;
```

In fp16, that stick holds 64 elements. Now recall that the head dimension of
essentially every transformer above 3B parameters is **128**. So the natural
unit of the hardware and the natural unit of the model are off by exactly a
factor of two, and that single mismatch propagates into every tensor layout
decision in the stack. It is not a bug to be fixed; it is the shape of the
machine, and the compiler has to metabolize it.

Then the scratchpad. Each core gets a 2 MiB local store — "LX" — of which the
vendor's own toolchain reserves a fifth:

```rust
pub const LX_CAPACITY_BYTES: u64 = 2 * 1024 * 1024;
pub const DXP_LX_FRAC_AVAIL: f64 = 0.2;
pub const USABLE_LX_BYTES: u64 = 1_677_721;
```

Every tile of every operation in the entire forward pass must fit in
1,677,721 bytes. Overflow it and you do not get an error message. You get, on
the card, after the bundle has been built and shipped and loaded:
**`DtException 1535`** — register-file and scratchpad over-subscription, thrown
from something called `L3DlOpsScheduler`. That number appears 23 times in
scratchy's source. Its siblings appear too: `DtException "Unrecognized opFunc:
multiply"` from `designSpaceConfig.cpp:7713`, and `DtException "Symbol already
reserved"` from `VariableDefinition.cpp:629`.

This is the real cost structure of exotic hardware, and it is not what people
assume. The expensive thing is not writing kernels. **The expensive thing is the
length of the feedback loop from an unnamed on-card fault back to the line of
math that caused it** — an integer, emitted by a closed C++ toolchain, about a
tile you can no longer inspect, minutes or hours after you wrote the code.

Here is what bespoke compilation does to that loop, in the codebase's own words:

> A per-core tile that does not fit `USABLE_LX_BYTES` is an on-card
> DtException 1535 (register-file / scratchpad over-subscription,
> L3DlOpsScheduler) — the whole point of the typed `TimeTile` witness is to turn
> that into a `cargo build` `Err` BEFORE the bundle is ever baked.

A scratchpad overflow on a remote accelerator becomes a **type error on your
laptop**. And this is available *only* because the specialization is total: tile
sizes are not runtime values, they were computed at macro-expansion time from
that one `config.json`, so the fit check is arithmetic on constants rather than
a hope about a future execution. A general-purpose runtime structurally cannot
do this. It does not know the shapes until it is already running, on the card,
where the only channel back to you is the exception number.

**And what got mined here was not code. It was the hardware's law.**

Nobody can derive that the usable scratchpad is 1,677,721 bytes. No datasheet
states it, because it is the product of a 2 MiB pad and a toolchain's 20%
working-set reservation — two facts that live in two different files in IBM's
`torch-spyre`, and scratchy cites both:

```rust
/// LX scratchpad bytes PER CORE on dd2 = 2 MiB. Confirmed against torch-spyre
/// `scratchpad/allocator.py:305` and `sentient_dd2_sysconfig.json`.
```

There are **62 distinct Python source citations inside the Spyre target alone**,
and **57 Rust files carrying `Copyright 2025 The Torch-Spyre Authors`** — an
honest Apache-2.0 port with the upstream headers preserved, not a rewrite
pretending to originality. `ktir-core`'s memory-view module describes itself, in
its own doc comment, as a "Rust port of the memref/tileref half of
`ktir_cpu/ir_types.py`."

So the asset reused from IBM is the **constraint set**: stick sizes, memory
spaces, scratchpad budgets, the exception taxonomy, the tiling laws. The
irreducible knowledge about a machine, which no amount of cleverness
substitutes for and which used to arrive welded to a Python framework you did
not want.

**The cost lands where the thesis says it should.** The Spyre target is about
**129,000 lines** of Rust — roughly twice CUDA (~62,000) or Metal (~62,000). It
is by far the most expensive thing in the repository. And it cost **zero lines
of model code**: the same 1,580 lines of DSL, the same 25 architectures, the
same 22-line LLaMA. An 8B model boots on Spyre in 12 seconds from a 330 MiB
image.

There is even a nice inversion in the pipeline itself. Scratchy performs
*instruction selection* — a solver, a cost model, the whole apparatus — only for
CUDA. Spyre and Metal don't have it, and don't need it: the instruction tape
comes straight out of the fused forward pass. The weird accelerator got the
*simpler* compiler. Instruction selection is a tax that generality imposes, and
a bespoke build has no generality to pay for.

Which points at the quietest cost of fifty years of code-as-the-unit-of-reuse.
In the shared-library era, hardware this unusual does not get supported at all —
because "support" means a vendor writing and indefinitely maintaining a general
backend inside somebody else's general framework, and the addressable market has
to justify the headcount. Reuse-by-code puts a **population threshold on what
hardware is allowed to exist**. Reuse-by-idea drops that threshold to one team
with a compiler, and the interesting consequence is not faster inference. It is
that novel silicon becomes viable at a scale where it previously wasn't.

## What this implies

If the unit of reuse is an idea, then a few things follow that are
uncomfortable for how we currently build and reward software.

**The durable artifacts are the ones that transmit ideas losslessly.** Papers.
Reference implementations that read like the math. Precise specifications.
Detailed comments explaining *why*. Not APIs — an API is a guess about which
generality the caller will want, and that guess gets cheaper to reject every
month.

**The thing worth hand-writing is the specializer.** Nobody else can mine it
from a repo, because nobody else has your target, your shapes, your device. In
scratchy, 22 lines describe LLaMA and 465,000 describe how to make LLaMA fast on
an IBM Spyre AIU. The second number is the moat, and it is the one that could
not have been borrowed.

**Faithful porting becomes a first-class engineering skill.** The instruction
that matters most when working this way is not "build me an inference server."
It is *port the algorithm; specialize the structure; never reinvent the logic*.
The failure mode is not code that doesn't compile. It is code that compiles,
passes, and quietly disagrees with the paper — and the only defense is citing
your source, at the site, by line, so that the disagreement is findable.

Fifty years of reuse was about not writing code someone else had written. That
was always a proxy. What we actually wanted was not to *rediscover* what someone
else had figured out. The code was just the only container we had for it.

We have a better one now.

---

*Scratchy is Apache-2.0. The serving-layer algorithms are derived from
[vLLM](https://github.com/vllm-project/vllm); the Spyre compilation and hardware
model are derived from IBM's `torch-spyre` and KTIR. Both are Apache-2.0, and
both are credited at each derivation site rather than in a NOTICE file.*
