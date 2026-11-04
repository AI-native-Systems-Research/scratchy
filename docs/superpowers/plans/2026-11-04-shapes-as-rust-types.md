# Shapes as Rust types (replaces phases 4 and 5 of the metal model-facts cleanup)

Status: DESIGN, for the user's approval. No code. Written against #1083 at `efb3e0780` (phase 1),
on main `e0cb89599`. Replaces phases 4 and 5 of
`2026-11-04-metal-model-facts-cleanup.md`, whose `const` assertions over emitted values were our own
shape verifier run a second time, not Rust types.

## Goal

Shape constraints are checked by **rustc's type checker**. A shape mismatch in a model is a type
error. There is no second, hand-written shape checker beside rustc's.

## What exists today: a shape type system outside Rust

- A model is torch-idiom Python (`crates/models/arch/dsl/<arch>.py`, 36 carriers), parsed by
  `parse_python.rs` and never executed by the compiler.
- `crates/compiler/macros/src/shape.rs` (3,441 lines) is a type checker for shapes:
  - symbolic dims: `Dim::Bound("hidden_size")`, `Mul`, `Div`, and `Var` for inference;
  - a union-find `Solver` that unifies them;
  - 33 per-op signature functions over 56 `OpKind`s, e.g. `sig_gemm`:
    `gemm(x: [..., K], w: [K, N]) -> [..., N]`, which unifies x's last dim with w's first.
- `shape::infer` (`lib.rs:756`) runs it against **concrete config bounds**: once on the first model
  to validate, then per model on its own clone with its own bounds. Its dims unify as numbers. When
  the numbers don't line up structurally, it rewrites the program with "reshape recovery" hints
  (`shape::apply_reshape_hints`; gemma-3-12b's per-head QK-norm). Every later pass reads its result:
  the FUF, the bridge, the lowering, the arena.

That solver is Hindley-Milner over dims, which is exactly what rustc's type inference does over type
parameters. The design moves it into rustc.

## Rule: a compiler, not per-arch code

The translator is a compiler pass: one translation rule per DSL construct (assignment, call, loop,
`if`, tuple unpack) and one Rust signature per op kind. It never names an arch, a model or a config.
Everything that differs between arches comes from their own inputs: the DSL text and `weights.json`.
An alias fix is data in `weights.json`, or the general `as_dim` construct whose law is a
`ModelFacts` law; it's never a branch for one arch. Gate on every phase: the translator and the
signature module contain no arch or model name (checked by a test that scans them for every
`dsl/*.py` stem and every config stem), the same discipline as "per-target surface = opcode lowering
only".

## Design

### 1. The op signatures become Rust generic functions

One module of op signatures, each a generic fn whose type parameters are symbolic dims:

```rust
pub struct Act<P, D>(/* trace handle */);    // an activation: prefix dims P, last dim D
pub struct W1<D>(..);  pub struct W2<K, N>(..); // weights by rank

pub fn rmsnorm<P, D>(x: Act<P, D>, w: W1<D>) -> Act<P, D>;
pub fn gemm<P, K, N>(x: Act<P, K>, w: W2<K, N>) -> Act<P, N>;
pub fn add<P, D>(a: Act<P, D>, b: Act<P, D>) -> Act<P, D>;
pub fn attention<P, H, Kv, Hd>(q: Act<P, Mul<H, Hd>>, k: Act<P, Mul<Kv, Hd>>, ..) -> Act<P, Mul<H, Hd>>;
```

- Dims are zero-sized marker types named for the config bound they read: `HiddenSize`, `NumHeads`,
  `HeadDim`, `IntermediateSize`, … Each implements `trait Bound { const NAME: &str; }`, one per
  bound name the 33 signatures use today. The bound names are the HF config vocabulary, so they're
  shared by every arch, not declared per model.
- Products and quotients are type constructors: `Mul<A, B>`, `Div<A, B>`, and `Add<A, B>` for
  concat. The translator below writes each in one canonical order (the order `shape.rs` already
  flattens `Mul` into), so equal dims are equal types.
- Equality of dims is type equality. The unification `sig_gemm` does by hand (`x.last() == w[0]`),
  rustc does by inferring `K`.

### 2. Each DSL carrier becomes a typed Rust function

A translator turns each `dsl/<arch>.py` into a Rust fn calling those signatures, line for line:

```rust
// dsl/llama.py:11-27, generated
pub fn llama<T>(ids: Tokens<T>, w: &LlamaWeights) -> Act<T, VocabSize> {
    let mut h = embed(ids, w.embed_tokens);
    for layer in w.layers() {
        let normed = rmsnorm(h, layer.input_layernorm);
        let q = gemm(normed, layer.q_proj);
        …
    }
    gemm(rmsnorm(h, w.norm), w.lm_head)
}
```

- Activations are never annotated. rustc infers every intermediate's dims from the signatures and the
  weights' types, which is the job `shape::infer` does today.
- **Weights carry declared types**, from the arch's `weights.json` (its names and symbolic shapes):
  `q_proj: W2<HiddenSize, Mul<NumHeads, HeadDim>>`. That declaration is independent of the DSL, so
  rustc checks the DSL's use of every weight against it. A mismatch, such as a q_proj fed to an op
  expecting the kv width, is a type error at the generated line, with the DSL line in a comment
  above it. (`shape::infer` reads the same manifest today, but checks it as numbers for one model at
  a time.)
- Python control flow translates directly: `for layer in range(..)` is a loop over typed layers. A
  hybrid's per-layer classes (sliding/global, MoE/dense, GDN/attention) are separate typed layer
  structs, and both branches of an `if` type-check.
- One generated fn per **arch**, not per model. Dims are symbols; their values come from each
  model's config only when the bake evaluates them. So rustc type-checks 36 straight-line functions,
  and the per-model compile blowup that sank #1081 can't occur.

### 3. Where it's compiled and how the bake uses it (no new crate)

- `scratchy-forward-compiler-macro` (the macros crate) gets a `build.rs` that runs the translator
  over every `crates/models/arch/dsl/*.py` and `include!`s the result. That's the same unfiltered,
  pre-bake source pattern `subtile/build.rs` already uses for configs. When the macros crate
  compiles, rustc type-checks every arch's shapes. A shape error fails the build, naming the DSL
  line.
- The macros crate is already scratchy-models' build-dependency, so the bake calls the compiled typed
  program. Each `Act<P, D>` is a trace handle carrying the `LocalId` the translator assigned. Running
  `llama(..)` with a model's config records, per `LocalId`, the shape its type states, evaluated
  against that config. That record is what `shape::infer` returns today, and every later pass reads
  it unchanged.
- **`shape.rs`'s `Solver` and the 33 `sig_*` functions are deleted** once parity holds (phase C).
  There is then one shape type system, and it's rustc's.

### 4. Kernel limits as types (refined newtypes)

A config value is data, so it enters the type system once, through a constructor that checks it.
After that the type carries the proof, and no command can be built with an illegal value. This is
parse-don't-validate, not an assertion beside the value:

- `kernel_constants.rs`'s typed structs (`AffineQmvConstants`, `RopeAppendNormedConstants`,
  `AffineEmbedConstants`, …) take refined types instead of raw `u32`:
  - `AffineBits` (only the widths a kernel is instantiated for) and `AffineGroup`;
  - `StagedHeadDim` (≤ 512, the RopeAppendNormed threadgroup staging);
  - the GDN conv width against the kernel's `GDN_CONV_BLOCK`;
  - the 11 affine bits/group sites, and the 4963 and 8099 asserts from the phase-0 inventory.
- The kernel's instantiation table is the refined type's constructor. One table, one law.
- These structs are compiled once, in the build script. No per-model cost.

### 5. What stays outside rustc, and why

- The value laws of config data (GQA divides, the page fits) are checked in `ModelFacts`'
  constructors, phases 1-2. That's the boundary where data becomes typed values.
- Lowering an op to kernel commands (the arena, the coloring, the binding lists) still walks the
  traced graph as data. Making each kernel arm take typed operands means the trace calls the metal
  lowering per op. That's a later step, named here and not part of this plan.

## Phases (on #1083, after phases 2, 3 and 6; each gated)

- **A. Spike: llama only.**
  - The op signatures llama uses as Rust fns; a translator for `llama.py`; typed weights from
    llama's `weights.json`.
  - Gates:
    1. The traced shapes equal `shape::infer`'s, per `LocalId`, for every llama config.
    2. A deliberately wrong DSL line (q_proj where k_proj belongs) fails to compile, and the error
       names the DSL line.
    3. Cold `--timings` of the macros crate: compile within noise of main.
    4. Every numeric alias and reshape-recovery hint llama's configs need is listed, with the
       resolution (a) or (b) taken for each.
  - Stop and report if any gate fails.
- **B. Every arch.**
  - All 33 signatures and all 36 carriers.
  - Parity of traced vs `shape::infer` shapes for every canonical in CI's metal scope (all, mtp,
    turboquant) and spyre's.
  - The hard cases are named up front: Gemma 4's per-class attention, `Div` in the vision merges,
    concat, MoE and recurrent layers, the DFlash2 drafter's ops, and a vision tower's 2-D rope.
- **C. Delete `shape.rs`'s Solver and signatures.** The bake reads traced shapes. Gates: the
  19-model bake byte-identical, the spyre identity gate, and cold timings.
- **D. Kernel limits as refined newtypes** (§4). Gates: bake identity, and one test per refined type
  that its constructor refuses each value its kernel lacks.

## Costs and risks

- **Numeric aliases: the main risk.** Today's solver equates dims by value, per model. Rust types
  equate them by symbol, for every model of the arch at once, which is stricter. Llama's
  `weights.json` declares `self_attn.q_proj: [hidden_size, hidden_size]`, but attention's signature
  needs `num_attention_heads * head_dim`. These are equal numbers for llama, but different types.
  - Each alias must be resolved in one of two ways:
    (a) `weights.json` states the precise symbolic shape (q_proj as `num_attention_heads * head_dim`).
        This is checked against the checkpoint as today, and preferred.
    (b) An explicit, named cast op in the DSL (`as_dim`) whose law, "these two bounds are equal for
        this arch", is a `ModelFacts` constructor law, so a config that breaks it fails the build.
  - Reshape recovery is the same problem: the reshapes it synthesizes from numbers become explicit
    DSL reshapes, typed by their signatures.
  - Phase A counts llama's aliases, and phase B counts every arch's before fixing any. If the count
    shows typed DSL needs a rewrite of most carriers, report it with numbers before going on.

- **Compile:** 36 straight-line generic fns, type-checked once in the macros crate. Measured in A
  before anything else.
- **Error messages:** rustc reports against the generated file. Each statement carries its DSL
  file:line as a comment, and A's gate 2 checks that the message is usable.
- **Canonical order of `Mul`:** the translator writes products in the order `shape.rs` flattens them.
  If a carrier needs commutative equality the order can't give, A or B reports which one before any
  workaround.
- **Spyre:** it reads the same traced shapes as metal, so the main goal gets rustc-checked shapes
  from phase C on, with no spyre-specific work.
