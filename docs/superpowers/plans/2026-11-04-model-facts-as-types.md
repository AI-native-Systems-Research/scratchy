# Model facts as types: metal first, then spyre

Supersedes #1079 (rejected: it lowered spyre's const-generic geometry to metal's `MetalModelConsts`
values). Here the direction is reversed. Metal's value struct is deleted, and every model and quant
fact reaches every lowering as a compile-time constant of a type parameter, over newtypes.

## Form (stable Rust)

- A const parameter can't be a newtype on stable (`adt_const_params` is nightly), and `f32` can't be
  a const parameter at all. So each **canonical is a type** `M: ModelFacts`, and every fact is an
  associated const of newtype type (`M::HEAD_DIM: HeadDim`, `M::ATTN_SCALE: AttnScale`,
  `M::KV_CODEC: KvCodec`).
- Lowerings are generic over **types** (`M`, later `Q: QuantFacts`, `R: Rung`), never over const
  `u32`s. On stable, `Shape<{M::HD}, …>` inside code generic over `M` is an error ("generic
  parameters may not be used in const operations"), so mixing the two styles would need
  `generic_const_exprs`.
- Derived quantities (GQA group, kv width, slabs) are associated consts that the generator emits as
  literals. Every law is a `const _: () = assert!(…)` over them, so a violation is a build error
  (E0080) when the canonical is baked.

## Staging (why there is a new crate)

Metal's bake runs inside scratchy-models' build script (`scratchy-forwards.rs` → `compile_carrier`).
A type that build script monomorphises over must exist when the script is **compiled**. Some facts
come from the DSL front end (`uses_rotary`, kv layers), so they can't be read from `config.json`
alone.

- **`scratchy-model-facts`** (new crate, approved 2026-11-04). It has one feature per model stem,
  forwarded from scratchy-models (a parity test keeps the two lists equal), and reads quant scope
  through `scratchy_quantizations::enabled_presets()` as scratchy-forwards.rs already does. Its
  `build.rs` runs the shared front end and the fill for the **enabled** canonicals only, and writes
  `pub struct <Canon>; impl ModelFacts for <Canon> { … }` plus a dispatch keyed by canonical name.
- `scratchy-forwards.rs` (build-dep on it) passes that dispatch into `compile_carrier`, which calls
  `metal_bake::<M>()`. A scoped build instantiates one model.
- The fill stays one function (`emit_canonical_params_impl`). The runtime `impl CanonicalParams for
  Weights` stops restating values: `CanonicalParams: WeightAccessors + ModelFacts`, and the emitted
  `impl ModelFacts for Weights` delegates each const to the stage-1 type.

## Phases (each gated: metal bake artifacts byte-identical to main, metal clippy scopes, chat smoke)

1. **Newtypes + `ModelFacts`** in `scratchy-ir`: one home for every model quantity. Metal's
   `tape::ids` model newtypes and subtile's `QueryHeads`/`KvHeads`/`HeadDim` move there, with
   private fields and a `get()` accessor. `CanonicalParams`' raw consts become `ModelFacts`
   newtype consts. All readers (cuda `eval.rs`, metal runtime) take `.get()` at their own seam.
2. **Stage-1 crate** + delegated runtime impl: one fill, two stages.
3. **Metal bake generic over `M`**: `lower_subtile_tape_to_metal`, `bake_bucket_tapes`,
   `lower_canonical_classed` and the off-tape statics take `M`. All ~290 `p.<field>` reads become
   `M::<FACT>`. `KvClass` selection becomes a type (`M::Base` / `M::Global` geometry), not a
   `bool`/mask branch. **`MetalModelConsts` is deleted.**
4. **Quant facts as types** (`Q: QuantFacts`: affine bits/group, MoE expert widths, scale dtype).
5. **Spyre onto the same `M`**: `AttnGeometry<NQH, NKVH, HD>` / `Shape<HD, NQH, NKVH, CAP>` /
   `Rung<MQ>` become `Shape<M, R>`. The `build.rs` geometry door in subtile and its config scan
   are deleted, because the stage-1 crate is that door for every fact.

## Gate artifacts

- Bake identity: on main and on the branch, each in **its own target dir**, run
  `cargo build -p scratchy-models --features metal,<stem>[,quant]` and `cmp` every
  `$OUT_DIR/*.rs`. Stems: smollm2-135m, llama-3.2-1b, qwen3.5-0.8b (gdn),
  gemma-4-26b-a4b-it (hybrid classes), gpt-oss-20b (sinks + moe), plus one MLX quant preset.
- `cargo clippy -p scratchy-models --features metal,all,mtp,turboquant -- -D warnings` (CI scope)
  once at phase end. Never per edit.
