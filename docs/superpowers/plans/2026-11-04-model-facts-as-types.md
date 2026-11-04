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
3. **Metal bake generic over `M`** (done): `lower_subtile_tape_to_metal`, `bake_bucket_tapes`,
   `lower_canonical_classed` and the off-tape statics take `M`. Every `p.<field>` read is now
   `M::<FACT>`. A layer's geometry is `Attention::class(mask.class())`, the one mask → class law
   in subtile (`AttnMask::class`), with no `is_global` branch. Runtime checks the types now prove
   are deleted (GQA divisibility in the split and verify paths). A step reading a fact group its
   canonical lacks is `LoweringError::ModelLacks(Fact)`, never a zero. **`MetalModelConsts` is
   deleted.** `emit_model` reaches stage 1 through `FactsDispatch` (`scratchy_model_facts::
   StageOne`), and asserts on every target that stage 1 generated its canonical. Tests lower at
   test types (`tape::facts::fixtures`: a `TestFacts` record made a type by `facts_type!`;
   value-parameterized cases are const-generic test types, e.g. `GdnAt<NK, NV, HK, HV, DRAFTS>`).
4. **Quant facts as types** (done): `ModelFacts::QUANT: Option<Quantization>`, `Some` exactly for
   a quantized canonical. It holds the embedding table's and the lm_head's `AffineStorage`
   (`AffineGroupSize` and `AffineBits`, each refused unless a kernel is instantiated for it) and
   every routed-expert bank's declared widths by `(base, layer)` as a `&'static [ExpertWidths]`
   table. Stage 1 derives them once, the derivations `emit_model` ran at bake time. The bake reads
   them off `M`: the recorder's affine embed, its shared-fused expert widths (a missing key is a
   refusal, not a fallback), the head's vocabulary suffix and the weights struct's embed storage.
   The `MoeExpertBits` closure over the program and `MetalStepFacts::embed_quant` are deleted. A
   weight's own storage (each matmul's group and bits) stays on its op, beside its `n`/`k`.
5. **Every stage around the metal lowering reads `M`** (done; review on #1081):
   - The DSL→tape bridge (`lower_decode_to_wavefront<M>`) runs behind `FactsDispatch`: eps is
     `M::RMS_NORM_EPS`, the scale and the per-op geometry come from `M::ATTENTION`'s classes, the
     sliding window from `Attention::window` (`attention_mask::<M>`). The `f32::NAN` scale
     stand-in, `geom_pair`, `sliding_window_bound` and `NoHeadGrouping` are deleted; an attention
     or MoE tile on a model without the group is `BridgeError::NoAttention` / `NoMoe`.
   - `ModelFacts::MOE: Option<Moe>`: experts, experts a token (≤ experts), the expert and shared
     widths, the router (`RouterKind`, `TopKScores`, routed scale) and the body (activation,
     gpt-oss's swiglu limit). The bridge's MoE arms fill the ops from it; its `RouterFlavor`
     copy of the router kind is deleted.
   - `emit_model` reads `M` for the KV cap ladder (`M::MAX_POSITIONS`, `kv_cap_ladder::<M>`; a
     pool-reading model without positions is `ModelLacks(Fact::MaxPositions)`), a drafter's
     result bucket (`M::DRAFTER`, `M::SPEC_VERIFY_ROWS`), every extern buffer's bytes
     (`ExternExtents`: vision, target hidden, multimodal splice, MRoPE rows, draft capture), the
     terminal row's width and the MRoPE rows' section, theta and rot window. Metal's
     `ids::MaxPositions` copy is deleted.
   - What stays a value: the per-op geometry and scale on the subtile attention ops
     (`ModelAttnGeometry`), now minted only from `M`. Metal never reads them; spyre's
     const-geometry door and the host reference do, so the carrier goes with phase 6.
6. **Spyre onto the same `M`**: `AttnGeometry<NQH, NKVH, HD>` / `Shape<HD, NQH, NKVH, CAP>` /
   `Rung<MQ>` become `Shape<M, R>`. The `build.rs` geometry door in subtile and its config scan
   are deleted, because the stage-1 crate is that door for every fact.

## Gate artifacts

- Bake identity: on main and on the branch, each in **its own target dir**, run
  `cargo build -p scratchy-models --features metal,<stem>[,quant]` and `cmp` every
  `$OUT_DIR/*.rs`. Stems: smollm2-135m, llama-3.2-1b, qwen3.5-0.8b (gdn),
  gemma-4-26b-a4b-it (hybrid classes), gpt-oss-20b (sinks + moe), plus one MLX quant preset.
- `cargo clippy -p scratchy-models --features metal,all,mtp,turboquant -- -D warnings` (CI scope)
  once at phase end. Never per edit.
