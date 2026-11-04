# Metal model-facts cleanup: one typed representation, shape laws checked by rustc

Status: PLAN rev 3 (approved) + phase 0 inventory, for audit. No code yet. Written from upstream main `e0cb89599`; nothing from #1081
(closed) is reused. Rev 2 answered auditer's six findings on rev 1; rev 3 answers the two blocking and
three non-blocking findings on rev 2.

## Why

Spyre attention is blocked on this, and spyre TP on spyre attention. The metal side has to stop
carrying model facts as loose values first, so spyre's attention work builds on one representation of
the facts and one shape-checking discipline.

## Goal

1. **One source of the model facts**, read by the metal bake, the metal runtime, and later spyre. Main
   has two:
   - `CanonicalParams`' 59 associated consts (`crates/compiler/ir/src/lib.rs:992`): raw `u32`/`f32`,
     with `0`/`-1`/`0.0` standing in for "absent".
   - `MetalModelConsts` (`tape/model_consts.rs`): 45 public raw fields mirroring them one for one,
     filled by the bake from `ModelParams` (`codegen.rs:8585`). Its other constructor,
     `from_canonical`, has only test callers (`worker.rs:2380`, `lowering.rs:9925`); the metal
     runtime reads `CanonicalParams` directly (phase 0, R1).

   After this PR there is ONE: `ModelFacts`. `CanonicalParams` keeps its overlapping consts only as
   trait defaults derived from it, in one place.
2. **Every shape constraint rustc can check, rustc checks**, by a `const` assertion at a use site or a
   trait bound. Each law that stays a runtime refusal gets a line saying why rustc cannot check it.
3. **No compile-time regression.** #1081 cost +130 s of build-script compile (cold CI scope 171 s →
   297 s) by making the lowering generic per model. Here the lowering compiles **once**, and no
   optimization level is touched.

Out of scope: cuda's `eval.rs`. It keeps reading `CanonicalParams`, whose consts keep their values, so
it compiles unchanged.

## Where each kind of constraint is checked

The bake runs in scratchy-models' build script and sees facts and graph shapes as **data**. There a law
can only be a checked constructor: a `const fn` that runs at build time and fails the build naming the
law. rustc checks a law only in code it compiles over the values, and code compiled once per model
anyway costs nothing extra. Two such places exist: the code the bake **emits** into scratchy-models,
and the metal **runtime** (generic over `W`, compiled per model).

| kind | example | where rustc checks it |
|---|---|---|
| fact law, one group | GQA divides, rot ≤ head dim, window > 0 | the group's `const fn` constructor, in `W::FACTS` (emitted) |
| fact law, across groups | TurboQuant codec ⇒ attention | `ModelFacts::new`, in `W::FACTS` |
| a runtime path needs a group | the attention pipelines need attention | `const { assert!(W::FACTS.<group>.is_some()) }`, only in a fn instantiated solely for models that have the group (see "Runtime" below); otherwise a runtime refusal |
| kernel ABI law | a kernel's supported head dims; affine `k % group == 0` | `const` assertion on each emitted command's constants (phase 4) |
| cross-step shape law | producer output extent == consumer input extent; a tensor fits its arena slot; a binding covers what the kernel reads | `const` assertion over independently emitted per-step extents and slot sizes (phase 5) |

`W::FACTS` re-checks values the build script already checked. On its own that adds no constraint;
what it adds is the values every runtime `const` assertion above is written against. A cross-step law
is a real rustc check only where its two sides come from **independent** derivations (the producer's
lowering vs the consumer's, the coloring's slot size vs the step's extent). Phase 0 says which ones
are; one derived from the other is the same number twice and is not emitted.

## Design

### The typed facts (`scratchy_ir::model_facts`)

- One newtype per quantity: private field, `const fn new` checking its law, `get()` the only exit
  (`QueryHeads`, `KvHeads`, `HeadDim`, `RotDim`, `BlockSize`, `HiddenSize`, `IntermediateSize`,
  `VocabSize`, `AttnWindow`, `AttnScale`, `SoftCap`, `RmsNormEps`, …; kv and query heads are
  different types).
- Groups for what a model may lack, each an `Option` on `ModelFacts`, never `0`/`-1`/NaN:
  - **attention**: a sliding class and a full class, each a head geometry whose constructor proves
    GQA divisibility and rot ≤ head dim (q heads, kv heads, head dim, rot dim, page); scale; window;
    attention softcap; rope pairing; rope-on-read; MRoPE section;
  - **vision tower**; **GDN layers**; **drafter**; **speculative drafts / capture slots / target**.
- Always present: hidden size, intermediate size, vocab, eps, norm offset, final softcap (an
  `Option`), dtypes, KV codec, KV cache layers.
- **Coverage:** every one of `MetalModelConsts`' 45 fields; everything `ModelAttnGeometry` carries
  (`nqh`, `nkvh`, `hd`, `gqa`, `model_geometry.rs`) plus what the subtile attention ops carry beside it
  (`scale: f32`, the mask's window, `subtile_ir.rs:1090,1270,605`). So the spyre PR adds no facts.
  Not covered, staying raw `CanonicalParams` consts because no metal or spyre path reads them: the MLA
  dims (`KV_LORA_RANK`, `QK_NOPE_HEAD_DIM`, `QK_ROPE_HEAD_DIM`, `V_HEAD_DIM`, `QK_HEAD_DIM`,
  `MLA_ATTN_SCALE`); and target/build knobs that are not model facts (`MAX_BLOCKS_PER_SEQ`,
  `PREFILL_TILE_Q`, `HAS_BIAS_ADD`, `HAS_MOE`).
- **TP:** `ModelFacts` is a canonical's facts **at its tensor-parallel width** (heads sharded), as
  `CanonicalParams` is today (the fill takes `tp_world_size`). Spyre TP reads exactly that.
- Storage: tables (KV cache layers) are `&'static` in the emitted const and owned at build time
  (`Cow<'static, [u32]>`), so one type serves both, and `ModelFacts` is `Clone`, not `Copy`. The
  table's law (strictly ascending layers) runs in both paths: the `const fn` on the borrowed slice in
  the emitted const, and the same check on the owned value when the bake builds it, which fails the
  build naming the law. A test covers the owned path (a descending table panics) as well as the
  borrowed one.

### One source: `CanonicalParams::FACTS`

- `CanonicalParams` gains `const FACTS: ModelFacts`. Each overlapping raw const becomes a trait
  **default** derived from `Self::FACTS`, all in one block of the trait: e.g. `HEAD_DIM` = the sliding
  class's head dim, else the vision tower's (the fill's vision-only rule, `codegen.rs` "is_vision_only"),
  else 0. cuda and serving read the same numbers they read today.
- The generated `impl CanonicalParams` emits only `FACTS` and the non-covered consts listed above.
- **Override risk, stated:** a trait default can be overridden, so the one-source property rests on
  the generator. It is held by a codegen unit test asserting that the emitted impl names none of the
  derived consts. A blanket extension trait (`impl<W: CanonicalParams> CanonicalConsts for W`), whose
  consts cannot be overridden, would hold it by type instead. But it changes every reader's path
  (`<W as CanonicalParams>::HEAD_DIM`), cuda's among them, which is out of scope here. It is the
  runtime PR's step, when cuda's readers move anyway.
- The bake builds the `ModelFacts` value once in the fill (no second front-end pass, no new crate, no
  feature table) and emits it as `FACTS` through one serializer of that value, so emission cannot drift
  from what the bake used.

### Who reads it

- **Bake:** `lower_subtile_tape_to_metal` (`tape/lowering.rs:153`), `lower_canonical` /
  `lower_canonical_classed` (`compiler/src/canonical.rs:290,528`) and `bake_bucket_tapes`
  (`static_tape.rs:434`) take `&ModelFacts` where they take `&MetalModelConsts`: one parameter type
  swapped, no generic added. Each `p.<field>` read (83 in `lowering.rs`) becomes a group read. A step
  needing a group the model lacks is `LoweringError::ModelLacks(group)`.
- **Runtime:** generic over `W: CanonicalParams` and compiled per model, it reads `W::FACTS`. A
  `const { assert!(..) }` in a generic fn is evaluated for **every** model that instantiates the fn,
  whichever runtime branch runs. So a group assertion goes only where the enclosing fn is
  instantiated solely for models that have the group. Phase 0 records, per `W::`-fact read site in
  `crates/targets/metal/src`, which models instantiate its fn, and gives each site one of three
  outcomes:
  - **assert**: the fn is reached only from code instantiated for models with the group;
  - **split**: the arm moves into its own generic fn, called only from code instantiated for those
    models, and the doc names that caller;
  - **stays a runtime refusal: shared fn**: the doc gives the reason.

  On main, `pipelines.rs:697` `constants_for<W>` is called from `pipeline_for_command<W>`
  (`pipelines.rs:200`) for every model, and it dispatches by the command's runtime `KernelId`. So its
  attention arms (`:731-772`) stay runtime refusals unless phase 0 finds a split whose caller is
  instantiated only for models with attention. This plan promises no assertion at `:731-772`; the
  same applies to `vision_arch.rs` and the GDN and drafter arms.
- `MetalModelConsts`, `tape/model_consts.rs` and `from_canonical` are deleted. The runtime fns and
  tests that took a `MetalModelConsts` take `&W::FACTS` / a `ModelFacts`.

## Phases (one PR, a commit per phase, every phase gated)

0. **Inventory** (a section of this doc, before code). Every shape law the metal path checks today,
   with file:line on main, classed by the table above. Sources:
   - the 122 `assert!`s in `lowering.rs`;
   - `LoweringError` and `steps_from_tape::Refused`;
   - `op_abi.rs`;
   - every `W::`-fact read in the metal runtime.

   For each cross-step law: are its two sides independently derived? Is it emittable as consts? For
   each law left as a runtime refusal: why rustc cannot check it.
1. **`ModelFacts` replaces `MetalModelConsts` in the bake.** Module, fill, lowering reads, deletion
   in the bake. The lowering tests' fixtures (`tp()`, `tq_consts()`, … `lowering.rs:9924+`) become
   `ModelFacts` values with struct-update, and each law gets a `should_panic` test at its constructor.
2. **`CanonicalParams::FACTS`, with defaults derived from it.** The generated impls shrink to `FACTS`
   plus the non-covered consts. Before the old literals are deleted, a local (unpushed) all-scope build
   asserts every derived default equals the literal the fill emits today, for every canonical. The
   result is reported in the PR.
3. **The metal runtime reads `W::FACTS`,** with the `const` assertions above. `MetalModelConsts`,
   `model_consts.rs` and `from_canonical` are deleted.
4. **Kernel-ABI laws as `const` assertions on emitted commands.** For each ABI law phase 0 finds, the
   kernel's constants type gets a `const fn` check, and the emitted command constants are emitted
   inside `const { … }`.
5. **Cross-step shape laws as `const` assertions** for the independently derived ones phase 0 finds.
   Each step's operand extents and its arena slot are emitted as consts beside the tape, with
   `const { assert!(..) }` over them. For each law found to be "the same number twice", the doc records
   the code that derives both sides (file:line), so a later change that makes them independent knows
   to add the assertion.
6. **Spyre's doors (shared crates, for spyre attention).** The head-dim door
   (`with_config_head_dim`, `model_geometry.rs:214`) gets its arms from `for_each_config_head_dim!`,
   which `subtile/build.rs` generates by scanning the configs. It cannot read the emitted facts:
   scratchy_subtile compiles before scratchy-models' build script runs the bake.
   - **Decision: keep the head-dim scan as the one pre-bake source** (option b).
   - Option (a), arms generated from `rope_at::<HD>`'s own law, has no law to generate from. The
     function's only constraints on `HD` (`lower_ktir_to_superdsc.rs:4790-4806`) are `HD > 0` and,
     for `HD` wider than one 64-element stick, an even stick count (`HD / 64` even: 128, 256, 512).
     Neither bounds the set; enumerating every `HD` it admits would be a declared bound plus one
     instantiation per value.
   - The law also can't become a `const` assertion inside `rope_at::<HD>` today. main's scan yields
     head dims 16, 32, 64, 96, 128, 192, 256, 512, and 96 and 192 break it (stick counts 1.5 and 3).
     The door instantiates `rope_at` for every scanned head dim, so such an assertion would fail every
     build, metal included. The law stays a runtime refusal for that reason, and this doc records it.
   - What changes: each generated arm is emitted through `HeadDim::new`, so the arms carry the
     positive-head-dim law.
   - The consumer-less attention-geometry door is deleted: `with_config_attn_geometry`,
     `OnAttnGeometry`, `for_each_config_attn_geometry!`, and the attention-geometry half of
     `subtile/build.rs`'s scan.
   - The head-dim scan keeps model_geometry.rs's guarantee: a new width arrives as a config.json,
     not as an edit here.

## Gate, every phase, before any push

- **Bake identity:** every scratchy-models `OUT_DIR/*.rs` byte-identical to main (tree paths
  normalized). Models:
  - dense: smollm2-135m, llama-3.2-1b, qwen3.5-0.8b, gemma-4-26b-a4b-it, gpt-oss-20b;
  - MLX-affine presets of: qwen3.5-122b-a10b, qwen3.6-35b-a3b, nemotron-3-super-120b-a12b,
    llama-3.2-1b, gemma-4-26b-a4b-it, gpt-oss-20b, glm-4.5-air;
  - heads, encoders and towers: gemma-4-26b-a4b-it-assistant, qwen3.6-35b-a3b-dflash2,
    modernbert-base, qwen3.5-9b, qwen2-vl-2b-instruct;
  - with `mtp-heads`: qwen3.6-35b-a3b and gemma-4-26b-a4b-it at MLX presets.

  Phases 2, 4 and 5 change emitted code by design. For those, the diff is exactly the `FACTS` / impl
  shrink / added `const` assertions, reviewed line by line, and the tapes' bytes are unchanged.
- **Spyre identity (phase 6, and whenever shared crates change):** the spyre bake's emitted
  KTIR/SDSC for llama-3.2-1b and granite-3.1-2b byte-identical to main, and the generated
  `for_each_config_head_dim!` arms equal to main's (16, 32, 64, 96, 128, 192, 256, 512).
- **Compile time:** cold, each tree in its own fresh target dir, `--timings`, main vs branch, for
  CI-scope `clippy -p scratchy-models --features metal,all,mtp,turboquant` and a scoped
  `-p scratchy-models --features metal,llama-3.2-1b`. Reported per unit:
  - the scratchy-models build-script compile: main's ±2 s;
  - the scratchy-models check: phases 2-5 add const evaluation there;
  - total wall time.

  A phase that misses these is not pushed, and its cost is reported.
- CI-scope metal clippy `-D warnings`; `cargo test -p scratchy-target-metal --lib`; the metal kernel
  tests locally behind the GPU lock; `cargo check` with `spyre`.

## Risks

- **The fill's derivations have edge cases**: vision-only bodies re-deriving text consts, Gemma 4's
  classes, drafter and head widths. Phase 2's equivalence check over every canonical in all scope is
  what proves the derived defaults keep them.
- **Phase 5 may find few independent cross-step laws** if most extents are derived from one source.
  Then this phase is small, and the doc records which laws were the same number twice.
- **Phase 4/5 const evaluation cost** in scratchy-models' check: measured per phase; a law whose cost
  shows is reported, not shipped.


## Phase 0: inventory (main `e0cb89599`)

Every shape law the metal path checks today, with its outcome under this plan. Sources:
- the production `assert!`s in `tape/lowering.rs`: 22, all before `mod tests` at `:9918`;
- `LoweringError` (`tape/lowered.rs`, 18 variants) and `steps_from_tape::Refused` (36 variants);
- `op_abi.rs`;
- every `W::`-fact read in `crates/targets/metal/src`.

Outcomes:
- **T**: a typed constructor (the law is in a group's `const fn`, and `W::FACTS` makes rustc evaluate
  it per model).
- **K**: a kernel-ABI law, emitted as a `const` assertion (phase 4).
- **X**: a cross-step law, emitted as a `const` assertion (phase 5).
- **R**: stays a runtime refusal, with the reason.

### Production assertions in `tape/lowering.rs`

| line | law | outcome |
|---|---|---|
| 5139, 5975 | a sliding attention's window > 0 (`debug_assert`) | **T**: `AttnWindow` positive, and its absence is `None`; a sliding step on a model without one is `ModelLacks`. The asserts are deleted. |
| 6138, 6193 | the final soft cap > 0 (`debug_assert`) | **T**: `SoftCap` finite and positive, absence `None`. The asserts are deleted. |
| 8099 | GDN conv kernel ≤ `GDN_CONV_BLOCK` + 1, and `GDN_CONV_BLOCK` > drafts | **K** over facts. `GDN_CONV_BLOCK` is metal's kernel constant, so the law can't live in the neutral `ModelFacts`. It is emitted in metal's per-canonical emission as `const _: () = assert!(..)` over `W::FACTS`' GDN group and drafts. The two sides are independent: the kernel's block vs the config's conv width. |
| 4963 | `RopeAppendNormed` staging needs head dim ≤ 512 | **K** over facts, emitted per canonical over `W::FACTS.attention` (each class's head dim), only for canonicals whose tape lowers a `RopeAppendNormed`. The bake knows which, and emits the assertion only there. |
| 4572, 4580, 7984, 7988 | affine embed: bits ∈ {2,3,4,8}, group ∈ {32,64,128} | **K** per command: the emitted command's bits and group vs the kernel's instantiation table (a const in the metal crate). Independent: the table vs the weight's storage. |
| 8632, 8636, 8675, 8679 | affine gather qmv / qmm_t: bits ∈ {2,3,4,8}, group ∈ {32,64,128} | **K** per command, as above. |
| 8710, 8714 | NAX gather qmm_t: bits ∈ {2,3,4,8}, group ∈ {64,128} | **K** per command, as above. |
| 3881 | `SiluMul` width > 0 | **R**. The width is the op's own value from the DSL bridge (subtile op data), not emitted as a separate const. Making it a positive type is a subtile op change and belongs to the spyre PR's bridge work. Recorded. |
| 4946 | `RopeAppendNormed` is NeoX-only | **R**: a matcher law (which DSL patterns fuse), not a shape. |
| 2217, 2218 | command / barrier / liveness vectors have equal lengths (`debug_assert`) | **R**: one function's internal invariant, not a shape law. |
| 2512, 5183 | already `const` assertions | unchanged. |

### `LoweringError` (`tape/lowered.rs`), each variant

These are laws about how the build script constructs a tape, unless a line says otherwise. They exist
only in the build script's data; the emitted tape is the lowered bytes after the law held.

| variant | law | outcome |
|---|---|---|
| `MalformedLoop` | a `StepRow::Loop` body lies within the step tape | **R**: tape construction. |
| `TurboQuantOffsetUnbound` | a TurboQuant codec command can bind the additive offset of the KV operand it compresses | **R**: the codec pass's wiring (which buffer the offset is), decided while building the tape. |
| `UnrotatedWriterOffset` | a TurboQuant-coded KV writer that writes un-rotated (`KvAppend`) carries no additive offset | **R**: a law about which writer form the codec pass meets, decided while building the tape. |
| `NoSource` | a step's site holds the weight source it binds | **R**: the step records and the opcode lowering agree. |
| `NoRotaryTable` | a rope-on-read attention's layer class has a rotary table | fact half **T** (rope-on-read is on the attention group); the table's presence is the bake's manifest data, **R**. |
| `DoubleGate` | a gated row's realization does not gate its commands again | **R**: construction. |
| `SplitPartialsSlotTaken` | a decode attention the TurboQuant split takes binds nothing at the split's partials slot | **K candidate (phase 4)**: the reserved slot is the split kernel's ABI constant, and the binding list comes from the lowering. These are independent, so the emitted command can carry `const` assertion "no binding at the partials slot". |
| `CacheWriterSkipped` | a cache-writing command's row is one a cache-only forward runs | **R**: the liveness walk and the lowering agree. |
| `RowZeroWithoutPerRowTwin` | a row-0-only command's step has a per-row twin | **R**: step structure. |
| `TurboQuantCompressesNothing` | a TurboQuant model's backbone compresses some KV | fact half **T** (`KvCodec`); the graph half (which binds compress) is **R**. |
| `CodecStepOnDenseModel` | a KV codec step reaches only a TurboQuant model | fact half **T**; graph half **R**. |
| `OneRowFold` | a one-row fold is in a one-row bucket | **R**: a fold rule over buckets, at construction. |
| `RopedAttentionShape` | a decode attention running its KV writer lowered to one command, and the writer binds nothing the fused command has no slot for | **R**, though it is named for a shape: it is structural (the command count and the fused command's binding slots at fusion time), not an extent relation between emitted values. Raised at `lowering.rs:2790`. |
| `RowProgramWeight` | a row program's instruction names a weight it holds a layer for | **R**: construction. |
| `CombineEndsUngathered` | an expert combine with ends is in a gathering bake | **R**: construction. |
| `NormedRowsUnread` | rows a MoE step normalizes on load are read by a command that normalizes | **R**: construction. |
| `ScratchTooLarge` | a KV-cap-sized scratch fits u32 byte offsets | **X candidate (phase 5)**, if the scratch size and the kernel's u32 offset bound are independent. |
| `TooManyCommands` | the command table fits a `CommandIx` | **R**: the refusal sits where the index is minted, and a table that doesn't fit is never emitted. |

### `steps_from_tape::Refused` (36 variants)

- Coloring and arena laws: `NoColour`, `Uncoloured`, `Clobbers`, `UnwrittenSlot`, `NoRegion`. These
  are the arena-slot cross-step laws.
  - A slot's size is the coloring's max over the tiles it holds, so "a tile fits its slot" is **the
    same number twice** (derived at `codegen.rs` `band_facts`: per-color max). It is not emitted.
    Recorded here per phase 5.
  - "A binding covers what the kernel reads" is **X** if a kernel's read extent is derived from its
    own constants (the kernel ABI) independently of the shape inference that sized the slot. Today
    no per-kernel read-extent function exists. Phase 5 writes one, as a `const fn` per kernel
    constants type beside phase 4's tables, for the kernels whose reads are a function of their
    constants (matvec/matmul/norm/rope/attention). The emitted assertion is
    `SLOT_BYTES[slot] >= <K as Reads>::bytes(&CONSTS)`. Kernels whose reads depend on runtime inputs
    (paged attention's block table) are **R**: their extent is not a compile-time value.
- Operand/step-matching laws (`MissingOperand`, `NotAStep`, `NotASource`, `ForeignActivation`,
  `PerLayerOperand`, `NoWeight`, `NotAWeight`, `Escaped`, `NoGatedFold`, `FusionShape`,
  `LoopUnbalanced`, `IncompleteMoeBlock`, …): **R**. They are tape-construction laws in the build
  script, with no emitted pair.
- `NoIntermediate`-style fact reads become `ModelLacks` (**T** for the fact half).
- `SplitSiluMul`, `TileLevel`, `DeviceSampler`, `ChainedHead`: **R**, wiring laws (a node that
  should not reach metal).

### `op_abi.rs`

One `const` assertion (`:301`), already a const. The `W::` matches in `:323-349` are an enum alias
(`W::Region`), not model facts.

### Metal runtime fact reads (R1)

The runtime's generic fns over `W: CanonicalParams`, and what each reads:

| site | fn, instantiated for | reads | outcome |
|---|---|---|---|
| `pipelines.rs:697-772` | `constants_for<W>`, called from `pipeline_for_command<W>` (`:200`, `:906`), called from `bake_bucket<W>` (`worker.rs:1554`, `:1676`): **every model** | `Q_SIZE`, `RMS_NORM_EPS`, `NORM_WEIGHT_OFFSET`, `INTERMEDIATE_SIZE`, `HIDDEN_SIZE`, and on the `RopeAppend` / `AttentionViaCache` / `AttentionPrefillSdpaPaged` arms `HEAD_DIM`, `NUM_Q_HEADS`, `NUM_KV_HEADS`, `ROT_DIM`, `BLOCK_SIZE`, `ATTN_SCALE` | **R: shared fn.** The arm is chosen by the command's runtime `KernelId`. No caller is instantiated only for models with attention, and a split would need the dispatch itself to be per group. The attention arms read `W::FACTS.attention` and refuse `ModelLacks(Attention)` instead of reading 0s. |
| `pool.rs:2080, 2084` | the step loop (`begin_step<W>`, every model) | `BLOCK_SIZE`, `GLOBAL_BLOCK_SIZE` | **R: shared fn.** These are TurboQuant's dequant max-blocks (`live_block_width` over the two classes' pages) and unrotated-blocks stores. They become conditional on `W::FACTS.attention`: when it is `Some`, they store from its classes' pages as today; when it is `None`, the stores are skipped (no stand-in page). The models that take the `None` path are the vision-only towers (qwen2-vl, qwen2.5-vl, qwen3.5-vl, locateanything, and the gemma3-mm tower): no attention group, dense KV, no TurboQuant state to update. |
| `worker.rs:1607` | `bake_bucket<W>`, every model | `METAL_DTYPE` | always present: no group, no assertion. |
| `worker.rs:1374` | every model | `MAX_BLOCKS_PER_SEQ` | not a model fact; stays a raw const. |
| `worker.rs:1858, 1878` | every model | `SOURCES` | a weight accessor, not a fact. |
| `layers.rs:108` | the norm loader | `SCALE_DTYPE` | always present. |

Result: on main the metal runtime has **no** site where a group-presence `const` assertion is sound;
every fact-reading fn is instantiated for every model. Phase 3's runtime change is: read `W::FACTS`
instead of the raw consts, with `ModelLacks` where a group is absent. The table row "a runtime path
needs a group" ships nothing in this PR, and this section is the record of why.

### What phases 4 and 5 ship, from this inventory

- **Phase 4 (K):**
  - affine embed / gather / NAX-gather bits and group vs the instantiation tables (per command,
    11 assertion sites);
  - the TurboQuant split's partials slot unbound by any other binding (per split command);
  - `RopeAppendNormed` head dim ≤ 512 (per canonical that lowers it);
  - GDN conv block vs conv width and drafts (per canonical with GDN).
- **Phase 5 (X):**
  - binding coverage, `SLOT_BYTES[slot] >= reads(consts)`, for kernels whose reads are a function of
    their constants. Each per-kernel `Reads` const fn is a second, hand-written claim about the
    shader, so each ships only with a kernel test tying it to the shader:
    - a run with `MTL_SHADER_VALIDATION=1` against a buffer sized exactly to `reads(consts)`, which
      passes;
    - a run with a buffer one element short, where validation fails;
    - both behind the GPU lock (`/Volumes/Builds/gpu-lock`).

    A `Reads` impl without that test does not ship;
  - `ScratchTooLarge` if its two sides are independent.

  Everything else is **R** or the same number twice, recorded above.
