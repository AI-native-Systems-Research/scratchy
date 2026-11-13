# EXPERIMENT 1 — the golden set, and what each Granite kernel does at each stage

**THE PIPELINE.** Python Triton → C++ KTIR lowering → our Rust lowering to SuperDSC →
`dxp_standalone`. Nothing here touches the deeptools dataflow-scheduler, senprogs, the
senulator, or any backend patch; none of that is in scope for this experiment.

**THIS DIRECTORY IS THE REFERENCE SET FOR EXPERIMENT 2** (the all-Rust path). Everything
under it was written by `run_experiment1.py` and nothing was hand-edited.

## → IF YOU ARE HERE FOR THE RUST FRONT END, `ktir/` IS THE ORACLE

The scope narrowed after this set was produced: KTIR→SuperDSC moved elsewhere, so **the 23
files in `ktir/` are the part of this directory that matters** — they are the acceptance
oracle for a pure-Rust path to KTIR. `bundles/` and the program digests remain a true record
of what the SuperDSC path did, but they are no longer what this set is *for*.

Read these three things before diffing:

1. **`index.json` records the C++ pass list per configuration**, under `provenance.pass_lists`
   — extracted from `backend/compiler.py`'s **source**, not retyped, so it cannot drift away
   from the passes that actually ran. **A diff against `ktir/<name>.ktir.mlir` is only
   meaningful against the same pass list.** `provenance.git` keeps
   `artifacts_produced_at_commit` distinct from `recorded_at_commit`, so a provenance refresh
   can never claim the goldens came from a commit that did not produce them.
2. **Every configuration has a `.ttir.mlir`; eleven of twelve have a `.ktir.mlir`.** The one
   missing is the record of `embedding_granite` refusing *inside* `make_ktir`, not an
   omission — so the TTIR checkpoint is diffable even where no KTIR oracle exists.
3. **Every golden here is `tt.func`.** `ToSchedulerKTIR` emits `func.func`. That difference
   already hid one silent parser hole (see the parser-holes section), and it means this set
   cannot by itself exercise the `func.func` reader — a structural blind spot in experiment 1
   as a check on experiment 2's input form, stated here rather than left to be discovered.

```bash
export DEEPTOOLS_PATH=/Users/nickm/tmp/dt_src     # required by stage 3
PYTHONPATH=python TRITON_BACKENDS_IN_TREE=1 python3 \
    third_party/spyre/test/experiment1/run_experiment1.py [name-substring]

# refresh `provenance` in index.json WITHOUT regenerating any artefact
PYTHONPATH=python TRITON_BACKENDS_IN_TREE=1 python3 \
    third_party/spyre/test/experiment1/run_experiment1.py --provenance-only
```

## ⛔ NOTHING HERE EXECUTED ARITHMETIC

Per configuration the outcome is exactly one of **"produces a program"** (with a census
and a byte size) or **"refuses"** (with the stage and the verbatim diagnostic). No
`allclose`, no tolerance, no hardware, no simulator. **Nothing below is called correct.**
A program is a program; whether it computes the right thing is not a question this
experiment asks.

## The three stages, and what each artefact is

| stage | what runs | artefact |
|---|---|---|
| `ktir` | `triton.compile`'s own `make_ir` → `make_ttir` → `make_ktir`. `make_ktir` **is** the C++ pass list — `add_dot_to_linalg`, `add_ktir_lowering(grid)`, `add_plan_corelets` (`third_party/spyre/backend/compiler.py:151-157`). Nothing reimplements a pass. | `ktir/<name>.ttir.mlir`, `ktir/<name>.ktir.mlir` |
| `bundle` | C++ KTIR **text** → SuperDSC, through `rust/triton-superdsc/triton-superdsc-lower`'s `emit_bundles` example, i.e. `lower(&text, &opts)` — which since `e2eade442` is `lower_value(&from_text(text)?, opts)`. The text boundary is the real one: it is how the C++ side hands off, and a parse at that *external* boundary is legitimate in a way a parse between our own stages would not be. | `bundles/<name>/bundle.mlir`, `sdsc_0.json`…`sdsc_N.json`, `manifest.json` |
| `program` | `DEEPTOOLS_PATH=… dxp_standalone -d <dir> -b sentient`, `FLEX_COMPUTE` unset, run in a scratch copy so the multi-megabyte `spyreCodeDir/` never lands in the tree. | `index.json`'s `stages.program` — per artefact file, its byte size and sha256 |

Three operational facts the driver had to encode, each of which would otherwise produce a
false pass:

* **The MLIR diagnostic is on fd 2, not in the Python exception.** `pm.run` raises a bare
  `PassManager::run failed`, which names nothing. The pass that refused is only in the C++
  diagnostic stream, so fd 2 is captured and read from *inside* the `except` block.
* **The program stage requires the ARTIFACT, not the exit code.** `dxp_standalone` prints
  nothing on success and **an abort returns 0 through a pipe**, so the existence of
  `spyreCodeDir/init_binary.bin` is the only acceptance signal.
* **`DBO_DEBUG=1` gives one `debug/sdsc_<i>/` per SCHEDULED SuperDSC.** That is the
  in-equals-out census. Counting `sdsc_*.json` in the emission directory instead would
  over-count, because `emit_bundles` does not clear its output — so the driver emits into a
  fresh directory, and a refusal leaves **no** directory rather than an empty one.

## THE RESULT, PER KERNEL AND PER CONFIGURATION

**Nine** of twelve configurations produce a program. **In equals out on every one of the
nine.** Two refuse, each naming its construct and its stage. One — `embedding_granite` —
reaches C++ KTIR and stops there **by scope, not by refusal**: the SuperDSC lowering moved to
another effort, so stages 2 and 3 were not run for it. Its KTIR golden is complete.

| kernel / configuration | KTIR | SuperDSCs in | scheduled | `init_binary.bin` | outcome |
|---|---|---|---|---|---|
| `attention_flash` non-causal | 230 L | 21 | **21** | 192 256 B | program |
| `attention_flash` causal | 328 L | 41 | **41** | 192 256 B | program |
| `swiglu_mlp` small | 121 L | 8 | **8** | 139 392 B | program |
| `swiglu_mlp` **Granite** | 121 L | 8 | **8** | **3 942 144 B** | program |
| `swiglu_mlp` tiled-K | 131 L | 12 | **12** | 197 888 B | program |
| **`rmsnorm` Granite** *(NEW)* | 91 L | 7 | **7** | 87 936 B | program |
| **`rope` 8 kv heads** *(NEW)* | 95 L | 6 | **6** | 81 664 B | program |
| **`rope` 32 query heads** *(NEW)* | 96 L | 24 | **24** | 102 400 B | program |
| **`embedding` Granite** *(NEW)* | 66 L | — | — | — | **KTIR ONLY** — stages 2–3 out of scope |
| **`embedding` BLOCK_M=128** *(control)* | 67 L | 1 | **1** | 76 416 B | program — **but see the two unproven choices below** |
| **`decoder_block` one layer** *(NEW)* | 455 L | ❌ | — | — | **refuses at the lowering: `tt.trans`** |
| **`decoder_block` two layers fused** *(NEW)* | 688 L | ❌ | — | — | **refuses at the lowering: `tt.trans`** |

Exact configurations, signatures, constexprs, launch grids, pinned grid points, per-file
byte sizes, the full `opFuncName` sequence per bundle and the sha256 of every program file
are in `index.json`. The table above is a summary of it, not a separate record.

### The op census of each program, by `opFuncName`

```
attention_flash_noncausal  21   batchmatmul 2  maxnonstick 1  sumnonstick 1  exp 2
                                mul 6  add 2  sub 2  maximum 1  realdiv 1  identity 3
attention_flash_causal     41   batchmatmul 4  maxnonstick 2  sumnonstick 2  exp 4
                                mul 11 add 5  sub 4  maximum 2  realdiv 1  identity 6
swiglu_mlp_small            8   batchmatmul 3  silu 1  mul 1  add 1  identity 2
swiglu_mlp_granite          8   batchmatmul 3  silu 1  mul 1  add 1  identity 2
swiglu_mlp_tiled_k         12   batchmatmul 3  silu 1  mul 1  add 3  identity 4
rmsnorm_granite             7   sumnonstick 1  mul 4  add 1  rsqrt 1
rope_kv8                    6   mul 4  sub 1  add 1
rope_q32                   24   mul 16 sub 4  add 4          (4 x the kv8 body)
embedding_bm128_control     1   mul 1   <- the gather RIDES ON this multiply
```

**The embedding gather is ONE SuperDSC for the whole kernel.** It is not its own
`opFuncName`: the gather rides on the op that consumes the gathered tile — here the
`EMB_SCALE` multiply — as an extra `computeOp_` field, and the store retargets onto that
multiply's output. So `load ids -> gather rows -> * 12.0 -> store` is one `mul`.

**`rmsnorm` needs no new device capability.** One `sumnonstick` for `tl.sum(x * x, 1)`,
one `rsqrt`, four multiplies and an add — the reduction is along the non-stick axis, which
is what `sumnonstick` is. **`rope` needs none either**: six elementwise ops, the two halves
reached by descriptor column offset rather than by any slice.

### `rope`'s two configurations take DIFFERENT PATHS, and that is a property of the width

`rope_kv8` is a 4 × 8 = **32**-block grid over 32 compute tiles, so the work loop is
`for %g = tile_id to 32 step 32` — **one trip**, induction variable folded to a constant by
the grid pin. `rope_q32` is 4 × 32 = **128** blocks — `for %g = tile_id to 128 step 32`,
**four trips per tile**, and `arith.remui %g, 4` (the grid delinearisation) survives the
pin. `remui` is not an affine function of a symbol, so no rolled address expression exists
and the loop is unrolled: 24 = 4 × 6. **The census difference is the unroll, not extra
work.** Both are in the experiment for exactly this reason, and the emission records the
unroll as a note rather than letting the count change silently.

## THE REFUSALS — TWO LEFT, TWO RESOLVED, AND ALL FOUR KEPT ON THE RECORD

A refusal is a result, and so is a refusal that later goes away: **nothing here is deleted
when it resolves.** Both `embedding` refusals are now gone and both keep their sections,
because in each case the *diagnosis* is the durable part — one of them was diagnosed by a
control that ran before any fix existed, and the fix then confirmed the control had read it
right. Two refusals remain, both on `decoder_block`'s `tt.trans`, and both are now owned by
another effort.

| # | configuration | construct | stage | status |
|---|---|---|---|---|
| 1 | `embedding` Granite | one-stick index tile | `PlanCorelets` (C++) | **resolved** — `single_corelet` plan |
| 2 | `embedding` BLOCK_M=128 | `construct_indirect_access_tile` | the lowering | **resolved** — produces a program |
| 3 | `decoder_block` one layer | `tt.trans` on a computed tile | the lowering | open, **not ours** |
| 4 | `decoder_block` two layers | `tt.trans` on a computed tile | the lowering | open, **not ours** |

### RESOLVED — `embedding` at the Granite width: `PlanCorelets` no longer refuses

`ktir/embedding_granite.ktir.mlir` now exists: **66 lines of C++ KTIR at the real Granite
width**, `V = 49159`, `D_MODEL = 4096`, `BLOCK_M = 64`. It used to refuse one stage before the
lowering, in our own C++ pass list:

```
embedding.py:109:15: error: 'ktdf.corelet_plan' op the two data_bounds ranges must be
disjoint and contiguous and non-empty (each lo < hi, one's hi == the other's lo),
but got [0, 0] and [0, 1]
```

Read from the pass rather than inferred: `PlanCorelets::fillSplit` computed
`k = tileSticks / kNumCorelets` as a **floor**, and the i32 index tile at `BLOCK_M = 64` is
exactly **one** 64-element stick — so `k = 0`, corelet 0 got the empty `[0, 0]`, and the
SP-E3-02 verifier rejected it.

**THE CONTROL THAT ESTABLISHED THE CAUSE, before the fix existed.**
`embedding_granite_bm128_control` is the same kernel at `BLOCK_M = 128`, which makes that
tile **two** sticks; it got *past* `PlanCorelets` while the Granite width did not. That is
what localised the bug to the one-stick tile rather than to the gather, to `V = 49159` or to
`D_MODEL = 4096`. **The fix confirms the control was reading it right** — a one-stick tile now
gets a `single_corelet` plan instead of a malformed two-way split:

```mlir
ktdf.corelet_plan pattern = "single_corelet" {
  ktdf.corelet 0 {data_bounds = [0, 1]}
}
```

One corelet with a non-empty range, so the disjoint/contiguous/non-empty invariant holds
**because the plan is right, not because the verifier was relaxed**. That distinction is the
whole value of the fix and is worth keeping in the record.

**The same one-stick tile is why `vector_add` and `mul` had no C++ KTIR either**, and they now
reach it too (60 lines each). Neither is a configuration of *this* set — this set is the
Granite kernels — so there is nothing to regenerate here; it is noted only so that an earlier
sentence in this file claiming they "already refuse" cannot be read as still true. The
crate-side goldens under `rust/triton-superdsc/triton-superdsc-lower/tests/goldens/` carry a
fuller version of that stale claim (a table headed "ONLY TWO OF THE FIVE NAMED FIXTURES
PRODUCE C++ KTIR TODAY"); that directory belongs to another effort and is theirs to correct.

The control keeps its place in the set. It is still **not a Granite configuration** — it is
labelled as such in `index.json` — and it is still the only width at which the *gather* has
been through stages 2 and 3, because those stages are now out of scope for the Granite width.

> **Scope, stated plainly:** `embedding_granite`'s outcome in `index.json` is **`KTIR ONLY`**,
> not `PROGRAM`. Stages 2 and 3 were not run for it. Writing them now would bake another
> session's in-flight lowering into a checked-in golden, which is the same reason a provenance
> refresh does not regenerate. The absence of `bundles/embedding_granite/` is a **scope
> boundary, not a refusal** — and `scope_note` on the entry says so, so the two can never be
> confused by someone reading only the JSON.


### RESOLVED — `embedding` past `PlanCorelets`: the gather PRODUCES A PROGRAM

`embedding_granite_bm128_control` now reaches `spyreCodeDir/init_binary.bin` at **76 416
bytes**, **1 SuperDSC in, 1 scheduled**. It used to refuse twice over: first in the emitter
(`no SuperDSC dataFormat_ for element type si32`), then after the value migration at the text
parse (`ktir-text-parse: unmodelled element type si32`), with
`ktdp.construct_indirect_access_tile` behind both.

**The gather is not its own `opFuncName`.** It rides on the op that consumes the gathered
tile as an extra `computeOp_` field, exactly as the vendor spells it in
`~/tmp/dt_src/dxp/test/test_gather_1core/sdsc_1.json`. What we emit, read back out of
`bundles/embedding_granite_bm128_control/sdsc_0.json`:

```
op mul     N_ = { mb_ 128, out_ 4096 }
  lds 0  Tensor0  INPUT       scale [1,1]  wl 2  SEN169_FP16  memOrg {hbm, lx}   the table
  lds 1  Tensor1  KERNEL_IDX  scale [1]    wl 4  SENUINT32    memOrg {hbm}       the ids
  lds 2  Tensor2  INPUT       scale [1,1]  wl 2  SEN169_FP16  memOrg {hbm, lx}   the splat
  lds 3  Tensor3  INPUT       scale [1,1]  wl 2  SEN169_FP16  memOrg {hbm, lx}   the output
  computeOp_  inputLabeledDs  ["Tensor0-idx0", "Tensor2-idx2"]
              outputLabeledDs ["Tensor3-idx3"]
              indirectAccessIndexLabeledDs ["Tensor1-idx1"]
  KERNEL_IDX class  layoutDimOrder_ ["mb"]  stickDimOrder_ ["mb"]  stickSize_ [32]
```

`SENUINT32`, `wordLength` 4, `memOrg_` `{hbm}` only, `scale_` `[1]`, and a **32**-element
stick — all as the vendor has it, and all four required by `dbo`'s
`GatherIndexConversion.cpp`, which synthesises the index-to-address SuperDSC
(`OpFuncs::INT32IDXTOADDR`) from ours and `DT_CHECK`s both the format (`:133`) and that every
`scale_` entry is 1 (`:135`).

**`ldsIdx_ 1` IS LOAD-BEARING, and this is the part that would not have been guessed.**
Appending the index **last** in `labeledDs_` passes our crate and is then rejected downstream
with `operand #1 does not dominate this use`, because the synthesised index-to-address op
feeds the gathered operand — so the index must appear *immediately after the tensor it
gathers*, which is what the vendor's `ldsIdx_ 1` is. An emitter that gets the field right and
the position wrong produces no artifact.

#### ⚠️ TWO CHOICES HERE ARE UNPROVEN, AND THE PROGRAM DOES NOT SETTLE THEM

Reaching `init_binary.bin` says `dxp_standalone` accepted the description. It does **not** say
the description is the right one, and two specifics are open — recorded here so a later reader
does not promote them by finding them in a golden:

* **The gathered tensor's RANK.** The vendor's is rank 3 (`N_` `mb_ 3`, `x_ 64`, `out_ 512`,
  class `["mb","out","x"]`) because *its* tensor is; ours is rank 2 (`mb_ 128`, `out_ 4096`,
  class `["out","mb"]`). A rank that cannot be justified is refused by name, and that refusal
  is itself a test — but "rank 2 is right for a rank-2 gather" is an inference from the
  vendor's rank 3, not a measurement.
* **The destination's class name.** Ours is `INPUT`; the vendor's destination is `OUTPUT`.

Neither is a numbers question and neither was executed. **The Granite width is still not
reached at all** — see the section above — so the gather has been exercised only at
`BLOCK_M = 128`, on a tile geometry chosen to get past `PlanCorelets`.

  assumption must not be applied to a `KERNEL_IDX` operand.

### 2 & 3. `decoder_block`, both configurations — `tt.trans` on a computed tile

> **Status: NO LONGER THIS EFFORT'S TO CLOSE.** All KTIR→SuperDSC work, `tt.trans`
> included, moved to the session on scratchy's KTIR→SuperDSC→`dxp_standalone` path. The
> refusal below is kept because it is a measured result and it names the construct — not
> because anyone here is still working it. **Both configurations reach C++ KTIR cleanly**
> (455 and 688 lines), so their `ktir/` goldens are complete and are oracle material
> exactly like the other ten.

```
REFUSED: KTIR operation `tt.trans` has NO SuperDSC mapping. `tt.trans` is not in
`OpFuncs` (deeptools sys-arch-spec/arch_enums.h) and this emitter will not guess.
```

Reported as `op 93` and `op 111` respectively — **`op N` is the operation's program-order
ordinal, not a source line.** A value path has no source lines; only the text parse, which
really is reading text, still says `line N`. The two are different coordinate systems and
the prefix is what distinguishes them.

The KTIR itself is `ktir/decoder_layer_one.ktir.mlir:140` and `:142` (and
`decoder_two_layers.ktir.mlir:124` onward, four of them, two per layer):

```mlir
%qk_54 = tt.trans %k1r_51 {order = array<i32: 1, 0>} : tensor<64x64xf16> -> tensor<64x64xf16>
```

It comes from `k1r.T` / `k2r.T` in the score contraction. **This is NOT the same construct
as `attention_flash.py`'s `.T`.** There the transpose is on a *descriptor load*, so it
folds into `access_tile_order` and is recorded as a host obligation — the caller presents
the tensor in the layout the DDL demands. Here the operand is a **computed register tile**
(the RoPE-rotated `k`), so the host cannot present it, and reusing the transposed-read note
would be a wrong answer dressed as a contract.

What exists on the device, read from the enum and the templates:
`OpFuncs::INTERSLICETRANSPOSE_FP16` (spelled `"interslicetranspose_fp16"`,
`arch_enums.cpp:494`), bound by `ddc/ddl_templates/inter_slice_transpose.ddl`, plus
`SFP_ReadLX_TRANSPOSE_FWDL0` and `PT_BLK_TRANSPOSE_LOAD`. **But unlike the gather there is
no bundle-input-level example**: the only shipped file
(`ddc/ddl_templates/test/sdsc_interslicetranspose.json`) is a fully *scheduled* DDC-level
file with folds and `coreIdToWkSlice_`, named `"mul_78-Gelu"`, in which the transpose is a
stage inside a larger op rather than a top-level one. So the transpose is a genuinely open
lowering question, and the gather is not.

**The two decoder configurations reach C++ KTIR cleanly** — 455 and 688 lines — so the
refusal is entirely in the SuperDSC lowering. Their KTIR is checked in and is usable golden
material as it stands.

## What changed in the lowering to get `rmsnorm` and `rope` through

Recorded here because it bears on how the goldens should be read, and because one of the
three was a **silent wrong answer** rather than a gap. Full reasoning and the vendor
evidence are in commit `0a74f5215` and in `tests/broadcast_axis.rs`.

1. **`scale_`'s `-1` sentinel now lands on the axis `tt.expand_dims` actually lifted.**
   `scale_` is positional in the operand's class's `layoutDimOrder_`, and the emitter
   previously wrote `('mb','out')` / stick `('mb',)` / `scale (1,-1)` for *every* reduced
   operand regardless of which axis was lifted. `rmsnorm.py` has both in one line —
   `x * r[:, None] * n1[None, :]` — so one of the two multiplies was mis-described and
   nothing said so. The stick axis is a **per-DSC** choice forced by the broadcast axis, and
   both assignments are in the vendor's own shipped inputs; a DSC that would need both at
   once is now refused by name.
2. **The rank-1 identity `affine_map<(d0) -> (d0)>` is a real access tile.** HF's RMSNorm
   weight is 1-D (`nn.Parameter(torch.ones(hidden_size))`), so its descriptor emits it.
3. **A multi-trip loop with no `iter_args` is unrolled** — see `rope_q32` above.

## THREE SILENT PARSER HOLES THE GATHER WORK EXPOSED — one of them on experiment 2's own input

None of these changes any artefact under this directory. They are recorded because the whole
point of a golden set is that a later reader can trust it, and each of these was a case where
the parse produced a *plausible wrong answer* rather than a refusal.

* **`func.func` fell through to the generic reader.** Only `tt.func` routed to the real
  function parser, so a `func.func` kernel was read with **no symbol name and no region
  arguments**, silently — surfacing later as "has no arguments". No fixture here caught it
  because **every `make_ktir` golden in this directory is `tt.func`**. But `func.func` is
  exactly what `ToSchedulerKTIR` produces, so the hole sat directly on experiment 2's input
  form while experiment 1 could never have found it. Both kinds now route.
* **The gather was typed with its INDEX VECTOR's type.** The generic reader took an
  operation's type tail from the last ` : ` on the opening line, and
  `indirect(%ids : memref<256xsi32>)` puts one *inside the operand list* — so the real result
  type on the closing line was never read, and anything asking the gathered tile's shape got
  `256xsi32` instead of `128x4096xf16`.
* **`ldsIdx_` position was load-bearing, and getting the fields right was not enough.** See
  the gather section above: appending the index last passed our crate and was rejected
  downstream with `operand #1 does not dominate this use`.

**The class, stated once:** a text reader with a generic fallback fails by *guessing*, not by
refusing, and the guess can be well-formed enough to reach a program. Every one of these three
was found by a control that needed a shape the fallback happened to get wrong — not by any of
the eleven kernel configurations.

## THE TEXT→VALUE MIGRATION CHANGED NO SUPERDSC, AND FIXED THE CALLER CONTRACT

The set was regenerated end to end after `e2eade442` moved the lowering onto KTIR **values**
with `from_text` at the text boundary. Compared against the pre-migration commit, across all
twelve configurations:

* **Not one `sdsc_*.json` byte changed.** Not one census, not one `opFuncName` sequence, not
  one `init_binary.bin` byte size, not one program `sha256`. Measured by comparing
  `index.json` field by field: **0 of 12 configurations differ** on any of those.
* **`bundle.mlir` and `manifest.json` changed in exactly two ways**, both in
  human-readable provenance text:

**(a) Float rendering is now EXACT, and that is a fix, not drift.** The caller contract said

```
arg 3: constant_splat -- caller must fill 64x1 f16 with 0.0000100136      <- lossy
arg 3: constant_splat -- caller must fill 64x1 f16 with 0.000010013580322265625
```

`0.0000100136` is a 6-significant-digit print, and it is **not the f16 value** — a caller
seeding it would seed a different number. The three constants that appear, checked against
`struct.pack('<e', …)` rather than taken on faith:

| constexpr | source | the f16 it becomes — and what the manifest now says |
|---|---|---|
| `EPS` | `1e-05` | `1.0013580322265625e-05` (subnormal: `168 × 2⁻²⁴`) |
| `INV_D` | `1/4096` | `0.000244140625` (a power of two, exact) |
| `LOG2E` | `1.44269504` | `1.4423828125` (`1 + 453 × 2⁻¹⁰`) |

Each of the three new strings round-trips through f16 to itself; none of the three old ones
did. This repo has been bitten here before — `35835c92a`, "the float-repr bug that only a
1e-05 constexpr could expose".

**(b) Notes say `op N` where they said `line N`.** `N` is the operation's program-order
ordinal; a value path has no source lines. The **text parse** still says `line N`, because it
is genuinely reading text. Different coordinate systems, distinguished by the prefix — so a
diff must not treat `op 41` and `line 49` as disagreeing about anything.

**What this buys experiment 2:** the SuperDSC files, the census and the program digests are
the stable part of this golden set and are safe to diff byte-for-byte. `manifest.json` and
`bundle.mlir` carry provenance prose whose *wording* is not a contract — diff their structure,
not their comments.

## Provenance and hazards for whoever diffs against this

* **The pinned grid point is part of the identity of a bundle.** `index.json` records it
  per configuration. It is **3, not 0**, for `attention_flash` causal: at query block 0 the
  off-band sweep is empty and would prove nothing about causal. Everything else is 0.
* **`ktir/` and `bundles/` are byte-comparable; the program is compared by sha256.** The
  program artefacts are megabytes and are deliberately not checked in; `index.json` carries
  each file's byte size and sha256 instead, which is what a diff actually needs.
* **`tests/broadcast_axis.rs` reads the `ktir/` goldens directly** rather than keeping a
  second copy under the crate's `tests/data/`, so a regeneration cannot leave that suite
  testing a stale duplicate.
* **A refusal leaves no `bundles/<name>/` directory.** Absence is the record; there is no
  partial bundle anywhere under here. A refusal also cannot *delete* one: stage 2 emits into
  a scratch directory and moves it into place only on success, because an early version
  cleared the output first and a run that failed for an unrelated reason took all 143
  committed bundle files with it.
* **All twelve configurations have a `.ttir.mlir`; eleven have a `.ktir.mlir`.**
  `embedding_granite` refuses inside `make_ktir`, so its C++ KTIR does not exist — but its
  TTIR does and is checked in, so experiment 2 can be diffed at that checkpoint even for
  the one configuration with no KTIR oracle. The absence of exactly one `.ktir.mlir` is the
  record of that refusal, not an omission.
* **A build failure is not a refusal.** If the lowering crate does not compile, the driver
  stops with `FATAL` before any kernel runs. It used to shell out per kernel and report the
  compiler's own error as twelve kernel refusals, which was a lie about every one of them.
* **Verified against `index.json`:** every configuration's on-disk `sdsc_*.json` count
  equals its recorded `sdsc_count`, and equals its scheduled count, for all nine programs.
  Zero mismatches.
* **THE GATHER TOUCHED SHARED EMITTER CODE AND REGRESSED NOTHING.** Checked field by field
  against the pre-gather commit -- outcome, `sdsc_count`, the full `opFuncName` sequence,
  `sdsc_scheduled`, `init_binary.bin` byte size, every program artefact's `sha256`, and every
  bundle file's byte size: **exactly one of twelve configurations changed, and it is the one
  that was supposed to** (`embedding_granite_bm128_control`, refusal -> program). The other
  eleven are identical, and on disk only that new bundle directory and `index.json` appeared.
  A gather fix landing in shared code is precisely where a silent regression would sit, so
  this is asserted rather than assumed.
