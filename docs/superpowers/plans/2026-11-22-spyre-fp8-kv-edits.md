# Spyre fp8 KV cache — the edit plan, restart 3 (stamp-ready)

Base: main `48c86605ae`, worktree `/Users/nickm/git/scratchy/.claude/worktrees/spyre-kv-fp8`.
Facts (binding): `docs/superpowers/plans/2026-11-20-fp8-kv-torch-spyre-facts.md` (**FACTS**, cited
`FACTS §X`). Restart 2's plan: `.claude/restart2-ref/2026-11-20-spyre-fp8-kv-edits.md` (**R2**, cited by
its edit IDs). The coder types every edit verbatim with the Edit tool and runs the step's gate. Paths
are relative to the worktree.

Abbreviations (R2's): `SPLICE` = `backends/spyre/crates/triton/splice/src/lib.rs`, `WF` =
`backends/spyre/crates/ktir/ktir-superdsc/src/emit/whole_function.rs`, `LK` =
`backends/spyre/crates/ktir/ktir-superdsc/src/emit/lower_ktir_to_superdsc.rs`, `OPSPEC` =
`backends/spyre/crates/ktir/ktir-superdsc/src/superdsc_opspec.rs`, `PW` =
`backends/spyre/crates/ktir/ktir-superdsc/src/ir/bridge/tiled_op_sdsc_op/pointwise.rs`, `EMIT` =
`backends/spyre/crates/ktir/ktir-superdsc/src/emit/mod.rs`, `BAKE` =
`backends/spyre/crates/spyre/src/lower_subtile_tape_to_superdsc.rs`.

**Anchors.** R2 was written against this same base. Verified on this tree: the whole of R2's diff
(`.claude/restart2-ref/restart2-vs-main.patch`) applies to main with no offset and no fuzz
(`patch --dry-run -p0 -i .claude/restart2-ref/restart2-vs-main.patch`, every file "patching file", no
"Hunk … succeeded at offset"). So every R2 anchor quoted below as "unchanged" is main's text, unique
where R2 said it was; the coder types those edits from R2 by ID. Edits whose text changes are written
out here in full.

**`Cargo.lock`.** `backends/spyre-hw/Cargo.lock` is never hand-edited: the first build after 1.2a-c
adds the `scratchy-kv` package entry itself.

---

# PART 1 — THE SALVAGE (Steps 1 and 2 of R2, checked against FACTS)

## 1.0 The carried-edit ledger

Every R2 edit of Steps 1–2 (and the fixes that touch them), its status here, and what FACTS says
about it. "unchanged" = type R2's text as written. "re-anchored" = typed from THIS file. "replaced" /
"dropped" = not typed; the reason is the FACTS line on the right.

| R2 ID | file | status | FACTS check |
|---|---|---|---|
| 1.1a | triton-frontend `codegen.rs` | **dropped** (superseded by 1.1a′, R2's own pod result) | — |
| 1.1a′ | triton-frontend `codegen.rs` | **re-anchored** (anchor = main's 3-line arm; text §1.1 below) | Triton's `tensor_descriptor_base.dtype` is its block's element type: needed for the reference's `k_page.dtype != q.dtype` reads (FACTS A.3 step 3, A.4, A.5) |
| 1.1b | triton-frontend `codegen.rs` | unchanged | a dtype comparison is Python's at trace time — the reference's `if k_page.dtype != q.dtype` (SI `page_attn_decode.py:94-96`) folds; no bool constexpr |
| 1.1c | triton-frontend `tests/dtype_compare.rs` (new) | unchanged | fixture of 1.1b |
| 1.2a–1.2j | `kv` crate, spyre-hw/spyre Cargo chain, `cli/main.rs`, splice `Cargo.toml` | unchanged | the build's one selection (`kv_cache_dtype="fp8"`, FACTS A.0) as a type; no runtime toggle |
| 1.3a | ktir-core `codec.rs` | unchanged | `torch.finfo(float8_e4m3fn).max` = 448 (FACTS A.2 `quantize_kv` step 1) |
| 1.3b | `OPSPEC` (`FloatFormat`) | unchanged | the bound per format, read by the write's `FMAX` (FACTS A.2) |
| 1.3c | `SPLICE` (`KvFormat`, `kv_pool_ptr`) | unchanged | the pools' element type `float8_e4m3fn` (FACTS A.1 step 1) |
| 1.3d | `SPLICE` (`float_df`/`stated_float`) | unchanged | `kFloat8_e4m3fn` ↔ `SEN143_FP8` (FACTS A.1, TS `types_mapping.h:37,82`) |
| 1.3e | `SPLICE` `lower_kv_write` | **split**: the pool-pointer half is **re-anchored** here (§1.3 below); the `FMAX` constexpr half moves to Part 2 W with the kernel that reads it | — |
| 1.3f, 1.3g | `SPLICE` `lower_attn` | unchanged | the reads take `k_pages`/`v_pages` at the cache dtype (FACTS A.3–A.5) |
| 1.3h | `SPLICE` comment | unchanged | describes 1.3f-g |
| 1.4a | `kernels/reshape_and_cache_head_major.py` | **replaced** by Part 2 W (it DEPARTS: it quantizes per head on loaded `[T,128]` column windows, where the reference runs ONE `quantize_kv` over the whole `[T,8,128]` key and only then the 8 per-head `index_copy_` — FACTS A.2 steps 1–2, SI `ops/reshape_and_cache.py:47-52`; torch-spyre realizes ONE `qfp8ch` over `x.get_size()` — FACTS B.1 "Lowering", TS `lowering.py:2969-2992`) | — |
| 1.4b | `kernels/page_attn_head_major_decode.py` | unchanged | the reference's widen at its position (SI `page_attn_decode.py:94-96`, FACTS A.3 step 3) |
| 1.4c | `kernels/page_attn_head_major_prefill.py` | unchanged | SI `page_attn_prefill.py:108-110` (FACTS A.4) |
| 1.4d | `kernels/batched_decode_head_major.py` (both loops, `replace_all`) | unchanged | SI `batched_decode.py:171-173` (FACTS A.5) |
| 1.5a–1.5e | triton-ktir `dot_to_linalg.rs` | unchanged | the reads' dots are fp16 dots on the widened page (FACTS A.6 "No fp8 matmul on the KV path"); the W8A8 leg stays the W8A8 fold's |
| 1.5f | `dot_to_linalg.rs` tests | unchanged | match / near-miss / neighbour fixtures |
| 1.6a–1.6k | ktir-core `ddl_binds.rs`, `OPSPEC`, `EMIT` | unchanged | `fp8todl16` (DDL `quantization_double_pad.ddl:60`, FACTS C.2); its output arrangement **STANDARD** (FACTS B.2: `ea_map` defaults to STANDARD for (fp8, fp16), `dtype_ops.py:223-236`) — 1.6f's row states `ElementArrangement::Standard` |
| 1.6l | `EMIT` `convert_opspec` | **dropped** (DEPARTS: it gives `fp8todl16` `dl16tofp32`'s RowBlocked rank-3 convert form; torch-spyre tags `dl16tofp32`'s output DL16_TO_FP32, a staggered EA (`dtype_ops.py:57`), and `fp8todl16`'s output STANDARD (FACTS B.2), and builds its output as a plain dense layout, never the staggered-preserve path (`propagate_layouts.py:691-748`). Part 2 R2-op states the convert's own form.) | — |
| 1.7a, 1.7b | `BAKE` `bundle_registry` | **dropped** (no reader: R2 needed it for the write's `±FMAX` `max`/`min` registry splats; Part 2 W lowers the saturation as the device's `clip`, whose bounds reach the descriptor's `constantInfo_` by name (`OpFunc::op_consts`, `OPSPEC:2009`), as the W8A8 chain's clip does — no registry slot) | torch-spyre's clamp is `clip` with `clipMin`/`clipMax` constants (FACTS B.5, TS `spyre_kernel.py:236-244`) |
| 2a, 2b | `placement.rs` `allocate_row_outermost<DF>` | unchanged | `head_major_kv_layout`: `[num_pages, block, ceil(hd/eps), eps]` at the cache's own `eps` (FACTS A.1; granite fp8 `[P,128,1,128]`, `P·16384` B) |
| 2c, 2d, 2e | `BAKE` pool set, `nbytes`, `allocate_row_outermost::<KvFormat>` | unchanged | pool bytes at the cache dtype (FACTS A.1) |
| 2f | `LK` `view_elem` / `indirect_view_elem` / `param_elem` | unchanged | the KTIR view's stated element type — the data→type door Part 2's arms read |
| 2g | triton-ktir-superdsc `layout.rs` `allocate_paged_cache` | unchanged | as 2b, for the one-kernel bundle |
| 2h | ktir-superdsc test `zz_the_rank_two_page_read_gathers_through_one_stick_permute.rs` | unchanged | fp16 fixture names its format |
| 2h′ | spyre test `zz_page_gather_sweep.rs` | **re-anchored** (R2 applied it under 2h's "any other caller" note; text §2 below) | fp16 fixture names its format |
| Fix A (`DataFormat: Copy + Debug`) | `OPSPEC` | moved to Part 2 (its readers are Part 2's generic frames) | — |
| Fixes B, C | `PW` re-export, `WF` | not carried (their targets, R2 4b/4c, are rebuilt in Part 2) | — |
| Fix 2a (`staggered_sources`) | `WF` | carried in Part 2 (a measured trap: the arrangement pre-pass classifies every op) | — |
| Fix 1 (pointwise tests' `assemble_index_select::<Fp16, …>`) | `PW` tests | carried in Part 2, with the generic gather it follows from (it touches no Step-2 code) | — |

**Pool zero fill (FACTS A.1 step 4, B.4).** Not an edit: main's runtime zero-fills every KV bank on the
device at boot (`backends/spyre-hw/crates/card/src/superdsc_exec.rs:3150-3153`, `Stream::fill_zero`,
the comment "the KV pool … holds only zeros at boot … a DEVICE-SIDE FILL"), and byte `0x00` is e4m3
`+0.0`. That is the reference's `p.view(torch.uint8).zero_()` (a device `MEMORY_FILL` over every
stick). Same.

## 1.1 Edit 1.1a′ (re-anchored) — `backends/spyre/crates/triton/triton-frontend/src/codegen.rs`

Anchor (main's arm, unique):
```rust
                    ("dtype", Val::Ir(id)) => {
                        return Ok(Val::Dtype(self.sem.ty(*id).scalar().clone()));
                    }
```
Replacement:
```rust
                    ("dtype", Val::Ir(id)) => {
                        return Ok(Val::Dtype(self.sem.ty(*id).scalar().clone()));
                    }
                    // A descriptor's dtype is its block's ELEMENT type (Triton's
                    // `tensor_descriptor_base.dtype`, `core.py`) — what `k_dst.dtype` names in
                    // `reshape_and_cache_head_major.py` — read off the handle's `!tt.tensordesc`.
                    ("dtype", Val::Desc { handle, .. }) => {
                        return match self.sem.ty(*handle) {
                            Type::TensorDesc(_, e) => Ok(Val::Dtype((*e).clone())),
                            other => Err(Error::new(
                                format!(
                                    "a tensor descriptor's handle is `{other}`, not a \
                                     `!tt.tensordesc`, so it states no element type"
                                ),
                                pos.line,
                                pos.col,
                            )),
                        };
                    }
```
Then type R2 1.1b and 1.1c unchanged.

## 1.2 Edits 1.2a–1.2j — unchanged (type from R2)

## 1.3 Edits 1.3a–1.3d unchanged; 1.3e re-anchored (pool pointers only); 1.3f–1.3h unchanged

**Edit 1.3e (pool-pointer half, re-anchored)** — `SPLICE`, `lower_kv_write`. Anchor (main, unique):
```rust
    for (p, ty) in [
        ("key", "*fp16"),
        ("value", "*fp16"),
        ("k_rows", "*fp16"),
        ("v_rows", "*fp16"),
        ("row_index", "*i32"),
    ] {
        signature.insert(
            p.to_string(),
            ArgSpec::parse(ty).map_err(|e| e.to_string())?,
        );
    }
```
Replacement:
```rust
    for (p, ty) in [("key", "*fp16"), ("value", "*fp16"), ("row_index", "*i32")] {
        signature.insert(
            p.to_string(),
            ArgSpec::parse(ty).map_err(|e| e.to_string())?,
        );
    }
    for p in ["k_rows", "v_rows"] {
        signature.insert(p.to_string(), kv_pool_ptr());
    }
```
(The `FMAX` constexpr R2 1.3e also inserted here is Part 2 W's, typed with the kernel parameter it
binds: a constexpr the kernel does not declare is a front-end refusal.)

Note for 1.3c: with 1.3e's `FMAX` half moved, `FloatFormat` is not yet read in `SPLICE` at the end of
Part 1, so 1.3c's `use ktir_superdsc::superdsc_opspec::{DataFormat, FloatFormat};` becomes
`use ktir_superdsc::superdsc_opspec::DataFormat;` until Part 2 W adds `FloatFormat` back (clippy
`-D warnings` refuses an unused import). Concretely, in 1.3c's replacement text, the first line is:
```rust
use ktir_superdsc::superdsc_opspec::DataFormat;
```
and Part 2 W's first edit restores `{DataFormat, FloatFormat}`.

## 1.4 Edits 1.4b, 1.4c, 1.4d unchanged; 1.4a NOT typed (Part 2 W)

## 1.5 Edits 1.5a–1.5f unchanged

## 1.6 Edits 1.6a–1.6k unchanged; 1.6l NOT typed

## 1.7 NOT typed

## 2 — Step 2: edits 2a–2h unchanged, plus 2h′

**Edit 2h′** — `backends/spyre/crates/spyre/tests/zz_page_gather_sweep.rs`. Anchor (unique):
```rust
    l.allocate_row_outermost(pool_tid)
```
Replacement:
```rust
    l.allocate_row_outermost::<ktir_superdsc::superdsc_opspec::Fp16>(pool_tid)
```

## Part 1 gate

1. `cd backends/spyre && cargo check --workspace` clean; then (pod)
   `cargo test -p triton-frontend --test dtype_compare`, `cargo test -p triton-ktir dot_to_linalg`,
   `cargo test -p ktir-core ddl_binds`, `cargo test -p ktir-superdsc`,
   `cargo test -p scratchy-target-spyre --test zz_page_gather_sweep`.
2. `cd backends/spyre-hw && cargo --config patches/oco.toml build --release -p scratchy-target-spyre-cli
   --features spyre-hw,model/granite-3.1-8b-instruct,quant/fp8-dynamic-per-channel,kv/fp8`.
   Expected at the end of Part 1: the bake's FIRST refusal is the KV write's, at its fp8 store — the
   kernel is still main's (1.4a is Part 2 W), so at a rung with `T ≥ 8` the front end refuses the
   scatter of an f16 value through an f8E4M3FN descriptor (`triton-frontend/src/semantic.rs:1485-1490`),
   and at `T < 8` the per-row `store` casts implicitly and the emitter refuses the f16→f8 `truncf`
   (`convert_kind_of`). Quote the refusal text. Any refusal in placement, pool sizing or footprint, or
   any refusal naming a read kernel's front end / `from_ttir` / `dot_to_linalg`, is a Part 1 defect.

## Part 1 purity

| edit | P1 const | P2 constexpr | P3 once | P4 no refusal | P5 newtypes | P6 const generic | P7 no model | P8 target surface | P10 one source | P11 hygiene | P12 tests |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1.1a′/1.1b | trace-time fold | — | Triton's dtype law once, in the front end | the ordered-compare refusal is kernel-source compile time | `Val::Dtype` | — | none | target-neutral front end | — | present tense | 1.1c fixture |
| 1.2 | build-time selection | — | ONE alias; the one `cfg` pair at `KvFormat`'s definition (tp-world precedent); no `cfg` in the ladder | — | a type, not a bool | `KvFormat` | none | spyre homes only | the alias is the only selection | no `#[allow]` | type carries it |
| 1.3 | consts | — | e4m3 bound once (`E4M3_FMAX`); Triton↔format map once (`float_df`, read both ways) | named compile-time panic in `stated_float` | `FloatKind`/`Df` | `FloatFormat` assoc const | — | format facts in the format module | `stated_float` derives from `float_df` | — | — |
| 1.4b-d | constexpr branch, folded | — | the reference's condition at its position | — | no `KV_FP8` | — | no head-size branch | kernels | port of the reference's lines | docstrings | — |
| 1.5 | — | — | ONE predicate `fp8_weight_load` | near-miss left unclaimed; attempted pattern refused by name at compile | — | — | over ops/types | shared Triton pass | the peel moved into the predicate | test named for what it states | match + near-miss + neighbour |
| 1.6a-k | const table | — | the admission pin compares formats (no pair list) | — | `DeviceOp`/`DdlDf` | `DEVICE_OPS` | — | one row + one `OpFunc` arm | spelling read from the enum | citation present | table consts are build errors |
| 2 | bake-time sizes | — | pool format = `KvFormat`; arrangement stick = the type's | named bake errors only; other formats refused by name at the door (lead ruling 11-20) | `DF` type parameter, no `df` argument | `allocate_row_outermost<DF>` | none | spyre homes | `flat_df` is the existing constructor; `view_elem` is the one reader | present tense | existing laws |

---

# PART 2 — THE EMITTER, REBUILT TO TORCH-SPYRE'S DEVICE-OP SEQUENCE

Lead rulings (11-22) this part implements: **Q-W1 = option A** (the kernel keeps its per-head
`clamp → narrow → store` chains; a principled column-family fold lowers the chains whose windows tile
the key as ONE whole-key `clip` + `clip` + `qfp8ch`, then main's coalesced scatter run); **dim order**
(the widen writes exactly main's fp16 page landing, EA STANDARD, recorded below as a justified
difference); **a card round-trip acceptance leg** (Step 6).

## 2.0 The governing table — every device op torch-spyre emits on this path, in order

Granite-3.1-8b: `T` token rows, 8 kv heads, head 128, block 128, `P = NB·8` folded page rows. TS = the
torch-spyre op (FACTS citation, or `superdsc.py` read here); SC = the op scratchy emits after this
plan. "Stick" = elements per stick of that operand's own format (TS `superdsc.py:1789-1790`:
`arg.device_dtype.elems_per_stick()`; SC: each `TensorArg`'s `with_df`).

### Write (one KV writer, per step; K and V each)

| # | TS op, operands, layouts, sticks | SC op (file:fn), operands, frames, sticks, dataformats | same / difference |
|---|---|---|---|
| W1 | `realdiv` by `k_scale`, skipped at 1.0 (FACTS A.2 `quantize_kv` step 2, B.5) | none: the scale is absent, read as 1.0 (R2 ruling Q2) | same (skipped) |
| W2 | `clip` (clamp_min): `clipMin = −448`, `clipMax = +DLFLOAT16_MAX` (`lower_clamp(x, min, None)`, TS `lowering.py:1505-1579`, bound at `:1537-1546`); input/output `[T,8,128]` SEN169_FP16, stick 64 (FACTS B.5, end-to-end step 2) | `clip` — `EMIT::assemble_clip`, called by `WF::narrow_chain` (Step 4) — `[T, 1024]` stick-major SEN169_FP16 in (the key parameter) and out (a minted `[T, 1024]` intermediate), stick 64; `constantInfo_` `clipMin = sen169(−448)`, `clipMax = sen169(DLFLOAT16_MAX)` ([`ClipBounds::at_least`]) | same op, same bounds. **Difference (inert):** scratchy's one SEN169 encoder (`EMIT::sen169_bits`) saturates DLFloat16's exponent 63 to its largest exponent-62 word `0x7DFF` (≈4.29·10⁹), and torch-spyre's own word for `DLFLOAT16_MAX` is UNKNOWN (its `generate_constant_info` encoder was not read). Both exceed every fp16 value (\|x\| ≤ 65504), so the bound never binds — **Q-R3, ruled 11-22:** scratchy's one encoder (`sen169_bits`) is the source of the word; no new encoder |
| W3 | `clip` (clamp_max): `clipMin = −DLFLOAT16_MAX`, `clipMax = 448` | `clip` (`ClipBounds::at_most`), `[T,1024]` → a second minted `[T,1024]` | same (as W2). The ORDER of W2/W3 in torch-spyre is UNKNOWN (FACTS B.5); scratchy's is the kernel's (`tl.minimum(tl.maximum(k, −FMAX), FMAX)`: clamp_min first). The values do not depend on the order: `min(max(x, lo), hi) = max(min(x, hi), lo)` for `lo ≤ hi` (derived) |
| W4 | `qfp8ch` over the WHOLE `[T,8,128]`: input stick 64 (SEN169_FP16), output stick 128 (SEN143_FP8), output dim order = input's with the stick-count dim rescaled 16 → 8 (`_qfp8ch_stl`, TS `propagate_layouts.py:494-525`), EA tag QFP8CH; one op (FACTS B.1, end-to-end step 3) | `qfp8ch` — `EMIT::assemble_convert(…, "qfp8ch", T, 1024, rb(clip_out), rb_df(q, Fp8))`, the W8A8 activation quantizer's own call (`ktir_matmul_fp8.rs:527-536`, `fq_afp8_op`) — rank-2 stick-major `[mb = T, out = 1024]` (rank-3 `[mb, out, y]` at `T = 1`, `convert_opspec`), input SEN169_FP16 stick 64, output SEN143_FP8 stick 128, two `primaryDsInfo_`; one op over the whole key | same. TS's input dim order is UNKNOWN (FACTS B.1: propagation over the fused-QKV view not traced). Scratchy's key is its own `[T, 1024]` tensor, stick-major = torch-spyre's DEFAULT layout of a 2-D `[T, 1024]` fp16 tensor, device `[16, T, 64]` (`spyre_tensor_impl.cpp:115-134`: `device_size[i−1] = extent(dim_order[i])`, `device_size[r−1] = extent(dim_order[0])`); `_qfp8ch_stl` of that is `[8, T, 128]` = scratchy's stick-major fp8 `[T, 1024]` (derived). **Q-R1 ruled 11-22:** no re-layout |
| W5 | 8 K + 8 V `identity` scatters, each `[T,128]` fp8 rows of `key[:, h]` through its own `[T]` index into the `[NB·8·128, 128]` QFP8CH rows view; one row = one 128-element SEN143_FP8 stick; bytes moved unconverted; no restickify for an fp8 source (FACTS B.3, A.2, end-to-end step 4) | `identity` + index — `LK::scatter_run::<Fp8>` → `assemble_scatter_copy` — from the `[T,1024]` stick-major fp8 intermediate, whose row `h·T + t` is head `h`'s token `t` (stick column `h` of row `t`, `(h·T + t)·128`); one entry = one fp8 stick, df SEN143_FP8, read in place (one stick a row: no relayout, `LK` `load_scatters`' own law); the 8 per-head stores COALESCE into one run of `8T` entries, cut into `index_runs(8T)` ops | same op, same rows, same bytes, no restickify. **Difference (op count):** scratchy's write index is ONE head-major buffer of the 8 `[T]` runs (`PagedStepTids::write_rows`, `SPLICE:1862-1864`), and an index op's base must be a whole index stick (`EntryBase::of_entries`); a per-head op would start at entry `h·T`. So the stores coalesce, which is main's own fp16 law (`LK:1076-1084`). torch-spyre's 8 index TENSORS each start at 0 (SI `spyre_attn.py:1930-1944`) |
| W6 | pool `[P,128,1,128]` SEN143_FP8, QFP8CH tag, `P·16384` B, zero-filled by a device fill (FACTS A.1, B.4) | `allocate_row_outermost::<KvFormat>` (Part 1, R2 2b) `[P·128 rows][1 stick][128]`, `P·16384` B; zeroed by the boot device fill (`superdsc_exec.rs:3150-3153`) | same; the QFP8CH tag is host metadata torch-spyre never puts into an SDSC (FACTS B.2: the codegen consults `element_arrangement` only for QFP8WT, `superdsc.py:1785-1817`), and scratchy states none |

### Read (per page tile; decode `[8,128,128]`, prefill `[8,128,128]`, batched decode `[E·8,128,128]`)

| # | TS op | SC op | same / difference |
|---|---|---|---|
| R1 | gather: `identity` + index over the SEN143_FP8 pool, whole 128-element sticks, indexed dim outermost (FACTS B.3, end-to-end step 5); whether it fuses into R2 is UNKNOWN (FACTS B.2) | `identity` + index — `PW::index_select_opspec::<Fp8>` (R2 4b) via `WF::page_legs::<Fp8>` — iteration `[mb = entries, y = blocks, x = rows, out = 128]`, input the pool read row-major (`["out","x","y","mb"]`), output an fp8 page local (a buffer minted for the gathered load's own value), at fp8 one stick a row, so its bytes are `[entries·blocks][rows][128]` = the gathered `[8,128,1,128]`; df SEN143_FP8, stick 128 | same op and bytes, IF torch-spyre does not fuse the gather into the widen — **Q-R2 ruled 11-22:** the unfused form stands |
| R2 | `fp8todl16` (`PointwiseOp("fp8todl16")`, TS `dtype_ops.py:127-129,145`, `spyre_kernel.py:398-427`): input the gathered fp8 (stick 128), output fp16 stick 64, EA STANDARD (`ea_map` default, `dtype_ops.py:223-236`), output layout the dense rebuild from the output host size with `dim_order = range(rank)` (`propagate_layouts.py:737-748`) — for host `[8,128,128]`, device `[128, 2, 8, 64]` (derived from `spyre_tensor_impl.cpp:115-134`); the iteration is the op's host dims, no virtual `mb` (`superdsc.py:2396-2414` injects `mb=1` only when the op has nothing but the stick dim); per-arg stick sizes (`superdsc.py:1789-1790`); no arrangement field in the SDSC (FACTS B.2) | `fp8todl16` — `PW::widen_landing_opspec` (R2 4b) via `WF` `PageFormat for Fp8` — input the fp8 page local (stick 128, SEN143_FP8, `Role::Input`), output main's fp16 page landing (stick 64, SEN169_FP16, `Role::Output`), both declared by `landing_arg` (the frame `index_select_opspec` lands a 16-bit page in); iteration `[mb, y, x, out]`; work division keeps a whole stick of the wider format (128, `convert_stick_df`); EA STANDARD (DeviceOp row, Part 1 1.6f); no arrangement field | **Difference 1 (dim order, justified — lead ruling 11-22):** the output is main's landing, device `[entries·blocks][2][rows][64]`, i.e. `[8, 2, 128, 64]` at decode = the STL with `dim_order [1,0,2]`, not torch-spyre's `[128, 2, 8, 64]`. Each side's convert lands in the order its own contraction reads: torch-spyre's propagation picks the dense order and its matmul consumes that order; scratchy's batched contraction and restickify read a page as per-block stick-major blocks (`WF:5840-5873`, `batch_blocks`), main's card-proven fp16 landing. The EA is STANDARD on both. **Difference 2 (iteration rank):** the landing's own `[entries, blocks, rows, w]` (`[8,1,128,128]` decode, `[1,8,128,128]` prefill, `[E,8,128,128]` batch), where torch-spyre iterates the `to` dims `[8,128,128]`: same element set, same bytes. It is kept 4-D because the arrangement authority compares every writer's and reader's declaration of a buffer (`PW` `landing_arg` doc), and the landing's declaration is the fp16 gather's 4-D one — **Q-R4 ruled 11-22:** 4-D kept |
| R3 | fp16 `matmul`s on the widened page; `attn · v_scale` skipped at 1.0 (FACTS A.3–A.5) | main's batched restickify + `batch_matmul` over the landing, unchanged | same |

---

## Step 3 — the shared prerequisites (formats Copy, the clip's bounds as op data, the convert's stick)

### 3a — Fix A (R2), unchanged
Type R2's "Fix A" (`OPSPEC`, `pub trait DataFormat: private::SealedDf + Copy + core::fmt::Debug`).

### 3b — `OPSPEC`: `DLFLOAT16_MAX`, `ClipBounds`, `OpInfo::Clip`

**Edit 3b-1.** Anchor (unique, `OPSPEC:2821-2823`):
```rust
/// Op-specific aux, serialized into `constantInfo_` ONLY when non-empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OpInfo {
```
Replacement:
```rust
/// ⭐ DLFloat16's LARGEST FINITE VALUE — torch-spyre's `DLFLOAT16_MAX`
/// (`torch_spyre/_inductor/constants.py:197`, `(1 + 510/512)·2³²`): the bound `lower_clamp` gives the
/// side of a one-sided clamp the program does not state (`lowering.py:1537-1546`, "DLFloat16, whose
/// infinity encoding is finite"). [`sen169_bits`](crate::emit::sen169_bits) saturates it to that
/// encoder's largest finite word, which exceeds every fp16 value just as this one does.
pub const DLFLOAT16_MAX: f32 = (1.0 + 510.0 / 512.0) * 4_294_967_296.0;

/// ⭐ A `clip`'s TWO NAMED EXTERNAL CONSTANTS — `clipMin` and `clipMax`, as SEN169 words. The clip body
/// is `FMAX(x, clipMin)` in PE (`unary_pipeline.ddl:370-377`) then `FMIN(·, clipMax)` on the SFP
/// (`:472-479`), its bounds read by `ddl.get_external_constant` out of `constantInfo_`. torch-spyre
/// states them per op (`PointwiseOp("clip", [x], {"constants": {"clipMin", "clipMax"}})`,
/// `spyre_kernel.py:236-244`), so they are each op's own data: [`OpInfo::Clip`] carries them, and
/// [`crate::emit::assemble_clip`] is the one door a `clip` descriptor is built by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ClipBounds {
    min: u16,
    max: u16,
}

impl ClipBounds {
    /// `spyre.clamp(x, min, max)` — both bounds stated (torch-spyre's fp8 quantize,
    /// `decompositions.py:3938-3941`).
    pub fn new(min: f32, max: f32) -> ClipBounds {
        ClipBounds {
            min: crate::emit::sen169_bits(min),
            max: crate::emit::sen169_bits(max),
        }
    }

    /// `clamp_min(x, min)` — `lower_clamp(x, min=min, max=None)`: the upper bound is
    /// [`DLFLOAT16_MAX`].
    pub fn at_least(min: f32) -> ClipBounds {
        ClipBounds::new(min, DLFLOAT16_MAX)
    }

    /// `clamp_max(x, max)` — `lower_clamp(x, min=None, max=max)`: the lower bound is
    /// `−`[`DLFLOAT16_MAX`].
    pub fn at_most(max: f32) -> ClipBounds {
        ClipBounds::new(-DLFLOAT16_MAX, max)
    }

    /// The descriptor's two constants, in the DDL's order.
    pub fn named_words(self) -> [(&'static str, u16); 2] {
        [("clipMin", self.min), ("clipMax", self.max)]
    }
}

/// Op-specific aux, serialized into `constantInfo_` ONLY when non-empty.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum OpInfo {
```

**Edit 3b-2.** Anchor (unique, `OPSPEC:2855`):
```rust
    Exx2Scale(u32),
```
Replacement:
```rust
    Exx2Scale(u32),
    /// A `clip`'s two bounds ([`ClipBounds`]), its `clipMin`/`clipMax` external constants.
    Clip(ClipBounds),
```

**Edit 3b-3.** The clip's constants leave the op-func table (they are each op's own now). Anchor
(unique, `OPSPEC:2003-2009`):
```rust
            // ⭐ THE fp8 E4M3 CLAMP BOUNDS, the same ±448 the `fq_chi`/`fq_cl` pair used to read as
            // `[1, stick]` H2D tensor operands. `unary_pipeline.ddl`'s clip body is
            // `FMAX(x, clipMin)` in PE (`:370-377`) then `FMIN(·, clipMax)` on the SFP (`:472-479`) —
            // the min/max pair encoded, in the same order-independent clamp form. The bounds are the
            // E4M3 finite range, so they are quant-format constants, not model constants: they belong
            // to the op, which is why they live in this table.
            OpFunc::Clip => &[("clipMin", -448.0), ("clipMax", 448.0)],
```
Replacement: (empty — delete the seven lines).

### 3c — `EMIT`: one writer of named constants, the clip's arm, the clip door

**Edit 3c-1.** Anchor (unique, inside `emit_sdsc`'s `constant_info`, `EMIT:4174-4188`):
```rust
        serde_json::Value::Object(
            op.op()
                .op_consts()
                .iter()
                .map(|&(cname, v)| {
                    serde_json::json!({
                        "dataFormat_": "SEN169_FP16",
                        "name_": cname,
                        "data_": fold_manager_const(sen169_bits(v) as u32, folds)
                    })
                })
                .enumerate()
                .map(|(i, entry)| (i.to_string(), entry))
                .collect(),
        )
```
Replacement:
```rust
        named_sen169_consts(
            op.op()
                .op_consts()
                .iter()
                .map(|&(cname, v)| (cname, sen169_bits(v))),
            folds,
        )
```

**Edit 3c-2.** Anchor (unique, `EMIT:4195`):
```rust
            OpInfo::None | OpInfo::FusedEpilogue { .. } => serde_json::json!({}),
```
Replacement:
```rust
            // A `clip`'s bounds are its own data, written exactly as a table's constants are.
            OpInfo::Clip(bounds) => named_sen169_consts(bounds.named_words().into_iter(), folds),
            OpInfo::None | OpInfo::FusedEpilogue { .. } => serde_json::json!({}),
```

**Edit 3c-3.** Anchor (unique, the doc line above `fold_manager_const`, `EMIT:7187`):
```rust
/// `constantInfo_.data_` as the NEW (post-2026-09) dxp contract demands it: a FoldManager
```
Replacement:
```rust
/// ⭐ NAMED SEN169 EXTERNAL CONSTANTS — `constantInfo_` as `{"<i>": {dataFormat_, name_, data_}}`, one
/// entry per name, in order, each value ONE SEN169 word: the one writer of an op func's table
/// ([`OpFunc::op_consts`]) and of a `clip`'s bounds ([`crate::superdsc_opspec::ClipBounds`]).
fn named_sen169_consts<'a>(
    entries: impl Iterator<Item = (&'a str, u16)>,
    folds: &SdscFoldSet,
) -> serde_json::Value {
    serde_json::Value::Object(
        entries
            .map(|(cname, word)| {
                serde_json::json!({
                    "dataFormat_": "SEN169_FP16",
                    "name_": cname,
                    "data_": fold_manager_const(u32::from(word), folds)
                })
            })
            .enumerate()
            .map(|(i, entry)| (i.to_string(), entry))
            .collect(),
    )
}

/// `constantInfo_.data_` as the NEW (post-2026-09) dxp contract demands it: a FoldManager
```

**Edit 3c-4.** A `clip` is not reachable through the string door (it has no bounds there). Anchor
(unique, `EMIT:7089`):
```rust
        "clip" => OpFunc::Clip,
```
Replacement: (empty — delete the line). `op_func_from_str("clip")` then panics by name at bake
(`EMIT:7098-7101`), so a bound-less clip cannot be built.

**Edit 3c-5.** The pointwise door states its op func and optional aux. Anchor (unique, `EMIT:2138-2143`):
```rust
pub fn assemble_pointwise_broadcast_divided<T: crate::collective::Tp, O: KindTag>(
    p: PointwiseDivided<'_, O>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp<T>, crate::superdsc_error::SuperDscError> {
    let PointwiseDivided {
```
Replacement:
```rust
pub fn assemble_pointwise_broadcast_divided<T: crate::collective::Tp, O: KindTag>(
    p: PointwiseDivided<'_, O>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp<T>, crate::superdsc_error::SuperDscError> {
    let func = op_func_from_str(p.op_func);
    assemble_pointwise_stated(p, func, None, sym_id_base, layout)
}

/// What one `clip` states: its name, its bounds, its extent, its one input and its output.
pub struct ClipOp<'a, A: KindTag, O: KindTag> {
    pub op_name: &'a str,
    pub bounds: crate::superdsc_opspec::ClipBounds,
    pub rows: crate::sdsc_abstract::RowCount,
    pub cols: crate::sdsc_abstract::BlockCols,
    pub a: In<'a, A>,
    pub o: &'a Stk<O>,
}

/// ⭐ A `clip` — `FMIN(FMAX(x, clipMin), clipMax)` over one input — with its bounds stated
/// ([`crate::superdsc_opspec::ClipBounds`], carried as [`OpInfo::Clip`]): the ONE door a clip
/// descriptor is built by. Otherwise the one-input pointwise op [`pw1`] states.
pub fn assemble_clip<T: crate::collective::Tp, A: KindTag, O: KindTag>(
    c: ClipOp<'_, A, O>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp<T>, crate::superdsc_error::SuperDscError> {
    let ClipOp {
        op_name,
        bounds,
        rows,
        cols,
        a,
        o,
    } = c;
    assemble_pointwise_stated(
        PointwiseDivided {
            op_name,
            op_func: OpFunc::Clip.name(),
            rows,
            cols,
            inputs: &[a.ew()],
            o,
            out_offset: crate::addr::OutCorner::ZERO,
            division: crate::work::EwDivision::Law,
        },
        OpFunc::Clip,
        Some(OpInfo::Clip(bounds)),
        sym_id_base,
        layout,
    )
}

/// The body of every pointwise door above: `func` is the op the descriptor runs (`p.op_func` its
/// spelling), and `info` the aux it states beyond the one its op func implies.
fn assemble_pointwise_stated<T: crate::collective::Tp, O: KindTag>(
    p: PointwiseDivided<'_, O>,
    func: OpFunc,
    info: Option<OpInfo>,
    sym_id_base: &mut i64,
    layout: Option<&BundleLayout>,
) -> Result<EmittedOp<T>, crate::superdsc_error::SuperDscError> {
    let PointwiseDivided {
```

**Edit 3c-6.** Same function body. Anchor (unique, `EMIT:2194-2209`):
```rust
    let mut op = pointwise_broadcast_opspec_df(
        op_func_from_str(op_func),
        rows,
        cols,
        inputs,
        o_name,
        out_offset,
        head_major,
        Df::Fp16,
        layout,
        division,
    )
    .map_err(|e| {
        crate::superdsc_error::SuperDscError(format!("assemble_pointwise_broadcast {op_name}: {e}"))
    })?;
    apply_lx_residency(&mut op, layout);
```
Replacement:
```rust
    let mut op = pointwise_broadcast_opspec_df(
        func,
        rows,
        cols,
        inputs,
        o_name,
        out_offset,
        head_major,
        Df::Fp16,
        layout,
        division,
    )
    .map_err(|e| {
        crate::superdsc_error::SuperDscError(format!("assemble_pointwise_broadcast {op_name}: {e}"))
    })?;
    if let Some(info) = info {
        op = op.with_op_info(info);
    }
    apply_lx_residency(&mut op, layout);
```

### 3d — the W8A8 clip states its bounds (byte-identical: same two names, order and words)

**Edit 3d-1.** `backends/spyre/crates/ktir/ktir-superdsc/src/emit/ktir_matmul_fp8.rs`. Anchor (unique, `:16-19`):
```rust
use super::{
    EmittedOp, In, assemble_convert, assemble_convert_col, assemble_pointwise_broadcast_off,
    bmm_site, emit_sdsc_tiled, op_func_from_str, pw1, pw2, rb, rb_df, rbo,
};
```
Replacement:
```rust
use super::{
    ClipOp, EmittedOp, In, assemble_clip, assemble_convert, assemble_convert_col,
    assemble_pointwise_broadcast_off, bmm_site, emit_sdsc_tiled, op_func_from_str, pw1, pw2, rb,
    rb_df, rbo,
};
```

**Edit 3d-2.** Same file. Anchor (unique, `:457-459`):
```rust
            // value-preserving by construction. The ±448 E4M3 bounds arrive as the external constants
            // `clipMin`/`clipMax` out of the descriptor's `constantInfo_` ([`OpFunc::op_consts`]),
```
Replacement:
```rust
            // value-preserving by construction. The ±448 E4M3 bounds arrive as the external constants
            // `clipMin`/`clipMax` out of the descriptor's `constantInfo_` (this op's own
            // [`crate::superdsc_opspec::ClipBounds`]),
```

**Edit 3d-3.** Same file. Anchor (unique, `:470-480`):
```rust
            let cl_h = rb(&cl, m, cl_reach);
            ops.push(pw1(
                &opn("fq_clip_op"),
                "clip",
                m_rows,
                k_cols,
                In::full(&sc_h),
                &cl_h,
                sym_id_base,
                layout,
            )?);
```
Replacement:
```rust
            let cl_h = rb(&cl, m, cl_reach);
            ops.push(assemble_clip(
                ClipOp {
                    op_name: &opn("fq_clip_op"),
                    bounds: crate::superdsc_opspec::ClipBounds::new(
                        -ktir_core::codec::E4M3_FMAX,
                        ktir_core::codec::E4M3_FMAX,
                    ),
                    rows: m_rows,
                    cols: k_cols,
                    a: In::full(&sc_h),
                    o: &cl_h,
                },
                sym_id_base,
                layout,
            )?);
```
(`&opn("fq_clip_op")` is a temporary of the `ops.push(…)` statement, alive until its end — the
pattern `pw1(&opn(…), …)` already uses.)

### 3e — `EMIT`: the convert's stick law, stated once

**Edit 3e-1.** Anchor (unique, `EMIT:4950-4954`):
```rust
    let stick_df = if out_df.elems_per_stick() >= in_df.elems_per_stick() {
        out_df
    } else {
        in_df
    };
```
Replacement:
```rust
    let stick_df = convert_stick_df(op);
```

**Edit 3e-2.** Anchor (unique, `EMIT:4921-4922`):
```rust
/// A dtype-convert's opspec: both operands keep their stick on `out`, addressed RowBlocked
/// at rows>1 — the rank-3 `[mb,out,1]` staging every residual-style consumer reads.
```
Replacement:
```rust
/// ⭐ THE STICK A CONVERT'S WORK DIVISION KEEPS WHOLE — the WIDER of its two formats' sticks
/// ([`convert_dtypes`]): a convert touches both dataspaces per core, so a core's slab must be a whole
/// stick of each. Read by every convert's opspec ([`convert_opspec`], the page widen's
/// `widen_landing_opspec`).
pub(crate) fn convert_stick_df(op: OpFunc) -> Df {
    let (in_df, out_df) = convert_dtypes(op);
    if out_df.elems_per_stick() >= in_df.elems_per_stick() {
        out_df
    } else {
        in_df
    }
}

/// A dtype-convert's opspec: both operands keep their stick on `out`, addressed RowBlocked
/// at rows>1 — the rank-3 `[mb,out,1]` staging every residual-style consumer reads.
```

### Step 3 gate
`cd backends/spyre && cargo test -p ktir-superdsc` and `cargo test -p scratchy-target-spyre` (pod);
then the kv/fp8 spyre-hw build. Expected: the same first refusal as Part 1 (the write's fp8 store),
and every W8A8 program bakes. A W8A8 refusal naming `clip` / `constantInfo_` is a Step 3 defect.

### Step 3 purity

| edit | P3 once | P4 | P5 | P8 | P10 | P11 | P12 |
|---|---|---|---|---|---|---|---|
| 3a | `Copy + Debug` stated by the trait, never per site | — | — | — | — | — | type |
| 3b/3c | a clip's bounds live in ONE place (its `OpInfo`); the table entry that stated them for every clip is deleted; one constants writer for table and clip | a bound-less clip is unconstructible (the string door panics by name at bake) | `ClipBounds` (private words, checked constructors), no raw pair | opcode lowering only (one door + one `OpInfo` row) | the old table closure moved into `named_sen169_consts`, not copied | present tense | type carries it |
| 3d | the E4M3 bound read from `E4M3_FMAX` (Part 1, 1.3a), not restated as `448.0` | — | — | — | — | comment rewritten to the new home | — |
| 3e | the wider-stick law stated once, read by both convert opspecs | — | — | — | moved, not copied | — | — |

---

## Step 4 — the write: the kernel's chains, the narrowing-store family fold, one scatter run

What the `kv/fp8` write program states after 4b (read off the kernel and the ladder; R2 3's own
reading, which the pod confirmed up to the emitter): for each kv head `h` of K, then of V (and below
the scatter floor, for each token `t`), `ktdp.load` of the f16 `key[:, h]` window `[T,128]` at
`(0, h·128)` (`[1,128]` at `(t, h·128)` below the floor) → `arith.maxnumf` against a splatted `−FMAX`
→ `arith.minnumf` against a splatted `FMAX` → `arith.truncf` to f8E4M3FN → `ktdp.store` through an
indirect tile into the fp8 pool view at index entry `h·T (+ t)`.

The fold (lead ruling Q-W1 = A): those chains are ONE family per `(source, pool, index)`; their
windows tile the source exactly, so the family lowers as torch-spyre lowers the reference's one
whole-tensor `quantize_kv` — `clip`, `clip`, `qfp8ch` over `[T, 1024]` (table rows W2–W4) — then ONE
scatter run of the narrowed rows (W5). It follows the fold rules of `~/.claude/CLAUDE.md`: its
target device ops are the chain exactly (each `maxnumf`/`minnumf` against a constant IS a one-sided
`clip`, `lower_clamp`'s own lowering; the `truncf` to e4m3 IS `qfp8ch`, as the reference's comment
states it, "`spyre.qfp8ch` on Spyre"); it is stated over op kinds and operand identity; every interior
value has one reader; ONE shape predicate (`tiles_the_view`); a family that does not close is left
unclaimed (its `truncf` is then `convert_kind_of`'s by-name refusal, `WF:2008-2019`); fixtures for
the match, each near-miss and the neighbour (4g).

### 4a — `SPLICE`: the `FMAX` constexpr (R2 1.3e's second half)

**Edit 4a-1.** Anchor (Part 1's 1.3c line, unique):
```rust
use ktir_superdsc::superdsc_opspec::DataFormat;
```
Replacement:
```rust
use ktir_superdsc::superdsc_opspec::{DataFormat, FloatFormat};
```

**Edit 4a-2.** Anchor (Part 1's 1.3e text, unique — `lower_attn`'s loops name `k_pages`/`v_pages`):
```rust
    for p in ["k_rows", "v_rows"] {
        signature.insert(p.to_string(), kv_pool_ptr());
    }
```
Replacement:
```rust
    for p in ["k_rows", "v_rows"] {
        signature.insert(p.to_string(), kv_pool_ptr());
    }
    // ⭐ THE POOL FORMAT'S SATURATION BOUND — `torch.finfo(k_rows.dtype).max`, the kernel's `FMAX`.
    // The kernel reads it only where the pool's element type is not the key's.
    signature.insert("FMAX".to_string(), ArgSpec::Constexpr);
    constexprs.insert(
        "FMAX".to_string(),
        Val::Float(f64::from(<KvFormat as FloatFormat>::FMAX)),
    );
```

### 4b — the kernel: R2 1.4a, with its docstring stating the fold

**Edit 4b-1** — `backends/spyre/kernels/reshape_and_cache_head_major.py`. Anchor (main, unique —
R2 1.4a's first anchor):
```python
Where Triton spells a torch op differently, the spelling is Triton's and the op is theirs: an index
of fewer than 8 rows is below Triton's scatter floor, so under 8 tokens each row is stored on its
own, at the row the index names. Their `key = key * 1.0` (materializing a strided view of the fused
QKV projection, whose per-head slice torch-spyre stores wrong) has no counterpart: `key` here is its
own tensor, not a view.
"""
```
Replacement:
```python
Where Triton spells a torch op differently, the spelling is Triton's and the op is theirs: an index
of fewer than 8 rows is below Triton's scatter floor, so under 8 tokens each row is stored on its
own, at the row the index names. Their `key = key * 1.0` (materializing a strided view of the fused
QKV projection, whose per-head slice torch-spyre stores wrong) has no counterpart: `key` here is its
own tensor, not a view.

A float8 cache takes their `quantize_kv` (spyre-inference 4a7d21d, `ops/reshape_and_cache.py:20-29`):
`/ scale` (the checkpoint's `k_scale`, absent, so no divide — their `if scale != 1.0`), the saturation
to `±FMAX` (`torch.finfo(k_rows.dtype).max`), and the cast to the cache's element type
(`spyre.qfp8ch`). Their condition, `k_scale is not None`, holds exactly for a float8 cache
(`_kernel_kv_scales`, `spyre_attn.py:1913-1920`); here it is the cache's own element type. Theirs is
ONE elementwise pass over the whole `key`, then the per-head stores. Triton slices no register value,
so each head's window is saturated and cast where it is loaded; the emitter's narrowing-store family
lowers those windows, which tile `key` exactly, as their one pass over the whole tensor (`clip`,
`clip`, `qfp8ch`) and then the stores. At a 16-bit cache the branch is not taken and nothing is
emitted for it.
"""
```
Then type R2 1.4a's other three edits (the `FMAX: tl.constexpr` parameter, the `T >= 8` per-head
chains, the `T < 8` per-row chains) unchanged.

### 4c — `LK`: the stored rows' source read once; the run law stated once; one run body

**Edit 4c-1** — `scatters_of`'s source reading moves into `param_rows`. Replace `LK:976-1048` (73
lines, md5 `e8effe32f76126a232aa8f26b5847701`; first line
`        let first_entry = indirect_tile_anchor(k, tile, &consts)?;`, last line `        });` — the
close of `out.push(Scatter { … })`) with:
```rust
        let first_entry = indirect_tile_anchor(k, tile, &consts)?;
        let src = param_rows(k, &consts, load)?;
        out.push(Scatter {
            index_tid,
            value_tid,
            src_tid: src.src_tid,
            entries,
            first_entry,
            sticks: src.width / Fp16::ELEMS_PER_STICK,
            src_row: src.src_row(),
            value,
        });
```

**Edit 4c-2.** Anchor (unique, `LK:1053`):
```rust
/// One store's share of a scatter run: where its entries start, how many rows it writes, and the
```
Replacement:
```rust
/// ⭐ THE ROWS A PLAIN LOAD READS OF A PARAMETER — the source side of a store through an indirect
/// tile: which parameter, the window's corner and extent, and the view it reads. Read once, here,
/// for every reader of a stored row's source ([`scatters_of`], [`narrow_families_of`]).
pub(crate) struct ParamRows {
    /// The parameter the rows are loaded from.
    pub src_tid: u32,
    /// The window's row corner.
    pub row: u32,
    /// The window's column corner, a multiple of its width.
    pub col: u32,
    /// The window's rows.
    pub rows: u32,
    /// The window's width, a whole number of fp16 sticks.
    pub width: u32,
    /// The view's rows.
    pub view_rows: u32,
    /// The view's columns.
    pub view_cols: u32,
}

impl ParamRows {
    /// The window's first row in the source's stick-major row grouping: the source is stick-major
    /// `[view_rows, view_cols]` and its stick columns group into rows `width` wide, so row `r` of the
    /// group at column `c` is `(c / width) · view_rows + r` — spyre-inference's `key[:, h]` row `t` is
    /// `h · T + t`, which is where their head-major `kv_write_index` puts its entry. At one stick a
    /// row it is the source's own stick.
    pub fn src_row(&self) -> u32 {
        (self.col / self.width) * self.view_rows + self.row
    }
}

/// [`ParamRows`] of `load`, a `ktdp.load` over a plain access tile of a parameter's view.
fn param_rows(
    k: &KtirNode,
    consts: &std::collections::HashMap<Ssa, i64>,
    load: &Operation<'_>,
) -> Result<ParamRows, Error> {
    let f = &k.func;
    let src_tile_v = load.operands.first().copied();
    let src_tile = f
        .operations
        .iter()
        .find(|o| o.result == src_tile_v && o.op_type == OpKind::KtdpConstructAccessTile)
        .ok_or_else(|| Error {
            message: format!(
                "{}: the scatter's stored rows are not loaded through a plain access tile",
                f.name
            ),
        })?;
    let src_view_v = src_tile.operands.first().copied();
    let src_view = f
        .operations
        .iter()
        .find(|o| o.result == src_view_v && o.op_type == OpKind::KtdpConstructMemoryView)
        .ok_or_else(|| Error {
            message: format!("{}: the scatter's source tile reads no memory view", f.name),
        })?;
    let ptr = src_view.operands.first().copied();
    let i = f
        .arguments
        .iter()
        .position(|(a, _)| Some(*a) == ptr)
        .ok_or_else(|| Error {
            message: format!(
                "{}: the scatter's source view does not reinterpret a parameter",
                f.name
            ),
        })?;
    let src_tid = k.bindings.get(i).map(|b| b.get()).ok_or_else(|| Error {
        message: format!("{}: parameter {i} has no bound buffer", f.name),
    })?;
    let corner = |slot: usize| {
        src_tile
            .operands
            .get(slot)
            .and_then(|r| consts.get(r).copied())
            .and_then(|r| u32::try_from(r).ok())
    };
    let (Some(row), Some(col)) = (corner(1), corner(2)) else {
        return err(format!(
            "{}: the scatter's source tile states no constant corner, so which source rows it \
             writes is unknown",
            f.name
        ));
    };
    let stick = Fp16::ELEMS_PER_STICK;
    let view_rows = src_view
        .result_type
        .and_then(|t| t.dims())
        .and_then(|d| d.first().copied())
        .unwrap_or(0) as u32;
    let view_cols = src_view
        .result_type
        .and_then(|t| t.dims())
        .and_then(|d| d.get(1).copied())
        .unwrap_or(0) as u32;
    let (rows, width) = shape_2d(src_tile).unwrap_or((0, 0));
    if width == 0 || !width.is_multiple_of(stick) || !col.is_multiple_of(width) {
        return err(format!(
            "{}: the scatter's source rows are {width} wide at column {col} — a scattered row is a \
             whole number of {stick}-element sticks of the source, its own head, at a column that \
             is a multiple of its width",
            f.name
        ));
    }
    Ok(ParamRows {
        src_tid,
        row,
        col,
        rows,
        width,
        view_rows,
        view_cols,
    })
}

/// One store's share of a scatter run: where its entries start, how many rows it writes, and the
```

**Edit 4c-3** — `load_scatters`' run law and run body, each stated once. Replace `LK:1134-1275` (142
lines, md5 `fba56cfe7038ccce59e4b67e530409f8`; first line `    let mut ops = Vec::new();`, last line
the `}` closing `load_scatters`) with:
```rust
    let mut ops = Vec::new();
    for ScatterRun {
        key: (src_tid, dst_tid, index_tid),
        sticks,
        value,
        mut pieces,
    } in runs
    {
        let RunExtent { e0, s0, total } = run_extent(name, &mut pieces)?;
        let (src, dst, index) = (region(src_tid)?, region(dst_tid)?, region(index_tid)?);
        let width = StickExtent::<Fp16>::new(sticks * stick).map_err(|message| Error {
            message: format!("{name}: {message}"),
        })?;
        ops.extend(scatter_run::<Fp16, T>(
            name,
            k,
            layout,
            RunCopy {
                src: &src.name(),
                src_rows: src.v_rows,
                groups: src.v_cols / width.elems(),
                width,
                s0,
                e0,
                total,
                dst,
                index,
                key: value,
            },
            sym_id_base,
        )?);
    }
    Ok(ops)
}

/// ⭐ WHERE ONE SCATTER RUN SITS — its first entry `e0`, the source row `s0` it reads first, and its
/// row count. The card's scatter copies source row `s0 + j` through entry `e0 + j` and nothing else,
/// so every store of the run continues it in order: its entry the run's end so far, its source row as
/// far past `s0` as its entry is past `e0`. A store that does not is refused by name. The one
/// statement of the run law, taken by a parameter's rows ([`load_scatters`]) and a narrowing family's
/// ([`NarrowFamily::extent`]).
pub(crate) struct RunExtent {
    pub e0: u32,
    pub s0: u32,
    pub total: u32,
}

fn run_extent(name: &str, pieces: &mut [RunPiece]) -> Result<RunExtent, Error> {
    pieces.sort_unstable();
    let (e0, s0) = pieces
        .first()
        .map(|p| (p.entry, p.src_row))
        .unwrap_or_default();
    let mut next = e0;
    for &RunPiece {
        entry,
        rows,
        src_row,
    } in pieces.iter()
    {
        if entry != next || src_row - s0 != entry - e0 {
            return err(format!(
                "{name}: a store writes source rows {src_row}.. through index entries \
                 {entry}.. where the run so far ends at {next} — the card's scatter copies \
                 source row `{s0} + j` through entry `{e0} + j`, so the program's stores must \
                 tile the run in order"
            ));
        }
        next += rows;
    }
    Ok(RunExtent {
        e0,
        s0,
        total: next - e0,
    })
}
```
followed by R2 3c's `RunCopy<'a, DF>` struct and `scatter_run<DF, T>` function, typed verbatim from
R2 (its text from `/// ⭐ WHAT ONE SCATTER RUN MOVES` through `scatter_run`'s closing `}`). At
`DF = Fp16` that is main's tail op for op and name for name (R2 3c's note).

**Edit 4c-4** — the narrowing-store family. Anchor (unique, `LK:1277`):
```rust
/// ⛔⛔⛔ THE GUARD FOR THE FACT THIS FILE USED TO DROP: an operand this emitter addresses at the
```
Replacement:
```rust
/// ⭐⭐⭐ A NARROWING-STORE FAMILY — spyre-inference's `quantize_kv` then `index_copy_`
/// (`ops/reshape_and_cache.py:20-29, 47-52`) as the KV write states it, one window at a time: each
/// member loads a window of ONE f16 parameter, saturates it (`arith.maxnumf` against a splatted `lo`,
/// then `arith.minnumf` against a splatted `hi` — `torch.clamp`'s `clamp_min` and `clamp_max`),
/// narrows it to f8E4M3FN (`arith.truncf`) and stores it through an indirect tile. When the members'
/// windows TILE the parameter's view exactly ([`tiles_the_view`]), the family is the reference's ONE
/// elementwise pass over the whole tensor followed by its stores, and it lowers as torch-spyre lowers
/// that: `clip` (`clamp_min`), `clip` (`clamp_max`) and `qfp8ch` over the whole view
/// (`lower_clamp`, `lowering.py:1505-1579`; `spyre.qfp8ch` over `x.get_size()`,
/// `lowering.py:2969-2992`), then ONE scatter run of the narrowed rows ([`scatter_run`]).
///
/// ⛔ THE FOLD'S LAWS: stated over op kinds and operand identity only; every interior value — the
/// load, both saturations, the narrowing — has exactly one reader; ONE shape predicate,
/// [`tiles_the_view`]; and a family that does not close — a member whose chain differs, members that
/// disagree on the bounds or the row width, windows that leave a gap or overlap — is LEFT
/// UNCLAIMED, its ops lowered one by one, where the f16→f8E4M3FN `arith.truncf` is the walk's own
/// by-name refusal (`convert_kind_of`).
pub(crate) struct NarrowFamily {
    /// The f16 parameter every member loads.
    pub src_tid: u32,
    /// The parameter every member stores through its index.
    pub dst_tid: u32,
    /// The parameter holding the index.
    pub index_tid: u32,
    /// The `clamp_min` bound every member states.
    pub lo: f32,
    /// The `clamp_max` bound every member states.
    pub hi: f32,
    /// One stored row's width, whole float8 sticks.
    pub width: StickExtent<crate::superdsc_opspec::Fp8>,
    /// The source view's `[rows, cols]`.
    pub view: (u32, u32),
    /// The members, in program order.
    members: Vec<NarrowMember>,
}

/// One member of a [`NarrowFamily`]: its three chain values, its share of the run, its window.
pub(crate) struct NarrowMember {
    /// The `arith.maxnumf` result.
    pub clamp_min: Ssa,
    /// The `arith.minnumf` result.
    pub clamp_max: Ssa,
    /// The `arith.truncf` result.
    pub narrow: Ssa,
    piece: RunPiece,
    /// `(row, col, rows, cols)` of the source window.
    window: (u32, u32, u32, u32),
}

impl NarrowFamily {
    /// The first member in program order — the values the family's whole-view buffers are minted
    /// by. A family is created with its first member.
    pub fn leader(&self) -> &NarrowMember {
        &self.members[0]
    }

    /// Every value the family's whole-view chain stands for: the walk lowers none of them.
    pub fn claimed(&self) -> impl Iterator<Item = Ssa> + '_ {
        self.members
            .iter()
            .flat_map(|m| [m.clamp_min, m.clamp_max, m.narrow])
    }

    /// The family's one scatter run ([`run_extent`]).
    pub fn extent(&self, name: &str) -> Result<RunExtent, Error> {
        let mut pieces: Vec<RunPiece> = self.members.iter().map(|m| m.piece).collect();
        run_extent(name, &mut pieces)
    }
}

/// ⭐ THE FAMILY'S ONE SHAPE PREDICATE — its members' source windows tile the source view exactly:
/// every window inside the view, no two overlapping, their areas summing to the view's.
fn tiles_the_view(fam: &NarrowFamily) -> bool {
    let (vr, vc) = fam.view;
    let w = &fam.members;
    let inside = w.iter().all(|m| {
        let (r, c, rows, cols) = m.window;
        r + rows <= vr && c + cols <= vc
    });
    let disjoint = w.iter().enumerate().all(|(i, a)| {
        w[i + 1..].iter().all(|b| {
            let (ar, ac, arn, acn) = a.window;
            let (br, bc, brn, bcn) = b.window;
            ar + arn <= br || br + brn <= ar || ac + acn <= bc || bc + bcn <= ac
        })
    });
    let area: u64 = w
        .iter()
        .map(|m| u64::from(m.window.2) * u64::from(m.window.3))
        .sum();
    inside && disjoint && area == u64::from(vr) * u64::from(vc)
}

/// [`NarrowFamily`] for every `(source, destination, index)` triple whose stores close the pattern
/// and tile their source.
pub(crate) fn narrow_families_of(k: &KtirNode) -> Result<Vec<NarrowFamily>, Error> {
    use crate::superdsc_opspec::Fp8;
    let f = &k.func;
    let consts = index_constants(f);
    let deep = f.ops_deep();
    let def_of = |v: Ssa| f.operations.iter().find(|o| o.result == Some(v));
    let one_reader = |v: Ssa| deep.iter().filter(|o| o.operands.contains(&v)).count() == 1;
    // Each triple in first-store order; `None` once its members disagree (left unclaimed).
    let mut groups: Vec<((u32, u32, u32), Option<NarrowFamily>)> = Vec::new();
    for store in f
        .operations
        .iter()
        .filter(|o| o.op_type == OpKind::KtdpStore)
    {
        let (Some(&narrow), Some(&tv)) = (store.operands.first(), store.operands.get(1)) else {
            continue;
        };
        let Some(tile) = f
            .operations
            .iter()
            .find(|o| o.result == Some(tv) && o.op_type == OpKind::KtdpConstructIndirectAccessTile)
        else {
            continue;
        };
        // The chain, link by link, from the store up: any other link is not this pattern.
        let Some(trunc) = def_of(narrow).filter(|o| {
            o.op_type == OpKind::ArithTruncf
                && o.result_type.and_then(|t| t.elem()) == Some(ktir_core::dtypes::DType::Fp8E4m3)
        }) else {
            continue;
        };
        let Some(&clamp_max) = trunc.operands.first() else {
            continue;
        };
        let Some(max_op) = def_of(clamp_max).filter(|o| o.op_type == OpKind::ArithMinnumf) else {
            continue;
        };
        let Some((hi, clamp_min)) = super::whole_function::splat_scale_of(f, max_op) else {
            continue;
        };
        let Some(min_op) = def_of(clamp_min).filter(|o| o.op_type == OpKind::ArithMaxnumf) else {
            continue;
        };
        let Some((lo, x)) = super::whole_function::splat_scale_of(f, min_op) else {
            continue;
        };
        let Some(load) = def_of(x).filter(|o| o.op_type == OpKind::KtdpLoad) else {
            continue;
        };
        if ![x, clamp_min, clamp_max, narrow]
            .into_iter()
            .all(one_reader)
        {
            continue;
        }
        let src = param_rows(k, &consts, load)?;
        let dst_tid = indirect_tile_param(k, tile, 0, "written")?;
        let index_tid = indirect_tile_param(k, tile, 1, "index")?;
        let (rows, _) = shape_2d(tile).ok_or_else(|| Error {
            message: format!(
                "{}: a narrowing store's indirect tile states no 2-D `shape`, so how many rows it \
                 writes is unknown",
                f.name
            ),
        })?;
        let entry = indirect_tile_anchor(k, tile, &consts)?;
        let member = NarrowMember {
            clamp_min,
            clamp_max,
            narrow,
            piece: RunPiece {
                entry,
                rows,
                src_row: src.src_row(),
            },
            window: (src.row, src.col, src.rows, src.width),
        };
        let key = (src.src_tid, dst_tid, index_tid);
        match groups.iter_mut().find(|(g, _)| *g == key) {
            None => groups.push((
                key,
                StickExtent::<Fp8>::new(src.width)
                    .ok()
                    .map(|width| NarrowFamily {
                        src_tid: src.src_tid,
                        dst_tid,
                        index_tid,
                        lo,
                        hi,
                        width,
                        view: (src.view_rows, src.view_cols),
                        members: vec![member],
                    }),
            )),
            Some((_, slot)) => {
                let joined = match slot.take() {
                    Some(mut fam)
                        if fam.lo == lo && fam.hi == hi && fam.width.elems() == src.width =>
                    {
                        fam.members.push(member);
                        Some(fam)
                    }
                    _ => None,
                };
                *slot = joined;
            }
        }
    }
    Ok(groups
        .into_iter()
        .filter_map(|(_, fam)| fam)
        .filter(tiles_the_view)
        .collect())
}

/// ⛔⛔⛔ THE GUARD FOR THE FACT THIS FILE USED TO DROP: an operand this emitter addresses at the
```
(`index_constants`, `indirect_tile_param`, `indirect_tile_anchor`, `shape_2d` are this file's,
`LK:442-566, 1666, 1813`; `Iterator::filter` hands `tiles_the_view` the `&NarrowFamily` it takes.)

### 4d — `WF`: the families before the walk, their members skipped, their runs after it

**Edit 4d-1.** Anchor (unique, `WF:4797-4799`):
```rust
    let mut batch_blocks: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();

    for (oi, op) in f.operations.iter().enumerate() {
```
Replacement:
```rust
    let mut batch_blocks: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
    // ⭐⭐⭐ THE NARROWING-STORE FAMILIES THE PROGRAM STATES
    // ([`super::lower_ktir_to_superdsc::narrow_families_of`]), read ONCE: each family's whole-view
    // chain — `clip`, `clip`, `qfp8ch` ([`narrow_chain`]) — is emitted here, into three buffers minted
    // for its leader's values; the members' own ops are skipped below, and its one scatter run follows
    // the walk.
    let narrow = super::lower_ktir_to_superdsc::narrow_families_of(k)?;
    let narrow_skip: std::collections::HashSet<Ssa> =
        narrow.iter().flat_map(|fam| fam.claimed()).collect();
    let mut narrowed: Vec<u32> = Vec::with_capacity(narrow.len());
    if !narrow.is_empty() {
        let Some(layout) = layout else {
            return err(format!(
                "{}: a narrowing-store family's whole-view buffers need a `BundleLayout` to hold them",
                f.name
            ));
        };
        let params = regions(k)?;
        for fam in &narrow {
            let x = params
                .iter()
                .find(|r| r.tid == fam.src_tid)
                .ok_or_else(|| Error {
                    message: format!(
                        "{}: t{} is not a parameter of the program",
                        f.name, fam.src_tid
                    ),
                })?;
            let (rows, cols) = fam.view;
            let lead = fam.leader();
            let lo_t = mint_inter(k, layout, floor, lead.clamp_min);
            layout.synth(crate::place::SynthId::Act(lo_t), &[rows, cols]);
            let hi_t = mint_inter(k, layout, floor, lead.clamp_max);
            layout.synth(crate::place::SynthId::Act(hi_t), &[rows, cols]);
            let q_t = mint_inter(k, layout, floor, lead.narrow);
            layout.synth_df(
                crate::place::SynthId::Act(q_t),
                &[rows, cols],
                <crate::superdsc_opspec::Fp8 as crate::superdsc_opspec::DataFormat>::DF,
            );
            out.extend(narrow_chain::<T>(
                f.name,
                fam,
                x,
                [lo_t, hi_t, q_t],
                sym_id_base,
                layout,
            )?);
            narrowed.push(q_t);
        }
    }

    for (oi, op) in f.operations.iter().enumerate() {
```

**Edit 4d-2.** Anchor (unique, `WF:4920-4922`):
```rust
                // A row-folded family's non-leader block: the leader's one whole-view descriptor
                // already computed its rows (see [`row_tiled_family`]).
                || row_fold_skip.contains(&r)
```
Replacement:
```rust
                // A row-folded family's non-leader block: the leader's one whole-view descriptor
                // already computed its rows (see [`row_tiled_family`]).
                || row_fold_skip.contains(&r)
                // A narrowing-store family's member: the family's whole-view chain computed it.
                || narrow_skip.contains(&r)
```

**Edit 4d-3.** Anchor (unique, `WF` — the doc line of `wide_blocks`):
```rust
/// Whether `tid` holds batch blocks wider than one stick — bytes no flat frame names
```
Replacement:
```rust
/// ⭐ ONE NARROWING FAMILY'S WHOLE-VIEW CHAIN — torch-spyre's three device ops for spyre-inference's
/// `quantize_kv` over the whole tensor: `clip` (`clamp_min`, [`crate::superdsc_opspec::ClipBounds::at_least`]),
/// `clip` (`clamp_max`, [`crate::superdsc_opspec::ClipBounds::at_most`]) and `qfp8ch` — the W8A8
/// activation quantizer's own convert door ([`super::assemble_convert`], `fq_afp8_op`) — over the
/// source's stick-major `[rows, cols]`, into the family's three minted buffers `[lo, hi, q]` (`q` at
/// float8).
fn narrow_chain<T: crate::collective::Tp>(
    name: &str,
    fam: &super::lower_ktir_to_superdsc::NarrowFamily,
    x: &Region,
    [lo_t, hi_t, q_t]: [u32; 3],
    sym_id_base: &mut i64,
    layout: &BundleLayout,
) -> Result<Vec<super::EmittedOp<T>>, Error> {
    let (rows, cols) = fam.view;
    let m_rows = crate::sdsc_abstract::RowCount::of_token_rows(rows);
    let k_cols = crate::sdsc_abstract::BlockCols::of_feature_cols(cols);
    let x_h = super::rb(&x.name(), rows, cols);
    let lo_h = super::rb(&crate::place::act_name(lo_t), rows, cols);
    let hi_h = super::rb(&crate::place::act_name(hi_t), rows, cols);
    let q_h = super::rb_df(&crate::place::act_name(q_t), rows, cols, Df::Fp8);
    let (lo_name, hi_name, q_name) = (
        format!("clipmin_o{lo_t}"),
        format!("clipmax_o{hi_t}"),
        format!("qfp8ch_o{q_t}"),
    );
    let lift = |e: crate::superdsc_error::SuperDscError| Error {
        message: format!("{name}: {}", e.0),
    };
    Ok(vec![
        super::assemble_clip(
            super::ClipOp {
                op_name: &lo_name,
                bounds: crate::superdsc_opspec::ClipBounds::at_least(fam.lo),
                rows: m_rows,
                cols: k_cols,
                a: super::In::full(&x_h),
                o: &lo_h,
            },
            sym_id_base,
            Some(layout),
        )
        .map_err(lift)?,
        super::assemble_clip(
            super::ClipOp {
                op_name: &hi_name,
                bounds: crate::superdsc_opspec::ClipBounds::at_most(fam.hi),
                rows: m_rows,
                cols: k_cols,
                a: super::In::full(&lo_h),
                o: &hi_h,
            },
            sym_id_base,
            Some(layout),
        )
        .map_err(lift)?,
        super::assemble_convert(
            &q_name,
            "qfp8ch",
            rows,
            cols,
            &hi_h,
            &q_h,
            sym_id_base,
            Some(layout),
        )
        .map_err(lift)?,
    ])
}

/// Whether `tid` holds batch blocks wider than one stick — bytes no flat frame names
```

**Edit 4d-4.** The runs after the walk, and the read guard over both kinds. Replace `WF:7821-7842`
(22 lines, md5 `5fac7c9c31b531c10b2565c61898eced`; first line
`    let scattered = super::lower_ktir_to_superdsc::load_scatters(f.name, k, sym_id_base, layout)?;`,
last line `    out.extend(scattered);`) with:
```rust
    let scattered = super::lower_ktir_to_superdsc::load_scatters(f.name, k, sym_id_base, layout)?;
    // ⭐ EACH NARROWING FAMILY'S ONE SCATTER RUN — its narrowed rows (the whole-view `qfp8ch`'s
    // stick-major float8 `[rows, cols]`, whose row `(c / width) · rows + r` is window `(r, c)`'s, the
    // source-row law a parameter's rows take) onto its destination through its index, the one card
    // form every run takes ([`super::lower_ktir_to_superdsc::scatter_run`]).
    let mut narrow_runs = Vec::new();
    if !narrow.is_empty() {
        let params = regions(k)?;
        let param = |tid: u32| {
            params.iter().find(|x| x.tid == tid).ok_or_else(|| Error {
                message: format!("{}: t{tid} is not a parameter of the program", f.name),
            })
        };
        for (fam, &q_t) in narrow.iter().zip(&narrowed) {
            let run = fam.extent(f.name)?;
            narrow_runs.extend(super::lower_ktir_to_superdsc::scatter_run::<
                crate::superdsc_opspec::Fp8,
                T,
            >(
                f.name,
                k,
                layout,
                super::lower_ktir_to_superdsc::RunCopy {
                    src: &crate::place::act_name(q_t),
                    src_rows: fam.view.0,
                    groups: fam.view.1 / fam.width.elems(),
                    width: fam.width,
                    s0: run.s0,
                    e0: run.e0,
                    total: run.total,
                    dst: param(fam.dst_tid)?,
                    index: param(fam.index_tid)?,
                    key: fam.leader().narrow,
                },
                sym_id_base,
            )?);
        }
    }
    for value_tid in super::lower_ktir_to_superdsc::scatters_of(k)?
        .iter()
        .map(|sc| sc.value_tid)
        .chain(narrow.iter().map(|fam| fam.dst_tid))
    {
        let read = gathers.iter().any(|g| g.value_tid == value_tid)
            || f.ops_deep().iter().any(|o| {
                o.op_type == OpKind::KtdpLoad
                    && o.result.is_some_and(|v| {
                        region_for_operand(k, v)
                            .ok()
                            .flatten()
                            .is_some_and(|r| r.tid == value_tid)
                    })
            });
        if read {
            return err(format!(
                "{}: the program scatters rows into t{value_tid} and also reads it — the scatter \
                 runs after the compute, so a read of its destination would see the rows before \
                 they land",
                f.name
            ));
        }
    }
    let scatters = !scattered.is_empty() || !narrow_runs.is_empty();
    out.extend(narrow_runs);
    out.extend(scattered);
```

**Edit 4d-5 — Fix 2a (R2), its comment naming this plan's arms.** `WF` (`staggered_sources`). Anchor
(unique, `WF:4239-4244` — the walk's own loop reads `oi, op`):
```rust
    for op in f.operations.iter() {
        if is_plumbing(op.op_type) {
            continue;
        }
        if op.result.is_some_and(|r| {
            silus.iter().any(|c| c.consumed.contains(&r))
```
Replacement: R2 Fix 2a's replacement text verbatim, EXCEPT its comment's first five lines
```rust
        // ⭐ A FLOAT8 VALUE IS NEVER A TABLE VALUE — an op whose result or operand states
        // f8E4M3FN is the walk's narrowing-store arm (`truncf → indirect store`) or its page arm
        // (`gather → reshape → extf`), each holding the float8 bytes as a typed local of its own
        // and never as a `Lowering`. The propagation has no arrangement to decide for it (the
        // 16-bit value the widen produces is standard, the convert's own table row); an fp8 op
```
become
```rust
        // ⭐ A FLOAT8 VALUE IS NEVER A TABLE VALUE — an op whose result or operand states
        // f8E4M3FN is a narrowing-store family's `arith.truncf` (lowered by the family's whole-view
        // `qfp8ch`, `narrow_families_of`) or a float8 page's `gather → reshape → extf` (the page
        // arm's `fp8todl16`), each holding the float8 bytes in a buffer of its own and never as a
        // `Lowering`. The propagation has no arrangement to decide for it (the 16-bit value the
        // widen produces is standard, the convert's own table row); an fp8 op
```
(the rest of R2's comment and code unchanged).

### 4e — `SPLICE` tests: the write lowered at either format; the fold's fixtures

**Edit 4e-1.** Replace `SPLICE:6735-6788` (54 lines, md5 `45f7ab3859c3c3782964cdc38d364432`; first
line ``    /// `reshape_and_cache_head_major` writing `t` tokens of `nkvh` kv heads at head dim `hd` into a``,
last line the `    }` closing `kv_write_ops`) with:
```rust
    /// `src` — `reshape_and_cache_head_major` or a variant of its source — writing `t` tokens of
    /// `nkvh` kv heads at head dim `hd` into a four-page-per-head pool held at format `DF`, through the
    /// card's lowering against its one-kernel layout.
    fn kv_write_lowered<DF: ktir_superdsc::superdsc_opspec::FloatFormat>(
        src: &str,
        t: i128,
        nkvh: i128,
        hd: i128,
    ) -> Result<Vec<ktir_superdsc::emit::EmittedOp<ktir_superdsc::collective::TpWorld<1>>>, String>
    {
        let TritonKernelRow::Spliced { file, entry, .. } = kv_write_row() else {
            panic!("the KV write row is spliced");
        };
        let mut signature = HashMap::new();
        let mut constexprs = HashMap::new();
        for p in ["key", "value"] {
            signature.insert(p.to_string(), ArgSpec::parse("*fp16").expect("spec"));
        }
        for p in ["k_rows", "v_rows"] {
            signature.insert(
                p.to_string(),
                ArgSpec::Ptr(triton_frontend::ttir::Type::Float(stated_float(DF::DF))),
            );
        }
        signature.insert(
            "row_index".to_string(),
            ArgSpec::parse("*i32").expect("spec"),
        );
        for (k, v) in [
            ("T", t),
            ("NUM_KV_HEADS", nkvh),
            ("HEAD_SIZE", hd),
            (
                "NUM_ROWS",
                4 * nkvh * i128::from(SchedulerBlockTokens::REFERENCE.get()),
            ),
        ] {
            signature.insert(k.to_string(), ArgSpec::Constexpr);
            constexprs.insert(k.to_string(), Val::Int(v));
        }
        signature.insert("FMAX".to_string(), ArgSpec::Constexpr);
        constexprs.insert("FMAX".to_string(), Val::Float(f64::from(DF::FMAX)));
        let spec = KernelSpec {
            kernel: entry.to_string(),
            signature,
            constexprs,
            file: file.to_string(),
        };
        let module = compile_kernel(src, &spec, &[1])?;
        let node = triton_ktir_superdsc::node_for(&module, Program::PagedKvWrite)
            .map_err(|e| e.to_string())?;
        let layout = triton_ktir_superdsc::layout::for_node(&node).map_err(|e| e.to_string())?;
        allocate_paged_cache(&node, &layout)?;
        let mut sym = 0i64;
        ktir_superdsc::emit::whole_function::lower_walk(
            &node,
            Some(&layout),
            &mut sym,
            &mut ktir_superdsc::emit::ktir_matmul_fp8::QuantizedActivations::default(),
        )
        .map_err(|e| e.message)
    }

    /// The write kernel's own source.
    fn kv_write_src() -> &'static str {
        let TritonKernelRow::Spliced { src, .. } = kv_write_row() else {
            panic!("the KV write row is spliced");
        };
        src
    }

    /// The write at the 16-bit pool, lowered.
    fn kv_write_ops(
        t: i128,
        nkvh: i128,
        hd: i128,
    ) -> Vec<ktir_superdsc::emit::EmittedOp<ktir_superdsc::collective::TpWorld<1>>> {
        kv_write_lowered::<ktir_superdsc::superdsc_opspec::Fp16>(kv_write_src(), t, nkvh, hd)
            .unwrap_or_else(|e| panic!("t={t} nkvh={nkvh} hd={hd}: the write lowers: {e}"))
    }
```

**Edit 4e-2.** Anchor (unique, the end of `a_two_stick_head_is_scattered_one_row_per_entry` and of the
module, `SPLICE:6820-6825`):
```rust
        let rows = kv_write_ops(1, 2, 128);
        assert_eq!(rows.len(), 2, "one scatter each for K and V: {:?}", {
            rows.iter().map(|e| e.op_name.clone()).collect::<Vec<_>>()
        });
    }
}
```
Replacement:
```rust
        let rows = kv_write_ops(1, 2, 128);
        assert_eq!(rows.len(), 2, "one scatter each for K and V: {:?}", {
            rows.iter().map(|e| e.op_name.clone()).collect::<Vec<_>>()
        });
    }

    /// ⭐ THE NARROWING FAMILY FOLD — MATCH. At a float8 pool the write's per-head (below the scatter
    /// floor, per-row) `clamp → narrow → store` chains tile the key and the value, so each lowers as
    /// the reference's one pass over the whole tensor — `clip`, `clip`, `qfp8ch` — and one scatter run
    /// of its narrowed rows: no per-head op, and no relayout (a float8 row is one stick, read in
    /// place). Decode, under the floor, and a prompt chunk, at granite-3.1-8b's 8 heads of 128.
    #[test]
    fn a_float8_write_quantizes_each_whole_tensor_once() {
        for t in [1i128, 4, 96] {
            let ops = kv_write_lowered::<ktir_superdsc::superdsc_opspec::Fp8>(
                kv_write_src(),
                t,
                8,
                128,
            )
            .unwrap_or_else(|e| panic!("t={t}: the float8 write lowers: {e}"));
            let names: Vec<&str> = ops.iter().map(|e| e.op_name.as_str()).collect();
            let count = |p: &str| names.iter().filter(|n| n.starts_with(p)).count();
            let runs = ktir_superdsc::emit::lower_ktir_to_superdsc::index_runs(t as u32 * 8).len();
            assert_eq!(count("clipmin_o"), 2, "t={t}: one clamp_min each for K and V: {names:?}");
            assert_eq!(count("clipmax_o"), 2, "t={t}: one clamp_max each: {names:?}");
            assert_eq!(count("qfp8ch_o"), 2, "t={t}: one qfp8ch each: {names:?}");
            assert_eq!(count("scatter_rows"), 0, "t={t}: nothing relaid: {names:?}");
            assert_eq!(count("scatter_run"), 2 * runs, "t={t}: one run each: {names:?}");
            assert_eq!(ops.len(), 6 + 2 * runs, "t={t}: nothing else: {names:?}");
        }
    }

    /// NEAR-MISS — a family whose windows do not tile the key (one kv head's chain absent) is not the
    /// reference's whole pass: the fold leaves it unclaimed, and its narrowing is the walk's own
    /// by-name refusal.
    #[test]
    fn a_float8_write_that_misses_a_head_is_left_unclaimed() {
        let src = kv_write_src().replace(
            "for h in tl.static_range(NUM_KV_HEADS):",
            // Seven of the test's eight kv heads.
            "for h in tl.static_range(7):",
        );
        let e = kv_write_lowered::<ktir_superdsc::superdsc_opspec::Fp8>(&src, 96, 8, 128)
            .expect_err("a family that leaves a head out is not folded");
        assert!(
            e.contains("is not one of the dtype-converts this door lowers"),
            "the narrowing's own refusal: {e}"
        );
    }

    /// NEAR-MISS — a chain without its saturation (`.to` straight off the load) is not the pattern:
    /// left unclaimed, refused by name at its narrowing. The value's chains still fold.
    #[test]
    fn a_float8_write_without_its_clamp_is_left_unclaimed() {
        let src = kv_write_src().replace(
            "k = tl.minimum(tl.maximum(k, -FMAX), FMAX).to(k_dst.dtype)",
            "k = k.to(k_dst.dtype)",
        );
        let e = kv_write_lowered::<ktir_superdsc::superdsc_opspec::Fp8>(&src, 96, 8, 128)
            .expect_err("an unsaturated narrowing is not folded");
        assert!(
            e.contains("is not one of the dtype-converts this door lowers"),
            "the narrowing's own refusal: {e}"
        );
    }

    /// NEAR-MISS — members that disagree on a bound are not one pass: left unclaimed, refused by
    /// name at the narrowing.
    #[test]
    fn a_float8_write_whose_heads_disagree_on_a_bound_is_left_unclaimed() {
        let src = kv_write_src().replace("tl.maximum(k, -FMAX)", "tl.maximum(k, -FMAX + h)");
        let e = kv_write_lowered::<ktir_superdsc::superdsc_opspec::Fp8>(&src, 96, 8, 128)
            .expect_err("members with different bounds are not folded");
        assert!(
            e.contains("is not one of the dtype-converts this door lowers"),
            "the narrowing's own refusal: {e}"
        );
    }

    /// NEIGHBOUR — the 16-bit write states no narrowing, so the fold claims nothing and the write is
    /// main's: one relaid head group per kv head at two sticks, one scatter run each for K and V, no
    /// clip and no convert.
    #[test]
    fn a_16_bit_write_is_not_a_narrowing_family() {
        let ops = kv_write_ops(96, 8, 128);
        let names: Vec<&str> = ops.iter().map(|e| e.op_name.as_str()).collect();
        assert!(
            names
                .iter()
                .all(|n| !n.starts_with("clip") && !n.starts_with("qfp8ch")),
            "{names:?}"
        );
    }
}
```
(The batched decode's `tl.maximum(sc, -65504.0)` clamp — a saturation with no narrowing store — is the
other neighbour; its lowering is pinned unchanged by this module's existing batched-decode tests.)

### Step 4 gate
`cd backends/spyre && cargo test -p scratchy-triton-splice` (the four new tests and every existing one),
`cargo test -p ktir-superdsc` (pod). Then the kv/fp8 spyre-hw build: every `kvwrite_s*` bakes; the
next refusal, if any, is the read's (a float8 page's `arith.extf` in `convert_kind_of`, or a page
landing law) — Step 5. A refusal naming a `kvwrite_s*` program is a Step 4 defect, reported with its
text.

### Step 4 purity

| edit | P1/P2 | P3 once | P4 | P5 | P7 | P8 | P10 | P11 | P12 |
|---|---|---|---|---|---|---|---|---|---|
| 4a | `FMAX` a kernel constexpr from `KvFormat` | the bound from `FloatFormat::FMAX` | — | — | — | splice | — | — | — |
| 4b | constexpr branch | the reference's condition | — | — | none | kernel | port | docstring states what the emitter does | — |
| 4c | — | a stored row's source read once (`param_rows`); the run law stated once (`run_extent`); one run body (`scatter_run<DF>`) | an unclosed family is left unclaimed; the run law's named refusal is main's | `StickExtent<Fp8>` is the width's type | stated over op kinds and identity; no head count or size | emitter | moved, not copied (both readers take the one function) | present tense | fixtures 4e |
| 4d | — | the family is read once and consumed by the emission, the skip and the run | — | — | — | emitter | the chain is two calls of the one clip door and the W8A8 convert call | — | — |
| 4e | — | — | — | — | — | tests | — | — | match, three near-misses, the neighbour |

---

## Step 5 — the read: the gather at the page's format, `fp8todl16` into main's landing

What the `kv/fp8` read programs state (Part 1's 1.4b-d through the ladder, R2 4's reading): a gathered
`ktdp.load` over the row-outermost fp8 pool → `tensor.reshape` to `[.., block, hd]` (f8E4M3FN) →
`arith.extf` to f16 (the reshape's ONE reader) → `linalg.transpose` (K) / `linalg.batch_matmul` (V).
The arm (table rows R1–R2): the reshape-of-a-gathered-page branch reads the gathered view's element
type — THE data→type door, one match. An f16 page lands exactly as main lands it. An fp8 page is
gathered at fp8 into a buffer of the gathered load's own value (`identity` + index over SEN143_FP8,
`LandingFrame::in_format::<Fp8>` — the TYPE that refuses a row not whole fp8 sticks), and ONE
`fp8todl16` widens it into main's 16-bit landing — the very buffer, frame, `batch_blocks` entry and mint
key main's f16 gather gives it — held as the widen's value; the widen op itself is skipped when the
walk reaches it. Every reader of the landing is main's, unchanged. Neither fp8 buffer is ever in the
walk's table.

### 5a — `OPSPEC`: `LandingFrame<DF>`, `block_extents`, `in_format` — R2 4a, unchanged
Type R2 4a verbatim (it needs Step 3a's `DataFormat: Copy + Debug`).

### 5b — `PW`: the landing's tensor stated once, the gather generic over its format, the widen — R2 4b
Type R2 4b verbatim (`PW:769-965`, 197 lines, md5 `f9bb2f125dd14b7baed963f3238a94b5`, its anchor is
main's), with ONE change inside `widen_landing_opspec`'s `dim` closure, so the widen's stick basis is
the convert law of Step 3e and not a second statement of it. In R2 4b's text, the lines
```rust
        df: if is_stick {
            <crate::superdsc_opspec::Fp8 as DataFormat>::DF
        } else {
            Df::Fp16
        },
```
are typed as
```rust
        df: if is_stick {
            crate::emit::convert_stick_df(OpFunc::Fp8ToDl16)
        } else {
            Df::Fp16
        },
```
(`OpFunc` is in scope in `PW`: `index_select_opspec` names `OpFunc::Identity`.)

### 5c — R2 Fix B (the `PW` re-export) and R2 Fix 1 (the pointwise tests' turbofish) — unchanged

### 5d — `WF`: the page arm — R2 4c, with the fp8 buffer minted by the gathered load's own value

**Edit 5d-1** = R2 4c(i), unchanged (its anchor is main's `head_major_skip` two lines).

**Edit 5d-2** (R2 4c(ii), re-anchored after Step 4d-2). Anchor (unique):
```rust
                || row_fold_skip.contains(&r)
                // A narrowing-store family's member: the family's whole-view chain computed it.
                || narrow_skip.contains(&r)
```
Replacement:
```rust
                || row_fold_skip.contains(&r)
                // A narrowing-store family's member: the family's whole-view chain computed it.
                || narrow_skip.contains(&r)
                // A float8 page's widen: the page arm wrote its 16-bit landing (`fp8todl16`).
                || widened.contains(&r)
```

**Edit 5d-3** = R2 4c(iii), unchanged: replace main's 149-line block (md5
`4e60ce1816898efca7faa86768220487`; first line
`                    // ⭐ THE CHUNK LEGS EVERY FRAME SHARES — one op a run of one index stick`, last
line the `                    }` closing `match frame {`, the line before
`                    gather_carried[gi] = true;`) with R2 4c(iii)'s text. (Step 4d-1 moved its line
numbers, not its text.)

**Edit 5d-4** = R2 4c(iv), anchored on the same line R2 names
(the doc line of `wide_blocks`, `/// Whether` … `holds batch blocks wider than one stick — bytes no flat frame names`, which Step
4d-3 left immediately after `narrow_chain`), typed verbatim EXCEPT these three changes inside R2's
text:

1. In `struct PageFill`, the doc line pair
   ```rust
   /// What a page fill reads: the gather, its frame in the table's 16-bit terms, the landing it fills,
   /// the reshape it lands, and the walk's mint state.
   ```
   is typed as
   ```rust
   /// What a page fill reads: the gather, its frame in the table's 16-bit terms, the landing it fills,
   /// and the walk's mint state.
   ```
   and its field line `    reshape: Ssa,` is not typed.
2. In `impl PageFormat for crate::superdsc_opspec::Fp8`'s `fill`, R2's
   ```rust
        let local = mint_scratch(
            p.k,
            p.layout,
            p.floor,
            p.reshape,
            crate::ktir_node::Scratch::NarrowLanding,
        );
        p.layout.synth_df(
            crate::place::SynthId::Act(local),
            &[narrow.landing_rows(), narrow.width().elems()],
            Self::DF,
        );
   ```
   is typed as
   ```rust
        // The float8 page's own buffer — the gathered load's value, as gathered.
        let local = mint_inter(p.k, p.layout, p.floor, p.g.load);
        p.layout.synth_df(
            crate::place::SynthId::Act(local),
            &[narrow.landing_rows(), narrow.width().elems()],
            <Self as crate::superdsc_opspec::DataFormat>::DF,
        );
   ```
   (R2 Fix C's spelling of the format; no `Scratch` variant is added — R2 3d is not typed.)
3. In `land_page`, both `PageFill { … }` literals are typed without their `reshape,` line.

### 5e — R2 4d (the `to_ktir_emit.rs` comment) — unchanged
### 5f — R2 Test 4e (`zz_the_landing_frame_is_the_reshape_law.rs`) — unchanged

### Step 5 gate — the feature's first full bake
`cd backends/spyre && cargo test -p ktir-superdsc -p scratchy-triton-splice -p scratchy-target-spyre`
(pod); clippy `-D warnings` on both workspaces as `CLAUDE.md` states it; the kv/fp8 spyre-hw build:
`BUILD_RC=0`, every rung of granite-3.1-8b-instruct fp8-dynamic-per-channel baked. Then `spyre chat` on nickm's card (short prompts and the 4711 recall, at decode and at a batch
rung, reply text and rc quoted — the verdict).

### Step 5 purity

| edit | P3 once | P4 | P5/P6 | P8 | P10 |
|---|---|---|---|---|---|
| 5a/5b | `block_extents` once (gather + widen read it); `landing_arg` once (both sides of the widen and the gather's output); the convert's stick basis read from `convert_stick_df` | `in_format` is the type that refuses a non-whole fp8 row, at bake | `LandingFrame<DF>`, `IndexSelectPage<DF>`, default `Fp16` so every fp16 caller is unchanged | emitter | the widen states the landing by the gather's own function |
| 5d | the page format decided once (`indirect_view_elem`, one match); the widen held once (`PageFormat::held`) | other formats refused by name at the bake (lead ruling 11-20) | `PageFormat` over the format TYPE | emitter | `land_page` IS main's page arm, generic; the fp8 buffer is an SSA value's own mint (`MintKey::Value`), no new key kind |

---

## Step 6 — the card acceptance leg: the KV round trip at production geometry

**Rulings (lead, 11-22):**
- **Q-C2:** the V half is the proof for both pools.
- **Q-C1′ — RULED:** route (a′), approved. The production programs are baked into bundles of their own, through `bundle_run`'s file inputs and outputs: the existing card door, a test's I/O and not a dump. The assertions are exact and in Rust, no human reads the bytes, and a written pool is only restaged, never interpreted. No new runtime API.

**What ONE run of the leg proves.** Six bundles are built and launched in one process. Each is the
program chat's own bake mints, compiled through the same splice function at the same format
(`Fp8`). Only its grouping and its one-kernel placement differ (`triton_ktir_superdsc::layout::for_node`,
the layout every one-kernel test in the tree takes):
- **REF:** the device's own `qfp8ch` — the W8A8 activation quantizer's convert door (`assemble_convert`, the card-proven `fq_afp8_op`) — over a host-staged `[rows, 128]` fp16 pool image. Its fp8 output IS a pool image, `[rows][128]` = row-outermost, made by no write path of ours.
- **W:** the KV write (`kv_write_program::<Fp8>`, what `lower_kv_write` mints).
- **D8 / D16:** the decode attention (`paged_attn_program`, decode) at an fp8 pool and, as the not-X control, at an fp16 pool.
- **P8 / P16:** the prefill attention (`paged_attn_program`, prefill) at both pools.

The phases and their verdicts (each check is bit-exact):

| phase | runs | asserts | localizes |
|---|---|---|---|
| 0 | REF over the K and V pattern images | — (yields the reference pool bytes) | — |
| 1 | W over the same K/V patterns as `[T, 1024]` keys | W's pool bytes == REF's pool bytes, byte for byte (an equality, nothing interpreted) | **the write**, alone |
| 2c | D16 and P16 over fp16 pools holding the same values | the output rows == the pattern, f16 bits | the attention's own arithmetic (`exp(0)`, the one-hot dot, the normalize) — the not-X control |
| 2 | D8 and P8 over REF's pools | the output rows == the pattern's f16 bits | **the read** (gather → `fp8todl16` → landing → dot), on bytes no write of ours made |
| 3 | D8 and P8 over W's pools | the output rows == the pattern's f16 bits | the round trip, write → read |

Reading the result:
- 1 fails → the write is guilty, whatever 2 says.
- 2c passes and 2 fails → the read is guilty.
- 2c fails → the harness's arithmetic is inexact, and 2/3 say nothing.
- 1 and 2 pass and 3 fails → the leg itself is wrong. (W and REF are byte-equal there, so 3 must equal 2.)

**The read made observable.** The query is all zeros, so every score is `0` exactly. The mask stack
leaves ONE key slot `s` per query row at `0` and sets every other slot to `−65504`, so after the max
the slot's `probs` is `exp(0)` and every other slot's is `exp(−65504) = 0`. Every query head `j`'s
output is then `v_page[j / 4, s, :]`. Decode reads slot `s ∈ SLOTS` (eight launches); prefill reads
slot `2p` for query row `p` (`p < 64`, one launch). The K pool's values never reach the output, since
`q = 0`, but its gather, widen and restickify run as chat's do.

**The pattern.** `x(which, kv, t, d) = ±(1 + m/8)·2^e`, with:
- the sign `−` when `t + d + which` is odd;
- `m = (7t + d + 3·which) mod 8`;
- `e = ((kv + d + which) mod 9) − 4`.

Every value is an e4m3 NORMAL with a 3-bit mantissa, `|x| ∈ [2⁻⁴, 30]`, so it is also exact in fp16
and in DLFloat16, and the saturation never binds. The leg checks this before any launch: each value
is equal to the decode of its own `f32_to_e4m3` byte (the encoder pinned to the card's `qfp8ch`,
`ktir-core/src/codec.rs`). K is `which = 0`, V is `which = 1`.

**Geometry.** granite-3.1-8b:
- 8 kv heads, 32 query heads, head width 128;
- block 128, a pool of 2 pages (`P = 16` folded rows, `2048` token rows);
- `T = 128`, one whole page written, the `T ≥ 8` branch;
- a decode walk of 1 page; a prefill of 64 query rows;
- `SCALE = 0.0078125` (the config's `attention_multiplier`).

Page 0 holds the pattern, written through `row_index[h·T + t] = h·128 + t` (head `h`, slot `t` of
page 0, spyre-inference's `kv_write_index`), and page 1 stays zero.

**On a failure:**
- each failing check prints ONE line, naming the phase, the program, the slot (or query row) and the first differing element index with its got/want words;
- every phase runs before the leg exits, and it exits non-zero if any check failed;
- no buffer is printed.

### 6a — `SPLICE`: the pool's pointer type at any format

**Edit 6a-1.** Anchor (Part 1's 1.3c text in the tree, unique, `SPLICE:123-131`):
```rust
/// The Triton float type the KV pools are stated as — [`KvFormat`]'s, found in the door's own
/// mapping ([`stated_float`]), so a format with no Triton float is a compile error here.
const KV_POOL_FLOAT: FloatKind = stated_float(<KvFormat as DataFormat>::DF);

/// ⭐ THE KV POOLS' POINTER TYPE — the one statement every paged kernel's `k_pages`/`v_pages` and
/// the write's `k_rows`/`v_rows` take.
fn kv_pool_ptr() -> ArgSpec {
    ArgSpec::Ptr(triton_frontend::ttir::Type::Float(KV_POOL_FLOAT))
}
```
Replacement:
```rust
/// The Triton float a pool held at format `P` is stated as — found in the door's own mapping
/// ([`stated_float`]) as an associated CONST, evaluated for every `P` a caller names, so a format
/// with no Triton float is a compile error at that caller.
struct PoolFloat<P>(core::marker::PhantomData<P>);

impl<P: DataFormat> PoolFloat<P> {
    const KIND: FloatKind = stated_float(P::DF);
}

/// ⭐ A PAGED POOL'S POINTER TYPE at format `P` — the one statement every paged kernel's
/// `k_pages`/`v_pages` and the write's `k_rows`/`v_rows` take.
fn pool_ptr<P: DataFormat>() -> ArgSpec {
    ArgSpec::Ptr(triton_frontend::ttir::Type::Float(PoolFloat::<P>::KIND))
}

/// The build's KV pools' pointer type: [`pool_ptr`] at [`KvFormat`].
fn kv_pool_ptr() -> ArgSpec {
    pool_ptr::<KvFormat>()
}
```

### 6b — `SPLICE`: the KV write's program, stated once, at any format

**Edit 6b-1.** Anchor (unique — the first lines of `lower_kv_write`'s body, `SPLICE:2432-2441`):
```rust
    let row = kv_write_row();
    let (src, file, entry, program) = match &row {
        TritonKernelRow::Spliced {
            src,
            file,
            entry,
            program,
        } => (*src, *file, *entry, *program),
        TritonKernelRow::Refused(why) => return Err(why.to_string()),
    };
    let SubOp::RopeAppend {
```
Replacement:
```rust
    let SubOp::RopeAppend {
```

**Edit 6b-2.** Replace `SPLICE:2487-2525` (39 lines, md5 `9456146ce299dcd3f869f48ecf64b9fd`; first
line `    let mut signature: HashMap<String, ArgSpec> = HashMap::new();` — the one inside
`lower_kv_write` — last line `        .map_err(|e| format!("triton splice: node_for: {e}"))?;`) with:
```rust
    let positional = kv_write_program::<KvFormat>(KvWriteGeometry {
        rows,
        nkvh,
        hd,
        pages,
    })?;
```

**Edit 6b-3.** Anchor (unique, `SPLICE:2542`):
```rust
/// A compiled program re-stated with this node's own name and bindings (`node_for` mints positional
```
Replacement:
```rust
/// ⭐ THE PAGED KV WRITE'S GEOMETRY — what `reshape_and_cache_head_major` is compiled at: the token
/// rows, the kv heads, the head width, and the pool's pages.
#[derive(Clone, Copy, Debug)]
pub struct KvWriteGeometry {
    pub rows: u32,
    pub nkvh: u32,
    pub hd: HeadWidth,
    pub pages: u32,
}

/// ⭐ THE KV WRITE'S PROGRAM — spyre-inference's `reshape_and_cache_head_major_kernel` compiled at
/// `g` with its pools held at format `P`, minted positionally (`[key, value, k_rows, v_rows,
/// row_index]`, buffers 0..5). The one statement of the write's launch contract: [`lower_kv_write`]
/// rebinds it at the build's [`KvFormat`], and the card's KV round-trip leg launches it as minted.
pub fn kv_write_program<P: FloatFormat>(g: KvWriteGeometry) -> Result<KtirNode, String> {
    let (src, file, entry, program) = match kv_write_row() {
        TritonKernelRow::Spliced {
            src,
            file,
            entry,
            program,
        } => (src, file, entry, program),
        TritonKernelRow::Refused(why) => return Err(why.to_string()),
    };
    let KvWriteGeometry {
        rows,
        nkvh,
        hd,
        pages,
    } = g;
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    for (p, ty) in [("key", "*fp16"), ("value", "*fp16"), ("row_index", "*i32")] {
        signature.insert(
            p.to_string(),
            ArgSpec::parse(ty).map_err(|e| e.to_string())?,
        );
    }
    for p in ["k_rows", "v_rows"] {
        signature.insert(p.to_string(), pool_ptr::<P>());
    }
    // ⭐ THE POOL FORMAT'S SATURATION BOUND — `torch.finfo(k_rows.dtype).max`, the kernel's `FMAX`.
    // The kernel reads it only where the pool's element type is not the key's.
    signature.insert("FMAX".to_string(), ArgSpec::Constexpr);
    constexprs.insert("FMAX".to_string(), Val::Float(f64::from(P::FMAX)));
    for (k, v) in [
        ("T", rows),
        ("NUM_KV_HEADS", nkvh),
        ("HEAD_SIZE", hd.get()),
        (
            "NUM_ROWS",
            pages * nkvh * SchedulerBlockTokens::REFERENCE.get(),
        ),
    ] {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), Val::Int(i128::from(v)));
    }
    let spec = KernelSpec {
        kernel: entry.to_string(),
        signature,
        constexprs,
        file: file.to_string(),
    };
    let module = compile_kernel(src, &spec, &[1])?;
    triton_ktir_superdsc::node_for_compiled(&module, program)
        .map_err(|e| format!("triton splice: node_for: {e}"))
}

/// Which of spyre-inference's two per-sequence paged kernels a program is: the decode kernel at
/// one query row, or the prefill kernel at a padded query length.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PagedSequence {
    Decode,
    Prefill { padded_query_len: u32 },
}

/// ⭐ A PER-SEQUENCE PAGED ATTENTION'S GEOMETRY — what `page_attn_head_major_decode` /
/// `_prefill` is compiled at.
#[derive(Clone, Copy, Debug)]
pub struct PagedAttnGeometry {
    pub seq: PagedSequence,
    /// `NUM_TOKENS` — the query's rows.
    pub rows: u32,
    pub nqh: u32,
    pub nkvh: u32,
    pub hd: HeadWidth,
    pub pages: u32,
    /// `NUM_BLOCKS` — the pages the walk reads ([`walk_pages`]).
    pub blocks: u32,
    pub scale: f32,
}

/// ⭐ THE PER-SEQUENCE PAGED ATTENTION'S PROGRAM — spyre-inference's kernel for `g.seq` compiled at
/// `g` with its pools held at format `P`, minted positionally (`[query, query_row_index, k_pages,
/// v_pages, page_index_table, (kv_row_pool,) mask_stack, out]`). The one statement of its launch
/// contract: [`lower_attn`] rebinds it at the build's [`KvFormat`] per window, and the card's KV
/// round-trip leg launches it as minted.
pub fn paged_attn_program<P: DataFormat>(g: PagedAttnGeometry) -> Result<KtirNode, String> {
    let PagedAttnGeometry {
        seq,
        rows,
        nqh,
        nkvh,
        hd,
        pages,
        blocks,
        scale,
    } = g;
    let decode = seq == PagedSequence::Decode;
    let row = if decode { attn_decode_row() } else { attn_row() };
    let (src, file, entry, program) = match row {
        TritonKernelRow::Spliced {
            src,
            file,
            entry,
            program,
        } => (src, file, entry, program),
        TritonKernelRow::Refused(why) => return Err(why.to_string()),
    };
    let mut signature: HashMap<String, ArgSpec> = HashMap::new();
    let mut constexprs: HashMap<String, Val> = HashMap::new();
    let mut pointers = vec![
        ("query", "*fp16"),
        ("query_row_index", "*i32"),
        ("page_index_table", "*i32"),
    ];
    if decode {
        pointers.push(("kv_row_pool", "*i32"));
    }
    pointers.extend([("mask_stack", "*fp16"), ("out", "*fp16")]);
    for (p, ty) in pointers {
        signature.insert(
            p.to_string(),
            ArgSpec::parse(ty).map_err(|e| e.to_string())?,
        );
    }
    for p in ["k_pages", "v_pages"] {
        signature.insert(p.to_string(), pool_ptr::<P>());
    }
    let mut sizes = vec![
        ("NUM_TOKENS", rows),
        ("NUM_HEADS", nqh),
        ("NUM_KV_HEADS", nkvh),
        ("HEAD_SIZE", hd.get()),
        ("BLOCK_SIZE", SchedulerBlockTokens::REFERENCE.get()),
        ("NUM_PAGES", pages),
        ("NUM_BLOCKS", blocks),
    ];
    // ⭐ ONE REQUEST'S QUERY WIDTH, NOT THE GRAPH'S ROWS — the prefill kernel gathers `P` rows
    // through `query_row_index` (a per-request table) and scatters them back through the same rows,
    // so its padded query length is the rung's per-request width.
    if let PagedSequence::Prefill { padded_query_len } = seq {
        sizes.push(("PADDED_QUERY_LEN", padded_query_len));
    }
    for (k, v) in sizes {
        signature.insert(k.to_string(), ArgSpec::Constexpr);
        constexprs.insert(k.to_string(), Val::Int(i128::from(v)));
    }
    signature.insert("SCALE".to_string(), ArgSpec::Constexpr);
    constexprs.insert("SCALE".to_string(), Val::Float(f64::from(scale)));
    let spec = KernelSpec {
        kernel: entry.to_string(),
        signature,
        constexprs,
        file: file.to_string(),
    };
    let module = compile_kernel(src, &spec, &[1])?;
    triton_ktir_superdsc::node_for_compiled(&module, program)
        .map_err(|e| format!("triton splice: node_for: {e}"))
}

/// A compiled program re-stated with this node's own name and bindings (`node_for` mints positional
```

### 6c — `SPLICE`: `lower_attn`'s sequence arm takes that program

**Edit 6c-1.** Replace `SPLICE:2294-2342` (49 lines, md5 `93e6ff90fa71048a66980b014277d252`; first
line `    let mut pointers = vec![`, last line
`        .map_err(|e| format!("triton splice: node_for: {e}"))?;` — the one before
`    // ⭐ THE WINDOW FAMILY`) with:
```rust
    let positional = paged_attn_program::<KvFormat>(PagedAttnGeometry {
        seq: if decode {
            PagedSequence::Decode
        } else {
            PagedSequence::Prefill {
                padded_query_len: query_rows.per_request().get(),
            }
        },
        rows: mq,
        nqh,
        nkvh,
        hd,
        pages,
        blocks,
        scale,
    })?;
```
(`lower_attn`'s `signature`, `constexprs` and `(src, file, entry, program)` stay; its batch arm reads
them. At `KvFormat` every program is the one `lower_attn` and `lower_kv_write` minted before: the
same spec, the same compile, the same memo key.)

### 6d — the bake crate: one bundle directory compiled outside a bake

`backends/spyre/crates/spyre/src/superdsc_bake.rs`. **Edit 6d-1.** Anchor (unique, `:1371-1373`):
```rust
    fn compile(&self, group: &SealedGroup, reaper: &Reaper) -> Result<(), String> {
        let explore = group.explore();
        let group = group.path();
```
Replacement:
```rust
    fn compile(&self, group: &SealedGroup, reaper: &Reaper) -> Result<(), String> {
        self.run_dxp(group.path(), group.explore(), reaper)
    }

    /// ⭐ ONE BUNDLE DIRECTORY COMPILED IN PLACE, OUTSIDE A BAKE — the same command, environment,
    /// knob guard and verdict every bake group takes ([`Self::run_dxp`]), for a card leg that bakes
    /// its own bundles (`scratchy-target-spyre-card`'s `kv_fp8_round_trip`). `explore` is the
    /// directory's own ([`ChunkExplore::of_group`] over its trips).
    pub fn compile_dir(&self, dir: &Path, explore: ChunkExplore) -> Result<(), String> {
        self.run_dxp(dir, explore, &Reaper::default())
    }

    /// The body of [`Self::compile`]: `group` compiled in place under `explore`, its child
    /// registered with `reaper`.
    fn run_dxp(&self, group: &Path, explore: ChunkExplore, reaper: &Reaper) -> Result<(), String> {
```
(The rest of `compile`'s old body follows unchanged and is now `run_dxp`'s: it reads `group` as a
`&Path`, plus `explore` and `reaper`, exactly the three names it read before.)

### 6e — the card crate's manifest

`backends/spyre-hw/crates/card/Cargo.toml`. **Edit 6e-1.** Anchor (unique):
```toml
tracing-subscriber = { workspace = true }
```
Replacement:
```toml
tracing-subscriber = { workspace = true }
# EXAMPLES ONLY. `kv_fp8_round_trip` mints the production KV write and paged attention programs
# (the splice's own constructors), lowers and places them as a one-kernel bundle, and reads the
# e4m3 codec it checks its pattern against.
scratchy-triton-splice = { path = "../../../spyre/crates/triton/splice" }
triton-ktir-superdsc = { path = "../../../spyre/crates/triton/triton-ktir-superdsc" }
ktir-core = { path = "../../../spyre/crates/ktir/ktir-core" }
```

**Edit 6e-2.** Anchor (unique, the file's last example):
```toml
# One process, N cards: an all-gather over the real flex card-to-card path, byte-exact on every card.
[[example]]
name = "cards_all_gather"
```
Replacement:
```toml
# One process, N cards: an all-gather over the real flex card-to-card path, byte-exact on every card.
[[example]]
name = "cards_all_gather"

# The fp8 KV cache's card leg: the production write and paged attention programs at granite-8b's
# geometry, ONE run separating the write from the read — the write's pool bytes against the
# device's own `qfp8ch`, the read over that reference pool and over the write's, an fp16-pool
# control, every check bit-exact.
[[example]]
name = "kv_fp8_round_trip"
```

### 6f — the leg: new file `backends/spyre-hw/crates/card/examples/kv_fp8_round_trip.rs`

```rust
// SPDX-License-Identifier: Apache-2.0

//! The fp8 KV cache's card leg: ONE run separates the write from the read.
//!
//! ```bash
//! cargo --config patches/oco.toml run --release -p scratchy-target-spyre-card \
//!   --example kv_fp8_round_trip -- <work-dir>
//! ```
//!
//! Six bundles, each a production program — the splice's own constructors, the ones chat's bake
//! calls — lowered and placed as a one-kernel bundle and compiled by the bake's own dxp door:
//!
//! * `ref` — the device's own `qfp8ch` (the W8A8 activation quantizer's convert door) over a host
//!   `[rows, 128]` fp16 pool image: an fp8 pool image no write path of ours made;
//! * `write` — `reshape_and_cache_head_major` at an fp8 pool;
//! * `decode8` / `prefill8` — the paged attention at an fp8 pool;
//! * `decode16` / `prefill16` — the same at an fp16 pool, the not-X control.
//!
//! The checks, every one bit-exact:
//! 1. the write's pool bytes equal the reference pool's, byte for byte — the write, alone;
//! 2c. the fp16-pool attention returns the pattern — the attention's own arithmetic;
//! 2. the fp8-pool attention over the REFERENCE pool returns the pattern — the read, alone;
//! 3. the fp8-pool attention over the WRITTEN pool returns the pattern — the round trip.
//!
//! The read is observed through the attention's own dot: the query is zero, so every score is 0,
//! and the mask leaves ONE slot `s` of each query row unmasked — so each query head `j`'s output is
//! `v[j / 4, s, :]`. A failing check prints the phase, the program, the slot or row, and the first
//! differing element; every phase runs, and the leg exits non-zero if any check failed.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result, anyhow, bail};
use ktir_superdsc::collective::TpWorld;
use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::ktir_node::KtirNode;
use ktir_superdsc::place::{PlaceId, act_name};
use ktir_superdsc::placement::{BundleLayout, SegRole, TensorPlacement};
use ktir_superdsc::reserved_tids::{PAGED_INT32_ZERO_TID, RESERVED_TID_FLOOR, scalarmul_scale_tid};
use ktir_superdsc::superdsc_opspec::{Df, Fp8, Fp16};
use scratchy_target_spyre::superdsc_bake::{ChunkExplore, DxpTool};
use scratchy_triton_splice::{
    HeadWidth, KvWriteGeometry, PagedAttnGeometry, PagedSequence, SchedulerBlockTokens,
    allocate_paged_cache, kv_write_program, paged_attn_program,
};

/// A one-card leg: a world of one.
type W1 = TpWorld<1>;

/// granite-3.1-8b's attention geometry: kv heads, query heads, head width.
const NKVH: u32 = 8;
const NQH: u32 = 32;
const HD: u32 = 128;
/// The scheduler's block, the page's slots.
const BLOCK: u32 = SchedulerBlockTokens::REFERENCE.get();
/// The pool: two pages; page 0 holds the pattern, page 1 stays zero.
const PAGES: u32 = 2;
/// The folded pool's token rows — `[PAGES · NKVH · BLOCK, HD]`.
const POOL_ROWS: u32 = PAGES * NKVH * BLOCK;
/// The write's token rows: one whole page (the kernel's `T >= 8` branch).
const T: u32 = BLOCK;
/// The decode slots read back — the first, a stick of index words' edges, a page's middle and end.
const SLOTS: [u32; 8] = [0, 1, 7, 8, 63, 64, 126, 127];
/// The prefill's query rows; row `p` reads slot `2p`.
const PREFILL_ROWS: u32 = 64;
/// granite-3.1-8b's `attention_multiplier`: the score scale chat's program is compiled at.
const SCALE: f32 = 0.0078125;
/// A masked slot's additive mask: `exp(−65504)` is 0 in fp16.
const MASKED: f32 = -65504.0;

/// ⭐ THE PATTERN — `±(1 + m/8)·2^e`, `m ∈ 0..8`, `e ∈ −4..=4`: an e4m3 NORMAL with a 3-bit
/// mantissa, so exact in e4m3, fp16 and DLFloat16, and inside the saturation. K is `which = 0`,
/// V `which = 1`.
fn pattern(which: u32, kv: u32, t: u32, d: u32) -> f32 {
    let sign = if (t + d + which) % 2 == 0 { 1.0 } else { -1.0 };
    let m = ((t * 7 + d + which * 3) % 8) as f32;
    let e = ((kv + d + which) % 9) as i32 - 4;
    sign * (1.0 + m / 8.0) * 2f32.powi(e)
}

/// The value of an e4m3fn byte (bias 7, no infinity).
fn e4m3(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0 } else { 1.0 };
    let e = i32::from((b >> 3) & 0xF);
    let m = f32::from(b & 7);
    if e == 0 {
        sign * (m / 8.0) * 2f32.powi(-6)
    } else {
        sign * (1.0 + m / 8.0) * 2f32.powi(e - 7)
    }
}

/// Every pattern value is its own e4m3 rounding — the card's `qfp8ch` encoder's byte, decoded.
fn check_pattern() -> Result<()> {
    for which in 0..2 {
        for kv in 0..NKVH {
            for t in 0..BLOCK {
                for d in 0..HD {
                    let v = pattern(which, kv, t, d);
                    let back = e4m3(ktir_core::codec::f32_to_e4m3(v));
                    if back != v {
                        bail!(
                            "the pattern is not e4m3-exact at which={which} kv={kv} t={t} d={d}: \
                             {v} rounds to {back}"
                        );
                    }
                }
            }
        }
    }
    Ok(())
}

/// A pool image `[POOL_ROWS, HD]`, row `(page·NKVH + kv)·BLOCK + t`: page 0 the pattern, page 1 zero.
fn pool_image(which: u32) -> Vec<f32> {
    let mut x = vec![0f32; (POOL_ROWS * HD) as usize];
    for kv in 0..NKVH {
        for t in 0..BLOCK {
            for d in 0..HD {
                x[((kv * BLOCK + t) * HD + d) as usize] = pattern(which, kv, t, d);
            }
        }
    }
    x
}

fn f16_bytes(vals: &[f32]) -> Vec<u8> {
    vals.iter()
        .flat_map(|v| half::f16::from_f32(*v).to_le_bytes())
        .collect()
}

fn u32_bytes(vals: &[u32]) -> Vec<u8> {
    vals.iter().flat_map(|v| v.to_le_bytes()).collect()
}

/// `bytes` written as `dir/name`; the path, as `bundle_run` takes it.
fn put(dir: &Path, name: &str, bytes: &[u8]) -> Result<String> {
    let p = dir.join(name);
    std::fs::write(&p, bytes).with_context(|| format!("write {}", p.display()))?;
    Ok(p.display().to_string())
}

/// A placement's size, by tid.
fn size_of(layout: &BundleLayout, tid: u32) -> Result<usize> {
    layout
        .placements
        .get(&tid)
        .map(|p| p.size as usize)
        .ok_or_else(|| anyhow!("t{tid} is not placed"))
}

/// `vals` as fp16, zero-padded to the placement's whole size (an fp16 bind is exactly its size).
fn f16_at(layout: &BundleLayout, tid: u32, vals: &[f32]) -> Result<Vec<u8>> {
    let mut b = f16_bytes(vals);
    b.resize(size_of(layout, tid)?, 0);
    Ok(b)
}

/// The directory written (`write_dir` + `write_placements`, after the lowering, so the synthetics
/// are in the segment sizes) and compiled in place by the bake's own dxp door.
fn bake(dir: &Path, ops: &[EmittedOp<W1>], layout: &BundleLayout, dxp: &DxpTool) -> Result<()> {
    let _ = std::fs::remove_dir_all(dir);
    triton_ktir_superdsc::bake::write_dir(dir, ops).map_err(|e| anyhow!("{}: {e}", dir.display()))?;
    triton_ktir_superdsc::bake::write_placements(dir, layout)
        .map_err(|e| anyhow!("{}: {e}", dir.display()))?;
    dxp.compile_dir(dir, ChunkExplore::of_group(ops.iter().map(|e| e.dsc())))
        .map_err(|e| anyhow!("{}: {e}", dir.display()))
}

/// A splice-minted program, placed by the one-kernel layout and lowered by the walk chat's bake runs.
fn lowered(node: &KtirNode) -> Result<(BundleLayout, Vec<EmittedOp<W1>>)> {
    let layout = triton_ktir_superdsc::layout::for_node(node).map_err(|e| anyhow!("{e}"))?;
    allocate_paged_cache(node, &layout).map_err(|e| anyhow!("{e}"))?;
    let mut sym = 0i64;
    let ops = ktir_superdsc::emit::whole_function::lower_walk::<W1>(
        node,
        Some(&layout),
        &mut sym,
        &mut ktir_superdsc::emit::ktir_matmul_fp8::QuantizedActivations::default(),
    )
    .map_err(|e| anyhow!("{}", e.message))?;
    Ok((layout, ops))
}

/// The reserved tensors the layout placed: each registry slot its value (`registry_constants`, slot
/// `i` at `scalarmul_scale_tid(i)`), the paged step's int32 zero stick its zeros. Any other reserved
/// placement is refused by name: this leg would not know what to stage there.
fn reserved_inputs(dir: &Path, node: &KtirNode, layout: &BundleLayout) -> Result<Vec<String>> {
    let consts = triton_ktir_superdsc::layout::registry_constants(node).map_err(|e| anyhow!("{e}"))?;
    let mut out = Vec::new();
    for (&tid, p) in layout.placements.iter().filter(|(t, _)| **t >= RESERVED_TID_FLOOR) {
        let bytes = match (0..consts.len()).find(|&i| scalarmul_scale_tid(i) == tid) {
            Some(i) => f16_bytes(&[consts[i]]),
            None if tid == PAGED_INT32_ZERO_TID => vec![0u8; p.size as usize],
            None => bail!("t{tid}: a reserved placement this leg does not stage"),
        };
        out.push(format!("t{tid}={}", put(dir, &format!("reserved_{tid}.bin"), &bytes)?));
    }
    Ok(out)
}

fn run(dir: &Path, inputs: Vec<String>, outputs: Vec<String>) -> Result<()> {
    scratchy_target_spyre_card::bundle_run::run(dir, &inputs, &outputs)
        .with_context(|| format!("bundle_run {}", dir.display()))
}

/// ⭐ PHASE 0 — the reference pool: the device's own `qfp8ch` over the pool image (the W8A8
/// activation quantizer's convert door, `fq_afp8_op`'s call), its `[POOL_ROWS, HD]` fp8 output
/// one stick a row — the row-outermost pool image itself.
fn reference_pool(root: &Path, dxp: &DxpTool, which: u32) -> Result<Vec<u8>> {
    let dir = root.join(format!("ref{which}"));
    let (src, dst) = (0u32, 1u32);
    let seg = SegRole::Activation.segment();
    let mut l = BundleLayout::default();
    let mut off = 0u64;
    for (tid, size) in [
        (src, u64::from(POOL_ROWS) * u64::from(HD) * 2),
        (dst, u64::from(POOL_ROWS) * u64::from(HD)),
    ] {
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: seg,
                bank: 0,
                offset: off,
                size,
            },
        );
        off += size;
    }
    // + the H2D window pad, as the page-gather sweep's hand layout states it.
    l.segment_bytes[seg] = off + 4096;
    let mut sym = 0i64;
    let op = ktir_superdsc::emit::assemble_convert::<W1>(
        "qfp8ch_ref",
        "qfp8ch",
        POOL_ROWS,
        HD,
        &ktir_superdsc::emit::rb(&act_name(src), POOL_ROWS, HD),
        &ktir_superdsc::emit::rb_df(&act_name(dst), POOL_ROWS, HD, Df::Fp8),
        &mut sym,
        Some(&l),
    )
    .map_err(|e| anyhow!("{e}"))?;
    bake(&dir, &[op], &l, dxp)?;
    let x = put(&dir, "image.bin", &f16_bytes(&pool_image(which)))?;
    let pool = dir.join("pool.bin");
    run(
        &dir,
        vec![format!("t{src}={x}"), format!("stick:t{src}={POOL_ROWS}")],
        vec![format!("raw:t{dst}={}", pool.display())],
    )?;
    std::fs::read(&pool).with_context(|| format!("read {}", pool.display()))
}

/// ⭐ PHASE 1 — the production KV write over the K and V patterns as `[T, NKVH·HD]` rows, into zeroed
/// fp8 pools; the written pools' bytes.
fn written_pools(root: &Path, dxp: &DxpTool) -> Result<(Vec<u8>, Vec<u8>)> {
    let dir = root.join("write");
    let node = kv_write_program::<Fp8>(KvWriteGeometry {
        rows: T,
        nkvh: NKVH,
        hd: HeadWidth::of_whole_sticks(HD),
        pages: PAGES,
    })
    .map_err(|e| anyhow!(e))?;
    let (layout, ops) = lowered(&node)?;
    bake(&dir, &ops, &layout, dxp)?;
    let rows = |which: u32| -> Vec<f32> {
        let mut x = vec![0f32; (T * NKVH * HD) as usize];
        for t in 0..T {
            for kv in 0..NKVH {
                for d in 0..HD {
                    x[(t * NKVH * HD + kv * HD + d) as usize] = pattern(which, kv, t, d);
                }
            }
        }
        x
    };
    // spyre-inference's `kv_write_index`: head `h` of slot `t` of page 0 is row `h·BLOCK + t`, one
    // `[T]` index per head, one after another.
    let index: Vec<u32> = (0..NKVH)
        .flat_map(|h| (0..T).map(move |t| h * BLOCK + t))
        .collect();
    let mut inputs = vec![
        format!("t0={}", put(&dir, "key.bin", &f16_bytes(&rows(0)))?),
        format!("stick:t0={T}"),
        format!("t1={}", put(&dir, "value.bin", &f16_bytes(&rows(1)))?),
        format!("stick:t1={T}"),
        format!("t2={}", put(&dir, "k_zero.bin", &vec![0u8; size_of(&layout, 2)?])?),
        "raw:t2".to_string(),
        format!("t3={}", put(&dir, "v_zero.bin", &vec![0u8; size_of(&layout, 3)?])?),
        "raw:t3".to_string(),
        format!("t4={}", put(&dir, "row_index.bin", &u32_bytes(&index))?),
        "raw:t4".to_string(),
    ];
    inputs.extend(reserved_inputs(&dir, &node, &layout)?);
    let (k, v) = (dir.join("k_pool.bin"), dir.join("v_pool.bin"));
    run(
        &dir,
        inputs,
        vec![
            format!("raw:t2={}", k.display()),
            format!("raw:t3={}", v.display()),
        ],
    )?;
    Ok((
        std::fs::read(&k).with_context(|| format!("read {}", k.display()))?,
        std::fs::read(&v).with_context(|| format!("read {}", v.display()))?,
    ))
}

/// The pools an attention bundle reads: fp8 pool bytes bound verbatim, or fp16 pool images staged
/// through the fp16 codec (`stick:…=1`, the identity map: a row-outermost pool is row-major).
enum Pools<'a> {
    Fp8 { k: &'a [u8], v: &'a [u8] },
    Fp16,
}

/// Which slot each query row reads, and the leg's own name for one launch's files.
struct Launch<'a> {
    slot_of: &'a dyn Fn(u32) -> u32,
    tag: &'a str,
}

/// One attention launch at `launch`'s slot law; the output row-major `[rows, NQH·HD]` fp16 words.
fn attend(
    dir: &Path,
    node: &KtirNode,
    layout: &BundleLayout,
    seq: PagedSequence,
    rows: u32,
    pools: &Pools<'_>,
    launch: Launch<'_>,
) -> Result<Vec<u16>> {
    let Launch { slot_of, tag } = launch;
    let decode = seq == PagedSequence::Decode;
    // `[query, query_row_index, k_pages, v_pages, page_index_table, (kv_row_pool,) mask_stack, out]`.
    let (mask, out) = if decode { (6u32, 7u32) } else { (5, 6) };
    let q = vec![0f32; (rows * NQH * HD) as usize];
    let mut m = vec![MASKED; (rows * BLOCK) as usize];
    for r in 0..rows {
        m[(r * BLOCK + slot_of(r)) as usize] = 0.0;
    }
    let qri: Vec<u32> = (0..rows).collect();
    // `page_index_table` `[NUM_BLOCKS, 32]`: block 0's page id, page 0, at column 0.
    let table = vec![0u32; 32];
    let mut inputs = vec![
        format!("t0={}", put(dir, &format!("{tag}_q.bin"), &f16_at(layout, 0, &q)?)?),
        format!("stick:t0={rows}"),
        format!("t1={}", put(dir, &format!("{tag}_qri.bin"), &u32_bytes(&qri))?),
        "raw:t1".to_string(),
        format!("t4={}", put(dir, &format!("{tag}_table.bin"), &u32_bytes(&table))?),
        "raw:t4".to_string(),
        format!("t{mask}={}", put(dir, &format!("{tag}_mask.bin"), &f16_at(layout, mask, &m)?)?),
        format!("stick:t{mask}={rows}"),
        format!("stick:t{out}={rows}"),
    ];
    if decode {
        let pool_rows: Vec<u32> = (0..PAGES * NKVH).collect();
        inputs.push(format!("t5={}", put(dir, &format!("{tag}_kvrp.bin"), &u32_bytes(&pool_rows))?));
        inputs.push("raw:t5".to_string());
    }
    match pools {
        Pools::Fp8 { k, v } => {
            inputs.push(format!("t2={}", put(dir, &format!("{tag}_k.bin"), k)?));
            inputs.push("raw:t2".to_string());
            inputs.push(format!("t3={}", put(dir, &format!("{tag}_v.bin"), v)?));
            inputs.push("raw:t3".to_string());
        }
        Pools::Fp16 => {
            inputs.push(format!("t2={}", put(dir, &format!("{tag}_k.bin"), &f16_at(layout, 2, &[])?)?));
            inputs.push("stick:t2=1".to_string());
            inputs.push(format!("t3={}", put(dir, &format!("{tag}_v.bin"), &f16_at(layout, 3, &pool_image(1))?)?));
            inputs.push("stick:t3=1".to_string());
        }
    }
    inputs.extend(reserved_inputs(dir, node, layout)?);
    let o = dir.join(format!("{tag}_out.bin"));
    run(dir, inputs, vec![format!("t{out}={}", o.display())])?;
    let bytes = std::fs::read(&o).with_context(|| format!("read {}", o.display()))?;
    Ok(bytes
        .chunks_exact(2)
        .take((rows * NQH * HD) as usize)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
}

/// The first element where `got` differs from the pattern's V row `slot_of(row)` of kv head
/// `j / (NQH/NKVH)` for every query head `j`: `(index, got, want)`.
fn first_wrong(got: &[u16], rows: u32, slot_of: &dyn Fn(u32) -> u32) -> Option<(usize, u16, u16)> {
    let qpk = NQH / NKVH;
    (0..rows).find_map(|r| {
        (0..NQH * HD).find_map(|c| {
            let (j, d) = (c / HD, c % HD);
            let i = (r * NQH * HD + c) as usize;
            let want = half::f16::from_f32(pattern(1, j / qpk, slot_of(r), d)).to_bits();
            (got.get(i) != Some(&want)).then(|| (i, got.get(i).copied().unwrap_or(0), want))
        })
    })
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .with_writer(std::io::stderr)
        .init();
    let root = PathBuf::from(
        std::env::args()
            .nth(1)
            .ok_or_else(|| anyhow!("usage: kv_fp8_round_trip <work-dir>"))?,
    );
    std::fs::create_dir_all(&root).with_context(|| format!("create {}", root.display()))?;
    let dxp = DxpTool::resolve()
        .ok_or_else(|| anyhow!("no dxp_standalone: DEEPTOOLS_PATH must name the SDK's share dir"))?;
    check_pattern()?;
    let mut failed: Vec<String> = Vec::new();

    // ── PHASE 0 and 1: the reference pools, the written pools, and the write's one check ──
    let (ref_k, ref_v) = (reference_pool(&root, &dxp, 0)?, reference_pool(&root, &dxp, 1)?);
    let (w_k, w_v) = written_pools(&root, &dxp)?;
    let image = (POOL_ROWS * HD) as usize;
    for (pool, w, r) in [("K", &w_k, &ref_k), ("V", &w_v, &ref_v)] {
        let (w, r) = (&w[..image.min(w.len())], &r[..image.min(r.len())]);
        match (0..image).find(|&i| w.get(i) != r.get(i)) {
            None => println!("PASS phase 1 (write): the {pool} pool is the device qfp8ch's, byte for byte"),
            Some(i) => failed.push(format!(
                "FAIL phase 1 (write): {pool} pool byte {i} (row {}, col {}) is {:?}, the reference's {:?}",
                i / HD as usize,
                i % HD as usize,
                w.get(i),
                r.get(i)
            )),
        }
    }

    // ── PHASES 2c, 2, 3: decode and prefill attention, at the fp16 control and at fp8 ──
    let hd = HeadWidth::of_whole_sticks(HD);
    let geometry = |seq: PagedSequence, rows: u32| PagedAttnGeometry {
        seq,
        rows,
        nqh: NQH,
        nkvh: NKVH,
        hd,
        pages: PAGES,
        blocks: 1,
        scale: SCALE,
    };
    let decode_slot = |s: u32| move |_r: u32| s;
    let prefill_slot = |r: u32| 2 * r;
    for (name, seq, rows) in [
        ("decode", PagedSequence::Decode, 1u32),
        (
            "prefill",
            PagedSequence::Prefill {
                padded_query_len: PREFILL_ROWS,
            },
            PREFILL_ROWS,
        ),
    ] {
        let at = |fmt: &str| root.join(format!("{name}{fmt}"));
        let n16 = paged_attn_program::<Fp16>(geometry(seq, rows)).map_err(|e| anyhow!(e))?;
        let n8 = paged_attn_program::<Fp8>(geometry(seq, rows)).map_err(|e| anyhow!(e))?;
        let (l16, ops16) = lowered(&n16)?;
        let (l8, ops8) = lowered(&n8)?;
        bake(&at("16"), &ops16, &l16, &dxp)?;
        bake(&at("8"), &ops8, &l8, &dxp)?;
        let slots: Vec<u32> = if seq == PagedSequence::Decode {
            SLOTS.to_vec()
        } else {
            vec![0]
        };
        for s in slots {
            let fixed = decode_slot(s);
            let slot_of: &dyn Fn(u32) -> u32 = if seq == PagedSequence::Decode {
                &fixed
            } else {
                &prefill_slot
            };
            for (phase, short, dir, node, layout, pools) in [
                ("2c (fp16 control)", "c", at("16"), &n16, &l16, Pools::Fp16),
                ("2 (read, reference pool)", "r", at("8"), &n8, &l8, Pools::Fp8 { k: &ref_k, v: &ref_v }),
                ("3 (round trip, written pool)", "w", at("8"), &n8, &l8, Pools::Fp8 { k: &w_k, v: &w_v }),
            ] {
                let tag = format!("{short}_s{s}");
                let got = attend(&dir, node, layout, seq, rows, &pools, Launch { slot_of, tag: &tag })?;
                match first_wrong(&got, rows, slot_of) {
                    None => println!("PASS phase {phase}: {name} slot law at s={s}"),
                    Some((i, g, w)) => failed.push(format!(
                        "FAIL phase {phase}: {name}, s={s}: element {i} (row {}, head {}, d {}) is \
                         0x{g:04x}, the pattern's 0x{w:04x}",
                        i / (NQH * HD) as usize,
                        (i % (NQH * HD) as usize) / HD as usize,
                        i % HD as usize
                    )),
                }
            }
        }
    }
    for f in &failed {
        println!("{f}");
    }
    if failed.is_empty() {
        println!("kv_fp8_round_trip: every check bit-exact");
        Ok(())
    } else {
        bail!("kv_fp8_round_trip: {} check(s) failed", failed.len())
    }
}
```

### 6g — the gate and the run, on nickm

From the synced tree on nickm (`oc rsync --delete` with the standard excludes — the pod sync law),
in the pod's standard environment (`DEEPTOOLS_PATH` set, as every bake needs), in the foreground:
```bash
cd /work/scratchy/backends/spyre-hw
cargo --config patches/oco.toml build --release -p scratchy-target-spyre-card --example kv_fp8_round_trip
cargo --config patches/oco.toml clippy -p scratchy-target-spyre-card --examples -- -D warnings
./target/release/examples/kv_fp8_round_trip /work/scratchy/backends/spyre-hw/target/kv-fp8-round-trip
echo "RC=$?"
```
and in `backends/spyre`:
```bash
cargo test -p scratchy-triton-splice
cargo clippy --workspace --features scratchy-models/granite-3.1-8b-instruct -- -D warnings
```
- **Verdict:** the leg's PASS/FAIL lines and `RC`, quoted whole.
- **Unchanged guarantees:** 6a–6c do not change the bake. The kv/fp8 build's programs are the same specs, so its gate is the same `BUILD_RC`. The coder confirms it with the kv/fp8 spyre-hw build of Step 5.

### Step 6 purity

| edit | P3 once | P4 | P8 | P10 | P11 | P12 |
|---|---|---|---|---|---|---|
| 6a | the pool pointer at a format is one function; its float found once per format, at compile time | a format with no Triton float is a compile error | splice | — | — | — |
| 6b/6c | the write's and the per-sequence attention's launch contracts each stated ONCE (`kv_write_program`, `paged_attn_program`), read by the bake's splice and by the leg | — | splice | the spec code moved out of `lower_kv_write`/`lower_attn`, not copied | present tense | — |
| 6d | the dxp command, env, guard and verdict stated once (`run_dxp`) for a bake group and a leg's dir | dxp's own refusal, by name | bake tool | `compile`'s body moved, not copied | — | — |
| 6f | the leg calls the production constructors, the walk and the one-kernel layout; it restates no law of theirs (its pattern, geometry and observability are its own statements) | a reserved placement it cannot stage is refused by name | card crate example (spyre-hw home) | — | no `#[allow]` (`attend` takes seven, its slot law and file tag one `Launch`) | in-tree Rust, exact assertions, not-X control first |


## Step 6 fixes (pod result 11-22: clippy 3 lints; the leg stopped at phase 0, "bundle ref0 places no Logits tensor")

**Cause, read in source.** `Executor::load` takes the read-back tensor from the bundle's one
`Logits` placement (`card/src/superdsc_exec.rs:1670-1674`: `code.layout.logits()`, refused by name
when absent). `bundle_run` reads each placement's role off `placements.json`
(`bundle_run.rs:431-444`: "read off the producer's role, not guessed"). The one-kernel layout
already marks every stored parameter `Logits` (`triton-ktir-superdsc/src/layout.rs:448-452`:
`if reg.is_out { SegRole::Logits }`), so W and the four attention bundles carry one. The REF bundle's
HAND layout marked its output `Activation`, and so stated no read-back tensor.

The existing door is the producer's role, so the fix is in the leg: the REF layout marks its `qfp8ch`
output `SegRole::Logits`, exactly as the one-kernel layout marks a stored output. There is no
card-runtime change.

All four edits are to `backends/spyre-hw/crates/card/examples/kv_fp8_round_trip.rs`.

**Fix 6-1 (the Logits role).** Anchor (unique):
```rust
    for (tid, size) in [
        (src, u64::from(POOL_ROWS) * u64::from(HD) * 2),
        (dst, u64::from(POOL_ROWS) * u64::from(HD)),
    ] {
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
```
Replacement:
```rust
    // The image is a per-launch input; the fp8 pool is the program's one output, which the launch
    // reads back — `SegRole::Logits`, the role the one-kernel layout gives every stored parameter.
    for (tid, size, role) in [
        (
            src,
            u64::from(POOL_ROWS) * u64::from(HD) * 2,
            SegRole::Activation,
        ),
        (dst, u64::from(POOL_ROWS) * u64::from(HD), SegRole::Logits),
    ] {
        l.ids.borrow_mut().insert(act_name(tid), PlaceId::Act(tid));
        l.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role,
```

**Fix 6-2 (`clippy::doc_lazy_continuation`, `:21`).** Anchor (unique, the module doc):
```rust
//! The checks, every one bit-exact:
//! 1. the write's pool bytes equal the reference pool's, byte for byte — the write, alone;
//! 2c. the fp16-pool attention returns the pattern — the attention's own arithmetic;
//! 2. the fp8-pool attention over the REFERENCE pool returns the pattern — the read, alone;
//! 3. the fp8-pool attention over the WRITTEN pool returns the pattern — the round trip.
```
Replacement:
```rust
//! The checks, every one bit-exact:
//! * phase 1 — the write's pool bytes equal the reference pool's, byte for byte: the write, alone;
//! * phase 2c — the fp16-pool attention returns the pattern: the attention's own arithmetic;
//! * phase 2 — the fp8-pool attention over the REFERENCE pool returns the pattern: the read, alone;
//! * phase 3 — the fp8-pool attention over the WRITTEN pool returns the pattern: the round trip.
```

**Fix 6-3 (`clippy::manual_is_multiple_of`, `:74`).** Anchor (unique):
```rust
    let sign = if (t + d + which) % 2 == 0 { 1.0 } else { -1.0 };
```
Replacement:
```rust
    let sign = if (t + d + which).is_multiple_of(2) {
        1.0
    } else {
        -1.0
    };
```

**Fix 6-4 (`clippy::chunks_exact_with_const_size`, `:386`).** Anchor (unique):
```rust
    Ok(bytes
        .chunks_exact(2)
        .take((rows * NQH * HD) as usize)
        .map(|c| u16::from_le_bytes([c[0], c[1]]))
        .collect())
```
Replacement:
```rust
    Ok(bytes
        .as_chunks::<2>()
        .0
        .iter()
        .take((rows * NQH * HD) as usize)
        .map(|&c| u16::from_le_bytes(c))
        .collect())
```

**Gate:** the two commands of 6g, unchanged (the example's build and its clippy `-D warnings`), then
the leg's run, with its PASS/FAIL lines and `RC` quoted whole.


## Step 6 fixes 2 (pod result 11-22: phase 0 PASSED; the write bundle stopped at `t4294967280: a reserved placement this leg does not stage`)

**The tid.** `t4294967280` = `u32::MAX − 15` = `FP8_INV448_TID`
(`ktir-superdsc/src/reserved_tids.rs:100`), one of the three W8A8 clamp-constant sticks
`FP8_{POS,NEG,INV}448_TID` (`:98-100`).
- **Producer:** the one-kernel layout places all three, each a `[1, 64]` fp16 stick (128 B), whenever any region states `is_fp8` (`triton-ktir-superdsc/src/layout.rs:576-591`). `regions()` stamps `is_fp8` iff the view's dtype is `Fp8E4m3` (`lower_ktir_to_superdsc.rs:1941-1949`). The fp8 KV pools make it true for W, decode8 and prefill8, although none of their ops reads the three.
- **Production staging:** the runtime stages them through ONE function, `scratchy_target_spyre::wiring::synthetic_constants(&ConstantEnv)` (`spyre/src/wiring.rs:2471-2504`). It returns "the synthetic constants a bundle needs, as `(tid, values)` in bind order": the three sticks from its statics `FP8_POS448`/`FP8_NEG448`/`FP8_INV448` (`:2384-2387`, 64 values each), "bound unconditionally", and the registry slots at `scalarmul_scale_tid(i)` from `env.scalarmul_scales` (`:2575-2583`).

The leg now stages every synthetic constant from that function, so there is no second statement of
any value. This includes the registry slots, which the leg used to look up itself.

**Every reserved placement the leg's bundles state.** `layout::for_regions` places exactly four kinds
(read in full, `layout.rs:464-591`):

| kind | condition | which bundles place it |
|---|---|---|
| the registry slots `scalarmul_scale_tid(i)` | the program reads any | the walks' seeds and constants: decode8/16, prefill8/16 |
| `PAGED_INT32_ZERO_TID` | a chained index gather | decode8/16, prefill8/16, as each states one |
| `FP8_{POS,NEG,INV}448_TID` | any `is_fp8` region | W, decode8, prefill8 |
| `ROPE_P_TID` | `Program::Rope` only | none of the leg's bundles |

REF's hand layout places none. So the synthetic list plus the int32 zero stick covers every one. The
zero stick is not a synthetic constant: production stages it as a run of zero int32 words beside
`kv_row_pool` (`spyre/src/paged_step.rs:782`, `(step.int32_zero, vec![0u8; run * 4])`), which the
leg's zeros already are.

All edits are to `backends/spyre-hw/crates/card/examples/kv_fp8_round_trip.rs`.

**Fix 2-1.** Anchor (unique, `:38`):
```rust
use ktir_superdsc::reserved_tids::{PAGED_INT32_ZERO_TID, RESERVED_TID_FLOOR, scalarmul_scale_tid};
```
Replacement:
```rust
use ktir_superdsc::reserved_tids::{PAGED_INT32_ZERO_TID, RESERVED_TID_FLOOR};
```

**Fix 2-2.** Replace the whole `reserved_inputs` function (unique; its doc's first line
`/// The reserved tensors the layout placed: each registry slot its value (\`registry_constants\`, slot`,
through its closing `}`, 16 lines) with:
```rust
/// The reserved tensors the layout placed, staged with the values production stages them with.
/// Every synthetic constant comes from the runtime's one list, `wiring::synthetic_constants`: the
/// W8A8 clamp sticks `FP8_{POS,NEG,INV}448`, the registry slots at `scalarmul_scale_tid(i)` (here
/// this program's own `registry_constants`), and the rest. The paged step's int32 zero stick is
/// zero words, as `paged_step` stages it. Any other reserved placement is refused by name: this leg
/// would not know what to stage there. `rows` is the program's query rows, the bundle's `mq_pad`.
fn reserved_inputs(
    dir: &Path,
    node: &KtirNode,
    layout: &BundleLayout,
    rows: u32,
) -> Result<Vec<String>> {
    // `ConstantEnv::scalarmul_scales` is `'static` (the bake emits it as a `static`); this leg's are
    // computed once per program, so they are leaked to that lifetime for the run.
    let scales: &'static [f32] = Box::leak(
        triton_ktir_superdsc::layout::registry_constants(node)
            .map_err(|e| anyhow!("{e}"))?
            .into_boxed_slice(),
    );
    let synthetic = scratchy_target_spyre::wiring::synthetic_constants(
        &scratchy_target_spyre::wiring::ConstantEnv {
            hidden: (NQH * HD) as usize,
            head_dim: HD as usize,
            mq_pad: rows as usize,
            uses_ones_reduce: false,
            ones_reduce_len: None,
            uses_identity: false,
            rope_class_hds: &[],
            attn_class_hds: &[],
            rms_invcols: &[],
            rows: rows as usize,
            scalarmul_scales: scales,
        },
    );
    let mut out = Vec::new();
    for (&tid, p) in layout.placements.iter().filter(|(t, _)| **t >= RESERVED_TID_FLOOR) {
        let size = p.size as usize;
        let bytes = match synthetic.iter().find(|(t, _)| *t == tid) {
            Some((_, values)) => {
                let mut b = f16_bytes(&values.to_vec());
                if b.len() > size {
                    bail!(
                        "t{tid}: production stages {} B of constants into a {size} B placement",
                        b.len()
                    );
                }
                b.resize(size, 0);
                b
            }
            None if tid == PAGED_INT32_ZERO_TID => vec![0u8; size],
            None => bail!("t{tid}: a reserved placement this leg does not stage"),
        };
        out.push(format!("t{tid}={}", put(dir, &format!("reserved_{tid}.bin"), &bytes)?));
    }
    Ok(out)
}
```

**Fix 2-3.** Anchor (unique, in `written_pools`, `:310`):
```rust
    inputs.extend(reserved_inputs(&dir, &node, &layout)?);
```
Replacement:
```rust
    inputs.extend(reserved_inputs(&dir, &node, &layout, T)?);
```

**Fix 2-4.** Anchor (unique, in `attend`, `:391`):
```rust
    inputs.extend(reserved_inputs(dir, node, layout)?);
```
Replacement:
```rust
    inputs.extend(reserved_inputs(dir, node, layout, rows)?);
```

**Gate:** 6g's build, clippy `-D warnings`, then the leg run, with its PASS/FAIL lines and `RC`
quoted whole.


## Step 6 fixes 3 (pod result 11-22: `bundle write places no Logits tensor`)

**Cause, read in source.** My fixes-1 claim was wrong: the one-kernel layout marks `Logits` only for a
parameter that a DIRECT store writes.
- `for_regions` takes the role from `reg.is_out` (`triton-ktir-superdsc/src/layout.rs:448-452`).
- `regions()` sets `is_out` from `store_views` (`ktir-superdsc/src/emit/lower_ktir_to_superdsc.rs:1795`, `:1941`).
- `store_views` maps a `ktdp.store` to a view only through a `ktdp.construct_access_tile` (`:2034-2046`), so an INDIRECT tile's destination is never `is_out`.

**Role facts, per bundle:**

| bundle | outputs | why | Logits before this fix |
|---|---|---|---|
| **write** | K pool `t2`, V pool `t3` | both written only by indirect scatters (`k_dst.scatter` / per-row `k_dst.store([row, 0], …)`, which `scatters_of` reads as indirect tiles) | none |
| **decode8 / decode16** | `out` `t7` | written by `out_rows.store([row, 0], attn)` at `row = tl.load(query_row_index)`, a data-dependent row, which the ladder lowers through an indirect tile (`out.index_copy_`) | none |
| **prefill8 / prefill16** | `out` `t6` | written by `out_rows.scatter(…, rows, 0)` (an indirect tile) | none |
| **REF** | `t1` | marked by the leg (fixes 1) | `t1` |

Every other parameter of these programs is a load, so `Activation`. `Executor::load` refuses a bundle
with no `Logits` placement (`card/src/superdsc_exec.rs:1670-1674`), so every bundle after REF stops
at load.

**Is it a layout defect?** It is a gap in the one-kernel layout's role law: a scatter's destination is
an output too. But the general fix (`store_views` through indirect tiles) changes `Region::is_out` for
every scatter destination that `regions()` serves to the production walk. That is a production-lowering
change outside this leg, and it would need its own proof, so it is NOT made here. It is **flagged for
the lead as a follow-on**. The leg marks its read-back tensors itself, as REF's was fixed: the role is
placement metadata only. The segment, offset, size and every descriptor stay the layout's, as REF's
working run shows (a `Logits` placement in segment 3).

**How the leg reads each output:**
- **Write:** the leg reads both pools as RAW outputs (`raw:t2=…`, `raw:t3=…`), which `bundle_run` reads off each placement's own window of the segment (`bundle_run.rs:1021-1040`), whatever the role. So `Logits` only has to EXIST, and the leg marks both pools, the two tensors it reads back. `Executor::load` takes the first (`BundleLayout::logits`, the first `is_logits` place: `t2`) as its logits id. `bundle_run` uses the `Logits` list only when no output is named (`:1048`), and the leg always names its outputs.
- **Attention:** the leg reads the one `out` (`t7` decode, `t6` prefill) as a plain fp16 output, and marks exactly it.

All edits are to `backends/spyre-hw/crates/card/examples/kv_fp8_round_trip.rs`.

**Fix 3-1** — the read-back marking and the attention's parameter positions, stated once. Anchor
(unique, the line opening `fn run`):
```rust
fn run(dir: &Path, inputs: Vec<String>, outputs: Vec<String>) -> Result<()> {
```
Replacement:
```rust
/// ⭐ THE READ-BACK ROLE. The one-kernel layout marks `Logits` only a parameter a DIRECT store writes
/// (`store_views` follows `ktdp.construct_access_tile`s alone), and every output of the write and of
/// the paged attention is written through an INDIRECT tile — the pools' scatters, `out.index_copy_`.
/// `Executor::load` takes a bundle's read-back tensor from its `Logits` placement, so the leg marks
/// the tensors it reads back. The role is placement metadata only: the segment, offset, size and every
/// descriptor are the layout's.
fn read_back(layout: &mut BundleLayout, tids: &[u32]) -> Result<()> {
    for &tid in tids {
        layout
            .placements
            .get_mut(&tid)
            .ok_or_else(|| anyhow!("t{tid} is not placed"))?
            .role = SegRole::Logits;
    }
    Ok(())
}

/// The per-sequence attention's `mask_stack` position in `paged_attn_program`'s positional mint
/// (`[query, query_row_index, k_pages, v_pages, page_index_table, (kv_row_pool,) mask_stack, out]`):
/// the decode kernel's `kv_row_pool` sits at 5, so its mask sits one later.
fn mask_tid(seq: PagedSequence) -> u32 {
    if seq == PagedSequence::Decode { 6 } else { 5 }
}

/// The per-sequence attention's `out` position — the one after its mask.
fn out_tid(seq: PagedSequence) -> u32 {
    mask_tid(seq) + 1
}

fn run(dir: &Path, inputs: Vec<String>, outputs: Vec<String>) -> Result<()> {
```

**Fix 3-2** — the write marks its two pools. Anchor (unique, in `written_pools`):
```rust
    let (layout, ops) = lowered(&node)?;
    bake(&dir, &ops, &layout, dxp)?;
```
Replacement:
```rust
    let (mut layout, ops) = lowered(&node)?;
    // Both pools are read back (raw, below): `[key, value, k_rows, v_rows, row_index]`.
    read_back(&mut layout, &[2, 3])?;
    bake(&dir, &ops, &layout, dxp)?;
```

**Fix 3-3** — `attend` reads its positions from the one statement. Anchor (unique, in `attend`):
```rust
    let (mask, out) = if decode { (6u32, 7u32) } else { (5, 6) };
```
Replacement:
```rust
    let (mask, out) = (mask_tid(seq), out_tid(seq));
```

**Fix 3-4** — the four attention bundles mark their `out`. Anchor (unique, in `main`):
```rust
        let (l16, ops16) = lowered(&n16)?;
        let (l8, ops8) = lowered(&n8)?;
        bake(&at("16"), &ops16, &l16, &dxp)?;
        bake(&at("8"), &ops8, &l8, &dxp)?;
```
Replacement:
```rust
        let (mut l16, ops16) = lowered(&n16)?;
        let (mut l8, ops8) = lowered(&n8)?;
        read_back(&mut l16, &[out_tid(seq)])?;
        read_back(&mut l8, &[out_tid(seq)])?;
        bake(&at("16"), &ops16, &l16, &dxp)?;
        bake(&at("8"), &ops8, &l8, &dxp)?;
```
(`attend` still binds `decode` for its `kv_row_pool` input, so that binding stays read.)

**Gate:** 6g's build, clippy `-D warnings`, then the leg run, with its PASS/FAIL lines and `RC`
quoted whole.


## Step 6 fixes 4 (pod result 11-22: `prepare: seg3: the 1053312 B shadow cannot cover a 1056768 B prefix`)

**Cause, read in source.** Two functions disagree on the unit of a segment's end.
- **The shadow:** `stage_weights` sizes a host-owned slot's shadow at EXACTLY the bake's segment size (`card/src/superdsc_exec.rs:2700-2701`, `HostBuf::zeroed(self.segment_bytes[i].max(128))`). Segment sizes are only 128-aligned (`triton-ktir-superdsc/src/layout.rs:992`, `align128(offset + size)`). Here seg3 is 1053312 B = 257.16 pages.
- **The prefix:** `send_prefix_and_fill` (`superdsc_exec.rs:987-1010`) computes `send = staged.min(shadow.len()).min(region).div_ceil(HOST_PAGE) * HOST_PAGE`. That is, it clamps the staged extent to the shadow FIRST and rounds up to a 4096-B page SECOND.
- **The overrun:** `staged` is the end of the last tensor `stage_weights` staged into the slot (`:2769-2771`). Every reserved constant (`t ≥ RESERVED_TID_FLOOR`) is staged as a weight by `bundle_run`. In the write bundle those are the three `FP8_*448` sticks, packed after the parameters at the segment's top (`layout.rs:576-591`). So `staged` falls inside the last partial page, and rounding it up passes the shadow's end by 3456 B.
- **REF ran clean** only because its hand layout added a whole 4096-B pad (`off + 4096`).
- **The doc's assumption:** the function's doc says "the bytes that rounding adds are the walk's own born zeros inside the shadow". That holds only when the shadow is whole pages, or when the staged bytes stop at least a page short of its end.

**Is it a production defect, and the chat's cause?** It is a **latent card-runtime defect**, NAMED
here for the lead: clamp-then-round in `send_prefix_and_fill`, against 128-aligned shadows. It is
**not the chat's cause**:
- the kv/fp8 build's chat passed `prepare` and ran with `rc=0`;
- production's activation segment does not end at its staged constants, because the walk's synthetics are bump-allocated past every placement and the segment is grown to their high-water (`lower_subtile_tape_to_superdsc.rs:6741`).

So production never stages into a segment's last partial page. The fix it deserves is typed, in the
runtime: a host shadow is whole pages, or `send` is rounded before it is clamped. That fix is a
runtime change, outside this leg, and needs its own proof, so it is a **follow-on for the lead**.

**The leg's fix: one host page of pad on every segment, stated once, in `bake`.** The bound:
- A staged tensor ends at or before the layout's own segment size `S` (it is a placement inside it). So the rounded prefix ends at or before `S + 4095`.
- With every nonzero segment grown by one 4096-B page, the shadow is at least `S + 4096`, whatever `write_placements`' synthetic high-water then adds. So the prefix always fits.

This is REF's `+ 4096` (the page-gather sweep's "H2D window pad"), moved from REF's hand layout into
the one function every bundle goes through. REF then states none of its own.

All edits are to `backends/spyre-hw/crates/card/examples/kv_fp8_round_trip.rs`.

**Fix 4-1.** Replace the whole `bake` function (unique; from its doc line
`/// The directory written (\`write_dir\` + \`write_placements\`, after the lowering, so the synthetics`
through its closing `}`, 10 lines) with:
```rust
/// The host page `prepare` rounds a staged prefix up to (`superdsc_exec::HOST_PAGE`).
const HOST_PAGE: u64 = 4096;

/// The directory written (`write_dir` + `write_placements`, after the lowering, so the synthetics
/// are in the segment sizes) and compiled in place by the bake's own dxp door.
///
/// ⭐ EVERY SEGMENT TAKES ONE HOST PAGE OF PAD. `prepare` sends a host-owned slot's staged prefix
/// rounded UP to a page, clamped to the shadow only before the rounding, and the shadow is the
/// segment's 128-aligned size. A reserved constant staged at a segment's top (the `FP8_*448` sticks)
/// would then round past it. A staged tensor ends at or before the layout's size `S`, so a page of
/// pad past `S` holds every rounded prefix: the page-gather sweep's H2D window pad.
fn bake(dir: &Path, ops: &[EmittedOp<W1>], layout: &mut BundleLayout, dxp: &DxpTool) -> Result<()> {
    for b in layout.segment_bytes.iter_mut().filter(|b| **b > 0) {
        *b += HOST_PAGE;
    }
    let _ = std::fs::remove_dir_all(dir);
    triton_ktir_superdsc::bake::write_dir(dir, ops).map_err(|e| anyhow!("{}: {e}", dir.display()))?;
    triton_ktir_superdsc::bake::write_placements(dir, layout)
        .map_err(|e| anyhow!("{}: {e}", dir.display()))?;
    dxp.compile_dir(dir, ChunkExplore::of_group(ops.iter().map(|e| e.dsc())))
        .map_err(|e| anyhow!("{}: {e}", dir.display()))
}
```

**Fix 4-2** (REF states no pad of its own). Anchor (unique, `:313-314`):
```rust
    // + the H2D window pad, as the page-gather sweep's hand layout states it.
    l.segment_bytes[seg] = off + 4096;
```
Replacement:
```rust
    // `bake` adds the one host page of pad every segment takes.
    l.segment_bytes[seg] = off;
```

**Fix 4-3.** Anchor (unique, `:327`): `    bake(&dir, &[op], &l, dxp)?;` → replacement
`    bake(&dir, &[op], &mut l, dxp)?;`

**Fix 4-4.** Anchor (unique, `:352`): `    bake(&dir, &ops, &layout, dxp)?;` → replacement
`    bake(&dir, &ops, &mut layout, dxp)?;`

**Fix 4-5.** Anchor (unique, `:554-555`):
```rust
        bake(&at("16"), &ops16, &l16, &dxp)?;
        bake(&at("8"), &ops8, &l8, &dxp)?;
```
Replacement:
```rust
        bake(&at("16"), &ops16, &mut l16, &dxp)?;
        bake(&at("8"), &ops8, &mut l8, &dxp)?;
```
(`l16`/`l8` and `layout` are already `mut` from fixes 3. Every later read of a layout — sizes, the
reserved placements — is unchanged by a larger segment: placements keep their offsets and sizes.)

**Gate:** 6g's build, clippy `-D warnings`, then the leg run, with its PASS/FAIL lines and `RC`
quoted whole.


## Read fix 1 (pod result 11-22: the write is exact, the read is guilty) — the probe REJECTED (no dumps); the widen's per-core start CONVICTED from source

### The conviction

**The widen's per-core START address is computed in a frame the landing is not in.**
- `per_core_addr` (`ktir-superdsc/src/emit/mod.rs:172-418`) folds each core's corner through `view_stick_layout` → `StickLayout::for_view_df` (`sdsc_abstract.rs:1620-1651`).
- For the landing's rank-4 `[mb, y, x, out]` view (`landing_arg`, `pointwise.rs:819-857`), that classifier returns `StickKind::Flat`: row-major over the LAYOUT order. So the per-core start strides `x` by the whole `out` width (128 elements) — `off = ((mb·Y + y)·X + x)·128 + out`.
- But the landing's bytes, and the walk dxp runs from the view's stated device order `["x","out","y","mb"]`, are `blocks` stick-major `[rows, cols]` blocks one after another (the frame `batch_blocks` states): within a block, `x` strides ONE stick (64 elements at fp16), and the second stick plane starts `rows·64` elements into the block.

The landing's own doc says why this held until now: "the frame changes the DECLARATION only — the
landing's per-core starts stay its blocks". The gather divides only its block axes
(`gather_copy_cores`), where Flat and the block law agree. The widen divides with
`work::distribute_cores` (`work.rs:446-499`), which splits `x`, and there the two laws part. **The
field:** the landing view's per-core start, `per_core_addr`'s `view_layout.off_view(..)` for a
`ViewFrame::Own`/`LandedBlocks` landing, has no address law of its own.

The fp8 INPUT side is right by accident: at one fp8 stick a row, Flat and stick-major are the same
bytes (`x` strides 128 either way). The write side (the fp16 landing) is wrong whenever `x` is split.

### The model reproduces every reported number

The model: core `c` of an x-chunk of `n` rows starting at `x0 = c·n` writes its row `x0 + k`, stick
`σ`, at element `A = Flat start + walk = x0·128 + σ·(rows·64) + k·64` within its block, where the true
place is `σ·(rows·64) + (x0 + k)·64`. An element at `A` lands in plane `A / 8192`, row `(A mod 8192)/64`
(rows = 128).

**Decode** (`[mb=8, y=1, x=128, out=128]`, one kv entry × 32 rows a core: `x0 = 32c`):

| core | its stick 0 lands on | its stick 1 lands on |
|---|---|---|
| c=0 | plane 0, rows 0..31 (right) | plane 1, rows 0..31 (right) |
| c=1 | plane 0, rows 64..95 | plane 1, rows 64..95 |
| c=2 | plane 1, rows 0..31 | the NEXT entry's plane 0, rows 0..31 |
| c=3 | plane 1, rows 64..95 | the next entry's plane 0, rows 64..95 |

- Plane 0 rows 32..63 and 96..127 of an entry are written by NO core. They keep the landing's boot zeros: **(ii) rows 63, 126, 127 read zero.**
- Plane 1 rows 0..31 are written by core 0 (right) and core 2 (row `64+k`'s stick 0). The read sees core 2's. The pattern gives rows `t` and `t+64` the same first half (sign parity `t+d+1`, mantissa `(7t+d+3) mod 8` with `7·64 ≡ 0 mod 8`, exponent independent of `t`). So row `k`'s stick 1 reads as its own stick 0: **(i) at s = 0, 1, 7, 8** (0xb180 / 0x3100 / 0x3200 / 0xb180, element 0's words, which differ from element 64's by one exponent step since 64 ≡ 1 mod 9).
- Row 64 (s=64): its plane 0 is written by core 1 with row 32's stick 0, the same first half (`7·32 ≡ 0 mod 8`), so it reads right. Its plane 1 is written by core 3 with row 96's stick 0: **again (i).**

**Prefill** (`[mb=1, y=8, x=128, out=128]`, `x` split 32, 4 rows a core: `x0 = 4c`): core `c`'s rows land
at rows `8c..8c+3`, both planes, within each block.
- Core 0 lands rows 0..3 where they belong: **query rows 0 and 1 (slots 0 and 2) fully right.**
- Rows 4..7 are written by no core: **(iii) query row 2 (slot 4) reads zero.**

Every reported row and word is reproduced, and no other reading of the source does.

**It is also a law violation:** the bytes a core writes depend on how the op is divided (the
gather's division is right, the widen's is wrong). The device form must not depend on the division.

### The fix — the landing addresses as its blocks, as a TYPE, for every division

A landing view carries its own address law: `PageLanding` (its blocks and its one block's stick-major
`StickLayout`), minted by `landing_arg` for EVERY landing. `per_core_addr` folds a landing's corner
through it.
- **Block-granular corners** (the gather's whole division) give exactly the Flat start, so every fp16 gather is byte-identical: `block · rows·cols`.
- **An `x`- or `out`-split corner** gets the block's stick-major offset (`StickLayout::dev_off`), the same law the walk reads.
- **The declaration is unchanged:** `PageLanding::declared` is the one block's arrangement exactly when the landing is one block several sticks wide (the condition `landing_arg` stated inline), and the view's own otherwise.

So the arrangement authority sees what it saw. Nothing is keyed to a division, and no core count is
forced.

**Edit RF1-1** — `OPSPEC`, the frame. Anchor (unique, `superdsc_opspec.rs:2273-2291`, from the line
`    /// ⭐ A page-gather LANDING of ONE block, several sticks wide: a stick-major \`[r, c]\` block`
through the enum's closing `}`):
```rust
    /// ⭐ A page-gather LANDING of ONE block, several sticks wide: a stick-major `[r, c]` block
    /// (each core's start at its own block, the view's own flat walk). The tensor's ONE
    /// arrangement is the stick-major `[r, c]` that block is — declared so, derived from the
    /// landing's own extents at the arrangement authority. A landing of SEVERAL blocks is NOT
    /// stated by this frame: its bytes are blocks-one-after-another, which no `StickKind` states
    /// (the walk carries that frame as `batch_blocks` / `BatchFrame::Blocks`), and stamping it
    /// `LandedBlocks` anyway would declare the batched transpose's read of the landing wrong —
    /// the arrangement authority refuses the conflict. So only a one-block landing several
    /// sticks wide carries this frame.
    /// ADDRESSING is not this frame's to change: the view keeps its own walk (a landing's per-core
    /// starts are its blocks), so it derives no layout of its own and stays classified as `Own`
    /// outside the authority's declaration.
    /// A one-stick landing is the flat frame's own bytes and never carries this frame.
    ///
    /// ⭐ IT CARRIES THE ARRANGEMENT IT DECLARES — the stick-major `[rows, cols]` of the one block,
    /// minted at the stamp from the landing's own extents. The authority reads it; nothing re-derives
    /// it from the view at emit, so the stamp and the declaration are one fact.
    LandedBlocks(crate::sdsc_abstract::StickLayout),
}
```
Replacement:
```rust
    /// ⭐ A PAGE LANDING — `blocks` stick-major `[rows, cols]` blocks one after another over the
    /// view's `[mb, y, x, out]` (the frame a page gather writes and the batched contraction reads,
    /// `batch_blocks` / `BatchFrame::Blocks`). It ADDRESSES as those blocks ([`PageLanding::off`]):
    /// a core's start is its corner's block, then that block's stick-major `(x, out)`, whatever axes
    /// the op divides — the start and the walk read one frame. Its DECLARATION is
    /// [`PageLanding::declared`]: the one block's stick-major arrangement when the landing is one block
    /// several sticks wide, and the view's own otherwise (several blocks are bytes no `StickKind`
    /// states, and stamping them would declare the batched transpose's read wrong — the arrangement
    /// authority refuses the conflict).
    Landing(PageLanding),
}

/// ⭐ A PAGE LANDING'S OWN FRAME — how many blocks, and the one block's stick-major `[rows, cols]` at
/// the landing's format. Minted by the landing's one stamp (`landing_arg`); the address law and the
/// declaration are both read off it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PageLanding {
    blocks: u32,
    block: crate::sdsc_abstract::StickLayout,
}

impl PageLanding {
    /// `blocks` blocks, each the stick-major `[rows, cols]` at `df`.
    pub fn of(blocks: u32, rows: usize, cols: usize, df: Df) -> PageLanding {
        PageLanding {
            blocks,
            block: crate::sdsc_abstract::StickLayout::row_blocked_df(rows, cols, df),
        }
    }

    /// The arrangement the landing declares to the authority: its one block's, when it is ONE block
    /// several sticks wide; otherwise none of its own (the view's).
    pub fn declared(self) -> Option<crate::sdsc_abstract::StickLayout> {
        (self.blocks == 1 && self.block.cols > self.block.lanes()).then_some(self.block)
    }

    /// The element offset of row `row`, column `col` of block `block`: the blocks one after another,
    /// each stick-major ([`crate::sdsc_abstract::StickLayout::dev_off`]).
    pub fn off(self, block: usize, row: usize, col: usize) -> usize {
        block * self.block.rows * self.block.cols + self.block.dev_off(row, col)
    }
}
```

**Edit RF1-2** — `EMIT`, the declaration. Anchor (unique, `emit/mod.rs:3937-3950`, from the comment
line `                    // ⭐ THE LANDING'S OWN FRAME — a page-gather landing of ONE block, several`
through `                    crate::superdsc_opspec::ViewFrame::LandedBlocks(landed) => landed,`)
→ replacement:
```rust
                    // ⭐ A PAGE LANDING declares what its frame states ([`PageLanding::declared`]):
                    // the one block's stick-major `[rows, cols]` when it is ONE block several sticks
                    // wide, and the view's own otherwise — several blocks are bytes no `StickKind`
                    // states, and declaring them stick-major would contradict the batched
                    // transpose's read of the same tensor, which the arrangement authority refuses.
                    crate::superdsc_opspec::ViewFrame::Landing(landing) => {
                        landing.declared().unwrap_or(own)
                    }
```
(The typed path `crate::superdsc_opspec::PageLanding` is in scope there through the doc link only;
the arm names the variant.)

**Edit RF1-3** — `EMIT`, the per-core start. Anchor (unique, in `per_core_addr`, `emit/mod.rs:410-412`):
```rust
        } else if view.frame == crate::superdsc_opspec::ViewFrame::FoldedRows {
            folded_rows_off(view, &extents, &corner, view_layout.lanes()) as u64
        } else {
```
Replacement:
```rust
        } else if view.frame == crate::superdsc_opspec::ViewFrame::FoldedRows {
            folded_rows_off(view, &extents, &corner, view_layout.lanes()) as u64
        } else if let crate::superdsc_opspec::ViewFrame::Landing(landing) = view.frame {
            landing_off(view, &extents, &corner, landing)? as u64
        } else {
```

**Edit RF1-4** — `EMIT`. Anchor (unique, `emit/mod.rs:426`):
```rust
/// A [`crate::superdsc_opspec::ViewFrame::FoldedRows`] view's element offset of `corner`: its non-stick
```
Replacement:
```rust
/// A [`crate::superdsc_opspec::ViewFrame::Landing`] view's element offset of `corner` over its
/// `[mb, y, x, out]` (the landing stamp's layout): block `mb·Y + y`, row `x`, column `out`
/// ([`crate::superdsc_opspec::PageLanding::off`]). A landing view of any other rank is refused by
/// name — the stamp mints only the four-axis view.
fn landing_off(
    view: &ArgView<'_>,
    extents: &crate::sdsc_abstract::DeviceExtents,
    corner: &[usize],
    landing: crate::superdsc_opspec::PageLanding,
) -> Result<usize, SuperDscError> {
    let ([mb, y, x, out], [_, y_ext, _, _]) = (corner, extents.dims()) else {
        return Err(SuperDscError(format!(
            "'{}': a page landing view is `[mb, y, x, out]`, and this one states {:?}",
            view.name, view.layout
        )));
    };
    Ok(landing.off(mb * y_ext + y, *x, *out))
}

/// A [`crate::superdsc_opspec::ViewFrame::FoldedRows`] view's element offset of `corner`: its non-stick
```
(`corner` and `extents.dims()` are `&[usize]`; the slice patterns `[mb, y, x, out]` and
`[_, y_ext, _, _]` bind references, so `mb * y_ext + y` is `&usize` arithmetic, which `usize`
implements.)

**Edit RF1-5** — `PW`, the stamp. Anchor (unique, `pointwise.rs:828-839`):
```rust
    let layout = ["mb", "y", "x", "out"];
    let frame = if landed_blocks == 1 && cols.elems() > DF::ELEMS_PER_STICK {
        crate::superdsc_opspec::ViewFrame::LandedBlocks(
            crate::sdsc_abstract::StickLayout::row_blocked_df(
                rows.get() as usize,
                cols.elems() as usize,
                DF::DF,
            ),
        )
    } else {
        crate::superdsc_opspec::ViewFrame::Own
    };
```
Replacement:
```rust
    let layout = ["mb", "y", "x", "out"];
    let frame = crate::superdsc_opspec::ViewFrame::Landing(crate::superdsc_opspec::PageLanding::of(
        landed_blocks,
        rows.get() as usize,
        cols.elems() as usize,
        DF::DF,
    ));
```

**Edit RF1-6** — `PW`, `landing_arg`'s doc (P11). Anchor (unique, `pointwise.rs:812-818`):
```rust
/// — so the 16-bit landing's arrangement, and every reader's, is the gather's own. A landing of ONE
/// block several sticks wide declares that block's stick-major `[rows, cols]`
/// (`ViewFrame::LandedBlocks`); a landing of several keeps `ViewFrame::Own` — its bytes are
/// blocks-one-after-another, which no stick layout states, the frame the walk carries as
/// `batch_blocks` / `BatchFrame::Blocks`; stamping it would declare the batched transpose's read of
/// it wrong, and the arrangement authority would refuse the conflict. The frame changes the
/// DECLARATION only — the landing's per-core starts stay its blocks.
```
Replacement:
```rust
/// — so the 16-bit landing's arrangement, and every reader's, is the gather's own. Every landing
/// carries its own frame (`ViewFrame::Landing`, [`crate::superdsc_opspec::PageLanding`]): it
/// ADDRESSES as its blocks under any division of `[mb, y, x, out]` (the gather divides its blocks,
/// the widen its rows too), and it DECLARES the one block's stick-major `[rows, cols]` exactly when
/// it is one block several sticks wide — several blocks are bytes no stick layout states, the frame
/// the walk carries as `batch_blocks` / `BatchFrame::Blocks`.
```

**Edit RF1-7 — the law's fixture.** It is in-tree, it is a law over a runtime corner, and it states the
division-independence. `backends/spyre/crates/ktir/ktir-superdsc/tests/zz_the_landing_frame_is_the_reshape_law.rs`,
appended at the end:
```rust
/// ⭐ A LANDING ADDRESSES AS ITS BLOCKS UNDER EVERY DIVISION — the start of `(block, row, col)` is
/// the block, then the block's stick-major `(row, col)`: a block-granular corner is the block's base
/// (the flat start the gather's division always gave), and a row- or column-split corner lands
/// inside the block's stick planes, never at a flat `row·cols`. granite's decode page: 8 blocks of
/// `[128, 128]` fp16 (two stick planes of `128·64`), and the fp8 page (one stick a row).
#[test]
fn a_page_landing_addresses_as_its_blocks_under_every_division() {
    use ktir_superdsc::superdsc_opspec::{Df, PageLanding};
    let f16 = PageLanding::of(8, 128, 128, Df::Fp16);
    assert_eq!(f16.off(3, 0, 0), 3 * 128 * 128, "a block-granular corner is the block's base");
    assert_eq!(f16.off(0, 32, 0), 32 * 64, "row 32, stick 0: one stick a row");
    assert_eq!(f16.off(0, 32, 64), 128 * 64 + 32 * 64, "row 32, stick 1: the second plane");
    assert_eq!(f16.off(0, 4, 0), 4 * 64, "the prefill core-1 corner");
    assert_eq!(f16.declared(), None, "several blocks declare none of their own");
    let fp8 = PageLanding::of(8, 128, 128, Df::Fp8);
    assert_eq!(fp8.off(0, 32, 0), 32 * 128, "one fp8 stick a row: row-major");
    let one = PageLanding::of(1, 128, 128, Df::Fp16);
    assert!(one.declared().is_some(), "one block several sticks wide declares its block");
    assert_eq!(PageLanding::of(1, 128, 64, Df::Fp16).declared(), None, "one stick declares none");
}
```

**Gate:**
- `cd backends/spyre && cargo test -p ktir-superdsc` (the new fixture and every landing/gather test), `cargo test -p scratchy-triton-splice`.
- Both workspace clippies.
- The kv/fp8 spyre-hw build (`BUILD_RC=0`).
- The card leg (Step 6) — expected: phases 1, 2c, 2 and 3 all PASS.
- Then `spyre chat` on nickm's card (short prompts and the 4711 recall, decode and a batch rung), with reply text and rc quoted: the verdict.

**Step purity.**
- P3: the landing's address law and its declaration are stated once, on `PageLanding`, read by `per_core_addr` and the authority; `landing_arg`'s inline declaration condition moves into `declared`.
- P4: a non-4-axis landing view is refused by name at bake.
- P5: `PageLanding` has private fields and one constructor.
- P7: there is no division-, model- or shape-keyed branch.
- P8: emitter only.
- P10: the Flat start for a landing is replaced, not duplicated.
- P11: both stale docs ("the per-core starts stay its blocks") are rewritten.
- P12: a fixture over the law, including the reported corners.
- Nothing dumps; the leg is unchanged.

---

## Purity — Part 2 as a whole

- **P1/P2:** every fact is a bake fact. `FMAX` is a kernel constexpr; the clip bounds are descriptor
  constants built at bake; the pool format is `KvFormat`.
- **P3:** each law is stated once. The run law is `run_extent`. A stored row's source is `param_rows`.
  The family's shape is `tiles_the_view`. A clip's bounds are its `OpInfo`. The convert's stick is
  `convert_stick_df`. The landing's tensor is `landing_arg`. The page's format comes from one
  `indirect_view_elem` match. The R2 copies this replaces (the per-head band converts, `NarrowRun`'s
  own coalescing loop, the `max`/`min` registry splats) are not typed.
- **P4:** the plan adds no runtime check. A form a law does not admit is a named bake error:
  - a chain that does not close is left unclaimed and refused by name at its `truncf`;
  - a page format other than f16/fp8 is refused by name;
  - a clip without bounds cannot be built.
- **P5/P6:** `ClipBounds`, `StickExtent<Fp8>`, `LandingFrame<DF>`, `IndexSelectPage<DF>`,
  `RunCopy<DF>`, and `PageFormat` over the format type.
- **P7:** nothing names a model, head count or size; the fold is over op kinds and identity.
- **P8:** opcode lowering only:
  - one `OpInfo` row and one door for `clip`;
  - one `DeviceOp` row and `OpFunc` arm for `fp8todl16` (Part 1);
  - the fold is an emitter pattern, a (chain → device ops) row, not a pass after lowering.
- **P11:** no `#[allow]` (`ClipOp` keeps `assemble_clip` at three parameters), and every comment is in
  the present tense.
- **P12:**
  - the fold has its match, three near-miss and neighbour fixtures;
  - the frames have R2 4e;
  - the card leg is Step 6 (typed; Q-C1′ and Q-C2 ruled).

## Open questions (Part 2)

- **Q-R1 — RULED 11-22: no re-layout of the key; its own `[T,1024]` tensor stands.** (Was: FACTS UNKNOWN, W4.) the dim order of torch-spyre's `qfp8ch` input over the fused-QKV
  `[T,8,128]` view (FACTS B.1). Scratchy's key is its own `[T,1024]` tensor in torch-spyre's default
  2-D layout. The design does not depend on the answer unless the lead wants the key itself
  re-laid as the reference's view is.
- **Q-R2 — RULED 11-22: the unfused form stands (`identity` + index, then `fp8todl16`; scratchy's gather's index form is the card-proven one).** (Was: FACTS UNKNOWN, R1.) whether Inductor fuses the page gather into `fp8todl16` as ONE
  `PointwiseOp("fp8todl16")` with the index prepended (TS `spyre_kernel.py:1527-1553`). The plan emits
  the unfused form, `identity` + index then `fp8todl16`: the only one whose index form scratchy's
  card-proven gather already states. If torch-spyre fuses, R1 + R2 become one op there.
- **Q-R3 — RULED 11-22:** the one-sided clip's other bound is DLFLOAT16_MAX through scratchy's existing
  `sen169_bits` (saturating to `0x7DFF`), one source, no new encoder; torch-spyre's own word is UNKNOWN;
  inert (both exceed every fp16 value).
- **Q-R4 — RULED 11-22: keep the 4-D iteration over the landing's own `[entries, blocks, rows, w]`, consistent with the dim-order ruling.** (Was: R2 iteration rank.) the widen iterates the landing's own 4-D `[entries, blocks, rows, w]`,
  so that it declares main's fp16 landing exactly, where torch-spyre iterates the 3-D `to` dims. Is
  the 3-D form wanted? If so, it must declare the same landing arrangement, which `landing_arg` states
  4-D today.
- **Q-C2 — RULED 11-22:** the V half is the round-trip proof for both pools. **Q-C1′ — RULED 11-22:** route (a′), the production programs in bundles of their own, through `bundle_run`'s file I/O (a test's I/O, not a dump); no new runtime API. Step 6 is typed.
- **Carried from FACTS, not design-blocking:** the device sizes of the write's two clip intermediates
  and of the fp8 key in torch-spyre are UNKNOWN (FACTS B.1 "Granite write (derived)"). Scratchy's are
  `[T,1024]` fp16 ×2 and `[T,1024]` fp8, minted per write program.
