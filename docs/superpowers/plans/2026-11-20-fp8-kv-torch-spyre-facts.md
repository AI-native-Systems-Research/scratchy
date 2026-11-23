# fp8 (float8_e4m3fn) KV cache on Spyre: what spyre-inference + torch-spyre do, as read in source

This file holds facts only. Each bullet is either read in source (cited `file:line`) or
marked **UNKNOWN**. Bullets marked **(derived)** are arithmetic over cited facts and
nothing else. It contains no design and no recommendations.

Sources and the commits read:

| tag | tree | HEAD |
|---|---|---|
| SI | `~/git/spyre-inference` | `4a7d21df4627eca90b7e4e57a596d3ea82d61119` |
| TS | `~/git/torch-spyre` | `f98503336881e3d5bdb328b6e40d725b32c4aade` |
| DT | `~/git/deeptools` | `3ce10beae47aa62ca90d737525753a5050aa80e2` |
| SC | `/Users/nickm/git/scratchy/.claude/worktrees/spyre-kv-fp8` | `48c86605aec2c39492fa34092a2a4888c4af76ae` (main) |

Paths below are relative to each tree's root: SI paths start at `spyre_inference/` or
`tests/`, TS paths at `torch_spyre/` or `tests/`, DT paths at `ddc/`, and SC paths at
`backends/`. Nothing under `~/tmp/dt_src` was read.

Granite-3.1-8b geometry used for the shapes below: `hidden_size` 4096,
`num_attention_heads` 32, `num_key_value_heads` 8 (SC
`crates/models/arch/configs/granite/granite-3.1-8b-instruct.json:12,19,21`), so head_dim =
4096/32 = 128 **(derived)**; qpk = 32/8 = 4 **(derived)**. Block size: SI
`platform.py:181` sets `_DEFAULT_BLOCK_SIZE = 128`, applied when the user did not set one
(`platform.py:661-668`). A user block size is rounded up to a multiple of
`_BLOCK_SIZE_MULTIPLE = 64` (`platform.py:180,670-680`) and must be a power of two
(`platform.py:685-689`). The SI on-card fp8 attention test also runs at `block_size=128`, 32 q
heads, 8 kv heads, head 128 (`tests/attention/test_spyre_attn.py:166-168,2283`). The
checkpoint's `torch_dtype` is bfloat16 (SC config `:28`), but the SI comment says "the
platform sets float16 for every model" (SI `v1/attention/backends/spyre_attn.py:1301-1302`).
The model dtype is therefore taken as fp16. Below, `T` is the token count, `NB` is
`num_blocks`, and `P = NB*8` is the folded page-row count.

---

## A. The reference's op sequence (spyre-inference)

### A.0 Eligibility, scales, compile mode
- An fp8 KV cache is accepted only for `kv_cache_dtype in ("fp8", "fp8_e4m3")`
  (`FP8_KV_CACHE_DTYPES`, SI `spyre_attn.py:243-244`, `:1424-1425`). The comment at `:243` says
  the storage is "float8_e4m3fn with one per-tensor (k_scale, v_scale) pair per layer".
- An fp8 cache requires `head_size % 128 == 0`; the comment is "A float8 stick holds 128
  elements" (SI `spyre_attn.py:1434-1439`). Test: `tests/attention/test_spyre_attn.py:1782-1793`.
- Scales are read as Python floats `layer._k_scale_float` / `_v_scale_float`, default 1.0. The
  docstring says "a [1] tensor fails torch-spyre's pointwise layout pass" (SI
  `spyre_attn.py:1993-2002`).
- Write scales `(k_scale, v_scale)` are passed to the store (`spyre_attn.py:1913-1920`,
  `:2022-2023`).
- Read side: `k_scale` is folded into the softmax scale (`scale * k_scale`), and `v_scale` goes
  to the kernels (`spyre_attn.py:1922-1928`).
- The store kernel is always `torch.compile`d (`spyre_attn.py:1459-1461`). The comment there
  says eager `index_copy_` rejects an int32 index and falls back to CPU with an int64 one.
- The attention kernels always run compiled on Spyre (`spyre_attn.py:1370-1374`,
  `:1482-1491`).

### A.1 Pool allocation
- `allocate_pages` does the following (SI `spyre_attn.py:1893-1911`):
  1. `dtype = torch.float8_e4m3fn if spec.dtype == torch.uint8 else spec.dtype`. Upstream
     sizes an fp8 cache as uint8 (`:1898-1899`).
  2. `layout = head_major_kv_layout(num_blocks * num_kv_heads, block_size, head_size, dtype)`
     (`:1900-1902`).
  3. Two tensors (K, V), each
     `torch.empty((num_blocks, num_kv_heads, block_size, head_size), dtype=dtype, device=device, device_layout=layout)`
     (`:1903-1907`).
  4. Zero fill through a byte view: `p.view(torch.uint8).zero_()`, with the comment "the device
     fill has no fp8 pattern" (`:1908-1910`).
- `head_major_kv_layout(num_pages, block_size, head_size, dtype)` (SI
  `v1/attention/ops/layout.py:43-72`):
  - `eps = get_elem_in_stick(dtype)`, `sticks = ceil(head_size/eps)` (`:60-61`);
  - `device_size=[num_pages, block_size, sticks, eps]` (`:63`);
  - `stride_map=[block_size*head_size, head_size, eps, 1]` (`:64`);
  - `device_dtype=get_device_dtype(dtype)` (`:65`);
  - `valid_elements={(0,): num_pages, (1,): block_size, (2, 3): head_size}` (`:66`);
  - `element_arrangement = ElementArrangement.QFP8CH if dtype == torch.float8_e4m3fn else STANDARD` (`:67-71`);
  - the docstring says "A float8 cache is tagged QFP8CH, the in-stick order `spyre.qfp8ch`
    writes and the fp8 -> fp16 read expects" (`:50-51`), and the indexed (page, kv_head) axis
    stays whole at device position 0 (`:44-48`).
- Granite fp8 pool, one of K or V **(derived from layout.py:60-71 with eps=128)**:
  `device_size=[P, 128, 1, 128]`, `stride_map=[16384, 128, 128, 1]`, `device_dtype=SEN143_FP8`,
  `valid_elements={(0,):P, (1,):128, (2,3):128}`, `element_arrangement=QFP8CH`. That is
  `P*16384` bytes. The fp16 pool for comparison is `[P,128,2,64]`, STANDARD.
  - eps=128 for fp8 is cited from SI `spyre_attn.py:1435`, TS
    `_inductor/propagate_layouts.py:497,891`, and SC `superdsc_opspec.rs:286`.
  - `SEN143_FP8` is cited from TS `csrc/types_mapping.h:37,82`: `kFloat8_e4m3fn` maps to
    `"fp8_143"` and then to `{SEN143_FP8, SEN143_FP8}`.
- Views made once, outside any graph:
  - `kv_slot_views`: `k_pages.view(-1, head_size)`, i.e. `[NB*8*128, 128]` rows (SI
    `spyre_attn.py:1946-1956`). The docstring says Inductor cannot lower a store through a view
    of a Spyre-layout tensor created inside a graph.
  - `_folded_pages`: `view(NB*8, 128, 128)`, "free under this layout" (`spyre_attn.py:1958-1963`).

### A.2 Write (store), in order
The entry is `do_kv_cache_update` → `self._reshape_fn(key, value, k_rows, v_rows, slot_mapping, k_scale=..., v_scale=...)`
(SI `spyre_attn.py:2004-2026`). `_reshape_fn` is `torch.compile(reshape_and_cache_kernel)`.

- Row index (host-side): head `h` of the token at `block*block_size + offset` goes to row
  `(block*num_kv_heads + h)*block_size + offset`. This is one index tensor per kv head, not one
  `[KV, T]` tensor (torch-spyre#3851) (`spyre_attn.py:1930-1944`). The kernel docstring calls
  each one "[T] int64" (`ops/reshape_and_cache.py:35-36`).
- `reshape_and_cache_kernel`, fp8 branch (`k_scale is not None`) (SI `ops/reshape_and_cache.py:32-52`):
  1. `key = quantize_kv(key, k_scale, k_rows.dtype)` and `value = quantize_kv(value, v_scale, v_rows.dtype)` (`:48-49`).
     The comment says quantizing is an elementwise pass that also materializes the strided
     fused-QKV view (`:40-43`).
  2. For each `h`, `idx` in `row_index`: `k_rows.index_copy_(0, idx, key[:, h])` and
     `v_rows.index_copy_(0, idx, value[:, h])` (`:50-52`). That is 8 K scatters and 8 V
     scatters for granite.
- `quantize_kv(x, scale, dtype)` (SI `ops/reshape_and_cache.py:20-29`):
  1. `fmax = torch.finfo(dtype).max` (`:23`). This is 448 for e4m3fn; the test asserts
     `[1e4,-1e4,3.0]/0.05` stores `[448,-448,60]` (`tests/attention/test_spyre_attn.py:1830-1834`).
  2. `x = x / scale` only `if scale != 1.0` (`:24-25`).
  3. `x = torch.clamp(x, -fmax, fmax)` (`:26`).
  4. On a spyre device: `return torch.ops.spyre.qfp8ch(x)` (`:27-28`). Otherwise `x.to(dtype)` (`:29`).
  5. The docstring says "`spyre.qfp8ch` on Spyre, where `.to` falls back to CPU and writes
     bytes in the wrong in-stick order" (`:21-22`). See B.2.c.
- Granite shapes **(derived from the cited code)**:

  | step | shape | dtype |
  |---|---|---|
  | `key`/`value` in | `[T, 8, 128]` | fp16 |
  | `/scale` (if ≠1) | `[T, 8, 128]` | fp16 |
  | clamp ±448 | `[T, 8, 128]` | fp16 |
  | `qfp8ch` | `[T, 8, 128]` | float8_e4m3fn |
  | `key[:, h]` | `[T, 128]` | float8_e4m3fn |
  | `idx` | `[T]` | int64 per the docstring (`reshape_and_cache.py:35-36`); its device dtype is **UNKNOWN** (it goes through `convert(...)`, `spyre_attn.py:1943`, not read) |
  | `k_rows` | `[NB*8*128, 128]` | float8_e4m3fn (view of the pool) |

  `T` is the scheduled token count; the caller's bucketing is not traced here (**UNKNOWN**).

### A.3 Read: per-sequence decode (`page_attn_decode_kernel`, SI `ops/page_attn_decode.py:31-148`)
- `q = query.index_select(0, query_row_index).reshape(num_kv_heads, num_queries_per_kv, head_size)` (`:81`). Granite: `[8,4,128]` fp16.
- Per page tile (`walk_tiles`, `:131-141`):
  1. `kv_rows = kv_row_pool.index_select(0, page_index[0, 0:1])` (`:88`). `kv_row_pool` is
     `[num_pages, num_kv_heads, 1]` int32, holding `page*KV + kv` (`spyre_attn.py:1965-1991`),
     uploaded with `temporary_chunk_major_page_index_layout` (`ops/layout.py:25-40`). Result `[1, 8, 1]`.
  2. `k_page = k_pages[kv_rows].reshape(num_kv_heads, block_size, head_size)`; the same for `v_page` (`:92-93`).
     `k_pages` is the folded `[P, 128, 128]` fp8 tensor. Result `[8,128,128]` fp8. The comment
     says this is subscripting, not `index_select`, so the split lands per kv head (`:89-91`).
  3. `if k_page.dtype != q.dtype: k_page = k_page.to(q.dtype); v_page = v_page.to(q.dtype)` (`:94-96`). Result `[8,128,128]` fp16.
  4. `scores = torch.matmul(q, k_page.permute(0, 2, 1)) * scale` (`:98`): `[8,4,128] @ [8,128,128]` → `[8,4,128]`.
     `scale` already carries `k_scale` (A.0).
  5. Mask add, amax, exp, sum, and `torch.matmul(probs, v_page)` (`:105-127`): `[8,4,128] @ [8,128,128]` → `[8,4,128]`.
- After the walk: `attn = normalize(...).reshape(1, num_heads, head_size)` (`:142`); then
  `if v_scale is not None and v_scale != 1.0: attn = attn * v_scale` (`:143-144`); then
  `out.index_copy_(0, query_row_index, attn)` (`:145-147`).

### A.4 Read: prefill (`page_attn_prefill_kernel`, SI `ops/page_attn_prefill.py:29-171`)
- `q` is `query.index_select(0, query_row_index)` → `unsqueeze(0).transpose(1,2).reshape(KV, qpk, Q, D)` (`:73-78`),
  plus `unsqueeze(0)` when grouped (`:79-80`). `_PAGE_GROUP = 8` (SI `spyre_attn.py:248`).
- Page gather on the unfolded `[NB, 8, 128, 128]` pool, using `index_select` (not
  subscripting; `:93-95`):
  - grouped: `k_pages.index_select(0, page_index[:, 0]).unsqueeze(2)` → `[G,8,1,128,128]` (`:96-101`);
  - ungrouped: `k_pages.index_select(0, page_index[0, 0:1]).squeeze(0).unsqueeze(1)` → `[8,1,128,128]` (`:103-106`).
- `if k_page.dtype != q.dtype: k_page = k_page.to(q.dtype)` and the same for V (`:108-110`).
- `scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale` (`:112`); `torch.matmul(tile_probs, v_page)` (`:126,136`).
- `attn * v_scale` when `v_scale` is not None and not 1.0 (`:164-165`); then
  `out.index_copy_(0, query_row_index, attn)` (`:166-170`).

### A.5 Read: batched decode (`batched_decode_kernel`, SI `ops/batched_decode.py:97-264`)
- `q = query.index_select(0, rep_row_ids).reshape(entries, KV, qpk, D)` (`:143-145`). Granite: `[E,8,4,128]`.
- `k_page = k_pages[page_ids].reshape(entries, num_kv_heads, block_size, head_size)`; the same for V (`:169-170`).
  This is a subscript on the unfolded `[NB,8,128,128]` pool, giving `[E,8,128,128]` fp8.
- `.to(q.dtype)` under the same dtype test (`:171-173`). The docstring says "A float8 page is
  cast to the query dtype after its gather and v_scale scales the output" (`:130-132`).
- `scores = torch.matmul(q, k_page.transpose(-2, -1)) * scale` (`:174`) → `[E,8,4,128]`;
  `torch.matmul(probs.reshape(E,KV,qpk,block), v_page)` (`:149-157`, `:49-52`).
- `attn * v_scale` when not None and not 1.0 (`:257-258`).

### A.6 What the reference does NOT do (read)
- No fp8 matmul on the KV path. Every attention matmul runs after the page is widened by
  `.to(q.dtype)` (A.3 step 3, A.4, A.5).
- No per-head or per-token scale: one float pair per layer (A.0).

---

## B. How torch-spyre lowers each op

### B.1 `torch.ops.spyre.qfp8ch`
- **Registration.** `@torch.library.custom_op("spyre::qfp8ch", mutates_args=(), device_types="spyre")`.
  The body is `pass`. The docstring says "Maps to: deeptools Qfp8ch operation" and "ONLY
  performs format conversion - scaling must be done separately" (TS `_inductor/customops.py:844-861`).
  - Fake: `torch.empty(input.size(), dtype=torch.float8_e4m3fn)` (`customops.py:864-867`).
  - The eager body is `pass` (returns None), so the op only works compiled. SI's store is
    always compiled (A.0).
- **Lowering.** `@register_spyre_lowering(torch.ops.spyre.qfp8ch)`: `Pointwise.create(dtype=torch.float8_e4m3fn, inner_fn=ops.qfp8ch(x_loader(index)), ranges=x.get_size())`, then `pw.realize()` (TS `_inductor/lowering.py:2969-2992`).
- **Kernel op.** `SpyreOpFuncs.qfp8ch(x) → PointwiseOp("qfp8ch", [x])` (TS `_inductor/spyre_kernel.py:340-341`).
  `"qfp8ch"` is listed in `SPYRE_FP8_OPS` (TS `_inductor/constants.py:221-226`).
- **SDSC opfunc.** `_get_op_func` returns the op name unchanged for a non-reduction, so the
  opfunc is `"qfp8ch"` (TS `_inductor/codegen/superdsc.py:1893-1912`).
  - A conversion that changes elements per stick (the comment lists "DL16TOFP32, FP32TODL16,
    FP8TODL16, qfp8ch, qfp8wt") gets a virtual `mb=1` outer dim when the op has only the stick
    dim (`superdsc.py:2396-2414`).
  - Each arg's stick size is `arg.device_dtype.elems_per_stick()` (`superdsc.py:1789-1790`):
    64 for the fp16 input, 128 for the fp8 output.
- **DDL.** The only template whose `opFuncName` is `qfp8ch` is
  `ddc/ddl_templates/quantization_double_pad.ddl:55`. `grep -ln 'qfp8ch\|fp8todl16'` over
  `ddc/ddl_templates/*.ddl` returns only that file. See C.
- **Output layout.** In `propagate_layouts._single_arg_op_layout`, case
  `spyreop.qfp8ch.default` → `_qfp8ch_stl(stl, output.dtype, coords)` (TS
  `_inductor/propagate_layouts.py:890-903`). `_qfp8ch_stl` (`propagate_layouts.py:494-525`):
  - starts from the input's `device_size`/`stride_map`;
  - sets the last dim to `out_eps = get_elem_in_stick(fp8)`;
  - takes the "stick-count dim" `i` from `_qfp8ch_count_dim` (`:462-491`) and, when its stride
    is `in_eps`, rescales `device_size[i] = ceil(device_size[i]*in_eps/out_eps)` (rounded UP,
    issue #3604) and `stride_map[i] = out_eps` (`:515-519`);
  - returns `SpyreTensorLayout(out_device_size, out_stride_map, get_device_dtype(fp8), ElementArrangement.QFP8CH)` (`:520-525`).
  - So the output is tagged **QFP8CH**, with the same dim order as the input and 128-element sticks.
  - Unit test: a `[1,4,64]` fp16 STL becomes `[1,4,128]`, stride `[128,64,1]`, QFP8CH (TS
    `tests/tensor/test_tensor_layout.py:1068-1095`).
- **Granite write (derived).** The input is the clamp output `[T,8,128]` fp16 (2 fp16
  sticks per head row), and the output is `[T,8,128]` fp8 (1 fp8 stick per head row), QFP8CH.
  The exact `device_size` torch-spyre picks for the fp16 input (dim order) is **UNKNOWN**:
  it depends on propagation over the fused-QKV view, which was not traced.
- **What the QFP8CH in-stick order is:**
  - The enum comment is "non-sequential order produced by on-device fp8 channel quantization
    (qfp8ch)" (TS `csrc/spyre_tensor_impl.h:50-51`), value 2 (`:46-57`), exposed to Python at
    `csrc/module.cpp:311`.
  - **The permutation itself is not written down anywhere read.** torch-spyre has no table
    or formula for it. The KTIR emitter passes QFP8CH through as "no layout change"
    (`_inductor/codegen/ktir.py:720-726`) and also has no fp8 element type (`ktir.py:219-225`).
    The DDL states no index list for qfp8ch (C.1). The numeric QFP8CH permutation is **UNKNOWN**.
  - Only evidence about the permutation (tests): host-uploaded STANDARD fp8 bytes read by
    `fp8todl16`, and `qfp8ch` output read back by `.cpu()`, both agree with logical order for
    rows of K=8. For K≥16 both fail strict-xfail. The xfail reason is "for K>=16 a host FP8
    copy is STANDARD but the FP8 kernels read and write QFP8CH, so from the 9th element of a
    row on, values come from the wrong offset" (torch-spyre#3033; TS
    `tests/inductor/fp8/test_fp8_quantize_dequantize.py:48-54,146-168,320-333,346-373`).
    Elements 0..7 of a row therefore read the same in both orders; element 8 onward differ.
    The rest of the map is **UNKNOWN**.

### B.2 `.to(torch.float16)` on an fp8 tensor that holds QFP8CH data (read-side widen)
- **Lowering.** `prims.convert_element_type` → TS `lowering.to_dtype` (`_inductor/lowering.py:2220-2279`).
  `DtypeOpTable.is_supported(fp8, fp16)` is true, so it calls stock `lowering.to_dtype(...)`
  (`:2261-2279`).
- **Device op.** `DtypeOpTable._FP8_TO_FP16_DTYPES = [(torch.float8_e4m3fn, torch.float16)]` maps to `FP8TODL16_OP = "fp8todl16"`
  (TS `_inductor/dtype_ops.py:127-129,145`; `_inductor/constants.py:48`).
  `SpyreOpFuncs.to_dtype` → `PointwiseOp(DtypeOpTable.get_operator(src, dst), [x])`, i.e.
  `PointwiseOp("fp8todl16", [x])` (TS `_inductor/spyre_kernel.py:398-427`).
  - `(fp8, bf16)` has no row (`dtype_ops.py:107-148`). A bf16 widen would take the CPU fallback
    of B.2.c. SI runs fp16 (header).
  - The `dequantize_fp8_with_scale` decomposition uses the same `.to(fp16)`, with the comment
    "triggers fp8todl16 via dtype_ops" (TS `_inductor/decompositions.py:3812-3825`).
- **Input arrangement handling.** In `_single_arg_op_layout`, `convert_element_type` with differing
  elems-per-stick (`propagate_layouts.py:855-888`) does:
  - `input_ea = stl.element_arrangement` (QFP8CH);
  - `fmt = DtypeOpTable.ea_map(fp8, fp16, QFP8CH)`;
  - `_convert_layout(...)`.
  - `ea_map` has entries only for fp16/bf16↔fp32 (`dtype_ops.py:51-62,153-154`). Every other
    pair "default[s] to STANDARD" (`dtype_ops.py:223-236`), so the **output EA is STANDARD**.
  - `_convert_layout` (`propagate_layouts.py:669-748`): QFP8CH is not in `STAGGERED_EAS =
    {DL16_TO_FP32, FP32_TO_DL16}` (TS `_inductor/constants.py:54-71`), so the preserve path is
    skipped. It takes the "Plain conversions (e.g. fp8->fp16 after qfp8ch)" path (`:691-696`):
    a dense layout rebuilt from the output host size,
    `SpyreTensorLayout(c_size, c_stride, output.dtype, list(range(len(c_size))), fmt)`,
    with the last dim forced to the input stick depth if the host last dim is not a stick
    multiple (`:737-748`).
  - The doc says the same: "Plain conversions rebuild a fresh dense layout from the output host
    size" (TS `docs/source/compiler/layout_optimization.md:96-102`).
  - **No reorder op is inserted for QFP8CH.** The restickify feasibility check rejects every
    non-STANDARD input EA (`_inductor/pass_utils.py:3173-3179`), and SEN143_FP8 is not a
    restickify dtype (`RESTICKIFY_SUPPORTED_DEVICE_DTYPES = {SEN169_FP16, IEEE_FP32, IEEE_INT32, SENUINT32}`,
    `_inductor/constants.py:28-34`).
  - The SDSC codegen consults `element_arrangement` only for QFP8WT
    (`_inductor/codegen/superdsc.py:1785-1817`). `grep element_arrangement` over
    `_inductor/codegen/*.py`, excluding `ktir.py`, finds only `superdsc.py:1791`. **So the SDSC
    for `fp8todl16` carries no arrangement field. torch-spyre relies on the device op itself
    to consume QFP8CH order and emit STANDARD order.**
  - Comments stating that contract:
    - TS `_inductor/constants.py:108-110`: "QFP8CH tensors are consumed by an fp8 matmul or
      the fp8->fp16 convert";
    - TS `_inductor/propagate_layouts.py:503-505`: "the fp8->fp16 conversion that consumes it
      rebuilds a dense layout";
    - SI `ops/layout.py:50-51` (B.1).
  - Test evidence: on-device `qfp8ch` → `fp8todl16` round trips are exact (atol=rtol=0) at
    K=16…4096, which includes `[1,4,32,128]` attention-shaped inputs
    (`tests/inductor/fp8/test_fp8_quantize_dequantize.py:684-757,907-926,972-995`). The same K
    fail when the fp8 bytes came from the host in STANDARD order (B.1).
- **fp16 output layout (granite).** Dense STANDARD, rebuilt from the host size of the
  gathered page; for the decode page this is `[8,128,128]` fp16 (A.3). The concrete
  `device_size` is **UNKNOWN**: it depends on the host size/stride Inductor gives the
  gathered view, which was not traced.
- **Whether the gather and the widen fuse into one device op is UNKNOWN.**
  - If one Pointwise body holds both the indirect load and `to_dtype`, the store path emits a
    single `PointwiseOp("fp8todl16")` with the index tensor prepended as an extra arg
    (`spyre_kernel.py:1527-1553`).
  - If the gather is realized first, it is a separate `identity` (B.3) followed by
    `fp8todl16`.
  - Which one Inductor produces for SI's kernels was not determined without running the
    compiler.

#### B.2.c The WRITE-side `.to(fp8)` that SI avoids (what the comment means)
- `DtypeOpTable` has no `(fp16, fp8)` row: only `(fp8, fp16)` (`dtype_ops.py:127-148`). So
  `lowering.to_dtype` takes `eager_fallback(torch.ops.spyre.to_dtype_cpu, x, dst)`
  (`lowering.py:2261-2265`).
- `spyre::to_dtype_cpu` is `input.cpu().to(dtype=dtype).to(input.device)` (TS `_inductor/customops.py:833-836`).
  The fp8 bytes are therefore produced on the host in logical (STANDARD) order and uploaded
  by host DMA.
- Host DMA has a special data-conversion path only for QFP8WT (`csrc/spyre_mem.cpp:789-790`
  and following); nothing in it reorders for QFP8CH.
  - TS `tests/models/op_registry.py:307-308`: "batchmatmulfp8 reads its activation in the
    element order qfp8ch writes, which neither a STANDARD nor a QFP8CH host DMA reproduces".
  - TS `_inductor/lowering.py:2840-2843`: "the D2H copy-out ignores element_arrangement and
    reads the buffer as STANDARD".
- Result: a pool tagged QFP8CH would receive STANDARD-order bytes. The device `fp8todl16`
  reads them as QFP8CH, which gives wrong values from the 9th element of each row on
  (torch-spyre#3033, B.1). This is what SI's "writes bytes in the wrong in-stick order" refers
  to (SI `ops/reshape_and_cache.py:21-22`).

### B.3 `index_copy_` into the fp8 rows view, and the page gather
- **No torch-spyre-specific lowering for `index_copy`, `index_select`, or `aten.index`
  was found.** `grep index_copy|index_put` over `torch_spyre/**/*.py` finds only comments and
  the `index_add` decomposition. The `index_add` decomposition states the primitives:
  "`index_select` -> on-device indirect gather; `index_put` -> on-device indirect overwrite
  store" (TS `_inductor/decompositions.py:4139-4162`). An `index_copy` scatter is an Inductor
  `ir.Scatter` (TS `_inductor/wsr/while_loop_bridge.py:1141-1142`). The upstream PyTorch
  decomposition `index_copy_` → `index_put` was not read (**UNKNOWN** in exact form).
- **Device op for a pure gather or scatter.** In `SpyreKernel.store`, a `TensorAccess` value
  that uses indirect symbols has the index tensor arg(s) prepended (index first, then value
  source, then destination). The op is `RESTICKIFY_OP` if the coordinates say restickify,
  else `IDENTITY_OP` (TS `_inductor/spyre_kernel.py:1554-1610`). `IDENTITY_OP = "identity"`
  (`_inductor/constants.py:20`). The SDSC places "the index tensor first" (`superdsc.py:2845-2846`).
  - DDL: `identity` binds typeless tensors (`[]`), with the comment "identity and shuffle only
    move data, so their input and output are free to arrange it differently"
    (`ddc/ddl_templates/unary_parallel.ddl:16-19,44`).
  - So an fp8 scatter or gather is an `identity` over SEN143_FP8 bytes with no conversion.
    That it moves the stored bytes unchanged is **(derived)** from the op being a data mover.
- **The indexed dim must be outermost in the value tensor's device layout.**
  `enforce_indirect_access_layout` rewrites the producer's layout or inserts a
  `spyre.restickify` copy to make that hold (TS `_inductor/enforce_indirect_access_layout.py:15-28`).
  - For an fp8 source this restickify is unavailable: `_dense_scatter_source_stl` raises
    `Unsupported("... ReStickifyOpHBM does not support {dtype}")` for any dtype outside
    `RESTICKIFY_SUPPORTED_DEVICE_DTYPES` (`enforce_indirect_access_layout.py:213-234`;
    constants `:28-34`).
  - When it does build a dense source layout, it keeps the source's `element_arrangement`
    (`:228-234`).
- **Element arrangement across the scatter.**
  - `validate_ops` checks EA compatibility only across the multi-input reads of a Pointwise
    (`_inductor/split_multi_ops.py:748-806`). A scatter's reads are an int32 index (STANDARD)
    and the fp8 source (QFP8CH), and `is_ea_compatible({STANDARD, QFP8CH})` is True (one
    non-STANDARD EA, not EXX2; `_inductor/constants.py:74-116`).
  - No code read compares the source EA against the destination pool's EA. The destination's
    STL is the pool's own (`device_layout=` at allocation, A.1).
  - So the arrangement is **carried as bytes, not checked**: QFP8CH bytes from `qfp8ch` land in
    a pool tagged QFP8CH because both tags say QFP8CH. That nothing enforces this is a fact
    about the files read: the arrangement consult sites are listed by `grep element_arrangement`
    over `_inductor/**/*.py`, and none of them is on the scatter path.
- **Whole sticks (derived).**
  - Write: the `k_rows` view `[NB*8*128, 128]` of a pool whose device dims are
    `[P, 128, 1, 128]` (A.1). One scattered row is one `(page·kv, token)` row of 128 fp8 =
    exactly one 128-element stick.
  - Decode gather: one entry is a `(page·kv)` row of the folded pool, i.e. `[128 tokens, 1
    stick, 128]` = 128 whole sticks, at device dim 0 (SI `ops/layout.py:44-48`).
  - Prefill and batched-decode gathers: one entry is a page `[8, 128, 128]` of the unfolded
    pool = 8·128 whole sticks.
- **Index-tensor constraints SI works around** (each cited as SI's own statement):
  - a row of a 2-D index is followed for its first int32 stick only (torch-spyre#3851; SI
    `spyre_attn.py:1936-1938`);
  - selecting a whole source faults the device (torch-spyre#4033; `spyre_attn.py:1469-1471`);
  - a single index over a flattened `(T, KV)` source fails with `UnalignedStickSplit`
    (`ops/reshape_and_cache.py:36-37`);
  - eager `aten.index` takes only one 1-D index (torch-spyre#4334; `spyre_attn.py:1371-1373`;
    TS `ops/eager.py:774-800`).

### B.4 The pool zero fill
- `aten::zero_` on spyre → `fill_tensor(self, 0.0)` (TS `ops/eager.py:741-745`) →
  `spyre_fill_tensor`. That function launches a device-side `MEMORY_FILL` DMA over every stick
  of the composite address, with `dtype = get_device_dtype(self.scalar_type())`, through
  `flex::createFillAsyncParams` (TS `csrc/spyre_mem.cpp:1876-1960`). It rejects partial and
  broadcast views (`:1889-1940`).
- On the `uint8` view the device dtype is `SENUINT32`: `{"uint8", {SENUINT8, SENUINT32}}` (TS
  `csrc/types_mapping.h:77-79`).
- The bit pattern `createFillAsyncParams` produces for 0.0 at SENUINT32 is **UNKNOWN** (not
  read). How a same-size `view(torch.uint8)` of a SEN143_FP8 STL is sized for the fill is
  **UNKNOWN** (not read).

### B.5 The other write-side ops
- `x / scale` (Python float): `aten.div.*` → `_lower_div_impl` → `lowering.div(x, y)` for
  true division (TS `_inductor/lowering.py:2715-2734,2708-2712`) → `PointwiseOp("realdiv", ...)`
  (`spyre_kernel.py:429-431`). How the scalar is bound (constant vs tensor) is **UNKNOWN** (not traced).
- `torch.clamp(x, -448, 448)`: the comment says "aten.clamp decomposes upstream into
  clamp_min + clamp_max". Each is lowered by `lower_clamp` with the other bound at
  `±DLFLOAT16_MAX` (TS `_inductor/lowering.py:180-181,219-242,1505-1579`) →
  `PointwiseOp("clip", [x], {"constants": {"clipMin", "clipMax"}})` (`spyre_kernel.py:236-244`).
  - DDL `clip` binds fp16 in `unary_pipeline.ddl:16`, with constants `clipMin`/`clipMax` at
    `:32-33`.
  - The order of the two clips is **UNKNOWN** (upstream decomposition, not read).

---

## C. The device ops' own definitions (DT `ddc/ddl_templates/quantization_double_pad.ddl`)

### C.1 `qfp8ch`
- Bind: `%qfp8ch_op = ddl.operation_bind([], [%inp_tensor], [%output_fp8_tensor]) {opFuncName="qfp8ch"}` (`:55`).
  - `%inp_tensor` allows `[%type_fp16, %type_fp32, %type_fp8]` (`:42-43`), with
    `%type_fp16 = SEN169_FP16` (`:32`).
  - `%output_fp8_tensor` is `[%type_fp8]` = `SEN143_FP8, bit_width=8` (`:35,47`), with layout
    `%slice_layout_out_padded` / `%global_layout_padded` (`:23,27,47`).
- Exactly one of {csqint8ch, qfp8ch, csqint4, dl16tofp32, fp32todl16, fp8todl16} per instance
  (`:64`). Constraints `out_front == 0` and `out_back < 65` (`:65-66`).
- Dataflow:
  - qfp8ch is in `%peforward_ops` and `%peforwardpadding_ops` (`:99-100`). The input stick goes
    LX→SFP via PE (`:226-230`).
  - Pad: "QFP8: pad as sending a FP16-0s to SFP from PE"; pad-front and pad-back each issue two
    `pe_zero_lrf` transfers per padded OUT position (`:199-207,266-275`). One-transfer padding
    on the stick back is at `:243-249`.
  - Output loop: `ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="assign", unit="sfp", repetition=8}` (`:291-294`).
    Then, for `%chdowncast_ops` (which includes qfp8ch, `:97`), SFP→LX (`:300-304`).
- **The template states no SHUFFLE, no index list, and no element order for qfp8ch.** The
  in-stick order that `assign`/`repetition=8` produces from two 64-element fp16 input sticks
  into one 128-element fp8 stick is **UNKNOWN** from the DDL. That two fp16 input sticks feed
  one fp8 output stick is **(derived)** from the 64/128 stick widths and the two-transfer
  padding at `:204-205,272-273`.

### C.2 `fp8todl16`
- Bind: `%fp8todl16_op = ddl.operation_bind([], [%inp_tensor], [%output_fp16_tensor]) {opFuncName="fp8todl16"}` (`:60`).
  `%output_fp16_tensor` is `[%type_fp16]` = SEN169_FP16 (`:32,50`).
- It is in `%chupcast_ops` (`:98`) and `%peforward_ops` (`:100`).
- Output loop over `%padded_dim_out` of the input (`:306`), when `fp8todl16` (`:315-323`):
  - comment: "Casting: fp8 one stick (top, bot) -> (fp16 stick0) (fp16 stick1)" (`:316`);
  - the input stick goes PE → SFP LRF (`:317-320`);
  - `ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="SHUFFLE", unit="sfp", repetition=8, indices=[0,1,2,3,4,5,6,7]}` (`:321`);
  - `ddl.compute([%sfp_src00], [%sfp_dst00]) {computetype="SHUFFLE", unit="sfp", repetition=8, indices=[8,9,10,11,12,13,14,15]}` (`:322`);
  - then SFP→LX (`:324-328`).
- For comparison:
  - `dl16tofp32`: SHUFFLE `indices=[0,1,2,3]` then `[4,5,6,7]` (`:307-313`). torch-spyre tags
    its output DL16_TO_FP32 (`dtype_ops.py:57`).
  - `fp32todl16`: `PACKMERGE ... indices=[0..7]` (`:295-299`).
- **What a SHUFFLE `indices` entry denotes (which lanes or bytes it selects) is not defined in
  any file read.** The SFP SHUFFLE semantics live in the deeptools compiler/simulator, which
  was not read. `sys-arch-spec/arch_enums.h:249` `SHUFFLE` is the opfunc alias of IDENTITY, a
  different thing. So the exact output order of `fp8todl16` for a given input order is
  **UNKNOWN** from the DDL.
- Read-side fact from torch-spyre tests (B.2): `fp8todl16` applied to a `qfp8ch` output gives
  logical (STANDARD) order exactly.
- Other DT references read: `docs/dsl-driven-lowering-plan.md:272-273,463-464` lists
  `fp8todl16` as an "explicit conversion" and `qfp8ch` as "a whole kernel".
  `sys-arch-spec/arch_enums.h:183` lists `FP8TODL16` as an OpFuncs entry.

### C.3 `identity` (gather/scatter mover)
- `%identity_op = ddl.operation_bind([], [%inptensor_shuf], [%outtensor_shuf]) {opFuncName="identity"}`
  on typeless tensors (`ddc/ddl_templates/unary_parallel.ddl:16-19,44`). No dtype or
  arrangement is stated.

---

## D. scratchy today (main @48c86605)

Correspondences (scratchy file:line ↔ torch-spyre piece):
- **SEN143_FP8 data format** ↔ TS `types_mapping.h:82`:
  - `pub enum Fp8 {}` with `ELEMS_PER_STICK = FLIT_BYTES / 1` (=128), `NAME = "SEN143_FP8"`,
    `WORD_LENGTH = 1` (`spyre/crates/ktir/ktir-superdsc/src/superdsc_opspec.rs:285-303`);
  - `Df::Fp8` (`superdsc_opspec.rs:363-414`);
  - `assert_df_stick_multiple(..., Df::Fp8)` (`superdsc_opspec.rs:1414-1416`).
- **`qfp8ch` emission** ↔ TS B.1:
  - `OpFunc::Qfp8ch` with the doc citing `quantization_*.ddl` opFuncName "qfp8ch"
    (`superdsc_opspec.rs:1727-1734`), spelled `"qfp8ch"` (`:1919`), `ReductionBinding::None`
    (`:2108`);
  - the convert dtype pair `OpFunc::Qfp8ch => (Df::Fp16, Df::Fp8)` (`ktir-superdsc/src/emit/mod.rs:4892-4897`);
  - the string→op mapping `"qfp8ch" => OpFunc::Qfp8ch` (`emit/mod.rs:7090`).
  - Emitted by the W8A8 activation quantize as one `assemble_convert(&opn("fq_afp8_op"), "qfp8ch", m, k_pad, &rb(&cl, m, k_pad), &rb_df(&afp8, m, k_pad, Df::Fp8), ...)`
    (`emit/ktir_matmul_fp8.rs:517-536`), and per pair at the half-stick `inn` (`:537-580`).
  - Its input is the clamped activation. The chain is `quantscalepertokenfp8` → … → `clip` →
    `qfp8ch` (`ktir_matmul_fp8.rs:1-13,104-122,255-300`), and mirrors torch-spyre's
    `quantize_fp8_with_scale` (`superdsc_opspec.rs:1728-1729`; TS `decompositions.py:3934-3941`).
- **How the W8A8 path states the QFP8CH arrangement for `matmulfp8`: it does not.**
  - `matmulfp8` is built as the plain matmul opspec, then every non-output arg gets
    `arg.set_df(Df::Fp8)` (`ktir_matmul_fp8.rs:633-675`). No arrangement is attached to the
    `qfp8ch` output or to the `matmulfp8` input.
  - torch-spyre also emits no QFP8CH field into the SDSC (B.2). It tags QFP8CH only on its
    host-side `SpyreTensorLayout` (B.1).
- **Element arrangement as a type** ↔ TS `ElementArrangement` (`spyre_tensor_impl.h:46-57`):
  - `pub enum ElementArrangement { Standard, Dl16ToFp32 }` — two variants, **no QFP8CH**
    (`spyre/crates/ktir/ktir-core/src/ddl_binds.rs:79-99`);
  - per-device-op output arrangement rows in `DeviceOpRow.arrangement`
    (`ddl_binds.rs:252-275`), read by `output_arrangement` (`ddl_binds.rs:839-844`);
  - the measured `DL16TOFP32_PERMUTATION` table and its staging
    (`ktir-superdsc/src/element_arrangement.rs:1-80`). That is scratchy's own card measurement
    (probe-8, 2026-10-26) of what torch-spyre calls DL16_TO_FP32. torch-spyre states no
    numbers for it (TS `codegen/ktir.py:727-733`).
- **e4m3 encoder matching the card's `qfp8ch` byte values** (`ktir-core/src/codec.rs:1008-1021`;
  fixture `spyre/crates/spyre/tests/fp8_e4m3_codec.rs`, named at `codec.rs:1020`). This is
  values only; it states nothing about in-stick order.
- **Indirect gather/scatter as `identity` + index** ↔ TS B.3:
  - `attach_gather_index` and `attach_scatter_index` (`superdsc_opspec.rs:4385-4410`). The
    scatter doc cites the vendor fixture `dxp/test/test_scatter_1core/sdsc_1.json` (an
    `identity` with the index in the middle, `:4397-4403`).
  - The index format is `SenUint32` (`superdsc_opspec.rs:305-340`). Whether these doors have
    been driven with a `Df::Fp8` value operand is **UNKNOWN** (not found in the files read).

torch-spyre pieces with **no** scratchy counterpart found:
- **`fp8todl16`.** No `OpFunc`, no `DeviceOp`, and no emission. The only mention is a doc line
  (`superdsc_opspec.rs:288`). `grep -rn -i 'fp8todl16|Fp8ToDl16'` over `backends/` and
  `crates/` (excluding `target/`) returns only that line. `DeviceOp` lists no `Qfp8ch` or
  fp8 convert either (`ddl_binds.rs:142-176`), and `DdlDf` has only `Fp16`, `Fp32`
  (`ddl_binds.rs:68-77`).
- **A QFP8CH arrangement variant**, and any statement that `qfp8ch` output / `fp8todl16` input
  carry it (`ddl_binds.rs:92-99`).
- **An fp8 KV pool** (head-major `[P, block, sticks, 128]` SEN143_FP8, QFP8CH-tagged) and its
  zero fill. `grep -i 'kv_cache_dtype|KvCacheDtype|CacheDType'` over
  `backends/spyre{,-hw}/**/*.rs` returns nothing. The serving config enum has `Fp8`/`Fp8E4m3`
  variants (`crates/core/config/src/cache.rs:25-27`), but no spyre consumer was found.
- **The per-layer `k_scale` fold into the softmax scale and the `v_scale` output multiply** on
  the attention path (SI A.0, A.3-A.5). No scratchy spyre counterpart was found (same grep).
- **The CPU-fallback hazard check** (fp16→fp8 `.to` falls back through host and writes
  STANDARD order, B.2.c). scratchy has no host-fallback path to compare.

---

## E. Tests

spyre-inference (SI):
- `tests/e2e/test_fp8_kv_cache.py:72-94`, `test_fp8_kv_cache_tracks_the_fp16_cache`. On the
  engine (`ibm-ai-platform/micro-g3.3-8b-instruct-1b`, `:29`), the `kv_cache_dtype="fp8"` run is
  asserted to differ from `"auto"` (`:83`). The first greedy token must match (`:86-88`). Each
  fp16 pick must be in the fp8 run's top-5, until the first flip (`:89-94`).
- `tests/attention/test_spyre_attn.py:2267-2289`, `test_spyre_attn_fp8_kv_cache` (on card,
  compiled, `block_size=128`, scales `(1,1)` and `(0.05,0.02)`). The body
  (`_run_spyre_attn_test`, `:157-345`) does the following:
  - writes through `do_kv_cache_update` (`:152-153,303-313`);
  - checks the stored pages first, via `pages.to(dtype).to("cpu")`, i.e. the device fp8→fp16
    conversion. The comment is "the stored order is QFP8CH, so a raw byte copy would read
    shuffled" (`:315-333`). This is compared against the host `quantize_kv` with
    `atol=2**-6, rtol=2**-3`, with the comment "The device conversion rounds a few values one
    e4m3 step (2^-3 relative) from the host cast, mostly below the smallest normal (2^-6)"
    (`:322-331`);
  - then checks attention against the reference computed on the dequantized values (`:334-345` onward).
- `tests/attention/test_spyre_attn.py:2066-2099` (prefill) and `:2102-2136` (batched decode),
  run on CPU (Python walk). These assert that a kernel fed fp8 pages with `scale*k_scale` and
  `v_scale` equals the kernel fed dequantized fp16 pages, at `atol=rtol=1e-2`.
- `tests/attention/test_spyre_attn.py:1782-1793` asserts that head_size 64 is refused for fp8;
  `:1830-1834` asserts clamp saturation to ±448; `:1815-1827` covers scale plumbing; `:764-770`
  asserts that other fp8 kinds are refused.

torch-spyre (TS):
- `tests/inductor/test_fp8_operations.py:72-119`, `test_qfp8ch_basic_conversion`: `qfp8ch` then
  `dequantize_fp8_with_scale` at `[1,2,128]`, with `atol=2**-6, rtol=0.125`, and ≤2% of
  normal-range values may differ from CPU. The comment says "The device rounds the input to
  DLFloat16 before FP8 … does not match the CPU on FP8 subnormals".
- `tests/inductor/test_fp8_operations.py:121-164`, `test_fp8todl16_basic_conversion`: the same
  round trip, at `atol=0.5, rtol=0.1`.
- `tests/inductor/fp8/test_fp8_quantize_dequantize.py`:
  - strict xfail #3033 for K≥16 when STANDARD host fp8 meets the QFP8CH kernels (`:48-54`):
    the `qfp8ch` → `.cpu()` direction (`:146-168`) and the host fp8 → `fp8todl16` direction
    (`:320-333,346-373`);
  - exact device round trips at K = 16…4096, including 3-D/4-D (`:684-757,907-926,972-995`).
- `tests/tensor/test_tensor_layout.py:1068-1095`: the `_qfp8ch_stl` output layout (rounds a
  partial stick up, tags QFP8CH).
- **No torch-spyre test of an fp8 `index_copy_` / `index_select` / subscript gather was found.**
  `grep -rln float8_e4m3fn tests | xargs grep -ln 'index_copy|index_select|index_put|[idx]|gather'`
  matches only `inductor/test_work_division.py`, `inductor/test_inductor_ops.py`, and
  `configs/upstream_tests_beta/test_torch.yaml`. In the first two files, the fp8 hits are
  the fp8 matmul work-division tests (`test_work_division.py:1925-2010`, with QFP8CH/QFP8WT
  operand deps), to_dtype param filters (`test_inductor_ops.py:740,761`), and a `_scaled_mm`
  CPU reference (`test_inductor_ops.py:14434-14490`). The `index_copy_` tests in
  `test_inductor_ops.py:14310-14370` use `cached_randn` caches `[1,8,2048,128]` with the
  default dtype (`:14328,14360`), not fp8. The yaml file was not read.

---

## What torch-spyre does, end to end (granite-3.1-8b, fp8 KV)

Each line traces to a bullet above.

**Allocation (once per layer, K and V each).**
- `SpyreTensorLayout(device_size=[P,128,1,128], stride_map=[16384,128,128,1], SEN143_FP8, valid_elements={(0,):P,(1,):128,(2,3):128}, QFP8CH)`.
  Host shape `[NB,8,128,128]` float8_e4m3fn (A.1).
- Zeroed by a device `MEMORY_FILL` DMA over the whole storage through a `uint8` view
  (device dtype SENUINT32); its fill pattern is UNKNOWN (A.1, B.4).
- Views made outside the graph: rows `[NB*8*128, 128]` and folded `[P,128,128]` (A.1).

**Write (compiled `reshape_and_cache_kernel`, per step).**
1. `key`/`value` `[T,8,128]` fp16 → `realdiv` by `k_scale`/`v_scale` (skipped when the scale
   is 1.0) → `[T,8,128]` fp16 (A.2, B.5).
2. → `clip` ×2 (min −448 / max 448, each with the other bound at ±DLFLOAT16_MAX) →
   `[T,8,128]` fp16 (A.2, B.5).
3. → device op **`qfp8ch`** (SDSC opfunc `qfp8ch`, DDL `quantization_double_pad.ddl:55`) →
   `[T,8,128]` SEN143_FP8, stick 128, same dim order as the input, stick-count dim rounded up,
   **EA QFP8CH**. The in-stick permutation is UNKNOWN; per #3033, elements 0-7 of a row match
   STANDARD and elements from 8 on do not (B.1, C.1).
4. → per kv head `h` (8×K, 8×V): device op **`identity`** with an index operand (an Inductor
   `Scatter`), writing `key[:, h]` `[T,128]` fp8 into rows `idx_h` of the `[NB*8*128,128]` QFP8CH
   view. Each row is one whole 128-element fp8 stick. The bytes move unconverted. No pass
   compares the source EA to the pool's EA. An fp8 source that needs re-alignment is refused
   (`Unsupported`), never restickified (A.2, B.3, C.3).

**Read (compiled attention kernels, per page tile).**
5. Gather from the QFP8CH pool by device op **`identity`** + index, or fused into step 6 —
   which one is UNKNOWN. Shapes: decode `k_pages[kv_rows]` → `[8,128,128]` fp8 (128 whole
   sticks per kv row); prefill `index_select` → `[8,1,128,128]` or `[G,8,1,128,128]`; batched
   decode `k_pages[page_ids]` → `[E,8,128,128]` (A.3-A.5, B.2, B.3).
6. `.to(fp16)` → device op **`fp8todl16`** (DDL `:60`, two SFP `SHUFFLE` computes with
   `indices=[0..7]` and `[8..15]`). The input arrangement QFP8CH is not passed to the SDSC, and
   no reorder op is inserted. torch-spyre relies on the op to read QFP8CH and write logical
   order. The output is **EA STANDARD**, a dense fp16 layout rebuilt from the host size
   (decode `[8,128,128]` fp16). The concrete `device_size` is UNKNOWN (B.2, C.2).
7. fp16 `matmul`s on the widened page: `q @ kᵀ * (softmax_scale·k_scale)` and `probs @ v`.
   After normalize, `attn * v_scale` (skipped at 1.0). Then `index_copy_` (or `copy_`) into the
   output (A.3-A.5).
