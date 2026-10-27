# ktir-superdsc — KTIR -> SuperDSC (SDSC) lowering

## OVERVIEW
Featureless leaf crate that lowers KTIR to the SuperDSC JSON wire format for the Spyre card. Third-party producers (IBM/triton-spyre) consume it without scratchy. Score ~22 (huge, Kani proofs, 48 law tests, external consumers).

## STRUCTURE
```
src/addr/       typed addressing: Nest, Win, View, DevOff (the only stride math)
src/emit/       per-node + whole-function lowering, pointwise/reduce assemblers
src/ir/island/  stable IR data: TileOp (+Kani proofs), TiledOp
src/ir/bridge/  translations: tile_op_tiled_op (tiler), tiled_op_sdsc_op/ (Bridge 3: attn, matmul/, pointwise, reduce, rmsnorm), span_overflow
src/sdsc_abstract.rs   ~11k LOC compile-time attention interpreter + Kani harnesses
tests/          48 zz_* law tests; fixtures/ = vendor JSON goldens
```

## WHERE TO LOOK
| Need | Location |
|------|----------|
| Whole program -> SDSC | `emit/whole_function.rs::lower_function` |
| Per-node lowering | `emit/lower_ktir_to_superdsc.rs` (`elementwise`, `rmsnorm`, `silumul`, `AttnAt`, rope) |
| fp8 W8A8 matmul | `emit/ktir_matmul_fp8.rs` |
| Attention emission | `ir/bridge/tiled_op_sdsc_op/attn.rs::assemble_attn::<NQH,NKVH,HD>` |
| Matmul family | `ir/bridge/tiled_op_sdsc_op/matmul/` (dims/opspec/assemble/walk; sealed `Walk2`/`Walk3`, `SharedKernelBmmForm`) |
| OpSpec / WorkPlan | `superdsc_opspec.rs` (`MAX_CORES=32`) |
| Core split | `work.rs` (`CORELETS_PER_CORE=2`) |
| Wire JSON | `wire.rs` (`Dsc`, `SdscOp`) |
| Placement | `placement.rs` (`BundleLayout`, a struct, not a trait), `place.rs` (`SynthRole`) |
| Reserved tensor ids | `reserved_tids.rs` (`u32::MAX-n` ABI) |
| KV span guard | `ir/bridge/span_overflow.rs` (torch-spyre port) |

## CONVENTIONS
- Builders: `*_opspec` -> `OpSpec`; `assemble_*` -> `Vec<EmittedOp>`; generic over `KindTag`/`DataFormat` tags.
- Const-generic doors: `attn_at::<NQH,NKVH,HD>`, `rope_at::<HD>`. Geometry is a type, not a runtime branch.
- Decompositions match torch-spyre literally (`spyre_rms_norm`, `spyre__sdpa_overrideable`, GQA by expansion).
- `emit` is a dir module only to seal `bmm_site::TapeLoweringSite::witness` (`pub(in crate::emit)`).
- Address mints: `View::win`, `win_at`, `BundleLayout::win`; `Win::minted` is pub(crate). Pin layouts with `const _: () = assert!`.
- Tests assert on raw vendor JSON fields (`dscs_`, `labeledDs_`, ...) via `serde_json::to_value`, not on mirror types. Vendor goldens live in `tests/fixtures/*.json`.
- Pin per-model geometry inline in tests (granite-3.1-2b: NQH=32, NKVH=8, HD=64; hd128 = granite-3.1-8b).

## ANTI-PATTERNS
- Adding a `[features]` entry, an edge to `ktir-emulator`, or any `Deserialize` (it's serialize-only).
- Stride arithmetic outside `Nest::off`; re-adding `DevOff::from_arg_raw` or `ex_unit`.
- A bare `64` stick width (fp8 elems/stick == hd 128), branching on `HD==64`, `(kv_dim/hd).max(1)`.
- `_` arms on elementwise kinds (a new kind must fail with E0004); defaulting `b_orient`; capping a too-small placement instead of returning Err.
- Splitting a reduction dim or cutting a physical stick; gathering a matmul operand directly.
- Silencing `unexpected_cfgs` (`cfg(kani)` is declared via check-cfg), or weakening a `zz_*` assertion.

## COMMANDS
```bash
cargo test -p ktir-superdsc                              # no features exist
cargo test -p ktir-superdsc --test zz_the_chunk_emission
cargo tree -p ktir-superdsc -e normal                    # must show only ktir-core, half, serde, serde_json
KANI_PKG=ktir-superdsc <kani cmd> <harness>              # NO --features (old spyre instruction is wrong)
```
