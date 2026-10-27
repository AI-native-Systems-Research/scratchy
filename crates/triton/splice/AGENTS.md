# splice - scratchy-triton-splice: SubOp -> spliced Triton kernel rows

## OVERVIEW
The only producer of the spliced op kinds for the spyre target. It maps each `scratchy-subtile` `SubOp` to a kernel in `crates/targets/spyre/kernels/*.py` and compiles it through the ladder.
Earned its file: the entry from the spyre target into the ladder; the whole crate is one 2.8k-LOC `src/lib.rs` (score ~14, distinct domain).

## WHERE TO LOOK
| Task | Location (src/lib.rs) |
|------|----------|
| Registry (total, per family) | `row`, `rmsnorm_row`, `matmul_row`, `elementwise_row`, `lmlast_row`, `attn_row` -> `TritonKernelRow` |
| Walk entry (fold/sandwich aware) | `lower_all` |
| One-op entry | `lower` |
| Attention | `lower_attn`, `attn_kernel_spec`, `inject_dead_cache_views`, `truncate_unused_mask_param` |
| Prefill lm-head fold | `is_prefill_lm_head_tail` (4 conjuncts), `lower_prefill_lm_head_fold` |
| Row-local dataflow (also used by the spyre numeric gate) | `tensor_feeds_result_through_row_local_ops` |
| Signatures + constexprs | `kernel_spec` (the biggest match) |
| Region-window laws, arity | `lower_one` |
| Compile memo | `Memo`, `memo_key`, `compile_kernel`; probe: `examples/cache_probe.rs` |

Callers: `crates/targets/spyre/src/lower_subtile_tape_to_{ktir,superdsc}.rs`. Golden: `crates/targets/spyre/tests/triton_splice_golden.rs`.

## CONVENTIONS
- Kernel sources are `include_str!` rows, so a missing `.py` is a compile error and every kernel edit rebuilds this crate.
- Compile chain: `codegen::compile` -> `opt::make_ttir` -> `from_ttir::convert` -> `make_ktir` -> `passes::to_ktir::run` (stated once in `compile_kernel_uncached`).
- Op names are `"{stem}_s{node.id}"`. `program_stem` reads the PRODUCER's `EwKind` (BiasAdd -> `add`).
- Bindings are the operand prefix with the output LAST. Attention instead uses door order (q, out, kc, kd, vc, vd, mask); mask tid = `ir.tensors.len()`.
- The memo is process-wide, one compile per key. The key includes the FULL source text, the full pointer type (`*fp16`/`*fp8`), sorted params and constexpr bits. `Val::{Dtype,Seq,Ir,Desc,Slice}` opt the spec out.

## ANTI-PATTERNS
- Totality by construction: no runtime lookup, no `_` arm in registry/arity/grid/`program_stem`. `Refused` arms only keep matches exhaustive.
- Never fold the `(1+w)` gain into weights: carry the `GAIN_OFFSET` constexpr.
- Never substitute `gelu` for `QuickGelu`/`GeluErf`; decompose producer-side.
- MoE is not a spliced kind: it lowers only from `kernels/moe.py` via KTIR.
- Window laws: pointwise kernels state only `C_START`; softcap states neither corner; matmul row corner = 0; rope's 4 regions share one row window.
- Width collision is not identity: the lm-head fold needs all four conjuncts, and `lower` refuses a fold tail.
- Attention post-compile edits only touch params that are unaddressed (dead views) or unused AND last (mask).
- HashMap order must never reach `memo_key`.
- Never claim byte-identity with the deleted builder. The gates are the golden, the emulator compares, and card A/Bs.

## COMMANDS
```bash
cargo test -p scratchy-target-spyre --features spyre-emu --test triton_splice_golden
cargo run -p scratchy-triton-splice --example cache_probe
```
