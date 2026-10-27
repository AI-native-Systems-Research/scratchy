# scratchy-target-metal-compiler

## OVERVIEW
Macro-time lowering: SubtileTape -> metal step tape -> per-bucket `static` tapes as TokenStreams. It's a separate crate so `syn`/`quote` never link into the runtime. Score ~14; earns its own file as a distinct crate boundary that `#[forward]` calls directly.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Macro entry, per decode canonical | `canonical.rs` `lower_canonical`, `lower_canonical_classed` (`MetalCanonical`, `CanonicalRefusal`, `RollOutcome`) |
| Opcode lowering core: step records, colouring projection, hazards | `steps_from_tape.rs` (2.9k lines): `assemble`, `cache_live`, `hazard_fences`, `proves` |
| Per-bucket static emission | `static_tape.rs` `bake_bucket_tapes`, `bake_module`, `kv_cap_ladder`, `emit_bucket_barriers_static`, `kernel_ref` |
| Value -> TokenStream emit helpers | `const_tokens.rs` `const_tokens` |
| Tape program wrapper | `tape_program.rs` `TapeProgram` |

## CONVENTIONS
- The only consumer is `crates/compiler/macros` (optional dep behind its `metal` feature). The macro calls `lower_canonical` once per decode canonical and `bake_bucket_tapes` once per bucket.
- Target-ABI facts come from `scratchy_target_metal::op_abi` tables. Read them as input; don't re-encode them as match arms here.
- It runs the same tape construction as the runtime (`scratchy_target_metal::tape`). Never fork a second lowering.
- There are no unit tests in this crate. Behavior is verified through model builds and `crates/models/arch/tests/metal_o2_logits.rs`.
- `WeightSlot` and other `syn`-dependent types stay here, never in `scratchy-subtile`.

## NEW-OPCODE CHECKLIST (where each piece lands)
1. Step arm: `steps_from_tape.rs`, an exhaustive match where a missing arm is a compile error.
2. Registry row and colour/fusion facts: `../src/op_abi.rs`.
3. Kernel identity and constants: `../src/tape/{lowered,kernel_identity,kernel_constants}.rs`.
4. Shader: `../shaders/`. Baked slots must match step 3.
5. Verify: build one scoped model that uses the op. Expansion runs this crate, so a refusal surfaces as a macro error.

## ANTI-PATTERNS
- 🛑 KNOWN DEFECT (`steps_from_tape.rs` ~605): the accessor base drops trailing `_<digits>`, so Qwen2-VL's `visual.merger.mlp.0`/`.2` collapse into one accessor. The fix belongs at accessor-name generation, not at keeping the numeral there. Read the comment before touching it.
- Refuse (`StepRefusal`/`BakeDefect`/`CanonicalRefusal`) rather than emit a best-effort tape.
- Linux CI excludes this crate along with `scratchy-target-metal`, because objc2 is reached through it. Never make this crate a dependency of anything on the linux-cuda graph.
