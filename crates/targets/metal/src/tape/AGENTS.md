# src/tape/

## OVERVIEW
The pure, objc-free half of the crate: the tape data model plus SubtileTape -> `LoweredMetalTape` lowering. `#[forward]` runs this at expansion, and the runtime plays the result. 13 files, ~21k LOC; score ~17 (symbol-dense, high centrality).

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Lowering entry | `lowering.rs` `lower_subtile_tape_to_metal` (~l.150), `BakePoint` |
| Tape types, closed `KernelId` set, `DispatchShape`, `RuntimeGate`, `Fence` | `lowered.rs` |
| One typed `MetalStep` variant per command-emitting step | `step.rs` (`MetalStepTape`, `StepRow`, `MoeBlock`, `KvWrite`) |
| Kernel identity ZSTs (`LIBRARY`/`FUNCTION`/`KERNEL_ID`) | `kernel_identity.rs` `MetalKernel` |
| Per-kernel baked constants (must match shader slots) | `kernel_constants.rs` |
| Binding sets per kernel | `kernel_bindings.rs` |
| qmv/qmm/w4a8 pickers and symbol names (MLX-faithful) | `quantized.rs` `pick_qmv_kernel{,_wide}`, `qmv_fast_covers` |
| Silicon profiles M1-M5, NAX gate | `targets.rs` `MetalTargetProfile`, `is_nax_capable` |
| Newtype ids (`LayerId`, `ArenaSlotIdx`, `BucketM`...) | `ids.rs` |
| Constant slot values | `constants.rs` `ConstantValue`, `ConstSlot` |

## CONVENTIONS
- Tapes hold no buffer pointers, only pipeline keys, dispatch shapes, scalars and slot ids. Workers bind against their own arena.
- Loops are fully unrolled at lowering. There is no runtime Loop interpreter, and the runtime never re-derives the analysis.
- Kernel choice is closed-form, from an analytical roofline. There's no empirical cost table.
- Newtype ids come with `get()`. f32 fields compare by bit pattern (`f32_bits!`).
- `lowered.rs` is the lower module and must not depend upward on `lowering.rs`.
- `src/interpreter/metal/{lowered,ids,kernel_*,continuation_witness}.rs` are re-export shims of this dir. Edit here.

## ANTI-PATTERNS
- ⛔ EXHAUSTIVE, NEVER `_` on layer-bearing variants (lowered.rs ~1673). A new step kind should fail with E0004 until its arm exists.
- ⛔ NOT ALWAYS 1: the layer body stride. Assuming 1 silently mis-rolls a model (lowered.rs ~1857).
- A wrong `_bq32_`-style symbol token with a correct `KernelId` silently picks the wrong specialization. Go through the `MetalKernel` ZSTs, never string-build names.
- BQ-blocks must not straddle a sequence boundary.
- A relocatable span must not attend the preamble at any sliding layer. Tail-unscaled partial softmax is NOT argmax-invariant.
- `BakePoint` chunked addressing is a parameter, never process state (a global caused a cross-model probe race).
- "Do NOT re-enable" notes in lowering.rs (e.g. the token-0/empty-past 2048 chunk) are binding.
- Bucket folding never merges buckets with differing matvec ends (quantized.rs).
- Qwen-MoE roll witness: two f32-bit fields NEVER compare equal by design (step.rs ~154). Don't "fix" it.

## NOTES
- `lowering.rs` is 12.8k lines. Navigate with `lsp_symbols`, not linear reads.
