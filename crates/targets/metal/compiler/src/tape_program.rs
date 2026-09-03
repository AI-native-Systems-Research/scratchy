// SPDX-License-Identifier: Apache-2.0
//! THE SHARED TAPE METAL LOWERS FROM, RE-ROLLED — in the target's own crate.
//!
//! ⛔ THIS USED TO LIVE IN `compiler/macros/codegen.rs`. It is metal codegen: it names
//! `scratchy_target_metal::from_tape` in every line. A `#[cfg(feature = "metal")]` island in the
//! shared compiler crate compiles only when that feature is on, which is how target code rots
//! there unnoticed.

/// Emit `impl ::scratchy_forward_compiler::CanonicalParams for Weights { … }` —
/// per-canonical model constants the universal `Instruction::eval`
/// reads as `W::HEAD_DIM`, `W::INTERMEDIATE_SIZE`, etc. Default
/// 0 / 0.0 / -1 for fields the canonical doesn't carry.
///
/// `tp_world_size` shards the column-parallel dims (`num_q_heads`,
/// `num_kv_heads`, `intermediate_size`) by `tp_world_size` — each
/// rank owns `1/tp` of those dims. The sharded values flow into the
/// emitted `Instruction::eval` body as `<W as CanonicalParams>::…`
/// constants, so kernel launches at tp>1 see the per-rank sizes
/// automatically. At `tp_world_size = 1` (every emission until task
/// #7's outer-loop fanout lands) sharding is identity — output is
/// byte-identical to single-rank builds.
///
/// Caller is responsible for ensuring `tp_world_size` evenly divides
/// every column-parallel dim (KV replication when
/// `num_kv_heads < tp_size` is task #6's loader-sharding work);
/// `compile()` skips a `(variant, tp)` tuple when divisibility fails.
/// Emit the `synthesized_kernel_sources()` override for the
/// `CanonicalParams` impl when this model can use compiler-driven
/// megakernel synthesis. Returns `quote! {}` (empty) when:
///   - the model isn't quantized with MLX-affine int4 (no AffineQmv
///     atoms to fuse), or
///   - the model lacks the standard transformer-decoder shape we know
///     how to synthesize for.
///
/// When emitting, calls `fuse_pass::synthesize_pre_attn_chunk` at
/// macro-expansion time to generate the MSL source, then bakes the
/// `(symbol, source)` pair into the generated arch as a `&'static
/// [(&'static str, &'static str)]`.
/// EMISSION ORDER, READ OFF THE SHARED `SubtileTape`.
///
/// Builds the tape through `scratchy_target_metal::from_tape` — `lower_region` →
/// `ValidatedGraph` → `lower_dag_to_tape`, the same three calls the SuperDSC emitter makes —
/// and projects each `Compute`'s position back onto the source op it came from.
///
/// ⛔ A REFUSAL HERE IS A BUILD DEFECT, NOT A FALLBACK. Every arch is tape-scheduled
/// (`is_tape_scheduled` is unconditionally true), so there is no second schedule to drop back
/// to; an arch the shared lowering cannot express has to be *expressed*, and the panic names it.
///
/// Ops the lowering folds away get `usize::MAX`: they sort last and the emitter skips them
/// (`fold_plan` marks them `fused_into`), so the value is never an order the device sees.
pub fn tape_levels_of(
    lowered: &scratchy_subtile::handoff::LoweredDecode,
    stem: &str,
    m: u64,
) -> Vec<usize> {
    tape_program(lowered, stem, m).levels
}

/// THE SHARED TAPE METAL LOWERS FROM, RE-ROLLED.
///
/// ⭐ ONE RE-ROLL, ON THE SHARED TAPE, BEFORE ANY TARGET LOWERS. `reroll_subtile_tape` finds the
/// repeating layer body once, on the `SubtileTape` — the same call the SuperDSC emitter makes,
/// on the same object. Both targets then READ the loop off it. Metal used to re-discover the
/// same fact afterwards from its own `Instruction` stream (`apply_loop_compression` +
/// `detect_repeating_run`): a second search over a second representation, which could disagree
/// with the first and which no test compared.
///
/// What metal needs to emit from the shared tape.
pub struct TapeProgram {
    /// Emission order per source op — `usize::MAX` for ops the lowering folds away.
    pub levels: Vec<usize>,
    /// The ROLLED tape's items: what metal emits from.
    pub rolled: Vec<scratchy_target_metal::from_tape::TapeItem>,
    /// The UN-rolled tape's items: the reference the roll is proven against.
    pub unrolled: Vec<scratchy_target_metal::from_tape::TapeItem>,
    /// How one cell divides into layers — `(length in steps, class id)`.
    pub segments: Vec<(usize, u64)>,
}

pub fn tape_program(
    lowered: &scratchy_subtile::handoff::LoweredDecode,
    stem: &str,
    m: u64,
) -> TapeProgram {
    use scratchy_target_metal::from_tape;
    let graph = from_tape::graph_for_metal(&lowered.input);
    let refuse = |what: &str, e: from_tape::TapeLoweringError| -> ! {
        panic!("[m2-flip] {} m={m}: {what}: {e}", stem)
    };
    let tape = from_tape::tape_for_metal(&graph)
        .unwrap_or_else(|e| refuse("the shared lowering could not build a tape", e));

    // ⛔ THE ORDER COMES FROM THE UNROLLED TAPE, AND IT HAS TO. A rolled tape holds ONE layer
    // body, so the source ops of layers 1..N appear in no step — `schedule_of` would report
    // `None` for them, they would take `usize::MAX`, and every later layer would sort to the end
    // of the emission. The order is per-source-op and the emitter is still unrolled, so it reads
    // the unrolled tape.
    let items = from_tape::items_of(&graph, &tape)
        .unwrap_or_else(|e| refuse("the tape holds a step metal cannot play", e));
    let steps = from_tape::steps_only(&items);
    let sched = from_tape::schedule_of(lowered.input.ops.len(), &steps);

    // ⭐ THE ROLLED TAPE IS WHAT METAL EMITS FROM. One re-roll, on the shared tape, by the same
    // call the SuperDSC emitter makes. The emitted instruction stream comes out rolled, so
    // nothing downstream has to search it for a repeating run.
    let rolled = match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        scratchy_subtile::subtile_tape::reroll_subtile_tape(&tape, &graph)
    })) {
        Ok(r) => r,
        Err(_) => panic!(
            "[m2-flip] {} m={m}: the SHARED reroll panicked on this tape",
            stem
        ),
    };
    let rolled_items = from_tape::items_of(&graph, &rolled)
        .unwrap_or_else(|e| refuse("the rolled tape holds a step metal cannot play", e));
    let segments = from_tape::cell_layer_segments(&graph, &tape, &rolled);
    TapeProgram {
        levels: sched
            .level
            .iter()
            .map(|l| l.unwrap_or(usize::MAX))
            .collect(),
        rolled: rolled_items,
        unrolled: items,
        segments,
    }
}
