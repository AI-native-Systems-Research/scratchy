# gemma-4 on spyre — campaign plan (worktree spyre-moe, 2026-11-17)

Mission order: **gemma-4 12B dense first, then gemma-4 26B MoE**, through the compiler
stack (triton→ttir→ktir→superdsc, spliced into the full-forward ladder). No model/kernel
hacks in the ladder — the ladder lowers ops. Pod: `nickm-54f8b56448-f67zh`, warm slot
`/work/scratchy` (fire tree), `oc rsync --delete` only, never cold, never a new worktree.

## Ground truth (measured, 2026-11-17)

`cargo build -p scratchy-target-spyre-models --features spyre-lower,
scratchy-models/gemma-4-12b-it,scratchy-quantizations/fp8-dynamic-per-channel`
in `backends/spyre` (log `/work/g4d-bake-1.log`): the whole dense ladder lowers —
1181 tiles, 48 AttnDecode, full superdsc mint — and fails at ONE wall:

```
wiring not lowerable — AttnDecode t1142: triton splice: a 1024-position sliding window —
spyre-inference's paged attention kernels attend every position up to the query's own,
and have no window
```

Checkpoint on the pod (HF_HOME=/work/huggingface): `RedHatAI/gemma-4-12B-it-FP8-Dynamic`
(preset `fp8-dynamic-per-channel`).

## Part 1 — dense: the window law

ONE fact: `AttnMask::SlidingWindow { window }` (already minted by the bridge,
`kv_codec.rs`) states the query's reach — a query at absolute position `p` attends keys
in `[p − W + 1, p]`. The reference (`spyre_attn.py`, `_build_active_tiles_with_skip` /
`_build_single_tile` / `_max_active_blocks`) states the whole law; our kernels already
take additive mask stacks, so **the kernels change nothing** — the window rides the mask
stack and the staging. Every reader of "causal to the sequence end" reads the window at
its one source:

1. **The mint** — `paged_step_of_ir` (lower_subtile_tape_to_ktir.rs:1589):
   `PagedStep.window: Option<NonZeroU32>` from the node's mask, inside the existing
   all-attentions-agree check. Per-class bodies mean only same-mask nodes share a step.
2. **The walk** — `walk_pages` (splice): a windowed op's swept extent is its SPAN, not
   the context: `span = W + mq − 1` positions; `walk = min(ceil(span/slots) + 1, pages,
   active_cap-resolved)` (the reference's `_max_active_blocks`, its `+1` the boundary
   block). `walk_buckets` for a windowed body top out at the span bound.
3. **The rungs** — the runtime's one selection line (`rung_for(valid_len)`,
   `SweptCols::covers`) is UNCHANGED: a windowed ladder's top rung (span bound) is the
   fallback that serves any length — the window law is what makes a span-bounded sweep
   serve any position. `prove_ladder_walks` proves span coverage; `check`'s reach law
   becomes the span law (`span(rows) ≤ walk·slots`, not `rows.end() ≤ walk·slots`).
4. **The mask law** — `mask_stack_of` (paged_step.rs): re-anchored frame — walk trip `i`
   covers absolute positions `(first_active + i)·slots + c`; row `r` attends iff
   `pos(r) − W + 1 ≤ kv ≤ pos(r)` (pos(r) already absolute via `StepRows`). Interior
   blocks come out zero by the law itself (the reference's zero-tile cache is invisible
   in the bytes).
5. **The page table** — windowed row `i` stages `blocks[first_active + i]`, zero past
   the request's blocks (null page + MASKED, the existing law). `first_active =
   max(0, rows.start − W + 1) / slots` — the EARLIEST query's window (prefill-correct),
   derived once, shared by every reader (mask + table + check).
6. **The batch law** — `batch_sources`: per sequence `first_active_s` from its own
   position; `mask_by_chunk` carries `window_start_s ≤ slot_pos ≤ pos_s`; chunk pages
   stage each sequence's active blocks. Chunking covers the max span.
7. **The splice** — un-refuse `SlidingWindow`; thread the window to the mint. Kernel
   compile gets `NUM_BLOCKS` = the span-bounded walk (same `walk_pages` call). Operands
   unchanged.

Every line passes the charter test: would it exist for a non-gemma op that states the
same mask? Yes — the law is over `AttnMask::SlidingWindow`, its window, and the shapes.

### Proof ladder (each its own commit, each gated by the warm bake)
a. Field + mint + agreement check (no behavior; bake still fails at the same wall).
b. Mask law + fixtures (pure function, unit-tested against the reference's tiles).
c. Walk law + page-table/check/reach laws + fixtures.
d. Ladder/rung/proof laws.
e. Batch law + fixtures.
f. Un-refuse the splice → the 12b bake goes GREEN (emu numeric gate included).
g. Pod: `spyre` CLI build (spyre-hw, spyre-emu) → `spyre chat` on the card, coherent
   reply, greedy parity vs the reference path. A/B vs main is N/A (main refuses) — the
   reference semantics gate (emu) + card chat is the proof.

## Part 2 — MoE: the moe.py port (after dense is proven)

State on main: the DSL states the MoE as ONE node (`gemma_moe(router_in, expert_in,
router, experts)` — gemma4-moe.py:72); the splice refuses every per-op MoE SubOp by law
(splice lib.rs:347-368); `kernels/moe.py` (239 ln) + `moe_prefill.py` exist with numeric
test scaffolding but are UNTRUSTED — the reference is
`~/git/spyre-inference/spyre_inference/moe.py` (958 ln, gemma-4 recipe,
`apply_monolithic`).

1. Port `moe.py` statement-for-statement from the reference (boundary: `x` and
   `router_logits` in; the router projection stays the plain matmul it is), as the
   head-major attention kernels port theirs. Replace, never patch.
2. The splice binds the one MoE node to it — operands, constexprs,
   gathered-vs-persistent by token count — exactly as `lower_attn` binds the attention
   node and as the reference's `apply_monolithic` dispatches.
3. A `Program` label routes to the generic walk (`lower_function`), as `PagedAttn` does.
   Nothing past the splice knows it is MoE.
4. Missing ops the port needs (gather, reduce, compare, batched contraction) are general
   lowerings of those ops, reached by any kernel, each with its own fixture.
5. Proof: parity-tiny bake → emu gate → card chat on `google/gemma-4-26B-A4B-it`
   (bf16, cached on the pod's home cache) / fp8 presets per the sweep law.

## Laws this plan obeys (charter, CLAUDE.md, memory)
- The ladder lowers OPS: every window reader is stated over `AttnMask` + shapes, never
  over gemma/layer kinds/head counts/model names.
- No runtime refusals over bake-known facts; no fallbacks; no `#[allow]`; no weakened
  tests. A named refusal only where a combination genuinely has no lowering yet.
- One fact, one source: `first_active` and the span are derived once and shared.
- Land one PR at a time; every commit gates on the warm pod bake + touched-crate lib
  tests; card proof before shipping.
