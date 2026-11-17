# gemma-4 campaign — plan & state (worktree spyre-moe)

Mission (user order 2026-11-17): gemma-4 12B dense end-to-end, then gemma-4 26B MoE —
through the compiler stack, no ladder hacks. Charter in this worktree's CLAUDE.md
(commit 8c7989e87a).

## Part 1 — 12b dense

- [x] census the hybrid's attention classes (40 sliding W=1024/nkvh=8/hd=256 + 8 global
      nkvh=1/hd=512; 48 layers, 1181 tiles, KV pool 390×128)
- [x] `PagedStep.window` field + mint + codegen (`f9c9dc93ae`)
- [x] windowed staging law — anchor/first_active, re-anchored mask stack, page-table skip,
      span admission, per-sequence batch anchoring (`484b7e7f1d`)
- [x] windowed walk stops at the reference's active-block bound (`441f584b68`)
- [x] un-refuse the splice (window rides the mask stack) + ONE paged step PER ATTENTION
      CLASS: `Wiring::paged: &'static [PagedStep]`, class-sorted, class 0 keeps the
      sentinels; `splice::paged_window_of_class` is the ONE window-family statement;
      scalar-operand census (`ArgSpec::Scalar` is an operand) — commit `8615433d5c`
- [x] local proof: 12b bake green (both classes, 48 launches, wiring lock 48/48),
      spyre-emu numeric gate PASSED, all local suites green
- [x] pod gates (f67zh, /work/scratchy): lib tests RC=0, clippy RC=0
- [x] pod bake GREEN after rung `aca666310e` (GUARD #14 write set = the binding's `!is_input`,
      not dsType_; wiring scan dedupes `(layer, ks, vs)` — window copies restate a layer's
      wiring): test RC=0, clippy RC=0, fp8 bake BAKE_RC=0 on f67zh
- [x] card tier: CARD GATE GREEN (GATE_RC=0, all 4 steps) against the auditor-ruled
      pairing /work/senlib-rs = a7f3380 (slot handed over by the senlib-rs session,
      pairing announced, restore when done). Two rungs of first-card-path finds:
      `d401e1826c` (KV-pool budget read `Wiring::paged` as Option — now `.first()`,
      classes share one pool), `e1f9936d83` (two card-path clippy lints; the card gate
      is these paths' only lint — spyre-hw never carried a clippy gate). Traps:
      rsync preserves mtimes → stale warm rlibs until the tree is touched; pod cargo
      needs `bash -lc`
- [x] EAR envelope restored — the fp8 card build's BUILD_RC=101 (`41_matmul_o2350`,
      `DtException: EAR overflow, MutableAddrSplitting.cpp:814`) was a REGRESSION, not
      a wall: #598 (10-17) landed the measured whole-dxp kernel-slab envelope, #714
      (10-19) deleted it one day after the last gemma fp8 bake, granite-only gates
      kept the deletion invisible (granite's head is 12× under the line); the CB
      campaign's m=7 decode rung (503 MB/core, 3.4× past the line) then tripped it.
      Fix: `WHOLE_DXP_EAR_KERNEL_SLAB_LIMIT` (150 MiB, measured bracket 125.8
      accepted / 176.2 refused) + `KernelSlab::of_plan` in `superdsc_opspec.rs`; the
      unbatched matmul arm in `matmul/opspec.rs` routes over-line ops through the
      SAME `OutAxis::NAME` tiler the batched arm rides (trips; in-envelope ops stay
      whole-dxp byte-identically). Wiring pin:
      `zz_the_lm_head_kernel_span_meets_the_ear.rs` (m=7 → 512 trips covering the
      vocab; m=1 12b + granite stay whole-dxp; 31b two-row reproduces #598's exact
      card-proven 128-trip form). Typed extents throughout (`MatM`/`MatK`/`MatN`,
      name derived from geometry). Local: 5/5 + ktir-superdsc suite + clippy green.
      Pod (gtv97): TEST_RC=0, CLIPPY_RC=0
- [ ] gemma-4-12b fp8 card build (in flight, /work/g4d-cardbuild2.log, BUILD_RC
      marker): `cargo --config patches/oco.toml build --release
      -p scratchy-target-spyre-cli --features spyre-hw,model/gemma-4-12b-it,
      quant/fp8-dynamic-per-channel` from /work/scratchy/backends/spyre-hw —
      coordinated with torch-spyre-senlib-rs (shares the nickm pod; no GPU runs
      queued from its side)
- [ ] card proof: `spyre chat` on the card, coherent reply — **BLOCKED: decode numerics defect**
      (first gemma card decode ever on the CB-era tree). Evidence (all fp8 card, gtv97,
      /work/spyre-g4d-hw): BUILD_RC=0 + load + prefill good; short answers coherent; long
      generations decay smoothly (top-2 logit gap: 3-9 healthy → knee → 0.02 by step 60;
      eventually all-NaN logits → minted sentinel 528448 = pad 524288 + 4160). EXONERATED:
      EAR fix (1-row rung, byte-identical), sampler (greedy decays identically), chained
      head on card (digit picks ~236770-832 count correctly 37 steps; 🦀 round-trips),
      block/rung boundary (knee at position ~50 mid-block-0; 256-rung weak-from-start at
      ~170). LOCALIZED: prefill-written KV readable at all positions (recall probes:
      "blue" at pos 5 through 160 tokens; summary start semantically right) — the model
      loses its OWN decode-written history (counting drift). EMU REPRODUCES (drifts by
      step ~6) ⇒ shared front: the KV-codec lowering (KvEncode/KvStage/RopeAppend) —
      exactly the ops on the numeric gate's SKIP list ("the KV-codec reference arms are
      the next rung"). Confound: emu's f16 stick doors violate the chain's f16-exactness
      law at gemma k=5 (6974/8258 s-values inexact, first at pick 26304) — emu-only
      corruption of high-id picks, may accelerate its decay. Onset not monotone in prompt
      length (12-token prompt: 37 good steps; ~30-token: broken from token 1; ~200:
      8 good) — prefill-rung lever suspected. NEXT: the eval_node KV-codec reference
      arms (the gate's own named missing rung) → gate the divergence at bake → fix the
      lowering; emu chat still grinding (/work/g4d-emuchat.log) as oracle

## Part 2 — 26b MoE (after dense is proven on the card)

- port `moe.py` statement-for-statement from
  `~/git/spyre-inference/spyre_inference/moe.py` (gemma-4 recipe); replace the tree's
  untrusted `moe.py`, never patch around it
- the tape states the MoE as ONE node; the splice binds it (operands, constexprs,
  gathered-vs-persistent by token count) as `lower_attn` does — nothing past the splice
  knows it is MoE
- missing ops (top-k = K rounds of plain ops; gather/scatter/reduce/compare/batched
  contraction) are GENERAL lowerings with their own fixtures
- proof on gemma-4-26b: bake → card chat, A/B vs main

## Dev-loop law

pod `nickm-54f8b56448-gtv97` (f67zh died 11-17; same deployment, /work survived),
warm tree `/work/scratchy`, SHARED with torch-spyre-senlib-rs — coordinate GPU/card
runs with it;
`oc -n a1-vllm-spyre rsync --delete --exclude='target' --exclude='.git'
--exclude='bench_results' --exclude='.claude' <worktree>/ <pod>:/work/scratchy`;
touch the changed files after rsync (stale-warm-target trap); gate scripts
`/work/g4d-{test,clippy,bake}.sh`; never background a pod build without an RC marker;
senlib-rs trouble → senlib-rs-auditor.
