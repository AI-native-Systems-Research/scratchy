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
- [ ] pod bake (g4d-bake.sh, fp8-dynamic-per-channel) — in flight after rsync
- [ ] card tier: first spyre-hw build in /work/scratchy/backends/spyre-hw
      (worker card/forward.rs per-class fan-out compiled only there); card gate
- [ ] card proof: `spyre chat` on the card, coherent reply

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

pod `nickm-54f8b56448-f67zh`, warm tree `/work/scratchy`;
`oc -n a1-vllm-spyre rsync --delete --exclude='target' --exclude='.git'
--exclude='bench_results' --exclude='.claude' <worktree>/ <pod>:/work/scratchy`;
touch the changed files after rsync (stale-warm-target trap); gate scripts
`/work/g4d-{test,clippy,bake}.sh`; never background a pod build without an RC marker;
senlib-rs trouble → senlib-rs-auditor.
