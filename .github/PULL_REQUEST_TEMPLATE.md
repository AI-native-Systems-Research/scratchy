<!--
PR titles use Conventional Commits, scoped to the crate/area: `fix(metal): ...`
See docs/CONTRIBUTING.md. Delete sections that genuinely don't apply.
-->

## Context

<!--
What is this about in the big picture? Link the issue / tracking issue.
This PR description is the provenance record for the change: what, why, evidence.
-->

Closes #

<!-- Tier 2 work only — link the spec/plan pair this implements, if any. -->
Spec/plan:

## What this delivers

<!-- Concretely: files, behavior, features. -->

## Input/output proof

<!--
The exact command a reviewer runs, INCLUDING the feature scope you built with —
without it they can't reproduce you on different hardware. What should they see?
-->

```bash
cargo build --release -p scratchy-cli --features metal,model/smollm2-135m
```

## No-regression proof

<!-- Which existing tests still pass; what guarantees prior behavior is unchanged. -->

## Spec drift

<!-- Tier 2 only. If implementation diverged from the spec, the spec is updated
     in this PR — specs are live documents, not write-once. -->

- [ ] Spec updated to match what was built
- [ ] N/A — no spec for this change

## CI gates run locally

- [ ] `cargo fmt --all -- --check`
- [ ] macOS/metal clippy — `cargo clippy -p scratchy-models --features metal,all -- -D warnings`
- [ ] linux/cuda clippy — see docs/CONTRIBUTING.md step 5
- [ ] e2e smoke (if serving/runtime touched)
- [ ] N/A — docs only

## Invariant check

<!-- See docs/CONTRIBUTING.md step 4. Confirm or explain. -->

- [ ] No new `#[allow]`, no new env-var behavior toggles
- [ ] No weakened tests, no relaxed numeric tolerances, no new `#[ignore]`
- [ ] Compile-time facts asserted at compile time (`const`/types), not at runtime
- [ ] Target crates: no net line growth (declared-data tables exempt) — delta: 
- [ ] Per-target change is opcode lowering only; no ISel added to metal/spyre
