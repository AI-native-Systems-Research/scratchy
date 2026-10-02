# `crates/triton/` — the Triton ladder, re-hosted

## What is here

| crate | what it is |
|---|---|
| `triton-frontend` | Triton `.py` → TTIR, in Rust. The ruff Python parser is behind the `ruff` feature, gated by `src/py/census.rs` (refuses out-of-census constructs by name at their source line). |
| `triton-ktir` | TTIR value → KTIR value: the nine KTDP/KTDF passes, ported for behaviour. |
| `triton-numeric` | The numeric verification harness: `.py` → TTIR → KTIR → `ktir-emulator`, scored against checked-in torch reference data in `test-numeric/`. |
| `triton-ktir-superdsc` | The adapter: our KTIR value → `ktir-superdsc`'s `KtirNode` + per-kind entry points, plus the case table (`cases.rs`) and the `bake_py` example that drives the whole chain. |

Test data (relocated from triton-spyre's `third_party/spyre/test/`):
`test-fixtures/` (kernel `.py`), `test-goldens/` (KTIR + TTIR goldens),
`test-experiment1/` (the C++ chain's checked-in KTIR), `test-numeric/`
(torch reference bytes + `meta.json` provenance per config).

## Where it came from, and the split

Re-hosted 2026-10-02 from **triton-spyre** (`github.com/ai-inference/triton-spyre`,
worktree `paged-attn-lowering`, branch `demo-latest-plus-walk`, tip `5c51a1d7a`,
pushed through `30016b28e`) — a Triton Conference 2025-era fork whose spyre
effort lives under `third_party/spyre/rust/`.

**The split, agreed with Nick 2026-10-02:**

- **scratchy owns the lowering crates** (the four here). This tree already
  owns `ktir-superdsc` — the KTIR→SuperDSC lowering triton-spyre vendored
  and re-vendored from us (at `ec18ac1da`, local delta zero, PRs #197/#215
  carried the door extensions back into main). The re-host completes the
  arrow reversal: one home for the whole Triton→SuperDSC ladder.
- **triton-spyre keeps the proof**: fixtures' numeric data generation
  (`gen_numeric_data.py`), card evidence, scorers, the parity table. It
  consumes these crates as a git dependency.
- The vendoring-management layer (their `vendor/PROVENANCE.md`, re-vendor
  queue, byte-identity controls) dissolves for new work on their side; the
  12-config byte-identity bar becomes *their* CI gate on *our* crate changes.

## The copy rules

1. **Edits to these crates happen HERE, and flow out** — never edit a
   triton-spyre-side copy. If a change originates there (as `5c51a1d7a`'s
   `OperandOrigin` call-site did), it lands here first or is ported here in
   the same change.
2. **No `[workspace]` tables**: these are members of scratchy's workspace.
   The offline-pod concern that made each crate its own workspace in
   triton-spyre is handled by feature discipline: everything except the
   Python parse step is dependency-free (`cargo test -p triton-frontend
   --no-default-features` still works); `ruff` is opt-in and pulls 84
   packages, so nothing offline-critical depends on it.
3. **Path roots**: every crate resolves test data via one `.parent()` from
   `CARGO_MANIFEST_DIR` to `crates/triton/` — the `test-*` directories are
   siblings of the crates, not inside any of them.
4. **Float constants are bit-pinned**: `0.011271055`, `1.44269504`,
   `0.0078125` and friends reproduce Python literals; their f64 rounding
   is golden-visible (see `triton-frontend/src/opt.rs`'s notes on
   `1.44269504` vs `LOG2_E`). Never "clean them up".

## Known-stale goldens (pinned, not hidden)

`attention_flash_noncausal`'s TTIR golden does not match the current front
end — **11 findings, identically, at triton-spyre's own clean tip
`5c51a1d7a` and here** (descriptor mangling `TDfp16` vs `Pfp16`, constant
placement). The re-host reproduces the source's failure exactly, which is
the fidelity proof. The staleness is pinned, fail-closed in both
directions:

- `triton-frontend/tests/fixture_status.rs` — `Expect::MatchesStaleGolden`
  (fails if the count moves to 0, meaning the golden was regenerated and
  the entry should go back to `Expect::MatchesAt`, or grows, meaning drift)
- `triton-frontend/tests/golden_diff.rs` — `check_case_stale_on`
- `triton-frontend/tests/divergence.rs` — the one test that cannot express
  a pin (it asserts an exact match as its FIRST step) carries `#[ignore]`
  with this file's provenance in its doc comment.

When triton-spyre regenerates the goldens, un-pin all three.

## What is deliberately NOT here

- Their `triton-superdsc/` and `triton-superdsc-lower/` crates (their own
  parallel KTIR→SuperDSC lowering, never card-proven, kept as reference) —
  superseded by our `ktir-superdsc`.
- The pod scorers and card-run harness (`third_party/spyre/test/pod/`) —
  the proof side of the split.
