# Testing: the five tiers, and the tests that stay tests

scratchy is a proc-macro compiler, and most of what its tests check is a fact the compiler
already knows at expansion time. #8 moves those facts out of `#[test]`s and into constants,
types and lints, where an edit can't quietly delete them. This page states the other half, the
floor that move must not cut through. Some tests encode empirical claims about hardware, about
clients we don't control, and about behaviour over time. No type can state those, and they are
not a backlog.

"No weakening tests" in the repo's `CLAUDE.md` means what this page says.

## The five tiers

Every test belongs to exactly one tier. The first four replace a test with something an edit
can't silently delete. The fifth keeps it.

| tier | what the test asserts | where it goes |
|---|---|---|
| **T1** | a compile-time fact re-derived at test time: a layout, offset, stride, arena size, slot colour, barrier class, trip count | `const _: () = assert!(…);` in the module that owns the fact. It fails the build, costs no test time, and can't be `#[ignore]`d. |
| **T2** | that two copies of the same arithmetic agree (`crates/compiler/subtile/tests/addr_one_law.rs` is the archetype) | one typed constructor and no second copy. Element counts and byte offsets become newtypes, not `usize`; residency, dtype and rank become typestate; target-ABI tables become sealed traits, not open matches. |
| **T3** | that the macro or a config parser rejects bad input (`rejects_*`, `*_is_error`, `bad_*`) | a `compile_error!` and a `trybuild` compile-fail fixture pinning the diagnostic |
| **T4** | an architectural guard that walks the source tree when someone runs it (`crates/tensors/tests/cfg_neutrality.rs`, `crates/cli/scr/tests/arch_feature_passthrough.rs`, the metal build's MTL3 guard) | a lint (`clippy.toml`, or the custom lint crate, #12 / #16), so the violation surfaces where it's typed |
| **T5** | behaviour no type can state | **stays a test, protected (below)** |

When moving an invariant out of a test, take the cheapest rung that can express it:
1. a `const` assertion;
2. a newtype, typestate or sealed trait;
3. a `compile_error!` from `#[forward]` or the config macros;
4. a `trybuild` fixture;
5. `clippy.toml`;
6. a custom lint;
7. last, `build.rs` source scanning.

## The protected set (T5)

Each zone is named by what its tests claim, not by a file list that would rot. A test belongs
to the zone when it makes the zone's claim, wherever it lives.

- **GPU numerics.** A kernel's output against a golden or reference implementation, on real
  silicon: gemm, softmax, rmsnorm, rope, quantized matvec/matmul, dequantization, attention,
  the samplers. Floating-point agreement with a reference is an empirical claim about hardware.
  - Today: `crates/targets/metal/tests/` and the metal crate's own kernel unit tests;
    `crates/targets/cuda/`; the `tools/vision_parity/` oracles;
    `crates/targets/spyre/tests/element_torchspyre_parity.rs`.
- **The wire surface.** What clients we don't control send and expect back: chat templates,
  tool and reasoning parsers, SSE streaming, the OpenAI- and Anthropic-compatible request and
  response shapes, their error codes. Compatibility with the world is an empirical claim, and
  these tests are its specification.
  - Today: `crates/serving/api/`.
- **Scheduler dynamics.** Behaviour over time under load: preemption, eviction, prefix-cache
  hits, block tables, batching, async-scheduling overlap.
  - Today: `crates/serving/scheduler/`, `crates/serving/engine/`.
- **End to end.** A real server answering real requests: serving, streaming, the quantization
  paths, multimodal, speculative decoding, LoRA, pooling, telemetry.
  - Today: `crates/e2e/`.
- **The allocator.** `crates/targets/spyre/flex-rs/` is T5 by default (#8's open question).
  Move a test out only on purpose, in its own PR, never as a side effect of another.

## What weakening a protected test means

A change to a T5 test is a weakening, and a review blocks it, when it:
- relaxes a numeric tolerance or widens an accepted range;
- adds `#[ignore]`, a skip, or an early return to a test that ran;
- deletes a golden or reference comparison, or replaces the reference with the code under test;
- swaps an end-to-end test for a unit test that asserts the same string;
- narrows the inputs (fewer shapes, dtypes, sizes, models) without saying what replaced them;
- turns an assertion into a log line, a warning, or a softer `assert!` (an equality into a
  containment, a value into "is not empty").

What it isn't:
- a fix to a test that was wrong, with the evidence in the PR: a reference that was itself
  wrong, a tolerance from the wrong dtype, a test that could never run;
- a move from one zone's file to another with the assertions unchanged;
- a T1–T4 move of a test that makes no T5 claim.

## Placing a new test

Ask what its assertion depends on.
- **Only on constants the compiler knows** (a size, an offset, a count): T1. Write the `const`
  assertion instead.
- **On two pieces of code agreeing:** T2. Remove one of them.
- **On rejecting bad input to a macro or config:** T3.
- **On the source tree's shape:** T4.
- **On hardware, on a client, or on time:** T5. Write the test, and it joins the protected set.

## Enforcement

Until the lint crate (#12) lands with `ignore_without_reason` and a tolerance-change rule, the
list above is a review obligation. A reviewer answers one question for every changed test:
"is this a T5 weakening?" The answer follows from the zone the test belongs to and the list
of what weakening means.

Parent: #8.
