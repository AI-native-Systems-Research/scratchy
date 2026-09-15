# Contributing — Scratchy PR Workflow

How we deliver PRs that actually close issues, with minimal ceremony.

Scratchy is a compiler: `dsl_math + weights → weight_loader_fn() + forward_fn()
+ instruction_tape`, with the whole pipeline running at proc-macro expansion
time. That shapes everything below — most notably, a change that *looks* like a
runtime fix is usually a change to what the macro emits.

It is also this org's **hyper-specialization** principle applied to inference
serving: a system optimized for how it is actually used in one specific
deployment, rather than a general runtime configured at startup. Two org-wide
practices follow from that and are reflected below — **spec-driven development**
(specs and plans are live documents that evolve with the system) and **governed
autonomy** (every change carries complete provenance: what, why, and evidence).
See [AI-native Systems Research](https://ai-native-systems-research.github.io/ai-native-systems-research/)
for the wider context.

## The Workflow

### 1. Worktree

Always work in an isolated git worktree. Never commit directly on main.

```bash
git worktree add -b fix/issue-1234 ../scratchy-issue-1234 main
cd ../scratchy-issue-1234
```

This keeps main clean and lets you juggle multiple issues in parallel. It also
matters more here than in most repos: builds are feature-scoped and expensive,
and separate worktrees keep separate `target/` dirs from thrashing each other.

### 2. Audit the Issue

Before writing any code, read the linked issue carefully:

- What is the actual problem or request?
- What's ambiguous or underspecified? Ask questions now, not after you've coded.
- What's the acceptance criteria — how will we know this is done?
- **If this is a sub-issue of a tracking issue:** read the parent first.
  Understand where your piece fits, what depends on yours, and what boundaries
  you must respect.

**Verify the issue's claims before building on them.** Issue bodies here carry
counts, file paths, and diagnoses that drift as the tree moves — several
currently cite numbers or files that no longer match `main`. Re-derive anything
load-bearing with your own `grep`/`ls` and correct the issue in a comment. A
plan built on a stale premise is wasted work.

If the issue is vague, comment on it to clarify scope before proceeding.

#### Issue Types — What to Focus On

**Bug fix:**
- Reproduce first. Understand the root cause, not just the symptom.
- Establish *where* it lives: macro expansion (wrong tape emitted), a shared
  pass, or one target's opcode lowering. Fixing the wrong layer is the most
  common wasted PR in this repo.
- Your test should reproduce the failure *before* your fix, and pass *after*.
- Scope: fix the bug. Don't refactor surrounding code unless it caused the bug.

**Feature (standalone):**
- Clarify acceptance criteria — what does "done" look like to a user?
- Consider edge cases and error paths upfront, not after review catches them.
- Scope: deliver the feature as described. Flag scope creep back to the issue.

**New model architecture:**
- Read [`docs/MODELS.md`](docs/MODELS.md) first. A new arch should touch
  `crates/models/arch/` plus its config, and touch a target crate *only* if it
  introduces a genuinely new opcode — then one emitter arm per target and one
  registry row, nothing more.
- Every checked-in `configs/<arch>/<stem>.json` needs its own `<stem>` feature.

**Feature sub-issue (part of a tracking issue):**
- Your PR must work independently (merge and pass CI on its own).
- Respect the interfaces/contracts defined by the parent plan or siblings.
- Don't solve problems that belong to other sub-issues — note them and move on.
- If your sub-issue reveals a gap in the tracking issue, comment there rather
  than expanding your PR.

### 3. Plan — get approval before coding

Pick the tier that matches the change. Both end the same way: **explicit
approval before you write code.**

#### Tier 1 — Micro-plan (most issues)

Write 3-5 bullets covering:

- **What** changes (files, behavior)
- **How** it works (approach, not line-by-line)
- **What tests** prove it works — and at which tier (see Tests below)
- **What you're NOT changing** (scope boundary)

**Present this to the reviewer (or issue owner) and get explicit approval before
writing code.** This avoids wasted work when the approach is wrong. A quick
"does this make sense?" saves hours of rework.

This is your contract with the reviewer. If you can't write this clearly, you
don't understand the issue yet — go back to step 2.

#### Tier 2 — Spec + plan (large or cross-cutting work)

For a new model architecture, a compiler-pass change, anything touching the
architecture invariants, or a tracking issue with sub-issues, write the org's
spec/plan pair before coding:

```
docs/superpowers/specs/YYYY-MM-DD-<slug>-design.md
docs/superpowers/plans/YYYY-MM-DD-<slug>.md
```

- **Spec** — *Purpose, Architecture, Components, Edge cases and failure modes,
  Testing, Scope — explicitly out.* The "explicitly out" section is not
  optional; in this repo it is where you commit to not growing target crates
  and not adding runtime analysis.
- **Plan** — *Goal, Architecture, Tech Stack, Spec link,* a **File Structure**
  table (`Path | Action | Responsibility`), notes on testing, then tasks as
  `- [ ]` checkboxes so progress is trackable. Name the feature scope each task
  verifies with.

These are live documents: when implementation contradicts the spec, update the
spec in the same PR rather than letting it drift. If an agentic worker will
execute the plan, say so at the top and name the sub-skill it should use, as the
org's existing plans do.

### 4. Implement + Test

#### Architecture invariants — non-negotiable

These are stated in [`CLAUDE.md`](CLAUDE.md) and a PR that violates one gets
sent back regardless of how well it works:

- **Everything is a compile-time constant.** `#[forward]` runs the whole
  pipeline at expansion and emits `static` tapes. No runtime lowering, no
  runtime analysis, no mirror types.
- **Everything common is shared.** One implementation of every target-neutral
  pass, consumed by both spyre and metal. Target-ABI facts are const tables the
  shared passes take as *input* — never logic, never scattered match arms.
- **Per-target surface = opcode lowering only.**
- **No instruction selection** for spyre or metal. ISel is cuda-gated and stays
  that way.
- **Metal 4 only.** Classic MTL3 command buffers are banned — enforced by
  `crates/targets/metal/build.rs` (`guard_no_classic_mtl3`) and by
  `disallowed-methods` in `clippy.toml`.
- No net line growth in target crates (declared-data tables exempt); no new
  env-var behavior toggles; no `#[allow]`; no weakening tests.

#### Build — scope to one model

Pick exactly one backend (`metal`, `cuda`, `spyre` — mutually exclusive) and
name at least one model. Naming zero models is a **build-time panic**, not a
silent empty binary.

```bash
cargo build --release -p scratchy-cli --features metal,model/smollm2-135m
```

**When iterating, scope to the single smallest model that exercises your
change** — `model/smollm2-135m` (135M, what CI's e2e smoke uses) or
`model/llama-3.2-1b`. Never build `model/<arch>` or `model/all` locally just to
test one thing: that forward-expands every config in scope, which is
minutes-to-hours and can OOM a laptop. Add `-Fquant/<preset>` only if the change
is quant-specific. See [`docs/BUILD.md`](docs/BUILD.md) for the full mechanics.

#### Tests

The bar: **if someone reverts your fix, a test should fail.** You don't need
strict TDD.

Which *kind* of test is a real decision here, not a formality. Because the
compiler decides layouts, strides, arena sizes, trip counts and kernel selection
at expansion time, a runtime `assert_eq!` about any of those is checking at
runtime a fact the compiler already knows — prefer `const _: () = assert!(…)`
beside the `const fn` that produces the value, or a type that makes the bad
state unrepresentable. For rejection laws, prefer a `trybuild` compile-fail
fixture over a unit test that pokes the macro's internals, because what a user
experiences is the diagnostic.

But some tests are load-bearing and must stay runtime tests: GPU numerics
against golden references, wire-format compatibility with clients we don't
control, scheduler dynamics under load, and end-to-end serving. These encode
empirical claims about hardware and about the world; no type asserts them.
**Never relax a numeric tolerance or add `#[ignore]` to make a failure go
away** — that is the one change most likely to look reasonable and be wrong.

See issues #8 and #13 for the tiering rationale and the protected set.

### 5. Verify like CI before you push

`.github/workflows/rust.yml` has three jobs. Reproduce the ones your change can
break:

```bash
# Format (fmt job) — cheap, run it always
cargo fmt --all -- --check

# macOS / metal job
cargo clippy -p scratchy-models --features metal,all -- -D warnings

# macOS / metal job, the hf-completions path (`all` doesn't enable it, so this
# is the only lint coverage the build-time HF query gets). Needs network.
cargo clippy -p scratchy-models --features metal,smollm2-135m,hf-completions -- -D warnings

# Linux / cuda job
SCRATCHY_GPU=h100 CUDA_COMPUTE_CAP=90 SCRATCHY_SKIP_CUDA_KERNELS=1 \
cargo clippy --workspace --features cuda,scratchy-models/all \
  --exclude scratchy-target-metal \
  --exclude scratchy-target-metal-compiler -- -D warnings

# e2e smoke, as CI runs it
cargo test -p scratchy-e2e -p scratchy-models \
  --features e2e,metal,scratchy-models/smollm2-135m \
  --test e1_basic_serving test_t1_smollm -- --ignored --test-threads=1
```

Both `all`-scoped clippy runs are the full-scope case: every config, every arch.
Expect them to be slow and RAM-heavy — that's CI-scale cost, not a regression.

Note the two `--exclude`s on the cuda gate are both required: `scratchy-target-metal`
and `scratchy-target-metal-compiler` each depend unconditionally on Apple-only
`objc2`, and `--workspace` builds every member as a root regardless of
`--features`, so excluding one still drags `objc2` onto Linux.

The x86_64 cuda gate reproduces in a container (see [`CLAUDE.md`](CLAUDE.md)
for the full `docker run`); Apple Silicon runs it amd64-emulated — slow but
faithful. The macOS metal job runs locally via `scripts/act-local.sh`.

Commit atomically — each commit should be a coherent unit.

### 6. PR Description Checklist (Medium+ PRs)

This is the provenance record for the change — what, why, and evidence. It must
include:

1. **Context** — what is this about in the big picture? A few sentences linking
   to the issue/tracking issue.
2. **What the PR delivers** — what changed, concretely (files, behavior, features).
3. **Input/output proof** — the exact command to run and what its output
   demonstrates, including the `--features` you built with. A reviewer on
   different hardware needs the feature scope to reproduce you at all.
4. **No-regression proof** — which existing tests still pass, and what
   guarantees previous behavior is unchanged.
5. **Invariant check** — if you touched a target crate, the line-count delta and
   which invariants from step 4 you considered.

## Commits & PR Titles

Use [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for
both, scoped to the crate or area touched:

```
fix(metal): dispatch rope through mtl4_dispatch
refactor(subtile): fold address formulas into addr::Nest
docs(build): document quant preset replacement semantics
```

## Principles

- **Understand before acting.** A PR that doesn't close the issue is wasted work.
- **Scope is sacred.** Fix the issue, nothing more. No drive-by refactors —
  especially in target crates, where line growth is itself a review failure.
- **Constants over assertions.** If the compiler knows it, prove it at compile
  time. A test is advisory; a type is load-bearing.
- **Tests prove intent.** Not coverage for coverage's sake — proof that the fix
  works.
- **No overengineering.** Three lines of straightforward code beats an abstraction.
- **Experiment, don't guess.** Performance claims need a measurement, a baseline,
  and stated hardware. Explore a space of candidates rather than shipping the
  first plausible fix, and label hypotheses as hypotheses until measured.

## Licensing

By contributing you agree that your contributions will be licensed under the
project's license. See [LICENSE](LICENSE) for details.

## Questions

Open an issue or contact the maintainers.
