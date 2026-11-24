---
name: scratchy-purity-audit
description: Audit a scratchy PR, branch or diff for Rust/compiler purity — everything is a constant; constants reach kernels as kernel compile-time constants; each invariant expressed once, as a type every code path shares (never re-stated as a runtime check along another path); no runtime refusals, ever; newtypes and non-cosmetic const generics; a derived compiler ladder with no per-model hacks and no cost model fit to a model; target surface = opcode lowering only, with zero target material outside the target homes; the device is a type of its constants; one source of truth; lock and comment hygiene; proper in-tree tests, never standing in for a type. Use when asked to audit, review or judge whether code "obeys the scratchy architecture", is "good scratchy", "pure", "locked down", or "high quality Rust".
---

# scratchy purity audit

The question this answers: **is the code a compiler, or hand-tuned code wearing a compiler's clothes?**
It audits architecture, not behavior. The e2e gate (release build + `scr chat` at the tip),
perf A/Bs and numerics are separate audits; mention them only to say they are separate.

Read the source yourself. No subagents, no workflows. Every finding is a line you read, at the
audited SHA, never an unread grep hit.

## 0. Pin what you audit

```bash
head=$(gh -R AI-native-Systems-Research/scratchy pr view <N> --json headRefOid --jq .headRefOid)
up=$(git remote -v | awk '/AI-native-Systems-Research\/scratchy(\.git)? \(fetch\)/ {print $1; exit}')
git fetch "$up" "+pull/<N>/head:refs/remotes/pr/<N>"   # `+`: a force-pushed head replaces the old ref
git fetch "$up" main
sha=$(git rev-parse refs/remotes/pr/<N>)
[ "$sha" = "$head" ] || echo "STOP: fetched $sha, the PR head is $head"
git diff --stat=200 "$up/main...$sha" -- . ':!**/third_party/**' ':!**/thunderkittens/**'
s=.claude/scratch/purity-<N>-${sha:0:8}
rm -rf "$s/src" && mkdir -p "$s/src" && printf '*\n' > "$s/.gitignore"
git archive -o "$s/src.tar" "$sha" <dirs…>
tar -xf "$s/src.tar" -C "$s/src"
```

- If it prints `STOP`, audit nothing: the fetched ref isn't the PR's head. Fetch again until `sha` equals `headRefOid`.
- `$up` is whichever remote fetches `AI-native-Systems-Research/scratchy` (`upstream` in some clones, `origin` in others).
- Scratch lives in the worktree, in a directory its own `.gitignore` (`*`) hides. Never `/tmp`: macOS culls it.
- The scratch path carries the head SHA, and `src` is emptied before extraction. A re-audit after a force-push never cites a file the new head deleted, and two audits never share a tree.
- Every `file:line` is at the pinned `headRefOid`; say so in the report's first line.
- Skip vendored trees. Do read hand-written kernel sources (`.cu`, `.metal`, `.cuh`).
- Run git from the worktree with plain, separate commands (no `cd` into other paths).
- **Read in this order:**
  1. the step and tape types;
  2. the facts types the target reads;
  3. the lowering;
  4. the target-private passes;
  5. the player and worker runtime;
  6. the kernel launch ABI and kernel sources;
  7. the root-side (shared-crate) changes;
  8. build scripts.
- **Main's precedent is the bar.** Compare against:
  - `crates/targets/metal/src/tape/{step.rs,ids.rs,kernel_constants.rs}`: typed steps, newtyped ids, typed function-constant structs;
  - `crates/compiler/ir/src/model_facts.rs`: `HeadDim(NonZeroU32)`, `Class<Kv, Hd, Rot, Page>` with laws as trait bounds, `Gdn`, `RopePairing`.

  A PR below main's bar is a finding even if it's internally consistent.
- **Main's known debt is not precedent.** Where main itself breaks a rule (the rule says so, as P9 does), a PR that copies or extends that code is a finding. Don't cite it as the bar, and don't flag main's existing code in a PR's audit.

## 1. The rules

Each rule lists the tells (what to grep) and the fix shape to name. Rank fixes as:
unrepresentable by structure > rustc type or bound > compile-fail test (nightly checks the error
code) > anything at runtime. For rules P3 and P4 a runtime check is never an acceptable fix.

### P1. Everything is a constant; `forward_fn` plays the tape and nothing else
`#[forward]` pins the model, quant and workload rung at build time, so there is no runtime model quantity.
- What varies is a rung: one baked tape per rung. The runtime only picks the rung and binds per-step data (live request count, page tables, positions, token ids, histories, step index).
- A kernel never re-derives a rung fact from live data. For example, a split count chosen from `kv_len` is a runtime decision in disguise.
- No load-time measuring, no runtime lowering or analysis, no mirror types.
- **Tells:**
  - kernel or variant choice inside the player (`match` guards on shape fields, `*_takes()`/`*_split*()`/`is_*()` predicates called from `play`/`forward`/`enqueue`);
  - `validate()` called in `new`/`build` on `&'static` tapes;
  - scratch or split sizes recomputed at play time.
- **Fix:** the step records the chosen kernel, split and tile as its variant or typed field, emitted once at expansion.

### P2. Every constant reaches the kernel as a kernel compile-time constant
Every model, quant, workload and device constant is a kernel constant:
- **Metal:** a `[[function_constant]]` slot set from a typed per-kernel struct (`kernel_constants.rs`).
- **CUDA:** a template argument of an instantiation generated from the tapes at build time (the models crate's `build.rs` already expands the selected models).

Forbidden:
- runtime shape arguments to a kernel, plus the launcher-side shape refusals that come with them;
- hand-picked instantiations in kernel source (`EXPERTS_LARGE`/`EXPERTS_SMALL`, one file per head dim);
- constant tables indexed at runtime;
- grid or threadgroup reads of a value the lowering knows;
- a constant stated twice (a Rust `const` and a kernel `constexpr`/`#define`, or the same `constexpr` in several kernel files).

The one sanctioned exception is a staging LUT that an A/B showed faster, and only when the user chose it.

- **Tells:**
  - `extern "C"` launchers taking `(m, n, k, …)`;
  - `constexpr` / `#define` values equal to a Rust const;
  - near-identical kernel files differing in one number;
  - `switch (experts)` / `if (head_dim == …)` in launchers.
- **Fix:** a generated instantiation unit, with each step naming its typed instantiation. Shape rules become `static_assert`s.

### P3. An invariant is expressed ONCE, as a type every code path shares
**The same invariant is never described more than once, along any number of code paths.** It is expressed once, as a type, and every path that relies on it takes that type. If fn A and fn B both need invariant I, then:
- a type T carries I;
- T is minted by one constructor (one law), in one place;
- A and B both take T, and neither re-states I.

A runtime `if` check is the commonest second description: the code path re-states, as a check, an invariant some other path already established. **No runtime `if` checks enforcing invariants.** A second description in any other form (a duplicated predicate, a mirrored constant, a re-validation, an assertion) is the same violation.

Forbidden:
- the same predicate evaluated in two layers (lowering and player, Rust and kernel, expansion and load);
- a Rust mirror of a kernel rule ("the launcher's rule", "mirrors `split_chunks`");
- re-validation of something already validated;
- `if`/`match`-guard checks that re-establish an invariant;
- `.expect("… already checked …")`, `.expect("`new` made …")`, `unreachable!` for a state a type should exclude, and `debug_assert_eq!(runtime, CONST)`;
- paired `Option`s whose `Some`-ness must agree (`tape.kv: Option<_>` beside `pool: Option<_>`);
- a struct shared by several variants whose meaning differs per variant (a "with m = 1" doc comment);
- a field "`None` on every other step".

**Method:** trace at least one invariant end to end and count every place it is described (Rust const, kernel constant, expansion check, validate, runtime launcher check, `expect`, test copy). Each path that re-states it counts once. The count is the finding. The fix gets it to one type, minted once.

- **Tells:** `rg -n 'expect\("|unreachable!|debug_assert|already (checked|refuses)|validate\(\)|the launcher.s rule|mirrors' <dirs>`
- **Fix shape for "every member must be handled"** (each buffer made resident, each field bound): destructure the struct with no `..` rest pattern, so a new field fails the build until it is handled. Never a per-site check that the list is complete.

### P4. No runtime refusals. Ever.
A runtime refusal is a deliberate runtime rejection of a model, shape, dtype, device, config, request parameter or capability:
- `Err(…Shape…)`, `ERR_SHAPE`, `bail!`, `anyhow!("unsupported …")`;
- a 4xx for a capability the backend lacks (sampling, logprobs, a cache dtype);
- `NotH100`-style device gates;
- `Missing`/`Unused`/`TooSmall`/`Disagrees` checks between artifacts the build produced together.

Two fixes only:
1. make it unrepresentable at build time (type, bound, or expansion-time compile error);
2. **build the missing capability.**

Never refuse, never fall back, and never add a gate that disables a path. A refusal is the deliverable not existing, dressed up as engineering judgement.

These are **not** refusals under this rule, and they are the only ones:
1. the runtime `config.json` → registry lookup that identifies which compiled model a checkpoint is (it stays, untouched);
2. the one user-approved data→type door (#1094): ONE loader function mints the typed weight from the checkpoint header and checks its shape there, and everything downstream takes the type;
3. true runtime failures that are not decisions: driver/launch errors, I/O, OOM, malformed wire JSON.

Anything else, including a second checkpoint check outside that one door, is a violation. When unsure, it is a violation.

- **Count them, classified.** Find every error enum in the runtime crates (`rg -n '^pub enum \w*(Error|Refusal)\b' <runtime crates>/src`). That includes the ones raised at load, such as a tape validator or a checkpoint loader, not just the launch and forward enums. Read every variant's doc and sort it into exactly one class:
  - **refusal**: a deliberate rejection of a shape, model, config, request parameter, capability or invariant. Only these count against P4.
  - **operational**: a true runtime failure that is not a decision (`Cuda(cudaError_t)`, a caught host exception, no device present, I/O). Exempt (exception 3). List these separately so the count reproduces, but never add them to the P4 count.
  - **wrapper**: any variant whose payload is another error enum (`Launch { error: KernelError }`, `Tape { error: TapeError }`, `Prepare { error: KernelError }`). Count it under the inner enum, never twice.

  A loader's checkpoint-header checks are exception 2 only if that one function mints a typed weight that everything downstream takes. If anything downstream re-checks the weight (its bytes, its shape, its presence), there is no door: the header checks count as refusals, and so does the re-check.

  Report `<enum>: N refusals (+ M operational, W wrappers)` per enum, with the total of refusals.
- **Also count:** shape-refusal sites in kernel sources (every form: `return ERR_SHAPE` and ternaries), `Result<(), String>` capability hooks, and refusal seams added to shared serving code.
- Report counts plus the worst examples, not a list of all of them.

### P5. Newtypes for every quantity; no raw scalars; no strings
- No naked `u32`/`u64`/`usize`/`f32`/`bool`/`&str` quantity crosses an API: `m`, `n`, `k`, `rows`, `cols`, `head_dim`, `kv_heads`, `top_k`, `bucket_m`, eps and scale are each their own type.
- A newtype with a `pub` inner field (`struct X(pub u64)`) is not a newtype. The one checking constructor is the only way in, and `get()` is the only exit.
- Floats stored as bits (`eps_bits: u32`, `theta_bits: u64`) are a raw scalar. Use a newtype that implements `Eq` over its bits internally.
- No sentinels: no NaN for absent, no `0` for none. Use `Option` or a type.
- No stringly-typed values: no `&[&str]` tables, `dim: "m"`, refusal `String`s carrying structure, `Vec<(_, String)>`, or arch/stem names past the one parse boundary.
- `bool` parameters that select behavior (`fn f(fp8: bool)`) become an enum or a type parameter.

### P6. Const generics: non-cosmetic, never demoted
- Anything knowable at compile time is a const generic or associated const whose law fails the build. Prefer it over a newtype.
- **Deletion test:** if the parameter can be deleted and the code still compiles, it is cosmetic. `Vec` plus a runtime count beside a const naming that count is always this bug.
- Shape constraints are **rustc type constraints**: shapes as const-generic or marker parameters on typed steps, with producer out == consumer in as the same type. `const _: () = assert!(..)` over emitted values is not the goal.
- Every lockdown is two-sided: hardware bounds the tile, and **the tape's own extent pins the product**. Watch for "did the legal set get bigger or the emitted extent smaller" (type weakened, or a degenerate program that satisfies an honest type).
- **A PR that demotes a const generic or newtype to a value is changes-requested; that is the verdict, not a finding.** This includes routing around a typed fact by adding a parallel raw-value copy, for example a second facts struct of `u32`s beside `model_facts::Class<…>`, or a second `RopePairing`.
- **A law names its numbers.** A build-time failure states the declared value, the input it came from, and which side broke ("192× the declared 64, from `n`"), never just "invalid".
- **A per-model generic design shows its cost.**
  - It measures cold build time and peak RSS against main, each in a fresh target dir, attributed with `--timings`, before it lands.
  - A compile-time regression is never answered by lowering opt-level or disabling an optimization.
  - The end state is ONE representation shared by every target and the runtime; a design that adds another copy of the facts is not progress.

### P7. A real compiler ladder: derived, never declared per model
Enforced in part by the p7_* lint in lints/scratchy-purity (scripts/purity.sh); a finding there is the verdict.
- No model, arch or stem names in logic: `match arch`, `.contains(&arch)`, `&[&str]` arch or preset allowlists, `RouteProgram::Qwen*`, `GEMMA_PROGRAM` template flags. Name the math ("top-k, scale, softmax"), never the model.
- A fact that follows from configs the compiler loads is derived in ONE shared place: no new `overrides.json` hand entries, no arch branches, no model-wide "has X anywhere" proxies.
- A kernel family covers the general shape space through instantiation, with a rung for off-tile shapes. If coverage is exactly the shapes of the checkpoints in hand, that is the finding: head dims {256, 512}, expert counts {128, 256}, "a tiny parity config's GEMMs sit off the tiles".
- A rule chosen by name when the shape already decides it (picking a tile by arch name rather than by `n`) is a finding.
- A checked-in model `config.json` is the verbatim checkpoint config: never edited. A sizing or capacity bug is never fixed by bumping a per-model constant until one test passes. That's reward hacking: the fix derives the value once, from the deployment or the config, for every model.
- **A cost model is derived, never fit (2026-10-01).**
  - An optimizer's cost inputs are tape facts (tiles, threads, bytes, dependencies) plus device facts measured by synthetic microbenchmarks with no model in them.
  - It is validated on several models, never fed one.
  - Each of these is reward hacking:
    - a curve fit to one model's measurements;
    - a cost table measured on one checkpoint;
    - an "optimizer" that reproduces hand-picked widths.
  - A hand-tuned constant is named as one, up front.
- **Tells:**
  - doc comments justifying a rule with a model ("Gemma's", "the 27B's");
  - tables of per-model shapes in comments;
  - the shared handoff carrying `arch: String` "for per-arch tables".

### P8. Shared vs target: the target's own surface is opcode lowering only
- Target ABI facts are **const tables** that the shared passes take as input. They are never logic, and never scattered match arms.
- No target-private passes after opcode lowering: fusion, liveness, arena or scratch placement, kernel re-selection, inserted ops. Each of those is a shared pass with a table row, or an op in the shared tape (e.g. activation quantization).
- The lowering is a dumb IR→tape translation; every transform is a pass. No instruction selection.
- Target material stays in its target's home (`crates/targets/<t>/` or `backends/<t>/`), and neutral code stays out of every target's home:
  - **Zero target material outside the target homes (2026-11-12).** That means no target `cfg` and no dependency on a target crate. It also rules out any target-named feature, and any target-specific field or type on a neutral item (`ktir_bundle`/`sengraph_bundle` on the neutral registry was the example). Main's `crates/tensors/tests/cfg_neutrality.rs` checks only the `cfg` part. The auditor reads the rest.
  - Neutral code never lives in a target-named crate. A backend's feature is never on for another backend.
- No new env-var behavior toggles.

### P9. The device is a type of its constants
- **The ruled design (2026-10-09):** the target is the chip's facts as const generics, `AppleGpu<const GEN: u32, const ULTRA: bool>` (the two facts `MTLDevice.architecture.name` carries). Every setting is a formula of them, read in the lowering as `T::…` with `T: MetalTarget`. There's one tape per variant, and the device picks its own at load.
- **Main doesn't have it yet; this is known P9 debt.** Main's Metal device is `MetalTargetProfile`, a runtime struct of `f64` facts (`crates/targets/metal/src/tape/targets.rs:27`), picked by `profile_for(class: GenClass)` (`crates/targets/metal/compiler/src/static_tape.rs:1261`; `GenClass` at `crates/targets/metal/src/tape/lowered.rs:3155`). That is the class enum this rule forbids. A PR that adds a profile field, a `GenClass` arm or a copy of the pattern for another target is a finding. A PR that moves it toward `AppleGpu<GEN, ULTRA>` is the fix.
- No hard-coded device numbers in Rust or kernel source (`H100_SXM_SMS = 132`, `TARGET_CTAS = 264`), no per-chip hand tables, and no class enums or bands.
- **Metal is MTL4 only.** Not a single line of Metal 3, not even in a test, an experiment or a microbench.
  - Main defines Metal 3 by the classic call, not the type. That means a command buffer from `queue.commandBuffer()` (MTL4 uses `device.newCommandBuffer()` and `beginCommandBufferWithAllocator`), or a classic `computeCommandEncoderWithDispatchType` or `memoryBarrierWithScope`.
  - Main enforces the first: `guard_no_classic_mtl3` fails the build (`crates/targets/metal/build.rs:403`), and `clippy.toml:15` bans `MTLCommandQueue::commandBuffer`.
  - A classic call those checks miss (another receiver, or a file outside the guard's scanned directories) is still a finding.
  - A plain `MTLCommandQueue` handle, like main's `MetalDevice.queue`, is not a finding.

### P10. One source of truth
- The same edit needed in two callers means the duplication is the bug: consolidate the routine, don't share a helper between the copies.
- A test that checks a copy against a copy can't catch drift; that counts even under `cfg(test)`.
- An invented schema in a data file (magic keys, `kind` strings, a second copy of the loader's naming) is a smell to reject.

### P11. Lock and comment hygiene
Report these as one grouped line after the rules above.
Enforced by the `p11_*` lints in `lints/scratchy-purity` (`scripts/purity.sh`); a finding there is the verdict.
- No `#[allow]` anywhere: not at a site, and not in a `[lints]` table. Find the design the lint doesn't flag. (Main's `[lints]` tables only declare check-cfg.)
- No escape-hatch constructors (`from_untyped`, `conflated_with_*`, a `pub` raw constructor on a checked type).
- No weakened tests or ratchets ("did the number get smaller?").
- Comments describe the code as it is, in the present tense:
  - no tombstones ("X used to be here", "these checks are gone");
  - no refactor narration ("now lives here", "moved from X");
  - no `/tmp/…` plan references.

  That applies to `Cargo.toml` comments too.

### P12. Proper tests: in-tree Rust, never standing in for a type
- A test is never written where a type could carry the fact (the ranking at the top of §1).
- Type proofs are `compile_fail` doctests that name their error code. Only nightly rustdoc checks the code (stable ignores it), so they're proved by the nightly run.
- No out-of-tree gate pipelines (Python or shell comparers, tape dumpers, ad-hoc gate scripts): "we are a rust project; we need proper tests".
  - Instead, the separate gates named at the top of this file check what a PR claims: CI, the nightly doctests, `scr chat` and a quiet perf A/B. This rule judges only that no out-of-tree pipeline stands in for them.
  - "Identical output" means an `OUT_DIR` diff of the emitted code against main.
- Not proper tests:
  - a test pinning a fake layer's error strings;
  - a test restating a helper;
  - a test running a mint on null or fake tensors;
  - a test checking a copy against a copy (P10).
- A golden is a hash (one line per model), never a dump. It is blessed by regeneration, never hand-edited.
- A test that "X changes the answer" first shows that not-X doesn't.

## 2. The report

```
**Purity audit at `<sha8>`** — <one-line verdict: compiler or not>.

| rule | verdict |
| P1 everything is a constant | met / half / not met |
…

### P<k>. <rule>: <verdict>
- **<claim in bold>** — `path:line` (and main's precedent it falls below, `path:line`). <fix shape, one sentence>.
```

- Lead with the verdict, then the rules that fail, worst first. Rules that are met get one line in the table, not a section.
- **Big things only.** Every finding is wrong behavior or an architecture violation. No clippy, no fmt, no style, no restating what the diff obviously does. Findings are durable, so no "CI pending".
- Give counts where the rule is about multiplicity (P3 derivations of one invariant, P4 refusal sites).
- End each failing rule with the fix shape, not a patch.
- Separate concerns the audit didn't cover (e2e, perf, numerics) get one closing line at most.
- **Don't post unless asked.** When asked to post:
  1. re-read `headRefOid`; if it moved, re-verify the cited lines first;
  2. use `gh -R AI-native-Systems-Research/scratchy pr comment <N> --body-file <file>` for "for discussion", or `gh pr review --request-changes` only when asked for a review (on the user's own PR, a comment saying "changes requested");
  3. rewrite "you"/"your" so the text reads as the user's own comment;
  4. no bare `@name`.

## 3. Worked example: #978 (CUDA on ThunderKittens) at `99c60d50`

- **P1:** the player picks kernels at play time: `gemv_splits` in 9 places, plus `norm_row_takes`, `attn_decode_split` and `gdn_scan_is_recurrent` in `runtime/worker.rs`. `CudaTape::validate()` runs at expansion (`compiler/src/emit.rs:1085`) and again in `CudaWorker::build` (`worker.rs:840`).
- **P2:** every `.cu` is compiled once, the same for any model. The kernels are already templates, but their instantiations are hand-picked, with eight attention files for head dims 256 and 512. `H100_SXM_SMS = 132` is copied into 3 files.
- **P3:** the GEMM tile rule is stated or checked in 5 places: Rust `GEMM_BF16_TILE`, the `.cu` `M_TILE`/`N_TILE`/`K_TILE`, `gemm_kernel` and `gemm_shape` at expansion, and the launcher's runtime `ERR_SHAPE`. Pool presence is two paired `Option`s, re-checked by `WorkerError::*PoolMissing/Unused` and 7 `.expect("`new` made a pool …")`.
- **P4:**
  - 99 runtime shape-refusal sites across all 44 `.cu` launchers: 98 `return SCRATCHY_CUDA_ERR_SHAPE` plus one in a ternary (`gemm_fp8.cu:859`). Grep for both forms;
  - **126 refusal variants** across the eight runtime error enums in `backends/cuda/crates/{cuda,worker}/src`, classified:
    - `KernelError`: 19 refusals (+ 2 operational: `HostException`, `Cuda`);
    - `WorkerError`: 11 refusals (+ 1 operational: `Cuda`; 2 wrappers: `Tape`, `Launch`);
    - `ForwardError`: 26 refusals (+ 1 operational: `Cuda`; 1 wrapper: `Launch`);
    - `PlanError`: 16 refusals;
    - `DeviceError`: 1 refusal, `NotH100` (+ 2 operational: `Cuda`, `NoDevice`; 1 wrapper: `Prepare`);
    - `TapeError` (`tape.rs:583`, raised at load by `validate()` at `worker.rs:840`): 45 refusals (+ 1 wrapper: `QuantChain`);
    - `QuantChainError` (`step.rs:439`): 2 refusals;
    - `LoadError` (`weights.rs:331`): 6 refusals: `TableLength`, `Missing`, `PerExpertTensors`, `Dtype`, `Shape` and `Bytes` (+ 2 operational: `NoCheckpoint`, `Upload`). The header checks are not the #1094 door, because the loader mints no type and `WorkerError::{MissingWeight,WeightTooSmall}` re-checks the bytes.

    The raw totals (21 + 14 + 28 + 16 + 4 + 46 + 2 + 8 = 139) = 126 refusals + 8 operational + 5 wrappers;
  - greedy-only sampling refused per request through a new stringly `WorkerFactory::check_sampling -> Result<(), String>` seam in shared serving.
- **P5–P7:** bare-`u32` step fields; a parallel raw `TargetModelFacts` beside the typed `model_facts::Class`; `CUDA_CAPABLE`, `CUDA_PRESETS` and `CUDA_GEMM_COL64_ARCHS` string allowlists; `RouteProgram::{QwenSoftmax, GemmaScaledSoftmax}`.
- **P8:** CUDA-private post-lowering passes (`gemv_tc`, `quantize_fp8_inputs`, `fp8_moe_prefill`, `fold_norm_quants`).
