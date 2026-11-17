---
name: scratchy-purity-audit
description: Audit a scratchy PR, branch or diff for Rust/compiler purity — everything is a constant; constants reach kernels as kernel compile-time constants; each invariant derived once into a type; no runtime checks and no runtime refusals, ever; newtypes and non-cosmetic const generics; a derived compiler ladder with no per-model hacks; target surface = opcode lowering only; the device is a type of its constants. Use when asked to audit, review or judge whether code "obeys the scratchy architecture", is "good scratchy", "pure", "locked down", or "high quality Rust".
---

# scratchy purity audit

The question this answers: **is the code a compiler, or hand-tuned code wearing a compiler's clothes?**
It audits architecture, not behavior. The e2e gate (release build + `scr chat` at the tip),
perf A/Bs and numerics are separate audits; mention them only to say they are separate.

Read the source yourself. No subagents, no workflows. Every finding is a line you read, at the
audited SHA, never an unread grep hit.

## 0. Pin what you audit

```bash
gh -R AI-native-Systems-Research/scratchy pr view <N> --json headRefOid,title,body,additions,changedFiles
git fetch upstream pull/<N>/head:refs/remotes/pr/<N>
git fetch upstream main
git diff --stat=200 upstream/main...refs/remotes/pr/<N> -- . ':!**/third_party/**' ':!**/thunderkittens/**'
mkdir -p /tmp/purity-<N>/src && git archive refs/remotes/pr/<N> <dirs…> | tar -x -C /tmp/purity-<N>/src
```

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

### P3. An invariant is derived ONCE, into a type
If fn A and fn B both need invariant I, a type T carries I. T is minted by one constructor (one law), and both take T. Forbidden:
- the same predicate evaluated in two layers (lowering and player, Rust and kernel, expansion and load);
- a Rust mirror of a kernel rule ("the launcher's rule", "mirrors `split_chunks`");
- re-validation of something already validated;
- `if`/`match`-guard checks that re-establish an invariant;
- `.expect("… already checked …")`, `.expect("`new` made …")`, `unreachable!` for a state a type should exclude, and `debug_assert_eq!(runtime, CONST)`;
- paired `Option`s whose `Some`-ness must agree (`tape.kv: Option<_>` beside `pool: Option<_>`);
- a struct shared by several variants whose meaning differs per variant (a "with m = 1" doc comment);
- a field "`None` on every other step".

**Method:** trace at least one invariant end to end and count every place it is stated or checked (Rust const, kernel constant, expansion check, validate, runtime launcher check, test copy). The count is the finding. The fix gets it to one.

- **Tells:** `rg -n 'expect\("|unreachable!|debug_assert|already (checked|refuses)|validate\(\)|the launcher.s rule|mirrors' <dirs>`

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

- **Count them:**
  - error-enum variants in runtime crates (`rg -n 'pub enum \w*(Error|Refusal)'` then count variants);
  - shape-refusal returns in kernel sources;
  - `Result<(), String>` capability hooks;
  - refusal seams added to shared serving code.
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

### P7. A real compiler ladder: derived, never declared per model
- No model, arch or stem names in logic: `match arch`, `.contains(&arch)`, `&[&str]` arch or preset allowlists, `RouteProgram::Qwen*`, `GEMMA_PROGRAM` template flags. Name the math ("top-k, scale, softmax"), never the model.
- A fact that follows from configs the compiler loads is derived in ONE shared place: no new `overrides.json` hand entries, no arch branches, no model-wide "has X anywhere" proxies.
- A kernel family covers the general shape space through instantiation, with a rung for off-tile shapes. If coverage is exactly the shapes of the checkpoints in hand, that is the finding: head dims {256, 512}, expert counts {128, 256}, "a tiny parity config's GEMMs sit off the tiles".
- A rule chosen by name when the shape already decides it (picking a tile by arch name rather than by `n`) is a finding.
- **Tells:**
  - doc comments justifying a rule with a model ("Gemma's", "the 27B's");
  - tables of per-model shapes in comments;
  - the shared handoff carrying `arch: String` "for per-arch tables".

### P8. Shared vs target: the target's own surface is opcode lowering only
- Target ABI facts are **const tables** that the shared passes take as input. They are never logic, and never scattered match arms.
- No target-private passes after opcode lowering: fusion, liveness, arena or scratch placement, kernel re-selection, inserted ops. Each of those is a shared pass with a table row, or an op in the shared tape (e.g. activation quantization).
- The lowering is a dumb IR→tape translation; every transform is a pass. No instruction selection.
- Neutral code never lives in a target-named crate or under a target `cfg`. A backend's feature is never on for another backend.
- No new env-var behavior toggles.

### P9. The device is a type of its constants
- The target is the chip's facts as const generics (`AppleGpu<GEN, ULTRA>`), and every setting is a formula of them. There's one tape per variant, and the device picks its own at load.
- No hard-coded device numbers in Rust or kernel source (`H100_SXM_SMS = 132`, `TARGET_CTAS = 264`), no per-chip hand tables, and no class enums or bands.

### P10. One source of truth
- The same edit needed in two callers means the duplication is the bug: consolidate the routine, don't share a helper between the copies.
- A test that checks a copy against a copy can't catch drift; that counts even under `cfg(test)`.
- An invented schema in a data file (magic keys, `kind` strings, a second copy of the loader's naming) is a smell to reject.

### P11. Lock hygiene (report only when it hides one of the above)
- No `#[allow]` at a site (a reasoned allow goes in `[lints]`).
- No escape-hatch constructors (`from_untyped`, `conflated_with_*`, a `pub` raw constructor on a checked type).
- No weakened tests or ratchets ("did the number get smaller?").
- No `/tmp/…` plan references in code.

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
- **P3:** the GEMM tile rule is stated or checked in 5 places: Rust `GEMM_BF16_TILE`, the `.cu` `M_TILE`/`N_TILE`/`K_TILE`, `gemm_kernel` and `gemm_shape` at expansion, and the launcher's runtime `ERR_SHAPE`. Pool presence is two paired `Option`s, re-checked by `WorkerError::*PoolMissing/Unused` and 6 `.expect("`new` made a pool …")`.
- **P4:**
  - 98 runtime shape-refusal returns across 44 `.cu` launchers;
  - `KernelError` (21 variants), `WorkerError` (14), `ForwardError` (28), `PlanError` (16);
  - greedy-only sampling refused per request through a new stringly `WorkerFactory::check_sampling -> Result<(), String>` seam in shared serving;
  - `DeviceError::NotH100`.
- **P5–P7:** bare-`u32` step fields; a parallel raw `TargetModelFacts` beside the typed `model_facts::Class`; `CUDA_CAPABLE`, `CUDA_PRESETS` and `CUDA_GEMM_COL64_ARCHS` string allowlists; `RouteProgram::{QwenSoftmax, GemmaScaledSoftmax}`.
- **P8:** CUDA-private post-lowering passes (`gemv_tc`, `quantize_fp8_inputs`, `fp8_moe_prefill`, `fold_norm_quants`).
