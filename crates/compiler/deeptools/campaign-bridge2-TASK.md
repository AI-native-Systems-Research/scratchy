Campaign: **bridge 2 — DataflowIR → SentientIR**, the D1–D28 span of the
deeptools compiler ladder.

# Mandatory questions

## Campaign

1. **Which repository and revision should this campaign use?**
   - Answer: `/Users/nickm/tmp/dt_src` — a local extract of the pod tree
     `nickm-59f46cc7fc-sj9vv:/project_src/deeptools`, revision
     `a0d29abbedfa2dd44ec7255e59440b06a429118c`
     (`stable-2026_07_24-142907-715-ga0d29abbed`, committed 2026-08-13).
     Contains `dcc/`, `sys-arch-spec/`, `dialects/`, `common/` — 53 MB, no
     `.git`. **The pod is the authority**; this extract exists only because
     CodeQL needs the source on the machine running wavefront.

2. **Should this campaign port the C implementation to Rust, or create safe
   Rust wrappers?**
   - Answer: `port`. A wrapper would keep the C++ on the critical path, and the
     whole point of the ratchet is removing the shell-out to `dbo-opt`.

3. **What should this campaign target?**
   - Answer: `named subsystems` — the closure that lowers DataflowIR to
     SentientIR, i.e. `dcc::buildDSCToSentientIRPipeline`'s D1–D28. In
     dependency order as the pipeline runs them:

     | subsystem | passes | lines |
     |---|---|---|
     | `dcc/src/Transform/Dataflow` | D1–D14 | 7,397 |
     | `dcc/src/Conversion/AgenToSentient` | D15 | 5,356 |
     | `dcc/src/Conversion/AffineToStandard` | D16 | 298 |
     | `dcc/src/Conversion/SCFToSentient` | D20 | 346 |
     | `dcc/src/Conversion/VectorChainLowering` | D21, D22 | 5,749 |
     | `dcc/src/Conversion/StandardToSentient` | D26 | 536 |
     | `dcc/src/Conversion/SymbolToSentient` | D27 | 256 |
     | `dcc/src/Conversion/DataflowToSentient` | D28 | 2,123 |

     **Weight the schedule toward the transfer lowering, not the compute
     lowering.** Measured on the real input (below), `group_0` is 98
     `dataflow.get_unit`, 44 `dataflow.get_logical_memory_view`, 22
     `agen.vector_load`, 22 `agen.composite_load_and_store`, 22 `agen.yield`
     and 7 `vectorchain.binary`. So `AgenToSentient` and the memory-view
     handling dominate; `VectorChainLowering` is comparatively thin here.

     🛑 **WHAT MAY AND MAY NOT BE DROPPED FROM THIS SPAN — read before excluding anything.**

     ⛔⛔ **"OUR RUST CANNOT EXPRESS THAT OP YET" IS NOT A REASON TO SKIP A PASS.** It is a
     statement about how far the emitter has got, not about what the machine needs. Two passes
     were excluded on exactly that reasoning and both exclusions were wrong and dangerous:

     * `TransformPagedMemView` (D9, 2,025 lines) matches
       `dataflow.GetPagedLogicalMemoryViewOp`, which the DataflowIR island does not declare — so
       it looked droppable. **Paged KV is central to this target**, and this pass is how a paged
       view becomes addresses. Dropping it would silently block paged KV on spyre.
     * `SymbolToSentient` (D27) matches only `symbol.*` ops, and the island has no `symbol`
       dialect. But symbols are how runtime-corrected addresses travel — `init.bin`'s
       `symbol_ids`, `dbo-correct-at-runtime` — so that absence is a gap to fill, not a fact.

     ⭐ **THE ONLY SAFE EXCLUSIONS ARE THE ONES SCRATCHY'S STRUCTURE MAKES IMPOSSIBLE**, not the
     ones it has not reached yet. Concretely, two:

     1. **The MLIR construction machinery inside every file** — `OpBuilder`, `mlir::Value`
        iterators, the `getOrCreate…` memoisers, the `std::map<int, std::map<int,
        std::map<SenComponents, mlir::Value>>>` lookup tables. Every one exists because the C++
        builds an IR at run time and must look SSA values back up later. `#[forward]` resolves the
        whole tape at expansion and emits `static` tables, so there is nothing to look up and
        nothing to memoise. **This is the real saving, and it is inside files rather than whole
        passes** — one measurement put ~79% of a single file here.
     2. **The generic MLIR cleanup passes** — `SCCP`, `SymbolDCE`, `Canonicalizer`,
        `AffineToStandard`. They exist to tidy after pattern rewriting; an emitter writes the tidy
        form directly. They are upstream MLIR in any case.

     ⛔ EVERYTHING ELSE IN THE EIGHT SUBSYSTEMS ABOVE IS IN SCOPE, including D9 and D27. Do not
     narrow this list by reasoning about which ops the current islands happen to declare, and do
     not narrow it by which passes fire on a particular model — `Model` is a const-generic trait
     precisely so that 164 configs across 25 architectures share one op vocabulary, and a pass
     that no-ops on the model you sampled will fire for some `(Arch, Model, Workload)` that has
     not been compiled yet.

     ⛔ **OUT OF SCOPE — do not schedule these.** They are the next rungs and
     pulling them in loses the seam that makes this verifiable:
     `dcc/src/Transform/Sentient` (D29–D75, 48,432 lines, Sentient→Sentient),
     `dcc/src/Conversion/SentientToProgIR` (D76), and anything under
     `dcg/` or `dbo/`.

     ⛔ **AND DO NOT SCHEDULE FROM `dcg/`.** Every citation to
     `dcgbeCodegen.cpp` or `L3DlOpsScheduler.cpp` is real C++ on the *wrong
     input path* — they consume or produce a PCFG, which this pipeline does not
     use. A previous attempt accumulated 61 such citations across 17 files, all
     legitimate-looking, none on our path, and the port had to be restarted.

4. **Which agentic backend and model should do the translation work?**
   - Answer: `Claude Code, claude-opus-5`.
     ⛔ NOT the README's recommended `gpt-5.6-sol` for translators — that is the
     OpenAI Codex backend, and this campaign uses Claude Code subscription
     billing (question 8), so a Codex translator would need separate API
     billing. Same backend throughout.

5. **Do you want agentic review after translated work lands?**
   - Answer: `Claude Code, claude-opus-5`, reviewing after each completed
     sub-campaign. Worth it here specifically because the house rules below are
     unusual — no runtime failure paths at all, no `#[allow]`, newtypes over
     raw scalars, machine facts read from `sys-arch-spec` rather than restated —
     and a translator working from C++ will not infer them. Review is where
     they get enforced per sub-campaign rather than discovered at the end.

6. **Should the campaign run the optional agentic UB audit pass?**
   - Answer: `no` for the agentic pass on this campaign. The deterministic
     `crustify-audit unsafe` check needs no model and should run; but this port
     is target-independent table and rule translation with no FFI and no
     `unsafe`, so an agentic UB reviewer has little to find. Revisit if any
     `unsafe` appears — under these house rules, that is itself a red flag.

7. **Should I run fully autonomously end to end?**
   - Answer: `no`. See questions 12–13: the campaign brief must be approved
     before setup and before translation, because the brief is where the
     scheduled unit list can be checked against the "already ported" boundary
     below. A bottom-up planner given `AgenToSentient` will otherwise discover
     the Agen dialect, then `arch_enums.h`, then the whole architecture spec —
     all of which are already in Rust.

8. **Which billing mode should agentic stages use?**
   - Answer: `subscription` — Claude Code.

## Autonomy

12. **Should I wait for your approval before starting the setup phase?**
    - Answer: `yes`
13. **Should I wait for your approval before starting the translation phase?**
    - Answer: `yes` — and present the scheduled unit list, so it can be checked
      against the terminal boundary below before anything is written.
14. **Should I wait for your approval between sub-campaigns?**
    - Answer: `yes`
15. **Should I wait for your approval before starting review passes?**
    - Answer: `no`
16. **Should I wait for your approval before starting UB audit passes?**
    - Answer: `not applicable`

# Optional questions

9. **Batching and parallelism?**
   - Answer: `orchestrator's choice`

10. **Review batch caps?**
   - Answer: `recommended 3x`

11. **Target unit budget per sub-campaign?**
    - Answer: `orchestrator's choice` — but note the natural seam is one
      sub-campaign per conversion above, since each has its own lit-test
      oracle.

# Benchmark recording questions

17. **Where and in what format should results be recorded?**
    - Answer: `crates/compiler/deeptools/docs/bridge2-results.md`, standard
      template.

# Target-repo context the orchestrator needs

**Where the port lands.** `crates/compiler/deeptools/src/bridges/dataflow_ir_to_sentient/`
in this worktree (`dfir-ratchet`). The crate has **no scratchy dependency and must
keep none** — it holds the deeptools IRs and the bridges between them.

**Both ends already exist, and the port must target them rather than inventing types.**

* Input: `islands::dataflow_ir` — six dialect modules (`dataflow`, `agen`,
  `vectorchain`, `affine`, `arith`, `scf`), one file per authoritative `.td`.
* Output: `islands::sentient` — `SentientOps.td`'s 29 ops over
  `SentientTypes.td`'s 16 enums, plus `ProgramUnit`/`ProgramUnits`.

⛔ **THE RUNG IS MIXED.** SentientIR is not "the DataflowIR ops replaced": after
D28 a real granite program still holds `dataflow.get_unit`,
`dataflow.get_logical_memory_view`, `agen.composite_load_and_store` and
`agen.yield` beside `sentient.scalar_constant`. And `dataflow.program_unit`
survives the whole rung — it is in **657 of the 668** `CHECK-SENT-IR`
expectations in `dcc/test/`. A port that assumes a clean dialect swap is wrong.

## 🛑 ALREADY PORTED — TERMINAL BOUNDARIES. WIRE TO THESE; DO NOT RE-PORT THEM.

⛔⛔ **THIS IS THE MAIN SCOPE RISK OF THE CAMPAIGN.** A bottom-up dependency
closure over the eight subsystems above will reach, in order: the Agen and
Dataflow dialect definitions, then `arch_enums.h`, then `sysdef.cpp`, then the
ISA. **Every one of those is already Rust in this repo.** Re-porting any of them
does not just waste the budget — it creates a second copy of a machine fact,
which is the specific failure `sys-arch-spec` exists to prevent: *"a fact about
the machine that two crates each held a copy of is a fact that can disagree
with itself"*.

When the closure reaches any C++ below, **stop and use the Rust named beside
it**:

| C++ the closure will reach | already ported to | do |
|---|---|---|
| `dcc/src/Dialect/Dataflow`, `Dataflow.td` | `islands::dataflow_ir::dialects::dataflow` | use |
| `dcc/src/Dialect/Agen`, `Agen.td`, `AgenEnums.td` | `islands::dataflow_ir::dialects::agen` | use |
| `VectorChain.td` | `islands::dataflow_ir::dialects::vectorchain` | use |
| `dcc/src/Dialect/Sentient/SentientOps.td` | `islands::sentient::dialects::sentient` (29 ops) | use |
| `dcc/src/Dialect/Sentient/SentientTypes.td` | same module (16 enums) | use |
| `sys-arch-spec/isa/isa.cpp`, `isa.hpp`, `isaSystemc.h` | `sys_arch_spec::{fields, values, operand}` | use |
| `sys-arch-spec/sysdef.cpp` | `sys_arch_spec::{regfile, memory}` | use |
| `sys-arch-spec/arch_enums.h` | `sys_arch_spec::arch_enums` | use |
| `sys-arch-spec/progir/progir.h`, `regvisitor.cpp` | `sys_arch_spec::{progir, reg_refs}`, `islands::progir` | use |
| anything reading `.ddl` / `.smc` template text | `deeptools`'s `ddl/` + `build.rs` const tables | use |
| MLIR itself (`OpBuilder`, `PatternRewriter`, `TypeConverter`, `applyPartialConversion`, SSA plumbing) | **nothing — not needed** | see below |

⛔⛔ **AND MLIR'S OWN MACHINERY IS NOT IN SCOPE AT ALL, WHICH IS MOST OF THE
LINE COUNT.** The C++ carries an `OpBuilder`, `mlir::Value` iterators,
`getOrCreate…` memoisers and `std::map<int, std::map<int, std::map<SenComponents,
mlir::Value>>>` lookup tables. **Every one of those exists because it constructs
an IR at run time** — SSA values it must later look up, memoisation because
construction is expensive, nested maps because it cannot index a constant. This
crate runs inside a proc macro with the tape already resolved as constants: it
*emits*, it does not *rewrite*. So port the **rule**, never the plumbing. One
prior measurement on a single file put ~79% of its lines in this category.

⭐ **CONFIGURE WAVEFRONT NARROWLY TO ENFORCE THIS.** The closure must terminate
at the boundaries above rather than discovering them. Per the README, a
sub-campaign may carry its own narrow `wavefront-config.json`; use one here, and
if the scheduled unit list at question 13 contains anything from the table above,
that is a scoping bug to fix before translation starts, not a batch to run.

## 🛑 TWO SETUP-PHASE ADAPTATIONS — decide these before Phase 1, not during it

The playbook's setup phase assumes a C library being migrated into a fresh `rust/`
tree of crates. This campaign is a subsystem of a C++ compiler being ported into an
**existing** crate. Placement is called orchestrator judgment in the playbook, so it
is decided here:

1. ⛔⛔ **NO `-sys` CRATES AND NO BINDGEN. THERE IS NO FFI BOUNDARY.** The playbook
   says to create a `<lib>-sys` placeholder per target and imported library, with a
   bindgen pipeline translators populate lazily. That exists so a partly-migrated
   tree can still call the C for symbols not yet ported. **This campaign has nothing
   to call.** `deeptools` links no C++ at all — the current path shells out to the
   `dbo-opt` *binary* as a subprocess, and the object of the ratchet is to stop doing
   that. An FFI shim here would be a dependency edge back into the thing being
   removed. Zero `unsafe`, zero `extern "C"`, zero bindgen.

2. ⛔ **THE TARGET IS AN EXISTING CRATE, NOT A NEW `rust/` TREE.** Everything lands in
   `crates/compiler/deeptools/src/bridges/dataflow_ir_to_sentient/` inside the
   workspace at the repo root. Do not author a `rust/` subtree, do not create a crate
   per `link_unit`, and do not add workspace members. `crates.json` should place every
   ported unit in the one existing crate `deeptools`.

   ⭐ The reason is not tidiness: both ends of this bridge are hand-designed Rust that
   already exists in that crate (`islands::dataflow_ir` and `islands::sentient`), and
   the port's whole job is to connect them. A separate crate could not see them without
   an inversion of the dependency the repo forbids.

## House rules the translated code must satisfy

These are not style preferences; each is load-bearing and each has cost real
time when broken.

1. **No runtime failure paths.** No `assert!`, `debug_assert!`, `panic!`,
   `unwrap`, `expect`, `unimplemented!`, `unreachable!`, and no
   `checked(..) -> Option` constructors. A refusal at run time is still a
   run-time failure. An invalid value must be *unconstructible*: enumerate the
   vendor's closed sets one variant per `def`, and bound indices with
   `Bounded::at::<I>()`, whose check is in a `const { }` block.
2. **No `#[allow]`.** Clippy clean.
3. **Newtypes, never raw scalars.** Transposing two extents must be `E0308`.
   `Elements`/`Bytes` exist; use them.
4. **No strings from the parsers.** Every closed set is a generated enum.
5. **Pairings are one value with two ends.** `islands::dataflow_ir::link`
   already provides `Link` → `(SendEnd, RecvEnd)` and `Rendezvous` →
   `(Half, Half)`. A send's consumer and the matching receive's producer must
   come from one value, not two lookups that agree — and the Sentient ops
   already take `SendEnd`/`RecvEnd`, so the pairing survives the lowering.
6. **Machine facts come from `sys-arch-spec`, never restated.** The ISA field
   and opcode tables, the register-file depths, `max_ibuff_entries`, the
   `RegType` enumerators and the ProgIR bounds are all there. Three separate
   times a fact was about to be hand-written that this crate already held.
7. **Const generics express *and* exploit.** A flag deciding which ops EXIST is
   a const generic that must REMOVE ops. `Exploit<A, M, W>` holds
   `IS_DECODE`, `FITS_LX`, `STICK_ALIGNED`, `NO_CACHE_WALK`, `CACHE_FITS_LX`;
   the bridge is where they are finally *read*, since an island must not branch
   on a workload.
8. **Cite by `path:line`.** The pod C++ is the authority. A lit test, a golden
   `.mlir` or a doc is evidence, never the authority.

## The oracles — this span has three, and they are cheap

1. **668 paired lit tests.** `dcc/test/` holds 825 `.mlir` tests, **668 with
   `CHECK-SENT-IR`** expectations, organised per unit (`PE`, `PT`, `SFP`,
   `L0LU/SU`, `L3LU/SU`, `LXLU/SU`). Each is DataflowIR input beside its
   expected SentientIR output — a per-conversion oracle the vendor already
   wrote. **Use these as the primary gate.**
2. **The reference pipeline runs on our input.** `dcc_standalone <bare>.mlir
   -kEmitProgIR` completes on our emitted DataflowIR (rc=0), and
   `--mlir-print-ir-after=dcc-dataflow-to-sentient` dumps the reference's own
   SentientIR for any module. So the port's output can be diffed against the
   reference's for the same input.
   ⚠️ Feed `dcc_standalone` a **bare, non-private `func.func @dataflowProgram()`** —
   the nested-module shape is for `dbo-opt`, and under `dcc-opt` `SymbolDCE`
   deletes a `private` body whose only caller is in another symbol table.
3. **The real input, which is the only one that counts.** A single-unit lit test
   is not representative. The bake
   `DBO_OPT=/project_src/deeptools/build/dbo/tools/dbo-opt/dbo-opt
   SCRATCHY_SKIP_SENDNN_CXX=1 cargo build
   -Fsuperdsc,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel -vv`
   stages the emitted DataflowIR at
   `/tmp/superdsc-stage-<pid>/<kernel>/group_N/group.mlir` — 32 groups, 18,050
   lines, two shapes (1,080-line and 88-line groups).

## What the pipeline actually does, measured

Running all 128 pass invocations over one real program: **34 change the IR, 94
are no-ops.** Of the 34, roughly half are generic MLIR cleanup (`SCCP` ×4,
`Canonicalizer` ×9, `AffineToStandard`) which a port that *emits* rather than
*rewrites* does not need, and a further nine are loop re-rolling that exists to
undo `TransformLoopToLegalizeForSentientLowering`'s full unroll. Do not port a
pass because it is in the list; port it because this span's output requires it.

⚠️ That measurement is from a single-unit PT program and therefore does **not**
exercise `DataflowToSentient` (no sends/receives), `AgenToSentient` (no
transfers) or the sync passes. Re-run it against a real granite group before
using it to scope anything.
