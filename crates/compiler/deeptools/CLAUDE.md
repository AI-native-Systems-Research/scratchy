# deeptools — the DataflowIR bridge
**GOAL:** compile the SubtileTape to DataflowIR, bake it to `init_binary` by invoking `dbo-opt`.
`SubtileIR tape ─bridge1─► DataflowIR ─► dbo-opt ─► init_binary`. `islands/<ir>/` is an IR and nothing else
(types, invariants, printer); `bridges/<from>_to_<to>/` is where two vocabularies meet. The DDL template is
the SCHEDULE (units, transfers, loop nest, computes) and knows no extents; the NODE is the SHAPE — deriving
one from the other is inventing it.
**ACCEPTANCE — the only test, both or neither:** `cargo build -Fsuperdsc,model/granite-3.1-{2b,8b}-instruct,quant/fp8-dynamic-per-channel`.
8b is hd=128/4096/12800 vs 2b's 64/2048/8192 — different door arm and tile geometry, so 2b passing hides
defects 8b finds. Read the `-vv` stream; cargo hides build-script output.
**WORKFLOW:** propose ONE precise task (file, function, exact change, why correct, what output proves it) →
approved → on the list. From the bake's output build a PRECISE failure list: one task per failure; resolve
one → complete it; find new ones → add tasks.
Note: We have patched dbo-opt to accept DataflowIr output via `--from-dfir`. We will need to figure out how to avoid this, but it is fine to use it for now.
## Rules
- CONST GENERICS EXPRESS *and* EXPLOIT. A flag deciding which OPS EXIST is a const generic and must REMOVE
  ops (`FITS_LX` the tiling loop, `IS_DECODE` the row nest, `STICK_ALIGNED` the mask *and* its consumer). A
  count that only appears inside an op is a value. Model, Arch, quant, sk/rung all flow. Every flag needs a
  test comparing TWO EMISSIONS structurally, carrying the VALUE not a ratio. A `const fn` on a runtime value
  folds to nothing.
- NEWTYPES, never raw scalars (`Rows`, `Cols`, `Contraction`, `Elements`, `Bytes`, `Sticks`, `Segment`,
  `GroupId`, `OpIndex`): transposing two extents must be E0308.
- NO STRINGS from the ddl/smc parsers — every closed set is a generated enum.
- An unread associated const NEVER EVALUATES; doctests do NOT run in `tests/*.rs`.
## 🛑🛑 NEVER RUNTIME REFUSE. ANYWHERE. THIS IS THE PRE-EMINENT RULE OF THIS CRATE.
**dbo-opt IS THE ONLY ORACLE.** The pipeline is `SubtileTape → DataflowIR → dbo-opt → init_binary`, and
every defect this bridge has ever fixed was named by a dbo-opt refusal on OUR emitted MLIR —
`vector<64xi1>` vs `i1`, "found no program to compile", the SSA-scoping pair, "Unable to generate
loops", the `ProgramUnitsReduction` assertion, "Dangling non-compute op has no use". A lowering that
returns `Err` stops **before the tape is emitted**: dbo-opt is never invoked, the sentence we needed is
never produced, and the build prints our message instead of the backend's.

**THIS HAS COST HOURS AND A REVERT FIVE TIMES.** The last one added ONE variant to an error type that
already existed — two lines, indistinguishable from the variants beside it — and the loop went blind
until the raw `-vv` stream was read line by line. Prose did not prevent any of the five.

- **There is no error type in the DataflowIR bridge.** `Err(`, `.ok_or` and `Result<` are frozen at
  **ZERO** by `crates/targets/spyre/tests/dfir_never_runtime_refuses.rs`. A `Result` is a VALUE — the
  caller can log it and carry on, and `codegen.rs` did exactly that. Bringing one back means adding a
  whole `enum` to a diff.
- **`panic!`/`todo!` are tolerated and capped**, ratcheted DOWN only. Loud and unswallowable, so nobody
  mistakes one for a lowering that ran — but still a stop before the oracle. The goal is zero.
- **NEVER substitute to dodge a panic.** An unbuilt op lowered as `Identity` "to keep the tape whole"
  is a program dbo-opt compiles happily and a model that emits garbage. `todo!` naming the op is right;
  a stand-in op-func is not.
- **The fact you are reaching for belongs in a TYPE.** An arity is an array length. A pairing is a
  witness consumed once. An op set is a const the door proved. If the lowering genuinely cannot produce
  a program yet, that is a decomposition to WRITE.

## Already fucked up here — do not repeat
- **Invented addresses** (everything into LX; programs read memory nothing fills). Addresses come from the
  placement authority derived from `&SubtileIR` — the same plan the worker H2Ds. Note this outranks the
  panic rule: a fabricated placement to avoid a stop is worse than the stop.
- **Read `EmittedOp`/`Dsc`** chasing `emit_bundle`. Opening `lower_subtile_tape_to_superdsc.rs` to thread
  something through means STOP. DEBT: retire that path; its tape-derived address authority is target-neutral
  and must not live in a target-named file.
- **Wrote MLIR straight to disk**, bypassing the bake's `reserve` (blocking disk bound), `submit` (queue +
  memoization) and staging deletion — megabytes uncompiled that looked like progress.
- **Depended on `DEEPTOOLS_PATH`** — not needed; `--from-dfir` never reaches `EnsureDeviceDeclaration`. You
  write/port Rust and invoke dbo-opt. Nothing else.
- **Scoped to one op** not the whole tape; built demo **examples** instead of using the build command;
  reported byte/node counts and "it compiles" as bakes; used `SCRATCHY_PLAN_ONLY_BAKE` (= NO DEVICE
  PROGRAMS) and called it a bake; passed off IBM's reference `dfir.mlir` output as ours. Say which input
  produced which artifact, and what was NOT verified.
