# Bridge-2 campaign brief — read this in full before you edit anything

You are an agent on the **crustify bridge-2 campaign**: port the 384 C++ functions of IBM Spyre
`dcc`'s `DataflowIR -> SentientIR` lowering (passes D1-D28) into the existing Rust crate
`crates/compiler/deeptools`, under `src/bridges/dataflow_ir_to_sentient/`.

Read [`TASK.md`](TASK.md) (this directory) for the campaign statement, and
[`../crates/compiler/deeptools/CLAUDE.md`](../crates/compiler/deeptools/CLAUDE.md) for the crate's
rules. Both outrank the generic C-porting conventions in your system prompt.

## 0. 🛑 THE BUDGET — MEASURED, AND IT IS WHY THIS SECTION IS FIRST

The first 144 functions cost **43 hours of wall clock, 89% of it model time, 586 MILLION cache-read
tokens against 5 million output** — 30 to 48 turns per function. That rate does not finish this
campaign. The measurement also says exactly where it went: of 30,991 lines produced, **5,306 were
implementation, 12,333 were doc comments and 12,575 were tests.** 82% of the output was not the port.

So the following are **HARD CAPS**, not guidance. A batch that exceeds them will be sent back.

| per ported function | cap |
|---|---|
| doc comment lines | **8** — the `/// Replaces: eNNN_name` anchor, one line of what it does, and any TRAP. Nothing else. |
| test functions | **1**, or 2 where the vendor's own case plus one negative both apply |
| `cargo check`/`cargo test` runs | **once per BATCH**, at the end — not per function |

⛔ **DO NOT RE-VERIFY CITATIONS.** Take the `file:line` from `UNITS.tsv` as given. Re-measuring line
numbers cost a large share of the turns above, and **the review pass owns that check** — it re-measured
every drifted citation in level 0 and will do the same for yours.

⛔ **DO NOT GREP THE CRATE TO DISCOVER TYPES.** Your `.rs` home's header names the island types your
units need. If something genuinely is not there, add it and say so in the commit message — do not go
looking first.

⛔ **NO TUTORIALS IN COMMENTS.** A trap is *"the non-L3 arm sets BOTH fields to stride_size"*. A
tutorial is three paragraphs on why MLIR builds IR at run time. The first earns its 8 lines; the second
is what 12,333 doc lines were.

⭐ WHAT IS NOT CAPPED: **correctness, and the emission.** The op a function emits IS the function —
see §3. Cutting the port to hit a cap is the one failure worse than being slow.

## 1. Where the C++ actually is

⛔ **The authority is the C++ tree, not the extract.**

```
/Users/nickm/git/deeptools-src/<file>:<line>      # repo_info.txt: deeptools|master|a0d29abbed…
```

That is exactly the revision every banner cites. `crustify-bridge2/source/bridge2.cpp` tells you
**which** functions are in scope and **in what order**; **its bodies are truncated at the tail.**
Measured: 366 of 384 bodies end in a blank line followed by bare closing braces, and a 48-entry
sample against the authority found 21 with real trailing statements dropped — a `return success();`
here, a `return rhs;` there, an entire `} else { … }` branch in `e217_lowerVectorLoadHelper`, the
`initMASData(mas_data, ad, max_mutable);` call in `e289_initialize`. **Port from the authority file
at the cited line.** `/Users/nickm/git/deeptools` is a *different* revision — do not use it.

`source/prelude.inc` is a pile of semantically empty stand-ins. It carries names and shapes so the
extract stands alone; it carries no behaviour and is not a porting input.

## 2. Resolving your worklist

Your worklist names units as `e<NNN>_<cppName>`. `NNN` is the entry number.

```bash
grep -P '^e217_' crustify-bridge2/UNITS.tsv     # entry, level, LoC, authority file:line,
                                                # extract line range, Rust home, callees
```

`crustify-bridge2/UNITS.tsv` is the map for all 384. `crates/compiler/deeptools/docs/bridge2-porting-order.md`
carries the same list with PORT/AUDIT checkboxes and the 106 exclusions with a reason each. Your
`.rs` home already exists, carries the citation table for its units in its module doc, and holds one
`// crustify:todo: e<NNN>_<name>` per scheduled unit. Replace each with the ported function carrying
`/// Replaces: e<NNN>_<name>`. **A surviving TODO is open work.**

## 3. What "ported" means here

⛔ **The whole function, including its emission.** A previous attempt extracted each function's
decision rule into a documented predicate (`is_l3()`, `BurstSetting::of()`, `Granularity::checked()`),
left out the part that emits an operation, and reported it done — nothing called any of it. **The op
a function emits IS the function**, with its exact attribute names and values, its branch order and
its early returns.

What you MAY drop is the *mechanism for reaching operands*: walking uses, memoising by
`(core, corelet, component)`, positioning an `OpBuilder`. Those exist because the C++ builds an IR at
run time and this crate builds a typed value at compile time.

⛔ **If the target IR cannot express a function's input, add the operation to the island**
(`src/islands/sentient/`, `src/islands/dataflow_ir/`) rather than concluding the function is
unnecessary. `src/bridges/dataflow_ir_to_sentient/mod.rs` shows the shape:
`dataflow_ir::Run<A>` in, `sentient::Run<A, M, W>` out.

⚠️ `src/bridges/dataflow_ir_to_sentient/agen_to_sentient.rs` is that earlier partial attempt.
Nothing in it counts as a ported unit; reuse what is right, and give every unit its own anchored
function.

## 4. What this campaign is NOT

Your system prompt's conventions cover C-to-Rust *wrapping*. **None of the following applies here**
— this is a pure-logic port of C++ into an existing crate, with no C left anywhere:

- ❌ no `bindgen`, no allowlist, no `-sys` crate (`crustify/rust/deeptools-sys` is an empty
  placeholder that exists only to satisfy an executor gate)
- ❌ no `ffi::`, no `mod ffi_export`, no `#[unsafe(no_mangle)] extern "C"`, no `CRUSTIFY_<FILE>` switch
- ❌ no `Foo`/`FooRef`/`FooMut` wrapped-layout triple, no `addr_of!` projection, no `unsafe`
- ❌ no C build, no sanitizers, no ASan/UBSan runner, no `io_equiv` C-vs-Rust harness (there is no C
  to call — the reference is a compiler that does not run on this host)

## 5. Crate rules that DO bind you (`crates/compiler/deeptools/CLAUDE.md`)

- 🛑 **NEVER RUNTIME REFUSE.** No `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
  A closed set is an `enum`; an invariant is a type. `crates/targets/spyre/tests/dfir_never_runtime_refuses.rs`
  freezes those at zero. `todo!("<op> not ported")` is tolerated, capped, and ratcheted down only.
- **Newtypes, never raw scalars** (`Rows`, `Cols`, `Elements`, `Bytes`, `Bits`, `Sticks`, …):
  transposing two extents must be an E0308.
- **No strings for closed sets** — every enumeration is an `enum`.
- Const-generic traits `Arch`, `Model`, `Workload` flow through; a flag that decides which ops
  EXIST is a const generic, not a runtime value.
- ⛔ Never substitute a stand-in op to dodge a `todo!` (an unbuilt op lowered as `Identity` compiles
  and emits garbage).

## 6. Tests — ⛔ ONE PER FUNCTION (see §0)

`#[cfg(test)] mod unit_tests` beside the code. **One test per ported function.** Where the vendor has a
case, port THEIRS rather than inventing one — the authority tree's `dcc/test/` has 668 files carrying
`CHECK-SENT-IR` expectations. Their *input* is MLIR text and **this crate has no parser and must not get
one**: build the typed input in the test and take the *expectation* from their `CHECK` lines.

A second test is justified only when the vendor's case plus one negative both apply. Level 0 averaged
87 lines of test per function; that is roughly ten times the budget.

## 7. Your gates — and the one you must NOT run

```bash
cargo check -p deeptools          # ONCE per batch, at the end — not per function
cargo test  -p deeptools          # ONCE per batch
```

⛔ **Do not run the workspace build or the acceptance build in your worktree.** Its `target/` is
~6 GB and this host has under 50 GB free; N agent worktrees doing that fills the disk and takes the
whole campaign down. The orchestrator owns that gate. (For the record it is
`cargo build -Fsuperdsc,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel`, which does not
even link on this host: `superdsc` pulls `flex-rs`, whose `build.rs` needs Spyre-only senlib headers.
The runnable prefix that still expands `#[forward]` over every staged program is
`cargo build -p scratchy-models --features superdsc,granite-3.1-2b-instruct,scratchy-quantizations/fp8-dynamic-per-channel`,
and it is the orchestrator's, not yours.)

Commit one changeset and land it on the session branch exactly as your task prompt describes.
