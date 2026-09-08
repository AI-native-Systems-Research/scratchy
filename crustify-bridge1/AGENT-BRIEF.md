# Bridge-1 campaign brief — read this in full before you edit anything

You are an agent on the **crustify bridge-1 campaign**: port the 110 C++ functions of IBM Spyre's
**SuperDSC → DataflowIR** conversion into the existing Rust crate `crates/compiler/deeptools`, under
`src/bridges/superdsc_to_dataflow_ir/`.

Read [`TASK.md`](TASK.md) (this directory) for the campaign statement, and
[`../crates/compiler/deeptools/CLAUDE.md`](../crates/compiler/deeptools/CLAUDE.md) for the crate's
rules. Both outrank the generic C-porting conventions in your system prompt.

## 0. 🛑 THE BUDGET — MEASURED ON BRIDGE 2, AND IT IS WHY THIS SECTION IS FIRST

Bridge 2's first 144 functions cost **43 hours of wall clock, 89% of it model time, 586 MILLION
cache-read tokens against 5 million output** — 30 to 48 turns per function. That rate does not
finish a campaign. The measurement also says exactly where it went: of 30,991 lines produced,
**5,306 were implementation, 12,333 were doc comments and 12,575 were tests. 82% of the output was
not the port.**

So the following are **HARD CAPS**, not guidance. A batch that exceeds them will be sent back.

| per ported function | cap |
|---|---|
| doc comment lines | **8** — the `/// Replaces: eNNN_name` anchor, one line of what it does, and any TRAP. Nothing else. |
| test functions | **1**, or 2 where the vendor's own case plus one negative both apply |
| `cargo check` / `cargo test` runs | **once per BATCH**, at the end — not per function |

⛔ **DO NOT RE-VERIFY CITATIONS.** Take the `file:line` from `UNITS.tsv` as given. Re-measuring line
numbers cost a large share of the turns above, and **the review pass owns that check.**

⛔ **DO NOT GREP THE CRATE TO DISCOVER TYPES.** Your `.rs` home's module header names the island
types your units need. If something genuinely is not there, add it and say so in the commit message
— do not go looking first.

⛔ **NO TUTORIALS IN COMMENTS.** A trap is *"the non-L3 arm sets BOTH fields to stride_size"*. A
tutorial is three paragraphs on why MLIR builds IR at run time. The first earns its 8 lines; the
second is what 12,333 doc lines were.

⭐ WHAT IS NOT CAPPED: **correctness, and the emission.** The op a function emits IS the function —
see §3. Cutting the port to hit a cap is the one failure worse than being slow.

## 1. Where the C++ actually is

⛔ **The authority is the C++ tree.**

```
/Users/nickm/git/deeptools-src/<file>:<line>     # repo_info.txt: deeptools|master|a0d29abbed…
```

That is exactly the revision every banner cites. ⛔ `/Users/nickm/git/deeptools` is a **different
revision** — do not use it.

`crustify-bridge1/cpp/bridge1.cpp` tells you **which** functions are in scope and **in what order**.

✅ **Unlike bridge 2's, this extract is verified untruncated** — all 110 bodies, by
`tools/verify_extract.py`, which re-derives each function's end from the authority with its own
brace matcher and compares LENGTHS and CONTENT. (Bridge 2's extract truncated 366 of 384 bodies and
its check reported "verbatim, 12 sampled, none differed" because it re-sliced the source with the
same length the extractor had used. Two negative controls confirm this one bites.) You may therefore
read a body from `bridge1.cpp`. Still cite the authority `file:line` in the anchor, and open the
authority when a body's surrounding context matters — the extract has no callers, no class members
and no headers.

`cpp/prelude.inc` catalogues the 116 qualified scopes and 217 external calls the bodies reach so the
unit is self-describing. It carries **no behaviour** and is **not a porting input**, and the unit is
not compile-clean — nothing builds it, and no gate asks you to.

## 2. Resolving your worklist

Your worklist names units as `e<NNN>_<cppName>`. `NNN` is the entry number.

```bash
grep -P '^e042_' crustify-bridge1/UNITS.tsv     # entry, level, LoC, authority file:line,
                                                # extract line range, Rust home, callees
```

`crustify-bridge1/UNITS.tsv` is the map for all 110. Your `.rs` home carries the citation table for
its units in its module doc and holds one `// crustify:todo: e<NNN>_<name>` per scheduled unit.
Replace each with the ported function carrying `/// Replaces: e<NNN>_<name>`. **A surviving TODO is
open work.**

## 3. What "ported" means here

⛔ **The whole function, including its emission.** A previous attempt on bridge 2 extracted each
function's decision rule into a documented predicate (`is_l3()`, `BurstSetting::of()`,
`Granularity::checked()`), left out the part that emits an operation, and reported it done — nothing
called any of it. **The op a function emits IS the function**, with its exact attribute names and
values, its branch order and its early returns.

What you MAY drop is the *mechanism for reaching operands*: walking uses, memoising by
`(core, corelet, component)`, positioning an `OpBuilder`. Those exist because the C++ builds an IR at
run time and this crate builds a typed value at compile time.

⛔ **If the target IR cannot express a function's input, ADD THE OP TO THE ISLAND**
(`src/islands/dataflow_ir/`) rather than concluding the function is unnecessary. **Deciding a
function is unneeded is not the porter's call.** A predicate is not a port.

⛔ **`src/bridges/subtile_to_dataflow_ir/` is the hand-written bridge this port REPLACES.** Read it,
reuse what is right — and do not edit or delete it. Switching the call site is a separate step.

## 4. What this campaign is NOT

Your system prompt's conventions cover C-to-Rust *wrapping*. **None of it applies here** — this is a
pure-logic port of C++ into an existing crate, with no C left anywhere:

- ❌ no `bindgen`, no allowlist, no `-sys` crate (`crustify/rust/deeptools-sys` is an empty
  placeholder that exists only to satisfy an executor gate)
- ❌ no `ffi::`, no `mod ffi_export`, no `#[unsafe(no_mangle)] extern "C"`, no `CRUSTIFY_<FILE>` switch
- ❌ no `Foo`/`FooRef`/`FooMut` wrapped-layout triple, no `addr_of!` projection, no `unsafe`
- ❌ no C build, no sanitizers, no ASan/UBSan runner, no `io_equiv` C-vs-Rust harness (there is no C
  to call — the reference is a compiler that does not run on this host)

## 5. Crate rules that DO bind you (`crates/compiler/deeptools/CLAUDE.md`)

- 🛑 **NEVER RUNTIME REFUSE. THIS IS THE PRE-EMINENT RULE OF THE CRATE.** No `Result`, no `Err(`, no
  `.ok_or`, no `assert!`, no `debug_assert!`. `crates/targets/spyre/tests/dfir_never_runtime_refuses.rs`
  freezes those at **zero**. A closed set is an `enum`; an invariant is a type.
  **Why:** a lowering that returns `Err` stops *before the tape is emitted*, so `dbo-opt` — the only
  oracle — is never invoked and the sentence we needed is never produced.
  `todo!("<op> not ported")` is tolerated, capped, and ratcheted down only; it is how the build names
  the next gap.
- ⛔ **NEVER substitute a stand-in op to dodge a `todo!`.** An unbuilt op lowered as `Identity` is a
  program dbo-opt compiles happily and a model that emits garbage.
- ⛔ **NEVER invent an address.** Addresses come from the placement authority derived from
  `&SubtileIR`. This outranks even the panic rule: a fabricated placement is worse than a stop.
- **Newtypes, never raw scalars** (`Rows`, `Cols`, `Contraction`, `Elements`, `Bytes`, `Bits`,
  `Sticks`, `Segment`, `GroupId`, `OpIndex`): transposing two extents must be an E0308.
- **No strings for closed sets** — every enumeration is an `enum`.
- Const-generic traits `Arch`, `Model`, `Workload` flow through; a flag that decides which ops
  **exist** is a const generic that must REMOVE ops, not a runtime value.
- ⭐ **THE DDL TEMPLATE IS THE SCHEDULE, THE NODE IS THE SHAPE.** The template says which units take
  part, what moves and in what loop nest, and knows no extents; the node says how big. A bridge that
  derives one from the other has invented it.

## 6. Tests — ⛔ ONE PER FUNCTION (see §0)

`#[cfg(test)] mod unit_tests` beside the code. **One test per ported function.** Where the vendor has
a case, port THEIRS rather than inventing one. Their *input* is MLIR text and **this crate has no
parser and must not get one**: build the typed input in the test and take the *expectation* from
their `CHECK` lines. A second test is justified only when the vendor's case plus one negative both
apply. Bridge 2's level 0 averaged 87 lines of test per function — roughly ten times the budget.

## 7. Your gates — and the one you must NOT run

```bash
cargo check -p deeptools          # ONCE per batch, at the end — not per function
cargo test  -p deeptools          # ONCE per batch
```

⛔ **Do not run the workspace build or the acceptance build in your worktree.** Its `target/` is ~6 GB,
this host has under 75 GB free, and a second campaign (bridge 2) is running concurrently with 8
agents; N agent worktrees doing that fills the disk and takes both campaigns down. The orchestrator
owns that gate.

For the record the crate's acceptance is **both models, both or neither** —
`cargo build -Fsuperdsc,model/granite-3.1-{2b,8b}-instruct,quant/fp8-dynamic-per-channel` — because
8b is hd=128/4096/12800 against 2b's 64/2048/8192, so 2b passing hides defects 8b finds. Neither
links on this Mac (`superdsc` pulls `flex-rs`, whose `build.rs` needs Spyre-only senlib headers). The
runnable prefix that still expands `#[forward]` over every staged program is
`cargo build -p scratchy-models --features superdsc,granite-3.1-2b-instruct,scratchy-quantizations/fp8-dynamic-per-channel`,
and it is the orchestrator's, not yours.

Commit one changeset and land it on the session branch exactly as your task prompt describes.
