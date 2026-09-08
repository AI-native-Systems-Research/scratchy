# Bridge-4 campaign brief — read this in full before you edit anything

You are an agent on the **crustify bridge-4 campaign**: port the 33 C++ functions of IBM Spyre
`sys-arch-spec`'s `ProgIR -> SenProg` lowering into the existing Rust crate
`crates/compiler/deeptools`, under `src/bridges/progir_to_senprog/`.

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

That is exactly the revision every banner cites. `/Users/nickm/git/deeptools` is a *different*
revision — do not use it.

⭐ **THIS CAMPAIGN'S EXTRACT IS VERIFIED VERBATIM — 33 of 33.** `crustify-bridge4/cpp/bridge4.cpp`
was checked by `tools/verify_extract.py`, which shares no code with the extractor: it re-derives each
body's end from the authority with its own scanner and compares length **and** content, asserts brace
depth reaches zero at the last character, and never appends a brace. Both negative controls fail as
required (drop 3 lines → length mismatch; append a brace → length mismatch). It also caught a real
defect on the way: 24 of the header in-class bodies had lost their leading indentation, so they were
not byte-verbatim until fixed.

⛔ SO YOU MAY READ THE EXTRACT, BUT CITE AND CHECK AGAINST THE AUTHORITY. Bridge 2's extract was
*not* verified this way and 366 of its 384 bodies turned out truncated at the tail — its check
re-sliced the source using the same length the extractor used, so it compared what was copied against
what was copied. Never trust an extract whose verification could not have failed.

## 1a. 🛑 THE EMISSION IS SPREAD OVER FOUR FILES, AND `dpc.cpp` HOLDS ONLY THREE UNITS

⛔ `Dpc::convertIr2Senprog` (`sys-arch-spec/dpc/dpc.cpp:615`) is the entry point, but the transitive
closure of the *emission* is **33 units across four files**: `dpc.cpp` (3), `progir.cpp` (4),
**`progir.h` (18, all IN-CLASS definitions)** and `isa.cpp` (6). A `.cpp`-only scan would miss
eighteen of the thirty-three. The operand text itself comes from `OperandAttr::print`
(`progir.cpp:25-62`), and the emission indexes six declared-data tables — `typeToFieldEncoding`,
`typeToFieldBitShift`, `senComponentsToString`, `regTypeToString`, `progFormatFeaturesMap`,
`typeToFieldName`.

⭐ **SENPROG IS A PRINT FORMAT.** What is being ported is the exact text: field order, spelling,
separators, and which fields are omitted. A function that decides *what* to write without writing it
is not a port.

⛔ AND A CORELET IS A SINGLE CHARACTER HERE. `dpc.cpp:638-639` is
`corelet = unitName.back(); unitName.pop_back();`, with L3LU/L3SU special-cased to `"0"` at
`:635-636`. Bridge 2's `ExtendUnitNameToCorelet` only ever writes `0` or `1`, so a third corelet is
silently mislabelled at both ends of the ladder.

## 2. Resolving your worklist

Your worklist names units as `e<NNN>_<cppName>`. `NNN` is the entry number.

```bash
grep -P '^e032_' crustify-bridge4/UNITS.tsv     # entry, level, LoC, authority file:line,
                                                # extract line range, Rust home, callees
```

`crustify-bridge4/UNITS.tsv` is the map for all 33 — entry, unit, qualified name, authority
`file:line`, LoC, the extract's own line range, and callees. Your
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

⛔ **If the target island cannot express something, extend it** rather than concluding the function is
unnecessary. This rung reads `src/islands/progir/` and writes SenProg text.

⚠️ **THE OUTPUT ISLAND IS AN OPEN DECISION, AND IT IS DELIBERATELY NOT YOURS TO SETTLE ALONE.** Branch
`deeptools-islands` carries a 2,193-line `islands/senprog.rs` whose writer (`:574-767`) and types
(`:291-476`) are densely cited to `dpc.cpp:615-777` and — uniquely in this project — were checked
byte-for-byte against a tool nobody here wrote (`dip_standalone -s`). Roughly 420 of those lines are
worth adopting; the other 1,426 (a guard/FIFO/latch layer at `:768-2193`) hold 16 `assert!`, 5
`panic!`, a `Result` swallowed by `.unwrap_or_default()`, six mutable global atomics and one law
tautologised to `assert!(true || …)`, none of which this crate permits. Its central type is also the
senulator's untyped `i32 operands[]` with an `0xFFFF`-absent sentinel, which makes the newtype rule
unsatisfiable until replaced. **Read it, port against the authority, and say in your commit message
which parts you adopted and why.**

⚠️ `bridges/progir_to_senprog.rs` on that same branch (1,930 lines) is ~40% real emission and ~60%
validation and senulator coupling. Reference only.

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
