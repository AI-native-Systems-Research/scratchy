# Bridge-3 campaign brief — read this in full before you edit anything

You are an agent on the **crustify bridge-3 campaign**: port the 130 C++ functions of IBM Spyre
`dcc`'s `SentientIR -> ProgIR` conversion (pass **D76**, the last MLIR rung) into the existing Rust
crate `crates/compiler/deeptools`, under `src/bridges/sentient_to_progir/`.

Read [`TASK.md`](TASK.md) (this directory) for the campaign statement, and
[`../crates/compiler/deeptools/CLAUDE.md`](../crates/compiler/deeptools/CLAUDE.md) for the crate's
rules. Both outrank the generic C-porting conventions in your system prompt.

## 0. 🛑 THE BUDGET — MEASURED ON BRIDGE 2, AND IT IS WHY THIS SECTION IS FIRST

Bridge 2's first 144 functions cost **43 hours of wall clock, 89% of it model time, 586 MILLION
cache-read tokens against 5 million output** — 30 to 48 turns per function. That rate does not finish
a campaign. The measurement also says exactly where it went: of 30,991 lines produced, **5,306 were
implementation, 12,333 were doc comments and 12,575 were tests.** 82% of the output was not the port.

So the following are **HARD CAPS**, not guidance. A batch that exceeds them will be sent back.

| per ported function | cap |
|---|---|
| doc comment lines | **8** — the `/// Replaces: eNNN_name` anchor, one line of what it does, and any TRAP. Nothing else. |
| test functions | **1**, or 2 where the vendor's own case plus one negative both apply |
| `cargo check`/`cargo test` runs | **once per BATCH**, at the end — not per function |

⛔ **DO NOT RE-VERIFY CITATIONS.** Take the `file:line` from `UNITS.tsv` as given. Re-measuring line
numbers cost a large share of the turns above, and **the review pass owns that check.**

⛔ **DO NOT GREP THE CRATE TO DISCOVER TYPES.** Your `.rs` home's header names the island types your
units need, and §5 below names the two islands. If something genuinely is not there, add it and say
so in the commit message — do not go looking first.

⛔ **NO TUTORIALS IN COMMENTS.** A trap is *"a REGULAR block always answers region index 0"*. A
tutorial is three paragraphs on why MLIR builds IR at run time. The first earns its 8 lines; the
second is what 12,333 doc lines were.

⭐ WHAT IS NOT CAPPED: **correctness, and the emission.** The op a function emits IS the function —
see §3. Cutting the port to hit a cap is the one failure worse than being slow.

## 1. Where the C++ actually is

⛔ **The authority is the C++ tree, not the extract.**

```
/Users/nickm/git/deeptools-src/<file>:<line>      # repo_info.txt: deeptools|master|a0d29abbed…
```

That is exactly the revision every banner cites. `crustify-bridge3/cpp/bridge3.cpp` tells you
**which** functions are in scope and **in what order**.

✅ **THIS EXTRACT IS VERIFIED UNTRUNCATED — 130 of 130.** `tools/verify_extract3.py` does not share
the extractor's slicing: it re-derives every function's end from the authority with its own
character-state-machine brace matcher and compares line counts, non-whitespace character counts and
then full content, and separately asserts each body's last line closes its function and was not
brace-padded. Two negative controls confirm it bites. So unlike bridge 2 — where 366 of 384 bodies
were silently cut at the tail — you may read the extract as complete. **Still port from the
authority file at the cited line** when a body is long: it has the surrounding declarations.

⛔ `/Users/nickm/git/deeptools` is a *different* revision. Do not use it.

`cpp/prelude.inc` is a catalogue of the 45 qualified scopes and 378 external calls the bodies reach.
It carries names only, no behaviour, and is not a porting input.

## 2. Resolving your worklist

Your worklist names units as `e<NNN>_<cppName>`. `NNN` is the entry number.

```bash
grep -P '^e086_' crustify-bridge3/UNITS.tsv    # unit, entry, level, LoC, authority file:line,
                                               # extract line range, Rust home, callees
```

`crustify-bridge3/UNITS.tsv` is the map for all 130; `crustify-bridge3/EXCLUSIONS.tsv` lists the 36
definitions deliberately NOT scheduled, with a reason each. Your `.rs` home already exists, carries
the citation table for its units in its module doc, and holds one `// crustify:todo: e<NNN>_<name>`
per scheduled unit. Replace each with the ported function carrying `/// Replaces: e<NNN>_<name>`.
**A surviving TODO is open work.**

## 3. What "ported" means here

⛔ **The whole function, including its emission.** An earlier hand attempt on bridge 2 extracted each
function's *decision rule* into a documented predicate, left out the part that emits an operation,
and reported it done — nothing called any of it. **The op a function emits IS the function**, with its
exact attribute names and values, its branch order and its early returns.

What you MAY drop is the *mechanism for reaching operands*: walking uses, memoising by
`(core, corelet, component)`, positioning an `OpBuilder`. Those exist because the C++ builds an IR at
run time and this crate builds a typed value at compile time.

⛔ **If the target island cannot express a function's output, EXTEND THE ISLAND**
(`src/islands/progir/`) rather than concluding the function is unnecessary. The ProgIR island is
**571 lines and deliberately small** — it will need extending, and that is expected work, not a
detour. Say what you added in the commit message.

## 4. What this campaign is NOT

Your system prompt's conventions cover C-to-Rust *wrapping*. **None of the following applies here**
— this is a pure-logic port of C++ into an existing crate, with no C left anywhere:

- ❌ no `bindgen`, no allowlist, no `-sys` crate
- ❌ no `ffi::`, no `mod ffi_export`, no `#[unsafe(no_mangle)] extern "C"`, no `CRUSTIFY_<FILE>` switch
- ❌ no `Foo`/`FooRef`/`FooMut` wrapped-layout triple, no `addr_of!` projection, no `unsafe`
- ❌ no C build, no sanitizers, no `io_equiv` C-vs-Rust harness (there is no C to call — the reference
  is a compiler that does not run on this host)

## 5. The two islands you work between

```
crates/compiler/deeptools/src/islands/sentient/   INPUT  — 29 ops, 16 enums, a printer
crates/compiler/deeptools/src/islands/progir/     OUTPUT — Instruction, Block, UnitProgram,
                                                           Program<A,M,W>, RegInit, UnitRegState,
                                                           OperandValue, RegType, Overflow, Invalid
```

The ProgIR island is the faithful shape of the reference's `ProgramAndStateInfo` (`progir.h:505-556`):
`per_unit: Vec<(Component, UnitProgram)>`, `reg_state`, `variable_definitions`. An `Instruction` is
`{opcode, symbolic_opcode, fields: Vec<(OperandField, OperandValue)>, dead, tag, comment}`, with
`OpCode` and `OperandField` re-exported from `sys_arch_spec` — **91 opcodes and 89 operand fields
already vendored**, so an opcode no unit has cannot be written down, and you must not add a
hand-transcribed second copy of either.

`Block` is **nested, not a `prev`/`next` graph** — a closed loop is a `body`. `CONDITION_END`,
`FORLOOP_END` and `DUMMY` are not variants, because nesting states what they marked.

## 6. Crate rules that DO bind you (`crates/compiler/deeptools/CLAUDE.md`)

- 🛑 **NEVER RUNTIME REFUSE.** No `Result`, no `Err(`, no `.ok_or`, no `assert!`, no `debug_assert!`.
  A closed set is an `enum`; an invariant is a type. `todo!("<op> not ported")` is tolerated, capped,
  and ratcheted down only.
  ⛔⛔ **THIS SPAN HAS FOUR CHECKING FUNCTIONS** — `checkRegDefs`, `verifyOnTheFlyConversions`,
  `checkProgramValidity`'s callers, and `runOnOperation`'s `signalPassFailure`. Port each as a
  function that **RETURNS THE OFFENDERS**, the way `progir::Program::overflowing` already does.
  Never as an assert, a `Result`, or a silent `false`.
- **Newtypes, never raw scalars** (`Rows`, `Cols`, `Elements`, `Bytes`, `Bits`, `Sticks`, `RegIndex`,
  …): transposing two extents must be an E0308. A `-1` sentinel is an `Option`, never a negative.
- **No strings for closed sets** — every enumeration is an `enum`.
- Const-generic traits `Arch`, `Model`, `Workload` flow through.
- ⛔ Never substitute a stand-in op to dodge a `todo!`.

## 7. Tests — ⛔ ONE PER FUNCTION (see §0)

`#[cfg(test)] mod unit_tests` beside the code. **One test per ported function.** Where the vendor has
a case, port THEIRS rather than inventing one — the authority tree's `dcc/test/` carries `CHECK`
expectations. Their *input* is MLIR text and **this crate has no parser and must not get one**: build
the typed input in the test and take the *expectation* from their `CHECK` lines.

## 8. Your gates — and the one you must NOT run

```bash
cargo check -p deeptools          # ONCE per batch, at the end — not per function
cargo test  -p deeptools          # ONCE per batch
```

⛔ **Do not run the workspace build or the acceptance build in your worktree.** Its `target/` is
~6 GB, this host is near capacity, and two sibling campaigns are running. The orchestrator owns that
gate.

Commit one changeset and land it on the session branch exactly as your task prompt describes.
