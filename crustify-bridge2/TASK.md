# Port bridge 2 of IBM Spyre deeptools to Rust

## Objective

Port the **384 C++ functions in `source/bridge2.cpp`** to Rust. That is the whole job.

They are the `DataflowIR -> SentientIR` lowering of IBM Spyre's `dcc` compiler — passes D1-D28 of its
76-pass pipeline. The Rust lands in an existing crate, `crates/compiler/deeptools`, under
`src/bridges/dataflow_ir_to_sentient/`.

## The source

`source/bridge2.cpp` — one self-contained C++17 translation unit, 16,788 lines.

- Every function body is **verbatim** from the reference.
- Each is preceded by a banner giving its entry number and its **original** location:
  `// ---- 371/384  constructLoadAndStoreStmt  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2167  (177L)`
- Functions appear in **dependency order**: level 0 calls nothing else in the span, level *N* only
  levels below it. Level banners separate them.
- Member definitions have been rewritten as free functions (`Class::foo` → `eNNN_foo`) so the unit
  needs no class declarations.

`source/prelude.inc` — plain-C++ stand-ins for every name the bodies call. **There is no MLIR, no
LLVM and no `dcc` header anywhere in this campaign.** The stand-ins carry the *names and shapes* of
the calls; they carry no behaviour, because the behaviour being ported is in the bodies.

⚠️ **The unit does not yet compile cleanly** — roughly 500 name-resolution errors remain, all inside
the generated `prelude.inc` (names needing to be declared as namespace vs struct vs template vs free
function, and collisions among those). The bodies themselves are complete and unmodified.

## What "ported" means here

⛔ **The whole function, including its emission.** A previous hand-porting attempt extracted each
function's *decision rule* into a documented predicate — `is_l3()`, `BurstSetting::of()`,
`Granularity::checked()` — left out the part that emits an operation, and reported it as done.
Nothing called any of it. **The op a function emits IS the function.**

What may be dropped is the *mechanism for reaching operands*: walking uses, memoising by
`(core, corelet, component)`, positioning an `OpBuilder`. Those exist because the C++ builds an IR at
run time. What must not be dropped is what gets emitted, or with which attribute names and values.

⛔ **If the target IR cannot express a function's input, add the operation to it** rather than
concluding the function is unnecessary.

## What the output must be

The Rust lowers a **typed value**, not text:

```
deeptools::islands::dataflow_ir::Run<A>   ──this port──►   deeptools::islands::sentient::Run<A, M, W>
```

Both islands already exist in the crate (`src/islands/dataflow_ir/`, `src/islands/sentient/`), with
their ops, types and printers. `src/bridges/dataflow_ir_to_sentient/` holds a partial spine to build
on. There is no MLIR parser in the crate and there must not be one.

Measured across the 417 programs a real build stages, the reference's SentientIR uses **seven** of the
dialect's 29 ops: `scalar_constant`, `load_and_store`, `load_and_send`, `vector_binary`,
`receive_and_store`, `vector_mac`, `for`. `dataflow.get_unit` and `dataflow.program_unit` survive into
the output — the rung is mixed, not a dialect swap.

## Crate conventions the port must honour

- **No runtime refusals.** No `Result`, no `Err`, no `.ok_or`, no `assert!`, no `debug_assert!`. A
  closed set is an `enum`; an invariant is a type. `todo!` naming an unported operation is allowed.
- **Newtypes, never raw scalars** — `Rows`, `Cols`, `Elements`, `Bytes`, `Bits`, `Sticks`. Transposing
  two extents must be a type error.
- **No strings for closed sets** — every enumeration is a generated or hand-written `enum`.
- Const-generic traits `Arch`, `Model`, `Workload` flow through.

## Acceptance

`cargo build -Fsuperdsc,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel` going through,
read from the `-vv` stream. It runs the lowering over all 417 staged programs, so a `todo!` stops it
and names the operation.

Unit tests come with each function. Where the vendor has a case for it, port theirs: `dcc/test/` holds
825 `.mlir` tests and **668 carry `CHECK-SENT-IR`** expectations — DataflowIR input beside the
SentientIR the reference produces for it. Their input is text and this crate has no parser, so build
the typed input in the test and take the *expectation* from their `CHECK` lines.

## Reference material in the repo

- `../crates/compiler/deeptools/docs/bridge2-porting-order.md` — all 384 entries with a PORT and an
  AUDIT checkbox each, in dependency order, plus the 106 excluded definitions with a reason for each.
- `../crates/compiler/deeptools/tests/sentient_corpus/` — 18 pairs of the crate's own emitted
  DataflowIR beside the reference's own SentientIR for the same program. The answer key.
- `../crates/compiler/deeptools/CLAUDE.md` — the crate's rules.

## Authority

The C++ in `source/bridge2.cpp` is extracted from the pod at `/project_src/deeptools`, revision
`a0d29abbed` (`stable-2026_07_24-142907-715-ga0d29abbed`). **That tree is the ultimate authority**; the
banners' `file:line` citations resolve against it, so any port can be diffed against the original.
