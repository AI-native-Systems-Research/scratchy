# Port bridge 3 of IBM Spyre deeptools to Rust

## Objective

Port the **130 C++ functions in `cpp/bridge3.cpp`** to Rust. That is the whole job.

They are the `SentientIR -> ProgIR` conversion of IBM Spyre's `dcc` compiler — pass **D76**, the last
rung of its 76-pass pipeline and the one that leaves MLIR. The Rust lands in an existing crate,
`crates/compiler/deeptools`, under `src/bridges/sentient_to_progir/`.

## The source

`cpp/bridge3.cpp` — one self-contained C++17 translation unit, 7,194 lines, holding 6,879
declaration lines of body.

- Every function body is **verbatim** from the reference, and **verified untruncated: 130 of 130**
  (see *Authority* below).
- Each is preceded by a banner giving its entry number and its **original** location:
  `// ---- 86/130  ConstructBinaryInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1715  (485L)`
- Functions appear in **dependency order**: level 0 calls nothing else in the span, level *N* only
  levels below it. Level banners separate them. ⛔ **Levels 0-7, with ONE mutually recursive
  component**: `LowerUniformOperations` and `GenerateProgIR` share level 5 and cannot be split by a
  wave barrier.
- Member definitions have been rewritten as free functions (`Class::foo` → `eNNN_foo`) so the unit
  needs no class declarations. `~UniformRegionContext` became `e070_dtor_UniformRegionContext`,
  because `~` is neither a legal identifier nor a match for the campaign's anchor regex.

`cpp/prelude.inc` — a catalogue of the 45 qualified scopes and 378 external calls the bodies reach,
so the unit is self-describing with no MLIR, LLVM or `dcc` header. It carries **names only**; the
behaviour being ported is in the bodies. ⚠️ The unit does not compile, and nothing builds it: these
bodies do heavy member access on opaque reference types. The **verified-verbatim bodies** are the
load-bearing property, not compilability.

## What is NOT in this campaign

⛔⛔ **`dcc/src/Transform/Sentient/` — 32,766 lines of D29-D75 passes — IS NOT YOURS.** Those passes
rewrite SentientIR *in place* before D76 runs: `RegisterAllocation.cpp`,
`SmartRegisterAllocation.cpp`, `RegisterTypeAssignment.cpp`, `AddressPinningAndToggle.cpp`,
`LiveRangeReduction.cpp`, `OpRerolling.cpp`, `LoopRolling.cpp`, and forty more. They are
island-internal transforms, not this conversion.

If a function you port depends on state those passes establish — register assignments, pinned
addresses, rerolled loops — **port the conversion as the reference writes it and record the
dependency in the commit message.** Do NOT port the pass, and do NOT invent the state.

`EXCLUSIONS.tsv` lists the 36 definitions inside the conversion directory that are also not
scheduled, with a reason each: 33 are 1-4 line header inlines over a member (a struct field in Rust,
not a function), and the three substantive ones are an ostream configuration helper and two MLIR
pass-registry factories.

## What "ported" means here

⛔ **The whole function, including its emission.** A previous hand-porting attempt on bridge 2
extracted each function's *decision rule* into a documented predicate, left out the part that emits
an operation, and reported it as done. Nothing called any of it. **The op a function emits IS the
function.**

What may be dropped is the *mechanism for reaching operands*: walking uses, memoising by
`(core, corelet, component)`, positioning an `OpBuilder`. Those exist because the C++ builds an IR at
run time. What must not be dropped is what gets emitted, or with which attribute names and values.

⛔ **If the target island cannot express a function's output, add it to the island** rather than
concluding the function is unnecessary.

## What the output must be

The Rust lowers a **typed value**, not text:

```
deeptools::islands::sentient::Run<A, M, W>   ──this port──►   deeptools::islands::progir::Program<A, M, W>
```

Both islands already exist in the crate. ⭐ **PROGIR IS NOT AN MLIR DIALECT** — `SentientToProgIR`
fills in a plain C++ structure, `std::map<int, ProgramAndStateInfo>` keyed by core id, and the MLIR
module it leaves behind holds only `init`'s reference to it (`replaceProgramBodyWithSmcOp`, twelve
lines). So this bridge produces a value; the only text is `init.smc`.

⭐ **AND THE OUTPUT IS TINY.** `kMaxCompIBuff = 256` — *"Maximum number of instructions on any
unit"* — with `kMaxCompRegs = 128` (`progir.h:506-510`). A complete int8 batched matmul compiles to
**eight** instructions on one unit. The real per-unit bound is `max_ibuff_entries(unit)`, per
component AND per arch (`sysdef.cpp:451-500`) — 256 for the L3 halves, 128 for PT/PE/SFP/L0/LXSU.

⛔ **The ProgIR island is 571 lines and WILL need extending.** That is scheduled work, not scope
creep: the island has `Instruction`, `Block`, `UnitProgram`, `Program<A,M,W>`, `RegInit`,
`UnitRegState`, `OperandValue`, `RegType`, `Overflow`, `Invalid`, and `OpCode`/`OperandField`
re-exported from `sys-arch-spec` (91 opcodes, 89 operand fields, already vendored — **never
hand-transcribe a second copy**). Whatever the uniform-instruction and uniform-block machinery needs
beyond that is to be added there.

## Crate conventions the port must honour

- **No runtime refusals.** No `Result`, no `Err`, no `.ok_or`, no `assert!`, no `debug_assert!`. A
  closed set is an `enum`; an invariant is a type. `todo!` naming an unported operation is allowed.
  ⛔⛔ This span has **four checking functions** (`checkRegDefs`, `verifyOnTheFlyConversions`, and
  the validity/`signalPassFailure` paths in `runOnOperation`). Each is ported as a function that
  **returns the offenders**, the way `progir::Program::overflowing` already does.
- **Newtypes, never raw scalars** — `Rows`, `Cols`, `Elements`, `Bytes`, `Bits`, `Sticks`, `RegIndex`.
  Transposing two extents must be a type error. A `-1` sentinel is an `Option`.
- **No strings for closed sets** — every enumeration is a generated or hand-written `enum`.
- Const-generic traits `Arch`, `Model`, `Workload` flow through.

## Acceptance

`cargo build -Fsuperdsc,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel` going through,
read from the `-vv` stream. Per-batch, the agent gate is `cargo check -p deeptools` +
`cargo test -p deeptools`, run **once at the end of the batch**.

## Authority

The C++ in `cpp/bridge3.cpp` is extracted from `/Users/nickm/git/deeptools-src`, revision
`a0d29abbed` (`repo_info.txt: deeptools|master|a0d29abbedfa2dd44ec7255e59440b06a429118c`). **That tree
is the ultimate authority**; every banner's `file:line` resolves against it.

✅ **The extraction is verified untruncated, 130 of 130.** `tools/verify_extract3.py` does not import
the extractor's scanner: it re-derives each function's end from the authority with its own
character-state-machine matcher and compares line counts, non-whitespace character counts and then
full content. It also asserts each body's last non-blank line closes its function and matches the
authority's own last line, so a body cannot have been brace-padded. **Two negative controls confirm
the check bites**: a tail cut re-balanced with a bare `}` and a brace-neutral 4-line cut from the
middle of a body were both caught.

This matters because bridge 2's extract truncated **366 of 384** bodies and its check reported
"verbatim, 12 sampled, none differed" — it re-sliced the source with the same (file, line, length)
the extractor had used, comparing what was copied against what was copied.
