# Port bridge 1 of IBM Spyre deeptools to Rust

## Objective

Port the **110 C++ functions in `cpp/bridge1.cpp`** to Rust. That is the whole job.

They are the **SuperDSC → DataflowIR** conversion of IBM Spyre's compiler, plus the four DDL-compiler
derivations that conversion depends on but never calls. The Rust lands in an existing crate,
`crates/compiler/deeptools`, under `src/bridges/superdsc_to_dataflow_ir/`.

## The two components

**1. `dsc-based-utils/DSC2ToDataflowIR` — 106 units, 7,556 declaration lines, 9 files.**
This IS the conversion: its header guard is literally `SUPERDSCTODATAFLOWIR`, and its constructor is
`DSC2ToDataflowIR(SuperDsc *s, MLIRContext&, bool uniformization, const DesignSpaceConfigGlobal&)`.

| file | units |
|---|---|
| `V3/SNTransferLowering.cpp` | 29 |
| `V3/SNComputeLowering.cpp` | 18 |
| `V3/SNControlFlowLowering.cpp` | 14 |
| `V3/SNDSCLowering.cpp` | 12 |
| `DSC2ToDataflowIR.cpp` | 11 |
| `DSC2ToDataflowIRUtils.hpp` | 11 |
| `DataflowIRConstructionUtils.hpp` | 6 |
| `V3/SNSyncLowering.cpp` | 4 |
| `V3/SNStickMaskLowering.cpp` | 1 |

**2. The DDL compiler's DERIVATIONS — 4 units.**
The conversion does not call these: in the reference they run *before* a `SuperDsc` exists, so a call
graph rooted at the conversion is blind to them. They are named explicitly because **scratchy
currently INVENTS them** in `crates/compiler/deeptools/src/islands/.../shape.rs`, and removing that
invention is why this campaign exists.

| unit | authority | notes |
|---|---|---|
| `checkConstraints` | `ddc/ddcv1.cpp:792` (130L) | a **lambda** nested in `Ddc::exploreAssignDataStages`; lexically CONTAINS `checkConstraintsImpl` (`:825`, 82L), so one unit covers both. Called at `:967`, `:1191`, `:1194`, `:1377`. |
| `createDataConnectMetadata` | `ddc/ddcv1.cpp:3283` (45L) | **the port identity.** |
| `getCumulativeStickSizes` | `dsc/dsc2.cpp:4109` (16L) | |
| `getStickSizes` | `dsc/dsc2.cpp:4066` (23L) | carries the actual per-dim stick arithmetic. |

⚠️ **A correction to the campaign brief, measured.** The brief lists `getCumulativeStickSizes` as
present in `ddc/ddcv1.cpp` at `:789`/`:1723`. Those are **call sites** on a `currDsc` object. The
definition is `DesignSpaceConfig::getCumulativeStickSizes` at `dsc/dsc2.cpp:4109` (declared
`dsc/designSpaceConfig.h:260`), and its body is a six-line fold over `getStickSizes` — so
`getStickSizes` comes with it. Those are the only 2 units taken from `dsc/`; the rest of that
39,440-line data model stays out of scope.

⭐ **ABSOLUTE STAGE EXTENTS ARE A CONSTRAINT SYSTEM, NOT A FORMULA** — ratios against a tensor,
multiples, stage-relative bounds, SETs, bare bounds, and a different rule per `ddl.if` arm. Port the
constraint system. Trip counts need none of it: a loop's own two stages are dimensionless.

## The source

`cpp/bridge1.cpp` — one self-contained C++17 translation unit, 7,918 lines.

- Every function body is **verbatim** from the reference, and **verified so**: all 110, by
  `tools/verify_extract.py`, which re-derives each function's end from the authority with its own
  brace matcher and compares LENGTHS and CONTENT. Two negative controls confirm the check fails when
  a body *is* truncated.
- Each body is preceded by a banner giving its entry number and its **original** location:
  `// ---- 70/110  GenerateConstantBitStreamAndShuffle  --  dsc-based-utils/…/SNTransferLowering.cpp:2475  (41L)`
- Functions appear in **dependency order**: level 0 calls nothing else in the span, level *N* only
  levels below it. Level banners separate them. 11 levels: L0=41, L1=30, L2=11, L3=10, L4=6, L5=4,
  L6=2, L7=1, L8=2, L9=2, L10=1.
- Member definitions are rewritten as free functions (`Class::foo` → `eNNN_foo`) so the unit needs no
  class declarations. **Only the declarator name is rewritten; bodies are untouched.**

`cpp/prelude.inc` catalogues the 116 qualified scopes and 217 external calls the bodies reach. It
carries **no behaviour**, is **not a porting input**, and does not make the unit compile — these
bodies do heavy member access on opaque reference types (`mlir::Value`, `OpBuilder`, `dsc2::*`) that
no behaviour-free stand-in satisfies without the real headers. Nothing builds it and no gate asks it
to.

## What "ported" means here

⛔ **The whole function, including its emission.** A previous attempt extracted each function's
*decision rule* into a documented predicate — `is_l3()`, `BurstSetting::of()` — left out the part
that emits an operation, and reported it as done. Nothing called any of it. **The op a function emits
IS the function**, with its exact attribute names and values.

What may be dropped is the *mechanism for reaching operands*: walking uses, memoising by
`(core, corelet, component)`, positioning an `OpBuilder`. Those exist because the C++ builds an IR at
run time. What must not be dropped is what gets emitted, or with which attribute names and values.

⛔ **If the target IR cannot express a function's input, add the operation to the island** rather than
concluding the function is unnecessary. **That call is not the porter's to make.**

## What the output must be

The Rust lowers a **typed value**, not text:

```
scratchy's SuperDSC (SdscOp / SdscFolds)   ──this port──►   deeptools::islands::dataflow_ir::Run<A>
```

- **Input**, already typed: `crates/targets/spyre/src/lower_subtile_tape_to_superdsc.rs` (12,597
  lines), carrying `SdscOp`/`SdscFolds` with C++-mirroring field names (`opConsts`, `backGapCore_`,
  `coordinates_`, `constantInfo_`), validated against the vendor's `dxp_standalone`.
  ⚠️ Several fields are untyped `serde_json::Value` escape hatches. A ported lowering that reads one
  needs it **typed**, and tightening those is **in scope**.
- **Output**, already exists: `crates/compiler/deeptools/src/islands/dataflow_ir/` — ops, types,
  printer (`ty.rs` alone is 3,271 lines).
- There is no MLIR text on either side and no parser in this crate. There must not be one.

⛔ **This lands BESIDE `src/bridges/subtile_to_dataflow_ir/`, which it is intended to replace.** Do
not delete or edit that module in this campaign; switching the call site is a separate, deliberate
step after the gate is green on both models.

## Crate conventions the port must honour

- 🛑 **No runtime refusals** — no `Result`, `Err(`, `.ok_or`, `assert!`, `debug_assert!`; frozen at
  zero by `crates/targets/spyre/tests/dfir_never_runtime_refuses.rs`. A closed set is an `enum`; an
  invariant is a type. `todo!` naming an unported op is allowed and is how the build names the next gap.
- **Newtypes, never raw scalars** — `Rows`, `Cols`, `Elements`, `Bytes`, `Sticks`, `Segment`.
  Transposing two extents must be a type error.
- **No strings for closed sets.**
- Const-generic traits `Arch`, `Model`, `Workload` flow through, and a flag that decides which ops
  exist must REMOVE ops.

## Acceptance

**Both models, both or neither** (`crates/compiler/deeptools/CLAUDE.md`):

```
cargo build -Fsuperdsc,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel
cargo build -Fsuperdsc,model/granite-3.1-8b-instruct,quant/fp8-dynamic-per-channel
```

8b is hd=128/4096/12800 against 2b's 64/2048/8192 — a different door arm and tile geometry, so 2b
passing hides defects 8b finds. Read the `-vv` stream; cargo hides build-script output. Neither links
on this Mac; the runnable prefix is in `build.json`. **dbo-opt is the only oracle** — every defect
this bridge has fixed was named by a dbo-opt refusal on our emitted MLIR.

## Authority

```
/Users/nickm/git/deeptools-src     repo_info.txt: deeptools|master|a0d29abbedfa2dd44ec7255e59440b06a429118c
```

**That tree is the ultimate authority**; every banner's `file:line` resolves against it.
⛔ `/Users/nickm/git/deeptools` is a **different revision**. Do not use it.
