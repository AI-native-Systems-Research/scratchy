# The ddc / L3-scheduler campaign — the scheduling and address-placement stage

**382 units · 4 stages of `runDdc` · dependency levels 0..9 · 25,529 in-scope body lines**

Authority, LOCAL, revision `a0d29abbed`: **`/Users/nickm/git/deeptools-src`**.
Not the pod. Not the other local `deeptools` checkout (a different revision).

---

## The spec is 60 lines. Read it first.

`dbo-opt` is the binary scratchy shells out to, and its per-program pipeline runs `runDdc` for
every program (`dbo/src/Transforms/sdsc_bundle/RunSchedulerOnSdsc.cpp:145`).
**`dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp` is the spec for this whole job.** Four stages:

| # | stage | authority | in campaign |
|---|---|---|---|
| 1 | `sbf::doCoreletSplitSdsc(SuperDsc*)` | `dbo/src/Utils/sdsc_bundle/SdscCoreletSplit.cpp:84` | ⛔ **excluded** — `SchedulerStages.cpp:25` returns early unless `numCoreletsPerCore == 2`, and scratchy emits `numCoreletsUsed_ = 1` in all 313 sampled SuperDSCs |
| 2a | `L3DlOpsScheduler(..).run(sdsc)` | `dcg/dcg_fe/scheduler/` — 8,033 `.cpp` lines | ✅ 144 units, **zero** previously ported |
| 2b | `ddc::Ddc(..).run_v1(sdsc)` | entry `ddc/ddcv1.cpp:3695`; `ddc/` — 18,830 `.cpp` lines | ✅ 222 units (178 `ddc` + 44 `ddl`), **four** previously ported |
| 3 | `DcgManager::runDcgForDlOpsStandalone(sdsc)` | `dcg/dcg_manager/dcg_manager.cpp:449` | ✅ 16 units |

`runDdc` raises when DDC finds no mapping (*"Scheduler failed to find a suitable op mapping"*), so
the mapping is not optional.

⛔ **Stage 3 is `runDcgForDlOpsStandalone`, NOT `runDcg`** — `SchedulerStages.cpp:53-57` picks it
whenever `dscs_` is non-empty, always true for scratchy's input.
⚠️ Its body (`dcg_manager.cpp:449-513`) delegates almost entirely to `dcg_fe/pcfg_gen/` and
`dcg_be/`, both **out of scope**. Its in-scope effect is the `mySDsc.pcfg_` and
`firstAvailGlobalGrpId` mutation. `todo!` naming the missing translator is correct there; **do not
invent a PCFG.**

⛔ Out of scope: `dcg/dcg_be/`, `dcg/dcg_fe/pcfg_gen/` (~48,000 lines) — the user has ruled the
PCFG / data-DSC path off our path. The four standalone `main()` drivers are harnesses, not the path.

---

## ⭐⭐ The acceptance criterion

These stages mutate the `SuperDsc` **in place**, and the observable result is that
**the `ScheduleNode` tree gains its LOOP, TRANSFER, SYNC and CONDITION nodes.**

Scratchy's SuperDSC today has only ALLOCATE nodes (one construction site:
`crates/targets/spyre/src/lower_subtile_tape_to_superdsc.rs:5127`) plus a flat `computeOp_` list —
which matches torch-spyre's own `generate_sdsc`, and is exactly why `dxp_standalone` works and the
Rust `sdscToDataflowIR` port yields nothing.

The **representative node-minting site to model** is `ddc/ddl/ddl_conversion.cpp:1065`: it mints a
`dsc2::LoopNode`, takes its dims from the DDL, names it `loop_ds<num>_ds<den>`, and registers it in
`ddlInterface.loop_labels_`.

⛔ **A port that documents the scheduling decision without adding nodes to that tree is not a port.**

## ⭐⭐ The target vocabulary is already typed

A wrong shape must be a **compile error**, not a judgement call. Emit into these existing types;
invent none:

| type | where |
|---|---|
| `Statement` | `src/bridges/superdsc_to_dataflow_ir/driver.rs:467` |
| `Scheduled` | `driver.rs:1152` |
| `Viewed` / `Viewing` | `driver.rs:1756-1790` |
| `ScheduleView::roots` | `driver.rs:1539` |
| `Dsc` | `driver.rs:1788` |

The single function it all plugs into is **`Schedule::roots`** in
`crates/targets/spyre/src/lower_superdsc_to_dataflow_ir.rs` — committed, compiles, and currently
yields nothing. Both the component and the DSC are already in hand there.

---

## Why this matters

`ddc.run_v1` is what **places addresses**, and the L3 scheduler's `run` commits LX allocations then
calls `fillAllocationStartAddrAndOffset` — literally *"Set start address, offset in allocations"*
(`dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:8000`). Our emitted views have printed
`start_address = 0` where the reference states a placed base; the backend has refused with
`Register initialization out of boundary`; and `crates/compiler/deeptools/src/reginit.rs`
(1,569 lines, on the **integ** branch) hand-computes placement from `ddc/ddcv1.cpp:132-360`.

This campaign replaces that guesswork with the real thing. The superseding units are those for
`Ddc::allocAllMem`, `Ddc::minimizeAllocations`, `Ddc::calculateClStartAddress` and
`Ddc::finalizeOps`; each carries a `NOTE` in `UNITS.tsv` and on its own anchor.

## ⚠️ The L3 scheduler — do not decide this, port it

Scratchy states its own schedule (our SuperDSC writes `coreIdToDscSchedule`) and
`L3DlOpsScheduler.cpp:415-425` reads that field. **That question is the user's, not the porter's**,
and being wrong about it costs a whole rediscovery.

Measured while scoping, and it points toward "reachable": `coreIdToDscSchedule` occurs in
`L3DlOpsScheduler.cpp` **only as a read** (`:425`), never a write, and nowhere in `ddc/` outside
test JSON. So the field is an **input** to both stages. What the L3 scheduler *produces* is the
dsc2 schedule tree, the LX buffer type, the committed LX allocations and their start addresses.

---

## Already ported — do not re-port, do not duplicate

All four in `crates/compiler/deeptools/src/bridges/superdsc_to_dataflow_ir/shape_constraints.rs`,
following the same `/// Replaces: eNNN_name` convention so cross-referencing works:

- `e001_checkConstraints` — `ddc/ddcv1.cpp:792`
- `e002_createDataConnectMetadata` — `ddc/ddcv1.cpp:3283`
- `e041_getStickSizes` — `dsc/dsc2.cpp:4066`, a `DesignSpaceConfig` method
- `e071_getCumulativeStickSizes` — `dsc/dsc2.cpp`, a `DesignSpaceConfig` method

The last two are **outside this campaign's file list**, so they were never enumerated — our units
*call* them. And `checkConstraints` is a **lambda inside `Ddc::exploreAssignDataStages`**, so
porting that unit means **calling** the existing port, not writing a second constraint checker.

---

## Enumeration — how the count was reached

Definitions were found **by brace matching** from each signature's opening paren, comment- and
string-aware, never by a signature regex (a signature regex undercounted bridge 2 by 72%). Headers
were scanned too, and in-class definitions with them — the enclosing class is recovered from the
brace structure, because an in-class member definition carries no `Class::` qualifier.

**444 definitions found · 382 in scope · 62 excluded**, each with a stated reason in
`EXCLUSIONS.tsv` (57 one-to-three-line field accessors, 2 trivial ctor/dtors, 2 named exclusions,
1 already ported) and **none for being hard**. Cross-checked independently against each file's
column-0 closing-brace count.

Levels are computed over the **SCC condensation** (Tarjan), never a plain longest-path fixpoint —
bridge 3's fixpoint printed "level 266" on a mutually recursive pair. This span is **acyclic**:
382 SCCs over 382 units, 617 resolved call edges, levels 0..9.

## Extraction — independently verified

`cpp/{l3,ddc,ddl,dcg}.cpp` — one self-contained TU per scope, bodies **verbatim**, each preceded by
a banner with its entry number and original `file:line`, in dependency order, plus `prelude.inc`
(a name inventory of 599 external names — **out of scope, nothing in it is to be ported**).

`tools/verify_extract.py` re-derives every body's end with **its own character state machine** — it
does not import the extractor's scanner and never re-slices with the extractor's own
`(file, line, length)` — and reports **382/382 bodies byte-identical, 1,069,773 bytes compared**,
with **two negative controls (a dropped last line, a blanked inner brace) both DETECTED**.

This is the check bridge 2 did not have: its extract silently truncated 366 of 384 bodies because
its check compared what was copied against what was copied.

---

## Running it

```bash
crustify/campaigns/ddc/driver.sh     # holds a pid lock; do not start a second
```

Gate, once per **batch**: `cargo check -p deeptools` and `cargo test -p deeptools`.
⛔ Never the workspace or acceptance build in an agent worktree (~6 GB of `target/` each).

Outstanding work is judged from **`UNITS.tsv`**, never from anchors in the tree.
Progress is counted only under `crates/compiler/deeptools/src/schedule/`.

## Regenerating

```bash
export DDC_CAMP=<worktree>/crustify-ddc DDC_TREE=<worktree>
python3 tools/enum_units.py && python3 tools/order.py && python3 tools/extract.py
python3 tools/verify_extract.py          # MUST pass, including both negative controls
python3 tools/gen_configs.py && python3 tools/gen_homes.py
python3 tools/validate_scaffold.py && python3 tools/gen_campaign.py
bash tools/dryrun_all.sh <worktree>
```

`gen_homes.py` **refuses to overwrite ported work**: every file it writes is recorded in
`scaffold-manifest.json` with its sha256, and a file is rewritten only when its current sha256 still
equals the recorded one. There is no `--force`. (Re-running bridge 4's generator destroyed its
finished 2,299-line port.)
