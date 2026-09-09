# What of the deeptools C++ compiler is ported, and what is not

Measured 2026-09-09 against `/Users/nickm/git/deeptools-src` @ `a0d29abbed`. Line counts exclude
`test/`, `tests/`, `unit_tests/` and vendored externals.

The authority for "what the compiler does" is not a document — it is the pass pipeline `dbo-opt`
builds, since `dbo-opt` is the binary scratchy shells out to. Three files define it:

- `dbo/src/Pipeline/Pipeline.cpp` — the run-level pipeline
- `dbo/src/Transforms/sdsc_bundle/RunSchedulerOnSdsc.cpp:144-170` — the per-program scheduling stages
- `dbo/src/Pipeline/RunProgramPipelines.cpp:154-175` — the per-program codegen stages

## The per-program path, stage by stage

| # | stage (`dbo-opt` pass) | C++ authority | lines | status |
|---|---|---|---|---|
| A | `sbf-dsm-cl-split` corelet split | `dsm` (entry `runDsmClSplit`) | — | ⛔ not ported |
| B | `sbf-ddc` L3 schedule + data-connect | `ddc` (`ddl` 4,956 · `transformations` 1,795 · `ddcv1.cpp`) | 20,869 | ⛔ 2 units only (bridge 1) |
| C | `sbf-dcg` L3 program generation → PCFG | `dcg` (`dcg_fe` 41,921 · `dcg_be` 11,481 · `dcg_manager` 1,241) | 56,893 | ✂️ **off our path** — see below |
| D | `sbf-perf-ideal-cycles` | `perfdsc` | 13,445 | performance estimate, not correctness |
| E | `dcc-prep-fill` / `dip-prep-fill` | `dsc-based-utils/dcc_prep` 447 · `dip_prep` 369 | 816 | ⛔ not ported |
| F | `sbf-sdsc-to-dataflow-ir` — three parts, see below | | 15,024 | ✅ the one part that applies |
| G | `dcc` D1–D28 → SentientIR | `Conversion/AgenToSentient` · `DataflowToSentient` · `VectorChainLowering` · `Transform/Dataflow` · `StandardToSentient` | ~16,000 | ✅ bridge 2, 384 units |
| H | `dcc` D29–D75 in place | `Transform/Sentient` | 34,438 | 🔄 campaign 5 running |
| I | `dcc` D76 → ProgIR | `Conversion/SentientToProgIR` | 8,380 | ✅ bridge 3, 130 units |
| J | `dcc-progir-opt` | `dsc-based-utils/dcc_prep/ProgirOpt.cpp` | 90 | ⛔ not ported |
| K | ProgIR → SenProg | `sys-arch-spec/progir` · `isa` · `dpc` | ~3,000 | ✅ bridge 4, 33 units |
| L | `dip-generate-init-packet` → `dbo-reduce-to-init-bin` | `dip` · `dbo/src/InitBin` | — | on `deeptools-islands`, not imported |

## Stage F is three parts and we ported one

`dsc-based-utils/SdscToDataflowIR/SdscToDataflowIR.cpp` is 48 lines and does exactly this:

1. `DSC2ToDataflowIR` over `sdsc->dscs_` — the **DL (compute) DSCs**. 7,156 lines. ✅ **bridge 1, 110 units.**
2. `PCFGToDFManager` over the **data DSCs** — 6,991 lines. ✂️ off our path.
3. `ModuleStitcher` — stitches DL + data modules into one. 877 lines. ✂️ off our path: a cited
   pass-through at `ModuleStitcher.cpp:54` when there are no data modules.

Part 2 consumes the PCFG that stage C (`dcg`, 56,893 lines) produces from `SuperDsc::dataOpdscs_`.

## C and F.2 are OFF OUR PATH — decided, and here is the evidence

**Decision (user, 2026-09-09): we do not need PCFG, because our input is a simple SuperDSC.**
What the reference itself says, so the decision is checkable rather than taken on faith:

`SuperDsc` (`dsc/superdsc.h:67,75,97,98`) carries `dscs_` (DL DSCs) and `dataOpdscs_` (data ops),
with `pcfg_`/`pcfgPool_` for what DCG generates. Our emission writes `dscs_`, `numCoresUsed_` and
`coreIdToDscSchedule` and nothing else: `subtile/src/superdsc_opspec.rs`'s `OpSpec` is a compute op,
and `grep -rio "datadsc|data_dsc|pcfg|l3_prog"` over `crates/compiler` returns no relevant hit.

1. **`PCFGToDFManager::run` converts `sdsc_.pcfg_` and `sdsc_.dataOpdscs_`** (`PCFGToDFManager.cpp:34,43`).
   Both are empty for us, so it emits no program-bearing module.
2. **`ModuleStitcher::stitch` returns the DL module unchanged** when there are no data modules —
   `if (dsc_module && data_modules.empty()) return dsc_module;` (`ModuleStitcher.cpp:54`). The
   reference's own answer for the DL-only case is "the DL module *is* the DataflowIR".
3. **The L3 transfers come out of the DL path, which we ported.** `runDcgForDlOpsStandalone`
   (`dcg_manager.cpp:449-513`) exists to fill `pcfg_[core][L3LU]`/`[L3SU]` — the L3 load- and
   store-unit programs — and its ACT3 codegen is skipped outright because `SchedulerStages.cpp:47`
   sets `createSenProg = false`. Those same programs are what `SNTransferLowering.cpp` emits on the
   DL path: bridge 1's port of it, `bridges/superdsc_to_dataflow_ir/transfer.rs`, carries 61 L3LU/L3SU
   references, with 26 more in `dsc_lowering.rs` and 20 in `utils.rs`. **DCG's L3 PCFG generation is
   the other pathway to the same programs, for data-op-only SuperDSCs — not an additional stage our
   input has to pass through.**

So stage C (56,893), F.2 (6,991) and F.3 (877) — **64,761 lines — are off our path**, and stage F is
bridge 1 alone.

## `dcc` directories no campaign covers

| lines | directory | what it is |
|---|---|---|
| 3,184 | `Dialect/Sentient` | op definitions — our `islands/sentient` |
| 2,764 | `Dialect/Uniform` | op definitions — uniformized programs |
| 754 | `Dialect/Agen` | op definitions |
| 1,991 | `Analysis` | ⛔ support code the passes draw on |
| 1,618 | `Utils` | ⛔ support code |
| 2,031 | `Conversion/SentientToTrace` | senulator input — not on the init_binary path |
| 1,978 | `tools/dcc-standalone` | the CLI — replaced by our pipeline |
| 1,822 | `tools/LitAutoTestGen` | test generation — not needed |
| 507 | `Driver` | the CLI driver |

## Totals

- **Ported: 657 units** over roughly **35,000 lines** of C++ (bridges 1–4).
- **Running: 34,438** (`Transform/Sentient`, campaign 5).
- **Off our path, decided: 64,761** — DCG + PCFGToDataflowIR + ModuleStitcher, per the section above.
- **Still unported and still needed: ~25,000** — `Ddc` 20,869 (bridge 1 took 2 units of `ddcv1.cpp`),
  `DsmClSplit`, the fill passes 816, `ProgirOpt` 90, and `dcc`'s `Analysis` 1,991 + `Utils` 1,618.
- `dcc` alone is 102,074 lines; the whole `deeptools-src` tree is far larger and most of it
  (`dsm` 236,994 · `dr5` 68,081 · `senulator` 52,031 · `dvs` 43,663) is off this path entirely.
