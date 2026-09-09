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
| C | **`sbf-dcg` L3 program generation → PCFG** | `dcg` (`dcg_fe` 41,921 · `dcg_be` 11,481 · `dcg_manager` 1,241) | **56,893** | ⛔ **never scoped** |
| D | `sbf-perf-ideal-cycles` | `perfdsc` | 13,445 | performance estimate, not correctness |
| E | `dcc-prep-fill` / `dip-prep-fill` | `dsc-based-utils/dcc_prep` 447 · `dip_prep` 369 | 816 | ⛔ not ported |
| F | `sbf-sdsc-to-dataflow-ir` — **three parts**, see below | | 15,024 | ◐ one of three |
| G | `dcc` D1–D28 → SentientIR | `Conversion/AgenToSentient` · `DataflowToSentient` · `VectorChainLowering` · `Transform/Dataflow` · `StandardToSentient` | ~16,000 | ✅ bridge 2, 384 units |
| H | `dcc` D29–D75 in place | `Transform/Sentient` | 34,438 | 🔄 campaign 5 running |
| I | `dcc` D76 → ProgIR | `Conversion/SentientToProgIR` | 8,380 | ✅ bridge 3, 130 units |
| J | `dcc-progir-opt` | `dsc-based-utils/dcc_prep/ProgirOpt.cpp` | 90 | ⛔ not ported |
| K | ProgIR → SenProg | `sys-arch-spec/progir` · `isa` · `dpc` | ~3,000 | ✅ bridge 4, 33 units |
| L | `dip-generate-init-packet` → `dbo-reduce-to-init-bin` | `dip` · `dbo/src/InitBin` | — | on `deeptools-islands`, not imported |

## Stage F is three parts and we ported one

`dsc-based-utils/SdscToDataflowIR/SdscToDataflowIR.cpp` is 48 lines and does exactly this:

1. `DSC2ToDataflowIR` over `sdsc->dscs_` — the **DL (compute) DSCs**. 7,156 lines. ✅ **bridge 1, 110 units.**
2. `PCFGToDFManager` over the **data DSCs** — 6,991 lines. ⛔ **not ported.**
3. `ModuleStitcher` — stitches DL + data modules into one. 877 lines. ⛔ **not ported.**

Part 2 consumes the PCFG that stage C (`dcg`, 56,893 lines) produces from `SuperDsc::dataOpdscs_`.

## Why C and F.2 may be structurally inapplicable — A CHECK, NOT A CONCLUSION

`SuperDsc` (`dsc/superdsc.h:67,75,97,98`) carries `dscs_` (DL DSCs) and `dataOpdscs_` (data ops),
with `pcfgMap_`/`pcfgPool_` for what DCG generates from the latter.

**Our emission produces `dscs_` only.** `crates/compiler/subtile/src/superdsc_opspec.rs`'s `OpSpec`
is a compute op (`op`, `iter`, `args`, epilogue); nothing in `crates/compiler` mentions a data op,
a PCFG, or an L3 program (`grep -rio "datadsc|data_dsc|pcfg|l3_prog"` → 0 relevant hits).
Transfers reach DataflowIR through the DL DSC's own lowering — `SNTransferLowering.cpp`, which is
29 of bridge 1's 110 units.

⛔ So the claim "DCG is not needed" rests on our SuperDSC never carrying data ops. That is true of
what we emit today; it is NOT established that the DL path expresses every transfer the reference
routes through DCG. **Do not treat C and F.2 as winnowed until an end-to-end run proves the DL path
alone reaches a correct `init_binary`.** The prior hand-written L3 work on `deeptools-islands`
(`bridges/1.rs` — burst derivation, L3 loop levels, coalescing) is evidence the L3 side *was* needed
there; that tree emitted DataflowIR directly and never had a SuperDSC to carry data ops.

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
- **Unscoped on the per-program path: ~65,000**, of which **56,893 is DCG** — see the caveat above.
- `dcc` alone is 102,074 lines; the whole `deeptools-src` tree is far larger and most of it
  (`dsm` 236,994 · `dr5` 68,081 · `senulator` 52,031 · `dvs` 43,663) is off this path entirely.
