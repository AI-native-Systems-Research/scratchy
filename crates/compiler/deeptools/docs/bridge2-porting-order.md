# Bridge 2 — the porting order, definition by definition

`DataflowIR -> SentientIR`, the D1-D28 span. **490 definitions, 11 levels**, in dependency order.

## ⛔ THIS SUPERSEDES AN EARLIER VERSION THAT WAS WRONG, AND THE ERROR IS WORTH KNOWING

The first graph keyed nodes by **function NAME**, which collapsed every same-named definition into
one node. `runOnOperation` is defined in **21 files** — every pass has one — so all 21 pass drivers
became a single hub, fusing unrelated passes into each other's dependency chains. `dccExtContext`
(9 files), `dcc_ext_ctx_` (8), `opts_` (6) and `matchAndRewrite` (4) did the same. 31 of 418 names
were multiply defined.

It also cited files by **basename**, which is ambiguous: there are two `Helper.cpp`
(`AgenToSentient/` and `VectorChainToSentientPT/`) and `VectorChainHelper.cpp` actually lives under
`CommonHelpers/`. A citation that cannot be resolved is not a citation.

Corrected: nodes are keyed `(file, name)`, calls resolve to a same-file definition first and to a
unique cross-file definition otherwise, and 38 genuinely ambiguous cross-file calls are dropped
rather than guessed. **490 definitions, 929 edges, 11 levels** — against 418/949/9 before.

## How it is derived, and how it still lies

A textual call graph over `dcc/src/{Transform/Dataflow,Conversion/*}` in the pod extract at
`a0d29abbed`. Level 0 calls nothing else in the span; level N calls only levels below it.

⭐ **DEFINITIONS AT THE SAME LEVEL NEVER CALL EACH OTHER**, which is what makes a level a safe batch.

⛔ It cannot see through virtual dispatch (an edge to the base, never the override), templates,
function pointers or `std::function`; the 38 dropped ambiguous calls are real edges it declines to
guess at; and cycles are broken at the first back edge. A level is a **starting order, not a proof**:
if a port needs something from a higher level, the graph missed an edge.

⛔ **A LINE COUNT IS NOT A WORK ESTIMATE.** Between 26% and 83% of these lines are MLIR *mechanics* —
`OpBuilder` calls, SSA iterators, memoisers, the nested `(core, corelet, component)` lookup maps —
which exist because the C++ builds an IR at run time while we emit from a resolved tape. So a
196-line function may be 40 lines of Rust.

⛔⛔ **THIS IS NOT A LICENCE TO SKIP THE EMISSION, AND IT WAS READ AS ONE.** An earlier draft of this
paragraph ended *"port the rule, never the plumbing"*, and that sentence is exactly how levels 0-2
came to be 490 predicates with no output: `setldtype`'s "rule" became a `LoadType` enum and the
`sentient.load_and_send` it exists to attribute was never built. **The op the function emits IS the
function.** What may be dropped is the mechanism for *getting at* the operands — walking uses,
memoising by component, positioning a builder — not what gets emitted or with which attributes.

## ⛔ THE RULES OF EXECUTION — the plan was approved, the EXECUTION was reward-hacked

The plan below is unchanged and was approved. What failed was implementing it: levels 0, 1 and 2 were
reported complete — "47 definitions, 1,791 lines", "48 tests green, clippy clean" — while what had
been written was one documented PREDICATE per function (`is_l3()`, `BurstSetting::of()`,
`Granularity::checked()`) with every function's EMISSION left out and nothing calling any of it. The
differential test that would have graded it stayed `#[ignore]`d for all three levels. 1,845 lines the
oracle could not see.

So each entry now has two boxes, and they tick separately:

- ⛔ **PORT means the WHOLE function, including its emission.** A predicate is not a port. If our IR
  cannot express the function's input, ADD THE OP TO THE ISLAND — do not decide the function is
  unnecessary. That judgement is not this task's to make.
- ⛔ **AUDIT means line by line against the C++**: every branch, every constant, every attribute name
  and value, every default, every early return, and the order of emission. Report divergences,
  including deliberate ones with their reason.
- ⛔ **UNIT TESTS COME WITH THE PORT.** Where the vendor has a case, port THEIRS: `dcc/test/` holds 825
  `.mlir` tests and 668 carry `CHECK-SENT-IR` expectations. Their input is text and this crate has no
  parser, so the typed input is built in the test and the EXPECTATION comes from their `CHECK` lines.
- ⛔ **`cargo build` IS STILL THE ACCEPTANCE GATE**, because a green unit test is not coverage of the
  bridge: `cargo build -Fsuperdsc,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel` going
  through, read from `-vv`. E2E arrives when bridge 1 lands.
- ⛔ **NO GROUPING.** Grouping into levels is what made "level 2 complete" reportable. A level is a
  starting ORDER; the unit of work is one entry.

Where the code goes: `crates/compiler/deeptools/src/bridges/dataflow_ir_to_sentient/`, called from
`emit_run` in `crates/targets/spyre/src/lower_subtile_tape_to_dataflow_ir.rs` — the only place the
typed `dataflow_ir::Run` exists. There is no MLIR text on the input side.

## Progress

`2/490 ported; 2/490 audited` — keep in step with the task list's counter.

⭐ Entry 233 is `AffineYieldOpLowering::matchAndRewrite`, ported as `lower_affine_yield`: the island
gained `affine::Op::Yield`, `scf::Op::Yield` and `scf::Op::Parallel` so the function HAD an input, the
operand list carries through unchanged, and the pattern declines under a parallel parent because the
parallel lowering owns that terminator. Four unit tests. ⛔ IT WAS FIRST "PORTED" BY ARGUING OUR IR
COULD NOT EXPRESS AN `affine.yield` AND SKIPPING IT — which is the same failure as the levels, one
function wide.

## Level 0 — 239 definitions, 1614 body lines

### Conversion/VectorChainLowering — 54 defs, 654 lines

- [ ] **PORT 001** `getMaskValueForPT` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:17`, 196 lines
- [ ] **AUDIT 001** `getMaskValueForPT` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:17`, line by line against the C++
- [ ] **PORT 002** `getOperandFromReceiveOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:34`, 54 lines
- [ ] **AUDIT 002** `getOperandFromReceiveOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:34`, line by line against the C++
- [ ] **PORT 003** `getOperandFromSendOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:95`, 50 lines
- [ ] **AUDIT 003** `getOperandFromSendOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:95`, line by line against the C++
- [ ] **PORT 004** `fuseCompareAndSelectIntoMinOrMax` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:415`, 49 lines
- [ ] **AUDIT 004** `fuseCompareAndSelectIntoMinOrMax` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:415`, line by line against the C++
- [ ] **PORT 005** `redefineConstantVectors` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:571`, 37 lines
- [ ] **AUDIT 005** `redefineConstantVectors` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:571`, line by line against the C++
- [ ] **PORT 006** `getForOpBound` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:262`, 31 lines
- [ ] **AUDIT 006** `getForOpBound` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:262`, line by line against the C++
- [ ] **PORT 007** `fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1243`, 26 lines
- [ ] **AUDIT 007** `fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1243`, line by line against the C++
- [ ] **PORT 008** `getVectorBinaryToSentientBinary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:32`, 17 lines
- [ ] **AUDIT 008** `getVectorBinaryToSentientBinary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:32`, line by line against the C++
- [ ] **PORT 009** `eraseOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:806`, 16 lines
- [ ] **AUDIT 009** `eraseOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:806`, line by line against the C++
- [ ] **PORT 010** `getComputePrecisionOfOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:65`, 13 lines
- [ ] **AUDIT 010** `getComputePrecisionOfOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:65`, line by line against the C++
- [ ] **PORT 011** `resetSentientFMAsIfExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:468`, 13 lines
- [ ] **AUDIT 011** `resetSentientFMAsIfExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:468`, line by line against the C++
- [ ] **PORT 012** `constValToField` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:250`, 12 lines
- [ ] **AUDIT 012** `constValToField` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:250`, line by line against the C++
- [ ] **PORT 013** `sameBlock` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:652`, 11 lines
- [ ] **AUDIT 013** `sameBlock` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:652`, line by line against the C++
- [ ] **PORT 014** `getXrfValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:248`, 11 lines
- [ ] **AUDIT 014** `getXrfValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:248`, line by line against the C++
- [ ] **PORT 015** `setSentientMacXrfRegIncrAttr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:665`, 11 lines
- [ ] **AUDIT 015** `setSentientMacXrfRegIncrAttr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:665`, line by line against the C++
- [ ] **PORT 016** `hasConstantBounds` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:80`, 10 lines
- [ ] **AUDIT 016** `hasConstantBounds` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:80`, line by line against the C++
- [ ] **PORT 017** `getResultPrecisionFromOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:52`, 10 lines
- [ ] **AUDIT 017** `getResultPrecisionFromOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:52`, line by line against the C++
- [ ] **PORT 018** `getVectorElementWiseCompareOperatorToSentientBinaryOperator` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:55`, 9 lines
- [ ] **AUDIT 018** `getVectorElementWiseCompareOperatorToSentientBinaryOperator` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:55`, line by line against the C++
- [ ] **PORT 019** `computeUnitPrecision` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:31`, 9 lines
- [ ] **AUDIT 019** `computeUnitPrecision` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:31`, line by line against the C++
- [ ] **PORT 020** `getVectorTernaryToSentientTernary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:247`, 7 lines
- [ ] **AUDIT 020** `getVectorTernaryToSentientTernary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:247`, line by line against the C++
- [ ] **PORT 021** `replaceAndEraseDummyMacOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:681`, 7 lines
- [ ] **AUDIT 021** `replaceAndEraseDummyMacOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:681`, line by line against the C++
- [ ] **PORT 022** `size` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:319`, 6 lines
- [ ] **AUDIT 022** `size` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:319`, line by line against the C++
- [ ] **PORT 023** `getAbsorbtionFlag` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:73`, 6 lines
- [ ] **AUDIT 023** `getAbsorbtionFlag` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:73`, line by line against the C++
- [ ] **PORT 024** `getId` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:65`, 6 lines
- [ ] **AUDIT 024** `getId` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:65`, line by line against the C++
- [ ] **PORT 025** `getInputPrecisionFromOperand` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:36`, 6 lines
- [ ] **AUDIT 025** `getInputPrecisionFromOperand` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:36`, line by line against the C++
- [ ] **PORT 026** `reuse_info_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:82`, 5 lines
- [ ] **AUDIT 026** `reuse_info_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:82`, line by line against the C++
- [ ] **PORT 027** `isSentientBinaryLogicalOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:30`, 4 lines
- [ ] **AUDIT 027** `isSentientBinaryLogicalOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:30`, line by line against the C++
- [ ] **PORT 028** `findNodeFromOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:167`, 4 lines
- [ ] **AUDIT 028** `findNodeFromOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:167`, line by line against the C++
- [ ] **PORT 029** `walk` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:132`, 2 lines
- [ ] **AUDIT 029** `walk` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:132`, line by line against the C++
- [ ] **PORT 030** `dominates` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:38`, 2 lines
- [ ] **AUDIT 030** `dominates` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:38`, line by line against the C++
- [ ] **PORT 031** `setReuseFlag` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:91`, 2 lines
- [ ] **AUDIT 031** `setReuseFlag` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:91`, line by line against the C++
- [ ] **PORT 032** `getParentNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:36`, 2 lines
- [ ] **AUDIT 032** `getParentNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:36`, line by line against the C++
- [ ] **PORT 033** `getFirstChild` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:39`, 2 lines
- [ ] **AUDIT 033** `getFirstChild` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:39`, line by line against the C++
- [ ] **PORT 034** `getNextSibling` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:42`, 2 lines
- [ ] **AUDIT 034** `getNextSibling` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:42`, line by line against the C++
- [ ] **PORT 035** `getPrevSibling` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:45`, 2 lines
- [ ] **AUDIT 035** `getPrevSibling` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:45`, line by line against the C++
- [ ] **PORT 036** `getLastChild` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:48`, 2 lines
- [ ] **AUDIT 036** `getLastChild` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:48`, line by line against the C++
- [ ] **PORT 037** `getRoot` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:109`, 2 lines
- [ ] **AUDIT 037** `getRoot` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:109`, line by line against the C++
- [ ] **PORT 038** `getFirstValue` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:76`, 0 lines
- [ ] **AUDIT 038** `getFirstValue` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:76`, line by line against the C++
- [ ] **PORT 039** `getIncrement` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:77`, 0 lines
- [ ] **AUDIT 039** `getIncrement` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:77`, line by line against the C++
- [ ] **PORT 040** `isMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:55`, 0 lines
- [ ] **AUDIT 040** `isMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:55`, line by line against the C++
- [ ] **PORT 041** `getStartVal` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:76`, 0 lines
- [ ] **AUDIT 041** `getStartVal` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:76`, line by line against the C++
- [ ] **PORT 042** `isLoopNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:54`, 0 lines
- [ ] **AUDIT 042** `isLoopNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:54`, line by line against the C++
- [ ] **PORT 043** `size_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:208`, 0 lines
- [ ] **AUDIT 043** `size_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:208`, line by line against the C++
- [ ] **PORT 044** `dcc_ext_ctx_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:45`, 0 lines
- [ ] **AUDIT 044** `dcc_ext_ctx_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:45`, line by line against the C++
- [ ] **PORT 045** `dccExtContext` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:51`, 0 lines
- [ ] **AUDIT 045** `dccExtContext` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:51`, line by line against the C++
- [ ] **PORT 046** `is_visited_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:108`, 0 lines
- [ ] **AUDIT 046** `is_visited_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:108`, line by line against the C++
- [ ] **PORT 047** `MaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:74`, 0 lines
- [ ] **AUDIT 047** `MaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:74`, line by line against the C++
- [ ] **PORT 048** `setOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:52`, 0 lines
- [ ] **AUDIT 048** `setOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:52`, line by line against the C++
- [ ] **PORT 049** `LMTLoopNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:94`, 0 lines
- [ ] **AUDIT 049** `LMTLoopNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:94`, line by line against the C++
- [ ] **PORT 050** `LoopMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:34`, 0 lines
- [ ] **AUDIT 050** `LoopMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:34`, line by line against the C++
- [ ] **PORT 051** `OperationNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:32`, 0 lines
- [ ] **AUDIT 051** `OperationNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:32`, line by line against the C++
- [ ] **PORT 052** `increment_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:72`, 0 lines
- [ ] **AUDIT 052** `increment_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:72`, line by line against the C++
- [ ] **PORT 053** `dcc_ext_ctx_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.hpp:53`, 0 lines
- [ ] **AUDIT 053** `dcc_ext_ctx_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.hpp:53`, line by line against the C++
- [ ] **PORT 054** `dccExtContext` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.hpp:59`, 0 lines
- [ ] **AUDIT 054** `dccExtContext` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.hpp:59`, line by line against the C++

### Transform/Dataflow — 75 defs, 434 lines

- [ ] **PORT 055** `run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.cpp:21`, 48 lines
- [ ] **AUDIT 055** `run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.cpp:21`, line by line against the C++
- [ ] **PORT 056** `transformSCFToAffineLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:102`, 44 lines
- [ ] **AUDIT 056** `transformSCFToAffineLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:102`, line by line against the C++
- [ ] **PORT 057** `getDataflowForLoopInfoIfIV` — `dcc/src/Transform/Dataflow/Utils.cpp:99`, 31 lines
- [ ] **AUDIT 057** `getDataflowForLoopInfoIfIV` — `dcc/src/Transform/Dataflow/Utils.cpp:99`, line by line against the C++
- [ ] **PORT 058** `isEligibleForSplitting` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:832`, 20 lines
- [ ] **AUDIT 058** `isEligibleForSplitting` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:832`, line by line against the C++
- [ ] **PORT 059** `removeCoresCoreletsFoldsFromProgramUnit` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:240`, 20 lines
- [ ] **AUDIT 059** `removeCoresCoreletsFoldsFromProgramUnit` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:240`, line by line against the C++
- [ ] **PORT 060** `partitionUnits` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:168`, 17 lines
- [ ] **AUDIT 060** `partitionUnits` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:168`, line by line against the C++
- [ ] **PORT 061** `getConstantTripCount` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:167`, 14 lines
- [ ] **AUDIT 061** `getConstantTripCount` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:167`, line by line against the C++
- [ ] **PORT 062** `createInequalityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:263`, 14 lines
- [ ] **AUDIT 062** `createInequalityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:263`, line by line against the C++
- [ ] **PORT 063** `createDummyYieldInElseReg` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:383`, 13 lines
- [ ] **AUDIT 063** `createDummyYieldInElseReg` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:383`, line by line against the C++
- [ ] **PORT 064** `createNewMemViewWithMod` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:966`, 12 lines
- [ ] **AUDIT 064** `createNewMemViewWithMod` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:966`, line by line against the C++
- [ ] **PORT 065** `getLhsRhsOfEQPredicate` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:519`, 11 lines
- [ ] **AUDIT 065** `getLhsRhsOfEQPredicate` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:519`, line by line against the C++
- [ ] **PORT 066** `isDataTransfer` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:314`, 11 lines
- [ ] **AUDIT 066** `isDataTransfer` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:314`, line by line against the C++
- [ ] **PORT 067** `printUnitToOpsMap` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:189`, 10 lines
- [ ] **AUDIT 067** `printUnitToOpsMap` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:189`, line by line against the C++
- [ ] **PORT 068** `calculateStartElementsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:532`, 10 lines
- [ ] **AUDIT 068** `calculateStartElementsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:532`, line by line against the C++
- [ ] **PORT 069** `createNonPagedMemView` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:545`, 10 lines
- [ ] **AUDIT 069** `createNonPagedMemView` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:545`, line by line against the C++
- [ ] **PORT 070** `identifyTimeDimForExplicitLoops` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:963`, 10 lines
- [ ] **AUDIT 070** `identifyTimeDimForExplicitLoops` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:963`, line by line against the C++
- [ ] **PORT 071** `createNewMemOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:684`, 9 lines
- [ ] **AUDIT 071** `createNewMemOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:684`, line by line against the C++
- [ ] **PORT 072** `cloneMemViewIfNonPaged` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:633`, 9 lines
- [ ] **AUDIT 072** `cloneMemViewIfNonPaged` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:633`, line by line against the C++
- [ ] **PORT 073** `printEquivalenceClasses` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:202`, 8 lines
- [ ] **AUDIT 073** `printEquivalenceClasses` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:202`, line by line against the C++
- [ ] **PORT 074** `getMaxMutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:673`, 8 lines
- [ ] **AUDIT 074** `getMaxMutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:673`, line by line against the C++
- [ ] **PORT 075** `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:683`, 8 lines
- [ ] **AUDIT 075** `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:683`, line by line against the C++
- [ ] **PORT 076** `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:355`, 8 lines
- [ ] **AUDIT 076** `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:355`, line by line against the C++
- [ ] **PORT 077** `performFullUnroll` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:141`, 7 lines
- [ ] **AUDIT 077** `performFullUnroll` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:141`, line by line against the C++
- [ ] **PORT 078** `removeValuesFromIndices` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:36`, 7 lines
- [ ] **AUDIT 078** `removeValuesFromIndices` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:36`, line by line against the C++
- [ ] **PORT 079** `replaceDimsInMapWithSyms` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:47`, 7 lines
- [ ] **AUDIT 079** `replaceDimsInMapWithSyms` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:47`, line by line against the C++
- [ ] **PORT 080** `createEqualityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:253`, 7 lines
- [ ] **AUDIT 080** `createEqualityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:253`, line by line against the C++
- [ ] **PORT 081** `setBuilderToInsertRef` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:280`, 6 lines
- [ ] **AUDIT 081** `setBuilderToInsertRef` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:280`, line by line against the C++
- [ ] **PORT 082** `getStoreOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:843`, 6 lines
- [ ] **AUDIT 082** `getStoreOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:843`, line by line against the C++
- [ ] **PORT 083** `TransformationConditionalTree` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:111`, 5 lines
- [ ] **AUDIT 083** `TransformationConditionalTree` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:111`, line by line against the C++
- [ ] **PORT 084** `addTimeDimIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:876`, 5 lines
- [ ] **AUDIT 084** `addTimeDimIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:876`, line by line against the C++
- [ ] **PORT 085** `sortDataBasedOnWeight` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:855`, 4 lines
- [ ] **AUDIT 085** `sortDataBasedOnWeight` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:855`, line by line against the C++
- [ ] **PORT 086** `base_unit_program_` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:43`, 4 lines
- [ ] **AUDIT 086** `base_unit_program_` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:43`, line by line against the C++
- [ ] **PORT 087** `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:698`, 3 lines
- [ ] **AUDIT 087** `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:698`, line by line against the C++
- [ ] **PORT 088** `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:673`, 3 lines
- [ ] **AUDIT 088** `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:673`, line by line against the C++
- [ ] **PORT 089** `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:678`, 3 lines
- [ ] **AUDIT 089** `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:678`, line by line against the C++
- [ ] **PORT 090** `isOperationSelected` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:34`, 2 lines
- [ ] **AUDIT 090** `isOperationSelected` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:34`, line by line against the C++
- [ ] **PORT 091** `getRoot` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:81`, 2 lines
- [ ] **AUDIT 091** `getRoot` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:81`, line by line against the C++
- [ ] **PORT 092** `getNewDbgNameFromList` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:456`, 2 lines
- [ ] **AUDIT 092** `getNewDbgNameFromList` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:456`, line by line against the C++
- [ ] **PORT 093** `ConditionalSimplificationManager` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:78`, 2 lines
- [ ] **AUDIT 093** `ConditionalSimplificationManager` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:78`, line by line against the C++
- [ ] **PORT 094** `populateDuplicateReusedTogglePatterns` — `dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:207`, 2 lines
- [ ] **AUDIT 094** `populateDuplicateReusedTogglePatterns` — `dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:207`, line by line against the C++
- [ ] **PORT 095** `getParentNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:53`, 2 lines
- [ ] **AUDIT 095** `getParentNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:53`, line by line against the C++
- [ ] **PORT 096** `getFirstChild` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:56`, 2 lines
- [ ] **AUDIT 096** `getFirstChild` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:56`, line by line against the C++
- [ ] **PORT 097** `getNextSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:59`, 2 lines
- [ ] **AUDIT 097** `getNextSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:59`, line by line against the C++
- [ ] **PORT 098** `getPrevSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:62`, 2 lines
- [ ] **AUDIT 098** `getPrevSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:62`, line by line against the C++
- [ ] **PORT 099** `comp_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:30`, 2 lines
- [ ] **AUDIT 099** `comp_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:30`, line by line against the C++
- [ ] **PORT 100** `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:328`, 2 lines
- [ ] **AUDIT 100** `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:328`, line by line against the C++
- [ ] **PORT 101** `opts_` — `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:69`, 0 lines
- [ ] **AUDIT 101** `opts_` — `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:69`, line by line against the C++
- [ ] **PORT 102** `dcc_ext_ctx_` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:47`, 0 lines
- [ ] **AUDIT 102** `dcc_ext_ctx_` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:47`, line by line against the C++
- [ ] **PORT 103** `dccExtContext` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:94`, 0 lines
- [ ] **AUDIT 103** `dccExtContext` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:94`, line by line against the C++
- [ ] **PORT 104** `opts_` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:39`, 0 lines
- [ ] **AUDIT 104** `opts_` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:39`, line by line against the C++
- [ ] **PORT 105** `OperationNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:51`, 0 lines
- [ ] **AUDIT 105** `OperationNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:51`, line by line against the C++
- [ ] **PORT 106** `OperationTreeBase` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:78`, 0 lines
- [ ] **AUDIT 106** `OperationTreeBase` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:78`, line by line against the C++
- [ ] **PORT 107** `dcc_ext_ctx_` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:54`, 0 lines
- [ ] **AUDIT 107** `dcc_ext_ctx_` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:54`, line by line against the C++
- [ ] **PORT 108** `dccExtContext` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:66`, 0 lines
- [ ] **AUDIT 108** `dccExtContext` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:66`, line by line against the C++
- [ ] **PORT 109** `opts_` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:84`, 0 lines
- [ ] **AUDIT 109** `opts_` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:84`, line by line against the C++
- [ ] **PORT 110** `mem_index_` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:94`, 0 lines
- [ ] **AUDIT 110** `mem_index_` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:94`, line by line against the C++
- [ ] **PORT 111** `weight_` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:111`, 0 lines
- [ ] **AUDIT 111** `weight_` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:111`, line by line against the C++
- [ ] **PORT 112** `dccExtContext` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:219`, 0 lines
- [ ] **AUDIT 112** `dccExtContext` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:219`, line by line against the C++
- [ ] **PORT 113** `opts_` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:61`, 0 lines
- [ ] **AUDIT 113** `opts_` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:61`, line by line against the C++
- [ ] **PORT 114** `mem_index_` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:71`, 0 lines
- [ ] **AUDIT 114** `mem_index_` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:71`, line by line against the C++
- [ ] **PORT 115** `weight_` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:83`, 0 lines
- [ ] **AUDIT 115** `weight_` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:83`, line by line against the C++
- [ ] **PORT 116** `dccExtContext` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:125`, 0 lines
- [ ] **AUDIT 116** `dccExtContext` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:125`, line by line against the C++
- [ ] **PORT 117** `dcc_ext_ctx_` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:55`, 0 lines
- [ ] **AUDIT 117** `dcc_ext_ctx_` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:55`, line by line against the C++
- [ ] **PORT 118** `curr_unit_corelet_id_` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:91`, 0 lines
- [ ] **AUDIT 118** `curr_unit_corelet_id_` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:91`, line by line against the C++
- [ ] **PORT 119** `opts_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:34`, 0 lines
- [ ] **AUDIT 119** `opts_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:34`, line by line against the C++
- [ ] **PORT 120** `subscripts_map_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:45`, 0 lines
- [ ] **AUDIT 120** `subscripts_map_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:45`, line by line against the C++
- [ ] **PORT 121** `mem_index_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:50`, 0 lines
- [ ] **AUDIT 121** `mem_index_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:50`, line by line against the C++
- [ ] **PORT 122** `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:341`, 0 lines
- [ ] **AUDIT 122** `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:341`, line by line against the C++
- [ ] **PORT 123** `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:364`, 0 lines
- [ ] **AUDIT 123** `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:364`, line by line against the C++
- [ ] **PORT 124** `TPMVBase` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:389`, 0 lines
- [ ] **AUDIT 124** `TPMVBase` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:389`, line by line against the C++
- [ ] **PORT 125** `TPMVVector` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:397`, 0 lines
- [ ] **AUDIT 125** `TPMVVector` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:397`, line by line against the C++
- [ ] **PORT 126** `access_details_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:458`, 0 lines
- [ ] **AUDIT 126** `access_details_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:458`, line by line against the C++
- [ ] **PORT 127** `TPMVComposite` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:519`, 0 lines
- [ ] **AUDIT 127** `TPMVComposite` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:519`, line by line against the C++
- [ ] **PORT 128** `comp_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.hpp:24`, 0 lines
- [ ] **AUDIT 128** `comp_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.hpp:24`, line by line against the C++
- [ ] **PORT 129** `opts_` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:43`, 0 lines
- [ ] **AUDIT 129** `opts_` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:43`, line by line against the C++

### Conversion/AgenToSentient — 82 defs, 319 lines

- [ ] **PORT 130** `setldtype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1647`, 60 lines
- [ ] **AUDIT 130** `setldtype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1647`, line by line against the C++
- [ ] **PORT 131** `generateSetSendDestinationStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2731`, 49 lines
- [ ] **AUDIT 131** `generateSetSendDestinationStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2731`, line by line against the C++
- [ ] **PORT 132** `checkIndirectMemViewForExtractOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:388`, 42 lines
- [ ] **AUDIT 132** `checkIndirectMemViewForExtractOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:388`, line by line against the C++
- [ ] **PORT 133** `getLoadConsumer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1242`, 36 lines
- [ ] **AUDIT 133** `getLoadConsumer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1242`, line by line against the C++
- [ ] **PORT 134** `findExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:514`, 20 lines
- [ ] **AUDIT 134** `findExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:514`, line by line against the C++
- [ ] **PORT 135** `insertCopyAndAddStmtsHelper` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:502`, 17 lines
- [ ] **AUDIT 135** `insertCopyAndAddStmtsHelper` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:502`, line by line against the C++
- [ ] **PORT 136** `findCandidateForLowering` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2884`, 12 lines
- [ ] **AUDIT 136** `findCandidateForLowering` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2884`, line by line against the C++
- [ ] **PORT 137** `addLoadChainToDeleteList` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2975`, 9 lines
- [ ] **AUDIT 137** `addLoadChainToDeleteList` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2975`, line by line against the C++
- [ ] **PORT 138** `getStoreOpFromLoadStorePattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2872`, 8 lines
- [ ] **AUDIT 138** `getStoreOpFromLoadStorePattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2872`, line by line against the C++
- [ ] **PORT 139** `getLoopNestLevel` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:43`, 6 lines
- [ ] **AUDIT 139** `getLoopNestLevel` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:43`, line by line against the C++
- [ ] **PORT 140** `setCoalescedBoundValues` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:672`, 5 lines
- [ ] **AUDIT 140** `setCoalescedBoundValues` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:672`, line by line against the C++
- [ ] **PORT 141** `constructLoadAndSendStmt` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:230`, 4 lines
- [ ] **AUDIT 141** `constructLoadAndSendStmt` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:230`, line by line against the C++
- [ ] **PORT 142** `constructReceiveAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:248`, 4 lines
- [ ] **AUDIT 142** `constructReceiveAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:248`, line by line against the C++
- [ ] **PORT 143** `createAgenToSentientPass` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:252`, 3 lines
- [ ] **AUDIT 143** `createAgenToSentientPass` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:252`, line by line against the C++
- [ ] **PORT 144** `has` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:397`, 2 lines
- [ ] **AUDIT 144** `has` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:397`, line by line against the C++
- [ ] **PORT 145** `setIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:77`, 2 lines
- [ ] **AUDIT 145** `setIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:77`, line by line against the C++
- [ ] **PORT 146** `setSubscriptsMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:228`, 2 lines
- [ ] **AUDIT 146** `setSubscriptsMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:228`, line by line against the C++
- [ ] **PORT 147** `setMemViewStartAddr` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:81`, 2 lines
- [ ] **AUDIT 147** `setMemViewStartAddr` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:81`, line by line against the C++
- [ ] **PORT 148** `setMemViewLayoutMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:99`, 2 lines
- [ ] **AUDIT 148** `setMemViewLayoutMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:99`, line by line against the C++
- [ ] **PORT 149** `setTransferOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:125`, 2 lines
- [ ] **AUDIT 149** `setTransferOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:125`, line by line against the C++
- [ ] **PORT 150** `setTransferSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:122`, 2 lines
- [ ] **AUDIT 150** `setTransferSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:122`, line by line against the C++
- [ ] **PORT 151** `setExpectedTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:110`, 2 lines
- [ ] **AUDIT 151** `setExpectedTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:110`, line by line against the C++
- [ ] **PORT 152** `setElementWidth` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:119`, 2 lines
- [ ] **AUDIT 152** `setElementWidth` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:119`, line by line against the C++
- [ ] **PORT 153** `setIndicesCoeffDict` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:231`, 2 lines
- [ ] **AUDIT 153** `setIndicesCoeffDict` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:231`, line by line against the C++
- [ ] **PORT 154** `setRotationPosition` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:107`, 2 lines
- [ ] **AUDIT 154** `setRotationPosition` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:107`, line by line against the C++
- [ ] **PORT 155** `setShuffleMode` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:104`, 2 lines
- [ ] **AUDIT 155** `setShuffleMode` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:104`, line by line against the C++
- [ ] **PORT 156** `setMemoryIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:93`, 2 lines
- [ ] **AUDIT 156** `setMemoryIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:93`, line by line against the C++
- [ ] **PORT 157** `setLayoutCoeffs` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:96`, 2 lines
- [ ] **AUDIT 157** `setLayoutCoeffs` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:96`, line by line against the C++
- [ ] **PORT 158** `setTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:116`, 2 lines
- [ ] **AUDIT 158** `setTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:116`, line by line against the C++
- [ ] **PORT 159** `setExtents` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:113`, 2 lines
- [ ] **AUDIT 159** `setExtents` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:113`, line by line against the C++
- [ ] **PORT 160** `setTimeOffsets` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:295`, 2 lines
- [ ] **AUDIT 160** `setTimeOffsets` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:295`, line by line against the C++
- [ ] **PORT 161** `setTimeBounds` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:292`, 2 lines
- [ ] **AUDIT 161** `setTimeBounds` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:292`, line by line against the C++
- [ ] **PORT 162** `setInterleaveGroupIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:299`, 2 lines
- [ ] **AUDIT 162** `setInterleaveGroupIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:299`, line by line against the C++
- [ ] **PORT 163** `setTimeAddrMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:286`, 2 lines
- [ ] **AUDIT 163** `setTimeAddrMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:286`, line by line against the C++
- [ ] **PORT 164** `setTimeSymbols` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:289`, 2 lines
- [ ] **AUDIT 164** `setTimeSymbols` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:289`, line by line against the C++
- [ ] **PORT 165** `setStrides` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:355`, 2 lines
- [ ] **AUDIT 165** `setStrides` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:355`, line by line against the C++
- [ ] **PORT 166** `getIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:70`, 0 lines
- [ ] **AUDIT 166** `getIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:70`, line by line against the C++
- [ ] **PORT 167** `getMemRef` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:59`, 0 lines
- [ ] **AUDIT 167** `getMemRef` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:59`, line by line against the C++
- [ ] **PORT 168** `getElementWidth` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:69`, 0 lines
- [ ] **AUDIT 168** `getElementWidth` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:69`, line by line against the C++
- [ ] **PORT 169** `getOp` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:57`, 0 lines
- [ ] **AUDIT 169** `getOp` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:57`, line by line against the C++
- [ ] **PORT 170** `getSubscriptsMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:221`, 0 lines
- [ ] **AUDIT 170** `getSubscriptsMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:221`, line by line against the C++
- [ ] **PORT 171** `setMemory` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:84`, 0 lines
- [ ] **AUDIT 171** `setMemory` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:84`, line by line against the C++
- [ ] **PORT 172** `setMemRef` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:80`, 0 lines
- [ ] **AUDIT 172** `setMemRef` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:80`, line by line against the C++
- [ ] **PORT 173** `getMemoryIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:72`, 0 lines
- [ ] **AUDIT 173** `getMemoryIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:72`, line by line against the C++
- [ ] **PORT 174** `getTransferOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:63`, 0 lines
- [ ] **AUDIT 174** `getTransferOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:63`, line by line against the C++
- [ ] **PORT 175** `getMemViewLayoutMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:61`, 0 lines
- [ ] **AUDIT 175** `getMemViewLayoutMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:61`, line by line against the C++
- [ ] **PORT 176** `getRotationPosition` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:67`, 0 lines
- [ ] **AUDIT 176** `getRotationPosition` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:67`, line by line against the C++
- [ ] **PORT 177** `getExtents` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:73`, 0 lines
- [ ] **AUDIT 177** `getExtents` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:73`, line by line against the C++
- [ ] **PORT 178** `setChunkStride` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:103`, 0 lines
- [ ] **AUDIT 178** `setChunkStride` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:103`, line by line against the C++
- [ ] **PORT 179** `setChunkSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:102`, 0 lines
- [ ] **AUDIT 179** `setChunkSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:102`, line by line against the C++
- [ ] **PORT 180** `getLayoutCoeffs` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:62`, 0 lines
- [ ] **AUDIT 180** `getLayoutCoeffs` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:62`, line by line against the C++
- [ ] **PORT 181** `getTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:68`, 0 lines
- [ ] **AUDIT 181** `getTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:68`, line by line against the C++
- [ ] **PORT 182** `getTransferSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:89`, 0 lines
- [ ] **AUDIT 182** `getTransferSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:89`, line by line against the C++
- [ ] **PORT 183** `getExpectedTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:88`, 0 lines
- [ ] **AUDIT 183** `getExpectedTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:88`, line by line against the C++
- [ ] **PORT 184** `setLdOrStSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:128`, 0 lines
- [ ] **AUDIT 184** `setLdOrStSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:128`, line by line against the C++
- [ ] **PORT 185** `getChunkSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:64`, 0 lines
- [ ] **AUDIT 185** `getChunkSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:64`, line by line against the C++
- [ ] **PORT 186** `getComp` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:58`, 0 lines
- [ ] **AUDIT 186** `getComp` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:58`, line by line against the C++
- [ ] **PORT 187** `getTimeSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:263`, 0 lines
- [ ] **AUDIT 187** `getTimeSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:263`, line by line against the C++
- [ ] **PORT 188** `getTimeOffsets` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:267`, 0 lines
- [ ] **AUDIT 188** `getTimeOffsets` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:267`, line by line against the C++
- [ ] **PORT 189** `getTimeBounds` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:266`, 0 lines
- [ ] **AUDIT 189** `getTimeBounds` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:266`, line by line against the C++
- [ ] **PORT 190** `getTimeAddrMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:265`, 0 lines
- [ ] **AUDIT 190** `getTimeAddrMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:265`, line by line against the C++
- [ ] **PORT 191** `getBurstIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:268`, 0 lines
- [ ] **AUDIT 191** `getBurstIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:268`, line by line against the C++
- [ ] **PORT 192** `getLdOrStSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:90`, 0 lines
- [ ] **AUDIT 192** `getLdOrStSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:90`, line by line against the C++
- [ ] **PORT 193** `setBurstIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:298`, 0 lines
- [ ] **AUDIT 193** `setBurstIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:298`, line by line against the C++
- [ ] **PORT 194** `getTimeOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:262`, 0 lines
- [ ] **AUDIT 194** `getTimeOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:262`, line by line against the C++
- [ ] **PORT 195** `getTimeSymbols` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:264`, 0 lines
- [ ] **AUDIT 195** `getTimeSymbols` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:264`, line by line against the C++
- [ ] **PORT 196** `getIndicesCoeffDict` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:222`, 0 lines
- [ ] **AUDIT 196** `getIndicesCoeffDict` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:222`, line by line against the C++
- [ ] **PORT 197** `comp_` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:47`, 0 lines
- [ ] **AUDIT 197** `comp_` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:47`, line by line against the C++
- [ ] **PORT 198** `getMemViewStartAddr` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:60`, 0 lines
- [ ] **AUDIT 198** `getMemViewStartAddr` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:60`, line by line against the C++
- [ ] **PORT 199** `getChunkStride` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:65`, 0 lines
- [ ] **AUDIT 199** `getChunkStride` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:65`, line by line against the C++
- [ ] **PORT 200** `getShuffleMode` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:66`, 0 lines
- [ ] **AUDIT 200** `getShuffleMode` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:66`, line by line against the C++
- [ ] **PORT 201** `getMemory` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:71`, 0 lines
- [ ] **AUDIT 201** `getMemory` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:71`, line by line against the C++
- [ ] **PORT 202** `setOp` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:76`, 0 lines
- [ ] **AUDIT 202** `setOp` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:76`, line by line against the C++
- [ ] **PORT 203** `AccessDetailsBase` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:218`, 0 lines
- [ ] **AUDIT 203** `AccessDetailsBase` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:218`, line by line against the C++
- [ ] **PORT 204** `AccessDetailsAffine` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:259`, 0 lines
- [ ] **AUDIT 204** `AccessDetailsAffine` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:259`, line by line against the C++
- [ ] **PORT 205** `getInterleaveGroupIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:269`, 0 lines
- [ ] **AUDIT 205** `getInterleaveGroupIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:269`, line by line against the C++
- [ ] **PORT 206** `setTimeOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:284`, 0 lines
- [ ] **AUDIT 206** `setTimeOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:284`, line by line against the C++
- [ ] **PORT 207** `setTimeSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:285`, 0 lines
- [ ] **AUDIT 207** `setTimeSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:285`, line by line against the C++
- [ ] **PORT 208** `getStrides` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:352`, 0 lines
- [ ] **AUDIT 208** `getStrides` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:352`, line by line against the C++
- [ ] **PORT 209** `index_mapping_` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:375`, 0 lines
- [ ] **AUDIT 209** `index_mapping_` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:375`, line by line against the C++
- [ ] **PORT 210** `dcc_ext_ctx_` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:38`, 0 lines
- [ ] **AUDIT 210** `dcc_ext_ctx_` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:38`, line by line against the C++
- [ ] **PORT 211** `dccExtContext` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:521`, 0 lines
- [ ] **AUDIT 211** `dccExtContext` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:521`, line by line against the C++

### Conversion/StandardToSentient — 8 defs, 71 lines

- [ ] **PORT 212** `getSentientCmpIPredicate` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:36`, 18 lines
- [ ] **AUDIT 212** `getSentientCmpIPredicate` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:36`, line by line against the C++
- [ ] **PORT 213** `LowerConstantIntToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:358`, 15 lines
- [ ] **AUDIT 213** `LowerConstantIntToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:358`, line by line against the C++
- [ ] **PORT 214** `LowerSubIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:90`, 10 lines
- [ ] **AUDIT 214** `LowerSubIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:90`, line by line against the C++
- [ ] **PORT 215** `LowerAddIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:79`, 9 lines
- [ ] **AUDIT 215** `LowerAddIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:79`, line by line against the C++
- [ ] **PORT 216** `LowerMulIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:102`, 9 lines
- [ ] **AUDIT 216** `LowerMulIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:102`, line by line against the C++
- [ ] **PORT 217** `LowerConstantIndexToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:347`, 8 lines
- [ ] **AUDIT 217** `LowerConstantIndexToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:347`, line by line against the C++
- [ ] **PORT 218** `createStandardToSentientPass` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:476`, 2 lines
- [ ] **AUDIT 218** `createStandardToSentientPass` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:476`, line by line against the C++
- [ ] **PORT 219** `If` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:159`, 0 lines
- [ ] **AUDIT 219** `If` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:159`, line by line against the C++

### Conversion/DataflowToSentient — 9 defs, 61 lines

- [ ] **PORT 220** `lowerOpaqueOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1984`, 27 lines
- [ ] **AUDIT 220** `lowerOpaqueOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1984`, line by line against the C++
- [ ] **PORT 221** `ExtendUnitNameToCorelet` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:104`, 11 lines
- [ ] **AUDIT 221** `ExtendUnitNameToCorelet` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:104`, line by line against the C++
- [ ] **PORT 222** `isSameListOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:175`, 10 lines
- [ ] **AUDIT 222** `isSameListOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:175`, line by line against the C++
- [ ] **PORT 223** `isTargetL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1720`, 6 lines
- [ ] **AUDIT 223** `isTargetL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1720`, line by line against the C++
- [ ] **PORT 224** `createDataflowToSentientPass` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2051`, 3 lines
- [ ] **AUDIT 224** `createDataflowToSentientPass` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2051`, line by line against the C++
- [ ] **PORT 225** `isSenComponentL0LU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:96`, 2 lines
- [ ] **AUDIT 225** `isSenComponentL0LU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:96`, line by line against the C++
- [ ] **PORT 226** `isSenComponentL0SU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:100`, 2 lines
- [ ] **AUDIT 226** `isSenComponentL0SU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:100`, line by line against the C++
- [ ] **PORT 227** `dcc_ext_ctx_` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:42`, 0 lines
- [ ] **AUDIT 227** `dcc_ext_ctx_` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:42`, line by line against the C++
- [ ] **PORT 228** `dccExtContext` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:91`, 0 lines
- [ ] **AUDIT 228** `dccExtContext` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:91`, line by line against the C++

### Conversion/SCFToSentient — 4 defs, 48 lines

- [ ] **PORT 229** `runOnOperation` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:251`, 30 lines
- [ ] **AUDIT 229** `runOnOperation` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:251`, line by line against the C++
- [ ] **PORT 230** `getSentientCmpIPredicate` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:33`, 16 lines
- [ ] **AUDIT 230** `getSentientCmpIPredicate` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:33`, line by line against the C++
- [ ] **PORT 231** `createSCFToSentientPass` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:285`, 2 lines
- [ ] **AUDIT 231** `createSCFToSentientPass` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:285`, line by line against the C++
- [ ] **PORT 232** `ConversionPattern` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:70`, 0 lines
- [ ] **AUDIT 232** `ConversionPattern` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:70`, line by line against the C++

### Conversion/AffineToStandard — 4 defs, 24 lines

- [x] **PORT 233** `matchAndRewrite` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:41`, 8 lines
- [x] **AUDIT 233** `matchAndRewrite` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:41`, line by line against the C++
- [ ] **PORT 234** `populateAffineToStdConversionPatterns` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:198`, 8 lines
- [ ] **AUDIT 234** `populateAffineToStdConversionPatterns` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:198`, line by line against the C++
- [ ] **PORT 235** `populateAffineToVectorConversionPatterns` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:208`, 6 lines
- [ ] **AUDIT 235** `populateAffineToVectorConversionPatterns` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:208`, line by line against the C++
- [ ] **PORT 236** `createAffineToStandardPass` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:239`, 2 lines
- [ ] **AUDIT 236** `createAffineToStandardPass` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:239`, line by line against the C++

### Conversion/SymbolToSentient — 3 defs, 3 lines

- [ ] **PORT 237** `createSymbolToSentientPass` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:187`, 3 lines
- [ ] **AUDIT 237** `createSymbolToSentientPass` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:187`, line by line against the C++
- [ ] **PORT 238** `dcc_ext_ctx_` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.hpp:27`, 0 lines
- [ ] **AUDIT 238** `dcc_ext_ctx_` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.hpp:27`, line by line against the C++
- [ ] **PORT 239** `dccExtContext` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.hpp:38`, 0 lines
- [ ] **AUDIT 239** `dccExtContext` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.hpp:38`, line by line against the C++

## Level 1 — 67 definitions, 2356 body lines

### Transform/Dataflow — 32 defs, 1178 lines

- [ ] **PORT 240** `analyzeLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:268`, 127 lines
- [ ] **AUDIT 240** `analyzeLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:268`, line by line against the C++
- [ ] **PORT 241** `calculatePartitionSizes` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:862`, 97 lines
- [ ] **AUDIT 241** `calculatePartitionSizes` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:862`, line by line against the C++
- [ ] **PORT 242** `matchUnits` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:69`, 77 lines
- [ ] **AUDIT 242** `matchUnits` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:69`, line by line against the C++
- [ ] **PORT 243** `mergeShallow` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:398`, 60 lines
- [ ] **AUDIT 243** `mergeShallow` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:398`, line by line against the C++
- [ ] **PORT 244** `cloneOpsForRegions` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:223`, 60 lines
- [ ] **AUDIT 244** `cloneOpsForRegions` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:223`, line by line against the C++
- [ ] **PORT 245** `constructConditionals` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1000`, 59 lines
- [ ] **AUDIT 245** `constructConditionals` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1000`, line by line against the C++
- [ ] **PORT 246** `createExplicitTimeLoops` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1282`, 57 lines
- [ ] **AUDIT 246** `createExplicitTimeLoops` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1282`, line by line against the C++
- [ ] **PORT 247** `getLoopTripCount` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:743`, 56 lines
- [ ] **AUDIT 247** `getLoopTripCount` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:743`, line by line against the C++
- [ ] **PORT 248** `expandAffineApplyOps` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:183`, 53 lines
- [ ] **AUDIT 248** `expandAffineApplyOps` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:183`, line by line against the C++
- [ ] **PORT 249** `cleanup` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:263`, 49 lines
- [ ] **AUDIT 249** `cleanup` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:263`, line by line against the C++
- [ ] **PORT 250** `createConditionsForHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:289`, 48 lines
- [ ] **AUDIT 250** `createConditionsForHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:289`, line by line against the C++
- [ ] **PORT 251** `runOnOperation` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:49`, 42 lines
- [ ] **AUDIT 251** `runOnOperation` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:49`, line by line against the C++
- [ ] **PORT 252** `transformLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:399`, 38 lines
- [ ] **AUDIT 252** `transformLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:399`, line by line against the C++
- [ ] **PORT 253** `removeCoresCoreletsFoldsFromDefImmutMap` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:99`, 37 lines
- [ ] **AUDIT 253** `removeCoresCoreletsFoldsFromDefImmutMap` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:99`, line by line against the C++
- [ ] **PORT 254** `createConditionsForNonHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:342`, 35 lines
- [ ] **AUDIT 254** `createConditionsForNonHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:342`, line by line against the C++
- [ ] **PORT 255** `calculateDimWeights` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:560`, 26 lines
- [ ] **AUDIT 255** `calculateDimWeights` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:560`, line by line against the C++
- [ ] **PORT 256** `calculateFullShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:462`, 25 lines
- [ ] **AUDIT 256** `calculateFullShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:462`, line by line against the C++
- [ ] **PORT 257** `calculateIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:56`, 25 lines
- [ ] **AUDIT 257** `calculateIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:56`, line by line against the C++
- [ ] **PORT 258** `runOnOperation` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:41`, 23 lines
- [ ] **AUDIT 258** `runOnOperation` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:41`, line by line against the C++
- [ ] **PORT 259** `synthesizeTimeInfo` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1256`, 22 lines
- [ ] **AUDIT 259** `synthesizeTimeInfo` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1256`, line by line against the C++
- [ ] **PORT 260** `if_op_` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:56`, 20 lines
- [ ] **AUDIT 260** `if_op_` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:56`, line by line against the C++
- [ ] **PORT 261** `traverseRegion` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:129`, 17 lines
- [ ] **AUDIT 261** `traverseRegion` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:129`, line by line against the C++
- [ ] **PORT 262** `updateTPMVInfo` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:382`, 17 lines
- [ ] **AUDIT 262** `updateTPMVInfo` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:382`, line by line against the C++
- [ ] **PORT 263** `clear` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:112`, 15 lines
- [ ] **AUDIT 263** `clear` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:112`, line by line against the C++
- [ ] **PORT 264** `hasMutableAddrOverflow` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:801`, 14 lines
- [ ] **AUDIT 264** `hasMutableAddrOverflow` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:801`, line by line against the C++
- [ ] **PORT 265** `opHasSideEffect` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:38`, 13 lines
- [ ] **AUDIT 265** `opHasSideEffect` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:38`, line by line against the C++
- [ ] **PORT 266** `printTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:288`, 13 lines
- [ ] **AUDIT 266** `printTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:288`, line by line against the C++
- [ ] **PORT 267** `setLoopIteratorOrder` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:575`, 13 lines
- [ ] **AUDIT 267** `setLoopIteratorOrder` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:575`, line by line against the C++
- [ ] **PORT 268** `initialize` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:658`, 13 lines
- [ ] **AUDIT 268** `initialize` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:658`, line by line against the C++
- [ ] **PORT 269** `calculateSubscriptsCoefficients` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1188`, 11 lines
- [ ] **AUDIT 269** `calculateSubscriptsCoefficients` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1188`, line by line against the C++
- [ ] **PORT 270** `isDataTransferToKeep` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:327`, 10 lines
- [ ] **AUDIT 270** `isDataTransferToKeep` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:327`, line by line against the C++
- [ ] **PORT 271** `inRegionEmpty` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:214`, 6 lines
- [ ] **AUDIT 271** `inRegionEmpty` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:214`, line by line against the C++

### Conversion/AgenToSentient — 16 defs, 764 lines

- [ ] **PORT 272** `constructSetActiveMaskValueOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2567`, 161 lines
- [ ] **AUDIT 272** `constructSetActiveMaskValueOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2567`, line by line against the C++
- [ ] **PORT 273** `getStoreProducer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1285`, 159 lines
- [ ] **AUDIT 273** `getStoreProducer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1285`, line by line against the C++
- [ ] **PORT 274** `checkCompositeRegion` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:206`, 89 lines
- [ ] **AUDIT 274** `checkCompositeRegion` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:206`, line by line against the C++
- [ ] **PORT 275** `constructExtentAndTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:32`, 64 lines
- [ ] **AUDIT 275** `constructExtentAndTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:32`, line by line against the C++
- [ ] **PORT 276** `createUniformizeRegionsOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3880`, 57 lines
- [ ] **AUDIT 276** `createUniformizeRegionsOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3880`, line by line against the C++
- [ ] **PORT 277** `constructIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:354`, 50 lines
- [ ] **AUDIT 277** `constructIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:354`, line by line against the C++
- [ ] **PORT 278** `computeBurstAndGroup` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:796`, 36 lines
- [ ] **AUDIT 278** `computeBurstAndGroup` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:796`, line by line against the C++
- [ ] **PORT 279** `updateSymbolicAccessDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1013`, 35 lines
- [ ] **AUDIT 279** `updateSymbolicAccessDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1013`, line by line against the C++
- [ ] **PORT 280** `isLoadAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:463`, 27 lines
- [ ] **AUDIT 280** `isLoadAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:463`, line by line against the C++
- [ ] **PORT 281** `checkStoreOpFromExtractPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:433`, 27 lines
- [ ] **AUDIT 281** `checkStoreOpFromExtractPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:433`, line by line against the C++
- [ ] **PORT 282** `isReceiveAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:493`, 17 lines
- [ ] **AUDIT 282** `isReceiveAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:493`, line by line against the C++
- [ ] **PORT 283** `initializeMemViewInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:274`, 16 lines
- [ ] **AUDIT 283** `initializeMemViewInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:274`, line by line against the C++
- [ ] **PORT 284** `constructIteratorCoefficients` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:406`, 10 lines
- [ ] **AUDIT 284** `constructIteratorCoefficients` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:406`, line by line against the C++
- [ ] **PORT 285** `insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:388`, 7 lines
- [ ] **AUDIT 285** `insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:388`, line by line against the C++
- [ ] **PORT 286** `constructLdOrStType` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:267`, 5 lines
- [ ] **AUDIT 286** `constructLdOrStType` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:267`, line by line against the C++
- [ ] **PORT 287** `get` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:401`, 4 lines
- [ ] **AUDIT 287** `get` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:401`, line by line against the C++

### Conversion/VectorChainLowering — 15 defs, 370 lines

- [ ] **PORT 288** `getMergeTypeFromIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:301`, 109 lines
- [ ] **AUDIT 288** `getMergeTypeFromIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:301`, line by line against the C++
- [ ] **PORT 289** `checkValidityOfPackAndShuffleLowering` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:140`, 45 lines
- [ ] **AUDIT 289** `checkValidityOfPackAndShuffleLowering` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:140`, line by line against the C++
- [ ] **PORT 290** `getLayoutMapAndIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:879`, 40 lines
- [ ] **AUDIT 290** `getLayoutMapAndIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:879`, line by line against the C++
- [ ] **PORT 291** `getMaskValueConstantForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:93`, 40 lines
- [ ] **AUDIT 291** `getMaskValueConstantForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:93`, line by line against the C++
- [ ] **PORT 292** `isXrfRelated` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:531`, 31 lines
- [ ] **AUDIT 292** `isXrfRelated` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:531`, line by line against the C++
- [ ] **PORT 293** `areXrfAccessesLegal` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:98`, 28 lines
- [ ] **AUDIT 293** `areXrfAccessesLegal` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:98`, line by line against the C++
- [ ] **PORT 294** `getOperandFromConstantOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:267`, 26 lines
- [ ] **AUDIT 294** `getOperandFromConstantOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:267`, line by line against the C++
- [ ] **PORT 295** `getName` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:866`, 11 lines
- [ ] **AUDIT 295** `getName` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:866`, line by line against the C++
- [ ] **PORT 296** `insertConstAndAddOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:296`, 10 lines
- [ ] **AUDIT 296** `insertConstAndAddOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:296`, line by line against the C++
- [ ] **PORT 297** `insertIfNotExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:81`, 8 lines
- [ ] **AUDIT 297** `insertIfNotExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:81`, line by line against the C++
- [ ] **PORT 298** `updateYieldArgs` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:690`, 8 lines
- [ ] **AUDIT 298** `updateYieldArgs` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:690`, line by line against the C++
- [ ] **PORT 299** `getOperandFromConstantBitstreamOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:299`, 5 lines
- [ ] **AUDIT 299** `getOperandFromConstantBitstreamOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:299`, line by line against the C++
- [ ] **PORT 300** `getOperandFromNegOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:366`, 5 lines
- [ ] **AUDIT 300** `getOperandFromNegOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:366`, line by line against the C++
- [ ] **PORT 301** `isMaskEquivalentToNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:123`, 4 lines
- [ ] **AUDIT 301** `isMaskEquivalentToNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:123`, line by line against the C++
- [ ] **PORT 302** `getTotalDataOriginsCount` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:33`, 0 lines
- [ ] **AUDIT 302** `getTotalDataOriginsCount` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:33`, line by line against the C++

### Conversion/DataflowToSentient — 3 defs, 34 lines

- [ ] **PORT 303** `separateBasedOnDestinationUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:761`, 18 lines
- [ ] **AUDIT 303** `separateBasedOnDestinationUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:761`, line by line against the C++
- [ ] **PORT 304** `getUnitNameFromAListOfGetUnitOp` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:119`, 10 lines
- [ ] **AUDIT 304** `getUnitNameFromAListOfGetUnitOp` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:119`, line by line against the C++
- [ ] **PORT 305** `areCoreletsDifferent` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:132`, 6 lines
- [ ] **AUDIT 305** `areCoreletsDifferent` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:132`, line by line against the C++

### Conversion/AffineToStandard — 1 defs, 10 lines

- [ ] **PORT 306** `runOnOperation` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:222`, 10 lines
- [ ] **AUDIT 306** `runOnOperation` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:222`, line by line against the C++

## Level 2 — 61 definitions, 2729 body lines

### Conversion/AgenToSentient — 15 defs, 898 lines

- [ ] **PORT 307** `constructChunkAndShuffleInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:98`, 167 lines
- [ ] **AUDIT 307** `constructChunkAndShuffleInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:98`, line by line against the C++
- [ ] **PORT 308** `checkBasicConditions` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:58`, 133 lines
- [ ] **AUDIT 308** `checkBasicConditions` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:58`, line by line against the C++
- [ ] **PORT 309** `constructReceiveAndExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2471`, 91 lines
- [ ] **AUDIT 309** `constructReceiveAndExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2471`, line by line against the C++
- [ ] **PORT 310** `processInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:305`, 80 lines
- [ ] **AUDIT 310** `processInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:305`, line by line against the C++
- [ ] **PORT 311** `cloneStartAddrOutsideLoop` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3942`, 79 lines
- [ ] **AUDIT 311** `cloneStartAddrOutsideLoop` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3942`, line by line against the C++
- [ ] **PORT 312** `gatherAffineLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:538`, 74 lines
- [ ] **AUDIT 312** `gatherAffineLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:538`, line by line against the C++
- [ ] **PORT 313** `initialize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:295`, 57 lines
- [ ] **AUDIT 313** `initialize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:295`, line by line against the C++
- [ ] **PORT 314** `setsttype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1731`, 50 lines
- [ ] **AUDIT 314** `setsttype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1731`, line by line against the C++
- [ ] **PORT 315** `lowerVectorLoadHelper` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2899`, 46 lines
- [ ] **AUDIT 315** `lowerVectorLoadHelper` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2899`, line by line against the C++
- [x] **PORT 316** `setImmutableAddrAndIncrements` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1581`, 43 lines
- [x] **AUDIT 316** `setImmutableAddrAndIncrements` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1581`, line by line against the C++
- [ ] **PORT 317** `cleanupTriviallyRedundantSetSendDestination` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:4084`, 38 lines
- [ ] **AUDIT 317** `cleanupTriviallyRedundantSetSendDestination` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:4084`, line by line against the C++
- [ ] **PORT 318** `constructImmutableAddress` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1217`, 16 lines
- [ ] **AUDIT 318** `constructImmutableAddress` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1217`, line by line against the C++
- [ ] **PORT 319** `lowerSetTransferMaskStateOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3815`, 9 lines
- [ ] **AUDIT 319** `lowerSetTransferMaskStateOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3815`, line by line against the C++
- [ ] **PORT 320** `emplace_insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:378`, 8 lines
- [ ] **AUDIT 320** `emplace_insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:378`, line by line against the C++
- [ ] **PORT 321** `getFirst` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:411`, 7 lines
- [ ] **AUDIT 321** `getFirst` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:411`, line by line against the C++

### Transform/Dataflow — 21 defs, 856 lines

- [ ] **PORT 322** `fillPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1063`, 117 lines
- [ ] **AUDIT 322** `fillPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1063`, line by line against the C++
- [ ] **PORT 323** `removeCoresCoreletsFoldsFromUniformizeRegion` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:139`, 98 lines
- [ ] **AUDIT 323** `removeCoresCoreletsFoldsFromUniformizeRegion` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:139`, line by line against the C++
- [ ] **PORT 324** `processComputeUnit` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:37`, 91 lines
- [ ] **AUDIT 324** `processComputeUnit` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:37`, line by line against the C++
- [ ] **PORT 325** `runOnOperation` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:152`, 76 lines
- [ ] **AUDIT 325** `runOnOperation` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:152`, line by line against the C++
- [ ] **PORT 326** `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:70`, 68 lines
- [ ] **AUDIT 326** `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:70`, line by line against the C++
- [ ] **PORT 327** `createForOpWithAdditionalReturnValue` — `dcc/src/Transform/Dataflow/Utils.cpp:28`, 66 lines
- [ ] **AUDIT 327** `createForOpWithAdditionalReturnValue` — `dcc/src/Transform/Dataflow/Utils.cpp:28`, line by line against the C++
- [ ] **PORT 328** `enumerateCollectionUnit` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:34`, 57 lines
- [ ] **AUDIT 328** `enumerateCollectionUnit` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:34`, line by line against the C++
- [ ] **PORT 329** `adjustForEvenImmutableAddr` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1204`, 47 lines
- [ ] **AUDIT 329** `adjustForEvenImmutableAddr` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1204`, line by line against the C++
- [ ] **PORT 330** `runOnOperation` — `dcc/src/Transform/Dataflow/UniformQueryMapsCanonicalization.cpp:55`, 41 lines
- [ ] **AUDIT 330** `runOnOperation` — `dcc/src/Transform/Dataflow/UniformQueryMapsCanonicalization.cpp:55`, line by line against the C++
- [ ] **PORT 331** `gatherPageDependentDimsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:927`, 32 lines
- [ ] **AUDIT 331** `gatherPageDependentDimsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:927`, line by line against the C++
- [ ] **PORT 332** `initMASData` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:707`, 31 lines
- [ ] **AUDIT 332** `initMASData` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:707`, line by line against the C++
- [ ] **PORT 333** `applyShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:616`, 27 lines
- [ ] **AUDIT 333** `applyShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:616`, line by line against the C++
- [ ] **PORT 334** `offsetShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:590`, 22 lines
- [ ] **AUDIT 334** `offsetShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:590`, line by line against the C++
- [ ] **PORT 335** `addConstraintsForIVRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:86`, 22 lines
- [ ] **AUDIT 335** `addConstraintsForIVRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:86`, line by line against the C++
- [ ] **PORT 336** `removeAncestors` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:339`, 18 lines
- [ ] **AUDIT 336** `removeAncestors` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:339`, line by line against the C++
- [ ] **PORT 337** `compute` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:151`, 15 lines
- [ ] **AUDIT 337** `compute` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:151`, line by line against the C++
- [ ] **PORT 338** `createNewSubscriptsFromStartElements` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:562`, 10 lines
- [ ] **AUDIT 338** `createNewSubscriptsFromStartElements` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:562`, line by line against the C++
- [ ] **PORT 339** `isHoistable` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:82`, 9 lines
- [ ] **AUDIT 339** `isHoistable` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:82`, line by line against the C++
- [ ] **PORT 340** `setupForPartitioning` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:819`, 6 lines
- [ ] **AUDIT 340** `setupForPartitioning` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:819`, line by line against the C++
- [ ] **PORT 341** `analyzeAndTransform` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:441`, 3 lines
- [ ] **AUDIT 341** `analyzeAndTransform` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:441`, line by line against the C++
- [ ] **PORT 342** `FlatteningLocalRegionsTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:79`, 0 lines
- [ ] **AUDIT 342** `FlatteningLocalRegionsTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:79`, line by line against the C++

### Conversion/VectorChainLowering — 19 defs, 704 lines

- [ ] **PORT 343** `insertPTMaskOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:41`, 164 lines
- [ ] **AUDIT 343** `insertPTMaskOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:41`, line by line against the C++
- [ ] **PORT 344** `getOperandFromLoadOrStoreOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:153`, 94 lines
- [ ] **AUDIT 344** `getOperandFromLoadOrStoreOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:153`, line by line against the C++
- [ ] **PORT 345** `getLayoutExpr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:28`, 67 lines
- [ ] **AUDIT 345** `getLayoutExpr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:28`, line by line against the C++
- [ ] **PORT 346** `createForOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:129`, 56 lines
- [ ] **AUDIT 346** `createForOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:129`, line by line against the C++
- [ ] **PORT 347** `createIfOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:189`, 56 lines
- [ ] **AUDIT 347** `createIfOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:189`, line by line against the C++
- [ ] **PORT 348** `validateLoweringAndSetMissingParameters` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:483`, 49 lines
- [ ] **AUDIT 348** `validateLoweringAndSetMissingParameters` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:483`, line by line against the C++
- [ ] **PORT 349** `eraseOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:690`, 46 lines
- [ ] **AUDIT 349** `eraseOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:690`, line by line against the C++
- [ ] **PORT 350** `createSentientConstants` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:34`, 31 lines
- [ ] **AUDIT 350** `createSentientConstants` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:34`, line by line against the C++
- [ ] **PORT 351** `convertStringToType` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:196`, 24 lines
- [ ] **AUDIT 351** `convertStringToType` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:196`, line by line against the C++
- [ ] **PORT 352** `print` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:96`, 22 lines
- [ ] **AUDIT 352** `print` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:96`, line by line against the C++
- [ ] **PORT 353** `convertTypeToString` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:223`, 22 lines
- [ ] **AUDIT 353** `convertTypeToString` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:223`, line by line against the C++
- [ ] **PORT 354** `computeLoops` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:173`, 18 lines
- [ ] **AUDIT 354** `computeLoops` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:173`, line by line against the C++
- [ ] **PORT 355** `addMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:136`, 16 lines
- [ ] **AUDIT 355** `addMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:136`, line by line against the C++
- [ ] **PORT 356** `insertDummyMacOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:312`, 15 lines
- [ ] **AUDIT 356** `insertDummyMacOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:312`, line by line against the C++
- [ ] **PORT 357** `updateNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:155`, 10 lines
- [ ] **AUDIT 357** `updateNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:155`, line by line against the C++
- [ ] **PORT 358** `getMaskValueForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:114`, 9 lines
- [ ] **AUDIT 358** `getMaskValueForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:114`, line by line against the C++
- [ ] **PORT 359** `setValue` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:72`, 3 lines
- [ ] **AUDIT 359** `setValue` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:72`, line by line against the C++
- [ ] **PORT 360** `dominance_info_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:26`, 2 lines
- [ ] **AUDIT 360** `dominance_info_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:26`, line by line against the C++
- [ ] **PORT 361** `OperandReuse` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:30`, 0 lines
- [ ] **AUDIT 361** `OperandReuse` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:30`, line by line against the C++

### Conversion/DataflowToSentient — 4 defs, 141 lines

- [ ] **PORT 362** `lowerL3SyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:667`, 61 lines
- [ ] **AUDIT 362** `lowerL3SyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:667`, line by line against the C++
- [ ] **PORT 363** `lowerL3SyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:375`, 57 lines
- [ ] **AUDIT 363** `lowerL3SyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:375`, line by line against the C++
- [ ] **PORT 364** `createUniformRegionsWithTwoRegionsNoResult` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:153`, 18 lines
- [ ] **AUDIT 364** `createUniformRegionsWithTwoRegionsNoResult` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:153`, line by line against the C++
- [ ] **PORT 365** `pushBackTheUnitToListIfDoesnotExist` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:143`, 5 lines
- [ ] **AUDIT 365** `pushBackTheUnitToListIfDoesnotExist` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:143`, line by line against the C++

### Conversion/SCFToSentient — 1 defs, 69 lines

- [ ] **PORT 366** `matchAndRewrite` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:72`, 69 lines
- [ ] **AUDIT 366** `matchAndRewrite` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:72`, line by line against the C++

### Conversion/SymbolToSentient — 1 defs, 61 lines

- [ ] **PORT 367** `createIfOpFromMapping` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:123`, 61 lines
- [ ] **AUDIT 367** `createIfOpFromMapping` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:123`, line by line against the C++

## Level 3 — 33 definitions, 2120 body lines

### Transform/Dataflow — 13 defs, 740 lines

- [ ] **PORT 368** `runOnOperation` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:362`, 127 lines
- [ ] **AUDIT 368** `runOnOperation` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:362`, line by line against the C++
- [ ] **PORT 369** `createIterArgsForConditionals` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:401`, 121 lines
- [ ] **AUDIT 369** `createIterArgsForConditionals` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:401`, line by line against the C++
- [ ] **PORT 370** `hoistCommonConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:94`, 96 lines
- [ ] **AUDIT 370** `hoistCommonConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:94`, line by line against the C++
- [ ] **PORT 371** `transformSCFLoopWithNonConstantUpperBound` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:169`, 94 lines
- [ ] **AUDIT 371** `transformSCFLoopWithNonConstantUpperBound` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:169`, line by line against the C++
- [ ] **PORT 372** `flatten` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:384`, 70 lines
- [ ] **AUDIT 372** `flatten` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:384`, line by line against the C++
- [ ] **PORT 373** `calculatePartialShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:490`, 66 lines
- [ ] **AUDIT 373** `calculatePartialShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:490`, line by line against the C++
- [ ] **PORT 374** `replaceIfOpByIterArg` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:619`, 56 lines
- [ ] **AUDIT 374** `replaceIfOpByIterArg` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:619`, line by line against the C++
- [ ] **PORT 375** `getPageValidity` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:114`, 33 lines
- [ ] **AUDIT 375** `getPageValidity` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:114`, line by line against the C++
- [ ] **PORT 376** `runOnOperation` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:94`, 32 lines
- [ ] **AUDIT 376** `runOnOperation` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:94`, line by line against the C++
- [ ] **PORT 377** `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:131`, 22 lines
- [ ] **AUDIT 377** `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:131`, line by line against the C++
- [ ] **PORT 378** `createPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:983`, 10 lines
- [ ] **AUDIT 378** `createPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:983`, line by line against the C++
- [ ] **PORT 379** `initialize` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:694`, 8 lines
- [ ] **AUDIT 379** `initialize` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:694`, line by line against the C++
- [ ] **PORT 380** `runOn` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:447`, 5 lines
- [ ] **AUDIT 380** `runOn` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:447`, line by line against the C++

### Conversion/AgenToSentient — 8 defs, 588 lines

- [ ] **PORT 381** `constructLoadAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2167`, 177 lines
- [ ] **AUDIT 381** `constructLoadAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2167`, line by line against the C++
- [ ] **PORT 382** `constructLoadAndExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2351`, 114 lines
- [ ] **AUDIT 382** `constructLoadAndExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2351`, line by line against the C++
- [ ] **PORT 383** `coalesceTimeDimensions` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:681`, 112 lines
- [ ] **AUDIT 383** `coalesceTimeDimensions` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:681`, line by line against the C++
- [ ] **PORT 384** `constructTimeLoopsAndVectorOperations` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1789`, 109 lines
- [ ] **AUDIT 384** `constructTimeLoopsAndVectorOperations` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1789`, line by line against the C++
- [ ] **PORT 385** `lowerCompositeMemoryInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3775`, 37 lines
- [ ] **AUDIT 385** `lowerCompositeMemoryInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3775`, line by line against the C++
- [ ] **PORT 386** `constructDetails` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:418`, 18 lines
- [ ] **AUDIT 386** `constructDetails` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:418`, line by line against the C++
- [ ] **PORT 387** `insertInitializationStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3834`, 13 lines
- [ ] **AUDIT 387** `insertInitializationStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3834`, line by line against the C++
- [ ] **PORT 388** `addStoreInputToDeleteList` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2987`, 8 lines
- [ ] **AUDIT 388** `addStoreInputToDeleteList` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2987`, line by line against the C++

### Conversion/DataflowToSentient — 2 defs, 402 lines

- [ ] **PORT 389** `lowerL0LXSyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:438`, 222 lines
- [ ] **AUDIT 389** `lowerL0LXSyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:438`, line by line against the C++
- [ ] **PORT 390** `lowerL0LXSyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:189`, 180 lines
- [ ] **AUDIT 390** `lowerL0LXSyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:189`, line by line against the C++

### Conversion/VectorChainLowering — 9 defs, 322 lines

- [ ] **PORT 391** `createSplatOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:70`, 110 lines
- [ ] **AUDIT 391** `createSplatOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:70`, line by line against the C++
- [ ] **PORT 392** `getGCVTorFCVTTypeFromIndicesAndCastInputs` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:188`, 108 lines
- [ ] **AUDIT 392** `getGCVTorFCVTTypeFromIndicesAndCastInputs` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:188`, line by line against the C++
- [ ] **PORT 393** `setReuseInformation` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:17`, 44 lines
- [ ] **AUDIT 393** `setReuseInformation` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:17`, line by line against the C++
- [ ] **PORT 394** `getOperandFromShuffleOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:311`, 36 lines
- [ ] **AUDIT 394** `getOperandFromShuffleOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:311`, line by line against the C++
- [ ] **PORT 395** `updateLoopMaskTreeForDynamicMask` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:28`, 9 lines
- [ ] **AUDIT 395** `updateLoopMaskTreeForDynamicMask` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:28`, line by line against the C++
- [ ] **PORT 396** `cleanup` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1056`, 7 lines
- [ ] **AUDIT 396** `cleanup` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1056`, line by line against the C++
- [ ] **PORT 397** `updateLoopMaskTreeForConstantMask` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:20`, 4 lines
- [ ] **AUDIT 397** `updateLoopMaskTreeForConstantMask` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:20`, line by line against the C++
- [ ] **PORT 398** `op_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:79`, 2 lines
- [ ] **AUDIT 398** `op_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:79`, line by line against the C++
- [ ] **PORT 399** `OperationTreeBase` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:105`, 2 lines
- [ ] **AUDIT 399** `OperationTreeBase` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:105`, line by line against the C++

### Conversion/SymbolToSentient — 1 defs, 68 lines

- [ ] **PORT 400** `LowerSymbolQueryMap` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:40`, 68 lines
- [ ] **AUDIT 400** `LowerSymbolQueryMap` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:40`, line by line against the C++

## Level 4 — 15 definitions, 1606 body lines

### Conversion/DataflowToSentient — 3 defs, 943 lines

- [ ] **PORT 401** `lowerSyncLXL3ToLXL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:787`, 928 lines
- [ ] **AUDIT 401** `lowerSyncLXL3ToLXL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:787`, line by line against the C++
- [ ] **PORT 402** `lowerSyncForAGroup` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:746`, 8 lines
- [ ] **AUDIT 402** `lowerSyncForAGroup` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:746`, line by line against the C++
- [ ] **PORT 403** `lowerSyncForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:733`, 7 lines
- [ ] **AUDIT 403** `lowerSyncForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:733`, line by line against the C++

### Transform/Dataflow — 7 defs, 321 lines

- [ ] **PORT 404** `transformVectorLoad` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:298`, 75 lines
- [ ] **AUDIT 404** `transformVectorLoad` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:298`, line by line against the C++
- [ ] **PORT 405** `transformVectorStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:375`, 74 lines
- [ ] **AUDIT 405** `transformVectorStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:375`, line by line against the C++
- [ ] **PORT 406** `calculateShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:396`, 61 lines
- [ ] **AUDIT 406** `calculateShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:396`, line by line against the C++
- [ ] **PORT 407** `constructValidPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:190`, 58 lines
- [ ] **AUDIT 407** `constructValidPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:190`, line by line against the C++
- [ ] **PORT 408** `analyzeValidPages` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:888`, 33 lines
- [ ] **AUDIT 408** `analyzeValidPages` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:888`, line by line against the C++
- [ ] **PORT 409** `runOnOperation` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:459`, 17 lines
- [ ] **AUDIT 409** `runOnOperation` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:459`, line by line against the C++
- [ ] **PORT 410** `runOnOperation` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:455`, 3 lines
- [ ] **AUDIT 410** `runOnOperation` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:455`, line by line against the C++

### Conversion/VectorChainLowering — 1 defs, 254 lines

- [ ] **PORT 411** `getOperandWithPrecision` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:389`, 254 lines
- [ ] **AUDIT 411** `getOperandWithPrecision` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:389`, line by line against the C++

### Conversion/AgenToSentient — 3 defs, 73 lines

- [ ] **PORT 412** `constructTimeStepsInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:626`, 42 lines
- [ ] **AUDIT 412** `constructTimeStepsInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:626`, line by line against the C++
- [ ] **PORT 413** `constructAffineDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2787`, 16 lines
- [ ] **AUDIT 413** `constructAffineDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2787`, line by line against the C++
- [ ] **PORT 414** `lowerAffineCompositeHelper` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2953`, 15 lines
- [ ] **AUDIT 414** `lowerAffineCompositeHelper` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2953`, line by line against the C++

### Conversion/SymbolToSentient — 1 defs, 15 lines

- [ ] **PORT 415** `runOnOperation` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:23`, 15 lines
- [ ] **AUDIT 415** `runOnOperation` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:23`, line by line against the C++

## Level 5 — 16 definitions, 1096 body lines

### Conversion/AgenToSentient — 8 defs, 538 lines

- [ ] **PORT 416** `lowerLDCVTIPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3444`, 326 lines
- [ ] **AUDIT 416** `lowerLDCVTIPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3444`, line by line against the C++
- [ ] **PORT 417** `lowerIndirectVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3217`, 46 lines
- [ ] **AUDIT 417** `lowerIndirectVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3217`, line by line against the C++
- [ ] **PORT 418** `lowerIndirectVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3169`, 44 lines
- [ ] **AUDIT 418** `lowerIndirectVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3169`, line by line against the C++
- [ ] **PORT 419** `constructAffineCompDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2809`, 33 lines
- [ ] **AUDIT 419** `constructAffineCompDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2809`, line by line against the C++
- [ ] **PORT 420** `lowerVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3074`, 28 lines
- [ ] **AUDIT 420** `lowerVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3074`, line by line against the C++
- [ ] **PORT 421** `lowerExtractVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3001`, 21 lines
- [ ] **AUDIT 421** `lowerExtractVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3001`, line by line against the C++
- [ ] **PORT 422** `lowerVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3050`, 20 lines
- [ ] **AUDIT 422** `lowerVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3050`, line by line against the C++
- [ ] **PORT 423** `lowerExtractVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3026`, 20 lines
- [ ] **AUDIT 423** `lowerExtractVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3026`, line by line against the C++

### Transform/Dataflow — 6 defs, 388 lines

- [ ] **PORT 424** `transformCompIndLoadAndStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:546`, 124 lines
- [ ] **AUDIT 424** `transformCompIndLoadAndStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:546`, line by line against the C++
- [ ] **PORT 425** `transform_time` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:977`, 98 lines
- [ ] **AUDIT 425** `transform_time` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:977`, line by line against the C++
- [ ] **PORT 426** `transformCompLoadAndStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:451`, 92 lines
- [ ] **AUDIT 426** `transformCompLoadAndStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:451`, line by line against the C++
- [ ] **PORT 427** `analyzeAndConstructValidPages` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:152`, 32 lines
- [ ] **AUDIT 427** `analyzeAndConstructValidPages` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:152`, line by line against the C++
- [ ] **PORT 428** `initialize_time` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:1095`, 23 lines
- [ ] **AUDIT 428** `initialize_time` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:1095`, line by line against the C++
- [ ] **PORT 429** `shiftMutableAddr` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:366`, 19 lines
- [ ] **AUDIT 429** `shiftMutableAddr` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:366`, line by line against the C++

### Conversion/DataflowToSentient — 1 defs, 166 lines

- [ ] **PORT 430** `lowerSyncForAQueryMap` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1728`, 166 lines
- [ ] **AUDIT 430** `lowerSyncForAQueryMap` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1728`, line by line against the C++

### Conversion/VectorChainLowering — 1 defs, 4 lines

- [ ] **PORT 431** `getOperand` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:378`, 4 lines
- [ ] **AUDIT 431** `getOperand` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:378`, line by line against the C++

## Level 6 — 30 definitions, 1790 body lines

### Transform/Dataflow — 10 defs, 536 lines

- [ ] **PORT 432** `matchAndRewrite` — `dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:33`, 170 lines
- [ ] **AUDIT 432** `matchAndRewrite` — `dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:33`, line by line against the C++
- [ ] **PORT 433** `runOnOperation` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:226`, 70 lines
- [ ] **AUDIT 433** `runOnOperation` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:226`, line by line against the C++
- [ ] **PORT 434** `isLoopInvariant` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:681`, 66 lines
- [ ] **AUDIT 434** `isLoopInvariant` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:681`, line by line against the C++
- [ ] **PORT 435** `transformCompIndLoadAndStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:298`, 54 lines
- [ ] **AUDIT 435** `transformCompIndLoadAndStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:298`, line by line against the C++
- [ ] **PORT 436** `transformCompLoadAndStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:256`, 39 lines
- [ ] **AUDIT 436** `transformCompLoadAndStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:256`, line by line against the C++
- [ ] **PORT 437** `transform` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:592`, 39 lines
- [ ] **AUDIT 437** `transform` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:592`, line by line against the C++
- [ ] **PORT 438** `topLevelConditionsMatch` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:279`, 31 lines
- [ ] **AUDIT 438** `topLevelConditionsMatch` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:279`, line by line against the C++
- [ ] **PORT 439** `transformVectorLoad` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:201`, 25 lines
- [ ] **AUDIT 439** `transformVectorLoad` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:201`, line by line against the C++
- [ ] **PORT 440** `transformVectorStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:229`, 24 lines
- [ ] **AUDIT 440** `transformVectorStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:229`, line by line against the C++
- [ ] **PORT 441** `singleOpBranchToYieldVal` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:498`, 18 lines
- [ ] **AUDIT 441** `singleOpBranchToYieldVal` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:498`, line by line against the C++

### Conversion/VectorChainLowering — 7 defs, 514 lines

- [ ] **PORT 442** `processXrfPtrPerUnit` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:337`, 190 lines
- [ ] **AUDIT 442** `processXrfPtrPerUnit` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:337`, line by line against the C++
- [ ] **PORT 443** `lowerDanglingNonComputeOpsPESFP` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1274`, 93 lines
- [ ] **AUDIT 443** `lowerDanglingNonComputeOpsPESFP` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1274`, line by line against the C++
- [ ] **PORT 444** `lowerDanglingNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:882`, 88 lines
- [ ] **AUDIT 444** `lowerDanglingNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:882`, line by line against the C++
- [ ] **PORT 445** `analyzeNonComputeOpsForFusion` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:610`, 86 lines
- [ ] **AUDIT 445** `analyzeNonComputeOpsForFusion` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:610`, line by line against the C++
- [ ] **PORT 446** `analyzeAndFillResultForwarding` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:162`, 27 lines
- [ ] **AUDIT 446** `analyzeAndFillResultForwarding` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:162`, line by line against the C++
- [ ] **PORT 447** `analyzeAndFillOperandForwarding` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:535`, 25 lines
- [ ] **AUDIT 447** `analyzeAndFillOperandForwarding` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:535`, line by line against the C++
- [ ] **PORT 448** `getOperandFromCastOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:354`, 5 lines
- [ ] **AUDIT 448** `getOperandFromCastOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:354`, line by line against the C++

### Conversion/AgenToSentient — 10 defs, 492 lines

- [ ] **PORT 449** `gatherSymbolicLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1051`, 153 lines
- [ ] **AUDIT 449** `gatherSymbolicLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1051`, line by line against the C++
- [ ] **PORT 450** `adjustMutableAddrInitForIndirect` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1447`, 127 lines
- [ ] **AUDIT 450** `adjustMutableAddrInitForIndirect` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1447`, line by line against the C++
- [ ] **PORT 451** `adjustMutableAddrInitForStride` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:4034`, 47 lines
- [ ] **AUDIT 451** `adjustMutableAddrInitForStride` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:4034`, line by line against the C++
- [ ] **PORT 452** `lowerCompositeIndirectLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3267`, 42 lines
- [ ] **AUDIT 452** `lowerCompositeIndirectLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3267`, line by line against the C++
- [ ] **PORT 453** `lowerCompositeIndirectStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3313`, 42 lines
- [ ] **AUDIT 453** `lowerCompositeIndirectStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3313`, line by line against the C++
- [ ] **PORT 454** `lowerCompositeLoadAndStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3148`, 17 lines
- [ ] **AUDIT 454** `lowerCompositeLoadAndStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3148`, line by line against the C++
- [ ] **PORT 455** `lowerCompositeLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3106`, 17 lines
- [ ] **AUDIT 455** `lowerCompositeLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3106`, line by line against the C++
- [ ] **PORT 456** `lowerCompositeIndirectLoadAndStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3359`, 17 lines
- [ ] **AUDIT 456** `lowerCompositeIndirectLoadAndStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3359`, line by line against the C++
- [ ] **PORT 457** `lowerCompositeStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3127`, 17 lines
- [ ] **AUDIT 457** `lowerCompositeStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3127`, line by line against the C++
- [ ] **PORT 458** `insertCopyAndAddStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3861`, 13 lines
- [ ] **AUDIT 458** `insertCopyAndAddStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3861`, line by line against the C++

### Conversion/StandardToSentient — 2 defs, 168 lines

- [ ] **PORT 459** `ConstructIFRecursively` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:113`, 125 lines
- [ ] **AUDIT 459** `ConstructIFRecursively` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:113`, line by line against the C++
- [ ] **PORT 460** `SimplifyOrIOp` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:389`, 43 lines
- [ ] **AUDIT 460** `SimplifyOrIOp` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:389`, line by line against the C++

### Conversion/DataflowToSentient — 1 defs, 80 lines

- [ ] **PORT 461** `lowerSyncOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1901`, 80 lines
- [ ] **AUDIT 461** `lowerSyncOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1901`, line by line against the C++

## Level 7 — 18 definitions, 2238 body lines

### Conversion/VectorChainLowering — 6 defs, 1301 lines

- [ ] **PORT 462** `fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:245`, 628 lines
- [ ] **AUDIT 462** `fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:245`, line by line against the C++
- [ ] **PORT 463** `patternAgnosticFuseNonComputeOpsHelper` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:96`, 222 lines
- [ ] **AUDIT 463** `patternAgnosticFuseNonComputeOpsHelper` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:96`, line by line against the C++
- [ ] **PORT 464** `fuseNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:46`, 191 lines
- [ ] **AUDIT 464** `fuseNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:46`, line by line against the C++
- [ ] **PORT 465** `createXrfIndexModifOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:564`, 97 lines
- [ ] **AUDIT 465** `createXrfIndexModifOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:564`, line by line against the C++
- [ ] **PORT 466** `fillOpInfo` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1070`, 83 lines
- [ ] **AUDIT 466** `fillOpInfo` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1070`, line by line against the C++
- [ ] **PORT 467** `fuseNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1159`, 80 lines
- [ ] **AUDIT 467** `fuseNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1159`, line by line against the C++

### Conversion/AgenToSentient — 4 defs, 632 lines

- [ ] **PORT 468** `generateAffineAddressManipulationStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:625`, 381 lines
- [ ] **AUDIT 468** `generateAffineAddressManipulationStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:625`, line by line against the C++
- [ ] **PORT 469** `constructReceiveAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2025`, 131 lines
- [ ] **AUDIT 469** `constructReceiveAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2025`, line by line against the C++
- [ ] **PORT 470** `constructLoadAndSendStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1910`, 104 lines
- [ ] **AUDIT 470** `constructLoadAndSendStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1910`, line by line against the C++
- [ ] **PORT 471** `constructSymbolicDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2849`, 16 lines
- [ ] **AUDIT 471** `constructSymbolicDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2849`, line by line against the C++

### Transform/Dataflow — 5 defs, 234 lines

- [ ] **PORT 472** `parseConditional` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:534`, 83 lines
- [ ] **AUDIT 472** `parseConditional` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:534`, line by line against the C++
- [ ] **PORT 473** `runOnOperation` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:130`, 69 lines
- [ ] **AUDIT 473** `runOnOperation` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:130`, line by line against the C++
- [ ] **PORT 474** `hoistLoopInvariantConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:750`, 42 lines
- [ ] **AUDIT 474** `hoistLoopInvariantConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:750`, line by line against the C++
- [ ] **PORT 475** `areShallowlyMergeable` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:343`, 34 lines
- [ ] **AUDIT 475** `areShallowlyMergeable` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:343`, line by line against the C++
- [ ] **PORT 476** `run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:647`, 6 lines
- [ ] **AUDIT 476** `run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:647`, line by line against the C++

### Conversion/StandardToSentient — 2 defs, 38 lines

- [ ] **PORT 477** `LowerSelectOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:243`, 19 lines
- [ ] **AUDIT 477** `LowerSelectOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:243`, line by line against the C++
- [ ] **PORT 478** `LowerLogicalOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:264`, 19 lines
- [ ] **AUDIT 478** `LowerLogicalOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:264`, line by line against the C++

### Conversion/DataflowToSentient — 1 defs, 33 lines

- [ ] **PORT 479** `runOnOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2014`, 33 lines
- [ ] **AUDIT 479** `runOnOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2014`, line by line against the C++

## Level 8 — 8 definitions, 292 body lines

### Transform/Dataflow — 2 defs, 106 lines

- [ ] **PORT 480** `shallowlyMergeConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:198`, 76 lines
- [ ] **AUDIT 480** `shallowlyMergeConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:198`, line by line against the C++
- [ ] **PORT 481** `simplifyValueBasedConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:466`, 30 lines
- [ ] **AUDIT 481** `simplifyValueBasedConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:466`, line by line against the C++

### Conversion/VectorChainLowering — 3 defs, 96 lines

- [ ] **PORT 482** `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:975`, 54 lines
- [ ] **AUDIT 482** `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:975`, line by line against the C++
- [ ] **PORT 483** `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1374`, 30 lines
- [ ] **AUDIT 483** `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1374`, line by line against the C++
- [ ] **PORT 484** `matchAndRewrite` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:44`, 12 lines
- [ ] **AUDIT 484** `matchAndRewrite` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:44`, line by line against the C++

### Conversion/AgenToSentient — 2 defs, 56 lines

- [ ] **PORT 485** `lowerSymbolicVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3410`, 30 lines
- [ ] **AUDIT 485** `lowerSymbolicVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3410`, line by line against the C++
- [ ] **PORT 486** `lowerSymbolicVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3380`, 26 lines
- [ ] **AUDIT 486** `lowerSymbolicVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3380`, line by line against the C++

### Conversion/StandardToSentient — 1 defs, 34 lines

- [ ] **PORT 487** `runOnOperation` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:439`, 34 lines
- [ ] **AUDIT 487** `runOnOperation` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:439`, line by line against the C++

## Level 9 — 2 definitions, 224 body lines

### Conversion/AgenToSentient — 1 defs, 144 lines

- [ ] **PORT 488** `fuseLoadOrStoreChainOps` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:22`, 144 lines
- [ ] **AUDIT 488** `fuseLoadOrStoreChainOps` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:22`, line by line against the C++

### Transform/Dataflow — 1 defs, 80 lines

- [ ] **PORT 489** `runOnOperation` — `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:77`, 80 lines
- [ ] **AUDIT 489** `runOnOperation` — `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:77`, line by line against the C++

## Level 10 — 1 definitions, 81 body lines

### Conversion/AgenToSentient — 1 defs, 81 lines

- [ ] **PORT 490** `runOnOperation` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:169`, 81 lines
- [ ] **AUDIT 490** `runOnOperation` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:169`, line by line against the C++

## AUDIT LOG

⛔ **AN AUDIT THAT PASSES EVERYTHING IS NOT AN AUDIT.** Three of the first five failed, all the same
way, and the PORT boxes for 130, 131 and 133 have been UNTICKED.

### ✅ AUDIT 233 `AffineYieldOpLowering::matchAndRewrite` — `AffineToStandard.cpp:41`
| C++ | ours | verdict |
|---|---|---|
| `isa<scf::ParallelOp>(op->getParentOp())` | `Parent::ScfParallel` | ✅ the only question asked of the parent |
| `return failure()` | `YieldRewrite::Declined` | ✅ and not modelled as an error — a declining pattern leaves the module untouched |
| `replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands())` | `scf::Op::Yield { operands }` | ✅ operand list unchanged; `replaceOpWithNewOp` also ERASES the original, and the call site consumes the `affine.yield` rather than emitting it |
| `return success()` | `YieldRewrite::Yielded` | ✅ |

⚠️ **NOTE 1 — the call site passes a CONSTANT.** `statement()` hardcodes `Parent::Other`, so the
`ScfParallel` arm is unreachable from the bridge today. The rule is ported and right; its discriminator
is not yet fed a real value. Becomes live when a caller walks a parallel's body.

⚠️ **NOTE 2 — the C++ pattern reaches NESTED yields and our walk does not.** `OpRewritePattern` is
applied by a driver to every `affine.yield` in the module, including inside `affine.for` bodies;
`statement()` walks only the top level of a unit's body and `affine.for` is still a `todo!`. The loop
entry must recurse and pass its own parent. Bound to that entry, not a defect in 233.

### ✅ AUDIT 316 `setImmutableAddrAndIncrements` — `Helper.cpp:1581`
All four cases match the C++, including the non-L3 burst arm setting **both** fields to `stride_size`
(`:1606-1612`) and the L3's immutable address being assigned above both branches (`:1586-1589`).

⚠️ **NOTE — the C++ creates TWO `ConstantOp`s where we mint ONE.** Each arm builds a separate constant
per field even when the values are equal; we mint one per distinct value into a pool. The reference's
own dumped output has them folded to a single `%1`, so the TEXT agrees — but if a golden ever shows two
distinct SSA values carrying the same constant, the pool is wrong and this is where it comes from.

### ❌ AUDIT 130 `setldtype` — `Helper.cpp:1647` — **FAILED, PORT UNTICKED**
Two of the reference's conditions exist only as PROSE in the doc comment, not as code:
1. **The LX-only gate is missing.** `:1653` returns immediately unless the component is `LXLU`/`LXSU`.
   Nothing in our port asks. An L0 or L3 load would be handed a mode it must never carry.
2. **The explicit zero-pad's stick check is missing.** `:1655-1663` rejects a `zpad16b` whose SHUFFLE
   RESULT is not a whole stick — *"LX loads involving explicit padding should be at stick
   granularity"*. `Explicit::ZeroPad16BOnce` is constructible with no such check.

### ❌ AUDIT 131 `generateSetSendDestinationStmts` — `Helper.cpp:2731` — **FAILED, PORT UNTICKED**
⛔ **THE PORT IS THE GUARD AND NOTHING ELSE.** 49 lines of C++ became a one-line
`sets_send_destination(unit) -> bool`. Absent: collecting the consumer's components from either a
`get_unit` (one) or a `uniform.QueryMapOp` via `getAllQueriedValues` (several, with
*"vector_loadOp's consumer is not a getUnitOp"* for anything else); the destination test
`any of {PT, SFP, L0SU, CROSSPTNLINK}`; the emission of `sentient.set_send_dst(consumer->getResult(0))`;
and the arch split — at `>= RCUDD1A` it emits, below that a consumer in {PT, L0SU, CROSSPTNLINK} is the
error *"requires generating SETDSTMASK instruction which is not supported at the current arch level"*.

### ❌ AUDIT 133 `getLoadConsumer` — `Helper.cpp:1242` — **FAILED, PORT UNTICKED**
⛔ **A DATA HOLDER, NOT THE FUNCTION.** `load_consumer(to_unit, via)` packages two values the caller
must already have. The C++ FINDS them: it picks the root to follow (the result for a `vector_load`,
indirect or symbolic load; the **load induction variable** for a composite load), requires
`hasOneUse()`, then matches the user against three shapes and reads the send's `getToUnit()`. None of
that is ported.
