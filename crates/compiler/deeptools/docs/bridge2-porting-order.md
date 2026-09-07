# Bridge 2 — the port list, winnowed to what scratchy's architecture needs

`DataflowIR -> SentientIR`, the D1-D28 span. **384 functions to port**, out of 490 definitions in the
span; **106 are excluded** and listed at the bottom with the reason for each.

## ⛔ WHAT THE EXCLUSION CRITERION IS, AND WHAT IT IS NOT

⭐ **EXCLUDED = SCRATCHY'S ARCHITECTURE WILL NEVER NEED IT**, not *we do not support it yet*. Three
kinds, and each is a fact about our design rather than a judgement about the function:

1. **C++ field accessors and data members (81).** `unsigned getElementWidth() { return
   element_width_; }` is a struct field in Rust. There is no function to port.
2. **MLIR pass machinery (19).** `createXPass`, `populateXConversionPatterns`, `getDependentDialects`,
   the pass context, pattern-driver configuration. `#[forward]` runs the whole pipeline at macro
   expansion — there is no pass manager, no pattern driver and no rewriter at run time.
3. **MLIR printing, parsing and verification (5).** This crate EMITS and never reads back; an island
   is emit-only and no MLIR text reaches it.

⛔⛔ **NOTHING IS EXCLUDED FOR BEING HARD, UNFINISHED, OR LOOP-DEPENDENT.** The 50 loop-sensitive
functions (22+14+14 across levels 0-2) are **DEFERRED, NOT EXCLUDED** — they stay on this list and
are ported when tiling lands and the corpus is regenerated.

⛔ **AND EVERY EXCLUSION IS LISTED, SO ANY ONE CAN BE OVERRULED.** See the bottom section.

## ⛔ THE RULES OF EXECUTION

- ⛔ **PORT means the WHOLE function, INCLUDING its emission.** A predicate is not a port. Levels 0-2
  were once reported complete when what existed was one documented predicate per function with every
  emission missing and nothing calling any of it.
- ⛔ **If our IR cannot express a function's input, ADD THE OP TO THE ISLAND.** Deciding a function is
  unnecessary is not the porter's judgement to make — that decision is this document's, above.
- ⛔ **AUDIT means line by line against the C++**: every branch, constant, attribute name and value,
  default, early return, and the order of emission.
- ⛔ **NEVER PORT ANYTHING NEW WHILE AN AUDIT IS OUTSTANDING.**
- ⛔ Unit tests come with the port; the vendor's own case where one exists (668 of `dcc/test`'s 825
  `.mlir` tests carry `CHECK-SENT-IR` expectations).
- ⛔ `cargo build -Fsuperdsc,model/granite-3.1-2b-instruct,quant/fp8-dynamic-per-channel` is the
  acceptance gate. E2E when bridge 1 lands.

## Progress

`57/384 ported; 57/384 audited`

Ported and audited: `AffineYieldOpLowering::matchAndRewrite` (`lower_affine_yield`, entry 001, in
`src/bridges/dataflow_ir_to_sentient/std_affine_to_standard.rs`), `setImmutableAddrAndIncrements`
(entry 214), and entries 002-024 — `setCoalescedBoundValues`, the six `AccessDetailsBase` setters
that establish its access state and the seven that establish its transfer state, the
`AccessDetailsAffine` constructor and its two setters (`setSubscriptsMap`, `setIndicesCoeffDict`),
and `AccessDetailsAffineComposite`'s constructor plus its time setters (`setTimeAddrMap`,
`setTimeSymbols`, `setTimeBounds`, `setTimeOffsets`, `setInterleaveGroupIndex`), all in
`src/bridges/dataflow_ir_to_sentient/agen_access_details.rs`, and entries 025-032 — the INDIRECT
(extract) pattern: `setStrides` and `has` (`agen_access_details.rs`), the two
`construct*Stmt` overloads and `insertCopyAndAddStmtsHelper` (`agen_agen_to_sentient.rs`), and
`getLoopNestLevel`, `checkIndirectMemViewForExtractOp` and `findExtractScalarOp`
(`agen_helper.rs`), and entries 033-040 — the AgenToSentient load-consumer chain
(`getLoadConsumer`, `setldtype`, `generateSetSendDestinationStmts`,
`getStoreOpFromLoadStorePattern`, `findCandidateForLowering`, `addLoadChainToDeleteList`, all in
`agen_helper.rs`) plus `isSenComponentL0LU`/`isSenComponentL0SU`
(`dfs_dataflow_to_sentient.rs`). And entries 081-088 — the Loop Mask Tree's node layer:
`LoopMaskNode`'s five delegating accessors (`getParentNode`, `getFirstChild`, `getNextSibling`,
`getPrevSibling`, `getLastChild`), `LoopMaskTree::getRoot`, and the two empty destructors
`~MaskNode`/`~LMTLoopNode` as a build-time `needs_drop` guard, in
`src/bridges/dataflow_ir_to_sentient/vc_loop_mask_tree.rs` — where the intrusive tree became an
arena, so the unchecked `static_cast` to the derived node is minting a `LoopMaskNodeId`, and
`mlir::OperationNode`'s own storage and walks sit beside them unanchored, awaiting entries
079/080/281 and 101-107.

⭐ AND ENTRIES 049-056 — the six `StandardToSentient` scalar lowerings (`LowerAddIOpToSentient`,
`LowerSubIOpToSentient`, `LowerMulIOpToSentient`, the `If` shape law, `LowerConstantIndexToSentient`,
`LowerConstantIntToSentient`) in `src/bridges/dataflow_ir_to_sentient/std_standard_to_sentient.rs`,
and `OperandReuse`'s two getters (`getId`, `getAbsorbtionFlag`) in `vc_operand_reuse.rs`.

⭐ 001/384 MOVED TO ITS OWN TRANSLATION UNIT'S HOME: `lower_affine_yield` was living in
`dataflow_ir_to_sentient/mod.rs`, and `crustify/crates.json` homes `e001_matchAndRewrite` in
`std_affine_to_standard.rs`. It carries its `/// Replaces:` anchor there and `mod.rs` imports it.

⭐ ONE VOCABULARY, NOT TWO: 002's port and 022's arrived with their own `TimeDim` and `TimeBound`.
022's landed first and is the one kept — its `TimeDim` is a `u32` with an `index()` accessor and its
`kInvalid` variant is spelled `Variable`. `set_coalesced_bound_values` was rebased onto it rather
than duplicating the pair, so `MemoryOperandIndex` and `LayoutCoeff` are all 002-008 adds to the
shared vocabulary block.

⚠️ THE AUDIT OF 002-008 CORRECTED ITS OWN CITATIONS, and one of 001's. Six field references pointed at
the doc comment above the declaration rather than the declaration; `kMax`'s container use is
`AccessDetails.hpp:375` not `:377`; `coalesced_bound *=` is `AccessDetails.cpp:777` not `:772`; the
time-loop trip rule is `Helper.cpp:1815-1820`. And 001's note credited the decline to an
`AffineParallelLowering` pattern — **dcc's copy of the pass registers no such pattern**
(`AffineToStandard.cpp:198-206`); the reference's comment is upstream MLIR's, and dcc leaves the
terminator alone because `scf` is already legal in its target.

⛔ **052 IS A COMMENT, NOT A FUNCTION** — `StandardToSentient.cpp:159` is the line
`// return If(lhs) {If(rhs) true_val; else false_val} else false_val;` inside
`ConstructIFRecursively`'s `and` branch, which the extractor read as a 0-line function called `If`.
What it states is the SHAPE a conjunction lowers to, and that is what `NestedIf` holds; the recursion
around it stays entry 338's. ⛔ The comment names `lhs` as the outer `if` while the code returns
`rhs_if_op` (`:175`) — the emitted nest has the RIGHT-hand conjunct outermost.

⭐ **THE THREE THAT WERE UNTICKED BY THEIR OWN AUDITS ARE NOW RE-PORTED WITH THEIR EMISSION.**
`getLoadConsumer`, `setldtype` and `generateSetSendDestinationStmts` had been reduced to documented
predicates; `generate_set_send_destination_stmts` now builds the `sentient.set_send_dst` and a test
checks the three the vendor's own golden expects, verbatim
(`dcc/test/Conversion/AgenToSentient/lx-to-sfp-bypass-1.mlir:47,55,63`).

⭐ **WHAT ENTRIES 033-040 NEEDED FROM THE ISLANDS, added rather than worked around.**
`islands/dataflow_ir/dialects/mod.rs` gained the operand/result/region/use census — `uses()` counts
one entry per USE and descends into regions, which is what `hasOneUse()` means and what a previous
attempt's shape-matching faked. `GenericComp` grew from 6 variants to the ones that are
`senCompToGenericComp`'s true image, because the port could not tell `l0lu` from `l0su` while they
folded onto one `L0`. `DfirUnit` gained `CrossPtnLink` (the fourth member of the SFP-bypass set) and
`vectorchain::Op::Rotate` (the third rearrangement the two chain functions name). `link.rs` gained
`PtRowUnit<ROW>`, `L0su`, `L0lu` and `CrossPtnLink` markers, because `dataflow.send %pt, %9` — a line
in the reference's own test input — was not constructible.

⛔ **THREE DISCREPANCIES THE 033-040 AUDITS FOUND IN THE REFERENCE**, each recorded beside the port
that carries it: `getLoadConsumer`'s comment promises a `storeOp` arm the code does not have (so a
stored load hits `emitError("unsupported loadOp consumer!")`); `generateSetSendDestinationStmts`
binds `map_op` and never reads it; `addLoadChainToDeleteList` dereferences
`*user->getUsers().begin()` on a rearrangement whose result nothing reads.


## Level 0

- [x] **PORT 001/384** `matchAndRewrite` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:41`, 8 lines
- [x] **AUDIT 001/384** `matchAndRewrite` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:41`, line by line against the C++
- [x] **PORT 002/384** `setCoalescedBoundValues` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:672`, 5 lines
- [x] **AUDIT 002/384** `setCoalescedBoundValues` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:672`, line by line against the C++
- [x] **PORT 003/384** `setIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:77`, 2 lines
- [x] **AUDIT 003/384** `setIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:77`, line by line against the C++
- [x] **PORT 004/384** `setMemViewStartAddr` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:81`, 2 lines
- [x] **AUDIT 004/384** `setMemViewStartAddr` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:81`, line by line against the C++
- [x] **PORT 005/384** `setMemoryIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:93`, 2 lines
- [x] **AUDIT 005/384** `setMemoryIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:93`, line by line against the C++
- [x] **PORT 006/384** `setLayoutCoeffs` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:96`, 2 lines
- [x] **AUDIT 006/384** `setLayoutCoeffs` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:96`, line by line against the C++
- [x] **PORT 007/384** `setMemViewLayoutMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:99`, 2 lines
- [x] **AUDIT 007/384** `setMemViewLayoutMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:99`, line by line against the C++
- [x] **PORT 008/384** `setShuffleMode` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:104`, 2 lines
- [x] **AUDIT 008/384** `setShuffleMode` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:104`, line by line against the C++
- [x] **PORT 009/384** `setRotationPosition` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:107`, 2 lines
- [x] **AUDIT 009/384** `setRotationPosition` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:107`, line by line against the C++
- [x] **PORT 010/384** `setExpectedTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:110`, 2 lines
- [x] **AUDIT 010/384** `setExpectedTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:110`, line by line against the C++
- [x] **PORT 011/384** `setExtents` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:113`, 2 lines
- [x] **AUDIT 011/384** `setExtents` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:113`, line by line against the C++
- [x] **PORT 012/384** `setTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:116`, 2 lines
- [x] **AUDIT 012/384** `setTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:116`, line by line against the C++
- [x] **PORT 013/384** `setElementWidth` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:119`, 2 lines
- [x] **AUDIT 013/384** `setElementWidth` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:119`, line by line against the C++
- [x] **PORT 014/384** `setTransferSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:122`, 2 lines
- [x] **AUDIT 014/384** `setTransferSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:122`, line by line against the C++
- [x] **PORT 015/384** `setTransferOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:125`, 2 lines
- [x] **AUDIT 015/384** `setTransferOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:125`, line by line against the C++
- [x] **PORT 016/384** `AccessDetailsBase` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:218`, 0 lines
- [x] **AUDIT 016/384** `AccessDetailsBase` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:218`, line by line against the C++
- [x] **PORT 017/384** `setSubscriptsMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:228`, 2 lines
- [x] **AUDIT 017/384** `setSubscriptsMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:228`, line by line against the C++
- [x] **PORT 018/384** `setIndicesCoeffDict` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:231`, 2 lines
- [x] **AUDIT 018/384** `setIndicesCoeffDict` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:231`, line by line against the C++
- [x] **PORT 019/384** `AccessDetailsAffine` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:259`, 0 lines
- [x] **AUDIT 019/384** `AccessDetailsAffine` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:259`, line by line against the C++
- [x] **PORT 020/384** `setTimeAddrMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:286`, 2 lines
- [x] **AUDIT 020/384** `setTimeAddrMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:286`, line by line against the C++
- [x] **PORT 021/384** `setTimeSymbols` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:289`, 2 lines
- [x] **AUDIT 021/384** `setTimeSymbols` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:289`, line by line against the C++
- [x] **PORT 022/384** `setTimeBounds` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:292`, 2 lines
- [x] **AUDIT 022/384** `setTimeBounds` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:292`, line by line against the C++
- [x] **PORT 023/384** `setTimeOffsets` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:295`, 2 lines
- [x] **AUDIT 023/384** `setTimeOffsets` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:295`, line by line against the C++
- [x] **PORT 024/384** `setInterleaveGroupIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:299`, 2 lines
- [x] **AUDIT 024/384** `setInterleaveGroupIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:299`, line by line against the C++
- [x] **PORT 025/384** `setStrides` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:355`, 2 lines
- [x] **AUDIT 025/384** `setStrides` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:355`, line by line against the C++
- [x] **PORT 026/384** `has` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:397`, 2 lines
- [x] **AUDIT 026/384** `has` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:397`, line by line against the C++
- [x] **PORT 027/384** `constructLoadAndSendStmt` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:230`, 4 lines
- [x] **AUDIT 027/384** `constructLoadAndSendStmt` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:230`, line by line against the C++
- [x] **PORT 028/384** `constructReceiveAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:248`, 4 lines
- [x] **AUDIT 028/384** `constructReceiveAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:248`, line by line against the C++
- [x] **PORT 029/384** `insertCopyAndAddStmtsHelper` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:502`, 17 lines
- [x] **AUDIT 029/384** `insertCopyAndAddStmtsHelper` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:502`, line by line against the C++
- [x] **PORT 030/384** `getLoopNestLevel` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:43`, 6 lines
- [x] **AUDIT 030/384** `getLoopNestLevel` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:43`, line by line against the C++
- [x] **PORT 031/384** `checkIndirectMemViewForExtractOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:388`, 42 lines
- [x] **AUDIT 031/384** `checkIndirectMemViewForExtractOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:388`, line by line against the C++
- [x] **PORT 032/384** `findExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:514`, 20 lines
- [x] **AUDIT 032/384** `findExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:514`, line by line against the C++
- [x] **PORT 033/384** `getLoadConsumer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1242`, 36 lines
- [x] **AUDIT 033/384** `getLoadConsumer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1242`, line by line against the C++
- [x] **PORT 034/384** `setldtype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1647`, 60 lines
- [x] **AUDIT 034/384** `setldtype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1647`, line by line against the C++
- [x] **PORT 035/384** `generateSetSendDestinationStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2731`, 49 lines
- [x] **AUDIT 035/384** `generateSetSendDestinationStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2731`, line by line against the C++
- [x] **PORT 036/384** `getStoreOpFromLoadStorePattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2872`, 8 lines
- [x] **AUDIT 036/384** `getStoreOpFromLoadStorePattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2872`, line by line against the C++
- [x] **PORT 037/384** `findCandidateForLowering` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2884`, 12 lines
- [x] **AUDIT 037/384** `findCandidateForLowering` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2884`, line by line against the C++
- [x] **PORT 038/384** `addLoadChainToDeleteList` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2975`, 9 lines
- [x] **AUDIT 038/384** `addLoadChainToDeleteList` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2975`, line by line against the C++
- [x] **PORT 039/384** `isSenComponentL0LU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:96`, 2 lines
- [x] **AUDIT 039/384** `isSenComponentL0LU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:96`, line by line against the C++
- [x] **PORT 040/384** `isSenComponentL0SU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:100`, 2 lines
- [x] **AUDIT 040/384** `isSenComponentL0SU` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:100`, line by line against the C++
- [ ] **PORT 041/384** `ExtendUnitNameToCorelet` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:104`, 11 lines
- [ ] **AUDIT 041/384** `ExtendUnitNameToCorelet` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:104`, line by line against the C++
- [ ] **PORT 042/384** `isSameListOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:175`, 10 lines
- [ ] **AUDIT 042/384** `isSameListOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:175`, line by line against the C++
- [ ] **PORT 043/384** `isTargetL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1720`, 6 lines
- [ ] **AUDIT 043/384** `isTargetL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1720`, line by line against the C++
- [ ] **PORT 044/384** `lowerOpaqueOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1984`, 27 lines
- [ ] **AUDIT 044/384** `lowerOpaqueOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1984`, line by line against the C++
- [ ] **PORT 045/384** `getSentientCmpIPredicate` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:33`, 16 lines
- [ ] **AUDIT 045/384** `getSentientCmpIPredicate` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:33`, line by line against the C++
- [ ] **PORT 046/384** `ConversionPattern` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:70`, 0 lines
- [ ] **AUDIT 046/384** `ConversionPattern` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:70`, line by line against the C++
- [ ] **PORT 047/384** `runOnOperation` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:251`, 30 lines
- [ ] **AUDIT 047/384** `runOnOperation` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:251`, line by line against the C++
- [ ] **PORT 048/384** `getSentientCmpIPredicate` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:36`, 18 lines
- [ ] **AUDIT 048/384** `getSentientCmpIPredicate` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:36`, line by line against the C++
- [x] **PORT 049/384** `LowerAddIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:79`, 9 lines
- [x] **AUDIT 049/384** `LowerAddIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:79`, line by line against the C++
- [x] **PORT 050/384** `LowerSubIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:90`, 10 lines
- [x] **AUDIT 050/384** `LowerSubIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:90`, line by line against the C++
- [x] **PORT 051/384** `LowerMulIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:102`, 9 lines
- [x] **AUDIT 051/384** `LowerMulIOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:102`, line by line against the C++
- [x] **PORT 052/384** `If` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:159`, 0 lines
- [x] **AUDIT 052/384** `If` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:159`, line by line against the C++
- [x] **PORT 053/384** `LowerConstantIndexToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:347`, 8 lines
- [x] **AUDIT 053/384** `LowerConstantIndexToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:347`, line by line against the C++
- [x] **PORT 054/384** `LowerConstantIntToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:358`, 15 lines
- [x] **AUDIT 054/384** `LowerConstantIntToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:358`, line by line against the C++
- [x] **PORT 055/384** `getId` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:65`, 6 lines
- [x] **AUDIT 055/384** `getId` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:65`, line by line against the C++
- [x] **PORT 056/384** `getAbsorbtionFlag` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:73`, 6 lines
- [x] **AUDIT 056/384** `getAbsorbtionFlag` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:73`, line by line against the C++
- [ ] **PORT 057/384** `setReuseFlag` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:91`, 2 lines
- [ ] **AUDIT 057/384** `setReuseFlag` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:91`, line by line against the C++
- [ ] **PORT 058/384** `dominates` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:38`, 2 lines
- [ ] **AUDIT 058/384** `dominates` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:38`, line by line against the C++
- [ ] **PORT 059/384** `isSentientBinaryLogicalOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:30`, 4 lines
- [ ] **AUDIT 059/384** `isSentientBinaryLogicalOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:30`, line by line against the C++
- [ ] **PORT 060/384** `getInputPrecisionFromOperand` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:36`, 6 lines
- [ ] **AUDIT 060/384** `getInputPrecisionFromOperand` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:36`, line by line against the C++
- [ ] **PORT 061/384** `getResultPrecisionFromOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:52`, 10 lines
- [ ] **AUDIT 061/384** `getResultPrecisionFromOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:52`, line by line against the C++
- [ ] **PORT 062/384** `getComputePrecisionOfOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:65`, 13 lines
- [ ] **AUDIT 062/384** `getComputePrecisionOfOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:65`, line by line against the C++
- [ ] **PORT 063/384** `hasConstantBounds` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:80`, 10 lines
- [ ] **AUDIT 063/384** `hasConstantBounds` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:80`, line by line against the C++
- [ ] **PORT 064/384** `size` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:319`, 6 lines
- [ ] **AUDIT 064/384** `size` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:319`, line by line against the C++
- [ ] **PORT 065/384** `fuseCompareAndSelectIntoMinOrMax` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:415`, 49 lines
- [ ] **AUDIT 065/384** `fuseCompareAndSelectIntoMinOrMax` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:415`, line by line against the C++
- [ ] **PORT 066/384** `resetSentientFMAsIfExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:468`, 13 lines
- [ ] **AUDIT 066/384** `resetSentientFMAsIfExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:468`, line by line against the C++
- [ ] **PORT 067/384** `redefineConstantVectors` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:571`, 37 lines
- [ ] **AUDIT 067/384** `redefineConstantVectors` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:571`, line by line against the C++
- [ ] **PORT 068/384** `getVectorBinaryToSentientBinary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:32`, 17 lines
- [ ] **AUDIT 068/384** `getVectorBinaryToSentientBinary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:32`, line by line against the C++
- [ ] **PORT 069/384** `getVectorElementWiseCompareOperatorToSentientBinaryOperator` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:55`, 9 lines
- [ ] **AUDIT 069/384** `getVectorElementWiseCompareOperatorToSentientBinaryOperator` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:55`, line by line against the C++
- [ ] **PORT 070/384** `getVectorTernaryToSentientTernary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:247`, 7 lines
- [ ] **AUDIT 070/384** `getVectorTernaryToSentientTernary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:247`, line by line against the C++
- [ ] **PORT 071/384** `getOperandFromReceiveOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:34`, 54 lines
- [ ] **AUDIT 071/384** `getOperandFromReceiveOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:34`, line by line against the C++
- [ ] **PORT 072/384** `getOperandFromSendOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:95`, 50 lines
- [ ] **AUDIT 072/384** `getOperandFromSendOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:95`, line by line against the C++
- [ ] **PORT 073/384** `constValToField` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:250`, 12 lines
- [ ] **AUDIT 073/384** `constValToField` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:250`, line by line against the C++
- [ ] **PORT 074/384** `sameBlock` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:652`, 11 lines
- [ ] **AUDIT 074/384** `sameBlock` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:652`, line by line against the C++
- [ ] **PORT 075/384** `eraseOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:806`, 16 lines
- [ ] **AUDIT 075/384** `eraseOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:806`, line by line against the C++
- [ ] **PORT 076/384** `fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1243`, 26 lines
- [ ] **AUDIT 076/384** `fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1243`, line by line against the C++
- [ ] **PORT 077/384** `walk` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:132`, 2 lines
- [ ] **AUDIT 077/384** `walk` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:132`, line by line against the C++
- [ ] **PORT 078/384** `findNodeFromOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:167`, 4 lines
- [ ] **AUDIT 078/384** `findNodeFromOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:167`, line by line against the C++
- [ ] **PORT 079/384** `OperationNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:32`, 0 lines
- [ ] **AUDIT 079/384** `OperationNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:32`, line by line against the C++
- [ ] **PORT 080/384** `LoopMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:34`, 0 lines
- [ ] **AUDIT 080/384** `LoopMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:34`, line by line against the C++
- [x] **PORT 081/384** `getParentNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:36`, 2 lines
- [x] **AUDIT 081/384** `getParentNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:36`, line by line against the C++
- [x] **PORT 082/384** `getFirstChild` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:39`, 2 lines
- [x] **AUDIT 082/384** `getFirstChild` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:39`, line by line against the C++
- [x] **PORT 083/384** `getNextSibling` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:42`, 2 lines
- [x] **AUDIT 083/384** `getNextSibling` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:42`, line by line against the C++
- [x] **PORT 084/384** `getPrevSibling` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:45`, 2 lines
- [x] **AUDIT 084/384** `getPrevSibling` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:45`, line by line against the C++
- [x] **PORT 085/384** `getLastChild` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:48`, 2 lines
- [x] **AUDIT 085/384** `getLastChild` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:48`, line by line against the C++
- [x] **PORT 086/384** `MaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:74`, 0 lines
- [x] **AUDIT 086/384** `MaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:74`, line by line against the C++
- [x] **PORT 087/384** `LMTLoopNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:94`, 0 lines
- [x] **AUDIT 087/384** `LMTLoopNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:94`, line by line against the C++
- [x] **PORT 088/384** `getRoot` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:109`, 2 lines
- [x] **AUDIT 088/384** `getRoot` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:109`, line by line against the C++
- [ ] **PORT 089/384** `getMaskValueForPT` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:17`, 196 lines
- [ ] **AUDIT 089/384** `getMaskValueForPT` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:17`, line by line against the C++
- [ ] **PORT 090/384** `getXrfValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:248`, 11 lines
- [ ] **AUDIT 090/384** `getXrfValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:248`, line by line against the C++
- [ ] **PORT 091/384** `getForOpBound` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:262`, 31 lines
- [ ] **AUDIT 091/384** `getForOpBound` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:262`, line by line against the C++
- [ ] **PORT 092/384** `setSentientMacXrfRegIncrAttr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:665`, 11 lines
- [ ] **AUDIT 092/384** `setSentientMacXrfRegIncrAttr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:665`, line by line against the C++
- [ ] **PORT 093/384** `replaceAndEraseDummyMacOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:681`, 7 lines
- [ ] **AUDIT 093/384** `replaceAndEraseDummyMacOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:681`, line by line against the C++
- [ ] **PORT 094/384** `computeUnitPrecision` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:31`, 9 lines
- [ ] **AUDIT 094/384** `computeUnitPrecision` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:31`, line by line against the C++
- [ ] **PORT 095/384** `isOperationSelected` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:34`, 2 lines
- [ ] **AUDIT 095/384** `isOperationSelected` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:34`, line by line against the C++
- [ ] **PORT 096/384** `createDummyYieldInElseReg` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:383`, 13 lines
- [ ] **AUDIT 096/384** `createDummyYieldInElseReg` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:383`, line by line against the C++
- [ ] **PORT 097/384** `getNewDbgNameFromList` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:456`, 2 lines
- [ ] **AUDIT 097/384** `getNewDbgNameFromList` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:456`, line by line against the C++
- [ ] **PORT 098/384** `getLhsRhsOfEQPredicate` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:519`, 11 lines
- [ ] **AUDIT 098/384** `getLhsRhsOfEQPredicate` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:519`, line by line against the C++
- [ ] **PORT 099/384** `ConditionalSimplificationManager` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:78`, 2 lines
- [ ] **AUDIT 099/384** `ConditionalSimplificationManager` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:78`, line by line against the C++
- [ ] **PORT 100/384** `TransformationConditionalTree` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:111`, 5 lines
- [ ] **AUDIT 100/384** `TransformationConditionalTree` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:111`, line by line against the C++
- [ ] **PORT 101/384** `OperationNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:51`, 0 lines
- [ ] **AUDIT 101/384** `OperationNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:51`, line by line against the C++
- [ ] **PORT 102/384** `getParentNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:53`, 2 lines
- [ ] **AUDIT 102/384** `getParentNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:53`, line by line against the C++
- [ ] **PORT 103/384** `getFirstChild` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:56`, 2 lines
- [ ] **AUDIT 103/384** `getFirstChild` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:56`, line by line against the C++
- [ ] **PORT 104/384** `getNextSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:59`, 2 lines
- [ ] **AUDIT 104/384** `getNextSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:59`, line by line against the C++
- [ ] **PORT 105/384** `getPrevSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:62`, 2 lines
- [ ] **AUDIT 105/384** `getPrevSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:62`, line by line against the C++
- [ ] **PORT 106/384** `OperationTreeBase` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:78`, 0 lines
- [ ] **AUDIT 106/384** `OperationTreeBase` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:78`, line by line against the C++
- [ ] **PORT 107/384** `getRoot` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:81`, 2 lines
- [ ] **AUDIT 107/384** `getRoot` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:81`, line by line against the C++
- [ ] **PORT 108/384** `partitionUnits` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:168`, 17 lines
- [ ] **AUDIT 108/384** `partitionUnits` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:168`, line by line against the C++
- [ ] **PORT 109/384** `performFullUnroll` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:141`, 7 lines
- [ ] **AUDIT 109/384** `performFullUnroll` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:141`, line by line against the C++
- [ ] **PORT 110/384** `getConstantTripCount` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:167`, 14 lines
- [ ] **AUDIT 110/384** `getConstantTripCount` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:167`, line by line against the C++
- [ ] **PORT 111/384** `getMaxMutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:673`, 8 lines
- [ ] **AUDIT 111/384** `getMaxMutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:673`, line by line against the C++
- [ ] **PORT 112/384** `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:683`, 8 lines
- [ ] **AUDIT 112/384** `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:683`, line by line against the C++
- [ ] **PORT 113/384** `isEligibleForSplitting` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:832`, 20 lines
- [ ] **AUDIT 113/384** `isEligibleForSplitting` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:832`, line by line against the C++
- [ ] **PORT 114/384** `sortDataBasedOnWeight` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:855`, 4 lines
- [ ] **AUDIT 114/384** `sortDataBasedOnWeight` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:855`, line by line against the C++
- [ ] **PORT 115/384** `createNewMemViewWithMod` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:966`, 12 lines
- [ ] **AUDIT 115/384** `createNewMemViewWithMod` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:966`, line by line against the C++
- [ ] **PORT 116/384** `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:355`, 8 lines
- [ ] **AUDIT 116/384** `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:355`, line by line against the C++
- [ ] **PORT 117/384** `transformSCFToAffineLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:102`, 44 lines
- [ ] **AUDIT 117/384** `transformSCFToAffineLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:102`, line by line against the C++
- [ ] **PORT 118/384** `removeValuesFromIndices` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:36`, 7 lines
- [ ] **AUDIT 118/384** `removeValuesFromIndices` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:36`, line by line against the C++
- [ ] **PORT 119/384** `replaceDimsInMapWithSyms` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:47`, 7 lines
- [ ] **AUDIT 119/384** `replaceDimsInMapWithSyms` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:47`, line by line against the C++
- [ ] **PORT 120/384** `createEqualityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:253`, 7 lines
- [ ] **AUDIT 120/384** `createEqualityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:253`, line by line against the C++
- [ ] **PORT 121/384** `createInequalityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:263`, 14 lines
- [ ] **AUDIT 121/384** `createInequalityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:263`, line by line against the C++
- [ ] **PORT 122/384** `setBuilderToInsertRef` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:280`, 6 lines
- [ ] **AUDIT 122/384** `setBuilderToInsertRef` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:280`, line by line against the C++
- [ ] **PORT 123/384** `calculateStartElementsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:532`, 10 lines
- [ ] **AUDIT 123/384** `calculateStartElementsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:532`, line by line against the C++
- [ ] **PORT 124/384** `createNonPagedMemView` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:545`, 10 lines
- [ ] **AUDIT 124/384** `createNonPagedMemView` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:545`, line by line against the C++
- [ ] **PORT 125/384** `cloneMemViewIfNonPaged` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:633`, 9 lines
- [ ] **AUDIT 125/384** `cloneMemViewIfNonPaged` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:633`, line by line against the C++
- [ ] **PORT 126/384** `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:673`, 3 lines
- [ ] **AUDIT 126/384** `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:673`, line by line against the C++
- [ ] **PORT 127/384** `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:678`, 3 lines
- [ ] **AUDIT 127/384** `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:678`, line by line against the C++
- [ ] **PORT 128/384** `createNewMemOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:684`, 9 lines
- [ ] **AUDIT 128/384** `createNewMemOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:684`, line by line against the C++
- [ ] **PORT 129/384** `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:698`, 3 lines
- [ ] **AUDIT 129/384** `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:698`, line by line against the C++
- [ ] **PORT 130/384** `getStoreOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:843`, 6 lines
- [ ] **AUDIT 130/384** `getStoreOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:843`, line by line against the C++
- [ ] **PORT 131/384** `addTimeDimIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:876`, 5 lines
- [ ] **AUDIT 131/384** `addTimeDimIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:876`, line by line against the C++
- [ ] **PORT 132/384** `identifyTimeDimForExplicitLoops` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:963`, 10 lines
- [ ] **AUDIT 132/384** `identifyTimeDimForExplicitLoops` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:963`, line by line against the C++
- [ ] **PORT 133/384** `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:328`, 2 lines
- [ ] **AUDIT 133/384** `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:328`, line by line against the C++
- [ ] **PORT 134/384** `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:341`, 0 lines
- [ ] **AUDIT 134/384** `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:341`, line by line against the C++
- [ ] **PORT 135/384** `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:364`, 0 lines
- [ ] **AUDIT 135/384** `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:364`, line by line against the C++
- [ ] **PORT 136/384** `TPMVBase` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:389`, 0 lines
- [ ] **AUDIT 136/384** `TPMVBase` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:389`, line by line against the C++
- [ ] **PORT 137/384** `TPMVVector` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:397`, 0 lines
- [ ] **AUDIT 137/384** `TPMVVector` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:397`, line by line against the C++
- [ ] **PORT 138/384** `TPMVComposite` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:519`, 0 lines
- [ ] **AUDIT 138/384** `TPMVComposite` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:519`, line by line against the C++
- [ ] **PORT 139/384** `run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.cpp:21`, 48 lines
- [ ] **AUDIT 139/384** `run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.cpp:21`, line by line against the C++
- [ ] **PORT 140/384** `removeCoresCoreletsFoldsFromProgramUnit` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:240`, 20 lines
- [ ] **AUDIT 140/384** `removeCoresCoreletsFoldsFromProgramUnit` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:240`, line by line against the C++
- [ ] **PORT 141/384** `isDataTransfer` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:314`, 11 lines
- [ ] **AUDIT 141/384** `isDataTransfer` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:314`, line by line against the C++
- [ ] **PORT 142/384** `getDataflowForLoopInfoIfIV` — `dcc/src/Transform/Dataflow/Utils.cpp:99`, 31 lines
- [ ] **AUDIT 142/384** `getDataflowForLoopInfoIfIV` — `dcc/src/Transform/Dataflow/Utils.cpp:99`, line by line against the C++

## Level 1

- [ ] **PORT 143/384** `constructExtentAndTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:32`, 64 lines
- [ ] **AUDIT 143/384** `constructExtentAndTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:32`, line by line against the C++
- [ ] **PORT 144/384** `constructLdOrStType` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:267`, 5 lines
- [ ] **AUDIT 144/384** `constructLdOrStType` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:267`, line by line against the C++
- [ ] **PORT 145/384** `initializeMemViewInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:274`, 16 lines
- [ ] **AUDIT 145/384** `initializeMemViewInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:274`, line by line against the C++
- [ ] **PORT 146/384** `constructIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:354`, 50 lines
- [ ] **AUDIT 146/384** `constructIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:354`, line by line against the C++
- [ ] **PORT 147/384** `constructIteratorCoefficients` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:406`, 10 lines
- [ ] **AUDIT 147/384** `constructIteratorCoefficients` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:406`, line by line against the C++
- [ ] **PORT 148/384** `computeBurstAndGroup` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:796`, 36 lines
- [ ] **AUDIT 148/384** `computeBurstAndGroup` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:796`, line by line against the C++
- [ ] **PORT 149/384** `insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:388`, 7 lines
- [ ] **AUDIT 149/384** `insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:388`, line by line against the C++
- [ ] **PORT 150/384** `get` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:401`, 4 lines
- [ ] **AUDIT 150/384** `get` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:401`, line by line against the C++
- [ ] **PORT 151/384** `checkCompositeRegion` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:206`, 89 lines
- [ ] **AUDIT 151/384** `checkCompositeRegion` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:206`, line by line against the C++
- [ ] **PORT 152/384** `checkStoreOpFromExtractPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:433`, 27 lines
- [ ] **AUDIT 152/384** `checkStoreOpFromExtractPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:433`, line by line against the C++
- [ ] **PORT 153/384** `isLoadAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:463`, 27 lines
- [ ] **AUDIT 153/384** `isLoadAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:463`, line by line against the C++
- [ ] **PORT 154/384** `isReceiveAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:493`, 17 lines
- [ ] **AUDIT 154/384** `isReceiveAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:493`, line by line against the C++
- [ ] **PORT 155/384** `updateSymbolicAccessDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1013`, 35 lines
- [ ] **AUDIT 155/384** `updateSymbolicAccessDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1013`, line by line against the C++
- [ ] **PORT 156/384** `getStoreProducer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1285`, 159 lines
- [ ] **AUDIT 156/384** `getStoreProducer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1285`, line by line against the C++
- [ ] **PORT 157/384** `constructSetActiveMaskValueOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2567`, 161 lines
- [ ] **AUDIT 157/384** `constructSetActiveMaskValueOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2567`, line by line against the C++
- [ ] **PORT 158/384** `createUniformizeRegionsOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3880`, 57 lines
- [ ] **AUDIT 158/384** `createUniformizeRegionsOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3880`, line by line against the C++
- [ ] **PORT 159/384** `getUnitNameFromAListOfGetUnitOp` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:119`, 10 lines
- [ ] **AUDIT 159/384** `getUnitNameFromAListOfGetUnitOp` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:119`, line by line against the C++
- [ ] **PORT 160/384** `areCoreletsDifferent` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:132`, 6 lines
- [ ] **AUDIT 160/384** `areCoreletsDifferent` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:132`, line by line against the C++
- [ ] **PORT 161/384** `separateBasedOnDestinationUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:761`, 18 lines
- [ ] **AUDIT 161/384** `separateBasedOnDestinationUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:761`, line by line against the C++
- [ ] **PORT 162/384** `insertIfNotExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:81`, 8 lines
- [ ] **AUDIT 162/384** `insertIfNotExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:81`, line by line against the C++
- [ ] **PORT 163/384** `getMaskValueConstantForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:93`, 40 lines
- [ ] **AUDIT 163/384** `getMaskValueConstantForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:93`, line by line against the C++
- [ ] **PORT 164/384** `checkValidityOfPackAndShuffleLowering` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:140`, 45 lines
- [ ] **AUDIT 164/384** `checkValidityOfPackAndShuffleLowering` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:140`, line by line against the C++
- [ ] **PORT 165/384** `getMergeTypeFromIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:301`, 109 lines
- [ ] **AUDIT 165/384** `getMergeTypeFromIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:301`, line by line against the C++
- [ ] **PORT 166/384** `getOperandFromConstantOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:267`, 26 lines
- [ ] **AUDIT 166/384** `getOperandFromConstantOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:267`, line by line against the C++
- [ ] **PORT 167/384** `getOperandFromConstantBitstreamOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:299`, 5 lines
- [ ] **AUDIT 167/384** `getOperandFromConstantBitstreamOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:299`, line by line against the C++
- [ ] **PORT 168/384** `getOperandFromNegOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:366`, 5 lines
- [ ] **AUDIT 168/384** `getOperandFromNegOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:366`, line by line against the C++
- [ ] **PORT 169/384** `getName` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:866`, 11 lines
- [ ] **AUDIT 169/384** `getName` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:866`, line by line against the C++
- [ ] **PORT 170/384** `getLayoutMapAndIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:879`, 40 lines
- [ ] **AUDIT 170/384** `getLayoutMapAndIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:879`, line by line against the C++
- [ ] **PORT 171/384** `isMaskEquivalentToNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:123`, 4 lines
- [ ] **AUDIT 171/384** `isMaskEquivalentToNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:123`, line by line against the C++
- [ ] **PORT 172/384** `areXrfAccessesLegal` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:98`, 28 lines
- [ ] **AUDIT 172/384** `areXrfAccessesLegal` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:98`, line by line against the C++
- [ ] **PORT 173/384** `insertConstAndAddOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:296`, 10 lines
- [ ] **AUDIT 173/384** `insertConstAndAddOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:296`, line by line against the C++
- [ ] **PORT 174/384** `isXrfRelated` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:531`, 31 lines
- [ ] **AUDIT 174/384** `isXrfRelated` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:531`, line by line against the C++
- [ ] **PORT 175/384** `updateYieldArgs` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:690`, 8 lines
- [ ] **AUDIT 175/384** `updateYieldArgs` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:690`, line by line against the C++
- [ ] **PORT 176/384** `opHasSideEffect` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:38`, 13 lines
- [ ] **AUDIT 176/384** `opHasSideEffect` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:38`, line by line against the C++
- [ ] **PORT 177/384** `mergeShallow` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:398`, 60 lines
- [ ] **AUDIT 177/384** `mergeShallow` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:398`, line by line against the C++
- [ ] **PORT 178/384** `runOnOperation` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:49`, 42 lines
- [ ] **AUDIT 178/384** `runOnOperation` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:49`, line by line against the C++
- [ ] **PORT 179/384** `clear` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:112`, 15 lines
- [ ] **AUDIT 179/384** `clear` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:112`, line by line against the C++
- [ ] **PORT 180/384** `traverseRegion` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:129`, 17 lines
- [ ] **AUDIT 180/384** `traverseRegion` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:129`, line by line against the C++
- [ ] **PORT 181/384** `inRegionEmpty` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:214`, 6 lines
- [ ] **AUDIT 181/384** `inRegionEmpty` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:214`, line by line against the C++
- [ ] **PORT 182/384** `cloneOpsForRegions` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:223`, 60 lines
- [ ] **AUDIT 182/384** `cloneOpsForRegions` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:223`, line by line against the C++
- [ ] **PORT 183/384** `expandAffineApplyOps` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:183`, 53 lines
- [ ] **AUDIT 183/384** `expandAffineApplyOps` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:183`, line by line against the C++
- [ ] **PORT 184/384** `getLoopTripCount` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:743`, 56 lines
- [ ] **AUDIT 184/384** `getLoopTripCount` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:743`, line by line against the C++
- [ ] **PORT 185/384** `hasMutableAddrOverflow` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:801`, 14 lines
- [ ] **AUDIT 185/384** `hasMutableAddrOverflow` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:801`, line by line against the C++
- [ ] **PORT 186/384** `calculatePartitionSizes` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:862`, 97 lines
- [ ] **AUDIT 186/384** `calculatePartitionSizes` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:862`, line by line against the C++
- [ ] **PORT 187/384** `constructConditionals` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1000`, 59 lines
- [ ] **AUDIT 187/384** `constructConditionals` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1000`, line by line against the C++
- [ ] **PORT 188/384** `calculateSubscriptsCoefficients` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1188`, 11 lines
- [ ] **AUDIT 188/384** `calculateSubscriptsCoefficients` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1188`, line by line against the C++
- [ ] **PORT 189/384** `synthesizeTimeInfo` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1256`, 22 lines
- [ ] **AUDIT 189/384** `synthesizeTimeInfo` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1256`, line by line against the C++
- [ ] **PORT 190/384** `createExplicitTimeLoops` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1282`, 57 lines
- [ ] **AUDIT 190/384** `createExplicitTimeLoops` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1282`, line by line against the C++
- [ ] **PORT 191/384** `calculateFullShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:462`, 25 lines
- [ ] **AUDIT 191/384** `calculateFullShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:462`, line by line against the C++
- [ ] **PORT 192/384** `calculateDimWeights` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:560`, 26 lines
- [ ] **AUDIT 192/384** `calculateDimWeights` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:560`, line by line against the C++
- [ ] **PORT 193/384** `matchUnits` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:69`, 77 lines
- [ ] **AUDIT 193/384** `matchUnits` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:69`, line by line against the C++
- [ ] **PORT 194/384** `analyzeLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:268`, 127 lines
- [ ] **AUDIT 194/384** `analyzeLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:268`, line by line against the C++
- [ ] **PORT 195/384** `transformLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:399`, 38 lines
- [ ] **AUDIT 195/384** `transformLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:399`, line by line against the C++
- [ ] **PORT 196/384** `runOnOperation` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:41`, 23 lines
- [ ] **AUDIT 196/384** `runOnOperation` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:41`, line by line against the C++
- [ ] **PORT 197/384** `calculateIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:56`, 25 lines
- [ ] **AUDIT 197/384** `calculateIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:56`, line by line against the C++
- [ ] **PORT 198/384** `createConditionsForHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:289`, 48 lines
- [ ] **AUDIT 198/384** `createConditionsForHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:289`, line by line against the C++
- [ ] **PORT 199/384** `createConditionsForNonHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:342`, 35 lines
- [ ] **AUDIT 199/384** `createConditionsForNonHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:342`, line by line against the C++
- [ ] **PORT 200/384** `updateTPMVInfo` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:382`, 17 lines
- [ ] **AUDIT 200/384** `updateTPMVInfo` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:382`, line by line against the C++
- [ ] **PORT 201/384** `setLoopIteratorOrder` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:575`, 13 lines
- [ ] **AUDIT 201/384** `setLoopIteratorOrder` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:575`, line by line against the C++
- [ ] **PORT 202/384** `initialize` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:658`, 13 lines
- [ ] **AUDIT 202/384** `initialize` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:658`, line by line against the C++
- [ ] **PORT 203/384** `removeCoresCoreletsFoldsFromDefImmutMap` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:99`, 37 lines
- [ ] **AUDIT 203/384** `removeCoresCoreletsFoldsFromDefImmutMap` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:99`, line by line against the C++
- [ ] **PORT 204/384** `cleanup` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:263`, 49 lines
- [ ] **AUDIT 204/384** `cleanup` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:263`, line by line against the C++
- [ ] **PORT 205/384** `isDataTransferToKeep` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:327`, 10 lines
- [ ] **AUDIT 205/384** `isDataTransferToKeep` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:327`, line by line against the C++

## Level 2

- [ ] **PORT 206/384** `constructChunkAndShuffleInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:98`, 167 lines
- [ ] **AUDIT 206/384** `constructChunkAndShuffleInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:98`, line by line against the C++
- [ ] **PORT 207/384** `initialize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:295`, 57 lines
- [ ] **AUDIT 207/384** `initialize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:295`, line by line against the C++
- [ ] **PORT 208/384** `emplace_insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:378`, 8 lines
- [ ] **AUDIT 208/384** `emplace_insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:378`, line by line against the C++
- [ ] **PORT 209/384** `getFirst` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:411`, 7 lines
- [ ] **AUDIT 209/384** `getFirst` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:411`, line by line against the C++
- [ ] **PORT 210/384** `checkBasicConditions` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:58`, 133 lines
- [ ] **AUDIT 210/384** `checkBasicConditions` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:58`, line by line against the C++
- [ ] **PORT 211/384** `processInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:305`, 80 lines
- [ ] **AUDIT 211/384** `processInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:305`, line by line against the C++
- [ ] **PORT 212/384** `gatherAffineLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:538`, 74 lines
- [ ] **AUDIT 212/384** `gatherAffineLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:538`, line by line against the C++
- [ ] **PORT 213/384** `constructImmutableAddress` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1217`, 16 lines
- [ ] **AUDIT 213/384** `constructImmutableAddress` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1217`, line by line against the C++
- [x] **PORT 214/384** `setImmutableAddrAndIncrements` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1581`, 43 lines
- [x] **AUDIT 214/384** `setImmutableAddrAndIncrements` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1581`, line by line against the C++
- [ ] **PORT 215/384** `setsttype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1731`, 50 lines
- [ ] **AUDIT 215/384** `setsttype` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1731`, line by line against the C++
- [ ] **PORT 216/384** `constructReceiveAndExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2471`, 91 lines
- [ ] **AUDIT 216/384** `constructReceiveAndExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2471`, line by line against the C++
- [ ] **PORT 217/384** `lowerVectorLoadHelper` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2899`, 46 lines
- [ ] **AUDIT 217/384** `lowerVectorLoadHelper` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2899`, line by line against the C++
- [ ] **PORT 218/384** `lowerSetTransferMaskStateOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3815`, 9 lines
- [ ] **AUDIT 218/384** `lowerSetTransferMaskStateOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3815`, line by line against the C++
- [ ] **PORT 219/384** `cloneStartAddrOutsideLoop` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3942`, 79 lines
- [ ] **AUDIT 219/384** `cloneStartAddrOutsideLoop` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3942`, line by line against the C++
- [ ] **PORT 220/384** `cleanupTriviallyRedundantSetSendDestination` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:4084`, 38 lines
- [ ] **AUDIT 220/384** `cleanupTriviallyRedundantSetSendDestination` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:4084`, line by line against the C++
- [ ] **PORT 221/384** `pushBackTheUnitToListIfDoesnotExist` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:143`, 5 lines
- [ ] **AUDIT 221/384** `pushBackTheUnitToListIfDoesnotExist` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:143`, line by line against the C++
- [ ] **PORT 222/384** `createUniformRegionsWithTwoRegionsNoResult` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:153`, 18 lines
- [ ] **AUDIT 222/384** `createUniformRegionsWithTwoRegionsNoResult` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:153`, line by line against the C++
- [ ] **PORT 223/384** `lowerL3SyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:375`, 57 lines
- [ ] **AUDIT 223/384** `lowerL3SyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:375`, line by line against the C++
- [ ] **PORT 224/384** `lowerL3SyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:667`, 61 lines
- [ ] **AUDIT 224/384** `lowerL3SyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:667`, line by line against the C++
- [ ] **PORT 225/384** `matchAndRewrite` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:72`, 69 lines
- [ ] **AUDIT 225/384** `matchAndRewrite` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:72`, line by line against the C++
- [ ] **PORT 226/384** `createIfOpFromMapping` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:123`, 61 lines
- [ ] **AUDIT 226/384** `createIfOpFromMapping` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:123`, line by line against the C++
- [ ] **PORT 227/384** `OperandReuse` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:30`, 0 lines
- [ ] **AUDIT 227/384** `OperandReuse` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:30`, line by line against the C++
- [ ] **PORT 228/384** `validateLoweringAndSetMissingParameters` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:483`, 49 lines
- [ ] **AUDIT 228/384** `validateLoweringAndSetMissingParameters` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:483`, line by line against the C++
- [ ] **PORT 229/384** `getMaskValueForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:114`, 9 lines
- [ ] **AUDIT 229/384** `getMaskValueForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:114`, line by line against the C++
- [ ] **PORT 230/384** `convertStringToType` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:196`, 24 lines
- [ ] **AUDIT 230/384** `convertStringToType` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:196`, line by line against the C++
- [ ] **PORT 231/384** `convertTypeToString` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:223`, 22 lines
- [ ] **AUDIT 231/384** `convertTypeToString` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:223`, line by line against the C++
- [ ] **PORT 232/384** `getOperandFromLoadOrStoreOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:153`, 94 lines
- [ ] **AUDIT 232/384** `getOperandFromLoadOrStoreOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:153`, line by line against the C++
- [ ] **PORT 233/384** `eraseOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:690`, 46 lines
- [ ] **AUDIT 233/384** `eraseOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:690`, line by line against the C++
- [ ] **PORT 234/384** `setValue` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:72`, 3 lines
- [ ] **AUDIT 234/384** `setValue` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:72`, line by line against the C++
- [ ] **PORT 235/384** `createSentientConstants` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:34`, 31 lines
- [ ] **AUDIT 235/384** `createSentientConstants` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:34`, line by line against the C++
- [ ] **PORT 236/384** `addMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:136`, 16 lines
- [ ] **AUDIT 236/384** `addMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:136`, line by line against the C++
- [ ] **PORT 237/384** `updateNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:155`, 10 lines
- [ ] **AUDIT 237/384** `updateNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:155`, line by line against the C++
- [ ] **PORT 238/384** `computeLoops` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:173`, 18 lines
- [ ] **AUDIT 238/384** `computeLoops` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:173`, line by line against the C++
- [ ] **PORT 239/384** `insertPTMaskOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:41`, 164 lines
- [ ] **AUDIT 239/384** `insertPTMaskOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:41`, line by line against the C++
- [ ] **PORT 240/384** `getLayoutExpr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:28`, 67 lines
- [ ] **AUDIT 240/384** `getLayoutExpr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:28`, line by line against the C++
- [ ] **PORT 241/384** `createForOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:129`, 56 lines
- [ ] **AUDIT 241/384** `createForOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:129`, line by line against the C++
- [ ] **PORT 242/384** `createIfOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:189`, 56 lines
- [ ] **AUDIT 242/384** `createIfOpWithReturnValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:189`, line by line against the C++
- [ ] **PORT 243/384** `insertDummyMacOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:312`, 15 lines
- [ ] **AUDIT 243/384** `insertDummyMacOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:312`, line by line against the C++
- [ ] **PORT 244/384** `isHoistable` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:82`, 9 lines
- [ ] **AUDIT 244/384** `isHoistable` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:82`, line by line against the C++
- [ ] **PORT 245/384** `enumerateCollectionUnit` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:34`, 57 lines
- [ ] **AUDIT 245/384** `enumerateCollectionUnit` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:34`, line by line against the C++
- [ ] **PORT 246/384** `FlatteningLocalRegionsTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:79`, 0 lines
- [ ] **AUDIT 246/384** `FlatteningLocalRegionsTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:79`, line by line against the C++
- [ ] **PORT 247/384** `compute` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:151`, 15 lines
- [ ] **AUDIT 247/384** `compute` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:151`, line by line against the C++
- [ ] **PORT 248/384** `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:70`, 68 lines
- [ ] **AUDIT 248/384** `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:70`, line by line against the C++
- [ ] **PORT 249/384** `processComputeUnit` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:37`, 91 lines
- [ ] **AUDIT 249/384** `processComputeUnit` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:37`, line by line against the C++
- [ ] **PORT 250/384** `initMASData` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:707`, 31 lines
- [ ] **AUDIT 250/384** `initMASData` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:707`, line by line against the C++
- [ ] **PORT 251/384** `setupForPartitioning` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:819`, 6 lines
- [ ] **AUDIT 251/384** `setupForPartitioning` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:819`, line by line against the C++
- [ ] **PORT 252/384** `fillPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1063`, 117 lines
- [ ] **AUDIT 252/384** `fillPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1063`, line by line against the C++
- [ ] **PORT 253/384** `adjustForEvenImmutableAddr` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1204`, 47 lines
- [ ] **AUDIT 253/384** `adjustForEvenImmutableAddr` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1204`, line by line against the C++
- [ ] **PORT 254/384** `offsetShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:590`, 22 lines
- [ ] **AUDIT 254/384** `offsetShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:590`, line by line against the C++
- [ ] **PORT 255/384** `applyShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:616`, 27 lines
- [ ] **AUDIT 255/384** `applyShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:616`, line by line against the C++
- [ ] **PORT 256/384** `runOnOperation` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:152`, 76 lines
- [ ] **AUDIT 256/384** `runOnOperation` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:152`, line by line against the C++
- [ ] **PORT 257/384** `analyzeAndTransform` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:441`, 3 lines
- [ ] **AUDIT 257/384** `analyzeAndTransform` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:441`, line by line against the C++
- [ ] **PORT 258/384** `addConstraintsForIVRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:86`, 22 lines
- [ ] **AUDIT 258/384** `addConstraintsForIVRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:86`, line by line against the C++
- [ ] **PORT 259/384** `createNewSubscriptsFromStartElements` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:562`, 10 lines
- [ ] **AUDIT 259/384** `createNewSubscriptsFromStartElements` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:562`, line by line against the C++
- [ ] **PORT 260/384** `gatherPageDependentDimsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:927`, 32 lines
- [ ] **AUDIT 260/384** `gatherPageDependentDimsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:927`, line by line against the C++
- [ ] **PORT 261/384** `runOnOperation` — `dcc/src/Transform/Dataflow/UniformQueryMapsCanonicalization.cpp:55`, 41 lines
- [ ] **AUDIT 261/384** `runOnOperation` — `dcc/src/Transform/Dataflow/UniformQueryMapsCanonicalization.cpp:55`, line by line against the C++
- [ ] **PORT 262/384** `removeCoresCoreletsFoldsFromUniformizeRegion` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:139`, 98 lines
- [ ] **AUDIT 262/384** `removeCoresCoreletsFoldsFromUniformizeRegion` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:139`, line by line against the C++
- [ ] **PORT 263/384** `removeAncestors` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:339`, 18 lines
- [ ] **AUDIT 263/384** `removeAncestors` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:339`, line by line against the C++
- [ ] **PORT 264/384** `createForOpWithAdditionalReturnValue` — `dcc/src/Transform/Dataflow/Utils.cpp:28`, 66 lines
- [ ] **AUDIT 264/384** `createForOpWithAdditionalReturnValue` — `dcc/src/Transform/Dataflow/Utils.cpp:28`, line by line against the C++

## Level 3

- [ ] **PORT 265/384** `constructDetails` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:418`, 18 lines
- [ ] **AUDIT 265/384** `constructDetails` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:418`, line by line against the C++
- [ ] **PORT 266/384** `coalesceTimeDimensions` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:681`, 112 lines
- [ ] **AUDIT 266/384** `coalesceTimeDimensions` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:681`, line by line against the C++
- [ ] **PORT 267/384** `constructTimeLoopsAndVectorOperations` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1789`, 109 lines
- [ ] **AUDIT 267/384** `constructTimeLoopsAndVectorOperations` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1789`, line by line against the C++
- [ ] **PORT 268/384** `constructLoadAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2167`, 177 lines
- [ ] **AUDIT 268/384** `constructLoadAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2167`, line by line against the C++
- [ ] **PORT 269/384** `constructLoadAndExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2351`, 114 lines
- [ ] **AUDIT 269/384** `constructLoadAndExtractScalarOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2351`, line by line against the C++
- [ ] **PORT 270/384** `addStoreInputToDeleteList` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2987`, 8 lines
- [ ] **AUDIT 270/384** `addStoreInputToDeleteList` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2987`, line by line against the C++
- [ ] **PORT 271/384** `lowerCompositeMemoryInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3775`, 37 lines
- [ ] **AUDIT 271/384** `lowerCompositeMemoryInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3775`, line by line against the C++
- [ ] **PORT 272/384** `insertInitializationStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3834`, 13 lines
- [ ] **AUDIT 272/384** `insertInitializationStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3834`, line by line against the C++
- [ ] **PORT 273/384** `lowerL0LXSyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:189`, 180 lines
- [ ] **AUDIT 273/384** `lowerL0LXSyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:189`, line by line against the C++
- [ ] **PORT 274/384** `lowerL0LXSyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:438`, 222 lines
- [ ] **AUDIT 274/384** `lowerL0LXSyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:438`, line by line against the C++
- [ ] **PORT 275/384** `LowerSymbolQueryMap` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:40`, 68 lines
- [ ] **AUDIT 275/384** `LowerSymbolQueryMap` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:40`, line by line against the C++
- [ ] **PORT 276/384** `setReuseInformation` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:17`, 44 lines
- [ ] **AUDIT 276/384** `setReuseInformation` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:17`, line by line against the C++
- [ ] **PORT 277/384** `getGCVTorFCVTTypeFromIndicesAndCastInputs` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:188`, 108 lines
- [ ] **AUDIT 277/384** `getGCVTorFCVTTypeFromIndicesAndCastInputs` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:188`, line by line against the C++
- [ ] **PORT 278/384** `getOperandFromShuffleOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:311`, 36 lines
- [ ] **AUDIT 278/384** `getOperandFromShuffleOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:311`, line by line against the C++
- [ ] **PORT 279/384** `createSplatOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:70`, 110 lines
- [ ] **AUDIT 279/384** `createSplatOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:70`, line by line against the C++
- [ ] **PORT 280/384** `cleanup` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1056`, 7 lines
- [ ] **AUDIT 280/384** `cleanup` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1056`, line by line against the C++
- [ ] **PORT 281/384** `OperationTreeBase` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:105`, 2 lines
- [ ] **AUDIT 281/384** `OperationTreeBase` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:105`, line by line against the C++
- [ ] **PORT 282/384** `updateLoopMaskTreeForConstantMask` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:20`, 4 lines
- [ ] **AUDIT 282/384** `updateLoopMaskTreeForConstantMask` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:20`, line by line against the C++
- [ ] **PORT 283/384** `updateLoopMaskTreeForDynamicMask` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:28`, 9 lines
- [ ] **AUDIT 283/384** `updateLoopMaskTreeForDynamicMask` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:28`, line by line against the C++
- [ ] **PORT 284/384** `hoistCommonConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:94`, 96 lines
- [ ] **AUDIT 284/384** `hoistCommonConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:94`, line by line against the C++
- [ ] **PORT 285/384** `replaceIfOpByIterArg` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:619`, 56 lines
- [ ] **AUDIT 285/384** `replaceIfOpByIterArg` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:619`, line by line against the C++
- [ ] **PORT 286/384** `runOnOperation` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:94`, 32 lines
- [ ] **AUDIT 286/384** `runOnOperation` — `dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:94`, line by line against the C++
- [ ] **PORT 287/384** `flatten` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:384`, 70 lines
- [ ] **AUDIT 287/384** `flatten` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:384`, line by line against the C++
- [ ] **PORT 288/384** `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:131`, 22 lines
- [ ] **AUDIT 288/384** `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:131`, line by line against the C++
- [ ] **PORT 289/384** `initialize` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:694`, 8 lines
- [ ] **AUDIT 289/384** `initialize` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:694`, line by line against the C++
- [ ] **PORT 290/384** `createPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:983`, 10 lines
- [ ] **AUDIT 290/384** `createPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:983`, line by line against the C++
- [ ] **PORT 291/384** `calculatePartialShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:490`, 66 lines
- [ ] **AUDIT 291/384** `calculatePartialShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:490`, line by line against the C++
- [ ] **PORT 292/384** `transformSCFLoopWithNonConstantUpperBound` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:169`, 94 lines
- [ ] **AUDIT 292/384** `transformSCFLoopWithNonConstantUpperBound` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:169`, line by line against the C++
- [ ] **PORT 293/384** `runOn` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:447`, 5 lines
- [ ] **AUDIT 293/384** `runOn` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:447`, line by line against the C++
- [ ] **PORT 294/384** `getPageValidity` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:114`, 33 lines
- [ ] **AUDIT 294/384** `getPageValidity` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:114`, line by line against the C++
- [ ] **PORT 295/384** `createIterArgsForConditionals` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:401`, 121 lines
- [ ] **AUDIT 295/384** `createIterArgsForConditionals` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:401`, line by line against the C++
- [ ] **PORT 296/384** `runOnOperation` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:362`, 127 lines
- [ ] **AUDIT 296/384** `runOnOperation` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:362`, line by line against the C++

## Level 4

- [ ] **PORT 297/384** `constructTimeStepsInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:626`, 42 lines
- [ ] **AUDIT 297/384** `constructTimeStepsInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:626`, line by line against the C++
- [ ] **PORT 298/384** `constructAffineDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2787`, 16 lines
- [ ] **AUDIT 298/384** `constructAffineDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2787`, line by line against the C++
- [ ] **PORT 299/384** `lowerAffineCompositeHelper` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2953`, 15 lines
- [ ] **AUDIT 299/384** `lowerAffineCompositeHelper` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2953`, line by line against the C++
- [ ] **PORT 300/384** `lowerSyncForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:733`, 7 lines
- [ ] **AUDIT 300/384** `lowerSyncForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:733`, line by line against the C++
- [ ] **PORT 301/384** `lowerSyncForAGroup` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:746`, 8 lines
- [ ] **AUDIT 301/384** `lowerSyncForAGroup` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:746`, line by line against the C++
- [ ] **PORT 302/384** `lowerSyncLXL3ToLXL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:787`, 928 lines
- [ ] **AUDIT 302/384** `lowerSyncLXL3ToLXL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:787`, line by line against the C++
- [ ] **PORT 303/384** `runOnOperation` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:23`, 15 lines
- [ ] **AUDIT 303/384** `runOnOperation` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:23`, line by line against the C++
- [ ] **PORT 304/384** `getOperandWithPrecision` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:389`, 254 lines
- [ ] **AUDIT 304/384** `getOperandWithPrecision` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:389`, line by line against the C++
- [ ] **PORT 305/384** `runOnOperation` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:459`, 17 lines
- [ ] **AUDIT 305/384** `runOnOperation` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:459`, line by line against the C++
- [ ] **PORT 306/384** `transformVectorLoad` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:298`, 75 lines
- [ ] **AUDIT 306/384** `transformVectorLoad` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:298`, line by line against the C++
- [ ] **PORT 307/384** `transformVectorStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:375`, 74 lines
- [ ] **AUDIT 307/384** `transformVectorStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:375`, line by line against the C++
- [ ] **PORT 308/384** `calculateShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:396`, 61 lines
- [ ] **AUDIT 308/384** `calculateShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:396`, line by line against the C++
- [ ] **PORT 309/384** `constructValidPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:190`, 58 lines
- [ ] **AUDIT 309/384** `constructValidPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:190`, line by line against the C++
- [ ] **PORT 310/384** `analyzeValidPages` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:888`, 33 lines
- [ ] **AUDIT 310/384** `analyzeValidPages` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:888`, line by line against the C++

## Level 5

- [ ] **PORT 311/384** `constructAffineCompDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2809`, 33 lines
- [ ] **AUDIT 311/384** `constructAffineCompDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2809`, line by line against the C++
- [ ] **PORT 312/384** `lowerExtractVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3001`, 21 lines
- [ ] **AUDIT 312/384** `lowerExtractVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3001`, line by line against the C++
- [ ] **PORT 313/384** `lowerExtractVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3026`, 20 lines
- [ ] **AUDIT 313/384** `lowerExtractVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3026`, line by line against the C++
- [ ] **PORT 314/384** `lowerVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3050`, 20 lines
- [ ] **AUDIT 314/384** `lowerVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3050`, line by line against the C++
- [ ] **PORT 315/384** `lowerVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3074`, 28 lines
- [ ] **AUDIT 315/384** `lowerVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3074`, line by line against the C++
- [ ] **PORT 316/384** `lowerIndirectVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3169`, 44 lines
- [ ] **AUDIT 316/384** `lowerIndirectVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3169`, line by line against the C++
- [ ] **PORT 317/384** `lowerIndirectVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3217`, 46 lines
- [ ] **AUDIT 317/384** `lowerIndirectVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3217`, line by line against the C++
- [ ] **PORT 318/384** `lowerLDCVTIPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3444`, 326 lines
- [ ] **AUDIT 318/384** `lowerLDCVTIPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3444`, line by line against the C++
- [ ] **PORT 319/384** `lowerSyncForAQueryMap` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1728`, 166 lines
- [ ] **AUDIT 319/384** `lowerSyncForAQueryMap` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1728`, line by line against the C++
- [ ] **PORT 320/384** `getOperand` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:378`, 4 lines
- [ ] **AUDIT 320/384** `getOperand` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:378`, line by line against the C++
- [ ] **PORT 321/384** `transformCompLoadAndStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:451`, 92 lines
- [ ] **AUDIT 321/384** `transformCompLoadAndStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:451`, line by line against the C++
- [ ] **PORT 322/384** `transformCompIndLoadAndStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:546`, 124 lines
- [ ] **AUDIT 322/384** `transformCompIndLoadAndStore` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:546`, line by line against the C++
- [ ] **PORT 323/384** `shiftMutableAddr` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:366`, 19 lines
- [ ] **AUDIT 323/384** `shiftMutableAddr` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:366`, line by line against the C++
- [ ] **PORT 324/384** `analyzeAndConstructValidPages` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:152`, 32 lines
- [ ] **AUDIT 324/384** `analyzeAndConstructValidPages` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:152`, line by line against the C++
- [ ] **PORT 325/384** `transform_time` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:977`, 98 lines
- [ ] **AUDIT 325/384** `transform_time` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:977`, line by line against the C++
- [ ] **PORT 326/384** `initialize_time` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:1095`, 23 lines
- [ ] **AUDIT 326/384** `initialize_time` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:1095`, line by line against the C++

## Level 6

- [ ] **PORT 327/384** `gatherSymbolicLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1051`, 153 lines
- [ ] **AUDIT 327/384** `gatherSymbolicLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1051`, line by line against the C++
- [ ] **PORT 328/384** `adjustMutableAddrInitForIndirect` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1447`, 127 lines
- [ ] **AUDIT 328/384** `adjustMutableAddrInitForIndirect` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1447`, line by line against the C++
- [ ] **PORT 329/384** `lowerCompositeLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3106`, 17 lines
- [ ] **AUDIT 329/384** `lowerCompositeLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3106`, line by line against the C++
- [ ] **PORT 330/384** `lowerCompositeStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3127`, 17 lines
- [ ] **AUDIT 330/384** `lowerCompositeStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3127`, line by line against the C++
- [ ] **PORT 331/384** `lowerCompositeLoadAndStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3148`, 17 lines
- [ ] **AUDIT 331/384** `lowerCompositeLoadAndStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3148`, line by line against the C++
- [ ] **PORT 332/384** `lowerCompositeIndirectLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3267`, 42 lines
- [ ] **AUDIT 332/384** `lowerCompositeIndirectLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3267`, line by line against the C++
- [ ] **PORT 333/384** `lowerCompositeIndirectStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3313`, 42 lines
- [ ] **AUDIT 333/384** `lowerCompositeIndirectStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3313`, line by line against the C++
- [ ] **PORT 334/384** `lowerCompositeIndirectLoadAndStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3359`, 17 lines
- [ ] **AUDIT 334/384** `lowerCompositeIndirectLoadAndStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3359`, line by line against the C++
- [ ] **PORT 335/384** `insertCopyAndAddStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3861`, 13 lines
- [ ] **AUDIT 335/384** `insertCopyAndAddStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3861`, line by line against the C++
- [ ] **PORT 336/384** `adjustMutableAddrInitForStride` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:4034`, 47 lines
- [ ] **AUDIT 336/384** `adjustMutableAddrInitForStride` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:4034`, line by line against the C++
- [ ] **PORT 337/384** `lowerSyncOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1901`, 80 lines
- [ ] **AUDIT 337/384** `lowerSyncOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1901`, line by line against the C++
- [ ] **PORT 338/384** `ConstructIFRecursively` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:113`, 125 lines
- [ ] **AUDIT 338/384** `ConstructIFRecursively` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:113`, line by line against the C++
- [ ] **PORT 339/384** `SimplifyOrIOp` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:389`, 43 lines
- [ ] **AUDIT 339/384** `SimplifyOrIOp` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:389`, line by line against the C++
- [ ] **PORT 340/384** `analyzeAndFillOperandForwarding` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:535`, 25 lines
- [ ] **AUDIT 340/384** `analyzeAndFillOperandForwarding` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:535`, line by line against the C++
- [ ] **PORT 341/384** `analyzeNonComputeOpsForFusion` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:610`, 86 lines
- [ ] **AUDIT 341/384** `analyzeNonComputeOpsForFusion` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:610`, line by line against the C++
- [ ] **PORT 342/384** `analyzeAndFillResultForwarding` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:162`, 27 lines
- [ ] **AUDIT 342/384** `analyzeAndFillResultForwarding` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:162`, line by line against the C++
- [ ] **PORT 343/384** `getOperandFromCastOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:354`, 5 lines
- [ ] **AUDIT 343/384** `getOperandFromCastOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:354`, line by line against the C++
- [ ] **PORT 344/384** `lowerDanglingNonComputeOpsPESFP` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1274`, 93 lines
- [ ] **AUDIT 344/384** `lowerDanglingNonComputeOpsPESFP` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1274`, line by line against the C++
- [ ] **PORT 345/384** `processXrfPtrPerUnit` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:337`, 190 lines
- [ ] **AUDIT 345/384** `processXrfPtrPerUnit` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:337`, line by line against the C++
- [ ] **PORT 346/384** `lowerDanglingNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:882`, 88 lines
- [ ] **AUDIT 346/384** `lowerDanglingNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:882`, line by line against the C++
- [ ] **PORT 347/384** `topLevelConditionsMatch` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:279`, 31 lines
- [ ] **AUDIT 347/384** `topLevelConditionsMatch` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:279`, line by line against the C++
- [ ] **PORT 348/384** `singleOpBranchToYieldVal` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:498`, 18 lines
- [ ] **AUDIT 348/384** `singleOpBranchToYieldVal` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:498`, line by line against the C++
- [ ] **PORT 349/384** `isLoopInvariant` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:681`, 66 lines
- [ ] **AUDIT 349/384** `isLoopInvariant` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:681`, line by line against the C++
- [ ] **PORT 350/384** `matchAndRewrite` — `dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:33`, 170 lines
- [ ] **AUDIT 350/384** `matchAndRewrite` — `dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:33`, line by line against the C++
- [ ] **PORT 351/384** `runOnOperation` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:226`, 70 lines
- [ ] **AUDIT 351/384** `runOnOperation` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:226`, line by line against the C++
- [ ] **PORT 352/384** `transformVectorLoad` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:201`, 25 lines
- [ ] **AUDIT 352/384** `transformVectorLoad` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:201`, line by line against the C++
- [ ] **PORT 353/384** `transformVectorStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:229`, 24 lines
- [ ] **AUDIT 353/384** `transformVectorStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:229`, line by line against the C++
- [ ] **PORT 354/384** `transformCompLoadAndStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:256`, 39 lines
- [ ] **AUDIT 354/384** `transformCompLoadAndStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:256`, line by line against the C++
- [ ] **PORT 355/384** `transformCompIndLoadAndStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:298`, 54 lines
- [ ] **AUDIT 355/384** `transformCompIndLoadAndStore` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:298`, line by line against the C++
- [ ] **PORT 356/384** `transform` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:592`, 39 lines
- [ ] **AUDIT 356/384** `transform` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:592`, line by line against the C++

## Level 7

- [ ] **PORT 357/384** `generateAffineAddressManipulationStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:625`, 381 lines
- [ ] **AUDIT 357/384** `generateAffineAddressManipulationStmts` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:625`, line by line against the C++
- [ ] **PORT 358/384** `constructLoadAndSendStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1910`, 104 lines
- [ ] **AUDIT 358/384** `constructLoadAndSendStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1910`, line by line against the C++
- [ ] **PORT 359/384** `constructReceiveAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2025`, 131 lines
- [ ] **AUDIT 359/384** `constructReceiveAndStoreStmt` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2025`, line by line against the C++
- [ ] **PORT 360/384** `constructSymbolicDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2849`, 16 lines
- [ ] **AUDIT 360/384** `constructSymbolicDetailsAndAddrs` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2849`, line by line against the C++
- [ ] **PORT 361/384** `runOnOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2014`, 33 lines
- [ ] **AUDIT 361/384** `runOnOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2014`, line by line against the C++
- [ ] **PORT 362/384** `LowerSelectOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:243`, 19 lines
- [ ] **AUDIT 362/384** `LowerSelectOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:243`, line by line against the C++
- [ ] **PORT 363/384** `LowerLogicalOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:264`, 19 lines
- [ ] **AUDIT 363/384** `LowerLogicalOpToSentient` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:264`, line by line against the C++
- [ ] **PORT 364/384** `patternAgnosticFuseNonComputeOpsHelper` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:96`, 222 lines
- [ ] **AUDIT 364/384** `patternAgnosticFuseNonComputeOpsHelper` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:96`, line by line against the C++
- [ ] **PORT 365/384** `fillOpInfo` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1070`, 83 lines
- [ ] **AUDIT 365/384** `fillOpInfo` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1070`, line by line against the C++
- [ ] **PORT 366/384** `fuseNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1159`, 80 lines
- [ ] **AUDIT 366/384** `fuseNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1159`, line by line against the C++
- [ ] **PORT 367/384** `createXrfIndexModifOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:564`, 97 lines
- [ ] **AUDIT 367/384** `createXrfIndexModifOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:564`, line by line against the C++
- [ ] **PORT 368/384** `fuseNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:46`, 191 lines
- [ ] **AUDIT 368/384** `fuseNonComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:46`, line by line against the C++
- [ ] **PORT 369/384** `fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:245`, 628 lines
- [ ] **AUDIT 369/384** `fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:245`, line by line against the C++
- [ ] **PORT 370/384** `areShallowlyMergeable` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:343`, 34 lines
- [ ] **AUDIT 370/384** `areShallowlyMergeable` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:343`, line by line against the C++
- [ ] **PORT 371/384** `hoistLoopInvariantConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:750`, 42 lines
- [ ] **AUDIT 371/384** `hoistLoopInvariantConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:750`, line by line against the C++
- [ ] **PORT 372/384** `runOnOperation` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:130`, 69 lines
- [ ] **AUDIT 372/384** `runOnOperation` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:130`, line by line against the C++
- [ ] **PORT 373/384** `run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:647`, 6 lines
- [ ] **AUDIT 373/384** `run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:647`, line by line against the C++

## Level 8

- [ ] **PORT 374/384** `lowerSymbolicVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3380`, 26 lines
- [ ] **AUDIT 374/384** `lowerSymbolicVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3380`, line by line against the C++
- [ ] **PORT 375/384** `lowerSymbolicVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3410`, 30 lines
- [ ] **AUDIT 375/384** `lowerSymbolicVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3410`, line by line against the C++
- [ ] **PORT 376/384** `runOnOperation` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:439`, 34 lines
- [ ] **AUDIT 376/384** `runOnOperation` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:439`, line by line against the C++
- [ ] **PORT 377/384** `matchAndRewrite` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:44`, 12 lines
- [ ] **AUDIT 377/384** `matchAndRewrite` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:44`, line by line against the C++
- [ ] **PORT 378/384** `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1374`, 30 lines
- [ ] **AUDIT 378/384** `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1374`, line by line against the C++
- [ ] **PORT 379/384** `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:975`, 54 lines
- [ ] **AUDIT 379/384** `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:975`, line by line against the C++
- [ ] **PORT 380/384** `shallowlyMergeConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:198`, 76 lines
- [ ] **AUDIT 380/384** `shallowlyMergeConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:198`, line by line against the C++
- [ ] **PORT 381/384** `simplifyValueBasedConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:466`, 30 lines
- [ ] **AUDIT 381/384** `simplifyValueBasedConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:466`, line by line against the C++

## Level 9

- [ ] **PORT 382/384** `fuseLoadOrStoreChainOps` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:22`, 144 lines
- [ ] **AUDIT 382/384** `fuseLoadOrStoreChainOps` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:22`, line by line against the C++
- [ ] **PORT 383/384** `runOnOperation` — `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:77`, 80 lines
- [ ] **AUDIT 383/384** `runOnOperation` — `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:77`, line by line against the C++

## Level 10

- [ ] **PORT 384/384** `runOnOperation` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:169`, 81 lines
- [ ] **AUDIT 384/384** `runOnOperation` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:169`, line by line against the C++

## Excluded — 106 definitions, with the reason for each

⛔ Overrule any of these by moving it into a level above; the criterion is stated at the top.

### a one-line C++ field accessor; in Rust the field itself — 47

- `getOp` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:57`
- `getComp` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:58`
- `getMemRef` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:59`
- `getMemViewStartAddr` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:60`
- `getMemViewLayoutMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:61`
- `getLayoutCoeffs` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:62`
- `getTransferOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:63`
- `getChunkSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:64`
- `getChunkStride` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:65`
- `getShuffleMode` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:66`
- `getRotationPosition` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:67`
- `getTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:68`
- `getElementWidth` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:69`
- `getIndices` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:70`
- `getMemory` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:71`
- `getMemoryIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:72`
- `getExtents` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:73`
- `setOp` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:76`
- `setMemRef` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:80`
- `setMemory` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:84`
- `getExpectedTotalElements` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:88`
- `getTransferSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:89`
- `getLdOrStSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:90`
- `setChunkSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:102`
- `setChunkStride` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:103`
- `setLdOrStSize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:128`
- `getSubscriptsMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:221`
- `getIndicesCoeffDict` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:222`
- `getTimeOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:262`
- `getTimeSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:263`
- `getTimeSymbols` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:264`
- `getTimeAddrMap` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:265`
- `getTimeBounds` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:266`
- `getTimeOffsets` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:267`
- `getBurstIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:268`
- `getInterleaveGroupIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:269`
- `setTimeOrder` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:284`
- `setTimeSet` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:285`
- `setBurstIndex` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:298`
- `getStrides` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:352`
- `getTotalDataOriginsCount` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:33`
- `getFirstValue` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:76`
- `setOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:52`
- `isLoopNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:54`
- `isMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:55`
- `getStartVal` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:76`
- `getIncrement` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:77`

### a C++ data MEMBER, not a function — the extractor caught the declaration — 34

- `comp_` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:47`
- `index_mapping_` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:375`
- `dcc_ext_ctx_` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:38`
- `dcc_ext_ctx_` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:42`
- `dcc_ext_ctx_` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.hpp:27`
- `dominance_info_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:26`
- `size_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:208`
- `op_` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:79`
- `dcc_ext_ctx_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:45`
- `reuse_info_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:82`
- `is_visited_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:108`
- `increment_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:72`
- `dcc_ext_ctx_` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.hpp:53`
- `if_op_` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:56`
- `opts_` — `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:69`
- `dcc_ext_ctx_` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:47`
- `opts_` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:39`
- `dcc_ext_ctx_` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:54`
- `opts_` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:84`
- `mem_index_` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:94`
- `weight_` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:111`
- `opts_` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:61`
- `mem_index_` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:71`
- `weight_` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:83`
- `base_unit_program_` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:43`
- `dcc_ext_ctx_` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:55`
- `curr_unit_corelet_id_` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:91`
- `opts_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:34`
- `comp_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:30`
- `subscripts_map_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:45`
- `mem_index_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:50`
- `access_details_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:458`
- `comp_` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.hpp:24`
- `opts_` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:43`

### returns the MLIR pass context — scratchy has no pass context — 9

- `dccExtContext` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:521`
- `dccExtContext` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:91`
- `dccExtContext` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.hpp:38`
- `dccExtContext` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.hpp:51`
- `dccExtContext` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.hpp:59`
- `dccExtContext` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:94`
- `dccExtContext` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:66`
- `dccExtContext` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:219`
- `dccExtContext` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:125`

### MLIR pass construction — no pass pipeline exists at runtime — 6

- `createAffineToStandardPass` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:239`
- `createAgenToSentientPass` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:252`
- `createDataflowToSentientPass` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2051`
- `createSCFToSentientPass` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:285`
- `createStandardToSentientPass` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:476`
- `createSymbolToSentientPass` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:187`

### MLIR printing/parsing/verification — this crate emits and never reads back — 5

- `print` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:96`
- `parseConditional` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:534`
- `printUnitToOpsMap` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:189`
- `printEquivalenceClasses` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:202`
- `printTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:288`

### fills an MLIR RewritePatternSet — no pattern driver exists — 3

- `populateAffineToStdConversionPatterns` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:198`
- `populateAffineToVectorConversionPatterns` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:208`
- `populateDuplicateReusedTogglePatterns` — `dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:207`

### configures an MLIR pattern driver — scratchy has no driver — 1

- `runOnOperation` — `dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:222`

### a pass driver whose whole body is pipeline plumbing — 1

- `runOnOperation` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:455`

