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

## ⛔ MEASURED FOR ENTRIES 230-253: THE EXTRACT DROPS THE STATEMENT THAT DOES THE WORK

Brace-matched from the authority at each unit's cited line and diffed against `source/bridge2.cpp`'s
body — **not** re-sliced with the recorded `loc`, which is itself one short in every one of the 24.
**11 of 24 bodies lose at least one real statement**, and in three of them what is lost is the call that
makes the function have an effect:

| unit | dropped from the extract's body |
|---|---|
| **251 `setupForPartitioning`** | **the WHOLE body** — `DT_CHECK(isEligibleForSplitting(all_mem_views))`, `sortDataBasedOnWeight`, `calculatePartitionSizes` (`:825-829`) |
| **252 `fillPartitions`** | `return nullptr;`, the lambda's `};`, and `dcc::CondNode::walk<kReverseBFS>(cond_tree.getRoot(), addOpsToPartitions)` (`:1180-1184`) — the extract never RUNS the lambda it builds |
| **239 `insertPTMaskOps`** | `pt_masking_tree->walk(analyzeAndInsertMaskOps);` (`:205`) — same defect |
| 235 `createSentientConstants` | `rewriter.getI32ArrayAttr(extended_vals));` and `return sentient_vconst_op.getResult();` — the op's values AND its result |
| 236 `addMaskNode` | `op_to_node_.insert(std::make_pair(mask_related_op, mask_node));` — the node is built and never registered |
| 253 `adjustForEvenImmutableAddr` | `AffineMap::get`'s `getNumSymbols()`/`getContext()` arguments |
| 250 `initMASData` | `max_mutable += weight;` |
| 230 / 241 / 243 / 244 | `llvm_unreachable("unknown type string")` · `return new_for_op;` · `return ret.getResult(0);` · `return true;` |

Benign: 232, 233, 245 and 249 lose only a nested `}` / `});` that the extractor's brace-balancer put
back. Identical to the authority: 231, 234, 237, 238, 240, 242, 246, 247, 248.

## ⛔ MEASURED FOR ENTRIES 289-296: WHAT THE EXTRACT DROPS, AND WHY THE `calls` COLUMN IS NOT THE CALL GRAPH

Brace-matched from the authority at each unit's cited line (a0d29abbed) and diffed against
`source/bridge2.cpp`'s body. All eight `UNITS.tsv` citations land exactly on the named definition, and
the recorded `loc` is 1-6 lines short of the real body in all eight. **6 of the 8 extract bodies lose a
real statement, and in five of them it is the statement that gives the function its effect:**

| unit | dropped from the extract's body |
|---|---|
| **289 `initialize`** (`:694-705`) | `initMASData(mas_data, ad, max_mutable);` (`:703-704`) — the extract only `DT_CHECK`s |
| **290 `createPartitions`** (`:983-998`) | `dcc::ConditionalTree cond_tree(*root_op); cond_tree.compute();` and `fillPartitions(..)` (`:993-997`) — it computes `root_op` and throws it away |
| **291 `calculatePartialShift`** (`:490-558`) | `if (remainder != 0) offsetShifts(shifts, ad, -remainder);` and `return total_shift;` (`:556-557`) — an `int64_t` function left with no return, minus the stick-remainder correction |
| **293 `runOn`** (`:447-453`) | the walk's whole lambda, `[&](LoopLikeOpInterface loop_op) { analyzeAndTransform(loop_op); });` (`:452`) — `unit.walk<WalkOrder::PostOrder>(` is left with no argument: it walks and calls nothing |
| **294 `getPageValidity`** (`:114-148`) | `return page_sel_constraints;` (`:147`) — the VALID-page return; both invalid-page returns survive |
| **295 `createIterArgsForConditionals`** (`:401-524`) | `curr_loop->erase();` (`:522`), with the balancer emitting `}}` — clone-and-erase becomes clone-only, doubling the nest |
| 292 `transformSCFLoopWithNonConstantUpperBound` (`:169-263`) | body IDENTICAL — but the extract starts at the CONTINUATION line and lost `LogicalResult TransformLoopToLegalizeForSentientLowering::` (`:168`), so it carries neither a return type nor an `e292_` name |
| 296 `runOnOperation` (`:362-489`) | nothing: identical to the authority |

⛔ AND THE `calls` COLUMN IS SHORT FOR SEVEN OF THE EIGHT, including callees the truncation did NOT
take: 290 omits `e187_constructConditionals` (`:990`, present in the extract) as well as the truncated
`e252_fillPartitions`/`e247_compute`; 291 omits `e192_calculateDimWeights` (`:520`) and
`e254_offsetShifts`; 292 and 293 omit `e257_analyzeAndTransform` (`:245`/`:250`, `:452`); 294 omits
`e258_addConstraintsForIVRanges` (`:135`); 295 omits `e201_setLoopIteratorOrder` (`:421`) and
`e200_updateTPMVInfo` (`:519`); 296 lists 2 of its 8, missing `e140`, `e203`, `e204`, `e205`, `e262` and
`e263`; 289's one in-span callee, `e250_initMASData`, is the line the truncation took. Across all 384,
**172 columns omit a name that appears in their own extract body.**

⭐ THE `level` COLUMN IS UNAFFECTED, AND IT IS THE ONE THE WAVES USE. For all eight, level 3 is exactly
1 + the highest level among the TRUE callees (e250, e252/e247, e254, e257, e258, e264, e262/e263 — all
level 2), which the recorded `calls` column could not have produced: from it, 289 would be level 1 and
290/293 level 0. Order the work by `level`; take the callees from the authority.

## ⛔ MEASURED FOR ENTRIES 297-310: THE EXTRACT DROPS BOTH DISPATCH ARMS OF TWO FUNCTIONS

Brace-matched from the authority at each unit's cited line (a0d29abbed) and diffed against
`source/bridge2.cpp`'s body. All 14 `UNITS.tsv` citations land exactly on the named definition, and the
recorded `loc` is 1-6 lines short of the real body in all 14. **10 of the 14 extract bodies lose a real
statement, and in five of them it is the statement that gives the function its effect:**

| unit | dropped from the extract's body |
|---|---|
| **300 `lowerSyncForAUnit`** (`:733-745`) | **BOTH dispatch arms** — `if (src_unit_name.substr(0, 2) != "l3") return lowerL0LXSyncOperationForAUnit(..)` (`:740-742`) and `return lowerL3SyncOperationForAUnit(..)` (`:744`). The extract constructs an `OpBuilder` and ends |
| **301 `lowerSyncForAGroup`** (`:746-759`) | the same two arms, group-flavoured (`:753-758`) — same defect, and this function IS the L3-vs-LX decision |
| **298 `constructAffineDetailsAndAddrs`** (`:2787-2807`) | `return gatherAffineLoadStoreDetails<AccessDetailsAffine>(src_op, unit, comp, access_details, mutable_addrs, immutable_addrs);` (`:2805-2806`) — the tail call that gathers what the name promises |
| **308 `calculateShifts`** (`:396-460`) | `offsetShifts(shifts, ad, -num_elems_in_stick);` (`:457`) — the odd-immutable-address correction. Its `DT_CHECK_MSG` guard survives, so the extract checks the precondition and then does not shift |
| **299 `lowerAffineCompositeHelper`** (`:2953-2973`) | `to_be_deleted.push_back(candidate_op);` (`:2970`) and `return success();` (`:2972`) — the lowered op is never queued for deletion. It also loses the `"vector operations");` continuation, leaving the extract with an unterminated string literal |
| 302 `lowerSyncLXL3ToLXL3` (`:787-1718`) | `return LogicalResult::success();` (`:1715`) AND the fallthrough `return LogicalResult::failure();` (`:1717`) |
| 304 `getOperandWithPrecision` (`:389-646`) | both `return std::nullopt;` (`:643`, `:645`) — the balancer's `}}` leaves the `Multiply`/`MAC`/`Binary` arm an EMPTY block, so a case the reference explicitly refuses reads as a case it accepts |
| 297 · 309 · 310 | `return LogicalResult::success();` (`:669` · `:250` · `:921`) |
| 303, 305, 306, 307 | nothing: identical to the authority |

## ⛔ ENTRIES 304 AND 320 ARE A CYCLE, AND THE `level` COLUMN PUTS THEM ON DIFFERENT LEVELS

`e304_getOperandWithPrecision` (level 4, `VectorOperands.cpp:389`) and `e320_getOperand` (level **5**,
`:378`) call each other. e320 is a forwarder whose whole body is
`return getOperandWithPrecision(dcc_ext_ctx, op, comp, is_precision_converted, traverse_upwards);`
(`:382-383`), and e304 calls e320 back **nine times** (`:468`, `:478`, `:494`, `:504`, `:547`, `:557`,
`:575`, `:586`, `:630` — the recursion that walks a cast, a user or a select's operand). Both `calls`
columns read `-`, which is why no level computation saw the edge. A strict level ordering cannot hold
across a cycle: schedule the pair as one item, or cut the edge deliberately and say which way.

## ⛔ AND THE `calls` COLUMN FOR 297-310: SHORT FOR ALL 14, PLUS SIX EDGES THAT ARE NOT CALLS

Recomputed from the authority body with comments and string literals stripped and the signature skipped.
**All 14 omit at least one true callee**, and only entry 302's declared list happens to imply its
recorded level — for the other 13 the column implies level 0, 1 or 3 against a recorded 4. The level
column is right and was not computed from this column; take callees from the authority.

Omitted, by unit: 297 `e266_coalesceTimeDimensions`/`e148_computeBurstAndGroup`; 298
`e265_constructDetails`/`e212_gatherAffineLoadStoreDetails`; 299 (declares nothing)
`e267_constructTimeLoopsAndVectorOperations`/`e037_findCandidateForLowering`; 300 (nothing)
`e273`/`e223`; 301 (nothing) `e274`/`e224`; 302 `e273_lowerL0LXSyncOperationForAUnit`/`e223`; 303
(nothing) `e275_LowerSymbolQueryMap`/`e077_walk`; 304 (nothing) all eight of `e071`, `e072`, `e166`,
`e167`, `e168`, `e232`, `e278` and `e320`; 305 `e287_flatten`/`e077_walk`; 306 and 307 `e290_createPartitions`,
`e265_constructDetails`, `e251_setupForPartitioning`, `e253_adjustForEvenImmutableAddr`,
`e185_hasMutableAddrOverflow`; 308 `e291_calculatePartialShift`, `e254_offsetShifts`,
`e191_calculateFullShift`; 309 seven, including `e295_createIterArgsForConditionals`,
`e198_createConditionsForHyperRectSubscripts` and `e199_createConditionsForNonHyperRectSubscripts`; 310
`e294_getPageValidity`/`e260_gatherPageDependentDimsForPage`.

⛔ **AND SIX DECLARED EDGES ARE NOT CALLS AT ALL**, three of them because the "callee" is not a function:

- **`e052_If`** (declared by 308 and 309) — `StandardToSentient.cpp:159` is a COMMENT,
  `// return If(lhs) {If(rhs) true_val; else false_val} else false_val;`. There is no function `If`, and
  both citing bodies match only on `If` as the first word of an English comment sentence.
- **`e064_size`** (297, 302, 310) — `VectorChainHelper.cpp:319` is `size(vec.size())`, a MEMBER
  INITIALISER in a local struct's constructor. Every `.size()` in those bodies is a container's own.
- **`e019_AccessDetailsAffine`** (298, 306, 307, 308) — `AccessDetails.hpp:259` is
  `AccessDetailsAffineComposite`'s base-class mem-initialiser. Every match in those bodies is the TYPE
  NAME in a declaration or a template argument (`AccessContainer<AccessDetailsAffine>`).
- **`e133_getUseChain`** (306, 307) — `TransformPagedMemViewImpl.hpp:328` is
  `virtual SmallVector<Operation *> getUseChain(Operation *mem_op) { return {}; }`, a `TPMVBase` virtual
  taking ONE argument. 306/307 call `op.getUseChain()` with NO argument on an agen load/store op,
  generated from `Agen.td:172`/`:265`. Different function, same name.
- **`e150_get`** (297) — a real function (`AccessDetails.hpp:401`, `AccessContainer<T>::get`), but 297
  never calls it; its only member call is `access_details.size()`.

## ⛔ AND WHAT THE LANDED RUST CLAIMED ABOUT THESE 14 — 7 FIXED IN THIS COMMIT

35 in-span `.cpp` citations were re-measured against a0d29abbed (14 module-doc banner rows, all exact;
21 in prose). Three claims were wrong about behaviour, not just about a line:

- `vc_vector_operands.rs` said `splat_` is touched only by e304 and left its shape open. e304 is its
  only WRITER and writes exactly one value, `"east"` (`VectorOperands.cpp:582`); the only READER is
  `e340_analyzeAndFillOperandForwarding`, which feeds it to `symbolizeSentientComputePort` and keeps a
  `SentientComputePortAttr` (`VectorChainHelper.cpp:544-547`). The field is a `sen::Port`, not a string.
- `tf_mutable_addr_splitting.rs` justified `TimeOrderIsNotAPermutation` with "a permutation by
  construction (`constructTimeStepsInfo`)". Nothing constructs it: `time_order` is the composite op's
  own attribute (`op.getTimeOrder()`) and e297 only reads it (`AccessDetails.cpp:642`). The permutation
  property is a GATE — `checkBasicConditions` requires `inversePermutation` to succeed, and only for
  four of the six composite kinds (`AgenToSentient/Helper.cpp:166-175`).
- `dfs_dataflow_to_sentient.rs` said the caller "asks whether three of the four are empty while the
  fourth is not". `:798-803` tests only the three empties; that the remaining list is non-empty comes
  from the `DT_CHECK` at `:796-797`, and the group case is a separate arm at `:825-826`.

Four citations drifted: `VectorOperands.hpp:112` (a blank line) for `splat_`'s declaration at `:70`;
`TransformPagedMemViewImpl.cpp:190-196` (the signature) for `insert_refs = mem_ops_` at `:201`;
`DataflowToSentient.cpp:796-800` in `islands/dataflow_ir/dialects/dataflow.rs` for the group arm at
`:825-826`; and `Helper.cpp:2963` for a quote that is the comment on `:2962`.

## ⛔ MEASURED FOR ENTRIES 311-326: ONE EXTRACT BODY IS EMPTY, AND ONE ENTRY IS SCHEDULED AHEAD OF ITS CALLEE

Brace-matched from the authority at each unit's cited line (a0d29abbed) and diffed against
`source/bridge2.cpp`'s body. All 16 `UNITS.tsv` citations land exactly on the named definition, and the
recorded `loc` is 1-6 lines short of the real body in all 16. **14 of the 16 extract bodies lose a real
statement, and in five it is the statement that gives the function its effect:**

| unit | dropped from the extract's body |
|---|---|
| **320 `getOperand`** (`VectorOperands.cpp:378-384`) | **THE WHOLE BODY.** The function is `bool is_precision_converted; return getOperandWithPrecision(dcc_ext_ctx, op, comp, is_precision_converted, traverse_upwards);` and the extract keeps only the declaration, so a 7-line forwarder reads as a function that computes nothing. This is the second reason the 304↔320 cycle below is invisible: the extract of 320 contains no call at all |
| **319 `lowerSyncForAQueryMap`** (`:1728-1899`) | **ITS FINAL DISPATCH ARM** — `DT_CHECK_MSG(implicit_sync_tile_size == -1, "L3 doesn't have implicit sync"); return lowerSyncLXL3ToLXL3(op, builder, src_vs, dst_vs, -1, /*is_src_l3 =*/true);` (`:1894-1897`). The src-is-L3 case, its precondition and its tail call are all gone; same defect as 300/301 |
| **323 `shiftMutableAddr`** (`:366-388`) | BOTH returns — the all-shifts-zero early exit `return AffineMap::get(mem_view_op->getContext());` (`:385`, the empty map that reports "no shift") and `return applyShifts(evaluator, shifts, unit, op, mem_view_op, ad);` (`:387`). The extract calculates shifts and discards them. It also drops the `const` qualifier from the signature |
| **311 `constructAffineCompDetailsAndAddrs`** (`:2809-2847`) | `.failed()) return failure();` (`:2842-2843`) and `return gatherAffineLoadStoreDetails<AccessDetailsAffineComposite>(src_op, unit, comp, access_details, mutable_addrs, immutable_addrs);` (`:2845-2846`) — same tail call 298 lost, composite-flavoured |
| **314 `lowerVectorLoadOp`** (`:3050-3072`) | `return lowerVectorLoadHelper<AccessDetailsAffine, VectorLoadOp>(candidate_op, store_op, unit, access_details, mutable_addrs, immutable_addrs, to_be_deleted);` (`:3069-3071`) — the helper that does the lowering |
| 318 `lowerLDCVTIPattern` (`:3444-3773`) | `to_be_deleted.push_back(element_shuffle_op);` and `to_be_deleted.push_back(element_load_op);` (`:3770-3771`) plus `return success();` — the fused ops are never queued for deletion |
| 312 · 313 · 315 · 316 · 317 · 324 | `return success();` / `return LogicalResult::success();` (`Helper.cpp:3023` · `:3047` · `:3103` · `:3214` · `:3264`, and 324 at `TransformPagedMemViewImpl.cpp:185`) |
| 321 · 322 | the `<< "\n");` tail of the closing `LLVM_DEBUG` (`:543` · `:670`), leaving an unterminated stream expression. Every statement that acts survives in both |
| 325, 326 | nothing: identical to the authority |

## ⛔ ENTRY 315 IS SCHEDULED TWO LEVELS AHEAD OF ITS OWN CALLEE — AN OVERLOAD, RESOLVED WRONG

`constructReceiveAndStoreStmt` and `constructLoadAndSendStmt` each have TWO declarations in
`AgenToSentient.hpp`: a primary template with five trailing defaults (`:221-227`, `:239-245` — defined in
`Helper.cpp:1910` = **e358** and `Helper.cpp:2025` = **e359**, both level **7**), and an inline forwarder
whose last parameter `Operation* extract_op` is REQUIRED (`:229-237`, `:247-255` — **e027**/**e028**, both
level **0**). Which one a call binds is decided by its argument count:

- **315** calls `constructReceiveAndStoreStmt<AccessDetailsAffine>(&builder, unit, candidate_op,
  element_type, access_details[0], mutable_addrs[0], immutable_addrs[0])` — **seven** arguments
  (`Helper.cpp:3090-3092`). The forwarder needs eight, so it is not viable; the call resolves to the
  primary and therefore to **e359, level 7**. The `calls` column declares `e028` (level 0). **315 is
  level 5 and calls a level-7 unit** — port it after 359, or the callee will not exist.
- **316** passes `extract_op` as its seventh argument (`:3202-3204`) and **317** as its eighth
  (`:3251-3253`); the primary's corresponding parameter is `unsigned burst_size`, which an `Operation*`
  cannot convert to, so both DO bind the level-0 forwarders. `e027`/`e028` are correct for those two, and
  the column simply copied 317's answer onto 315.

`e320_getOperand` is the same 304↔320 cycle already recorded above, seen from the level-5 side: its
declared `calls` is `-` because the extract left it with no body to read.

## ⛔ AND THE `calls` COLUMN FOR 311-326: 26 OF ITS 39 EDGES ARE NOT CALLS

Recomputed from the authority body with comments and string literals stripped and the signature skipped.
**All 16 omit at least one true callee, and in every one of the 16 the omitted set contains the callee
that fixes the level** — no declared list implies the recorded level 5; they imply 0, 1, 2 or 3. Three
entries — **318, 323 and 324** — have NO true edge in their declared list at all. The `level` column is
right for 15 of the 16 (315 is the exception above) and was not computed from this column; take callees
from the authority.

Of the 39 declared edges, 12 are real calls, one names the wrong overload (315's `e028`) and **26 are not
calls**:

- **`e019_AccessDetailsAffine`** (312, 313, 314, 315, 316, 317, 318, 323 — 8 of the 16, its worst run) —
  and it is wrong twice over. `UNITS.tsv` gives it `loc` 0 and a one-line extract span, because
  `AccessDetails.hpp:259` is a MEM-INITIALISER: `explicit AccessDetailsAffineComposite(Operation* op,
  SenComponents comp) : AccessDetailsAffine(op, comp) {}`. So the entry is named after the BASE class in
  a delegation while the entity is the DERIVED class's constructor — and in all 8 bodies the match is
  only the type NAME as a template argument (`AccessContainer<AccessDetailsAffine>`,
  `constructLoadAndSendStmt<AccessDetailsAffine>`) or a parameter type (323's `agen::AccessDetailsAffine
  &ad`). Nothing in these 16 constructs an `AccessDetailsAffineComposite`; where one IS constructed it is
  inside `e208_emplace_insert`.
- **`e064_size`** (312, 313, 315, 316, 317, 318, 319, 324, 325, 326 — 10 of the 16) — `size(vec.size())`,
  a member initialiser (`VectorChainHelper.cpp:319`). Every `size()` in these bodies is a container's own.
- **`e052_If`** (318, 322, 323, 325) — the word "If" starting a comment: *"// on. If ldtype_shuffle
  existed…"* (`Helper.cpp:3733`), *"// If the indirect part…"* (`MutableAddrSplitting.cpp:575`), *"// If
  all the shifts are 0…"* (`MutableStartAddrShifting.cpp:381`), *"// If the TPMVInfo isn't for a paged mem
  view…"* (`TransformPagedMemViewImpl.cpp:997`). There is no function `If`.
- **`e026_has`** (318, 319) — a COMMENT in 318 (*"…that has a vector_load input"*, `:3516`) and two STRING
  LITERALS in 319 (*"sync buffer size has to be a constant op"*, *"the target has to be a single unit."*).
- **`e150_get`** (318, 319) — a real function (`AccessDetails.hpp:401`, `AccessContainer<T>::get`), but
  neither body has a single member `get(`: every match is an MLIR static builder — `IndexType::get`,
  `ArrayAttr::get`, `SentientSyncModeAttr::get`, `SentientLoadConsumerAttr::get`.

Omitted true callees, by unit: 311 `e265_constructDetails`/`e297_constructTimeStepsInfo`/
`e212_gatherAffineLoadStoreDetails`; 312 `e298_constructAffineDetailsAndAddrs`/`e037_findCandidateForLowering`/
`e269_constructLoadAndExtractScalarOp`; 313 the same two plus `e216_constructReceiveAndExtractScalarOp`;
314 `e298`/`e037`/`e036_getStoreOpFromLoadStorePattern`/`e217_lowerVectorLoadHelper`; 315 `e298`/`e037`/
`e270_addStoreInputToDeleteList` (and `e359` for `e028`); 316 `e298`/`e037`/`e032_findExtractScalarOp`/
`e038_addLoadChainToDeleteList`; 317 `e298`/`e037`/`e032`/`e270`; 318 `e298`/`e035_generateSetSendDestinationStmts`/
`e214_setImmutableAddrAndIncrements`; 319 `e302_lowerSyncLXL3ToLXL3` — the only in-span callee it has, and
the one that sets its level; 320 `e304_getOperandWithPrecision`; 321 and 322 `e289_initialize`/`e265`/`e297`/
`e290_createPartitions`/`e251_setupForPartitioning`/`e189_synthesizeTimeInfo`/`e185_hasMutableAddrOverflow`/
`e253_adjustForEvenImmutableAddr`/`e190_createExplicitTimeLoops`; 323 `e308_calculateShifts`/`e255_applyShifts`;
324 `e309_constructValidPage`/`e294_getPageValidity`; 325 `e310_analyzeValidPages`/`e197_calculateIndicesRanges`/
`e132_identifyTimeDimForExplicitLoops`/`e119_replaceDimsInMapWithSyms`/`e131_addTimeDimIndicesRanges`;
326 `e265`/`e297`.

## ⛔ AND WHAT THE LANDED RUST CLAIMED ABOUT THESE 16 — 4 FIXED IN THIS COMMIT

66 in-span `.cpp` citations were re-measured against a0d29abbed: 16 module-doc banner rows and the 32
checkbox rows below, all exact, plus 18 in prose. Two claims were wrong, two citations drifted:

- `agen_agen_to_sentient.rs` called 314 and 315 "level 8". `UNITS.tsv` records both at level **5**. The
  note now says 5 and carries the 315 inversion above, which is the schedule's error and not the module's.
- `agen_helper.rs` said `findCandidateForLowering` is "instantiated with **eleven** different classes" and
  then cited **twelve** sites. The twelve carry **ten** distinct classes: `VectorLoadOp` appears at
  `:3013` AND `:3063`, `VectorStoreOp` at `:3038` AND `:3086`. (The two symbolic lowerings use neither
  template for their candidate — `lowerSymbolicVectorLoadOp` takes it off
  `access_details.get(kDirSrc).getOp()` with a `dyn_cast_or_null`, `Helper.cpp:3393-3394`.)
- `tf_mutable_addr_splitting.rs` cited `:451-546` for 321's `initialize` → `synthesizeTimeInfo` →
  `hasMutableAddrOverflow` gate. 321's body ends at `:544` and `:546` is 322's first line; the sequence is
  `:475-479`.
- `tf_transform_paged_mem_view_impl.rs` quoted one comment for both of its callers. `:603-605` reads
  *"…valid for mem_ops_."*; `:1002-1004` reads `mem_op_`, singular.

Everything else in prose measured exact, including the four `findExtractScalarOp` instantiations and
their four error-report ranges (`:3195`/`:3196-3199`, `:3243`/`:3244-3247`, `:3293`/`:3294-3297`,
`:3339`/`:3340-3343`), 321's `:488-490`, `:512-515`, `:526-529` and `:533`, the four
`getBytesPerStick() * 8 / ad.getElementWidth()` sites in `MutableStartAddrShifting.cpp`
(`:373`, `:405`, `:485`, `:553`), and `DataflowToSentient.cpp:1838-1860` as the merged-versus-split pair.

## ⛔ MEASURED FOR ENTRIES 327-350: THREE PREDICATES LOSE `return true;` AND FOUR LOSE THEIR DISPATCH TAIL

Brace-matched from the authority at each unit's cited line (a0d29abbed) and diffed against
`source/bridge2.cpp`'s body. **22 of the 24 extract bodies lose real content** (339 and 340 lose only
nesting braces the extractor's balancer put back), and in ten of them what is lost is the statement
that gives the function its effect:

| unit | dropped from the extract's body |
|---|---|
| **343 `getOperandFromCastOp`** (`VectorOperands.cpp:354-361`) | **ITS ENTIRE COMPUTATION** — `auto operand = getOperand(dcc_ext_ctx, parent, comp); return operand;` (`:359-360`). An 8-line forwarder that walks to the cast's parent and re-asks for its operand reads as a function that finds the parent and returns nothing. Same defect as `e320_getOperand` above, and the same call it loses |
| **341 `analyzeNonComputeOpsForFusion`** (`:610-702`) | **BOTH ASSIGNMENTS THAT REPORT THE ANSWER** — `is_fusion_respected = false;` at `:696` AND at `:699`, with the `} else if (!to.has_value()) {` that guards the second (`:698`). The extract keeps every `dominates` query and discards both verdicts, so the out-parameter is never written on the failing path |
| **346 `lowerDanglingNonComputeOps`** (`:882-973`) | **THE WHOLE ERASE LOOP** — `for (auto op : tobe_deleted) { VectorOperand::eraseOp(op); }` (`:970-972`). The extract collects the dangling ops it lowered and never deletes them; the ops it replaced stay in the unit |
| **335 `insertCopyAndAddStmts`** (`:3861-3877`) | ONE OF ITS TWO DISPATCH ARMS plus its fallthrough — `return insertCopyAndAddStmtsHelper<scf::ForOp>(scf_for, index, imm_val);`, the `} else` and `llvm_unreachable("unhandeled type of loop")` (`:3874-3876`). The `affine::AffineForOp` arm survives, so the extract reads as an affine-only helper |
| **345 `processXrfPtrPerUnit`** (`:337-528`) | `return vector_op_to_xrfptr_map;` (`:527`) — 190 lines build the map from vector op to XRF pointer and the extract discards it |
| **338 `ConstructIFRecursively`** (`:113-241`) | ITS FAILURE ARM — `signalPassFailure(); return nullptr;` (`:238-239`), after the *"CMPI/AND/OR operations or an op with one return value"* diagnosis. The extract reports and then falls off the end of a function returning `Operation *` |
| **337 `lowerSyncOperation`** (`:1901-1982`) | `return LogicalResult::failure();` (`:1981`) — the fallthrough under all three `lowerSyncForA*` dispatch arms, so an unmatched sync reads as a success |
| **347 · 348 · 349** | `return true;` (`:310` · `:516` · `:747`). All three are **predicates whose only `true` is the last line**: `topLevelConditionsMatch`, `singleOpBranchToYieldVal`, `isLoopInvariant`. In the extract each one falls off the end after its `return false;` guards, so all three read as always-false — and 349 is recursive, so a port from the extract makes its own recursion vacuous |
| **342 `analyzeAndFillResultForwarding`** (`.hpp:162-194`) | the `else` half of its dispatch — `context, symbolizeSentientComputePort(dest).value());` closes the logical-port assignment, then `else result_forwarding.push_back(SentientComputePortAttr::get(context, symbolizeSentientComputePort(dest).value()));` (`:189-192`) is gone. The extract fills the logical port and never fills the vector |
| **327 `gatherSymbolicLoadStoreDetails`** (`:1051-1207`) | its diagnosis string `"Unable to construct immutable addresses"` (`:1204`, leaving `emitError(` unterminated) and `return success();` (`:1206`) |
| **344 `lowerDanglingNonComputeOpsPESFP`** (`:1273-1368`) | `return result;` (`:1368`) — **and its RETURN TYPE.** `UNITS.tsv` cites `:1274`, which is the qualified name; `mlir::LogicalResult` is alone on `:1273` and outside the span. Both ends of the signature-to-return contract are missing |
| **350 `matchAndRewrite`** (`DuplicateReusedToggle.cpp:33-205`) | `return success();` (`:204`) and the `const` qualifier from `PatternRewriter &rewriter) const {` (`:35`) — the same `const` loss as 323 |
| 328 · 332 · 333 · 336 | `return LogicalResult::success();` / `return success();` (`:1574` · `:3310` · `:3356` · `:4081`) |
| 329 · 330 · 331 · 334 | the ARGUMENTS of the `lowerAffineCompositeHelper<…>` tail call — `candidate_op, unit, access_details, mutable_addrs, immutable_addrs, to_be_deleted);` (`:3123-3124` · `:3144-3145` · `:3165-3166` · `:3376-3377`). The template name survives with an empty argument list |
| 339 · 340 | nothing that acts: 339 is identical to the authority plus a blank line, 340 loses four nesting `}` its balancer replaced with `}}}}` |

⛔ **AND `loc` MEASURES THE EXTRACT, NOT THE FUNCTION.** For all 24, `loc` is exactly the recorded
extract span minus 2, and so 2 to 7 lines short of the authority body (327: 153 vs 157; 341: 86 vs 93;
342: 27 vs 33). All 24 landed banner rows mirror `loc`, so each states a length short by the amount the
extract truncated — the numbers are consistent with `UNITS.tsv` and none is a length of the real
function. Campaign-wide, `loc == span - 2` for **356 of 384**.

## ⛔ FOUR OF THESE 24 ARE NOT IN THE EXTRACT UNDER THEIR OWN NAME — AND 118 OF 384 ARE NOT

The extract rewrites each definition's name to `eNNN_<cppName>`, which is how a worklist item is found
in it. **118 of 384 are never rewritten**, in four mechanically distinct classes; four of them are in
this span, one per class:

| class | count | in this span |
|---|---|---|
| the return type shares the line with `Class::name` | 7 | **338** — `Operation *StandardToSentientLoweringPass::ConstructIFRecursively(` (`:113`). The rewrite expects the qualified name to start the line |
| the citation points one line PAST the return type | 11 | **344** — `UNITS.tsv` says `:1274`, and `mlir::LogicalResult` is on `:1273`. The extract's span starts inside the signature |
| an unqualified free or `static` function | 29 | **347** — `static bool topLevelConditionsMatch(…)` (`:279`); nothing to strip a class from |
| a template, whose `template <…>` header is above the cited line | 71 (indented in-class) + this | **342** — `VectorChainHelper.hpp:162` is `void analyzeAndFillResultForwarding(` and `template <typename BuilderType>` is on `:161`. The extract shows a plain function taking a `BuilderType&` that no longer has a declaration |

⛔ **AND 17 OF 384 HAVE `loc=0` — THE ENTRY NAMES SOMETHING THAT IS NOT A FUNCTION.** Two of them
matter to this span, because the `calls` column below declares them as callees — with a third that is
not a function either, though its `loc` is not 0:

- **`e052_If`** (`StandardToSentient.cpp:159`, level 0) is a **COMMENT LINE** —
  `// return If(lhs) {If(rhs) true_val; else false_val} else false_val;` — and it sits INSIDE
  `e338_ConstructIFRecursively`'s own body (`:113-241`), which is entry **338** of this batch. There is
  no function `If`.
- **`e064_size`** (`VectorChainHelper.cpp:319`, `loc` 6) is the member initialiser `size(vec.size()) {`;
  its six lines are a constructor's tail.
- **`e227_OperandReuse`** (`OperandReuse.hpp:30`, `loc` 0) is a constructor declaration, cited where the
  previous batch found it named after the constructor and cited at the destructor.

The other fifteen are base-class mem-initialisers (`e046_ConversionPattern` —
`: mlir::ConversionPattern(mlir::scf::ForOp::getOperationName(), 1, ctx) {}`, `SCFToSentient.cpp:70`;
`e079`, `e080`, `e086`, `e087`, `e101`, `e106`, `e136`-`e138`, `e246`, and `e016`/`e019` already
recorded) and pure-virtual declarations with no body (`e134_cloneUseChain`, `e135_eraseMemOpAndUseChain`,
`TransformPagedMemViewImpl.hpp:341`, `:364`).

## ⛔ AND THE `calls` COLUMN FOR 327-350: 37 OF ITS 46 EDGES ARE NOT CALLS

Recomputed from the authority body with comments and string literals stripped, the signature skipped and
each name resolved against its receiver's type. Of the 46 declared edges, **6 are real calls, 3 name the
wrong definition of a shared name, and 37 are not calls at all.** Only **5 of the 24** — 327, 335, 338,
341, 350 — declare a single true callee; the other **19 declare none**.

The 37 are the same substring classes the previous batches measured, plus two new ones:

- **`e064_size`** (327, 329, 330, 331, 334, 336, 337, 345, 348, 349, 350 — 11 of the 24) — every match
  is a container's own `.size()`: `access_details.size()`, `bb.getOperations().size()`,
  `iter_arg_chain.size()`.
- **`e052_If`** (327, 328, 336, 337, 338, 341, 350 — 7 of the 24) — the word "If" opening a comment, and
  the entry is itself a comment (above). 328 alone has seven such matches (`Helper.cpp:1477`, `:1479`,
  `:1482`, `:1489`, `:1541`, `:1543`, `:1545`).
- **`e150_get`** (338, 340, 342, 344, 345, 346, 347 — 7) — MLIR static builders in six of them
  (`SentientComputePortAttr::get`, `ArrayAttr::get`, `CmpIPredicateAttr::get`, `IndexType::get`), and in
  **347** it is `std::get<0>(pair)` / `std::get<1>(pair)` (`:288-289`). Not one is
  `AccessContainer<T>::get`.
- **`e026_has`** (327, 328, 337, 344, 346, 350 — 6) — English in comments (*"has been updated"*,
  *"it has been used but not absorbed"*) and in **337** and **344** a STRING LITERAL: *"Src unit types
  has to be the same."* (`:1930`), *"Dangling non-compute op has no use\n"* (`:1344`).
- **`e227_OperandReuse`** (341, 344, 346) — the PARAMETER TYPE `OperandReuse &reuse_info` (`:612`,
  `:1276`, `:884`). Nothing in the three constructs one.
- **`e149_insert`** (328, 345) — 🆕 comments only: *"insert an add op"* (`:1543`, `:1545`), *"insert xrf
  offset ops"* (`:393`, `:438`, …). 345's real insertions are `insertConstAndAddOps` and
  `insertDummyMacOp`, which are **e173** and **e243** and are not declared. 327's `e149_insert` IS real
  (`mem_view_start_addrs.insert(…)`, `:1188`, `:1194`).
- **`e058_dominates`** (328) — 🆕 the comment *"Note: The extract_op dominates the indirect op."*
  (`:1476`). 341's `e058` is real (`reuse_info.dominates(left, right)`, `:656`).

⛔ **AND THREE EDGES NAME THE WRONG DEFINITION OF A SHARED NAME** — the column is keyed on the name, so
where two units share one it declares both or picks either:

- **338** declares `e045_getSentientCmpIPredicate` AND `e048_getSentientCmpIPredicate` for the SINGLE
  unqualified call at `:130`. e045 is `SCFToSentient.cpp:33`, e048 is `StandardToSentient.cpp:36`; 338 is
  in `StandardToSentient.cpp`, so the call is **e048** and e045 is not reachable from it.
- **349** declares `e082_getFirstChild` AND `e103_getFirstChild` — and **both are wrong.** e082 is
  `LoopMaskNode::getFirstChild` and e103 is `LocalOpNode::getFirstChild`; 349's receivers are
  `then_node`/`else_node` of type `CondNode*`, whose `getFirstChild` is `Analysis/ConditionalTree.hpp:49`
  — **outside the D1-D28 span and not one of the 384.** 349's only in-span call is its own recursion.
- **350** is one of FOUR units named `matchAndRewrite` (e001 L0, e225 L2, e350 L6, e377 L8). Its column
  happens not to declare one; nothing in the column could tell them apart if it did.

⭐ **WHAT THE LEVEL COLUMN GETS RIGHT, AND WHY IT IS NOT FROM THIS COLUMN.** Level 6 needs a level-5
callee. **13 of the 24 have one** — 329-334 through `e311_constructAffineCompDetailsAndAddrs`
(`Helper.cpp:3111`, `:3132`, `:3153`, `:3275`, `:3321`, `:3364`), 337 through
`e319_lowerSyncForAQueryMap` (`:1977`), and 340, 341, 342, 343, 344, 346 through
`e320_getOperand` — the `VectorOperand::getOperand(dcc_ext_ctx, …)` free function, distinguished from
MLIR's `Operation::getOperand` by its first argument (`:554`, `:618`/`:667`, `:169`, `:359`, `:1288`,
`:897`). **Every one of those 13 edges is omitted from the `calls` column**, whose deepest declared
callee anywhere in the 24 is level 2. The remaining **11 are over-levelled**: 327 (deepest real callee
`e213_constructImmutableAddress`, L2 ⇒ L3), 345 (`e243_insertDummyMacOp`, L2 ⇒ L3), 350 (`e264`, L2 ⇒ L3),
335 (`e029`, L0 ⇒ L1), 338 (`e048`, L0 ⇒ L1), and 328, 336, 339, 347, 348, 349 with no in-span callee at
all (⇒ L0). **This errs safe** — unlike 315 above, nothing in this batch is scheduled ahead of a callee;
the 11 are merely scheduled later than they need to be.

Omitted true callees, by unit: 327 `e155_updateSymbolicAccessDetails`/`e213_constructImmutableAddress`;
329, 330, 331, 334 `e311`/`e037_findCandidateForLowering`/`e299_lowerAffineCompositeHelper`; 332 and 333
`e311`/`e037`/`e032_findExtractScalarOp`/`e267_constructTimeLoopsAndVectorOperations`; 337
`e300_lowerSyncForAUnit`/`e301_lowerSyncForAGroup`/`e319`; 340 and 342 `e320`/`e169_getName`; 341 and 343
`e320`; 344 `e320`/`e169`/`e075_eraseOp`/`e055_getId`/`e056_getAbsorbtionFlag`/
`e060_getInputPrecisionFromOperand`; 345 `e090_getXrfValue`/`e091_getForOpBound`/`e173_insertConstAndAddOps`/
`e174_isXrfRelated`/`e175_updateYieldArgs`/`e243_insertDummyMacOp`; 346 those six of 344 plus
`e094_computeUnitPrecision`/`e282_updateLoopMaskTreeForConstantMask`. 328, 335, 336, 338, 339, 347, 348,
349 and 350 omit nothing — they have no undeclared in-span callee. **15 of 24 omit at least one.**

## ⛔ AND WHAT THE LANDED RUST CLAIMED ABOUT THESE 24 — 4 FIXED IN THIS COMMIT

56 in-span `.cpp` citations were re-measured against a0d29abbed: the 24 module-doc banner rows and the
48 checkbox rows below, all exact, plus 8 in prose. **No unit in this span is ported** — 0
`/// Replaces:` anchors, 0 surviving `// crustify:todo:`, all 48 PORT/AUDIT boxes `[ ]`, which is what
the schedule records. What the landed Rust claims ABOUT them, however, had four defects:

- ⛔ **`ExtractScalarOp` GAVE THE RECEIVE FLAVOUR A RESULT IT DOES NOT HAVE.** `agen_helper.rs` carried
  `pub addr: Val` and `pub data: Val` for both kinds under *"IT IS THE OP'S TWO RESULTS THAT MATTER"*.
  `Sentient_LoadAndExtractScalarOp` has two (`let results = (outs Index:$addr, Index:$data);`,
  `SentientOps.td:622`, with `getAddrResult()` = 0 and `getDataResult()` = 1) — but
  `Sentient_ReceiveAndExtractScalarOp` has **one** (`let results = (outs Index:$result);`, `:684`), and
  it is the DATUM. `e328_adjustMutableAddrInitForIndirect` is where the difference shows: it reads
  `load_and_extract_op.getDataResult()` for the load and `extract_op->getResult(0)` for the receive
  (`Helper.cpp:1467-1470`) — result **1** in one case and result **0** in the other. A mandatory `addr`
  field made result 0 of a receive an address, which is the datum. The results are now an enum carrying
  what each op actually binds, with one accessor for the value the indirect access consumes.
- ⛔ **THE `agen.composite_load_and_store` EMISSION IS NOT `e331`'s.** `agen_agen_to_sentient.rs` said
  *"THE EMISSION IS `e331_lowerCompositeLoadAndStoreOp`'s"*. `e331` (`Helper.cpp:3148-3167`) emits
  nothing: it builds details, re-finds the candidate and tail-calls
  `lowerAffineCompositeHelper<CompositeLoadAndStoreOp>` (`:3164-3166`) = **e299** (`:2953`, level 4),
  which emits nothing either and calls `constructTimeLoopsAndVectorOperations` (`:2959`) = **e267**
  (`:1789`), which dispatches `CompositeLoadAndStoreOp` to `constructLoadAndStoreStmt` (`:1882`) =
  **e268** (`:2167`), where `sentient::LoadAndStoreOp::create` is (`:2323`). The emission is **e268's,
  four calls below `e331`**.
- ⛔ **NEITHER `e327` NOR `e374` BRANCHES ON `has` TO TELL AN INDIRECT ACCESS FROM A DIRECT ONE.**
  `agen_access_details.rs` said so twice (on `has` and on `get`). `e327_gatherSymbolicLoadStoreDetails`
  does not call `has` at all — its three matches are English in comments. `e374_lowerSymbolicVectorLoadOp`
  calls it once, `has(kDirDst)` (`Helper.cpp:3399`), and what that asks is whether the load has a paired
  STORE. The function that does branch indirect-vs-direct is **`e268_constructLoadAndStoreStmt`**, on
  `has(kIndSrc)`/`has(kIndDst)` at `Helper.cpp:2187`, `:2200`, `:2213`, `:2220`, `:2257`, `:2259` and
  `:2337` — the only seven `kInd` `has` sites in the tree.
- ⚠️ **"WITH NOTHING IN BETWEEN" OVERSTATES `e328`.** `agen_helper.rs` justified the 1D-identity,
  based-at-0 view by saying `e328` *"adds it to the mutable address with nothing in between"*. `e328`
  inserts a `sentient::AddOp` at one of **four** sites, and only the first adds to the mutable address
  itself (`:1496`); the iter_arg-chain walk adds the extracted scalar to the loop's INITIALISER instead
  (`:1522`, `:1533`, then `curr_loop->setOperand(arg_idx, new_init)`), and the non-iter_arg case walks
  back to whichever of the two ops comes last before inserting (`:1569`). What is unscaled and
  unoffset is the extracted scalar, not the address it joins — the vendor's own output shows
  `%34 = sentient.scalar_add %33, %22` where `%33` is an `arith.addi` from the address chain
  (`lx_indirect_loads_stores_composite.mlir:42-43`).

Everything else in prose measured exact, including both `lx_indirect_loads_stores_composite.mlir`
citations (`:34` binds `%21, %22 = sentient.load_and_extract_scalar` and `:43` consumes `%22` — the
`AgenToSentient/` copy, not the `SentientToProgIR/` one of the same basename), `e335`'s two
`insertCopyAndAddStmtsHelper` instantiations (`Helper.cpp:3871`, `:3874`) and the index-from-the-end rule
they carry, and the four `findExtractScalarOp` error ranges already re-measured for 311-326.

## ⛔ MEASURED FOR ENTRIES 351-356: ALL FOUR SHIFTING TRANSFORMS LOSE `op->erase();`, AND 351 IS 372

Re-measured body-by-body against the authority at `a0d29abbed`. **351 and 356 are identical to it**
modulo the `eNNN_` rename. The other four each lose exactly one statement and it is the SAME statement:
`op->erase();` — `MutableStartAddrShifting.cpp:226` (352), `:253` (353), `:295` (354), `:352` (355) —
the last line of each body and the only one that removes the operation being replaced. In 353, 354 and
355 the clone is `(void)`-discarded, so that erase was the ONLY mutation of the existing IR those three
perform: as extracted they compute a shifted subscripts map, build a clone nothing reads, and leave the
original transfer in place. All four also drop the `const` member qualifier (`:201-202`, `:229-230`,
`:256-257`, `:298-299`), the same drop already recorded for 323.

⭐ `loc` IS EXACT FOR ALL SIX under the column's own convention — authority body lines minus the
signature lines: 71-1=70 (351), 27-2=25, 26-2=24, 41-2=39, 56-2=54, 40-1=39 (356). All six cited
`file:line`s land on the first line of the signature.

⭐ **351 AND 372 ARE THE SAME FUNCTION.** Diffed whole (`MutableAddrSplitting.cpp:226-296` against
`MutableStartAddrShifting.cpp:130-198`) they differ in four things and nothing else: the candidate type
(`MASCandidate` `:91-100` versus `MSASCandidate` `:68-77` — field-for-field identical, same
constructor), 351's per-unit `num_conditionals_ = 0` reset (`:244-246`, which the shifting pass has no
counterpart for because it has no conditional budget), the one-use check's form (`DT_CHECK` at
MAS`:255` versus `DT_CHECK_MSG(…, "Expecting L3 memory views to be used in one memory operand.")` at
MSAS`:156-158`), and the four dispatch targets. ⛔ AND `DisableThisPass` IS A SEPARATE `cl::opt` IN
EACH FILE — `-dcc-mutable-addr-splitting-disable` (MAS`:51-54`) versus
`-dcc-mutable-start-addr-shifting-disable` (MSAS`:41-45`), the same trap already recorded for
`MaxImmutableSize`. One shared Rust collector must take the flag, the reset and the message as
parameters, or one flag will disable both passes. ⭐ AND THE ONE BODY SITS AT TWO LEVELS — 351 is 6
(arms 306/307 at 4, 321/322 at 5) and 372 is 7 (arms 352-355 at 6, because 352 and 353 reach
`e323_shiftMutableAddr` at 5). Both are right; the depth is the arms', not the dispatcher's.

⛔ **AND THE EXTRACT SENDS 351'S DISPATCH NOWHERE.** Its four calls are bare
`transformVectorLoad(candidate)` … `transformCompIndLoadAndStore(candidate)`, and
`prelude.inc:2263-2268` declares empty stand-ins under exactly those four names while the extract's
eight real bodies all carry `eNNN_` prefixes. So in the extract 351 dispatches to no-ops, and nothing
in it says WHICH four it means: they are 306/307/321/322 in `tf_mutable_addr_splitting.rs` (MAS`:286`,
`:288`, `:290`, `:292`), not the identically named 352-355 of this worklist, which belong to 372
(MSAS`:189`, `:191`, `:193`, `:195`).

## ⛔ ENTRY 351 COLLECTS CANDIDATES ITS OWN CALLEES `DT_CHECK` ON

`isCandidateMemView` (351's lambda, `MutableAddrSplitting.cpp:229-238`) admits a view when the unit is
`HBM` and the start address is not bound by `symbol::CreateSymbolOp`/`SymbolQueryMapOp` — and because
`isa_and_nonnull` is false for a null defining op, `!isa_and_nonnull<…>` also ADMITS a start with no
defining op at all. Entry 113 `isEligibleForSplitting` (`:832-853`) requires strictly more: every view
in the chain is a `GetLogicalMemoryViewOp` whose start is an `arith::ConstantOp`. Entry 251
`setupForPartitioning` opens with `DT_CHECK(isEligibleForSplitting(all_mem_views))` (`:826`), and all
four callees reach it directly once `hasMutableAddrOverflow` is true (`:315`→`:333`, `:392`→`:410`,
`:477`→`:484`, `:571`→`:589`). So a toggled or region-argument start on an overflowing HBM transfer is
a candidate 351 collects and its own callee aborts on. In Rust that gap cannot be a runtime refusal:
351's collector has to hand its callees the constant-start proof `SplitCandidateView::ConstantStart`
already carries — and 372's collector must NOT, because 352-355 have no such precondition.

## ⛔ AND THE `calls` COLUMN FOR 351-356: 8 OF ITS 13 EDGES ARE NOT CALLS

The `level` column is right for all six and derivable from none of them. Every one omits the callee
that sets its level, and the declared lists imply 2, 3, 3, 3, 3 and 2 against a recorded 6.

- **8 of 13 declared edges are not calls.** `e052_If` (352, 353, 354, 355) is the word "If" opening the
  comment *"If the new subscripts map is empty…"* (`:217`, `:245`, `:274`, `:325`); there is no
  function `If`. `e150_get` (356) is the word "get" opening *"get cloned into multiple conditional
  branches."* (`TransformPagedMemViewImpl.cpp:626`) — the body has no `get` call of any kind.
  `e149_insert` (351) is `analyzed_candidates.insert(mem_op)` (`MutableAddrSplitting.cpp:275`), a
  one-argument `std::unordered_set::insert`, against `AccessContainer::insert(MemoryOperandIndex, T)`
  (`AccessDetails.hpp:388`). `e019_AccessDetailsAffine` (352, 353) matches only the type NAME in a
  declaration and a template argument — and is inverted twice over: `AccessDetails.hpp:259` is a
  MEM-INITIALISER, so the entry denotes `AccessDetailsAffineComposite`'s constructor, which is what
  354 and 355 construct and neither declares, while 352 and 353 construct plain `AccessDetailsAffine`
  and both do.
- **The 5 real edges are `e208_emplace_insert` (352-355) and `e135_eraseMemOpAndUseChain` (356).** 135
  is the static edge only: it is `virtual` (`hpp:364`) with four `override final`s, of which just
  `TPMVVectorLoad`'s (129, `cpp:698`) is scheduled — `:747`, `:817` and `:1266` are not.
- **Omitted true callees.** 351: `e306`/`e307` (level 4) and `e321`/`e322` (level 5) — its own four
  dispatch arms, which is exactly why 6 is right. 352, 353: `e265_constructDetails` (3) and
  `e323_shiftMutableAddr` (5). 354, 355: `e323_shiftMutableAddr` (5). 356: `e119` (0), `e197` (1),
  `e324_analyzeAndConstructValidPages` (5) and `agen::utils::replaceConstOpsInSubscriptsMap`
  (`Dialect/Agen/Utils.cpp:46`), which is in neither the 384 nor the 106.
- ⛔ **354 AND 355'S `constructDetails` IS UNSCHEDULED AND UNEXCLUDED.** `ad` is an
  `AccessDetailsAffineComposite&`, so `ad.constructDetails(…)` binds
  `AccessDetailsAffineComposite::constructDetails` (`AccessDetails.cpp:834-853`) — a distinct 20-line
  body sequencing six virtuals, not a delegation to entry 265 (`:418`, the `AccessDetailsAffine`
  override that 352 and 353 bind). It appears in neither the 384 nor the 106.

## ⛔ AND 356'S OTHER CALLER IS OUTSIDE THE CAMPAIGN

`TPMVBase::transform` is not virtual (`hpp:72`) and is called exactly twice: `TPMVVector::run`
(`cpp:650`, entry 373, level 7) and `TPMVComposite::run` (`cpp:871`), which is scheduled nowhere. The
composite path is also the one that reaches 356 with re-initialised state — `initialize()` `:857`,
`initialize_time()` `:859`, `transform_time()` `:862` (entry 325), then `tpmv_info_.clear()` `:864`
and `initialize()` AGAIN at `:867` before `transform()` at `:871`. A port that treats 356's input as
whatever the first `initialize()` left will be wrong on every composite.

⚠️ AND 356'S ERASE LOOP RUNS OVER THE PREVIOUS ITERATION'S OUTPUT. `for (auto &mem_op : mem_ops_)
eraseMemOpAndUseChain(mem_op);` (`:618`) sits inside `for (auto &info : tpmv_info_)`, whose last
statement is `mem_ops_ = new_mem_ops;` (`:627`). With two `TPMVInfo` entries — what
`TPMVCompositeLoadStore::initialize` produces (`:1193`, `:1200`) — the second pass erases the clones
the first one built, which is the intent the comment states and not a leak.

## ⛔ AND WHAT THE LANDED RUST CLAIMED ABOUT THESE 6 — 3 FIXED IN THIS COMMIT

All six module-doc banner rows (entry, `loc`, path:line) measured EXACT, as did every in-span prose
citation in `tf_mutable_addr_splitting.rs` — `num_conditionals_` at `:222`, its reset at `:244-246`,
`MaxNumConditionals` at `:70-75`, the *"value of -1 indicates no maximum"* comment at `:956` and the
`int`/`int64_t` comparison at `:960`. Fixed, all in `tf_transform_paged_mem_view_impl.rs`, all about
who fills the `tpmv_info_` vector 356 loops over:

- **Entry 326 does not write `tpmv_info_`.** It is `TPMVCompositeLoad::initialize_time` (`cpp:1095`),
  an override of a DIFFERENT pure virtual (`hpp:468`), and it writes `access_details_`, `time_order_`,
  `time_set_` and `tpmv_comp_info_`. The composites' `tpmv_info_` is written by their `initialize()`
  (`cpp:1080`, `:1133`, `:1186`), none of which is scheduled.
- **Entry 202 is one override, not "the vector classes".** `TPMVVectorStore::initialize` (`cpp:706`)
  and `TPMVVectorLoadStore::initialize` (`cpp:755`) are separate `override final`s.
- **The one that fills TWO is `TPMVCompositeLoadStore::initialize`** (`cpp:1193`, `:1200`), not 326.

And entry 137 (`hpp:397`) is only `TPMVVectorLoad`'s constructor — cited at its mem-initialiser and so
named after the base, the same defect as `e019` above; the note no longer reads as if it covered all
three vector classes.

## ⛔ MEASURED FOR ENTRIES 357-373: TEN EXTRACT BODIES LOSE REAL CONTENT, AND 370 LOSES ITS ONLY `return true;`

Brace-matched from the authority at each unit's cited line (`a0d29abbed`) and diffed line-by-line
against `source/bridge2.cpp`'s body. ⭐ **`loc` IS EXACT FOR ALL 17** under the column's convention
(body span minus signature lines): 387-6, 109-5, 136-5, 21-5, 34-1, 20-1, 21-2, 226-4, 88-5, 83-3,
99-2, 195-4, 632-4, 36-2, 43-1, 70-1, 7-1. **Seven bodies are verbatim** — 361, 362, 371, 372, 373,
plus 363 (two closing braces the extractor's balancer put back) and 366 (the `return;` of its
`applyPartialConversion` failure guard, `:1239`, which is also its last statement, so nothing is lost).
The other **ten lose real content, and in seven of them it is the statement that acts**:

| unit | dropped from the extract's body |
|---|---|
| **370 `areShallowlyMergeable`** (`CFGSDataflowConditionalTree.cpp:343-378`) | **ITS ONLY `return true;`** (`:377`). Four `return false;` guards survive and the extract ends at the walk's closing braces (`bridge2.cpp:16366-16368`), so a `bool` predicate reads ALWAYS-FALSE — the same defect as 347, 348 and 349, and 370's one real callee IS `e347_topLevelConditionsMatch` (`:351`), which has it too. Two nested always-false predicates make the merge step unreachable |
| **360 `constructSymbolicDetailsAndAddrs`** (`Helper.cpp:2849-2869`) | **ITS ENTIRE PAYLOAD** — `return gatherSymbolicLoadStoreDetails(comp, access_details, mutable_addrs, immutable_addrs);` (`:2867-2868`). A 16-line function whose whole point is to build two `AccessDetailsSymbolic` and hand them on; the extract ends `}}` after the dst error return (`bridge2.cpp:14931-14932`) and forwards nothing |
| **367 `createXrfIndexModifOps`** (`LoweringXRF.cpp:564-662`) | **ITS RESULT** — `return vector_op_to_xrfptr_map;` (`:661`). 97 lines build the `XrfPtrMap` returned BY VALUE and the extract discards it, exactly as 345 discards the map it builds |
| **369 `fuseComputeOps`** (`VectorChainToSentientPT.cpp:245-876`) | `VectorOperand::eraseOperands(from_operands);` (`:873`) AND `LoweringXRF::replaceAndEraseDummyMacOps(mac_op_to_xrfptr_map);` (`:875`). The 628-line body places every PT compute op and the extract drops both cleanups: the dummy MACs the XRF lowering left as placeholders are never replaced, and the from-side operands are never erased |
| **368 `fuseNonComputeOps`** (`VectorChainToSentientPT.cpp:46-240`) | `LoweringXRF::replaceAndEraseDummyMacOps(mac_op_to_xrfptr_map);` (`:239`) — the same dummy-MAC replacement, the last line of this body too |
| **365 `fillOpInfo`** (`VectorChainToSentientPESFP.cpp:1070-1157`) | THREE STATEMENTS, one of them leaving a **DANGLING `else`** — `op_info.opC_precision_ = op_info.compute_precision_;` (`:1153`) is gone while its `else` (`:1152`) stays, so the extract reads `else` / blank / `}` (`bridge2.cpp:15321-15323`); also `op_info.op_dbg_name_ = dataflow::getDbgNameAttr(op);` (`:1155`) and `return success();` (`:1156`). The opB pair above it survives intact, so a port from the extract gives opC no fallback and every op no debug name |
| **358 `constructLoadAndSendStmt`** (`Helper.cpp:1910-2018`) | `builder->restoreInsertionPoint(insert_pt);` (`:2015`) **while KEEPING the comment that describes it** (`:2013-2014`), plus `return LogicalResult::success();` (`:2017`). The one statement that puts the builder back where it found it is the one dropped |
| **357 `generateAffineAddressManipulationStmts`** (`Helper.cpp:625-1011`) | `return LogicalResult::success();` (`:1010`). Its `else` arm returns success at `:827` and survives; the 381-line `if` arm's own return does not, so the extract's last statement is the `LogicalResult::failure()` of the indirect warning path and then `#endif` / `}}}}` (`bridge2.cpp:14669-14672`) |
| **359 `constructReceiveAndStoreStmt`** (`Helper.cpp:2025-2160`) | `return LogicalResult::success();` (`:2159`) |
| **364 `patternAgnosticFuseNonComputeOpsHelper`** (`VectorChainToSentientPESFP.cpp:96-321`) | `return success();` (`:320`) **and the `const` member qualifier** — `bool is_precision_converted_global) const {` (`:99`) is rewritten without it, the same signature loss already recorded for 323, 350 and 352-355 |

## ⛔ ENTRY 357'S `TOGGLE_INDIRECT_IMPL1` REGIONS READ DEAD IN THE EXTRACT — AND 268'S READS LIVE

`TOGGLE_INDIRECT_IMPL1` is a file-local `#define … 1` at `Helper.cpp:34`. **The extract does not
define it**: `prelude.inc:2348` declares `inline Any TOGGLE_INDIRECT_IMPL1;`, a VARIABLE, which does
not satisfy `#if defined(…)`. The extract kept all four guarded regions verbatim, so the preprocessor
inverts every one of them:

- **80 of 357's 381 lines go dark.** `#if defined` at `Helper.cpp:906`/`:924`/`:954` (extract
  `:14570`/`:14588`/`:14618`) guards `SmallVector<std::pair<int, Value>> post_process_list;`
  (`:918`), the kIndSrc/kIndDst collection that fills it including its `llvm_unreachable("unexpected
  mem_view_start addr index for kIndSrc/Dst")` (`:925-939`), and the 53-line `findInvariantLoop`
  lambda plus the patch loop that inserts `mlir::arith::AddIOp` and rewrites the loop's iter-arg
  INITIALISER — `loop->setOperand(loop->getNumOperands() - v.first - 1, new_init)` (`:999`), the same
  index-from-the-end rule already recorded for `insertCopyAndAddStmtsHelper`. So an IBR access whose
  `mem_view_start` addr is non-constant (a toggling JCR) gets no offset at all in the extract.
- **And 268's 12-line region turns ON.** `#if !defined(TOGGLE_INDIRECT_IMPL1)` at `Helper.cpp:2272`
  (extract `:7739`) guards two `sentient::AddOp::create` calls adding `indirect_src_offset` /
  `indirect_dst_offset` straight to `load_immutable_addr` / `store_immutable_addr` (`:2277-2284`),
  under a comment saying it *"depends on enhancements tracked in issue 2013"*.

⛔ **THESE ARE THE TWO MUTUALLY EXCLUSIVE PLACEMENTS OF THE SAME OFFSET** — at the loop's iter-arg
initialiser (the authority's choice) or on the immutable address at the transfer (the unshipped one) —
and the extract picks the wrong one in BOTH directions. A port that reads the extract adds the indirect
offset in the wrong place and drops the invariant-level hoist entirely.

## ⛔ AND THE `calls` COLUMN FOR 357-373: 37 OF ITS 51 EDGES DO NOT NAME A REAL CALLEE

Each declared edge was resolved against the brace-matched body and against the cited definition of the
entry it names. **14 of 51 are real, 15 name the wrong definition of a shared name, and 22 are not
calls at all.** 361 and 373 declare `-` and each has two real callees; 366 declares four edges and not
one is a call.

- **22 absent.** `e052_If` (357, 364, 365, 366, 368, 369, 371 — seven of the seventeen) is the comment
  word *"If"*, already recorded. `e149_insert` (357, 358, 365, 366, 369) and `e026_has` (364, 366, 368,
  370) match comment prose the same way. `e227_OperandReuse` (366, 368, 369) is a CONSTRUCTOR
  (`OperandReuse.hpp:30`) and all three take `OperandReuse&` as a PARAMETER — the name is in the
  signature, never in a call. 371 declares `e079` and `e101` `OperationNode`, matching only
  `OperationNode::WalkOrder::kReverseBFS` in the template argument at `:790`.
- **15 wrong-definition.** `e064_size` (357, 364, 365, 367, 368, 369) is
  `VectorChainHelper.cpp:319`, which is `size(vec.size())` in `merge_and_pack_type`'s
  member-initialiser list — a DATA MEMBER, not a callable, so no `.size()` call can be that edge.
  `e150_get` (358, 359, 364, 368, 369) is `AccessContainer::get` (`AccessDetails.hpp:401`), while every
  match is an MLIR static factory — `IndexType::get(context)`, `SentientComputePortAttr::get(…)`; 369
  has 47 of them and no `AccessContainer::get` at all. `e149_insert` (367, 372) is a one-argument set
  insert (`LoweringXRF.cpp:590`, `:600`; `MutableStartAddrShifting.cpp:178`) against
  `insert(MemoryOperandIndex, T)`, the same inversion already recorded for 351. 371 declares BOTH
  `e088` and `e107` `getRoot` for its two `getRoot()` calls (`:761`, `:790`) and both are wrong: the
  receiver is the tree itself, whose `getRoot` is `dcc/src/Analysis/ConditionalTree.hpp:143`, outside
  the span — the same defect as 349's `getFirstChild`.
- **The 14 real edges** are `e004_setMemViewStartAddr`, `e030_getLoopNestLevel` and
  `e264_createForOpWithAdditionalReturnValue` (357), `e208_emplace_insert` (360),
  `e338_ConstructIFRecursively` (362, 363), `e279_createSplatOperation` (364),
  `e229_getMaskValueForNonPT` + `e342_analyzeAndFillResultForwarding` (365), `e068`/`e069`/`e070` +
  `e342` (369) and `e347_topLevelConditionsMatch` (370).
- ⛔ **AND FOUR OMITTED CALLEES ARE EXACTLY THE LINES THE EXTRACT TRUNCATED.** 360 omits
  `e327_gatherSymbolicLoadStoreDetails` (`Helper.cpp:1051`, level 6) — its lost `return`. 368 and 369
  omit `e093_replaceAndEraseDummyMacOps` (`LoweringXRF.cpp:681`, level 0) and 369 also
  `e233_eraseOperands` (level 2) — their lost cleanups. An agent working from the extract sees neither
  the statement nor the edge.
- ⛔ **372 DECLARES ONE EDGE AND OMITS ALL FOUR OF ITS ARMS** — `e352`-`e355`
  (`MutableStartAddrShifting.cpp:189`, `:191`, `:193`, `:195`, all level 6), which is exactly what
  makes 372 level 7; the sole declared edge is the set `insert` above. 371 omits
  `e349_isLoopInvariant` (`:783`, level 6), its only in-set callee and the reason IT is level 7. 361
  omits `e337_lowerSyncOperation` (level 6) and `e044_lowerOpaqueOperation` (level 0). 373 omits
  `e356_transform` (`:650`, level 6). ⭐ So the `level` column is right for all 17 and derivable from
  the `calls` column for none of them.

## ⛔ THE SCHEDULE IS KEYED BY BARE METHOD NAME, SO 43 LATER OVERRIDES WERE SILENTLY DROPPED

Mechanical, and it explains a class of missing work the 106 exclusions do not cover. Across every
`.cpp` cited by the 384 there are **82 top-level `Class::name(` definitions that are not entries, and
43 of them are a LATER definition of a name an EARLIER entry in the SAME FILE already took.** None of
the 43 appears in the 106.

- **13 are in `VectorChainToSentientPESFP.cpp`** (357-373's own file for 364/365/366) and they are the
  per-op lowering bodies: `BinaryOpLowering` (`:328`), `MultiplyOpLowering` (`:569`),
  `ShuffleOpLowering` (`:613`), `PackOpLowering` (`:675`), `ScanWithGapOpLowering` (`:743`),
  `FastExpOpLowering` (`:804`), `ExpEstimateOpLowering` (`:835`), `FloorOpLowering` (`:869`),
  `RecEstimateOpLowering` (`:899`), `LnEstimateOpLowering` (`:929`), `TanhEstimateOpLowering`
  (`:1022`), `StoreOpLowering` (`:61`) and `VectorStoreOpLowering` (`:78`) — all
  `matchAndRewrite`, all shadowed by `e377_matchAndRewrite` (`:44`, `SendOpLowering`). **The op a
  function emits IS the function**, and for the PESFP component thirteen of those emissions are not in
  the campaign at all.
- **18 are in 373's own translation unit** (`TransformPagedMemViewImpl.cpp`): 51 definitions, 33
  entries, and all 18 non-entries are later same-named overrides. `initialize()` has SIX overrides
  (`:658`, `:706`, `:755`, `:1080`, `:1133`, `:1186`) and only `TPMVVectorLoad`'s is scheduled
  (`e202`); `initialize_time()` has three (`:1095`, `:1148`, `:1211`) and only `e326`;
  `createNewMemOp` six (`:684`, `:732`, `:779`, `:1120`, `:1173`, `:1242`) and only `e128`;
  `eraseMemOpAndUseChain` four and only `e129`. ⛔ **So 373's
  own two calls cannot both resolve**: `TPMVVector::run()` is `initialize()` then `transform()`
  (`:648`, `:650`), and its `initialize()` is a pure virtual whose overrides are 1-of-6 present. Its
  sibling `TPMVComposite::run()` (`:855`) is shadowed by 373 itself.
- **2 are 363's own siblings.** `LowerLogicalOpToSentient` is THREE overloads —
  `mlir::arith::AndIOp` (`StandardToSentient.cpp:264`, which is `e363`), `OrIOp` (`:286`) and
  `CmpIOp` (`:308`) — declared at `:68-70` and all three called by `e376_runOnOperation`
  (`:460`, `:462`, `:464`). Only the AndIOp body is in the 384. A port of `e363` alone lowers
  `arith.andi` and leaves `arith.ori` and `arith.cmpi` to a pass that dispatches to nothing.

## ⛔ AND WHAT THE LANDED RUST CLAIMED ABOUT THESE 17 — 1 FIXED IN THIS COMMIT

⭐ **NO PORTED SUBJECT.** 0 `/// Replaces:` anchors and 0 surviving `// crustify:todo:` for 357-373;
all 34 PORT/AUDIT boxes are `[ ]`. All 17 module-doc citation rows measured exact, as did the
`agen_helper.rs` `todo!` block for 357 — eight citations including the `:827`-versus-`:1010` pair of
returns, the empty-coefficient-table walk through `:631-632` → `:761` → the zero-trip loop, and the
non-L3 all-zeros row that takes neither arm at `:788`/`:795`. `tf_cfg_simplification_dataflow_level.rs`
measured exact for 371 (`:107`, and the four scheduled versus two unscheduled steps of that pass), as
did `agen_agen_to_sentient.rs` on 315's 7-argument call (`Helper.cpp:3090-3092`) binding `e359` rather
than the 8-parameter `e028`, on `e358`'s optional `extract_op` test (`:2006`) against
`lowerIndirectVectorLoadOp`'s own report (`:3196-3199`), and on both indirect callers (`:3202`,
`:3251`). One claim was wrong:

- **`agen_agen_to_sentient.rs` said `e028`'s extra `data_elem_type` exists because `setsttype` derives
  `element_width` and the shuffle mode from it, citing `Helper.cpp:2124`.** Neither half holds.
  `data_elem_type` occurs FOUR times in the whole authority — `AgenToSentient.hpp:242`, `:250`, `:253`
  and the primary's own parameter list at `Helper.cpp:2027` — and **never in a body**: `e359` reads
  `element_width` from `access_details.getElementWidth()` (`:2032`) and calls
  `setsttype(builder, comp, producer_info.first, total_elements, element_width, shuffle_mode)`
  (`:2092`), whose signature (`hpp:206-209`, body `:1731-1784`) takes no element type at all — it
  derives the shuffle mode from the producer input OP (a `vectorchain::ShuffleOp`'s splat shape or a
  `dataflow::ReceiveOp`'s result type) and WRITES `total_elements` from `element_width`. `:2124` is an
  argument line of the `sentient::ReceiveAndStoreOp::create` call, not a `setsttype` use. The store
  overload's extra parameter is DEAD in the reference, which is a stronger reason for the port to drop
  it than the one the doc gave; the implementation was already right.

## ⛔ MEASURED FOR ENTRIES 374-381: 374'S EXTRACT KEEPS ITS RETURN AND DROPS THE ARGUMENTS, AND 380 LOSES ITS ONLY `return`

Brace-matched from the authority at each unit's cited line (`a0d29abbed`) and diffed line-by-line
against `source/bridge2.cpp`'s body. ⭐ **`loc` IS EXACT FOR ALL 8** under the column's convention
(body span minus signature lines): 29-3, 33-3, 35-1, 15-3, 31-1, 55-1, 78-2, 31-1. **Four bodies are
verbatim** — 376, 378, 379 and 381, each differing only in the renamed signature and trailing blank
lines. The other four lose content, and two of them lose the statement that answers:

| unit | dropped from the extract's body |
|---|---|
| **374 `lowerSymbolicVectorLoadOp`** (`Helper.cpp:3380-3408`) | ⛔ **THE ARGUMENTS OF ITS ONLY RETURN, WITH THE RETURN LEFT STANDING** — a shape not seen before in this corpus. `return lowerVectorLoadHelper<AccessDetailsSymbolic, SymbolicVectorLoadOp>(` survives and `candidate_op, store_op, unit, access_details, mutable_addrs, immutable_addrs, to_be_deleted);` (`:3406-3407`) does not, so the extract reads `return lowerVectorLoadHelper<…>(` / blank / `}`. A truncation that drops a whole statement at least reads as missing; this one reads as a CALL WITH NO ARGUMENTS, and the three containers entry 360 filled are exactly what it fails to name |
| **380 `shallowlyMergeConditionals`** (`CFGSDataflowConditionalTree.cpp:198-275`) | ⛔ **ITS ONLY `return cur_merged_if_op;`** (`:274`). The function's whole contract is *"return the merged if-op so we may resume merging from there, or nullptr to indicate there are no more merge opportunities"* (`:207-210`); the extract ends after the `if (cur_merged_if_op) { recompute(); getOE().clearCache(); }` block, so an `Operation *`-returning function reads as returning nothing at all — the same defect class as 370's lost `return true;`, one rung up: 370 reads always-false and its only caller, 380, reads always-null |
| **375 `lowerSymbolicVectorStoreOp`** (`Helper.cpp:3410-3442`) | `return success();` (`:3441`) |
| **377 `matchAndRewrite`** (`VectorChainToSentientPESFP.cpp:44-58`) | `return success();` (`:57`) **and the `const` member qualifier** — `mlir::ConversionPatternRewriter &rewriter) const {` (`:46`) is rewritten without it, the same signature loss already recorded for 323, 350, 352-355 and 364 |

## ⭐ AND THE `level` COLUMN IS RIGHT FOR ALL 8 — WHAT RAN OUT OF ORDER IS THE CAMPAIGN, NOT THE SCHEDULE

The port commit for these eight reported *"all eight are scheduled ahead of a callee they need —
reported as a schedule violation"*. ⛔ **THAT IS THE WRONG DIAGNOSIS, AND IT MATTERS BECAUSE IT BLAMES
THE COLUMN INSTEAD OF THE WAVE ORDER.** Every unported callee the eight stop at sits STRICTLY BELOW
level 8, which is exactly the order a topological schedule promises:

`e217` L2 · `e227` L2 · `e228` L2 · `e239` L2 · `e270` L3 · `e285` L3 · `e304` L4 · `e339` L6 ·
`e344` L6 · `e346` L6 · `e347` L6 · `e360` L7 · `e362` L7 · `e363` L7 · `e364` L7 · `e366` L7 ·
`e367` L7 · `e368` L7 · `e369` L7 · `e370` L7

All twenty are `[ ]` PORT. So the schedule put them first and **the campaign ported level 8 while
level 7 (357-373) has zero ported entries** — 17 unported units directly below eight that need nine of
them. The corpus convention of stopping at a `todo!` naming the first gap is the right response to
that, but the finding to record is an EXECUTION-ORDER one: level 7 must land before level 8 stops
being a wall of `todo!`.

## ⛔ AND THE `calls` COLUMN FOR 374-381: 19 OF ITS 27 EDGES DO NOT NAME A REAL CALLEE, AND 29 REAL ONES ARE MISSING

Each declared edge was resolved against the brace-matched body and against the cited definition of the
entry it names. **8 of 27 are real.** Four failure shapes:

- **`e052_If` — a COMMENT LINE, declared by 379, 380 and 381.** `StandardToSentient.cpp:159` is
  `// return If(lhs) {If(rhs) true_val; else false_val} else false_val;`, and the column gives it
  `0L`. Nothing in any of the three calls anything named `If`; the extractor matched prose.
- **`e064_size` — a CONSTRUCTOR MEMBER-INITIALISER, declared by 375 and 379.**
  `VectorChainHelper.cpp:319` is `size(vec.size()) {` inside an init list — a data member being set,
  not a callable. 375's real `access_details.size() == 1` (`:3419-3420`) is `AccessContainer::size` in
  `AccessDetails.hpp`, and 379 calls no `size()` of ours at all (`unit.getUnits().size()` at `:993` is
  MLIR's).
- **THE NODE ACCESSORS NAME THE WRONG TREE, TWICE OVER — 12 edges across 380 and 381.**
  `getFirstChild`/`getNextSibling`/`getRoot`/`OperationNode` are declared as `e082`/`e083`/`e088`/`e079`
  (`.../VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:39`/`:42`/`:109`/`:32`) AND as
  `e103`/`e104`/`e107`/`e101` (`.../FlatteningLocalRegions.cpp:56`/`:59`/`:81`/`:51`) — two unrelated
  trees' same-named accessors, each listed once per row. The accessors these two actually walk are
  `CondNode`'s, at `dcc/src/Analysis/ConditionalTree.hpp:49`/`:52`/`:143`, which are **not among the
  384** because `dcc/src/Analysis/` is outside the D1-D28 span. Also `e026_has` in 380: real for 374
  (`:3399`), a non-call there.
- ⛔ **`e099` NAMES THE DESTRUCTOR AND 381 CALLS THE CONSTRUCTOR.**
  `CFGSDataflowConditionalTree.hpp:78` is `~ConditionalSimplificationManager() { if (val_array_)
  delete[] val_array_; }` — 2L, which is why the column matched it. `:477`'s
  `ConditionalSimplificationManager instance(child);` calls the CONSTRUCTOR at `.hpp:55-76`, 22 lines
  that decide `is_candidate_` from `getLhsRhsOfEQPredicate` and `getDataflowForLoopInfoIfIV` and
  allocate `val_array_[num_iterations]`. **That constructor is not one of the 384 under any entry**,
  and it is not optional: `parseConditional`'s very first test is `if (!is_candidate_ || …) return
  false;` (`:537`), and `is_candidate_` is written NOWHERE ELSE — default `true` at `.hpp:30`, set
  `false` only at `.hpp:61` and `.hpp:69`, both inside that constructor. So the answer 381 branches on
  is decided by a function the schedule cannot see.

And the omissions run the other way in every row — **29 real in-set callees are undeclared**, including
all seven of 376's (`e339` `:443`, `e049` `:448`, `e050` `:450`, `e362` `:456`, `e053` `:458`, `e363`
`:460`/`:462`/`:464`, `e054` `:466`) against a declared `-`, eight of 379's nine, five of 378's seven,
`e360` in both 374 and 375, `e370`+`e177` in 380 and `e285` in 381. ⭐ 376's four undeclared-but-PORTED
callees (`e049`, `e050`, `e053`, `e054`) are all called by the landed Rust anyway, as is `e036` by 374
and `e067` by 378/379 — the implementations found them; only the column did not.

## ⛔ AND WHAT THE LANDED RUST CLAIMED ABOUT THESE 8 — 1 IMPLEMENTATION AND 13 CITATIONS FIXED IN THIS COMMIT

8 `/// Replaces:` anchors, 0 surviving `// crustify:todo:`, 16/16 boxes `[x]`. ⭐ **376, 378, 379 AND
380 MEASURED EXACT — every citation, all 34 of them**, including 376's fifteen (`:439`, `:441-445`,
`:447-471` and one per arm), 378's six, 379's ten (the `WalkResult::interrupt()`-is-a-discarded-
temporary catch at `:1019` and the `insertPTMaskOps` that still runs after it at `:1024`) and 380's
twenty. So did every claim the island growth forced, each re-derived from the reference:
`isDataTransfer`'s nineteen classes DO include both symbolic ops (`UnitFiltering.cpp:314-325`);
`getLoadConsumer` roots a `SymbolicVectorLoadOp` at `getResult(0)` (`Helper.cpp:1245-1246`);
`constructChunkAndShuffleInfo`'s gate excludes them (`AccessDetails.cpp:207-208`);
`AccessDetailsAffine::initialize` (`:295`) drops them into
`return op->emitError("unsupported operation")` (`:343-345`); `is_memory_op`'s `isa<>` names five ops and no symbolic one
(`TransformLoopToLegalizeForSentientLowering.cpp:329-331`); `TpmvManager`'s `dyn_cast` chain ends in
`llvm_unreachable("memory operation is not supported for static paged tensors")`
(`TransformPagedMemViewManager.cpp:64-66`); no `SymbolicVector*` appears in any `Utils.cpp`; and the
emission test's expected text is `symbolic_vector_load_store.mlir:224`/`:302` verbatim, over
`#map`/`#set` from `:184-185`, with two indices on a one-dimensional memref exactly as the vendor
writes it. **One implementation was wrong and is fixed here:**

- ⛔ **381 ENUMERATED ROOT'S CHILDREN AS THE UNIT'S TOP-LEVEL STATEMENT LIST, AND THE SAME FILE
  ALREADY SAYS WHY THAT IS WRONG.** The landed loop was
  `tree.unit.body.iter().enumerate().filter(|(_, op)| ConditionalKind::of(op).is_some())`, with a doc
  bullet calling `:473-474` *"a walk over the unit's TOP-LEVEL conditionals and nothing deeper"* and a
  test, `a_nested_conditional_is_not_a_root_child`, asserting that an `scf.if` inside a top-level
  `affine.for` is NOT reached. `ConditionalTree::compute` parents every selected op at
  `findClosestParent(op)`, which **skips every UNSELECTED ancestor** —
  `while ((curr_op = curr_op->getParentOp())) if (isOperationSelected(*curr_op)) return curr_op;`
  (`dcc/src/Analysis/ConditionalTree.hpp:165-169`) — and `isOperationSelected` is
  `isa<AffineIfOp, scf::IfOp, sentient::IfOp>` (`ConditionalTree.cpp:226`), which names no loop. A
  conditional under a top-level `affine.for` therefore has NO selected ancestor, so its parent node is
  the root (`ConditionalTree.cpp:200-208`) and it IS a root child. ⛔ **380, forty lines above it in
  the same file, gets this right and says so**: its `sibling_group` helper recurses through unselected
  ops and its doc reads *"a conditional directly in a `then` region and one nested inside an
  `affine.for` in that same region are SIBLINGS"* — the two ports disagreed about one tree. 381 now
  calls `sibling_group` for root's children, and the test is inverted to assert the conditional under
  the loop DOES reach `parseConditional`.

⚠️ And two counts in the port commit's own message do not match the tree, both by one and neither
affecting code: it reported *"208/384 ported and audited"* where the boxes are **209/209** (`e001`
plus 002-024 and the rest), and *"839 passed"* where `cargo test -p deeptools --lib` reports **840**
(unchanged by this commit — `#[test]` count is 840 at `ec81458b0` and 840 here; one test is renamed
and inverted, none added).

⭐ AND THE ONLY THING 381 GETS FROM THE ROOT IS ITS CHILD LIST, so the fix is contained: nothing else
in the body reads `at` except the call-local `PROCESSED_SIMPLIFICATIONS` set, which is now keyed by
position in the root sibling group rather than by position in the statement list.

**13 drifted citations re-measured and corrected** (small offsets are the norm; these were rewritten
rather than merely noted because each names the wrong line's content):

| where | said | is |
|---|---|---|
| 374, the container fill | `:3386-3393` | `:3386-3391` — `:3393` starts the next statement |
| 374, taking the candidate from `kDirSrc` | `:3395` | `:3393-3394`; `:3395` is the `DT_CHECK` on it |
| 374, the second store read | `:3397-3402` | `:3399-3402` — `:3397-3398` are its comment |
| 375, `DT_CHECK(size == 1)` on all three | `:3418-3419` | `:3419-3420` |
| 375, the `[0]` indexing | `:3427-3428` | `:3429-3430` |
| 375, element type from the memref | `:3424` | `:3427`; `:3424` is the `DT_CHECK` |
| 375, the width taken from the VALUE instead | `:874-878` (no file) | `AccessDetails.cpp:879-880` — a different file from the one the block cites throughout |
| 375, `constructSymbolicDetailsAndAddrs(op, nullptr, …)` | `:3413-3417` | `:3413-3418` |
| 375, entry 028's three arguments | `:3425-3429` | `:3428-3431` |
| 375, the store pushed to the delete list | `:3434` | `:3436`; `:3434` is inside the error string |
| 375, entry 270 on the value's producer | `:3436-3437` | `:3438-3439` |
| 377, the four fusible producer classes | `:100-102` | `:101-103` |
| 377, `is_precision_converted_global` by value | `:98` | `:99` |

Two more claims were narrowed rather than corrected: `AccessDetailsSymbolic::initialize` spans
`:858-896`, not `:855-897` (the cited range opens on the file's banner comment), and 380's
`deleteAncestorsIfPossible`/`recompute`/`clearCache` note said *"the first two live in
`dcc/src/Analysis/`"* when **all three** do (`TransformationConditionalTree.cpp:155`,
`ConditionalTree.hpp:151`, `OperationEquivalence.hpp:71`).

## ⛔ AND WHAT THE LANDED RUST CLAIMED ABOUT ENTRY 384 — THE PORT HOLDS, 8 CITATIONS AND 7 COUNTS FIXED IN THIS COMMIT

1 `/// Replaces:` anchor, 0 surviving `// crustify:todo:`, 2/2 boxes `[x]`. ⭐ **THE SIX-STEP SPINE AND
ITS GATE ARE EXACT**, each re-derived from `AgenToSentient.cpp:169-250` (81 lines, as `UNITS.tsv`
says): the gate at `:174-176`, the LDCVTI pre-pass `:178-210`, the fusion `:213-216`, the interleave
sweep `:218-231`, the mask-state sweep `:233-245` and the cleanup at `:248`, in that order. So is the
gate's membership test — `getUnitType` runs `stringToSenComponents` then `senCompToGenericComp`
(`sys-arch-spec/arch_enums.cpp:124+`) and no specific component outside the six collapses into one, so
`TransferComp::of`'s twelve `None` arms are the reference's `return`. ⭐ **AND THE PORT IS RIGHT ABOUT
THE ONE PLACE THE REFERENCE'S OWN COMMENT IS WRONG**: `:183`/`:185` call the LDCVTI anchor a
`vectorchain.multiply`, and the walk at `:192-198` matches `vectorchain::BinaryOp` with
`getBinaryOp() == mul`. Both island additions answer their ten-way fan-out correctly —
`isDataTransfer`'s nineteen classes include `CompositeMemoryInterleaveOp` and do NOT include
`SetTransferMaskStateOp` (`UnitFiltering.cpp:314-325`) — and both `emit` arms match
`comp_mem_interleave.mlir:309` and `set_transfer_mask_state.mlir:24` attribute for attribute, over
`Agen.td:982-1039` and `:1041-1116`.

⭐ THE CORPUS MEASUREMENT REPRODUCES INDEPENDENTLY, and says more than the port claimed: over
`tests/sentient_corpus/`'s 18 pairs the `agen` ops sit on `l3lu` (72), `lxlu` (36) and `lxsu` (18) and
nowhere else, so the gate emits nothing new; and every `vectorchain.binary` with operator `mul` sits
on `sfp` (9 of them, none anywhere else), which the gate excludes — so **the LDCVTI arm cannot fire on
a staged program even at `Sen1p5`**, on top of being inert because `lower()`'s one caller is `Dd2`. The
`set_send_dst` `todo!` is unreachable for a second reason: e035 `generate_set_send_destination_stmts`
(`agen_helper.rs:1325`) has no caller outside its own `#[cfg(test)]` module.

⚠️ ONE DELIBERATE DIVERGENCE, CONFIRMED UNOBSERVABLE. Entry 382's dispatch grew a `Consumed(1)` arm
for the two new ops, which DROPS them (and the mask state's result binding) from the lowered body
where the reference leaves them in place for steps 4 and 5. Nothing can see it: `run_on_operation`
counts them over the INPUT, including regions, and `todo!`s whenever the count is non-zero, so no
program reaches the fusion with one present.

**8 citations corrected** — the four step spans were each short by the reference's closing brace or
its comment, and two claims named the wrong function entirely:

| where | said | is |
|---|---|---|
| the LDCVTI candidate walk (2 sites) | `:183-189` | `:192-198` |
| the LDCVTI pre-pass block (2 sites) | `:178-198` | `:178-210` |
| the mask-state sweep (2 sites) | `:233-244` | `:233-245` |
| the `set_send_dst` cleanup (2 sites) | `:246-247` | `:247-248` |
| the "multiply" comment | `:180` | `:183`/`:185`; `:180` reads "agen.vector_load operations" |
| the twelve transfer classes (2 sites) | `findCandidateForLowering`, `:29-52` | `fuseLoadOrStoreChainOps`'s candidate walk, `:33-37` — `findCandidateForLowering` (`Helper.cpp:2884`) is instantiated with ten distinct classes at twelve sites, which is what `:275` above already measured |
| the `slice_mask_map` grammar (2 sites) | `Agen.td:1060-1073` | `:1058-1072` |
| the *add the op to the island* rule (5 sites, all pre-existing) | `AGENT-BRIEF.md:57` | `:87` — correct when written, drifted by `ba46ed418` inserting the budget section |

**7 counts corrected**, each a number the tree contradicts: *"FOUR OF ITS SEVEN STEPS"* → **SIX** (the
gate and the five steps it guards, which is what the module doc and this entry's own list name); *"for steps 5
and 6"* → **steps 4 and 5**; *"nine total matches"* → **TEN** (`construct_chunk_and_shuffle_info` and
`AccessDetailsAffine::initialize` are two sites in one file); *"NINE OF THE REFERENCE'S TWELVE `agen`
TRANSFER CLASSES HAVE NO ISLAND OP"* → **SEVEN** (the symbolic pair landed with 374/375);
`is_data_transfer`'s header *"NINETEEN CLASSES, ELEVEN OF WHICH THIS ISLAND DECLARES"* → **TWELVE**
(its own body says six `dataflow` and six `agen`); *"the four indirect composites"* → **three**
(`composite_indirect_load`, `_store`, `_load_and_store`); and *"a `_ => false` would make the tenth
`agen` op silently non-transferring"* is now count-free — the island declares eight and the sentence
was stale the moment it was written.

⛔ ONE CLAIM WAS INCOMPLETE, NOT DRIFTED. The cleanup comment said e220 *"collapses a unit's identical
ones into one at the top"*. It also erases them ALL and emits NONE when the shared destination unit's
type is `"sfp"` — *"if the mode is only ever set to the default (sfp), then we don't even need to
generate any setdstmask instructions"* (`Helper.cpp:4110-4112`) — which is the case a port of e220 is
most likely to get wrong, since every other arch level emits the one survivor.

## Progress

`225/384 ported; 225/384 audited`

Ported and audited: `AffineYieldOpLowering::matchAndRewrite` (`lower_affine_yield`, entry 001, in
`src/bridges/dataflow_ir_to_sentient/std_affine_to_standard.rs`), and entries 002-024 —
`setCoalescedBoundValues`, the six `AccessDetailsBase` setters
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
079/080/281 — and entries 106-107 have since taken it up, so the arena now carries both derived
families.

⭐ AND ENTRIES 049-056 — the six `StandardToSentient` scalar lowerings (`LowerAddIOpToSentient`,
`LowerSubIOpToSentient`, `LowerMulIOpToSentient`, the `If` shape law, `LowerConstantIndexToSentient`,
`LowerConstantIntToSentient`) in `src/bridges/dataflow_ir_to_sentient/std_standard_to_sentient.rs`,
and `OperandReuse`'s two getters (`getId`, `getAbsorbtionFlag`) in `vc_operand_reuse.rs`.

⭐ AND ENTRIES 041-048 — the `dataflow.opaque` lowering and the two SCF/Standard predicate maps:
`ExtendUnitNameToCorelet`, `isSameListOfUnits`, `isTargetL3` and `lowerOpaqueOperation` in
`src/bridges/dataflow_ir_to_sentient/dfs_dataflow_to_sentient.rs`; `getSentientCmpIPredicate`, the
`ForOpLowering` pattern registration and `SCFToSentientLoweringPass::runOnOperation` in
`std_scf_to_sentient.rs`; and the second `getSentientCmpIPredicate` in
`std_standard_to_sentient.rs`.

⛔ THE ISLAND GREW FOR 044, AS THE BRIEF REQUIRES. `dataflow.opaque` carried no `dbgName`, and the
reference forwards one into the `sentient.opaque` it creates (`DataflowToSentient.cpp:2006-2007`);
`Dataflow.td:342` declares the attribute and IBM's own input writes it
(`dcc/test/Conversion/DataflowToSentient/opaque.mlir:35`). `Op::Opaque` became a struct payload so
the lowering takes one typed input, and the printer emits `dbgName = ".."` first as MLIR's key order
requires.

⚠️ AND THAT ANSWER KEY FALSIFIED THREE PRINTER LINES IN THE SENTIENT ISLAND. `sentient.opaque` was
printing `func_name = "RECIPROCAL"` (the generated enum's spelling rather than the reference's
lowercase), `{P0 = "0"}` for the register dictionaries (the address with no `R` — `ddcv1.cpp:3350`
writes `"R" + startAddress` and `dcc/src/Dialect/Sentient/Utils.cpp:157` strips it back off by
position), and both dictionaries unsorted. All three are fixed, and `opaque.mlir:18` is now
reproduced byte for byte by a unit test.

⛔ 042 HAS NO CALLER AT `a0d29abbed`. A grep of every `.cpp`/`.hpp`/`.h` in the authority tree finds
`isSameListOfUnits` exactly once, at its own definition. It is ported anyway — this document's rule
is that deciding a function is unnecessary is not the porter's judgement — and the note on it says so.

⛔ 041'S ERROR ARM IS UNREACHABLE BY CONSTRUCTION, NOT UNCHECKED. `!unit->hasAttr("corelet")` is the
failure, and `residency_of` sends both LX halves to `Residency::Corelet { .. }` unconditionally
(`src/units.rs:594`), so the port takes a `Corelet` and the arm has no input. Both call sites
(`:409-411`, `:709-713`) guard on `dst_comp == LXLU || dst_comp == LXSU` before calling, which is what
the `LxHalf` enum is.

⛔ AND 047 CANNOT LOWER A MODULE YET, BY DESIGN OF THE SCHEDULE RATHER THAN OF THE PORT. Its three
patterns are `ForOpLowering::matchAndRewrite` (entry 225, unported), `IfOpLowering::matchAndRewrite`
(`SCFToSentient.cpp:158`) and `YieldOpLowering::matchAndRewrite` (`:241`) — and the last two are in
neither the 384 nor the 106 exclusions. The pass is wired in with a `todo!` naming the pattern that
owes the rewrite, because the alternative is leaving an `scf` op in a module the Sentient rung is then
asked to schedule. The `ConversionTarget` is a THREE-valued `Legality`: `applyPartialConversion` fails
only on ops marked ILLEGAL, and `dataflow`/`affine`/`agen` are named by neither list.

⭐ AND ENTRIES 097-104 — the conditional-tree pair (`getNewDbgNameFromList`,
`getLhsRhsOfEQPredicate`), `ConditionalSimplificationManager`'s destructor and
`CFGSDataflowConditionalTree`'s constructor in `tf_cfgs_dataflow_conditional_tree.rs`, and
`LocalOpNode`'s constructor with its three link reads (`getParentNode`, `getFirstChild`,
`getNextSibling`) in `tf_flattening_local_regions.rs`.

⭐ AND ENTRIES 105-108 — `LocalOpNode::getPrevSibling`, `FlatteningLocalRegionsTree`'s constructor
and its `getRoot`, and `partitionUnits`, in `tf_flattening_local_regions.rs`. ⛔⛔ THE FAMILY MOVED
ONTO THE SHARED ARENA HERE, which is the reconciliation entries 101-104 could not make: entry 106 is
the `: OperationTreeBase()` the C++ names, so the tree it constructs is
`OperationTreeBase<LocalOpNode>` from `vc_loop_mask_tree.rs` — as that layer's own banner instructs —
instead of a second set of links, a second `getPrevSibling` walk and, at entry 180, a second
`insertChildNode`. Entries 102-104 therefore moved from `LocalOpNode` onto the tree as one-line
delegations, exactly as 081-083 read on `LoopMaskTree`, and `LocalOpNode` is now the payload the
derived class adds (`units`, `is_in_region_num`). ⭐ ENTRY 107'S `DT_CHECK` SPLIT THREE WAYS: two
conjuncts hold by construction, and `root_ != nullptr` stays an `Option` because `flatten` really does
observe a rootless tree (`FlatteningLocalRegions.cpp:385-387`). ⛔ AND 108'S EQUALITY IS POINTER
IDENTITY — `std::vector<Operation *>::operator==` — which is what keeps the vendor's `@diff_groups`
four structurally identical bodies in four separate regions, in `MapVector` key order (`%0, %2, %1,
%3`).

⭐ AND ENTRIES 109-112, THE TWO RANGE PAIRS. `performFullUnroll` + `getConstantTripCount` in
`tf_loop_unroll_for_shuffle_op.rs`, and `getMaxMutableRange` + `getMaxImmutableRange` in
`tf_mutable_addr_splitting.rs`. ⛔⛔ 109 IS THE DECISION AND NOT THE DUPLICATION, AND THAT IS THE
REFERENCE'S OWN SHAPE: neither overload clones a loop body — the scf one reconstructs a trip count and
calls `loopUnrollByFactor(for_op, *trip_count)`, the affine one is the single line
`return loopUnrollFull(for_op)` (`LoopUnrollForShuffleOp.cpp:151-165`). Those two utilities are
upstream MLIR (`mlir/Dialect/{SCF,Affine}/Utils`), not among these 384, so `Unroll::ByFactor` /
`Unroll::Fully` name the request the function issues and `Unroll::failed()` answers the caller's one
question (`:135`). ⭐ THE THREE C++ FUNCTIONS COLLAPSED INTO ONE because an overload set over two loop
kinds IS a match on a closed set — and `llvm_unreachable("unsupported loop type")` (`:148`) moved into
`Loop::of`, where "not a loop" is an answer rather than undefined behaviour. ⛔ THE ISLAND GAINED
`scf::Op::For` FOR EXACTLY THAT REASON (its input was inexpressible: `scf.for`'s bounds are three SSA
VALUES where `affine.for`'s are maps, which is the whole reason one overload needs a helper and the
other does not), wired through `operands`/`block_args`/`regions` and the printer. ⭐ AND THE NEW
VARIANT MADE ENTRY 046's `Pattern::ForOpLowering` REACHABLE: `std_scf_to_sentient.rs`'s `pattern_for`
was total over an `scf` that had no `scf.for` in it, so it now answers `ForOpLowering` for the op the
pattern is registered under, and `walk_preorder` descends its region like every other. ⛔ AND 110 CARRIES
TWO REFERENCE DEFECTS, ONE REPRODUCED AND ONE NOT: the truncating `(ub - lb) / step` under-counts when
the step does not divide the span, and is reproduced because the factor is what the utility is asked
for; the unguarded zero or negative trip count reaches a `uint64_t` factor behind
`assert(unrollFactor > 0)`, and is refused instead (`TripCount` cannot hold one, and
`Unroll::EmptyOrReversedRange` keeps the divergence visible). ⭐ 110 ALSO REFUSES EVERY ORDINARY
`scf.for`: `arith::ConstantIntOp::classof` wants a SIGNLESS INTEGER, so `index`-typed bounds are not
constants to it — which the island already splits as `ConstantInt` vs `Constant`, and which is
consistent with the vendor's only test for this pass being affine throughout
(`dcc/test/Transform/LoopUnrolForShuffleOp/ldcvti_pattern.mlir`). ⭐⭐ 111 AND 112 ARE THE SAME
FUNCTION OVER TWO REGISTERS, AND THE REGISTERS DIVERGE BY ARCH: `EAR` is 21 bits everywhere
(`sysdef.cpp:313`, `:336`) but `EBR` is 30 on RCUDD1A and **32** from SEN1P5 (`:321-332`, `:344-355`),
so the immutable range is 2^40 on DD2 and 2^42 on SEN1P5 while the mutable range is 2^31 on both.
`Arch` gained `L3_EAR_BITS`/`L3_EBR_BITS` as `Bounded<53>` — the bound is the reference's own `int64_t`
return type, so `2^bits * bytesPerStick * 8` cannot overflow and needs no check. ⛔ THE `DT_CHECK(
is_any_of(comp, L3LU, L3SU))` BECAME `L3Half`, and the finding that justifies it is that the two halves
declare IDENTICAL `EAR` and `EBR` rows: the component's only function in these two queries is the
abort, so the type is the whole of it. ⛔ AND `cl::init(-1)` IS AN `Option`, not a negative size —
`MaxMutableSize < 0` is a sentinel test on the same variable that carries the value.

⭐ 097'S CITATION IS A CALL SITE, NOT A FUNCTION. `CFGSDataflowConditionalTree.cpp:456` is inside
`mergeConditionalBranchesInSubtree`; the function itself is `dataflow::utils::getNewDbgNameFromList`
(`dcc/src/Dialect/dialect_utils/Dataflow/Utils.cpp:167`, 19 lines) and that is what was ported —
head-plus-rest rather than a first-iteration flag, and a per-operation `Option<&str>` because the
whole name is abandoned the moment one operation carries no `dbgName`.

⭐ THE ISLAND GREW FOR 098, AS THE BRIEF REQUIRES. `getLhsRhsOfEQPredicate` tests
`cmpi_op.getPredicate() != eq`, and `arith::Op::Compare` carried no predicate at all — it printed
`arith.cmpi eq` unconditionally. `CmpIPredicate` now names the SIX signed forms
`getSentientCmpIPredicate` accepts (`StandardToSentient.cpp:36-53`, entry 048), whose `else` is
`DT_CHECK(0)`; the four unsigned MLIR forms are absent BY CONSTRUCTION rather than refused. `Compare`
had no construction site crate-wide, so emission is unchanged for `Eq`.

⚠️ AND THE EXTRACT HAD TRUNCATED 098: its bodies stop at `rhs = cmpi_op.getRhs(); return true;`, i.e.
at the half of the function that produces the answer. Ported from the authority at
`CFGSDataflowConditionalTree.cpp:519`.

⭐ AND ENTRIES 073-080 — the constant-splat port name (`constValToField`), the same-block guard
(`sameBlock`) and the use-erasing walk (`eraseOp`) in
`src/bridges/dataflow_ir_to_sentient/vc_vector_operands.rs`; the PESFP compute-fusion pass
(`fuseComputeOps`) in `vc_vector_chain_to_sentient_pesfp.rs`; and the Loop Mask Tree's walk, lookup
and node layer (`LoopMaskTree::walk`, `findNodeFromOp`, `LoopMaskNode`'s constructor and its virtual
destructor) in `vc_loop_mask_tree.rs`.

⛔ 073'S RUNTIME REFUSAL IS DEAD CODE IN THE REFERENCE, AND THAT IS WHY THE DOMAIN IS A TYPE.
`DT_ERROR` expands to `DT_CHECK_MSG(false, …)`, which THROWS (`util/dt_exception.hpp:110-121`), so
`constValToField`'s `return ""` (`VectorOperands.cpp:261`) is unreachable and so are both callers'
`if (value == "") { emitError; return nullopt; }` arms (`:284-287`, entries 166/167). The accepted
domain is exactly `{0, 1, 2, 3}` — a four-variant `ConstantOperandValue` — and the `double`
comparison chain went with it: what the chain decides is WHICH OF FOUR pseudo-unit ports
(`SentientTypes.td:100-106`) the splat is read from, not a number.

⛔⛔ 074 PROVED `kind == Constant` IS NOT `isa<arith::ConstantOp>`, AND THE TWO DISAGREE BOTH WAYS.
`getOperandFromConstantBitstreamOp` tags a `vectorchain.constant_bitstream` operand `Constant`
(`VectorOperands.cpp:305`) and the trivial-shuffle path re-tags an operand built over an
`arith.constant` as `NFWD`/`ConstantBitstream` (`:317-336`). `sameBlock` asks the OP, so the port
takes the enclosing scope as a parameter and interrogates the op class there — the precedent
`agen_helper.rs`'s entry 038 set. ⚠️ The exemption is unreachable from the one caller today:
`OperandReuse::setReuseInformation` guards with `type_ != Constant` (`OperandReuse.cpp:31`). Ported
anyway.

⛔ 075 CAN DELETE AN INNOCENT OP IN THE REFERENCE. `for (auto &use : op->getUses())` yields ONE ENTRY
PER USE, so an op that reads the same result twice is pushed to `to_be_erased` twice and `e->erase()`d
twice (`:809-819`) — a double free there, and in an arena a second erase at a stale position. The port
deduplicates, and erases in DESCENDING position order because every position a removal invalidates is
lexicographically greater than the one removed.

⭐⭐ 076 SPLIT LEGALITY IN TWO, BECAUSE `applyPartialConversion` DOES. Sixteen patterns are installed
(`VectorChainToSentientPESFP.cpp:1247-1255`) but only thirteen op classes are `addIllegalOp`
(`:1262-1265`); an op that is neither legal nor explicitly illegal is still offered to the patterns
and left alone if none matches. So `ElementWiseCompareOpLowering`, `ElementWiseSelectionOpLowering`
and `ShuffleOpLowering` DO rewrite when they match while their ops are not required to lower —
`Unlowered::Required` and `Unlowered::BestEffort` keep that asymmetry auditable instead of merging it
away, and only the required work reaches the `todo!`.

⭐⭐ 079 GAVE THE ARENA NODE AN `OpId`, NOT A BORROW, AND THE REFERENCE FORCED IT. A `LoopMaskTree` is
built once per PT program unit and then threaded through three MUTATING passes — `fuseNonComputeOps`,
`fuseComputeOps`, `lowerDanglingNonComputeOps` (`VectorChainToSentientPT.cpp:1003-1014`) — while
`insertPTMaskOps` walks it inserting `set_mask`/`incrmask` ops (`LoweringPTMasks.cpp:74-100`). A
`&DfirOp` held across those calls forbids exactly the rewrites the tree exists to drive, and
`updateNode` (entry 237) exists BECAUSE a lowering replaces the op a node names. ⚠️ This diverges
deliberately from `LocalOpNode::new` (entry 101), which took `&'p DfirOp`: that node is built, read
and dropped inside one non-mutating walk and never asked which op it holds. And `push_child` takes the
op by value rather than as an `Option`, so *only the synthetic root names no operation* is a type
rather than a comment.

⭐ 078 DROPPED THE `DenseMap` MEMO FOR AN ARENA SCAN, and the two answers coincide exactly: the map
holds an entry for every node except the root (`LoopMaskTree.cpp:152`, `:179`, while `:175` inserts
nothing) and the scan skips the root because its operation is `None`. What the map would cost is a
second source of truth for its two writers (entries 236, 237) to keep in step — which is the bug its
own `"op already in map - should not happen"` check exists to catch.

⛔ 077'S ACTION RETURNS NOTHING, AND ONE CALLER READS AS THOUGH IT DOES NOT. `kBFS` discards the
action's result (`(void)action(curr_node)`, `OperationTree.cpp:190`), so
`analyzeAndInsertMaskOps`'s `signalPassFailure(); … return nullptr;` (`LoweringPTMasks.cpp:193-197`)
does **not** stop the walk — the rest of the queue is visited and further masks are still lowered.
Only the two GUIDED orders steer by the return value, and they are unscheduled. A test pins it.

⭐ 079 PUT THE OPERATION IN THE **BASE**, WHICH IS WHERE THE REFERENCE KEEPS IT
(`OperationNode::operation_op_`, `OperationTree.hpp:185`; `LocalOpNode(Operation *op)
: OperationNode(op)`, `FlatteningLocalRegions.cpp:51`) — but entry 101 had already put it in the
PAYLOAD, as a `&'p DfirOp`. In C++ there is one `insertChildNode` because the node's constructor,
which the caller runs, has already set `operation_op_`; here the arena mints the node, so those two
constructors reach it as TWO inserts over one shared body: `push_named_child` (an `OpId` in the base,
so `find_node_from_op` can search it) and `push_child` (the base's `op` stays `None`; the payload
names it). ⚠️ A tree built through the second must not be searched with `find_by_op`.

⛔ **FIVE SCOPE HOLES FOUND IN THIS BATCH, NONE IN THE 384 AND NONE IN THE 106 EXCLUSIONS.**
- The sixteen PESFP compute-pattern `matchAndRewrite` bodies (`VectorChainToSentientPESFP.cpp:328`,
  `:569`, …) — entry 076 installs them and nothing ports them, so `fuseComputeOps` reaches a `todo!`
  naming the first pattern it needs.
- `VectorOperand::sameBlock`'s SECOND overload, over a
  `SmallVectorImpl<optional<VectorOperand>>&` (`VectorOperands.cpp:670-685`), which is what
  `VectorChainToSentientPT.cpp:142` and `VectorChainToSentientPESFP.cpp:121` actually call. Its body
  is entry 074's applied to every element — same two failure arms, one `return success()` at the end
  — so it is a fold over the ported one, but no caller can be built without it.
- `VectorOperand::eraseOp`'s SECOND overload, taking a `ConversionPatternRewriter` and an
  `erased_list` (`:829`), called from `VectorChainToSentientPESFP.cpp:1062`. It is the one that
  deduplicates — `std::find` against both lists (`:837-841`) — which is what entry 075 had to decide
  for itself.
- `OperationNode::breadthFirstWalk` (`OperationTree.cpp:183`) and the six other
  `LoopMaskNode::walk<>` specializations (`LoopMaskTree.cpp:26-93`). The BFS one is written here
  unanchored, beside the rest of the base layer; the other six orders are not.
- `LoopMaskNode`'s and `MaskNode`'s COPY constructors (`LoopMaskTree.hpp:33`, `:73`), and
  `MaskNode`'s is a reference DEFECT: `MaskNode(const MaskNode &n) : LoopMaskNode(n.getOperation())`
  copies the op and silently drops `start_val_` and `increment_`, leaving both uninitialised. Nothing
  in the tree calls it today.

⛔ **TWO HOLES IN THE VALUE-BASED SIMPLIFICATION PATH, NEITHER OF THEM MINE TO FILL.** Entry 381
(`simplifyValueBasedConditionals`) needs the manager to be constructible and parseable, and:
- `parseConditional` (`CFGSDataflowConditionalTree.cpp:534`) is excluded above under *MLIR
  printing/parsing/verification*, which is a name-based misclassification — it parses no text. It
  walks the conditional tree and fills `val_array_` with the value each iteration yields, and it is
  the only writer of the array entry 099's destructor frees.
- `ConditionalSimplificationManager`'s CONSTRUCTOR (`CFGSDataflowConditionalTree.hpp:55-76`, 22
  lines) is in no batch and on no exclusion list. Entry 099 is the DESTRUCTOR at `:78`. The
  constructor is where `is_candidate_` is decided and where `num_iterations` sizes the array, so
  without it the destructor has nothing to own.

⛔ **THREE MORE SCOPE HOLES, ALL UNDER ENTRY 261 AND ALL IN `dcc/src/Dialect/Uniform/Utils.cpp`.**
`UniformQueryMapsCanonicalization.cpp:59` calls `simplifyQueryMapWithSameTarget` (`:1263`), which is
the WHOLE of what the pass does to a query map — the pass without it only erases — and which in turn
calls `getRegionOpAndIndex` (`:343`) and `getUnitsOfRegion` (`:526`). None of the three is in the 384
or in the 106 exclusions. All three are written unanchored beside entry 261's port, the last of them
narrowed to the two region-owning ops this island carries: the reference's walk also stops at a
`uniform.equalize_pattern`, which `islands/dataflow_ir/dialects/uniform.rs` deliberately omits.

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

⚠️ **THE RE-AUDIT OF 001-024 FOUND FIVE DOCUMENTED CLAIMS THE AUTHORITY CONTRADICTS**, all in
`agen_access_details.rs`, and no defect in the ported code itself. (1) ⛔ **024'S PROMOTION MOVES THE
BURST OUTWARD, NOT INWARD.** `computeBurstAndGroup` scans `for (int i = time_bounds.size() - 1;
i >= 0; --i)` (`AccessDetails.cpp:798`) over a vector whose index 0 is the OUTERMOST time dimension —
`constructTimeLoops` builds loop `idx` inside loop `idx - 1` and stops at `loop_num = burst_index`
(`Helper.cpp:1807-1857`), so the burst is claimed on the innermost valid dimension first and the
promotion hands THAT dimension to the interleave group. The anchor doc said "moves the burst inward"
and the test asserted the mirror image (burst 0 → 1, group 0), a state the scan cannot reach; it now
replays burst 1 → 0, group 1. (2) `TimeOffsets`' `Default` was grounded in "`calculateTimeOffsets`'s
own `const_value` fallback" — that function has no fallback, it takes `tmp_time_offsets.back()`
unconditionally (`dialect_utils/Agen/Utils.cpp:116`); the `? … : 0` ternary at the cited lines is
`constructIteratorCoeffDict`'s, and the real ground is that `getFlattenedAffineExpr` always emits a
trailing constant coefficient. (3) `IndicesCoeffDict`'s `Vec` was justified by the producer's
insertion order — a `DenseMap` preserves none; the ordered walk is `initMASData`'s over
`ad.getIndices()` (`MutableAddrSplitting.cpp:724-737`) and the ground is positional pairing with
`indices_`. (4) The struct's wave list stopped at entry 008 while entries 009-015's seven members are
declared beside those. (5) The composite fixture cited `/tmp/ktir_ref/export/debug/dfir.mlir:78-84`,
which does not exist on this host; re-pointed at
`tests/sentient_corpus/group_0__g0_7_matmul.dfir.mlir:25-30`, with the extents it does NOT share
stated rather than implied.

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

⭐ AND ENTRIES 057-064 — `OperandReuse`'s `setReuseFlag` and `dominates` (`vc_operand_reuse.rs`),
and the six `VectorChainHelper` units in `vc_vector_chain_helper.rs`:
`isSentientBinaryLogicalOp`, `getInputPrecisionFromOperand`, `getResultPrecisionFromOperands`,
`getComputePrecisionOfOp`, `hasConstantBounds` and the `merge_and_pack_type` constructor.

⭐ ONE VOCABULARY, NOT TWO (AGAIN): 057/058 landed after 055/056 and were rebased onto that wave's
`OperandTag`, `DataId` and `DataOriginId` rather than carrying a second `OperandTag` with a bare
`int id_`. The table stays keyed by the `Val` a data origin produces — the sentinel lives in
`DataId::Unassigned` and nowhere else, so `setReuseFlag`'s default-inserting `operator[]` gets the
reference's `{-1, false}` from the derived `Default`. ⛔ AND `dominance_info_` DOES NOT ARRIVE WITH
058 after all: `OpId` carries an op's path through the region tree, so `dominates` is a pure function
of its two arguments and the reference's cached `DominanceInfo` has nothing left to hold. `dominates`
takes two `OpId`s while the table takes a `Val`, which is the same split the C++ has (`Operation *`
as a position here, as a map key there).

⛔ 062's PACK/MERGE SHORT-CIRCUIT RUNS BEFORE THE ELEMENT-TYPE QUERY, and that order is the whole
function: `MergeOp` appears in neither `getVectorType` nor `getCustomVectorType` (`Utils.cpp:520-702`),
so reaching `getElementType` for one is `DT_CHECK_MSG("Type is not a known vector type")`. It is also a
DIFFERENT answer — IBM's `gcvt.mlir` packs two `vector<128xf8E4M3FN>` and the compute reads
`ComputePrecision = #sentient<precision fp16>`. `vectorchain.pack` and `vectorchain.merge` were added
to the DataflowIR island for it; the pack printer is byte-exact against `fpuop.mlir:217` and the merge
printer is derived from the `.td` and NOT byte-checked, because no vendor text writes a
`vectorchain.merge`.

⛔ AN EMPTY `affine_set` STILL HAS CONSTANT BOUNDS, and 063 must say so.
`affine_set<(d0) : (d0 - 64 >= 0, -d0 + 63 >= 0)>` admits no integer, yet its `LB` is 64 and its `UB`
is 63 — and it is the all-lanes-off mask of twenty `create_affine_mask` ops in
`Conversion/VectorChainToSentientPESFP/mixed_precision.mlir`, each lowering to
`sentient.scalar_constant {value = 0 : si64}` (`:288`, `:296`). A `LB <= UB` test in
`hasConstantBounds` would turn twenty legal masks into an `emitOpError`.

⚠️ 060/061's `fp80 -> fp8` REMAP IS UNREPRESENTABLE, WHICH IS STRONGER THAN DROPPED. MLIR's
`Float80Type` stands in for fp8 (`VectorOperands.cpp:227-232`); `ElemType` has no F80 and
`sen::Precision` has no `Fp80`, so the remap collapses into the type. `dlfp16` is dead in the
reference too — `getPrecisionInString` can only produce `int<n>`, `bf16`, `mxfp<n>` or `fp<n>`.

⭐ AND ENTRIES 129-136 — `TransformPagedMemView`'s de-paging base class and its use chains, all in
`src/bridges/dataflow_ir_to_sentient/tf_transform_paged_mem_view_impl.rs` (the file's first code):
`TPMVBase`'s three virtuals (`getUseChain`, `cloneUseChain`, `eraseMemOpAndUseChain`),
`TPMVVector`'s constructor, `TPMVVectorLoad`'s `eraseMemOpAndUseChain` override, and three
`TPMVComposite`/`TPMVVectorLoadStore` members — `getStoreOp`, `addTimeDimIndicesRanges`,
`identifyTimeDimForExplicitLoops`. 13 unit tests, built from the vendor's own
`dcc/test/Transform/TransformPagedMemView/paged_mem_view_{loads,load_and_store}.mlir` inputs. No
equivalence tests: there is no C to call.

⭐⭐ THE HEADER'S USE-CHAIN DIRECTION CONTRACT IS WRONG, AND THE TYPE FIXES IT. `TPMVBase::getUseChain`
documents *"If `mem_op` is the first element of the returned vector, it is in order. If `mem_op` is
the last element, it is in reverse order"* (`TransformPagedMemViewImpl.hpp:321-327`) — but BOTH dialect
implementations put `mem_op` first, and `VectorStoreOp::cloneUseChainToNewOp` says the opposite about
its own input two files away (*"The use chain is stored in reverse order"*, `Agen.cpp:229-230`).
Measured: `VectorLoadOp::getUseChain` walks consumer-ward (`:115-136`) and `VectorStoreOp::getUseChain`
walks producer-ward (`:207-222`), which is why their two `eraseOpAndUseChain` loops run in opposite
directions (`:176-177` reversed, `:257` as-is) to compute the SAME thing — teardown consumer-first.
`UseChain` carries the direction and `consumer_first()` is that one rule.

⭐ AND THE REFERENCE'S DEAD BRANCH BECAME THE LIVE ONE. `eraseOpAndUseChain` opens
`if (use_chain.empty()) { op->erase(); }` (`Agen.cpp:173-175`), unreachable in C++ because
`getUseChain` asserts on a non-linear chain first. Those three asserts are one question — is the
chain linear — and the header's own *"Empty if there isn't a use chain"* is its answer, so entry 129
answers `UseChain::None` where the reference aborts and then that branch is exactly right: erase the
load, leave what reads it alone.

⛔ 131 AND 132 JOIN THROUGH A SYMBOL NUMBERING, AND IT IS NOW CHECKED BY A TEST.
`addTimeDimIndicesRanges` APPENDS after `calculateIndicesRanges` on the same vector
(`TransformPagedMemViewImpl.cpp:1010-1017`), so time dim `i` lands in slot `i + num_non_time_dims`,
which is exactly the symbol `identifyTimeDimForExplicitLoops` looks up (`:967`) and the index
`addConstraintsForIVRanges` reads (`:94-102`). `NonTimeDims::sym_for` holds that arithmetic once.
⛔ The call site has TWO adjacent `int` dimension counts — `time_set_.getNumDims()` and
`subscripts_map_.getNumDims()` (`:966`, `:1024-1025`) — so the time count arrives as the `IntegerSet`
it is read off and the two cannot be exchanged.

⛔ 131'S `DT_CHECK_MSG(b - 1 >= 0, "no special time bound values should exist")` IS A TYPE. `TimeSteps`
is a `NonZeroU32` behind an `of(TimeBound)` door, so `kInvalid`, `kCoalesced` and the reachable
`Steps(0)` cannot reach the subtraction and `last_index()` is total. `IvRange` widens the reference's
`std::pair<int, int>` to `i64`: `calculateIndicesRanges` narrows an `int64_t`
`getSingleConstantResult()` into it (`:73-75`), and the constraints these become take
`AffineExpr::Const(i64)`.

⚠️ 132'S `auto it` IS A `bool`. `if (auto it = page_dependent_time_syms_.find(...) != ...end())`
binds the comparison, not the iterator, because `=` is looser than `!=`. The behaviour is the
intended one; only the name misleads. Nothing was ported around it.

⛔ **`TPMVBase`'S CONSTRUCTOR IS MISCLASSIFIED IN THE EXCLUSIONS.** It is binned as `comp_`
(`…/TransformPagedMemViewImpl.hpp:30`) under *"a one-line C++ field accessor; in Rust the field
itself"*. It is not an accessor: it is a member-initialising constructor that also seeds `mem_ops_`
with a one-element list — and `initialize()` asserts `mem_ops_.size() == 1` in all six concrete
classes (`:659`, `:707`, `:756`, `:1081`, `:1134`, `:1187`), so the singleton is an invariant that
constructor establishes. Entry 136 delegates to it, so `TpmvBase::new` exists and carries NO anchor;
the exclusion is reported rather than overruled.

⛔ **AND THE EXTRACTOR KEPT ONLY TWO DEFINITIONS PER OVERRIDDEN NAME, so EIGHTEEN `TPMV*` member
definitions are in neither the 384 nor the 106 exclusions.** Counted in the authority: this file defines
`eraseMemOpAndUseChain` five times (`hpp:364`, `cpp:698`, `cpp:747`, `cpp:817`, `cpp:1266`) and the
campaign scheduled two (135 and 129); `getUseChain` three times (`hpp:328`, `cpp:673`, `cpp:721`) and
scheduled two (133 and 126); `cloneUseChain` three times (`hpp:341`, `cpp:678`, `cpp:726`) and
scheduled two (134 and 127); `createNewMemOp` six times over one pure virtual (`hpp:357` `= 0`, then
`cpp:684`, `:732`, `:779`, `:1120`, `:1173`, `:1242`) and scheduled one (128). Unscheduled and
unexcluded:
`TPMVVectorStore::{getUseChain, cloneUseChain, eraseMemOpAndUseChain}` (`cpp:721`, `:726`, `:747`),
`TPMVVectorLoadStore::{createNewMemOp, eraseMemOpAndUseChain}` (`cpp:779`, `:817`),
`TPMVVectorStore::createNewMemOp` (`cpp:732`) and the three composite `createNewMemOp` overrides
(`cpp:1120`, `:1173`, `:1242`) plus `TPMVCompositeLoadStore::eraseMemOpAndUseChain` (`cpp:1266`).
⛔ RE-COUNTED FOR ENTRY 356, EIGHT MORE: `initialize()` is a pure virtual (`hpp:66`) with SIX
`override final`s (`cpp:658` = 202, `:706`, `:755`, `:1080`, `:1133`, `:1186`) and only 202 is
scheduled; `initialize_time()` is a second pure virtual (`hpp:468`) with three (`cpp:1095` = 326,
`:1148`, `:1211`) and only 326 is; and `TPMVComposite::run` (`cpp:855`) is unscheduled while its twin
`TPMVVector::run` (`cpp:647`) is entry 373. Totals: 51 `TPMV*` member definitions in the `.cpp`, 33 of
them scheduled, and the exclusions carry four `TPMV*` items — all `hpp` fields (`:30`, `:45`, `:50`,
`:458`).
Reported, not filled — a `Replaces:` anchor on an entry nothing scheduled would count as coverage no
worklist asked for.

⚠️ AND FOUR OF THE SIX CONCRETE CLASSES INHERIT 133/134/135 UNCHANGED, which makes the empty bodies
live behaviour rather than fallbacks. Only `TPMVVectorLoad` and `TPMVVectorStore` override
`getUseChain`/`cloneUseChain`; `TPMVVectorLoadStore` and every composite genuinely have no linear
chain — a composite transfer's consumers live inside its own region, and a load-and-store pattern's
store is reached through entry 130 instead. ⚠️ There is no trait to dispatch through yet: entries 137
and 138 are the derived constructors, so the three virtuals are inherent methods on `TpmvBase` today
and `erase_vector_load_and_use_chain` is `TPMVVectorLoad`'s override standing beside them as a free
function.

⭐ AND ENTRIES 121-128 — the paged-memory-view transform's page-guard and use-chain layer, all in
`tf_transform_paged_mem_view_impl.rs`: `createInequalityCondition`, `setBuilderToInsertRef`,
`calculateStartElementsForPage`, `createNonPagedMemView`, `cloneMemViewIfNonPaged`, `getUseChain`,
`cloneUseChain` and `createNewMemOp`.

⛔ A MUTATED `OpBuilder &` IS NOT A RETURN VALUE, and both 121 and 122 hand one back. The reference
reassigns the caller's builder to the then-region of the `scf.if` it just built
(`TransformPagedMemViewImpl.cpp:275`) so that whatever is emitted next lands inside the guard. Here the
guard is a VALUE: `Condition` holds the per-dimension `BoundGuard`s and `Condition::wrap(guarded)`
nests the statements inside-out. 122's two arms are then `InsertRef::Conditional` (wrap) and
`InsertRef::MemOp` (hand the statements back unwrapped, for the caller to splice at the op's own
position) — no cursor, and no way to emit into a region that was never opened.

⭐ 121 IS TWO NESTED ONE-ARMED `scf.if`s AND NOT ONE `arith.andi`. The reference builds the `sge`
guard, descends into its then-body and builds the `sle` guard there, so the four values are minted
lb_const, lb_cond, ub_const, ub_cond and the SSA numbering follows. `BoundGuard` carries its
predicate rather than hard-coding the pair, because 120 (`createEqualityCondition`) is the `eq`
sibling one guard wide and belongs in this same type rather than a second one.

⛔ THE ISLAND GREW `dataflow.get_paged_logical_memory_view`, WHICH IS THIS TRANSFORM'S ENTIRE INPUT —
without it none of 123-128 had anything to run on. `PagedMemView` carries `pages: Vec<Page>`, and
`Page` PAIRS an `idx_set` with a `start_addr`, which turns the verifier's *"there should be a start
address and idx_set for every page"* into the pairing. `PageRect`'s per-dimension
`PageSpan { lo, hi }` turns both *"idx_set should be hyper rectangular"* and 123's *"expected
constant lower bound"* into the parameter type: 123 is `spans.iter().map(|span| span.lo)`, and its
test EXECUTES that equivalence against `IntegerSet::constant_bound(Lb, dim)` rather than asserting
it. The printer reproduces `DataflowOps.cpp:239-267` and is checked verbatim against
`dcc/test/Dialect/Dataflow/paged_mem_view.mlir:19-22`.

⚠️ AND THAT TEST FOUND A PRINTER DEFECT ONE LEVEL DOWN: `AffineExpr::Add(d1, -2)` printed
`d1 + -2` where MLIR writes `d1 - 2`, and that is the lower half of every page whose span does not
start at zero (`#set1` of the same file). A negative addend is now a subtraction — the same reason
`d2 * -1` was already printed `-d2`.

⭐ THE THREE ASSERTS IN 126'S WALK BECAME STOPPING CONDITIONS. `getUseChain` aborts on an op with
more than one result and on a result with more than one use; `let [res] = results[..] else { break }`
stops the walk at exactly those shapes, so the caller gets a chain that ends where the reference's
assert would have fired — which is what a chain that cannot be cloned means. 127's
`assert(isa<VectorLoadOp>(new_op))` became a parameter type instead: `VectorLoadRef` does the
`cast<agen::VectorLoadOp>` once and 126/127/128 all take one.

⭐ ONE VOCABULARY, NOT TWO (AGAIN): 121-128 landed after 129-136, in the same file, and were rebased
onto that batch's names rather than carrying a second set. Entry 129 ported the DIALECT walk
(`agen::VectorLoadOp::getUseChain`, `Agen.cpp:114-134`) as `vector_load_use_chain` because its
teardown needs the same chain — so 126, whose C++ is a cast and a forward, IS that forward, and it
answers in the `UseChain` that 133's `TPMVBase::use_chain` returns. ⛔ AND THE FORK CASE IS 129'S
READING, WHICH IS THE STRONGER ONE: a value with two readers means there is NO chain
(`UseChain::None`, the contract's own "Empty if there isn't a use chain",
`TransformPagedMemViewImpl.hpp:322`), not a chain truncated at the fork — half a chain cloned into a
guarded branch is a worse answer than none. 129's `VectorLoadOp` replaced the `VectorLoadRef` this
batch had narrowed for itself, and gained the `getResult().getType()` that 128 rebuilds the load
with.

⭐ AND THE ISLAND GAINED THE MUTABLE HALF OF ITS OWN CENSUS, for 125/127/128: `operands_mut`,
`results_mut`, `replace_uses_of_with` (one entry per USE, mirroring `uses()`) and
`clone_with_fresh_results`. ⛔ `operands_mut` EXCLUDES TWO OPERANDS BY DESIGN — a link end (`Send`'s
`to`, `Receive`'s `from`), because `link::Link` hands its two ends out once by consuming itself, and
`vectorchain::Predicate`'s mask, which carries the type it was defined at. A send's `data` IS
included, and that is what a chain clone re-points. ⛔ `clone_with_fresh_results` DOES NOT REMAP
REGIONS; both callers are region-free chain ops (`Agen.cpp:114-134`).


⭐ AND ENTRIES 065-072 — the rest of `VectorChainHelper`'s fusion and mapping layer in
`vc_vector_chain_helper.rs` (`fuseCompareAndSelectIntoMinOrMax`, `resetSentientFMAsIfExists`,
`redefineConstantVectors`, `getVectorBinaryToSentientBinary`,
`getVectorElementWiseCompareOperatorToSentientBinaryOperator`, `getVectorTernaryToSentientTernary`)
and `VectorOperand`'s two link resolvers (`getOperandFromReceiveOp`, `getOperandFromSendOp`) in
`vc_vector_operands.rs`.

⛔ 065's TWO OUT-BOOLS BECAME ONE THREE-STATE ANSWER. `bool& fusion_to_min, bool& fusion_to_max` has
a fourth state the reference reaches only to `DT_ERROR` on it; `MinOrMaxFusion` is
`NotFused | ToMin | ToMax` and the impossible pair is gone from the type. The arm order is the
reference's and it carries information: `compare_gt`/`compare_ge` with the arms in the compare's own
order is a MAX and reversed is a MIN, and `compare_lt`/`compare_le` is the transpose of that — eight
rows, all eight pinned by a table test. `compare_eq`/`compare_neq` is the one `todo!`, and it is the
reference's `DT_ERROR` at `VectorChainHelper.cpp:460-462`.

⭐ AND 065 IS UNPORTABLE WITHOUT OPERATION EQUIVALENCE, whose comment says why:
*"some times constant operands are duplicated, and direct match may result in spurious mismatches"*.
`relu.mlir` is that case — `:50-51` are two separate `dense<0.0>` constants, the compare reads `%cst`
and the selection reads `%cst_1` (`:73-74`), and it still fuses to the `max` at `:35`/`:37`.
`dcc::OperationEquivalence` (`dcc/src/Analysis/OperationEquivalence.cpp`) is not on this list, so what
landed is the minimum 065 needs: a `skeleton()` that clones an op, clears its regions and blanks every
`Val`, then compares operands pairwise through their defining ops. ⛔ IT IS BLANK-AND-COMPARE RATHER
THAN A HAND-WRITTEN PER-VARIANT TEST SO THAT NO COMPARISON CAN FALL BEHIND THE `Op` ENUM — a new
attribute is compared the day it is added. Three divergences are documented at the function: the
reference's `dbgName` filter has nothing to filter here, `IntegerSetAttr` equality is plain rather
than order-insensitive, and the equivalence-class memo is omitted.

⭐ 066'S WALK IS PROVABLY COMPLETE, AND THAT IS A FACT ABOUT THE ISLAND. `unit.walk<PreOrder>` visits
every nested operation; the Rust recursion has exactly three arms because `sentient::Op::For` and
`sentient::Op::If` are the ONLY region-carrying variants in the whole sentient island (checked across
all seven of its dialects). `si32 = -1` is `data_id: None` — the sentinel is the absence. MAC and
BINARY only: a unary or a ternary keeps its data IDs, which is a test.

⛔ 067 IS POSITIONAL REWRITING, NOT VALUE SUBSTITUTION, and that is the difference between a port and
a paraphrase. The reference clones the constant with an `OpBuilder` positioned at EACH USER, records
`{owner, new result, use.getOperandNumber()}`, and only then calls `setOperand` — so one op reading a
constant twice gets TWO clones, and a nested user gets its clone inside its own block. Both are tests.
`!use_empty()` is a real guard: an unused vector constant is neither cloned nor erased.
`const-vector-multiple-uses.mlir` is the golden — `%cst_1` is read six times at four depths in the
input (`:179`, `:184`, `:185`, down to the `multiply_and_accumulate` at `:310`) and NO `dense<…>`
survives into the expectation at `:15-24`. ⭐ AND THE SCOPE IS THE MODULE, NOT A UNIT: both call sites
pass `module_op` (`VectorChainToSentientPT.cpp:983`, `VectorChainToSentientPESFP.cpp:1385`), so the
port takes the whole `Program` — preamble and every unit. The `isa<VectorType,
dataflow::CustomVectorType>` test is discharged by the variant, because `arith::Op::DenseConstant`'s
`ty` IS a `Vector` while `Constant` is an `index` and `ConstantInt` an `i<n>`.

⭐ 068 CARRIES NO `todo!` AT ALL: `vc::BinaryOp` has exactly twelve variants and the reference names
all twelve, so its `DT_ERROR` is unreachable once the argument is a Rust enum. It is still not an
identity — `sen::BinaryOp` has twenty, the two converts, `merge`, `pack` and the four `fcmp`s having
no `vectorchain` counterpart. 069 is four arms for six inputs BY DESIGN: the ISA has no element-wise
`gt`/`ge` and the caller reorders the operands instead (`VectorChainToSentientPESFP.cpp:450-485`),
which `fcmp_select.mlir` shows both halves of — input `:136` is `compare_gt` and expectation `:48` is
`fcmp_lt` with `opA`/`opB` swapped. ⛔ 069'S `todo!` MESSAGE SAYS "Ternary" BECAUSE THE REFERENCE'S
DOES: it is a copy-paste from 070, kept verbatim so a grep for the text finds the C++ line.

⛔ 071/072 ARE THE LINK NAME, AND THE LINK NAME IS THE OPERAND. Both resolve a peer unit plus the
asking component into a `sen::Port`, and the reference's string arms became total matches over
`DfirUnit`, so a nineteenth unit has to say which arm it belongs to. `generic == PT` is every
`PtRow(_)` and `generic == CROSSPTNLINK` is one unit, both checked against `senCompToGenericComp`
(`sys-arch-spec/arch_enums.cpp:124-211`). Goldens: `loweringXRF_with_if_branch.mlir:145` receives from
an `l0lu` and expects `opA = #sentient<compute_port west>` (`:49`); `xrf_increments.mlir:413-457`
receives from an `lxlu` on a PT and expects `opC = … north` (`:71`); `xrf_increments.mlir:422` sends
to a `ptrow1` and expects `ResultForwarding = [#sentient<compute_port south>]` (`:81`).
⭐ THE ASYMMETRY IS THE REFERENCE'S: a PE/SFP may SEND to the L0 and to either LX half, and may not
RECEIVE from the L0 at all. ⛔ AND THE PT BRANCH OF 072 HAS NO `else` — an unmatched destination falls
out with `link` empty into the bottom `if (link.empty())`, whose message says *"Unsupported
destination for PE/SFP FMA"* even though the unit asking is the PT; both branches are written and the
message is reproduced as the reference words it. Two of the four failure paths in each are
unreachable from a resolved `DfirUnit` and are documented rather than written: `findUnitType`'s empty
optional is a disagreement between attributes the caller resolved earlier, and
*"Unknown receiver"*/*"Unknown destination"* is a lookup into the same table `DfirUnit::spelling`
came out of.

⛔ THE `sfpring` RE-CHECK IS A TAUTOLOGY IN THE REFERENCE ITSELF: 071's
`stringToSenComponents.at(unit_str) == SFP` sits inside `else if (record->second == SFP)`, where
`record->second` IS that lookup. And 072's `emitWarning` arm — SFP→SFP below DD1 — is unreachable on
every arch this crate builds for, which is what `supports_sfp_ring` says; it is written because it is
the function and it is where an older generation lands the day one is added to `IsaGen`.

⭐ THE ISLAND GREW FOR 066/067, AS THE BRIEF REQUIRES. Both are IN-PLACE REWRITES and the islands were
emit-only, so `dialects::vals_mut`/`operands_mut`/`regions_mut` arrived in the DataflowIR island
(`dialects/mod.rs`) as ONE total match with no wildcard arm — a new op cannot be silently skipped by
the rewriters — plus `Role` to tell an operand from a result, `ProgramUnits::iter_mut`, and a
`pub(super) val_mut` on `SendEnd`/`RecvEnd`/`Predicate` for `setOperand`. ⛔ `Predicate::val_mut`
CANNOT TOUCH ITS `ty`, so the invariant that type exists for still holds.

⭐ AND ENTRIES 137-142 — the two leaf constructors of the de-paging hierarchy and its manager, the
unit filter, and the loop-info query: `TPMVVectorLoad`/`TPMVCompositeLoad` in
`tf_transform_paged_mem_view_impl.rs`, `TransformPagedMemViewManager::run` in
`tf_transform_paged_mem_view_manager.rs`, `removeCoresCoreletsFoldsFromProgramUnit` and
`isDataTransfer` in `tf_unit_filtering.rs`, and `getDataflowForLoopInfoIfIV` in `tf_utils.rs`.
49 unit tests. No equivalence tests: there is no C to call.

⛔ 137 AND 138 ARE THE **DERIVED** CONSTRUCTORS, NOT THE CLASSES THEY ARE NAMED AFTER, which is the
same reading entry 136's anchor records: `hpp:397` is the mem-initializer `: TPMVVector(mem_op, comp)
{}`, so `e137_TPMVVector` is `TPMVVectorLoad`'s constructor and `e138_TPMVComposite` (`hpp:519`) is
`TPMVCompositeLoad`'s. Six leaf types exist because entry 139's ENTIRE content is choosing which one
to build — a single struct would make that function return the same thing six times — and the five
siblings of 137/138 carry no anchor because the extractor deduplicated them by text
(`: TPMVVector(mem_op, comp) {}` at `hpp:397`, `:415`, `:433`; `: TPMVComposite(…)` at `:519`, `:534`,
`:549`).

⛔ 137'S SIBLING DROPS AN ARGUMENT ON PURPOSE. `TPMVVectorLoadStore(Operation *mem_op,
agen::VectorStoreOp &store_op, SenComponents comp)` forwards `mem_op` and `comp` and keeps the store
NOWHERE (`hpp:431-433`); `initialize` re-derives it through entry 130's `getStoreOp` and appends it to
`mem_ops_` (`Impl.cpp:512-520`). The parameter and its fate are kept in the port, because a signature
quietly narrowed to two arguments would hide the reason entry 130 exists.

⛔ 138'S OWN INPUT HAS NO ISLAND OP, AND THAT IS RECORDED RATHER THAN INVENTED. `agen.composite_load`
and `agen.composite_store` (`paged_mem_view_loads.mlir:331`, `paged_mem_view_stores.mlir:361`) are two
of the eleven `agen` ops the island does not declare; only `composite_load_and_store` is present, so
two of the three composite leaves are reachable from a vendor test and from nothing this crate emits.
The campaign's *add the op to the island* rule was applied to entry 139's actual input — the paged view
itself, which the 121-128 wave added — and minting two composite ops no emitter produces would be the
stand-in the crate rules forbid.

⛔ 139's DISPATCH IS ASYMMETRIC AND THE ASYMMETRY IS THE PORT. Both load-and-store arms build a
`TPMVVectorLoadStore` over the **load** (`TransformPagedMemViewManager.cpp:32`, `:46`) — even the arm
that reached the pattern from the store and had to walk BACK to the load, where the reference passes
`load_op` and not the op it was handed. `Selection` carries that choice; a port that passed `op`
through both arms would type-check and de-page the wrong op.

⛔ 140's TWO FILTER FINDINGS, both recorded beside the port. `getCoreId` returns **-1** for a unit with
no `core` attribute (`DccExtContext.cpp:78-124`) and -1 is in no filter set, so a non-empty
`filter-cores-except` ERASES the HBM handle — modelled as `Residency::Global` having no core. And a
`Residency::CoreWide` unit carries `corelet = 0` explicitly while a `Residency::Scratchpad` carries
none, so `filter-corelets-except=1` erases the first and keeps the second; that is the reason
`Residency` distinguishes them. The vendor's own `core_filtering_edge_case.mlir` (32 `lxlu-CL0`
handles, `filter-cores-except=0`, one survivor) is reproduced as a test.

⛔ 142 DEPARTS FROM THE REFERENCE ON ONE VALUE, DELIBERATELY. `(ub - lb) / step` is an `int64_t` there
and its only consumer feeds it to `new std::optional<int64_t>[num_iterations]`
(`CFGSDataflowConditionalTree.hpp:74-75`), so `4 to 0` sizes an array with **-4** and a zero step
divides by zero. `Iterations` is unsigned and saturates at zero; `LoopStep` is positive by
construction. The truncating division is kept as the reference's (`0 to 7 step 2` is three, not four),
because the array the caller sizes with it is the reference's too. ⛔ Its `scf` arm is ported and not
declined: `dyn_cast<scf::ForOp>` is the FIRST cast in the body (`Utils.cpp:99`) and case 3 of
`Transform/CFGSimplificationDataflowLevel/simplify-conditional.mlir:241` is *"Same as 2. but with
scf.for instead of affine.for"* — answering `None` there would have been a port that declines the
reference's own test.

⚠️ AND A CARRYING `scf.for` FALSIFIED THE `scf.yield` PRINTER. It printed `scf.yield %3` while the
reference writes `scf.yield %20 : index` (`simplify-conditional.mlir:311`) and every `scf.yield` with
operands under `dcc/test` carries its types — the same mandatory `type($results)` the `affine.yield`
printer already carried a note about, unreachable until an `scf.for` could carry an `iter_args`.
Fixed, with the two vendor loops as printer tests, and 001's terminator test now asserts the type list.

⭐ AND ENTRIES 159-166 — the sync's destination sort and the SFP's permutation namer:
`getUnitNameFromAListOfGetUnitOp`, `areCoreletsDifferent` and `separateBasedOnDestinationUnits` in
`dfs_dataflow_to_sentient.rs`, `OperandReuse::insertIfNotExists` in `vc_operand_reuse.rs`,
`getMaskValueConstantForNonPT`, `checkValidityOfPackAndShuffleLowering` and
`getMergeTypeFromIndices` in `vc_vector_chain_helper.rs`, and `getOperandFromConstantOp` in
`vc_vector_operands.rs`. 32 unit tests. No equivalence tests: there is no C to call.

⛔ THE ISLAND GREW THREE TIMES FOR THIS WAVE, EACH FOR A FUNCTION WHOSE INPUT IT COULD NOT SPELL.
(1) `dataflow.create_group` — 161's FOURTH bucket is `dyn_cast<CreateGroupOp>` and nothing else
(`DataflowToSentient.cpp:780-782`), and its only caller branches on that list being non-empty
(`:796-800`), so without the op the port would have been the same function with one arm deleted; the
vendor writes it six units wide at `dcc/test/L3SU/sync-op-l3su.mlir:75`. (2)
`arith::Op::DenseConstant` held `one: bool`, and 166 accepts 0, 1, **2 and 3** through
`constValToField` (`VectorOperands.cpp:250-262`) with `dense<2>` real input
(`VectorChainToSentientPESFP/splat.mlir:68`) — a boolean made two of the four compute ports
unreachable, so the field is now an `i64` splat and MLIR's `%e` float spelling is emitted from it.
(3) `vectorchain::LaneMask::as_set` — 089 and 163 each reconstructed the prefix form's elided
`mask_set`, and two derivations of one set is one too many; 089 now reads it from there.

⛔⛔ 161'S `// corelet = 1` COMMENT IS WRONG AND THE CODE IS WHAT IS PORTED. The test is
`getAttr("corelet") == getI32IntegerAttr(0)`, so a destination with NO `corelet` — the LX scratchpad,
`Residency::Scratchpad` — lands in `src_dst_lx_corelet1` beside the genuine corelet-1 units, and
`lowerSyncLXL3ToLXL3` then treats that list as a corelet-1 region. The same null means `false` in 160,
where both disjuncts need an attribute; and `Residency::CoreWide` prints `corelet = 0 : i32`
(`UnitMaterializer.cpp:62-80` against `:142-152`), so an L3 half IS on corelet 0 to `getAttr` — which
is why the `substr(0, 2) != "l3"` prefix test has to be decided first.

⛔ 163 EMITS NOTHING AND THAT IS THE REFERENCE'S OWN SPLIT: the `sentient.scalar_constant` is created
by the header template `getMaskValueForNonPT` (entry 229, `VectorChainHelper.hpp:114-125`), which owns
the builder. Its answer is always 0 across the corpus — every one-dimensional `mask_set` a
`create_affine_mask` carries in the 825 `.mlir` files is the all-lanes-live
`(d0 - N >= 0, -d0 + (N-1) >= 0)`, so `from_slice` is one past the last slice and the range is EMPTY
(`fnms_with_cast.mlir:11-12`, `:20`). ⛔ WHICH IS WHY `from_slice` MUST NOT BE UPPER-BOUNDED: refusing
`64 / 8 == 8` would refuse every mask IBM writes. A negative `lb / lanes_per_slice` wraps through
`unsigned` there and answers 0, reproduced as 0; a `to_slice` past the eight-bit slice mask is `1 << i`
out of range and is declined, which no mask in the corpus reaches. ⛔ AND THE MASK **PARAMETER** IS
IGNORED ON THIS SIDE where the PT folds it in — `fold_mode_df.mlir:92` carries a `%c0` this path never
reads.

⛔⛔ 165'S TABLE ORDER IS LOAD-BEARING AND IBM'S OWN TEST PROVES IT. `pack0` and `pack16` have
IDENTICAL rows (`VectorChainHelper.cpp:374`, `:383`) and the scan returns the FIRST match, so `pack16`
is unreachable: `dcc/test/SFP/merge_and_pack.mlir:232` writes a pack it NAMES `%pack16` and the
reference lowers it to `binary_operator pack0`. So the 34 rows are a `Vec` and not a map, and all 34
are pinned as one golden against that file's ordered `CHECK-SENT-IR`. ⭐ `scale` is why one table
serves four element widths, and a row holding `-1` cannot widen; the checksum filter applies only at
`scale == 1`; `repetition` is a member DEFAULT of eight that no row overrides (`:310`).

⭐ 164'S OUT-PARAMETER BECAME A TYPE, WHICH IS WHAT 165 AND 277 TAKE. `ValidPackIndices` can only be
minted by the four gates passing, so the checked list cannot be swapped for another on the way to the
scan, and `DT_CHECK_MSG((pack_op || shuffle_op))` is discharged by the `PackOrShuffle` witness.
⛔ `index > 2 * (int)indices.size()` is STRICTLY greater, so `index == 2 * size` is admitted; that is
reproduced and noted, and nothing downstream indexes with these.

⚠️ 160 HAS NO CALLER AT `a0d29abbed` — a grep of the authority tree finds the symbol once, at its own
definition; its neighbours decide the corelet split inline or through 161. Ported anyway, as 042 was.

⭐ AND ENTRIES 175-182 — `updateYieldArgs` in `vc_lowering_xrf.rs`; `opHasSideEffect` and
`mergeShallow` in `tf_cfgs_dataflow_conditional_tree.rs`; `CanonicalizeToggleDataflowPass::
runOnOperation` in `tf_canonicalize_toggle.rs`; and the Flattening tree's `clear`, `traverseRegion`,
`inRegionEmpty` and `cloneOpsForRegions` in `tf_flattening_local_regions.rs`. ⛔⛔ THE ISLAND GAINED A
WHOLE DIALECT HERE, AS THE BRIEF REQUIRES: `uniform.uniformize_regions` is the op this entire pass
family is about, and it was inexpressible. `islands/dataflow_ir/dialects/uniform.rs` declares its four
ops — `uniformize_regions` with a `LocalRegion` per arm (a unit list, a block argument and a body),
`yield`, `def_immutable_mapping` and `query_map` — as the eighth `dialects::Op` variant, wired through
`operands`/`results`/`regions`/`block_args` and the printer, and answered in 22 census arms across the
bridge. Every one of those arms is a `dyn_cast` in the reference that a `uniform.` op fails, and each
carries the reason it fails rather than a catch-all.

⛔⛔ 180 AND 182 ARE THE FLATTENING ITSELF, AND 182 IS WHERE THE BINDER MOVES. `traverseRegion` walks a
local region attributing every operation to every unit and stamping its parent's region index;
`cloneOpsForRegions` then rebuilds ONE equivalence class's body, dropping the nested
`uniform.uniformize_regions` and splicing in the operations of whichever of its regions belongs to that
class (`:235-238`). ⛔ THE REGION AN OPERATION CAME FROM HAS TO BE FOUND BY POINTER IDENTITY, NOT BY
`is_in_region_num`: `compute` stamps `false` — 0 — for EVERY region of a uniformized op (`:164`), so
the field cannot tell region 1 from region 0, and `:250-251` asks the parent which of its bodies holds
this operation instead. That is what lets `uniform.query_map(map:%285, key:%arg48)` come out reading the
NEW region's block argument (`flatten_local_region4.mlir:757` becomes `:355`), which is the whole point
of the pass. ⭐ AND `if (block.empty()) block.erase()` (`:269`) IS AN EMPTY `Vec` HERE, because the
island already records a blockless region as an empty `else_body` — the vendor's own flattened
`scf.if` at `:353-357` prints no `else`, so the case is live rather than theoretical.

⛔ 180 ENDS AT A `todo!` NAMING ENTRY 247, ON THE REFERENCE'S OWN `isa<>`. `traverseRegion` hands a
nested `uniform.uniformize_regions` to `compute` (`:137-138`), which is entry 247 at level 2 and
unported; the walk therefore refuses exactly the input the reference routes elsewhere, and every region
of fixtures 1-3 and fixture 4's outer region walk completely. Same shape as 178, whose one rewrite is
`DuplicateReusedToggle`'s pattern at level 6: the driver is ported, the `todo!` names the entry that
owes the rewrite, and it is gated on the real match condition so a program with no reused toggle is a
checked no-op.

⛔ 181 HAS NO CALLER ANYWHERE IN THE AUTHORITY TREE, AND THE REASON IS A DEFECT WORTH RECORDING. It is
declared (`:88`), defined (`:214`) and referenced nowhere else. The place that wants it is `:269` — is
this region empty — and it would have answered WRONGLY there, twice over: `is_in_region_num` is 0 for
every local region's children (`:164`) so a full region reads empty, and the predicate cannot see the
unit-class filter at `:233` that decides what actually gets cloned. Ported anyway, per this document's
rule that deciding a function is unnecessary is not the porter's judgement, with a test that pins the
wrong answer rather than a port that quietly corrects it.

⭐ THE ISLAND ALSO GAINED `Values::clone_without_regions` (MLIR's `Operation::cloneWithoutRegions`),
`scf::Op::If::results` and `affine::Op::If::dbg_name`. The result list is not cosmetic: 177 keeps the
DESTINATION's terminator when `dst->getNumResults() != 0` and the source's otherwise (`:420-428`), its
caller picks which of two candidates is the destination by the same question (`:237-247`), and
`areShallowlyMergeable` declines outright when both bind something (`:348`) — three decisions that a
census answering "none" for every `scf.if` would have made constant. And the `dbgName` on an
`affine.if` is there because `mergeShallow`'s candidates are `isa<affine::AffineIfOp, scf::IfOp>`
(`:34`), so an `affine.if` reaches `setDbgNameAttr` on the same path.

⛔ AND `UNITS.tsv`'s CALLEE COLUMN IS WRONG FOR FOUR OF THESE EIGHT. 175's is `-` because the extract
truncated the body one line early, at `setOperands`, dropping `return getXrfValue(yield_op, idx)` — so
the real edge is 175 → 090, and the tail is the half that makes the function a function rather than a
mutation. 178's is `e001_matchAndRewrite`, but `AffineYieldOpLowering` has nothing to do with this pass;
the pattern it adds is `e350_matchAndRewrite`. 177's names `e052_If`, which `mergeShallow` does not
call. 180's is `-` although the body calls `compute` at `:138`. The ports follow the authority, not the
column.

⚠️ AND ONE OF THIS BATCH'S OWN TESTS PASSED FOR THE WRONG REASON BEFORE IT PASSED FOR THE RIGHT ONE.
182's case-4 test built its `Values` counter at zero, so the clone minted `%4` while the fixture it was
cloning still read the ORIGINAL `%4` — printing `%4 = arith.subi %2, %4`, which looks exactly like a
clone correctly reading a value from outside the region. It was caught by reading the output rather than
the assertion, and the fixture now starts its counter past the program (`values_past(300)`), which is
also why the pinned text can be read against `flatten_local_region4.mlir:345-358` line for line.

⭐ AND ENTRIES 191-198 — `calculateFullShift` and `calculateDimWeights` in
`tf_mutable_start_addr_shifting.rs`; `ProgramUnitsReductionPass::matchUnits` in
`tf_program_units_reduction.rs`; `analyzeLoop` and `transformLoop` in
`tf_transform_loop_to_legalize_for_sentient_lowering.rs`; `TransformPagedMemViewPass::runOnOperation`
in `tf_transform_paged_mem_view.rs`; and `calculateIndicesRanges` and
`createConditionsForHyperRectSubscripts` in `tf_transform_paged_mem_view_impl.rs`.

⛔ 191'S TERNARY IS NOT A BOUNDS CHECK, AND ITS CONSTANT COLUMN IS NOT A LITERAL IN THE SUBSCRIPT.
`coeffs.size() != num_dims ? coeffs.back() : 0` (`:478`) reads like an index guard; the flattened row
is `num_dims + num_syms + num_locals + 1` wide, so it is longer than `num_dims` for every expression
that flattens at all and the `: 0` arm is dead. What the guard shields is FAILURE — a non-affine
subscript leaves `coeffs` EMPTY and `coeffs.back()` is then undefined behaviour. The island gained
MLIR's flattener for this (`AffineExpr::flatten` and `FlatAffineExpr` in `islands/dataflow_ir/ty.rs`,
`getFlattenedAffineExpr`), which names the constant column instead of counting the row's width: it is
the whole reason 191 cannot pattern-match an `Add` against a literal, because `(d0 + 5) floordiv 8`
has a 5 in it and a constant column of ZERO, and shifting 5 out of it would move the start address
eight times too far. The locals carry what they stand for, so `(x floordiv 2) + (x floordiv 2)` is ONE
column of coefficient 2 — a difference an enclosing `mod 2` can see.

⛔⛔ 195'S `ub_const.value()` IS SAFE ONLY BECAUSE 194 ORDERS ITS TESTS THE WAY IT DOES. The three
`getDefiningOp<arith::ConstantIndexOp>()`s at `:417-421` are read with no null check, and an `scf.for`
whose bound is an `arith.select` would fault there — it cannot arrive, because `analyzeLoop` tests
`!constant_bounds` at `:340-343` and the register file only at `:347-390`, so a non-constant `scf.for`
on a unit that owns one leaves as `KSplitParent` and never reaches `KUnroll`. ⭐⭐ THE VENDOR PROVES IT
ON AN INPUT WHERE BOTH TESTS MATCH: `dyn-loops-cond-bound.mlir` runs on `ptrow0` with an
`arith.select` bound whose induction variable is read by a store on `pt_lrfreg`, and the expectation
is an `scf.if`, not an unrolled loop. ⭐ `KSplitParent` is therefore `scf.for`-only —
`!constant_bounds && !affine_non_const_maps` is `!cb && cb` on the affine path — and 195's
`affine.for` arm for it is unreachable rather than merely unused.

⛔⛔ AND 198'S `num_dim_vars++` RUNS BEFORE THE `continue`, which is what makes the new subscripts map
renumber correctly for a dimension whose selected range is the whole range: the position is spent
whether or not a condition is emitted for it. `getConstantBound(LB/UB, dim)` there is a bound on the
loop ITERATOR, not on a view axis — `page_sel_constraints` is the page set with its dimensions
replaced by the symbol-form subscripts map's results, one symbol per iterator — which is why
`arg1 * 3` confined to `[0, 1]` prints `cmpi eq %arg1, 0` in
`paged_mem_view_loads.mlir:45-56`, and why the reference's two-field skip test is a single `==` on two
values of the same kind here. ⭐ 197 REUSES ENTRY 142 rather than re-walking the nest, and 142 is
STRICTER than the reference's own walk — it requires the value to BE the induction variable, so a
carried `iter_arg` answers "not an IV" — and it appends, because the time dimensions follow at
entry 131.

⭐ AND ENTRIES 151-158 — the AgenToSentient helpers for the composite regions, the extract-scalar
pattern and the SAMV: `checkCompositeRegion`, `checkStoreOpFromExtractPattern`,
`isLoadAndExtractScalarPattern`, `isReceiveAndExtractScalarPattern`, `updateSymbolicAccessDetails`,
`getStoreProducer`, `constructSetActiveMaskValueOp` and `createUniformizeRegionsOp`, all in
`agen_helper.rs`.

⛔⛔ 157'S `numvalidentry` IS TWO COUNTS IN ONE FIELD AND A FULL COUNT ENCODES AS ZERO. The inner
dimension's valid count is shifted by the bits the outer needs (or the reverse when the cross-slice
mask is the inner one), and both counts are rewritten to 0 when they equal their own width
(`Helper.cpp:2681`, `:2708`) — so the field never has to hold the width, and reading it as a plain
count would place the mask a whole dimension out. ⭐ THE VENDOR'S FIVE CASES ARE THE TEST:
`set_transfer_mask_state.mlir` gives `numvalidentry`/`sliceid_xsl`/`xslinner`/`wsllen` for three
generic maps and the two degenerate ones, and all five are asserted. The three-pattern
`slice_mask_map` enum is what types away `isUnmask`, `isFullMask`, `isGenericSAMV`, `getSliceIDXsl`
and the reference's four mask-attribute `DT_CHECK_MSG`s; the reference also computes `mask_wsl_elems`
and never reads it (`:2648`).

⛔ 155 NEEDS AN OPERATION MAPPING, NOT JUST A VALUE ONE. `ir_map.lookupOrDefault(ad.getOp())` answers
with the clone because `Operation::clone(IRMapping&)` records `map(this, newOp)` — so `OpMapping`
joins `ValueMapping` in `agen_helper.rs`, keyed by address as the C++ keys on `Operation*`. This is
also what promoted `AccessDetailsSymbolic` to carry its `AccessDetailsBase`: five of the six handles
entry 155 rewrites live on the base, and `mem_ref_` had no field at all.

⛔ 151 TAKES THE TERMINATOR'S OPERANDS BESIDE THE REGION. The island's `agen.yield` is a unit
variant, and 151's store arms ask what the region YIELDS (`hasOneUse` against the yield, at
`Helper.cpp:283-288`) — so `yielded: &[Val]` is passed alongside the body rather than reshaping the op
at twenty match sites. Its load arm's send_data check is unreachable in the reference itself and gets
no outcome, on the precedent entry 031's `ViewStart` set.

⛔ TWO ISLAND GAPS ARE RECORDED RATHER THAN STOOD IN FOR, both in 156: there is no
`vectorchain.coalesce`, so the coalesce-store arm and its *"must be preceded by
dataflow.receiveOp."* have nothing to match, and no `dataflow.create_multicast_group`, so two of the
four producer classes its `DT_CHECK` accepts cannot arrive. Each outcome exists and carries the
reference's message; neither is given a substitute op. 158'S `"active"` ATTRIBUTE IS NOT IR — it is
set, searched for backwards from the loop and removed by the last region, all within the pass, so it
is a field on the returned record instead of an attribute on the emitted op.

⭐ AND ENTRIES 206-213 — the chunking and shuffle info, the affine record's own initialization, the
container's two accessors, and the four `Helper.cpp` gatekeepers: `constructChunkAndShuffleInfo`,
`initialize`, `emplace_insert`, `getFirst` (all in `agen_access_details.rs`),
`checkBasicConditions`, `processInterleaveOp`, `gatherAffineLoadStoreDetails` and
`constructImmutableAddress` (all in `agen_helper.rs`).

⛔⛔ 206 WORKS ON A ROW-MAJOR **COPY** AND ITS TWO CURSORS START AT `-1`. A column-major layout is
reversed all but its trailing constant term, and the extents with it (`AccessDetails.cpp:111-123`),
while the members keep their own order. The `chunk_dim_idx`/`chunk_stride_dim_idx` pair is
`Option<usize>` here, and the legality test the reference indexes with one (`:191`) is skipped when
there is none — observably the same, because `chunk_stride` is 0 in exactly that case and its `&&`
fails. `dim == 0`'s multiplier is `INT32_MAX` verbatim (`:147-149`); a zero stride divides by zero
there, and with no ratio the extent has nothing to fit under, so it lands on the reference's own
*"Extent in load/store set is larger than from the layout"*.

⛔ 212 PUTS THE SAME START ADDRESS IN **TWO** CONTAINERS (`Helper.cpp:570-573`) — `mutable_addrs`,
which entry 357 rewrites, and a local copy entry 213 then reads as `updated_mem_view_start_addrs` —
and 213'S TWO ARMS DO NOT READ THE SAME WAY: the L3 arm reads the record, the other indexes
`updated_mem_view_start_addrs[i]` BY POSITION (`:1229`), which is in range only because of the size
test above it and means the same thing only because 212 filled both containers from one walk. 212's
coefficient rows are zero-filled as each iterator is first seen and written at column `i`, so a
column means "record `i`" even for an iterator the earlier records never mentioned.

⛔ ONE `todo!` IS ADDED, AND IT IS GATED ON THE ONE INPUT ENTRY 357 PROVABLY LEAVES ALONE: with no
access details, `generateAffineAddressManipulationStmts`' own `DT_CHECK` holds trivially, both of its
loops (`Helper.cpp:698`, `:771`) run zero times and it returns `success()`. Every other input reaches
statements this port does not have, and no stand-in op is substituted for them.

⭐ LEVEL 8, ENTRIES 374-381. Every one of the eight sits on an unported callee, so each is the
faithful control flow plus every reachable ported callee, stopping at a `todo!` that names the first
gap: 374 reaches entry 036's store search then `e360_constructSymbolicDetailsAndAddrs`; 375 goes
straight to 360; 376 lowers the four `arith` kinds it has and names `e339`/`e362`/`e363`; 377 finds
the send's producer then names `e304`; 378 runs entry 067 per PE/SFP unit then names `e227`; 379 runs
067 ONCE for the module (`:983`, OUTSIDE the walk — unlike 378's, which is inside it) then names
`e367`; 380 walks the sibling groups to the first candidate pair and names `e370`; 381 loops the root
children and names `parseConditional`. 374 and 375 return `!`: no outcome type exists before 360.

⛔ THE ISLAND GREW TWO OPS FOR 374/375. `agen.symbolic_vector_load` and `agen.symbolic_vector_store`
(`Agen.td:1119-1229`) had no island variant, which would have made both entries inputless — the
brief's rule is to add the op, not to declare the function unneeded. They carry their strides as
operands and the load carries a one-or-none multicast handle (`Agen.td:1163-1172`), and every total
match over `agen::Op` now answers for them: `is_data_transfer` says TRUE (they are two of the
nineteen `isa<>` classes), `AgenLoad::of` roots the load at its result (`Helper.cpp:1245-1247`),
`getVectorType` says None (no `SymbolicVector*Op` appears in `Utils.cpp` at all),
`constructChunkAndShuffleInfo` walks no users (its gate is the four non-symbolic loads, `:207-208`)
and `AccessDetailsAffine::initialize` refuses them (a symbolic subscript is a runtime value, which is
what `AccessDetailsSymbolic::initialize` at `:855-897` exists for). Entry 382's dispatch grew rows 11
and 12, and took the enclosing statement as a parameter so entry 036 can ask about the operation.

⛔ SEVEN OF THE REFERENCE'S TWELVE `agen` TRANSFER CLASSES HAVE NO ISLAND OP (the symbolic pair
gained one with entries 374/375, the interleave with entry 384). 210 is handed a `CheckedOp` (any DataflowIR op, or the already-lowered
`sentient.receive_and_store` whose only readable state is the mark) and 211 a `MemoryInterleave` (the
`granularity` attribute and the region's ops), on the precedent of `IndirectMemView` and
`UniformizeSource`; both matches are total and neither invents an op. 211'S IDENTITY TEST IS THE OP
**NAME** (`:361`, `:380-381`), not the attributes, so a region holding a `load_and_send` beside a
`receive_and_store` with matching burst and count is still refused — and its granularity default is
the maximum and is never checked, so `l3BurstSize` itself is legal and 0 is not.

⭐ LEVEL 10, ENTRY 384 — THE PASS ITSELF, and it is the first unit that puts the component gate in.
`AgenToSentient.cpp:174-176` returns for any unit outside `L0LU, L0SU, LXLU, LXSU, L3SU, L3LU`, and
`agen_agen_to_sentient::run_on_operation` is now what `program()` calls for the per-unit walk, so
entry 382's dispatch sees the `agen` ops of a transfer unit only. The gate is a PARAMETER of
`body`/`statement` rather than a `continue`, because the reference's `return` skips one pass of
seventy-six while this spine lowers every dialect of a unit in one walk; an `agen` access on a PE or
SFP unit now reaches a `todo!` naming `VectorChainToSentientPESFP`'s `VectorLoadOpLowering`, which
owns it. Measured against the golden corpus, no unit outside `l3lu`/`lxlu`/`lxsu` carries an `agen`
op, so the gate changes nothing that is emitted today.

⛔ FOUR OF ITS SIX STEPS SIT ON UNPORTED CALLEES, each a counted `todo!` over the input that would
reach it: the LDCVTI pre-pass (`:178-210`, `e318`, LXLU at SEN1P5 and above, anchored on a
`vectorchain.binary` whose operator is `mul` — NOT `vectorchain.multiply`, which is what the comment
at `:183`/`:185` says and the walk at `:192-198` contradicts), the interleave sweep (`:218-231`, `e271`), the
mask-state sweep (`:233-245`, `e218`) and the `set_send_dst` cleanup (`:247-248`, `e220`, whose own
`getArch() < RCUDD1A_ISA` early return is vacuous here — `IsaGen` has no lower generation). Only the
fusion (`e382`) lowers. The LDCVTI arm is inert in production today because the one caller of
`lower()` is `Dd2`.

⛔ THE ISLAND GREW TWO MORE OPS, for steps 4 and 5. `agen.composite_memory_interleave`
(`Agen.td:982-1039`) and `agen.set_transfer_mask_state` (`:1041-1116`) had no island variant, which
would have made both sweeps inputless. The interleave carries `granularity: Option<Elements>` (absent
is the hardware maximum, not zero) and a region of DataflowIR transfers — distinct from
`agen_helper::MemoryInterleave`, whose region is the LOWERED SentientIR the check reads. The mask
state carries `slices: Vec<SliceMask>` for `slice_mask_map`, so `(0)`, `(1)`, `(A)` and `(A|B)` are the
four productions the grammar has (`:1058-1072`) and a fifth is unspellable, with `num_slices` derived
as `slices.len()` rather than an attribute that could disagree; its `maskA`/`maskB` patterns are the
two optional `i32` arrays zipped into `MaskPattern`, and `MaskId` derives the letter rather than
storing it. Both `emit` arms reproduce the vendor's rendered text
(`comp_mem_interleave.mlir:309`, `set_transfer_mask_state.mlir:24`). Declaring them forced an answer
out of ten total matches: `is_data_transfer` (interleave TRUE, mask state FALSE — the fourteenth
class, outside the nineteen), `agen_op_kind`/`AgenLoad::of`/`VectorLoadOp::of`/`VectorStoreOp::of`
None, `is_memory_op` false, `TpmvManager::run` unsupported, `vector_type_of` None (the chain names the
two accesses only), `constructChunkAndShuffleInfo` no user walk, and
`AccessDetailsAffine::initialize` `UnsupportedOperation`. Entry 382's dispatch consumes both without
lowering: they are not among the twelve classes `fuseLoadOrStoreChainOps`'s candidate walk looks for
(`AgenToSentient.cpp:33-37`), and `e271`'s `todo!` fires before anything an interleave's region holds
could be dropped.


⭐ LEVEL 5, ENTRIES 222-229 — THE L3 SYNC LOWERING, THE `scf.for` PATTERN, THE SYMBOL QUERY CHAIN AND
THE VECTORCHAIN CLOSE-OUT. Eight units in five homes, and three of them needed the islands to grow.

⛔ 223 AND 224 READ AN ATTRIBUTE THE ISLAND WAS INVENTING. Both derive the emitted `sentient.sync`'s
`soft` from `wait_immediately_for_async_transfers` on the `dataflow.sync_send`
(`DataflowToSentient.cpp:385-388`, `:677-680`), and `dataflow::Op::SyncSend` had no such field — the
printer hard-coded `= true`. It is now a mandatory `wait: AsyncTransferWait`, which is what makes
the reference's `DT_CHECK` on the attribute's presence unreachable rather than a runtime refusal;
all 292 `dataflow.sync_send` ops under `dcc/test` carry it and none omits it. The SOURCE-component
guard (`:377-380`, `:669-672`) is the receiver type `L3Half`, so a caller cannot reach either
lowering from a non-L3 unit, and `L3SyncDst`'s three cases cover all eight accepted destination
components — which leaves both `emitError("Unknown lowering of the L3 sync ...")` arms and 224's
`break` with no input. 224 inlines 221's dedup rule (`:143-150`) without claiming that entry's
anchor: its peer list keeps first-occurrence order, matched against the six-unit soft group at
`dcc/test/L3SU/sync-op-l3su.mlir:14` and the single-peer hard sync at `:24`.
`sen::SyncHalf::of_lowered_destination` is the constructor those two need: `rendezvous` mints from
`SyncPeer`, whose four kinds cannot spell `lxlu0`/`l3su`, and "both sides or neither" is not a
property of a LOWERED sync's peer list.

⭐ 222 RETURNS THE OP THAT STANDS IN FOR TWO BUILDERS. `uniform.uniformize_regions` with two regions,
one block argument each, `mlir::TypeRange()` results — and the two unit slices instead of a flat
`units` plus `list_sizes` is what makes the op's three prefix-sum asserts unwritable. The sentient
union gained a `Uniform` variant so a lowered program can hold one at all, which forced `raised()`
(total, `dataflow_ir::dialects::Op` in) and turned `replace_all_uses_with`'s shared-dialect arm from
a no-op into a real rewrite.

⛔ 225'S INDUCTION VARIABLE IS INVERTED AND THE EXCEPTION LIST IS THE SUBTLETY.
`SCFToSentient.cpp:130-139` builds `i' = nIterations - i` as the loop's FIRST op and then replaces
every OTHER use of `i` with `i'`; skipping body index 0 IS that `exception_list`, and a port that
rewrote the whole body would rewrite the definition into itself. `hasSingleElement` holds because
one `Vec` is one block, so `emitError("expected scf.forop to have one block")` has no input; the
LCCR `push_back` stays commented out (`:101-103`) and each result takes one
`SentientRegTypeAttr(unknown)`. ⛔ AND 047 STILL CANNOT LOWER A MODULE: `match_and_rewrite` returns
ops of the SENTIENT union, so an earlier note promising 047's parameter would become `&mut` when 225
landed was wrong and is corrected in place.

⭐ 226 IS A RIGHT-NESTED `sentient.if` CHAIN, and `QueryMapping`'s `{first, middle, last_value}`
shape is what discharges `DT_CHECK_MSG(key_list.size() >= 2)` (`SymbolToSentient.cpp:134`) and
encodes `if (rhs_val == last_map_key) break;` (`:153`) — the last pair contributes its VALUE and
never a comparison. Results are minted before the recursion so the outermost `if` binds the lowest
values, as the reference's builder order does; the intermediate `else` yields the nested `if`'s
results (`:167-168`) and the innermost yields `val_list.back()` `num_results` times.

⛔ 227 IS THE DESTRUCTOR, NOT THE CLASS DECLARATION — `data_origins_.clear()` at
`OperandReuse.hpp:30`, written as a real `Drop` rather than left to the field's drop glue, because
one `OperandReuse` is constructed per `dataflow.program_unit` lowering
(`VectorChainToSentientPT.cpp:1006`) and a table that outlived it would carry the previous unit's
origins into the next unit's numbering. A `static` table would still compile against every method on
it and silently fail exactly this. The doc note that called 227 the class declaration is corrected.
`total_data_origins_count` (`:33`) came with it: NOT a scheduled unit, and 228 starts its fresh ids
at that count.

⛔ 228'S COUNTER IS SHARED ACROSS THE WHOLE UNIT, which is why the walk order is the port. A, B then
C of one `sentient.vector_mac`, then the next compute's — and an operand that already carries an id
is left alone (`== -1` is `None` here, through `DataId`, so the sentinel is not restated). The nine
op kinds the reference refuses with `emitError` + `WalkResult::interrupt()`
(`VectorChainHelper.cpp:483-533`) are four `todo!` arms naming them: a build failure, not a runtime
refusal, and none stood in for.

⭐ 229 HANDS BACK THE OP AND THE VALUE, because `builder.setInsertionPoint(op)` is the dropped
mechanism and the caller now owes the placement. `sentient.scalar_constant {value = 0 : si64} :
index` for entry 163's all-lanes-live set is the vendor's own expectation
(`dcc/test/Conversion/VectorChainToSentientPESFP/fnms_with_cast.mlir:11`); the shape mirrors
`vc_helper::MaskValue::Constant`, the PT side of the same question.

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
- [x] **PORT 041/384** `ExtendUnitNameToCorelet` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:104`, 11 lines
- [x] **AUDIT 041/384** `ExtendUnitNameToCorelet` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:104`, line by line against the C++
- [x] **PORT 042/384** `isSameListOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:175`, 10 lines
- [x] **AUDIT 042/384** `isSameListOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:175`, line by line against the C++
- [x] **PORT 043/384** `isTargetL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1720`, 6 lines
- [x] **AUDIT 043/384** `isTargetL3` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1720`, line by line against the C++
- [x] **PORT 044/384** `lowerOpaqueOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1984`, 27 lines
- [x] **AUDIT 044/384** `lowerOpaqueOperation` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1984`, line by line against the C++
- [x] **PORT 045/384** `getSentientCmpIPredicate` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:33`, 16 lines
- [x] **AUDIT 045/384** `getSentientCmpIPredicate` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:33`, line by line against the C++
- [x] **PORT 046/384** `ConversionPattern` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:70`, 0 lines
- [x] **AUDIT 046/384** `ConversionPattern` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:70`, line by line against the C++
- [x] **PORT 047/384** `runOnOperation` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:251`, 30 lines
- [x] **AUDIT 047/384** `runOnOperation` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:251`, line by line against the C++
- [x] **PORT 048/384** `getSentientCmpIPredicate` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:36`, 18 lines
- [x] **AUDIT 048/384** `getSentientCmpIPredicate` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:36`, line by line against the C++
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
- [x] **PORT 057/384** `setReuseFlag` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:91`, 2 lines
- [x] **AUDIT 057/384** `setReuseFlag` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:91`, line by line against the C++
- [x] **PORT 058/384** `dominates` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:38`, 2 lines
- [x] **AUDIT 058/384** `dominates` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:38`, line by line against the C++
- [x] **PORT 059/384** `isSentientBinaryLogicalOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:30`, 4 lines
- [x] **AUDIT 059/384** `isSentientBinaryLogicalOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:30`, line by line against the C++
- [x] **PORT 060/384** `getInputPrecisionFromOperand` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:36`, 6 lines
- [x] **AUDIT 060/384** `getInputPrecisionFromOperand` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:36`, line by line against the C++
- [x] **PORT 061/384** `getResultPrecisionFromOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:52`, 10 lines
- [x] **AUDIT 061/384** `getResultPrecisionFromOperands` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:52`, line by line against the C++
- [x] **PORT 062/384** `getComputePrecisionOfOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:65`, 13 lines
- [x] **AUDIT 062/384** `getComputePrecisionOfOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:65`, line by line against the C++
- [x] **PORT 063/384** `hasConstantBounds` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:80`, 10 lines
- [x] **AUDIT 063/384** `hasConstantBounds` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:80`, line by line against the C++
- [x] **PORT 064/384** `size` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:319`, 6 lines
- [x] **AUDIT 064/384** `size` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:319`, line by line against the C++
- [x] **PORT 065/384** `fuseCompareAndSelectIntoMinOrMax` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:415`, 49 lines
- [x] **AUDIT 065/384** `fuseCompareAndSelectIntoMinOrMax` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:415`, line by line against the C++
- [x] **PORT 066/384** `resetSentientFMAsIfExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:468`, 13 lines
- [x] **AUDIT 066/384** `resetSentientFMAsIfExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:468`, line by line against the C++
- [x] **PORT 067/384** `redefineConstantVectors` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:571`, 37 lines
- [x] **AUDIT 067/384** `redefineConstantVectors` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:571`, line by line against the C++
- [x] **PORT 068/384** `getVectorBinaryToSentientBinary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:32`, 17 lines
- [x] **AUDIT 068/384** `getVectorBinaryToSentientBinary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:32`, line by line against the C++
- [x] **PORT 069/384** `getVectorElementWiseCompareOperatorToSentientBinaryOperator` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:55`, 9 lines
- [x] **AUDIT 069/384** `getVectorElementWiseCompareOperatorToSentientBinaryOperator` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:55`, line by line against the C++
- [x] **PORT 070/384** `getVectorTernaryToSentientTernary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:247`, 7 lines
- [x] **AUDIT 070/384** `getVectorTernaryToSentientTernary` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:247`, line by line against the C++
- [x] **PORT 071/384** `getOperandFromReceiveOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:34`, 54 lines
- [x] **AUDIT 071/384** `getOperandFromReceiveOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:34`, line by line against the C++
- [x] **PORT 072/384** `getOperandFromSendOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:95`, 50 lines
- [x] **AUDIT 072/384** `getOperandFromSendOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:95`, line by line against the C++
- [x] **PORT 073/384** `constValToField` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:250`, 12 lines
- [x] **AUDIT 073/384** `constValToField` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:250`, line by line against the C++
- [x] **PORT 074/384** `sameBlock` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:652`, 11 lines
- [x] **AUDIT 074/384** `sameBlock` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:652`, line by line against the C++
- [x] **PORT 075/384** `eraseOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:806`, 16 lines
- [x] **AUDIT 075/384** `eraseOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:806`, line by line against the C++
- [x] **PORT 076/384** `fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1243`, 26 lines
- [x] **AUDIT 076/384** `fuseComputeOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1243`, line by line against the C++
- [x] **PORT 077/384** `walk` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:132`, 2 lines
- [x] **AUDIT 077/384** `walk` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:132`, line by line against the C++
- [x] **PORT 078/384** `findNodeFromOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:167`, 4 lines
- [x] **AUDIT 078/384** `findNodeFromOp` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:167`, line by line against the C++
- [x] **PORT 079/384** `OperationNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:32`, 0 lines
- [x] **AUDIT 079/384** `OperationNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:32`, line by line against the C++
- [x] **PORT 080/384** `LoopMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:34`, 0 lines
- [x] **AUDIT 080/384** `LoopMaskNode` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:34`, line by line against the C++
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
- [x] **PORT 089/384** `getMaskValueForPT` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:17`, 196 lines
- [x] **AUDIT 089/384** `getMaskValueForPT` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:17`, line by line against the C++
- [x] **PORT 090/384** `getXrfValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:248`, 11 lines
- [x] **AUDIT 090/384** `getXrfValue` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:248`, line by line against the C++
- [x] **PORT 091/384** `getForOpBound` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:262`, 31 lines
- [x] **AUDIT 091/384** `getForOpBound` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:262`, line by line against the C++
- [x] **PORT 092/384** `setSentientMacXrfRegIncrAttr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:665`, 11 lines
- [x] **AUDIT 092/384** `setSentientMacXrfRegIncrAttr` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:665`, line by line against the C++
- [x] **PORT 093/384** `replaceAndEraseDummyMacOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:681`, 7 lines
- [x] **AUDIT 093/384** `replaceAndEraseDummyMacOps` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:681`, line by line against the C++
- [x] **PORT 094/384** `computeUnitPrecision` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:31`, 9 lines
- [x] **AUDIT 094/384** `computeUnitPrecision` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:31`, line by line against the C++
- [x] **PORT 095/384** `isOperationSelected` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:34`, 2 lines
- [x] **AUDIT 095/384** `isOperationSelected` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:34`, line by line against the C++
- [x] **PORT 096/384** `createDummyYieldInElseReg` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:383`, 13 lines
- [x] **AUDIT 096/384** `createDummyYieldInElseReg` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:383`, line by line against the C++
- [x] **PORT 097/384** `getNewDbgNameFromList` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:456`, 2 lines
- [x] **AUDIT 097/384** `getNewDbgNameFromList` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:456`, line by line against the C++
- [x] **PORT 098/384** `getLhsRhsOfEQPredicate` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:519`, 11 lines
- [x] **AUDIT 098/384** `getLhsRhsOfEQPredicate` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:519`, line by line against the C++
- [x] **PORT 099/384** `ConditionalSimplificationManager` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:78`, 2 lines
- [x] **AUDIT 099/384** `ConditionalSimplificationManager` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:78`, line by line against the C++
- [x] **PORT 100/384** `TransformationConditionalTree` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:111`, 5 lines
- [x] **AUDIT 100/384** `TransformationConditionalTree` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:111`, line by line against the C++
- [x] **PORT 101/384** `OperationNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:51`, 0 lines
- [x] **AUDIT 101/384** `OperationNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:51`, line by line against the C++
- [x] **PORT 102/384** `getParentNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:53`, 2 lines
- [x] **AUDIT 102/384** `getParentNode` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:53`, line by line against the C++
- [x] **PORT 103/384** `getFirstChild` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:56`, 2 lines
- [x] **AUDIT 103/384** `getFirstChild` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:56`, line by line against the C++
- [x] **PORT 104/384** `getNextSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:59`, 2 lines
- [x] **AUDIT 104/384** `getNextSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:59`, line by line against the C++
- [x] **PORT 105/384** `getPrevSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:62`, 2 lines
- [x] **AUDIT 105/384** `getPrevSibling` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:62`, line by line against the C++
- [x] **PORT 106/384** `OperationTreeBase` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:78`, 0 lines
- [x] **AUDIT 106/384** `OperationTreeBase` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:78`, line by line against the C++
- [x] **PORT 107/384** `getRoot` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:81`, 2 lines
- [x] **AUDIT 107/384** `getRoot` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:81`, line by line against the C++
- [x] **PORT 108/384** `partitionUnits` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:168`, 17 lines
- [x] **AUDIT 108/384** `partitionUnits` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:168`, line by line against the C++
- [x] **PORT 109/384** `performFullUnroll` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:141`, 7 lines
- [x] **AUDIT 109/384** `performFullUnroll` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:141`, line by line against the C++
- [x] **PORT 110/384** `getConstantTripCount` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:167`, 14 lines
- [x] **AUDIT 110/384** `getConstantTripCount` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:167`, line by line against the C++
- [x] **PORT 111/384** `getMaxMutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:673`, 8 lines
- [x] **AUDIT 111/384** `getMaxMutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:673`, line by line against the C++
- [x] **PORT 112/384** `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:683`, 8 lines
- [x] **AUDIT 112/384** `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:683`, line by line against the C++
- [x] **PORT 113/384** `isEligibleForSplitting` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:832`, 20 lines
- [x] **AUDIT 113/384** `isEligibleForSplitting` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:832`, line by line against the C++
- [x] **PORT 114/384** `sortDataBasedOnWeight` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:855`, 4 lines
- [x] **AUDIT 114/384** `sortDataBasedOnWeight` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:855`, line by line against the C++
- [x] **PORT 115/384** `createNewMemViewWithMod` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:966`, 12 lines
- [x] **AUDIT 115/384** `createNewMemViewWithMod` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:966`, line by line against the C++
- [x] **PORT 116/384** `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:355`, 8 lines
- [x] **AUDIT 116/384** `getMaxImmutableRange` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:355`, line by line against the C++
- [x] **PORT 117/384** `transformSCFToAffineLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:102`, 44 lines
- [x] **AUDIT 117/384** `transformSCFToAffineLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:102`, line by line against the C++
- [x] **PORT 118/384** `removeValuesFromIndices` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:36`, 7 lines
- [x] **AUDIT 118/384** `removeValuesFromIndices` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:36`, line by line against the C++
- [x] **PORT 119/384** `replaceDimsInMapWithSyms` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:47`, 7 lines
- [x] **AUDIT 119/384** `replaceDimsInMapWithSyms` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:47`, line by line against the C++
- [x] **PORT 120/384** `createEqualityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:253`, 7 lines
- [x] **AUDIT 120/384** `createEqualityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:253`, line by line against the C++
- [x] **PORT 121/384** `createInequalityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:263`, 14 lines
- [x] **AUDIT 121/384** `createInequalityCondition` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:263`, line by line against the C++
- [x] **PORT 122/384** `setBuilderToInsertRef` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:280`, 6 lines
- [x] **AUDIT 122/384** `setBuilderToInsertRef` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:280`, line by line against the C++
- [x] **PORT 123/384** `calculateStartElementsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:532`, 10 lines
- [x] **AUDIT 123/384** `calculateStartElementsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:532`, line by line against the C++
- [x] **PORT 124/384** `createNonPagedMemView` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:545`, 10 lines
- [x] **AUDIT 124/384** `createNonPagedMemView` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:545`, line by line against the C++
- [x] **PORT 125/384** `cloneMemViewIfNonPaged` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:633`, 9 lines
- [x] **AUDIT 125/384** `cloneMemViewIfNonPaged` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:633`, line by line against the C++
- [x] **PORT 126/384** `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:673`, 3 lines
- [x] **AUDIT 126/384** `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:673`, line by line against the C++
- [x] **PORT 127/384** `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:678`, 3 lines
- [x] **AUDIT 127/384** `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:678`, line by line against the C++
- [x] **PORT 128/384** `createNewMemOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:684`, 9 lines
- [x] **AUDIT 128/384** `createNewMemOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:684`, line by line against the C++
- [x] **PORT 129/384** `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:698`, 3 lines
- [x] **AUDIT 129/384** `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:698`, line by line against the C++
- [x] **PORT 130/384** `getStoreOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:843`, 6 lines
- [x] **AUDIT 130/384** `getStoreOp` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:843`, line by line against the C++
- [x] **PORT 131/384** `addTimeDimIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:876`, 5 lines
- [x] **AUDIT 131/384** `addTimeDimIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:876`, line by line against the C++
- [x] **PORT 132/384** `identifyTimeDimForExplicitLoops` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:963`, 10 lines
- [x] **AUDIT 132/384** `identifyTimeDimForExplicitLoops` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:963`, line by line against the C++
- [x] **PORT 133/384** `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:328`, 2 lines
- [x] **AUDIT 133/384** `getUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:328`, line by line against the C++
- [x] **PORT 134/384** `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:341`, 0 lines
- [x] **AUDIT 134/384** `cloneUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:341`, line by line against the C++
- [x] **PORT 135/384** `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:364`, 0 lines
- [x] **AUDIT 135/384** `eraseMemOpAndUseChain` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:364`, line by line against the C++
- [x] **PORT 136/384** `TPMVBase` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:389`, 0 lines
- [x] **AUDIT 136/384** `TPMVBase` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:389`, line by line against the C++
- [x] **PORT 137/384** `TPMVVector` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:397`, 0 lines
- [x] **AUDIT 137/384** `TPMVVector` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:397`, line by line against the C++
- [x] **PORT 138/384** `TPMVComposite` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:519`, 0 lines
- [x] **AUDIT 138/384** `TPMVComposite` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:519`, line by line against the C++
- [x] **PORT 139/384** `run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.cpp:21`, 48 lines
- [x] **AUDIT 139/384** `run` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.cpp:21`, line by line against the C++
- [x] **PORT 140/384** `removeCoresCoreletsFoldsFromProgramUnit` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:240`, 20 lines
- [x] **AUDIT 140/384** `removeCoresCoreletsFoldsFromProgramUnit` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:240`, line by line against the C++
- [x] **PORT 141/384** `isDataTransfer` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:314`, 11 lines
- [x] **AUDIT 141/384** `isDataTransfer` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:314`, line by line against the C++
- [x] **PORT 142/384** `getDataflowForLoopInfoIfIV` — `dcc/src/Transform/Dataflow/Utils.cpp:99`, 31 lines
- [x] **AUDIT 142/384** `getDataflowForLoopInfoIfIV` — `dcc/src/Transform/Dataflow/Utils.cpp:99`, line by line against the C++

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
- [x] **PORT 151/384** `checkCompositeRegion` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:206`, 89 lines
- [x] **AUDIT 151/384** `checkCompositeRegion` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:206`, line by line against the C++
- [x] **PORT 152/384** `checkStoreOpFromExtractPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:433`, 27 lines
- [x] **AUDIT 152/384** `checkStoreOpFromExtractPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:433`, line by line against the C++
- [x] **PORT 153/384** `isLoadAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:463`, 27 lines
- [x] **AUDIT 153/384** `isLoadAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:463`, line by line against the C++
- [x] **PORT 154/384** `isReceiveAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:493`, 17 lines
- [x] **AUDIT 154/384** `isReceiveAndExtractScalarPattern` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:493`, line by line against the C++
- [x] **PORT 155/384** `updateSymbolicAccessDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1013`, 35 lines
- [x] **AUDIT 155/384** `updateSymbolicAccessDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1013`, line by line against the C++
- [x] **PORT 156/384** `getStoreProducer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1285`, 159 lines
- [x] **AUDIT 156/384** `getStoreProducer` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1285`, line by line against the C++
- [x] **PORT 157/384** `constructSetActiveMaskValueOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2567`, 161 lines
- [x] **AUDIT 157/384** `constructSetActiveMaskValueOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:2567`, line by line against the C++
- [x] **PORT 158/384** `createUniformizeRegionsOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3880`, 57 lines
- [x] **AUDIT 158/384** `createUniformizeRegionsOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3880`, line by line against the C++
- [x] **PORT 159/384** `getUnitNameFromAListOfGetUnitOp` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:119`, 10 lines
- [x] **AUDIT 159/384** `getUnitNameFromAListOfGetUnitOp` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:119`, line by line against the C++
- [x] **PORT 160/384** `areCoreletsDifferent` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:132`, 6 lines
- [x] **AUDIT 160/384** `areCoreletsDifferent` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:132`, line by line against the C++
- [x] **PORT 161/384** `separateBasedOnDestinationUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:761`, 18 lines
- [x] **AUDIT 161/384** `separateBasedOnDestinationUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:761`, line by line against the C++
- [x] **PORT 162/384** `insertIfNotExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:81`, 8 lines
- [x] **AUDIT 162/384** `insertIfNotExists` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:81`, line by line against the C++
- [x] **PORT 163/384** `getMaskValueConstantForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:93`, 40 lines
- [x] **AUDIT 163/384** `getMaskValueConstantForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:93`, line by line against the C++
- [x] **PORT 164/384** `checkValidityOfPackAndShuffleLowering` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:140`, 45 lines
- [x] **AUDIT 164/384** `checkValidityOfPackAndShuffleLowering` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:140`, line by line against the C++
- [x] **PORT 165/384** `getMergeTypeFromIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:301`, 109 lines
- [x] **AUDIT 165/384** `getMergeTypeFromIndices` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:301`, line by line against the C++
- [x] **PORT 166/384** `getOperandFromConstantOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:267`, 26 lines
- [x] **AUDIT 166/384** `getOperandFromConstantOp` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:267`, line by line against the C++
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
- [x] **PORT 175/384** `updateYieldArgs` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:690`, 8 lines
- [x] **AUDIT 175/384** `updateYieldArgs` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:690`, line by line against the C++
- [x] **PORT 176/384** `opHasSideEffect` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:38`, 13 lines
- [x] **AUDIT 176/384** `opHasSideEffect` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:38`, line by line against the C++
- [x] **PORT 177/384** `mergeShallow` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:398`, 60 lines
- [x] **AUDIT 177/384** `mergeShallow` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:398`, line by line against the C++
- [x] **PORT 178/384** `runOnOperation` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:49`, 42 lines
- [x] **AUDIT 178/384** `runOnOperation` — `dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:49`, line by line against the C++
- [x] **PORT 179/384** `clear` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:112`, 15 lines
- [x] **AUDIT 179/384** `clear` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:112`, line by line against the C++
- [x] **PORT 180/384** `traverseRegion` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:129`, 17 lines
- [x] **AUDIT 180/384** `traverseRegion` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:129`, line by line against the C++
- [x] **PORT 181/384** `inRegionEmpty` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:214`, 6 lines
- [x] **AUDIT 181/384** `inRegionEmpty` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:214`, line by line against the C++
- [x] **PORT 182/384** `cloneOpsForRegions` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:223`, 60 lines
- [x] **AUDIT 182/384** `cloneOpsForRegions` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:223`, line by line against the C++
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
- [x] **PORT 191/384** `calculateFullShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:462`, 25 lines
- [x] **AUDIT 191/384** `calculateFullShift` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:462`, line by line against the C++
- [x] **PORT 192/384** `calculateDimWeights` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:560`, 26 lines
- [x] **AUDIT 192/384** `calculateDimWeights` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:560`, line by line against the C++
- [x] **PORT 193/384** `matchUnits` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:69`, 77 lines
- [x] **AUDIT 193/384** `matchUnits` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:69`, line by line against the C++
- [x] **PORT 194/384** `analyzeLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:268`, 127 lines
- [x] **AUDIT 194/384** `analyzeLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:268`, line by line against the C++
- [x] **PORT 195/384** `transformLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:399`, 38 lines
- [x] **AUDIT 195/384** `transformLoop` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:399`, line by line against the C++
- [x] **PORT 196/384** `runOnOperation` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:41`, 23 lines
- [x] **AUDIT 196/384** `runOnOperation` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:41`, line by line against the C++
- [x] **PORT 197/384** `calculateIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:56`, 25 lines
- [x] **AUDIT 197/384** `calculateIndicesRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:56`, line by line against the C++
- [x] **PORT 198/384** `createConditionsForHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:289`, 48 lines
- [x] **AUDIT 198/384** `createConditionsForHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:289`, line by line against the C++
- [x] **PORT 199/384** `createConditionsForNonHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:342`, 35 lines
- [x] **AUDIT 199/384** `createConditionsForNonHyperRectSubscripts` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:342`, line by line against the C++
- [x] **PORT 200/384** `updateTPMVInfo` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:382`, 17 lines
- [x] **AUDIT 200/384** `updateTPMVInfo` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:382`, line by line against the C++
- [x] **PORT 201/384** `setLoopIteratorOrder` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:575`, 13 lines
- [x] **AUDIT 201/384** `setLoopIteratorOrder` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:575`, line by line against the C++
- [x] **PORT 202/384** `initialize` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:658`, 13 lines
- [x] **AUDIT 202/384** `initialize` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:658`, line by line against the C++
- [x] **PORT 203/384** `removeCoresCoreletsFoldsFromDefImmutMap` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:99`, 37 lines
- [x] **AUDIT 203/384** `removeCoresCoreletsFoldsFromDefImmutMap` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:99`, line by line against the C++
- [x] **PORT 204/384** `cleanup` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:263`, 49 lines
- [x] **AUDIT 204/384** `cleanup` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:263`, line by line against the C++
- [x] **PORT 205/384** `isDataTransferToKeep` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:327`, 10 lines
- [x] **AUDIT 205/384** `isDataTransferToKeep` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:327`, line by line against the C++

## Level 2

- [x] **PORT 206/384** `constructChunkAndShuffleInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:98`, 167 lines
- [x] **AUDIT 206/384** `constructChunkAndShuffleInfo` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:98`, line by line against the C++
- [x] **PORT 207/384** `initialize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:295`, 57 lines
- [x] **AUDIT 207/384** `initialize` — `dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:295`, line by line against the C++
- [x] **PORT 208/384** `emplace_insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:378`, 8 lines
- [x] **AUDIT 208/384** `emplace_insert` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:378`, line by line against the C++
- [x] **PORT 209/384** `getFirst` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:411`, 7 lines
- [x] **AUDIT 209/384** `getFirst` — `dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:411`, line by line against the C++
- [x] **PORT 210/384** `checkBasicConditions` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:58`, 133 lines
- [x] **AUDIT 210/384** `checkBasicConditions` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:58`, line by line against the C++
- [x] **PORT 211/384** `processInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:305`, 80 lines
- [x] **AUDIT 211/384** `processInterleaveOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:305`, line by line against the C++
- [x] **PORT 212/384** `gatherAffineLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:538`, 74 lines
- [x] **AUDIT 212/384** `gatherAffineLoadStoreDetails` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:538`, line by line against the C++
- [x] **PORT 213/384** `constructImmutableAddress` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1217`, 16 lines
- [x] **AUDIT 213/384** `constructImmutableAddress` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1217`, line by line against the C++
- [ ] **PORT 214/384** `setImmutableAddrAndIncrements` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1581`, 43 lines
- [ ] **AUDIT 214/384** `setImmutableAddrAndIncrements` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:1581`, line by line against the C++
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
- [x] **PORT 222/384** `createUniformRegionsWithTwoRegionsNoResult` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:153`, 18 lines
- [x] **AUDIT 222/384** `createUniformRegionsWithTwoRegionsNoResult` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:153`, line by line against the C++
- [x] **PORT 223/384** `lowerL3SyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:375`, 57 lines
- [x] **AUDIT 223/384** `lowerL3SyncOperationForAUnit` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:375`, line by line against the C++
- [x] **PORT 224/384** `lowerL3SyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:667`, 61 lines
- [x] **AUDIT 224/384** `lowerL3SyncOperationForAGroupOfUnits` — `dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:667`, line by line against the C++
- [x] **PORT 225/384** `matchAndRewrite` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:72`, 69 lines
- [x] **AUDIT 225/384** `matchAndRewrite` — `dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:72`, line by line against the C++
- [x] **PORT 226/384** `createIfOpFromMapping` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:123`, 61 lines
- [x] **AUDIT 226/384** `createIfOpFromMapping` — `dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:123`, line by line against the C++
- [x] **PORT 227/384** `OperandReuse` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:30`, 0 lines
- [x] **AUDIT 227/384** `OperandReuse` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:30`, line by line against the C++
- [x] **PORT 228/384** `validateLoweringAndSetMissingParameters` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:483`, 49 lines
- [x] **AUDIT 228/384** `validateLoweringAndSetMissingParameters` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:483`, line by line against the C++
- [x] **PORT 229/384** `getMaskValueForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:114`, 9 lines
- [x] **AUDIT 229/384** `getMaskValueForNonPT` — `dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:114`, line by line against the C++
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
- [x] **PORT 246/384** `FlatteningLocalRegionsTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:79`, 0 lines
- [x] **AUDIT 246/384** `FlatteningLocalRegionsTree` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:79`, line by line against the C++
- [x] **PORT 247/384** `compute` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:151`, 15 lines
- [x] **AUDIT 247/384** `compute` — `dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:151`, line by line against the C++
- [x] **PORT 248/384** `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:70`, 68 lines
- [x] **AUDIT 248/384** `runOnOperation` — `dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:70`, line by line against the C++
- [x] **PORT 249/384** `processComputeUnit` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:37`, 91 lines
- [x] **AUDIT 249/384** `processComputeUnit` — `dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:37`, line by line against the C++
- [x] **PORT 250/384** `initMASData` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:707`, 31 lines
- [x] **AUDIT 250/384** `initMASData` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:707`, line by line against the C++
- [x] **PORT 251/384** `setupForPartitioning` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:819`, 6 lines
- [x] **AUDIT 251/384** `setupForPartitioning` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:819`, line by line against the C++
- [x] **PORT 252/384** `fillPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1063`, 117 lines
- [x] **AUDIT 252/384** `fillPartitions` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1063`, line by line against the C++
- [x] **PORT 253/384** `adjustForEvenImmutableAddr` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1204`, 47 lines
- [x] **AUDIT 253/384** `adjustForEvenImmutableAddr` — `dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1204`, line by line against the C++
- [x] **PORT 254/384** `offsetShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:590`, 22 lines
- [x] **AUDIT 254/384** `offsetShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:590`, line by line against the C++
- [x] **PORT 255/384** `applyShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:616`, 27 lines
- [x] **AUDIT 255/384** `applyShifts` — `dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:616`, line by line against the C++
- [x] **PORT 256/384** `runOnOperation` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:152`, 76 lines
- [x] **AUDIT 256/384** `runOnOperation` — `dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:152`, line by line against the C++
- [x] **PORT 257/384** `analyzeAndTransform` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:441`, 3 lines
- [x] **AUDIT 257/384** `analyzeAndTransform` — `dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:441`, line by line against the C++
- [x] **PORT 258/384** `addConstraintsForIVRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:86`, 22 lines
- [x] **AUDIT 258/384** `addConstraintsForIVRanges` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:86`, line by line against the C++
- [x] **PORT 259/384** `createNewSubscriptsFromStartElements` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:562`, 10 lines
- [x] **AUDIT 259/384** `createNewSubscriptsFromStartElements` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:562`, line by line against the C++
- [x] **PORT 260/384** `gatherPageDependentDimsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:927`, 32 lines
- [x] **AUDIT 260/384** `gatherPageDependentDimsForPage` — `dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:927`, line by line against the C++
- [x] **PORT 261/384** `runOnOperation` — `dcc/src/Transform/Dataflow/UniformQueryMapsCanonicalization.cpp:55`, 41 lines
- [x] **AUDIT 261/384** `runOnOperation` — `dcc/src/Transform/Dataflow/UniformQueryMapsCanonicalization.cpp:55`, line by line against the C++
- [x] **PORT 262/384** `removeCoresCoreletsFoldsFromUniformizeRegion` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:139`, 98 lines
- [x] **AUDIT 262/384** `removeCoresCoreletsFoldsFromUniformizeRegion` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:139`, line by line against the C++
- [x] **PORT 263/384** `removeAncestors` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:339`, 18 lines
- [x] **AUDIT 263/384** `removeAncestors` — `dcc/src/Transform/Dataflow/UnitFiltering.cpp:339`, line by line against the C++
- [x] **PORT 264/384** `createForOpWithAdditionalReturnValue` — `dcc/src/Transform/Dataflow/Utils.cpp:28`, 66 lines
- [x] **AUDIT 264/384** `createForOpWithAdditionalReturnValue` — `dcc/src/Transform/Dataflow/Utils.cpp:28`, line by line against the C++

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

- [x] **PORT 374/384** `lowerSymbolicVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3380`, 26 lines
- [x] **AUDIT 374/384** `lowerSymbolicVectorLoadOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3380`, line by line against the C++
- [x] **PORT 375/384** `lowerSymbolicVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3410`, 30 lines
- [x] **AUDIT 375/384** `lowerSymbolicVectorStoreOp` — `dcc/src/Conversion/AgenToSentient/Helper.cpp:3410`, line by line against the C++
- [x] **PORT 376/384** `runOnOperation` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:439`, 34 lines
- [x] **AUDIT 376/384** `runOnOperation` — `dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:439`, line by line against the C++
- [x] **PORT 377/384** `matchAndRewrite` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:44`, 12 lines
- [x] **AUDIT 377/384** `matchAndRewrite` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:44`, line by line against the C++
- [x] **PORT 378/384** `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1374`, 30 lines
- [x] **AUDIT 378/384** `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1374`, line by line against the C++
- [x] **PORT 379/384** `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:975`, 54 lines
- [x] **AUDIT 379/384** `runOnOperation` — `dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:975`, line by line against the C++
- [x] **PORT 380/384** `shallowlyMergeConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:198`, 76 lines
- [x] **AUDIT 380/384** `shallowlyMergeConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:198`, line by line against the C++
- [x] **PORT 381/384** `simplifyValueBasedConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:466`, 30 lines
- [x] **AUDIT 381/384** `simplifyValueBasedConditionals` — `dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:466`, line by line against the C++

## Level 9

- [ ] **PORT 382/384** `fuseLoadOrStoreChainOps` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:22`, 144 lines
- [ ] **AUDIT 382/384** `fuseLoadOrStoreChainOps` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:22`, line by line against the C++
- [ ] **PORT 383/384** `runOnOperation` — `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:77`, 80 lines
- [ ] **AUDIT 383/384** `runOnOperation` — `dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:77`, line by line against the C++

## Level 10

- [x] **PORT 384/384** `runOnOperation` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:169`, 81 lines
- [x] **AUDIT 384/384** `runOnOperation` — `dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:169`, line by line against the C++

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

