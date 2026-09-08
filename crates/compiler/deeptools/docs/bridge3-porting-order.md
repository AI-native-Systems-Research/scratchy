# Bridge 3 porting order — `SentientIR -> ProgIR`, dcc pass D76

130 units from `dcc/src/Conversion/SentientToProgIR/`, 6,879 declaration lines, in dependency
order. Authority: `/Users/nickm/git/deeptools-src` at revision `a0d29abbed`.

## ⛔⛔ The methodology

**PORT ONE fn → AUDIT it line by line against the citation → tick BOTH.** Never port new work
with an audit outstanding. A predicate is not a port: the op a function emits IS the function.
If the target island cannot express something, ADD IT TO THE ISLAND — deciding a function is
unnecessary is not the porter's call.

**PORT** = the whole function including its emission, anchored `/// Replaces: eNNN_name`.
**AUDIT** = a different agent instance read the port beside the authority at the cited line and
confirmed the branch order, the early returns, and every attribute name and value.

## ⛔ Not in this span

`dcc/src/Transform/Sentient/` — 32,766 lines of D29-D75 passes that rewrite SentientIR *in
place* — is a separate, larger body of work. A ported function that depends on state those
passes establish (register assignments, pinned addresses, rerolled loops) is ported **as the
reference writes it**, with the dependency recorded in the commit message. Do not port the pass,
and do not invent the state.

## ⛔ One mutually recursive component

`LowerUniformOperations` and `GenerateProgIR` call each other, so they share level 5 and cannot
be split by a wave barrier. Neither may be ported as if the other were a leaf.

## Level 0 — 41 units, 833 declaration lines

| | PORT | AUDIT | unit | LoC | authority | Rust home |
|---|---|---|---|---|---|---|
| 001/130 | [ ] | [ ] | `e001_ConstructNOPInstr` | 10 | `ConstructProgIRHelper.cpp:329` | `construct/scalar.rs` |
| 002/130 | [ ] | [ ] | `e002_ConstructReturnInstr` | 7 | `ConstructProgIRHelper.cpp:340` | `construct/scalar.rs` |
| 003/130 | [ ] | [ ] | `e003_normalizeBurstSize` | 13 | `ConstructProgIRHelper.cpp:2562` | `construct/transfer.rs` |
| 004/130 | [ ] | [ ] | `e004_rtrim` | 6 | `ConstructProgIRHelper.cpp:3859` | `construct/opaque.rs` |
| 005/130 | [ ] | [ ] | `e005_ConstructIncrMaskInstr` | 11 | `ConstructProgIRHelper.cpp:4118` | `construct/mask_and_splat.rs` |
| 006/130 | [ ] | [ ] | `e006_addToRegsToInit` | 61 | `ConstructProgIRHelper.cpp:4157` | `construct/reg_init.rs` |
| 007/130 | [ ] | [ ] | `e007_addPESFPLRFImmcopyToRegInit` | 228 | `ConstructProgIRHelper.cpp:4226` | `construct/reg_init.rs` |
| 008/130 | [ ] | [ ] | `e008_addToRegInit` | 41 | `LowerSentientHelper.cpp:96` | `lower/labels_and_regs.rs` |
| 009/130 | [ ] | [ ] | `e009_GetAddressScale` | 16 | `LowerSentientHelper.cpp:502` | `lower/labels_and_regs.rs` |
| 010/130 | [ ] | [ ] | `e010_GetOpCodePrefix` | 18 | `LowerSentientHelper.cpp:519` | `lower/labels_and_regs.rs` |
| 011/130 | [ ] | [ ] | `e011_fillUnitToIdMap` | 17 | `LowerSentientHelper.cpp:967` | `lower/control.rs` |
| 012/130 | [ ] | [ ] | `e012_setRegDef` | 7 | `RegDefTracker.cpp:53` | `reg_def_tracker.rs` |
| 013/130 | [ ] | [ ] | `e013_addRegDefsForUnit` | 15 | `RegDefTracker.cpp:61` | `reg_def_tracker.rs` |
| 014/130 | [ ] | [ ] | `e014_checkRegDefs` | 63 | `RegDefTracker.cpp:106` | `reg_def_tracker.rs` |
| 015/130 | [ ] | [ ] | `e015_initializeUtilizedRegisters` | 39 | `SentientToProgIR.cpp:429` | `driver.rs` |
| 016/130 | [ ] | [ ] | `e016_replaceProgramBodyWithSmcOp` | 49 | `SentientToProgIR.cpp:560` | `driver.rs` |
| 017/130 | [ ] | [ ] | `e017_createNOPInstr` | 8 | `UniformInstrAndBlock.cpp:43` | `uniform/instr.rs` |
| 018/130 | [ ] | [ ] | `e018_getUniformizedInstr` | 21 | `UniformInstrAndBlock.cpp:218` | `uniform/instr.rs` |
| 019/130 | [ ] | [ ] | `e019_getRegularInstr` | 11 | `UniformInstrAndBlock.cpp:240` | `uniform/instr.rs` |
| 020/130 | [ ] | [ ] | `e020_getCommonField` | 5 | `UniformInstrAndBlock.cpp:252` | `uniform/instr.rs` |
| 021/130 | [ ] | [ ] | `e021_setCommonField` | 5 | `UniformInstrAndBlock.cpp:257` | `uniform/instr.rs` |
| 022/130 | [ ] | [ ] | `e022_hasCommonField` | 3 | `UniformInstrAndBlock.cpp:262` | `uniform/instr.rs` |
| 023/130 | [ ] | [ ] | `e023_getUnitInstrList` | 12 | `UniformInstrAndBlock.cpp:266` | `uniform/block.rs` |
| 024/130 | [ ] | [ ] | `e024_insertInstruction` | 5 | `UniformInstrAndBlock.cpp:322` | `uniform/block.rs` |
| 025/130 | [ ] | [ ] | `e025_setCurrentRegion` | 4 | `UniformInstrAndBlock.cpp:327` | `uniform/block.rs` |
| 026/130 | [ ] | [ ] | `e026_getMaxInstrSize` | 8 | `UniformInstrAndBlock.cpp:331` | `uniform/block.rs` |
| 027/130 | [ ] | [ ] | `e027_getCurrentRegionInstrSize` | 5 | `UniformInstrAndBlock.cpp:360` | `uniform/block.rs` |
| 028/130 | [ ] | [ ] | `e028_getInstr` | 8 | `UniformInstrAndBlock.cpp:365` | `uniform/block.rs` |
| 029/130 | [ ] | [ ] | `e029_getLastInstr` | 3 | `UniformInstrAndBlock.cpp:374` | `uniform/block.rs` |
| 030/130 | [ ] | [ ] | `e030_appendEmptyUniformRegion` | 4 | `UniformInstrAndBlock.cpp:378` | `uniform/block.rs` |
| 031/130 | [ ] | [ ] | `e031_getMaxInstrSize` | 7 | `UniformInstrAndBlock.cpp:426` | `uniform/block.rs` |
| 032/130 | [ ] | [ ] | `e032_appendRegularBlock` | 3 | `UniformInstrAndBlock.cpp:442` | `uniform/block.rs` |
| 033/130 | [ ] | [ ] | `e033_appendUniformBlock` | 3 | `UniformInstrAndBlock.cpp:445` | `uniform/block.rs` |
| 034/130 | [ ] | [ ] | `e034_getUniformInstr` | 6 | `UniformInstrAndBlock.cpp:449` | `uniform/block.rs` |
| 035/130 | [ ] | [ ] | `e035_doesInstrWithThisIndexExist` | 7 | `UniformInstrAndBlock.cpp:456` | `uniform/block.rs` |
| 036/130 | [ ] | [ ] | `e036_appendEmptyUniformRegion` | 4 | `UniformInstrAndBlock.cpp:497` | `uniform/block.rs` |
| 037/130 | [ ] | [ ] | `e037_getUnitRegionIndex` | 6 | `UniformInstrAndBlock.hpp:208` | `uniform/block.rs` |
| 038/130 | [ ] | [ ] | `e038_verifyOnTheFlyConversions` | 56 | `Utils.cpp:27` | `utils.rs` |
| 039/130 | [ ] | [ ] | `e039_getProperConsumer` | 14 | `Utils.cpp:86` | `utils.rs` |
| 040/130 | [ ] | [ ] | `e040_updateProperConsumer` | 14 | `Utils.cpp:102` | `utils.rs` |
| 041/130 | [ ] | [ ] | `e041_getAddrWraparounded` | 10 | `Utils.cpp:117` | `utils.rs` |

## Level 1 — 39 units, 2412 declaration lines

| | PORT | AUDIT | unit | LoC | authority | Rust home |
|---|---|---|---|---|---|---|
| 042/130 | [ ] | [ ] | `e042_ConstructMVLoopInstr` | 63 | `ConstructProgIRHelper.cpp:48` | `construct/scalar.rs` |
| 043/130 | [ ] | [ ] | `e043_ConstructAssignInstr` | 204 | `ConstructProgIRHelper.cpp:113` | `construct/scalar.rs` |
| 044/130 | [ ] | [ ] | `e044_ConstructBranchExitInstr` | 9 | `ConstructProgIRHelper.cpp:319` | `construct/scalar.rs` |
| 045/130 | [ ] | [ ] | `e045_ConstructSyncInstr` | 96 | `ConstructProgIRHelper.cpp:348` | `construct/scalar.rs` |
| 046/130 | [ ] | [ ] | `e046_ConstructJMPInstr` | 11 | `ConstructProgIRHelper.cpp:446` | `construct/scalar.rs` |
| 047/130 | [ ] | [ ] | `e047_ConstructJCMPInstr` | 227 | `ConstructProgIRHelper.cpp:458` | `construct/scalar.rs` |
| 048/130 | [ ] | [ ] | `e048_ConstructJADDInstr` | 33 | `ConstructProgIRHelper.cpp:687` | `construct/scalar.rs` |
| 049/130 | [ ] | [ ] | `e049_ConstructXRFADDInstr` | 47 | `ConstructProgIRHelper.cpp:892` | `construct/scalar.rs` |
| 050/130 | [ ] | [ ] | `e050_ConstructXRFADDInstr` | 22 | `ConstructProgIRHelper.cpp:940` | `construct/scalar.rs` |
| 051/130 | [ ] | [ ] | `e051_ConstructJSUBInstr` | 34 | `ConstructProgIRHelper.cpp:964` | `construct/scalar.rs` |
| 052/130 | [ ] | [ ] | `e052_setSentientComputeInputProgIROperands` | 125 | `ConstructProgIRHelper.cpp:1184` | `construct/compute.rs` |
| 053/130 | [ ] | [ ] | `e053_setSentientComputeOutputProgIROperands` | 112 | `ConstructProgIRHelper.cpp:1310` | `construct/compute.rs` |
| 054/130 | [ ] | [ ] | `e054_ConstructL3LoadInstr` | 95 | `ConstructProgIRHelper.cpp:2576` | `construct/transfer.rs` |
| 055/130 | [ ] | [ ] | `e055_ConstructL3StoreInstr` | 150 | `ConstructProgIRHelper.cpp:2672` | `construct/transfer.rs` |
| 056/130 | [ ] | [ ] | `e056_ConstructZRAssignInstr` | 14 | `ConstructProgIRHelper.cpp:2828` | `construct/transfer.rs` |
| 057/130 | [ ] | [ ] | `e057_ConstructL3LoadAndStoreInstr` | 179 | `ConstructProgIRHelper.cpp:2843` | `construct/transfer.rs` |
| 058/130 | [ ] | [ ] | `e058_ConstructLoadComputeInstr` | 99 | `ConstructProgIRHelper.cpp:3287` | `construct/transfer.rs` |
| 059/130 | [ ] | [ ] | `e059_ConstructLRFCopyInstr` | 49 | `ConstructProgIRHelper.cpp:3509` | `construct/transfer.rs` |
| 060/130 | [ ] | [ ] | `e060_ConstructSetDstMaskInstr` | 41 | `ConstructProgIRHelper.cpp:3559` | `construct/mask_and_splat.rs` |
| 061/130 | [ ] | [ ] | `e061_ConstructSetDestInstr` | 54 | `ConstructProgIRHelper.cpp:3601` | `construct/mask_and_splat.rs` |
| 062/130 | [ ] | [ ] | `e062_ConstructImmCopyInstrFromSplatOp` | 60 | `ConstructProgIRHelper.cpp:3657` | `construct/mask_and_splat.rs` |
| 063/130 | [ ] | [ ] | `e063_ConstructOpaqueInstr` | 167 | `ConstructProgIRHelper.cpp:3866` | `construct/opaque.rs` |
| 064/130 | [ ] | [ ] | `e064_ConstructSAMVInstr` | 33 | `ConstructProgIRHelper.cpp:4034` | `construct/mask_and_splat.rs` |
| 065/130 | [ ] | [ ] | `e065_ConstructSAMVResetInstruction` | 26 | `ConstructProgIRHelper.cpp:4068` | `construct/mask_and_splat.rs` |
| 066/130 | [ ] | [ ] | `e066_ConstructSetMaskInstr` | 22 | `ConstructProgIRHelper.cpp:4095` | `construct/mask_and_splat.rs` |
| 067/130 | [ ] | [ ] | `e067_fillImmField` | 26 | `ConstructProgIRHelper.cpp:4130` | `construct/reg_init.rs` |
| 068/130 | [ ] | [ ] | `e068_getRegImmVals` | 102 | `LowerSentientHelper.cpp:1140` | `lower/labels_and_regs.rs` |
| 069/130 | [ ] | [ ] | `e069_recordOpRegDefs` | 30 | `RegDefTracker.cpp:22` | `reg_def_tracker.rs` |
| 070/130 | [ ] | [ ] | `e070_dtor_UniformRegionContext` | 28 | `RegDefTracker.cpp:77` | `reg_def_tracker.rs` |
| 071/130 | [ ] | [ ] | `e071_createJmpInstr` | 12 | `UniformInstrAndBlock.cpp:30` | `uniform/instr.rs` |
| 072/130 | [ ] | [ ] | `e072_addEntryToOperandMap` | 121 | `UniformInstrAndBlock.cpp:52` | `uniform/instr.rs` |
| 073/130 | [ ] | [ ] | `e073_addEntryToOperandMap` | 43 | `UniformInstrAndBlock.cpp:174` | `uniform/instr.rs` |
| 074/130 | [ ] | [ ] | `e074_getMaxInstrRegionIndex` | 12 | `UniformInstrAndBlock.cpp:340` | `uniform/block.rs` |
| 075/130 | [ ] | [ ] | `e075_getRegionInstrSize` | 7 | `UniformInstrAndBlock.cpp:353` | `uniform/block.rs` |
| 076/130 | [ ] | [ ] | `e076_getUnitUniformInstrList` | 9 | `UniformInstrAndBlock.cpp:383` | `uniform/block.rs` |
| 077/130 | [ ] | [ ] | `e077_getLastBlockCurrentInstrSize` | 3 | `UniformInstrAndBlock.cpp:433` | `uniform/block.rs` |
| 078/130 | [ ] | [ ] | `e078_addInstructionToLastBlock` | 6 | `UniformInstrAndBlock.cpp:436` | `uniform/block.rs` |
| 079/130 | [ ] | [ ] | `e079_getNextInstrIndex` | 19 | `UniformInstrAndBlock.cpp:464` | `uniform/block.rs` |
| 080/130 | [ ] | [ ] | `e080_setFCValueFromFoldMode` | 22 | `Utils.cpp:128` | `utils.rs` |

## Level 2 — 17 units, 2125 declaration lines

| | PORT | AUDIT | unit | LoC | authority | Rust home |
|---|---|---|---|---|---|---|
| 081/130 | [ ] | [ ] | `e081_ConstructLRFADDInstr` | 88 | `ConstructProgIRHelper.cpp:721` | `construct/scalar.rs` |
| 082/130 | [ ] | [ ] | `e082_ConstructLARorEARADDInstr` | 81 | `ConstructProgIRHelper.cpp:810` | `construct/scalar.rs` |
| 083/130 | [ ] | [ ] | `e083_ConstructLRFSUBInstr` | 136 | `ConstructProgIRHelper.cpp:999` | `construct/scalar.rs` |
| 084/130 | [ ] | [ ] | `e084_ConstructLARorEARSUBInstr` | 46 | `ConstructProgIRHelper.cpp:1136` | `construct/scalar.rs` |
| 085/130 | [ ] | [ ] | `e085_ConstructFMAInstr` | 291 | `ConstructProgIRHelper.cpp:1423` | `construct/compute.rs` |
| 086/130 | [ ] | [ ] | `e086_ConstructBinaryInstr` | 485 | `ConstructProgIRHelper.cpp:1715` | `construct/compute.rs` |
| 087/130 | [ ] | [ ] | `e087_ConstructUnaryInstr` | 237 | `ConstructProgIRHelper.cpp:2201` | `construct/compute.rs` |
| 088/130 | [ ] | [ ] | `e088_ConstructTernaryInstr` | 122 | `ConstructProgIRHelper.cpp:2439` | `construct/compute.rs` |
| 089/130 | [ ] | [ ] | `e089_ConstructLoadInstr` | 201 | `ConstructProgIRHelper.cpp:3024` | `construct/transfer.rs` |
| 090/130 | [ ] | [ ] | `e090_ConstructLoadInstr` | 60 | `ConstructProgIRHelper.cpp:3226` | `construct/transfer.rs` |
| 091/130 | [ ] | [ ] | `e091_ConstructStoreInstr` | 121 | `ConstructProgIRHelper.cpp:3387` | `construct/transfer.rs` |
| 092/130 | [ ] | [ ] | `e092_ConstructSplatInstrFromSplatOp` | 65 | `ConstructProgIRHelper.cpp:3719` | `construct/mask_and_splat.rs` |
| 093/130 | [ ] | [ ] | `e093_AddToLabelsMap` | 8 | `LowerSentientHelper.cpp:32` | `lower/labels_and_regs.rs` |
| 094/130 | [ ] | [ ] | `e094_updateLabelAndAddToCodeGraph` | 40 | `LowerSentientHelper.cpp:47` | `lower/labels_and_regs.rs` |
| 095/130 | [ ] | [ ] | `e095_equalizeProgramLength` | 90 | `SentientToProgIR.cpp:469` | `driver.rs` |
| 096/130 | [ ] | [ ] | `e096_getUniformizedUnitInstrList` | 42 | `UniformInstrAndBlock.cpp:279` | `uniform/block.rs` |
| 097/130 | [ ] | [ ] | `e097_flattenIndex` | 12 | `UniformInstrAndBlock.cpp:484` | `uniform/block.rs` |

## Level 3 — 24 units, 658 declaration lines

| | PORT | AUDIT | unit | LoC | authority | Rust home |
|---|---|---|---|---|---|---|
| 098/130 | [ ] | [ ] | `e098_ConstructSplatPadInstr` | 65 | `ConstructProgIRHelper.cpp:3792` | `construct/mask_and_splat.rs` |
| 099/130 | [ ] | [ ] | `e099_LowerLoadAndSendOperation` | 29 | `LowerSentientHelper.cpp:144` | `lower/transfer.rs` |
| 100/130 | [ ] | [ ] | `e100_LowerReceiveAndStoreOperation` | 49 | `LowerSentientHelper.cpp:174` | `lower/transfer.rs` |
| 101/130 | [ ] | [ ] | `e101_LowerLoadAndStoreOperation` | 55 | `LowerSentientHelper.cpp:224` | `lower/transfer.rs` |
| 102/130 | [ ] | [ ] | `e102_LowerLoadAndExtractScalarOperation` | 24 | `LowerSentientHelper.cpp:280` | `lower/transfer.rs` |
| 103/130 | [ ] | [ ] | `e103_LowerReceiveAndExtractScalarOperation` | 13 | `LowerSentientHelper.cpp:305` | `lower/transfer.rs` |
| 104/130 | [ ] | [ ] | `e104_LowerLoadComputeAndSendOperation` | 33 | `LowerSentientHelper.cpp:319` | `lower/transfer.rs` |
| 105/130 | [ ] | [ ] | `e105_LowerSyncOperation` | 37 | `LowerSentientHelper.cpp:353` | `lower/control.rs` |
| 106/130 | [ ] | [ ] | `e106_LowerNOPOperation` | 14 | `LowerSentientHelper.cpp:391` | `lower/control.rs` |
| 107/130 | [ ] | [ ] | `e107_LowerCommonOperations` | 26 | `LowerSentientHelper.cpp:538` | `lower/compute.rs` |
| 108/130 | [ ] | [ ] | `e108_LowerBinaryOperation` | 8 | `LowerSentientHelper.cpp:565` | `lower/compute.rs` |
| 109/130 | [ ] | [ ] | `e109_LowerUnaryOperation` | 7 | `LowerSentientHelper.cpp:574` | `lower/compute.rs` |
| 110/130 | [ ] | [ ] | `e110_LowerTernaryOperation` | 8 | `LowerSentientHelper.cpp:582` | `lower/compute.rs` |
| 111/130 | [ ] | [ ] | `e111_LowerForOperation` | 66 | `LowerSentientHelper.cpp:591` | `lower/control.rs` |
| 112/130 | [ ] | [ ] | `e112_LowerMACOperation` | 31 | `LowerSentientHelper.cpp:658` | `lower/compute.rs` |
| 113/130 | [ ] | [ ] | `e113_LowerSubOperation` | 40 | `LowerSentientHelper.cpp:690` | `lower/compute.rs` |
| 114/130 | [ ] | [ ] | `e114_LowerAddOperation` | 50 | `LowerSentientHelper.cpp:731` | `lower/compute.rs` |
| 115/130 | [ ] | [ ] | `e115_LowerReturnOperation` | 7 | `LowerSentientHelper.cpp:900` | `lower/control.rs` |
| 116/130 | [ ] | [ ] | `e116_LowerSetSendDestinationOperation` | 18 | `LowerSentientHelper.cpp:908` | `lower/control.rs` |
| 117/130 | [ ] | [ ] | `e117_LowerOpaqueOperation` | 12 | `LowerSentientHelper.cpp:939` | `lower/compute.rs` |
| 118/130 | [ ] | [ ] | `e118_LowerSAMVOperation` | 14 | `LowerSentientHelper.cpp:952` | `lower/compute.rs` |
| 119/130 | [ ] | [ ] | `e119_LowerSetMaskOperation` | 10 | `LowerSentientHelper.cpp:1118` | `lower/compute.rs` |
| 120/130 | [ ] | [ ] | `e120_LowerIncrMaskOperation` | 10 | `LowerSentientHelper.cpp:1129` | `lower/compute.rs` |
| 121/130 | [ ] | [ ] | `e121_getUniformizedUnitUniformInstrList` | 32 | `UniformInstrAndBlock.cpp:393` | `uniform/block.rs` |

## Level 4 — 4 units, 284 declaration lines

| | PORT | AUDIT | unit | LoC | authority | Rust home |
|---|---|---|---|---|---|---|
| 122/130 | [ ] | [ ] | `e122_LowerCopyOperation` | 95 | `LowerSentientHelper.cpp:406` | `lower/transfer.rs` |
| 123/130 | [ ] | [ ] | `e123_LowerYieldOperation` | 117 | `LowerSentientHelper.cpp:782` | `lower/control.rs` |
| 124/130 | [ ] | [ ] | `e124_LowerSplatOperation` | 11 | `LowerSentientHelper.cpp:927` | `lower/compute.rs` |
| 125/130 | [ ] | [ ] | `e125_LowerUniformYieldOperation` | 61 | `LowerSentientHelper.cpp:1056` | `lower/control.rs` |

## Level 5 — 3 units, 246 declaration lines

| | PORT | AUDIT | unit | LoC | authority | Rust home |
|---|---|---|---|---|---|---|
| 126/130 | [ ] | [ ] | `e126_LowerUniformOperations` | 70 | `LowerSentientHelper.cpp:985` | `lower/control.rs` |
| 127/130 | [ ] | [ ] | `e127_GenerateProgIR` | 65 | `SentientToProgIR.cpp:53` | `driver.rs` |
| 128/130 | [ ] | [ ] | `e128_GenerateProgIR` | 111 | `SentientToProgIR.cpp:119` | `driver.rs` |

## Level 6 — 1 units, 186 declaration lines

| | PORT | AUDIT | unit | LoC | authority | Rust home |
|---|---|---|---|---|---|---|
| 129/130 | [ ] | [ ] | `e129_GenerateProgIRForProgramUnit` | 186 | `SentientToProgIR.cpp:234` | `driver.rs` |

## Level 7 — 1 units, 135 declaration lines

| | PORT | AUDIT | unit | LoC | authority | Rust home |
|---|---|---|---|---|---|---|
| 130/130 | [ ] | [ ] | `e130_runOnOperation` | 135 | `SentientToProgIR.cpp:610` | `driver.rs` |

## Excluded — 36 definitions, with a reason each

The criterion is what a Rust struct field or a `const` already is, NOT what is hard.

| definition | authority | LoC | reason |
|---|---|---|---|
| `regDefChecking` | `RegDefTracker.hpp:25` | 8 | `return false;` -- a compile-time debug constant, a `const bool` in Rust |
| `enabled` | `RegDefTracker.hpp:48` | 1 | 1-line accessor over two flags |
| `dccExtContext` | `RegDefTracker.hpp:49` | 1 | 1-line field accessor |
| `progStateInfo` | `RegDefTracker.hpp:50` | 1 | 1-line field accessor |
| `regs` | `RegDefTracker.hpp:69` | 4 | context-chain lookup guarded by DT_CHECK_MSG -- the operand-reaching mechanism the brief allows dropping, and a runtime abort this crate forbids |
| `regs` | `RegDefTracker.hpp:73` | 4 | the const overload of the same |
| `initializeOutputStream` | `SentientToProgIR.cpp:421` | 7 | ostream plumbing (copyfmt/rdbuf); this crate emits and never configures a C++ stream |
| `mlir::sentient::createSentientToProgIRPass` | `SentientToProgIR.cpp:746` | 12 | MLIR pass factory -- #[forward] runs the pipeline at expansion, there is no pass manager |
| `mlir::sentient::createSentientToProgIRPass` | `SentientToProgIR.cpp:759` | 8 | MLIR pass-registry factory over a dummy context; same reason |
| `dccExtContext` | `SentientToProgIR.hpp:107` | 1 | 1-line field accessor -- a struct field in Rust, not a function |
| `GetAddressScale` | `SentientToProgIR.hpp:438` | 3 | 3-line overload that stringifies its operand and forwards to the .cpp GetAddressScale (entry kept) |
| `begin` | `UniformInstrAndBlock.hpp:54` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `end` | `UniformInstrAndBlock.hpp:55` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `empty` | `UniformInstrAndBlock.hpp:56` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `find` | `UniformInstrAndBlock.hpp:57` | 3 | 3-line header inline over a member -- a struct field in Rust, not a function |
| `getInstn` | `UniformInstrAndBlock.hpp:120` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `setInstn` | `UniformInstrAndBlock.hpp:121` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `getTag` | `UniformInstrAndBlock.hpp:122` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `setTag` | `UniformInstrAndBlock.hpp:123` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `setUniformizedComment` | `UniformInstrAndBlock.hpp:124` | 4 | comment bookkeeping; setComment drops the text unless enable_debug_flag (progir.h:322-330) so a comment is never load-bearing |
| `setCommonComment` | `UniformInstrAndBlock.hpp:128` | 4 | the same, for the common comment |
| `getOperandMap` | `UniformInstrAndBlock.hpp:132` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `setOperandMap` | `UniformInstrAndBlock.hpp:133` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `getDeadCode` | `UniformInstrAndBlock.hpp:134` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `setDeadCode` | `UniformInstrAndBlock.hpp:135` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `getCommonFields` | `UniformInstrAndBlock.hpp:159` | 3 | 3-line header inline over a member -- a struct field in Rust, not a function |
| `getUnitToComment` | `UniformInstrAndBlock.hpp:162` | 3 | 3-line header inline over a member -- a struct field in Rust, not a function |
| `getCommonComment` | `UniformInstrAndBlock.hpp:165` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `block_type_` | `UniformInstrAndBlock.hpp:178` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `empty` | `UniformInstrAndBlock.hpp:179` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `getInstrLists` | `UniformInstrAndBlock.hpp:194` | 3 | 3-line header inline over a member -- a struct field in Rust, not a function |
| `getCurrentRegion` | `UniformInstrAndBlock.hpp:197` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `getType` | `UniformInstrAndBlock.hpp:200` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `setUnitRegionIndex` | `UniformInstrAndBlock.hpp:205` | 3 | 3-line header inline over a member -- a struct field in Rust, not a function |
| `getBlocks` | `UniformInstrAndBlock.hpp:267` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |
| `empty` | `UniformInstrAndBlock.hpp:268` | 1 | 1-line header inline over a member -- a struct field in Rust, not a function |

## Kept despite living in a header

`getUnitRegionIndex` (`UniformInstrAndBlock.hpp:208`, 6L) is scheduled: its body is a real
decision rule — a REGULAR block always answers 0, a missing entry answers -1 — and a `-1`
sentinel is an `Option` in this crate, never a negative number.
