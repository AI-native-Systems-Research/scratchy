// ================================================================================================
// DDC / L3-SCHEDULER CAMPAIGN - consolidated translation unit: dcg
//
// STAGE 3: DcgManager::runDcgForDlOpsStandalone(sdsc) -- the branch SchedulerStages.cpp:53-57 takes whenever dscs_ is non-empty (always, for us). NOT runDcg.
//   dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29-41 is the call site.
//
// 16 of the campaign's 382 function bodies, from dcg/dcg_manager/, extracted VERBATIM
// and emitted in dependency order over the SCC condensation (level 0 calls
// nothing else in the span; level N only calls levels below N).
// Entry numbers are GLOBAL across the campaign's three TUs - one ladder.
//
// AUTHORITY, and the ONLY thing to port from: /Users/nickm/git/deeptools-src
//   revision a0d29abbed
// The other local deeptools checkout is a DIFFERENT revision. Do not use it.
// The pod is not reachable from this host and is not the authority here.
//
// Each entry's banner gives its unit symbol, level, and original file:line. The
// signature is rewritten to the unit symbol so symbol == unit name; the body
// between `{` and its matching `}` is byte-for-byte the authority's.
// ================================================================================================

#include "prelude.inc"

// ------------------------------------------------------------------------------------------------
// entry 188/382   level 0   scc 151   13 body lines
// unit: e188_runDcgComputeTransfer
// authority: dcg/dcg_manager/dcg_manager.cpp:112
// original: void DcgManager::runDcgComputeTransfer(SuperDsc& mySDsc)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e188_runDcgComputeTransfer(SuperDsc& mySDsc)
{
  if (verbose > 0)
    std::cout << "Running DCG's Compute Transfer for DataOp: Node-name:"
              << mySDsc.name_ << std::endl;

  DT_CHECK(senCompToISAptr != nullptr);
  DT_CHECK(!isInpFetchNeigh);
  int c = 0;
  for (auto& myDataOpDsc : mySDsc.dataOpdscs_) {
    dcg_fe_.computeTranferforDataOp(mySDsc, myDataOpDsc, c);
    c++;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 189/382   level 0   scc 154   52 body lines
// unit: e189_runDcgForDlOps
// authority: dcg/dcg_manager/dcg_manager.cpp:216
// original: void DcgManager::runDcgForDlOps(SuperDsc& mySDsc)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e189_runDcgForDlOps(SuperDsc& mySDsc)
{
  if (verbose > 0)
    std::cout << "Running DCG for DLOp: Node-name:" << mySDsc.name_
              << std::endl;
  DT_CHECK(mySDsc.dscs_.size() >= 1);
  DT_CHECK(!isInpFetchNeigh);
  DT_CHECK(mySDsc.dataOpdscs_.size() == 0);
  DT_CHECK(senCompToISAptr != nullptr);
  // mySDsc.exportJson(std::cout, false);

  std::vector<SenPcfg> pcfgL3lu;  // core idx..
  std::vector<SenPcfg> pcfgL3su;

  dcg_fe_.generatePcfgIRForDLOp(mySDsc, pcfgL3lu, pcfgL3su);

  bool useDt1 = false;

  if (dscGlobal->dtVersion > 1 && !useDt1) {
    // need to insert pcfgL3lu and pcfgL3su in sdsc pcfg
    for (auto& kv : mySDsc.coreIdToDsc_) {
      auto coreID = kv.first;
      for (int i = 0; i < 2; i++) {
        SenComponents senCompType =
            (i == 0) ? SenComponents::L3LU : SenComponents::L3SU;
        SenPcfg* localPcfg =
            (i == 0) ? &pcfgL3lu.at(coreID) : &pcfgL3su.at(coreID);

        int pcfgId;
        mySDsc.pcfgMap_.try_emplace(coreID);
        if (mySDsc.pcfgMap_.at(coreID).count(senCompType)) {
          pcfgId = mySDsc.pcfgMap_.at(coreID).at(senCompType);
        } else {
          pcfgId = mySDsc.pcfgPool_.empty()
                       ? 0
                       : mySDsc.pcfgPool_.rbegin()->first + 1;
          mySDsc.pcfgPool_.try_emplace(pcfgId);
          mySDsc.pcfgMap_.at(coreID)[senCompType] = pcfgId;
        }

        auto& sdscPcfg = mySDsc.pcfgPool_.at(pcfgId);
        sdscPcfg.mergeSenPcfg(sdscPcfg, *localPcfg);  // copy
      }
    }
  }

  if (createSenProg) {
    DT_CHECK_MSG(mySDsc.sdscFoldProps_.empty(),
                 "Codegen for Folded Super-DSC is not supported");
    DT_CHECK(progIRcodeGen == DCGProgIRGen::DCG);
    dcg_be_.convertToProgIRDlOp(mySDsc, pcfgL3lu, pcfgL3su);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 190/382   level 0   scc 158   64 body lines
// unit: e190_runDcgForDlOpsStandalone
// authority: dcg/dcg_manager/dcg_manager.cpp:449
// original: void DcgManager::runDcgForDlOpsStandalone(SuperDsc& mySDsc)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e190_runDcgForDlOpsStandalone(SuperDsc& mySDsc)
{
  DT_CHECK(senCompToISAptr != nullptr);  // check sdsc

  // ACT1: generate pcfg for DL ops
  DT_CHECK(mySDsc.dscs_.size() >= 1);
  DT_CHECK(senCompToISAptr != nullptr);

  // create PCFG for l3-lu, l3-su
  int maxCoreId = 0;
  for (auto& entry : mySDsc.coreIdToDsc_) {
    if (maxCoreId < entry.first) {
      maxCoreId = entry.first;
    }
  }
  std::vector<SenPcfg> pcfgL3lu(maxCoreId + 1);  // core idx..
  std::vector<SenPcfg> pcfgL3su(maxCoreId + 1);
  int maxGrpIDinL3 = 0;
  for (auto& kv : mySDsc.coreIdToDsc_) {
    auto coreId = kv.first;
    mySDsc.pcfg_[coreId][SenComponents::L3LU];
    mySDsc.pcfg_[coreId][SenComponents::L3SU];
    const auto dsc = kv.second;
    const bool hasDSC2 = dsc->isDSC2();
    int maxGrpIDLU;
    int maxGrpIDSU;

    if (enableL3DlScheduler && hasDSC2) {
      DscPcfgTranslator::PcfgInfo infoL3lu = {
          mySDsc.pcfg_[coreId][SenComponents::L3LU], coreId, 0,
          SenComponents::L3LU, SenComponents::L3LU};
      DscPcfgTranslator::PcfgInfo infoL3su = {
          mySDsc.pcfg_[coreId][SenComponents::L3SU], coreId, 0,
          SenComponents::L3SU, SenComponents::L3SU};
      DscPcfgTranslator::transformDscCompToPcfg(mySDsc, *dscGlobal, *dsc,
                                                infoL3lu);
      DscPcfgTranslator::transformDscCompToPcfg(mySDsc, *dscGlobal, *dsc,
                                                infoL3su);
      maxGrpIDLU = mySDsc.pcfg_[coreId][SenComponents::L3LU].getMaxGTRGroupId();
      maxGrpIDSU = mySDsc.pcfg_[coreId][SenComponents::L3SU].getMaxGTRGroupId();
    } else {
      maxGrpIDLU = dcg_fe_.createPcfgForUnitPerCore(
          mySDsc, pcfgL3lu.at(coreId), SenComponents::L3LU, coreId);
      maxGrpIDSU = dcg_fe_.createPcfgForUnitPerCore(
          mySDsc, pcfgL3su.at(coreId), SenComponents::L3SU, coreId);
    }

    maxGrpIDinL3 = maxGrpIDinL3 < maxGrpIDLU ? maxGrpIDLU : maxGrpIDinL3;
    maxGrpIDinL3 = maxGrpIDinL3 < maxGrpIDSU ? maxGrpIDSU : maxGrpIDinL3;
  }

  // DT_CHECK(maxGrpIDinL3 <= maxGrpIDL3);
  firstAvailGlobalGrpId += maxGrpIDinL3;
  firstAvailGlobalGrpId = firstAvailGlobalGrpId % sysDef.maxGroupID;

  // create merged pcfg for
  // copy to sdsc pcfg
  if (createSenProg) {
    DT_CHECK_MSG(mySDsc.sdscFoldProps_.empty(),
                 "Codegen for Folded Super-DSC is not supported");
    DT_CHECK(progIRcodeGen != DCGProgIRGen::DCG);
    // ACT3: create sen programs
    dcg_be_.fillAndCreateSenProgInfoUsingSuperDSC(mySDsc);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 191/382   level 0   scc 152   60 body lines
// unit: e191_removeextraPTrows
// authority: dcg/dcg_manager/dcg_manager.cpp:568
// original: void DcgManager::removeextraPTrows(SuperDsc& mySDsc)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e191_removeextraPTrows(SuperDsc& mySDsc)
{
  // delete extra pcfgs
  bool removePTPcfgNode = false;
  std::set<SenComponents> pcfgUnitToRemove = {
      SenComponents::PTROW1_0, SenComponents::PTROW2_0, SenComponents::PTROW3_0,
      SenComponents::PTROW4_0, SenComponents::PTROW5_0, SenComponents::PTROW6_0,
      SenComponents::PTROW7_0, SenComponents::PTROW1_1, SenComponents::PTROW2_1,
      SenComponents::PTROW3_1, SenComponents::PTROW4_1, SenComponents::PTROW5_1,
      SenComponents::PTROW6_1, SenComponents::PTROW7_1};

  for (auto& myDataOpDsc : mySDsc.dataOpdscs_) {
    if (myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTLx ||
        myDataOpDsc.op->name == OpFuncs::ReStickifyOpWithPTHBM) {
      removePTPcfgNode = true;
      for (auto& pcfgVecPerCore : myDataOpDsc.pcfg_) {
        for (auto it = pcfgVecPerCore.begin(); it != pcfgVecPerCore.end();) {
          if (pcfgUnitToRemove.count((*it).second)) {
            it = pcfgVecPerCore.erase(it);
          } else {
            ++it;
          }
        }
      }
    }
  }

  if (removePTPcfgNode) {
    if (dscGlobal->dtVersion > 1) {
      for (auto& kv : mySDsc.pcfgMap_) {
        int coreId = kv.first;
        std::vector<std::pair<int, SenComponents>> idsToRemove;
        std::vector<SenComponents> unitsToRemove;
        for (auto& kv2 : kv.second) {
          SenComponents unitType = kv2.first;
          if (pcfgUnitToRemove.count(unitType)) {
            idsToRemove.emplace_back(kv2.second, unitType);
          }
        }
        for (auto& entry : idsToRemove) {
          mySDsc.pcfgPool_.erase(entry.first);
          mySDsc.pcfgMap_.at(coreId).erase(entry.second);
        }
      }
    } else if (mySDsc.dataOpdscs_.size() > 1) {
      for (auto& kv : mySDsc.pcfg_) {
        int coreId = kv.first;
        std::vector<SenComponents> unitsToRemove;
        for (auto& kv2 : kv.second) {
          auto unitType = kv2.first;
          if (pcfgUnitToRemove.count(unitType)) {
            unitsToRemove.push_back(unitType);
          }
        }
        for (auto& entry : unitsToRemove) {
          kv.second.erase(entry);
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 192/382   level 0   scc 162   16 body lines
// unit: e192_printSenProgram
// authority: dcg/dcg_manager/dcg_manager.cpp:959
// original: void DcgManager::printSenProgram(SuperDsc* mySDsc, Dpc& myDpc, std::string fileName, bool use_smc /*= false*/)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e192_printSenProgram(SuperDsc* mySDsc, Dpc& myDpc,
                                 std::string fileName,
                                 bool use_smc /*= false*/)
{
  std::ofstream fileFP(fileName);
  if (!fileFP.is_open()) {
    DT_ERROR_FMT("Unable to open %s for writing SenProgs", fileName.c_str());
  }

  if (!use_smc) {
    myDpc.convertIr2Senprog(mySDsc->progstateinfo_, *senCompToISAptr, fileFP,
                            std::nullopt);
  } else {
    myDpc.convertIr2SMC(mySDsc->progstateinfo_, *senCompToISAptr, fileFP,
                        std::nullopt);
  }

  fileFP.close();
}

// ------------------------------------------------------------------------------------------------
// entry 193/382   level 0   scc 163   3 body lines
// unit: e193_runDcgForGenKG3
// authority: dcg/dcg_manager/dcg_manager.h:94
// original: void runDcgForGenKG3(SuperDsc& mySDsc)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e193_runDcgForGenKG3(SuperDsc& mySDsc)
{
    dcg_be_.runDcgForGenKG3(mySDsc, createSenProg);
  }

// ------------------------------------------------------------------------------------------------
// entry 194/382   level 0   scc 164   3 body lines
// unit: e194_runDcgForSparseKG3CONV2D
// authority: dcg/dcg_manager/dcg_manager.h:98
// original: void runDcgForSparseKG3CONV2D(SuperDsc& mySDsc)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e194_runDcgForSparseKG3CONV2D(SuperDsc& mySDsc)
{
    dcg_be_.runDcgForSparseKG3CONV2D(mySDsc, createSenProg);
  }

// ------------------------------------------------------------------------------------------------
// entry 195/382   level 0   scc 165   3 body lines
// unit: e195_runDcgForSparseKG3BMM
// authority: dcg/dcg_manager/dcg_manager.h:102
// original: void runDcgForSparseKG3BMM(SuperDsc& mySDsc)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e195_runDcgForSparseKG3BMM(SuperDsc& mySDsc)
{
    dcg_be_.runDcgForSparseKG3BMM(mySDsc, createSenProg);
  }

// ------------------------------------------------------------------------------------------------
// entry 196/382   level 0   scc 166   3 body lines
// unit: e196_printTrafficPerCore
// authority: dcg/dcg_manager/dcg_manager.h:120
// original: void printTrafficPerCore(SuperDsc& sdsc, std::string fileName)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e196_printTrafficPerCore(SuperDsc& sdsc, std::string fileName)
{
    dcg_fe_.printTrafficPerCore(sdsc, fileName);
  }

// ------------------------------------------------------------------------------------------------
// entry 280/382   level 1   scc 153   70 body lines
// unit: e280_runDcgGenerateProgIR
// authority: dcg/dcg_manager/dcg_manager.cpp:145
// original: void DcgManager::runDcgGenerateProgIR(SuperDsc& mySDsc)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e280_runDcgGenerateProgIR(SuperDsc& mySDsc)
{
  if (verbose > 0)
    std::cout << "Running DCG's PROG IR Gen for DataOp: Node-name:"
              << mySDsc.name_ << std::endl;

  DT_CHECK(senCompToISAptr != nullptr);
  DT_CHECK(!isInpFetchNeigh);
  DT_CHECK(progIRcodeGen == DCGProgIRGen::DCG);

  // SenPrograms
  if (createSenProg) {
    DT_CHECK_MSG(mySDsc.sdscFoldProps_.empty(),
                 "Codegen for Folded Super-DSC is not supported");
    if (mySDsc.dataOpdscs_.size() <= 1) {
      if (verbose > 0) {
        std::cout << "Creating SenPrograms..." << std::endl;
      }

      DT_CHECK(mySDsc.dataOpdscs_.size());
      DataOpDsc& myDataOpDsc = mySDsc.dataOpdscs_[0];

      if (myDataOpDsc.op->name == OpFuncs::GatherOpHBM) {
        DT_CHECK((*senCompToISAptr).count(SenComponents::L3LU));
        DT_CHECK((*senCompToISAptr).at(SenComponents::L3LU).sysDef.coreArch >=
                 IsaCoreGen::MPW4_ISA);
      }
      auto coreIDTostatus =
          dcg_be_.createSenProgramSTCDPOp(&myDataOpDsc, &mySDsc);
      bool ibuffViolation = false;
      for (auto& kv : coreIDTostatus) {
        auto status = kv.second;
        if (status.first == ProgramAndStateInfo::ErrorType::IBUFF_OVERFLOW &&
            myDataOpDsc.op->name == OpFuncs::STCDPOpLx) {
          ibuffViolation = true;
        } else {
          if (enable_prog_verification_)
            DT_ERROR("Program verification failed for core " +
                     std::to_string(kv.first) + " node " + mySDsc.name_ +
                     "\nError message: " + status.second);
        }
      }

      if (ibuffViolation) {
        if (verbose > 0)
          std::cout << "I-buff violation detected : Re-generating senprogs\n";
        coreIDTostatus.clear();
        // need to re-set as now we can have dynamic mvloops
        dcg_fe_.finalizeBurstInfo((STCDPOpLx*)myDataOpDsc.op);
        dcg_fe_.createPcfgsSTCDPOp(&myDataOpDsc, true);
        coreIDTostatus = dcg_be_.createSenProgramSTCDPOp(&myDataOpDsc, &mySDsc);
      }

      if (enable_prog_verification_) {
        for (auto& kv : coreIDTostatus) {
          auto status = kv.second;
          DT_ERROR("Program verification failed for core " +
                   std::to_string(kv.first) + " node " + mySDsc.name_ +
                   "\nError message: " + status.second);
        }
      }

    } else {
      // senprog gen routine using superDSC's pcfg
      dcg_be_.fillAndCreateSenProgInfoUsingSuperDSC(mySDsc);
    }
  }

  // delete extra pcfgs
  removeextraPTrows(mySDsc);
}

// ------------------------------------------------------------------------------------------------
// entry 281/382   level 1   scc 159   52 body lines
// unit: e281_runDcgForInputFetchNeighbor
// authority: dcg/dcg_manager/dcg_manager.cpp:514
// original: void DcgManager::runDcgForInputFetchNeighbor(SuperDsc& mySDscMain, SuperDsc* mySDscPre)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e281_runDcgForInputFetchNeighbor(SuperDsc& mySDscMain,
                                             SuperDsc* mySDscPre)
{
  if (verbose > 0)
    std::cout << "Running DCG for InputFetchNeighbor: Node-name:"
              << mySDscMain.name_ << std::endl;

  myIFNInfo.dataDsc_Idx = 0;
  DataOpDsc& ddsc =
      *dcg_fe_.generatePcfgIRForDataOpInpFetch(mySDscMain, mySDscPre, 0);
  isInpFetchNeigh = true;
  // ACT2 generate Programs
  // SenPrograms
  if (createSenProg) {
    DT_CHECK_MSG(mySDscMain.sdscFoldProps_.empty(),
                 "Codegen for Folded Super-DSC is not supported");
    if (verbose > 0) {
      std::cout << "Creating SenPrograms..." << std::endl;
    }

    auto coreIDTostatus = dcg_be_.createSenProgramSTCDPOp(&ddsc, &mySDscMain);
    bool ibuffViolation = false;
    for (auto& kv : coreIDTostatus) {
      auto status = kv.second;
      if (status.first == ProgramAndStateInfo::ErrorType::IBUFF_OVERFLOW &&
          ddsc.op->name == OpFuncs::STCDPOpLx) {
        ibuffViolation = true;
      } else {
        if (enable_prog_verification_)
          DT_ERROR("Program verification failed for core " +
                   std::to_string(kv.first) + " node " + mySDscMain.name_ +
                   "\nError message: " + status.second);
      }
    }

    if (ibuffViolation) {
      if (verbose > 0)
        std::cout << "I-buff violation detected : Re-generating senprogs\n";
      coreIDTostatus.clear();
      DT_CHECK(0);  // not ready
      dcg_fe_.createPcfgsSTCDPOp(&ddsc, true);
      coreIDTostatus = dcg_be_.createSenProgramSTCDPOp(&ddsc, &mySDscMain);
    }

    if (enable_prog_verification_) {
      for (auto& kv : coreIDTostatus) {
        auto status = kv.second;
        DT_ERROR("Program verification failed for core " +
                 std::to_string(kv.first) + " node " + mySDscMain.name_ +
                 "\nError message: " + status.second);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 282/382   level 1   scc 155   259 body lines
// unit: e282_mergePcfgInSuperDSC
// authority: dcg/dcg_manager/dcg_manager.cpp:635
// original: void DcgManager::mergePcfgInSuperDSC(SuperDsc& mySDsc, std::vector<SenPcfg>& pcfgL3lu, std::vector<SenPcfg>& pcfgL3su, bool useActualDLpcfg /*= false*/)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e282_mergePcfgInSuperDSC(SuperDsc& mySDsc,
                                     std::vector<SenPcfg>& pcfgL3lu,
                                     std::vector<SenPcfg>& pcfgL3su,
                                     bool useActualDLpcfg /*= false*/)
{
  std::set<SenComponents> allUnits = {L3LU, L3SU, LXLU0, LXSU0, LXLU1, LXSU1};
  std::set<SenComponents> lxUnits = {LXLU0, LXSU0, LXLU1, LXSU1};
  mySDsc.pcfg_.clear();

  // Check if should force DT1
  bool useDt1 = false;

  for (auto& mapkv : mySDsc.coreIdToDscSchedule) {
    auto coreID = mapkv.first;
    auto& dscSchedule = mapkv.second;
    bool seenDLDsc = false;
    bool hasDlDsc = false;
    for (auto& scheduleStep : mapkv.second) {
      if (scheduleStep.dldsc_idx >= 0) {
        hasDlDsc = true;
      }
    }

    // Get DataDSC units
    std::unordered_set<SenComponents> dataDscUnits;
    if (hasDlDsc) {
      for (auto& ddsc : mySDsc.dataOpdscs_) {
        // find coreIDX
        int coreIDX = -1;
        for (int idx2 = 0; idx2 < ddsc.coreIdsUsed_.size(); idx2++) {
          if (ddsc.coreIdsUsed_.at(idx2) == coreID) {
            coreIDX = idx2;
            break;
          }
        }
        if (coreIDX >= 0) {
          auto& pcfgs = ddsc.pcfg_.at(coreIDX);
          for (auto& [_, comp] : pcfgs) {
            dataDscUnits.insert(comp);
          }
        }
      }
    }

    for (auto& scheduleStep : dscSchedule) {
      std::set<SenComponents> unitSyncInserted;

      // Merge function
      auto mergePcfg = [&](SenPcfg& myPcfg, SenComponents senCompType,
                           bool dldsc) {
        if (mySDsc.pcfg_.count(coreID) == 0) {
          mySDsc.pcfg_[coreID][senCompType];
        } else if (mySDsc.pcfg_.at(coreID).count(senCompType) == 0) {
          mySDsc.pcfg_.at(coreID)[senCompType];
        }
        SenPcfg& masterSenPCFG = mySDsc.pcfg_.at(coreID).at(senCompType);

        // add syncs if needed
        if (scheduleStep.before_sync && allUnits.count(senCompType)) {
          std::string tag =
              dldsc ? "before_dsc" + std::to_string(scheduleStep.dldsc_idx)
                    : "before_ddsc" + std::to_string(scheduleStep.datadsc_idx);

          // do this all units
          dcg_fe_.addSyncNode(masterSenPCFG, senCompType, coreID, tag);
          unitSyncInserted.insert(senCompType);
        }

        masterSenPCFG.mergeSenPcfg(masterSenPCFG, myPcfg);  // copy

        // add syncs if needed
        if (scheduleStep.after_sync && allUnits.count(senCompType)) {
          std::string tag =
              dldsc ? "after_dsc" + std::to_string(scheduleStep.dldsc_idx)
                    : "after_ddsc" + std::to_string(scheduleStep.datadsc_idx);

          // do this all units
          dcg_fe_.addSyncNode(masterSenPCFG, senCompType, coreID, tag);
          unitSyncInserted.insert(senCompType);
        }
      };

      if (scheduleStep.datadsc_idx != -1) {
        DT_CHECK(scheduleStep.datadsc_idx >= 0);
        DT_CHECK(scheduleStep.datadsc_idx < mySDsc.dataOpdscs_.size());
        DataOpDsc& ddsc = mySDsc.dataOpdscs_[scheduleStep.datadsc_idx];

        // find coreIDX
        int coreIDX = -1;
        for (int idx2 = 0; idx2 < ddsc.coreIdsUsed_.size(); idx2++) {
          if (ddsc.coreIdsUsed_.at(idx2) == coreID) {
            coreIDX = idx2;
            break;
          }
        }
        DT_CHECK(coreIDX >= 0);
        for (auto& kv2 : ddsc.pcfg_.at(coreIDX)) {
          auto& SenPcfgToCopy = kv2.first;
          auto& senCompType = kv2.second;
          mergePcfg(SenPcfgToCopy, senCompType, false);
        }
      }

      if (scheduleStep.dldsc_idx != -1 &&
          (scheduleStep.datadsc_idx == -1 ||
           mySDsc.target_ == SenTargets::SENPCFG)) {
        if (mySDsc.target_ == SenTargets::SENPCFG &&
            scheduleStep.datadsc_idx != -1)
          DT_CHECK(useActualDLpcfg);

        if (useActualDLpcfg) {
          DT_CHECK(scheduleStep.dldsc_idx >= 0);
          DT_CHECK(scheduleStep.dldsc_idx < mySDsc.dscs_.size());
          auto& dsc = mySDsc.dscs_.at(scheduleStep.dldsc_idx);

          // find coreIDX
          int coreIDX = -1;
          for (int idx2 = 0; idx2 < dsc.coreIdsUsed_.size(); idx2++) {
            if (dsc.coreIdsUsed_.at(idx2) == coreID) {
              coreIDX = idx2;
              break;
            }
          }
          DT_CHECK(coreIDX >= 0);

          // Merge DSC PCFG
          if (dsc.pcfg_.size() > coreIDX) {
            SenPcfg& myPcfg = dsc.pcfg_.at(coreIDX);
            if (myPcfg.srcNode != nullptr) {
              mergePcfg(myPcfg, SenComponents::L3LU, true);
            }
          } else if (mySDsc.pcfgMap_.count(coreID) > 0) {
            DT_CHECK(dscGlobal->dtVersion > 1 && !useDt1);
            for (auto& kv : mySDsc.pcfgMap_.at(coreID)) {
              SenComponents senCompType = kv.first;
              SenPcfg& myPcfg = mySDsc.pcfgPool_.at(kv.second);

              // Check if merging is needed: codegen or DL+DataDSC
              bool needsMerge = dataDscUnits.count(senCompType) > 0 ||
                                allUnits.count(senCompType) > 0 ||
                                progIRcodeGen == DCGProgIRGen::DCC;
              if (needsMerge && myPcfg.srcNode != nullptr) {
                mergePcfg(myPcfg, senCompType, true);
              }
            }
          }
        } else if (pcfgL3lu.size() > 0 && pcfgL3su.size() > 0) {
          for (int i = 0; i < 2; i++) {
            SenComponents senCompType =
                (i == 0) ? SenComponents::L3LU : SenComponents::L3SU;
            SenPcfg& myPcfg =
                (i == 0) ? pcfgL3lu.at(coreID) : pcfgL3su.at(coreID);
            if (myPcfg.srcNode != nullptr) {
              mergePcfg(myPcfg, senCompType, true);
            }
          }
        }
      }

      if (scheduleStep.dldsc_idx >= 0) {
        DT_CHECK(seenDLDsc == false);
        seenDLDsc = true;
      }

      // add missing unit if sync is required
      if (scheduleStep.after_sync || scheduleStep.before_sync) {
        for (auto& u : allUnits) {
          if (!unitSyncInserted.count(u)) {
            unitSyncInserted.insert(u);

            if (mySDsc.pcfg_.at(coreID).count(u) == 0) {
              mySDsc.pcfg_.at(coreID)[u];
            }
            auto& masterSenPCFG = mySDsc.pcfg_.at(coreID).at(u);
            SenPcfgSyncNode* newSyncNode = nullptr;
            if (scheduleStep.before_sync)
              newSyncNode = dcg_fe_.addSyncNode(
                  masterSenPCFG, u, coreID,
                  "before_ddsc" + std::to_string(scheduleStep.datadsc_idx) +
                      std::to_string(
                          scheduleStep.dldsc_idx));  // do this all units

            if (scheduleStep.after_sync)
              newSyncNode = dcg_fe_.addSyncNode(
                  masterSenPCFG, u, coreID,
                  "after_ddsc" + std::to_string(scheduleStep.datadsc_idx) +
                      std::to_string(
                          scheduleStep.dldsc_idx));  // do this all units

            if (hasDlDsc && lxUnits.count(u)) {
              newSyncNode->startNewProgIRBlock = true;
              if (seenDLDsc) {
                // after
                newSyncNode->isBeforeBlock = false;
              } else {
                // before
                newSyncNode->isBeforeBlock = true;
              }
            }
          }
        }
      }

      // insert marking for before code blocks (that should be insert before DVS
      // lib templates)
      if (scheduleStep.dldsc_idx >= 0) {  // current is dl-dsc
        if (mySDsc.pcfg_.count(coreID)) {
          for (auto& unitPcfg : mySDsc.pcfg_.at(coreID)) {
            auto& u = unitPcfg.first;
            auto& masterSenPCFG = unitPcfg.second;
            if (is_any_of(u, SenComponents::L3LU, SenComponents::L3SU))
              continue;  // ignore

            if (masterSenPCFG.srcNode != nullptr) {
              masterSenPCFG.srcNode->startNewProgIRBlock = true;
              masterSenPCFG.srcNode->isBeforeBlock = true;
              auto tailNode = masterSenPCFG.getPcfgGraphTailNode();
              if (tailNode != nullptr) {
                tailNode->endBeforeBlock = true;
              }
            }
          }
        }
      }
    }
  }

  // Copy back to PCFG pool
  if (dscGlobal->dtVersion > 1 && !useDt1 &&
      progIRcodeGen != DCGProgIRGen::DCC) {
    int pcfgId =
        mySDsc.pcfgPool_.empty() ? 0 : mySDsc.pcfgPool_.rbegin()->first + 1;

    // Create new PCFGs when needed
    for (auto& kv : mySDsc.pcfg_) {
      int coreID = kv.first;
      for (auto& kv2 : kv.second) {
        SenComponents comp = kv2.first;
        SenPcfg& pcfg = kv2.second;
        mySDsc.pcfgMap_[coreID][comp] = pcfgId;
        mySDsc.pcfgPool_[pcfgId] = pcfg;
        pcfgId++;
      }
    }

    // Get used PCFG ids
    std::unordered_set<int> usedPcfgIds;
    for (auto& [core, unitMap] : mySDsc.pcfgMap_) {
      for (auto& [unit, id] : unitMap) {
        usedPcfgIds.insert(id);
      }
    }

    // Erase unused PCFGs
    auto& pool = mySDsc.pcfgPool_;
    for (auto it = pool.cbegin(), nextIt = it; it != pool.cend(); it = nextIt) {
      ++nextIt;
      if (usedPcfgIds.count(it->first) == 0) {
        pool.erase(it);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 326/382   level 2   scc 156   5 body lines
// unit: e326_mergePcfgInSuperDSC
// authority: dcg/dcg_manager/dcg_manager.cpp:629
// original: void DcgManager::mergePcfgInSuperDSC(SuperDsc& mySDsc)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e326_mergePcfgInSuperDSC(SuperDsc& mySDsc)
{
  std::vector<SenPcfg> pcfgL3lu;
  std::vector<SenPcfg> pcfgL3su;
  mergePcfgInSuperDSC(mySDsc, pcfgL3lu, pcfgL3lu);
}

// ------------------------------------------------------------------------------------------------
// entry 327/382   level 2   scc 161   13 body lines
// unit: e327_printSenProgram
// authority: dcg/dcg_manager/dcg_manager.cpp:945
// original: void DcgManager::printSenProgram(SuperDsc* mySDsc, std::string fileName)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e327_printSenProgram(SuperDsc* mySDsc, std::string fileName)
{
  std::ofstream fileFP(fileName);
  if (!fileFP.is_open()) {
    DT_ERROR_FMT("Unable to open %s for writing SenProgs", fileName.c_str());
  }

  for (auto& psCore : mySDsc->progstateinfo_) {
    fileFP << "Program for coreID : " << psCore.first << "\n";
    psCore.second.print(fileFP);
    fileFP << "\n";
  }
  fileFP.close();
}

// ------------------------------------------------------------------------------------------------
// entry 348/382   level 3   scc 157   179 body lines
// unit: e348_runDcgForDataOpsDlOps
// authority: dcg/dcg_manager/dcg_manager.cpp:269
// original: void DcgManager::runDcgForDataOpsDlOps(SuperDsc& mySDsc)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e348_runDcgForDataOpsDlOps(SuperDsc& mySDsc)
{
  mySDsc.pcfg_.clear();
  if (verbose > 0)
    std::cout << "Running DCG for DL+Data Op: Node-name:" << mySDsc.name_
              << std::endl;
  // check sdsc
  DT_CHECK(senCompToISAptr != nullptr);

  // requires Input-Fetch
  bool reqInpFetch = false;
  std::pair<int, int> data_dldscIdx_inf = {-1, -1};
  bool reqDataOp = false;
  bool reqDLOp = false;
  bool hasSyncs = false;
  for (auto& mapkv : mySDsc.coreIdToDscSchedule) {
    auto coreID = mapkv.first;
    auto& dscSchedule = mapkv.second;
    DT_CHECK(dscSchedule.size() > 0);
    for (auto& scheduleStep : dscSchedule) {
      hasSyncs =
          hasSyncs || scheduleStep.after_sync || scheduleStep.before_sync;
      if (scheduleStep.datadsc_idx != -1 && scheduleStep.dldsc_idx != -1) {
        if (mySDsc.target_ == SenTargets::SENPCFG) {
          reqDLOp = true;
          reqDataOp = true;
        } else {
          if (reqInpFetch)  // we only allow 1 InpFetch op
            DT_CHECK(data_dldscIdx_inf.first == scheduleStep.datadsc_idx);
          reqInpFetch = true;
          data_dldscIdx_inf.first = scheduleStep.datadsc_idx;
          data_dldscIdx_inf.second = scheduleStep.dldsc_idx;
        }
      } else if (scheduleStep.datadsc_idx != -1) {
        reqDataOp = true;
      } else {
        // DT_CHECK(!reqDLOp);// we only allow 1 DL op
        reqDLOp = true;
        if (progIRcodeGen != DCGProgIRGen::DCC) {
          if (dscGlobal->psumRing == "sfpring" &&
              mySDsc.target_ != SenTargets::SENPCFG) {
            auto& dsc = mySDsc.dscs_.at(0);
            DT_CHECK(dsc.computeOp_.size() > 0);
            auto& myOp = dsc.computeOp_.at(0).opFuncName;
            if (myOp == OpFuncs::CONV2D_INT8_FWD ||
                myOp == OpFuncs::CONV2D_FWD ||
                myOp == OpFuncs::CONV2D_INT8_FWD_SPARSEKG3 ||
                myOp == OpFuncs::BATCHMATMUL_INT8_FWD_SPARSEKG3 ||
                myOp == OpFuncs::BATCHMATMUL_FWD ||
                myOp == OpFuncs::BATCHMATMUL_INT8_FWD) {
              bool allLxPinned = true;
              for (auto& lds : dsc.labeledDs_) {
                if (is_any_of(lds.pinnedComponent(), HBM, NO_COMPONENT)) {
                  allLxPinned = false;
                  break;
                }
              }
              reqDLOp = !allLxPinned;
              if (!reqDLOp) {
                DT_CHECK(!scheduleStep.after_sync);
                DT_CHECK(!scheduleStep.before_sync);
              }
            }
          }
        }
      }
    }
  }

  if (!(reqDLOp || reqInpFetch || reqDataOp)) {
    return;
  }
  if (!reqDataOp) DT_CHECK(reqInpFetch ^ reqDLOp);
  // ACT1: call dataOp
  if (!reqDLOp && !reqInpFetch && !hasSyncs) {
    // check if it is only in
    runDcg(mySDsc);
  } else {
    DT_CHECK(senCompToISAptr != nullptr);
    DT_CHECK(!isInpFetchNeigh);
    int c = 0;
    for (int idx = 0; idx < mySDsc.dataOpdscs_.size(); idx++) {
      auto& myDataOpDsc = mySDsc.dataOpdscs_.at(idx);
      if (idx == data_dldscIdx_inf.first && reqInpFetch)
        continue;  // this for input fetch
      dcg_fe_.computeTranferforDataOp(mySDsc, myDataOpDsc, c);
      dcg_fe_.generatePcfgIRForDataOp(mySDsc, myDataOpDsc, c);
      c++;
    }

    std::vector<SenPcfg> pcfgL3lu;  // core idx..
    std::vector<SenPcfg> pcfgL3su;
    if (reqDLOp) {
      // ACT2: generate pcfg for DL ops
      DT_CHECK(mySDsc.dscs_.size() >= 1);
      DT_CHECK(senCompToISAptr != nullptr);

      // create PCFG for l3-lu, l3-su
      if (createSenProg || progIRcodeGen == DCGProgIRGen::DCC) {
        dcg_fe_.generatePcfgIRForDLOp(mySDsc, pcfgL3lu, pcfgL3su);
      }

      if (progIRcodeGen == DCGProgIRGen::DCC) {
        // need to insert pcfgL3lu and pcfgL3su in sdsc pcfg
        for (auto& kv : mySDsc.coreIdToDsc_) {
          auto coreID = kv.first;
          for (int i = 0; i < 2; i++) {
            SenComponents senCompType =
                (i == 0) ? SenComponents::L3LU : SenComponents::L3SU;
            SenPcfg* localPcfg =
                (i == 0) ? &pcfgL3lu.at(coreID) : &pcfgL3su.at(coreID);

            int pcfgId;
            mySDsc.pcfgMap_.try_emplace(coreID);
            if (mySDsc.pcfgMap_.at(coreID).count(senCompType)) {
              pcfgId = mySDsc.pcfgMap_.at(coreID).at(senCompType);
            } else {
              pcfgId = mySDsc.pcfgPool_.empty()
                           ? 0
                           : mySDsc.pcfgPool_.rbegin()->first + 1;
              mySDsc.pcfgPool_.try_emplace(pcfgId);
              mySDsc.pcfgMap_.at(coreID)[senCompType] = pcfgId;
            }

            auto& sdscPcfg = mySDsc.pcfgPool_.at(pcfgId);
            sdscPcfg.mergeSenPcfg(sdscPcfg, *localPcfg);  // copy
          }
        }
      }
    } else if (reqInpFetch) {
      DT_CHECK(reqInpFetch);
      DT_CHECK(mySDsc.dscs_.size() > data_dldscIdx_inf.second);
      myIFNInfo.dataDsc_Idx = data_dldscIdx_inf.first;
      DataOpDsc& ddsc = *dcg_fe_.generatePcfgIRForDataOpInpFetch(
          mySDsc, nullptr, data_dldscIdx_inf.first);
    } else {
      DT_CHECK(hasSyncs);
    }
    // create merged pcfg for
    // copy to sdsc pcfg
    if (createSenProg) {
      DT_CHECK_MSG(mySDsc.sdscFoldProps_.empty(),
                   "Codegen for Folded Super-DSC is not supported");
      DT_CHECK(progIRcodeGen == DCGProgIRGen::DCG);
      mergePcfgInSuperDSC(mySDsc, pcfgL3lu, pcfgL3su);

      // ACT3: create sen programs
      if (reqInpFetch) isInpFetchNeigh = true;
      dcg_be_.fillAndCreateSenProgInfoUsingSuperDSC(mySDsc);
    }

    if (progIRcodeGen != DCGProgIRGen::DCC) {
      removeextraPTrows(mySDsc);

      // Final act merge pcfg for senpcfg simulation
      if (reqDLOp) DT_CHECK(!mySDsc.dscs_.empty());

      // no need to merge if backend is dcc
      bool hasDlPcfg = false;
      if (mySDsc.dscs_.size()) {
        auto& dsc0 = mySDsc.dscs_.at(0);
        hasDlPcfg = !dsc0.pcfg_.empty() || !mySDsc.pcfgPool_.empty();
      }
      if (reqDLOp && hasDlPcfg) {
        // without --genpcfg, no need to merge pcfg in SDSC..
        mergePcfgInSuperDSC(mySDsc, pcfgL3lu, pcfgL3su, true);
      } else {
        mergePcfgInSuperDSC(mySDsc, pcfgL3lu, pcfgL3su);
      }
    }
  }

  // Keep only pool
  if (dscGlobal->dtVersion > 1 && progIRcodeGen != DCGProgIRGen::DCC) {
    mySDsc.pcfg_.clear();
    for (auto& ddsc : mySDsc.dataOpdscs_) {
      ddsc.pcfg_.clear();
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 349/382   level 3   scc 160   46 body lines
// unit: e349_convertToProgIRDataOp
// authority: dcg/dcg_manager/dcg_manager.cpp:898
// original: void DcgManager::convertToProgIRDataOp(SuperDsc& mySDsc)
// class: DcgManager
// rust home: crates/compiler/deeptools/src/schedule/dcg/manager.rs
// ------------------------------------------------------------------------------------------------
void e349_convertToProgIRDataOp(SuperDsc& mySDsc)
{
  if (verbose > 0)
    std::cout << "Creating SenPrograms from input sdsc for dataOps..."
              << std::endl;

  DT_CHECK(senCompToISAptr != nullptr);
  DT_CHECK(!isInpFetchNeigh);
  DT_CHECK(createSenProg);

  if (mySDsc.pcfg_.empty()) {
    DT_CHECK(!mySDsc.dataOpdscs_.empty());
  }

  // determine if we need to stitch dataDSCs
  if (mySDsc.dataOpdscs_.size() > 1 && mySDsc.pcfg_.empty())
    mergePcfgInSuperDSC(mySDsc);

  // SenPrograms
  if (mySDsc.dataOpdscs_.size() <= 1 && mySDsc.pcfg_.empty()) {
    if (verbose > 0) {
      std::cout << "Creating SenPrograms..." << std::endl;
    }

    DT_CHECK(mySDsc.dataOpdscs_.size());
    DataOpDsc& myDataOpDsc = mySDsc.dataOpdscs_[0];

    if (myDataOpDsc.op->name == OpFuncs::GatherOpHBM) {
      DT_CHECK((*senCompToISAptr).count(SenComponents::L3LU));
      DT_CHECK((*senCompToISAptr).at(SenComponents::L3LU).sysDef.coreArch >=
               IsaCoreGen::MPW4_ISA);
    }
    auto coreIDTostatus =
        dcg_be_.createSenProgramSTCDPOp(&myDataOpDsc, &mySDsc);
    if (enable_prog_verification_) {
      for (auto& kv : coreIDTostatus) {
        auto status = kv.second;
        DT_ERROR("Program verification failed for core " +
                 std::to_string(kv.first) + " node " + mySDsc.name_ +
                 "\nError message: " + status.second);
      }
    }
  } else {
    // senprog gen routine using superDSC's pcfg
    dcg_be_.fillAndCreateSenProgInfoUsingSuperDSC(mySDsc);
  }
}

