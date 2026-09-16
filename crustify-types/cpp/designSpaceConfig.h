/************************************************************
 * IBM Confidential
 * (C) Copyright IBM Corp. 2018, 2025
 ************************************************************/

/*
 * Description:
 *
 */

#ifndef DESIGN_SPACE_CONFIG_
#define DESIGN_SPACE_CONFIG_

#include <stdint.h>
#include <stdio.h>
#include <util/variabledefinition/VariableDefinition.h>
#include <util/sendefs/numeric_convert.h>
#include <util/sendefs/sendefs.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <list>
#include <map>
#include <sstream>
#include <stdexcept>
#include <thread>
#include <tuple>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#ifdef SEN
#include <json11/json11.hpp>
#else
#include <external/json11/json11.hpp>
#endif

#include "dims.h"
#include "dsc2.h"
#include "dscdefn.h"
#include "pcfg.h"
#include "sys-arch-spec/dscglobal/dscglobal.h"
#include "sys-arch-spec/sysdef.h"


class DesignSpaceConfig {
 public:
  // Design Space Configuration: A language to define how each op
  // needs to be executed in SENTIENT
  // --> Specifically, we can (succintly) express in DSC
  // the sequence in which data corresponding to the different data-structures
  // need to be moved across the memory heirarchy levels
  // through the different links in the sentient architecture
  // until it reaches the processing elements (either SFP/PE/PT)

  // A description of the DSC language can be found in:

  // ** DSC (including the name) is under development **
  // The current definition of DSC has the following restrictions apply:
  // (i) Every core does identical amount of work
  // (ii) Only one compute function in the op -- can be either SFP/PE/PT
  // (iii) A single nested sequence of loops -- no hierarchical nesting
  // (iv) The inputs to the "op" should be dervied from a single primary op
  // (v) All computations are fp16

  // To be filled by DSM (graph modifier)
  std::string name_;
  int numCoresUsed_;
  int numCoreletsUsed_;
  std::vector<int> coreIdsUsed_;
  std::map<PrimaryDimTypes, std::vector<VariableSymbol>>
      dimToSymbolMapping_;  // single value for pure symbolic and pivot dims,
                            // multiple entries (max-pivot) for irregular dims

  // To be filled by DGP
  DataStructDims N_;
  DataStructDims unpadN_;
  DataStructDims dscN_;  // total parameters performed by this DSC

  // filled by DSM/DM/DSI
  std::vector<LabeledDsInfo> labeledDs_;

  // filled by DSM/DM
  std::vector<ComputeOpInfo> computeOp_;
  std::map<int, dsc2::ConstantInfo> constantInfo_;

  // To be filled by DSM
  std::map<DsTypes, PrimaryDsInfo> primaryDsInfo_;
  PrimaryDsRelationInfo pdsRelation_;
  DataStructDims ChipD_;
  DataStructDims ChipletD_;
  DataStructDims CoreD_;
  DataStructDims CoreletD_;
  std::map<PrimaryDimTypes, std::vector<std::pair<int, int>>>
      coordinateMasking_;    // pair<unmasked, masked>
  int maskingConstId_ = -1;  // assuming all tensors use same masking constant

  // To be filled by DM
  int numCoreletsUsed_DSC2_ = -1;
  std::map<int, dsc2::DataStage> dataStageParam_;
  DataStructDims B_;
  DataStructDims T_;
  DataStructDims Tel_;
  DataStructDims P_;
  DataStructDims Pel_;
  std::vector<DataStructDims> sc_;
  std::vector<LoopNames> loopOrder_;
  std::map<LoopNames, LoopProperties> loopProperties_;
  std::map<LoopNames, std::vector<AuxLoopSetInfo>> auxLoopOrder_;
  dsc2::ScheduleTree scheduleTree_;
  std::set<int> gtrIdsUsed_;
  bool l0TetheredMode_ = false;

  // PCFG for each core
  std::vector<SenPcfg> pcfg_;
  SenTargets target_ = SenTargets::UNDEFINED;

  // Program frame for DT2SL
  // To be filled by different DCS tools based on target_:
  // SENTIENT, SENULATOR, SYSTEMC --> Codegen
  // SENPCFG --> PCFG transform function
  // SENTF --> DCS_SENTF
  // HOST --> DSM
  struct ProgramFrame {
    std::shared_ptr<void> ptr_;
    size_t size_ = 0;
  };
  std::map<SenTargets, ProgramFrame> prog_frame_ptr_;

  DesignSpaceConfig();
  bool isDSC2() const;
  void copyDsc(const DesignSpaceConfig&, const SuperDsc* sdsc);
  void customizeFoldFromRef(const DesignSpaceConfig& iDsc,
                            std::deque<int64_t> coord,
                            std::deque<const FoldDimProp*> ccFoldProp,
                            std::deque<const FoldDimProp*> sdscFoldProp,
                            std::string name, int level);
  void printPrimaryDsInfo(std::ostream& os, const PrimaryDsInfo&,
                          std::string) const;
  void printComputeOpInfo(std::ostream& os, const ComputeOpInfo&,
                          std::string) const;
  void printMemOrg(std::ostream& os, const MemOrg&, std::string) const;
  void printDt(std::ostream& os, const DtInfo&, std::string) const;
  void printCoreDs(std::ostream& os, const CoreDsInfo&, std::string) const;
  void printLabeledDs(std::ostream& os, const LabeledDsInfo&,
                      std::string) const;
  void printDsc(bool verbose = false, std::ostream& out = std::cout) const;
  void createPcfgs(std::ostream& os);

  void transformDscToPcfg(int cidx = 0);
  void buildPcfgLoop(SenPcfgNode**, SenPcfgNode**, std::string, AuxLoopSetInfo*,
                     int, int);
  void buildPcfgCond(SenPcfgNode**, SenPcfgNode**, std::string, AuxLoopSetInfo*,
                     int, int);
  int getPcfgLoopCount(std::string, AuxLoopSetInfo*);
  void populatePcfgDynLoop(SenPcfgMvloopNode*);
  void pcfgInsertAbove(SenPcfgNode*, SenPcfgNode*);
  void pcfgInsertBelow(SenPcfgNode*, SenPcfgNode*);
  void fillPcfgProgramFrame(const DesignSpaceConfig& refDsc);
  SenPcfgNode* createPcfgNode(SenPcfgNode::Type, int c = 0);
  bool isPartStickDt(DtInfo*);
  void populatePcfgPartStickDtNode(SenPcfgDtNode*);
  void populatePcfgDtNode(SenPcfgDtNode*, int);
  void pcfgCheckAndAddSamvInfo(SenPcfgDtNode*);
  void populatePcfgSisterLoopDtNode(SenPcfgDtNode*, int);
  void populatePcfgXCoreDtNode(SenPcfgXCoreDtNode*);
  void populatePcfgSfpComputeNode(SenPcfgSfpComputeNode*);
  void populatePcfgPtComputeNode(SenPcfgPtComputeNode*);
  void populatePcfgRingDtNode(SenPcfgRingDtNode*, int);
  void populatePcfgZeroDtNode(SenPcfgMemPadConstNode*, DtInfo*);
  SenPcfgNode* addPcfgCondPadding(LabeledDsInfo*, DtInfo*, SenPcfgNode*,
                                  int c = 0, int cidx = 0);
  std::vector<std::pair<void*, SenPcfgNode::Type>> sortDtCompute(int);
  SenPcfgMemPadConstNode* createPcfgIjVerPadNode(std::string, LabeledDsInfo*,
                                                 DtInfo*, bool, bool, int c = 0,
                                                 int cidx = 0);
  std::string exportJsonStr(int verbosity = 1, const std::string& ps = "",
                            bool addNewLine = true) const;
  void exportJson(const std::string& FileName, int verbosity = 1) const;
  void exportJson(std::ostream& json, int verbosity = 1,
                  const std::string& ps = "", bool addNewLine = true) const;
  bool importJsonObj(const json11::Json&, const SuperDsc* sdsc,
                     int skipModule = 0);
  bool importJsonStr(const std::string&, const SuperDsc* sdsc,
                     int skipModule = 0);
  bool importJson(const std::string& fileName, const SuperDsc* sdsc,
                  int skipModule = 0);
  bool importJson(std::istream& inputStream, const SuperDsc* sdsc,
                  int skipModule = 0);
  bool verify(bool verbose = false, int level = 9);
  bool checkAssumption(bool verbose = false, int level = 9);
  bool checkDataStructDims(DataStructDims& d1, DataStructDims& d2,
                           std::string& msg);
  bool checkLoopStage(LoopNames LName, int LoopStage);
  std::string getDimPrefix4LxTransfer_lDs(LabeledDsInfo, PrimaryDimTypes);
  std::string getDimPrefix4LxTransfer(std::size_t, PrimaryDimTypes);

  // Pure PrimaryDsInfo stick queries. Relocated here from dsi (they operate
  // only on PrimaryDsInfo, a dsc type) so callers no longer need to depend on
  // dsi just for these.
  static double get_stick(PrimaryDsInfo pdi, PrimaryDimTypes stick_dim);
  static int get_stick_repl(PrimaryDsInfo pdi, PrimaryDimTypes stick_dim);
  static double get_stick_srpdt(PrimaryDsInfo pdi, PrimaryDimTypes stick_dim);
  std::map<PrimaryDimTypes, double> getDtSize(DtInfo*, LabeledDsInfo*,
                                              std::string format = "element",
                                              bool addZeroPad = false);
  std::map<PrimaryDimTypes, double> getAuxLoopDtSize(
      DtInfo*, LabeledDsInfo*, std::string format = "element");
  std::vector<double> getDsDimShapes(LabeledDsInfo*, bool, bool);
  std::vector<std::string> getDimOrderStr(LabeledDsInfo*, bool, bool);
  AuxLoopSetInfo* getAuxLoopSet(std::string);
  int getLoopCount(LoopNames);
  int getDimIndexInLayoutOrder(DsTypes dstype, PrimaryDimTypes dim) const;
  int64_t getAddress(LabeledDsInfo& lds, SenComponents comp, int coreid) const;

  double getInpRowInHBM(std::string) const;
  double getInpColInHBM(std::string) const;
  double getInpInHBM(std::string) const;

  int getBlockTransferSize(const dsc2::TransferNode& ti, SenComponents comp,
                           int clId = -1, bool epilogue = false,
                           bool sizeInNumberOfLoads = false) const;
  std::unordered_map<PrimaryDimTypes, int> getBlockTransferSizePerDim(
      const dsc2::TransferNode& ti, SenComponents comp, int clId = -1,
      bool epilogue = false, bool sizeInNumberOfLoads = false,
      bool doNotRound = false) const;
  std::unordered_map<PrimaryDimTypes, int> getImplicitSyncTileSizePerDim(
      const dsc2::SyncNode& sn, int clId = -1, bool epilogue = false) const;
  void setRelevantCompCoreCl();
  void finalizeScheduleTree(SuperDsc& sdsc, const SenSystemDef& sysDef);
  std::vector<PrimaryDimTypes> getLayoutDims(int ldsIdx) const;
  std::set<PrimaryDimTypes> getLayoutDimSet(DsTypes dsType) const;
  inline std::set<PrimaryDimTypes> getLayoutDimSet(int ldsIdx) const {
    return getLayoutDimSet(labeledDs_.at(ldsIdx).dsType_);
  }
  inline std::vector<PrimaryDimTypes> getStickDims(DsTypes dsType) const {
    return primaryDsInfo_.at(dsType).stickDimOrder_;
  };
  inline std::vector<PrimaryDimTypes> getStickDims(int ldsIdx) const {
    return getStickDims(labeledDs_.at(ldsIdx).dsType_);
  }
  std::set<PrimaryDimTypes> getStickDimSet(DsTypes dsType) const;
  inline std::set<PrimaryDimTypes> getStickDimSet(int ldsIdx) const {
    return getStickDimSet(labeledDs_.at(ldsIdx).dsType_);
  }

  std::vector<PrimaryDimTypes> getNonBroadcastLdsDims(int ldsIdx) const;
  std::set<PrimaryDimTypes> getNonBroadcastLdsDimSet(int ldsIdx) const;
  bool hasBroadcastLdsDims(int ldsIdx) const;

  std::vector<std::pair<PrimaryDimTypes, int>> getStickSizes(
      DsTypes dsType, bool stickSliceOnly = false,
      bool stickWithoutSlices = false, bool l0SliceOnly = false,
      int numL0Slices = -1) const;
  std::unordered_map<PrimaryDimTypes, int> getCumulativeStickSizes(
      DsTypes dsType, bool stickSliceOnly = false,
      bool stickWithoutSlices = false, bool l0SliceOnly = false,
      int numL0Slices = -1) const;
  dsc2::DataStage getSizeDataStageForNode(
      const dsc2::ScheduleNode* node, const dsc2::AllocateNode* alloc) const;
  dsc2::DataStage getSizeDataStageForNode(const dsc2::ScheduleNode* node,
                                          const int ldsIdx,
                                          const PaddingFormType& padding) const;

  std::vector<std::pair<PrimaryDimTypes, int>>
  getBufferCapacityForNodePerDimCustomLocation(
      const dsc2::ScheduleNode* nodeForLocation, const dsc2::ScheduleNode* node,
      int ldsIdx, SenComponents comp, int corelet, int row,
      const bool doNotRound = false, const bool includeGaps = true,
      bool allowSymbolicVolumeLimit = false) const;
  std::vector<std::pair<PrimaryDimTypes, int>> getBufferCapacityForNodePerDim(
      const dsc2::ScheduleNode* node, int ldsIdx, SenComponents comp,
      int corelet, int row, const bool doNotRound = false,
      const bool includeGaps = true,
      bool allowSymbolicVolumeLimit = false) const;
  int64_t getBufferCapacityForNode(const dsc2::ScheduleNode* node, int ldsIdx,
                                   SenComponents comp, int corelet, int row,
                                   const uint64_t bytesPerStick = 0,
                                   const bool forceEvenNumSticks = false,
                                   const bool doNotRound = false,
                                   const bool includeGaps = true) const;

  bool allocationExists(const dsc2::DataInfo& di,
                        const SenComponents storage) const;
  const dsc2::AllocateNode* getAllocation(const dsc2::DataInfo& di,
                                          const SenComponents storage,
                                          bool allowMissingAlloc = false) const;

  dsc2::AllocateNode* getMutableAllocation(const dsc2::DataInfo& di,
                                           const SenComponents storage,
                                           bool allowMissingAlloc = false);

  // Removes usage-link between an allocateNode and a user node. In case the
  // allocateNode no longer has any user, the allocateNode is removed from the
  // schedule tree. Returns:
  //   true  - if the allocateNode is removed from the schedule tree,
  //   false - otherwise.
  bool reduceUsersOrDeleteAllocation(const dsc2::DataInfo& di,
                                     const SenComponents storage,
                                     const dsc2::ScheduleNode* userNode,
                                     bool canDelete = true);
  void cleanupAllocationForCompute(dsc2::ComputeNode* computeNode);
  void cleanupAllocationForTransfer(dsc2::TransferNode* transferNode);
  void cleanupAllocation(dsc2::ScheduleNode* node);

  static void getInnermostCommonAncestor(
      const dsc2::ScheduleNode* firstNode, const dsc2::ScheduleNode* secondNode,
      std::vector<const dsc2::ScheduleNode*>& pathToFirstNode,
      std::vector<const dsc2::ScheduleNode*>& pathToSecondNode);

  // Utility..
  DataStructDims& getDsdFromStr(std::string dsdstr) {
    std::transform(dsdstr.begin(), dsdstr.end(), dsdstr.begin(), ::tolower);
    if (dsdstr == "n") {
      return N_;
    } else if (dsdstr == "d") {
      return CoreD_;
    } else if (dsdstr == "b") {
      return B_;
    } else if (dsdstr == "t") {
      return T_;
    } else if (dsdstr == "p") {
      return P_;
    } else if (dsdstr == "coreletd") {
      return CoreletD_;
    } else if (dsdstr == "tel") {
      return Tel_;
    } else if (dsdstr == "pel") {
      return Pel_;
    } else if (dsdstr == "unpadn") {
      return unpadN_;
    } else if (dsdstr == "chipd") {
      return ChipD_;
    } else if (dsdstr == "chipletd") {
      return ChipletD_;
    } else if (dsdstr == "dscn") {
      return dscN_;
    } else {
      DT_ERROR("Unknow string input to getDsdFromStr()");
    }
  }
  static const std::map<LoopNames, std::string> loopNameToString;
  static const std::map<std::string, LoopNames> stringToLoopName;
  // Following DM convention..
  static const std::map<std::string, LoopNames> stringToLoopNameDm;
  static const std::map<DsTypes, std::string> dsTypeToString;
  static const std::map<std::string, DsTypes> stringToDsType;
  static const std::map<DtType, std::string> dtTypeToString;
  static const std::map<std::string, DtType> stringToDtType;
  static const std::map<ExternalSenOps, std::string> exSenOpsToString;
  static const std::map<std::string, ExternalSenOps> stringToExSenOps;
  static const std::map<OpFuncs, std::pair<int, int>> opFuncsToInOuts;
  std::map<std::string, double*> paramNameToVal = {
      {"nin", &N_.in_},
      {"nout", &N_.out_},
      {"nmb", &N_.mb_},
      {"ni", &N_.i_},
      {"nj", &N_.j_},
      {"nij", &N_.ij_},
      {"nki", &N_.ki_},
      {"nkj", &N_.kj_},
      {"nkij", &N_.kij_},
      {"nx", &N_.x_},
      {"ny", &N_.y_},
      {"nr", &N_.r_},
      {"nc", &N_.c_},
      {"nrc", &N_.rc_},
      {"nsi", &N_.si_},
      {"nsj", &N_.sj_},
      {"nsij", &N_.sij_},
      {"nzi", &N_.zi_},
      {"nzj", &N_.zj_},
      {"nzij", &N_.zij_},

      {"unpadnin", &unpadN_.in_},
      {"unpadnout", &unpadN_.out_},
      {"unpadnmb", &unpadN_.mb_},
      {"unpadni", &unpadN_.i_},
      {"unpadnj", &unpadN_.j_},
      {"unpadnij", &unpadN_.ij_},
      {"unpadnki", &unpadN_.ki_},
      {"unpadnkj", &unpadN_.kj_},
      {"unpadnkij", &unpadN_.kij_},
      {"unpadnx", &unpadN_.x_},
      {"unpadny", &unpadN_.y_},
      {"unpadnr", &unpadN_.r_},
      {"unpadnc", &unpadN_.c_},
      {"unpadnrc", &unpadN_.rc_},
      {"unpadnsi", &unpadN_.si_},
      {"unpadnsj", &unpadN_.sj_},
      {"unpadnsij", &unpadN_.sij_},
      {"unpadnzi", &unpadN_.zi_},
      {"unpadnzj", &unpadN_.zj_},
      {"unpadnzij", &unpadN_.zij_},

      {"dscnin", &dscN_.in_},
      {"dscnout", &dscN_.out_},
      {"dscnmb", &dscN_.mb_},
      {"dscni", &dscN_.i_},
      {"dscnj", &dscN_.j_},
      {"dscnij", &dscN_.ij_},
      {"dscnki", &dscN_.ki_},
      {"dscnkj", &dscN_.kj_},
      {"dscnkij", &dscN_.kij_},
      {"dscnx", &dscN_.x_},
      {"dscny", &dscN_.y_},
      {"dscnr", &dscN_.r_},
      {"dscnc", &dscN_.c_},
      {"dscnrc", &dscN_.rc_},
      {"dscnsi", &dscN_.si_},
      {"dscnsj", &dscN_.sj_},
      {"dscnsij", &dscN_.sij_},
      {"dscnzi", &dscN_.zi_},
      {"dscnzj", &dscN_.zj_},
      {"dscnzij", &dscN_.zij_},

      {"chipdin", &ChipD_.in_},
      {"chipdout", &ChipD_.out_},
      {"chipdmb", &ChipD_.mb_},
      {"chipdi", &ChipD_.i_},
      {"chipdj", &ChipD_.j_},
      {"chipdij", &ChipD_.ij_},
      {"chipdki", &ChipD_.ki_},
      {"chipdkj", &ChipD_.kj_},
      {"chipdkij", &ChipD_.kij_},
      {"chipdx", &ChipD_.x_},
      {"chipdy", &ChipD_.y_},
      {"chipdr", &ChipD_.r_},
      {"chipdc", &ChipD_.c_},
      {"chipdrc", &ChipD_.rc_},
      {"chipdsi", &ChipD_.si_},
      {"chipdsj", &ChipD_.sj_},
      {"chipdsij", &ChipD_.sij_},
      {"chipdzi", &ChipD_.zi_},
      {"chipdzj", &ChipD_.zj_},
      {"chipdzij", &ChipD_.zij_},

      {"chipletdin", &ChipletD_.in_},
      {"chipletdout", &ChipletD_.out_},
      {"chipletdmb", &ChipletD_.mb_},
      {"chipletdi", &ChipletD_.i_},
      {"chipletdj", &ChipletD_.j_},
      {"chipletdij", &ChipletD_.ij_},
      {"chipletdki", &ChipletD_.ki_},
      {"chipletdkj", &ChipletD_.kj_},
      {"chipletdkij", &ChipletD_.kij_},
      {"chipletdx", &ChipletD_.x_},
      {"chipletdy", &ChipletD_.y_},
      {"chipletdr", &ChipletD_.r_},
      {"chipletdc", &ChipletD_.c_},
      {"chipletdrc", &ChipletD_.rc_},
      {"chipletdsi", &ChipletD_.si_},
      {"chipletdsj", &ChipletD_.sj_},
      {"chipletdsij", &ChipletD_.sij_},
      {"chipletdzi", &ChipletD_.zi_},
      {"chipletdzj", &ChipletD_.zj_},
      {"chipletdzij", &ChipletD_.zij_},

      {"din", &CoreD_.in_},
      {"dout", &CoreD_.out_},
      {"dmb", &CoreD_.mb_},
      {"di", &CoreD_.i_},
      {"dj", &CoreD_.j_},
      {"dij", &CoreD_.ij_},
      {"dki", &CoreD_.ki_},
      {"dkj", &CoreD_.kj_},
      {"dkij", &CoreD_.kij_},
      {"dx", &CoreD_.x_},
      {"dy", &CoreD_.y_},
      {"dr", &CoreD_.r_},
      {"dc", &CoreD_.c_},
      {"drc", &CoreD_.rc_},
      {"dsi", &CoreD_.si_},
      {"dsj", &CoreD_.sj_},
      {"dsij", &CoreD_.sij_},
      {"dzi", &CoreD_.zi_},
      {"dzj", &CoreD_.zj_},
      {"dzij", &CoreD_.zij_},

      {"coreletdin", &CoreletD_.in_},
      {"coreletdout", &CoreletD_.out_},
      {"coreletdmb", &CoreletD_.mb_},
      {"coreletdi", &CoreletD_.i_},
      {"coreletdj", &CoreletD_.j_},
      {"coreletdij", &CoreletD_.ij_},
      {"coreletdki", &CoreletD_.ki_},
      {"coreletdkj", &CoreletD_.kj_},
      {"coreletdkij", &CoreletD_.kij_},
      {"coreletdx", &CoreletD_.x_},
      {"coreletdy", &CoreletD_.y_},
      {"coreletdr", &CoreletD_.r_},
      {"coreletdc", &CoreletD_.c_},
      {"coreletdrc", &CoreletD_.rc_},
      {"coreletdsi", &CoreletD_.si_},
      {"coreletdsj", &CoreletD_.sj_},
      {"coreletdsij", &CoreletD_.sij_},
      {"coreletdzi", &CoreletD_.zi_},
      {"coreletdzj", &CoreletD_.zj_},
      {"coreletdzij", &CoreletD_.zij_},

      {"bin", &B_.in_},
      {"bout", &B_.out_},
      {"bmb", &B_.mb_},
      {"bi", &B_.i_},
      {"bj", &B_.j_},
      {"bij", &B_.ij_},
      {"bki", &B_.ki_},
      {"bkj", &B_.kj_},
      {"bkij", &B_.kij_},
      {"bx", &B_.x_},
      {"by", &B_.y_},
      {"br", &B_.r_},
      {"bc", &B_.c_},
      {"brc", &B_.rc_},
      {"bsi", &B_.si_},
      {"bsj", &B_.sj_},
      {"bsij", &B_.sij_},
      {"bzi", &B_.zi_},
      {"bzj", &B_.zj_},
      {"bzij", &B_.zij_},

      {"tin", &T_.in_},
      {"tout", &T_.out_},
      {"tmb", &T_.mb_},
      {"ti", &T_.i_},
      {"tj", &T_.j_},
      {"tij", &T_.ij_},
      {"tki", &T_.ki_},
      {"tkj", &T_.kj_},
      {"tkij", &T_.kij_},
      {"tx", &T_.x_},
      {"ty", &T_.y_},
      {"tr", &T_.r_},
      {"tc", &T_.c_},
      {"trc", &T_.rc_},
      {"tsi", &T_.si_},
      {"tsj", &T_.sj_},
      {"tsij", &T_.sij_},
      {"tzi", &T_.zi_},
      {"tzj", &T_.zj_},
      {"tzij", &T_.zij_},

      {"telin", &Tel_.in_},
      {"telout", &Tel_.out_},
      {"telmb", &Tel_.mb_},
      {"teli", &Tel_.i_},
      {"telj", &Tel_.j_},
      {"telij", &Tel_.ij_},
      {"telki", &Tel_.ki_},
      {"telkj", &Tel_.kj_},
      {"telkij", &Tel_.kij_},
      {"telx", &Tel_.x_},
      {"tely", &Tel_.y_},
      {"telr", &Tel_.r_},
      {"telc", &Tel_.c_},
      {"telrc", &Tel_.rc_},
      {"telsi", &Tel_.si_},
      {"telsj", &Tel_.sj_},
      {"telsij", &Tel_.sij_},
      {"telzi", &Tel_.zi_},
      {"telzj", &Tel_.zj_},
      {"telzij", &Tel_.zij_},

      {"pelin", &Pel_.in_},
      {"pelout", &Pel_.out_},
      {"pelmb", &Pel_.mb_},
      {"peli", &Pel_.i_},
      {"pelj", &Pel_.j_},
      {"pelij", &Pel_.ij_},
      {"pelki", &Pel_.ki_},
      {"pelkj", &Pel_.kj_},
      {"pelkij", &Pel_.kij_},
      {"pelx", &Pel_.x_},
      {"pely", &Pel_.y_},
      {"pelr", &Pel_.r_},
      {"pelc", &Pel_.c_},
      {"pelrc", &Pel_.rc_},
      {"pelsi", &Pel_.si_},
      {"pelsj", &Pel_.sj_},
      {"pelsij", &Pel_.sij_},
      {"pelzi", &Pel_.zi_},
      {"pelzj", &Pel_.zj_},
      {"pelzij", &Pel_.zij_},

      {"pin", &P_.in_},
      {"pout", &P_.out_},
      {"pmb", &P_.mb_},
      {"pi", &P_.i_},
      {"pj", &P_.j_},
      {"pij", &P_.ij_},
      {"pki", &P_.ki_},
      {"pkj", &P_.kj_},
      {"pkij", &P_.kij_},
      {"px", &P_.x_},
      {"py", &P_.y_},
      {"pr", &P_.r_},
      {"pc", &P_.c_},
      {"prc", &P_.rc_},
      {"psi", &P_.si_},
      {"psj", &P_.sj_},
      {"psij", &P_.sij_},
      {"pzi", &P_.zi_},
      {"pzj", &P_.zj_},
      {"pzij", &P_.zij_}};
  /**
   * TODO: add all of fields.
   * current fields only for getInpInHBM()
   */
  void updateParamNameToVal() {
    paramNameToVal["nout"] = &N_.out_;
    paramNameToVal["nmb"] = &N_.mb_;
    paramNameToVal["nij"] = &N_.ij_;
    paramNameToVal["ni"] = &N_.i_;
    paramNameToVal["nj"] = &N_.j_;
    paramNameToVal["nx"] = &N_.x_;
    paramNameToVal["ny"] = &N_.y_;
    paramNameToVal["nrc"] = &N_.rc_;
    paramNameToVal["nr"] = &N_.r_;
    paramNameToVal["nc"] = &N_.c_;
    paramNameToVal["nzi"] = &N_.zi_;
    paramNameToVal["nzj"] = &N_.zj_;

    paramNameToVal["dout"] = &CoreD_.out_;
    paramNameToVal["dmb"] = &CoreD_.mb_;
    paramNameToVal["dij"] = &CoreD_.ij_;
    paramNameToVal["di"] = &CoreD_.i_;
    paramNameToVal["dj"] = &CoreD_.j_;
    paramNameToVal["dx"] = &CoreD_.x_;
    paramNameToVal["dy"] = &CoreD_.y_;
    paramNameToVal["drc"] = &CoreD_.rc_;
    paramNameToVal["dr"] = &CoreD_.r_;
    paramNameToVal["dc"] = &CoreD_.c_;

    paramNameToVal["coreletdout"] = &CoreletD_.out_;
    paramNameToVal["coreletdmb"] = &CoreletD_.mb_;
    paramNameToVal["coreletdij"] = &CoreletD_.ij_;
    paramNameToVal["coreletdi"] = &CoreletD_.i_;
    paramNameToVal["coreletdj"] = &CoreletD_.j_;
    paramNameToVal["coreletdx"] = &CoreletD_.x_;
    paramNameToVal["coreletdy"] = &CoreletD_.y_;
    paramNameToVal["coreletdrc"] = &CoreletD_.rc_;
    paramNameToVal["coreletdr"] = &CoreletD_.r_;
    paramNameToVal["coreletdc"] = &CoreletD_.c_;

    paramNameToVal["bout"] = &B_.out_;
    paramNameToVal["bmb"] = &B_.mb_;
    paramNameToVal["bij"] = &B_.ij_;
    paramNameToVal["bi"] = &B_.i_;
    paramNameToVal["bj"] = &B_.j_;
    paramNameToVal["bx"] = &B_.x_;
    paramNameToVal["by"] = &B_.y_;
    paramNameToVal["brc"] = &B_.rc_;
    paramNameToVal["br"] = &B_.r_;
    paramNameToVal["bc"] = &B_.c_;

    paramNameToVal["tout"] = &T_.out_;
    paramNameToVal["tmb"] = &T_.mb_;
    paramNameToVal["tij"] = &T_.ij_;
    paramNameToVal["ti"] = &T_.i_;
    paramNameToVal["tj"] = &T_.j_;
    paramNameToVal["tx"] = &T_.x_;
    paramNameToVal["ty"] = &T_.y_;
    paramNameToVal["trc"] = &T_.rc_;
    paramNameToVal["tr"] = &T_.r_;
    paramNameToVal["tc"] = &T_.c_;

    paramNameToVal["telr"] = &Tel_.r_;
    paramNameToVal["telc"] = &Tel_.c_;

    paramNameToVal["pout"] = &P_.out_;
    paramNameToVal["pmb"] = &P_.mb_;
    paramNameToVal["pij"] = &P_.ij_;
    paramNameToVal["pi"] = &P_.i_;
    paramNameToVal["pj"] = &P_.j_;
    paramNameToVal["px"] = &P_.x_;
    paramNameToVal["py"] = &P_.y_;
    paramNameToVal["prc"] = &P_.rc_;
    paramNameToVal["pr"] = &P_.r_;
    paramNameToVal["pc"] = &P_.c_;

    paramNameToVal["pelr"] = &Pel_.r_;
    paramNameToVal["pelc"] = &Pel_.c_;
  }

 private:
  bool importJsonObjDSC2Fields(const json11::Json&, const SuperDsc* sdsc,
                               int skipModule = 0);
  void exportJsonDSC2Fields(std::ostream& json, int verbosity = 1,
                            std::string ps = "", bool addNewLine = true) const;
  std::unordered_map<PrimaryDimTypes, int>
  getBlockTransferSizePerDimCustomLocation(
      const dsc2::ScheduleNode* nodeForLocation, const dsc2::TransferNode& ti,
      SenComponents comp, int clId = -1, bool epilogue = false,
      bool sizeInNumberOfLoads = false, bool doNotRound = false) const;
};

std::ostream& operator<<(std::ostream& os,
                         const DesignSpaceConfig::ProgramFrame& pf);

#endif
