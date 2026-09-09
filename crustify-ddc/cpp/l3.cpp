// ================================================================================================
// DDC / L3-SCHEDULER CAMPAIGN - consolidated translation unit: l3
//
// STAGE 1 of runDdc: L3DlOpsScheduler(dscGlobal, memTrackers, {executionStep}, verbose).run(sdsc)
//   dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29-41 is the call site.
//
// 144 of the campaign's 382 function bodies, from dcg/dcg_fe/scheduler/, extracted VERBATIM
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
// entry 001/382   level 0   scc 0   4 body lines
// unit: e001_isSameDscGroup
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:60
// original: [[maybe_unused]] static bool isSameDscGroup(SuperDsc &mySDsc)
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
[[maybe_unused]] static bool e001_isSameDscGroup(SuperDsc &mySDsc)
{
  DT_CHECK_MSG(mySDsc.dscs_.size() >= 1, "Expect at least one DSC.");
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 002/382   level 0   scc 1   6 body lines
// unit: e002_isLabeledDsDimensionBroadcast
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:65
// original: static inline bool isLabeledDsDimensionBroadcast(const DesignSpaceConfig &dsc, const LabeledDsInfo &lds, PrimaryDimTypes dim)
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
inline bool e002_isLabeledDsDimensionBroadcast(const DesignSpaceConfig &dsc,
                                                 const LabeledDsInfo &lds,
                                                 PrimaryDimTypes dim)
{
  int scaleIdx = dsc.getDimIndexInLayoutOrder(lds.dsType_, dim);
  DT_CHECK_MSG((scaleIdx >= 0 && scaleIdx < lds.scale_.size()),
               "Invalid layoutDimOrder_ index.");
  return (lds.scale_.at(scaleIdx) < 1);
}

// ------------------------------------------------------------------------------------------------
// entry 003/382   level 0   scc 2   12 body lines
// unit: e003_isDimensionCoreletSplit
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:74
// original: bool L3DlOpsScheduler::isDimensionCoreletSplit(const DesignSpaceConfig &dsc, PrimaryDimTypes dim)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e003_isDimensionCoreletSplit(const DesignSpaceConfig &dsc,
                                               PrimaryDimTypes dim)
{
  if (dsc.numCoreletsUsed_ <= 1) return false;
  if (dsc.dataStageParam_.count(dataStageCoreIdx)) {
    auto &cdtsg = dsc.dataStageParam_.at(dataStageCoreIdx).ss_;

    return cdtsg.primaryDimToVal_st(dim, SenComponents::NO_COMPONENT, -1, 0) <
           cdtsg.primaryDimToVal_st(dim, SenComponents::NO_COMPONENT, -1, -1);
  } else {
    return dsc.CoreletD_.primaryDimToVal_st(dim) <
           dsc.CoreD_.primaryDimToVal_st(dim);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 004/382   level 0   scc 7   20 body lines
// unit: e004_voidPaddingIfChunking
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:128
// original: static void voidPaddingIfChunking(DataStructDims &ds, const DataStructDims &refDs)
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e004_voidPaddingIfChunking(DataStructDims &ds,
                                  const DataStructDims &refDs)
{
  DT_CHECK_MSG((!ds.empty() && !refDs.empty()),
               "Expect non-empty data-stage parameters.");
  for (auto &[dim, padInfo] : ds.paddingSizes_) {
    if (refDs.primaryDimToVal_st(dim) != ds.primaryDimToVal_st(dim) ||
        (padInfo.windowDim_ != PrimaryDimTypesCount &&
         refDs.primaryDimToVal_st(padInfo.windowDim_) !=
             ds.primaryDimToVal_st(padInfo.windowDim_))) {
      if (padInfo.padBack_ != 0 || padInfo.padFront_ != 0)
        padInfo.padBack_ = padInfo.padFront_ = -1;
      // TODO: This is a temporary workaround to carry the unneededPad_ from
      // core to chunk data stage due to the limitation of the current L3 PCFG
      // generation. We will set unneededPad_ to 0 when the DSC2ToPCFG
      // translator is available.
      if (!carryUnneededPadToChunk)
        padInfo.unneededPad_ = padInfo.unneededPadFront_ =
            padInfo.unneededPadBack_ = 0;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 005/382   level 0   scc 9   15 body lines
// unit: e005_addOrUpdateSymbolicInfoInParams
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:158
// original: static void addOrUpdateSymbolicInfoInParams(DataStructDims &chunkParams, const DataStructDims &coreParams)
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e005_addOrUpdateSymbolicInfoInParams(DataStructDims &chunkParams,
                                            const DataStructDims &coreParams)
{
  DT_CHECK_MSG((!chunkParams.empty() && !coreParams.empty()),
               "Expect non-empty data-stage parameters.");
  // Add symbolicDimInfo_ for dims that are not chunked
  for (const auto &[dim, symbolicInfo] : coreParams.symbolicDimInfo_) {
    if (chunkParams.primaryDimToVal_st(dim) ==
        coreParams.primaryDimToVal_st(dim)) {
      chunkParams.symbolicDimInfo_[dim] = symbolicInfo;
    }
  }

  // Add and update maxSymbolicVolume_.
  chunkParams.maxSymbolicVolume_ = coreParams.maxSymbolicVolume_;
  chunkParams.pruneMaxSymbolicVolumes(coreParams);
}

// ------------------------------------------------------------------------------------------------
// entry 006/382   level 0   scc 10   39 body lines
// unit: e006_getLabeledDsWkSliceMulticastDegree
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:176
// original: static unsigned getLabeledDsWkSliceMulticastDegree( const SuperDsc &mySDsc, const int ldsIdx, const std::vector<int> &dscIndices)
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
unsigned e006_getLabeledDsWkSliceMulticastDegree(
    const SuperDsc &mySDsc, const int ldsIdx,
    const std::vector<int> &dscIndices)
{
  DT_CHECK_MSG(!dscIndices.empty(), "Expect valid DSCs.");
  // Use the first DSC to get labeledDsInfo.
  const int dscMainIdx = dscIndices[0];
  const DesignSpaceConfig &dscMain = mySDsc.dscs_.at(dscMainIdx);

  // Use the work slice in the first coreId of the first DSC to compute the
  // multicast degree, since all work slices should have the same value in
  // the specified DSCs.
  int mainCoreId = dscMain.coreIdsUsed_[0];

  // All coreIds in the specified DSCs.
  std::unordered_set<int> processingCoreIds;
  for (int dscIdx : dscIndices) {
    const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
    for (int coreId : dsc.coreIdsUsed_) processingCoreIds.insert(coreId);
  }

  int multicastDegree = 0;
  for (const auto &coreEntry : mySDsc.coreIdToWkSlice_) {
    const int coreId = coreEntry.first;

    // Skip if the coreId is not in the processing coreIds.
    if (!processingCoreIds.count(coreId)) continue;

    bool match = true;
    for (PrimaryDimTypes dim : dscMain.getLayoutDims(ldsIdx)) {
      if (coreEntry.second.at(dim) !=
          mySDsc.coreIdToWkSlice_.at(mainCoreId).at(dim)) {
        match = false;
        break;
      }
    }

    if (match) ++multicastDegree;
  }

  return multicastDegree;
}

// ------------------------------------------------------------------------------------------------
// entry 007/382   level 0   scc 12   16 body lines
// unit: e007_scheduleDimTypeToString
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:282
// original: std::string L3DlOpsScheduler::scheduleDimTypeToString(ScheduleDimTypes type)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::string e007_scheduleDimTypeToString(ScheduleDimTypes type)
{
  switch (type) {
    case ScheduleDimTypes::ELEMENTWISE:
      return "Elementwise";
    case ScheduleDimTypes::BROADCAST:
      return "Broadcast";
    case ScheduleDimTypes::REDUCTION:
      return "Reduction";
    case ScheduleDimTypes::WINDOW_PADDED:
      return "Window/Padded";
    case ScheduleDimTypes::REUSE:
      return "Reuse";
    default:
      DT_ERROR("Unsupported ScheduleDimTypes.");
  }
}

// ------------------------------------------------------------------------------------------------
// entry 008/382   level 0   scc 13   19 body lines
// unit: e008_hasDimensionReuse
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:303
// original: bool L3DlOpsScheduler::hasDimensionReuse(const DesignSpaceConfig& dsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e008_hasDimensionReuse(const DesignSpaceConfig& dsc)
{
  bool hasReuse = false;
  if (dsc.primaryDsInfo_.size() > 1 &&
      dsc.primaryDsInfo_.count(DsTypes::KERNEL)) {
    std::map<PrimaryDimTypes, int> countPerDim;
    for (auto& kv : dsc.primaryDsInfo_) {
      for (auto& entry : kv.second.layoutDimOrder_) {
        countPerDim[entry]++;
      }
    }
    for (auto& kv : countPerDim) {
      if (kv.second < dsc.primaryDsInfo_.size()) {
        hasReuse = true;
        break;
      }
    }
  }
  return hasReuse;
}

// ------------------------------------------------------------------------------------------------
// entry 009/382   level 0   scc 14   12 body lines
// unit: e009_getStickSize
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:327
// original: int L3DlOpsScheduler::getStickSize(const DesignSpaceConfig &dsc, DsTypes dsType, PrimaryDimTypes dim)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
int e009_getStickSize(const DesignSpaceConfig &dsc, DsTypes dsType,
                                   PrimaryDimTypes dim)
{
  int stickSize = 1;
  const auto stickSizePerDim = dsc.getCumulativeStickSizes(dsType);
  for (const auto &entry : stickSizePerDim) {
    if (dim == entry.first) {
      stickSize = entry.second;
      break;
    }
  }

  return stickSize;
}

// ------------------------------------------------------------------------------------------------
// entry 010/382   level 0   scc 15   25 body lines
// unit: e010_getCoreSplitDimensions
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:342
// original: std::unordered_set<PrimaryDimTypes> L3DlOpsScheduler::getCoreSplitDimensions( const SuperDsc &mySDsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::unordered_set<PrimaryDimTypes> e010_getCoreSplitDimensions(
    const SuperDsc &mySDsc) const
{
  std::unordered_set<PrimaryDimTypes> dims;
  const DesignSpaceConfig &dscMain = mySDsc.dscs_.at(0);
  for (const auto &entry : EnumsConversion::primaryDimToString) {
    PrimaryDimTypes dim = entry.first;
    // Ignore the combined IJ and KIJ dimensions.
    if (is_any_of(dim, PrimaryDimTypes::IJ, PrimaryDimTypes::KIJ,
                  PrimaryDimTypes::PrimaryDimTypesCount))
      continue;

    DT_CHECK_MSG(dscMain.dataStageParam_.count(dataStageCoreIdx),
                 "Expect dataStageParam_ entry for the core data stage.");
    auto &cdtsg = dscMain.dataStageParam_.at(dataStageCoreIdx).ss_;
    long param = cdtsg.primaryDimToVal_st(dim);
    for (const DesignSpaceConfig &dsc : mySDsc.dscs_) {
      auto &dscCdtsg = dsc.dataStageParam_.at(dataStageCoreIdx).ss_;
      if (param != dscCdtsg.primaryDimToVal_st(dim)) {
        dims.insert(dim);
        break;
      }
    }
  }

  return dims;
}

// ------------------------------------------------------------------------------------------------
// entry 011/382   level 0   scc 16   6 body lines
// unit: e011_getLabeledDsWithDsType
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:369
// original: void L3DlOpsScheduler::getLabeledDsWithDsType(std::vector<int> &indices, DesignSpaceConfig &dsc, DsTypes dsType)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e011_getLabeledDsWithDsType(std::vector<int> &indices,
                                              DesignSpaceConfig &dsc,
                                              DsTypes dsType)
{
  for (int idx = 0; idx < dsc.labeledDs_.size(); ++idx) {
    LabeledDsInfo &lds = dsc.labeledDs_.at(idx);
    if (lds.dsType_ == dsType) indices.push_back(idx);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 012/382   level 0   scc 17   7 body lines
// unit: e012_getAllLabeledDsIndicesSet
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:378
// original: std::unordered_set<int> L3DlOpsScheduler::getAllLabeledDsIndicesSet( const DesignSpaceConfig& dsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::unordered_set<int> e012_getAllLabeledDsIndicesSet(
    const DesignSpaceConfig& dsc) const
{
  std::unordered_set<int> indices;
  for (const auto& lds : dsc.labeledDs_) {
    indices.insert(lds.ldsIdx_);
  }
  return indices;
}

// ------------------------------------------------------------------------------------------------
// entry 013/382   level 0   scc 18   7 body lines
// unit: e013_getHbmPinnedLabeledDsIndicesSet
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:387
// original: std::unordered_set<int> L3DlOpsScheduler::getHbmPinnedLabeledDsIndicesSet( const DesignSpaceConfig& dsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::unordered_set<int> e013_getHbmPinnedLabeledDsIndicesSet(
    const DesignSpaceConfig& dsc) const
{
  std::unordered_set<int> indices;
  for (const auto& lds : dsc.labeledDs_) {
    if (lds.isHbmPinned()) indices.insert(lds.ldsIdx_);
  }
  return indices;
}

// ------------------------------------------------------------------------------------------------
// entry 014/382   level 0   scc 19   26 body lines
// unit: e014_isLabeledDsLXNeighbor
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:406
// original: bool L3DlOpsScheduler::isLabeledDsLXNeighbor(const SuperDsc &mySDsc, const int dscIndex, const LabeledDsInfo &lds) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e014_isLabeledDsLXNeighbor(const SuperDsc &mySDsc,
                                             const int dscIndex,
                                             const LabeledDsInfo &lds) const
{
  if (!lds.isLxPinned()) return false;

  // LX-neighbor (input neighbor fetch) only applies to the input tensor with
  // DsType INPUT.
  if (lds.dsType_ != DsTypes::INPUT) return false;

  // Look at coreIdToDscSchedule.
  // Each DscScheduleStep has four fields as below:
  //   [int dataDsc_index, int dlDsc_index, bool after_sync, bool before_sync]

  // For a given DSC, any cores that it uses must have the same information in
  // their DscScheduleStep. Therefore, we only need to look at one of the
  // coreIds. Pick the first coreId.
  const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIndex);
  int coreId = dsc.coreIdsUsed_[0];
  const std::vector<DscScheduleStep> &dscSchedule =
      mySDsc.coreIdToDscSchedule.at(coreId);

  // Note: A DSC can only be used in one schedule step.
  for (const DscScheduleStep &step : dscSchedule) {
    if (step.dldsc_idx == dscIndex && step.datadsc_idx >= 0) return true;
  }

  return false;
}

// ------------------------------------------------------------------------------------------------
// entry 015/382   level 0   scc 24   10 body lines
// unit: e015_getParentLoopNodes
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:532
// original: std::vector<const dsc2::LoopNode *> L3DlOpsScheduler::getParentLoopNodes( const dsc2::ScheduleNode &node, const DesignSpaceConfig &dsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::vector<const dsc2::LoopNode *> e015_getParentLoopNodes(
    const dsc2::ScheduleNode &node, const DesignSpaceConfig &dsc) const
{
  DT_CHECK_MSG(!dsc.scheduleTree_.empty(), "Expect valid schedule tree.");
  std::vector<const dsc2::LoopNode *> loops;
  auto parentLoop = node.getOwnerLoop();
  while (parentLoop && parentLoop->getPrev()) {
    loops.push_back(parentLoop);
    parentLoop = parentLoop->getOwnerLoop();
  }
  return loops;
}

// ------------------------------------------------------------------------------------------------
// entry 016/382   level 0   scc 25   49 body lines
// unit: e016_createAllocateNode
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:544
// original: dsc2::AllocateNode *L3DlOpsScheduler::createAllocateNode( DesignSpaceConfig &dsc, const int ldsIdx, enum SenComponents component, const int numBuffers, const std::string &name, const int dscIdx)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
dsc2::AllocateNode *e016_createAllocateNode(
    DesignSpaceConfig &dsc, const int ldsIdx, enum SenComponents component,
    const int numBuffers, const std::string &name, const int dscIdx)
{
  dsc2::AllocateNode *node = new dsc2::AllocateNode();
  node->name_ = name;
  node->ldsIdx_ = ldsIdx;
  node->component_ = component;
  node->numBuffers_ = numBuffers;

  // Set layout
  const LabeledDsInfo &lds = dsc.labeledDs_.at(ldsIdx);
  node->layoutDimOrder_ = dsc.getLayoutDims(ldsIdx);
  DT_CHECK_MSG(
      std::set(node->layoutDimOrder_.begin(), node->layoutDimOrder_.end())
              .size() == node->layoutDimOrder_.size(),
      "Handling of external allocations with repeated dimensions is not yet "
      "implemented");
  node->maxDimSizes_.resize(node->layoutDimOrder_.size(), -1);

  // Set the padding_ field if the labeledDs has padding.
  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
               "Expect valid core data stage params.");
  const dsc2::DataStage &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);
  // padding.
  if (component == SenComponents::LX && lds.memOrg_.count(SenComponents::LX) &&
      lds.memOrg_.at(SenComponents::LX).isPadded) {
    for (auto &dim : node->layoutDimOrder_) {
      if (dsCore.ss_.paddingSizes_.count(dim)) {
        node->padding_.setPadding(dim, PadType::PADDED_FULLSPAN_WUNNEEDED);
      }
    }
  }

  // Update metadata if it is allocation for HBM transfer. We use the metadata
  // to allocate LX memory for the chunks.
  if (lds.isHbmPinned() && component == SenComponents::LX) {
    DT_CHECK_MSG(dscMetadata.count(dscIdx), "Expect a metadata entry.");
    if (dscMetadata.at(dscIdx).newAllocations_.find(component) ==
        dscMetadata.at(dscIdx).newAllocations_.end())
      dscMetadata.at(dscIdx).newAllocations_.emplace(component,
                                                     Metadata::Allocation());

    auto &allocMetadata =
        dscMetadata.at(dscIdx).newAllocations_.at(component).ldsIdxAndAllocNode;
    DT_CHECK_MSG(allocMetadata.find(ldsIdx) == allocMetadata.end(),
                 "Do not expect the labeledDs index was previously added.");
    allocMetadata[ldsIdx] = node;
  }

  return node;
}

// ------------------------------------------------------------------------------------------------
// entry 017/382   level 0   scc 26   22 body lines
// unit: e017_createTransferNode
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:596
// original: dsc2::TransferNode *L3DlOpsScheduler::createTransferNode( const SenComponents srcUnit, const SenComponents srcStorage, const std::vector<SenComponents> &dstUnits, const std::vector<SenComponents> &dstStorage, const int srcLdsIndex, const std::vector<int> &dstLdsIndices, const std::string &name)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
dsc2::TransferNode *e017_createTransferNode(
    const SenComponents srcUnit, const SenComponents srcStorage,
    const std::vector<SenComponents> &dstUnits,
    const std::vector<SenComponents> &dstStorage, const int srcLdsIndex,
    const std::vector<int> &dstLdsIndices, const std::string &name)
{
  dsc2::TransferNode *node = new dsc2::TransferNode();
  node->name_ = name;
  node->src_.unit_ = srcUnit;
  node->src_.storage_ = srcStorage;

  DT_CHECK_MSG(dstUnits.size() == dstStorage.size(),
               "Destination unit and storage numbers do not match.");
  for (int i = 0; i < dstUnits.size(); ++i) {
    node->dstVias_.emplace_back();
    node->dstVias_.back().loc_.unit_ = dstUnits[i];
    node->dstVias_.back().loc_.storage_ = dstStorage[i];
  }

  node->srcLdsAndLoopOffsets_.myLdsIdx_ = srcLdsIndex;
  for (const int index : dstLdsIndices) {
    node->dstLdsAndLoopOffsets_.emplace_back();
    node->dstLdsAndLoopOffsets_.back().myLdsIdx_ = index;
  }

  return node;
}

// ------------------------------------------------------------------------------------------------
// entry 018/382   level 0   scc 27   19 body lines
// unit: e018_createLoopNode
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:623
// original: dsc2::LoopNode *L3DlOpsScheduler::createLoopNode( const DesignSpaceConfig &dsc, const std::vector<PrimaryDimTypes> &dims, const int numeratorId, const int denominatorId, const std::string &name)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
dsc2::LoopNode *e018_createLoopNode(
    const DesignSpaceConfig &dsc, const std::vector<PrimaryDimTypes> &dims,
    const int numeratorId, const int denominatorId, const std::string &name)
{
  dsc2::LoopNode *node = new dsc2::LoopNode();
  node->name_ = name;
  node->numId_ = numeratorId;
  node->denId_ = denominatorId;

  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
               "Expect valid core data stage.");
  const dsc2::DataStage &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);
  for (auto dim : dims) {
    if (dim == PrimaryDimTypes::KI && dsCore.ss_.ki_ > 0)
      node->dims_.emplace_back(dim, MetaDimKind::WindowDim);
    else if (dim == PrimaryDimTypes::KJ && dsCore.ss_.kj_ > 0)
      node->dims_.emplace_back(dim, MetaDimKind::WindowDim);
    else
      node->dims_.emplace_back(dim);
  }
  return node;
}

// ------------------------------------------------------------------------------------------------
// entry 019/382   level 0   scc 28   6 body lines
// unit: e019_createBlockNode
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:645
// original: dsc2::BlockNode *L3DlOpsScheduler::createBlockNode(const std::string &name)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
dsc2::BlockNode *e019_createBlockNode(const std::string &name)
{
  dsc2::BlockNode *node = new dsc2::BlockNode();
  node->name_ = name;

  return node;
}

// ------------------------------------------------------------------------------------------------
// entry 020/382   level 0   scc 29   10 body lines
// unit: e020_createSyncNode
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:652
// original: dsc2::SyncNode* L3DlOpsScheduler::createSyncNode( const std::unordered_set<SenComponents>& units, const std::string& name, const bool isReceive, const bool isSoft) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
dsc2::SyncNode* e020_createSyncNode(
    const std::unordered_set<SenComponents>& units, const std::string& name,
    const bool isReceive, const bool isSoft) const
{
  dsc2::SyncNode *node = new dsc2::SyncNode();
  node->name_ = name;
  if (isReceive) node->isReceive_ = true;
  if (isSoft) node->isSoft_ = true;

  for (const auto &unit : units) node->units_.insert(unit);

  return node;
}

// ------------------------------------------------------------------------------------------------
// entry 021/382   level 0   scc 30   4 body lines
// unit: e021_getOpFuncName
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:665
// original: OpFuncs L3DlOpsScheduler::getOpFuncName(const DesignSpaceConfig& dsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
OpFuncs e021_getOpFuncName(const DesignSpaceConfig& dsc) const
{
  DT_CHECK(hasComputeOp(dsc));
  return dsc.computeOp_.front().opFuncName;
}

// ------------------------------------------------------------------------------------------------
// entry 022/382   level 0   scc 32   12 body lines
// unit: e022_addOrUpdateDataStageParam
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:721
// original: void L3DlOpsScheduler::addOrUpdateDataStageParam(DesignSpaceConfig &dsc, const DataStructDims &ssParam, const std::string &ssName, const DataStructDims &elParam, const std::string &elName, const int index)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e022_addOrUpdateDataStageParam(DesignSpaceConfig &dsc,
                                                 const DataStructDims &ssParam,
                                                 const std::string &ssName,
                                                 const DataStructDims &elParam,
                                                 const std::string &elName,
                                                 const int index)
{
  DT_CHECK_MSG((!ssParam.empty() && !elParam.empty()),
               "Expect non-empty data-stage parameters.");
  // Add a new dataStageParam_ entry if it does not exist.
  if (dsc.dataStageParam_.count(index) == 0)
    dsc.dataStageParam_.emplace(index, dsc2::DataStage());

  dsc.dataStageParam_[index].ss_ = ssParam;
  dsc.dataStageParam_[index].ss_.name_ = ssName;
  dsc.dataStageParam_[index].el_ = elParam;
  dsc.dataStageParam_[index].el_.name_ = elName;
}

// ------------------------------------------------------------------------------------------------
// entry 023/382   level 0   scc 33   6 body lines
// unit: e023_isOpFuncConv2dInt4
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:739
// original: bool L3DlOpsScheduler::isOpFuncConv2dInt4(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e023_isOpFuncConv2dInt4(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncConv2dInt4 = {
      OpFuncs::CONV2D_INT4_FWD, OpFuncs::CONV2D_INT4_FWD_GENKG3,
      OpFuncs::CONV2D_INT4_FWD_SPARSEKG3};
  return opFuncConv2dInt4.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 024/382   level 0   scc 34   6 body lines
// unit: e024_isOpFuncConv2dOs1
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:746
// original: bool L3DlOpsScheduler::isOpFuncConv2dOs1(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e024_isOpFuncConv2dOs1(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncConv2dOs1 = {
      OpFuncs::CONV2D_FWD_OS1, OpFuncs::CONV2D_XRF_INT8_FWD_OS1,
      OpFuncs::CONV2D_FWD_GEN_OS1, OpFuncs::CONV2D_INT8_FWD_OS1};
  return opFuncConv2dOs1.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 025/382   level 0   scc 36   6 body lines
// unit: e025_isOpFuncBmmInt4
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:769
// original: bool L3DlOpsScheduler::isOpFuncBmmInt4(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e025_isOpFuncBmmInt4(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncBmmInt4 = {
      OpFuncs::BATCHMATMUL_INT4_FWD, OpFuncs::BATCHMATMUL_INT4_FWD_SPARSEKG3,
      OpFuncs::BATCHMATMUL_XRF_INT4_FWD, OpFuncs::BATCHMATMUL_XRFCH_INT4_FWD};
  return opFuncBmmInt4.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 026/382   level 0   scc 37   7 body lines
// unit: e026_isOpFuncBmmInt8
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:776
// original: bool L3DlOpsScheduler::isOpFuncBmmInt8(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e026_isOpFuncBmmInt8(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncBmmInt8 = {
      OpFuncs::BATCHMATMUL_INT8_FWD, OpFuncs::BATCHMATMUL_INT8_FWD_MBKG3,
      OpFuncs::BATCHMATMUL_INT8_FWD_SPARSEKG3,
      OpFuncs::BATCHMATMUL_XRF_INT8_FWD, OpFuncs::BATCHMATMUL_XRFCH_INT8_FWD};
  return opFuncBmmInt8.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 027/382   level 0   scc 38   6 body lines
// unit: e027_isOpFuncBmmFp8NonXrf
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:784
// original: bool L3DlOpsScheduler::isOpFuncBmmFp8NonXrf(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e027_isOpFuncBmmFp8NonXrf(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncBmmFp8NonXrf = {
      OpFuncs::BATCHMATMUL_FP8_FWD, OpFuncs::BATCHMATMUL_FP8_FWD_MB,
      OpFuncs::BATCHMATMUL_FP8_FWD_SPARSEKG3};
  return opFuncBmmFp8NonXrf.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 028/382   level 0   scc 39   5 body lines
// unit: e028_isOpFuncBmmFp8Xrf
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:791
// original: bool L3DlOpsScheduler::isOpFuncBmmFp8Xrf(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e028_isOpFuncBmmFp8Xrf(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncBmmFp8Xrf = {
      OpFuncs::BATCHMATMUL_XRF_FP8_FWD, OpFuncs::BATCHMATMUL_XRFCH_FP8_FWD};
  return opFuncBmmFp8Xrf.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 029/382   level 0   scc 40   6 body lines
// unit: e029_isOpFuncBmmFp16
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:797
// original: bool L3DlOpsScheduler::isOpFuncBmmFp16(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e029_isOpFuncBmmFp16(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncBmmFp16 = {
      OpFuncs::BATCHMATMUL_FWD, OpFuncs::BATCHMATMUL_FWD_SPARSEKG3,
      OpFuncs::BATCHMATMUL_XRF_FWD, OpFuncs::BATCHMATMUL_XRFCH_FWD};
  return opFuncBmmFp16.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 030/382   level 0   scc 42   18 body lines
// unit: e030_isOpFuncScalarBroadcast
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:810
// original: bool L3DlOpsScheduler::isOpFuncScalarBroadcast(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e030_isOpFuncScalarBroadcast(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncScalarBroadcast = {
      OpFuncs::RELU_FWD,
      OpFuncs::RELU6_FWD,
      OpFuncs::LEAKYRELU_FWD,
      OpFuncs::GELU_FWD,
      OpFuncs::TANH_FWD,
      OpFuncs::SIGMOID_FWD,
      OpFuncs::FAST_SIGMOID_FWD,
      OpFuncs::ADD,
      OpFuncs::STRIDED_ADD,
      OpFuncs::MUL,
      OpFuncs::SUB,
      OpFuncs::REVSUB,
      OpFuncs::BIASADD,
      OpFuncs::BATCHNORM_FWD};
  return opFuncScalarBroadcast.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 031/382   level 0   scc 43   7 body lines
// unit: e031_isOpFuncReduction
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:829
// original: bool L3DlOpsScheduler::isOpFuncReduction(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e031_isOpFuncReduction(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncReduction = {
      OpFuncs::SUM,           OpFuncs::MAX,          OpFuncs::MEAN,
      OpFuncs::EXX2,          OpFuncs::SUM_NONSTICK, OpFuncs::MAX_NONSTICK,
      OpFuncs::MEAN_NONSTICK, OpFuncs::PROD_NONSTICK};
  return opFuncReduction.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 032/382   level 0   scc 44   5 body lines
// unit: e032_isOpFuncPooling
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:837
// original: bool L3DlOpsScheduler::isOpFuncPooling(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e032_isOpFuncPooling(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncPooling = {
      OpFuncs::MAXPOOL_FWD, OpFuncs::AVGPOOL_FWD, OpFuncs::AVGPOOL_NMAP_FWD};
  return opFuncPooling.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 033/382   level 0   scc 45   5 body lines
// unit: e033_isOpFuncDepthwiseConv
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:843
// original: bool L3DlOpsScheduler::isOpFuncDepthwiseConv(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e033_isOpFuncDepthwiseConv(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncDepthwiseConv = {
      OpFuncs::DEPTHWISE_CONV_FWD};
  return opFuncDepthwiseConv.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 034/382   level 0   scc 46   8 body lines
// unit: e034_isOpFuncQuantization
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:849
// original: bool L3DlOpsScheduler::isOpFuncQuantization(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e034_isOpFuncQuantization(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncQuantization = {
      OpFuncs::Q_FP8,       OpFuncs::Q_FP8_CH,      OpFuncs::Q_FP8_CHIL,
      OpFuncs::Q_FP8_WT,    OpFuncs::CSQ_INT8,      OpFuncs::CSQ_INT8_CH,
      OpFuncs::CSQ_INT8_WT, OpFuncs::CSQ_INT8_CHIL, OpFuncs::CSQ_INT8_MB,
      OpFuncs::CSQ_INT4,    OpFuncs::CSQ_INT4_WT,   OpFuncs::CSQ_INT4_CHIL};
  return opFuncQuantization.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 035/382   level 0   scc 47   5 body lines
// unit: e035_isOpFuncConversionDl16AndFp32
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:858
// original: bool L3DlOpsScheduler::isOpFuncConversionDl16AndFp32( const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e035_isOpFuncConversionDl16AndFp32(
    const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncConversionDl16AndFp32 = {
      OpFuncs::DL16TOFP32, OpFuncs::FP32TODL16};
  return opFuncConversionDl16AndFp32.count(opFuncName) > 0;
}

// ------------------------------------------------------------------------------------------------
// entry 036/382   level 0   scc 52   16 body lines
// unit: e036_getMinParamScalarBroadcast
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1008
// original: long L3DlOpsScheduler::getMinParamScalarBroadcast( const DesignSpaceConfig& dsc, const PrimaryDimTypes dim) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
long e036_getMinParamScalarBroadcast(
    const DesignSpaceConfig& dsc, const PrimaryDimTypes dim) const
{
  constexpr long defaultParam = 1;
  const auto coreParam =
      dsc.dataStageParam_.at(dataStageCoreIdx).ss_.primaryDimToVal_st(dim);

  switch (dim) {
    case PrimaryDimTypes::J:
      return coreParam;
    default: {
      const auto outputLdsStickSizes =
          dsc.getCumulativeStickSizes(dsc.labeledDs_.back().dsType_);
      return outputLdsStickSizes.count(dim) ? outputLdsStickSizes.at(dim)
                                            : defaultParam;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 037/382   level 0   scc 53   12 body lines
// unit: e037_getMinParamReduction
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1026
// original: long L3DlOpsScheduler::getMinParamReduction(const DesignSpaceConfig& dsc, const PrimaryDimTypes dim) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
long e037_getMinParamReduction(const DesignSpaceConfig& dsc,
                                            const PrimaryDimTypes dim) const
{
  constexpr long defaultParam = 1;
  const auto coreParam =
      dsc.dataStageParam_.at(dataStageCoreIdx).ss_.primaryDimToVal_st(dim);

  switch (dim) {
    case PrimaryDimTypes::J:
      return coreParam;
    default:
      return defaultParam;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 038/382   level 0   scc 54   22 body lines
// unit: e038_getMinParamPoolingAndDepthwiseConv
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1040
// original: long L3DlOpsScheduler::getMinParamPoolingAndDepthwiseConv( const DesignSpaceConfig& dsc, const PrimaryDimTypes dim, const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
long e038_getMinParamPoolingAndDepthwiseConv(
    const DesignSpaceConfig& dsc, const PrimaryDimTypes dim,
    const OpFuncs opFuncName) const
{
  constexpr long defaultParam = 1;
  const auto coreParam =
      dsc.dataStageParam_.at(dataStageCoreIdx).ss_.primaryDimToVal_st(dim);

  switch (dim) {
    case PrimaryDimTypes::IN: {
      if (opFuncName == OpFuncs::DEPTHWISE_CONV_FWD)
        return coreParam;
      else
        return defaultParam;
    }
    case PrimaryDimTypes::OUT:
      return 64;
    case PrimaryDimTypes::J:
    case PrimaryDimTypes::KI:
    case PrimaryDimTypes::KJ:
      return coreParam;
    default:
      return defaultParam;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 039/382   level 0   scc 55   31 body lines
// unit: e039_getMinParamQuantization
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1065
// original: long L3DlOpsScheduler::getMinParamQuantization(const DesignSpaceConfig& dsc, const PrimaryDimTypes dim, const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
long e039_getMinParamQuantization(const DesignSpaceConfig& dsc,
                                               const PrimaryDimTypes dim,
                                               const OpFuncs opFuncName) const
{
  constexpr long defaultParam = 1;
  const auto& dsCoreSs = dsc.dataStageParam_.at(dataStageCoreIdx).ss_;
  const auto coreParam = dsCoreSs.primaryDimToVal_st(dim);
  const bool hasPadding =
      dsCoreSs.paddingSizes_.count(dim) &&
      dsCoreSs.primaryDimToVal_st(dim, SenComponents::NO_COMPONENT, -1, -1,
                                  {dim, PadType::PADDED_FULLSPAN_WUNNEEDED}) !=
          dsCoreSs.primaryDimToVal_st(dim);

  switch (dim) {
    case PrimaryDimTypes::J:
      return coreParam;
    case PrimaryDimTypes::IN:
    case PrimaryDimTypes::OUT: {
      if (hasPadding)
        return coreParam;
      else if (opFuncName == OpFuncs::CSQ_INT4 ||
               opFuncName == OpFuncs::CSQ_INT4_CHIL)
        return 128;
      else
        return 64;
    }
    default: {
      if (hasPadding)
        return coreParam;
      else
        return defaultParam;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 040/382   level 0   scc 56   17 body lines
// unit: e040_getMinParamConversionDl16AndFp32
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1099
// original: long L3DlOpsScheduler::getMinParamConversionDl16AndFp32( const DesignSpaceConfig& dsc, const PrimaryDimTypes dim, const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
long e040_getMinParamConversionDl16AndFp32(
    const DesignSpaceConfig& dsc, const PrimaryDimTypes dim,
    const OpFuncs opFuncName) const
{
  constexpr long defaultParam = 1;
  const auto coreParam =
      dsc.dataStageParam_.at(dataStageCoreIdx).ss_.primaryDimToVal_st(dim);

  switch (dim) {
    case PrimaryDimTypes::J:
      return coreParam;
    default: {
      const auto& lds = (opFuncName == OpFuncs::DL16TOFP32)
                            ? dsc.labeledDs_.front()
                            : dsc.labeledDs_.back();
      const auto ldsStickSizes = dsc.getCumulativeStickSizes(lds.dsType_);
      return ldsStickSizes.count(dim) ? ldsStickSizes.at(dim) : defaultParam;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 041/382   level 0   scc 64   12 body lines
// unit: e041_getChunkParamsFromCandidates
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1423
// original: void L3DlOpsScheduler::getChunkParamsFromCandidates( DataStructDims &params, const DscParamCandidateIndicesType &selectedIndices, const DscParamCandidatesType &dscCandidates, const int dscIdx, const std::vector<PrimaryDimTypes> &primaryDims)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e041_getChunkParamsFromCandidates(
    DataStructDims &params, const DscParamCandidateIndicesType &selectedIndices,
    const DscParamCandidatesType &dscCandidates, const int dscIdx,
    const std::vector<PrimaryDimTypes> &primaryDims)
{
  for (PrimaryDimTypes dim : primaryDims) {
    const unsigned selectedIdx = selectedIndices[dscIdx].at(dim);
    DT_CHECK_MSG(selectedIdx < dscCandidates[dscIdx].at(dim).size(),
                 "Index is out of range.");
    params.primaryDimToValHandler_st(dim) =
        dscCandidates[dscIdx].at(dim).at(selectedIdx);
  }

  // Compute and update the compound IJ, KIJ, and RC dimensions.
  params.compound();
}

// ------------------------------------------------------------------------------------------------
// entry 042/382   level 0   scc 75   17 body lines
// unit: e042_getBurstEfficiency
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1609
// original: double L3DlOpsScheduler::getBurstEfficiency(const unsigned burstSize, const unsigned multicastDegree)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
double e042_getBurstEfficiency(const unsigned burstSize,
                                            const unsigned multicastDegree)
{
  DT_CHECK_MSG(burstSize >= 1 && burstSize <= dscGlobal.sysDef.l3BurstSize,
               "Invalid Burst size.");
  DT_CHECK_MSG(
      multicastDegree >= 1 && multicastDegree <= dscGlobal.sysDef.numCores,
      "Invalid multicast degree.");
  DT_CHECK_MSG(burstEfficiency.size() == dscGlobal.sysDef.l3BurstSize,
               "Incorrect number of burst sizes in the table.");
  const int maxNumCores = 32;
  for (auto &entry : burstEfficiency)
    DT_CHECK_MSG(entry.size() == maxNumCores,
                 "Incorrect number of multicast degrees in the table.");
  // Index starts with zero.
  const unsigned burstIdx = burstSize - 1;
  const unsigned multicastIdx = multicastDegree - 1;
  return burstEfficiency.at(burstIdx).at(multicastIdx);
}

// ------------------------------------------------------------------------------------------------
// entry 043/382   level 0   scc 74   37 body lines
// unit: e043_getLabeledDsNumOfStickVolumesInCore
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1694
// original: unsigned long L3DlOpsScheduler::getLabeledDsNumOfStickVolumesInCore( const DesignSpaceConfig &dsc, const int ldsIdx, const unsigned long stickVolume, const std::vector<PrimaryDimTypes> &primaryDims)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
unsigned long e043_getLabeledDsNumOfStickVolumesInCore(
    const DesignSpaceConfig &dsc, const int ldsIdx,
    const unsigned long stickVolume,
    const std::vector<PrimaryDimTypes> &primaryDims)
{
  // Compute the number of stick volumes in CoreD for this tensor.
  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
               "Core data stage parameters are unavailable.");
  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageChunkIdx),
               "Chunk data stage parameters are unavailable.");
  DT_CHECK_MSG(stickVolume > 0, "Invalid stick volume.");
  const LabeledDsInfo &lds = dsc.labeledDs_.at(ldsIdx);
  DT_CHECK_MSG(lds.memOrg_.count(SenComponents::LX),
               "Expect LX in labeledDs memOrg_.");
  const auto *allocNode = lds.memOrg_.at(SenComponents::LX).allocateNode_;
  DT_CHECK_MSG(allocNode && allocNode->component_ == SenComponents::LX,
               "Expect a valid LX allocate node.");

  int64_t chunkSizeInBytes =
      dsc.getBufferCapacityForNode(allocNode, ldsIdx, allocNode->component_, -1,
                                   -1, dscGlobal.sysDef.bytesPerStick);
  DT_CHECK_MSG(chunkSizeInBytes % dscGlobal.sysDef.bytesPerStick == 0,
               "Incorrect chunk size in bytes.");
  int64_t chunkSizeInSticks = chunkSizeInBytes / dscGlobal.sysDef.bytesPerStick;
  DT_CHECK_MSG(chunkSizeInSticks % stickVolume == 0,
               "Incorrect number of stick volumes in a chunk.");

  const auto &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);
  const auto &dsChunk = dsc.dataStageParam_.at(dataStageChunkIdx);
  int numChunks = 1;
  for (PrimaryDimTypes dim : dsc.getNonBroadcastLdsDims(ldsIdx)) {
    DT_CHECK(dsCore.ss_.primaryDimToVal_st(dim) %
                     dsChunk.ss_.primaryDimToVal_st(dim) ==
                 0 &&
             "Incorrect chunk parameter.");
    numChunks *= dsCore.ss_.primaryDimToVal_st(dim) /
                 dsChunk.ss_.primaryDimToVal_st(dim);
  }

  return (long)(numChunks * chunkSizeInSticks / stickVolume);
}

// ------------------------------------------------------------------------------------------------
// entry 044/382   level 0   scc 57   13 body lines
// unit: e044_getOpReducedDimSet
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2719
// original: std::set<PrimaryDimTypes> L3DlOpsScheduler::getOpReducedDimSet( const SuperDsc &mySDsc, const DesignSpaceConfig &dsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::set<PrimaryDimTypes> e044_getOpReducedDimSet(
    const SuperDsc &mySDsc, const DesignSpaceConfig &dsc) const
{
  std::set<PrimaryDimTypes> inputLdsNonBroadcastDimSet;
  std::set<PrimaryDimTypes> outputLdsNonBroadcastDimSet;
  for (const auto &lds : dsc.labeledDs_) {
    if (isOutputLabeledDs(lds.ldsIdx_, dsc))
      for (auto dim : dsc.getNonBroadcastLdsDims(lds.ldsIdx_))
        outputLdsNonBroadcastDimSet.insert(dim);
    else
      for (auto dim : dsc.getNonBroadcastLdsDims(lds.ldsIdx_))
        inputLdsNonBroadcastDimSet.insert(dim);
  }
  return set_diff(inputLdsNonBroadcastDimSet, outputLdsNonBroadcastDimSet);
}

// ------------------------------------------------------------------------------------------------
// entry 045/382   level 0   scc 66   12 body lines
// unit: e045_addSuperChunkDataStage
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2806
// original: void L3DlOpsScheduler::addSuperChunkDataStage(DesignSpaceConfig& dsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e045_addSuperChunkDataStage(DesignSpaceConfig& dsc)
{
  DT_CHECK_MSG(dataStageSuperChunkIdx >= 0 &&
                   dsc.dataStageParam_.count(dataStageSuperChunkIdx),
               "Expect a valid SuperChunk data stage id created.");
  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageChunkIdx),
               "Expect chunk data stage.");
  const auto& dsChunk = dsc.dataStageParam_.at(dataStageChunkIdx);
  dsc2::DataStage ds = dsChunk;
  const std::string superChunkName = "superchunk";
  ds.ss_.name_ = ds.el_.name_ = superChunkName;
  dsc.dataStageParam_[dataStageSuperChunkIdx] = ds;
}

// ------------------------------------------------------------------------------------------------
// entry 046/382   level 0   scc 84   11 body lines
// unit: e046_getLxBelowBlockNode
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3468
// original: dsc2::BlockNode *L3DlOpsScheduler::getLxBelowBlockNode( dsc2::ScheduleTree &scheduleTree) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
dsc2::BlockNode *e046_getLxBelowBlockNode(
    dsc2::ScheduleTree &scheduleTree) const
{
  dsc2::BlockNode *lxBelowBlockNode = nullptr;
  for (auto schedNode : scheduleTree.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::BLOCK})) {
    if (schedNode->name_ == lxBelowBlockNodeName) {
      lxBelowBlockNode = static_cast<dsc2::BlockNode *>(schedNode);
      break;
    }
  }
  return lxBelowBlockNode;
}

// ------------------------------------------------------------------------------------------------
// entry 047/382   level 0   scc 93   25 body lines
// unit: e047_collectAllDimensionsForLoopOrder
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3991
// original: std::vector<PrimaryDimTypes> L3DlOpsScheduler::collectAllDimensionsForLoopOrder( const DesignSpaceConfig &dsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::vector<PrimaryDimTypes> e047_collectAllDimensionsForLoopOrder(
    const DesignSpaceConfig &dsc)
{
  // try to rely less on DsTypes names
  std::vector<int> ldsOrder, ldsOrderLowPriority;
  for (auto& lds : dsc.labeledDs_) {
    if (is_any_of(&lds, dsc.computeOp_.at(0).indirectAccessIndexLabeledDs))
      ldsOrderLowPriority.push_back(lds.ldsIdx_);
    else
      ldsOrder.push_back(lds.ldsIdx_);
  }
  ldsOrder.insert(ldsOrder.end(), ldsOrderLowPriority.begin(),
                  ldsOrderLowPriority.end());
  std::vector<PrimaryDimTypes> dims;
  for (const auto ldsIdx : ldsOrder) {
    for (PrimaryDimTypes dim : dsc.getLayoutDims(ldsIdx)) {
      DT_CHECK_MSG((dim != PrimaryDimTypes::IJ || dim != PrimaryDimTypes::KIJ),
                   "Do not expect combined IJ or KIJ dimensions.");
      // We only collect each dimension once.
      if (DCGUtils::isValPresent(dims, dim)) continue;

      dims.push_back(dim);
    }
  }

  return dims;
}

// ------------------------------------------------------------------------------------------------
// entry 048/382   level 0   scc 103   43 body lines
// unit: e048_getSharesAndGroupName
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4673
// original: std::pair<size_t, size_t> L3DlOpsScheduler::getSharesAndGroupName( const SuperDsc &mySDsc, const DesignSpaceConfig &dsc, const LabeledDsInfo &lds, const std::map<PrimaryDimTypes, int> &currWkSlices, const std::vector<int> &processingCoreIds)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::pair<size_t, size_t> e048_getSharesAndGroupName(
    const SuperDsc &mySDsc, const DesignSpaceConfig &dsc,
    const LabeledDsInfo &lds,
    const std::map<PrimaryDimTypes, int> &currWkSlices,
    const std::vector<int> &processingCoreIds)
{
  // Entries sorted by coreId to make unique combination of coreIds.
  std::set<int> sharesCoreIds;
  for (const auto &[coreId, wkSlices] : mySDsc.coreIdToWkSlice_) {
    // Skip if the coreId is not in the processing coreIds.
    if (!DCGUtils::isValPresent(processingCoreIds, coreId)) continue;

    bool match = true;
    // Broadcast dimension for the current labeledDs is irrelevant.
    for (PrimaryDimTypes dim : dsc.getNonBroadcastLdsDims(lds.ldsIdx_)) {
      DT_CHECK_MSG(currWkSlices.at(dim) >= 0 && wkSlices.at(dim) >= 0,
                   "Invalid wkSlice Id.");
      if (currWkSlices.at(dim) != wkSlices.at(dim)) {
        match = false;
        break;
      }
    }

    if (match) sharesCoreIds.insert(coreId);
  }
  DT_CHECK_MSG(!sharesCoreIds.empty(), "Expect at least one core in shares.");

  // Shares is the number of cores that takes the same work slices in
  // groupInfo_.
  const size_t shares = sharesCoreIds.size();
  // When shares > 1, we select the unique group name for the set of shared
  // coreIds. Otherwise, we select the default group name which is (maxGroupID +
  // 1).
  size_t groupName = dscGlobal.sysDef.maxGroupID + 1;
  if (shares > 1) {
    if (coresSetToGtrGroupNameMap.count(sharesCoreIds))
      groupName = coresSetToGtrGroupNameMap.at(sharesCoreIds);
    else {
      DT_CHECK_MSG(gtrCurrGroupName <= dscGlobal.sysDef.maxGroupID,
                   "gtr_->groupName_ exceeds the limit.");
      groupName = gtrCurrGroupName;
      coresSetToGtrGroupNameMap.emplace(sharesCoreIds, groupName);
      ++gtrCurrGroupName;
    }
  }

  return std::make_pair(shares, groupName);
}

// ------------------------------------------------------------------------------------------------
// entry 049/382   level 0   scc 105   82 body lines
// unit: e049_calculateCoreletOffsetInByte
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4842
// original: std::vector<int64_t> L3DlOpsScheduler::calculateCoreletOffsetInByte( const DesignSpaceConfig &dsc, dsc2::AllocateNode *allocNode) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::vector<int64_t> e049_calculateCoreletOffsetInByte(
    const DesignSpaceConfig &dsc, dsc2::AllocateNode *allocNode) const
{
  std::vector<int64_t> coreletOffsets(dsc.numCoreletsUsed_DSC2_, 0);
  if (dsc.numCoreletsUsed_DSC2_ < 2) return coreletOffsets;

  DT_CHECK_MSG(allocNode, "Expect a valid allocate node.");
  const auto dsNode = dsc.getSizeDataStageForNode(allocNode, allocNode).ss_;
  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageChunkIdx),
               "Expect chunk data stage.");
  const auto &dsChunk = dsc.dataStageParam_.at(dataStageChunkIdx).ss_;
  const LabeledDsInfo &lds = dsc.labeledDs_.at(allocNode->ldsIdx_);
  // Use chunk data stage to get corelet split dimensions.
  std::vector<PrimaryDimTypes> ldsCoreletSplitDim;
  for (PrimaryDimTypes dim : dsc.getNonBroadcastLdsDims(allocNode->ldsIdx_)) {
    if (dsChunk.coreletSplit_.count(dim)) ldsCoreletSplitDim.push_back(dim);
  }
  DT_CHECK_MSG(ldsCoreletSplitDim.size() <= 1,
               "Support maximal one corelet split dimension for a tensor");

  if (!ldsCoreletSplitDim.empty()) {
    const auto stickSizePerDim = dsc.getCumulativeStickSizes(lds.dsType_);
    int overallOffset = 0;
    for (int coreletId = 1; coreletId < dsc.numCoreletsUsed_DSC2_;
         ++coreletId) {
      PrimaryDimTypes coreletSplitDim = ldsCoreletSplitDim.front();
      // Compute corelet split offset by iterate from the innermost to outer
      // dimensions in layoutDimOrder_.
      int currOffset = dscGlobal.sysDef.bytesPerStick;
      for (PrimaryDimTypes dim :
           dsc.getNonBroadcastLdsDims(allocNode->ldsIdx_)) {
        const int stickSize =
            stickSizePerDim.count(dim) ? stickSizePerDim.at(dim) : 1;
        if (dim == coreletSplitDim) {
          // Use chunk data stage if the current dimension is corelet
          // splitted.
          if (allocNode->padding_.getPadding(dim) != PadType::NOPAD) {
            // This is the case that the corelet split dimension has
            // padding. Currently the implementation here only supports the
            // I dimension. The corelet offset for I dimension is calculated
            // as:
            //   offset_in_element = size_of_i * stride
            DT_CHECK_MSG(allocNode->padding_.getPadding(dim) ==
                             PadType::PADDED_FULLSPAN_WUNNEEDED,
                         "Support PadType::PADDED_FULLSPAN_WUNNEEDED only.");
            DT_CHECK_MSG(dsNode.paddingSizes_.count(dim),
                         "Expect padding sizes in chunk data stage params.");
            DT_CHECK_MSG(dim == PrimaryDimTypes::I,
                         "Support I dimension only.");
            DT_CHECK_MSG(dsNode.paddingSizes_.at(dim).stride_ > 0,
                         "Expect a valid stride.");
            currOffset *= dsChunk.coreletSplit_.at(dim).at(coreletId - 1) *
                          dsChunk.paddingSizes_.at(dim).stride_ / stickSize;
          } else {
            // No padding on the corelet split dimension.
            currOffset *=
                dsChunk.primaryDimToVal_st(dim, allocNode->component_, -1,
                                           coreletId - 1, allocNode->padding_) /
                stickSize;
          }

          // The corelet offset is fully computed when we get to the corelet
          // split dimension.
          break;
        } else {
          // Otherwise, when the current dimension is an inner dimension or
          // the corelet split dimension but without padding, the corelet
          // offset is the size of this dimension; calculated as:
          //   offset_in_element = size_of_dim
          currOffset *=
              dsNode.primaryDimToVal_st(dim, allocNode->component_, -1,
                                        coreletId - 1, allocNode->padding_) /
              stickSize;
        }
      }

      // Update offset
      overallOffset += currOffset;
      coreletOffsets[coreletId] = overallOffset;
    }
  }

  return coreletOffsets;
}

// ------------------------------------------------------------------------------------------------
// entry 050/382   level 0   scc 106   31 body lines
// unit: e050_getInitialStartAddressAndOffset
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4926
// original: std::pair<int64_t, int64_t> L3DlOpsScheduler::getInitialStartAddressAndOffset( DesignSpaceConfig &dsc, const int ldsIdx, std::deque<int64_t> coord) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::pair<int64_t, int64_t> e050_getInitialStartAddressAndOffset(
    DesignSpaceConfig &dsc, const int ldsIdx, std::deque<int64_t> coord) const
{
  const LabeledDsInfo &lds = dsc.labeledDs_.at(ldsIdx);
  DT_CHECK_MSG(lds.memOrg_.count(SenComponents::LX), "Expect LX in memOrg_.");
  int64_t startAddr = -1;
  int64_t bufferOffset = -1;
  // Get initial start address and buffer offset from allocate node.
  // Use corelet 0 to get them.
  const int corelet0Id = 0;
  coord.at(1) = corelet0Id;
  dsc2::AllocateNode *allocNode =
      lds.memOrg_.at(SenComponents::LX).allocateNode_;
  if (lds.isHbmPinned()) {
    DT_CHECK_MSG(allocNode, "Expect a valid allocate node.");
    DT_CHECK_MSG((allocNode->numBuffers_ == 1 || allocNode->numBuffers_ == 2),
                 "Expect no buffering or double buffering.");
    startAddr = allocNode->startAddressCoreCorelet_.getData(coord);
    bufferOffset =
        allocNode->bufferOffsetCoreCorelet_.at(coord.at(0)).at(corelet0Id);
  } else {
    if (allocNode != nullptr) {
      DT_CHECK_MSG((allocNode->numBuffers_ == 1), "Expect no buffering.");
      startAddr = allocNode->startAddressCoreCorelet_.getData(coord);
    } else
      DT_ERROR("Expect a valid LX allocate node.");
    // There is always only one buffer in this case, so the offset is zero.
    bufferOffset = 0;
  }
  DT_CHECK_MSG(startAddr >= 0 && bufferOffset >= 0,
               "Invalid start address or buffer offset.");
  return std::make_pair(startAddr, bufferOffset);
}

// ------------------------------------------------------------------------------------------------
// entry 051/382   level 0   scc 71   10 body lines
// unit: e051_getLdsOrConstNameOfAllocNode
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5493
// original: std::string L3DlOpsScheduler::getLdsOrConstNameOfAllocNode( DesignSpaceConfig *currDsc, dsc2::AllocateNode *anode)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::string e051_getLdsOrConstNameOfAllocNode(
    DesignSpaceConfig *currDsc, dsc2::AllocateNode *anode)
{
  if (anode->tempStorageForCompute_ != nullptr) {
    return anode->tempStorageForCompute_->name_;
  } else if (anode->ldsIdx_ >= 0) {
    return currDsc->labeledDs_.at(anode->ldsIdx_).dsName_;
  } else if (anode->constIdx_ >= 0) {
    return currDsc->constantInfo_.at(anode->constIdx_).name_;
  }
  return "";
}

// ------------------------------------------------------------------------------------------------
// entry 052/382   level 0   scc 100   17 body lines
// unit: e052_verifyLoopOrder
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6366
// original: bool L3DlOpsScheduler::verifyLoopOrder( std::vector<PrimaryDimTypes> &loopOrder)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e052_verifyLoopOrder(
    std::vector<PrimaryDimTypes> &loopOrder)
{
  bool isGood = true;

  // Check if all dimensions are unique.
  std::unordered_set<PrimaryDimTypes> visitedDims;
  for (auto dim : loopOrder) {
    if (!visitedDims.insert(dim).second) {
      if (verbose > 0)
        std::cout << "Duplicate dimension in loop order: "
                  << EnumsConversion::primaryDimToString.at(dim) << std::endl;

      isGood = false;
    }
  }

  return isGood;
}

// ------------------------------------------------------------------------------------------------
// entry 053/382   level 0   scc 118   22 body lines
// unit: e053_verifyScheduleTree
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6385
// original: bool L3DlOpsScheduler::verifyScheduleTree(const DesignSpaceConfig &dsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e053_verifyScheduleTree(const DesignSpaceConfig &dsc)
{
  if (dsc.scheduleTree_.empty()) return false;

  std::vector<const dsc2::ScheduleNode *> allScheduleNodes =
      dsc.scheduleTree_.traverseTreeDFS();

  // Check if all nodes have unique names.
  bool hasUniqueName = true;
  std::unordered_set<std::string> allNames;
  for (auto *node : allScheduleNodes) {
    bool isUnique = allNames.insert(node->name_).second;
    if (!isUnique) {
      if (verbose > 0)
        std::cout << "Schedule node does not have unique name: " << node->name_
                  << std::endl;

      hasUniqueName = false;
    }
  }

  return hasUniqueName;
}

// ------------------------------------------------------------------------------------------------
// entry 054/382   level 0   scc 119   13 body lines
// unit: e054_prepDsc
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6411
// original: void L3DlOpsScheduler::prepDsc(SuperDsc &mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e054_prepDsc(SuperDsc &mySDsc)
{
  // Update numCoreletsUsed_DSC2_.
  for (auto &dsc : mySDsc.dscs_)
    // do imbalanced corelet split in the future
    dsc.numCoreletsUsed_DSC2_ = dsc.numCoreletsUsed_;

  // Create dscMetadata entry for each DSC.
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    dscMetadata.emplace(dscIdx, Metadata());
    dscMetadata.at(dscIdx).core_dstgid = dataStageCoreIdx;
    dscMetadata.at(dscIdx).chunk_dstgid = dataStageChunkIdx;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 055/382   level 0   scc 120   94 body lines
// unit: e055_computeMinHMICoreGroupSizeForSEN1P5
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6486
// original: int L3DlOpsScheduler::computeMinHMICoreGroupSizeForSEN1P5( const SuperDsc &mySDsc, const std::vector<int> &hbmLdsIndices) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
int e055_computeMinHMICoreGroupSizeForSEN1P5(
    const SuperDsc &mySDsc, const std::vector<int> &hbmLdsIndices) const
{
  DT_CHECK_MSG(dscGlobal.sysDef.coreArch == IsaCoreGen::SEN1P5_ISA,
               "Expecting SEN1P5_ISA.");

  // Collect all allocate nodes from all DSCs.
  using DscAllocNodeMapT =
      std::unordered_map<int, std::vector<const dsc2::ScheduleNode *>>;
  DscAllocNodeMapT dscToAllocNodes;
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    const auto &dsc = mySDsc.dscs_.at(dscIdx);
    // Traverse the schedule tree to find AllocateNode with matching ldsIdx
    auto allocNodes = dsc.scheduleTree_.traverseTreeDFS(
        nullptr, {dsc2::ScheduleNode::ALLOCATE});
    dscToAllocNodes.emplace(dscIdx, allocNodes);
  }

  int minNumRequests = std::numeric_limits<int>::max();
  // For each tensor in hbmLdsIndices, collect addresses and group by HMI bits
  for (const auto ldsIdx : hbmLdsIndices) {
    // Map from core ID to address
    std::map<int, int64_t> coreToAddress;

    // Iterate through all DSCs to find allocate nodes for this tensor
    for (const auto &[dscIdx, allocNodes] : dscToAllocNodes) {
      const auto &dsc = mySDsc.dscs_.at(dscIdx);
      std::unordered_set<int> dscCoreIdsUsedSet;
      for (const auto coreId : dsc.coreIdsUsed_) {
        dscCoreIdsUsedSet.insert(coreId);
      }

      for (const auto *node : allocNodes) {
        const auto *allocNode = dynamic_cast<const dsc2::AllocateNode *>(node);
        DT_CHECK_MSG(allocNode, "Expect an allocate node.");
        if (allocNode->ldsIdx_ == ldsIdx &&
            allocNode->component_ == SenComponents::HBM) {
          // Extract addresses for each core from startAddressCoreCorelet_
          // Get all data and coordinates from the FoldManager
          auto addressData =
              allocNode->startAddressCoreCorelet_.getDataAndFoldCoordinates();

          for (const auto &[coord, address] : addressData) {
            // coord contains [coreId, coreletId, ...] based on fold dimensions
            // We need the core ID (first coordinate)
            if (!coord.empty()) {
              int coreId = static_cast<int>(coord[0]);
              // Make sure the core ID is used by the current DSC. If multiple
              // addresses per core, just pick the first one.
              if (dscCoreIdsUsedSet.count(coreId) &&
                  (coreToAddress.find(coreId) == coreToAddress.end())) {
                coreToAddress[coreId] = address;
              }
            }
          }
        }
      }
    }

    // Get the unique addresses from coreToAddress. Each unique address
    // corresponds to a work slice that requires a HMI request.
    std::unordered_set<int64_t> uniqueAddresses;
    for (const auto &[coreId, address] : coreToAddress) {
      uniqueAddresses.insert(address);
    }

    // Group unique addresses by HMI bits [39, 36, 35, 34]
    // Extract these bits and use as group key
    constexpr int bitPos39 = 39;
    constexpr int bitPos36 = 36;
    constexpr int bitPos35 = 35;
    constexpr int bitPos34 = 34;
    std::map<int, int> hmiGroupSizes;
    for (const auto &address : uniqueAddresses) {
      // Extract bits 39, 36, 35, 34 from the address
      int hmiBits = ((address >> bitPos39) & 0x1) << 3 |
                    ((address >> bitPos36) & 0x1) << 2 |
                    ((address >> bitPos35) & 0x1) << 1 |
                    ((address >> bitPos34) & 0x1);
      hmiGroupSizes[hmiBits]++;
    }

    // Find the smallest group size for this tensor
    int minGroupSize = std::numeric_limits<int>::max();
    for (const auto &[hmi, groupSize] : hmiGroupSizes) {
      minGroupSize = std::min(minGroupSize, groupSize);
    }

    // Update the overall minimum across all tensors
    if (minGroupSize < std::numeric_limits<int>::max()) {
      minNumRequests = std::min(minNumRequests, minGroupSize);
    }
  }

  return minNumRequests;
}

// ------------------------------------------------------------------------------------------------
// entry 056/382   level 0   scc 21   13 body lines
// unit: e056_isIndexLds
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6582
// original: bool L3DlOpsScheduler::isIndexLds(const LabeledDsInfo &lds) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e056_isIndexLds(const LabeledDsInfo &lds) const
{
  if (lds.memOrg_.count(SenComponents::HBM)) {
    if (const auto allocNode = lds.memOrg_.at(SenComponents::HBM).allocateNode_;
        allocNode && allocNode->indirectAllocType_ ==
                         dsc2::AllocateNode::IndirectAllocType::INDEX_TENSOR) {
      DT_CHECK_MSG(allocNode->indexTensorType_ ==
                       dsc2::AllocateNode::IndexTensorType::ADDRESS,
                   "Only index tensors of type address are supported");
      return true;
    }
  }
  return false;
}

// ------------------------------------------------------------------------------------------------
// entry 057/382   level 0   scc 112   9 body lines
// unit: e057_isPagedLds
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6596
// original: bool L3DlOpsScheduler::isPagedLds(const LabeledDsInfo &lds) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e057_isPagedLds(const LabeledDsInfo &lds) const
{
  if (lds.memOrg_.count(SenComponents::HBM)) {
    if (const auto allocNode = lds.memOrg_.at(SenComponents::HBM).allocateNode_;
        allocNode && allocNode->indirectAllocType_ ==
                         dsc2::AllocateNode::IndirectAllocType::VALUE_TENSOR)
      return true;
  }
  return false;
}

// ------------------------------------------------------------------------------------------------
// entry 058/382   level 0   scc 96   18 body lines
// unit: e058_getNewDataStageIndex
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6606
// original: int L3DlOpsScheduler::getNewDataStageIndex(SuperDsc& mySDsc, DesignSpaceConfig& dsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
int e058_getNewDataStageIndex(SuperDsc& mySDsc,
                                           DesignSpaceConfig& dsc) const
{
  int newIdx = dataStageChunkIdx + 1;
  bool found = false;
  while (!found) {
    while (dsc.dataStageParam_.count(newIdx)) ++newIdx;
    // Make sure the index is new to the other DSCs.
    bool newToOthers = true;
    for (const auto& otherDsc : mySDsc.dscs_) {
      if (&otherDsc != &dsc && otherDsc.dataStageParam_.count(newIdx)) {
        newToOthers = false;
        break;
      }
    }
    if (newToOthers) found = true;
  }
  dsc.dataStageParam_[newIdx];
  return newIdx;
}

// ------------------------------------------------------------------------------------------------
// entry 059/382   level 0   scc 61   20 body lines
// unit: e059_getPagedDimensions
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6709
// original: std::vector<PrimaryDimTypes> L3DlOpsScheduler::getPagedDimensions( const DesignSpaceConfig &dsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::vector<PrimaryDimTypes> e059_getPagedDimensions(
    const DesignSpaceConfig &dsc) const
{
  std::vector<PrimaryDimTypes> dims;
  for (const auto &lds : dsc.labeledDs_) {
    if (lds.memOrg_.count(SenComponents::HBM)) {
      const auto allocNode = lds.memOrg_.at(SenComponents::HBM).allocateNode_;
      DT_CHECK_MSG(allocNode, "Expect a valid HBM allocate node.");
      if (allocNode->indirectAllocType_ ==
          dsc2::AllocateNode::IndirectAllocType::INDEX_TENSOR) {
        // Add its dimensions.
        DT_CHECK_MSG(!allocNode->layoutDimOrder_.empty(),
                     "Expect valid layoutDimOrder_.");
        const auto pageSize = allocNode->getPageSize();
        for (const auto dim : allocNode->layoutDimOrder_) {
          if (pageSize.count(dim) && !is_any_of(dim, dims)) dims.push_back(dim);
        }
      }
    }
  }
  return dims;
}

// ------------------------------------------------------------------------------------------------
// entry 060/382   level 0   scc 133   13 body lines
// unit: e060_getHbmAllocations
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7152
// original: std::vector<dsc2::AllocateNode *> L3DlOpsScheduler::getHbmAllocations( const DesignSpaceConfig &dsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::vector<dsc2::AllocateNode *> e060_getHbmAllocations(
    const DesignSpaceConfig &dsc) const
{
  std::vector<dsc2::AllocateNode *> hbmAllocations;
  for (const auto &lds : dsc.labeledDs_) {
    if (lds.isHbmPinned()) {
      DT_CHECK_MSG(lds.memOrg_.count(SenComponents::HBM),
                   "Expect HBM in memOrg_.");
      auto allocNode = lds.memOrg_.at(SenComponents::HBM).allocateNode_;
      DT_CHECK_MSG(allocNode, "Expect a valid allocate node.");
      hbmAllocations.push_back(allocNode);
    }
  }
  return hbmAllocations;
}

// ------------------------------------------------------------------------------------------------
// entry 061/382   level 0   scc 135   12 body lines
// unit: e061_gatherFoldParams
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7248
// original: void L3DlOpsScheduler::gatherFoldParams( const FoldManager<CoordinateBaseType> &cfm, std::vector<dsc2::FoldParamInfoType> &foldParams) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e061_gatherFoldParams(
    const FoldManager<CoordinateBaseType> &cfm,
    std::vector<dsc2::FoldParamInfoType> &foldParams) const
{
  // The following const_cast is a temporary workaround. Some required methods
  // in Foldmanager (getNumDims for example) are not marked as const.
  FoldManager<CoordinateBaseType> &fm =
      const_cast<FoldManager<CoordinateBaseType> &>(cfm);
  for (int i = 0; i < fm.getNumDims(); ++i) {
    CoordinateBaseType alpha, beta;
    fm.getAlphaBeta(alpha, beta, i);
    foldParams.push_back(
        {alpha, beta, fm.getFoldDimSize(i), fm.getFoldDimProp(i)->Label()});
  }
}

// ------------------------------------------------------------------------------------------------
// entry 062/382   level 0   scc 137   38 body lines
// unit: e062_getEnclosingLoopsAndRelatedDims
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7263
// original: void L3DlOpsScheduler::getEnclosingLoopsAndRelatedDims( dsc2::ScheduleNode *node, const DesignSpaceConfig *dsc, std::vector<dsc2::LoopNode *> &loopChain, std::unordered_set<PrimaryDimAndKind> &relatedDims) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e062_getEnclosingLoopsAndRelatedDims(
    dsc2::ScheduleNode *node, const DesignSpaceConfig *dsc,
    std::vector<dsc2::LoopNode *> &loopChain,
    std::unordered_set<PrimaryDimAndKind> &relatedDims) const
{
  auto *nodeOwnerLoop = node->getMutableOwnerLoop();
  auto *loop = nodeOwnerLoop;
  while (loop != nullptr) {
    loopChain.push_back(loop);
    relatedDims.insert(loop->dims_.begin(), loop->dims_.end());
    loop = loop->getMutableOwnerLoop();
  }
  // Exclude the root loop
  loopChain.pop_back();

  // Include dimensions from the allocateNode.
  if (auto allocNode = dynamic_cast<const dsc2::AllocateNode *>(node)) {
    relatedDims.insert(allocNode->layoutDimOrder_.begin(),
                       allocNode->layoutDimOrder_.end());
  } else if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
    const dsc2::TransferNode *transferNode =
        static_cast<const dsc2::TransferNode *>(node);
    std::vector<PrimaryDimTypes> transferDims;

    if (transferNode->srcLdsAndLoopOffsets_.myLdsIdx_ != -1) {
      transferDims =
          dsc->getLayoutDims(transferNode->srcLdsAndLoopOffsets_.myLdsIdx_);
    } else {
      for (auto &di : transferNode->dstLdsAndLoopOffsets_) {
        if (di.myLdsIdx_ != -1) {
          transferDims = dsc->getLayoutDims(di.myLdsIdx_);
          break;
        }
      }
    }
    relatedDims.insert(transferDims.begin(), transferDims.end());
  } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    // TO DO: ComputeNode's dimset depends on which input or output is being
    // processed.
  }
  return;
}

// ------------------------------------------------------------------------------------------------
// entry 063/382   level 0   scc 136   21 body lines
// unit: e063_findAndStoreLoopWithDim
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7307
// original: void L3DlOpsScheduler::findAndStoreLoopWithDim( DesignSpaceConfig *currDsc, const PrimaryDimAndKind dimToFind, dsc2::LoopNode *loop, const std::unordered_set<PrimaryDimAndKind> &relatedDims, dsc2::VectorOfLoopAndDim &relatedLoops, PadType accessPadType) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e063_findAndStoreLoopWithDim(
    DesignSpaceConfig *currDsc, const PrimaryDimAndKind dimToFind,
    dsc2::LoopNode *loop,
    const std::unordered_set<PrimaryDimAndKind> &relatedDims,
    dsc2::VectorOfLoopAndDim &relatedLoops, PadType accessPadType) const
{
  for (auto dimAndKind : loop->dims_) {
    // find out loops which contain dim or its related dim.
    if (dimAndKind.dim_ == dimToFind) {
      // general case: when dim is found in loop, record it.
      relatedLoops.emplace_back(
          loop, dimAndKind,
          dsc2::LoopDistributionInfo::LoopDistributionCat::ABOVE_CHUNK);
    } else if (accessPadType != PadType::NOPAD &&
               dimAndKind.kind_ == MetaDimKind::WindowDim) {
      auto &padding =
          currDsc->dataStageParam_.at(loop->denId_).ss_.paddingSizes_;
      if (padding.count(dimToFind.dim_) &&
          padding.at(dimToFind.dim_).windowDim_ == dimAndKind.dim_) {
        relatedLoops.emplace_back(
            loop, dimAndKind,
            dsc2::LoopDistributionInfo::LoopDistributionCat::ABOVE_CHUNK);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 064/382   level 0   scc 139   11 body lines
// unit: e064_constructDatastage
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7724
// original: int L3DlOpsScheduler::constructDatastage(DesignSpaceConfig *currDsc, dsc2::DataStage &refDataStage) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
int e064_constructDatastage(DesignSpaceConfig *currDsc,
                                         dsc2::DataStage &refDataStage) const
{
  auto id = currDsc->dataStageParam_.size();
  while (currDsc->dataStageParam_.count(id)) id++;
  auto &newDs = currDsc->dataStageParam_[id];
  newDs = refDataStage;
  newDs.ss_.name_ = std::to_string(id);
  newDs.el_.name_ = std::to_string(id) + "el";
  // Do we need to update the metadata info?
  // metadata.datastages_[id] = metadata.Datastage version of newDs;
  return id;
}

// ------------------------------------------------------------------------------------------------
// entry 065/382   level 0   scc 140   18 body lines
// unit: e065_constructLoopNode
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7737
// original: dsc2::LoopNode *L3DlOpsScheduler::constructLoopNode( int numId, int denId, std::vector<PrimaryDimAndKind> dims) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
dsc2::LoopNode *e065_constructLoopNode(
    int numId, int denId, std::vector<PrimaryDimAndKind> dims) const
{
  if (dims.empty()) {
    DT_ERROR("Cannot construct loop with no dimensions");
  }
  dsc2::LoopNode *newLoopNode = new dsc2::LoopNode();
  newLoopNode->numId_ = numId;
  newLoopNode->denId_ = denId;
  newLoopNode->dims_ = dims;

  newLoopNode->name_ = "loop_ds" + std::to_string(newLoopNode->numId_) + "_ds" +
                       std::to_string(newLoopNode->denId_);
  for (const auto &entry : newLoopNode->dims_) {
    newLoopNode->name_ +=
        "_" + EnumsConversion::primaryDimToString.at(entry.dim_);
  }

  return newLoopNode;
}

// ------------------------------------------------------------------------------------------------
// entry 066/382   level 0   scc 80   4 body lines
// unit: e066_addCore
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:29
// original: void addCore(const int coreId, const int slice)
// class: CrossCoreReductionGroup
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e066_addCore(const int coreId, const int slice)
{
    coreIds.resize(slice + 1, -1);
    coreIds.at(slice) = coreId;
  }

// ------------------------------------------------------------------------------------------------
// entry 067/382   level 0   scc 147   9 body lines
// unit: e067_getStartCoreAtCorelet
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:34
// original: int getStartCoreAtCorelet(int coreletId) const
// class: CrossCoreReductionGroup
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
int e067_getStartCoreAtCorelet(int coreletId) const
{
    DT_CHECK(!coreIds.empty());
    if (coreletId == 0)
      return coreIds.front();
    else if (coreletId == 1)
      return coreIds.back();
    else
      DT_ERROR("Unknown corelet id.");
  }

// ------------------------------------------------------------------------------------------------
// entry 068/382   level 0   scc 82   9 body lines
// unit: e068_getEndCoreAtCorelet
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:43
// original: int getEndCoreAtCorelet(int coreletId) const
// class: CrossCoreReductionGroup
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
int e068_getEndCoreAtCorelet(int coreletId) const
{
    DT_CHECK(!coreIds.empty());
    if (coreletId == 0)
      return coreIds.back();
    else if (coreletId == 1)
      return coreIds.front();
    else
      DT_ERROR("Unknown corelet id.");
  }

// ------------------------------------------------------------------------------------------------
// entry 069/382   level 0   scc 148   3 body lines
// unit: e069_updateMin
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:117
// original: inline void updateMin(float newVal)
// class: Constraints
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e069_updateMin(float newVal)
{
          min_ = min_ ? std::max(*min_, newVal) : newVal;
        }

// ------------------------------------------------------------------------------------------------
// entry 070/382   level 0   scc 149   3 body lines
// unit: e070_updateMax
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:120
// original: inline void updateMax(float newVal)
// class: Constraints
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e070_updateMax(float newVal)
{
          max_ = max_ ? std::min(*max_, newVal) : newVal;
        }

// ------------------------------------------------------------------------------------------------
// entry 071/382   level 0   scc 150   3 body lines
// unit: e071_updateValues
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:123
// original: inline void updateValues(std::set<float> newVals)
// class: Constraints
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e071_updateValues(std::set<float> newVals)
{
          values_ = values_ ? set_intersect(*values_, newVals) : newVals;
        }

// ------------------------------------------------------------------------------------------------
// entry 072/382   level 0   scc 69   9 body lines
// unit: e072_getTripCount
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.h:487
// original: int getTripCount(const DesignSpaceConfig &dsc, const PrimaryDimTypes dim, const int dataStageNumId, const int dataStageDenId) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
int e072_getTripCount(const DesignSpaceConfig &dsc, const PrimaryDimTypes dim,
                   const int dataStageNumId, const int dataStageDenId) const
{
    auto num =
        dsc.dataStageParam_.at(dataStageNumId).ss_.primaryDimToVal_st(dim);
    auto den =
        dsc.dataStageParam_.at(dataStageDenId).ss_.primaryDimToVal_st(dim);
    int tripCount =
        std::ceil(static_cast<double>(num) / static_cast<double>(den));
    return tripCount;
  }

// ------------------------------------------------------------------------------------------------
// entry 197/382   level 1   scc 3   15 body lines
// unit: e197_getCoreletSplitDimensions
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:88
// original: static std::vector<PrimaryDimTypes> getCoreletSplitDimensions( const DesignSpaceConfig &dsc)
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::vector<PrimaryDimTypes> e197_getCoreletSplitDimensions(
    const DesignSpaceConfig &dsc)
{
  std::vector<PrimaryDimTypes> dims;
  if (dsc.numCoreletsUsed_ > 1) {
    for (const auto &entry : EnumsConversion::primaryDimToString) {
      if (is_any_of(entry.first, PrimaryDimTypes::IJ, PrimaryDimTypes::KIJ,
                    PrimaryDimTypes::PrimaryDimTypesCount))
        continue;

      if (L3DlOpsScheduler::isDimensionCoreletSplit(dsc, entry.first))
        dims.push_back(entry.first);
    }
  }

  return dims;
}

// ------------------------------------------------------------------------------------------------
// entry 198/382   level 1   scc 8   6 body lines
// unit: e198_addOrUpdatePaddingSizesInChunkParams
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:150
// original: static void addOrUpdatePaddingSizesInChunkParams( DataStructDims &chunkParams, const DataStructDims &coreParams)
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e198_addOrUpdatePaddingSizesInChunkParams(
    DataStructDims &chunkParams, const DataStructDims &coreParams)
{
  DT_CHECK_MSG((!chunkParams.empty() && !coreParams.empty()),
               "Expect non-empty data-stage parameters.");
  chunkParams.paddingSizes_ = coreParams.paddingSizes_;
  voidPaddingIfChunking(chunkParams, coreParams);
}

// ------------------------------------------------------------------------------------------------
// entry 199/382   level 1   scc 11   49 body lines
// unit: e199_getLabeledDsNumOfWkSlices
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:219
// original: static unsigned getLabeledDsNumOfWkSlices(const SuperDsc &mySDsc, const int ldsIdx, const std::vector<int> &dscIndices)
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
unsigned e199_getLabeledDsNumOfWkSlices(const SuperDsc &mySDsc,
                                          const int ldsIdx,
                                          const std::vector<int> &dscIndices)
{
  DT_CHECK_MSG(!dscIndices.empty(), "Expect valid DSCs.");
  // Use the first DSC to get labeledDsInfo.
  const int dscMainIdx = dscIndices[0];
  const DesignSpaceConfig &dscMain = mySDsc.dscs_.at(dscMainIdx);

  // If all DSCs in SuperDsc are specified, we can simply compute using the
  // numWkSlicesPerDim_.
  if (dscIndices.size() == mySDsc.dscs_.size()) {
    unsigned numWkSlices = 1;
    // For the broadcast dimension for the current labeledDs, the number of
    // slices with regard to this dimension is one.
    for (PrimaryDimTypes dim : dscMain.getNonBroadcastLdsDims(ldsIdx))
      numWkSlices *= mySDsc.numWkSlicesPerDim_.at(dim);

    return numWkSlices;
  }

  // All coreIds in the specified DSCs.
  std::unordered_set<int> processingCoreIds;
  for (int dscIdx : dscIndices) {
    const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
    for (int coreId : dsc.coreIdsUsed_) processingCoreIds.insert(coreId);
  }

  // Use a map to represent each unique work slice.
  std::set<std::map<PrimaryDimTypes, int>> workSlices;
  for (int coreId : processingCoreIds) {
    std::map<PrimaryDimTypes, int> wkSlice;
    for (PrimaryDimTypes dim : dscMain.getLayoutDims(ldsIdx)) {
      int wkSliceId = -1;
      if (isLabeledDsDimensionBroadcast(dscMain, dscMain.labeledDs_.at(ldsIdx),
                                        dim))
        // If the current dimension is broadcast for the current labeledDs,
        // there is only one work slice with regard to this dimension.
        // Therefore, its work slice id is irrelevant to the value in
        // coreIdToWkSlice_. Here we use id 0.
        wkSliceId = 0;
      else
        wkSliceId = mySDsc.coreIdToWkSlice_.at(coreId).at(dim);

      wkSlice.emplace(dim, wkSliceId);
    }

    if (!workSlices.count(wkSlice)) workSlices.insert(wkSlice);
  }

  return workSlices.size();
}

// ------------------------------------------------------------------------------------------------
// entry 200/382   level 1   scc 20   7 body lines
// unit: e200_getLxNeighborLabeledDsIndicesSet
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:396
// original: std::unordered_set<int> L3DlOpsScheduler::getLxNeighborLabeledDsIndicesSet( const SuperDsc& mySDsc, const DesignSpaceConfig& dsc, const int dscIdx) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::unordered_set<int> e200_getLxNeighborLabeledDsIndicesSet(
    const SuperDsc& mySDsc, const DesignSpaceConfig& dsc,
    const int dscIdx) const
{
  std::unordered_set<int> indices;
  for (const auto& lds : dsc.labeledDs_) {
    if (isLabeledDsLXNeighbor(mySDsc, dscIdx, lds)) indices.insert(lds.ldsIdx_);
  }
  return indices;
}

// ------------------------------------------------------------------------------------------------
// entry 201/382   level 1   scc 22   55 body lines
// unit: e201_computeLdsAllocateSiblingLoopNode
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:435
// original: dsc2::BlockNode *L3DlOpsScheduler::computeLdsAllocateSiblingLoopNode( const SuperDsc &mySDsc, const int dscIdx, const LabeledDsInfo &lds, const dsc2::BlockNode *startNode, const std::vector<const dsc2::LoopNode *> &parentInnerToOuterLoopNodes) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
dsc2::BlockNode *e201_computeLdsAllocateSiblingLoopNode(
    const SuperDsc &mySDsc, const int dscIdx, const LabeledDsInfo &lds,
    const dsc2::BlockNode *startNode,
    const std::vector<const dsc2::LoopNode *> &parentInnerToOuterLoopNodes)
    const
{
  DT_CHECK_MSG(startNode && startNode->name_ == lxBelowBlockNodeName,
               "Expect the lx-below block node.");
  const auto &dsc = mySDsc.dscs_.at(dscIdx);
  dsc2::BlockNode *siblingNode = nullptr;
  if (lds.isHbmPinned()) {
    // For HBM tensor, put the allocate node inside the outermost loop that has
    // its related dimensions.
    DT_CHECK_MSG(!isIndexLds(lds), "Do not expect index tensor.");
    const auto ldsNonBroadcastDimSet =
        dsc.getNonBroadcastLdsDimSet(lds.ldsIdx_);
    if (lxBufferType == BufferType::SPATIAL_DOUBLE) {
      // For spatial-double buffering, the allocate node is inside a
      // core/SuperChunk loop.
      for (auto loopNode : parentInnerToOuterLoopNodes) {
        if (auto parentLoop = loopNode->getOwnerLoop();
            loopNode->numId_ == dataStageSuperChunkIdx &&
            loopNode->denId_ == dataStageChunkIdx &&
            parentLoop->numId_ == dataStageCoreIdx &&
            parentLoop->denId_ == dataStageSuperChunkIdx) {
          siblingNode = const_cast<dsc2::LoopNode*>(loopNode);
        } else if (loopNode->numId_ == dataStageCoreIdx &&
                   loopNode->denId_ == dataStageSuperChunkIdx) {
          DT_CHECK_MSG(loopNode->dims_.size() == 1,
                       "Currently only support one dimension in a loop node.");
          const auto dim = loopNode->dims_.front().dim_;
          if (ldsNonBroadcastDimSet.count(dim)) break;
          siblingNode = const_cast<dsc2::LoopNode*>(loopNode);
        }
      }
    } else {
      // For double buffering, the allocate node is inside a core/chunk loop.
      siblingNode = const_cast<dsc2::BlockNode*>(startNode);
      for (auto loopNode : parentInnerToOuterLoopNodes) {
        DT_CHECK_MSG(loopNode->numId_ == dataStageCoreIdx &&
                         loopNode->denId_ == dataStageChunkIdx,
                     "Expect a core-by-chunk loop.");
        DT_CHECK_MSG(loopNode->dims_.size() == 1,
                     "Currently only support one dimension in a loop node.");
        const auto dim = loopNode->dims_.front().dim_;
        if (ldsNonBroadcastDimSet.count(dim)) break;
        siblingNode = const_cast<dsc2::LoopNode*>(loopNode);
      }
    }
  } else if (isLabeledDsLXNeighbor(mySDsc, dscIdx, lds)) {
    // For LX-neighbor, put the allocate node inside the innermost loop.
    siblingNode = const_cast<dsc2::BlockNode *>(startNode);
  } else if (lds.isLxPinned()) {
    // For LX-local, put the allocate node outside the outermost loop.
    siblingNode =
        const_cast<dsc2::LoopNode *>(parentInnerToOuterLoopNodes.back());
  }

  return siblingNode;
}

// ------------------------------------------------------------------------------------------------
// entry 202/382   level 1   scc 23   32 body lines
// unit: e202_computeLdsTransferSiblingLoopNode
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:495
// original: dsc2::BlockNode *L3DlOpsScheduler::computeLdsTransferSiblingLoopNode( const SuperDsc &mySDsc, const int dscIdx, const LabeledDsInfo &lds, const dsc2::BlockNode *startNode, const std::vector<const dsc2::LoopNode *> &parentInnerToOuterLoopNodes) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
dsc2::BlockNode *e202_computeLdsTransferSiblingLoopNode(
    const SuperDsc &mySDsc, const int dscIdx, const LabeledDsInfo &lds,
    const dsc2::BlockNode *startNode,
    const std::vector<const dsc2::LoopNode *> &parentInnerToOuterLoopNodes)
    const
{
  DT_CHECK_MSG(startNode && startNode->name_ == lxBelowBlockNodeName,
               "Expect the lx-below block node.");
  const auto &dsc = mySDsc.dscs_.at(dscIdx);
  dsc2::BlockNode *siblingNode = nullptr;
  if (lds.isHbmPinned()) {
    // For HBM tensor, put the transfer node inside the outermost loop that has
    // its related dimensions.
    DT_CHECK_MSG(!isIndexLds(lds), "Do not expect index tensor.");
    const auto ldsNonBroadcastDimSet =
        dsc.getNonBroadcastLdsDimSet(lds.ldsIdx_);
    siblingNode = const_cast<dsc2::BlockNode*>(startNode);
    for (auto loopNode : parentInnerToOuterLoopNodes) {
      if (loopNode->denId_ == dataStageChunkIdx) {
        DT_CHECK_MSG(loopNode->dims_.size() == 1,
                     "Currently only support one dimension in a loop node.");
        const auto dim = loopNode->dims_.front().dim_;
        if (ldsNonBroadcastDimSet.count(dim)) break;
        siblingNode = const_cast<dsc2::LoopNode*>(loopNode);
      }
    }
  } else if (isLabeledDsLXNeighbor(mySDsc, dscIdx, lds)) {
    // For LX-neighbor, put the transfer node inside the innermost loop.
    siblingNode = const_cast<dsc2::BlockNode *>(startNode);
  } else if (lds.isLxPinned()) {
    // For LX-local, put the transfer node outside the outermost loop.
    siblingNode =
        const_cast<dsc2::LoopNode *>(parentInnerToOuterLoopNodes.back());
  }

  return siblingNode;
}

// ------------------------------------------------------------------------------------------------
// entry 203/382   level 1   scc 31   49 body lines
// unit: e203_getOpFuncDataFormat
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:670
// original: std::string L3DlOpsScheduler::getOpFuncDataFormat( const DesignSpaceConfig &dsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::string e203_getOpFuncDataFormat(
    const DesignSpaceConfig &dsc)
{
  OpFuncs name = getOpFuncName(dsc);
  switch (name) {
    case OpFuncs::CONV2D_INT4_FWD:
    case OpFuncs::CONV2D_INT4_FWD_GENKG3:
    case OpFuncs::CONV2D_INT4_FWD_SPARSEKG3:
    case OpFuncs::BATCHMATMUL_INT4_FWD:
    case OpFuncs::BATCHMATMUL_INT4_FWD_SPARSEKG3:
    case OpFuncs::BATCHMATMUL_XRF_INT4_FWD:
    case OpFuncs::BATCHMATMUL_XRFCH_INT4_FWD:
    case OpFuncs::CSQ_INT4:
    case OpFuncs::CSQ_INT4_WT:
    case OpFuncs::CSQ_INT4_CHIL:
      return "int4";

    case OpFuncs::CONV2D_INT8_FWD:
    case OpFuncs::CONV2D_INT8_FWD_GENKG3:
    case OpFuncs::CONV2D_INT8_FWD_SPARSEKG3:
    case OpFuncs::CONV2D_INT8_FWD_OS1:
    case OpFuncs::CONV2D_XRF_INT8_FWD_OS1:
    case OpFuncs::BATCHMATMUL_INT8_FWD:
    case OpFuncs::BATCHMATMUL_INT8_FWD_MBKG3:
    case OpFuncs::BATCHMATMUL_INT8_FWD_SPARSEKG3:
    case OpFuncs::BATCHMATMUL_XRF_INT8_FWD:
    case OpFuncs::BATCHMATMUL_XRFCH_INT8_FWD:
    case OpFuncs::CSQ_INT8:
    case OpFuncs::CSQ_INT8_CH:
    case OpFuncs::CSQ_INT8_WT:
    case OpFuncs::CSQ_INT8_CHIL:
    case OpFuncs::CSQ_INT8_MB:
      return "int8";

    case OpFuncs::CONV2D_FP8_FWD:
    case OpFuncs::CONV2D_FP8_FWD_GENKG3:
    case OpFuncs::CONV2D_FP8_FWD_SPARSEKG3:
    case OpFuncs::BATCHMATMUL_FP8_FWD:
    case OpFuncs::BATCHMATMUL_FP8_FWD_SPARSEKG3:
    case OpFuncs::BATCHMATMUL_XRF_FP8_FWD:
    case OpFuncs::BATCHMATMUL_XRFCH_FP8_FWD:
    case OpFuncs::Q_FP8:
    case OpFuncs::Q_FP8_CH:
    case OpFuncs::Q_FP8_WT:
    case OpFuncs::Q_FP8_CHIL:
      return "fp8";

    default:
      return "fp16";
  }
}

// ------------------------------------------------------------------------------------------------
// entry 204/382   level 1   scc 35   15 body lines
// unit: e204_isOpFuncConv2d
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:753
// original: bool L3DlOpsScheduler::isOpFuncConv2d(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e204_isOpFuncConv2d(const OpFuncs opFuncName) const
{
  static const std::unordered_set<OpFuncs> opFuncConv2dOthers = {
      OpFuncs::CONV2D_FWD,
      OpFuncs::CONV2D_FP8_FWD,
      OpFuncs::CONV2D_INT8_FWD,
      OpFuncs::CONV2D_FWD_GENKG3,
      OpFuncs::CONV2D_FP8_FWD_GENKG3,
      OpFuncs::CONV2D_INT8_FWD_GENKG3,
      OpFuncs::CONV2D_FWD_SPARSEKG3,
      OpFuncs::CONV2D_FP8_FWD_SPARSEKG3,
      OpFuncs::CONV2D_INT8_FWD_SPARSEKG3};

  return isOpFuncConv2dInt4(opFuncName) || isOpFuncConv2dOs1(opFuncName) ||
         (opFuncConv2dOthers.count(opFuncName) > 0);
}

// ------------------------------------------------------------------------------------------------
// entry 205/382   level 1   scc 41   5 body lines
// unit: e205_isOpFuncBmm
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:804
// original: bool L3DlOpsScheduler::isOpFuncBmm(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e205_isOpFuncBmm(const OpFuncs opFuncName) const
{
  return isOpFuncBmmFp16(opFuncName) || isOpFuncBmmFp8Xrf(opFuncName) ||
         isOpFuncBmmFp8NonXrf(opFuncName) || isOpFuncBmmInt4(opFuncName) ||
         isOpFuncBmmInt8(opFuncName);
}

// ------------------------------------------------------------------------------------------------
// entry 206/382   level 1   scc 51   80 body lines
// unit: e206_getMinParamBmm
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:925
// original: long L3DlOpsScheduler::getMinParamBmm(const DesignSpaceConfig& dsc, const PrimaryDimTypes dim, const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
long e206_getMinParamBmm(const DesignSpaceConfig& dsc,
                                      const PrimaryDimTypes dim,
                                      const OpFuncs opFuncName) const
{
  constexpr long defaultParam = 1;
  const auto coreParam =
      dsc.dataStageParam_.at(dataStageCoreIdx).ss_.primaryDimToVal_st(dim);

  auto& inp0_ldo =
      dsc.primaryDsInfo_.at(dsc.labeledDs_.front().dsType_).layoutDimOrder_;
  auto& inp1_ldo =
      dsc.primaryDsInfo_.at(dsc.labeledDs_.at(1).dsType_).layoutDimOrder_;
  auto& out_ldo =
      dsc.primaryDsInfo_.at(dsc.labeledDs_.back().dsType_).layoutDimOrder_;
  std::set<PrimaryDimTypes> inp0_reuse_dim;
  for (auto& cdim : inp1_ldo) {
    if ((std::find(out_ldo.begin(), out_ldo.end(), cdim) != out_ldo.end()) &&
        (std::find(inp0_ldo.begin(), inp0_ldo.end(), cdim) == inp0_ldo.end())) {
      inp0_reuse_dim.insert(cdim);
    }
  }
  DT_CHECK(inp0_reuse_dim.size() == 1);
  std::set<PrimaryDimTypes> inp1_reuse_dim;
  for (auto& cdim : inp0_ldo) {
    if (std::find(out_ldo.begin(), out_ldo.end(), cdim) != out_ldo.end() &&
        (std::find(inp1_ldo.begin(), inp1_ldo.end(), cdim) == inp1_ldo.end())) {
      inp1_reuse_dim.insert(cdim);
    }
  }
  std::set<PrimaryDimTypes> out_reuse_dim;
  for (auto& cdim : inp0_ldo) {
    if (std::find(inp1_ldo.begin(), inp1_ldo.end(), cdim) != inp1_ldo.end() &&
        (std::find(out_ldo.begin(), out_ldo.end(), cdim) == out_ldo.end())) {
      out_reuse_dim.insert(cdim);
    }
  }
  DT_CHECK(out_reuse_dim.size() == 1);
  if (dim == (*out_reuse_dim.begin())) {
    // this is input channel
    if (isOpFuncBmmInt4(opFuncName)) {
      return 128;
    } else if (isOpFuncBmmInt8(opFuncName) ||
               isOpFuncBmmFp8NonXrf(opFuncName)) {
      if (coreParam % 256 == 0) {
        return 256;
      } else if (coreParam % 128 == 0) {
        return 128;
      } else {
        return 64;
      }
    } else if (isOpFuncBmmFp16(opFuncName)) {
      if (dscGlobal.sysDef.coreArch > IsaCoreGen::RCUDD1A_ISA &&
          coreParam >= 1024 && coreParam % 256 == 0) {
        return 256;
      } else if (coreParam % 128 == 0) {
        return 128;
      } else {
        return 64;
      }
    } else if (isOpFuncBmmFp8Xrf(opFuncName)) {
      return 64;
    } else {
      return defaultParam;
    }
  } else if (dim == (*inp0_reuse_dim.begin())) {
    // this is output channel
    return 64;
  } else if (dim == (*inp1_reuse_dim.begin())) {
    // weight reuse dims
    // consider below-lx xrf-reuse in setting the min value
    constexpr int maxReuseNeeded = 64;  // 80% util for int8
    for (int reuse = maxReuseNeeded; reuse > 0; reuse--) {
      if (coreParam % reuse == 0) {
        return reuse;
      }
    }
    return defaultParam;
  } else if (coreParam < 0) {
    return coreParam;
  } else {
    return defaultParam;
    }
}

// ------------------------------------------------------------------------------------------------
// entry 207/382   level 1   scc 62   204 body lines
// unit: e207_generateDscParamCandidates
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1172
// original: auto L3DlOpsScheduler::generateDscParamCandidates( const SuperDsc &mySDsc, const std::vector<DataStructDims> &dscParams, const std::vector<PrimaryDimTypes> &primaryDims, const std::unordered_set<PrimaryDimTypes> &chunkDims, const std::unordered_set<PrimaryDimTypes> &coreSplitDims) -> DscParamCandidatesType
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
auto e207_generateDscParamCandidates(
    const SuperDsc &mySDsc, const std::vector<DataStructDims> &dscParams,
    const std::vector<PrimaryDimTypes> &primaryDims,
    const std::unordered_set<PrimaryDimTypes> &chunkDims,
    const std::unordered_set<PrimaryDimTypes> &coreSplitDims)
    -> DscParamCandidatesType
{
  DscParamCandidatesType dscCandidates(mySDsc.dscs_.size());

  for (auto dim : primaryDims) {
    // When it is not a chunk dimension, the only candidate value is its CoreD
    // value.
    if (!chunkDims.count(dim)) {
      for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
        const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
        DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
                     "Expect dataStageParam_ entry for the core data stage.");
        dscCandidates[dscIdx][dim] = {dsc.dataStageParam_.at(dataStageCoreIdx)
                                          .ss_.primaryDimToVal_st(dim)};
      }
      continue;
    }

    // Generate candidate values for all chunk dimensions.
    for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
      const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
      DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
                   "Expect dataStageParam_ entry for the core data stage.");
      const auto &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);
      const long lBound = dscParams.at(dscIdx).primaryDimToVal_st(dim);
      // Currently we are missing the transformation to break a block transfer
      // with symbolic size in ALxS, so for now we force chunking of symbolic
      // dim if there is a transfer LX->HBM related to that dim. Transformation
      // is available in DDC, it needs to be made common
      const auto &outputLds = dsc.labeledDs_.back();
      const bool chunkSymbolicDim =
          outputLds.isHbmPinned() &&
          dsc.getNonBroadcastLdsDimSet(outputLds.ldsIdx_).count(dim);
      double scaleDownFactor = 1.0;
      if (outputLds.scaledLdsCategory_ ==
              LabeledDsInfo::ScaledLdsCategory::SCALE_TENSOR &&
          outputLds.mxInfo_.dim == dim) {
        scaleDownFactor = ((double)1.0) / outputLds.mxInfo_.blkSize;
      }

      const long uBound = dsCore.ss_.primaryDimToVal_st(
          dim, SenComponents::NO_COMPONENT, -1, -1, {}, scaleDownFactor,
          chunkSymbolicDim);
      DT_CHECK_MSG((lBound > 0 && uBound >= lBound),
                   "Expect valid lower and upper bound parameters.");
      const auto pagedDims = getPagedDimensions(dsc);

      if (lBound == uBound) {
        // When lBound == uBound, there is effectively no chunking on this
        // dimension with only one candidate being the CoreD value. Hence,
        // this is a shortcut to get the candidate without doing extra checks,
        // with the assumption that the CoreD value meets all requirements.
        // Another benefit of this shortcut is that, getMinParamForDim may
        // enforce no chunking on a specific dimension by setting its lBound
        // == uBound. This shortcut save us from doing complicated checks. For
        // example, in the case of padding for the I and J dimension, we need
        // to use the R and C dimensions instead. But depending on whether I/J
        // is used as windowed or padded-only (in the case of CSQ computeOp)
        // dimension, the R/C parameters are derived differently, which makes
        // it complicated to check the I/J parameter. Therefore, in the case
        // of no chunking on I or J dimension, getMinParamForDim enforce the
        // chunk parameter equal to its CoreD value.
        dscCandidates[dscIdx][dim] = {lBound};

        // Verify that the CoreD value meets all requirements.
        for (const auto &lds : dsc.labeledDs_) {
          if (isIndexLds(lds)) continue;
          auto ss = getStickSize(dsc, lds.dsType_, dim);
          auto &lxMemOrg = lds.memOrg_.at(SenComponents::LX);
          auto &lxAlloc = lxMemOrg.allocateNode_;
          auto ubound_lds = dsCore.ss_.primaryDimToVal_st(
              dim, SenComponents::NO_COMPONENT, -1, -1, lxAlloc->padding_);
          DT_CHECK_MSG((ubound_lds % ss == 0),
                       "Expect parameter value is multiple of stick size.");
        }
      } else {
        for (long param = lBound; param <= uBound; ++param) {
          // We support equal chunk size only, meaning that the uBound must be
          // multiple of the chunk parameter value.
          if (uBound % param == 0) {
            // If the current dimension is symbolic, the candidate value must
            // meet the requirements below.
            auto isSymbolicValid = [&dim, &dsc, &dsCore, &uBound,
                                    &scaleDownFactor](long param) {
              const bool isSymbolicDimDsCore =
                  dsCore.ss_.symbolicDimInfo_.count(dim);
              if (isSymbolicDimDsCore) {
                // The valid value must be a divisor of the dimension's
                // granularity or max size (which is uBound).
                if (param < uBound) {
                  const auto &granularity = dsCore.ss_.primaryDimToVal_st(
                      dim, SenComponents::NO_COMPONENT, -1, -1, {},
                      scaleDownFactor,
                      /* getSymbolicGranularity */ true);
                  if (param > granularity || (granularity % param != 0))
                    return false;
                }
              }
              return true;
            };
            if (!isSymbolicValid(param)) continue;

            // If the current dimension is paged, the candidate value must meet
            // the requirements below.
            auto isPageValid = [&dim, &param, &dsc, this]() {
              // The value must be the size of one or multiple of a page size.
              const auto onePageSize =
                  dsc.dataStageParam_.at(this->dataStageOnePageIdx)
                      .ss_.primaryDimToVal_st(dim);
              if (param % onePageSize != 0) return false;
              // The value must be a divisor of the steady-state size
              // represented in the index tensor stick. It guarantees that every
              // chunk of data is fully within the total data represented in one
              // index tensor stick.
              const auto steadyStateIbrSize =
                  dsc.dataStageParam_.at(this->dataStageIbrIdx)
                      .ss_.primaryDimToVal_st(dim);
              if (steadyStateIbrSize % param != 0) return false;
              // The value must be a divisor of the epilogue size represented in
              // the index tensor stick. It guarantees that every chunk of data
              // is fully within the total data represented in one index tensor
              // stick.
              const auto epilogueIbrSize =
                  dsc.dataStageParam_.at(this->dataStageIbrIdx)
                      .el_.primaryDimToVal_st(dim);
              if (epilogueIbrSize % param != 0) return false;
              return true;
            };
            if (is_any_of(dim, pagedDims) && !isPageValid()) continue;

            // This check is accurate for a non-padded dimension but not for a
            // padded dimension.
            // TODO: If we need accurate check for a padded dimension, it
            // requires a complete DstaStructDims to compute the padding size.
            // Therefore, we need to do that in each chunk exploration rather
            // than here.
            auto isDimensionParamMultipleOfStickSize =
                [this](PrimaryDimTypes dim, long param,
                       const DesignSpaceConfig &dsc) {
                  for (const auto &lds : dsc.labeledDs_) {
                    if (isIndexLds(lds)) continue;
                    auto minDimSize = this->getStickSize(dsc, lds.dsType_, dim);
                    if (lds.scaledLdsCategory_ ==
                            LabeledDsInfo::ScaledLdsCategory::SCALE_TENSOR &&
                        lds.mxInfo_.dim == dim) {
                      minDimSize *= lds.mxInfo_.blkSize;
                    }
                    if (param % minDimSize != 0) return false;
                  }
                  return true;
                };

            // When we do corelet split on this dimension, do not add this
            // candidate when the parameter value cannot be equally split for
            // each corelet. For a core-split dimension, this applies to the
            // current DSC. For a non-core-split dimension, this applies to all
            // DSCs. Also, do not add it when its value per corelet is not
            // multiple of the stick size.
            auto isParamCoreletSplitValid =
                [&isDimensionParamMultipleOfStickSize](
                    PrimaryDimTypes dim, long param,
                    const DesignSpaceConfig &dsc) {
                  const int numCoreletsPerCore = dsc.numCoreletsUsed_;
                  if (isDimensionCoreletSplit(dsc, dim) &&
                      (param % numCoreletsPerCore != 0 ||
                       !isDimensionParamMultipleOfStickSize(
                           dim, param / numCoreletsPerCore, dsc)))
                    return false;
                  return true;
                };

            const bool isCoreSplitDim = coreSplitDims.count(dim);
            bool isCoreletSplitValid = true;
            if (isCoreSplitDim) {
              isCoreletSplitValid = isParamCoreletSplitValid(dim, param, dsc);
            } else {
              for (const auto &currDsc : mySDsc.dscs_) {
                if (!isParamCoreletSplitValid(dim, param, currDsc)) {
                  isCoreletSplitValid = false;
                  break;
                }
              }
            }
            if (!isCoreletSplitValid) continue;

            // Do not add the candidate if its value is not multiple of the
            // stick size.
            if (!isDimensionParamMultipleOfStickSize(dim, param, dsc)) continue;

            // Add the value as candidate.
            if (dscCandidates[dscIdx].count(dim))
              dscCandidates[dscIdx].at(dim).push_back(param);
            else
              dscCandidates[dscIdx][dim] = {param};
          }
        }
      }

      DT_CHECK_MSG(!dscCandidates[dscIdx][dim].empty(),
                   "There must be at least one valid candidate.");
    }
  }

  return dscCandidates;
}

// ------------------------------------------------------------------------------------------------
// entry 208/382   level 1   scc 68   34 body lines
// unit: e208_getLdsL3TransferNodes
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1571
// original: std::vector<const dsc2::TransferNode *> L3DlOpsScheduler::getLdsL3TransferNodes( const SuperDsc &mySDsc, const int dscIdx, const int ldsIdx, const std::vector<SenComponents> &srcStorages, const std::vector<SenComponents> &dstStorages) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::vector<const dsc2::TransferNode *> e208_getLdsL3TransferNodes(
    const SuperDsc &mySDsc, const int dscIdx, const int ldsIdx,
    const std::vector<SenComponents> &srcStorages,
    const std::vector<SenComponents> &dstStorages) const
{
  auto &dsc = mySDsc.dscs_.at(dscIdx);
  const auto &lds = dsc.labeledDs_.at(ldsIdx);

  // Return empty result if the tensor is LX pinned because of no L3 transfers.
  if (lds.isLxPinned()) return std::vector<const dsc2::TransferNode *>();

  dsc2::AllocateNode *allocNode = nullptr;
  if (lds.isHbmPinned()) {
    DT_CHECK_MSG(lds.memOrg_.count(SenComponents::HBM),
                 "Expect HBM in memOrg_.");
    allocNode = lds.memOrg_.at(SenComponents::HBM).allocateNode_;
  } else {
    DT_CHECK_MSG(isLabeledDsLXNeighbor(mySDsc, dscIdx, lds),
                 "Expect input neighbor fetch.");
    DT_CHECK_MSG(lds.memOrg_.count(SenComponents::LX), "Expect LX in memOrg_.");
    allocNode = lds.memOrg_.at(SenComponents::LX).allocateNode_;
  }

  DT_CHECK_MSG(allocNode, "Expect a valid allocate node.");
  DT_CHECK_MSG(allocNode->hasAllocUsers(), "Expect valid alloc users.");
  std::vector<const dsc2::TransferNode *> l3TransNodes;
  for (auto *node : dsc.scheduleTree_.traverseTreeDFS(
           nullptr, {dsc2::ScheduleNode::TRANSFER})) {
    DT_CHECK_MSG(node->nodeType_ == dsc2::ScheduleNode::TRANSFER,
                 "Expect a transfer node.");
    auto transNode = static_cast<const dsc2::TransferNode *>(node);
    if (allocNode->hasAllocUser(transNode) &&
        is_any_of(transNode->src_.storage_, srcStorages) &&
        is_any_of(transNode->dstVias_.front().loc_.storage_, dstStorages))
      l3TransNodes.push_back(transNode);
  }
  return l3TransNodes;
}

// ------------------------------------------------------------------------------------------------
// entry 209/382   level 1   scc 76   64 body lines
// unit: e209_getLabeledDsChunkStickVolume
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1628
// original: unsigned long L3DlOpsScheduler::getLabeledDsChunkStickVolume( const DesignSpaceConfig &dsc, const int ldsIdx)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
unsigned long e209_getLabeledDsChunkStickVolume(
    const DesignSpaceConfig &dsc, const int ldsIdx)
{
  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
               "Expect dataStageParam_ entry for the core data stage.");
  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageChunkIdx),
               "Expect dataStageParam_ entry for the chunk data stage.");
  const LabeledDsInfo &lds = dsc.labeledDs_.at(ldsIdx);
  double scaleDownFactor = 1.0;
  if (lds.scaledLdsCategory_ ==
      LabeledDsInfo::ScaledLdsCategory::SCALE_TENSOR) {
    scaleDownFactor = ((double)1.0) / lds.mxInfo_.blkSize;
  }
  DT_CHECK_MSG(lds.memOrg_.count(SenComponents::LX),
               "Expect LX in labeledDs memOrg_.");
  const auto *allocNode = lds.memOrg_.at(SenComponents::LX).allocateNode_;
  DT_CHECK_MSG(allocNode, "Expect a valid allocate node.");
  DT_CHECK_MSG(
      !isIndexLds(lds),
      "Cannot getLabeledDsChunkStickVolume on indirect access index tensor");
  const auto pageSizes = allocNode->getPageSize();

  // stickVolume is the number of consecutive sticks.
  // For a broadcast dimension for this labeledDs, there is only one stick in
  // this dimension, so it must be a consecutive stick.
  unsigned long stickVolume = 1;
  const auto &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);
  const auto &dsChunk = dsc.dataStageParam_.at(dataStageChunkIdx);
  for (PrimaryDimTypes dim : dsc.getNonBroadcastLdsDims(ldsIdx)) {
    int stickSize = getStickSize(dsc, lds.dsType_, dim);
    // If the current dimension is symbolic at chunk data stage, we use its
    // granularity value to estimate the stick volume. This is because we are
    // conservative to estimate the smallest volume and the granularity is the
    // smallest value the symbolic dimension can have. Similarly, we use its
    // granularity value at the core data stage as the upper bound.
    long param = dsChunk.ss_.primaryDimToVal_st(
        dim, SenComponents::NO_COMPONENT, -1, -1, allocNode->padding_,
        lds.mxInfo_.dim == dim ? scaleDownFactor : 1.0,
        /* getSymbolicGranularity */ true);
    DT_CHECK_MSG(isValidDimParam(param), "Expect a valid parameter value.");
    const auto upperBound = dsCore.ss_.primaryDimToVal_st(
        dim, SenComponents::NO_COMPONENT, -1, -1, allocNode->padding_,
        lds.mxInfo_.dim == dim ? scaleDownFactor : 1.0,
        /* getSymbolicGranularity */ true);

    // If the current dimension is paged, it is param value is capped by the
    // page size.
    if (pageSizes.count(dim)) param = std::min<long>(param, pageSizes.at(dim));
    DT_CHECK_MSG(param % stickSize == 0,
                 "The parameter value must be multiple of stick size.");

    stickVolume *= param / stickSize;

    // When the current parameter is less than its CoreD_ value (the upper
    // bound), the memory is no longer contiguous for the outer dimensions.
    // Hence, the stickVolume compute stops in this dimension.
    if (param < upperBound) break;
    // When current parameter is not symbolic but CoreD_ is symbolic, the memory
    // may no longer be contiguous for outer dimensions. Stop here
    if (dsChunk.ss_.symbolicDimInfo_.count(dim) !=
        dsCore.ss_.symbolicDimInfo_.count(dim))
      break;
  }

  return stickVolume;
}

// ------------------------------------------------------------------------------------------------
// entry 210/382   level 1   scc 58   7 body lines
// unit: e210_isOpCrossCoreReduction
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2734
// original: bool L3DlOpsScheduler::isOpCrossCoreReduction( const SuperDsc &mySDsc, const DesignSpaceConfig &dsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e210_isOpCrossCoreReduction(
    const SuperDsc &mySDsc, const DesignSpaceConfig &dsc) const
{
  const auto reducedDimSet = getOpReducedDimSet(mySDsc, dsc);
  return std::any_of(reducedDimSet.begin(), reducedDimSet.end(),
                     [&mySDsc](PrimaryDimTypes dim) {
                       return mySDsc.numWkSlicesPerDim_.at(dim) > 1;
                     });
}

// ------------------------------------------------------------------------------------------------
// entry 211/382   level 1   scc 88   89 body lines
// unit: e211_getInsertionNode
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3376
// original: dsc2::ScheduleNode* L3DlOpsScheduler::getInsertionNode( const std::unordered_set<dsc2::ScheduleNode*>& refNodeSet, const bool insertBefore) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
dsc2::ScheduleNode* e211_getInsertionNode(
    const std::unordered_set<dsc2::ScheduleNode*>& refNodeSet,
    const bool insertBefore) const
{
  if (refNodeSet.empty()) return nullptr;

  std::deque<dsc2::ScheduleNode*> parentNodesOuterToInner;
  std::unordered_set<dsc2::ScheduleNode*> insertionNodeSet;
  // Initial insertionNodeSet and parentNodesOuterToInner with the first ref
  // node.
  const auto firstRefNode = *refNodeSet.begin();
  DT_CHECK_MSG(firstRefNode, "Expect a valid schedule node.");
  insertionNodeSet.insert(firstRefNode);
  auto currNode = firstRefNode;
  while (currNode) {
    auto parentNode = currNode->getMutableParent();
    if (parentNode) parentNodesOuterToInner.push_front(parentNode);

    currNode = parentNode;
  }

  // Visit the other ref nodes to find the common parent and update
  // insertionNodeSet with the child nodes that have the same parent.
  bool hasCommonParent = true;
  for (auto refNode : refNodeSet) {
    if (refNode != firstRefNode) {
      DT_CHECK_MSG(refNode, "Expect a valid schedule node.");
      auto currNode = refNode;
      while (currNode) {
        auto parentNode = currNode->getMutableParent();
        if (auto it = std::find(parentNodesOuterToInner.begin(),
                                parentNodesOuterToInner.end(), parentNode);
            it != parentNodesOuterToInner.end()) {
          // Common parent is found.
          if ((it + 1 != parentNodesOuterToInner.end())) {
            // If the common parent is not the innermost in
            // parentNodesOuterToInner, we need to reset insertionNodeSet and
            // update it by adding the common parent's previous child.
            insertionNodeSet.clear();
            insertionNodeSet.insert(*(it + 1));
            // Update parentNodesOuterToInner by erasing the inner nodes after
            // the common parent.
            parentNodesOuterToInner.erase(it + 1,
                                          parentNodesOuterToInner.end());
          }
          // Update insertionNodeSet by adding the current node.
          insertionNodeSet.insert(currNode);

          // Stop traversing parent nodes because the common parent is found.
          break;
        }
        currNode = parentNode;
      }
      if (currNode == nullptr) {
        // The current ref node does not have common parent with the other ref
        // nodes. Hence, no valid insertion node. Stop here.
        hasCommonParent = false;
        break;
      }
    }
  }

  dsc2::ScheduleNode *insertionNode = nullptr;
  if (hasCommonParent && !insertionNodeSet.empty()) {
    const auto *commonParentNode = (*insertionNodeSet.begin())->getPrev();
    DT_CHECK_MSG(commonParentNode, "Parent node must be a block node.");
    for (const auto node : insertionNodeSet)
      DT_CHECK_MSG(node->getPrev() == commonParentNode,
                   "Expect node to have the same parent.");
    if (insertBefore) {
      // Search for the first child in insertionNodeSet.
      for (auto &childNodePtr : commonParentNode->next_) {
        auto childNode = childNodePtr.get();
        if (insertionNodeSet.count(childNode)) {
          insertionNode = childNode;
          break;
        }
      }
    } else {
      // Search for the last child in insertionNodeSet.
      for (auto rit = commonParentNode->next_.rbegin();
           rit != commonParentNode->next_.rend(); ++rit) {
        auto childNode = rit->get();
        if (insertionNodeSet.count(childNode)) {
          insertionNode = childNode;
          break;
        }
      }
    }
  }
  return insertionNode;
}

// ------------------------------------------------------------------------------------------------
// entry 212/382   level 1   scc 89   45 body lines
// unit: e212_addL3LUAndLXLUSyncNodeSequence
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3912
// original: void L3DlOpsScheduler::addL3LUAndLXLUSyncNodeSequence( const dsc2::ScheduleNode *insertAfterNode) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e212_addL3LUAndLXLUSyncNodeSequence(
    const dsc2::ScheduleNode *insertAfterNode) const
{
  dsc2::SyncNode *l3luToLxluSend = createSyncNode(
      {SenComponents::L3LU},
      "sync_send_" +
          EnumsConversion::senComponentsToString.at(SenComponents::L3LU) +
          "_to_" +
          EnumsConversion::senComponentsToString.at(SenComponents::LXLU));
  dsc2::SyncNode *l3luToLxluReceive = createSyncNode(
      {SenComponents::LXLU},
      "sync_receive_" +
          EnumsConversion::senComponentsToString.at(SenComponents::LXLU) +
          "_from_" +
          EnumsConversion::senComponentsToString.at(SenComponents::L3LU),
      true);
  l3luToLxluSend->otherEndOfTheSignals_.push_back(l3luToLxluReceive);
  l3luToLxluReceive->otherEndOfTheSignals_.push_back(l3luToLxluSend);

  dsc2::SyncNode *lxluToL3luSend = createSyncNode(
      {SenComponents::LXLU},
      "sync_send_" +
          EnumsConversion::senComponentsToString.at(SenComponents::LXLU) +
          "_to_" +
          EnumsConversion::senComponentsToString.at(SenComponents::L3LU));
  dsc2::SyncNode *lxluToL3luReceive = createSyncNode(
      {SenComponents::L3LU},
      "sync_receive_" +
          EnumsConversion::senComponentsToString.at(SenComponents::L3LU) +
          "_from_" +
          EnumsConversion::senComponentsToString.at(SenComponents::LXLU),
      true);
  lxluToL3luSend->otherEndOfTheSignals_.push_back(lxluToL3luReceive);
  lxluToL3luReceive->otherEndOfTheSignals_.push_back(lxluToL3luSend);

  auto insertAfterNodeMutable =
      const_cast<dsc2::ScheduleNode*>(insertAfterNode);
  auto parentNode = insertAfterNodeMutable->getMutableParent();
  parentNode->addChildNode(l3luToLxluSend, /*addBefore*/ false,
                           insertAfterNode);
  parentNode->addChildNode(l3luToLxluReceive, /*addBefore*/ false,
                           l3luToLxluSend);
  parentNode->addChildNode(lxluToL3luSend, /*addBefore*/ false,
                           l3luToLxluReceive);
  parentNode->addChildNode(lxluToL3luReceive, /*addBefore*/ false,
                           lxluToL3luSend);
}

// ------------------------------------------------------------------------------------------------
// entry 213/382   level 1   scc 90   26 body lines
// unit: e213_addL3LUAndLXLUSoftSyncNodeSequence
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3959
// original: void L3DlOpsScheduler::addL3LUAndLXLUSoftSyncNodeSequence( const dsc2::ScheduleNode* insertAfterNode) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e213_addL3LUAndLXLUSoftSyncNodeSequence(
    const dsc2::ScheduleNode* insertAfterNode) const
{
  dsc2::SyncNode* l3luToLxluSend = createSyncNode(
      {SenComponents::L3LU},
      "sync_soft_send_" +
          EnumsConversion::senComponentsToString.at(SenComponents::L3LU) +
          "_to_" +
          EnumsConversion::senComponentsToString.at(SenComponents::LXLU),
      false, true);
  dsc2::SyncNode* l3luToLxluReceive = createSyncNode(
      {SenComponents::LXLU},
      "sync_soft_receive_" +
          EnumsConversion::senComponentsToString.at(SenComponents::LXLU) +
          "_from_" +
          EnumsConversion::senComponentsToString.at(SenComponents::L3LU),
      true, true);

  l3luToLxluSend->otherEndOfTheSignals_.push_back(l3luToLxluReceive);
  l3luToLxluReceive->otherEndOfTheSignals_.push_back(l3luToLxluSend);
  auto insertAfterNodeMutable =
      const_cast<dsc2::ScheduleNode*>(insertAfterNode);
  auto parentNode = insertAfterNodeMutable->getMutableParent();
  parentNode->addChildNode(l3luToLxluSend, /*addBefore*/ false,
                           insertAfterNode);
  parentNode->addChildNode(l3luToLxluReceive, /*addBefore*/ false,
                           l3luToLxluSend);
}

// ------------------------------------------------------------------------------------------------
// entry 214/382   level 1   scc 94   94 body lines
// unit: e214_optimizeHbmLdsOutputInScheduleTree
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4018
// original: void L3DlOpsScheduler::optimizeHbmLdsOutputInScheduleTree(SuperDsc &mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e214_optimizeHbmLdsOutputInScheduleTree(SuperDsc &mySDsc)
{
  for (auto &dsc : mySDsc.dscs_) {
    DT_CHECK_MSG(!dsc.scheduleTree_.empty(), "Expect a valid schedule tree.");
    DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
                 "Expect valid core data stage parameters.");
    DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageChunkIdx),
                 "Expect valid chunk data stage parameters.");
    //if (dsc.labeledDs_.empty()) return;
    const LabeledDsInfo &ldsOutput = dsc.labeledDs_.back();

    // Return if the output tensor does not reside in HBM because there is no
    // HBM->LX transfer node to optimize.
    if (!ldsOutput.isHbmPinned()) return;

    // Get the output tensor's HBM->LX transfer node.
    dsc2::TransferNode *ldsOutputTransferInNode = nullptr;
    const auto transferNodes = dsc.scheduleTree_.traverseTreeDFSMutable(
        nullptr, {dsc2::ScheduleNode::TRANSFER});
    for (auto node : transferNodes) {
      DT_CHECK_MSG(node->nodeType_ == dsc2::ScheduleNode::TRANSFER,
                   "Expect a transfer node.");
      dsc2::TransferNode *transNode = static_cast<dsc2::TransferNode *>(node);
      if (transNode->srcLdsAndLoopOffsets_.myLdsIdx_ == ldsOutput.ldsIdx_ &&
          transNode->src_.storage_ == SenComponents::HBM) {
        ldsOutputTransferInNode = transNode;
        break;
      }
    }

    // Check if we can remove the HBM->LX transfer node in the schedule tree.
    // We can safely remove it when the trip counts of its unrelated parent
    // chunk loops are all one.
    bool optimize = true;
    const auto ldsNonBroadcastDimsSet =
        dsc.getNonBroadcastLdsDimSet(ldsOutput.ldsIdx_);
    const auto parentLoops = getParentLoopNodes(*ldsOutputTransferInNode, dsc);
    for (const auto loopNode : parentLoops) {
      for (const auto &dimKind : loopNode->dims_) {
        const auto currDim = dimKind.dim_;
        const bool isLdsUnrelatedDim = !ldsNonBroadcastDimsSet.count(currDim);
        if (isLdsUnrelatedDim) {
          const int tripCount =
              getTripCount(dsc, currDim, loopNode->numId_, loopNode->denId_);
          if (tripCount > 1) {
            optimize = false;
            break;
          }
        }
      }
      if (!optimize) break;
    }

    // Skip if the trip count is not one.
    if (!optimize) continue;

    // We can safely remove the HBM->LX transfer node.
    ldsOutputTransferInNode->getMutableParent()->deleteChildNode(
        &dsc, ldsOutputTransferInNode);

    // We can also safely remove the L3SU <-> L3LU sync nodes.
    std::vector<dsc2::ScheduleNode *> syncNodes =
        dsc.scheduleTree_.traverseTreeDFSMutable(nullptr,
                                                 {dsc2::ScheduleNode::SYNC});
    std::vector<dsc2::SyncNode *> syncNodesToRemove;
    for (auto node : syncNodes) {
      DT_CHECK_MSG(node->nodeType_ == dsc2::ScheduleNode::SYNC,
                   "Expect a sync node.");
      dsc2::SyncNode *syncNode = static_cast<dsc2::SyncNode *>(node);
      DT_CHECK_MSG(syncNode->units_.size() == 1,
                   "Only support one unit in the sync node for now.");
      DT_CHECK_MSG(syncNode->otherEndOfTheSignals_.size() == 1,
                   "Only support one entry in otherEndOfTheSignals_ for now.");
      const dsc2::SyncNode *dstSyncNode =
          syncNode->otherEndOfTheSignals_.front();
      DT_CHECK_MSG(dstSyncNode->units_.size() == 1,
                   "Only support one unit in the sync node for now.");
      if ((syncNode->units_.count(SenComponents::L3SU) &&
           dstSyncNode->units_.count(SenComponents::L3LU)) ||
          (syncNode->units_.count(SenComponents::L3LU) &&
           dstSyncNode->units_.count(SenComponents::L3SU))) {
        DT_CHECK(((syncNode->units_.count(SenComponents::L3SU) &&
                   syncNode->isReceive_ == false) ||
                  (syncNode->units_.count(SenComponents::L3LU) &&
                   syncNode->isReceive_ == true)) &&
                 "Expect L3SU send sync or L3LU receive sync.");
        syncNodesToRemove.push_back(syncNode);
      }
    }
    for (auto syncNode : syncNodesToRemove) {
      // Remove this sync node from the schedule tree.
      syncNode->getMutableParent()->deleteChildNode(&dsc, syncNode);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 215/382   level 1   scc 98   103 body lines
// unit: e215_buildScheduleDimensionsTable
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4236
// original: auto L3DlOpsScheduler::buildScheduleDimensionsTable( SuperDsc &mySDsc, const int dscIdx, std::vector<PrimaryDimTypes> &dims, const bool isReuse) -> ScheduleDimTableType
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
auto e215_buildScheduleDimensionsTable(
    SuperDsc &mySDsc, const int dscIdx, std::vector<PrimaryDimTypes> &dims,
    const bool isReuse) -> ScheduleDimTableType
{
  ScheduleDimTableType schedDimTypesTable;
  DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);

  // Collect all tensors to be analyzed. We do not analyze index tensors for
  // determining the chunk loop order.
  std::vector<int> analyzedLdsIndices;
  for (const auto &lds : dsc.labeledDs_) {
    if (!isIndexLds(lds)) analyzedLdsIndices.push_back(lds.ldsIdx_);
  }

  // Identify the scheduling type of each dimension for each tensor by
  // interating all tensors.
  for (const auto ldsIdx : analyzedLdsIndices) {
    LabeledDsInfo &lds = dsc.labeledDs_.at(ldsIdx);

    // Get the associated dimensions for this tensor.
    auto &pdsInfo = dsc.primaryDsInfo_.at(lds.dsType_);
    std::unordered_set<PrimaryDimTypes> ldsDims;
    ScheduleDimMapType schedPrimDimTypesMap;

    for (PrimaryDimTypes dim : dsc.getLayoutDims(ldsIdx)) {
      DT_CHECK_MSG((dim != PrimaryDimTypes::IJ || dim != PrimaryDimTypes::KIJ),
                   "Do not expect combined IJ or KIJ dimensions.");
      // Identify the schedule type of this dimension.
      ScheduleDimTypes schedDimType;
      if ((dim == PrimaryDimTypes::I || dim == PrimaryDimTypes::J) &&
          (dsc.N_.paddingSizes_.count(dim) &&
           dsc.N_.paddingSizes_.at(dim).windowDim_ !=
               PrimaryDimTypes::PrimaryDimTypesCount))
        // Dimension IJ is used in a strided-window or padded fashion.
        schedDimType = ScheduleDimTypes::WINDOW_PADDED;
      else {
        int scaleIdx = dsc.getDimIndexInLayoutOrder(lds.dsType_, dim);
        DT_CHECK_MSG((scaleIdx >= 0 && scaleIdx < lds.scale_.size()),
                     "Invalid index in layoutDimOrder_.");
        if (lds.scale_.at(scaleIdx) == 1)
          schedDimType = ScheduleDimTypes::ELEMENTWISE;
        else if (lds.scale_.at(scaleIdx) < 1) {
          if (ldsIdx < dsc.labeledDs_.size() - 1)
            schedDimType = ScheduleDimTypes::BROADCAST;
          else
            schedDimType = ScheduleDimTypes::REDUCTION;
        } else
          DT_ERROR("Invalid scale_ number");
      }

      // Assign the dimension to its corresponding type.
      if (schedPrimDimTypesMap.count(schedDimType))
        schedPrimDimTypesMap.at(schedDimType).push_back(dim);
      else {
        std::vector<PrimaryDimTypes> addedDim = {dim};
        schedPrimDimTypesMap.emplace(schedDimType, addedDim);
      }

      // Record this dimension.
      ldsDims.insert(dim);
    }

    // When there are more than one primaryDsInfo_, a dimension may have the
    // reuse attribute. The reuse attribute applies to the dimensions that are
    // not included in the current tensor but are included in another tensor
    // in the DSC.
    if (isReuse) {
      std::vector<PrimaryDimTypes> addedDims;
      for (PrimaryDimTypes dim : dims) {
        DT_CHECK_MSG(
            (dim != PrimaryDimTypes::IJ || dim != PrimaryDimTypes::KIJ),
            "Do not expect combined IJ or KIJ dimensions.");

        if (ldsDims.count(dim)) continue;

        addedDims.push_back(dim);
      }

      schedPrimDimTypesMap.insert(
          std::make_pair(ScheduleDimTypes::REUSE, addedDims));
    }

    // Add a table entry.
    DT_CHECK_MSG(!schedPrimDimTypesMap.empty(), "The entry must not be empty.");
    schedDimTypesTable.emplace(ldsIdx, schedPrimDimTypesMap);
  }
  DT_CHECK_MSG(schedDimTypesTable.size() == analyzedLdsIndices.size(),
               "Missing table entry for LabeledDs.");

  if (verbose > 0) {
    std::cout << "Dimension types to LabeledDs table:" << std::endl;
    for (const auto &[ldsIdx, schedPrimDimTypesMap] : schedDimTypesTable) {
      std::cout << "LabeledDs index: " << ldsIdx << std::endl;
      for (auto &entry : schedPrimDimTypesMap) {
        std::cout << "Dimension type " << scheduleDimTypeToString(entry.first)
                  << ": ";
        for (auto &dim : entry.second)
          std::cout << EnumsConversion::primaryDimToString.at(dim) << " ";
        std::cout << std::endl;
      }
      std::cout << std::endl;
    }
  }

  return schedDimTypesTable;
}

// ------------------------------------------------------------------------------------------------
// entry 216/382   level 1   scc 101   231 body lines
// unit: e216_buildLoopOrder
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4377
// original: std::vector<PrimaryDimTypes> L3DlOpsScheduler::buildLoopOrder( SuperDsc &mySDsc, const int dscIdx, const std::vector<PrimaryDimTypes> &dims, const ScheduleDimTableType &schedDimTypesTable, const bool isReuse)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::vector<PrimaryDimTypes> e216_buildLoopOrder(
    SuperDsc &mySDsc, const int dscIdx,
    const std::vector<PrimaryDimTypes> &dims,
    const ScheduleDimTableType &schedDimTypesTable, const bool isReuse)
{
  DT_CHECK_MSG(!schedDimTypesTable.empty(), "Expect valid schedDimTypesTable.");
  std::vector<PrimaryDimTypes> loopOrder;
  std::set<PrimaryDimTypes> remainingDims;
  for (auto dim : dims) remainingDims.insert(dim);

  auto pushBackToLoopOrder = [&loopOrder, &remainingDims](PrimaryDimTypes dim) {
    loopOrder.push_back(dim);
    remainingDims.erase(dim);
  };

  DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);

  // Process the output tensor first.
  LabeledDsInfo &ldsOutput = dsc.labeledDs_.back();
  if (ldsOutput.isHbmPinned()) {
    // Process all REUSE and REDUCTION dimensions in the given order for the
    // output tensor.
    std::vector<ScheduleDimTypes> processingTypes{ScheduleDimTypes::REUSE,
                                                  ScheduleDimTypes::REDUCTION};
    for (ScheduleDimTypes type : processingTypes) {
      if (schedDimTypesTable.at(ldsOutput.ldsIdx_).count(type)) {
        for (PrimaryDimTypes dim :
             schedDimTypesTable.at(ldsOutput.ldsIdx_).at(type)) {
          if (remainingDims.count(dim)) pushBackToLoopOrder(dim);
        }
      }
    }
  }

  // Process the input tensors excluding the index tensors.
  std::vector<int> analyzedInputLdsIndices;
  for (int ldsIdx = 0; ldsIdx < dsc.labeledDs_.size() - 1; ldsIdx++) {
    LabeledDsInfo &lds = dsc.labeledDs_.at(ldsIdx);
    if (!isIndexLds(lds)) analyzedInputLdsIndices.push_back(ldsIdx);
  }
  for (const auto ldsIdx : analyzedInputLdsIndices) {
    LabeledDsInfo &lds = dsc.labeledDs_.at(ldsIdx);
    if (isReuse) {
      // The case of multiple DsType INPUT/KERNEL/OUTPUT.
      if (lds.dsType_ == DsTypes::INPUT &&
          (lds.isHbmPinned() || isLabeledDsLXNeighbor(mySDsc, dscIdx, lds))) {
        std::vector<int> targetIndices;
        getLabeledDsWithDsType(targetIndices, dsc, DsTypes::KERNEL);

        if (targetIndices.size() > 0) {
          // primaryDsTypes_ has INPUT/KERNEL/OUTPUT.
          DT_CHECK_MSG(ldsIdx == 0,
                       "The DsType INPUT tensor must be at index 0.");
          DT_CHECK_MSG(targetIndices.size() == 1,
                       "Expect only one LabeledDs with DsType KERNEL.");

          // Push back REUSE dimensions for the DsType KERNEL tensor.
          int ldsKernelIdx = targetIndices[0];
          if (schedDimTypesTable.at(ldsKernelIdx)
                  .count(ScheduleDimTypes::REUSE)) {
            const std::vector<PrimaryDimTypes> &targetDims =
                schedDimTypesTable.at(ldsKernelIdx).at(ScheduleDimTypes::REUSE);
            // Process all WINDOW_PADDED and ELEMENTWISE dimensions in the given
            // order for the DsType INPUT tensor that appear as REUSE for the
            // DsType KERNEL tensor.
            std::vector<ScheduleDimTypes> processingTypes{
                ScheduleDimTypes::WINDOW_PADDED, ScheduleDimTypes::ELEMENTWISE};
            for (ScheduleDimTypes type : processingTypes) {
              if (schedDimTypesTable.at(ldsIdx).count(type)) {
                for (auto dim : schedDimTypesTable.at(ldsIdx).at(type)) {
                  for (auto targetDim : targetDims) {
                    if (dim == targetDim && remainingDims.count(dim))
                      pushBackToLoopOrder(dim);
                  }
                }
              }
            }

            // Process all other dimensions for the DsType INPUT tensor that
            // appear as REUSE for the DsType KERNEL tensor.
            std::vector<PrimaryDimTypes> layoutDimOrder =
                dsc.getLayoutDims(ldsIdx);
            for (auto dim : layoutDimOrder) {
              for (auto targetDim : targetDims) {
                if (dim == targetDim && remainingDims.count(dim))
                  pushBackToLoopOrder(dim);
              }
            }

            // Process the remaining dimensions for this tensor.
            for (auto dim : layoutDimOrder) {
              if (remainingDims.count(dim)) pushBackToLoopOrder(dim);
            }
          }
        } else {
          // primaryDsTypes_ has INPUT/OUTPUT.
          // Process all BROADCAST and WINDOW_PADDED dimensions in the given
          // order for this tensor.
          // Note: Effectively there is no dimension reuse in this case.
          std::vector<ScheduleDimTypes> processingTypes{
              ScheduleDimTypes::BROADCAST, ScheduleDimTypes::WINDOW_PADDED};
          for (ScheduleDimTypes type : processingTypes) {
            if (schedDimTypesTable.at(ldsIdx).count(type)) {
              for (PrimaryDimTypes dim :
                   schedDimTypesTable.at(ldsIdx).at(type)) {
                if (remainingDims.count(dim)) pushBackToLoopOrder(dim);
              }
            }
          }
        }
      } else if (lds.dsType_ == DsTypes::KERNEL && lds.isHbmPinned()) {
        // Push back REUSE dimensions for the DsType INPUT tensor.
        std::vector<int> targetIndices;
        getLabeledDsWithDsType(targetIndices, dsc, INPUT);
        DT_CHECK_MSG(targetIndices.size() <= 1,
                     "Expect at most one LabeledDs with DsType INPUT.");
        if (!targetIndices.empty()) {
          int ldsInputIdx = targetIndices[0];
          if (schedDimTypesTable.at(ldsInputIdx)
                  .count(ScheduleDimTypes::REUSE)) {
            const std::vector<PrimaryDimTypes> &targetDims =
                schedDimTypesTable.at(ldsInputIdx).at(ScheduleDimTypes::REUSE);
            std::vector<PrimaryDimTypes> layoutDimOrder =
                dsc.getLayoutDims(ldsIdx);

            // Process all dimensions for the DsType KERNEL tensor that appear
            // as REUSE for the DsType INPUT tensor.
            for (auto dim : layoutDimOrder) {
              for (auto targetDim : targetDims) {
                if (dim == targetDim && remainingDims.count(dim))
                  pushBackToLoopOrder(dim);
              }
            }

            // Process the remaining dimensions for this tensor.
            for (auto dim : layoutDimOrder) {
              if (remainingDims.count(dim)) pushBackToLoopOrder(dim);
            }
          }
        }
      } else if ((lds.dsType_ == DsTypes::OUTPUT ||
                  lds.dsType_ == DsTypes::KERNEL_IDX) &&
                 lds.isHbmPinned()) {
        // Process all REUSE and BROADCAST dimensions in the given order for
        // this tensor.
        std::vector<ScheduleDimTypes> processingTypes{
            ScheduleDimTypes::REUSE, ScheduleDimTypes::BROADCAST};
        for (ScheduleDimTypes type : processingTypes) {
          if (schedDimTypesTable.at(ldsIdx).count(type)) {
            for (PrimaryDimTypes dim : schedDimTypesTable.at(ldsIdx).at(type)) {
              if (remainingDims.count(dim)) pushBackToLoopOrder(dim);
            }
          }
        }
      }
    } else {
      // The case of single DsType OUTPUT, or two DsType INPUT/OUTPUT.
      if (lds.isHbmPinned()) {
        // Process all BROADCAST and WINDOW_PADDED dimensions in the given
        // order for this tensor.
        std::vector<ScheduleDimTypes> processingTypes{
            ScheduleDimTypes::BROADCAST, ScheduleDimTypes::WINDOW_PADDED};
        for (ScheduleDimTypes type : processingTypes) {
          if (schedDimTypesTable.at(ldsIdx).count(type)) {
            for (PrimaryDimTypes dim : schedDimTypesTable.at(ldsIdx).at(type)) {
              if (remainingDims.count(dim)) pushBackToLoopOrder(dim);
            }
          }
        }
      }
    }
  }

  // Process the remaining dimensions.
  // Push back all remaining dimensions in the DsType INPUT tensor, followed by
  // those in the KERNEL tensor, and followed by the OUTPUT tensors, and
  // followed by the KERNEL_IDX tensors.
  {
    // try to rely less on DsTypes names
    std::vector<int> ldsOrder, ldsOrderLowPriority;
    for (auto& lds : dsc.labeledDs_) {
      if (is_any_of(&lds, dsc.computeOp_.at(0).indirectAccessIndexLabeledDs))
        ldsOrderLowPriority.push_back(lds.ldsIdx_);
      else
        ldsOrder.push_back(lds.ldsIdx_);
    }
    ldsOrder.insert(ldsOrder.end(), ldsOrderLowPriority.begin(),
                    ldsOrderLowPriority.end());
    for (const auto ldsIdx : ldsOrder) {
      if (remainingDims.empty()) break;

      std::vector<PrimaryDimTypes> layoutDimOrder = dsc.getLayoutDims(ldsIdx);
      for (auto dim : layoutDimOrder) {
        if (remainingDims.count(dim)) pushBackToLoopOrder(dim);
      }
    }
  }

  // With index tensor, ensure that stick of index tensor is inner compared all
  // of its layout dimension -> TO BE REMOVED
  const auto pagedDims = getPagedDimensions(dsc);
  if (pagedDims.size() > 1) {
    for (int ldsIdx = 0; ldsIdx < dsc.labeledDs_.size() - 1; ldsIdx++) {
      LabeledDsInfo &lds = dsc.labeledDs_.at(ldsIdx);
      if (isIndexLds(lds)) {
        const auto indexStickDims = dsc.getStickDims(ldsIdx);
        DT_CHECK_MSG(indexStickDims.size() == 1,
                     "Support only one stick dimension for now.");
        PrimaryDimTypes indexStickDim = indexStickDims.front();
        auto it = std::find(loopOrder.begin(), loopOrder.end(), indexStickDim);
        DT_CHECK(it != loopOrder.end());
        auto indexStickPos = std::distance(loopOrder.begin(), it);
        for (int l = 0; l < indexStickPos; l++) {
          if (is_any_of(loopOrder.at(l), pagedDims)) {
            std::swap(loopOrder.at(l), loopOrder.at(indexStickPos));
            break;
          }
        }
      }
    }
  }

  DT_CHECK_MSG(remainingDims.empty(), "All dimensions should be processed.");

  if (verbose > 0) {
    std::cout << "Loop Order (innermost to outermost):" << std::endl;
    for (auto dim : loopOrder)
      std::cout << EnumsConversion::primaryDimToString.at(dim) << " ";
    std::cout << std::endl;
  }
  // Verify.
  DT_CHECK_MSG(verifyLoopOrder(loopOrder), "Invalid loop order.");

  return loopOrder;
}

// ------------------------------------------------------------------------------------------------
// entry 217/382   level 1   scc 97   58 body lines
// unit: e217_createChunkLoopNodes
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4612
// original: void L3DlOpsScheduler::createChunkLoopNodes( SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &loopOrderInnerToOuter)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e217_createChunkLoopNodes(
    SuperDsc &mySDsc,
    const std::vector<PrimaryDimTypes> &loopOrderInnerToOuter)
{
  DT_CHECK_MSG(!loopOrderInnerToOuter.empty(), "Expect valid loop order.");
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
    // Initialize the root node by setting its denId_ to the core datastage
    // index.
    dsc.scheduleTree_.getHeadMutable()->denId_ = dataStageCoreIdx;

    // A vector of loop nodes. The first entry represents the level outside the
    // outermost loop, and the last entry represents the level inside the
    // innermost loop.
    std::vector<dsc2::BlockNode *> loopNodes;

    auto addLoopNodes = [&loopNodes, this, &dsc, &loopOrderInnerToOuter](
                            const int numId, const int denId) {
      // Reserve space to avoid reallocations.
      loopNodes.reserve(loopNodes.size() + loopOrderInnerToOuter.size());

      for (auto it = loopOrderInnerToOuter.rbegin();
           it != loopOrderInnerToOuter.rend(); ++it) {
        const PrimaryDimTypes dim = *it;
        const std::string name = "loop_ds" + std::to_string(numId) + "_ds" +
                                 std::to_string(denId) + "_" +
                                 EnumsConversion::primaryDimToString.at(dim);
        dsc2::LoopNode* loopNode =
            createLoopNode(dsc, {dim}, numId, denId, name);
        loopNodes.push_back(loopNode);
      }
    };

    // Create loop nodes.
    if (lxBufferType == BufferType::SPATIAL_DOUBLE) {
      // Add a new SuperChunk data stage.
      if (dataStageSuperChunkIdx == -1)
        dataStageSuperChunkIdx = getNewDataStageIndex(mySDsc, dsc);
      addLoopNodes(dataStageCoreIdx, dataStageSuperChunkIdx);
      addLoopNodes(dataStageSuperChunkIdx, dataStageChunkIdx);
    } else
      addLoopNodes(dataStageCoreIdx, dataStageChunkIdx);

    // Add a dummy blockNode in the end to represent the level inside the
    // innermost loop.
    dsc2::BlockNode* dummyBlockNode = createBlockNode(lxBelowBlockNodeName);
    loopNodes.push_back(dummyBlockNode);
    DT_CHECK_MSG((lxBufferType == BufferType::DOUBLE &&
                  loopNodes.size() == loopOrderInnerToOuter.size() + 1) ||
                     (lxBufferType == BufferType::SPATIAL_DOUBLE &&
                      loopNodes.size() == 2 * loopOrderInnerToOuter.size() + 1),
                 "Incorrect number of loop nodes.");

    // Add the loop nodes in schedule tree.
    dsc2::BlockNode *currNode = dsc.scheduleTree_.getHeadMutable();
    for (auto newNode : loopNodes) {
      currNode->addChildNode(newNode);
      currNode = newNode;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 218/382   level 1   scc 104   114 body lines
// unit: e218_setCondGtr
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4721
// original: void L3DlOpsScheduler::setCondGtr( SuperDsc &mySDsc, const int dscIdx, const int ldsIdx, const int currCoreId, dsc2::TransferNode &ldsL3LUTransNode, const std::vector<std::tuple<const dsc2::LoopNode *, PrimaryDimTypes, int, int>> &loopDimTripCounts)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e218_setCondGtr(
    SuperDsc &mySDsc, const int dscIdx, const int ldsIdx, const int currCoreId,
    dsc2::TransferNode &ldsL3LUTransNode,
    const std::vector<std::tuple<const dsc2::LoopNode *, PrimaryDimTypes, int,
                                 int>> &loopDimTripCounts)
{
  std::unordered_set<PrimaryDimTypes> coreSplitDims =
      getCoreSplitDimensions(mySDsc);
  // Currently the hardware only supports one condGtr_ entry. That means, we
  // can only process one core split dimension for its multicast condition.
  DT_CHECK_MSG(coreSplitDims.size() < 2,
               "Unsupported number of core split dimensions in condGtr_.");

  DesignSpaceConfig &currDsc = mySDsc.dscs_.at(dscIdx);
  LabeledDsInfo &currLds = currDsc.labeledDs_.at(ldsIdx);
  const auto currLdsNonBroadcastDimsSet =
      currDsc.getNonBroadcastLdsDimSet(ldsIdx);
  DT_CHECK_MSG(!ldsL3LUTransNode.dstVias_.empty(), "Expect valid dstVias_.");
  const auto dstStorage = ldsL3LUTransNode.dstVias_.front().loc_.storage_;
  DT_CHECK_MSG(currLds.memOrg_.count(dstStorage),
               "Expect " +
                   EnumsConversion::senComponentsToString.at(dstStorage) +
                   " in memOrg_.");
  dsc2::AllocateNode *allocDstNode =
      currLds.memOrg_.at(dstStorage).allocateNode_;
  DT_CHECK_MSG(allocDstNode, "Expect a valid allocate node.");
  DT_CHECK_MSG(currLds.memOrg_.count(SenComponents::HBM),
               "Expect HBM in memOrg_.");
  dsc2::AllocateNode *allocHbmNode =
      currLds.memOrg_.at(SenComponents::HBM).allocateNode_;
  DT_CHECK_MSG(allocHbmNode, "Expect a valid allocate node.");

  for (const auto &[loopNode, dim, currTripCount, otherTripCount] :
       loopDimTripCounts) {
    const auto &currWkSlices = mySDsc.coreIdToWkSlice_.at(currCoreId);
    std::pair<size_t, size_t> sharesAndGroupName = getSharesAndGroupName(
        mySDsc, currDsc, currLds, currWkSlices, currDsc.coreIdsUsed_);

    // Also put the condGtr_ info to the corresponding transfer node in
    // the schedule tree, by duplicating the transfer node and guarding
    // it by a condition node.
    DT_CHECK_MSG(
        ldsL3LUTransNode.src_.storage_ == SenComponents::HBM &&
            !ldsL3LUTransNode.dstVias_.empty() &&
            ldsL3LUTransNode.dstVias_[0].loc_.storage_ == SenComponents::LX,
        "Expect L3LU transfer node.");
    // Create a new transfer node by duplicating the current one.
    std::vector<SenComponents> dstUnits;
    std::vector<SenComponents> dstStorage;
    for (const auto &dstVia : ldsL3LUTransNode.dstVias_) {
      dstUnits.push_back(dstVia.loc_.unit_);
      dstStorage.push_back(dstVia.loc_.storage_);
    }
    std::vector<int> dstLdsIndices;
    for (const auto &dataInfo : ldsL3LUTransNode.dstLdsAndLoopOffsets_)
      dstLdsIndices.push_back(dataInfo.myLdsIdx_);
    // TODO: Maybe it is better to clone from the original transfer
    // node instead of creating an empty new node and copy individual
    // field.
    const std::string newTransNodeName =
        ldsL3LUTransNode.name_ + "_condition_" + loopNode->name_ + "_" +
        EnumsConversion::primaryDimToString.at(dim);
    dsc2::TransferNode *newTransNode = createTransferNode(
        ldsL3LUTransNode.src_.unit_, ldsL3LUTransNode.src_.storage_, dstUnits,
        dstStorage, ldsL3LUTransNode.srcLdsAndLoopOffsets_.myLdsIdx_,
        dstLdsIndices, newTransNodeName);
    newTransNode->srcIndirect_ = ldsL3LUTransNode.srcIndirect_;
    newTransNode->srcIndirectLdsAndLoopOffsets_ =
        ldsL3LUTransNode.srcIndirectLdsAndLoopOffsets_;
    for (int i = 0; i < ldsL3LUTransNode.dstVias_.size(); ++i) {
      newTransNode->dstVias_.at(i).locIndirect_ =
          ldsL3LUTransNode.dstVias_.at(i).locIndirect_;
    }
    newTransNode->dstIndirectLdsAndLoopOffsets_ =
        ldsL3LUTransNode.dstIndirectLdsAndLoopOffsets_;
    allocDstNode->addAllocUser(newTransNode);
    allocHbmNode->addAllocUser(newTransNode);

    // Insert the new transfer node and create the condition nodes to
    // form the schedule tree as below:
    //   condition_if
    //     transfer_original
    //   condition_else
    //     transfer_condition
    dsc2::BlockNode *currParent = ldsL3LUTransNode.getMutableParent();
    auto ifNode = new dsc2::ConditionNode();
    currParent->addChildNode(ifNode, true, &ldsL3LUTransNode);
    ifNode->name_ = "condition_separate_" + ldsL3LUTransNode.name_ + "_" +
                    loopNode->name_ + "_" +
                    EnumsConversion::primaryDimToString.at(dim);

    dsc2::LoopCond loopCond(loopNode, dim, CondOp::LT,
                            dsc2::LoopCond::CondValType::INT, otherTripCount);
    std::vector<dsc2::LoopCond> loopConds;
    loopConds.emplace_back(loopCond);
    ifNode->loopCond_.twoLevelOrOfAnds_.emplace_back(loopConds);

    auto thenBlockNode = new dsc2::BlockNode();
    thenBlockNode->name_ = ifNode->name_ + "_then_region";
    ifNode->addThenRegion(thenBlockNode);
    ldsL3LUTransNode.moveNode(&currDsc, thenBlockNode);

    auto elseBlockNode = new dsc2::BlockNode();
    elseBlockNode->name_ = ifNode->name_ + "_else_region";
    ifNode->addElseRegion(elseBlockNode);
    elseBlockNode->addChildNode(newTransNode);

    dsc2::GroupTagRegInfo gtrInfo;
    gtrInfo.numSharers_ = sharesAndGroupName.first;
    // If there is only one sharer (meaning no share), we do not need
    // a group id so setting it to -1.
    gtrInfo.groupId_ =
        sharesAndGroupName.first > 1 ? sharesAndGroupName.second : -1;
    if (gtrInfo.groupId_ != -1) {
      currDsc.gtrIdsUsed_.insert(gtrInfo.groupId_);
    }
    newTransNode->coreIdToGTRInfo_.emplace(currCoreId, gtrInfo);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 219/382   level 1   scc 107   137 body lines
// unit: e219_fillFinalStartAddressAndOffset
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4959
// original: void L3DlOpsScheduler::fillFinalStartAddressAndOffset( DesignSpaceConfig &dsc, const int ldsIdx, const std::vector<PrimaryDimTypes> &coreletSplitDims) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e219_fillFinalStartAddressAndOffset(
    DesignSpaceConfig &dsc, const int ldsIdx,
    const std::vector<PrimaryDimTypes> &coreletSplitDims) const
{
  const LabeledDsInfo &lds = dsc.labeledDs_.at(ldsIdx);
  DT_CHECK_MSG(lds.memOrg_.count(SenComponents::LX), "Expect LX in memOrg_.");
  dsc2::AllocateNode *allocNode =
      lds.memOrg_.at(SenComponents::LX).allocateNode_;
  DT_CHECK_MSG(allocNode, "Expect a valid allocate node.");
  DT_CHECK_MSG((allocNode->numBuffers_ == 1 || allocNode->numBuffers_ == 2),
               "Expect no buffering or double buffering.");

  auto coordinates =
      allocNode->startAddressCoreCorelet_.getFlattenedCoordinates(
          {{0, 0}, {1, 0}});
  DT_CHECK(coordinates.begin()->size() >= 2);
  for (auto &coord : coordinates) {
    for (auto coreId : dsc.coreIdsUsed_) {
      coord[0] = coreId;
      std::pair<int64_t, int64_t> startAddrAndOffset =
          getInitialStartAddressAndOffset(dsc, ldsIdx, coord);
      const int64_t startAddr = startAddrAndOffset.first;
      const int64_t bufferOffset = startAddrAndOffset.second;

      // Assign the values to startAddressCoreCorelet_ and
      // bufferOffsetCoreCorelet_ in the allocate node, in particular for
      // corelet1, because we only have the values for corelet0 at this point.
      DT_CHECK_MSG(dsc.numCoreletsUsed_DSC2_ > 0,
                   "Expect at least one corelet used.");
      if (dsc.numCoreletsUsed_DSC2_ == 1) {
        const int corelet0Id = 0;
        coord[1] = corelet0Id;
        allocNode->startAddressCoreCorelet_.insertData(startAddr, coord);
        allocNode->bufferOffsetCoreCorelet_[coreId][corelet0Id] = bufferOffset;
      } else {
        std::vector<PrimaryDimTypes> ldsCoreletSplitDim;
        for (PrimaryDimTypes dim : dsc.getNonBroadcastLdsDims(ldsIdx)) {
          if (DCGUtils::isValPresent(coreletSplitDims, dim))
            ldsCoreletSplitDim.push_back(dim);
        }
        DT_CHECK_MSG(
            ldsCoreletSplitDim.size() <= 1,
            "Support maximal one corelet split dimension for a tensor");
        if (ldsCoreletSplitDim.size() == 0) {
          // Corelet split is on an unrelated dimension. The start addresses for
          // all corelets are the same. And the offset is the buffer size.
          for (int coreletId = 0; coreletId < dsc.numCoreletsUsed_DSC2_;
               ++coreletId) {
            coord[1] = coreletId;
            allocNode->startAddressCoreCorelet_.insertData(startAddr, coord);
            allocNode->bufferOffsetCoreCorelet_[coreId][coreletId] =
                bufferOffset;
          }
        } else {
          // Corelet split is on a related dimension. The start address for each
          // corelet has an offset. If the tensor is HBM double buffering, we
          // use the chunk data stage to compute the offset. Otherwise, if the
          // tensor is LX pinned or input neighbor fetch, we use the core data
          // stage for all non-corelet-split dimensions and the chunk data stage
          // for the corelet split dimension.
          PrimaryDimTypes coreletSplitDim = ldsCoreletSplitDim.front();
          const auto &dataStage =
              lds.isHbmPinned() ? dsc.dataStageParam_.at(dataStageChunkIdx)
                                : dsc.dataStageParam_.at(dataStageCoreIdx);
          DT_CHECK(
              dataStage.ss_.coreletSplit_.count(coreletSplitDim) &&
              "Expect corelet split dimension in chunk data stage parameters.");
          int64_t addr = startAddr;
          const auto stickSizePerDim = dsc.getCumulativeStickSizes(lds.dsType_);
          DT_CHECK_MSG(allocNode->startAddressCoreCorelet_.getFuncType(1) ==
                           BaseFuncType::Map,
                       "AllocateNode start address cl fold must be map type if "
                       "corelets need to work on different data");
          for (int coreletId = 0; coreletId < dsc.numCoreletsUsed_DSC2_;
               ++coreletId) {
            coord[1] = coreletId;
            allocNode->startAddressCoreCorelet_.insertData(startAddr, coord);
            allocNode->bufferOffsetCoreCorelet_[coreId][coreletId] =
                bufferOffset;

            // Compute corelet split offset by iterate from the innermost to
            // outer dimensions in layoutDimOrder_.
            int64_t coreletOffset = 1;
            for (auto dim : dsc.getNonBroadcastLdsDims(ldsIdx)) {
              const int stickSize =
                  stickSizePerDim.count(dim) ? stickSizePerDim.at(dim) : 1;
              if (dim == coreletSplitDim) {
                // Use chunk data stage if the current dimension is corelet
                // splitted.
                const auto &dsChunk = dsc.dataStageParam_.at(dataStageChunkIdx);
                if (allocNode->padding_.getPadding(dim) != PadType::NOPAD) {
                  // This is the case that the corelet split dimension has
                  // padding. Currently the implementation here only supports
                  // the I dimension. The corelet offset for I dimension is
                  // calculated as:
                  //   offset_in_element = size_of_i * stride
                  DT_CHECK(allocNode->padding_.getPadding(dim) ==
                               PadType::PADDED_FULLSPAN_WUNNEEDED &&
                           "Support PadType::PADDED_FULLSPAN_WUNNEEDED only.");
                  DT_CHECK_MSG(
                      dsChunk.ss_.paddingSizes_.count(dim),
                      "Expect padding sizes in chunk data stage params.");
                  DT_CHECK_MSG(dim == PrimaryDimTypes::I,
                               "Support I dimension only.");
                  DT_CHECK_MSG(dsChunk.ss_.paddingSizes_.at(dim).stride_ > 0,
                               "Expect a valid stride.");
                  coreletOffset *=
                      dsChunk.ss_.coreletSplit_.at(dim).at(coreletId) *
                      dsChunk.ss_.paddingSizes_.at(dim).stride_ / stickSize;
                } else {
                  // No padding on the corelet split dimension.
                  coreletOffset *= dsChunk.ss_.primaryDimToVal_st(
                                       dim, SenComponents::NO_COMPONENT, -1,
                                       coreletId, allocNode->padding_) /
                                   stickSize;
                }

                // The corelet offset is fully computed when we get to the
                // corelet split dimension.
                break;
              } else
                // Otherwise, when the current dimension is an inner dimension
                // or the corelet split dimension but without padding, the
                // corelet offset is the size of this dimension; calculated as:
                //   offset_in_element = size_of_dim
                coreletOffset *= dataStage.ss_.primaryDimToVal_st(
                                     dim, SenComponents::NO_COMPONENT, -1,
                                     coreletId, allocNode->padding_) /
                                 stickSize;
            }
            coreletOffset *= dscGlobal.sysDef.bytesPerStick;

            // Update address.
            addr += coreletOffset;
          }
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 220/382   level 1   scc 108   43 body lines
// unit: e220_fillIBRStartAddressAndOffset
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5100
// original: void L3DlOpsScheduler::fillIBRStartAddressAndOffset(const SuperDsc &sdsc, DesignSpaceConfig &dsc, const int ldsIdx) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e220_fillIBRStartAddressAndOffset(const SuperDsc &sdsc,
                                                    DesignSpaceConfig &dsc,
                                                    const int ldsIdx) const
{
  const LabeledDsInfo &lds = dsc.labeledDs_.at(ldsIdx);
  // Nothing to fill if it is not an index tensor.
  if (!isIndexLds(lds)) return;

  DT_CHECK_MSG(lds.memOrg_.count(SenComponents::L3LUIBR) ||
                   lds.memOrg_.count(SenComponents::L3SUIBR),
               "Expect IBR in memOrg_.");

  auto fillAllocIbrNode = [&sdsc, &dsc](dsc2::AllocateNode *allocIbrNode) {
    // Start address is zero.
    auto &addrFM = allocIbrNode->startAddressCoreCorelet_;
    DT_CHECK(addrFM.hasZeroFoldDim());
    std::deque<const FoldDimProp *> sdscFoldProps{sdsc.coreFoldProp_.get(),
                                                  sdsc.coreletFoldProp_.get()};
    for (const auto &foldProp : sdsc.sdscFoldProps_) {
      sdscFoldProps.push_back(foldProp.get());
    }
    std::deque<BaseFuncType> foldTypes(sdscFoldProps.size(),
                                       BaseFuncType::Constant);
    addrFM.buildFoldSpace(sdscFoldProps, foldTypes);
    std::deque<int64_t> coord(sdscFoldProps.size(), 0);
    addrFM.insertData(0, coord);

    // Buffer offset is zero because we load the full IBR in one transfer.
    for (auto coreId : dsc.coreIdsUsed_) {
      for (int coreletId = 0; coreletId < dsc.numCoreletsUsed_DSC2_;
           ++coreletId) {
        allocIbrNode->bufferOffsetCoreCorelet_[coreId][coreletId] = 0;
      }
    }
  };

  const std::vector<SenComponents> ibrRegs{SenComponents::L3LUIBR,
                                           SenComponents::L3SUIBR};
  for (const auto ibr : ibrRegs) {
    if (lds.memOrg_.count(ibr)) {
      auto allocIbrNode = lds.memOrg_.at(ibr).allocateNode_;
      DT_CHECK_MSG(allocIbrNode, "Expect a valid allocate node.");
      fillAllocIbrNode(allocIbrNode);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 221/382   level 1   scc 111   196 body lines
// unit: e221_fillTransferZeroPaddingInfo
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5296
// original: void L3DlOpsScheduler::fillTransferZeroPaddingInfo(SuperDsc &mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e221_fillTransferZeroPaddingInfo(SuperDsc &mySDsc)
{
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
    const auto &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx).ss_;
    const auto &dsChunk = dsc.dataStageParam_.at(dataStageChunkIdx).ss_;

    // We do not support both paging and windowed-padding on the same dimension.
    const auto pagedDims = getPagedDimensions(dsc);
    if (!pagedDims.empty()) {
      for (const auto &[dim, paddingSizes] : dsc.N_.paddingSizes_) {
        if (paddingSizes.windowDim_ != PrimaryDimTypes::PrimaryDimTypesCount)
          DT_CHECK_MSG(std::find(pagedDims.begin(), pagedDims.end(),
                                 paddingSizes.windowDim_) == pagedDims.end(),
                       "Do not support both paging and windowed-padding on the "
                       "same dimension.");
      }
    }

    // Find the L3 transfer nodes that need zero padding in window dimensions in
    // LX.
    std::vector<dsc2::TransferNode *> zeroPadTransNodes;
    for (const auto &lds : dsc.labeledDs_) {
      // The L3 transfer needs LX zero padding when the tensor meets the
      // following conditions:
      //   1. memOrg_.at(LX): isPadded == 1, isZeroPadded == 1
      //   2. Has a padded dimension that is related to a window dimension.
      //   3. Transfer from NO_COMPONENT or HBM to LX.
      if (lds.memOrg_.count(SenComponents::LX) &&
          lds.memOrg_.at(SenComponents::LX).isZeroPadded != ZpType::NOZEROPAD) {
        DT_CHECK_MSG(lds.memOrg_.at(SenComponents::LX).isPadded,
                     "Expect memOrg_ LX isPadded is true.");
        const dsc2::AllocateNode *allocLxNode =
            lds.memOrg_.at(SenComponents::LX).allocateNode_;
        DT_CHECK_MSG(allocLxNode, "Expect a valid LX allocate node.");
        bool hasWindowPad = false;
        for (PrimaryDimTypes dim : dsc.getLayoutDims(lds.ldsIdx_)) {
          if (allocLxNode->padding_.getPadding(dim) != PadType::NOPAD) {
            DT_CHECK_MSG(dsc.N_.paddingSizes_.count(dim),
                         "Expect paddingSizes_ entry.");
            if (dsc.N_.paddingSizes_.at(dim).windowDim_ !=
                PrimaryDimTypes::PrimaryDimTypesCount) {
              hasWindowPad = true;
              break;
            }
          }
        }
        if (hasWindowPad) {
          for (const auto &[allocUser, count] : allocLxNode->allocUsers_) {
            if (allocUser->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
              const dsc2::TransferNode *transNode =
                  static_cast<const dsc2::TransferNode *>(allocUser);
              if ((transNode->src_.storage_ == SenComponents::HBM ||
                   transNode->src_.storage_ == SenComponents::NO_COMPONENT) &&
                  transNode->dstVias_.front().loc_.storage_ ==
                      SenComponents::LX)
                zeroPadTransNodes.push_back(
                    const_cast<dsc2::TransferNode *>(transNode));
            }
          }
        }
      }
    }

    // Fill zero padding info in each transfer node.
    for (auto transNode : zeroPadTransNodes) {
      // Initialize the paddingInfo_ field in the transfer node.
      const bool isHbmTransfer = transNode->src_.storage_ == SenComponents::HBM;

      auto getDataStageLxSize = [isHbmTransfer, &dsCore, &dsChunk](
                                    const PrimaryDimTypes dim,
                                    const PaddingFormType &paddingType = {}) {
        const auto &dataStage = isHbmTransfer ? dsChunk : dsCore;
        return dataStage.primaryDimToVal_st(dim, SenComponents::NO_COMPONENT,
                                            -1, -1, paddingType);
      };

      // Collect padded dimensions for a window operation from N_.
      std::vector<PrimaryDimTypes> paddedDims;
      for (auto [dim, paddingSizes] : dsc.N_.paddingSizes_) {
        if ((paddingSizes.padFront_ > 0 || paddingSizes.padBack_ > 0) &&
            paddingSizes.windowDim_ != PrimaryDimTypes::PrimaryDimTypesCount)
          paddedDims.push_back(dim);
      }

      for (auto dim : paddedDims) {
        const int numWkSlice = mySDsc.numWkSlicesPerDim_.at(dim);
        DT_CHECK_MSG(
            dsCore.primaryDimToVal_st(dim) % getDataStageLxSize(dim) == 0,
            "Invalid chunk parameter.");
        const int numChunks =
            dsCore.primaryDimToVal_st(dim) / getDataStageLxSize(dim);

        std::vector<int> sizes(
            dsc2::TransferPadInfo::FoldDimPosition::TOTAL_FOLDDIM_NUM);
        std::vector<int> alphasPadFront(
            dsc2::TransferPadInfo::FoldDimPosition::TOTAL_FOLDDIM_NUM);
        std::vector<int> betasPadFront(
            dsc2::TransferPadInfo::FoldDimPosition::TOTAL_FOLDDIM_NUM);
        std::vector<int> alphasPadBack(
            dsc2::TransferPadInfo::FoldDimPosition::TOTAL_FOLDDIM_NUM);
        std::vector<int> betasPadBack(
            dsc2::TransferPadInfo::FoldDimPosition::TOTAL_FOLDDIM_NUM);

        // Set sizes for each folded dimension in fold manager.
        sizes[dsc2::TransferPadInfo::FoldDimPosition::WORK_SLICE_FOLDDIM] =
            numWkSlice;
        sizes[dsc2::TransferPadInfo::FoldDimPosition::CHUNK_FOLDDIM] =
            numChunks;

        // Compute the alpha and beta values for the affine fold dimensions.

        const auto &paddingSizesN = dsc.N_.paddingSizes_;
        const auto &paddingSizesCore = dsCore.paddingSizes_;
        const auto &paddingSizesChunk = dsChunk.paddingSizes_;
        const int padFrontN = paddingSizesN.at(dim).padFront_;
        const int padBackN = paddingSizesN.at(dim).padBack_;

        // Alpha for the padFront wkslice fold dimension is the negated core
        // stage offset.
        const int coreOffset =
            dsCore.primaryDimToVal_st(dim) * paddingSizesCore.at(dim).stride_;
        DT_CHECK_MSG(coreOffset > 0, "Expect a positive core offset.");
        const int negCoreOffset = (-1) * coreOffset;
        alphasPadFront
            [dsc2::TransferPadInfo::FoldDimPosition::WORK_SLICE_FOLDDIM] =
                negCoreOffset;

        // Beta for the padFront wkslice fold dimension is the padding size in
        // N_.
        betasPadFront
            [dsc2::TransferPadInfo::FoldDimPosition::WORK_SLICE_FOLDDIM] =
                padFrontN;

        // Alpha for the padBack wkslice fold dimension is the core stage
        // offset.
        alphasPadBack
            [dsc2::TransferPadInfo::FoldDimPosition::WORK_SLICE_FOLDDIM] =
                coreOffset;

        // Beta for the padBack wkslice fold dimension is computed as below:
        //   beta = padBackN + negCoreOffset * (numWkSlice - 1) + negChunkOffset
        //   * (numChunks - 1)
        const int chunkOffset =
            getDataStageLxSize(dim) * paddingSizesChunk.at(dim).stride_;
        DT_CHECK_MSG(chunkOffset > 0, "Expect a positive chunk offset.");
        const int negChunkOffset = (-1) * chunkOffset;
        const int padBackWkSliceFoldDimBeta = padBackN +
                                              negCoreOffset * (numWkSlice - 1) +
                                              negChunkOffset * (numChunks - 1);
        betasPadBack
            [dsc2::TransferPadInfo::FoldDimPosition::WORK_SLICE_FOLDDIM] =
                padBackWkSliceFoldDimBeta;

        // Alpha for the padFront chunk fold dimension is the negated chunk
        // stage offset.
        alphasPadFront[dsc2::TransferPadInfo::FoldDimPosition::CHUNK_FOLDDIM] =
            negChunkOffset;

        // Beta for the padFront chunk fold dimension is zero.
        const int padFrontChunkFoldDimBeta = 0;
        betasPadFront[dsc2::TransferPadInfo::FoldDimPosition::CHUNK_FOLDDIM] =
            padFrontChunkFoldDimBeta;

        // Alpha for the padBack chunk fold dimension is the chunk stage offset.
        alphasPadBack[dsc2::TransferPadInfo::FoldDimPosition::CHUNK_FOLDDIM] =
            chunkOffset;

        // Beta for the padBack chunk fold dimension is zero.
        const int padBackChunkFoldDimBeta = 0;
        betasPadBack[dsc2::TransferPadInfo::FoldDimPosition::CHUNK_FOLDDIM] =
            padBackChunkFoldDimBeta;

        // Build pad front sizes.
        transNode->paddingInfo_.buildPadFrontSizes(dim, sizes, alphasPadFront,
                                                   betasPadFront);

        // Build pad back sizes.
        transNode->paddingInfo_.buildPadBackSizes(dim, sizes, alphasPadBack,
                                                  betasPadBack);
      }

      // Set src unit_/storage_ to SenComponent::CONSTANT if it used to be
      // SenComponent::NO_COMPONENT.
      if (transNode->src_.unit_ == SenComponents::NO_COMPONENT)
        transNode->src_.unit_ = SenComponents::CONSTANT;
      if (transNode->src_.storage_ == SenComponents::NO_COMPONENT)
        transNode->src_.storage_ = SenComponents::CONSTANT;

      // Set dst unit_ to SenComponent::L3LU if not set yet.
      DT_CHECK_MSG(!transNode->dstVias_.empty(),
                   "Expect valid entry in dstVias_.");
      if (transNode->dstVias_.front().loc_.unit_ != SenComponents::L3LU)
        transNode->dstVias_.front().loc_.unit_ = SenComponents::L3LU;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 222/382   level 1   scc 72   236 body lines
// unit: e222_allocAllMem
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5508
// original: bool L3DlOpsScheduler::allocAllMem(const SuperDsc &mySDsc, DesignSpaceConfig *currDsc, const int dscIdx, bool commitIfValid)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// NOTE: The L3 scheduler's own allocation commit (distinct from Ddc::allocAllMem at
//       ddc/ddcv1.cpp:132 -- two different functions with the same name; the unit number
//       disambiguates).
// ------------------------------------------------------------------------------------------------
bool e222_allocAllMem(const SuperDsc &mySDsc,
                                   DesignSpaceConfig *currDsc, const int dscIdx,
                                   bool commitIfValid)
{
  auto &metadata = dscMetadata.at(dscIdx);
  std::map<dsc2::AllocateNode *,
           std::map<int, std::map<int, std::vector<int64_t>>>>
      startAddressCoreCorelet_;
  std::map<dsc2::AllocateNode *, std::map<int, std::map<int, int64_t>>>
      bufferOffsetCoreCorelet_;
  std::map<dsc2::AllocateNode *, std::pair<bool, bool>> copyToCoreCl;
  std::map<DsTrackInMem *, std::vector<std::vector<DsTrackInMem::DsMemInfo>>>
      trackerBackups;
  auto tryAlloc = [&]() {
    for (auto &[comp, allocMetadata] : metadata.newAllocations_) {
      std::vector<int> cores;
      bool copyCore = false;
      if (comp == SenComponents::LX) {
        cores = currDsc->coreIdsUsed_;
      } else {
        // for non-lx, we will use first core as proxy..
        cores.push_back(currDsc->coreIdsUsed_.front());
        copyCore = true;
      }
      // for corelets, rows use 0 as proxy
      bool copyCorelet = true;
      std::vector<int> corelets(1, 0);
      std::vector<int> rows(1, 0);
      for (auto &core : cores) {
        for (auto &corelet : corelets) {
          for (auto &row : rows) {
            auto myTracker = memTrackers->getTracker(comp, core, corelet, row);
            auto [it, didInsert] = trackerBackups.try_emplace(myTracker);
            if (didInsert) {
              for (const auto &exphase : exphases)
                it->second.push_back(myTracker->backupEps(exphase));
            }
            std::vector<std::pair<dsc2::AllocateNode *, int64_t>>
                nodeAndSize;  // in bytes..
            // currDsc->labeledDs_.at(acand.first).dsName_
            for (auto &[ldsIdx, allocNode] : allocMetadata.ldsIdxAndAllocNode) {
              // This is to compute the buffer size on LX for double-buffering.
              // The buffer size must be even number of sticks for ring porality
              // (refer to DSI).
              DT_CHECK_MSG(allocNode->component_ == SenComponents::LX,
                           "Expect only LX.");
              int numBuffers = allocNode->numBuffers_;
              if (numBuffers == -1)
                numBuffers = 2;  // reserve at least 2 buffers
              nodeAndSize.emplace_back(
                  allocNode,
                  numBuffers * currDsc->getBufferCapacityForNode(
                                   allocNode, ldsIdx, allocNode->component_,
                                   corelet, row, dscGlobal.sysDef.bytesPerStick,
                                   /*forceEvenNumSticks*/ true));
            }
            for (auto &acand : allocMetadata.consIdAndAllocNode) {
              DT_ERROR("No support");
              nodeAndSize.emplace_back(acand.second,
                                       dscGlobal.sysDef.bytesPerStick);
            }
            for (auto &acand : allocMetadata.compAndAllocNode) {
              DT_ERROR("No support");
              DT_CHECK(acand.second->numBuffers_ == 1);
              auto &opMetadata = metadata.opaqueOps_.at(acand.first);
              // for now assuming linear growth of registers
              auto size = currDsc->getBufferCapacityForNode(
                  acand.second, acand.second->ldsIdx_, acand.second->component_,
                  corelet, row);
              auto numSticks = size / dscGlobal.sysDef.bytesPerStick;
              // fail if more than the opaque op can handle or if not a power of
              // 2
              if (numSticks > opMetadata.max_unroll_ || numSticks == 0 ||
                  (numSticks & (numSticks - 1)))
                return false;
              nodeAndSize.emplace_back(acand.second,
                                       size * opMetadata.internalRegs_.size());
            }
            // Sort nodeAndSize by the size from largest to smallest. This is
            // required for LX memory to make sure that we allocate LX memory
            // for the nodes in this order. The reason is that, there can be
            // tensors from other nodes residing in LX, which causes LX memory
            // fragmentation for the current allocation. By allocating in this
            // order, we reduce the impact from the fragmentation, so that we
            // can put larger or more buffers in LX. For the other memory types,
            // there is no impact from sorting.
            std::sort(nodeAndSize.begin(), nodeAndSize.end(),
                      [](std::pair<dsc2::AllocateNode *, int64_t> &a,
                         std::pair<dsc2::AllocateNode *, int64_t> &b) {
                        return a.second > b.second;
                      });
            if (verbose > 0) {
              std::cout << "Comp: "
                        << EnumsConversion::senComponentsToString.at(comp)
                        << std::endl;
              for (auto &kv1 : nodeAndSize) {
                std::cout << getLdsOrConstNameOfAllocNode(currDsc, kv1.first)
                          << " " << kv1.second << std::endl;
              }
            }
            for (auto &kv : nodeAndSize) {
              myTracker->removeDs(
                  getLdsOrConstNameOfAllocNode(currDsc, kv.first), exphases);
            }
            for (auto &kv : nodeAndSize) {
              auto mySize = kv.second;
              if (kv.first->numBuffers_ == -1) {
                // full capacity reserved for circular buffer..
                mySize = std::max(mySize, myTracker->memCapacity);
              }
              std::vector<int64_t> addresses;
              auto tryDsAlloc = [&](const std::vector<int> &eps) {
                auto addr = myTracker->checkAndAddDs(
                    getLdsOrConstNameOfAllocNode(currDsc, kv.first), mySize,
                    eps);
                DT_CHECK(addr != EXISTS);
                if (addr == DOESNT_FIT) {
                  if (verbose > 0) {
                    std::cout << getLdsOrConstNameOfAllocNode(currDsc, kv.first)
                              << " " << mySize << std::endl;
                    for (const auto &ep : eps)
                      myTracker->printExPhase(std::cout, ep);
                  }
                  return false;
                }
                addresses.push_back(addr);
                return true;
              };
              bool success = false;
              if (!success) {
                // try allocating all eps separately
                for (const auto &exphase : exphases) {
                  success = tryDsAlloc({exphase});
                  if (!success) return false;
                }
              }
              if (commitIfValid) {
                if (std::equal(addresses.begin() + 1, addresses.end(),
                               addresses.begin())) {
                  // if all addresses are same, keep only one
                  addresses.resize(1);
                }
                if (dscGlobal.sysDef.coreArch <= MPW4_ISA && comp == PTARF &&
                    is_any_of(currDsc->labeledDs_.at(0).dataFormat_,
                              DataFormats::SENINT8, DataFormats::SENINT4)) {
                  for (auto &addr : addresses) addr += 4 * 128;
                }
                startAddressCoreCorelet_[kv.first][core][corelet] =
                    std::move(addresses);
                int numBuffers = kv.first->numBuffers_;
                if (numBuffers == -1) numBuffers = 2;
                bufferOffsetCoreCorelet_[kv.first][core][corelet] =
                    kv.second / numBuffers;
                copyToCoreCl[kv.first] = std::make_pair(copyCore, copyCorelet);
              }
            }
          }
        }
      }
    }
    return true;
  };
  bool success = tryAlloc();
  if (success && commitIfValid) {
    std::deque<const FoldDimProp *> sdscFoldProps{
        mySDsc.coreFoldProp_.get(), mySDsc.coreletFoldProp_.get()};
    for (const auto &foldProp : mySDsc.sdscFoldProps_) {
      sdscFoldProps.push_back(foldProp.get());
    }
    for (auto &[alloc, addrMap] : startAddressCoreCorelet_) {
      const BaseFuncType defaultFold = [&addrMap = addrMap] {
        for (auto &[core, clAddr] : addrMap) {
          for (auto &[cl, addr] : clAddr) {
            if (addr.size() > 1) return BaseFuncType::Map;
          }
        }
        return BaseFuncType::Constant;
      }();

      std::deque<BaseFuncType> foldTypes(sdscFoldProps.size(), defaultFold);
      auto &addrFM = alloc->startAddressCoreCorelet_;
      DT_CHECK(addrFM.hasZeroFoldDim());
      // use const fold if address does not change across core and/or corelet
      const auto &[copyCore, copyCl] = copyToCoreCl.at(alloc);
      DT_CHECK(!copyCore || addrMap.size() == 1);
      foldTypes[0] = copyCore ? BaseFuncType::Constant : BaseFuncType::Map;
      DT_CHECK(!copyCl || addrMap.begin()->second.size() == 1);
      // always set corelet fold as map, cl1 will be calculated later
      foldTypes[1] = BaseFuncType::Map;
      // build inner to outer
      addrFM.buildFoldSpace(sdscFoldProps, foldTypes);
      auto sdscFoldCoords = addrFM.getFlattenedCoordinates({{0, 0}, {1, 0}});
      for (auto &[core, clAddr] : addrMap) {
        for (auto &[cl, addr] : clAddr) {
          const auto size = addr.size();
          DT_CHECK(size == 1 || size == sdscFoldCoords.size());
          const bool multiAddr = size > 1;
          auto addrIt = addr.cbegin();
          for (auto &coord : sdscFoldCoords) {
            coord[0] = core;
            coord[1] = cl;
            addrFM.insertData(*addrIt, coord);
            if (multiAddr) addrIt++;
          }
        }
      }
    }
    for (auto &kv : bufferOffsetCoreCorelet_) {
      kv.first->bufferOffsetCoreCorelet_ = kv.second;
    }
    for (auto &kv : copyToCoreCl) {
      if (kv.second.second) {
        // copy to other corelets used..
        for (auto &corekv : kv.first->bufferOffsetCoreCorelet_) {
          for (int cl = 1; cl < currDsc->numCoreletsUsed_; cl++) {
            corekv.second[cl] = corekv.second.at(0);
          }
        }
      }
      if (kv.second.first) {
        // copy to other cores..
        auto &headCore = currDsc->coreIdsUsed_.front();
        for (int cidx = 1; cidx < currDsc->coreIdsUsed_.size(); cidx++) {
          auto &newCore = currDsc->coreIdsUsed_.at(cidx);
          if (kv.first->numBuffers_ != 1) {
            kv.first->bufferOffsetCoreCorelet_[newCore] =
                kv.first->bufferOffsetCoreCorelet_.at(headCore);
          }
        }
      }
    }
  } else {
    for (const auto &[myTracker, backupInfo] : trackerBackups) {
      for (int i = 0; i < exphases.size(); i++)
        myTracker->restoreEps(exphases.at(i), backupInfo.at(i));
    }
  }
  return success;
}

// ------------------------------------------------------------------------------------------------
// entry 223/382   level 1   scc 121   30 body lines
// unit: e223_getHbmLdsTransferHMIRequestEstimate
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6452
// original: int L3DlOpsScheduler::getHbmLdsTransferHMIRequestEstimate( const SuperDsc &mySDsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
int e223_getHbmLdsTransferHMIRequestEstimate(
    const SuperDsc &mySDsc) const
{
  // Use the first DSC to get all HBM pinned tensors.
  const auto &mainDsc = mySDsc.dscs_.front();
  std::vector<int> hbmLdsIndices;
  for (const auto &lds : mainDsc.labeledDs_) {
    if (lds.isHbmPinned()) hbmLdsIndices.push_back(lds.ldsIdx_);
  }

  // Compute the minimum number of HMI requests among all HBM tensors depending
  // on SENARCH.
  if (dscGlobal.sysDef.coreArch < IsaCoreGen::SEN1P5_ISA) {
    // SENARCH prior to SEN1P5_ISA only has one HMI. Therefore, all work
    // slices share the same HMI and we use the number of work slices to
    // estimate the number of HMI requests.
    int minNumRequests = std::numeric_limits<int>::max();
    for (const auto ldsIdx : hbmLdsIndices) {
      const auto ldsNonBroadcastDims = mainDsc.getNonBroadcastLdsDims(ldsIdx);
      int ldsNumWkSlices = 1;
      for (const auto dim : ldsNonBroadcastDims) {
        ldsNumWkSlices *= mySDsc.numWkSlicesPerDim_.at(dim);
      }
      minNumRequests = std::min(minNumRequests, ldsNumWkSlices);
    }
    return minNumRequests;
  } else if (dscGlobal.sysDef.coreArch == IsaCoreGen::SEN1P5_ISA) {
    return computeMinHMICoreGroupSizeForSEN1P5(mySDsc, hbmLdsIndices);
  } else {
    DT_ERROR("Unsupported SENARCH.");
  }
}

// ------------------------------------------------------------------------------------------------
// entry 224/382   level 1   scc 124   8 body lines
// unit: e224_getAllPagedLdsIndices
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6731
// original: std::vector<int> L3DlOpsScheduler::getAllPagedLdsIndices( const DesignSpaceConfig &dsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::vector<int> e224_getAllPagedLdsIndices(
    const DesignSpaceConfig &dsc) const
{
  std::vector<int> ldsIndices;
  for (const auto &lds : dsc.labeledDs_) {
    if (isPagedLds(lds)) ldsIndices.push_back(lds.ldsIdx_);
  }

  return ldsIndices;
}

// ------------------------------------------------------------------------------------------------
// entry 225/382   level 1   scc 126   62 body lines
// unit: e225_createPagedDimChunkLoops
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6804
// original: std::unordered_map<PrimaryDimTypes, dsc2::LoopNode *> L3DlOpsScheduler::createPagedDimChunkLoops( DesignSpaceConfig &dsc, const std::vector<PrimaryDimTypes> &pagedDims, std::set<dsc2::LoopNode *> &chunkLoopNodes)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::unordered_map<PrimaryDimTypes, dsc2::LoopNode *>
e225_createPagedDimChunkLoops(
    DesignSpaceConfig &dsc, const std::vector<PrimaryDimTypes> &pagedDims,
    std::set<dsc2::LoopNode *> &chunkLoopNodes)
{
  // Process each paged dimension.
  std::unordered_map<PrimaryDimTypes, dsc2::LoopNode *> origChunkLoopNodes;
  std::unordered_map<PrimaryDimTypes, dsc2::LoopNode *> newChunkLoopNodes;
  for (const auto dim : pagedDims) {
    // Expand the chunk loop into a loop nest.
    dsc2::LoopNode *origChunkLoopNode = nullptr;
    for (auto &node : chunkLoopNodes) {
      // TODO: Support multiple dimensions in a chunk loop node. In this case,
      // we must split the paged dimension out of this loop node at first.
      DT_CHECK_MSG(node->dims_.size() == 1,
                   "Only support one dimension in a chunk loop node for now.");
      const auto loopDim = node->dims_.front().dim_;
      if (dim == loopDim) {
        origChunkLoopNode = node;
        break;
      }
    }
    DT_CHECK_MSG(origChunkLoopNode &&
                     origChunkLoopNode->numId_ == dataStageCoreIdx &&
                     ((lxBufferType == BufferType::DOUBLE &&
                       origChunkLoopNode->denId_ == dataStageChunkIdx) ||
                      (lxBufferType == BufferType::SPATIAL_DOUBLE &&
                       origChunkLoopNode->denId_ == dataStageSuperChunkIdx)),
                 "Expect a valid chunk loop.");
    origChunkLoopNodes.emplace(dim, origChunkLoopNode);

    // Create a core/ibr loop node.
    std::string coreIbrLoopName =
        "loop_core_ibr_ds" + std::to_string(origChunkLoopNode->numId_) + "_ds" +
        std::to_string(dataStageIbrIdx) + "_" +
        EnumsConversion::primaryDimToString.at(dim);
    dsc2::LoopNode* newIbrLoopNode =
        createLoopNode(dsc, {dim}, origChunkLoopNode->numId_, dataStageIbrIdx,
                       coreIbrLoopName);

    // Create a ibr/chunk or ibr/SuperChunk loop node.
    std::string chunkIbrLoopName =
        "loop_ibr_chunk_ds" + std::to_string(dataStageIbrIdx) + "_ds" +
        std::to_string(origChunkLoopNode->denId_) + "_" +
        EnumsConversion::primaryDimToString.at(dim);
    dsc2::LoopNode* newChunkLoopNode =
        createLoopNode(dsc, {dim}, dataStageIbrIdx, origChunkLoopNode->denId_,
                       chunkIbrLoopName);
    newChunkLoopNodes.emplace(dim, newChunkLoopNode);

    // Connect the new loop nest in schedule tree to replace the original
    // chunk loop node.
    auto parentOrigChunkLoopNode = origChunkLoopNode->getMutableParent();
    parentOrigChunkLoopNode->addChildNode(newIbrLoopNode, /*addBefore*/ true,
                                          origChunkLoopNode);
    newIbrLoopNode->addChildNode(newChunkLoopNode);
    origChunkLoopNode->moveChildren(newChunkLoopNode);

    // TODO: Remove origChunkLoopNode if it only has the current dimension.
    // Otherwise, we need to keep this node alive for the other dimensions.
    chunkLoopNodes.erase(origChunkLoopNode);
    parentOrigChunkLoopNode->deleteChildNode(&dsc, origChunkLoopNode);
  }

  return newChunkLoopNodes;
}

// ------------------------------------------------------------------------------------------------
// entry 226/382   level 1   scc 128   62 body lines
// unit: e226_createStoreIndexTensorToIbr
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7037
// original: void L3DlOpsScheduler::createStoreIndexTensorToIbr( DesignSpaceConfig &dsc, const int dscIdx, const int indexLdsIdx, dsc2::AllocateNode &indexLdsHbmAllocNode, dsc2::LoopNode &newChunkLoopNode, const bool isTransferIn)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e226_createStoreIndexTensorToIbr(
    DesignSpaceConfig &dsc, const int dscIdx, const int indexLdsIdx,
    dsc2::AllocateNode &indexLdsHbmAllocNode, dsc2::LoopNode &newChunkLoopNode,
    const bool isTransferIn)
{
  auto &indexLds = dsc.labeledDs_.at(indexLdsIdx);
  // Create the IBR allocate and transfer node for its index tensor. Also
  // add an MemOrg entry for IBR for the index tensor's labeledDs if not
  // exists.
  const SenComponents srcUnit =
      isTransferIn ? SenComponents::L3LU : SenComponents::L3SU;
  const SenComponents dstUnit =
      isTransferIn ? SenComponents::L3LU : SenComponents::L3SU;
  const SenComponents srcStorage =
      isTransferIn ? SenComponents::HBM : SenComponents::LX;
  const SenComponents dstStorage =
      isTransferIn ? SenComponents::L3LUIBR : SenComponents::L3SUIBR;
  const std::string allocIbrNodeName =
      "allocate_lds" + std::to_string(indexLds.ldsIdx_) + "_" +
      EnumsConversion::senComponentsToString.at(dstStorage);
  dsc2::AllocateNode *newIbrAllocNode =
      createAllocateNode(dsc, indexLdsIdx, dstStorage, 1 /* no buffering */,
                         allocIbrNodeName, dscIdx);
  newIbrAllocNode->indirectAllocType_ = indexLdsHbmAllocNode.indirectAllocType_;
  newIbrAllocNode->relatedIndirectAccessAlloc_ =
      indexLdsHbmAllocNode.relatedIndirectAccessAlloc_;
  // Add memorg entry if it doesn't exist.
  indexLds.memOrg_[dstStorage].allocateNode_ = newIbrAllocNode;
  const std::string transIbrNodeName =
      "transfer_lds" + std::to_string(indexLds.ldsIdx_) +
      "_src:" + EnumsConversion::senComponentsToString.at(srcStorage) +
      "_dst:" + EnumsConversion::senComponentsToString.at(dstStorage);
  dsc2::TransferNode *newIbrTransNode =
      createTransferNode(srcUnit, srcStorage, {dstUnit}, {dstStorage},
                         indexLdsIdx, {indexLdsIdx}, transIbrNodeName);
  indexLdsHbmAllocNode.addAllocUser(newIbrTransNode);
  newIbrAllocNode->addAllocUser(newIbrTransNode);

  // Create a pair of L3LU or L3SU self-sync sync nodes.
  dsc2::SyncNode *newL3SyncSendNode = createSyncNode(
      {srcUnit},
      "sync_send_" + EnumsConversion::senComponentsToString.at(srcUnit) +
          "_to_" + EnumsConversion::senComponentsToString.at(srcUnit) +
          "_paged_index_" + std::to_string(indexLdsIdx));
  dsc2::SyncNode *newL3SyncRecvNode = createSyncNode(
      {srcUnit},
      "sync_receive_" + EnumsConversion::senComponentsToString.at(srcUnit) +
          "_from_" + EnumsConversion::senComponentsToString.at(srcUnit) +
          "_paged_index_" + std::to_string(indexLdsIdx),
      true);
  newL3SyncSendNode->otherEndOfTheSignals_.push_back(newL3SyncRecvNode);
  newL3SyncRecvNode->otherEndOfTheSignals_.push_back(newL3SyncSendNode);

  // Add the new allocate, transfer and sync nodes before the new chunk loop
  // node (ibr/chunk) for this dimension as siblings.
  auto parentNewChunkLoopNode = newChunkLoopNode.getMutableParent();
  DT_CHECK_MSG(parentNewChunkLoopNode, "Expect a valid parent node.");
  parentNewChunkLoopNode->addChildNode(newIbrAllocNode, /*addBefore*/ true,
                                       &newChunkLoopNode);
  parentNewChunkLoopNode->addChildNode(newIbrTransNode, /*addBefore*/ true,
                                       &newChunkLoopNode);
  parentNewChunkLoopNode->addChildNode(newL3SyncSendNode, /*addBefore*/ true,
                                       &newChunkLoopNode);
  parentNewChunkLoopNode->addChildNode(newL3SyncRecvNode, /*addBefore*/ true,
                                       &newChunkLoopNode);
}

// ------------------------------------------------------------------------------------------------
// entry 227/382   level 1   scc 127   45 body lines
// unit: e227_convertTransferDirectToIndirect
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7103
// original: void L3DlOpsScheduler::convertTransferDirectToIndirect( dsc2::TransferNode &transNode, DesignSpaceConfig &dsc, const int indexLdsIdx, const PrimaryDimTypes indexStickDim, const bool isTransferIn)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e227_convertTransferDirectToIndirect(
    dsc2::TransferNode &transNode, DesignSpaceConfig &dsc,
    const int indexLdsIdx, const PrimaryDimTypes indexStickDim,
    const bool isTransferIn)
{
  DT_CHECK_MSG(transNode.getTransferType() ==
                   dsc2::TransferNode::TransferType::TENSOR_TO_TENSOR,
               "Expect a tensor transfer.");
  const auto transOrigOwnerLoopDenId = transNode.getOwnerLoop()->denId_;
  DT_CHECK_MSG(
      transOrigOwnerLoopDenId == dataStageChunkIdx ||
          transOrigOwnerLoopDenId == dataStageSuperChunkIdx,
      "Expect the transfer owner loop to be a chunk or SuperChunk loop.");
  // Create a new chunk/1page or SuperChunk/1page loop node.
  const std::string onePageLoopNameSuffix = isTransferIn ? "_to_lx" : "_to_hbm";
  const std::string onePageLoopName =
      "loop_chunk_1page_ds" + std::to_string(transOrigOwnerLoopDenId) + "_ds" +
      std::to_string(dataStageOnePageIdx) + "_" +
      EnumsConversion::primaryDimToString.at(indexStickDim) +
      onePageLoopNameSuffix;
  dsc2::LoopNode *newOnePageLoopNode =
      createLoopNode(dsc, {indexStickDim}, transOrigOwnerLoopDenId,
                     dataStageOnePageIdx, onePageLoopName);

  // Connect the new chunk/1page loop and the transfer node in schedule
  // tree.
  auto parentOrigTransNode = transNode.getMutableParent();
  parentOrigTransNode->addChildNode(newOnePageLoopNode,
                                    /*addBefore*/ true, &transNode);
  transNode.moveNode(&dsc, newOnePageLoopNode);

  // Add indirection info in the transfer node.
  if (isTransferIn) {
    transNode.srcIndirect_.storage_ = SenComponents::L3LUIBR;
    transNode.srcIndirect_.unit_ = SenComponents::L3LU;
    transNode.srcIndirectLdsAndLoopOffsets_.myLdsIdx_ = indexLdsIdx;
  } else {
    DT_CHECK_MSG(transNode.dstVias_.size() == 1,
                 "Expect one entry in dstVias_.");
    transNode.dstVias_.front().locIndirect_.storage_ = SenComponents::L3SUIBR;
    transNode.dstVias_.front().locIndirect_.unit_ = SenComponents::L3SU;
    DT_CHECK_MSG(transNode.dstIndirectLdsAndLoopOffsets_.empty(),
                 "Expect empty dstIndirectLdsAndLoopOffsets_.");
    transNode.dstIndirectLdsAndLoopOffsets_.emplace_back();
    DT_CHECK_MSG(transNode.dstIndirectLdsAndLoopOffsets_.size() == 1,
                 "Expect one entry in dstIndirectLdsAndLoopOffsets_.");
    transNode.dstIndirectLdsAndLoopOffsets_.front().myLdsIdx_ = indexLdsIdx;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 228/382   level 1   scc 138   181 body lines
// unit: e228_buildCoordinateFromAllocation
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7333
// original: void L3DlOpsScheduler::buildCoordinateFromAllocation( DesignSpaceConfig &dsc, dsc2::AllocateNode *refAllocNode, dsc2::ScheduleNode *node, dsc2::CoordinateType<CoordinateBaseType> &coordinate) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e228_buildCoordinateFromAllocation(
    DesignSpaceConfig &dsc, dsc2::AllocateNode *refAllocNode,
    dsc2::ScheduleNode *node,
    dsc2::CoordinateType<CoordinateBaseType> &coordinate) const
{
  // TODO: Add an option to reconstruct the coordinate if it is constructed.
  if (coordinate.foldConstructed()) {
    return;
  }

  if (verbose > 0) {
    std::cout << "\n\n>>> buildFoldFromAllocation:"
              << "\n  node= " << node->name_ << "(" << node << ")"
              << "\n  refAllocNode= " << refAllocNode->name_ << "("
              << refAllocNode << ")";

    if (verbose > 2) {
      std::cout << "\n\n===================================================="
                << "\n\nReference fold: ";
      const_cast<dsc2::AllocateNode *>(refAllocNode)
          ->allocateCoordinates_.printCoordinates(std::cout);
    }
  }

  // Find the enclosing loop chain. Collect the associated dimensions.
  std::vector<dsc2::LoopNode *> loopChain;
  std::unordered_set<PrimaryDimAndKind> relatedDims;
  getEnclosingLoopsAndRelatedDims(node, &dsc, loopChain, relatedDims);

  auto &refCoord = refAllocNode->allocateCoordinates_;
  const auto &allocLds = dsc.labeledDs_.at(refAllocNode->ldsIdx_);

  for (auto &[dim, cfm] : refCoord.coordinates_) {
    bool coordDimIsRelatedToNode = false;
    for (auto &[relatedDim, relatedDimKind] : relatedDims) {
      if (dim == relatedDim) {
        coordDimIsRelatedToNode = true;
        break;
      }
    }

    if (!coordDimIsRelatedToNode) {
      // Is this an error condition?
      if (verbose > 1) {
        std::cout << "\n[buildFoldFromAllocation] scheduleNode " << node->name_
                  << " does not include dimension "
                  << EnumsConversion::primaryDimToString.at(dim)
                  << " even though reference allocateNode "
                  << refAllocNode->name_ << " includes the dimension."
                  << std::endl;
      }
      continue;
    }

    if (coordinate.coordinates_.count(dim)) {
      if (verbose > 1) {
        std::cout << "\n[buildFoldFromAllocation] scheduleNode " << node->name_
                  << "'s coordinates already includes fold for dimension "
                  << EnumsConversion::primaryDimToString.at(dim) << std::endl;
      }
      continue;
    }

    std::vector<dsc2::FoldParamInfoType> foldParams;
    gatherFoldParams(cfm, foldParams);

    dsc2::VectorOfLoopAndDim relatedLoops;
    int dimIdx = dsc.getDimIndexInLayoutOrder(allocLds.dsType_, dim);
    int scale = dimIdx < 0 ? 1 : allocLds.scale_.at(dimIdx);
    if (scale > 0) {
      // The dimension is a non-broadcast dimension.
      for (auto loop : loopChain) {
        // Note: dim is of type PrimaryDimTypes, metaDimKind is missing here.
        findAndStoreLoopWithDim(&dsc, dim, loop, relatedDims, relatedLoops,
                                coordinate.getPadding(dim));
      }
    }

    int spatialFoldEnds = refCoord.getNumOfSpatialFolds(dim) - 1;
    int refTemporalCount = refCoord.getNumOfTemporalFolds(dim);
    int temporalFoldEnds = spatialFoldEnds + refTemporalCount;

    if (refTemporalCount < relatedLoops.size()) {
      // The allocNode has more enclosing loops than the reference
      // allocateNode. Distribute the additional loops over the element
      // arrangement levels.
      int temporalDiff = relatedLoops.size() - refTemporalCount;

      // Insert fold parameters for the loop differential.

      dsc2::VectorOfLoopAndDim loopsToDistribute;
      // Scan order is from the innermost loop towards the outermost loop.
      for (int i = 0; i < temporalDiff; ++i) {
        loopsToDistribute.push_back(relatedLoops.at(i));
      }
      std::vector<dsc2::FoldParamInfoType> elemArrParamsAfterDistribution;
      std::vector<dsc2::FoldParamInfoType> elemArr;
      for (int i = foldParams.size() - 1; i > temporalFoldEnds; --i) {
        elemArr.push_back(foldParams.at(i));
      }

      // Distribute temporal loops over element arrangements and compute the
      // fold parameters for temporal and element arrangement folds.
      dsc2::LoopDistributionParamPerNodeType loopParamsAfterDistribution;
      const SenComponents sizeRefComp = SenComponents::L3LU;
      const SenComponents propRefComp = SenComponents::L3LU;
      distributeElemArrToTemporalLoops(
          &dsc, dim, node, refAllocNode->ldsIdx_, coordinate.getPadding(dim),
          coordinate.getPadding(dim), sizeRefComp, propRefComp,
          loopsToDistribute, elemArr, loopParamsAfterDistribution,
          elemArrParamsAfterDistribution, -1 /* for both corelets */, verbose);

      // Distribution may remove some original element arrangement levels. Clear
      // the old element arrangement folds from foldParams and add the element
      // arrangement levels from the result of distribution.
      foldParams.erase(foldParams.begin() + temporalFoldEnds + 1,
                       foldParams.end());
      // The innermost element arrangement is at the beginning
      // of elemArrParamsAfterDistribution.
      for (int j = elemArrParamsAfterDistribution.size() - 1; j >= 0; --j) {
        foldParams.push_back({elemArrParamsAfterDistribution.at(j).alpha,
                              elemArrParamsAfterDistribution.at(j).beta,
                              elemArrParamsAfterDistribution.at(j).cardinality,
                              "elem_arr_" + std::to_string(j)});
      }

      // Construct an iterator for the insertion point (beginning of element
      // arrangement folds).
      auto iter = foldParams.begin() + temporalFoldEnds + 1;
      std::string foldDimStr = EnumsConversion::primaryDimToString.at(dim);

      for (auto &[loop, loopDim, _] : loopsToDistribute) {
        int iterationCount = -1;
        if (loop->isParametricLoop()) {
          iterationCount = loop->parametricIterCount(
              &dsc, 0 /* To generalize*/, SenComponents::NO_COMPONENT, -1);
        } else {
          auto &numDs = dsc.dataStageParam_.at(loop->numId_).ss_;
          auto &denDs = dsc.dataStageParam_.at(loop->denId_).ss_;
          // Assumption: Spatial fold includes a level for corelets. Therefore,
          // get the datastage values per corelet.
          //
          // To do:
          //   Need to have special case when the corelets may have imbalanced
          //   distribution.
          auto denVal =
              denDs.dataStageDimToVal_compView_st(loopDim.dim_, propRefComp, 0);
          iterationCount = numDs.dataStageDimToVal_compView_st(loopDim.dim_,
                                                               propRefComp, 0) /
                           denVal;
        }
        iter = foldParams.insert(
            iter, {loopParamsAfterDistribution.at(loop).at(dim).alpha,
                   loopParamsAfterDistribution.at(loop).at(dim).beta,
                   iterationCount, loop->name_ + " " + foldDimStr});
      }
      temporalFoldEnds += temporalDiff;
    }

    // Construct the folds
    dsc2::CoordinateCategory coordCat = dsc2::CoordinateCategory::UNKNOWN_COORD;
    for (int i = foldParams.size() - 1; i >= 0; --i) {
      std::string foldDimLabel = foldParams.at(i).foldDimLabel;
      if (i > temporalFoldEnds) {
        coordCat = dsc2::CoordinateCategory::ELEM_ARR_COORD;
        foldDimLabel = "elem_arr_" + std::to_string(foldParams.size() - 1 - i);
      } else if (i > spatialFoldEnds) {
        coordCat = dsc2::CoordinateCategory::TEMPORAL_COORD;
      } else {
        coordCat = dsc2::CoordinateCategory::SPATIAL_COORD;
      }
      coordinate.addFold(dim, coordCat, foldParams.at(i).cardinality,
                         foldDimLabel, foldParams.at(i).alpha,
                         foldParams.at(i).beta, 0);
    }
  }
  coordinate.completeFoldConstruction();
  if (verbose > 2) {
    std::cout << "\n\n===================================================="
              << "\n\n[buildCoordinateFromAllocation] fold at end:"
              << "\n  refAllocNode = " << refAllocNode->name_
              << "\n  node         = " << node->name_;
    coordinate.printCoordinates(std::cout);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 229/382   level 1   scc 141   203 body lines
// unit: e229_sliceCoordinateForCorelet
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7518
// original: void L3DlOpsScheduler::sliceCoordinateForCorelet( SuperDsc &mySDsc, DesignSpaceConfig *currDsc, dsc2::AllocateNode *allocNode) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e229_sliceCoordinateForCorelet(
    SuperDsc &mySDsc, DesignSpaceConfig *currDsc,
    dsc2::AllocateNode *allocNode) const
{
  if (currDsc->numCoreletsUsed_ == 1) {
    return;
  }

  if (allocNode->component_ != SenComponents::LX) {
    return;
  }

  auto &chunkDs = currDsc->dataStageParam_.at(dataStageChunkIdx);
  if (chunkDs.ss_.coreletSplit_.empty()) {
    return;
  }
  PrimaryDimTypes coreletSplitDim = chunkDs.ss_.coreletSplit_.begin()->first;
  if (!allocNode->allocateCoordinates_.coordinates_.count(coreletSplitDim)) {
    return;
  }

  // Check if the coreletSplitdim is a broadcast dimension.
  const auto &allocLds = currDsc->labeledDs_.at(allocNode->ldsIdx_);
  int dimIdx =
      currDsc->getDimIndexInLayoutOrder(allocLds.dsType_, coreletSplitDim);
  int scale = dimIdx < 0 ? 1 : allocLds.scale_.at(dimIdx);
  if (scale < 0) {
    return;
  }

  // Construct an artificial loop to simulate corelet level slicing.
  int denId = constructDatastage(currDsc, chunkDs);

  // Setup the denominator dataStage according to the corelet-slicing strategry.
  //
  // Slicing strategy: Chunk half-and-half.
  currDsc->dataStageParam_.at(denId).ss_.primaryDimToValHandler_st(
      coreletSplitDim) /= currDsc->numCoreletsUsed_;
  currDsc->dataStageParam_.at(denId).el_.primaryDimToValHandler_st(
      coreletSplitDim) /= currDsc->numCoreletsUsed_;
  if (currDsc->dataStageParam_.at(denId).ss_.coreletSplit_.count(
          coreletSplitDim)) {
    for (auto &splitDimSize :
         currDsc->dataStageParam_.at(denId).ss_.coreletSplit_.at(
             coreletSplitDim)) {
      splitDimSize /= currDsc->numCoreletsUsed_;
    }
  }
  if (currDsc->dataStageParam_.at(denId).el_.coreletSplit_.count(
          coreletSplitDim)) {
    for (auto &splitDimSize :
         currDsc->dataStageParam_.at(denId).el_.coreletSplit_.at(
             coreletSplitDim)) {
      splitDimSize /= currDsc->numCoreletsUsed_;
    }
  }

  dsc2::LoopNode *newLoop =
      constructLoopNode(dataStageChunkIdx, denId, {coreletSplitDim});

  // Distribute the existing element arrangements.
  dsc2::LoopDistributionParamPerNodeType loopParamsAfterDistribution;
  std::vector<dsc2::FoldParamInfoType> elemArrParamsAfterDistribution;
  PadType allocPadding =
      allocNode->allocateCoordinates_.getPadding(coreletSplitDim);
  PrimaryDimAndKind coreletSplitDimAndKind = {
      coreletSplitDim, (allocPadding == PadType::NOPAD ? MetaDimKind::Unpadded
                                                       : MetaDimKind::Padded)};
  dsc2::VectorOfLoopAndDim relatedLoops;
  relatedLoops.push_back(
      {newLoop, coreletSplitDimAndKind,
       dsc2::LoopDistributionInfo::LoopDistributionCat::ABOVE_CHUNK});

  bool isLxPinned = allocNode->allocateCoordinates_.getNumOfTemporalFolds(
                        coreletSplitDim) == 0;
  if (isLxPinned) {
    // LX-pinned allocation.
    //   Include all chunk loops for distribution. The folds for these chunk
    //   loops would become outer element arrangements in the allocateNode's
    //   coordinate. Order in relatedLoops is from inner (pos 0) to outer (end
    //   of list).
    auto *lxBelowBlockNode = getLxBelowBlockNode(currDsc->scheduleTree_);
    DT_CHECK_MSG(lxBelowBlockNode, "Expect a valid lx_below block node.");
    dsc2::LoopNode *currLoop = lxBelowBlockNode->getMutableOwnerLoop();
    while (currLoop) {
      if (!currLoop->getMutableOwnerLoop()) {
        // Exclude the root loop.
        break;
      }
      if (dsc2::loopRelevantForDim(currDsc, coreletSplitDimAndKind, currLoop,
                                   allocPadding)) {
        relatedLoops.push_back(
            {currLoop, coreletSplitDimAndKind,
             dsc2::LoopDistributionInfo::LoopDistributionCat::ABOVE_CHUNK});
      }
      currLoop = currLoop->getMutableOwnerLoop();
    }
  }

  std::vector<dsc2::FoldParamInfoType> foldParams;
  gatherFoldParams(
      allocNode->allocateCoordinates_.coordinates_.at(coreletSplitDim),
      foldParams);

  int spatialFoldEnds =
      allocNode->allocateCoordinates_.getNumOfSpatialFolds(coreletSplitDim) - 1;
  int temporalFoldEnds =
      spatialFoldEnds +
      allocNode->allocateCoordinates_.getNumOfTemporalFolds(coreletSplitDim);

  std::vector<dsc2::FoldParamInfoType> elemArr;
  for (int i = foldParams.size() - 1; i > temporalFoldEnds; --i) {
    elemArr.push_back(foldParams.at(i));
  }

  // Temporarily make the allocNode a child of the new loop. The original
  // scheduleTree will be restored after the distribution. For computing custom
  // dataStage, the distribution process requires that the loop has a childNode.
  allocNode->getMutableParent()->addChildNode(newLoop, true, allocNode);
  allocNode->getMutableParent()->moveChildNode(currDsc, allocNode, newLoop);

  // Distribute temporal loops over element arrangements and compute the fold
  // parameters for temporal and element arrangement folds.
  dsc2::distributeElemArrToTemporalLoops(
      currDsc, coreletSplitDim, allocNode, allocNode->ldsIdx_, allocPadding,
      allocPadding, allocNode->component_, allocNode->component_, relatedLoops,
      elemArr, loopParamsAfterDistribution, elemArrParamsAfterDistribution,
      -1 /* for both corelets */, verbose);

  // Update wkSlice fold.
  auto &loopParams =
      loopParamsAfterDistribution.at(newLoop).at(coreletSplitDim);
  constexpr int wkSliceFoldPos =
      static_cast<int>(dsc2::CoordinateFoldPosition::Core);
  const int wkSliceFoldCardinality =
      currDsc->numCoreletsUsed_DSC2_ *
      mySDsc.numWkSlicesPerDim_.at(coreletSplitDim);
  foldParams.at(wkSliceFoldPos) = {loopParams.alpha, loopParams.beta,
                                   wkSliceFoldCardinality, "workslice_fold"};

  // Update temporal folds.
  for (int i = foldParams.size() - 1; i >= 0; --i) {
    const bool isTemporalFold = (i > spatialFoldEnds && i <= temporalFoldEnds);
    if (isTemporalFold) {
      // Adjust the alpha value based on the corelet slicing strategy.
      // Slicing strategy: Chunk half-and-half.
      foldParams.at(i).alpha /= currDsc->numCoreletsUsed_DSC2_;
    }
  }

  // Restore the original schedule tree.
  allocNode->getMutableParent()->moveChildNode(
      currDsc, allocNode, newLoop->getMutableParent(), true, newLoop);
  allocNode->getMutableParent()->deleteChildNode(currDsc, newLoop);

  // Cleanup the new dataStage.
  currDsc->dataStageParam_.erase(denId);

  // Update the coordinate folds.
  foldParams.erase(foldParams.begin() + temporalFoldEnds + 1, foldParams.end());
  if (isLxPinned) {
    // Order in relatedLoops: inner (beginning) to outer (end).
    for (auto &relatedLoopInfo : reverse(relatedLoops)) {
      auto &currLoop = relatedLoopInfo.loopNode;
      if (currLoop == newLoop) {
        // This is the updated corelet fold processed above.
        continue;
      }
      int iterationCnt = currDsc->dataStageParam_.at(currLoop->numId_)
                             .ss_.dataStageDimToVal_compView_st(
                                 coreletSplitDim, allocNode->component_) /
                         currDsc->dataStageParam_.at(currLoop->denId_)
                             .ss_.dataStageDimToVal_compView_st(
                                 coreletSplitDim, allocNode->component_);
      auto &currLoopParams =
          loopParamsAfterDistribution.at(currLoop).at(coreletSplitDim);
      foldParams.push_back({currLoopParams.alpha, currLoopParams.beta,
                            iterationCnt, "elem_arr_chunk_level"});
    }
  }

  // The innermost element arrangement is at the beginning
  // of elemArrParamsAfterDistribution.
  for (int j = elemArrParamsAfterDistribution.size() - 1; j >= 0; --j) {
    foldParams.push_back({elemArrParamsAfterDistribution.at(j).alpha,
                          elemArrParamsAfterDistribution.at(j).beta,
                          elemArrParamsAfterDistribution.at(j).cardinality,
                          "elem_arr_" + std::to_string(j)});
  }

  allocNode->allocateCoordinates_.clearFoldForDim(coreletSplitDim);
  dsc2::CoordinateCategory coordCat = dsc2::CoordinateCategory::UNKNOWN_COORD;
  for (int i = foldParams.size() - 1; i >= 0; --i) {
    std::string foldDimLabel = foldParams.at(i).foldDimLabel;
    if (i > temporalFoldEnds) {
      coordCat = dsc2::CoordinateCategory::ELEM_ARR_COORD;
    } else if (i > spatialFoldEnds) {
      coordCat = dsc2::CoordinateCategory::TEMPORAL_COORD;
    } else {
      coordCat = dsc2::CoordinateCategory::SPATIAL_COORD;
    }
    allocNode->allocateCoordinates_.addFold(
        coreletSplitDim, coordCat, foldParams.at(i).cardinality, foldDimLabel,
        foldParams.at(i).alpha, foldParams.at(i).beta, 0);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 283/382   level 2   scc 6   20 body lines
// unit: e283_addOrUpdateCoreletSplitInParams
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:105
// original: static void addOrUpdateCoreletSplitInParams(DataStructDims &params, const DesignSpaceConfig &dsc)
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e283_addOrUpdateCoreletSplitInParams(DataStructDims &params,
                                            const DesignSpaceConfig &dsc)
{
  std::vector<PrimaryDimTypes> coreletSplitDims =
      getCoreletSplitDimensions(dsc);
  for (auto dim : coreletSplitDims) {
    const int numCoreletsPerCore = dsc.numCoreletsUsed_;
    if (params.primaryDimToVal_st(dim) == -1) {
      continue;
    }
    DT_CHECK_MSG(params.primaryDimToVal_st(dim) % numCoreletsPerCore == 0,
                 "Invalid corelet split.");
    const int coreletVal = params.primaryDimToVal_st(dim) / numCoreletsPerCore;
    if (params.coreletSplit_.count(dim) == 0)
      params.coreletSplit_.emplace(dim, std::vector<int>());
    else
      params.coreletSplit_.at(dim).clear();

    for (int i = 0; i < numCoreletsPerCore; ++i)
      params.coreletSplit_.at(dim).push_back(coreletVal);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 284/382   level 2   scc 48   4 body lines
// unit: e284_isOpFuncStridedWindow
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:865
// original: bool L3DlOpsScheduler::isOpFuncStridedWindow(const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
bool e284_isOpFuncStridedWindow(const OpFuncs opFuncName) const
{
  return isOpFuncConv2d(opFuncName) || isOpFuncPooling(opFuncName) ||
         isOpFuncDepthwiseConv(opFuncName);
}

// ------------------------------------------------------------------------------------------------
// entry 285/382   level 2   scc 77   267 body lines
// unit: e285_calculateBurstEfficiency
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1738
// original: double L3DlOpsScheduler::calculateBurstEfficiency( const SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &primaryDims)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
double e285_calculateBurstEfficiency(
    const SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &primaryDims)
{
  // A map from each pair of burst size (1 to 32) and multicast degree (1 to 32)
  // to the number of data transfer requests for this pair.
  std::unordered_map<unsigned, std::unordered_map<unsigned, unsigned long>>
      burstSizeAndMulticastToNumReqMap;

  auto addToMap = [&burstSizeAndMulticastToNumReqMap](
                      const unsigned burstSize, const unsigned multicastDegree,
                      const unsigned long numRequests) {
    if (!burstSizeAndMulticastToNumReqMap.count(burstSize))
      burstSizeAndMulticastToNumReqMap.emplace(
          burstSize, std::unordered_map<unsigned, unsigned long>());
    if (!burstSizeAndMulticastToNumReqMap.at(burstSize).count(multicastDegree))
      burstSizeAndMulticastToNumReqMap.at(burstSize).emplace(multicastDegree,
                                                             0);

    burstSizeAndMulticastToNumReqMap[burstSize][multicastDegree] += numRequests;
  };

  // Since all labeledDs are the same for all DSCs in a group, use DSC 0 to
  // collect all labeledDs that need chunking.
  const int dscMainIdx = 0;
  const DesignSpaceConfig &dscMain = mySDsc.dscs_.at(dscMainIdx);
  std::vector<int> labeledDsIndices;
  for (int ldsIdx = 0; ldsIdx < dscMain.labeledDs_.size(); ++ldsIdx) {
    const LabeledDsInfo &lds = dscMain.labeledDs_.at(ldsIdx);
    if ((lds.isHbmPinned() || isLabeledDsLXNeighbor(mySDsc, dscMainIdx, lds)) &&
        !isIndexLds(lds))
      labeledDsIndices.push_back(ldsIdx);
  }

  // Core split dimensions among DSCs.
  std::unordered_set<PrimaryDimTypes> coreSplitDims =
      getCoreSplitDimensions(mySDsc);
  DT_CHECK_MSG(
      coreSplitDims.size() <= 1 && mySDsc.dscs_.size() <= 2,
      "Currently support at most one core split dimension with two DSCs.");

  // Compute the following for each collected labeledDs:
  //   1) Total number of data transfer requests
  //   2) Burst size of each request
  //   3) Multicast degree of each request
  for (const int ldsIdx : labeledDsIndices) {
    // Use dscMain to check if this labeledDs has core split dimensions.
    bool ldsHasCoreSplitDim = false;
    for (PrimaryDimTypes dim : dscMain.getLayoutDims(ldsIdx)) {
      ldsHasCoreSplitDim = coreSplitDims.count(dim);
      if (ldsHasCoreSplitDim) break;
    }

    // Compute the number of data transfer requests for this tensor. The
    // maximum burst size per request is 32. When the stick volume is greater
    // than 32, there must be more than one request. In this case, currently
    // the code gen strategy is to have as many requests of burst size 32 as
    // possible. For exmaple, if the stick volume is 65, there will be three
    // requests of burst size-32, 32, and 1. We have to update here if the
    // code gen strategy changes in future.
    const unsigned maxBurstSize = 32;
    if (ldsHasCoreSplitDim) {
      // The coreD of this labeledDs in each DSC must be different. The same
      // work slice must only be used in one DSC. Hence, we compute by
      // processing each DSC separately.
      for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
        const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
        const auto &lds = dsc.labeledDs_.at(ldsIdx);
        const auto ldsNonBroadcastDimsSet =
            dsc.getNonBroadcastLdsDimSet(ldsIdx);
        DT_CHECK(dsc.dataStageParam_.count(dataStageCoreIdx) &&
                 dsc.dataStageParam_.count(dataStageChunkIdx) &&
                 "Expect valid dataStageParam_ entries.");
        unsigned long stickVolume = getLabeledDsChunkStickVolume(dsc, ldsIdx);
        const unsigned long numRequestsOfMaxBurst = stickVolume / maxBurstSize;
        const unsigned otherBurstSize = stickVolume % maxBurstSize;

        // Number of stick volumes for the coreD amount of work.
        const unsigned long numStickVolumes =
            getLabeledDsNumOfStickVolumesInCore(dsc, ldsIdx, stickVolume,
                                                primaryDims);

        // Get the tensor's L3 transfer nodes. If there are multiple nodes, they
        // should be on the same loop level. Hence, use any one of them.
        // TODO: We should consider all transfer nodes in the total amount of
        // data transferred.
        const auto l3TransNodes = getLdsL3TransferNodes(
            mySDsc, dscIdx, ldsIdx,
            {SenComponents::HBM, SenComponents::NO_COMPONENT},
            {SenComponents::LX});
        DT_CHECK_MSG(!l3TransNodes.empty(), "Expect valid transfer nodes.");
        const auto currTransNode = l3TransNodes.front();

        // Number of repeated transfers for this tensor. Each tensor needs at
        // least one transfer. The tensor may be repeatedly transferred when
        // its parent loops have unrelated dimensions.
        const auto parentLoops = getParentLoopNodes(*currTransNode, dsc);
        long numRepeats = 1;
        for (const auto loopNode : parentLoops) {
          for (const auto &dimKind : loopNode->dims_) {
            const auto dim = dimKind.dim_;
            const bool isLdsUnrelatedDim = !ldsNonBroadcastDimsSet.count(dim);
            if (isLdsUnrelatedDim) {
              const long tripCount =
                  getTripCount(dsc, dim, loopNode->numId_, loopNode->denId_);
              numRepeats *= tripCount;
            }
          }
        }

        unsigned numWkSlices =
            getLabeledDsNumOfWkSlices(mySDsc, ldsIdx, {dscIdx});
        unsigned shares =
            getLabeledDsWkSliceMulticastDegree(mySDsc, ldsIdx, {dscIdx});

        // Record the number of requests.
        if (numRequestsOfMaxBurst)
          addToMap(maxBurstSize, shares,
                   numRequestsOfMaxBurst * numStickVolumes * numRepeats *
                       numWkSlices);
        if (otherBurstSize)
          addToMap(otherBurstSize, shares,
                   numStickVolumes * numRepeats * numWkSlices);
      }
    } else {
      // The coreD of this labeledDs in each DSC must be the same. The same
      // work slice can be used by multiple DSCs. We use dscMain to compute
      // the stick volume and the number of requests to transfer the stick
      // volume.
      unsigned long stickVolume = getLabeledDsChunkStickVolume(dscMain, ldsIdx);
      const unsigned long numRequestsOfMaxBurst = stickVolume / maxBurstSize;
      const unsigned otherBurstSize = stickVolume % maxBurstSize;

      // Number of stick volumes for the coreD amount of work.
      const unsigned long numStickVolumes = getLabeledDsNumOfStickVolumesInCore(
          dscMain, ldsIdx, stickVolume, primaryDims);

      // Number of repeated transfers for this tensor. Each tensor needs at
      // least one transfer. The tensor may be repeatedly transferred when its
      // parent loops have unrelated dimensions.
      std::map<int, std::map<PrimaryDimTypes, long>> dscToParentDimsTripCount;
      for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
        const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
        const auto &lds = dsc.labeledDs_.at(ldsIdx);
        const auto ldsNonBroadcastDimsSet =
            dsc.getNonBroadcastLdsDimSet(ldsIdx);
        DT_CHECK(dsc.dataStageParam_.count(dataStageCoreIdx) &&
                 dsc.dataStageParam_.count(dataStageChunkIdx) &&
                 "Expect valid dataStageParam_ entries.");
        dscToParentDimsTripCount.emplace(dscIdx,
                                         std::map<PrimaryDimTypes, long>());
        // Get the tensor's L3 transfer nodes. If there are multiple nodes, they
        // should be on the same loop level. Hence, use any one of them.
        // TODO: We should consider all transfer nodes in the total amount of
        // data transferred.
        const auto l3TransNodes = getLdsL3TransferNodes(
            mySDsc, dscIdx, ldsIdx,
            {SenComponents::HBM, SenComponents::NO_COMPONENT},
            {SenComponents::LX});
        DT_CHECK_MSG(!l3TransNodes.empty(), "Expect valid transfer nodes.");
        const auto currTransNode = l3TransNodes.front();

        const auto parentLoops = getParentLoopNodes(*currTransNode, dsc);
        for (const auto loopNode : parentLoops) {
          for (const auto &dimKind : loopNode->dims_) {
            const auto dim = dimKind.dim_;
            const bool isLdsUnrelatedDim = !ldsNonBroadcastDimsSet.count(dim);
            if (isLdsUnrelatedDim) {
              const long tripCount =
                  getTripCount(dsc, dim, loopNode->numId_, loopNode->denId_);
              dscToParentDimsTripCount[dscIdx][dim] = tripCount;
            }
          }
        }
      }
      DT_CHECK_MSG(dscToParentDimsTripCount.size() == mySDsc.dscs_.size(),
                   "Number of DSCs does not match.");

      std::vector<long> dscNumRepeats(mySDsc.dscs_.size());
      for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
        const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
        long numRepeats = 1;
        for (const auto &entry : dscToParentDimsTripCount.at(dscIdx))
          numRepeats *= entry.second;

        dscNumRepeats[dscIdx] = numRepeats;
      }

      if (dscNumRepeats.size() == 1 || dscNumRepeats[0] == dscNumRepeats[1]) {
        const long numRepeats = dscNumRepeats[0];
        std::vector<int> dscIndices;
        for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx)
          dscIndices.push_back(dscIdx);
        unsigned numWkSlices =
            getLabeledDsNumOfWkSlices(mySDsc, ldsIdx, dscIndices);
        unsigned shares =
            getLabeledDsWkSliceMulticastDegree(mySDsc, ldsIdx, dscIndices);

        // Record the number of requests.
        if (numRequestsOfMaxBurst)
          addToMap(maxBurstSize, shares,
                   numRequestsOfMaxBurst * numStickVolumes * numRepeats *
                       numWkSlices);
        if (otherBurstSize)
          addToMap(otherBurstSize, shares,
                   numStickVolumes * numRepeats * numWkSlices);
      } else {
        DT_CHECK_MSG(dscNumRepeats.size() == 2,
                     "Currently only support at most two DSCs.");
        const int lowCountDscIdx = dscNumRepeats[0] < dscNumRepeats[1] ? 0 : 1;
        const int highCountDscIdx = dscNumRepeats[0] < dscNumRepeats[1] ? 1 : 0;

        // Both DSCs have the lower number of repeats. Hence, the multicast is
        // across the two DSCs.
        {
          const long numRepeats = std::min(dscNumRepeats[0], dscNumRepeats[1]);
          unsigned numWkSlices = getLabeledDsNumOfWkSlices(
              mySDsc, ldsIdx, {lowCountDscIdx, highCountDscIdx});
          unsigned shares = getLabeledDsWkSliceMulticastDegree(
              mySDsc, ldsIdx, {lowCountDscIdx, highCountDscIdx});

          // Record the number of requests.
          if (numRequestsOfMaxBurst)
            addToMap(maxBurstSize, shares,
                     numRequestsOfMaxBurst * numStickVolumes * numRepeats *
                         numWkSlices);
          if (otherBurstSize)
            addToMap(otherBurstSize, shares,
                     numStickVolumes * numRepeats * numWkSlices);
        }

        // Only the DSC with higher repeat number has additional repeats. The
        // multicast is only on this DSC accordingly.
        {
          const long numRepeats = std::max(dscNumRepeats[0], dscNumRepeats[1]) -
                                  std::min(dscNumRepeats[0], dscNumRepeats[1]);
          unsigned numWkSlices =
              getLabeledDsNumOfWkSlices(mySDsc, ldsIdx, {highCountDscIdx});
          unsigned shares = getLabeledDsWkSliceMulticastDegree(
              mySDsc, ldsIdx, {highCountDscIdx});

          // Record the number of requests.
          if (numRequestsOfMaxBurst)
            addToMap(maxBurstSize, shares,
                     numRequestsOfMaxBurst * numStickVolumes * numRepeats *
                         numWkSlices);
          if (otherBurstSize)
            addToMap(otherBurstSize, shares,
                     numStickVolumes * numRepeats * numWkSlices);
        }
      }
    }
  }

  // Compute the average efficiency per request.
  double efficiency = 0.0;
  unsigned long totalNumRequests = 0;
  for (const auto &burstSizeEntry : burstSizeAndMulticastToNumReqMap) {
    const unsigned burstSize = burstSizeEntry.first;
    for (const auto &multicastEntry : burstSizeEntry.second) {
      const unsigned multicastDegree = multicastEntry.first;
      const unsigned long numRequests = multicastEntry.second;
      efficiency +=
          getBurstEfficiency(burstSize, multicastDegree) * numRequests;
      totalNumRequests += numRequests;
    }
  }

  return efficiency / totalNumRequests;
}

// ------------------------------------------------------------------------------------------------
// entry 286/382   level 2   scc 70   230 body lines
// unit: e286_calculateFlopPerByte
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2253
// original: double L3DlOpsScheduler::calculateFlopPerByte( const SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &primaryDims)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
double e286_calculateFlopPerByte(
    const SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &primaryDims)
{
  // Compute the total flops by the steps below:
  //  1) For each DSC, compute the flops of a chunk per work slice, and
  //  multiply it by the number of work slices in the DSC (which should
  //  effectively be equal to the number of cores in the DSC) to have the
  //  total flops of a chunk in the DSC. 2) Compute the total flops for all
  //  DSCs by summing up the flops in 1).
  long long totalFlops = 0;
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
    DT_CHECK(dsc.dataStageParam_.count(dataStageCoreIdx) &&
             dsc.dataStageParam_.count(dataStageChunkIdx) &&
             "Expect valid dataStageParam_ entries.");

    // For padded dimensions, the pad type should be the same in any allocate
    // node for LX.
    PaddingFormType padding;
    for (const auto &lds : dsc.labeledDs_) {
      if (lds.memOrg_.count(SenComponents::LX)) {
        const auto *allocNode = lds.memOrg_.at(SenComponents::LX).allocateNode_;
        DT_CHECK_MSG(allocNode, "Expect a valid LX allocate node.");
        padding = allocNode->padding_;
        break;
      }
    }

    long long flops = 1;
    for (PrimaryDimTypes dim : primaryDims) {
      const long param =
          dsc.dataStageParam_.at(dataStageChunkIdx)
              .ss_.primaryDimToVal_st(dim, SenComponents::NO_COMPONENT, -1, -1,
                                      padding);
      if (isValidDimParam(param)) flops *= param;
    }

    // Each element is computed by MAC, which has two operations–a
    // multiplication and an add.
    flops *= 2;

    // Flops for all work slices.
    flops *= dsc.numCoresUsed_;

    // Total flops.
    totalFlops += flops;
  }
  DT_CHECK_MSG(totalFlops > 0, "Invalid total flops.");

  // Compute the total bytes transferred by the steps below:
  //  1) For each labeledDs that requires HBM transfer:
  //    1.1) Compute the chunk size, the number of repeated transfers and the
  //    multicast degree of the transfers to have the total bytes for this
  //    labeledDs per work slice.
  //    1.2) Multiply the bytes in 1.1) by the number of work slices to have
  //    the total bytes for this labeledDs.
  //  2) Compute the total bytes for all such labeledDs by summing up the
  //  bytes in 1).
  long long totalBytes = 0;
  const int dscMainIdx = 0;
  const DesignSpaceConfig &dscMain = mySDsc.dscs_.at(dscMainIdx);
  std::unordered_set<PrimaryDimTypes> coreSplitDims =
      getCoreSplitDimensions(mySDsc);
  DT_CHECK_MSG(
      coreSplitDims.size() <= 1 && mySDsc.dscs_.size() <= 2,
      "Currently support at most one core split dimension with two DSCs.");
  for (int ldsIdx = 0; ldsIdx < dscMain.labeledDs_.size(); ++ldsIdx) {
    const LabeledDsInfo &lds = dscMain.labeledDs_.at(ldsIdx);
    DT_CHECK_MSG(!isLabeledDsLXNeighbor(mySDsc, dscMainIdx, lds),
                 "Do not expect input neighbor fetch.");

    // Skip if the labeledDs does not require HBM transfer.
    if (!lds.isHbmPinned()) continue;

    // Compute the number of bytes in a chunk.
    // Use dscMain to check if this labeledDs has core split dimensions.
    bool ldsHasCoreSplitDim = false;
    for (PrimaryDimTypes dim : dscMain.getLayoutDims(ldsIdx)) {
      ldsHasCoreSplitDim = coreSplitDims.count(dim);
      if (ldsHasCoreSplitDim) break;
    }

    DT_CHECK_MSG(lds.memOrg_.count(SenComponents::LX),
                 "Expect LX in labeledDs memOrg_.");
    const auto *allocNode = lds.memOrg_.at(SenComponents::LX).allocateNode_;
    DT_CHECK_MSG(allocNode, "Expect a valid allocate node.");
    DT_CHECK_MSG(allocNode->component_ == SenComponents::LX,
                 "Expect an LX allocation.");
    long long ldsBytes = 0;
    if (ldsHasCoreSplitDim) {
      // The chunk size may be different in each DSC. We compute for each DSC
      // separately.
      for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
        const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
        const auto &currLds = dsc.labeledDs_.at(ldsIdx);
        const auto ldsNonBroadcastDimsSet =
            dsc.getNonBroadcastLdsDimSet(ldsIdx);
        DT_CHECK(dsc.dataStageParam_.count(dataStageCoreIdx) &&
                 dsc.dataStageParam_.count(dataStageChunkIdx) &&
                 "Expect valid dataStageParam_ entries.");
        int64_t bytes = dsc.getBufferCapacityForNode(
            allocNode, ldsIdx, allocNode->component_, -1, -1,
            dscGlobal.sysDef.bytesPerStick);

        // Get the tensor's L3 transfer nodes. If there are multiple nodes, they
        // should be on the same loop level. Hence, use any one of them.
        // TODO: We should consider all transfer nodes in the total amount of
        // data transferred.
        const auto l3TransNodes = getLdsL3TransferNodes(
            mySDsc, dscIdx, ldsIdx, {SenComponents::HBM}, {SenComponents::LX});
        DT_CHECK_MSG(!l3TransNodes.empty(), "Expect valid transfer nodes.");
        const auto currTransNode = l3TransNodes.front();
        const auto parentLoops = getParentLoopNodes(*currTransNode, dsc);

        // Number of repeated transfers for this tensor. Each tensor needs at
        // least one transfer. The tensor may be repeatedly transferred when
        // its parent loops have unrelated dimensions.
        long numRepeats = 1;
        for (const auto loopNode : parentLoops) {
          for (const auto &dimKind : loopNode->dims_) {
            const auto dim = dimKind.dim_;
            const bool isLdsUnrelatedDim = !ldsNonBroadcastDimsSet.count(dim);
            if (isLdsUnrelatedDim) {
              const long tripCount =
                  getTripCount(dsc, dim, loopNode->numId_, loopNode->denId_);
              numRepeats *= tripCount;
            }
          }
        }
        unsigned numWkSlices =
            getLabeledDsNumOfWkSlices(mySDsc, ldsIdx, {dscIdx});

        ldsBytes += bytes * numRepeats * numWkSlices;
      }
    } else {
      // The chunk size must be the same for all DSCs. We use dscMain to compute
      // the chunk size.
      int64_t bytes = dscMain.getBufferCapacityForNode(
          allocNode, ldsIdx, allocNode->component_, -1, -1,
          dscGlobal.sysDef.bytesPerStick);

      // Number of repeated transfers for this tensor. Each tensor needs at
      // least one transfer. The tensor may be repeatedly transferred when its
      // parent loops have unrelated dimensions.
      std::map<int, std::map<PrimaryDimTypes, long>> dscToParentDimsTripCount;
      for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
        const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
        const auto &currLds = dsc.labeledDs_.at(ldsIdx);
        const auto ldsNonBroadcastDimsSet =
            dsc.getNonBroadcastLdsDimSet(ldsIdx);
        DT_CHECK(dsc.dataStageParam_.count(dataStageCoreIdx) &&
                 dsc.dataStageParam_.count(dataStageChunkIdx) &&
                 "Expect valid dataStageParam_ entries.");
        dscToParentDimsTripCount.emplace(dscIdx,
                                         std::map<PrimaryDimTypes, long>());
        // Get the tensor's L3 transfer nodes. If there are multiple nodes, they
        // should be on the same loop level. Hence, use any one of them.
        // TODO: We should consider all transfer nodes in the total amount of
        // data transferred.
        const auto l3TransNodes = getLdsL3TransferNodes(
            mySDsc, dscIdx, ldsIdx, {SenComponents::HBM}, {SenComponents::LX});
        const auto currTransNode = l3TransNodes.front();
        const auto parentLoops = getParentLoopNodes(*currTransNode, dsc);
        for (const auto loopNode : parentLoops) {
          for (const auto &dimKind : loopNode->dims_) {
            const auto dim = dimKind.dim_;
            const bool isLdsUnrelatedDim = !ldsNonBroadcastDimsSet.count(dim);
            if (isLdsUnrelatedDim) {
              const long tripCount =
                  getTripCount(dsc, dim, loopNode->numId_, loopNode->denId_);
              dscToParentDimsTripCount[dscIdx][dim] = tripCount;
            }
          }
        }
      }
      DT_CHECK_MSG(dscToParentDimsTripCount.size() == mySDsc.dscs_.size(),
                   "Number of DSCs does not match.");

      std::vector<long> dscNumRepeats(mySDsc.dscs_.size());
      for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
        const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
        long numRepeats = 1;
        for (const auto &entry : dscToParentDimsTripCount.at(dscIdx))
          numRepeats *= entry.second;

        dscNumRepeats[dscIdx] = numRepeats;
      }

      if (dscNumRepeats.size() == 1 || dscNumRepeats[0] == dscNumRepeats[1]) {
        const long numRepeats = dscNumRepeats[0];
        std::vector<int> dscIndices;
        for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx)
          dscIndices.push_back(dscIdx);
        unsigned numWkSlices =
            getLabeledDsNumOfWkSlices(mySDsc, ldsIdx, dscIndices);

        ldsBytes += bytes * numRepeats * numWkSlices;
      } else {
        DT_CHECK_MSG(dscNumRepeats.size() == 2,
                     "Currently only support at most two DSCs.");
        const int lowCountDscIdx = dscNumRepeats[0] < dscNumRepeats[1] ? 0 : 1;
        const int highCountDscIdx = dscNumRepeats[0] < dscNumRepeats[1] ? 1 : 0;

        // Both DSCs have the lower number of repeats. Hence, the multicast is
        // across the two DSCs.
        {
          const long numRepeats = std::min(dscNumRepeats[0], dscNumRepeats[1]);
          unsigned numWkSlices = getLabeledDsNumOfWkSlices(
              mySDsc, ldsIdx, {lowCountDscIdx, highCountDscIdx});

          ldsBytes += bytes * numRepeats * numWkSlices;
        }

        // Only the DSC with higher repeat number has additional repeats. The
        // multicast is only on this DSC accordingly.
        {
          const long numRepeats = std::max(dscNumRepeats[0], dscNumRepeats[1]) -
                                  std::min(dscNumRepeats[0], dscNumRepeats[1]);
          unsigned numWkSlices =
              getLabeledDsNumOfWkSlices(mySDsc, ldsIdx, {highCountDscIdx});

          ldsBytes += bytes * numRepeats * numWkSlices;
        }
      }
    }

    totalBytes += ldsBytes;
  }
  DT_CHECK_MSG(totalBytes > 0, "Invalid total bytes.");

  return (double)totalFlops / totalBytes;
}

// ------------------------------------------------------------------------------------------------
// entry 287/382   level 2   scc 81   27 body lines
// unit: e287_getCrossCoreReductionGroupInfo
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2743
// original: std::vector<CrossCoreReductionGroup> L3DlOpsScheduler::getCrossCoreReductionGroupInfo( const SuperDsc &mySDsc, const DesignSpaceConfig &dsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::vector<CrossCoreReductionGroup>
e287_getCrossCoreReductionGroupInfo(
    const SuperDsc &mySDsc, const DesignSpaceConfig &dsc) const
{
  DT_CHECK_MSG(isOpCrossCoreReduction(mySDsc, dsc),
               "Expect cross-core reduction dataflow.");
  // Collect the reduced dimensions.
  const auto reducedDimSet = getOpReducedDimSet(mySDsc, dsc);
  int numGroups = 1;
  for (const auto& [dim, numSlices] : mySDsc.numWkSlicesPerDim_)
    if (!reducedDimSet.count(dim)) numGroups *= numSlices;

  // Put cores in groups based on reduce dimensions. Use slice ids to order them
  std::vector<CrossCoreReductionGroup> groupInfo(numGroups);
  for (const auto& [coreId, wkSlice] : mySDsc.coreIdToWkSlice_) {
    int group = 0, groupCardinality = 1, reduceSlice = 0, reduceCardinality = 1;
    for (const auto& [dim, dimSlice] : wkSlice) {
      if (reducedDimSet.count(dim)) {
        reduceSlice += dimSlice * reduceCardinality;
        reduceCardinality *= mySDsc.numWkSlicesPerDim_.at(dim);
      } else {
        group += dimSlice * groupCardinality;
        groupCardinality *= mySDsc.numWkSlicesPerDim_.at(dim);
      }
    }
    groupInfo.at(group).addCore(coreId, reduceSlice);
  }

  return groupInfo;
}

// ------------------------------------------------------------------------------------------------
// entry 288/382   level 2   scc 91   423 body lines
// unit: e288_createSynchronizationDSC
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3487
// original: void L3DlOpsScheduler::createSynchronizationDSC(SuperDsc& mySDsc, const int dscIdx)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e288_createSynchronizationDSC(SuperDsc& mySDsc,
                                                const int dscIdx)
{
  auto& dsc = mySDsc.dscs_.at(dscIdx);

  // Check tensor residencies in the DSC.
  std::unordered_set<int> allLdsIndexSet = getAllLabeledDsIndicesSet(dsc);
  std::unordered_set<int> hbmPinnedLdsIndexSet =
      getHbmPinnedLabeledDsIndicesSet(dsc);
  std::unordered_set<int> lxNeighborLdsIndexSet =
      getLxNeighborLabeledDsIndicesSet(mySDsc, dsc, dscIdx);
  const bool hasHbmPinnedTensor = !hbmPinnedLdsIndexSet.empty();
  const bool hasLxNeighborTensor = !lxNeighborLdsIndexSet.empty();
  const bool hasAllLxPinnedTensors =
      !hasHbmPinnedTensor && !hasLxNeighborTensor;
  DT_CHECK_MSG(!(hasHbmPinnedTensor && hasLxNeighborTensor),
               "Do not support both HBM pinned tensor and input-neighbor "
               "fetch tensor existing "
               "in the same DSC.");

  bool syncL3LUAndLXLUInserted = false;
  // Start to add sync nodes.
  if (hasLxNeighborTensor) {
    DT_CHECK_MSG(lxNeighborLdsIndexSet.size() == 1,
                 "Currently support only one LX input-neighbor fetch tensor.");
    const auto ldsIdx = *lxNeighborLdsIndexSet.begin();
    const auto& lds = dsc.labeledDs_.at(ldsIdx);
    DT_CHECK_MSG(lds.memOrg_.count(SenComponents::LX), "Expect LX in memOrg_.");

    // Find the tensor's NO_COMPONENT to LX transfer node.
    auto allocNode = lds.memOrg_.at(SenComponents::LX).allocateNode_;
    DT_CHECK_MSG(allocNode, "Expect a valid allocate node.");
    dsc2::TransferNode* transNode = nullptr;
    for (auto& [node, count] : allocNode->allocUsers_) {
      if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
        auto currTransNode = static_cast<const dsc2::TransferNode*>(node);
        const bool isNoCompToLxTransfer =
            currTransNode->src_.storage_ == NO_COMPONENT &&
            (!currTransNode->dstVias_.empty() &&
             currTransNode->dstVias_.front().loc_.storage_ ==
                 SenComponents::LX);
        if (isNoCompToLxTransfer) {
          transNode = const_cast<dsc2::TransferNode*>(currTransNode);
        }
      }
    }
    DT_CHECK_MSG(transNode, "Expect a valid transfer node.");

    // Create a pair of L3LU->LXLU soft sync nodes and insert them after the
    // transfer node.
    addL3LUAndLXLUSoftSyncNodeSequence(transNode);
  } else if (hasHbmPinnedTensor) {
    auto getHbmToLxTransNodes = [&hbmPinnedLdsIndexSet, &dsc]() {
      std::vector<dsc2::TransferNode*> hbmToLxTransNodes;
      for (auto ldsIdx : hbmPinnedLdsIndexSet) {
        const auto& lds = dsc.labeledDs_.at(ldsIdx);
        DT_CHECK_MSG(lds.memOrg_.count(SenComponents::HBM),
                     "Expect HBM in memOrg_.");
        auto allocNode = lds.memOrg_.at(SenComponents::HBM).allocateNode_;
        DT_CHECK_MSG(allocNode, "Expect a valid HBM allocate node.");
        for (auto& [userNode, count] : allocNode->allocUsers_) {
          if (userNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
            auto transNode = static_cast<const dsc2::TransferNode*>(userNode);
            const bool isHbmToLxTransfer =
                transNode->src_.storage_ == SenComponents::HBM &&
                (!transNode->dstVias_.empty() &&
                 transNode->dstVias_.front().loc_.storage_ ==
                     SenComponents::LX);
            if (isHbmToLxTransfer)
              hbmToLxTransNodes.push_back(
                  const_cast<dsc2::TransferNode*>(transNode));
          }
        }
      }
      return hbmToLxTransNodes;
    };

    // Insert L3LU<->LXLU sync nodes if there are HBM->LX transfers.
    const auto hbmToLxTransNodes = getHbmToLxTransNodes();
    const bool hasHbmToLxTransfer = !hbmToLxTransNodes.empty();
    if (hasHbmToLxTransfer) {
      auto getInnerHbmToLxTransNodeSet = [&hbmToLxTransNodes, &dsc]() {
        DT_CHECK_MSG(!hbmToLxTransNodes.empty(),
                     "Expect HBM to LX transfer nodes.");
        std::map<int, std::unordered_set<dsc2::ScheduleNode*>>
            numParentLoopsToTransNodes;
        for (auto transNode : hbmToLxTransNodes) {
          int numParentLoops = 0;
          const dsc2::ScheduleNode* currNode = transNode;
          while (currNode && currNode->getOwnerLoop()) {
            ++numParentLoops;
            currNode = currNode->getOwnerLoop();
          }
          numParentLoopsToTransNodes[numParentLoops].insert(transNode);
        }

        auto innerHbmToLxTransNodes =
            std::prev(numParentLoopsToTransNodes.end())->second;
        return innerHbmToLxTransNodes;
      };

      // Add L3LU<->LXLU sync nodes after the innermost HBM->LX transfer.
      auto innerHbmToLxTransNodeSet = getInnerHbmToLxTransNodeSet();
      constexpr bool insertL3LUAndLXLUSyncNodeBefore = false;
      auto insertAfterNode = getInsertionNode(innerHbmToLxTransNodeSet,
                                              insertL3LUAndLXLUSyncNodeBefore);
      if (lxBufferType == BufferType::SPATIAL_DOUBLE) {
        auto insertAfterOwnerLoopNode = insertAfterNode->getMutableOwnerLoop();
        DT_CHECK_MSG(insertAfterOwnerLoopNode, "Expect a valid loop node.");
        if (insertAfterOwnerLoopNode == dsc.scheduleTree_.getHead() ||
            (insertAfterOwnerLoopNode->numId_ == dataStageCoreIdx &&
             insertAfterOwnerLoopNode->denId_ == dataStageSuperChunkIdx))
          // Add hard sync nodes only after the transfer when the innermost
          // HBM->LX transfer is inside a core/SuperChunk loop.
          addL3LUAndLXLUSyncNodeSequence(insertAfterNode);
        else {
          DT_CHECK_MSG(
              insertAfterOwnerLoopNode->numId_ == dataStageSuperChunkIdx &&
                  insertAfterOwnerLoopNode->denId_ == dataStageChunkIdx,
              "Expect a SuperChunk-by-chunk loop.");
          // Add soft sync nodes after the transfer.
          addL3LUAndLXLUSoftSyncNodeSequence(insertAfterNode);
          // Add hard sync nodes at the end of the innermost
          // Core-by-SuperChunk loop.
          auto* lxBelowBlockNode = getLxBelowBlockNode(dsc.scheduleTree_);
          DT_CHECK_MSG(lxBelowBlockNode, "Expect a valid lx_below block node.");
          dsc2::LoopNode* outermostSuperChunkByChunkLoop =
              lxBelowBlockNode->getMutableOwnerLoop();
          DT_CHECK_MSG(
              outermostSuperChunkByChunkLoop &&
                  outermostSuperChunkByChunkLoop->numId_ ==
                      dataStageSuperChunkIdx &&
                  outermostSuperChunkByChunkLoop->denId_ == dataStageChunkIdx,
              "Unexpected SuperChunk-by-chunk loop.");
          while (
              !(outermostSuperChunkByChunkLoop->getMutableOwnerLoop()->numId_ ==
                    dataStageCoreIdx &&
                outermostSuperChunkByChunkLoop->getMutableOwnerLoop()->denId_ ==
                    dataStageSuperChunkIdx)) {
            outermostSuperChunkByChunkLoop =
                outermostSuperChunkByChunkLoop->getMutableOwnerLoop();
          }
          dsc2::LoopNode* innermostCoreBySuperChunkLoop =
              outermostSuperChunkByChunkLoop->getMutableOwnerLoop();
          addL3LUAndLXLUSyncNodeSequence(
              innermostCoreBySuperChunkLoop->getNextView(SenComponents::ALL)
                  .back());
        }
      } else
        addL3LUAndLXLUSyncNodeSequence(insertAfterNode);

      syncL3LUAndLXLUInserted = true;
    }

    // Check if the output tensor has HBM->LX or LX->HBM transfers.
    dsc2::TransferNode* lxToHbmOutputLdsTransNode = nullptr;
    dsc2::TransferNode* hbmToLxOutputLdsTransNode = nullptr;
    for (const auto ldsIdx : hbmPinnedLdsIndexSet) {
      if (isOutputLabeledDs(ldsIdx, dsc)) {
        const auto& lds = dsc.labeledDs_.at(ldsIdx);
        DT_CHECK_MSG(lds.memOrg_.count(SenComponents::HBM),
                     "Expect HBM in memOrg_.");
        const auto allocNode = lds.memOrg_.at(SenComponents::HBM).allocateNode_;
        DT_CHECK_MSG(allocNode, "Expect a valid allocate node.");

        for (auto& [userNode, count] : allocNode->allocUsers_) {
          if (userNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
            auto transNode = static_cast<const dsc2::TransferNode*>(userNode);
            const bool isLxToHbmTransfer =
                transNode->src_.storage_ == SenComponents::LX &&
                (!transNode->dstVias_.empty() &&
                 transNode->dstVias_.front().loc_.storage_ ==
                     SenComponents::HBM);
            const bool isHbmToLxTransfer =
                transNode->src_.storage_ == SenComponents::HBM &&
                (!transNode->dstVias_.empty() &&
                 transNode->dstVias_.front().loc_.storage_ ==
                     SenComponents::LX);
            if (isLxToHbmTransfer) {
              DT_CHECK_MSG(lxToHbmOutputLdsTransNode == nullptr,
                           "Expect only one LX to HBM transfer node.");
              lxToHbmOutputLdsTransNode =
                  const_cast<dsc2::TransferNode*>(transNode);
            }
            if (isHbmToLxTransfer) {
              DT_CHECK_MSG(hbmToLxOutputLdsTransNode == nullptr,
                           "Expect only one HBM to LX transfer node.");
              hbmToLxOutputLdsTransNode =
                  const_cast<dsc2::TransferNode*>(transNode);
            }
          }
        }
        // Break because we expect only one output tensor.
        break;
      }
    }

    // Insert L3SU<->LXSU sync nodes if we have LX->HBM transfers.
    const bool hasLxToHbmTransfer = (lxToHbmOutputLdsTransNode != nullptr);
    if (hasLxToHbmTransfer) {
      auto parentLxToHbmTransNode =
          lxToHbmOutputLdsTransNode->getMutableParent();

      // Add one pair of LXSU->L3SU sync nodes before the LX->HBM transfer
      // node.
      {
        dsc2::SyncNode* lxsuToL3suSend = createSyncNode(
            {SenComponents::LXSU},
            "sync_send_" +
                EnumsConversion::senComponentsToString.at(SenComponents::LXSU) +
                "_to_" +
                EnumsConversion::senComponentsToString.at(SenComponents::L3SU));
        dsc2::SyncNode* lxsuToL3suReceive = createSyncNode(
            {SenComponents::L3SU},
            "sync_receive_" +
                EnumsConversion::senComponentsToString.at(SenComponents::L3SU) +
                "_from_" +
                EnumsConversion::senComponentsToString.at(SenComponents::LXSU),
            true);

        lxsuToL3suSend->otherEndOfTheSignals_.push_back(lxsuToL3suReceive);
        lxsuToL3suReceive->otherEndOfTheSignals_.push_back(lxsuToL3suSend);
        parentLxToHbmTransNode->addChildNode(lxsuToL3suSend, /*addBefore*/ true,
                                             lxToHbmOutputLdsTransNode);
        parentLxToHbmTransNode->addChildNode(
            lxsuToL3suReceive, /*addBefore*/ true, lxToHbmOutputLdsTransNode);
      }

      // Add one pair of L3SU->LXSU sync nodes at a location depending on the
      // buffering type.
      {
        dsc2::SyncNode* l3suToLxsuSend = createSyncNode(
            {SenComponents::L3SU},
            "sync_send_" +
                EnumsConversion::senComponentsToString.at(SenComponents::L3SU) +
                "_to_" +
                EnumsConversion::senComponentsToString.at(SenComponents::LXSU));
        dsc2::SyncNode* l3suToLxsuReceive = createSyncNode(
            {SenComponents::LXSU},
            "sync_receive_" +
                EnumsConversion::senComponentsToString.at(SenComponents::LXSU) +
                "_from_" +
                EnumsConversion::senComponentsToString.at(SenComponents::L3SU),
            true);

        l3suToLxsuSend->otherEndOfTheSignals_.push_back(l3suToLxsuReceive);
        l3suToLxsuReceive->otherEndOfTheSignals_.push_back(l3suToLxsuSend);

        if (lxBufferType == BufferType::SPATIAL_DOUBLE) {
          // Add after the allocate node.
          const auto ldsIdx =
              lxToHbmOutputLdsTransNode->srcLdsAndLoopOffsets_.myLdsIdx_;
          const auto outputLdsAllocLxNode = dsc.labeledDs_.at(ldsIdx)
                                                .memOrg_.at(SenComponents::LX)
                                                .allocateNode_;
          DT_CHECK_MSG(outputLdsAllocLxNode, "Expect allocate node.");
          auto parentOutputLdsAllocLxNode =
              outputLdsAllocLxNode->getMutableParent();
          parentOutputLdsAllocLxNode->addChildNode(
              l3suToLxsuSend, /*addBefore*/ false, outputLdsAllocLxNode);
          parentOutputLdsAllocLxNode->addChildNode(l3suToLxsuReceive,
                                                   /*addBefore*/ false,
                                                   l3suToLxsuSend);

        } else {
          // Add before the LX->HBM transfer node.
          parentLxToHbmTransNode->addChildNode(
              l3suToLxsuSend, /*addBefore*/ true, lxToHbmOutputLdsTransNode);
          parentLxToHbmTransNode->addChildNode(l3suToLxsuReceive,
                                               /*addBefore*/ true,
                                               lxToHbmOutputLdsTransNode);
        }
      }

      // Insert L3SU->L3LU sync nodes if the output tensor also has HBM->LX
      // transfer.
      bool hasOutputLdsHbmToLxTransfer = (hbmToLxOutputLdsTransNode != nullptr);
      if (hasOutputLdsHbmToLxTransfer) {
        // Add one pair of L3SU->L3LU sync nodes at allocation loop level.
        {
          dsc2::SyncNode* l3suToL3luSend =
              createSyncNode({SenComponents::L3SU},
                             "sync_send_" +
                                 EnumsConversion::senComponentsToString.at(
                                     SenComponents::L3SU) +
                                 "_to_" +
                                 EnumsConversion::senComponentsToString.at(
                                     SenComponents::L3LU));
          dsc2::SyncNode* l3suToL3luReceive =
              createSyncNode({SenComponents::L3LU},
                             "sync_receive_" +
                                 EnumsConversion::senComponentsToString.at(
                                     SenComponents::L3LU) +
                                 "_from_" +
                                 EnumsConversion::senComponentsToString.at(
                                     SenComponents::L3SU),
                             true);

          l3suToL3luSend->otherEndOfTheSignals_.push_back(l3suToL3luReceive);
          l3suToL3luReceive->otherEndOfTheSignals_.push_back(l3suToL3luSend);

          const auto ldsIdx =
              hbmToLxOutputLdsTransNode->srcLdsAndLoopOffsets_.myLdsIdx_;
          const auto outputLdsAllocLxNode = dsc.labeledDs_.at(ldsIdx)
                                                .memOrg_.at(SenComponents::LX)
                                                .allocateNode_;
          DT_CHECK_MSG(outputLdsAllocLxNode, "Expect allocate node.");
          auto parentOutputLdsAllocLxNode =
              outputLdsAllocLxNode->getMutableParent();
          parentOutputLdsAllocLxNode->addChildNode(l3suToL3luReceive,
                                                   /*addBefore*/ true,
                                                   outputLdsAllocLxNode);
          // Add the L3SU send node at the end of the allocation owner loop
          // (which can be the schedule tree root node).
          auto allocParentLoop = outputLdsAllocLxNode->getMutableOwnerLoop();
          allocParentLoop->addChildNode(l3suToL3luSend,
                                        /*addBefore*/ false);
        }

        // Add two pairs of L3SU->L3LU sync nodes at the root level.
        for (int i = 0; i < 2; ++i) {
          dsc2::SyncNode* l3suToL3luSend =
              createSyncNode({SenComponents::L3SU},
                             "sync_send_" +
                                 EnumsConversion::senComponentsToString.at(
                                     SenComponents::L3SU) +
                                 "_to_" +
                                 EnumsConversion::senComponentsToString.at(
                                     SenComponents::L3LU) +
                                 "_outermost_" + std::to_string(i));
          dsc2::SyncNode* l3suToL3luReceive =
              createSyncNode({SenComponents::L3LU},
                             "sync_receive_" +
                                 EnumsConversion::senComponentsToString.at(
                                     SenComponents::L3LU) +
                                 "_from_" +
                                 EnumsConversion::senComponentsToString.at(
                                     SenComponents::L3SU) +
                                 "_outermost_" + std::to_string(i),
                             true);

          l3suToL3luSend->otherEndOfTheSignals_.push_back(l3suToL3luReceive);
          l3suToL3luReceive->otherEndOfTheSignals_.push_back(l3suToL3luSend);

          auto schedRootNode = dsc.scheduleTree_.getHeadMutable();
          schedRootNode->addChildNode(l3suToL3luSend, /*addBefore*/ true);
          schedRootNode->addChildNode(l3suToL3luReceive,
                                      /*addBefore*/ false);
        }
      }
    }
/*   } else {
    // All tensors are LX pinned.
    for (const auto ldsIdx : allLdsIndexSet) {
      const auto& lds = dsc.labeledDs_.at(ldsIdx);
      DT_CHECK_MSG(lds.isLxPinned(), "Expect LX pinned tensor.");
    } */
  }

  // L3 padding requires L3LU/LXLU sync nodes if they haven't been inserted.
  if (!syncL3LUAndLXLUInserted) {
    // Check if any tensor has L3 padding. The L3 padding applies to
    // operations with a strided window, including CONV, MaxPool, AveragePool
    // and depth-wise operations.
    std::vector<int> l3PaddingLdsIndices;
    for (auto& ldsIdx : allLdsIndexSet) {
      LabeledDsInfo& lds = dsc.labeledDs_.at(ldsIdx);
      if (lds.memOrg_.count(SenComponents::LX) &&
          lds.memOrg_.at(SenComponents::LX).isZeroPadded != ZpType::NOZEROPAD) {
        bool hasDimZeroPad = false;
        for (auto& dim : dsc.primaryDsInfo_.at(lds.dsType_).layoutDimOrder_) {
          if (dsc.N_.paddingSizes_.count(dim)) {
            auto& dimPad = dsc.N_.paddingSizes_.at(dim);
            if ((dimPad.padFront_ > 0 || dimPad.padBack_ > 0) &&
                dimPad.windowDim_ != PrimaryDimTypes::PrimaryDimTypesCount) {
              hasDimZeroPad = true;
              break;
            }
          }
        }

        if (hasDimZeroPad) {
          l3PaddingLdsIndices.push_back(ldsIdx);
          DT_CHECK(lds.memOrg_.at(SenComponents::LX).isPresent &&
                   lds.memOrg_.at(SenComponents::LX).isPadded &&
                   "Invalid zero padded.");
        }
      }
    }

    // Insert L3LU<->LXLU sync nodes if the input LX pinned tensor needs L3
    // padding.
    const bool hasL3Padding = !l3PaddingLdsIndices.empty();
    if (hasL3Padding) {
      DT_CHECK_MSG(
          l3PaddingLdsIndices.size() == 1 && l3PaddingLdsIndices.front() == 0,
          "Currently expect only the input tensor at index 0 has L3 "
          "padding.");
      const int ldsIdx = l3PaddingLdsIndices.front();
      // Add L3LU<->LXLU sync nodes after the L3 padding tensor's NO_COMPONENT
      // to LX transfer node.
      const auto& lds = dsc.labeledDs_.at(ldsIdx);
      DT_CHECK_MSG(lds.memOrg_.count(SenComponents::LX),
                   "Expect LX in memOrg_.");
      const auto allocNode = lds.memOrg_.at(SenComponents::LX).allocateNode_;
      DT_CHECK_MSG(allocNode, "Expect a valid allocate node.");
      dsc2::TransferNode* noCompToLxTransNode = nullptr;
      for (auto& [userNode, count] : allocNode->allocUsers_) {
        if (userNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
          auto transNode = static_cast<const dsc2::TransferNode*>(userNode);
          const bool isNoCompToLxTransfer =
              transNode->src_.storage_ == SenComponents::NO_COMPONENT &&
              (!transNode->dstVias_.empty() &&
               transNode->dstVias_.front().loc_.storage_ == SenComponents::LX);
          if (isNoCompToLxTransfer) {
            noCompToLxTransNode = const_cast<dsc2::TransferNode*>(transNode);
            break;
          }
        }
      }
      DT_CHECK_MSG(noCompToLxTransNode, "Expect a valid transfer node.");
      addL3LUAndLXLUSyncNodeSequence(noCompToLxTransNode);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 289/382   level 2   scc 95   76 body lines
// unit: e289_optimizeHbmTransfers
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4113
// original: void L3DlOpsScheduler::optimizeHbmTransfers(SuperDsc& mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e289_optimizeHbmTransfers(SuperDsc& mySDsc)
{
  for (auto& dsc : mySDsc.dscs_) {
    // Do not support cross-core reduction dataflow for now. To support, we need
    // to consider transfer node and its enclosing condition node.
    if (isOpCrossCoreReduction(mySDsc, dsc)) continue;

    DT_CHECK_MSG(!dsc.scheduleTree_.empty(), "Expect a valid schedule tree.");
    DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
                 "Expect valid core data stage parameters.");
    DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageChunkIdx),
                 "Expect valid chunk data stage parameters.");
    DT_CHECK_MSG(lxBufferType != BufferType::SPATIAL_DOUBLE ||
                     dsc.dataStageParam_.count(dataStageSuperChunkIdx),
                 "Expect valid SuperChunk data stage parameters.");

    // Get all HBM<->LX transfer nodes.
    const auto allTransferNodes = dsc.scheduleTree_.traverseTreeDFSMutable(
        nullptr, {dsc2::ScheduleNode::TRANSFER});
    for (auto node : allTransferNodes) {
      DT_CHECK_MSG(node->nodeType_ == dsc2::ScheduleNode::TRANSFER,
                   "Expect a transfer node.");
      dsc2::TransferNode* transNode = static_cast<dsc2::TransferNode*>(node);
      if (transNode->getTransferType() == dsc2::TransferNode::TENSOR_TO_TENSOR) {
        const auto ldsIdx = transNode->srcLdsAndLoopOffsets_.myLdsIdx_;
        const auto ldsNonBroadcastDimSet = dsc.getNonBroadcastLdsDimSet(ldsIdx);
        const auto &lds = dsc.labeledDs_.at(ldsIdx);
        const auto allocLxNode = lds.memOrg_.at(SenComponents::LX).allocateNode_;
        const auto allocOwnerLoop = allocLxNode->getOwnerLoop();
        auto currOwnerLoop = transNode->getMutableOwnerLoop();
        dsc2::LoopNode* siblingLoopNode = nullptr;
        while (currOwnerLoop && (currOwnerLoop != allocOwnerLoop)) {
          DT_CHECK_MSG(currOwnerLoop->dims_.size() == 1,
                       "Expect only one dimension.");
          const auto currDim = currOwnerLoop->dims_.back().dim_;
          if (ldsNonBroadcastDimSet.count(currDim) &&
              (dsc.dataStageParam_.at(currOwnerLoop->numId_)
                   .ss_.primaryDimToVal_st(currDim) !=
               dsc.dataStageParam_.at(currOwnerLoop->denId_)
                   .ss_.primaryDimToVal_st(currDim)))
            break;

          siblingLoopNode = currOwnerLoop;
          currOwnerLoop = currOwnerLoop->getMutableOwnerLoop();
        }
        // Hoist the transfer node to an outer loop.
        if (siblingLoopNode) {
          bool addBefore = true;
          if ((transNode->src_.storage_ == SenComponents::HBM &&
               !transNode->dstVias_.empty() &&
               transNode->dstVias_.front().loc_.storage_ ==
                   SenComponents::LX) ||
              (transNode->src_.storage_ == SenComponents::HBM &&
               !transNode->dstVias_.empty() &&
               transNode->dstVias_.front().loc_.storage_ ==
                   SenComponents::L3LUIBR) ||
              (transNode->src_.storage_ == SenComponents::LX &&
               !transNode->dstVias_.empty() &&
               transNode->dstVias_.front().loc_.storage_ ==
                   SenComponents::L3SUIBR))
            addBefore = true;
          else if (transNode->src_.storage_ == SenComponents::LX &&
                   !transNode->dstVias_.empty() &&
                   transNode->dstVias_.front().loc_.storage_ ==
                       SenComponents::HBM)
            addBefore = false;
          else
            DT_ERROR("Unexpected transfer.");

          transNode->getMutableParent()->moveChildNode(
              &dsc, transNode, siblingLoopNode->getMutableParent(), addBefore,
              siblingLoopNode);
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 290/382   level 2   scc 102   38 body lines
// unit: e290_createChunkLoops
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:4190
// original: void L3DlOpsScheduler::createChunkLoops(SuperDsc &mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e290_createChunkLoops(SuperDsc &mySDsc)
{
  // The loop order and data transfers for all DSCs in the same group should
  // be the same. Therefore, we use the first DSC in the group to determine
  // the loop order and data transfers. However, the chunk parameters could be
  // different among DSCs in a group, because each DSC may have different
  // amount of work.
  const int dscMainIdx = 0;
  auto &dsc = mySDsc.dscs_.at(dscMainIdx);

  // Return if no labeledDs is in the DSC.
  if (dsc.labeledDs_.size() < 1) return;
  /*
  if (dsc.labeledDs_.size() < 1) {
    dsc2::BlockNode* dummyBlockNode = createBlockNode(lxBelowBlockNodeName);
    dsc2::BlockNode *currNode = dsc.scheduleTree_.getHeadMutable();
    currNode->addChildNode(dummyBlockNode);
    return;
  }
    */

  const bool isReuse = hasDimensionReuse(dsc);

  // Collect the associated dimensions for the loop order.
  std::vector<PrimaryDimTypes> loopOrderDims =
      collectAllDimensionsForLoopOrder(dsc);

  // Build the schedule LabeledDs to dimensions table with all associated
  // dimensions.
  ScheduleDimTableType schedDimTypesTable =
      buildScheduleDimensionsTable(mySDsc, dscMainIdx, loopOrderDims, isReuse);

  // Build loop order using the schedule LabeledDs to dimensions table.
  const auto loopOrderInnerToOuter = buildLoopOrder(
      mySDsc, dscMainIdx, loopOrderDims, schedDimTypesTable, isReuse);

  // Create loop nodes in the schedule tree.
  createChunkLoopNodes(mySDsc, loopOrderInnerToOuter);
}

// ------------------------------------------------------------------------------------------------
// entry 291/382   level 2   scc 109   127 body lines
// unit: e291_fillTransferMulticastInfo
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5146
// original: void L3DlOpsScheduler::fillTransferMulticastInfo(SuperDsc &mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e291_fillTransferMulticastInfo(SuperDsc &mySDsc)
{
  // Traverse all labeledDs that resides in HBM in all DSCs.
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);

    // We only process HBM pinned tensors, including:
    //  1. Any value tensors that are double-buffered in LX.
    //  2. Any index tensors that contains address info for paged value
    //  tensors.
    //    2.1. The index tensor is transferred from HBM to L3LUIBR in the
    //    granularify of one stick.
    //    2.2. The index tensor is transferred from HBM to LX entirely or in
    //    the granularify of one stick.
    std::vector<int> hbmLdsIndices;
    for (const auto &lds : dsc.labeledDs_) {
      if (lds.isHbmPinned()) hbmLdsIndices.push_back(lds.ldsIdx_);
    }
    for (const auto ldsIdx : hbmLdsIndices) {
      auto &lds = dsc.labeledDs_.at(ldsIdx);
      DT_CHECK_MSG(lds.memOrg_.count(SenComponents::HBM),
                   "Expect HBM in memOrg_.");
      dsc2::AllocateNode *allocNode =
          lds.memOrg_.at(SenComponents::HBM).allocateNode_;
      DT_CHECK_MSG(allocNode, "Expect a valid LX allocate node.");

      // Find the L3LU transfer nodes for the current labeledDs.
      DT_CHECK_MSG(allocNode->hasAllocUsers(), "Expect valid alloc users.");
      std::vector<dsc2::TransferNode *> ldsL3LUTransferNodes;
      for (auto *node : dsc.scheduleTree_.traverseTreeDFSMutable(
               nullptr, {dsc2::ScheduleNode::TRANSFER})) {
        DT_CHECK_MSG(node->nodeType_ == dsc2::ScheduleNode::TRANSFER,
                     "Expect a transfer node.");
        dsc2::TransferNode *transNode = static_cast<dsc2::TransferNode *>(node);
        // L3 transfer nodes have transfers from HBM to LX or L3LUIBR.
        if (allocNode->hasAllocUser(transNode) &&
            (transNode->src_.storage_ == SenComponents::HBM &&
             (transNode->dstVias_[0].loc_.storage_ == SenComponents::LX ||
              transNode->dstVias_[0].loc_.storage_ == SenComponents::L3LUIBR)))
          ldsL3LUTransferNodes.push_back(transNode);
      }

      if (!ldsL3LUTransferNodes.empty()) {
        const auto ldsNonBroadcastDimsSet =
            dsc.getNonBroadcastLdsDimSet(ldsIdx);

        for (auto transNode : ldsL3LUTransferNodes) {
          // Check if there is limitation to use condGtr_. When there are more
          // than one unrelated parent chunk loop have different loop trip
          // counts between the DSCs, we are unable to use condGtr_. As a
          // result, the data transfer is broadcasted to only the cores in its
          // own DSC.
          // A data structure to store {loopNode, dim, currTripCount,
          // otherTripCount}.
          std::vector<
              std::tuple<const dsc2::LoopNode *, PrimaryDimTypes, int, int>>
              loopDimTripCounts;
          const auto parentLoops = getParentLoopNodes(*transNode, dsc);
          for (const auto loopNode : parentLoops) {
            for (const auto &dimKind : loopNode->dims_) {
              const auto currDim = dimKind.dim_;
              const bool isLdsUnrelatedDim =
                  !ldsNonBroadcastDimsSet.count(currDim);
              if (isLdsUnrelatedDim) {
                const int currTripCount = getTripCount(
                    dsc, currDim, loopNode->numId_, loopNode->denId_);
                for (const auto &otherDsc : mySDsc.dscs_) {
                  const int otherTripCount = getTripCount(
                      otherDsc, currDim, loopNode->numId_, loopNode->denId_);
                  if (currTripCount > otherTripCount) {
                    loopDimTripCounts.emplace_back(
                        loopNode, currDim, currTripCount, otherTripCount);
                    // Break because we only expect at most two different trip
                    // counts.
                    break;
                  }
                }
              }
            }
          }
          const bool isCondGtrLegal = loopDimTripCounts.size() < 2;

          // Set GTR info in the L3LU transfer node. No multicast support on
          // L3SU.
          for (const int currCoreId : dsc.coreIdsUsed_) {
            std::vector<int> allCoreIds;
            if (isCondGtrLegal) {
              // Look at all coreIds in the DSC group for the shares_ and
              // groupName_.
              for (const auto &[coreId, _] : mySDsc.coreIdToDsc_)
                allCoreIds.push_back(coreId);
            } else {
              // Look at coreIds in the current DSC for the shares_ and
              // groupName_.
              for (const auto coreId : dsc.coreIdsUsed_)
                allCoreIds.push_back(coreId);
            }
            const auto &currWkSlices = mySDsc.coreIdToWkSlice_.at(currCoreId);
            std::pair<size_t, size_t> sharesAndGroupName =
                getSharesAndGroupName(mySDsc, dsc, lds, currWkSlices,
                                      allCoreIds);

            // Put sharers and group name to the correponding transfer node in
            // the schedule tree.
            DT_CHECK_MSG(!transNode->coreIdToGTRInfo_.count(currCoreId),
                         "Do not expect an entry created.");
            dsc2::GroupTagRegInfo gtrInfo;
            gtrInfo.numSharers_ = sharesAndGroupName.first;
            // If there is only one sharer (meaning no share), we do not need a
            // group id so setting it to -1.
            gtrInfo.groupId_ =
                sharesAndGroupName.first > 1 ? sharesAndGroupName.second : -1;
            if (gtrInfo.groupId_ != -1) {
              dsc.gtrIdsUsed_.insert(gtrInfo.groupId_);
            }
            transNode->coreIdToGTRInfo_.emplace(currCoreId, gtrInfo);

            // Fill condGtr_.
            const bool doCondGtr = !loopDimTripCounts.empty();
            if (isCondGtrLegal && doCondGtr)
              setCondGtr(mySDsc, dscIdx, ldsIdx, currCoreId, *transNode,
                         loopDimTripCounts);
          }
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 292/382   level 2   scc 110   20 body lines
// unit: e292_fillAllocationStartAddrAndOffset
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5274
// original: void L3DlOpsScheduler::fillAllocationStartAddrAndOffset( SuperDsc &mySDsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// NOTE: THIS IS THE `start_address = 0` DEFECT. L3DlOpsScheduler::run calls this as "Set start
//       address, offset in allocations" (L3DlOpsScheduler.cpp:8000). Our emitted views print
//       start_address = 0 where the reference states a placed base. The effect of this function IS
//       the port.
// ------------------------------------------------------------------------------------------------
void e292_fillAllocationStartAddrAndOffset(
    SuperDsc &mySDsc) const
{
  // Fill start address and buffer offset in allocate node.
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
    std::vector<PrimaryDimTypes> coreletSplitDims =
        getCoreletSplitDimensions(dsc);

    for (auto &lds : dsc.labeledDs_) {
      // Fill the LX allocate node for the value tensors or the index tensor
      // that has to be loaded to LX.
      if ((lds.isHbmPinned() &&
           !(isIndexLds(lds) && !lds.memOrg_.count(SenComponents::LX))) ||
          lds.isLxPinned())
        fillFinalStartAddressAndOffset(dsc, lds.ldsIdx_, coreletSplitDims);

      if ((lds.isHbmPinned() && isIndexLds(lds)))
        fillIBRStartAddressAndOffset(mySDsc, dsc, lds.ldsIdx_);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 293/382   level 2   scc 122   26 body lines
// unit: e293_setLxBufferType
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6425
// original: void L3DlOpsScheduler::setLxBufferType(const SuperDsc &mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e293_setLxBufferType(const SuperDsc &mySDsc)
{
  // Check if buffer type is forced
  if (lxBufferTypeMode == LxBufferTypeMode::FORCE_SPATIAL_DOUBLE) {
    lxBufferType = BufferType::SPATIAL_DOUBLE;
    return;
  }
  if (lxBufferTypeMode == LxBufferTypeMode::FORCE_DOUBLE) {
    lxBufferType = BufferType::DOUBLE;
    return;
  }

  // FIXME: Temporarily force double buffering for target rcudd1a and below.
  // Remove when fixed.
  if (dscGlobal.sysDef.coreArch <= IsaCoreGen::RCUDD1A_ISA) {
    lxBufferType = BufferType::DOUBLE;
    return;
  }

  // AUTO mode: Use spatial-double buffering if the number of requests from a
  // core group to an HMI is not greater than a heuristic threshold.
  const int numRequests = getHbmLdsTransferHMIRequestEstimate(mySDsc);
  constexpr int heuristicThreshold = 2;
  lxBufferType = (numRequests <= heuristicThreshold)
                     ? BufferType::SPATIAL_DOUBLE
                     : BufferType::DOUBLE;
}

// ------------------------------------------------------------------------------------------------
// entry 294/382   level 2   scc 129   85 body lines
// unit: e294_createStoreIndexTensorToLx
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6948
// original: void L3DlOpsScheduler::createStoreIndexTensorToLx( SuperDsc &mySDsc, const int dscIdx, const int pagedLdsIdx, const int indexLdsIdx, dsc2::AllocateNode &indexLdsHbmAllocNode, dsc2::LoopNode &newChunkLoopNode)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e294_createStoreIndexTensorToLx(
    SuperDsc &mySDsc, const int dscIdx, const int pagedLdsIdx,
    const int indexLdsIdx, dsc2::AllocateNode &indexLdsHbmAllocNode,
    dsc2::LoopNode &newChunkLoopNode)
{
  auto &dsc = mySDsc.dscs_.at(dscIdx);
  auto &pagedLds = dsc.labeledDs_.at(pagedLdsIdx);
  auto &indexLds = dsc.labeledDs_.at(indexLdsIdx);
  // Create LX allocate node for preloading its index tensor as the default
  // strategy. Also add an MemOrg entry for LX for the index tensor's labeledDs
  // if not exists.
  const SenComponents allocLxStorage = SenComponents::LX;
  const std::string allocLxNodeName =
      "allocate_lds" + std::to_string(indexLds.ldsIdx_) + "_" +
      EnumsConversion::senComponentsToString.at(allocLxStorage);
  dsc2::AllocateNode *indexLdsLxAllocNode =
      createAllocateNode(dsc, indexLdsIdx, allocLxStorage, 1 /* no buffering */,
                         allocLxNodeName, dscIdx);
  indexLdsLxAllocNode->indirectAllocType_ =
      indexLdsHbmAllocNode.indirectAllocType_;
  DT_CHECK_MSG(pagedLds.memOrg_.count(SenComponents::LX),
               "Expect LX in memOrg_.");
  const auto pagedLdsLxAllocNode =
      pagedLds.memOrg_.at(SenComponents::LX).allocateNode_;
  DT_CHECK_MSG(pagedLdsLxAllocNode,
               "Expect valid paged tensor LX allocate node.");
  indexLdsLxAllocNode->relatedIndirectAccessAlloc_ = pagedLdsLxAllocNode;
  // Add memorg entry if it doesn't exist.
  indexLds.memOrg_[SenComponents::LX].allocateNode_ = indexLdsLxAllocNode;
  // Add LX allocate node to schedule tree after its HBM allocate node.
  auto parentInsertNode = indexLdsHbmAllocNode.getMutableParent();
  DT_CHECK_MSG(parentInsertNode, "Expect a valid parent node.");
  parentInsertNode->addChildNode(indexLdsLxAllocNode, /*addBefore*/ false,
                                 &indexLdsHbmAllocNode);

  const bool isLxMemSufficient = allocAllMem(mySDsc, &dsc, dscIdx, false);
  // If the remaining LX memory cannot fit the entire index tensor, use the
  // fallback strategy that loads one stick of the index tensor to LX at a time
  // in double buffering form, by changing the LX allocate node.
  if (!isLxMemSufficient) {
    // Change the number of buffers to two.
    indexLdsLxAllocNode->numBuffers_ = 2;
    // Move the allocate node to the first child in the core/ibr loop which is
    // the parent of the newChunkLoopNode.
    parentInsertNode = newChunkLoopNode.getMutableParent();
    indexLdsLxAllocNode->moveNode(&dsc, parentInsertNode, true);
    DT_CHECK_MSG(allocAllMem(mySDsc, &dsc, dscIdx, false),
                 "Memory allocation must be valid to commit.");
  }

  // Create HBM->LX transfer node for its index tensor.
  const SenComponents transLxSrcStorage = SenComponents::HBM;
  const SenComponents transLxDstStorage = SenComponents::LX;
  const std::string transLxNodeName =
      "transfer_lds" + std::to_string(indexLds.ldsIdx_) +
      "_src:" + EnumsConversion::senComponentsToString.at(transLxSrcStorage) +
      "_dst:" + EnumsConversion::senComponentsToString.at(transLxDstStorage);
  dsc2::TransferNode *indexLdsLxTransNode = createTransferNode(
      SenComponents::L3LU, transLxSrcStorage, {SenComponents::L3LU},
      {transLxDstStorage}, indexLdsIdx, {indexLdsIdx}, transLxNodeName);
  indexLdsHbmAllocNode.addAllocUser(indexLdsLxTransNode);
  indexLdsLxAllocNode->addAllocUser(indexLdsLxTransNode);

  // Create a pair of L3LU to L3SU sync nodes.
  constexpr SenComponents srcUnit = SenComponents::L3LU;
  constexpr SenComponents dstUnit = SenComponents::L3SU;
  dsc2::SyncNode *newL3LUSyncSendNode = createSyncNode(
      {srcUnit},
      "sync_send_" + EnumsConversion::senComponentsToString.at(srcUnit) +
          "_to_" + EnumsConversion::senComponentsToString.at(dstUnit) +
          "_paged_index_" + std::to_string(indexLdsIdx));
  dsc2::SyncNode *newL3SUSyncRecvNode = createSyncNode(
      {dstUnit},
      "sync_receive_" + EnumsConversion::senComponentsToString.at(dstUnit) +
          "_from_" + EnumsConversion::senComponentsToString.at(srcUnit) +
          "_paged_index_" + std::to_string(indexLdsIdx),
      true);
  newL3LUSyncSendNode->otherEndOfTheSignals_.push_back(newL3SUSyncRecvNode);
  newL3SUSyncRecvNode->otherEndOfTheSignals_.push_back(newL3LUSyncSendNode);

  // Add the index tensor LX transfer nodes as well as the sync nodes after its
  // LX allocate node in schedule tree.
  parentInsertNode->addChildNode(indexLdsLxTransNode, /*addBefore*/ false,
                                 indexLdsLxAllocNode);
  parentInsertNode->addChildNode(newL3LUSyncSendNode, /*addBefore*/ false,
                                 indexLdsLxTransNode);
  parentInsertNode->addChildNode(newL3SUSyncRecvNode, /*addBefore*/ false,
                                 newL3LUSyncSendNode);
}

// ------------------------------------------------------------------------------------------------
// entry 295/382   level 2   scc 145   35 body lines
// unit: e295_fillExplicitTransferSize
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7875
// original: void L3DlOpsScheduler::fillExplicitTransferSize(SuperDsc &mySDsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e295_fillExplicitTransferSize(SuperDsc &mySDsc) const
{
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
    if (dsc.numCoreletsUsed_DSC2_ > 1 && isOpCrossCoreReduction(mySDsc, dsc)) {
      auto allTransNodes = dsc.scheduleTree_.traverseTreeDFSMutable(
          nullptr, {dsc2::ScheduleNode::TRANSFER});
      std::vector<dsc2::TransferNode *> outputLdsL3SUTransNodes;
      for (auto currNode : allTransNodes) {
        auto currTransNode = dynamic_cast<dsc2::TransferNode *>(currNode);
        DT_CHECK_MSG(currTransNode, "Expect a valid transfer node.");
        if (currTransNode->getTransferType() ==
                dsc2::TransferNode::TransferType::TENSOR_TO_TENSOR &&
            isOutputLabeledDs(currTransNode->srcLdsAndLoopOffsets_.myLdsIdx_,
                              dsc) &&
            currTransNode->src_.storage_ == SenComponents::LX &&
            !currTransNode->dstVias_.empty() &&
            currTransNode->dstVias_.front().loc_.storage_ == SenComponents::HBM)
          outputLdsL3SUTransNodes.push_back(currTransNode);
      }
      DT_CHECK_MSG(outputLdsL3SUTransNodes.size() <= 1,
                   "Currently support at most one L3SU transfer.");
      for (auto transNode : outputLdsL3SUTransNodes) {
        DT_CHECK_MSG(transNode->transferSize_.empty(),
                     "Expect empty transferSize_ field.");
        // Set the explicit transfer size to the size for one corelet.
        constexpr int corelet0Id = 0;
        const auto transSizePerDim = dsc.getBlockTransferSizePerDim(
            *transNode, transNode->src_.storage_,
            /* for any one of the corelets */ corelet0Id);
        for (const auto &[dim, size] : transSizePerDim)
          transNode->transferSize_.emplace(dim, size);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 328/382   level 3   scc 49   24 body lines
// unit: e328_computeMinParamForPaddedDim
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:870
// original: long L3DlOpsScheduler::computeMinParamForPaddedDim( const DesignSpaceConfig& dsc, const PrimaryDimTypes dim) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
long e328_computeMinParamForPaddedDim(
    const DesignSpaceConfig& dsc, const PrimaryDimTypes dim) const
{
  const auto opFuncName = getOpFuncName(dsc);
  DT_CHECK_MSG(isOpFuncStridedWindow(opFuncName),
               "Expect a strided-window op.");
  constexpr long defaultParam = 1;
  const auto& dsCoreSs = dsc.dataStageParam_.at(dataStageCoreIdx).ss_;
  const auto coreParam = dsCoreSs.primaryDimToVal_st(dim);
  if (dsCoreSs.paddingSizes_.count(dim)) {
    auto& padInfo = dsCoreSs.paddingSizes_.at(dim);
    const int totPadding = padInfo.padFront_ + padInfo.padBack_;
    const int hyPadChunks = totPadding / padInfo.stride_;
    const int hyPadThresh = (hyPadChunks > 10) ? 5 : 6;
    if (hyPadChunks > hyPadThresh) {
      const int minChkSize = std::ceil(static_cast<double>(hyPadChunks) /
                                       static_cast<double>(hyPadThresh));
      for (int cand = minChkSize; cand <= coreParam; cand++) {
        if (coreParam % cand == 0) {
          return cand;
        }
      }
    }
  }
  return defaultParam;
}

// ------------------------------------------------------------------------------------------------
// entry 329/382   level 3   scc 65   13 body lines
// unit: e329_addChunkDataStageFromCandidates
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1405
// original: void L3DlOpsScheduler::addChunkDataStageFromCandidates( DataStructDims &chunkParams, DesignSpaceConfig &dsc, const int dscIdx, const DscParamCandidateIndicesType &selectedIndices, const DscParamCandidatesType &dscCandidates, const std::vector<PrimaryDimTypes> &primaryDims)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e329_addChunkDataStageFromCandidates(
    DataStructDims &chunkParams, DesignSpaceConfig &dsc, const int dscIdx,
    const DscParamCandidateIndicesType &selectedIndices,
    const DscParamCandidatesType &dscCandidates,
    const std::vector<PrimaryDimTypes> &primaryDims)
{
  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
               "Core data stage parameters are unavailable.");
  const std::string chunkDsName = "chunk";
  const auto &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);
  getChunkParamsFromCandidates(chunkParams, selectedIndices, dscCandidates,
                               dscIdx, primaryDims);
  addOrUpdateCoreletSplitInParams(chunkParams, dsc);
  addOrUpdatePaddingSizesInChunkParams(chunkParams, dsCore.ss_);
  addOrUpdateSymbolicInfoInParams(chunkParams, dsCore.ss_);
  addOrUpdateDataStageParam(dsc, chunkParams, chunkDsName, chunkParams,
                            chunkDsName, dataStageChunkIdx);
}

// ------------------------------------------------------------------------------------------------
// entry 330/382   level 3   scc 83   17 body lines
// unit: e330_getLdsTransferCoreIds
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2773
// original: std::vector<int> L3DlOpsScheduler::getLdsTransferCoreIds( const SuperDsc &mySDsc, const DesignSpaceConfig &dsc, const LabeledDsInfo &lds) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
std::vector<int> e330_getLdsTransferCoreIds(
    const SuperDsc &mySDsc, const DesignSpaceConfig &dsc,
    const LabeledDsInfo &lds) const
{
  std::vector<int> selectedCoreIds;
  if (isOutputLabeledDs(lds.ldsIdx_, dsc) &&
      isOpCrossCoreReduction(mySDsc, dsc)) {
    const auto coreGroupInfo = getCrossCoreReductionGroupInfo(mySDsc, dsc);
    for (const auto &coreGroup : coreGroupInfo) {
      for (int coreletId = 0; coreletId < dsc.numCoreletsUsed_DSC2_;
           ++coreletId)
        selectedCoreIds.push_back(coreGroup.getEndCoreAtCorelet(coreletId));
    }
  } else {
    for (const auto &[coreId, wkSlice] : mySDsc.coreIdToWkSlice_)
      selectedCoreIds.push_back(coreId);
  }

  return selectedCoreIds;
}

// ------------------------------------------------------------------------------------------------
// entry 331/382   level 3   scc 85   140 body lines
// unit: e331_exploreSuperChunkDataStageParams
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2829
// original: void L3DlOpsScheduler::exploreSuperChunkDataStageParams(SuperDsc& mySDsc, const int dscIdx)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e331_exploreSuperChunkDataStageParams(SuperDsc& mySDsc,
                                                        const int dscIdx)
{
  auto& dsc = mySDsc.dscs_.at(dscIdx);
  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageSuperChunkIdx),
               "Expect SuperChunk data stage.");
  // TODO: Add proper exploration. Also, need to consider paged tensors. For
  // example, the superchunk size cannot exceeds the amount that the IBR
  // register can represent. A temporary set of SuperChunk parameters for
  // testing purpose.
  const auto* lxBelowBlockNode = getLxBelowBlockNode(dsc.scheduleTree_);
  const auto& dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);
  const auto& dsChunk = dsc.dataStageParam_.at(dataStageChunkIdx);
  auto& dsSuperChunk = dsc.dataStageParam_.at(dataStageSuperChunkIdx);

  if (enableSuperChunkExplore) {
    std::vector<const dsc2::LoopNode *> innerToOuterChunkLoops;
    auto currLoop = lxBelowBlockNode->getOwnerLoop();
    while (currLoop && currLoop->denId_ == dataStageChunkIdx) {
      innerToOuterChunkLoops.push_back(currLoop);
      currLoop = currLoop->getOwnerLoop();
    }
    const auto pagedDims = getPagedDimensions(dsc);

    // Initialize the SuperChunk parameters to the maximum.
    for (auto currLoop : reverse(innerToOuterChunkLoops)) {
      const auto currDim = currLoop->dims_.back().dim_;
      if (!pagedDims.empty() && is_any_of(currDim, pagedDims)) {
        // For a paged dimension, the maximum is the IBR data stage value.
        const auto currDimChunkSsVal = dsChunk.ss_.primaryDimToVal_st(currDim);
        DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageIbrIdx),
                     "Expect IBR data stage.");
        const auto& dsIbr = dsc.dataStageParam_.at(dataStageIbrIdx);
        const auto currDimIbrSsVal = dsIbr.ss_.primaryDimToVal_st(currDim);
        const auto currDimIbrElVal = dsIbr.el_.primaryDimToVal_st(currDim);
        // Use min(ibr.ss_, ibr.el_) as the maximum SuperChunk ss_ value.
        const auto currDimSuperChunkSsVal =
            std::min(currDimIbrSsVal, currDimIbrElVal);
        DT_CHECK_MSG(
            currDimSuperChunkSsVal % currDimChunkSsVal == 0,
            "The SuperChunk value must be multiple of the chunk value.");
        dsSuperChunk.ss_.primaryDimToValHandler_st(currDim) =
            currDimSuperChunkSsVal;
        // Compute the SuperChunk el_ value.
        const auto currDimSuperChunkElVal =
            (currDimIbrSsVal % currDimSuperChunkSsVal != 0)
                ? currDimIbrSsVal % currDimSuperChunkSsVal
                : currDimSuperChunkSsVal;
        DT_CHECK_MSG(
            currDimSuperChunkElVal % currDimChunkSsVal == 0,
            "The SuperChunk value must be multiple of the chunk value.");
        dsSuperChunk.el_.primaryDimToValHandler_st(currDim) =
            currDimSuperChunkElVal;
      } else {
        // For a non-paged dimension, the maximum is the core data stage value.
        const auto currDimCoreSsVal = dsCore.ss_.primaryDimToVal_st(currDim);
        const auto currDimCoreElVal = dsCore.el_.primaryDimToVal_st(currDim);
        DT_CHECK_MSG(currDimCoreSsVal == currDimCoreElVal,
                     "Expect the same ss_ and el_ values.");
        dsSuperChunk.ss_.primaryDimToValHandler_st(currDim) = currDimCoreSsVal;
        dsSuperChunk.el_.primaryDimToValHandler_st(currDim) = currDimCoreElVal;
      }
    }
    addOrUpdateCoreletSplitInParams(dsSuperChunk.ss_, dsc);
    addOrUpdateCoreletSplitInParams(dsSuperChunk.el_, dsc);

    // Explore by decreasing the SuperChunk parameters from the outer loop
    // dimension to the inner loop dimension until the LX allocation fits in
    // available LX space.
    if (!allocAllMem(mySDsc, &dsc, dscIdx, false)) {
      bool found = false;
      for (auto currLoop : reverse(innerToOuterChunkLoops)) {
        const auto currDim = currLoop->dims_.back().dim_;
        const auto& refDs =
            (!pagedDims.empty() && is_any_of(currDim, pagedDims))
                ? dsc.dataStageParam_.at(dataStageIbrIdx)
                : dsCore;
        const auto currDimRefSsVal = refDs.ss_.primaryDimToVal_st(currDim);
        const auto currDimRefElVal = refDs.el_.primaryDimToVal_st(currDim);
        const auto currDimChunkVal = dsChunk.ss_.primaryDimToVal_st(currDim);
        auto currDimSuperChunkSsVal =
            dsSuperChunk.ss_.primaryDimToVal_st(currDim);
        while (currDimSuperChunkSsVal > currDimChunkVal) {
          currDimSuperChunkSsVal -= currDimChunkVal;
          if (currDimRefSsVal % currDimSuperChunkSsVal ==
              currDimRefElVal % currDimSuperChunkSsVal) {
            const auto currDimSuperChunkElVal =
                (currDimRefSsVal % currDimSuperChunkSsVal != 0)
                    ? currDimRefSsVal % currDimSuperChunkSsVal
                    : currDimSuperChunkSsVal;
            if (enableSuperChunkEpilogue ||
                (currDimSuperChunkSsVal == currDimSuperChunkElVal)) {
              dsSuperChunk.ss_.primaryDimToValHandler_st(currDim) =
                  currDimSuperChunkSsVal;
              dsSuperChunk.el_.primaryDimToValHandler_st(currDim) =
                  currDimSuperChunkElVal;
              addOrUpdateCoreletSplitInParams(dsSuperChunk.ss_, dsc);
              addOrUpdateCoreletSplitInParams(dsSuperChunk.el_, dsc);
              if (allocAllMem(mySDsc, &dsc, dscIdx, false)) {
                found = true;
                break;
              }
            }
          }
        }
        if (found) break;
      }
      DT_CHECK_MSG(found,
                   "A valid set of SuperChunk parameters must be found.");
    }
  } else {
    const auto* innermostLoopNode = lxBelowBlockNode->getOwnerLoop();
    DT_CHECK(innermostLoopNode &&
             innermostLoopNode->getOwnerLoop()->numId_ ==
                 dataStageSuperChunkIdx &&
             innermostLoopNode->getOwnerLoop()->denId_ == dataStageChunkIdx);
    auto currLoop = const_cast<dsc2::LoopNode*>(innermostLoopNode);
    while (currLoop && currLoop->numId_ != dataStageCoreIdx) {
      const auto currDim = currLoop->dims_.back().dim_;
      const auto currDimCoreVal = dsCore.ss_.primaryDimToVal_st(currDim);
      const auto currDimSuperChunkVal =
          dsSuperChunk.ss_.primaryDimToVal_st(currDim);
      if ((currDimCoreVal > currDimSuperChunkVal) &&
          (currDimCoreVal % currDimSuperChunkVal == 0)) {
        // Update SuperChunk parameter and test its capacity.
        const auto newCurrDimSuperChunkVal = currDimSuperChunkVal * 2;
        dsSuperChunk.ss_.primaryDimToValHandler_st(currDim) =
            dsSuperChunk.el_.primaryDimToValHandler_st(currDim) =
                newCurrDimSuperChunkVal;
        if (allocAllMem(mySDsc, &dsc, dscIdx, false)) break;

        // Capacity test fails, recover the SuperChunk parameters.
        dsSuperChunk.ss_.primaryDimToValHandler_st(currDim) =
            dsSuperChunk.el_.primaryDimToValHandler_st(currDim) =
                currDimSuperChunkVal;
      }
      currLoop = currLoop->getMutableOwnerLoop();
    }
  }

  return;
}

// ------------------------------------------------------------------------------------------------
// entry 332/382   level 3   scc 92   5 body lines
// unit: e332_createSynchronization
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3481
// original: void L3DlOpsScheduler::createSynchronization(SuperDsc &mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e332_createSynchronization(SuperDsc &mySDsc)
{
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    createSynchronizationDSC(mySDsc, dscIdx);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 333/382   level 3   scc 117   613 body lines
// unit: e333_fillLoopOffsetsAndAddresses
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:5750
// original: void L3DlOpsScheduler::fillLoopOffsetsAndAddresses( SuperDsc &mySDsc, const int dscIdx, const bool allowUnpaddedIndexingAtPaddedNoZeroPad)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e333_fillLoopOffsetsAndAddresses(
    SuperDsc &mySDsc, const int dscIdx,
    const bool allowUnpaddedIndexingAtPaddedNoZeroPad)
{
  // Below is the L3DlOpsScheduler specific implementation for now.

  DesignSpaceConfig *currDsc = &mySDsc.dscs_.at(dscIdx);
  auto &metadata = dscMetadata.at(dscIdx);
  DT_CHECK_MSG(metadata.datatransfers_.empty(),
               "Expect empty datatransfers_ in metadata for now.");

  // End of L3DlOpsScheduler specific implementation.
  // The original fillLoopOffsetsAndAddresses in DDC starts here.

  auto scheduleTreeTraverse = currDsc->scheduleTree_.traverseTreeDFSMutable();
  // identify loops below chunk boundary
  std::unordered_set<const dsc2::LoopNode *> loopsBelowChunkBoundary;
  // there are no loops below chunk boundary in ALxS
  // dsc2::LoopNode *lastFoundChunkLoop = nullptr;
  // for (auto *node : scheduleTreeTraverse) {
  //   if (node->nodeType_ == dsc2::ScheduleNode::LOOP) {
  //     auto *loop = static_cast<dsc2::LoopNode *>(node);
  //     if (loop->numId_ == metadata.core_dstgid &&
  //         loop->denId_ == metadata.chunk_dstgid) {
  //       lastFoundChunkLoop = loop;
  //       continue;
  //     }
  //     if (loop->getOwnerLoop() == lastFoundChunkLoop ||
  //         loopsBelowChunkBoundary.count(loop->getOwnerLoop())) {
  //       loopsBelowChunkBoundary.insert(loop);
  //     }
  //   }
  // }

  auto fillDataInfo = [&](dsc2::DataInfo &di, dsc2::DataInfo *indirectDi,
                          DataLocation loc, DataLocation *indirectLoc,
                          const dsc2::LoopNode *loopLocation, bool isProducer) {
    if (di.myLdsIdx_ < 0 && di.constantId_ < 0) return;
    if (dsc2::memories.count(loc.storage_) == 0) return;
    bool hasIndirect = (indirectDi != nullptr);
    if (hasIndirect) DT_CHECK_MSG(indirectLoc, "Expect valid indirect loc.");

    // fill start address here
    if (di.myLdsIdx_ >= 0 &&
        !currDsc->labeledDs_.at(di.myLdsIdx_).memOrg_.count(loc.storage_)) {
      di.print(std::cerr, 0);
      if (loopLocation) {
        loopLocation->print(std::cerr, 0);
      }
      currDsc->printLabeledDs(std::cerr, currDsc->labeledDs_.at(di.myLdsIdx_),
                              "");
      DT_ERROR("Error in fillDataInfo: LabeldDs[" +
               std::to_string(di.myLdsIdx_) + "]" +
               " does not have memOrg_ entry for DataLocation storge " +
               EnumsConversion::senComponentsToString.at(loc.storage_));
      DT_ERROR("Error in fillDataInfo: LabeldDs[" +
               std::to_string(di.myLdsIdx_) + "]" +
               " does not have memOrg_ entry for DataLocation storge " +
               EnumsConversion::senComponentsToString.at(loc.storage_));
    }

    const auto &allocation = di.myLdsIdx_ >= 0
                                 ? currDsc->labeledDs_.at(di.myLdsIdx_)
                                       .memOrg_.at(loc.storage_)
                                       .allocateNode_
                                 : currDsc->constantInfo_.at(di.constantId_)
                                       .allocations_.at(loc.storage_);
    di.startAddr_.clone(allocation->startAddressCoreCorelet_);
    di.isStartAddrSymbolic_ = allocation->isStartAddrSymbolic_;

    // Adjust the HBM address for cross-core reduction with corelet split (more
    // than one corelet).
    if (currDsc->numCoreletsUsed_DSC2_ > 1 &&
        isOpCrossCoreReduction(mySDsc, *currDsc) &&
        isOutputLabeledDs(di.myLdsIdx_, *currDsc) &&
        loc.storage_ == SenComponents::HBM) {
      const auto crossCoreGroupInfo =
          getCrossCoreReductionGroupInfo(mySDsc, *currDsc);
      constexpr int corelet1Id = 1;
      std::unordered_set<int> corelet1EndCoreIds;
      for (const auto &coreGroup : crossCoreGroupInfo) {
        DT_CHECK_MSG(!coreGroup.isEmpty(), "Expect valid core group.");
        corelet1EndCoreIds.insert(coreGroup.getEndCoreAtCorelet(corelet1Id));
      }
      if (di.isStartAddrSymbolic_)
        DT_ERROR("Currently no support; work in progress");
      else {
        // Compute the core offsets.
        const auto coreOffsets =
            calculateCoreletOffsetInByte(*currDsc, allocation);
        // Add the offset to startAddr_.
        for (auto &[coord, addr] :
             di.startAddr_.getDataAndFoldCoordinates({{1, 0}})) {
          constexpr int coreFoldPos = 0;
          if (corelet1EndCoreIds.count(coord[coreFoldPos])) {
            di.startAddr_.insertData(addr + coreOffsets[corelet1Id], coord);
          }
        }
      }
    }

    auto genericUnit = EnumsConversion::senCompToGenericComp.at(loc.unit_);
    const auto &addrScale = dscGlobal.sysDef.addressGranularityScalePerUnit.at(
        {genericUnit, loc.storage_});
    DT_CHECK(addrScale > 0);
    if (addrScale != 1) {
      if (di.isStartAddrSymbolic_) {
        di.startAddr_.apply({}, [&](auto &&sym) {
          return mySDsc.symbolDefinitions_.addVar(
              VariableOperator::DIV, {{true, sym}, {false, addrScale}});
        });
      } else {  // non symbolic address
        di.startAddr_.apply({}, std::divides<int64_t>(), addrScale);
      }
    }

    dsc2::AllocateNode *indAllocation = nullptr;
    if (hasIndirect) {
      const int indexLdsIdx = indirectDi->myLdsIdx_;
      DT_CHECK_MSG(indexLdsIdx >= 0, "Expect a valid index tensor.");
      const auto &indexLds = currDsc->labeledDs_.at(indexLdsIdx);
      indAllocation = indexLds.memOrg_.at(indirectLoc->storage_).allocateNode_;
      DT_CHECK_MSG(indAllocation, "Expect a valid indirect allocate node.");
      DT_CHECK_MSG(indirectLoc->storage_ == SenComponents::L3LUIBR ||
                       indirectLoc->storage_ == SenComponents::L3SUIBR,
                   "Expect indirection storage as L3LUIBR or L3SUIBR.");
      auto foldProps = indAllocation->startAddressCoreCorelet_.getFoldDimProp();
      auto foldTypes = indAllocation->startAddressCoreCorelet_.getFuncType();
      DT_CHECK(foldTypes.size() >= 2);
      foldTypes.front() = BaseFuncType::Map;
      indirectDi->startAddr_.buildFoldSpace(foldProps, foldTypes);
      indirectDi->startAddr_.apply(indAllocation->startAddressCoreCorelet_, {},
                                   [](auto &&l, auto &&r) { return r; });
      DT_CHECK(!indAllocation->isStartAddrSymbolic_);
      // Get the index tensor stick dimension size at IBR datastage without
      // rounding to a full stick.
      const auto indexLdsIbrSizesPerDimNoRounding =
          currDsc->getBufferCapacityForNodePerDim(
              indAllocation, indexLdsIdx, indirectLoc->storage_, -1, -1, true);
      const auto indexLdsStickDimSet = currDsc->getStickDimSet(indexLdsIdx);
      const auto indexLdsCumulativeStickSizes =
          currDsc->getCumulativeStickSizes(indexLds.dsType_);
      std::unordered_map<PrimaryDimTypes, int>
          indexLdsStickDimIbrSizesNoRounding;
      const auto pageSize = indAllocation->getPageSize();
      std::map<PrimaryDimTypes, VariableSymbol> dimToIndexCoreSizeSymbol;
      for (const auto &[dim, size] : indexLdsIbrSizesPerDimNoRounding) {
        if (indexLdsStickDimSet.count(dim)) {
          DT_CHECK_MSG(!indexLdsStickDimIbrSizesNoRounding.count(dim),
                       "Same stick dimension in multiple coordinates in the "
                       "layout is currently not supported.");
          DT_CHECK_MSG(size <= indexLdsCumulativeStickSizes.at(dim),
                       "IBR stick dimension size without rounding should "
                       "always be not greater than the stick size.");
          indexLdsStickDimIbrSizesNoRounding.emplace(dim, size);
          if (currDsc->dimToSymbolMapping_.count(dim) &&
              mySDsc.numWkSlicesPerDim_.at(dim) > 1) {
            const auto &symbols = currDsc->dimToSymbolMapping_.at(dim);
            DT_CHECK(symbols.size() == 1);
            dimToIndexCoreSizeSymbol.emplace(
                dim, mySDsc.symbolDefinitions_.addVar(
                         VariableOperator::DIV_CEIL,
                         {{true, symbols.at(0)}, {false, pageSize.at(dim)}}));
          }
        }
      }
      bool needSymbolicAddress = !dimToIndexCoreSizeSymbol.empty();
      if (needSymbolicAddress) indirectDi->isStartAddrSymbolic_ = true;

      const auto indexLdsStickSizes = currDsc->getStickSizes(indexLds.dsType_);
      DT_CHECK_MSG(
          indexLdsCumulativeStickSizes.size() == indexLdsStickSizes.size(),
          "Same stick dimension in multiple coordinates in the stick layout is "
          "currently not supported.");
      auto genericUnit =
          EnumsConversion::senCompToGenericComp.at(indirectLoc->unit_);
      const auto &addrScale =
          dscGlobal.sysDef.addressGranularityScalePerUnit.at(
              {genericUnit, indirectLoc->storage_});
      DT_CHECK(indexLds.wordLength == 4);

      // Compute the core offset for startAddr_ in the IBR register for each
      // core.
      for (auto &[coord, startAddr] :
           indirectDi->startAddr_.getDataAndFoldCoordinates()) {
        const auto &coreId = coord.at(0);
        if (!mySDsc.coreIdToWkSlice_.count(coreId)) {
          continue;
        }
        int64_t offsetInBytes = 0;
        VariableDefinition::OperandsType operands;  // for affine expression
        const int64_t numBytesInStick = dscGlobal.sysDef.bytesPerStick;
        for (const auto &[stickDim, stickSize] : indexLdsStickSizes) {
          const auto wkSliceId =
              mySDsc.coreIdToWkSlice_.at(coreId).at(stickDim);
          DT_CHECK_MSG(indexLdsStickDimIbrSizesNoRounding.count(stickDim),
                       "Expect the stick dimension size available.");
          const auto dimIbrSize =
              indexLdsStickDimIbrSizesNoRounding.at(stickDim);
          if (wkSliceId == 0) continue;
          if (needSymbolicAddress) {
            operands.push_back(
                {false, int64_t(wkSliceId * indexLds.wordLength)});
            if (dimToIndexCoreSizeSymbol.count(stickDim)) {
              operands.push_back({true, dimToIndexCoreSizeSymbol.at(stickDim)});
            } else {
              operands.push_back({false, dimIbrSize});
            }
          } else {
            offsetInBytes += wkSliceId * dimIbrSize * indexLds.wordLength;
          }
        }
        // Adjust the offset as the relative to the start of a stick.
        int64_t newAddr;
        if (needSymbolicAddress) {
          if (operands.empty()) {
            newAddr = mySDsc.symbolDefinitions_.addVar(VariableOperator::CONST,
                                                       {{false, startAddr}});
          } else {
            newAddr = mySDsc.symbolDefinitions_.addVar(VariableOperator::MACC,
                                                       operands);
            newAddr = mySDsc.symbolDefinitions_.addVar(
                VariableOperator::MOD,
                {{true, newAddr}, {false, numBytesInStick}});
            if (addrScale != 1) {
              newAddr = mySDsc.symbolDefinitions_.addVar(
                  VariableOperator::DIV, {{true, newAddr}, {false, addrScale}});
            }
            if (startAddr != 0) {
              newAddr = mySDsc.symbolDefinitions_.addVar(
                  VariableOperator::ADD, {{true, newAddr}, {false, startAddr}});
            }
          }
        } else {
          offsetInBytes = offsetInBytes % numBytesInStick;
          newAddr = startAddr + (offsetInBytes / addrScale);
        }
        indirectDi->startAddr_.insertData(newAddr, coord);
      }
    }

    // fill offsets
    if (di.myLdsIdx_ < 0) return;  // no offsets for constants
    DT_CHECK(di.loopEleOffsets_.empty());
    const auto &myLds = currDsc->labeledDs_.at(di.myLdsIdx_);
    const auto pageSizes = allocation->getPageSize();
    auto loopPtr = loopLocation;
    bool belowChunkLimit = loopsBelowChunkBoundary.count(loopPtr);
    bool reachIndAllocOwnerLoop = false;
    // const auto& dataConnectLoops =
    //     isProducer ? metadata.dataConnects_.at(di.dataConnect_).consumers_
    //                : metadata.dataConnects_.at(di.dataConnect_).producers_;
    auto *allocOwnerLoop = allocation->getOwnerLoop();
    const dsc2::LoopNode *indAllocOwnerLoop =
        hasIndirect ? indAllocation->getOwnerLoop() : nullptr;
    while (loopPtr != allocOwnerLoop) {
      if (loopPtr->prev_ == nullptr) {  // this is the root node
        DT_ERROR("Cannot find allocation in parent loops: " + di.dataConnect_);
        DT_ERROR("Cannot find allocation in parent loops: " + di.dataConnect_);
      }
      if (loopPtr == indAllocOwnerLoop) reachIndAllocOwnerLoop |= true;
      if (belowChunkLimit && !loopsBelowChunkBoundary.count(loopPtr)) {
        belowChunkLimit = false;
      }
      for (auto &[dim, kind] : loopPtr->dims_) {
        PadType allocPadding = allocation->padding_.getPadding(dim);
        bool relevant = is_any_of(dim, allocation->layoutDimOrder_);
        PrimaryDimTypes relatedPadDim = PrimaryDimTypesCount;
        if (!relevant && !loopPtr->isParametricLoop()) {
          for (auto &[padDim, padInfo] :
               currDsc->dataStageParam_.at(loopPtr->denId_).ss_.paddingSizes_) {
            if (padInfo.windowDim_ == dim &&
                allocation->padding_.getPadding(padDim) != PadType::NOPAD) {
              // Accessing a padded dim using the corresponding window dim
              // Use case: Iterating within a single window.
              relevant = true;
              allocPadding = allocation->padding_.getPadding(padDim);
              relatedPadDim = padDim;
              break;
            }
          }
        }
        if (relevant) {
          int dimIdx = currDsc->getDimIndexInLayoutOrder(myLds.dsType_, dim);
          int scale = dimIdx < 0 ? 1 : myLds.scale_.at(dimIdx);
          if (scale > 0) {
            for (int clId = 0; clId < currDsc->numCoreletsUsed_DSC2_; clId++) {
              bool storeLoopEleOffs = true;
              bool isIndirectLoopEleOffs = false;
              int loopEleOffs = 0;
              if (loopPtr->isParametricLoop()) {
                loopEleOffs = loopPtr->parametricStride(currDsc);
              } else {
                double scaleDownFactor = 1.0;
                if (myLds.scaledLdsCategory_ ==
                    LabeledDsInfo::ScaledLdsCategory::SCALE_TENSOR) {
                  if (myLds.mxInfo_.dim == dim) {
                    scaleDownFactor = ((double)1.0) / myLds.mxInfo_.blkSize;
                  }
                }
                const auto &ds =
                    currDsc->dataStageParam_.at(loopPtr->denId_).ss_;
                loopEleOffs = ds.dataStageDimToVal_compView_st(
                    dim, loc.unit_, belowChunkLimit ? clId : -1, {},
                    scaleDownFactor);

                if (allocPadding == PadType::PADDED_NOZEROPAD) {
                  // - Index along padded dimension in a parametric loop that
                  // accesses the valid region only.
                  //   Offset: index(dim_padded) * data_stage_size(dim_padded)
                  //   No update is needed to loopEleOffs that was computed
                  //   above.
                  if (kind != MetaDimKind::PadValid) {
                    if (is_any_of(kind, MetaDimKind::Unpadded,
                                  MetaDimKind::WindowDim) &&
                        allowUnpaddedIndexingAtPaddedNoZeroPad) {
                      if (ds.paddingSizes_.count(dim)) {
                        // - Index is into the unpadded dimension (i or j) of
                        // the
                        //   result of a window-based operation.
                        //   Offset: index(dim_unpadded) *
                        //           data_stage_size(dim_unpadded) * stride(dim)
                        DT_CHECK(ds.paddingSizes_.at(dim).windowDim_ !=
                                     PrimaryDimTypes::PrimaryDimTypesCount &&
                                 "Expect a window-based dimension.");
                        loopEleOffs *= ds.paddingSizes_.at(dim).stride_;
                      }
                    } else {
                      DT_ERROR(
                          "Unsupported access into a dimension that is in "
                          "padded_nozeropad form.");
                    }
                  }
                } else if (is_any_of(allocPadding, PadType::PADDED_WZEROPAD,
                                     PadType::PADDED_FULLSPAN,
                                     PadType::PADDED_FULLSPAN_WUNNEEDED)) {
                  if (kind == MetaDimKind::Padded) {
                    // - Index into the padded dimension (direct indexing into
                    // the padded dimension)
                    //   Offset: index(dim_padded) * data_stage_size(dim_padded)
                    //   No update is needed to loopEleOffs that was computed
                    //   above.

                    loopEleOffs = ds.dataStageDimToVal_compView_st(
                        dim, loc.unit_, belowChunkLimit ? clId : -1,
                        allocation->padding_, scaleDownFactor);
                  } else if (kind == MetaDimKind::PadValid) {
                    // - Index into the padded dimension that access only the
                    // valid part.
                    //   Example: Transfer from Padded_nozeropad to
                    //   Padded_wzeropad where a dimension of a parametric loop
                    //   that index into the Padded_nozeropad version. Offset:
                    //   index(dim) * data_stage_size(dim) + Zpf
                    //           The offset Zpf will be added by a later stage
                    //           after this loop processing completes.
                    loopEleOffs = ds.dataStageDimToVal_compView_st(
                        dim, loc.unit_, belowChunkLimit ? clId : -1,
                        allocation->padding_, scaleDownFactor);
                  } else if (ds.paddingSizes_.count(dim)) {
                    // - Index is into the unpadded dimension (i or j) of the
                    //   result of a window-based operation.
                    //   Offset: index(dim_unpadded) *
                    //           data_stage_size(dim_unpadded) * stride(dim)
                    loopEleOffs *= ds.paddingSizes_.at(dim).stride_;
                  } else if (relatedPadDim != PrimaryDimTypesCount) {
                    // - Index into a window dimension (iteration along ki or kj
                    //   inside a window)
                    //   Offset: index(dim_window) * data_stage_size(dim_window)
                    //   * dilation
                    loopEleOffs *= ds.paddingSizes_.at(relatedPadDim).dilation_;
                  }
                } else if (allocPadding == PadType::LOWERED_PADDED) {
                  // - Index into an unpadded dimension (i or j) of the result
                  // of
                  //   a window-based operation.
                  //   Offset: index(dim_unpadded) * window_size(dim_unpadded) *
                  //   size(dim_unpadded)
                  if (ds.paddingSizes_.count(dim)) {
                    if (kind == MetaDimKind::Unpadded) {
                      loopEleOffs *= ds.primaryDimToVal_st(
                          ds.paddingSizes_.at(dim).windowDim_);
                    } else {
                      DT_ERROR("Unsupported access using dimension " +
                               EnumsConversion::primaryDimToString.at(dim) +
                               " into a dimension that is in " +
                               "lowered_padded form.");
                    }
                  } else if (relatedPadDim != PrimaryDimTypesCount) {
                    // - Index into a window dimension (iteration along ki or kj
                    //   inside a window).
                    //   Offset: index(dim_window) * data_stage_size(dim_window)
                    //   * dilation
                    loopEleOffs *= ds.paddingSizes_.at(relatedPadDim).dilation_;
                  }
                } else {
                  // Padding style is NOPAD. Only unpadded dimension-indices (i
                  // or j) can be used to access the storage along the current
                  // dimension.
                  //   Offset: index(dim) * size(dim)
                  // This offset value is computed above.

                  // We may scale loopEleOffs for a paged dimension.
                  if (pageSizes.count(dim)) {
                    DT_CHECK_MSG(
                        allocPadding == PadType::NOPAD,
                        "Do not expect a dimension is both paged and padded.");
                    const auto pageSize = pageSizes.at(dim);
                    auto scaleLoopEleOffs = [&pageSize](const int loopEleOffs) {
                      // Scale loopEleOffs by the page size.
                      return static_cast<int>(
                          std::floor(float(loopEleOffs) / pageSize));
                    };
                    if (isIndexLds(myLds))
                      loopEleOffs = scaleLoopEleOffs(loopEleOffs);
                    else if (isPagedLds(myLds)) {
                      if (loopEleOffs >= pageSize) {
                        if (reachIndAllocOwnerLoop)
                          storeLoopEleOffs = false;
                        else {
                          isIndirectLoopEleOffs = true;
                          loopEleOffs = scaleLoopEleOffs(loopEleOffs);
                        }
                      }
                    } else
                      DT_ERROR("Unhandled indirect alloc type");
                  }
                }
              }

              // Store loopEleOffs in DataInfo.
              if (storeLoopEleOffs) {
                if (isIndirectLoopEleOffs) {
                  DT_CHECK_MSG(indirectDi, "Expect a valid indirect DataInfo");
                  indirectDi->loopEleOffsets_[clId][loopPtr][dim] = loopEleOffs;
                } else
                  di.loopEleOffsets_[clId][loopPtr][dim] = loopEleOffs;
              }
            }
          }
        }
      }
      loopPtr = loopPtr->getOwnerLoop();
    }

    // Compute constant offsets
    loopPtr = loopLocation;
    while (loopPtr != allocOwnerLoop) {
      if (loopPtr->prev_ == nullptr) {  // this is the root node
        DT_ERROR("Cannot find allocation in parent loops: " + di.dataConnect_);
        DT_ERROR("Cannot find allocation in parent loops: " + di.dataConnect_);
      }
      for (auto &[dim, kind] : loopPtr->dims_) {
        if (is_any_of(dim, allocation->layoutDimOrder_)) {
          int dimIdx = currDsc->getDimIndexInLayoutOrder(myLds.dsType_, dim);
          int scale = dimIdx < 0 ? 1 : myLds.scale_.at(dimIdx);
          if (scale > 0) {
            PadType allocPadding = allocation->padding_.getPadding(dim);
            const auto &ds =
                currDsc->dataStageParam_
                    .at(loopPtr->isParametricLoop() ? metadata.core_dstgid
                                                    : loopPtr->denId_)
                    .ss_;
            for (auto coreId : currDsc->coreIdsUsed_) {
              for (int clId = 0; clId < currDsc->numCoreletsUsed_DSC2_;
                   clId++) {
                if (is_any_of(allocPadding, PadType::PADDED_WZEROPAD,
                              PadType::PADDED_FULLSPAN,
                              PadType::PADDED_FULLSPAN_WUNNEEDED) &&
                    kind == MetaDimKind::PadValid) {
                  // - Index into the padded dimension that access only the
                  // valid
                  //   part.
                  //   Example: Transfer from Padded_nozeropad to
                  //   Padded_wzeropad where a dimension of a parametric loop
                  //   that index into the Padded_nozeropad version. Offset:
                  //   index(dim) * data_stage_size(dim) + Zpf
                  //           Add the Zpf part to constEleOffs.
                  di.constEleOffsets_[coreId][clId][dim] =
                      ds.paddingSizes_.at(dim).padFront_;
                  if (ds.paddingSizes_.at(dim).padFront_ < 0) {
                    DT_ERROR("Requested pad type in loop " + loopPtr->name_ +
                             " is not legal");
                    DT_ERROR("Requested pad type in loop " + loopPtr->name_ +
                             " is not legal");
                  }
                } else if (is_any_of(allocPadding, PadType::PADDED_WZEROPAD,
                                     PadType::PADDED_FULLSPAN,
                                     PadType::PADDED_FULLSPAN_WUNNEEDED) &&
                           kind == MetaDimKind::PadBack) {
                  // - Index into the padded dimension that access only the
                  //   padding part at end.
                  //   Offset: index(dim) * data_stage_size(dim) + Zpf +
                  //   padValid
                  //           Add the (Zpf + padValid) part to constEleOffs.
                  di.constEleOffsets_[coreId][clId][dim] =
                      ds.primaryDimToVal_st(dim, SenComponents::NO_COMPONENT,
                                            -1, -1, allocation->padding_) -
                      ds.paddingSizes_.at(dim).padBack_;
                  if (ds.paddingSizes_.at(dim).padBack_ < 0) {
                    DT_ERROR("Requested pad type in loop " + loopPtr->name_ +
                             " is not legal");
                    DT_ERROR("Requested pad type in loop " + loopPtr->name_ +
                             " is not legal");
                  }
                }
              }
            }
          }
        }
      }
      loopPtr = loopPtr->getOwnerLoop();
    }

    if (allocation->numBuffers_ != 1) {
      DT_CHECK_MSG(loopPtr->getOwnerLoop() != nullptr,
                   "Do not expect the root node.");
      // fill buffer switching info
      di.bufferSwitchPosition_ = loopPtr;
      di.bufferAddrOffset_ = allocation->bufferOffsetCoreCorelet_;
      for (auto &corePair : di.bufferAddrOffset_) {
        for (auto &clPair : corePair.second) {
          clPair.second /= addrScale;
          if (genericUnit == L0LU) clPair.second /= dscGlobal.sysDef.numPTRows;
        }
      }
    }
  };

  for (auto *node : scheduleTreeTraverse) {
    if (metadata.externalNodes_.count(node)) continue;

    auto *ownerLoop = node->getOwnerLoop();
    if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      auto *transfer = static_cast<dsc2::TransferNode *>(node);
      auto findLastFusableLoop = [&](SenComponents unit) {
        const dsc2::LoopNode *lastFusableLoop = nullptr;
        if (transfer->isNodeRelevant(unit)) {
          auto *parent = transfer->getPrev();
          while (parent && parent->getPrev() &&
                 parent->getNextView(unit).size() == 1) {
            if (parent->nodeType_ == dsc2::ScheduleNode::LOOP)
              lastFusableLoop = static_cast<const dsc2::LoopNode *>(parent);
            parent = parent->getPrev();
          }
        }
        return lastFusableLoop;
      };

      if (transfer->src_.unit_ == NO_COMPONENT &&
          transfer->paddingInfo_.isEmpty())
        continue;

      dsc2::DataInfo *srcIndirectLdsAndLoopOffsets =
          transfer->isSrcIndirect() ? &transfer->srcIndirectLdsAndLoopOffsets_
                                    : nullptr;
      DataLocation *srcIndirect =
          transfer->isSrcIndirect() ? &transfer->srcIndirect_ : nullptr;
      fillDataInfo(transfer->srcLdsAndLoopOffsets_,
                   srcIndirectLdsAndLoopOffsets, transfer->src_, srcIndirect,
                   ownerLoop, false);
      transfer->lastFusableParentLoopSrc_ =
          findLastFusableLoop(transfer->src_.unit_);
      if (transfer->dstLdsAndLoopOffsets_.size() != transfer->dstVias_.size()) {
        DT_ERROR("Transfer node destinations missing information");
        DT_ERROR("Transfer node destinations missing information");
      }
      transfer->lastFusableParentLoopDst_.clear();
      for (int i = 0; i < transfer->dstVias_.size(); i++) {
        const bool isIndirectDst = transfer->isDstIndirectAtIndex(i);
        dsc2::DataInfo *dstIndirectLdsAndLoopOffsets =
            isIndirectDst ? &transfer->dstIndirectLdsAndLoopOffsets_[i]
                          : nullptr;
        DataLocation *dstIndirect =
            isIndirectDst ? &transfer->dstVias_[i].locIndirect_ : nullptr;
        fillDataInfo(transfer->dstLdsAndLoopOffsets_[i],
                     dstIndirectLdsAndLoopOffsets, transfer->dstVias_[i].loc_,
                     dstIndirect, ownerLoop, true);
        transfer->lastFusableParentLoopDst_.push_back(
            findLastFusableLoop(transfer->dstVias_[i].loc_.unit_));
      }
      if (auto tnMetaIt = metadata.datatransfers_.find(transfer);
          tnMetaIt != metadata.datatransfers_.end() &&
          tnMetaIt->second.apply_row_offset_) {
        for (auto coreId : currDsc->coreIdsUsed_) {
          for (int clId = 0; clId < currDsc->numCoreletsUsed_DSC2_; clId++) {
            transfer->srcLdsAndLoopOffsets_
                .constEleOffsets_[coreId][clId][metadata.rowSplitDim] =
                currDsc->getBlockTransferSizePerDim(
                    *transfer, transfer->src_.unit_,
                    clId)[metadata.rowSplitDim] *
                EnumsConversion::senCompToRowId.at(
                    transfer->dstVias_.at(0).loc_.unit_);
          }
        }
      }
    } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      auto *compute = static_cast<dsc2::ComputeNode *>(node);
      if (compute->inputsLdsAndLoopOffsets_.size() != compute->inputs_.size() ||
          compute->outputsLdsAndLoopOffsets_.size() !=
              compute->outputs_.size()) {
        DT_ERROR("Compute node input/output missing information");
        DT_ERROR("Compute node input/output missing information");
      }
      for (int i = 0; i < compute->inputs_.size(); i++) {
        fillDataInfo(compute->inputsLdsAndLoopOffsets_[i], nullptr,
                     {compute->exUnit_, compute->inputs_[i]}, nullptr,
                     ownerLoop, false);
      }
      for (int i = 0; i < compute->outputs_.size(); i++) {
        fillDataInfo(compute->outputsLdsAndLoopOffsets_[i], nullptr,
                     {compute->exUnit_, compute->outputs_[i]}, nullptr,
                     ownerLoop, true);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 334/382   level 3   scc 123   47 body lines
// unit: e334_addIbrDataStage
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6626
// original: void L3DlOpsScheduler::addIbrDataStage( SuperDsc& mySDsc, DesignSpaceConfig& dsc, const std::vector<PrimaryDimTypes>& pagedDims)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e334_addIbrDataStage(
    SuperDsc& mySDsc, DesignSpaceConfig& dsc,
    const std::vector<PrimaryDimTypes>& pagedDims)
{
  // Get a new datastage index if not exists.
  if (dataStageIbrIdx == -1)
    dataStageIbrIdx = getNewDataStageIndex(mySDsc, dsc);
  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
               "Expect the core datastage available.");
  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageOnePageIdx),
               "Expect the OnePage datastage available.");
  const auto &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);
  const auto &dsOnePage = dsc.dataStageParam_.at(dataStageOnePageIdx);
  std::vector<int> indexLdsIndices;
  for (const auto &lds : dsc.labeledDs_) {
    if (isIndexLds(lds)) indexLdsIndices.push_back(lds.ldsIdx_);
  }
  DT_CHECK_MSG(indexLdsIndices.size() == 1, "Support only one index tensor.");
  const auto &indexLds = dsc.labeledDs_[indexLdsIndices.front()];
  const auto indexLdsStickSizes = dsc.getCumulativeStickSizes(indexLds.dsType_);
  // This datastage could have epilogue (el_) when coreD is greater than one
  // stick (in index tensor) of data but not a multiple of one stick (in
  // index tensor) of data.
  DataStructDims ssParams, elParams;
  for (const auto dim : pagedDims) {
    // The max number of pages in IBR is the stick size of this dimension in the
    // index tensor if it is stick dimension, or one otherwise.
    const int maxNumPagesInIbr =
        indexLdsStickSizes.count(dim) ? indexLdsStickSizes.at(dim) : 1;
    const int pageSize = dsOnePage.ss_.primaryDimToVal_st(dim);
    DT_CHECK(dsCore.ss_.primaryDimToVal_st(dim) % pageSize == 0);
    const int numPagesInCore = dsCore.ss_.primaryDimToVal_st(dim) / pageSize;
    auto ssNumPages = std::min(numPagesInCore, maxNumPagesInIbr);
    ssParams.primaryDimToValHandler_st(dim) = ssNumPages * pageSize;
    auto elNumPages = (numPagesInCore % ssNumPages > 0)
                          ? numPagesInCore % ssNumPages
                          : ssNumPages;
    elParams.primaryDimToValHandler_st(dim) = elNumPages * pageSize;
  }
  ssParams.name_ = "ibr";
  elParams.name_ = "ibr";
  addOrUpdateCoreletSplitInParams(ssParams, dsc);
  addOrUpdateCoreletSplitInParams(elParams, dsc);
  addOrUpdateSymbolicInfoInParams(ssParams, dsCore.ss_);
  addOrUpdateSymbolicInfoInParams(elParams, dsCore.el_);
  dsc2::DataStage ds;
  ds.ss_ = ssParams;
  ds.el_ = elParams;
  dsc.dataStageParam_[dataStageIbrIdx] = ds;
}

// ------------------------------------------------------------------------------------------------
// entry 335/382   level 3   scc 125   30 body lines
// unit: e335_addOnePageDataStage
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6676
// original: void L3DlOpsScheduler::addOnePageDataStage( SuperDsc& mySDsc, DesignSpaceConfig& dsc, const std::vector<PrimaryDimTypes>& pagedDims)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e335_addOnePageDataStage(
    SuperDsc& mySDsc, DesignSpaceConfig& dsc,
    const std::vector<PrimaryDimTypes>& pagedDims)
{
  // Get a new datastage index if not exists.
  if (dataStageOnePageIdx == -1)
    dataStageOnePageIdx = getNewDataStageIndex(mySDsc, dsc);
  DataStructDims params;
  const auto allPagedLdsIndices = getAllPagedLdsIndices(dsc);
  for (const auto dim : pagedDims) {
    bool visitedDim = false;
    for (const auto ldsIdx : allPagedLdsIndices) {
      const auto &lds = dsc.labeledDs_.at(ldsIdx);
      DT_CHECK_MSG(lds.memOrg_.count(SenComponents::HBM),
                   "Exepect HBM in memOrg_.");
      const auto allocNode = lds.memOrg_.at(SenComponents::HBM).allocateNode_;
      DT_CHECK_MSG(allocNode, "Expect a valid HBM allocate node.");
      const auto pageSizes = allocNode->getPageSize();
      DT_CHECK_MSG(pageSizes.count(dim), "Expect paged dim in paged tensor.");
      const auto pageSize = pageSizes.at(dim);
      DT_CHECK_MSG(!(visitedDim && pageSize != params.primaryDimToVal_st(dim)),
                   "Page size does not match for this dimension.");
      params.primaryDimToValHandler_st(dim) = pageSize;
      visitedDim |= true;
    }
  }
  params.name_ = "1page";
  addOrUpdateCoreletSplitInParams(params, dsc);
  dsc2::DataStage ds;
  ds.ss_ = params;
  ds.el_ = params;
  dsc.dataStageParam_[dataStageOnePageIdx] = ds;
}

// ------------------------------------------------------------------------------------------------
// entry 336/382   level 3   scc 130   71 body lines
// unit: e336_processPagedTensorTransfers
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6870
// original: void L3DlOpsScheduler::processPagedTensorTransfers( SuperDsc &mySDsc, const int dscIdx, const std::vector<dsc2::TransferNode *> &l3TransferNodesAllTensors, const std::vector<dsc2::AllocateNode *> &pagedLdsHbmAllocNodes, const std::unordered_map<PrimaryDimTypes, dsc2::LoopNode *> &newPagedDimChunkLoopNodes, PrimaryDimTypes innerIndexDimInChunkLoops)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e336_processPagedTensorTransfers(
    SuperDsc &mySDsc, const int dscIdx,
    const std::vector<dsc2::TransferNode *> &l3TransferNodesAllTensors,
    const std::vector<dsc2::AllocateNode *> &pagedLdsHbmAllocNodes,
    const std::unordered_map<PrimaryDimTypes, dsc2::LoopNode *>
        &newPagedDimChunkLoopNodes,
    PrimaryDimTypes innerIndexDimInChunkLoops)
{
  DT_CHECK_MSG(pagedLdsHbmAllocNodes.size() <= 1,
               "Support no more than one paged tensor for now.");
  auto &dsc = mySDsc.dscs_.at(dscIdx);
  for (const auto &pagedLdsHbmAllocNode : pagedLdsHbmAllocNodes) {
    // Find its index tensor.
    const auto pagedLdsIdx = pagedLdsHbmAllocNode->ldsIdx_;
    DT_CHECK_MSG(pagedLdsIdx >= 0, "Expect a valid paged tensor ldsIdx.");
    auto &pagedLds = dsc.labeledDs_.at(pagedLdsIdx);
    const auto indexLdsHbmAllocNode =
        pagedLdsHbmAllocNode->relatedIndirectAccessAlloc_;
    DT_CHECK_MSG(indexLdsHbmAllocNode, "Expect a valid HBM allocate node.");
    const int indexLdsIdx = indexLdsHbmAllocNode->ldsIdx_;
    DT_CHECK_MSG(indexLdsIdx >= 0, "Expect a valid index tensor.");
    auto &indexLds = dsc.labeledDs_.at(indexLdsIdx);

    // Find the paged tensor's HBM<->LX transfer nodes.
    std::vector<dsc2::TransferNode *> pagedLdsL3TransNodes;
    for (auto &transNode : l3TransferNodesAllTensors) {
      if (transNode->srcLdsAndLoopOffsets_.myLdsIdx_ == pagedLdsIdx)
        pagedLdsL3TransNodes.push_back(transNode);
    }
    DT_CHECK_MSG(
        !pagedLdsL3TransNodes.empty() && pagedLdsL3TransNodes.size() <= 2,
        "Expect a HBM->LX and/or a LX->HBM transfer node.");

    // Convert its transfer node to a chunk/1page loop containing the original
    // transfer node with indirection info.
    for (auto transNode : pagedLdsL3TransNodes) {
      // Transfer-in is HBM->LX transfer. Otherwise, it is LX->HBM transfer.
      const bool isTransferIn = transNode->src_.storage_ == SenComponents::HBM;

      // TODO: Support multiple stick dimensions.
      const auto indexStickDims = dsc.getStickDims(indexLdsIdx);
      DT_CHECK_MSG(indexStickDims.size() == 1,
                   "Support only one stick dimension for now.");
      PrimaryDimTypes indexStickDim = indexStickDims.front();
      // The dim in the stick of index tensor, must be the innermost in the
      // chunk loop order So the we consume the IBR stick fully (and not
      // reload), before moving the next. 2 pros: a) more performant, b) we
      // dont have ability to offset start addr above ibr allocation
      DT_CHECK_MSG(
          innerIndexDimInChunkLoops == indexStickDim,
          "Expect index stick dim to be innermost in chunk loop order");

      // The IBR allocation happens at the innermost (new) chunk loop related
      // to the index tensor. Add the new allocate, transfer and sync nodes
      // before the loop node as siblings.
      DT_CHECK_MSG(newPagedDimChunkLoopNodes.count(innerIndexDimInChunkLoops),
                   "Expect the new chunk loop for the current dimension.");
      auto newChunkLoopNode =
          newPagedDimChunkLoopNodes.at(innerIndexDimInChunkLoops);

      // Create the store of the index tensor to LX if it is LX->HBM transfer
      // for the paged tensor.
      if (!isTransferIn) {
        createStoreIndexTensorToLx(mySDsc, dscIdx, pagedLdsIdx, indexLdsIdx,
                                   *indexLdsHbmAllocNode, *newChunkLoopNode);
      }

      // Create the store of the index tensor to IBR register.
      createStoreIndexTensorToIbr(dsc, dscIdx, indexLdsIdx,
                                  *indexLdsHbmAllocNode, *newChunkLoopNode,
                                  isTransferIn);

      // Convert transfer node from direct to indirect.
      convertTransferDirectToIndirect(*transNode, dsc, indexLdsIdx,
                                      indexStickDim, isTransferIn);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 350/382   level 4   scc 50   26 body lines
// unit: e350_getMinParamConv2d
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:896
// original: long L3DlOpsScheduler::getMinParamConv2d(const DesignSpaceConfig& dsc, const PrimaryDimTypes dim, const OpFuncs opFuncName) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
long e350_getMinParamConv2d(const DesignSpaceConfig& dsc,
                                         const PrimaryDimTypes dim,
                                         const OpFuncs opFuncName) const
{
  constexpr long defaultParam = 1;
  const auto coreParam =
      dsc.dataStageParam_.at(dataStageCoreIdx).ss_.primaryDimToVal_st(dim);

  switch (dim) {
    case PrimaryDimTypes::IN: {
      if (isOpFuncConv2dInt4(opFuncName))
        return 128;
      else if (isOpFuncConv2dOs1(opFuncName))
        return coreParam;
      else
        return 64;
    }
    case PrimaryDimTypes::OUT:
      return 64;
    case PrimaryDimTypes::I:
      return computeMinParamForPaddedDim(dsc, dim);
    case PrimaryDimTypes::J:
    case PrimaryDimTypes::KI:
    case PrimaryDimTypes::KJ:
      return coreParam;
    default:
      return defaultParam;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 351/382   level 4   scc 86   12 body lines
// unit: e351_setSuperChunkDataStageParams
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2793
// original: void L3DlOpsScheduler::setSuperChunkDataStageParams(SuperDsc& mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e351_setSuperChunkDataStageParams(SuperDsc& mySDsc)
{
  if (lxBufferType != BufferType::SPATIAL_DOUBLE) return;

  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    auto& dsc = mySDsc.dscs_.at(dscIdx);
    addSuperChunkDataStage(dsc);
    exploreSuperChunkDataStageParams(mySDsc, dscIdx);
    auto& dsSuperChunk = dsc.dataStageParam_.at(dataStageSuperChunkIdx);
    addOrUpdateCoreletSplitInParams(dsSuperChunk.ss_, dsc);
    addOrUpdateCoreletSplitInParams(dsSuperChunk.el_, dsc);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 352/382   level 4   scc 67   5 body lines
// unit: e352_updateChunkDataStagesFromCandidates
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2819
// original: void L3DlOpsScheduler::updateChunkDataStagesFromCandidates( DataStructDims &chunkParams, DesignSpaceConfig &dsc, const int dscIdx, const DscParamCandidateIndicesType &selectedIndices, const DscParamCandidatesType &dscCandidates, const std::vector<PrimaryDimTypes> &primaryDims)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e352_updateChunkDataStagesFromCandidates(
    DataStructDims &chunkParams, DesignSpaceConfig &dsc, const int dscIdx,
    const DscParamCandidateIndicesType &selectedIndices,
    const DscParamCandidatesType &dscCandidates,
    const std::vector<PrimaryDimTypes> &primaryDims)
{
  addChunkDataStageFromCandidates(chunkParams, dsc, dscIdx, selectedIndices,
                                  dscCandidates, primaryDims);
  if (lxBufferType == BufferType::SPATIAL_DOUBLE) addSuperChunkDataStage(dsc);
}

// ------------------------------------------------------------------------------------------------
// entry 353/382   level 4   scc 87   254 body lines
// unit: e353_createAllocationAndTransfer
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:3121
// original: void L3DlOpsScheduler::createAllocationAndTransfer(SuperDsc &mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e353_createAllocationAndTransfer(SuperDsc &mySDsc)
{
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    auto &dsc = mySDsc.dscs_.at(dscIdx);

    // Find the lx-below block node in schedule tree. This node should be inside
    // the innermost chunk loop. And we use it as the starting point to find the
    // insertion point for the allocation and transfer of each tensor.
    dsc2::BlockNode *lxBelowBlockNode = getLxBelowBlockNode(dsc.scheduleTree_);
    DT_CHECK_MSG(lxBelowBlockNode != nullptr,
                 "Expect lx_below_schedule block node.");
    // Collect parent loop nodes from inner to outer loops.
    const auto parentInnerToOuterLoopNodes =
        getParentLoopNodes(*lxBelowBlockNode, dsc);

    // Collect all value tensors to be analyzed. Skip index tensors.
    std::vector<int> allValueLdsIndices;
    for (const auto &lds : dsc.labeledDs_) {
      if (!isIndexLds(lds)) allValueLdsIndices.push_back(lds.ldsIdx_);
    }

    // Create allocate nodes and transfer nodes by looking at each tensor. Only
    // process tensors reside in HBM, LX-neighbor or LX-local.
    for (auto &ldsIdx : allValueLdsIndices) {
      LabeledDsInfo &lds = dsc.labeledDs_.at(ldsIdx);
      const bool isLdsOutput = isOutputLabeledDs(ldsIdx, dsc);

      if (lds.isHbmPinned()) {
        DT_CHECK_MSG(lds.memOrg_.count(SenComponents::HBM),
                     "Expect HBM in memOrg_.");
        dsc2::AllocateNode *allocHbmNode =
            lds.memOrg_.at(SenComponents::HBM).allocateNode_;
        DT_CHECK_MSG(allocHbmNode, "Expect a valid HBM allocate node.");

        // Determine the sibling loop node for allocate node insertion.
        auto allocSiblingLoopNode = computeLdsAllocateSiblingLoopNode(
            mySDsc, dscIdx, lds, lxBelowBlockNode, parentInnerToOuterLoopNodes);
        DT_CHECK_MSG(allocSiblingLoopNode, "Expect a valid node.");

        const bool isOutsideOutermostLoop =
            allocSiblingLoopNode->getOwnerLoop() == dsc.scheduleTree_.getHead();
        const int numBuffers = isOutsideOutermostLoop ? 1 : 2;
        const SenComponents allocStorage = SenComponents::LX;
        const std::string allocNodeName =
            "allocate_lds" + std::to_string(lds.ldsIdx_) + "_" +
            EnumsConversion::senComponentsToString.at(allocStorage);
        dsc2::AllocateNode* allocNode = createAllocateNode(
            dsc, ldsIdx, allocStorage, numBuffers, allocNodeName, dscIdx);
        auto& lxMemOrg = lds.memOrg_[SenComponents::LX];
        lxMemOrg.isPresent = true;
        lxMemOrg.allocateNode_ = allocNode;

        dsc2::BlockNode* allocParentNode =
            allocSiblingLoopNode->getMutableParent();
        // Insert allocate node before ref node.
        allocParentNode->addChildNode(allocNode, /*addBefore*/ true,
                                      allocSiblingLoopNode);

        // Determine the sibling loop node for transfer node insertion.
        auto transSiblingLoopNode = computeLdsTransferSiblingLoopNode(
            mySDsc, dscIdx, lds, lxBelowBlockNode, parentInnerToOuterLoopNodes);
        DT_CHECK_MSG(transSiblingLoopNode, "Expect a valid node.");
        if (transSiblingLoopNode->nodeType_ == dsc2::ScheduleNode::LOOP) {
          const auto loopTypeSiblingNode =
              static_cast<dsc2::LoopNode*>(transSiblingLoopNode);
          DT_CHECK_MSG(loopTypeSiblingNode->denId_ == dataStageChunkIdx,
                       "Expect loop denominator to be chunk.");
        }
        dsc2::BlockNode* transParentNode =
            transSiblingLoopNode->getMutableParent();

        // Check if the transfer is for cross-core reduction scenario.
        const bool isTransferCrossCoreReduction =
            isLdsOutput && isOpCrossCoreReduction(mySDsc, dsc);

        const SenComponents transInSrcStorage = SenComponents::HBM;
        const SenComponents transInDstStorage = SenComponents::LX;
        std::string transInNodeName =
            "transfer_lds" + std::to_string(lds.ldsIdx_) + "_src:" +
            EnumsConversion::senComponentsToString.at(transInSrcStorage) +
            "_dst:" +
            EnumsConversion::senComponentsToString.at(transInDstStorage);
        dsc2::TransferNode* transInNode = createTransferNode(
            SenComponents::L3LU, transInSrcStorage, {SenComponents::L3LU},
            {transInDstStorage}, ldsIdx, {ldsIdx}, transInNodeName);
        allocNode->addAllocUser(transInNode);
        allocHbmNode->addAllocUser(transInNode);

        if (isTransferCrossCoreReduction) {
          // Create a core condition block enclosing the transferIn node and
          // insert it before sibling loop node.
          const auto transferCoreIds = getLdsTransferCoreIds(mySDsc, dsc, lds);
          auto ifNode = new dsc2::ConditionNode();
          std::string ifNodeName =
              "condition_separate_" + transInNode->name_ + "_core";
          for (const auto coreId : transferCoreIds)
            ifNodeName += "_" + std::to_string(coreId);
          ifNode->name_ = ifNodeName;
          transParentNode->addChildNode(ifNode, /*addBefore*/ true,
                                        transSiblingLoopNode);
          for (const auto coreId : transferCoreIds) {
            for (int coreletId = 0; coreletId < dsc.numCoreletsUsed_DSC2_;
                 ++coreletId)
              ifNode->coreClCond_[coreId].insert(coreletId);
          }

          auto thenBlockNode = new dsc2::BlockNode();
          thenBlockNode->name_ = ifNode->name_ + "_then_region";
          ifNode->addThenRegion(thenBlockNode);
          thenBlockNode->addChildNode(transInNode);
        } else
          // Insert transferIn node before sibling loop node.
          transParentNode->addChildNode(transInNode, /*addBefore*/ true,
                                        transSiblingLoopNode);

        // Output tensor is the last lds in labeledDs_. Insert transferOut node
        // after the sibling loop node.
        if (isLdsOutput) {
          const SenComponents transOutSrcStorage = SenComponents::LX;
          const SenComponents transOutDstStorage = SenComponents::HBM;
          std::string transOutNodeName =
              "transfer_lds" + std::to_string(lds.ldsIdx_) + "_src:" +
              EnumsConversion::senComponentsToString.at(transOutSrcStorage) +
              "_dst:" +
              EnumsConversion::senComponentsToString.at(transOutDstStorage);
          dsc2::TransferNode* transOutNode = createTransferNode(
              SenComponents::L3SU, transOutSrcStorage, {SenComponents::L3SU},
              {transOutDstStorage}, ldsIdx, {ldsIdx}, transOutNodeName);
          allocNode->addAllocUser(transOutNode);
          allocHbmNode->addAllocUser(transOutNode);

          if (isTransferCrossCoreReduction) {
            // Create a core condition block enclosing the transferOut node and
            // insert it after sibling loop node.
            const auto transferCoreIds =
                getLdsTransferCoreIds(mySDsc, dsc, lds);
            auto ifNode = new dsc2::ConditionNode();
            std::string ifNodeName =
                "condition_separate_" + transOutNode->name_ + "_core";
            for (const auto coreId : transferCoreIds)
              ifNodeName += "_" + std::to_string(coreId);
            ifNode->name_ = ifNodeName;
            transParentNode->addChildNode(ifNode, /*addBefore*/ false,
                                          transSiblingLoopNode);
            for (const auto coreId : transferCoreIds) {
              for (int coreletId = 0; coreletId < dsc.numCoreletsUsed_DSC2_;
                   ++coreletId)
                ifNode->coreClCond_[coreId].insert(coreletId);
            }

            auto thenBlockNode = new dsc2::BlockNode();
            thenBlockNode->name_ = ifNode->name_ + "_then_region";
            ifNode->addThenRegion(thenBlockNode);
            thenBlockNode->addChildNode(transOutNode);
          } else
            // Insert transferOut node after sibling loop node.
            transParentNode->addChildNode(transOutNode, /*addBefore*/ false,
                                          transSiblingLoopNode);
        }
      } else if (isLabeledDsLXNeighbor(mySDsc, dscIdx, lds)) {
        DT_CHECK_MSG(lds.dsType_ == INPUT,
                     "Invalid DsType for input-neighbor fetch.");
        DT_CHECK_MSG(
            ldsIdx == 0,
            "Expect input-neighbor-fetch tensor to be at labeledDs index 0.");
        const int numBufferForNoBuffer = 1;
        DT_CHECK_MSG(lds.memOrg_.count(SenComponents::LX),
                     "Expect memOrg_ LX entry.");
        auto& allocNode = lds.memOrg_.at(SenComponents::LX).allocateNode_;
        if (allocNode == nullptr) {
          const SenComponents allocStorage = SenComponents::LX;
          const std::string allocNodeName =
              "allocate_lds" + std::to_string(lds.ldsIdx_) + "_" +
              EnumsConversion::senComponentsToString.at(allocStorage);
          allocNode =
              createAllocateNode(dsc, ldsIdx, allocStorage,
                                 numBufferForNoBuffer, allocNodeName, dscIdx);
          // Insert allocate node at outermost level before the sibling loop
          // node.
          dsc2::LoopNode* outermostSiblingLoopNode =
              const_cast<dsc2::LoopNode*>(parentInnerToOuterLoopNodes.back());
          dsc2::BlockNode* outermostParentNode =
              outermostSiblingLoopNode->getMutableParent();
          outermostParentNode->addChildNode(allocNode, /*addBefore*/ true,
                                            outermostSiblingLoopNode);
        }

        std::string transInNodeName = "transfer_lds" +
                                      std::to_string(lds.ldsIdx_) + "_src:" +
                                      EnumsConversion::senComponentsToString.at(
                                          SenComponents::NO_COMPONENT) +
                                      "_dst:" +
                                      EnumsConversion::senComponentsToString.at(
                                          SenComponents::NO_COMPONENT) +
                                      "_lx_neighbor";
        // Create a dummy transfer node with src ldsIdx == -1.
        constexpr int srcLdsIdx = -1;
        dsc2::TransferNode* transInNode = createTransferNode(
            SenComponents::NO_COMPONENT, SenComponents::NO_COMPONENT,
            {SenComponents::NO_COMPONENT}, {SenComponents::LX}, srcLdsIdx,
            {ldsIdx}, transInNodeName);
        allocNode->addAllocUser(transInNode);

        // Insert transfer node at the innermost level before the lx-below block
        // node.
        dsc2::BlockNode* transRefNode = lxBelowBlockNode;
        dsc2::BlockNode* transParentNode = transRefNode->getMutableParent();
        transParentNode->addChildNode(transInNode, /*addBefore*/ true,
                                      transRefNode);
      } else if (lds.isLxPinned()) {
        const int numBufferForNoBuffer = 1;
        DT_CHECK_MSG(lds.memOrg_.count(SenComponents::LX),
                     "Expect memOrg_ LX entry.");
        dsc2::LoopNode* refNode =
            const_cast<dsc2::LoopNode*>(parentInnerToOuterLoopNodes.back());
        dsc2::BlockNode* parentNode = refNode->getMutableParent();

        auto& allocNode = lds.memOrg_.at(SenComponents::LX).allocateNode_;
        if (allocNode == nullptr) {
          const SenComponents allocStorage = SenComponents::LX;
          const std::string allocNodeName =
              "allocate_lds" + std::to_string(lds.ldsIdx_) + "_" +
              EnumsConversion::senComponentsToString.at(allocStorage);
          allocNode =
              createAllocateNode(dsc, ldsIdx, allocStorage,
                                 numBufferForNoBuffer, allocNodeName, dscIdx);
          // Insert allocate node at root level before the ref node.
          parentNode->addChildNode(allocNode, /*addBefore*/ true, refNode);
        }

        SenComponents srcStorage =
            isLdsOutput ? SenComponents::LX : SenComponents::NO_COMPONENT;
        SenComponents dstStorage =
            isLdsOutput ? SenComponents::NO_COMPONENT : SenComponents::LX;
        const int srcLdsIdx = isLdsOutput ? ldsIdx : -1;
        const int dstLdsIdx = isLdsOutput ? -1 : ldsIdx;
        std::string transNodeName =
            "transfer_lds" + std::to_string(lds.ldsIdx_) +
            "_src:" + EnumsConversion::senComponentsToString.at(srcStorage) +
            "_dst:" + EnumsConversion::senComponentsToString.at(dstStorage) +
            "_lx_local";
        dsc2::TransferNode* transNode =
            createTransferNode(SenComponents::NO_COMPONENT, srcStorage,
                               {SenComponents::NO_COMPONENT}, {dstStorage},
                               srcLdsIdx, {dstLdsIdx}, transNodeName);
        allocNode->addAllocUser(transNode);

        // Insert transfer node at root level before the ref node if it is an
        // input tensor. Otherwise, insert after the ref node.
        const bool addBefore = isLdsOutput ? false : true;
        parentNode->addChildNode(transNode, addBefore, refNode);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 354/382   level 4   scc 131   56 body lines
// unit: e354_processDscHbmPagedTensors
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6746
// original: void L3DlOpsScheduler::processDscHbmPagedTensors(SuperDsc &mySDsc, const int dscIdx)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e354_processDscHbmPagedTensors(SuperDsc &mySDsc,
                                                 const int dscIdx)
{
  auto &dsc = mySDsc.dscs_.at(dscIdx);

  // Return if there is no paged tensor.
  const auto pagedDims = getPagedDimensions(dsc);
  if (pagedDims.empty()) return;

  // Collect all chunk loop nodes, L3 transfer nodes and paged tensor HBM
  // allocate nodes.
  DT_CHECK_MSG(!dsc.scheduleTree_.empty(), "Expect valid schedule tree.");
  std::set<dsc2::LoopNode *> chunkLoopNodes;
  auto innerIndexDimInChunkLoops = PrimaryDimTypes::PrimaryDimTypesCount;
  std::vector<dsc2::TransferNode *> l3TransferNodesAllTensors;
  std::vector<dsc2::AllocateNode *> pagedLdsHbmAllocNodes;
  for (auto& node : dsc.scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::LOOP, dsc2::ScheduleNode::TRANSFER,
                     dsc2::ScheduleNode::ALLOCATE})) {
    auto loopNode = dynamic_cast<dsc2::LoopNode*>(node);
    if (loopNode && loopNode->numId_ == dataStageCoreIdx &&
        ((lxBufferType == BufferType::DOUBLE &&
          loopNode->denId_ == dataStageChunkIdx) ||
         (lxBufferType == BufferType::SPATIAL_DOUBLE &&
          loopNode->denId_ == dataStageSuperChunkIdx))) {
      chunkLoopNodes.insert(loopNode);
      DT_CHECK_MSG(loopNode->dims_.size() == 1,
                   "Only support one dimension in a chunk loop node for now.");
      const auto loopDim = loopNode->dims_.front().dim_;
      if (is_any_of(loopDim, pagedDims)) {
        innerIndexDimInChunkLoops = loopDim;
      }
    }
    auto transNode = dynamic_cast<dsc2::TransferNode *>(node);
    if (transNode && !transNode->dstVias_.empty() &&
        ((transNode->src_.storage_ == SenComponents::HBM &&
          transNode->dstVias_.front().loc_.storage_ == SenComponents::LX) ||
         (transNode->src_.storage_ == SenComponents::LX &&
          transNode->dstVias_.front().loc_.storage_ == SenComponents::HBM)))
      l3TransferNodesAllTensors.push_back(transNode);
    auto allocNode = dynamic_cast<dsc2::AllocateNode *>(node);
    if (allocNode &&
        allocNode->indirectAllocType_ ==
            dsc2::AllocateNode::IndirectAllocType::VALUE_TENSOR &&
        allocNode->component_ == SenComponents::HBM)
      pagedLdsHbmAllocNodes.push_back(allocNode);
  }
  DT_CHECK(innerIndexDimInChunkLoops != PrimaryDimTypes::PrimaryDimTypesCount);

  // Create new loop nodes for the paged dimensions.
  auto newPagedDimChunkLoopNodes =
      createPagedDimChunkLoops(dsc, pagedDims, chunkLoopNodes);

  // Process the transfers for each paged tensor.
  processPagedTensorTransfers(mySDsc, dscIdx, l3TransferNodesAllTensors,
                              pagedLdsHbmAllocNodes, newPagedDimChunkLoopNodes,
                              innerIndexDimInChunkLoops);
}

// ------------------------------------------------------------------------------------------------
// entry 355/382   level 4   scc 134   47 body lines
// unit: e355_fillCoordinateCustomWkSliceId
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7198
// original: void L3DlOpsScheduler::fillCoordinateCustomWkSliceId( SuperDsc &mySDsc, DesignSpaceConfig &dsc, const int ldsIdx, dsc2::CoordinateType<CoordinateBaseType> &coordinate) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e355_fillCoordinateCustomWkSliceId(
    SuperDsc &mySDsc, DesignSpaceConfig &dsc, const int ldsIdx,
    dsc2::CoordinateType<CoordinateBaseType> &coordinate) const
{
  const auto &lds = dsc.labeledDs_.at(ldsIdx);
  auto ldsCoreIdsAscendingOrder = getLdsTransferCoreIds(mySDsc, dsc, lds);
  std::sort(ldsCoreIdsAscendingOrder.begin(), ldsCoreIdsAscendingOrder.end());

  const auto &refCoreIdToWkSlice = mySDsc.coreIdToWkSlice_;
  auto &tgtCoreIdToWkSlice = coordinate.coreIdToWkSlice_;
  // Initialize tgtCoreIdToWkSlice entry with the refCoreIdToWkSlice.
  for (const auto coreId : ldsCoreIdsAscendingOrder) {
    if (refCoreIdToWkSlice.count(coreId)) {
      tgtCoreIdToWkSlice[coreId] = refCoreIdToWkSlice.at(coreId);
    }
  }

  // Set the new wkSliceId for each corelet split dimension.
  for (const auto &[dim, coreletSplitArr] :
       dsc.dataStageParam_.at(dataStageCoreIdx).ss_.coreletSplit_) {
    DT_CHECK_MSG(
        dsc.numCoreletsUsed_DSC2_ = coreletSplitArr.size(),
        "Number of corelet split entries does not match number of corelets.");
    std::unordered_set<int> visitedCoreIds;
    for (int currWkSliceId = 0;
         currWkSliceId < mySDsc.numWkSlicesPerDim_.at(dim); ++currWkSliceId) {
      int wkSliceIdOffset = dsc.numCoreletsUsed_DSC2_ - 1;
      // Visit the core Ids in the ascending order. The wkSliceId is assigned in
      // the same order as corelet ID. In the hardware, the SFP ring direction
      // of corelet 0 is from smaller core ID to larger core ID. It is vice
      // versa for corelet 1. As a result, the larger coreId gets the smaller
      // wkSliceId.
      for (auto coreId : ldsCoreIdsAscendingOrder) {
        DT_CHECK_MSG(tgtCoreIdToWkSlice.count(coreId), "Core ID not found.");
        auto &dimsWkSlices = tgtCoreIdToWkSlice.at(coreId);
        if (!visitedCoreIds.count(coreId)) {
          const int origWkSliceId = dimsWkSlices.at(dim);
          if (currWkSliceId == origWkSliceId) {
            const int newWkSliceId =
                dsc.numCoreletsUsed_DSC2_ * origWkSliceId + wkSliceIdOffset;
            dimsWkSlices[dim] = newWkSliceId;
            wkSliceIdOffset = (wkSliceIdOffset + 1) % dsc.numCoreletsUsed_DSC2_;
            visitedCoreIds.insert(coreId);
          }
        }
      }
    }
    const auto numCorelets = coreletSplitArr.size();
  }
}

// ------------------------------------------------------------------------------------------------
// entry 365/382   level 5   scc 59   33 body lines
// unit: e365_getMinParamForDimFromOpFunc
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1137
// original: long L3DlOpsScheduler::getMinParamForDimFromOpFunc(const DesignSpaceConfig& dsc, PrimaryDimTypes dim) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
long e365_getMinParamForDimFromOpFunc(const DesignSpaceConfig& dsc,
                                                   PrimaryDimTypes dim) const
{
  // Default valid minimum parameter value is one.
  constexpr long defaultParam = 1;
  const auto opFuncName = getOpFuncName(dsc);
  // Conv2D
  if (isOpFuncConv2d(opFuncName)) {
    return getMinParamConv2d(dsc, dim, opFuncName);
  }
  // BMM
  else if (isOpFuncBmm(opFuncName)) {
    return getMinParamBmm(dsc, dim, opFuncName);
  }
  // Scalar Ops and Broadcast Ops.
  else if (isOpFuncScalarBroadcast(opFuncName)) {
    return getMinParamScalarBroadcast(dsc, dim);
  }
  // Reduction Ops.
  else if (isOpFuncReduction(opFuncName)) {
    return getMinParamReduction(dsc, dim);
  }
  // MaxPooling, AvgPool, Depthwise Conv.
  else if (isOpFuncPooling(opFuncName) || isOpFuncDepthwiseConv(opFuncName)) {
    return getMinParamPoolingAndDepthwiseConv(dsc, dim, opFuncName);
  }
  // Quantization Ops.
  else if (isOpFuncQuantization(opFuncName)) {
    return getMinParamQuantization(dsc, dim, opFuncName);
  } else if (isOpFuncConversionDl16AndFp32(opFuncName)) {
    return getMinParamConversionDl16AndFp32(dsc, dim, opFuncName);
  }

  return defaultParam;
}

// ------------------------------------------------------------------------------------------------
// entry 366/382   level 5   scc 78   228 body lines
// unit: e366_findBestParamsForMemoryBandwidth
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2018
// original: void L3DlOpsScheduler::findBestParamsForMemoryBandwidth( DscParamCandidateIndicesType &dscCandidateIndices, const DscParamCandidatesType &dscCandidates, SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &primaryDims, const std::unordered_set<PrimaryDimTypes> &coreSplitDims, const bool isInputNeighborFetch)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e366_findBestParamsForMemoryBandwidth(
    DscParamCandidateIndicesType &dscCandidateIndices,
    const DscParamCandidatesType &dscCandidates, SuperDsc &mySDsc,
    const std::vector<PrimaryDimTypes> &primaryDims,
    const std::unordered_set<PrimaryDimTypes> &coreSplitDims,
    const bool isInputNeighborFetch)
{
  // We need information from the schedule nodes (specifically the allocate
  // nodes) to determine if there is enough space in LX.
  for (const DesignSpaceConfig &dsc : mySDsc.dscs_) {
    DT_CHECK_MSG(!dsc.scheduleTree_.empty(),
                 "Expect that schedule nodes have already been created.");
  }

  // Compute avgBestBurstEfficiency.
  for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
    DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
                 "Core data stage parameters are unavailable.");
    const auto &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);
    // Set up the temporary parameters, and add the chunk parameters to
    // dataStageParams_ for checking available LX space and computing the
    // efficiency.
    DataStructDims params = dsCore.ss_;
    params.symbolicDimInfo_.clear();
    params.maxSymbolicVolume_.clear();
    updateChunkDataStagesFromCandidates(
        params, dsc, dscIdx, dscCandidateIndices, dscCandidates, primaryDims);
    DT_CHECK_MSG(
        (isInputNeighborFetch || allocAllMem(mySDsc, &dsc, dscIdx, false)),
        "Expect valid chunk size that fits in LX.");
  }
  double avgBestBurstEfficiency = calculateBurstEfficiency(mySDsc, primaryDims);
  DT_CHECK_MSG(avgBestBurstEfficiency > 0.0,
               "Expect positive efficiency value.");

  if (verbose > 0) {
    std::cout << "Start exploring best chunk parameters for memory bandwidth"
              << std::endl;
    std::cout << "Starting parameters:" << std::endl;
    for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
      std::cout << "DSC index: " << dscIdx << std::endl;
      DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
      dsc.dataStageParam_.at(dataStageChunkIdx).ss_.printShort(std::cout);
      std::cout << std::endl;
    }
    std::cout << "Burst efficiency: " << avgBestBurstEfficiency << std::endl;
  }

  // Collect all dimensions to be explored. Since all DSCs in a group have the
  // same labeledDs and primaryDsInfo information, we collect it from one DSC.
  const unsigned dscMainIdx = 0;
  const DesignSpaceConfig &dscMain = mySDsc.dscs_.at(dscMainIdx);

  // For each DsType, count the number of tensors that use it.
  std::unordered_map<DsTypes, unsigned> dsTypeCountMap;

  auto addToDsTypeCountMap = [&dsTypeCountMap](DsTypes type) {
    if (dsTypeCountMap.count(type))
      ++dsTypeCountMap[type];
    else
      dsTypeCountMap.emplace(type, 1);
  };

  for (int ldsIdx = 0; ldsIdx < dscMain.labeledDs_.size(); ++ldsIdx) {
    const LabeledDsInfo &lds = dscMain.labeledDs_.at(ldsIdx);
    if (lds.isHbmPinned() || isLabeledDsLXNeighbor(mySDsc, dscMainIdx, lds))
      addToDsTypeCountMap(lds.dsType_);
  }

  auto compareDsTypeCount = [](const std::pair<DsTypes, unsigned> &a,
                               const std::pair<DsTypes, unsigned> &b) {
    return a.second > b.second;
  };

  // DsTypes sorted based on the number of counts, with the most used DsType
  // in the beginning.
  std::vector<std::pair<DsTypes, unsigned>> sortedDsTypes;
  for (auto &entry : dsTypeCountMap) sortedDsTypes.push_back(entry);
  std::sort(sortedDsTypes.begin(), sortedDsTypes.end(), compareDsTypeCount);

  // Explore with the chunk dimensions in the following order:
  //   1) Innermost dimensions from the most used DsType to the least used
  //   DsType.
  //   2) Repeat on one-level outer dimensions from the most used DsType
  //   to the least used DsType, until all dimensions in all DsTypes are
  //   included.
  // There could be a dimension that exists in multiple DsTypes, and it is
  // explored multiple times as a result. This may help find a better
  // parameter for this dimension at the subsequent attempts, when other
  // dimensions are explored in between.
  std::vector<PrimaryDimTypes> sortedExploringChunkDims;
  unsigned layoutDimOrderIdx = 0;
  bool hasDim = false;
  do {
    hasDim = false;
    for (auto &entry : sortedDsTypes) {
      const auto &ldo = dscMain.primaryDsInfo_.at(entry.first).layoutDimOrder_;
      if (layoutDimOrderIdx < ldo.size()) {
        PrimaryDimTypes dim = ldo.at(layoutDimOrderIdx);
        DT_CHECK_MSG(
            (dim != PrimaryDimTypes::IJ || dim != PrimaryDimTypes::KIJ),
            "Do not expect combined IJ or KIJ dimensions.");
        sortedExploringChunkDims.push_back(dim);
        hasDim = true;
      }
    }

    ++layoutDimOrderIdx;
  } while (hasDim);

  DT_CHECK_MSG(!sortedExploringChunkDims.empty(),
               "Expect dimensions to explore.");

  // Start to explore. We explore all candidate parameters for all dimensions
  // identified in sortedExploringChunkDims.
  for (auto dim : sortedExploringChunkDims) {
    // Prepare all possible sets of candidate parameters to be explored in an
    // array.
    std::vector<DscParamCandidateIndicesType> selectedIndicesArray;
    const bool isCoreSplitDim = coreSplitDims.count(dim);
    if (!isCoreSplitDim) {
      // For a non-core-split dimension, all DSCs must have the same candidates.
      // Hence, we use dscMain to iterate through the candidates.
      const unsigned startIdx = dscCandidateIndices[dscMainIdx][dim] + 1;
      for (unsigned idx = startIdx;
           idx < dscCandidates[dscMainIdx].at(dim).size(); ++idx) {
        DscParamCandidateIndicesType selectedIndices = dscCandidateIndices;
        for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
          DT_CHECK_MSG(idx < dscCandidates[dscIdx].at(dim).size(),
                       "Index is out of range.");
          selectedIndices[dscIdx][dim] = idx;
        }
        selectedIndicesArray.push_back(selectedIndices);
      }
    } else {
      // For a core-split dimension, the candidate values are explored
      // independently for each DSC.
      for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
        const unsigned startIdx = dscCandidateIndices[dscIdx][dim] + 1;
        for (unsigned idx = startIdx;
             idx < dscCandidates[dscIdx].at(dim).size(); ++idx) {
          DscParamCandidateIndicesType selectedIndices = dscCandidateIndices;
          selectedIndices[dscIdx][dim] = idx;
          selectedIndicesArray.push_back(selectedIndices);
        }
      }
    }

    // Iterate through all possible sets of candidate parameters, compute
    // avgBurstEfficiency, compare with avgBestBurstEfficiency, and select the
    // best set if any.
    unsigned selectedIndicesArrayIdx = selectedIndicesArray.size();
    for (unsigned i = 0; i < selectedIndicesArray.size(); ++i) {
      bool allocSuccess = true;
      const DscParamCandidateIndicesType &selectedIndices =
          selectedIndicesArray[i];
      for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
        DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
        DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
                     "Core data stage parameters are unavailable.");
        const auto &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);

        // Set up the temporary parameters to be explored, and add them to
        // dataStageParams_ for checking available LX space and computing the
        // efficiency.
        DataStructDims params = dsCore.ss_;
        params.symbolicDimInfo_.clear();
        params.maxSymbolicVolume_.clear();
        updateChunkDataStagesFromCandidates(
            params, dsc, dscIdx, selectedIndices, dscCandidates, primaryDims);

        // Skip if there is not enough space in LX.
        // For input-neighbor fetch, enough space in LX is reserved, so we
        // don't need to check the size.
        if (!isInputNeighborFetch &&
            !allocAllMem(mySDsc, &dsc, dscIdx, false)) {
          allocSuccess = false;
          break;
        }
      }
      double avgBurstEfficiency =
          allocSuccess ? calculateBurstEfficiency(mySDsc, primaryDims) : 0.0;

      if (verbose > 0) {
        std::cout << "Current parameters:" << std::endl;
        for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
          std::cout << "DSC index: " << dscIdx << std::endl;
          DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
          dsc.dataStageParam_.at(dataStageChunkIdx).ss_.printShort(std::cout);
          std::cout << std::endl;
        }
        std::cout << "Burst efficiency: " << avgBurstEfficiency << std::endl;
      }

      // Update avgBestBurstEfficiency and the selected indices.
      if (avgBurstEfficiency > avgBestBurstEfficiency) {
        avgBestBurstEfficiency = avgBurstEfficiency;
        selectedIndicesArrayIdx = i;
      }
    }

    // Update the candidate parameter.
    if (selectedIndicesArrayIdx != selectedIndicesArray.size())
      dscCandidateIndices = selectedIndicesArray[selectedIndicesArrayIdx];
  }

  // Add the selected parameters to dataStageParam_.
  for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
    DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
                 "Core data stage parameters are unavailable.");
    const auto &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);
    DataStructDims params = dsCore.ss_;
    params.symbolicDimInfo_.clear();
    params.maxSymbolicVolume_.clear();
    updateChunkDataStagesFromCandidates(
        params, dsc, dscIdx, dscCandidateIndices, dscCandidates, primaryDims);
    DT_CHECK_MSG(
        (isInputNeighborFetch || allocAllMem(mySDsc, &dsc, dscIdx, false)),
        "Expect valid chunk size that fits in LX.");
  }

  if (verbose > 0) {
    std::cout << "Selected parameters:" << std::endl;
    for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
      std::cout << "DSC index: " << dscIdx << std::endl;
      DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
      dsc.dataStageParam_.at(dataStageChunkIdx).ss_.printShort(std::cout);
      std::cout << std::endl;
    }
    std::cout << "Burst efficiency: " << avgBestBurstEfficiency << std::endl;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 367/382   level 5   scc 73   214 body lines
// unit: e367_findBestParamsForArithmeticIntensity
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:2500
// original: void L3DlOpsScheduler::findBestParamsForArithmeticIntensity( DscParamCandidateIndicesType &dscCandidateIndices, const DscParamCandidatesType &dscCandidates, SuperDsc &mySDsc, const std::vector<PrimaryDimTypes> &primaryDims, const std::unordered_set<PrimaryDimTypes> &coreSplitDims)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e367_findBestParamsForArithmeticIntensity(
    DscParamCandidateIndicesType &dscCandidateIndices,
    const DscParamCandidatesType &dscCandidates, SuperDsc &mySDsc,
    const std::vector<PrimaryDimTypes> &primaryDims,
    const std::unordered_set<PrimaryDimTypes> &coreSplitDims)
{
  for (const DesignSpaceConfig &dsc : mySDsc.dscs_) {
    // We need information from the schedule nodes (specifically the allocate
    // nodes) to determine if there is enough space in LX.
    DT_CHECK_MSG(!dsc.scheduleTree_.empty(),
                 "Expect that schedule nodes have already been created.");
  }

  // Use any DSC to obtain the system Flops/Byte. The system Flops/Byte
  // includes all cores and corelets. Therefore, we need to adjust it based on
  // the current numbers of cores and corelets in all DSCs.
  int totalNumCores = 0;
  for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    const DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
    totalNumCores += dsc.numCoresUsed_;
  }
  const DesignSpaceConfig &dscMain = mySDsc.dscs_.at(0);
  std::string dataFormat = getOpFuncDataFormat(dscMain);
  const double sysFlopPerByte =
      dscGlobal.sysDef.sysFlopsPerByte.at(dataFormat) *
      (dscMain.numCoreletsUsed_ * totalNumCores) /
      (dscGlobal.sysDef.numCoreletsPerCore * dscGlobal.sysDef.numCores);

  // Compute bestWkloadFlopPerByte.
  for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
    DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
                 "Core data stage parameters are unavailable.");
    const auto &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);

    // Set up the temporary parameters, and add the chunk parameters to
    // dataStageParams_ for checking available LX space and computing the
    // flop/byte.
    DataStructDims params = dsCore.ss_;
    params.symbolicDimInfo_.clear();
    params.maxSymbolicVolume_.clear();
    updateChunkDataStagesFromCandidates(
        params, dsc, dscIdx, dscCandidateIndices, dscCandidates, primaryDims);
    DT_CHECK_MSG((allocAllMem(mySDsc, &dsc, dscIdx, false)),
                 "Expect valid chunk size that fits in LX.");
  }
  double bestWkloadFlopPerByte = calculateFlopPerByte(mySDsc, primaryDims);

  if (verbose > 0) {
    std::cout
        << "Start exploring best chunk parameters for arithmetic intensity"
        << std::endl;
    std::cout << "Starting parameters:" << std::endl;
    for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
      std::cout << "DSC index: " << dscIdx << std::endl;
      DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
      dsc.dataStageParam_.at(dataStageChunkIdx).ss_.printShort(std::cout);
      std::cout << std::endl;
    }
    std::cout << "System flops/byte: " << sysFlopPerByte << std::endl;
    std::cout << "Starting flops/byte: " << bestWkloadFlopPerByte << std::endl;
  }

  auto isCurrFlopPerByteBetterThanBest = [&sysFlopPerByte,
                                          &bestWkloadFlopPerByte](double curr) {
    if (curr <= 0.0) return false;

    // We prefer the bestWkloadFlopPerByte to be sufficiently compute bound.
    // Therefore, we use an adjusted sysFlopPerByte to represent the optimal
    // workload flops/byte value.
    // When the workload flops/byte is less than the system flops/byte, we refer
    // this workload as memory bound. Otherwise, when workload flops/byte is
    // greater than the system flops/byte, we refer it as compute bound. In our
    // heuristics, the best workload flops/byte value is within the range
    // [sysFlopPerByte, adjSysFlopPerByte] if possible. Therefore, we determine
    // that the current value is better in the following cases:
    //   1. When the previous best value is memory bound, the current value is
    //   greater than that.
    //   2. When the previous best value is compute bound beyond
    //   adjSysFlopPerByte, the current value is also compute bound and closer
    //   to sysFlopPerByte.
    //   3. When the previous best value is within the range [sysFlopPerByte,
    //   adjSysFlopPerByte], the current value is also within the range and
    //   closer to adjSysFlopPerByte.
    const double alpha = 1.1;  // A heuristic value.
    const double adjSysFlopPerByte = alpha * sysFlopPerByte;
    if ((bestWkloadFlopPerByte < sysFlopPerByte &&
         curr > bestWkloadFlopPerByte) ||
        (bestWkloadFlopPerByte > adjSysFlopPerByte && curr > sysFlopPerByte &&
         curr < bestWkloadFlopPerByte) ||
        (curr > bestWkloadFlopPerByte && curr <= adjSysFlopPerByte))
      return true;

    return false;
  };

  // Explore the best set of parameters, by updating the parameters until
  // there is no update that improves avgBestWkloadFlopPerByte.
  while (true) {
    // Prepare all possible sets of candidate paramters to be explored in an
    // array.
    std::vector<DscParamCandidateIndicesType> selectedIndicesArray;
    for (PrimaryDimTypes dim : primaryDims) {
      const bool isCoreSplitDim = coreSplitDims.count(dim);
      if (!isCoreSplitDim) {
        DscParamCandidateIndicesType selectedIndices = dscCandidateIndices;
        bool isOutOfRange = false;
        for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
          if (selectedIndices[dscIdx][dim] + 1 <
              dscCandidates[dscIdx].at(dim).size())
            ++selectedIndices[dscIdx][dim];
          else {
            isOutOfRange = true;
            break;
          }
        }

        // Add to array if the index is within range.
        if (!isOutOfRange) selectedIndicesArray.push_back(selectedIndices);
      } else {
        for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
          DscParamCandidateIndicesType selectedIndices = dscCandidateIndices;

          // Add to array if the index is within range.
          if (selectedIndices[dscIdx][dim] + 1 <
              dscCandidates[dscIdx].at(dim).size()) {
            ++selectedIndices[dscIdx][dim];
            selectedIndicesArray.push_back(selectedIndices);
          }
        }
      }
    }

    // Iterate through all possible sets of candidate parameters, compute
    // wkloadFlopPerByte, compare with bestWkloadFlopPerByte, and select the
    // best set if any.
    unsigned selectedIndicesArrayIdx = selectedIndicesArray.size();
    for (unsigned i = 0; i < selectedIndicesArray.size(); ++i) {
      bool allocSuccess = true;
      const DscParamCandidateIndicesType &selectedIndices =
          selectedIndicesArray[i];
      for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
        DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
        DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
                     "Core data stage parameters are unavailable.");
        const auto &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);

        // Set up the temporary parameters to be explored, and add them to
        // dataStageParam_ for checking available LX space and compute
        // flop/byte.
        DataStructDims params = dsCore.ss_;
        params.symbolicDimInfo_.clear();
        params.maxSymbolicVolume_.clear();
        updateChunkDataStagesFromCandidates(
            params, dsc, dscIdx, selectedIndices, dscCandidates, primaryDims);

        // Skip if there is not enough space in LX.
        if (!allocAllMem(mySDsc, &dsc, dscIdx, false)) {
          allocSuccess = false;
          break;
        }
      }
      double wkloadFlopPerByte =
          allocSuccess ? calculateFlopPerByte(mySDsc, primaryDims) : 0.0;

      if (verbose > 0) {
        std::cout << "Current parameters:" << std::endl;
        for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
          std::cout << "DSC index: " << dscIdx << std::endl;
          DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
          dsc.dataStageParam_.at(dataStageChunkIdx).ss_.printShort(std::cout);
          std::cout << std::endl;
        }
        std::cout << "flops/byte: " << wkloadFlopPerByte << std::endl;
      }

      // Update avgBestWkloadFlopPerByte and the selected indices.
      if (isCurrFlopPerByteBetterThanBest(wkloadFlopPerByte)) {
        bestWkloadFlopPerByte = wkloadFlopPerByte;
        selectedIndicesArrayIdx = i;
      }
    }

    // There is no more update on the parameters to improve
    // avgBestWkloadFlopPerByte. We settle with the current parameters and
    // stop exploring.
    if (selectedIndicesArrayIdx == selectedIndicesArray.size()) break;

    // There is a new dimension parameter selected. Update the candidate
    // parameter.
    dscCandidateIndices = selectedIndicesArray[selectedIndicesArrayIdx];
  }

  // Add the selected parameters to dataStageParam_.
  for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
    DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
                 "Core data stage parameters are unavailable.");
    const auto &dsCore = dsc.dataStageParam_.at(dataStageCoreIdx);
    DataStructDims params = dsCore.ss_;
    params.symbolicDimInfo_.clear();
    params.maxSymbolicVolume_.clear();
    updateChunkDataStagesFromCandidates(
        params, dsc, dscIdx, dscCandidateIndices, dscCandidates, primaryDims);
    DT_CHECK_MSG((allocAllMem(mySDsc, &dsc, dscIdx, false)),
                 "Expect valid chunk size that fits in LX.");
  }

  if (verbose > 0) {
    std::cout << "Selected parameters:" << std::endl;
    for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
      std::cout << "DSC index: " << dscIdx << std::endl;
      DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
      dsc.dataStageParam_.at(dataStageChunkIdx).ss_.printShort(std::cout);
      std::cout << std::endl;
    }
    std::cout << "flops/byte: " << bestWkloadFlopPerByte << std::endl;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 368/382   level 5   scc 132   4 body lines
// unit: e368_processHbmPagedTensors
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:6741
// original: void L3DlOpsScheduler::processHbmPagedTensors(SuperDsc &mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e368_processHbmPagedTensors(SuperDsc &mySDsc)
{
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx)
    processDscHbmPagedTensors(mySDsc, dscIdx);
}

// ------------------------------------------------------------------------------------------------
// entry 369/382   level 5   scc 142   28 body lines
// unit: e369_buildCoordinateForAllocation
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7167
// original: void L3DlOpsScheduler::buildCoordinateForAllocation( SuperDsc &mySDsc, DesignSpaceConfig &dsc, dsc2::AllocateNode *allocNode, dsc2::CoordPropInfoType &coordPropInfo) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e369_buildCoordinateForAllocation(
    SuperDsc &mySDsc, DesignSpaceConfig &dsc, dsc2::AllocateNode *allocNode,
    dsc2::CoordPropInfoType &coordPropInfo) const
{
  // Check if the target allocation is for the output tensor's LX cross-core
  // reduction scenario.
  const bool isAllocationForLxCrossCoreReduction =
      (allocNode->component_ == SenComponents::LX &&
       isOutputLabeledDs(allocNode->ldsIdx_, dsc) &&
       isOpCrossCoreReduction(mySDsc, dsc));
  // Fill the custom coreIdToWkSlice_ in the coordinate if the allocation is for
  // the output tensor's LX cross-core reduction.
  const bool needCustomWkSliceId = isAllocationForLxCrossCoreReduction;
  if (needCustomWkSliceId)
    fillCoordinateCustomWkSliceId(mySDsc, dsc, allocNode->ldsIdx_,
                                  allocNode->allocateCoordinates_);

  // Build coordinates.
  allocNode->allocateCoordinates_.setPadding(allocNode->padding_);
  if (coordPropInfo.refNode->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
    dsc2::AllocateNode *refAllocNode =
        static_cast<dsc2::AllocateNode *>(coordPropInfo.refNode);
    buildCoordinateFromAllocation(dsc, refAllocNode, allocNode,
                                  allocNode->allocateCoordinates_);
  } else
    DT_ERROR("Unsupported schedule node type.");

  // Adjust the coordinates for the output tensor's LX cross-core reduction.
  if (isAllocationForLxCrossCoreReduction)
    sliceCoordinateForCorelet(mySDsc, &dsc, allocNode);
}

// ------------------------------------------------------------------------------------------------
// entry 373/382   level 6   scc 60   15 body lines
// unit: e373_getMinParamForDim
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1119
// original: long L3DlOpsScheduler::getMinParamForDim(const SuperDsc& mySDsc, const DesignSpaceConfig& dsc, PrimaryDimTypes dim) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
long e373_getMinParamForDim(const SuperDsc& mySDsc,
                                         const DesignSpaceConfig& dsc,
                                         PrimaryDimTypes dim) const
{
  // FIXME: Remove this limitation when a fix is available.
  // for ops with partial output reduction, dont chunk corelet split dim
  // 2 different limitations:
  //    a) when output is lxopted -> input cannot be chunk half/half -> ALxS
  //    to be enhanced b) when output is not-lxopted -> using workslice fold
  //    to indicate psum corner core coords will be not faithful ->
  //    indirectly addressed with 1)
  if (const auto& dsCoreSs = dsc.dataStageParam_.at(dataStageCoreIdx).ss_;
      isOpCrossCoreReduction(mySDsc, dsc) &&
      dsCoreSs.coreletSplit_.count(dim)) {
    return dsCoreSs.primaryDimToVal_st(dim);
  }
  return getMinParamForDimFromOpFunc(dsc, dim);
}

// ------------------------------------------------------------------------------------------------
// entry 374/382   level 6   scc 143   109 body lines
// unit: e374_propagateCoordinateDSC
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7761
// original: void L3DlOpsScheduler::propagateCoordinateDSC(SuperDsc &mySDsc, DesignSpaceConfig &dsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e374_propagateCoordinateDSC(SuperDsc &mySDsc,
                                              DesignSpaceConfig &dsc) const
{
  const auto hbmAllocations = getHbmAllocations(dsc);
  std::unordered_set<dsc2::ScheduleNode *> visitedNodes;
  std::deque<dsc2::ScheduleNode *> refNodes;

  for (const auto hbmAllocNode : hbmAllocations) {
    refNodes.push_back(hbmAllocNode);
    visitedNodes.insert(hbmAllocNode);
  }

  while (!refNodes.empty()) {
    auto currRefNode = refNodes.front();
    refNodes.pop_front();

    auto collectTargetNodes = [&dsc](const dsc2::ScheduleNode *refNode) {
      std::vector<dsc2::ScheduleNode *> targetNodes;
      // Collect target nodes to propagate coordinates.
      if (refNode->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
        const auto refAllocNode =
            static_cast<const dsc2::AllocateNode *>(refNode);
        // Find the target allocate nodes by the transfer nodes in the reference
        // allocation users.
        std::vector<const dsc2::TransferNode *> userTransNodes;
        for (auto &[userNode, count] : refAllocNode->allocUsers_) {
          if (userNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
            // Currently we only collect transfers between HBM and LX.
            const auto transNode =
                static_cast<const dsc2::TransferNode *>(userNode);
            if ((transNode->src_.storage_ == SenComponents::HBM &&
                 (!transNode->dstVias_.empty() &&
                  transNode->dstVias_.front().loc_.storage_ ==
                      SenComponents::LX)) ||
                (transNode->src_.storage_ == SenComponents::LX &&
                 (!transNode->dstVias_.empty() &&
                  transNode->dstVias_.front().loc_.storage_ ==
                      SenComponents::HBM)))
              userTransNodes.push_back(transNode);
          }
        }

        for (auto transNode : userTransNodes) {
          auto addAllocNodesToTarget = [&dsc, &targetNodes, &refAllocNode](
                                           const int ldsIdx,
                                           const SenComponents storage) {
            if (storage == SenComponents::HBM || storage == SenComponents::LX) {
              DT_CHECK_MSG(ldsIdx < dsc.labeledDs_.size(),
                           "Expect a valid labeledDs_ entry.");
              const auto &lds = dsc.labeledDs_.at(ldsIdx);
              DT_CHECK_MSG(lds.memOrg_.count(storage),
                           "Expect the storage entry in memOrg_.");
              const auto currAllocNode = lds.memOrg_.at(storage).allocateNode_;
              DT_CHECK_MSG(currAllocNode, "Expect a valid allocate node.");
              if (currAllocNode != refAllocNode)
                targetNodes.push_back(currAllocNode);
            }
          };

          if (transNode->isSrcLabeledDs()) {
            const auto srcLdsIdx = transNode->srcLdsAndLoopOffsets_.myLdsIdx_;
            const SenComponents srcStorage = transNode->src_.storage_;
            addAllocNodesToTarget(srcLdsIdx, srcStorage);
          }
          if (transNode->isDstLabeledDs()) {
            DT_CHECK_MSG(!transNode->dstLdsAndLoopOffsets_.empty(),
                         "Expect valid dstLdsAndLoopOffsets_ entry.");
            const auto dstLdsIdx =
                transNode->dstLdsAndLoopOffsets_.front().myLdsIdx_;
            DT_CHECK_MSG(!transNode->dstVias_.empty(),
                         "Expect valid dstVias_ entry.");
            const SenComponents dstStorage =
                transNode->dstVias_.front().loc_.storage_;
            addAllocNodesToTarget(dstLdsIdx, dstStorage);
          }
        }
      } else
        DT_ERROR("Unsupported schedule node type.");

      return targetNodes;
    };

    const auto targetNodes = collectTargetNodes(currRefNode);
    for (auto currTargetNode : targetNodes) {
      // Build target node's coordinates if it has not been visited. Also, add
      // it as a new ref node.
      if (!visitedNodes.count(currTargetNode)) {
        refNodes.push_back(currTargetNode);
        visitedNodes.insert(currTargetNode);

        // Build coordinates.
        if (auto targetAllocNode =
                dynamic_cast<dsc2::AllocateNode *>(currTargetNode);
            targetAllocNode) {
          // Build for allocate node.
          if (auto refAllocNode =
                  dynamic_cast<dsc2::AllocateNode *>(currRefNode);
              refAllocNode) {
            dsc2::CoordPropInfoType coordPropInfo;
            coordPropInfo.refNode = refAllocNode;
            coordPropInfo.nodeToFold = targetAllocNode;
            buildCoordinateForAllocation(mySDsc, dsc, targetAllocNode,
                                         coordPropInfo);
          } else
            DT_ERROR("Unsupported schedule node type.");
        } else
          DT_ERROR("Unsupported schedule node type.");
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 377/382   level 7   scc 63   20 body lines
// unit: e377_getInitialChunkParams
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1382
// original: DataStructDims L3DlOpsScheduler::getInitialChunkParams( const SuperDsc &mySDsc, const DesignSpaceConfig &dsc, const std::unordered_set<PrimaryDimTypes> &chunkDims)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
DataStructDims e377_getInitialChunkParams(
    const SuperDsc &mySDsc, const DesignSpaceConfig &dsc,
    const std::unordered_set<PrimaryDimTypes> &chunkDims)
{
  DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
               "Expect dataStageParam_ entry for the core data stage.");
  // Initialize with the CoreD parameters.
  DataStructDims params = dsc.dataStageParam_.at(dataStageCoreIdx).ss_;
  params.symbolicDimInfo_.clear();
  params.maxSymbolicVolume_.clear();

  // Set each chunk dimension's parameter to its minimum value.
  for (auto dim : chunkDims) {
    DT_CHECK(isValidDimParam(dsc.dataStageParam_.at(dataStageCoreIdx)
                                 .ss_.primaryDimToVal_st(dim)) &&
             "Expect the chunk dimension has a valid parameter value.");
    params.primaryDimToValHandler_st(dim) = getMinParamForDim(mySDsc, dsc, dim);
  }

  params.compound();

  return params;
}

// ------------------------------------------------------------------------------------------------
// entry 378/382   level 7   scc 144   3 body lines
// unit: e378_propagateCoordinate
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7757
// original: void L3DlOpsScheduler::propagateCoordinate(SuperDsc &mySDsc) const
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e378_propagateCoordinate(SuperDsc &mySDsc) const
{
  for (auto &dsc : mySDsc.dscs_) propagateCoordinateDSC(mySDsc, dsc);
}

// ------------------------------------------------------------------------------------------------
// entry 380/382   level 8   scc 79   129 body lines
// unit: e380_setChunkDataStageParams
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:1439
// original: void L3DlOpsScheduler::setChunkDataStageParams(SuperDsc &mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e380_setChunkDataStageParams(SuperDsc &mySDsc)
{
  const std::string chunkDsName = "chunk";

  // Check the tensors residency and the chunk dimensions.
  // The residency and the chunk dimensions hould be identical for all
  // DSCs in a group. Therefore, we check them using only one DSC.
  const int dscMainIndex = 0;
  const DesignSpaceConfig &dscMain = mySDsc.dscs_.at(dscMainIndex);
  const bool isReuse = hasDimensionReuse(dscMain);
  bool isDoubleBuffering = false, isInputNeighborFetch = false;
  std::unordered_set<PrimaryDimTypes> chunkDims;
  for (auto &lds : dscMain.labeledDs_) {
    if (lds.isHbmPinned() || isLabeledDsLXNeighbor(mySDsc, dscMainIndex, lds)) {
      for (PrimaryDimTypes dim : dscMain.getNonBroadcastLdsDims(lds.ldsIdx_)) {
        DT_CHECK_MSG(
            (dim != PrimaryDimTypes::IJ || dim != PrimaryDimTypes::KIJ),
            "Do not expect combined IJ or KIJ dimensions.");
        chunkDims.insert(dim);
      }

      if (lds.isHbmPinned()) isDoubleBuffering = true;
      if (isLabeledDsLXNeighbor(mySDsc, dscMainIndex, lds))
        isInputNeighborFetch = true;
    }
  }
  DT_CHECK_MSG(
      !(isDoubleBuffering && isInputNeighborFetch),
      "Do not support double buffering and input-neighbor fetch coexisting "
      "in the same DSC.");

  // When all tensors are LX-local, the chunk parameters are the same as the
  // core parameters. Set the chunk dataStageParam_ and return.
  const bool isAllLxLocal = !isDoubleBuffering && !isInputNeighborFetch;
  if (isAllLxLocal) {
    for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
      DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
      DT_CHECK_MSG(dsc.dataStageParam_.count(dataStageCoreIdx),
                   "Core data stage parameters are unavailable.");
      addOrUpdateDataStageParam(
          dsc, dsc.dataStageParam_.at(dataStageCoreIdx).ss_, chunkDsName,
          dsc.dataStageParam_.at(dataStageCoreIdx).el_, chunkDsName,
          dataStageChunkIdx);
    }
  }
  // Explore chunk data stage parameters.
  else {
    // All primary dimensions.
    std::vector<PrimaryDimTypes> primaryDims;
    for (const auto &entry : EnumsConversion::primaryDimToString) {
      if (is_any_of(entry.first, PrimaryDimTypes::IJ, PrimaryDimTypes::KIJ,
                    PrimaryDimTypes::PrimaryDimTypesCount))
        continue;

      primaryDims.push_back(entry.first);
    }

    // A vector of chunk parameters sets for each DSC in a DSC group.
    std::vector<DataStructDims> dscChunkParams;

    // Get the initial chunk parameters.
    for (auto &dsc : mySDsc.dscs_) {
      dscChunkParams.emplace_back(
          getInitialChunkParams(mySDsc, dsc, chunkDims));
    }
    DT_CHECK_MSG(dscChunkParams.size() == mySDsc.dscs_.size(),
                 "Number of DSCs does not match.");

    // Get the core split dimensions.
    std::unordered_set<PrimaryDimTypes> coreSplitDims =
        getCoreSplitDimensions(mySDsc);

    // Generate the parameter candidates for each chunk dimension in each DSC.
    DscParamCandidatesType dscCandidates = generateDscParamCandidates(
        mySDsc, dscChunkParams, primaryDims, chunkDims, coreSplitDims);
    DT_CHECK_MSG(dscCandidates.size() == mySDsc.dscs_.size(),
                 "Number of DSCs does not match.");

    // The indices for the selected candidates of all primary dimensions.
    // The indices all start with 0.
    DscParamCandidateIndicesType dscCandidateIndices(dscCandidates.size());
    for (auto &entry : dscCandidateIndices) {
      for (PrimaryDimTypes dim : primaryDims) entry.emplace(dim, 0);
    }

    // We temporarily add the initial set of chunk parameters to
    // dataStageParams_. If it is double buffering, the chunks should fit in
    // LX.
    for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
      DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
      updateChunkDataStagesFromCandidates(dscChunkParams[dscIdx], dsc, dscIdx,
                                          dscCandidateIndices, dscCandidates,
                                          primaryDims);

      if (isDoubleBuffering) {
        if (!allocAllMem(mySDsc, &dsc, dscIdx, false)) {
          DT_ERROR_FMT(
              "Unable to map graph within architecture constraints: The "
              "initial chunk parameters must fit in LX for SuperDSC: %s",
              mySDsc.name_.c_str());
        }
      }
    }

    if (enableChunkExplore) {
      // Find the best set of parameters for memory bandwidth.
      findBestParamsForMemoryBandwidth(dscCandidateIndices, dscCandidates,
                                       mySDsc, primaryDims, coreSplitDims,
                                       isInputNeighborFetch);

      // Find the best set of parameters for the Flop/Byte arithmetic intensity
      // in the case of tensor reuse and HBM transfers.
      if (isReuse && !isInputNeighborFetch)
        findBestParamsForArithmeticIntensity(dscCandidateIndices, dscCandidates,
                                             mySDsc, primaryDims,
                                             coreSplitDims);
    }

    // Set the best set of chunk parameters to dataStageParam_ for each DSC.
    for (unsigned dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
      DesignSpaceConfig &dsc = mySDsc.dscs_.at(dscIdx);
      updateChunkDataStagesFromCandidates(dscChunkParams[dscIdx], dsc, dscIdx,
                                          dscCandidateIndices, dscCandidates,
                                          primaryDims);

      DT_CHECK_MSG(allocAllMem(mySDsc, &dsc, dscIdx, false),
                   "Memory allocation must be valid to commit.");
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 382/382   level 9   scc 146   122 body lines
// unit: e382_run
// authority: dcg/dcg_fe/scheduler/L3DlOpsScheduler.cpp:7912
// original: void L3DlOpsScheduler::run(SuperDsc &mySDsc)
// class: L3DlOpsScheduler
// rust home: crates/compiler/deeptools/src/schedule/l3/dl_ops.rs
// ------------------------------------------------------------------------------------------------
void e382_run(SuperDsc &mySDsc)
{
  if (verbose > 0)
    std::cout << "Running L3 DL Ops Scheduler: Node-name: " << mySDsc.name_
              << std::endl;

  // Return if no DSC is in SuperDsc.
  if (mySDsc.dscs_.empty()) return;

  // Preprocess DSCs.
  prepDsc(mySDsc);

  // A debug env var to disable chunk parameters exploration.
  auto disableChunkExplorePtr =
      dtGetEnv<std::string>("DISABLE_ABOVE_LX_CHUNK_EXPLORE");
  if (disableChunkExplorePtr.has_value()) {
    std::string disableChunkExploreStr(disableChunkExplorePtr.value());
    if (disableChunkExploreStr == "1") enableChunkExplore = false;
  }

  // TODO: Currently DeepTools supports multiple DSCs in a SuperDSC, but only
  // when each DSC is doing its part of the same work (i.e. work A is splitted
  // and presented in different DSCs), meaning that they have the same
  // operations and tensors; the only differences are coreIdsUsed_ and coreD_.
  // On the other hand, it's currently unsupported when different DSCs are
  // doing different work. For example, some DSCs are part of work A, while
  // the others are part of work B. This scenario may be supported in future.
  DT_CHECK_MSG(isSameDscGroup(mySDsc), "Expect DSCs in the same group");

  // Set LX buffering type.
  setLxBufferType(mySDsc);

  // Create chunk loop nodes in schedule tree.
  createChunkLoops(mySDsc);

  // Create allocation and transfer in schedule tree.
  createAllocationAndTransfer(mySDsc);

  // Create new datastages for paged dimensions for each DSC.
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    auto &currDsc = mySDsc.dscs_.at(dscIdx);
    DT_CHECK_MSG(currDsc.dataStageParam_.count(dataStageCoreIdx),
                 "Expect a core data stage entry.");

    // Add new datastages if there are page dimensions.
    const auto pagedDims = getPagedDimensions(currDsc);
    if (!pagedDims.empty()) {
      // Add the 1Page datastage.
      addOnePageDataStage(mySDsc, currDsc, pagedDims);

      // Add the IBR datastage.
      addIbrDataStage(mySDsc, currDsc, pagedDims);
    }
  }

  // Compute the chunk data stage parameters and add a dataStageParam_ entry
  // for it.
  setChunkDataStageParams(mySDsc);

  // Compute the SuperChunk data stage parameters.
  setSuperChunkDataStageParams(mySDsc);

  // Optimize the transfer node for the HBM output tensor in the schedule
  // tree.
  optimizeHbmLdsOutputInScheduleTree(mySDsc);

  // Optimize HBM transfer node location based on the selected data stage
  // parameters.
  optimizeHbmTransfers(mySDsc);

  // Create synchronization in schedule tree.
  createSynchronization(mySDsc);

  // If there are HBM paged tensors, process them by modifying the schedule tree
  // to create indirect transfers.
  processHbmPagedTensors(mySDsc);

  // Commit all LX memory allocations. After this point, no new LX allocation
  // can be created.
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    const bool commitLxMem =
        allocAllMem(mySDsc, &mySDsc.dscs_[dscIdx], dscIdx, true);
    DT_CHECK_MSG(commitLxMem, "Memory allocation must be valid to commit.");
  }

  // Fill zero padding info for L3 transfers.
  fillTransferZeroPaddingInfo(mySDsc);

  // Fill multicast info in L3 transfers.
  fillTransferMulticastInfo(mySDsc);

  // Set start address, offset in allocations.
  fillAllocationStartAddrAndOffset(mySDsc);

  // Finalize the schedule tree by filling the loop address and offset in the
  // transfer nodes.
  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx)
    fillLoopOffsetsAndAddresses(mySDsc, dscIdx, true);

  dsc2::transformLxZeroPadInfoInScheduleTree(mySDsc);

  // Propagate coordinates.
  propagateCoordinate(mySDsc);

  // FIXME: A temporary solution to fill explicit transfer size for cross-core
  // partial reduction. Revisit for a proper solution.
  fillExplicitTransferSize(mySDsc);

  if (verbose > 0) {
    for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
      std::cout << "Schedule tree in DSC " << dscIdx << ":" << std::endl;
      const auto &dsc = mySDsc.dscs_.at(dscIdx);
      for (auto &node : dsc.scheduleTree_.traverseTreeDFS()) {
        std::cout << "Schedule node: " << node->name_ << std::endl;
      }
    }
  }

  for (int dscIdx = 0; dscIdx < mySDsc.dscs_.size(); ++dscIdx) {
    DT_CHECK_MSG(verifyScheduleTree(mySDsc.dscs_.at(dscIdx)),
                 "Invalid scheduleTree.");
  }
}

