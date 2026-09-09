// ================================================================================================
// DDC / L3-SCHEDULER CAMPAIGN - consolidated translation unit: ddc
//
// STAGE 2 of runDdc: ddc::Ddc(dscGlobal, ..).run_v1(sdsc)
//   dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29-41 is the call site.
//
// 178 of the campaign's 382 function bodies, from ddc/, extracted VERBATIM
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
// entry 073/382   level 0   scc 167   22 body lines
// unit: e073_addPropInfo
// authority: ddc/ddc.h:402
// original: void addPropInfo(dsc2::ScheduleNode* refNode, dsc2::ScheduleNode* nodeToFold, const std::vector<PrimaryDimTypes> dims, const std::string dataConnect = "", const bool refIsProducer = true, const bool scaleDown = false)
// class: CoordPropTracker
// rust home: crates/compiler/deeptools/src/schedule/ddc/mod.rs
// ------------------------------------------------------------------------------------------------
void e073_addPropInfo(dsc2::ScheduleNode* refNode,
                     dsc2::ScheduleNode* nodeToFold,
                     const std::vector<PrimaryDimTypes> dims,
                     const std::string dataConnect = "",
                     const bool refIsProducer = true,
                     const bool scaleDown = false)
{
      std::vector<PrimaryDimTypes> unseenDims;
      for (auto& dim : dims) {
        if (refsAdded_.count(nodeToFold) &&
            refsAdded_.at(nodeToFold).count(refNode) &&
            refsAdded_.at(nodeToFold).at(refNode).count(dim)) {
          // This propagation step has already been included.
          continue;
          ;
        }
        unseenDims.push_back(dim);
        refsAdded_[nodeToFold][refNode][dim] = 0;
      }
      if (unseenDims.empty()) {
        // No remaining dimension for propagation.
        return;
      }
      itemsToProcess_.push_back(dsc2::CoordPropInfoType{
          refNode, nodeToFold, dataConnect, refIsProducer,
          dsc2::CoordPropInfoType::PropStateType::NOT_PROCESSED, unseenDims,
          scaleDown});
    }

// ------------------------------------------------------------------------------------------------
// entry 074/382   level 0   scc 169   23 body lines
// unit: e074_retry
// authority: ddc/ddc.h:436
// original: void retry(const dsc2::CoordPropInfoType& propInfo, const std::vector<PrimaryDimTypes> dims)
// class: CoordPropTracker
// rust home: crates/compiler/deeptools/src/schedule/ddc/mod.rs
// ------------------------------------------------------------------------------------------------
void e074_retry(const dsc2::CoordPropInfoType& propInfo,
               const std::vector<PrimaryDimTypes> dims)
{
      itemsToProcess_.push_back(dsc2::CoordPropInfoType{
          propInfo.refNode, propInfo.nodeToFold, propInfo.dataConnect,
          propInfo.refIsProducer,
          dsc2::CoordPropInfoType::PropStateType::NOT_PROCESSED, dims});
      for (auto& dim : dims) {
        int retryCount = 0;
        if (refsAdded_[propInfo.nodeToFold].count(propInfo.refNode) &&
            refsAdded_[propInfo.nodeToFold].at(propInfo.refNode).count(dim)) {
          retryCount =
              refsAdded_[propInfo.nodeToFold].at(propInfo.refNode).at(dim);
          if (retryCount == 15) {
            DT_ERROR(
                "Retry threshold for propagation reached for " +
                propInfo.refNode->name_ + " -> " + propInfo.nodeToFold->name_ +
                ", dim= " + EnumsConversion::primaryDimToString.at(dim) + ".");
          }
          ++retryCount;
        }

        refsAdded_[propInfo.nodeToFold][propInfo.refNode][dim] = retryCount;
      }
    }

// ------------------------------------------------------------------------------------------------
// entry 075/382   level 0   scc 170   10 body lines
// unit: e075_getCurrItem
// authority: ddc/ddc.h:463
// original: bool getCurrItem(dsc2::CoordPropInfoType& nextCoordPropInfo)
// class: CoordPropTracker
// rust home: crates/compiler/deeptools/src/schedule/ddc/mod.rs
// ------------------------------------------------------------------------------------------------
bool e075_getCurrItem(dsc2::CoordPropInfoType& nextCoordPropInfo)
{
      ++currItemToProcess_;
      if (currItemToProcess_ >= itemsToProcess_.size()) {
        return false;
      }
      itemsToProcess_.at(currItemToProcess_).propState =
          dsc2::CoordPropInfoType::PropStateType::COMPLETE;
      nextCoordPropInfo = itemsToProcess_.at(currItemToProcess_);
      return true;
    }

// ------------------------------------------------------------------------------------------------
// entry 076/382   level 0   scc 113   29 body lines
// unit: e076_print
// authority: ddc/ddc.h:580
// original: void print(std::ostream& out) const
// class: RowGroupInfo
// rust home: crates/compiler/deeptools/src/schedule/ddc/mod.rs
// ------------------------------------------------------------------------------------------------
void e076_print(std::ostream& out) const
{
      out << "\nRowgroup: "
          << "\n  Category= ";
      switch (cat) {
        case (Category::ROW_TO_NONROW):
          out << "Row-to-NonRow";
          break;
        case (Category::ROW_TO_SAME_ROW):
          out << "Row-to-SameRow";
          break;
        case (Category::NONROW_TO_ROW):
          out << "NonRow-to-Row";
          break;
        case (Category::ROW_NORTH_SOUTH):
          out << "Row-North-South";
          break;
        case (Category::NO_BUNDLING):
          out << "No-Bundling";
          break;
        default:
          break;
      }
      out << "\n  Group elements:";
      for (auto& [node, row, beta] : nodeInfo) {
        out << " (" << node->name_ << ", row= " << row << ", beta=" << beta
            << ")";
      }
      out.flush();
    }

// ------------------------------------------------------------------------------------------------
// entry 077/382   level 0   scc 174   6 body lines
// unit: e077_printFoldParams
// authority: ddc/ddc.h:611
// original: void printFoldParams(std::vector<dsc2::FoldParamInfoType>& foldParams)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/mod.rs
// ------------------------------------------------------------------------------------------------
void e077_printFoldParams(std::vector<dsc2::FoldParamInfoType>& foldParams)
{
    for (auto& fpInfo : foldParams) {
      std::cout << "(" << fpInfo.alpha << ", " << fpInfo.beta << ", "
                << fpInfo.cardinality << ", " << fpInfo.foldDimLabel << ") ";
    }
  }

// ------------------------------------------------------------------------------------------------
// entry 078/382   level 0   scc 175   25 body lines
// unit: e078_dbgPrint
// authority: ddc/ddc_fold.cpp:20
// original: void dbgPrint(const dsc2::ComputeNode *computeNode)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e078_dbgPrint(const dsc2::ComputeNode *computeNode)
{
  std::cout << " ComputeNode: " << computeNode->name_ << "(" << computeNode
            << ") "
            << EnumsConversion::computeTypeToString.at(computeNode->type_)
            << " [";
  for (int i = 0, e = computeNode->inputsLdsAndLoopOffsets_.size(); i < e;
       ++i) {
    std::cout << " '"
              << EnumsConversion::senComponentsToString.at(
                     computeNode->inputs_.at(i))
              << "(" << computeNode->inputsLdsAndLoopOffsets_.at(i).dataConnect_
              << ")'";
  }
  std::cout << "] -> [";
  for (int i = 0, e = computeNode->outputsLdsAndLoopOffsets_.size(); i < e;
       ++i) {
    std::cout << " '"
              << EnumsConversion::senComponentsToString.at(
                     computeNode->outputs_.at(i))
              << "("
              << computeNode->outputsLdsAndLoopOffsets_.at(i).dataConnect_
              << ")'";
  }
  std::cout << "]";
}

// ------------------------------------------------------------------------------------------------
// entry 079/382   level 0   scc 176   16 body lines
// unit: e079_dbgPrint
// authority: ddc/ddc_fold.cpp:46
// original: void dbgPrint(const dsc2::TransferNode *transferNode)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e079_dbgPrint(const dsc2::TransferNode *transferNode)
{
  std::cout << " TransferNode: " << transferNode->name_ << "(" << transferNode
            << ") ['"
            << EnumsConversion::senComponentsToString.at(
                   transferNode->src_.unit_)
            << "(" << transferNode->srcLdsAndLoopOffsets_.dataConnect_
            << ")'] -> [";
  for (int i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e; ++i) {
    std::cout << " '"
              << EnumsConversion::senComponentsToString.at(
                     transferNode->dstVias_.at(i).loc_.unit_)
              << "(" << transferNode->dstLdsAndLoopOffsets_.at(i).dataConnect_
              << ")'";
  }
  std::cout << "]";
}

// ------------------------------------------------------------------------------------------------
// entry 080/382   level 0   scc 178   9 body lines
// unit: e080_allDimsCovered
// authority: ddc/ddc_fold.cpp:75
// original: bool allDimsCovered(const DesignSpaceConfig *currDsc, const dsc2::CoordinateType<CoordinateBaseType> &coord, const int ldsIdx)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
bool e080_allDimsCovered(const DesignSpaceConfig *currDsc,
                    const dsc2::CoordinateType<CoordinateBaseType> &coord,
                    const int ldsIdx)
{
  auto allDims = currDsc->getLayoutDims(ldsIdx);
  for (auto &dim : allDims) {
    if (!coord.coordinates_.count(dim)) {
      return false;
    }
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 081/382   level 0   scc 179   19 body lines
// unit: e081_buildFoldForBroadcastDim
// authority: ddc/ddc_fold.cpp:87
// original: void buildFoldForBroadcastDim(const DesignSpaceConfig *currDsc, dsc2::CoordinateType<CoordinateBaseType> &coord, const LabeledDsInfo &lds, PrimaryDimTypes dim, int scale)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e081_buildFoldForBroadcastDim(const DesignSpaceConfig *currDsc,
                              dsc2::CoordinateType<CoordinateBaseType> &coord,
                              const LabeledDsInfo &lds, PrimaryDimTypes dim,
                              int scale)
{
  std::string foldDimStr = EnumsConversion::primaryDimToString.at(dim);
  int elemArrCard = 1;
  if (scale == -2) {
    elemArrCard = currDsc->getCumulativeStickSizes(lds.dsType_).at(dim);
  }
  // Element arrangement fold
  coord.addFold(dim, dsc2::CoordinateCategory::ELEM_ARR_COORD, elemArrCard,
                "elem_arr_0", 0, 0, 0);
  // Build rowsplit fold.
  coord.addFold(dim, dsc2::CoordinateCategory::SPATIAL_COORD, 1,
                "rowsplit_fold_" + foldDimStr, elemArrCard, 0, 0);
  // Build corelet fold.
  coord.addFold(dim, dsc2::CoordinateCategory::SPATIAL_COORD, 1,
                "corelet_fold_" + foldDimStr, elemArrCard, 0, 0);
  // Build core workslice fold.
  coord.addFold(dim, dsc2::CoordinateCategory::SPATIAL_COORD, 1,
                "core_workslice_fold_" + foldDimStr, elemArrCard, 0, 0);
}

// ------------------------------------------------------------------------------------------------
// entry 082/382   level 0   scc 180   7 body lines
// unit: e082_getCompRowId
// authority: ddc/ddc_fold.cpp:110
// original: int getCompRowId(SenComponents comp)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
int e082_getCompRowId(SenComponents comp)
{
  int ptRowId = -1;
  if (EnumsConversion::senCompToRowId.count(comp)) {
    ptRowId = EnumsConversion::senCompToRowId.at(comp);
  }
  return ptRowId;
}

// ------------------------------------------------------------------------------------------------
// entry 083/382   level 0   scc 181   17 body lines
// unit: e083_getComponent
// authority: ddc/ddc_fold.cpp:118
// original: SenComponents getComponent(const dsc2::ScheduleNode *node, bool getSrc = true)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
SenComponents e083_getComponent(const dsc2::ScheduleNode *node, bool getSrc = true)
{
  if (node->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
    return static_cast<const dsc2::AllocateNode *>(node)->component_;
  } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    return static_cast<const dsc2::ComputeNode *>(node)->exUnit_;
  } else if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
    if (getSrc) {
      return static_cast<const dsc2::TransferNode *>(node)->src_.unit_;
    } else {
      return static_cast<const dsc2::TransferNode *>(node)
          ->dstVias_.at(0)
          .loc_.unit_;
    }
  } else {
    DT_ERROR("[getComponent] Unsupported node type: " + node->name_);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 084/382   level 0   scc 182   51 body lines
// unit: e084_getLayoutDimsFromNode
// authority: ddc/ddc_fold.cpp:136
// original: std::vector<PrimaryDimTypes> getLayoutDimsFromNode( const DesignSpaceConfig *currDsc, const dsc2::ScheduleNode *node, int inputPos = -1, int outputPos = -1)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
std::vector<PrimaryDimTypes> e084_getLayoutDimsFromNode(
    const DesignSpaceConfig *currDsc, const dsc2::ScheduleNode *node,
    int inputPos = -1, int outputPos = -1)
{
  std::vector<PrimaryDimTypes> nodeDims;
  if (node->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
    const dsc2::AllocateNode *allocNode =
        static_cast<const dsc2::AllocateNode *>(node);
    if (allocNode->ldsIdx_ != -1) {
      nodeDims = currDsc->getLayoutDims(allocNode->ldsIdx_);
    }
  } else if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
    const dsc2::TransferNode *transferNode =
        static_cast<const dsc2::TransferNode *>(node);
    if (inputPos != -1) {
      DT_CHECK(inputPos == 0);
      if (transferNode->srcLdsAndLoopOffsets_.myLdsIdx_ != -1) {
        nodeDims = currDsc->getLayoutDims(
            transferNode->srcLdsAndLoopOffsets_.myLdsIdx_);
      }
    } else if (outputPos != -1) {
      if (transferNode->dstLdsAndLoopOffsets_.at(outputPos).myLdsIdx_ != -1) {
        nodeDims = currDsc->getLayoutDims(
            transferNode->dstLdsAndLoopOffsets_.at(outputPos).myLdsIdx_);
      }
    } else {
      DT_ERROR(
          "[getLayoutDimsFromNode] Neither source nor destination was "
          "specified for transferNode " +
          node->name_ + ".");
    }
  } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    const dsc2::ComputeNode *computeNode =
        static_cast<const dsc2::ComputeNode *>(node);
    if (inputPos != -1) {
      if (computeNode->inputsLdsAndLoopOffsets_.at(inputPos).myLdsIdx_ != -1) {
        nodeDims = currDsc->getLayoutDims(
            computeNode->inputsLdsAndLoopOffsets_.at(inputPos).myLdsIdx_);
      }
    } else if (outputPos != -1) {
      if (computeNode->outputsLdsAndLoopOffsets_.at(outputPos).myLdsIdx_ !=
          -1) {
        nodeDims = currDsc->getLayoutDims(
            computeNode->outputsLdsAndLoopOffsets_.at(outputPos).myLdsIdx_);
      }
    } else {
      DT_ERROR(
          "[getLayoutDimsFromNode] Neither input nor output was "
          "specified for computeNode " +
          node->name_ + ".");
    }
  }
  return nodeDims;
}

// ------------------------------------------------------------------------------------------------
// entry 085/382   level 0   scc 183   7 body lines
// unit: e085_needToConsiderRowBundling
// authority: ddc/ddc_fold.cpp:190
// original: bool needToConsiderRowBundling( const dsc2::DataStage &coreDs, const dsc2::CoordinateType<CoordinateBaseType> &refCoord, const std::vector<PrimaryDimTypes> &workingDims)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
bool e085_needToConsiderRowBundling(
    const dsc2::DataStage &coreDs,
    const dsc2::CoordinateType<CoordinateBaseType> &refCoord,
    const std::vector<PrimaryDimTypes> &workingDims)
{
  if (coreDs.ss_.rowSplit_.empty()) {
    return false;
  }
  const PrimaryDimTypes rowSplitDim = coreDs.ss_.rowSplit_.begin()->first;
  return is_any_of(rowSplitDim, workingDims);
}

// ------------------------------------------------------------------------------------------------
// entry 086/382   level 0   scc 184   28 body lines
// unit: e086_isAllocateIncoming
// authority: ddc/ddc_fold.cpp:201
// original: bool isAllocateIncoming(const DesignSpaceConfig *currDsc, const dsc2::AllocateNode *allocNode, const dsc2::ScheduleNode *node)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
bool e086_isAllocateIncoming(const DesignSpaceConfig *currDsc,
                        const dsc2::AllocateNode *allocNode,
                        const dsc2::ScheduleNode *node)
{
  if (!allocNode->hasAllocUser(node)) {
    DT_ERROR("ScheduleNode " + node->name_ +
             " is not a user of the allocateNode " + allocNode->name_ + ".");
  }

  if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
    const dsc2::TransferNode *transferNode =
        static_cast<const dsc2::TransferNode *>(node);
    const dsc2::AllocateNode *srcAlloc = currDsc->getAllocation(
        transferNode->srcLdsAndLoopOffsets_, transferNode->src_.storage_, true);
    return (srcAlloc == allocNode);
  } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    const dsc2::ComputeNode *computeNode =
        static_cast<const dsc2::ComputeNode *>(node);
    for (int i = 0, e = computeNode->inputsLdsAndLoopOffsets_.size(); i < e;
         ++i) {
      const dsc2::AllocateNode *inputAlloc =
          currDsc->getAllocation(computeNode->inputsLdsAndLoopOffsets_.at(i),
                                 computeNode->inputs_.at(i), true);
      if (inputAlloc == allocNode) {
        return true;
      }
    }
    return false;
  }
  DT_ERROR("Unsupported nodeType for scheduleNode " + node->name_ + ".");
}

// ------------------------------------------------------------------------------------------------
// entry 087/382   level 0   scc 185   32 body lines
// unit: e087_findCommonAncestor
// authority: ddc/ddc_fold.cpp:232
// original: const dsc2::ScheduleNode *findCommonAncestor( const std::vector<dsc2::ScheduleNode *> &nodeList, std::vector<dsc2::ScheduleNode::NodeType> ancestorType)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
const dsc2::ScheduleNode *e087_findCommonAncestor(
    const std::vector<dsc2::ScheduleNode *> &nodeList,
    std::vector<dsc2::ScheduleNode::NodeType> ancestorType)
{
  std::vector<const dsc2::ScheduleNode *> candidateAncestors;
  std::map<const dsc2::ScheduleNode *, int> descendantCount;
  if (nodeList.empty()) {
    return nullptr;
  }
  const dsc2::ScheduleNode *currParent = nodeList.at(0)->getPrev();
  while (currParent) {
    if (is_any_of(currParent->nodeType_, ancestorType)) {
      candidateAncestors.push_back(currParent);
      descendantCount[currParent] = 0;
    }
    currParent = currParent->getPrev();
  }

  for (auto node : nodeList) {
    currParent = node->getPrev();
    while (currParent) {
      if (descendantCount.count(currParent)) {
        ++descendantCount.at(currParent);
      }
      currParent = currParent->getPrev();
    }
  }
  int nodeCount = nodeList.size();
  for (auto ancs : candidateAncestors) {
    if (descendantCount.at(ancs) == nodeCount) {
      return ancs;
    }
  }
  return nullptr;
}

// ------------------------------------------------------------------------------------------------
// entry 088/382   level 0   scc 186   25 body lines
// unit: e088_orderDescendants
// authority: ddc/ddc_fold.cpp:267
// original: void orderDescendants( const dsc2::ScheduleNode *commonAncestor, const std::vector<const dsc2::ScheduleNode *> &nodeList, std::vector<const dsc2::ScheduleNode *> &orderedNodeList)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e088_orderDescendants(
    const dsc2::ScheduleNode *commonAncestor,
    const std::vector<const dsc2::ScheduleNode *> &nodeList,
    std::vector<const dsc2::ScheduleNode *> &orderedNodeList)
{
  DT_CHECK_MSG(commonAncestor->nodeType_ == dsc2::ScheduleNode::BLOCK,
               "[orderDescendants] Common ancestor node " +
                   commonAncestor->name_ + " is not a block node.");
  std::vector<const dsc2::ScheduleNode *> ancestorChildren;
  const dsc2::ScheduleNode *currNode = nullptr;
  for (auto node : nodeList) {
    // Find the child of the commonAncestor that contains node.
    currNode = node;
    while (currNode) {
      if (currNode->getPrev() == commonAncestor) {
        ancestorChildren.push_back(currNode);
        break;
      }
      currNode = currNode->getPrev();
    }
  }
  const dsc2::BlockNode *commonAncestorBlock =
      static_cast<const dsc2::BlockNode *>(commonAncestor);
  for (auto child : commonAncestorBlock->getNextView(SenComponents::ALL)) {
    if (is_any_of(child, ancestorChildren)) {
      orderedNodeList.push_back(child);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 089/382   level 0   scc 187   87 body lines
// unit: e089_constructAllocElemArrLayout
// authority: ddc/ddc_fold.cpp:296
// original: void constructAllocElemArrLayout( const DesignSpaceConfig *currDsc, const DesignSpaceConfigGlobal &dscGlobal, const dsc2::AllocateNode *allocNode, PrimaryDimTypes coordDim, std::vector<dsc2::FoldParamInfoType> &foldParams, int &numElemArrFoldsOfAllocNode, dsc2::LoopDistributionParamPerNodeType &loopParamInfo)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e089_constructAllocElemArrLayout(
    const DesignSpaceConfig *currDsc, const DesignSpaceConfigGlobal &dscGlobal,
    const dsc2::AllocateNode *allocNode, PrimaryDimTypes coordDim,
    std::vector<dsc2::FoldParamInfoType> &foldParams,
    int &numElemArrFoldsOfAllocNode,
    dsc2::LoopDistributionParamPerNodeType &loopParamInfo)
{
  if (allocNode->ldsIdx_ < 0) {
    DT_ERROR("AllocateNode " + allocNode->name_ +
             " must have a valid index to labeledDs.");
  }
  const auto &myLds = currDsc->labeledDs_.at(allocNode->ldsIdx_);
  const auto &primDsInfo = currDsc->primaryDsInfo_.at(myLds.dsType_);
  const auto &layoutOrder = allocNode->layoutDimOrder_;
  const auto &maxDimSizes = allocNode->maxDimSizes_;
  const auto stickDimSizeVect = currDsc->getStickSizes(myLds.dsType_);
  const int stickDimsNum = stickDimSizeVect.size();
  int maxStickSize = 1, stickSize = 1;
  for (const auto &stickDimSize : primDsInfo.stickSize_) {
    maxStickSize *= stickDimSize;
  }
  if (EnumsConversion::senCompToGenericComp.count(allocNode->component_) &&
      EnumsConversion::senCompToGenericComp.at(allocNode->component_) == L0LU) {
    maxStickSize /= dscGlobal.sysDef.numPTRows;
  }
  // build sizes for stick dimensions
  std::unordered_map<PrimaryDimTypes, int> stickDimSize;
  for (int i = 0; i < stickDimsNum && stickSize < maxStickSize; i++) {
    auto [dim, size] = stickDimSizeVect[i];
    size = std::min(size, maxStickSize / stickSize);
    stickSize *= size;
    stickDimSize[dim] =
        (stickDimSize.count(dim) ? stickDimSize[dim] : 1) * size;
  }
  // build sizes for layout dimensions
  int currElemArrIndex = foldParams.size() - 1;
  int outerMostElemArrIndex = foldParams.size() - numElemArrFoldsOfAllocNode;
  while (currElemArrIndex >= outerMostElemArrIndex &&
         foldParams.at(currElemArrIndex).cardinality == 1) {
    --currElemArrIndex;
  }
  if (currElemArrIndex < outerMostElemArrIndex) {
    // All element arrangement levels have cardinality 1. Can not split any
    // element arranagement.
    return;
  }
  int currElemArrLevel = foldParams.size() - currElemArrIndex;

  int64_t &remainingDimSizes = foldParams.at(currElemArrIndex).cardinality;
  std::unordered_set<PrimaryDimTypes> layoutDims(layoutOrder.begin(),
                                                 layoutOrder.end());
  const auto datastage = currDsc->getSizeDataStageForNode(allocNode, allocNode);
  int innerCard = 1;
  for (int i = 0; i < layoutOrder.size(); i++) {
    auto dim = layoutOrder.at(i);
    if (dim != coordDim) {
      continue;
    }
    auto maxDimSize = maxDimSizes.at(i);
    int size = remainingDimSizes;
    if (stickDimSize.count(dim)) {
      // size = std::ceil(float(size) / stickDimSize[dim]);
    }
    if (maxDimSize > 0 && size > maxDimSize) {
      // in case that the dim size is larger than the max allowed size
      // specified by user, set dim size to maxDimSize and reset
      // remainingDimSizes
      DT_CHECK(remainingDimSizes % maxDimSize == 0);
      remainingDimSizes /= maxDimSize;
      // size = maxDimSize;
      dsc2::FoldParamInfoType newFoldParam;
      newFoldParam.cardinality = maxDimSize;
      newFoldParam.alpha = datastage.ss_.calculate_padded(
          coordDim, innerCard, allocNode->padding_, false);
      newFoldParam.beta = 0;
      newFoldParam.foldDimLabel = "elem_arr_layout_split";
      foldParams.insert(foldParams.begin() + currElemArrIndex + 1,
                        newFoldParam);
      ++numElemArrFoldsOfAllocNode;
      innerCard *= maxDimSize;
      // Adjust elemArr indices from previous iterations.
      for (auto &[loopNode, loopInfo] : loopParamInfo) {
        if (loopInfo.count(coordDim) &&
            loopInfo.at(coordDim).relatedElemArrLevel >= currElemArrLevel) {
          ++loopInfo.at(coordDim).relatedElemArrLevel;
        }
      }
    } else {
      break;
    }
  }
  foldParams.at(currElemArrIndex).alpha *= innerCard;
}

// ------------------------------------------------------------------------------------------------
// entry 090/382   level 0   scc 188   34 body lines
// unit: e090_combineContigousLevels
// authority: ddc/ddc_fold.cpp:395
// original: void combineContigousLevels(std::vector<dsc2::FoldParamInfoType> &foldParams, int start = 0, int end = -1)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e090_combineContigousLevels(std::vector<dsc2::FoldParamInfoType> &foldParams,
                            int start = 0, int end = -1)
{
  //                     Outermost   innermost
  //                        |         |
  //   foldParams: 0 1 ... start ... end
  //                       <-----------|
  //                       Scan from inner to outer.
  if (end == -1) {
    end = foldParams.size() - 1;
  }
  int i = end;
  while (i > start) {
    if (foldParams.at(i - 1).alpha ==
        foldParams.at(i).alpha * foldParams.at(i).cardinality) {
      //   [Outer] elemArr(i-1) : ( alpha_(i) * card_(i), card_(i-1))
      //   [Inner] elemArr(i)   : ( alpha_(i)           , card_(i) )
      //     =>
      //   [Outer] elemArr(i-1) : ( alpha_(i) * card_(i), card_(i-1) * card_(i))
      //   [Inner] elemArr(i)   : Removed
      foldParams.at(i - 1).alpha = foldParams.at(i).alpha;
      foldParams.at(i - 1).cardinality *= foldParams.at(i).cardinality;
      foldParams.at(i - 1).beta += foldParams.at(i).beta;

      // Remove element arrangement fold at fold index i.
      // Caution: Ensure that this erase operation does not invalidate other
      // operations on foldParams in the current loop.
      foldParams.erase(foldParams.begin() + i);
    } else if (foldParams.at(i - 1).cardinality == 1) {
      // Remove folds of cardinality 1.
      foldParams.erase(foldParams.begin() + i - 1);
    }
    // Move to the next outer fold level.
    --i;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 091/382   level 0   scc 189   42 body lines
// unit: e091_matchDataStream
// authority: ddc/ddc_fold.cpp:430
// original: bool matchDataStream(const DesignSpaceConfig *currDsc, const dsc2::CoordPropInfoType &coordPropInfo, const dsc2::DataInfo &di, const SenComponents unit, bool checkRefForAllocate)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
bool e091_matchDataStream(const DesignSpaceConfig *currDsc,
                     const dsc2::CoordPropInfoType &coordPropInfo,
                     const dsc2::DataInfo &di, const SenComponents unit,
                     bool checkRefForAllocate)
{
  if (coordPropInfo.dataConnect != "" &&
      coordPropInfo.dataConnect == di.dataConnect_) {
    return true;
  }
  const dsc2::AllocateNode *allocNodeToMatch = nullptr;
  auto comp = SenComponents::NO_COMPONENT;
  bool otherNodeIsCompute = false;
  if (checkRefForAllocate &&
      coordPropInfo.refNode->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
    allocNodeToMatch =
        static_cast<const dsc2::AllocateNode *>(coordPropInfo.refNode);
    otherNodeIsCompute =
        coordPropInfo.nodeToFold->nodeType_ == dsc2::ScheduleNode::COMPUTE;
    comp = static_cast<const dsc2::ComputeNode *>(coordPropInfo.nodeToFold)
               ->exUnit_;
  } else if (!checkRefForAllocate && coordPropInfo.nodeToFold->nodeType_ ==
                                         dsc2::ScheduleNode::ALLOCATE) {
    allocNodeToMatch =
        static_cast<const dsc2::AllocateNode *>(coordPropInfo.nodeToFold);
    otherNodeIsCompute =
        coordPropInfo.refNode->nodeType_ == dsc2::ScheduleNode::COMPUTE;
    comp =
        static_cast<const dsc2::ComputeNode *>(coordPropInfo.refNode)->exUnit_;
  }
  if (!allocNodeToMatch) {
    return false;
  }
  if (otherNodeIsCompute && comp != SenComponents::LXLU &&
      allocNodeToMatch->ldsIdx_ != -1 &&
      currDsc->labeledDs_.at(allocNodeToMatch->ldsIdx_).scaledLdsCategory_ ==
          LabeledDsInfo::ScaledLdsCategory::SCALE_TENSOR) {
    // Compute nodes consume/produce the value tensors. The scale tensors
    // are implicitly connected through the corresponding value tensors.
    //
    // Special case:
    //   An FMUL computeNode is used on LXLU for auto conversion of FP4 to
    //   BFloat16. This computeNode accesses the scale allocation directly.
    allocNodeToMatch = dsc2::getValueAllocation(currDsc, allocNodeToMatch);
  }
  return (allocNodeToMatch == currDsc->getAllocation(di, unit, true));
}

// ------------------------------------------------------------------------------------------------
// entry 092/382   level 0   scc 197   18 body lines
// unit: e092_getNumElementsInPTSlice
// authority: ddc/ddc_fold.cpp:1519
// original: int Ddc::getNumElementsInPTSlice(int ldsIdx, PrimaryDimTypes dim)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
int e092_getNumElementsInPTSlice(int ldsIdx, PrimaryDimTypes dim)
{
  // Determine how many elements are placed in one slice. Distribution of
  // element arrangements within a slice does not need scaling by the number
  // of PT rows.
  if (ldsIdx < 0 || ldsIdx > currDsc->labeledDs_.size()) {
    DT_ERROR("[getNumElementsInPTSlice] Invalid labeledDs index " +
             std::to_string(ldsIdx) + ".");
  }

  auto stickSizesPerDim = currDsc->getCumulativeStickSizes(
      currDsc->labeledDs_.at(ldsIdx).dsType_, false, false, true,
      dscGlobal.sysDef.numPTRows);
  auto it = stickSizesPerDim.find(dim);
  if (it != stickSizesPerDim.end())
    return it->second;
  else
    return 1;
}

// ------------------------------------------------------------------------------------------------
// entry 093/382   level 0   scc 209   41 body lines
// unit: e093_buildSpatialFold
// authority: ddc/ddc_fold.cpp:2109
// original: void Ddc::buildSpatialFold( dsc2::ScheduleNode *node, const PrimaryDimTypes &currDim, const PadType &currPadType, dsc2::CoordinateType<CoordinateBaseType> &nodeCoordinates)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e093_buildSpatialFold(
    dsc2::ScheduleNode *node, const PrimaryDimTypes &currDim,
    const PadType &currPadType,
    dsc2::CoordinateType<CoordinateBaseType> &nodeCoordinates)
{
  auto &coreDataStage = currDsc->dataStageParam_.at(metadata.core_dstgid).ss_;
  CoordinateBaseType coreletAlpha, coreletBeta;
  int64_t stride = 1;
  if (is_any_of(currPadType, PadType::PADDED_WZEROPAD, PadType::PADDED_FULLSPAN,
                PadType::PADDED_FULLSPAN_WUNNEEDED)) {
    if (coreDataStage.paddingSizes_.count(currDim)) {
      stride = coreDataStage.paddingSizes_.at(currDim).stride_;
    }
  }

  if (currDsc->dataStageParam_.at(metadata.core_dstgid)
          .ss_.coreletSplit_.count(currDim)) {
    int64_t firstCoreletSize = coreDataStage.coreletSplit_.at(currDim).at(0);
    coreletAlpha = firstCoreletSize * stride;
    coreletBeta = 0;
  } else {
    coreletAlpha = 0;
    coreletBeta = 0;
  }

  // Corelet spatial fold
  nodeCoordinates.addFold(currDim, dsc2::CoordinateCategory::SPATIAL_COORD,
                          currDsc->numCoreletsUsed_, "corelet_fold_dim",
                          coreletAlpha, coreletBeta, 0);

  // Core workslice fold
  // Assumption: Each core is assigned the same number of elements.
  CoordinateBaseType coreAlpha =
      currDsc->dataStageParam_.at(metadata.core_dstgid)
          .ss_.dataStageDimToVal_compView_st(
              currDim, SenComponents::NO_COMPONENT, -1, {});
  coreAlpha *= stride;

  CoordinateBaseType coreBeta = 0;
  nodeCoordinates.addFold(currDim, dsc2::CoordinateCategory::SPATIAL_COORD,
                          sdsc_->numWkSlicesPerDim_.at(currDim),
                          "core_workslice_fold_dim", coreAlpha, coreBeta, 0);

  return;
}

// ------------------------------------------------------------------------------------------------
// entry 094/382   level 0   scc 198   6 body lines
// unit: e094_getDefaultRowSplitFold
// authority: ddc/ddc_fold.cpp:2154
// original: void getDefaultRowSplitFold(dsc2::FoldParamInfoType &resultFoldParams)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e094_getDefaultRowSplitFold(dsc2::FoldParamInfoType &resultFoldParams)
{
  resultFoldParams.alpha = 0;
  resultFoldParams.beta = 0;
  resultFoldParams.cardinality = 1;
  resultFoldParams.foldDimLabel = "rowsplit_fold";
}

// ------------------------------------------------------------------------------------------------
// entry 095/382   level 0   scc 190   12 body lines
// unit: e095_gatherFoldParams
// authority: ddc/ddc_fold.cpp:2224
// original: void Ddc::gatherFoldParams(const FoldManager<CoordinateBaseType> &cfm, std::vector<dsc2::FoldParamInfoType> &foldParams)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e095_gatherFoldParams(const FoldManager<CoordinateBaseType> &cfm,
                           std::vector<dsc2::FoldParamInfoType> &foldParams)
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
// entry 096/382   level 0   scc 213   3 body lines
// unit: e096_updateMin
// authority: ddc/ddc_metadata.h:40
// original: inline void updateMin(float newVal)
// class: Constraints
// rust home: crates/compiler/deeptools/src/schedule/ddc/metadata.rs
// ------------------------------------------------------------------------------------------------
void e096_updateMin(float newVal)
{
        min_ = min_ ? std::max(*min_, newVal) : newVal;
      }

// ------------------------------------------------------------------------------------------------
// entry 097/382   level 0   scc 214   3 body lines
// unit: e097_updateMax
// authority: ddc/ddc_metadata.h:43
// original: inline void updateMax(float newVal)
// class: Constraints
// rust home: crates/compiler/deeptools/src/schedule/ddc/metadata.rs
// ------------------------------------------------------------------------------------------------
void e097_updateMax(float newVal)
{
        max_ = max_ ? std::min(*max_, newVal) : newVal;
      }

// ------------------------------------------------------------------------------------------------
// entry 098/382   level 0   scc 215   3 body lines
// unit: e098_updateValues
// authority: ddc/ddc_metadata.h:46
// original: inline void updateValues(std::set<float> newVals)
// class: Constraints
// rust home: crates/compiler/deeptools/src/schedule/ddc/metadata.rs
// ------------------------------------------------------------------------------------------------
void e098_updateValues(std::set<float> newVals)
{
        values_ = values_ ? set_intersect(*values_, newVals) : newVals;
      }

// ------------------------------------------------------------------------------------------------
// entry 099/382   level 0   scc 216   23 body lines
// unit: e099_dump
// authority: ddc/ddc_metadata.h:49
// original: void dump() const
// class: Constraints
// rust home: crates/compiler/deeptools/src/schedule/ddc/metadata.rs
// ------------------------------------------------------------------------------------------------
void e099_dump() const
{
        std::cerr << "mustBeMultiple_= " << (mustBeMultiple_ ? "T " : "F ");
        std::cerr << "loopDimKind_= "
                  << (loopDimKind_ == MetaDimKind::Count
                          ? "NOT_SET"
                          : EnumsConversion::metaDimKindToString.at(
                                loopDimKind_))
                  << " ";
        std::cerr << ", min_= ";
        if (min_)
          std::cerr << min_.value() << " ";
        else
          std::cerr << "-inf ";
        std::cerr << ", max_= ";
        if (max_)
          std::cerr << max_.value() << " ";
        else
          std::cerr << "inf ";
        std::cerr << ", values_= {";
        if (values_)
          for (auto v : values_.value()) std::cerr << v << " ";
        std::cerr << "}\n";
      }

// ------------------------------------------------------------------------------------------------
// entry 100/382   level 0   scc 217   5 body lines
// unit: e100_insertProducer
// authority: ddc/ddc_metadata.h:147
// original: void insertProducer(dsc2::ScheduleNode* node)
// class: DataConnect
// rust home: crates/compiler/deeptools/src/schedule/ddc/metadata.rs
// ------------------------------------------------------------------------------------------------
void e100_insertProducer(dsc2::ScheduleNode* node)
{
      if (!is_any_of(node, producers_)) {
        producers_.push_back(node);
      }
    }

// ------------------------------------------------------------------------------------------------
// entry 101/382   level 0   scc 218   5 body lines
// unit: e101_insertConsumer
// authority: ddc/ddc_metadata.h:153
// original: void insertConsumer(dsc2::ScheduleNode* node)
// class: DataConnect
// rust home: crates/compiler/deeptools/src/schedule/ddc/metadata.rs
// ------------------------------------------------------------------------------------------------
void e101_insertConsumer(dsc2::ScheduleNode* node)
{
      if (!is_any_of(node, consumers_)) {
        consumers_.push_back(node);
      }
    }

// ------------------------------------------------------------------------------------------------
// entry 102/382   level 0   scc 114   11 body lines
// unit: e102_print
// authority: ddc/ddc_metadata.h:166
// original: void print(std::ostream& outs) const
// class: DataConnect
// rust home: crates/compiler/deeptools/src/schedule/ddc/metadata.rs
// ------------------------------------------------------------------------------------------------
void e102_print(std::ostream& outs) const
{
      outs << " Consumers= [";
      for (auto consumer : consumers_) {
        outs << " " << consumer->name_;
      }
      outs << "] Producers= [";
      for (auto producer : producers_) {
        outs << " " << producer->name_;
      }
      outs << "]\n";
    }

// ------------------------------------------------------------------------------------------------
// entry 103/382   level 0   scc 219   13 body lines
// unit: e103_getLoops
// authority: ddc/ddc_metadata.h:179
// original: std::unordered_set<const dsc2::LoopNode*> getLoops( const std::vector<dsc2::ScheduleNode*>& baseNodes) const
// class: DataConnect
// rust home: crates/compiler/deeptools/src/schedule/ddc/metadata.rs
// ------------------------------------------------------------------------------------------------
std::unordered_set<const dsc2::LoopNode*> e103_getLoops(
        const std::vector<dsc2::ScheduleNode*>& baseNodes) const
{
      std::unordered_set<const dsc2::LoopNode*> loops;
      for (auto baseNode : baseNodes) {
        auto loop = baseNode->getOwnerLoop();
        while (loop != nullptr) {
          if (!loops.insert(loop).second) {
            break;
          }
          loop = loop->getOwnerLoop();
        }
      }
      return loops;
    }

// ------------------------------------------------------------------------------------------------
// entry 104/382   level 0   scc 4   4 body lines
// unit: e104_clear
// authority: ddc/ddc_metadata.h:225
// original: void clear()
// class: Metadata
// rust home: crates/compiler/deeptools/src/schedule/ddc/metadata.rs
// ------------------------------------------------------------------------------------------------
void e104_clear()
{
    this->~Metadata();
    new (this) Metadata();
  }

// ------------------------------------------------------------------------------------------------
// entry 105/382   level 0   scc 229   17 body lines
// unit: e105_isReleventComputeToPackStickDim
// authority: ddc/ddc_transformation.cpp:693
// original: std::pair<bool, int> Ddc::isReleventComputeToPackStickDim( dsc2::ComputeNode *&releventComputeNode)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
std::pair<bool, int> e105_isReleventComputeToPackStickDim(
    dsc2::ComputeNode *&releventComputeNode)
{
  bool relevent = false;
  int numberOfInputs = 1;
  for (auto &child : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::COMPUTE}, ALL, -1, -1)) {
    DT_CHECK(child->nodeType_ == dsc2::ScheduleNode::COMPUTE &&
             "Compute Node is expected");
    auto cn = static_cast<const dsc2::ComputeNode *>(child);
    if (cn->type_ == ComputeOpType::RECIPROCAL ||
        cn->type_ == ComputeOpType::LAYERNORMSCALE) {
      relevent = true;
      releventComputeNode = const_cast<dsc2::ComputeNode *>(cn);
      if (cn->type_ == ComputeOpType::LAYERNORMSCALE) numberOfInputs = 2;
    }
  }
  return std::make_pair(relevent, numberOfInputs);
}

// ------------------------------------------------------------------------------------------------
// entry 106/382   level 0   scc 230   72 body lines
// unit: e106_findReleventStickPackingTransfers
// authority: ddc/ddc_transformation.cpp:712
// original: void Ddc::findReleventStickPackingTransfers( dsc2::ComputeNode *releventComputeNode, std::vector<dsc2::TransferNode *> &releventInputTransfers, dsc2::TransferNode *&releventOutputTransfer, std::vector<int> &releventTransferDstIdx)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
void e106_findReleventStickPackingTransfers(
    dsc2::ComputeNode *releventComputeNode,
    std::vector<dsc2::TransferNode *> &releventInputTransfers,
    dsc2::TransferNode *&releventOutputTransfer,
    std::vector<int> &releventTransferDstIdx)
{
  auto isConnected = [&](const std::vector<dsc2::DataInfo> &dataInfos,
                         const std::string &dataConnect) {
    for (auto &dataInfo : dataInfos)
      if (dataInfo.dataConnect_ == dataConnect) return true;
    return false;
  };
  for (auto &child : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::TRANSFER}, ALL, -1, -1)) {
    DT_CHECK(child->nodeType_ == dsc2::ScheduleNode::TRANSFER &&
             "Transfer Node is expected");
    auto transferNode = static_cast<dsc2::TransferNode *>(child);
    if (transferNode->srcLdsAndLoopOffsets_.myLdsIdx_ < 0) continue;
    bool lx_to_exUnit = false;
    if (transferNode->src_.unit_ == LXLU) {
      int idx = 0;
      for (auto dst : transferNode->dstVias_) {
        if (dst.loc_.unit_ == releventComputeNode->exUnit_) {
          lx_to_exUnit = true;
          releventTransferDstIdx.push_back(idx);
          break;
        }
        idx++;
      }
    }
    bool exUnit_to_lx = false;
    if (transferNode->src_.unit_ == releventComputeNode->exUnit_) {
      int idx = 0;
      for (auto dst : transferNode->dstVias_) {
        if (dst.loc_.unit_ == LXSU) {
          exUnit_to_lx = true;
          releventTransferDstIdx.push_back(idx);
          break;
        }
        idx++;
      }
    }
    if (!lx_to_exUnit && !exUnit_to_lx) continue;

    // Check data connect of opaque vs the transfers
    bool relevent = false;
    if (lx_to_exUnit) {
      for (auto &dstDataInfo : transferNode->dstLdsAndLoopOffsets_)
        if (isConnected(releventComputeNode->inputsLdsAndLoopOffsets_,
                        dstDataInfo.dataConnect_))
          relevent = true;
    } else {
      if (isConnected(releventComputeNode->outputsLdsAndLoopOffsets_,
                      transferNode->srcLdsAndLoopOffsets_.dataConnect_))
        relevent = true;
    }

    // get the tensor, check scale.at(i) = -2 where i is index of stick
    // dimension
    bool skip = false;
    for (auto dst : transferNode->dstLdsAndLoopOffsets_)
      if (dst.myLdsIdx_ != transferNode->srcLdsAndLoopOffsets_.myLdsIdx_)
        skip = true;
    if (skip) continue;
    auto &myLds =
        currDsc->labeledDs_.at(transferNode->srcLdsAndLoopOffsets_.myLdsIdx_);
    skip = true;
    for (auto scale : myLds.scale_)
      if (scale == -2) skip = false;
    if (skip) continue;

    if (lx_to_exUnit)
      releventInputTransfers.emplace_back(transferNode);
    else
      releventOutputTransfer = transferNode;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 107/382   level 0   scc 231   19 body lines
// unit: e107_isEligibleCompute
// authority: ddc/ddc_transformation.cpp:789
// original: bool Ddc::isEligibleCompute( std::vector<dsc2::TransferNode *> &releventInputTransfers, int numberOfInputs, std::vector<int> &ldsInputIdx)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e107_isEligibleCompute(
    std::vector<dsc2::TransferNode *> &releventInputTransfers,
    int numberOfInputs, std::vector<int> &ldsInputIdx)
{
  auto getLdsType = [&](dsc2::TransferNode *tr) {
    return currDsc->labeledDs_.at(tr->srcLdsAndLoopOffsets_.myLdsIdx_).dsType_;
  };

  // LDS type and scale of all input transfers has to be the same to continue
  if (releventInputTransfers.size() > 1) {
    for (auto &tr : releventInputTransfers) {
      if (getLdsType(releventInputTransfers.at(0)) != getLdsType(tr))
        return false;
    }
    for (int i = 0; i < numberOfInputs; i++) {
      if (currDsc->labeledDs_.at(ldsInputIdx.at(0)).scale_ !=
          currDsc->labeledDs_.at(ldsInputIdx.at(i)).scale_)
        return false;
    }
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 108/382   level 0   scc 240   29 body lines
// unit: e108_cloneComputeForOffsetAdjustment
// authority: ddc/ddc_transformation.cpp:1356
// original: void Ddc::cloneComputeForOffsetAdjustment(dsc2::ComputeNode *node)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
void e108_cloneComputeForOffsetAdjustment(dsc2::ComputeNode *node)
{
  std::vector<dsc2::AllocateNode *> spread_added_allocs;
  for (int idx = 0; idx < node->repetitionWithOffset_.forOutputs_.size();
       idx++) {
    for (int rep = 0; rep < node->repetitionWithOffset_.forOutputs_.at(idx) - 1;
         rep++) {
      dsc2::ComputeNode *newNode =
          static_cast<dsc2::ComputeNode *>(node->clone());
      node->getMutableParent()->addChildNode(newNode, false, node);
      // resetting repetition for the cloned code
      newNode->repetitionWithOffset_.forOutputs_.at(idx) = 1;

      metadata.nodeCloningMap_[static_cast<dsc2::ScheduleNode *>(node)]
          .push_back(newNode);
      auto &myLds = currDsc->labeledDs_.at(
          node->outputsLdsAndLoopOffsets_.at(idx).myLdsIdx_);
      auto alloc = myLds.memOrg_.at(node->outputs_.at(idx)).allocateNode_;
      alloc->allocUsers_.push_back({newNode, 1});

      if (std::find(spread_added_allocs.begin(), spread_added_allocs.end(),
                    alloc) != spread_added_allocs.end())
        continue;
      auto dim = alloc->layoutDimOrder_.at(0);
      auto spread = node->repetitionWithOffset_.forOutputs_.at(idx);
      alloc->gapStickSpread_[dim] = spread;
      spread_added_allocs.push_back(alloc);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 109/382   level 0   scc 293   12 body lines
// unit: e109_transformAComputeNodeForInterSliceRestickify
// authority: ddc/ddc_transformation.cpp:2383
// original: bool Ddc::transformAComputeNodeForInterSliceRestickify( dsc2::ComputeNode *compNode, int none_trivial_input_idx, PrimaryDimTypes dimForLoop)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e109_transformAComputeNodeForInterSliceRestickify(
    dsc2::ComputeNode *compNode, int none_trivial_input_idx,
    PrimaryDimTypes dimForLoop)
{
  // unroll transfernode and create masked compute nodes
  compNode->name_ += "_masked";
  auto loopNode = compNode->getParentDimLoop(dimForLoop);
  for (int clId = 0; clId < currDsc->numCoreletsUsed_DSC2_; clId++) {
    for (auto &[dim, kind] : loopNode->dims_) {
      compNode->instrAttribute_.computeMaskLoopOffsets_[clId][loopNode][dim] =
          1;
    }
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 110/382   level 0   scc 246   50 body lines
// unit: e110_constructAllocation
// authority: ddc/ddc_transformation_util.cpp:20
// original: dsc2::AllocateNode *Ddc::constructAllocation( const dsc2::DataInfo &di, const SenComponents storage, const PaddingFormType &paddingPerDim, const dsc2::ScheduleNode *userNode)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
dsc2::AllocateNode *e110_constructAllocation(
    const dsc2::DataInfo &di, const SenComponents storage,
    const PaddingFormType &paddingPerDim, const dsc2::ScheduleNode *userNode)
{
  if (di.myLdsIdx_ < 0) {
    DT_ERROR("DataInfo is not associated with any labeled dataStage.");
  }
  if (dsc2::memories.count(storage) == 0) {
    DT_ERROR("Error in setAllocation: Requested storage " +
             EnumsConversion::senComponentsToString.at(storage) +
             " is not a memory.");
  }

  if (currDsc->labeledDs_.at(di.myLdsIdx_).memOrg_.count(storage)) {
    const auto &memOrg =
        currDsc->labeledDs_.at(di.myLdsIdx_).memOrg_.at(storage);
    if (memOrg.allocateNode_) {
      DT_ERROR("Error in constructAllocation: LabeldDs[" +
               std::to_string(di.myLdsIdx_) + "]" +
               " already has an allocation for memOrg_ entry for DataLocation "
               "storge " +
               EnumsConversion::senComponentsToString.at(storage));
    }
  }

  const auto &labelledDS = currDsc->labeledDs_.at(di.myLdsIdx_);
  auto allocNode = new dsc2::AllocateNode();
  allocNode->layoutDimOrder_ = currDsc->getLayoutDims(labelledDS.ldsIdx_);
  DT_CHECK_MSG(std::set(allocNode->layoutDimOrder_.begin(),
                        allocNode->layoutDimOrder_.end())
                       .size() == allocNode->layoutDimOrder_.size(),
               "Handling of external allocations with repeated dimensions is "
               "not yet implemented");
  allocNode->maxDimSizes_.resize(allocNode->layoutDimOrder_.size(), -1);
  allocNode->ldsIdx_ = di.myLdsIdx_;
  allocNode->component_ = storage;
  allocNode->name_ =
      "allocate_lds" + std::to_string(allocNode->ldsIdx_) + "_" +
      EnumsConversion::senComponentsToString.at(allocNode->component_);
  auto &allocMeta =
      metadata.newAllocations_[allocNode->component_].ldsIdxAndAllocNode;
  DT_CHECK(allocMeta.find(allocNode->ldsIdx_) == allocMeta.end());
  allocMeta[allocNode->ldsIdx_] = allocNode;

  // Associate allocation with labelled datastage.
  // Construct a new memOrg entry.
  auto &memOrg = currDsc->labeledDs_.at(di.myLdsIdx_).memOrg_[storage];
  memOrg.isPresent = true;
  memOrg.allocateNode_ = allocNode;
  allocNode->padding_ = paddingPerDim;
  allocNode->addAllocUser(userNode);
  return allocNode;
}

// ------------------------------------------------------------------------------------------------
// entry 111/382   level 0   scc 226   41 body lines
// unit: e111_reduceUsersOrDeleteAllocationAndMetadata
// authority: ddc/ddc_transformation_util.cpp:73
// original: void Ddc::reduceUsersOrDeleteAllocationAndMetadata( const dsc2::DataInfo &di, const SenComponents storage, const dsc2::ScheduleNode *userNode)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
void e111_reduceUsersOrDeleteAllocationAndMetadata(
    const dsc2::DataInfo &di, const SenComponents storage,
    const dsc2::ScheduleNode *userNode)
{
  auto allocNode = currDsc->getMutableAllocation(di, storage);
  if (!allocNode) {
    DT_ERROR("Error in reduceUsersOrDeleteAllocationAndMetadata: LabeldDs[" +
             std::to_string(di.myLdsIdx_) +
             "]'s memOrg_ entry for DataLocation storge " +
             EnumsConversion::senComponentsToString.at(storage) +
             " does not have any allocation.");
  }

  SenComponents comp = allocNode->component_;
  std::string name = allocNode->name_;
  int ldsOrConstIdx =
      allocNode->ldsIdx_ >= 0 ? allocNode->ldsIdx_ : allocNode->constIdx_;
  bool canDelete = !isExternalNode(allocNode);
  if (!currDsc->reduceUsersOrDeleteAllocation(di, storage, userNode,
                                              canDelete)) {
    // AllocateNode still exists in schedule tree because
    //   - either some ScheduleNode still uses the allocateNode or
    //   - the allocateNode is external and was not deleted.
    return;
  }

  // The allocateNode has no users in the schedule tree.
  // Remove allocateNode from metadata.
  auto allocationMetaEntry = metadata.newAllocations_.find(comp);
  if (allocationMetaEntry == metadata.newAllocations_.end()) {
    DT_ERROR(
        "Error in reduceUsersOrDeleteAllocationAndMetadata: Missing metadata "
        "for component " +
        EnumsConversion::senComponentsToString.at(comp));
  }
  auto &allocMeta = metadata.newAllocations_.at(comp).ldsIdxAndAllocNode;
  if (allocMeta.find(ldsOrConstIdx) == allocMeta.end()) {
    DT_ERROR(
        "Error in reduceUsersOrDeleteAllocationAndMetadata: Missing metadata "
        "for allocateNode " +
        name);
  }
  allocMeta.erase(ldsOrConstIdx);
}

// ------------------------------------------------------------------------------------------------
// entry 112/382   level 0   scc 234   8 body lines
// unit: e112_constructDatastage
// authority: ddc/ddc_transformation_util.cpp:117
// original: int Ddc::constructDatastage()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
int e112_constructDatastage()
{
  auto id = currDsc->dataStageParam_.size();
  while (currDsc->dataStageParam_.count(id)) id++;
  auto &newDs = currDsc->dataStageParam_[id];
  newDs.ss_.name_ = std::to_string(id);
  newDs.el_.name_ = std::to_string(id) + "el";
  return id;
}

// ------------------------------------------------------------------------------------------------
// entry 113/382   level 0   scc 232   11 body lines
// unit: e113_constructDatastage
// authority: ddc/ddc_transformation_util.cpp:126
// original: int Ddc::constructDatastage(dsc2::DataStage &refDataStage)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
int e113_constructDatastage(dsc2::DataStage &refDataStage)
{
  auto id = currDsc->dataStageParam_.size();
  while (currDsc->dataStageParam_.count(id)) id++;
  auto &newDs =
      currDsc->dataStageParam_.emplace(id, refDataStage).first->second;
  newDs.ss_.name_ = std::to_string(id);
  newDs.el_.name_ = std::to_string(id) + "el";
  // Do we need to update the metadata info?
  // metadata.datastages_[id] = metadata.Datastage version of newDs;
  return id;
}

// ------------------------------------------------------------------------------------------------
// entry 114/382   level 0   scc 233   18 body lines
// unit: e114_constructLoopNode
// authority: ddc/ddc_transformation_util.cpp:138
// original: dsc2::LoopNode *Ddc::constructLoopNode(int numId, int denId, std::vector<PrimaryDimAndKind> dims, dsc2::ScheduleNode *baseNode)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
dsc2::LoopNode *e114_constructLoopNode(int numId, int denId,
                                       std::vector<PrimaryDimAndKind> dims,
                                       dsc2::ScheduleNode *baseNode)
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
// entry 115/382   level 0   scc 251   14 body lines
// unit: e115_collectLoopReferences
// authority: ddc/ddc_transformation_util.cpp:318
// original: void Ddc::collectLoopReferences( const dsc2::ScheduleNode *node, std::set<const dsc2::LoopNode *> &referencedLoops)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
void e115_collectLoopReferences(
    const dsc2::ScheduleNode *node,
    std::set<const dsc2::LoopNode *> &referencedLoops)
{
  if (node->nodeType_ == dsc2::ScheduleNode::CONDITION) {
    const dsc2::ConditionNode *condNode =
        static_cast<const dsc2::ConditionNode *>(node);
    if (condNode->hasCoreClCond()) {
      return;
    }
    for (auto &orIt : condNode->loopCond_.twoLevelOrOfAnds_) {
      for (auto &andIt : orIt) {
        referencedLoops.insert(andIt.loopComp_);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 116/382   level 0   scc 247   49 body lines
// unit: e116_getPaddingPerDim
// authority: ddc/ddc_transformation_util.cpp:709
// original: PaddingFormType Ddc::getPaddingPerDim(const dsc2::TransferNode *transferNode, bool forSrc) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
PaddingFormType e116_getPaddingPerDim(const dsc2::TransferNode *transferNode,
                                      bool forSrc) const
{
  PaddingFormType result;
  const dsc2::AllocateNode *refAllocNode = nullptr;
  if (forSrc) {
    refAllocNode = currDsc->getAllocation(transferNode->srcLdsAndLoopOffsets_,
                                          transferNode->src_.storage_, true);
    if (!refAllocNode) {
      for (size_t i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e;
           ++i) {
        refAllocNode = currDsc->getAllocation(
            transferNode->dstLdsAndLoopOffsets_.at(i),
            transferNode->dstVias_.at(i).loc_.storage_, true);
        if (refAllocNode) {
          break;
        }
      }
    }
  } else {
    for (size_t i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e;
         ++i) {
      refAllocNode = currDsc->getAllocation(
          transferNode->dstLdsAndLoopOffsets_.at(i),
          transferNode->dstVias_.at(i).loc_.storage_, true);
      if (refAllocNode) {
        break;
      }
    }
    if (!refAllocNode) {
      refAllocNode = currDsc->getAllocation(transferNode->srcLdsAndLoopOffsets_,
                                            transferNode->src_.storage_, true);
    }
  }
  if (refAllocNode) {
    result = refAllocNode->padding_;
  }

  // In case, access pattern is specified on the transfer operation, overwrite
  // the corresponding padding information in the result.
  if (metadata.datatransfers_.find(transferNode) !=
      metadata.datatransfers_.end()) {
    auto &accessPatternPerDim =
        metadata.datatransfers_.at(transferNode).getAccessPatternList();
    for (auto &[dim, accPat] : accessPatternPerDim) {
      PadType padding = forSrc ? accPat.first : accPat.second;
      result.setPadding(dim, padding);
    }
  }
  return result;
}

// ------------------------------------------------------------------------------------------------
// entry 117/382   level 0   scc 220   25 body lines
// unit: e117_getNodeDescription
// authority: ddc/ddc_transformation_util.cpp:1094
// original: std::string Ddc::getNodeDescription(const dsc2::ScheduleNode *node) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
std::string e117_getNodeDescription(const dsc2::ScheduleNode *node) const
{
  std::string description = node->name_;

  if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
    auto transferNode = static_cast<const dsc2::TransferNode *>(node);
    description = description + " [ src: " +
                  transferNode->srcLdsAndLoopOffsets_.dataConnect_ + " dst: ";
    for (auto dstIt : transferNode->dstLdsAndLoopOffsets_) {
      description = description + " " + dstIt.dataConnect_;
    }
    description = description + " ]";
  } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    auto computeNode = static_cast<const dsc2::ComputeNode *>(node);
    description = description + " [ input: ";
    for (auto inpIt : computeNode->inputsLdsAndLoopOffsets_) {
      description = description + " " + inpIt.dataConnect_;
    }
    description = description + " output: ";
    for (auto outIt : computeNode->outputsLdsAndLoopOffsets_) {
      description = description + " " + outIt.dataConnect_;
    }
    description = description + "]";
  }
  return description;
}

// ------------------------------------------------------------------------------------------------
// entry 118/382   level 0   scc 242   114 body lines
// unit: e118_cloneForPeSfpWorkSplit
// authority: ddc/ddc_transformation_util.cpp:1306
// original: dsc2::AllocateNode *Ddc::cloneForPeSfpWorkSplit( dsc2::AllocateNode *node, bool skipMetadataUpdate /* = false */)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
dsc2::AllocateNode *e118_cloneForPeSfpWorkSplit(
    dsc2::AllocateNode *node, bool skipMetadataUpdate /* = false */)
{
  if (!toggleMap.count(node->component_)) {
    return nullptr;
  }

  if (metadata.nodeCloningMap_.count(node)) {
    DT_ERROR("[PeSfpWorkSplit] allocateNode " + node->name_ +
             " has already been cloned.");
  }

  if (node->ldsIdx_ < 0 && node->constIdx_ < 0) {
    DT_ERROR("[PeSfpWorkSplit] Can not clone allocateNode " + node->name_ +
             " as none of ldsIdx_ and constIdx_ is set.");
  }
  dsc2::AllocateNode *newNode =
      static_cast<dsc2::AllocateNode *>(node->clone());
  std::string suffix =
      is_any_of(node->component_, SenComponents::PELRF, SenComponents::PESTATE)
          ? "_sfp_parallel"
          : "_pe_parallel";
  newNode->name_ += suffix;
  newNode->component_ = toggleMap.at(node->component_);

  newNode->clearAllocUsers();
  newNode->tempStorageForCompute_ = nullptr;

  if (!skipMetadataUpdate) {
    // Update DDC Metadata by following the placement of the original allocNode
    // (the usage of the allocNode is not known at the time of cloning). The
    // Opaque operation case has to be handled during the cloning of the opaque
    // operation. The pointer to ComputeNode is needed to access the metaData
    // Map compAndAllocateNode.
    auto &metadataNew = metadata.newAllocations_[newNode->component_];

    const auto &metadataRef = metadata.newAllocations_.at(node->component_);
    if (node->ldsIdx_ >= 0) {
      auto &allocMeta = metadataNew.ldsIdxAndAllocNode;
      DT_CHECK(allocMeta.find(newNode->ldsIdx_) == allocMeta.end());
      allocMeta[newNode->ldsIdx_] = newNode;
      if (!metadataRef.ldsIdxAndAllocNode.count(node->ldsIdx_) ||
          metadataRef.ldsIdxAndAllocNode.at(node->ldsIdx_) != node) {
        DT_ERROR("[PeSfpWorkSplit] Can not clone allocateNode " + node->name_ +
                 " as the original allocateNode was not found in the DDC "
                 "metadata(ldxIDx).");
      }
    } else if (node->constIdx_ >= 0) {
      auto &allocMeta = metadataNew.consIdAndAllocNode;
      DT_CHECK(allocMeta.find(newNode->constIdx_) == allocMeta.end());
      allocMeta[newNode->constIdx_] = newNode;
      if (!metadataRef.consIdAndAllocNode.count(node->constIdx_) ||
          metadataRef.consIdAndAllocNode.at(node->constIdx_) != node) {
        DT_ERROR("[PeSfpWorkSplit] Can not clone allocateNode " + node->name_ +
                 " as the original allocateNode was not found in the DDC "
                 "metadata(constIDx).");
      }
    }
  }

  // Add the new node to the schedule tree immediately after node.
  node->getMutableParent()->addChildNode(newNode, false, node);
  metadata.nodeCloningMap_[node].push_back(newNode);

  if (newNode->ldsIdx_ >= 0) {
    // Ensure that labeled DS has memOrg entry for the reference component (from
    // the original allocateNode)
    if (!currDsc->labeledDs_.at(node->ldsIdx_)
             .memOrg_.count(node->component_)) {
      DT_ERROR("[PeSfpWorkSplit] Can not clone allocateNode " + node->name_ +
               " as labeledDs[" + std::to_string(node->ldsIdx_) +
               "] does not have memOrg_ entry for reference storage " +
               EnumsConversion::senComponentsToString.at(node->component_) +
               ".");
    }

    // Add a new memOrg entry for the new component in case this does not exist
    // yet.
    auto &newMemOrg = currDsc->labeledDs_.at(newNode->ldsIdx_).memOrg_;
    if (!newMemOrg.count(newNode->component_)) {
      newMemOrg[newNode->component_] = newMemOrg[node->component_];
      currDsc->labeledDs_.at(newNode->ldsIdx_)
          .memOrg_.at(newNode->component_)
          .allocateNode_ = nullptr;
    }

    // Ensure that labeled DS does not have allocateNode for the memOrg entry
    // for the new component
    if (currDsc->labeledDs_.at(newNode->ldsIdx_)
            .memOrg_.at(newNode->component_)
            .allocateNode_) {
      DT_ERROR(
          "[PeSfpWorkSplit] Can not clone allocateNode " + newNode->name_ +
          " as labeledDs[" + std::to_string(newNode->ldsIdx_) +
          "] already includes an allocation for memOrg_ entry for storage " +
          EnumsConversion::senComponentsToString.at(newNode->component_) + ".");
    }
    currDsc->labeledDs_.at(newNode->ldsIdx_)
        .memOrg_.at(newNode->component_)
        .allocateNode_ = newNode;
  } else {
    // Constant
    if (currDsc->constantInfo_.at(newNode->constIdx_)
            .allocations_.count(newNode->component_)) {
      DT_ERROR("[PeSfpWorkSplit] Can not clone allocateNode " + node->name_ +
               " as constantInfo[" + std::to_string(node->constIdx_) +
               "] already includes an allocation for memOrg_ entry for new "
               "storage " +
               EnumsConversion::senComponentsToString.at(newNode->component_) +
               ".");
    }
    currDsc->constantInfo_.at(newNode->constIdx_)
        .allocations_[newNode->component_] = newNode;
  }
  return newNode;
}

// ------------------------------------------------------------------------------------------------
// entry 119/382   level 0   scc 244   115 body lines
// unit: e119_cloneForPeSfpWorkSplit
// authority: ddc/ddc_transformation_util.cpp:1422
// original: dsc2::ComputeNode *Ddc::cloneForPeSfpWorkSplit(dsc2::ComputeNode *node)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
dsc2::ComputeNode *e119_cloneForPeSfpWorkSplit(dsc2::ComputeNode *node)
{
  std::vector<SenComponents> comps = {SenComponents::PE, SenComponents::SFP};
  if (!is_any_of(node->exUnit_, comps)) {
    return nullptr;
  }

  for (auto input : node->inputs_) {
    if (is_any_of(input, comps)) {
      // Input to PE comes from SFP or vice versa. Can not split work across PE
      // and SFP.
      DT_ERROR("[Pe/Sfp worksplit] Can not split computeNode " + node->name_ +
               " as input is sent from " +
               EnumsConversion::senComponentsToString.at(input) + " to " +
               EnumsConversion::senComponentsToString.at(node->exUnit_) + ".");
    }
  }

  for (auto output : node->outputs_) {
    if (is_any_of(output, comps)) {
      // Output from PE goes to SFP or vice versa. Can not split work across PE
      // and SFP.
      DT_ERROR("[Pe/Sfp worksplit] Can not split computeNode " + node->name_ +
               " as output is sent from " +
               EnumsConversion::senComponentsToString.at(node->exUnit_) +
               " to " + EnumsConversion::senComponentsToString.at(output) +
               ".");
    }
  }

  if (metadata.nodeCloningMap_.count(node)) {
    DT_ERROR("[PeSfpWorkSplit] computeNode " + node->name_ +
             " has already been cloned.");
  }

  dsc2::ComputeNode *newNode = static_cast<dsc2::ComputeNode *>(node->clone());
  std::string suffix =
      (node->exUnit_ == SenComponents::PE) ? "_sfp_parallel" : "_pe_parallel";
  newNode->name_ += suffix;
  newNode->exUnit_ = (node->exUnit_ == SenComponents::PE) ? SenComponents::SFP
                                                          : SenComponents::PE;

  // Update allocation tracking info.
  dsc2::AllocateNode *allocNode = nullptr;

  if (!newNode->isOpaqueOp_) {
    for (int i = 0, e = newNode->inputsLdsAndLoopOffsets_.size(); i < e; ++i) {
      if (!newNode->inputsLdsAndLoopOffsets_.at(i).dataConnect_.empty()) {
        newNode->inputsLdsAndLoopOffsets_.at(i).dataConnect_ += suffix;
      }

      if (toggleMap.count(newNode->inputs_.at(i))) {
        newNode->inputs_.at(i) = toggleMap.at(newNode->inputs_.at(i));
        if ((allocNode = currDsc->getMutableAllocation(
                 newNode->inputsLdsAndLoopOffsets_.at(i),
                 newNode->inputs_.at(i), true))) {
          allocNode->addAllocUser(newNode);
        }
      }
    }

    for (int i = 0, e = newNode->outputsLdsAndLoopOffsets_.size(); i < e; ++i) {
      if (!newNode->outputsLdsAndLoopOffsets_.at(i).dataConnect_.empty()) {
        newNode->outputsLdsAndLoopOffsets_.at(i).dataConnect_ += suffix;
      }

      if (toggleMap.count(newNode->outputs_.at(i))) {
        newNode->outputs_.at(i) = toggleMap.at(newNode->outputs_.at(i));
        if ((allocNode = currDsc->getMutableAllocation(
                 newNode->outputsLdsAndLoopOffsets_.at(i),
                 newNode->outputs_.at(i), true))) {
          allocNode->addAllocUser(newNode);
        }
      }
    }
  } else {
    metadata.opaqueOps_[newNode] = metadata.opaqueOps_.at(node);
    DT_CHECK(metadata.nodeCloningMap_
                 .at(metadata.opaqueOps_.at(node).internalRegAlloc_)
                 .size() == 1);
    dsc2::AllocateNode *newAllocNode = static_cast<dsc2::AllocateNode *>(
        metadata.nodeCloningMap_
            .at(metadata.opaqueOps_.at(node).internalRegAlloc_)
            .at(0));
    metadata.opaqueOps_.at(newNode).internalRegAlloc_ = newAllocNode;
    metadata.newAllocations_[newAllocNode->component_]
        .compAndAllocNode[newNode] = newAllocNode;
    newAllocNode->tempStorageForCompute_ = newNode;
    newAllocNode->addAllocUser(newNode);

    auto &newInOutRegsMetadata =
        metadata.opaqueOps_.at(newNode).inOutRegAllocs_;
    for (auto &[name, allocNode] : newInOutRegsMetadata) {
      DT_CHECK(metadata.nodeCloningMap_.at(allocNode).size() == 1);
      allocNode = static_cast<dsc2::AllocateNode *>(
          metadata.nodeCloningMap_.at(allocNode).at(0));
      allocNode->addAllocUser(newNode);
    }

    for (auto &dc : newNode->instrAttribute_.input_data_connects_) {
      if (!dc.empty()) {
        dc += suffix;
      }
    }
    for (auto &dc : newNode->instrAttribute_.output_data_connects_) {
      if (!dc.empty()) {
        dc += suffix;
      }
    }
  }

  // Add the new node to the schedule tree immediately after node.
  node->getMutableParent()->addChildNode(newNode, false, node);
  metadata.nodeCloningMap_[node].push_back(newNode);
  return newNode;
}

// ------------------------------------------------------------------------------------------------
// entry 120/382   level 0   scc 223   32 body lines
// unit: e120_storageOrDatastreamIsExternal
// authority: ddc/ddc_transformation_util.cpp:1652
// original: bool Ddc::storageOrDatastreamIsExternal(const dsc2::DataInfo &dataInfo, SenComponents storage, bool isIncoming) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e120_storageOrDatastreamIsExternal(const dsc2::DataInfo &dataInfo,
                                        SenComponents storage,
                                        bool isIncoming) const
{
  const dsc2::AllocateNode *allocNode;
  auto &dc = dataInfo.dataConnect_;
  std::string streamType = isIncoming ? "Incoming" : "Outgoing";
  allocNode = currDsc->getAllocation(dataInfo, storage, true);
  if (allocNode && isExternalNode(allocNode)) {
    if (transformationReportLevel_ > 2) {
      std::cerr << "\n  " << streamType << " datastream [data_connect=" << dc
                << "] is associated with external allocation "
                << allocNode->name_ << ".";
    }
    return true;
  }

  // DataConnects for constants may not have entries in the metadata.
  if (metadata.dataConnects_.count(dc)) {
    auto &nodeList = isIncoming ? metadata.dataConnects_.at(dc).producers_
                                : metadata.dataConnects_.at(dc).consumers_;
    for (auto &node : nodeList) {
      if (isExternalNode(node)) {
        if (transformationReportLevel_ > 2) {
          std::cerr << "\n  " << streamType
                    << " datastream [data_connect=" << dc
                    << "] is associated with external node " << node->name_
                    << ".";
        }
        return true;
      }
    }
  }
  return false;
}

// ------------------------------------------------------------------------------------------------
// entry 121/382   level 0   scc 300   8 body lines
// unit: e121_relatedToExternalNodes
// authority: ddc/ddc_transformation_util.cpp:1748
// original: bool Ddc::relatedToExternalNodes(const dsc2::SyncNode *node) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e121_relatedToExternalNodes(const dsc2::SyncNode *node) const
{
  for (auto otherEnd : node->otherEndOfTheSignals_) {
    if (isExternalNode(otherEnd)) {
      return true;
    }
  }
  return false;
}

// ------------------------------------------------------------------------------------------------
// entry 122/382   level 0   scc 206   30 body lines
// unit: e122_isNodeRelatedToComps
// authority: ddc/ddc_transformation_util.cpp:1778
// original: bool Ddc::isNodeRelatedToComps(const dsc2::ScheduleNode *node, const std::vector<SenComponents> &comps, SenComponents &nodeComp)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e122_isNodeRelatedToComps(const dsc2::ScheduleNode *node,
                               const std::vector<SenComponents> &comps,
                               SenComponents &nodeComp)
{
  nodeComp = SenComponents::NO_COMPONENT;
  if (node->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
    if (is_any_of(static_cast<const dsc2::AllocateNode *>(node)->component_,
                  comps)) {
      nodeComp = static_cast<const dsc2::AllocateNode *>(node)->component_;
      return true;
    }
  } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    if (is_any_of(static_cast<const dsc2::ComputeNode *>(node)->exUnit_,
                  comps)) {
      nodeComp = static_cast<const dsc2::ComputeNode *>(node)->exUnit_;
      return true;
    }
  } else if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
    if (is_any_of(static_cast<const dsc2::TransferNode *>(node)->src_.unit_,
                  comps)) {
      nodeComp = static_cast<const dsc2::TransferNode *>(node)->src_.unit_;
      return true;
    }
    for (auto &dstVia :
         static_cast<const dsc2::TransferNode *>(node)->dstVias_) {
      if (is_any_of(dstVia.loc_.unit_, comps)) {
        nodeComp = dstVia.loc_.unit_;
        return true;
      }
    }
  }
  return false;
}

// ------------------------------------------------------------------------------------------------
// entry 123/382   level 0   scc 238   57 body lines
// unit: e123_updateNodesWithNewLds
// authority: ddc/ddc_transformation_util.cpp:1919
// original: void Ddc::updateNodesWithNewLds(int newLdsIdx, int oldLdsIdx, dsc2::ScheduleNode *startNode)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
void e123_updateNodesWithNewLds(int newLdsIdx, int oldLdsIdx,
                                dsc2::ScheduleNode *startNode)
{
  // Use newLds for all childs of start node
  for (auto &child : currDsc->scheduleTree_.traverseTreeDFSMutable(
           startNode,
           {dsc2::ScheduleNode::ALLOCATE, dsc2::ScheduleNode::TRANSFER,
            dsc2::ScheduleNode::COMPUTE},
           ALL, -1, -1)) {
    if (child->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      auto cn = static_cast<dsc2::ComputeNode *>(child);
      if (cn->isOpaqueOp_) {
        auto &ldsIdx = metadata.opaqueOps_.at(cn).ldsIdx_;
        if (ldsIdx == oldLdsIdx) ldsIdx = newLdsIdx;
      }
      for (auto &in : cn->inputsLdsAndLoopOffsets_) {
        if (in.myLdsIdx_ == oldLdsIdx) in.myLdsIdx_ = newLdsIdx;
      }
      for (auto &out : cn->outputsLdsAndLoopOffsets_) {
        if (out.myLdsIdx_ == oldLdsIdx) out.myLdsIdx_ = newLdsIdx;
      }
    } else if (child->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      auto tn = static_cast<dsc2::TransferNode *>(child);
      if (tn->srcLdsAndLoopOffsets_.myLdsIdx_ == oldLdsIdx)
        tn->srcLdsAndLoopOffsets_.myLdsIdx_ = newLdsIdx;
      for (auto &out : tn->dstLdsAndLoopOffsets_) {
        if (out.myLdsIdx_ == oldLdsIdx) out.myLdsIdx_ = newLdsIdx;
      }
    } else if (child->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
      auto an = static_cast<dsc2::AllocateNode *>(child);
      if (an->ldsIdx_ == oldLdsIdx) {
        an->ldsIdx_ = newLdsIdx;
        // *** find memorg of old lds, and replace the pointer to allocate node
        // to new lds.
        auto &memorgOld = currDsc->labeledDs_.at(oldLdsIdx).memOrg_;
        auto &memorgNew = currDsc->labeledDs_.at(newLdsIdx).memOrg_;
        for (auto &compMemOrg : memorgOld) {
          if (compMemOrg.second.allocateNode_ == an) {
            memorgNew[compMemOrg.first] = compMemOrg.second;
            compMemOrg.second.allocateNode_ = nullptr;
          }
        }
        for (auto &compAlloc : metadata.newAllocations_) {
          auto &ldsToAlloc = compAlloc.second.ldsIdxAndAllocNode;
          std::vector<int> entries_to_be_deleted;
          for (auto &ldsAlloc : ldsToAlloc) {
            if (ldsAlloc.second == an) {
              ldsToAlloc[newLdsIdx] = an;
              entries_to_be_deleted.push_back(ldsAlloc.first);
            }
          }
          for (auto i : entries_to_be_deleted) ldsToAlloc.erase(i);
        }
      }
    } else {
      DT_ERROR("the node has to be either compute, transfer or allocate.");
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 124/382   level 0   scc 302   10 body lines
// unit: e124_getLdsOrConstNameOfAllocNode
// authority: ddc/ddcv1.cpp:20
// original: std::string Ddc::getLdsOrConstNameOfAllocNode(dsc2::AllocateNode* anode)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
std::string e124_getLdsOrConstNameOfAllocNode(dsc2::AllocateNode* anode)
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
// entry 125/382   level 0   scc 303   101 body lines
// unit: e125_minimizeAllocations
// authority: ddc/ddcv1.cpp:30
// original: void Ddc::minimizeAllocations(bool has_auto_shuffling)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// NOTE: Part of the ddcv1.cpp:30-360 placement span that reginit.rs hand-transcribes. The authority
//       is this body.
// ------------------------------------------------------------------------------------------------
void e125_minimizeAllocations(bool has_auto_shuffling)
{
  struct Interval {
    int start;
    int end;
  };
  std::map<dsc2::ScheduleNode*, Interval> live_range;
  std::vector<std::tuple<dsc2::ScheduleNode*, Interval, bool>>
      live_range_ordered_list;
  std::map<int, const dsc2::ScheduleNode*> index_to_node;
  std::map<const dsc2::ScheduleNode*, int> node_to_index;
  int index = 0;
  for (auto& node :
       currDsc->scheduleTree_.traverseTreeDFSMutable(nullptr, {})) {
    index_to_node[index] = node;
    node_to_index[node] = index;
    index++;
  }
  for (auto& node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::ALLOCATE})) {
    auto* allocateNode = static_cast<dsc2::AllocateNode*>(node);
    int idx_start = INT_MAX;
    int idx_end = 0;
    for (auto [userNode, n] : allocateNode->allocUsers_) {
      if (node_to_index.at(userNode) < idx_start)
        idx_start = node_to_index.at(userNode);
      if (node_to_index.at(userNode) > idx_end)
        idx_end = node_to_index.at(userNode);
    }
    Interval range = {idx_start, idx_end};
    live_range[node] = range;
    live_range_ordered_list.push_back(std::make_tuple(node, range, false));
  }

  std::sort(live_range_ordered_list.begin(), live_range_ordered_list.end(),
            [&](std::tuple<dsc2::ScheduleNode*, Interval, bool>& lrA,
                std::tuple<dsc2::ScheduleNode*, Interval, bool>& lrB) {
              return std::get<1>(lrA).start < std::get<1>(lrB).start;
            });
  /*
  for (auto index_node : index_to_node) {
    std::cout << index_node.first << " : " << index_node.second << " -> "
              << index_node.second->name_ << "\n";
  }
  for (auto node_live_range : live_range) {
    std::cout << node_live_range.first << " : [" << node_live_range.second.start
              << ", " << node_live_range.second.end << ")  -> "
              << node_live_range.first->name_ << "\n";
  }
  */
  // overlap [As, Ae) and [Bs, Be)?
  auto overlapInterval = [](Interval intervalA, Interval intervalB) {
    if ((intervalA.end > intervalB.start && intervalA.end <= intervalB.end) ||
        (intervalA.start >= intervalB.start && intervalA.start < intervalB.end))
      return true;
    return false;
  };
  auto overlap = [&](dsc2::ScheduleNode* node,
                     std::vector<dsc2::AllocateNode*>& shadow_allocs) {
    for (auto alloc : shadow_allocs)
      if (overlapInterval(live_range.at(node), live_range.at(alloc)))
        return true;
    return false;
  };
  for (auto& [node, range, node_added] : live_range_ordered_list) {
    auto* allocateNode = static_cast<dsc2::AllocateNode*>(node);
    if (has_auto_shuffling && (allocateNode->component_ == SFPLRF ||
                               allocateNode->component_ == PELRF ||
                               allocateNode->component_ == PTXRF)) {
      if (allocateNode->ldsIdx_ == -1) continue;
      // if (allocateNode->isOpaque) continue;
      // if allocUsers_ has a compute with is computeNode->isOpaqueOp_ = true ->
      // continue;
      for (auto& shadow_allocs : metadata.shadowAllocations_) {
        if (node_added) break;
        if (shadow_allocs.size() == 0) continue;
        if (allocateNode->component_ != shadow_allocs.at(0)->component_)
          continue;
        // if (allocateNode->getPrev() != shadow_allocs.at(0)->getPrev())
        // continue;
        if (overlap(node, shadow_allocs)) continue;
        if (std::find(shadow_allocs.begin(), shadow_allocs.end(),
                      allocateNode) == shadow_allocs.end()) {
          shadow_allocs.push_back(allocateNode);
        }
        node_added = true;
      }
    }
    if (!node_added) {
      metadata.shadowAllocations_.push_back({allocateNode});
    }
  }
  /*
  for (auto& shadow_allocs : metadata.shadowAllocations_) {
    std::cout << "entry:\n";
    for (auto alloc : shadow_allocs) {
      std::cout << "  " << alloc << " -> " << alloc->name_ << "\n";
    }
  }
  std::cout << "end\n";
  */
}

// ------------------------------------------------------------------------------------------------
// entry 126/382   level 0   scc 305   115 body lines
// unit: e126_populateUnitTimeTransfers
// authority: ddc/ddcv1.cpp:439
// original: void Ddc::populateUnitTimeTransfers()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e126_populateUnitTimeTransfers()
{
  for (auto& node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::TRANSFER})) {
    auto* transfNode = static_cast<dsc2::TransferNode*>(node);
    int numElemLimit = -1;
    if (auto dtMetaIt = metadata.datatransfers_.find(transfNode);
        dtMetaIt != metadata.datatransfers_.end())
      numElemLimit = dtMetaIt->second.force_num_elements_;
    const auto transfType = transfNode->getTransferType();
    if (transfType == dsc2::TransferNode::TransferType::CONSTANT_TO_CONSTANT) {
      // special handling for constant transfers
      // DT_CHECK_MSG(transfNode->src_.unit_ == SenComponents::CONSTANT,
      //              "Expect CONSTANT as src unit.");
      const auto& constant = currDsc->constantInfo_.at(
          transfNode->srcLdsAndLoopOffsets_.constantId_);
      const int numElemInConst = constant.data_.getSingleData().size();
      transfNode->replicationFactor_ =
          8 * dscGlobal.sysDef.bytesPerStick / numElemInConst /
          EnumsConversion::dataFormatsToBitWidth.at(constant.dataFormat_);
      if (numElemLimit > 0) {
        if (numElemLimit % numElemInConst != 0) {
          DT_ERROR(
              "Number of elements requested is not compatible with the "
              "constant selected in transfer: " +
              transfNode->name_);
        }
        transfNode->replicationFactor_ = std::min(
            transfNode->replicationFactor_, numElemLimit / numElemInConst);
      }
      continue;
    }
    const bool constantToTensor =
        transfType == dsc2::TransferNode::TransferType::CONSTANT_TO_TENSOR;
    const bool tensorToTensor =
        transfType == dsc2::TransferNode::TransferType::TENSOR_TO_TENSOR;
    const bool noTransToTensor =
        transfType == dsc2::TransferNode::TransferType::NO_TRANSFER_TO_TENSOR;
    const bool noTransFromTensor =
        transfType == dsc2::TransferNode::TransferType::NO_TRANSFER_FROM_TENSOR;
    DT_CHECK_MSG((constantToTensor || tensorToTensor || noTransToTensor ||
                  noTransFromTensor),
                 "Unexpected transfer type.");
    int ldsIdx = ((constantToTensor || noTransToTensor)
                      ? transfNode->dstLdsAndLoopOffsets_.front().myLdsIdx_
                      : transfNode->srcLdsAndLoopOffsets_.myLdsIdx_);
    auto unit = ((constantToTensor || noTransToTensor)
                     ? transfNode->dstVias_.front().loc_.unit_
                     : transfNode->src_.unit_);
    if (unit == NO_COMPONENT || ldsIdx < 0) continue;

    const auto& lds = currDsc->labeledDs_.at(ldsIdx);
    bool l0SliceOnly = EnumsConversion::senCompToGenericComp.at(unit) == L0LU;
    auto stickSizes = currDsc->getStickSizes(
        lds.dsType_, false, false, l0SliceOnly, dscGlobal.sysDef.numPTRows);
    bool do2BSplat =
        (is_any_of(-2, lds.scale_) && unit == LXLU && stickSizes.size() == 1) ||
        transfNode->src_.unit_ == SenComponents::CONSTANT;
    if (transfNode->src_.unit_ != SenComponents::CONSTANT && unit == LXLU &&
        stickSizes.size() > 1) {
      do2BSplat = true;
      for (auto dimSize : stickSizes) {
        if (lds.scale_.at(currDsc->getDimIndexInLayoutOrder(
                lds.dsType_, dimSize.first)) == -2)
          continue;
        do2BSplat = false;
      }
    }
    int elemSoFar = 1;
    // assume unitTimeTransfer is continuous within a stick boundary
    // chunk-strided load will be evaluated at the end of
    // exploreAssignDataStages.
    for (int i = 0; i < stickSizes.size() &&
                    (numElemLimit < 0 || elemSoFar < numElemLimit);
         i++) {
      int size = stickSizes[i].second;
      elemSoFar *= size;
      if (numElemLimit >= 0 && elemSoFar > numElemLimit) {
        if (elemSoFar % numElemLimit != 0) {
          DT_ERROR(
              "Number of elements speficied for transfer is incompatible with "
              "stick composition in node: " +
              transfNode->name_);
        }
        size /= (elemSoFar / numElemLimit);
      }
      transfNode->unitTimeTransferChunkSize_.push_back(
          {{stickSizes[i].first, do2BSplat ? 1 : size}, i, i});
    }
    if (do2BSplat) {
      if (stickSizes.size() > 1) DT_CHECK(constantToTensor);
      if (numElemLimit >= 0) {
        transfNode->replicationFactor_ = numElemLimit;
      } else {
        transfNode->replicationFactor_ = 1;
        for (const auto& [dim, size] : stickSizes) {
          transfNode->replicationFactor_ *= size;
        }
        if (transfNode->src_.unit_ != SenComponents::CONSTANT &&
            lds.dataFormat_ != DataFormats::SEN169_FP16) {
          DT_CHECK_MSG(
              (lds.dataFormat_ == DataFormats::IEEE_FP32 ||
               lds.dataFormat_ == DataFormats::IEEE_INT32 ||
               lds.dataFormat_ == DataFormats::SENUINT32),
              "No system in place for splatting of data formats with <16b");
          // adjust to be a 16B splat to support fp32 version of the op
          DT_CHECK(transfNode->unitTimeTransferChunkSize_[0].sizeDim_.size_ ==
                   1);
          transfNode->unitTimeTransferChunkSize_[0].sizeDim_.size_ *= 4;
          DT_CHECK(transfNode->replicationFactor_ % 4 == 0);
          transfNode->replicationFactor_ /= 4;
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 127/382   level 0   scc 307   27 body lines
// unit: e127_spreadDataInAllocate
// authority: ddc/ddcv1.cpp:1682
// original: void Ddc::spreadDataInAllocate()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e127_spreadDataInAllocate()
{
  bool is_restickify_sen1p5 = false;
  for (auto& computeOp : currDsc->computeOp_) {
    if ((computeOp.opFuncName == OpFuncs::ReStickifyOpLx ||
         computeOp.opFuncName == OpFuncs::ReStickifyOpHBM) &&
        dscGlobal.sysDef.coreArch == SEN1P5_ISA)
      is_restickify_sen1p5 = true;
  }
  if (is_restickify_sen1p5) {
    for (const auto& compAlloc : metadata.newAllocations_) {
      if (compAlloc.first != PTXRF) continue;
      for (const auto& [ldsIdx, alloc] : compAlloc.second.ldsIdxAndAllocNode) {
        bool used_in_compute_masked = false;
        for (auto [usr, idx] : alloc->allocUsers_) {
          if (usr->nodeType_ != dsc2::ScheduleNode::NodeType::COMPUTE) continue;
          auto* cn = static_cast<const dsc2::ComputeNode*>(usr);
          if (!cn->instrAttribute_.computeMaskLoopOffsets_.empty()) {
            used_in_compute_masked = true;
            break;
          }
        }
        if (!used_in_compute_masked) continue;
        alloc->gapStickSpread_[alloc->layoutDimOrder_.at(0)] = 8;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 128/382   level 0   scc 308   30 body lines
// unit: e128_finalizeAllocateLayouts
// authority: ddc/ddcv1.cpp:1712
// original: void Ddc::finalizeAllocateLayouts()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e128_finalizeAllocateLayouts()
{
  for (const auto& compAlloc : metadata.newAllocations_) {
    for (const auto& [ldsIdx, alloc] : compAlloc.second.ldsIdxAndAllocNode) {
      if (alloc->maxDimSizes_.size() != alloc->layoutDimOrder_.size()) {
        DT_ERROR("Mismatch in allocate layout vectors");
      }
      for (int i = 0; i < alloc->maxDimSizes_.size(); i++) {
        if (alloc->maxDimSizes_[i] >= 0) {
          int size = currDsc->dataStageParam_.at(alloc->maxDimSizes_[i])
                         .ss_.primaryDimToVal_st(alloc->layoutDimOrder_[i],
                                                 alloc->component_, 0, 0);
          auto stickSizes = currDsc->getCumulativeStickSizes(
              currDsc->labeledDs_.at(alloc->ldsIdx_).dsType_);
          auto stickIt = stickSizes.find(alloc->layoutDimOrder_[i]);
          if (stickIt != stickSizes.end()) {
            size /= stickIt->second;
          }
          alloc->maxDimSizes_[i] = size;
        }
      }
    }
  }
  for (auto& shadowAllocs : metadata.shadowAllocations_) {
    for (auto alloc : shadowAllocs) {
      if (alloc != shadowAllocs.at(0))
        alloc->startAddressCoreCorelet_.clone(
            shadowAllocs.at(0)->startAddressCoreCorelet_);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 129/382   level 0   scc 309   19 body lines
// unit: e129_getClSplitDim
// authority: ddc/ddcv1.cpp:1799
// original: std::set<PrimaryDimTypes> getClSplitDim(const DesignSpaceConfig& dsc)
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
std::set<PrimaryDimTypes> e129_getClSplitDim(const DesignSpaceConfig& dsc)
{
  std::set<PrimaryDimTypes> clSplitDims;
  if (dsc.numCoreletsUsed_ > 1) {
    for (const auto& dim : EnumsConversion::primaryDimToString) {
      if (is_any_of(dim.first, IJ, KIJ, PrimaryDimTypesCount)) continue;
      if (dsc.CoreD_.primaryDimToVal_st(dim.first) !=
          dsc.CoreletD_.primaryDimToVal_st(dim.first)) {
        clSplitDims.insert(dim.first);
      }
    }
    if (clSplitDims.empty()) {
      DT_ERROR(
          "dsc: " + dsc.name_ +
          " NumCorelets in DSC > 1, but not able to identify corelet split "
          "dimension");
    }
  }
  return clSplitDims;
}

// ------------------------------------------------------------------------------------------------
// entry 130/382   level 0   scc 310   47 body lines
// unit: e130_getPeSfpSplitDim
// authority: ddc/ddcv1.cpp:1819
// original: std::set<PrimaryDimTypes> getPeSfpSplitDim( const DesignSpaceConfig& dsc, const DesignSpaceConfigGlobal& dscGlobal)
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
std::set<PrimaryDimTypes> e130_getPeSfpSplitDim(
    const DesignSpaceConfig& dsc, const DesignSpaceConfigGlobal& dscGlobal)
{
  std::set<PrimaryDimTypes> peSfpSplitDims;
  bool splitNotPossible = dsc.computeOp_.front().attributes_.dataFormat_ ==
                              DataFormats::IEEE_FP32 &&
                          dscGlobal.sysDef.coreArch <= RCUDD1A_ISA;
  auto& chunkSs = dsc.dataStageParam_.at(1).ss_;
  if (!chunkSs.peSfpSplit_.empty()) {
    DT_CHECK_MSG(!splitNotPossible,
                 "PE/SFP split requested, but hardware cannot perform it");
    for (auto& [dim, info] : chunkSs.peSfpSplit_) {
      peSfpSplitDims.insert(dim);
    }
    return peSfpSplitDims;
  }

  if (splitNotPossible) return peSfpSplitDims;

  auto& compOpFunc = dsc.computeOp_.front().opFuncName;
  std::set<OpFuncs> opFuncsForPeSfpSplit = {
      OpFuncs::RECIPROCAL,     OpFuncs::SQRT_FWD,    OpFuncs::RSQRT,
      OpFuncs::GELU_FWD,       OpFuncs::TANH_FWD,    OpFuncs::LAYERNORM_SCALE,
      OpFuncs::INT32IDXTOADDR, OpFuncs::SIGMOID_FWD, OpFuncs::SILU_FWD,
  };
  if (compOpFunc == OpFuncs::LAYERNORM_SCALE) {
    if (dtGetEnv<bool>("ENABLE_LN32").value_or(false)) {
      opFuncsForPeSfpSplit.erase(compOpFunc);
    }
  }
  if (!opFuncsForPeSfpSplit.count(compOpFunc)) {
    return peSfpSplitDims;
  }

  auto& inpLds = dsc.labeledDs_.at(0);
  auto stickSizePerDim = dsc.getCumulativeStickSizes(inpLds.dsType_);
  for (auto dim : reverse(dsc.getLayoutDims(inpLds.ldsIdx_))) {
    int dimStickSize =
        (stickSizePerDim.count(dim) ? stickSizePerDim.at(dim) : 1);
    int dimSizeForSplit =
        chunkSs.primaryDimToVal_st(dim, SenComponents::NO_COMPONENT, -1, 0) /
        dimStickSize;
    if (dimSizeForSplit > 1 && !(dimSizeForSplit % 2)) {
      peSfpSplitDims.insert(dim);
      break;
    }
  }
  return peSfpSplitDims;
}

// ------------------------------------------------------------------------------------------------
// entry 131/382   level 0   scc 311   22 body lines
// unit: e131_setPeFoldsIfPtInteraction
// authority: ddc/ddcv1.cpp:1868
// original: void setPeFoldsIfPtInteraction(DesignSpaceConfig* currDsc, const DesignSpaceConfigGlobal& dscGlobal)
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e131_setPeFoldsIfPtInteraction(DesignSpaceConfig* currDsc,
                               const DesignSpaceConfigGlobal& dscGlobal)
{
  if (dscGlobal.sysDef.coreArch < IsaCoreGen::SEN1P5_ISA) return;
  const auto allComputeNodes = currDsc->scheduleTree_.traverseTreeDFSMutable(
      nullptr, {dsc2::ScheduleNode::COMPUTE});
  bool doChange = false;
  for (const auto* node : allComputeNodes) {
    auto* cn = static_cast<const dsc2::ComputeNode*>(node);
    if (EnumsConversion::senCompToGenericComp.at(cn->exUnit_) == PT) {
      doChange = true;
      break;
    }
  }
  if (!doChange) return;
  for (auto* node : allComputeNodes) {
    auto* cn = static_cast<dsc2::ComputeNode*>(node);
    const auto genericComp =
        EnumsConversion::senCompToGenericComp.at(cn->exUnit_);
    if (genericComp == PE) {
      cn->numFoldsEngaged = dscGlobal.sysDef.numFoldsPerUnit.at(genericComp);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 132/382   level 0   scc 314   5 body lines
// unit: e132_restoreDsc
// authority: ddc/ddcv1.cpp:2272
// original: void Ddc::restoreDsc()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e132_restoreDsc()
{
  if (metadata.opFuncBackup_ != OpFuncs::NONE) {
    currDsc->computeOp_.at(0).opFuncName = metadata.opFuncBackup_;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 133/382   level 0   scc 317   81 body lines
// unit: e133_adjustLoopOffsetsAndAddresses
// authority: ddc/ddcv1.cpp:3201
// original: void Ddc::adjustLoopOffsetsAndAddresses()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e133_adjustLoopOffsetsAndAddresses()
{
  // adjust this offset for sen1p5 restickify
  // transfer l0lu->pt -> 2 loop around it
  // value of datastage? inner of size 8 and outer size 2
  // loop offset of inner has to be 2 out
  // loop offset of outer has to be 1 out
  auto adjustLoopOffsetsForRestickify = [&]() {
    std::vector<SenComponents> skip_units = {NO_COMPONENT, CONSTANT};
    std::vector<PrimaryDimTypes> inputStickDim;
    for (auto* node : currDsc->scheduleTree_.traverseTreeDFSMutable(
             nullptr, {dsc2::ScheduleNode::TRANSFER})) {
      auto* transfer = static_cast<dsc2::TransferNode*>(node);
      if (is_any_of(transfer->src_.unit_, skip_units)) continue;
      if (EnumsConversion::senCompToGenericComp.at(transfer->src_.unit_) ==
          LXLU) {
        inputStickDim =
            currDsc->getStickDims(transfer->srcLdsAndLoopOffsets_.myLdsIdx_);
        break;
      }
    }
    for (auto* node : currDsc->scheduleTree_.traverseTreeDFSMutable(
             nullptr, {dsc2::ScheduleNode::TRANSFER})) {
      auto* transfer = static_cast<dsc2::TransferNode*>(node);
      if (is_any_of(transfer->src_.unit_, skip_units)) continue;
      if (EnumsConversion::senCompToGenericComp.at(transfer->src_.unit_) !=
          L0LU)
        continue;
      bool releventTransfer = false;
      for (auto dstVia : transfer->dstVias_) {
        if (is_any_of(dstVia.loc_.unit_, skip_units)) continue;
        if (EnumsConversion::senCompToGenericComp.at(dstVia.loc_.unit_) == PT) {
          releventTransfer = true;
        }
      }
      if (!releventTransfer) continue;
      auto innerLoop = transfer->getOwnerLoop();
      auto outerLoop = innerLoop->getOwnerLoop();
      for (int cl = 0; cl < currDsc->numCoreletsUsed_DSC2_; cl++) {
        auto& loopEleOffsets = transfer->srcLdsAndLoopOffsets_.loopEleOffsets_;
        DT_CHECK(loopEleOffsets.at(cl).size() == 2);
        DT_CHECK(loopEleOffsets.at(cl).count(innerLoop) == 1);
        DT_CHECK(loopEleOffsets.at(cl).count(outerLoop) == 1);
        DT_CHECK(inputStickDim.size() == 1);
        auto dim = inputStickDim.at(0);
        loopEleOffsets.at(cl).at(innerLoop).at(dim) = 2;
        loopEleOffsets.at(cl).at(outerLoop).at(dim) = 1;
      }
    }
  };

  auto adjustLoopOffsetsForLXLUCompute = [&](dsc2::ComputeNode* compute) {
    DT_CHECK(compute->inputs_.size() == 2);
    int idx = 0;
    if (compute->inputs_.at(0) == LXLUSCALEREG &&
        compute->inputs_.at(1) == LATCH)
      idx = 1;
    auto parentLoop = compute->getOwnerLoop();
    for (int cl = 0; cl < currDsc->numCoreletsUsed_DSC2_; cl++) {
      auto& loopEleOffsets =
          compute->inputsLdsAndLoopOffsets_.at(idx).loopEleOffsets_;
      loopEleOffsets[cl][parentLoop][IN] = 1;
    }
  };

  if (dscGlobal.sysDef.coreArch < SEN1P5_ISA) return;
  for (auto& computeOp : currDsc->computeOp_) {
    if (computeOp.opFuncName == OpFuncs::ReStickifyOpLx ||
        computeOp.opFuncName == OpFuncs::ReStickifyOpHBM) {
      adjustLoopOffsetsForRestickify();
    }
  }

  for (auto* node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::COMPUTE})) {
    auto* compute = static_cast<dsc2::ComputeNode*>(node);
    if (compute->type_ == FMUL && compute->exUnit_ == LXLU)
      adjustLoopOffsetsForLXLUCompute(compute);
  }

  return;
}

// ------------------------------------------------------------------------------------------------
// entry 134/382   level 0   scc 319   27 body lines
// unit: e134_simplifyScheduleTree
// authority: ddc/ddcv1.cpp:3457
// original: void Ddc::simplifyScheduleTree()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e134_simplifyScheduleTree()
{
  DT_CHECK(!currDsc->scheduleTree_.getHead()->relevantComps_.empty());
  // explore tree in reverse so that if we delete something we don't risk to
  // iterate over deleted children
  for (auto* node : reverse(currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::CONDITION}))) {
    auto* cn = static_cast<dsc2::ConditionNode*>(node);
    if (isExternalNode(cn)) {
      continue;
    }
    auto cnRelevantCoreCl = cn->getRelevantCoreCl();
    if (cnRelevantCoreCl.empty()) {  // empty condition node
      cn->prev_->deleteChildNode(currDsc, cn);
      continue;
    }
    if (!cn->hasCoreClCond()) continue;
    // if "then" or "else" branch have exactly same core/cl then the
    // condition can be reduced to a true or false and simplified away
    for (const auto& child : cn->next_) {
      if (cnRelevantCoreCl == child->getRelevantCoreCl()) {
        child->moveNode(currDsc, cn->getMutableParent(), false, cn);
        cn->prev_->deleteChildNode(currDsc, cn);
        break;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 135/382   level 0   scc 321   5 body lines
// unit: e135_updateLdsIdxMetadata
// authority: ddc/ddcv1.cpp:3667
// original: void Ddc::updateLdsIdxMetadata(DesignSpaceConfig& dsc)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e135_updateLdsIdxMetadata(DesignSpaceConfig& dsc)
{
  for (auto& lds : dsc.labeledDs_) {
    metadata.ldsIdxAfterDdc[lds.ldsIdx_] = lds.ldsIdx_;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 136/382   level 0   scc 322   9 body lines
// unit: e136_initGlobalData
// authority: ddc/ddcv1.cpp:3673
// original: void Ddc::initGlobalData()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e136_initGlobalData()
{
  coreletSplitDim = PrimaryDimTypes::PrimaryDimTypesCount;
  if (!(currDsc->dataStageParam_.at(metadata.core_dstgid)
            .ss_.coreletSplit_.empty())) {
    coreletSplitDim = currDsc->dataStageParam_.at(metadata.core_dstgid)
                          .ss_.coreletSplit_.begin()
                          ->first;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 137/382   level 0   scc 288   9 body lines
// unit: e137_int_log2
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:95
// original: int int_log2(int x)
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
int e137_int_log2(int x)
{
  // TODO: surely there's a better way?
  int i = 0;
  while (x > 0) {
    i++;
    x = x >> 1;
  };
  return i - 1;
}

// ------------------------------------------------------------------------------------------------
// entry 138/382   level 0   scc 99   1 body lines
// unit: e138_swap
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:121
// original: void swap()
// class: SwapBuffer
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e138_swap()
{ flag = !flag; }

// ------------------------------------------------------------------------------------------------
// entry 139/382   level 0   scc 366   9 body lines
// unit: e139_packmerge_psuedostring
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:124
// original: std::string packmerge_psuedostring(const std::vector<int> indices, const std::string& in_reg_1, const std::string& in_reg_2, const std::string& out_reg_name)
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::string e139_packmerge_psuedostring(const std::vector<int> indices,
                                   const std::string& in_reg_1,
                                   const std::string& in_reg_2,
                                   const std::string& out_reg_name)
{
  std::stringstream ss;
  ss << out_reg_name << " = packmerge " << in_reg_1 << " " << in_reg_2 << " [ ";
  for (auto i : indices) {
    ss << i << " ";
  }
  ss << "]";
  return ss.str();
}

// ------------------------------------------------------------------------------------------------
// entry 140/382   level 0   scc 369   26 body lines
// unit: e140_repeat_over_dims
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:178
// original: void ComputationOp::repeat_over_dims( const std::vector<DimSymbol>& dims, std::vector<ComputationOp>& append_to) const
// class: ComputationOp
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e140_repeat_over_dims(
    const std::vector<DimSymbol>& dims,
    std::vector<ComputationOp>& append_to) const
{
  int original_size = append_to.size();
  append_to.push_back(*this);  // a copy.
  std::set<DimSymbol> dim_set;
  for (const auto& in : inputs) {
    for (auto dim_idx : in) {
      dim_set.insert(dim_idx.dim);
    }
  }
  for (const auto& dim : dims) {
    if (dim_set.count(dim)) continue;

    int dim_stop = append_to.size();
    for (int i = original_size; i < dim_stop; i++) {
      append_to.push_back(append_to[i]);
      for (auto& in : append_to[i].inputs) {
        in.insert({dim, false});
      }
      append_to[i].output.insert({dim, false});
      for (auto& in : append_to.back().inputs) {
        in.insert({dim, true});
      }
      append_to.back().output.insert({dim, true});
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 141/382   level 0   scc 285   16 body lines
// unit: e141_act
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:249
// original: AbstractLayout act(const AbstractLayout& input) override
// class: MergeAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
AbstractLayout e141_act(const AbstractLayout& input)
{
    DT_CHECK(!swap || input.sliceDims()[slice_idx] == slice_dim);
    AbstractLayout output(input);
    if (!stick_dim.is_dummy()) {
      output.stick_dims.erase(stick_dim);
    }
    auto evicted_dim = output.sliceDims()[slice_idx];
    output.slice_dims[slice_idx] = stick_dim;

    if (swap) {
      DT_CHECK(!evicted_dim.is_dummy());
      output.stick_dims.insert(evicted_dim);
    }

    return output;
  }

// ------------------------------------------------------------------------------------------------
// entry 142/382   level 0   scc 279   6 body lines
// unit: e142_cost
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:266
// original: double cost(const AbstractLayout& input) override
// class: MergeAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
double e142_cost(const AbstractLayout& input)
{
    if (!swap) {
      return (double)(input.numSticks() / 2);
    }
    return (double)(input.numSticks());
  }

// ------------------------------------------------------------------------------------------------
// entry 143/382   level 0   scc 370   29 body lines
// unit: e143_get_shuffle_indices
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:275
// original: std::vector<int> get_shuffle_indices(bool high)
// class: MergeAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::vector<int> e143_get_shuffle_indices(bool high)
{
    if (high) {
      switch (slice_idx) {
        case dim_64bit:
          return merge64h;
        case dim_32bit:
          return merge32h;
        case dim_16bit:
          return merge16h;
        case dim_8bit:
          return merge8h;
        default:
          DT_ERROR("Tried to use illegal merge instruction");
      }
    } else {
      switch (slice_idx) {
        case dim_64bit:
          return merge64l;
        case dim_32bit:
          return merge32l;
        case dim_16bit:
          return merge16l;
        case dim_8bit:
          return merge8l;
        default:
          DT_ERROR("Tried to use illegal merge instruction");
      }
    }
  }

// ------------------------------------------------------------------------------------------------
// entry 144/382   level 0   scc 260   10 body lines
// unit: e144_act
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:351
// original: AbstractLayout act(const AbstractLayout& input) override
// class: PackAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
AbstractLayout e144_act(const AbstractLayout& input)
{
    AbstractLayout output(input);
    if (!stick_dim.is_dummy()) {
      output.stick_dims.erase(stick_dim);
    }
    auto& slice = output.slice_dims;
    slice.insert(slice.begin() + slice_idx + 1, stick_dim);
    slice.erase(slice.begin() + dim_8bit);
    return output;
  }

// ------------------------------------------------------------------------------------------------
// entry 145/382   level 0   scc 372   12 body lines
// unit: e145_get_indices
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:366
// original: std::vector<int> get_indices()
// class: PackAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::vector<int> e145_get_indices()
{
    switch (slice_idx) {
      case dim_64bit:
        return pack25;
      case dim_32bit:
        return pack26;
      case dim_16bit:
        return pack27;
      default:
        DT_ERROR("Tried to use illegal pack action");
    }
  }

// ------------------------------------------------------------------------------------------------
// entry 146/382   level 0   scc 262   13 body lines
// unit: e146_act
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:409
// original: AbstractLayout act(const AbstractLayout& input) override
// class: ShiftLeftAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
AbstractLayout e146_act(const AbstractLayout& input)
{
    DT_CHECK((extract_dim.is_dummy() ||
              extract_dim == input.sliceDims()[dim_64bit]));

    AbstractLayout output(input);
    if (!extract_dim.is_dummy()) {
      output.stick_dims.insert(output.sliceDims()[dim_64bit]);
    }
    auto& slice = output.slice_dims;
    slice.erase(slice.begin() + dim_64bit);
    slice.insert(slice.begin() + dim_8bit, DimSymbol::getDummy());
    return output;
  }

// ------------------------------------------------------------------------------------------------
// entry 147/382   level 0   scc 284   4 body lines
// unit: e147_cost
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:423
// original: double cost(const AbstractLayout& input) override
// class: ShiftLeftAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
double e147_cost(const AbstractLayout& input)
{
    return (!extract_dim.is_dummy()) ? input.numSticks() * 2
                                     : input.numSticks();
  }

// ------------------------------------------------------------------------------------------------
// entry 148/382   level 0   scc 266   10 body lines
// unit: e148_act
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:476
// original: AbstractLayout act(const AbstractLayout& input) override
// class: Pack8Action
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
AbstractLayout e148_act(const AbstractLayout& input)
{
    AbstractLayout output = input;
    output.stick_dims.erase(dim);
    auto& slice = output.slice_dims;
    for (int i : {dim_8bit, dim_4bit}) {
      slice.erase(slice.begin() + i);
    }
    slice.insert(slice.end(), {dim, DimSymbol::getDummy()});
    return output;
  }

// ------------------------------------------------------------------------------------------------
// entry 149/382   level 0   scc 283   8 body lines
// unit: e149_act
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:523
// original: AbstractLayout act(const AbstractLayout& input) override
// class: Pack9Action
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
AbstractLayout e149_act(const AbstractLayout& input)
{
    AbstractLayout output = input;
    output.stick_dims.erase(dim);
    auto& slice = output.slice_dims;
    slice[dim_4bit] = dim;
    slice[dim_8bit] = DimSymbol::getDummy();
    return output;
  }

// ------------------------------------------------------------------------------------------------
// entry 150/382   level 0   scc 259   4 body lines
// unit: e150_cost
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:532
// original: double cost(const AbstractLayout& input) override
// class: Pack9Action
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
double e150_cost(const AbstractLayout& input)
{
    int num_input_sticks = int_pow2(input.stickDims().size());
    return (double)(num_input_sticks / 2);
  }

// ------------------------------------------------------------------------------------------------
// entry 151/382   level 0   scc 263   11 body lines
// unit: e151_act
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:573
// original: AbstractLayout act(const AbstractLayout& input) override
// class: Pack24Action
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
AbstractLayout e151_act(const AbstractLayout& input)
{
    AbstractLayout output = input;
    output.stick_dims.erase(dim);
    auto& slice = output.slice_dims;
    for (int i : {dim_8bit, dim_4bit, dim_2bit}) {
      slice.erase(slice.begin() + i);
    }
    slice.insert(slice.end(),
                 {dim, DimSymbol::getDummy(), DimSymbol::getDummy()});
    return output;
  }

// ------------------------------------------------------------------------------------------------
// entry 152/382   level 0   scc 272   8 body lines
// unit: e152__out_format
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:598
// original: static std::optional<DataFormats> _out_format(DataFormats in, const AbstractLayout& goal)
// class: GCVTF16F8PackAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::optional<DataFormats> e152__out_format(DataFormats in,
                                                const AbstractLayout& goal)
{
    if (is_any_of(in, DataFormats::IEEE_FP16, DataFormats::SEN169_FP16) &&
        is_any_of(goal.format, DataFormats::SEN143_FP8,
                  DataFormats::SEN152_FP8)) {
      return goal.format;
    }
    return std::nullopt;
  }

// ------------------------------------------------------------------------------------------------
// entry 153/382   level 0   scc 278   7 body lines
// unit: e153_act
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:620
// original: AbstractLayout act(const AbstractLayout& input) override
// class: GCVTF16F8PackAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
AbstractLayout e153_act(const AbstractLayout& input)
{
    AbstractLayout output = input;
    output.stick_dims.erase(dim);
    output.slice_dims.erase(output.slice_dims.begin() + dim_8bit);
    output.slice_dims.push_back(dim);
    return output;
  }

// ------------------------------------------------------------------------------------------------
// entry 154/382   level 0   scc 271   8 body lines
// unit: e154__out_format
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:644
// original: static std::optional<DataFormats> _out_format(DataFormats in, const AbstractLayout& goal)
// class: GCVTF16F8MergeAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::optional<DataFormats> e154__out_format(DataFormats in,
                                                const AbstractLayout& goal)
{
    if (is_any_of(in, DataFormats::IEEE_FP16, DataFormats::SEN169_FP16) &&
        is_any_of(goal.format, DataFormats::SEN143_FP8,
                  DataFormats::SEN152_FP8)) {
      return goal.format;
    }
    return std::nullopt;
  }

// ------------------------------------------------------------------------------------------------
// entry 155/382   level 0   scc 261   6 body lines
// unit: e155_act
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:666
// original: AbstractLayout act(const AbstractLayout& input) override
// class: GCVTF16F8MergeAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
AbstractLayout e155_act(const AbstractLayout& input)
{
    AbstractLayout output = input;
    output.stick_dims.erase(dim);
    output.slice_dims[dim_8bit] = dim;
    return output;
  }

// ------------------------------------------------------------------------------------------------
// entry 156/382   level 0   scc 281   5 body lines
// unit: e156_contains
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:684
// original: bool AbstractLayout::contains(DimSymbol dim) const
// class: AbstractLayout
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
bool e156_contains(DimSymbol dim) const
{
  return (stick_dims.find(dim) != stick_dims.end()) ||
         (std::find(slice_dims.begin(), slice_dims.end(), dim) !=
          slice_dims.end());
}

// ------------------------------------------------------------------------------------------------
// entry 157/382   level 0   scc 265   11 body lines
// unit: e157_get_node
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:717
// original: std::shared_ptr<GraphNode> AutoShuffler::get_node( const AbstractLayout& layout)
// class: AutoShuffler
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::shared_ptr<GraphNode> e157_get_node(
    const AbstractLayout& layout)
{
  auto it = layout_to_nodes.find(layout);
  if (it == layout_to_nodes.end()) {
    // add a new node to the graph
    auto node = std::make_shared<GraphNode>(layout);
    layout_to_nodes.insert(std::make_pair(layout, node));
    return node;
  } else {
    return it->second;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 158/382   level 0   scc 375   14 body lines
// unit: e158_make_stick_number_key
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:730
// original: auto make_stick_number_key(const std::vector<DimSymbol>& stick_ordering)
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
auto e158_make_stick_number_key(const std::vector<DimSymbol>& stick_ordering)
{
  DT_CHECK(stick_ordering.size() < 32);
  std::unordered_map<DimSymbol, uint32_t> masks;
  for (int i = 0; i < stick_ordering.size(); i++) {
    masks[stick_ordering[i]] = (1u << i);
  }
  return [masks = std::move(masks)](const StickIndex& idx) {
    uint32_t key = 0u;
    for (auto x : idx) {
      if (x.high) key = (key | masks.at(x.dim));
    }
    return key;
  };
}

// ------------------------------------------------------------------------------------------------
// entry 159/382   level 0   scc 287   3 body lines
// unit: e159_all_one
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:1043
// original: bool all_one(std::vector<int> vec)
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
bool e159_all_one(std::vector<int> vec)
{
  return std::all_of(vec.begin(), vec.end(), [](int x) { return x == 1; });
}

// ------------------------------------------------------------------------------------------------
// entry 160/382   level 0   scc 264   7 body lines
// unit: e160_update_worklist
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:1139
// original: void AutoShuffler::update_worklist(std::shared_ptr<GraphNode> node)
// class: AutoShuffler
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e160_update_worklist(std::shared_ptr<GraphNode> node)
{
  // We have to record a frozen version of the cost since the cost
  // stored on the node itself may be mutated.
  // Use a monotonic counter as tie-breaker for deterministic ordering.
  // TODO replace this with reduce_key
  worklist.push(std::make_tuple(node->estimated_cost(), worklist_counter++, node));
}

// ------------------------------------------------------------------------------------------------
// entry 161/382   level 0   scc 380   42 body lines
// unit: e161_codegen_psuedocode
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:1224
// original: std::vector<std::string> AutoShuffler::codegen_psuedocode( const ConcreteLayout& input, const ConcreteLayout& output, const std::vector<std::shared_ptr<GraphNode>>& shuffle)
// class: AutoShuffler
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::vector<std::string> e161_codegen_psuedocode(
    const ConcreteLayout& input, const ConcreteLayout& output,
    const std::vector<std::shared_ptr<GraphNode>>& shuffle)
{
  std::vector<std::string> code_lines;
  int reg_id = 0;
  auto get_new_reg_name = [&reg_id]() {
    std::stringstream ss;
    ss << "r" << reg_id;
    reg_id++;
    return ss.str();
  };
  std::vector<std::string> input_regs;
  for (int i = 0; i < int_pow2(input.stick_dims.size()); i++) {
    input_regs.push_back(get_new_reg_name());
  }

  const std::function<std::vector<std::string>(std::shared_ptr<GraphNode> node)>
      create_node_allocations =
          [&get_new_reg_name](std::shared_ptr<GraphNode> node) {
            std::vector<std::string> regs;
            for (int i = 0; i < node->layout.numSticks(); i++) {
              regs.push_back(get_new_reg_name());
            }
            // Codegen pulls in order from the back. Register names are
            // interchangeable for psuedocode, but this order is easier to read.
            std::reverse(regs.begin(), regs.end());
            return regs;
          };

  std::function<void(ComputationOp&, const std::vector<std::string>&,
                     std::string&, std::pair<std::vector<int>, int>&, bool)>
      do_codegen = [&code_lines](
                       ComputationOp& op,
                       const std::vector<std::string>& input_regs,
                       std::string& output_reg,
                       std::pair<std::vector<int>, int>& input_output_stick_id,
                       bool output_reg_added) {
        auto newlines = op.codegen_psuedocode(input_regs, output_reg);
        code_lines.insert(code_lines.end(), newlines.begin(), newlines.end());
      };
  codegen_generic<std::string>(input, output, shuffle, input_regs,
                               create_node_allocations, do_codegen);
  return code_lines;
}

// ------------------------------------------------------------------------------------------------
// entry 162/382   level 0   scc 381   8 body lines
// unit: e162_getDefaultSymbols
// authority: ddc/transformations/automatic_shuffle/shuffle.h:48
// original: static std::vector<DimSymbol> getDefaultSymbols(int n)
// class: DimSymbol
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::vector<DimSymbol> e162_getDefaultSymbols(int n)
{
    std::vector<DimSymbol> symbols;
    symbols.reserve(n);
    for (int i = 1; i < n + 1; i++) {
      symbols.push_back(DimSymbol(i));
    }
    return symbols;
  }

// ------------------------------------------------------------------------------------------------
// entry 163/382   level 0   scc 376   3 body lines
// unit: e163_constexpr
// authority: ddc/transformations/automatic_shuffle/shuffle.h:160
// original: if constexpr (sizeof(size_t) < sizeof(uint64_t))
// class: std
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
if e163_constexpr (sizeof(size_t) < sizeof(uint64_t))
{
      hash = hash ^ (hash >> 32);
    }

// ------------------------------------------------------------------------------------------------
// entry 164/382   level 0   scc 291   7 body lines
// unit: e164_insert_before
// authority: ddc/transformations/automatic_shuffle/shuffle.h:175
// original: void insert_before(dsc2::BlockNode* parent, dsc2::ScheduleNode* insert_point)
// class: DataEdge
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e164_insert_before(dsc2::BlockNode* parent,
                     dsc2::ScheduleNode* insert_point)
{
    if (allocation.has_value() && allocation.value() != nullptr &&
        !alloc_added) {
      parent->addChildNode(allocation.value(), true, insert_point);
      alloc_added = true;
    }
  }

// ------------------------------------------------------------------------------------------------
// entry 230/382   level 1   scc 168   4 body lines
// unit: e230_addPropInfo
// authority: ddc/ddc.h:430
// original: void addPropInfo(const dsc2::CoordPropInfoType& rhs, const std::vector<PrimaryDimTypes> dims)
// class: CoordPropTracker
// rust home: crates/compiler/deeptools/src/schedule/ddc/mod.rs
// ------------------------------------------------------------------------------------------------
void e230_addPropInfo(const dsc2::CoordPropInfoType& rhs,
                     const std::vector<PrimaryDimTypes> dims)
{
      addPropInfo(rhs.refNode, rhs.nodeToFold, dims, rhs.dataConnect,
                  rhs.refIsProducer, rhs.scaleDown);
    }

// ------------------------------------------------------------------------------------------------
// entry 231/382   level 1   scc 171   36 body lines
// unit: e231_rollbackToPos
// authority: ddc/ddc.h:474
// original: void rollbackToPos(int newPos)
// class: CoordPropTracker
// rust home: crates/compiler/deeptools/src/schedule/ddc/mod.rs
// ------------------------------------------------------------------------------------------------
void e231_rollbackToPos(int newPos)
{
      if (currItemToProcess_ < newPos || currItemToProcess_ < 0) {
        return;
      }
      if (newPos < 0 || newPos > currItemToProcess_) {
        DT_ERROR("Invalid rollback position " + std::to_string(newPos) +
                 ", acceptable range is [0 - " +
                 std::to_string(currItemToProcess_) + "].");
      }
      // Clear the computed coordinates up to the rollback position.
      for (int i = newPos; i <= currItemToProcess_; ++i) {
        itemsToProcess_.at(currItemToProcess_).propState =
            dsc2::CoordPropInfoType::PropStateType::ROLLED_BACK;
        if (itemsToProcess_[i].nodeToFold->nodeType_ ==
            dsc2::ScheduleNode::ALLOCATE) {
          auto allocNode =
              static_cast<dsc2::AllocateNode*>(itemsToProcess_[i].nodeToFold);
          allocNode->allocateCoordinates_.clear();
          allocNode->sliceViewCoordinates_.clear();
        } else if (itemsToProcess_[i].nodeToFold->nodeType_ ==
                   dsc2::ScheduleNode::TRANSFER) {
          auto transferNode =
              static_cast<dsc2::TransferNode*>(itemsToProcess_[i].nodeToFold);
          transferNode->transferCoordinates_.clear();
        } else if (itemsToProcess_[i].nodeToFold->nodeType_ ==
                   dsc2::ScheduleNode::COMPUTE) {
          auto computeNode =
              static_cast<dsc2::ComputeNode*>(itemsToProcess_[i].nodeToFold);
          computeNode->outputCoordinate_.clear();
          for (auto& inputCoord : computeNode->inputCoordinates_) {
            inputCoord.clear();
          }
        }
      }
      currItemToProcess_ = newPos - 1;
    }

// ------------------------------------------------------------------------------------------------
// entry 232/382   level 1   scc 173   5 body lines
// unit: e232_reset
// authority: ddc/ddc.h:528
// original: void reset()
// class: CoordPropTracker
// rust home: crates/compiler/deeptools/src/schedule/ddc/mod.rs
// ------------------------------------------------------------------------------------------------
void e232_reset()
{
      itemsToProcess_.clear();
      refsAdded_.clear();
      currItemToProcess_ = -1;
    }

// ------------------------------------------------------------------------------------------------
// entry 233/382   level 1   scc 177   11 body lines
// unit: e233_dbgPrint
// authority: ddc/ddc_fold.cpp:63
// original: void dbgPrint(const dsc2::ScheduleNode *node)
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e233_dbgPrint(const dsc2::ScheduleNode *node)
{
  if (node->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
    std::cout << " AllocateNode: " << node->name_ << "(" << node << ")";
  } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    dbgPrint(static_cast<const dsc2::ComputeNode *>(node));
  } else if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
    dbgPrint(static_cast<const dsc2::TransferNode *>(node));
  } else {
    DT_ERROR("[dbgPrint] Unsupported node type.");
  }
}

// ------------------------------------------------------------------------------------------------
// entry 234/382   level 1   scc 191   118 body lines
// unit: e234_scaleUpCoord
// authority: ddc/ddc_fold.cpp:476
// original: void Ddc::scaleUpCoord(dsc2::ScheduleNode *lhsNode, dsc2::CoordinateType<CoordinateBaseType> &lhsCoord, dsc2::CoordinateType<CoordinateBaseType> &rhsCoord, const int ldsIdx, const SenComponents comp)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e234_scaleUpCoord(dsc2::ScheduleNode *lhsNode,
                       dsc2::CoordinateType<CoordinateBaseType> &lhsCoord,
                       dsc2::CoordinateType<CoordinateBaseType> &rhsCoord,
                       const int ldsIdx, const SenComponents comp)
{
  if (ldsIdx == -1) {
    DT_ERROR("[scaleUpCoord] Invalid labeledDs index.");
  }

  const LabeledDsInfo &lds = currDsc->labeledDs_.at(ldsIdx);
  if (lds.scaledLdsCategory_ !=
      LabeledDsInfo::ScaledLdsCategory::VALUE_TENSOR) {
    // Nothing to do.
    DT_ERROR(
        "[scaleUpCoord] Invalid labeledDs category " +
        LabeledDsInfo::scaledLdsCategoryToString.at(lds.scaledLdsCategory_));
  }

  const PrimaryDimTypes dim = lds.mxInfo_.dim;
  if (!rhsCoord.hasCoordForDim(dim)) {
    return;
  }
  int scaleUpFactor = lds.mxInfo_.blkSize;
  std::vector<dsc2::FoldParamInfoType> foldParams;
  gatherFoldParams(rhsCoord.coordinates_.at(dim), foldParams);
  int spatialFoldEnds = rhsCoord.getNumOfSpatialFolds(dim) - 1;
  int temporalFoldEnds = spatialFoldEnds + rhsCoord.getNumOfTemporalFolds(dim);

  if (coordPropReportLevel_ > 1) {
    std::cout << "\n[scaleUpCoord] dim= "
              << EnumsConversion::primaryDimToString.at(dim) << ", Lds("
              << lds.ldsIdx_ << ")= " << lds.dsName_
              << "\nReference input coordinate:";
    rhsCoord.debugPrint(std::cout);
    std::cout << std::endl;
  }

  for (auto &foldParam : foldParams) {
    foldParam.alpha *= scaleUpFactor;
    foldParam.beta *= scaleUpFactor;
  }

  // Add a new element-arangement fold for scale block.
  foldParams.insert(foldParams.end(),
                    {1, 0, scaleUpFactor, "elem_arr_scaleup"});

  // Figure out all innermost levels that use the same scale element. Exclude
  // the newly inserted element arrangement.
  int scaleChangePos = foldParams.size() - 2;
  while (foldParams.at(scaleChangePos).cardinality == 1 ||
         foldParams.at(scaleChangePos).alpha == 0) {
    --scaleChangePos;
  }
  if (scaleChangePos <= temporalFoldEnds) {
    dsc2::LoopDistributionParamPerNodeType loopParamsAfterDistribution;
    std::vector<dsc2::FoldParamInfoType> elemArrParamsAfterDistribution;
    std::vector<dsc2::FoldParamInfoType> elemArr;
    for (int i = foldParams.size() - 1; i > temporalFoldEnds; --i) {
      elemArr.push_back(foldParams.at(i));
    }

    // Find the enclosing loop chain. Collect the associated dimensions.
    dsc2::VectorOfLoopAndDim loopChain;
    std::unordered_set<PrimaryDimAndKind> relatedDims;
    getEnclosingLoopsAndRelatedDims(currDsc, lhsNode, loopChain,
                                    loopsBelowChunkBoundary);

    dsc2::VectorOfLoopAndDim relatedLoops;
    for (auto loop : loopChain) {
      // Assumption: mx-scale dimension does not have padding.
      dsc2::collectRelatedLoops(currDsc, dim, loopChain, relatedLoops,
                                PadType::NOPAD);
    }
    dsc2::VectorOfLoopAndDim relatedLoopsToDistribute =
        dsc2::VectorOfLoopAndDim(
            relatedLoops.begin(),
            relatedLoops.begin() + temporalFoldEnds - scaleChangePos);

    // Distribute temporal loops over element arrangements and compute the fold
    // parameters for temporal and element arrangement folds.
    dsc2::distributeElemArrToTemporalLoops(
        currDsc, dim, lhsNode, ldsIdx, PadType::NOPAD, PadType::NOPAD, comp,
        comp, relatedLoopsToDistribute, elemArr, loopParamsAfterDistribution,
        elemArrParamsAfterDistribution, 0 /* for one corelet */,
        coordPropReportLevel_);

    // Update the relevant foldParams levels with the newly computed values.
    for (int i = scaleChangePos + 1, j = relatedLoopsToDistribute.size() - 1;
         i <= temporalFoldEnds, j >= 0; ++i, --j) {
      auto loop = relatedLoopsToDistribute.at(j).loopNode;
      foldParams.at(i).alpha =
          loopParamsAfterDistribution.at(loop).at(dim).alpha;
      foldParams.at(i).beta = loopParamsAfterDistribution.at(loop).at(dim).beta;
    }

    foldParams.erase(foldParams.begin() + temporalFoldEnds + 1,
                     foldParams.end());
    for (auto &newElemArr : elemArrParamsAfterDistribution) {
      foldParams.push_back(newElemArr);
    }
  }

  lhsCoord.clearFoldForDim(dim);
  for (int i = foldParams.size() - 1; i >= 0; --i) {
    dsc2::CoordinateCategory coordCat = dsc2::CoordinateCategory::UNKNOWN_COORD;
    if (i > temporalFoldEnds) {
      coordCat = dsc2::CoordinateCategory::ELEM_ARR_COORD;
    } else if (i > spatialFoldEnds) {
      coordCat = dsc2::CoordinateCategory::TEMPORAL_COORD;
    } else {
      coordCat = dsc2::CoordinateCategory::SPATIAL_COORD;
    }
    lhsCoord.addFold(dim, coordCat, foldParams.at(i).cardinality,
                     foldParams.at(i).foldDimLabel, foldParams.at(i).alpha,
                     foldParams.at(i).beta, 0);
  }

  if (coordPropReportLevel_ > 1) {
    std::cout << "\n[scaleUpCoord] Scaled up coordinate at the end:";
    lhsCoord.debugPrint(std::cout);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 235/382   level 1   scc 192   87 body lines
// unit: e235_scaleDownCoord
// authority: ddc/ddc_fold.cpp:598
// original: void Ddc::scaleDownCoord(dsc2::CoordinateType<CoordinateBaseType> &lhsCoord, dsc2::CoordinateType<CoordinateBaseType> &rhsCoord, const int ldsIdx)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e235_scaleDownCoord(dsc2::CoordinateType<CoordinateBaseType> &lhsCoord,
                         dsc2::CoordinateType<CoordinateBaseType> &rhsCoord,
                         const int ldsIdx)
{
  if (ldsIdx == -1) {
    DT_ERROR("[scaleDownCoord] Invalid labeledDs index.");
  }

  const LabeledDsInfo &lds = currDsc->labeledDs_.at(ldsIdx);
  if (lds.scaledLdsCategory_ !=
      LabeledDsInfo::ScaledLdsCategory::SCALE_TENSOR) {
    // Nothing to do.
    DT_ERROR(
        "[scaleDownCoord] Invalid labeledDs category " +
        LabeledDsInfo::scaledLdsCategoryToString.at(lds.scaledLdsCategory_));
  }

  const PrimaryDimTypes dim = lds.mxInfo_.dim;
  if (!rhsCoord.hasCoordForDim(dim)) {
    return;
  }
  int scaleBlkSize = lds.mxInfo_.blkSize;
  double scaleDownFactor = ((double)1.0) / scaleBlkSize;
  std::vector<dsc2::FoldParamInfoType> foldParams;
  gatherFoldParams(rhsCoord.coordinates_.at(dim), foldParams);
  int spatialFoldEnds = rhsCoord.getNumOfSpatialFolds(dim) - 1;
  int temporalFoldEnds = spatialFoldEnds + rhsCoord.getNumOfTemporalFolds(dim);

  if (coordPropReportLevel_ > 1) {
    std::cout << "\n[scaleDownCoord] dim= "
              << EnumsConversion::primaryDimToString.at(dim) << ", Lds("
              << lds.ldsIdx_ << ")= " << lds.dsName_
              << "\nReference input coordinate:";
    rhsCoord.debugPrint(std::cout);
    std::cout << std::endl;
  }

  int accumulatedCardinality = 1;
  for (int i = foldParams.size() - 1; i > temporalFoldEnds; --i) {
    accumulatedCardinality *= foldParams.at(i).cardinality;
    if (accumulatedCardinality < scaleBlkSize) {
      // check if the next level is part of a contiguous element arrangement.
      if (foldParams.at(i - 1).alpha != accumulatedCardinality) {
        DT_ERROR(
            "[scaleDownCoord] Non-contiguous element arrangment within a scale "
            "block is not supported.");
      }
      // Accumulate more folds to reach the scale block size.
      continue;
    }

    if (accumulatedCardinality > scaleBlkSize) {
      foldParams.at(i).cardinality = accumulatedCardinality / scaleBlkSize;
      foldParams.at(i).alpha = 1;
      foldParams.erase(foldParams.begin() + i + 1, foldParams.end());
    } else {
      // accumulatedCardinality == scaleBlkSize
      foldParams.erase(foldParams.begin() + i, foldParams.end());
    }

    for (int alphaInd = i - 1; alphaInd >= 0; --alphaInd) {
      if (foldParams.at(alphaInd).alpha > 1) {
        foldParams.at(alphaInd).alpha *= scaleDownFactor;
        foldParams.at(alphaInd).beta *= scaleDownFactor;
      }
    }
    // Scaling down is now complete.
    break;
  }

  lhsCoord.clearFoldForDim(dim);
  for (int i = foldParams.size() - 1; i >= 0; --i) {
    dsc2::CoordinateCategory coordCat = dsc2::CoordinateCategory::UNKNOWN_COORD;
    if (i > temporalFoldEnds) {
      coordCat = dsc2::CoordinateCategory::ELEM_ARR_COORD;
    } else if (i > spatialFoldEnds) {
      coordCat = dsc2::CoordinateCategory::TEMPORAL_COORD;
    } else {
      coordCat = dsc2::CoordinateCategory::SPATIAL_COORD;
    }
    lhsCoord.addFold(dim, coordCat, foldParams.at(i).cardinality,
                     foldParams.at(i).foldDimLabel, foldParams.at(i).alpha,
                     foldParams.at(i).beta, 0);
  }

  if (coordPropReportLevel_ > 1) {
    std::cout << "\n[scaleDownCoord] Compressed coordinate at the end:";
    lhsCoord.debugPrint(std::cout);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 236/382   level 1   scc 193   50 body lines
// unit: e236_needNonRowBundling
// authority: ddc/ddc_fold.cpp:688
// original: bool Ddc::needNonRowBundling(std::string &dataConnect, bool checkProducers) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
bool e236_needNonRowBundling(std::string &dataConnect,
                             bool checkProducers) const
{
  auto &nodeSet = checkProducers
                      ? metadata.dataConnects_.at(dataConnect).producers_
                      : metadata.dataConnects_.at(dataConnect).consumers_;
  if (nodeSet.size() <= 1) {
    return false;
  }
  std::vector<dsc2::ScheduleNode *> nodeList(nodeSet.begin(), nodeSet.end());
  for (auto node : nodeList) {
    if (getCompRowId(getComponent(node, !checkProducers)) != -1) {
      return false;
    }
  }
  const dsc2::ScheduleNode *commonAncestor =
      findCommonAncestor(nodeList, {dsc2::ScheduleNode::LOOP});
  if (!commonAncestor) {
    // TO DO: Is this an error condition?
    return false;
  }
  std::map<const dsc2::ScheduleNode *, int> descendantCount;
  const dsc2::ScheduleNode *currNode = nullptr;
  for (auto node : nodeSet) {
    currNode = node;
    while (currNode != commonAncestor) {
      if (currNode->prev_ &&
          currNode->prev_->nodeType_ == dsc2::ScheduleNode::CONDITION) {
        // Non-row operation is enclosed in conditionals. Keep track of the
        // number of descendants per conditional branch.
        ++descendantCount[currNode];
        break;
      }
      currNode = currNode->getPrev();
    }
  }

  if (descendantCount.empty()) {
    // None of the nodes in nodeSet are enclosed in any conditional up to the
    // common ancestor.
    return true;
  }
  for (auto &[node, cnt] : descendantCount) {
    if (cnt > 1) {
      // Multiple nodes are under the same conditional. This is a non-PT-row
      // bundling situation.
      return true;
    }
  }
  // Each node in the group is under a separate conditional. No need to bundle.
  return false;
}

// ------------------------------------------------------------------------------------------------
// entry 237/382   level 1   scc 194   252 body lines
// unit: e237_sameCoordinateRange
// authority: ddc/ddc_fold.cpp:1260
// original: bool Ddc::sameCoordinateRange( const dsc2::ScheduleNode *lhsNode, const dsc2::CoordinateType<CoordinateBaseType> &lhs, const dsc2::ScheduleNode *rhsNode, const dsc2::CoordinateType<CoordinateBaseType> &rhs, SenComponents comp /* = SenComponents::ALL*/, int ldsIdx /*= -1*/, bool commonDimsOnly /* = false*/)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
bool e237_sameCoordinateRange(
    const dsc2::ScheduleNode *lhsNode,
    const dsc2::CoordinateType<CoordinateBaseType> &lhs,
    const dsc2::ScheduleNode *rhsNode,
    const dsc2::CoordinateType<CoordinateBaseType> &rhs,
    SenComponents comp /* = SenComponents::ALL*/, int ldsIdx /*= -1*/,
    bool commonDimsOnly /* = false*/)
{
  std::vector<dsc2::FoldParamInfoType> lhsFoldParams, rhsFoldParams;
  if (!commonDimsOnly && lhs.coordinates_.size() != rhs.coordinates_.size()) {
    return false;
  }

  auto isConsistentCoord =
      [](const std::vector<dsc2::FoldParamInfoType> &lhsFoldParams,
         const std::vector<dsc2::FoldParamInfoType> &rhsFoldParams,
         PrimaryDimTypes coordDim, int coordPropReportLevel) -> bool {
    if (lhsFoldParams.size() != rhsFoldParams.size()) {
      if (coordPropReportLevel > 2) {
        std::cout << "\n    [CoordCompare] LHS and RHS do not have the same "
                     "number of coordinates levels after combining for dim= "
                  << EnumsConversion::primaryDimToString.at(coordDim)
                  << std::endl;
        std::cout << "\n        LHS:";
        for (auto fp : lhsFoldParams) {
          std::cout << " (" << fp.alpha << ", " << fp.cardinality << ")";
        }
        std::cout << "\n        RHS:";
        for (auto fp : rhsFoldParams) {
          std::cout << " (" << fp.alpha << ", " << fp.cardinality << ")";
        }
        std::cout << std::endl;
      }
      return false;
    }
    int lhsBeta = 0, rhsBeta = 0;
    for (int i = 0, e = lhsFoldParams.size(); i < e; ++i) {
      if (lhsFoldParams.at(i).cardinality != rhsFoldParams.at(i).cardinality ||
          (lhsFoldParams.at(i).cardinality > 1 &&
           lhsFoldParams.at(i).alpha != rhsFoldParams.at(i).alpha)) {
        if (coordPropReportLevel > 2) {
          std::cout << "\n    [CoordCompare] LHS(" << i << "): ("
                    << lhsFoldParams.at(i).alpha << ", "
                    << lhsFoldParams.at(i).beta << ", "
                    << lhsFoldParams.at(i).cardinality << ") RHS(" << i
                    << "): (" << rhsFoldParams.at(i).alpha << ", "
                    << rhsFoldParams.at(i).beta << ", "
                    << rhsFoldParams.at(i).cardinality << ")";
          std::cout << "\n    LHS and RHS params do not match for dim= "
                    << EnumsConversion::primaryDimToString.at(coordDim)
                    << std::endl;
        }
        return false;
      }
      lhsBeta += lhsFoldParams.at(i).beta;
      rhsBeta += rhsFoldParams.at(i).beta;
    }
    if (lhsBeta != rhsBeta) {
      if (coordPropReportLevel > 2) {
        std::cout
            << "\n    [CoordCompare] LHS and RHS betas do not match for dim= "
            << EnumsConversion::primaryDimToString.at(coordDim)
            << "\n                   LHS-beta= " << lhsBeta
            << " RHS-beta= " << rhsBeta << std::endl;
      }
      return false;
    }

    return true;
  };

  for (auto &[coordDim, lhsFm] : lhs.coordinates_) {
    if (!rhs.coordinates_.count(coordDim)) {
      if (commonDimsOnly) {
        continue;
      } else {
        if (coordPropReportLevel_ > 2) {
          std::cout
              << "\n    [CoordCompare] RHS does not have coordinates for dim= "
              << EnumsConversion::primaryDimToString.at(coordDim) << std::endl;
        }
        return false;
      }
    }
    lhsFoldParams.clear();
    rhsFoldParams.clear();
    gatherFoldParams(lhsFm, lhsFoldParams);
    gatherFoldParams(rhs.coordinates_.at(coordDim), rhsFoldParams);

    const bool lhsHasCustomSlicing =
        !lhs.coreIdToWkSlice_.empty() &&
        lhs.coreIdToWkSlice_ != sdsc_->coreIdToWkSlice_;
    const bool rhsHasCustomSlicing =
        !rhs.coreIdToWkSlice_.empty() &&
        rhs.coreIdToWkSlice_ != sdsc_->coreIdToWkSlice_;
    if ((!lhsHasCustomSlicing && !rhsHasCustomSlicing) ||
        lhs.coreIdToWkSlice_ == rhs.coreIdToWkSlice_) {
      combineContigousLevels(lhsFoldParams);
      combineContigousLevels(rhsFoldParams);
      if (!isConsistentCoord(lhsFoldParams, rhsFoldParams, coordDim,
                             coordPropReportLevel_)) {
        return false;
      }
    } else {
      // Collects <coreId, <work-slice, set of corelets>> for a given
      // scheduleNode and coordinate.
      auto fillCoreIdToWkSlice =
          [](const SuperDsc *sdsc, PrimaryDimTypes coordDim,
             const dsc2::ScheduleNode *node,
             const dsc2::CoordinateType<CoordinateBaseType> &coord,
             SenComponents comp,
             std::map<int, std::pair<int, std::set<int>>> &result) {
            auto relevantCoreCl = node->getRelevantCoreCl(comp);
            if (relevantCoreCl.empty()) {
              relevantCoreCl =
                  node->getRelevantCoreCl(SenComponents::NO_COMPONENT);
            }
            const auto &coreIdToWkSlice = coord.coreIdToWkSlice_.empty()
                                              ? sdsc->coreIdToWkSlice_
                                              : coord.coreIdToWkSlice_;
            for (auto &[coreId, coreletInfo] : relevantCoreCl) {
              if (!coreIdToWkSlice.count(coreId)) {
                continue;
              }
              result[coreId] = {coreIdToWkSlice.at(coreId).at(coordDim),
                                coreletInfo};
            }
          };

      if (ldsIdx != -1) {
        const auto &lds = currDsc->labeledDs_.at(ldsIdx);
        int dimIdx = currDsc->getDimIndexInLayoutOrder(lds.dsType_, coordDim);
        int scale = dimIdx < 0 ? 1 : lds.scale_.at(dimIdx);
        if (scale < 0) {
          // Skip broadcast dimensions.
          continue;
        }
      }

      // <coreId, <work-slice, set of corelets>>
      std::map<int, std::pair<int, std::set<int>>> lhsCoreToWkSliceIds,
          rhsCoreToWkSliceIds;
      fillCoreIdToWkSlice(sdsc_, coordDim, lhsNode, lhs, comp,
                          lhsCoreToWkSliceIds);
      fillCoreIdToWkSlice(sdsc_, coordDim, rhsNode, rhs, comp,
                          rhsCoreToWkSliceIds);
      std::vector<dsc2::FoldParamInfoType> workingLhsFoldParams,
          workingRhsFoldParams;

      // First, check that the size of the two lists are same.
      if (lhsCoreToWkSliceIds.size() != rhsCoreToWkSliceIds.size()) {
        DT_ERROR("Mismatch in number of coreIdToWkSlice entries for nodes " +
                 lhsNode->name_ + " and " + rhsNode->name_ + ".");
      }
      for (auto &[lhsCoreId, lhsCoreInfo] : lhsCoreToWkSliceIds) {
        const int &lhsSliceId = lhsCoreInfo.first;
        const std::set<int> &lhsCoreletInfo = lhsCoreInfo.second;
        if (!rhsCoreToWkSliceIds.count(lhsCoreId)) {
          DT_ERROR(
              "Workslice information for coreId=" + std::to_string(lhsCoreId) +
              " was not found for node " + rhsNode->name_ + ".");
        }
        const int &rhsSliceId = rhsCoreToWkSliceIds.at(lhsCoreId).first;
        const std::set<int> &rhsCoreletInfo =
            rhsCoreToWkSliceIds.at(lhsCoreId).second;

        workingLhsFoldParams = lhsFoldParams;
        // Fix core workslice.
        workingLhsFoldParams.at(FOLD_POS_CORE).beta +=
            (lhsSliceId * workingLhsFoldParams.at(FOLD_POS_CORE).alpha);
        workingLhsFoldParams.at(FOLD_POS_CORE).alpha = 0;
        workingLhsFoldParams.at(FOLD_POS_CORE).cardinality = 1;

        workingRhsFoldParams = rhsFoldParams;
        // Fix core workslice.
        workingRhsFoldParams.at(FOLD_POS_CORE).beta +=
            (rhsSliceId * workingRhsFoldParams.at(FOLD_POS_CORE).alpha);
        workingRhsFoldParams.at(FOLD_POS_CORE).alpha = 0;
        workingRhsFoldParams.at(FOLD_POS_CORE).cardinality = 1;

        // Check if need to fix corelets.
        int fixedLhsCorelet = -1, fixedRhsCorelet = -1;
        if (lhsCoreletInfo.size() <
            workingLhsFoldParams.at(FOLD_POS_CORELET).cardinality) {
          fixedLhsCorelet = *lhsCoreletInfo.begin();
        } else if (rhsCoreletInfo.size() <
                   workingRhsFoldParams.at(FOLD_POS_CORELET).cardinality) {
          fixedRhsCorelet = *rhsCoreletInfo.begin();
        } else {
          DT_CHECK_MSG(
              workingLhsFoldParams.at(FOLD_POS_CORELET).cardinality ==
                  workingRhsFoldParams.at(FOLD_POS_CORELET).cardinality,
              "Unexpected corelet cardinality mismatch for nodes " +
                  lhsNode->name_ + " and " + rhsNode->name_);
        }

        // Fix corelet workslice.
        //   Assumption: only one corelet is active.
        if (fixedLhsCorelet != -1) {
          workingLhsFoldParams.at(FOLD_POS_CORELET).beta +=
              (fixedLhsCorelet *
               workingLhsFoldParams.at(FOLD_POS_CORELET).alpha);
          workingLhsFoldParams.at(FOLD_POS_CORELET).alpha = 0;
          workingLhsFoldParams.at(FOLD_POS_CORELET).cardinality = 1;
        }
        if (fixedRhsCorelet != -1) {
          workingRhsFoldParams.at(FOLD_POS_CORELET).beta +=
              (fixedRhsCorelet *
               workingRhsFoldParams.at(FOLD_POS_CORELET).alpha);
          workingRhsFoldParams.at(FOLD_POS_CORELET).alpha = 0;
          workingRhsFoldParams.at(FOLD_POS_CORELET).cardinality = 1;
        }

        if (true) {
          // Verifying full range of coordinates requires the coordinates for
          // both steady-state and epilogue. The coordinate framework does not
          // keep track of the epilogue coordinates. As a temporary workaround
          // for PSUM, we are going to check for the start coordinate only.

          std::map<int64_t, int64_t> lhsPosToFixCoord, rhsPosToFixCoord;
          lhsPosToFixCoord[FOLD_POS_CORE] = lhsSliceId;
          rhsPosToFixCoord[FOLD_POS_CORE] = rhsSliceId;
          if (fixedLhsCorelet != -1) {
            lhsPosToFixCoord[FOLD_POS_CORELET] = fixedLhsCorelet;
          }
          if (fixedRhsCorelet != -1) {
            rhsPosToFixCoord[FOLD_POS_CORELET] = fixedRhsCorelet;
          }
          auto lhsStartCoord =
              lhs.coordinates_.at(coordDim).getSingleData(lhsPosToFixCoord);
          auto rhsStartCoord =
              rhs.coordinates_.at(coordDim).getSingleData(rhsPosToFixCoord);
          if (lhsStartCoord != rhsStartCoord) {
            std::cerr << "\n    Mismatch in starting coordinate("
                      << EnumsConversion::primaryDimToString.at(coordDim)
                      << "): LHS(" << lhsNode->name_ << ")[slice=" << lhsSliceId
                      << ", corelet="
                      << (fixedLhsCorelet != -1 ? fixedLhsCorelet : 0)
                      << "]= " << lhsStartCoord << " RHS(" << rhsNode->name_
                      << ")[slice=" << rhsSliceId << ", corelet="
                      << (fixedRhsCorelet != -1 ? fixedRhsCorelet : 0)
                      << "]= " << rhsStartCoord << std::endl;
            return false;
          }
        } else {
          combineContigousLevels(workingLhsFoldParams);
          combineContigousLevels(workingRhsFoldParams);

          if (!isConsistentCoord(workingLhsFoldParams, workingRhsFoldParams,
                                 coordDim, coordPropReportLevel_)) {
            return false;
          }
        }
      }
    }
  }

  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 238/382   level 1   scc 205   129 body lines
// unit: e238_getRelatedComputeCoord
// authority: ddc/ddc_fold.cpp:1976
// original: dsc2::CoordinateType<CoordinateBaseType> &Ddc::getRelatedComputeCoord( dsc2::ComputeNode *computeNode, dsc2::CoordPropInfoType &refFoldInfo, std::vector<int> &constructedInputCoords, std::vector<int> &constructedOutputCoords, int &selectedLdsIdx)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
dsc2::CoordinateType<CoordinateBaseType> &e238_getRelatedComputeCoord(
    dsc2::ComputeNode *computeNode, dsc2::CoordPropInfoType &refFoldInfo,
    std::vector<int> &constructedInputCoords,
    std::vector<int> &constructedOutputCoords, int &selectedLdsIdx)
{
  constructedInputCoords.clear();
  constructedOutputCoords.clear();
  selectedLdsIdx = -1;

  // Future work: Should we associate allocateNodes with data_connects?

  if (computeNode->isOpaqueOp_) {
    if (refFoldInfo.dataConnect == "") {
      DT_ERROR("[getRelatedComputeCoord] ComputeNode(opaque)= " +
               computeNode->name_ + ", refNode= " + refFoldInfo.refNode->name_ +
               ": data_connect is missing.");
    }

    for (size_t i = 0,
                e = computeNode->instrAttribute_.input_data_connects_.size();
         i < e; ++i) {
      if (computeNode->instrAttribute_.input_data_connects_.at(i) ==
          refFoldInfo.dataConnect) {
        constructedInputCoords.push_back(i);
        selectedLdsIdx = metadata.opaqueOps_.at(computeNode).ldsIdx_;
        return computeNode->inputCoordinates_.at(i);
      }
    }
    for (size_t i = 0,
                e = computeNode->instrAttribute_.output_data_connects_.size();
         i < e; ++i) {
      if (computeNode->instrAttribute_.output_data_connects_.at(i) ==
          refFoldInfo.dataConnect) {
        constructedOutputCoords.push_back(i);
        selectedLdsIdx = metadata.opaqueOps_.at(computeNode).ldsIdx_;
        return computeNode->outputCoordinate_;
      }
    }
    DT_ERROR("[getRelatedComputeCoord] Coordinate reference node " +
             refFoldInfo.refNode->name_ +
             ", data_connect= " + refFoldInfo.dataConnect +
             " does not correspond to any datastream of the computeNode " +
             computeNode->name_);
  }

  if (refFoldInfo.refNode->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
    // Assumption: All datastreams that are related to a given allocate node
    // must have the same coordinate arrangement. Therefore, picking the first
    // matching datastream is sufficient for the purpose of this function.
    auto refAllocNode = static_cast<dsc2::AllocateNode *>(refFoldInfo.refNode);
    if (computeNode->exUnit_ != SenComponents::LXLU &&
        refAllocNode->ldsIdx_ != -1 &&
        currDsc->labeledDs_.at(refAllocNode->ldsIdx_).scaledLdsCategory_ ==
            LabeledDsInfo::ScaledLdsCategory::SCALE_TENSOR) {
      // Compute nodes consume/produce the value tensors. The scale tensors
      // are implicitly connected through the corresponding value tensors.
      //
      // Special case:
      //   An FMUL computeNode is used on LXLU for auto conversion of FP4 to
      //   BFloat16. This computeNode accesses the scale allocation directly.
      refAllocNode = dsc2::getValueAllocation(currDsc, refAllocNode);
    }
    for (size_t i = 0, e = computeNode->inputsLdsAndLoopOffsets_.size(); i < e;
         ++i) {
      const dsc2::AllocateNode *allocNode =
          currDsc->getAllocation(computeNode->inputsLdsAndLoopOffsets_.at(i),
                                 computeNode->inputs_.at(i), true);
      if (allocNode == refAllocNode) {
        constructedInputCoords.push_back(i);
        selectedLdsIdx = computeNode->inputsLdsAndLoopOffsets_.at(i).myLdsIdx_;
        return computeNode->inputCoordinates_.at(i);
      }
    }
    for (size_t i = 0, e = computeNode->outputsLdsAndLoopOffsets_.size(); i < e;
         ++i) {
      const dsc2::AllocateNode *allocNode =
          currDsc->getAllocation(computeNode->outputsLdsAndLoopOffsets_.at(i),
                                 computeNode->outputs_.at(i), true);
      if (allocNode == refAllocNode) {
        constructedOutputCoords.push_back(i);
        selectedLdsIdx = computeNode->outputsLdsAndLoopOffsets_.at(i).myLdsIdx_;
        return computeNode->outputCoordinate_;
      }
    }
    DT_ERROR("[getRelatedComputeCoord] Coordinate reference node " +
             refFoldInfo.refNode->name_ +
             " does not correspond to any datastream of the computeNode " +
             computeNode->name_);
  }

  if (refFoldInfo.refNode->nodeType_ != dsc2::ScheduleNode::TRANSFER &&
      refFoldInfo.refNode->nodeType_ != dsc2::ScheduleNode::COMPUTE) {
    DT_ERROR("[getRelatedComputeCoord] Coordinate reference node " +
             refFoldInfo.refNode->name_ + " of type" +
             dsc2::ScheduleNode::nodeTypeToString.at(
                 refFoldInfo.refNode->nodeType_) +
             " for the computeNode " + computeNode->name_ +
             " is not supported.");
  }

  if (refFoldInfo.dataConnect == "") {
    DT_ERROR("[getRelatedComputeCoord] Reference node= " +
             refFoldInfo.refNode->name_ + ", computeNode " +
             computeNode->name_ + ": dataconnect is missing.");
  }

  // Assumption: All datastreams that are related to a given data_connect must
  // have the same coordinate arrangement. Therefore, picking the first matching
  // datastream is sufficient for the purpose of this function.
  for (size_t i = 0, e = computeNode->inputsLdsAndLoopOffsets_.size(); i < e;
       ++i) {
    if (computeNode->inputsLdsAndLoopOffsets_.at(i).dataConnect_ ==
        refFoldInfo.dataConnect) {
      constructedInputCoords.push_back(i);
      selectedLdsIdx = computeNode->inputsLdsAndLoopOffsets_.at(i).myLdsIdx_;
      return computeNode->inputCoordinates_.at(i);
    }
  }
  for (size_t i = 0, e = computeNode->outputsLdsAndLoopOffsets_.size(); i < e;
       ++i) {
    if (computeNode->outputsLdsAndLoopOffsets_.at(i).dataConnect_ ==
        refFoldInfo.dataConnect) {
      constructedOutputCoords.push_back(i);
      selectedLdsIdx = computeNode->outputsLdsAndLoopOffsets_.at(i).myLdsIdx_;
      return computeNode->outputCoordinate_;
    }
  }

  DT_ERROR("[getRelatedComputeCoord] Coordinate reference node " +
           refFoldInfo.refNode->name_ +
           " does not correspond to any datastream of the computeNode " +
           computeNode->name_);
}

// ------------------------------------------------------------------------------------------------
// entry 239/382   level 1   scc 200   60 body lines
// unit: e239_computeParamsForRowSplitFold
// authority: ddc/ddc_fold.cpp:2161
// original: void Ddc::computeParamsForRowSplitFold( const PrimaryDimTypes &currDim, dsc2::FoldParamInfoType &resultFoldParams, RowGroupInfo &rowGroup)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e239_computeParamsForRowSplitFold(
    const PrimaryDimTypes &currDim, dsc2::FoldParamInfoType &resultFoldParams,
    RowGroupInfo &rowGroup)
{
  // Setup parameters for the rowsplit fold. The default parameters represent an
  // identity fold.
  if (!currDsc->dataStageParam_.at(metadata.core_dstgid)
           .ss_.rowSplit_.count(currDim) ||
      rowGroup.nodeInfo.empty()) {
    getDefaultRowSplitFold(resultFoldParams);
    return;
  }
  resultFoldParams.foldDimLabel = "rowsplit_fold";
  // Jump in coordinate from one PT-row to the next.
  CoordinateBaseType alpha = 0, beta = 0;
  if (rowGroup.cat == RowGroupInfo::Category::ROW_TO_SAME_ROW) {
    alpha = 0;
    beta = rowGroup.nodeInfo.at(0).beta;
  } else {
    alpha = rowGroup.nodeInfo.at(1).beta - rowGroup.nodeInfo.at(0).beta;
    beta = rowGroup.nodeInfo.at(0).beta;
  }
  int numNodes = rowGroup.nodeInfo.size();

  if (rowGroup.cat == RowGroupInfo::Category::ROW_TO_SAME_ROW ||
      rowGroup.cat == RowGroupInfo::Category::ROW_NORTH_SOUTH) {
    // Rowsplit fold is a single PT row.
    resultFoldParams.alpha = alpha;
    resultFoldParams.beta = beta;
    resultFoldParams.cardinality = 1;
  } else if (numNodes == dscGlobal.sysDef.numPTRows) {
    // Rowsplit fold is for all PT rows.
    if (rowGroup.nodeInfo.at(0).row < rowGroup.nodeInfo.at(1).row) {
      // Monotonically increasing row numbers.
      for (int i = 1; i < numNodes; ++i) {
        if (rowGroup.nodeInfo.at(i - 1).row >= rowGroup.nodeInfo.at(i).row) {
          DT_ERROR(
              "Break found in expected monotonically increasing row sequence. "
              "Related scheduleNodes: " +
              rowGroup.nodeInfo.at(i - 1).node->name_ + ", " +
              rowGroup.nodeInfo.at(i).node->name_ + ".");
        }
      }
    } else {
      // Monotonically decreasing order.
      for (int i = 1; i < numNodes; ++i) {
        if (rowGroup.nodeInfo.at(i - 1).row <= rowGroup.nodeInfo.at(i).row) {
          DT_ERROR(
              "Break found in expected monotonically decreasing row sequence. "
              "Related scheduleNodes: " +
              rowGroup.nodeInfo.at(i - 1).node->name_ + ", " +
              rowGroup.nodeInfo.at(i).node->name_ + ".");
        }
      }
    }
    resultFoldParams.alpha = alpha;
    resultFoldParams.beta = rowGroup.nodeInfo.at(0).beta;
    resultFoldParams.cardinality = dscGlobal.sysDef.numPTRows;
  } else {
    DT_ERROR("Unsupported number of rows " +
             std::to_string(rowGroup.nodeInfo.size()) + ".");
  }
}

// ------------------------------------------------------------------------------------------------
// entry 240/382   level 1   scc 210   156 body lines
// unit: e240_buildFoldForExternalAllocation
// authority: ddc/ddc_fold.cpp:2238
// original: void Ddc::buildFoldForExternalAllocation(dsc2::AllocateNode *allocNode)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e240_buildFoldForExternalAllocation(dsc2::AllocateNode *allocNode)
{
  if (allocNode->allocateCoordinates_.foldConstructed()) {
    if (coordPropReportLevel_ > 1) {
      std::cout << "\n[buildFoldForExternalAllocation] Upstream-filled "
                   "coordinates of external allocNode "
                << allocNode->name_ << ":\n";
      allocNode->allocateCoordinates_.debugPrint(std::cout);
    }
    return;
  }

  if (allocNode->ldsIdx_ < 0) {
    return;
  }

  // Find the enclosing loop chain. Collect the associated dimensions.
  dsc2::VectorOfLoopAndDim loopChain;
  getEnclosingLoopsAndRelatedDims(currDsc, allocNode, loopChain,
                                  loopsBelowChunkBoundary);

  if (loopChain.empty()) {
    // TO DO: Determine a robust response.
    return;
  }

  if (coordPropReportLevel_ > 0) {
    std::cout << "\n>>> buildFoldForExternalAllocation: allocNode= "
              << allocNode->name_ << "(" << allocNode << ")";
  }

  allocNode->allocateCoordinates_.setPadding(allocNode->padding_);
  const auto &allocLds = currDsc->labeledDs_.at(allocNode->ldsIdx_);
  // Construct a foldManager for each relevant dimension.
  for (auto allocDim : allocNode->layoutDimOrder_) {
    if (allocNode->allocateCoordinates_.coordinates_.count(allocDim)) {
      continue;
    }
    PrimaryDimAndKind currDim = allocDim;
    auto &allocFoldCurrDim =
        allocNode->allocateCoordinates_.coordinates_[currDim.dim_];
    std::string foldDimStr =
        EnumsConversion::primaryDimToString.at(currDim.dim_);
    PadType allocPaddingForCurrDim =
        allocNode->allocateCoordinates_.getPadding(currDim.dim_);

    // Build element arrangement info along with the temporal fold.

    // TO DO: The following is a placeholder and includes only the monotonically
    // increasing contiguous case. Enhance the following simple case to reflect
    // the actual element arrangement layout.
    // Future work: The following datastructure should be part of dsc.
    std::vector<dsc2::FoldParamInfoType> elemArr;
    int64_t coreDDimSize = currDsc->dataStageParam_.at(metadata.core_dstgid)
                               .ss_.dataStageDimToVal_compView_st(
                                   currDim.dim_, SenComponents::NO_COMPONENT, 0,
                                   {currDim.dim_, allocPaddingForCurrDim});
    elemArr.push_back({/*stride*/ 1, /*beta*/ 0, /*card*/ coreDDimSize, ""});

    dsc2::VectorOfLoopAndDim relatedLoops;
    int dimIdx = currDsc->getDimIndexInLayoutOrder(allocLds.dsType_, allocDim);
    int scale = dimIdx < 0 ? 1 : allocLds.scale_.at(dimIdx);
    if (scale < 0) {
      // This is a broadcast dimension.
      buildFoldForBroadcastDim(currDsc, allocNode->allocateCoordinates_,
                               allocLds, currDim.dim_, scale);

      continue;
    }

    // The dimension is a non-broadcast dimension.
    collectRelatedLoops(currDsc, currDim, loopChain, relatedLoops,
                        allocPaddingForCurrDim);

    dsc2::LoopDistributionParamPerNodeType &loopParamsAfterDistribution =
        loopDistributionParamInfo[allocNode][allocNode];
    std::vector<dsc2::FoldParamInfoType> elemArrParamsAfterDistribution;

    // Distribute temporal loops over element arrangements and compute the fold
    // parameters for temporal and element arrangement folds.
    dsc2::distributeElemArrToTemporalLoops(
        currDsc, currDim, allocNode, allocNode->ldsIdx_, allocPaddingForCurrDim,
        allocPaddingForCurrDim, allocNode->component_, allocNode->component_,
        relatedLoops, elemArr, loopParamsAfterDistribution,
        elemArrParamsAfterDistribution, 0 /* for one corelet */,
        coordPropReportLevel_);

    // Add the element arrangement folds
    for (int64_t i = 0, e = elemArrParamsAfterDistribution.size(); i < e; ++i) {
      CoordinateBaseType alpha = elemArrParamsAfterDistribution.at(i).alpha;
      CoordinateBaseType beta = elemArrParamsAfterDistribution.at(i).beta;
      allocNode->allocateCoordinates_.addFold(
          currDim.dim_, dsc2::CoordinateCategory::ELEM_ARR_COORD,
          elemArrParamsAfterDistribution.at(i).cardinality,
          "elem_arr_" + std::to_string(i), alpha, beta, 0);
    }

    // Temporal folds for the loops
    //   Order: innermost to outermost
    for (auto &loopInfo : relatedLoops) {
      auto loopNode = loopInfo.loopNode;
      auto loopDim = loopInfo.dimAndKind;

      CoordinateBaseType loopAlpha = -1;
      CoordinateBaseType loopBeta = -1;
      int iterationCount = -1;
      if (loopNode->isParametricLoop()) {
        loopAlpha = loopNode->parametricStride(currDsc);
        loopBeta = 0;
        iterationCount = loopNode->parametricIterCount(
            currDsc, 0 /* To generalize*/, SenComponents::NO_COMPONENT, -1);
      } else {
        auto &numDs = currDsc->dataStageParam_.at(loopNode->numId_).ss_;
        auto &denDs = currDsc->dataStageParam_.at(loopNode->denId_).ss_;
        // To decide: Do we need to consider the metaDimKind
        // Assumption: Spatial fold includes a level for corelets. Therefore,
        // get the datastage values per corelet. To do:
        //   Need to have special case when the corelets may have imbalanced
        //   distribution.
        auto denVal = denDs.primaryDimToVal_st(
            loopDim.dim_, SenComponents::NO_COMPONENT, -1, 0);
        iterationCount = numDs.primaryDimToVal_st(
                             loopDim.dim_, SenComponents::NO_COMPONENT, -1, 0) /
                         denVal;

        // Set alpha and beta for new fold
        loopAlpha = denVal;
        loopBeta = 0;
        if (loopParamsAfterDistribution.count(loopNode)) {
          loopAlpha =
              loopParamsAfterDistribution.at(loopNode).at(currDim.dim_).alpha;
        }
      }

      allocNode->allocateCoordinates_.addFold(
          currDim.dim_, dsc2::CoordinateCategory::TEMPORAL_COORD,
          iterationCount, loopNode->name_ + " " + foldDimStr, loopAlpha,
          loopBeta, 0);
    }

    // Build rowsplit fold.
    allocNode->allocateCoordinates_.addFold(
        currDim.dim_, dsc2::CoordinateCategory::SPATIAL_COORD, 1,
        "rowsplit_fold_" + foldDimStr, 0, 0, 0);
    // Add spatial folds for core and corelet
    buildSpatialFold(allocNode, currDim.dim_, allocPaddingForCurrDim,
                     allocNode->allocateCoordinates_);
  }

  allocNode->allocateCoordinates_.completeFoldConstruction();
  if (coordPropReportLevel_ > 2) {
    std::cout << "\n\n===================================================="
              << "\nallocNodeCoordinates (" << allocNode->name_ << ") at end: ";
    allocNode->allocateCoordinates_.debugPrint(std::cout);
    std::cout << std::endl;
  }
}

// ------------------------------------------------------------------------------------------------
// entry 241/382   level 1   scc 199   166 body lines
// unit: e241_relateLoopsToAllocElemArr
// authority: ddc/ddc_fold.cpp:2872
// original: void Ddc::relateLoopsToAllocElemArr( dsc2::CoordPropInfoType &coordPropInfo, const PrimaryDimTypes dim, const dsc2::CoordinateType<CoordinateBaseType> &coordinate, const dsc2::VectorOfLoopAndDim &refLoopChain, dsc2::LoopDistributionParamPerNodeType &loopParamsAfterDistribution)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e241_relateLoopsToAllocElemArr(
    dsc2::CoordPropInfoType &coordPropInfo, const PrimaryDimTypes dim,
    const dsc2::CoordinateType<CoordinateBaseType> &coordinate,
    const dsc2::VectorOfLoopAndDim &refLoopChain,
    dsc2::LoopDistributionParamPerNodeType &loopParamsAfterDistribution)
{
  // coordPropInfo has the information in reverse direction.
  //   - coordPropInfo.nodeToFold: An allocateNode. The coordinates are already
  //   constructed and can not be modified. The existing element arrangments are
  //   going to be associated with the loops around the other scheduleNode.
  //   - coordPropInfo.refNode: A non-alloc node. Loops around this node are
  //   going to be associated with the element arrangement of the allocate node.
  DT_CHECK_MSG(
      coordPropInfo.nodeToFold->nodeType_ == dsc2::ScheduleNode::ALLOCATE,
      "[relateLoopsToAllocElemArr] Base node " +
          coordPropInfo.nodeToFold->name_ + " is not an allocateNode.");
  dsc2::AllocateNode *baseAllocNode =
      static_cast<dsc2::AllocateNode *>(coordPropInfo.nodeToFold);
  dsc2::ScheduleNode *refNode = coordPropInfo.refNode;
  if (coordPropReportLevel_ > 0) {
    std::cout << "\n\n>>> [relateLoopsToAllocElemArr]:"
              << "\n  node= " << refNode->name_ << "(" << refNode << ")"
              << "\n  baseAllocNode= " << baseAllocNode->name_ << "("
              << baseAllocNode << ")";

    if (coordPropReportLevel_ > 2) {
      std::cout << "\n\n===================================================="
                << "\n\nallocation coordinate: ";
      const_cast<dsc2::AllocateNode *>(baseAllocNode)
          ->allocateCoordinates_.debugPrint(std::cout);
      if (baseAllocNode->sliceViewCoordinates_.foldConstructed()) {
        std::cout << "\n\nslice-view coordinates:"
                  << "\n--------------------------";
        const_cast<dsc2::AllocateNode *>(baseAllocNode)
            ->sliceViewCoordinates_.debugPrint(std::cout);
      }
    }
  }

  DT_CHECK_MSG(baseAllocNode->ldsIdx_ >= 0,
               "Base allocateNode " + baseAllocNode->name_ +
                   " does not have a valid labeledDs index.");

  SenComponents sizeRefComp = SenComponents::ALL;
  SenComponents propRefComp = baseAllocNode->component_;

  // The following calculation of sizeRefComp and propRefComp is a candidate for
  // refactoring.
  if (refNode->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    auto computeNode = static_cast<dsc2::ComputeNode *>(refNode);
    sizeRefComp = computeNode->exUnit_;
    if (propRefComp == SenComponents::PTXRF ||
        propRefComp == SenComponents::PTARF) {
      // *** This is a kludge. FIND A GENERAL SOLUTION ***
      //  PTXRF and PTARF represents allocation for all PT rows. On the other
      //  hand, a computeNode on the PT represents computation on an individual
      //  PT row.
      propRefComp = computeNode->exUnit_;
    }
  } else if (refNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
    int allocPosInDest = -1;
    auto transferNode = static_cast<dsc2::TransferNode *>(refNode);
    if (coordPropInfo.refIsProducer) {
      // Transfer -> allocate
      for (int i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e;
           ++i) {
        if (matchDataStream(currDsc, coordPropInfo,
                            transferNode->dstLdsAndLoopOffsets_.at(i),
                            transferNode->dstVias_.at(i).loc_.storage_,
                            false /* !checkRefForAllocate */)) {
          sizeRefComp = transferNode->dstVias_.at(i).loc_.unit_;
          propRefComp = sizeRefComp;
          allocPosInDest = i;
          break;
        }
      }
    } else {
      // allocate -> transfer
      sizeRefComp = transferNode->src_.unit_;
      propRefComp = sizeRefComp;
    }
    // Override the comp in case transfer is for a single PT row.
    // Example: transfer from LXLU to PTRow4.
    //          - Src unit is lxlu.
    //          - Dest unit is ptrow4.
    //          In this case, coordinate should be for the PT row 4 only.
    if (!EnumsConversion::senCompToRowId.count(sizeRefComp)) {
      if (EnumsConversion::senCompToRowId.count(transferNode->src_.unit_)) {
        sizeRefComp = transferNode->src_.unit_;
      } else if (EnumsConversion::senCompToRowId.count(
                     transferNode->dstVias_.at(0).loc_.unit_)) {
        sizeRefComp = transferNode->dstVias_.at(0).loc_.unit_;
      }
    }

    // Override the comp in case transfer involves PE or SFP. This is needed to
    // account for PE-SFP splitting.
    //   In case the component already represents an individual PT row, that
    //   setting has higher priority than PE or SFP.
    //   Example: transfer_lds4_src:ptrow6_dst:pe in Conv_0 in resnet. The size
    //   should correspond to row6 only.
    if (!EnumsConversion::senCompToRowId.count(sizeRefComp) &&
        !is_any_of(sizeRefComp, SenComponents::PE, SenComponents::SFP)) {
      if (is_any_of(transferNode->src_.unit_, SenComponents::PE,
                    SenComponents::SFP)) {
        sizeRefComp = transferNode->src_.unit_;
      } else if (is_any_of(transferNode->dstVias_.at(0).loc_.unit_,
                           SenComponents::PE, SenComponents::SFP)) {
        sizeRefComp = transferNode->dstVias_.at(0).loc_.unit_;
      }
    }
  }

  int ptRowId = -1;
  if (EnumsConversion::senCompToRowId.count(sizeRefComp)) {
    ptRowId = EnumsConversion::senCompToRowId.at(sizeRefComp);
  }
  auto &effectiveRefCoord =
      (ptRowId != -1 && baseAllocNode->sliceViewCoordinates_.foldConstructed()
           ? baseAllocNode->sliceViewCoordinates_
           : baseAllocNode->allocateCoordinates_);

  const auto &allocLds = currDsc->labeledDs_.at(baseAllocNode->ldsIdx_);

  auto &cfm = effectiveRefCoord.coordinates_.at(dim);
  std::vector<dsc2::FoldParamInfoType> foldParams;
  gatherFoldParams(cfm, foldParams);

  dsc2::VectorOfLoopAndDim relatedLoops;
  int dimIdx = currDsc->getDimIndexInLayoutOrder(allocLds.dsType_, dim);
  int scale = dimIdx < 0 ? 1 : allocLds.scale_.at(dimIdx);
  if (scale > 0) {
    // The dimension is a non-broadcast dimension.
    //   Note: dim is of type PrimaryDimTypes, metaDimKind is missing here.
    collectRelatedLoops(currDsc, dim, refLoopChain, relatedLoops,
                        coordinate.getPadding(dim));
  }

  int spatialFoldEnds = effectiveRefCoord.getNumOfSpatialFolds(dim) - 1;
  int refTemporalCount = effectiveRefCoord.getNumOfTemporalFolds(dim);
  int temporalFoldEnds = spatialFoldEnds + refTemporalCount;

  if (refTemporalCount < relatedLoops.size()) {
    // The node has more enclosing loops than the reference
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
    distributeElemArrToTemporalLoops(
        currDsc, dim, refNode, baseAllocNode->ldsIdx_,
        effectiveRefCoord.getPadding(dim), coordinate.getPadding(dim),
        sizeRefComp, propRefComp, loopsToDistribute, elemArr,
        loopParamsAfterDistribution, elemArrParamsAfterDistribution,
        0 /* for one corelet */, coordPropReportLevel_);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 242/382   level 1   scc 221   270 body lines
// unit: e242_canUseFifo
// authority: ddc/ddc_transformation.cpp:15
// original: bool Ddc::canUseFifo(const dsc2::TransferNode *transferNode, size_t dstIndex, const dsc2::ScheduleNode *consumer) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e242_canUseFifo(const dsc2::TransferNode *transferNode, size_t dstIndex,
                     const dsc2::ScheduleNode *consumer) const
{
  if (transferNode->dstVias_.at(dstIndex).loc_.unit_ ==
      SenComponents::LXLUVALUE)
    return false;

  // Communication channel that would be used for the result after the
  // transformation.
  std::pair<SenComponents, SenComponents> communicationChannel;
  if (transferNode->dstVias_.at(dstIndex).via_.empty()) {
    communicationChannel.first = transferNode->src_.unit_;
  } else {
    communicationChannel.first =
        transferNode->dstVias_.at(dstIndex).via_.back();
  }
  communicationChannel.second = transferNode->dstVias_.at(dstIndex).loc_.unit_;
  if (is_any_of(communicationChannel.first, SenComponents::NO_COMPONENT,
                SenComponents::CONSTANT)) {
    if (transformationReportLevel_ > 1) {
      std::cerr << "\nSkip register transformation can not be performed on "
                   "transferNode "
                << getNodeDescription(transferNode) << " as the source "
                << EnumsConversion::senComponentsToString.at(
                       communicationChannel.first)
                << " can not be used as a FIFO.";
    }
    return false;
  }

  std::set<SenComponents> otherConsumerInputs;
  if (consumer->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    const dsc2::ComputeNode *compConsumer =
        static_cast<const dsc2::ComputeNode *>(consumer);
    // collect other inputs of the consumer to avoid triangular dependencies
    // that would rely on fifo depth to avoid deadlocks
    for (auto &input : compConsumer->inputs_) {
      if (input != transferNode->dstVias_.at(dstIndex).loc_.storage_)
        otherConsumerInputs.insert(input);
    }

    // Temporary check to work around a DCC transformation where DCC transforms
    // a compute with
    //    "inputs_" : ["lxlu", "lxlu", "pelrf"],
    // to an FMA that reads two different FIFO elements, as follows.
    //
    //   PE_FMA mask=255 src0=lxlu src1=lxlu src2=lxlu tgtrf=no unrlfldsrc0=0
    //     unrlfldsrc1=0 unrlfldsrc2=0 unrlfldtgt=0 unroll=x1
    //   PE_FMA be=be mask=255 src0=reuse src1=lxlu src2=R0 tgtrf=R0
    //     unrlfldsrc0=0 unrlfldsrc1=0 unrlfldsrc2=0 unrlfldtgt=0 unroll=x1
    //
    const std::string &resultDc =
        transferNode->dstLdsAndLoopOffsets_.at(dstIndex).dataConnect_;
    int count = 0;

    for (auto &inputIt : compConsumer->inputsLdsAndLoopOffsets_) {
      if (inputIt.dataConnect_ == resultDc) {
        if (count > 0) {
          return false;
        }
        ++count;
      }
    }
  }

  std::vector<const dsc2::ScheduleNode *> pathToTransfer, pathToConsumer;
  currDsc->getInnermostCommonAncestor(transferNode, consumer, pathToTransfer,
                                      pathToConsumer);

  // Ensure that none of the paths includes an unsupported node.
  auto checkUnsupportedNode = [&](std::vector<const dsc2::ScheduleNode *> &path,
                                  const std::string &pathType) -> bool {
    std::string errMsg =
        (!transformationReportLevel_
             ? ""
             : "\nSkip register transformation can not be performed on "
               "transferNode " +
                   Ddc::getNodeDescription(transferNode) + " as the path to " +
                   pathType + " " + Ddc::getNodeDescription(path.at(0)) +
                   " includes node ");

    for (size_t i = 1, e = path.size(); i < e; ++i) {
      const dsc2::ScheduleNode *currNode = path.at(i);
      if (currNode->nodeType_ == dsc2::ScheduleNode::CONDITION) {
        if (transformationReportLevel_ > 1) {
          std::cerr << errMsg << currNode->name_ << " of type condition.";
        }
        return false;
      } else if (currNode->nodeType_ == dsc2::ScheduleNode::LOOP &&
                 static_cast<const dsc2::LoopNode *>(currNode)
                     ->isParametricLoop()) {
        if (transformationReportLevel_ > 1) {
          std::cerr << errMsg << currNode->name_ << " of type parametric-loop.";
        }
        return false;
      }
    }
    return true;
  };

  if (!checkUnsupportedNode(pathToConsumer, "consumer")) {
    return false;
  }
  if (!checkUnsupportedNode(pathToTransfer, "transfer")) {
    return false;
  }

  auto emitSkipRegisterError = [&](const dsc2::ScheduleNode *node) {
    if (transformationReportLevel_ > 1) {
      std::cerr << "\nSkip register transformation can not be "
                   "performed on transferNode as the path between transfer "
                << Ddc::getNodeDescription(transferNode) << " and the consumer "
                << Ddc::getNodeDescription(consumer)
                << " includes a scheduleNode " << Ddc::getNodeDescription(node)
                << " that uses the intended communication channel "
                << EnumsConversion::senComponentsToString.at(
                       communicationChannel.first)
                << " FIFO of "
                << EnumsConversion::senComponentsToString.at(
                       communicationChannel.second);
    }
  };

  auto usesCommunicationChannel =
      [&](const dsc2::ScheduleNode *producerNode,
          const dsc2::ScheduleNode *node,
          const std::pair<SenComponents, SenComponents> &communicationChannel)
      -> bool {
    auto usesCommunicationChannelImpl =
        [&](const dsc2::ScheduleNode *producerNode,
            const dsc2::ScheduleNode *node,
            const std::pair<SenComponents, SenComponents> &communicationChannel,
            auto &self) -> bool {
      if (node == producerNode) {
        // Skip checking the original producer node.
        return false;
      }
      if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
        const dsc2::ComputeNode *compNode =
            static_cast<const dsc2::ComputeNode *>(node);
        if (compNode->exUnit_ == communicationChannel.second ||
            otherConsumerInputs.count(compNode->exUnit_)) {
          for (auto it : compNode->inputs_) {
            if (it == communicationChannel.first) {
              emitSkipRegisterError(compNode);
              return true;
            }
          }
        }

        if (compNode->exUnit_ == communicationChannel.first) {
          for (auto it : compNode->outputs_) {
            if (it == communicationChannel.second) {
              emitSkipRegisterError(compNode);
              return true;
            }
          }
        }
      } else if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
        const dsc2::TransferNode *transferNode =
            static_cast<const dsc2::TransferNode *>(node);
        SenComponents comp = SenComponents::NO_COMPONENT;
        for (auto dstViaIt : transferNode->dstVias_) {
          comp = transferNode->src_.unit_;
          for (auto it : dstViaIt.via_) {
            if (comp == communicationChannel.first &&
                (it == communicationChannel.second ||
                 otherConsumerInputs.count(it))) {
              emitSkipRegisterError(transferNode);
              return true;
            }
            comp = it;
          }
          if (comp == communicationChannel.first &&
              (dstViaIt.loc_.unit_ == communicationChannel.second ||
               otherConsumerInputs.count(dstViaIt.loc_.unit_))) {
            emitSkipRegisterError(transferNode);
            return true;
          }
        }
      } else if (auto *currBlock =
                     dynamic_cast<const dsc2::BlockNode *>(node)) {
        for (auto childNode : currBlock->getNextView(SenComponents::ALL)) {
          if (self(producerNode, childNode, communicationChannel, self)) {
            return true;
          }
        }
      }
      return false;
    };
    return usesCommunicationChannelImpl(
        producerNode, node, communicationChannel, usesCommunicationChannelImpl);
  };

  const dsc2::BlockNode *currParentBlock =
      static_cast<const dsc2::BlockNode *>(pathToTransfer.back());
  const dsc2::ScheduleNode *startChildNode =
      pathToTransfer.at(pathToTransfer.size() - 2);
  const dsc2::ScheduleNode *endChildNode =
      pathToConsumer.at(pathToConsumer.size() - 2);

  bool checkFifoUsage = false;
  for (auto currChild : currParentBlock->getNextView(SenComponents::ALL)) {
    if (currChild == startChildNode) {
      checkFifoUsage = true;
    } else if (!checkFifoUsage) {
      continue;
    }
    if (usesCommunicationChannel(transferNode, currChild,
                                 communicationChannel)) {
      return false;
    }

    if (currChild == endChildNode) {
      // The path to consumer (inclusive) does not access the communication
      // channel of interest.
      break;
    }
  }

  if (transferNode->getPrev() == consumer->getPrev()) {
    return true;
  }

  auto transferDims = currDsc->getNonBroadcastLdsDimSet(
      transferNode->srcLdsAndLoopOffsets_.myLdsIdx_);

  auto loopsWithoutReuse =
      [&](const std::vector<const dsc2::ScheduleNode *> &path,
          const std::set<PrimaryDimTypes> &relevantDims) {
        // Collect dimension-info for all loops on the path.
        for (size_t i = 0, e = path.size() - 1; i < e; ++i) {
          if (path.at(i)->nodeType_ == dsc2::ScheduleNode::LOOP) {
            const dsc2::LoopNode *loopNode =
                static_cast<const dsc2::LoopNode *>(path.at(i));
            for (auto dimIt : loopNode->dims_) {
              if (!relevantDims.count(dimIt.dim_) ||
                  dimIt.kind_ == MetaDimKind::WindowDim) {
                // Loop dimension is not related to the transfer or is a window
                // dimension that may cause reuse. As the transfer result is
                // reused, the skip-register transformation can not be done
                // unless the dimension is known to be 1.
                if (!dataStageExplorationDone_) {
                  return false;
                } else {
                  // Check if the dimension size is 1
                  const auto &numDs =
                      currDsc->dataStageParam_.at(loopNode->numId_).ss_;
                  const auto &denDs =
                      currDsc->dataStageParam_.at(loopNode->denId_).ss_;
                  if (numDs.primaryDimToVal_st(dimIt.dim_) !=
                      denDs.primaryDimToVal_st(dimIt.dim_)) {
                    return false;
                  }
                }
              }
            }
          }
        }
        return true;
      };

  if (!loopsWithoutReuse(pathToConsumer, transferDims)) {
    return false;
  }

  if (!loopsWithoutReuse(pathToTransfer, transferDims)) {
    return false;
  }

  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 243/382   level 1   scc 222   321 body lines
// unit: e243_canUseLatch
// authority: ddc/ddc_transformation.cpp:287
// original: bool Ddc::canUseLatch(const dsc2::TransferNode *transferNode, size_t dstIndex, std::vector<dsc2::ScheduleNode *> consumers) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e243_canUseLatch(const dsc2::TransferNode *transferNode, size_t dstIndex,
                      std::vector<dsc2::ScheduleNode *> consumers) const
{
  if (transferNode->dstVias_.at(dstIndex).loc_.unit_ ==
      SenComponents::LXLUVALUE)
    return true;
  if (transferNode->dstVias_.at(dstIndex).loc_.storage_ ==
      SenComponents::LXLUSCALEREG)
    return false;

  if (!dataStageExplorationDone_) {
    DT_ERROR(
        "\n[Use-latch]: Transformation can not be done before dataStage "
        "exploration is complete.");
  }

  if (transferNode->src_.unit_ == SenComponents::CONSTANT) return false;

  for (int i = 0; i < currDsc->numCoreletsUsed_DSC2_; ++i) {
    int transferSize = currDsc->getBlockTransferSize(
        *transferNode, transferNode->src_.unit_, i, false, true);
    if (transferSize != 1) {
      if (transformationReportLevel_ > 1) {
        std::cerr << "\n  [Use-latch]: Can not perform transformation on "
                  << getNodeDescription(transferNode)
                  << " as the transfer size " << transferSize
                  << " is greater than 1 (element).";
      }
      return false;
    }
  }

  DT_CHECK(!consumers.empty());
  int computeConsumersCount = 0;
  for (auto consumer : consumers) {
    if (consumer->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      ++computeConsumersCount;
      const dsc2::ComputeNode *compConsumer =
          static_cast<const dsc2::ComputeNode *>(consumer);
      const std::string &resultDc =
          transferNode->dstLdsAndLoopOffsets_.at(dstIndex).dataConnect_;
      int count = 0;

      for (const auto& inputIt : compConsumer->inputsLdsAndLoopOffsets_) {
        if (inputIt.dataConnect_ == resultDc) {
          if (count > 0) {
            return false;
          }
          ++count;
        }
      }
    }
  }

  if (computeConsumersCount != consumers.size()) {
    if (transformationReportLevel_ > 1) {
      std::cerr << "\n  [Use-latch]: Can not perform transformation as "
                   "consumers include non computeNodes.";
    }
    return false;
  }

  SenComponents destUnit = transferNode->dstVias_.at(dstIndex).loc_.unit_;
  const std::string &ResultDataConnect =
      transferNode->dstLdsAndLoopOffsets_.at(dstIndex).dataConnect_;

  // Check:
  // - All computes have the same computeType. This allows latched data to be
  //   used on the same input port for all compute-consumers.
  // - There is no non-consumer compute in between the producer and consumers
  //   (includes entire loop-body).

  // Ensure that all consumers are computeNodes and have the same computeType
  // and the computation is not on PT.
  const dsc2::ComputeNode *prevCompConsumer = nullptr;
  for (auto consumerIt : consumers) {
    if (consumerIt->nodeType_ != dsc2::ScheduleNode::COMPUTE) {
      continue;
    }
    const dsc2::ComputeNode *currCompConsumer =
        static_cast<dsc2::ComputeNode *>(consumerIt);
    if (currCompConsumer->exUnit_ == SenComponents::PT) {
      // Operations on PT may not have access to all FIFOs for all sources.
      // Skip the transformation.
      return false;
    }
    if (!prevCompConsumer) {
      prevCompConsumer = currCompConsumer;
    } else if (prevCompConsumer->type_ != currCompConsumer->type_) {
      if (transformationReportLevel_ > 1) {
        std::cerr
            << "\n  [Use-latch]: Can not perform transformation as consumer "
            << getNodeDescription(prevCompConsumer)
            << " has a different computeType "
            << EnumsConversion::computeTypeToString.at(prevCompConsumer->type_)
            << " than the computeType "
            << EnumsConversion::computeTypeToString.at(currCompConsumer->type_)
            << " of consumer " << getNodeDescription(currCompConsumer);
      }
      return false;
    }
  }

  // Check that all consumers use the data_connect for the same source
  // position(s).
  std::set<int> srcIndex;
  bool firstConsumer = true;
  for (auto consumerIt : consumers) {
    if (consumerIt->nodeType_ != dsc2::ScheduleNode::COMPUTE) {
      continue;
    }
    dsc2::ComputeNode *currComputeNode =
        static_cast<dsc2::ComputeNode *>(consumerIt);
    if (currComputeNode->isOpaqueOp_) {
      if (transformationReportLevel_ > 1) {
        std::cerr
            << "\n  [Use-latch]: Can not perform transformation as consumer "
            << getNodeDescription(currComputeNode)
            << " is an opaque operation.";
      }
      return false;
    }
    int useCount = 0;
    for (size_t i = 0, e = currComputeNode->inputsLdsAndLoopOffsets_.size();
         i < e; ++i) {
      if (currComputeNode->inputsLdsAndLoopOffsets_.at(i).dataConnect_ ==
          ResultDataConnect) {
        ++useCount;
        if (firstConsumer) {
          srcIndex.insert(i);
          continue;
        } else if (!srcIndex.count(i)) {
          if (transformationReportLevel_ > 1) {
            std::cerr << "\n  [Use-latch]: Can not perform transformation on "
                         "transfer "
                      << getNodeDescription(transferNode)
                      << " as input position " << i << " of consumer "
                      << getNodeDescription(currComputeNode)
                      << " does not match the corresponding input positions "
                      << " of previous consumers [";
            for (auto it : srcIndex) {
              std::cerr << " " << it;
            }
            std::cerr << "]";
          }
          return false;
        }
      }
    }
    if (!firstConsumer && srcIndex.size() != useCount) {
      if (transformationReportLevel_ > 1) {
        std::cerr << "\n  [Use-latch]: Can not perform transformation on "
                     "transfer "
                  << getNodeDescription(transferNode)
                  << " consumers use data_connect " << ResultDataConnect
                  << " for varying number of inputs.";
      }
      return false;
    }
    // Collection of data_connect usage from the first consumer is complete.
    firstConsumer = false;
  }
  DT_CHECK(!srcIndex.empty());

  // Analyze the schedule tree for the scope of interest between the producer
  // and the consumers.

  for (auto consumer : consumers) {
    std::vector<const dsc2::ScheduleNode *> pathToTransfer, pathToConsumer;
    currDsc->getInnermostCommonAncestor(transferNode, consumer, pathToTransfer,
                                        pathToConsumer);

    // Ensure that none of the paths includes an unsupported node.
    auto checkUnsupportedNode =
        [&](std::vector<const dsc2::ScheduleNode *> &path,
            const std::string &pathType) -> bool {
      std::string errMsg =
          (!transformationReportLevel_
               ? ""
               : "\n  [Use-latch]: Transformation can not be performed on "
                 "transferNode " +
                     Ddc::getNodeDescription(transferNode) +
                     " as the path to consumer " +
                     Ddc::getNodeDescription(consumer) + " includes node ");

      for (size_t i = 1, e = path.size() - 1; i < e; ++i) {
        const dsc2::ScheduleNode *currNode = path.at(i);
        if (currNode->nodeType_ == dsc2::ScheduleNode::CONDITION) {
          if (transformationReportLevel_ > 1) {
            std::cerr << errMsg << currNode->name_ << " of type condition.";
          }
          return false;
        } else if (currNode->nodeType_ == dsc2::ScheduleNode::LOOP &&
                   static_cast<const dsc2::LoopNode *>(currNode)
                       ->isParametricLoop()) {
          if (transformationReportLevel_ > 1) {
            std::cerr << errMsg << currNode->name_
                      << " of type parametric-loop.";
          }
          return false;
        }
      }
      return true;
    };

    if (!checkUnsupportedNode(pathToConsumer, "consumer")) {
      return false;
    }
    if (!checkUnsupportedNode(pathToTransfer, "transfer")) {
      return false;
    }

    auto emitUseLatchError = [&](const dsc2::ScheduleNode *node, int srcIndex,
                                 std::string dataConnect) {
      if (transformationReportLevel_ > 1) {
        std::cerr << "\n  [Use-latch]: Transformation can not be "
                     "performed on transferNode as the path between transfer "
                  << Ddc::getNodeDescription(transferNode)
                  << " and the consumer " << Ddc::getNodeDescription(consumer)
                  << " includes a scheduleNode "
                  << Ddc::getNodeDescription(node)
                  << " that has incompatible usage of the indended latch for "
                     "data_connect "
                  << dataConnect << " for source " << srcIndex;
      }
    };

    auto analyzeLatchUsage = [&](const dsc2::ScheduleNode *producerNode,
                                 const dsc2::ScheduleNode *node,
                                 const SenComponents unit,
                                 const std::set<int> &srcIndex,
                                 const std::string &dataConnect) -> bool {
      auto analyzeLatchUsageImpl =
          [&](const dsc2::ScheduleNode *producerNode,
              const dsc2::ScheduleNode *node, const SenComponents unit,
              const std::set<int> &srcIndex, const std::string &dataConnect,
              auto &self) -> bool {
        if (node == producerNode) {
          // Skip checking the original producer node.
          return true;
        }

        if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
          const dsc2::ComputeNode *compNode =
              static_cast<const dsc2::ComputeNode *>(node);
          if (!is_any_of(const_cast<dsc2::ScheduleNode *>(node), consumers)) {
            if (transformationReportLevel_ > 1) {
              std::cerr << "\n  [Use-latch]: Can not perform transformation as "
                           "an unrelated computeNode "
                        << getNodeDescription(compNode)
                        << " exists in the scope of the producer "
                        << getNodeDescription(transferNode)
                        << " and the consumers.";
            }
            return false;
          }
          // Future work: When we allow non-consumer computeNodes to appear in
          // the scope between the producer and consumer, we need to check if
          // a non-consumer's sources are consistent with the intended usage of
          // the latched data.
        } else if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
          if (!is_any_of(const_cast<dsc2::ScheduleNode *>(node), consumers)) {
            // Check that the transfer does not latch any result.
            for (auto dstIt :
                 static_cast<const dsc2::TransferNode *>(node)->dstVias_) {
              if (dstIt.loc_.storage_ == SenComponents::LATCH) {
                // Conservatively assume that the latch destination results in a
                // conflict.
                if (transformationReportLevel_ > 1) {
                  std::cerr
                      << "\n  [Use-latch]: Can not perform transformation as "
                         "a transferNode "
                      << getNodeDescription(node)
                      << " latches results in the scope of the producer "
                      << getNodeDescription(transferNode)
                      << " and the consumers.";
                }
                return false;
              }
            }
          }
        } else if (auto currBlock =
                       dynamic_cast<const dsc2::BlockNode *>(node)) {
          for (auto childNode : currBlock->getNextView(SenComponents::ALL)) {
            if (!self(producerNode, childNode, unit, srcIndex, dataConnect,
                      self)) {
              return false;
            }
          }
        }
        return true;
      };
      return analyzeLatchUsageImpl(producerNode, node, unit, srcIndex,
                                   dataConnect, analyzeLatchUsageImpl);
    };

    const dsc2::BlockNode *currParentBlock =
        static_cast<const dsc2::BlockNode *>(pathToTransfer.back());
    const dsc2::ScheduleNode *startChildNode =
        pathToTransfer.at(pathToTransfer.size() - 2);
    const dsc2::ScheduleNode *endChildNode =
        pathToConsumer.at(pathToConsumer.size() - 2);

    bool checkFifoUsage = false;
    for (auto currChild : currParentBlock->getNextView(SenComponents::ALL)) {
      if (currChild == startChildNode) {
        checkFifoUsage = true;
      } else if (!checkFifoUsage) {
        continue;
      }
      if (!analyzeLatchUsage(transferNode, currChild, destUnit, srcIndex,
                             ResultDataConnect)) {
        return false;
      }

      if (currChild == endChildNode) {
        break;
      }
    }
  }

  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 244/382   level 1   scc 241   8 body lines
// unit: e244_cloneForOffsetAdjustment
// authority: ddc/ddc_transformation.cpp:1386
// original: bool Ddc::cloneForOffsetAdjustment()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e244_cloneForOffsetAdjustment()
{
  for (auto &child : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::COMPUTE}, ALL, -1, -1)) {
    dsc2::ComputeNode *computeNode = static_cast<dsc2::ComputeNode *>(child);
    cloneComputeForOffsetAdjustment(computeNode);
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 245/382   level 1   scc 256   34 body lines
// unit: e245_setSizeForFixedSizeTransfers
// authority: ddc/ddc_transformation.cpp:1731
// original: void Ddc::setSizeForFixedSizeTransfers()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
void e245_setSizeForFixedSizeTransfers()
{
  // For each transfer node, check whether any dst dataConnect feeds an LXLU
  // compute node.  Use the pre-built metadata.dataConnects_ map for O(1)
  // consumer lookup instead of re-traversing the tree.
  for (auto *node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::TRANSFER})) {
    auto *transferNode = static_cast<dsc2::TransferNode *>(node);
    for (const auto &dstDataInfo : transferNode->dstLdsAndLoopOffsets_) {
      if (dstDataInfo.dataConnect_.empty()) continue;
      auto dcIt = metadata.dataConnects_.find(dstDataInfo.dataConnect_);
      if (dcIt == metadata.dataConnects_.end()) continue;
      bool hasLxluConsumer = false;
      for (const auto *consumer : dcIt->second.consumers_) {
        if (consumer->nodeType_ == dsc2::ScheduleNode::COMPUTE &&
            static_cast<const dsc2::ComputeNode *>(consumer)->exUnit_ ==
                SenComponents::LXLU) {
          hasLxluConsumer = true;
          break;
        }
      }
      if (!hasLxluConsumer) continue;
      int ldsIdx = dstDataInfo.myLdsIdx_;
      DT_CHECK(ldsIdx >= 0 &&
               ldsIdx < static_cast<int>(currDsc->labeledDs_.size()));
      const auto &labeledDs = currDsc->labeledDs_.at(ldsIdx);
      auto stickSizes = currDsc->getStickSizes(labeledDs.dsType_);
      transferNode->transferSize_.clear();
      for (const auto &[dim, size] : stickSizes) {
        transferNode->transferSize_[dim] = size;
      }
      break;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 246/382   level 1   scc 294   28 body lines
// unit: e246_transformForInterSliceRestickify
// authority: ddc/ddc_transformation.cpp:2398
// original: bool Ddc::transformForInterSliceRestickify()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e246_transformForInterSliceRestickify()
{
  bool didTransformation = false;
  // logic to identify if the transformation is triggered
  for (auto &node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::COMPUTE})) {
    auto *compNode = static_cast<dsc2::ComputeNode *>(node);
    if (EnumsConversion::senCompToGenericComp.at(compNode->exUnit_) !=
        SenComponents::PT)
      continue;
    if (compNode->inputsLdsAndLoopOffsets_.size() != 3) continue;
    if (compNode->inputs_.at(0) == SenComponents::ZERO &&
        compNode->inputs_.at(1) == SenComponents::ONE &&
        compNode->inputsLdsAndLoopOffsets_.at(2).myLdsIdx_ != -1) {
      int varInput_ldsidx = compNode->inputsLdsAndLoopOffsets_.at(2).myLdsIdx_;
      int none_trivial_input_idx = 2;
      int output_ldsidx = compNode->outputsLdsAndLoopOffsets_.at(0).myLdsIdx_;
      auto inputDsType = currDsc->labeledDs_.at(varInput_ldsidx).dsType_;
      auto outputDsType = currDsc->labeledDs_.at(output_ldsidx).dsType_;
      auto inputStickDimOrder = currDsc->getStickSizes(inputDsType);
      auto outputStickDimOrder = currDsc->getStickSizes(outputDsType);
      if (inputStickDimOrder != outputStickDimOrder) {
        didTransformation |= transformAComputeNodeForInterSliceRestickify(
            compNode, none_trivial_input_idx, outputStickDimOrder.at(0).first);
      }
    }
  }
  return didTransformation;
}

// ------------------------------------------------------------------------------------------------
// entry 247/382   level 1   scc 249   123 body lines
// unit: e247_splitLoopBandOnDim
// authority: ddc/ddc_transformation_util.cpp:159
// original: dsc2::LoopNode *Ddc::splitLoopBandOnDim( dsc2::LoopNode *baseLoop, const std::vector<std::vector<PrimaryDimAndKind>> &inputSplitDimSetsOuterToInner, bool unspecifiedDimsInnermost)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
dsc2::LoopNode *e247_splitLoopBandOnDim(
    dsc2::LoopNode *baseLoop,
    const std::vector<std::vector<PrimaryDimAndKind>>
        &inputSplitDimSetsOuterToInner,
    bool unspecifiedDimsInnermost)
{
  if (isExternalNode(baseLoop)) {
    DT_ERROR("[splitLoopBandOnDim] Can not split external loop " +
             baseLoop->name_);
  }
  std::vector<std::vector<PrimaryDimAndKind>> splitDimSets =
      inputSplitDimSetsOuterToInner;
  std::unordered_map<PrimaryDimAndKind, int> baseDims;
  for (auto baseDim : baseLoop->dims_) {
    baseDims[baseDim] = 0;
  }

  // Check: The sets do not have pairwise overlapping
  for (auto dimSet : splitDimSets) {
    for (auto dim : dimSet) {
      if (!baseDims.count(dim)) {
        DT_ERROR("Base loop is not associated with (" +
                 EnumsConversion::primaryDimToString.at(dim.dim_) + ", " +
                 EnumsConversion::metaDimKindToString.at(dim.kind_) + ")");
      }
      if (++baseDims[dim] > 1) {
        std::cerr << "\nDimension sets for loop-splitting:";
        for (auto dimSet : splitDimSets) {
          std::cerr << "\n{";
          for (auto dim : dimSet) {
            std::cerr << " ("
                      << EnumsConversion::primaryDimToString.at(dim.dim_)
                      << ", "
                      << EnumsConversion::metaDimKindToString.at(dim.kind_)
                      << ")";
          }
          std::cerr << "}";
        }
        DT_ERROR("Overlapping dimension sets specified for loop-splitting.");
      }
    }
  }

  // Check if any dimension of the base loop is not specified in the dimension
  // sets. These remaining dimensions result in a separate loop.
  std::vector<PrimaryDimAndKind> remainingDims;
  for (auto dimInfo : baseDims) {
    if (dimInfo.second == 0) {
      remainingDims.push_back(dimInfo.first);
    }
  }

  if (remainingDims.empty()) {
    if (splitDimSets.size() == 1) {
      // Only one dimension set is specified and the set matches the original
      // set. No transformation is needed.
      return baseLoop;
    }
  } else {
    // The loop with the remaining dimensions becomes innermost or outermost
    // depending on caller's decision.
    if (unspecifiedDimsInnermost) {
      splitDimSets.push_back(remainingDims);
    } else {
      splitDimSets.insert(splitDimSets.begin(), remainingDims);
    }
  }

  // Construct new loops starting with the second dimension set. The first
  // dimension set is for the (to-be-modified) base loop.
  // The first entry is the outermost loop and the last entry is the
  // innermost loop.
  std::vector<dsc2::LoopNode *> newLoops(splitDimSets.size());
  newLoops[0] = baseLoop;
  for (int i = 1, e = splitDimSets.size(); i < e; ++i) {
    newLoops[i] = constructLoopNode(baseLoop->numId_, baseLoop->denId_,
                                    splitDimSets[i], baseLoop);
    if (i > 1) {
      newLoops[i - 1]->addChildNode(newLoops[i]);
    }
  }

  // At this point, all new loops, except for the first and last loop, have
  // their parent/child associations setup.

  dsc2::LoopNode *innermostNewLoop = newLoops.back();
  // Move all children of the baseLoop to the newly constructed innermost loop
  baseLoop->moveChildren(innermostNewLoop);
  // Make the first new loop the only child of the base loop.
  baseLoop->addChildNode(newLoops[1]);
  // Replace dimension list of the base loop with the first entry of the
  // dimension set.
  baseLoop->dims_ = splitDimSets[0];
  baseLoop->name_ = "loop_ds" + std::to_string(baseLoop->numId_) + "_ds" +
                    std::to_string(baseLoop->denId_);
  for (const auto &entry : baseLoop->dims_) {
    baseLoop->name_ += "_" + EnumsConversion::primaryDimToString.at(entry.dim_);
  }
  baseLoop->name_ += "__split";

  // Update loop references in ConditionNodes inside the base loop.
  // Note: The original children of the base loop have been moved inside the
  // innermost new loop.
  for (auto *node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           innermostNewLoop, {dsc2::ScheduleNode::CONDITION})) {
    dsc2::ConditionNode *condNode = static_cast<dsc2::ConditionNode *>(node);
    for (auto &loopCondVec : condNode->loopCond_.twoLevelOrOfAnds_) {
      for (auto &loopCond : loopCondVec) {
        if (loopCond.loopComp_ != baseLoop) {
          continue;
        }
        for (const auto *loopNode : newLoops) {
          for (const auto &loopDim : loopNode->dims_) {
            if (loopDim.dim_ == loopCond.dim_) {
              // loopCond refers to a dimension that has been moved to the child
              // loop. Change the loop reference in loopCond to the new child
              // loop.
              loopCond.loopComp_ = loopNode;
              break;
            }
          }
        }
      }
    }
  }

  return innermostNewLoop;
}

// ------------------------------------------------------------------------------------------------
// entry 248/382   level 1   scc 237   30 body lines
// unit: e248_splitLoopBandOnDatastage
// authority: ddc/ddc_transformation_util.cpp:287
// original: dsc2::LoopNode *Ddc::splitLoopBandOnDatastage(dsc2::LoopNode *baseLoop)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
dsc2::LoopNode *e248_splitLoopBandOnDatastage(dsc2::LoopNode *baseLoop)
{
  if (isExternalNode(baseLoop)) {
    DT_ERROR("[splitLoopBandOnDatastage] Can not split external loop " +
             baseLoop->name_);
  }
  int refDataStageId =
      constructDatastage(currDsc->dataStageParam_.at(baseLoop->numId_));
  if (dataStageExplorationDone_) {
    DT_ERROR("Loop splitting after datastage exploration is not supported.");
  }

  // Construct the new loop as a prefectly-nested child of the base loop.
  dsc2::LoopNode *newLoopNode = constructLoopNode(
      refDataStageId, baseLoop->denId_, baseLoop->dims_, baseLoop);
  baseLoop->insertPerfectlyNestedBlockNode(newLoopNode);
  // Update the denominator DS of the original loop
  baseLoop->denId_ = refDataStageId;

  // Update loop references in ConditionNodes inside the base loop.
  // Note: The original children of the base loop have been moved inside the
  //       new loop.
  for (auto *node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           newLoopNode, {dsc2::ScheduleNode::CONDITION})) {
    dsc2::ConditionNode *condNode = static_cast<dsc2::ConditionNode *>(node);
    condNode->loopCond_.adjustConditionForSplitLoop(baseLoop,
                                                    {baseLoop, newLoopNode});
  }

  return newLoopNode;
}

// ------------------------------------------------------------------------------------------------
// entry 249/382   level 1   scc 250   372 body lines
// unit: e249_moveTransferNode
// authority: ddc/ddc_transformation_util.cpp:335
// original: void Ddc::moveTransferNode(dsc2::TransferNode *transferNode, dsc2::LoopNode *newParentLoop)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
void e249_moveTransferNode(dsc2::TransferNode *transferNode,
                           dsc2::LoopNode *newParentLoop)
{
  if (isExternalNode(transferNode)) {
    DT_ERROR("[moveTransferNode] Can not move external transferNode " +
             transferNode->name_);
  }
  // Cases where we may need sinking transfer node inside loops:
  //  Transfer:
  //            - padded to lower
  //            - block transfer requires transfers in smaller chunks
  //            - noncontiguous transfer: (e.g. transfer crosses stick boundary)

  DT_CHECK(newParentLoop);

  // Construct the loop chain from current owner loop to the new parent loop.
  std::vector<dsc2::LoopNode *> loopChain;
  auto *transferOwnerLoop = transferNode->getMutableOwnerLoop();

  if (transferOwnerLoop == newParentLoop) {
    return;
  }

  auto *loop = transferOwnerLoop;
  while (loop != nullptr) {
    loopChain.push_back(loop);
    if (loop == newParentLoop) {
      break;
    }
    loop = loop->getMutableOwnerLoop();
  }

  // Check if hoisting up or sinking down the transfer node along the loop
  // chain.
  bool hoistingUp = true;

  if (loopChain.back() != newParentLoop) {
    // New parent loop is not in the list of ancestors. Assume that we are
    // moving the transfer node down the loop chain.
    hoistingUp = false;
    loopChain.clear();
    loop = newParentLoop;
    while (loop != transferOwnerLoop && loop != nullptr) {
      loopChain.push_back(loop);
      if (loop == transferOwnerLoop) {
        break;
      }
      loop = loop->getMutableOwnerLoop();
    }
    if (loopChain.back() != transferOwnerLoop) {
      DT_ERROR("Can not move transfer node " + transferNode->name_ +
               ". The requested destination loopNode " + newParentLoop->name_ +
               " and the transfer node are not in the same loop nest.");
    }
  }

  // Layout of the loopChain

  // hoistingUp   : transferOwnerLoop, ... , newParentNode
  // hoistingDown : newParentNode,     ... , transferOwnerLoop

  // Verify if the requested transfer is semantically valid in terms of
  // data_connect.
  if (hoistingUp && transferNode->hasNonMemoryResult()) {
    DT_ERROR("Can not move transfer node " + transferNode->name_ +
             " with a FIFO destination.");
  }

  if (hoistingUp && metadata.dataConnects_.count(
                        transferNode->srcLdsAndLoopOffsets_.dataConnect_)) {
    // Check that the transfer is not hoisted up above a loop that contains the
    // data_connect of the source of the transfer.
    auto producerLoops =
        metadata.dataConnects_
            .at(transferNode->srcLdsAndLoopOffsets_.dataConnect_)
            .getProducerLoops();
    for (size_t i = 0, e = loopChain.size(); i < e - 1; ++i) {
      if (producerLoops.count(loopChain[i])) {
        DT_ERROR("Can not move transfer node " + transferNode->name_ +
                 " above the producer of the transfer-source (loop: " +
                 loopChain[i]->name_ + "), due to data_connect=" +
                 transferNode->srcLdsAndLoopOffsets_.dataConnect_ + ".");
      } else if (loopChain[i]->isParametricLoop()) {
        DT_ERROR("Can not move transfer node " + transferNode->name_ +
                 " above the parametric loop " + loopChain[i]->name_ + ".");
      }
    }
  }
  if (!hoistingUp) {
    // Sinking the transfer node.
    // check that the transfer node is not moved below a loop that contains a
    // data_connect of one of the destinations of the transfer.
    for (auto dst : transferNode->dstLdsAndLoopOffsets_) {
      auto consumerLoops =
          metadata.dataConnects_.at(dst.dataConnect_).getConsumerLoops();
      for (size_t i = 1, e = loopChain.size(); i < e; ++i) {
        // TO DO: The current check is more restrictive than necessary.
        // A given loop may not directly contain a usage of a data_connect.
        // However, the loop may still be marked as a consumer because one of
        // the nested loops directly contains a usage of the data_connect.

        // TO DO:
        //  - Disallow (down)movement across conditionals.
        //  - Find the lowest common ancestor `LCAncs` of the direct consumer
        //    and the newParentLoop. Determine the relative position of the
        //    children of `LCAncs` that contain the consumer and the
        //    newParentLoop. Since, transfer can not move across conditionals,
        //    the child containing newParentLoop must be before the child
        //    containing the consumer.
        //  - In case, newParentLoop itself is the lowest common ancestor,
        //    placing the transferNode at the beginning of newParentLoop will
        //    preserve the data-dependency.
        if (consumerLoops.count(loopChain[i])) {
          DT_ERROR("Can not move transfer node " + transferNode->name_ +
                   " below the consumer of the transfer-destination (loop: " +
                   loopChain[i]->name_ +
                   "), due to data_connect=" + dst.dataConnect_ + ".");
        }
      }
    }
  }

  // Verify validity of the transformation with respect to relevant condition
  // nodes.
  //   core-corelet condition:
  //     Construct a copy of the condition and bring the new condition node
  //     with the transfer node at the new location.
  //
  //   loop condition:
  //     Relevant only if the dimensions on the condition relates to the
  //     tensors in the transfer.
  //
  const dsc2::ScheduleNode *start, *end;
  if (hoistingUp) {
    start = static_cast<dsc2::ScheduleNode *>(transferNode);
    end = static_cast<dsc2::ScheduleNode *>(newParentLoop);
  } else {
    start = static_cast<dsc2::ScheduleNode *>(newParentLoop);
    end = static_cast<dsc2::ScheduleNode *>(transferNode);
  }

  struct ConditionInfo {
    const dsc2::ConditionNode *condition;
    bool useThenBranchForCondition;
  };
  std::vector<ConditionInfo> loopConditions;
  const dsc2::ScheduleNode *lastSeenNode = start;

  dsc2::ConditionNode *coreConditionToMove = nullptr;

  for (const dsc2::ScheduleNode *curr = start; curr != end;
       curr = curr->getPrev()) {
    if (curr->nodeType_ == dsc2::ScheduleNode::CONDITION) {
      const dsc2::ConditionNode *condNode =
          static_cast<const dsc2::ConditionNode *>(curr);
      if (condNode->hasCoreClCond()) {
        if (condNode->getThenBranchNode() == lastSeenNode) {
          // THEN branch of IF node. Condition expression can be combined
          // directly using intersection.
          if (!coreConditionToMove) {
            coreConditionToMove = new dsc2::ConditionNode();
            coreConditionToMove->name_ =
                "core_corelet_reuse_" + transferNode->name_;
            coreConditionToMove->coreClCond_ = condNode->coreClCond_;
          } else {
            for (auto &corePair : coreConditionToMove->coreClCond_) {
              auto clsIt = condNode->coreClCond_.find(corePair.first);
              if (clsIt != condNode->coreClCond_.end()) {
                corePair.second = set_intersect(corePair.second, clsIt->second);
                if (corePair.second.empty()) {
                  coreConditionToMove->coreClCond_.erase(corePair.first);
                }
              } else {
                coreConditionToMove->coreClCond_.erase(corePair.first);
              }
            }
          }
        } else {
          // ELSE branch of IF node. Condition expression has to be negated.
          if (!coreConditionToMove) {
            coreConditionToMove = new dsc2::ConditionNode();
            coreConditionToMove->name_ =
                "core_corelet_reuse_" + transferNode->name_;
            // Compute Universe minus condNode->coreClCond_
            std::set<int> allCorelets;
            for (int cl = 0; cl < currDsc->numCoreletsUsed_DSC2_; cl++) {
              allCorelets.insert(cl);
            }

            for (const auto &coreId : currDsc->coreIdsUsed_) {
              auto clsIt = condNode->coreClCond_.find(coreId);
              if (clsIt != condNode->coreClCond_.end()) {
                if (clsIt->second != allCorelets) {
                  coreConditionToMove->coreClCond_[coreId] =
                      set_diff(allCorelets, clsIt->second);
                }
              } else {
                coreConditionToMove->coreClCond_[coreId] = allCorelets;
              }
            }
          } else {
            for (auto &corePair : condNode->coreClCond_) {
              auto clsIt =
                  coreConditionToMove->coreClCond_.find(corePair.first);
              if (clsIt != coreConditionToMove->coreClCond_.end()) {
                clsIt->second = set_diff(clsIt->second, corePair.second);
                if (clsIt->second.empty()) {
                  coreConditionToMove->coreClCond_.erase(clsIt);
                }
              }
            }
          }
        }
      } else {
        // Loop condition
        for (auto &orIt : condNode->loopCond_.twoLevelOrOfAnds_) {
          for (auto &andIt : orIt) {
            // Consider all conditionNodes, even if they are not related to the
            // transferNode. Check whether the condition refers to a loop in the
            // loop-chain.
            for (auto loopIt : loopChain) {
              if (loopIt != newParentLoop && andIt.loopComp_ == loopIt) {
                DT_ERROR("Can not move transfer node " + transferNode->name_ +
                         " beyond loop conditional node " + condNode->name_ +
                         " associated with " + andIt.loopComp_->name_);
              }
            }
          }
        }

        // The condition node is not associated with any loop in the loop chain.
        // Record the condition. A later step will construct a copy of the
        // condition and place the new transfer node enclosed inside the
        // condition copy (possibly nested in other similar loop-conditions).
        loopConditions.push_back(
            {condNode, condNode->getThenBranchNode() == lastSeenNode});
      }
    }
    lastSeenNode = curr;
  }

  dsc2::ConditionNode *firstLoopCond = nullptr, *lastLoopCond = nullptr;
  dsc2::BlockNode *lastLoopCondThen = nullptr;

  for (const auto &cond : reverse(loopConditions)) {
    dsc2::ConditionNode *newCond =
        static_cast<dsc2::ConditionNode *>(cond.condition->clone());
    newCond->name_ = cond.condition->name_ + "_reuse_" + transferNode->name_;
    // Subsequent conditions would become descendants of THEN block.
    dsc2::BlockNode *thenBranch = new dsc2::BlockNode();
    thenBranch->name_ = newCond->name_ + "_then_region";
    dsc2::BlockNode *elseBranch = new dsc2::BlockNode();
    elseBranch->name_ = newCond->name_ + "_else_region";
    newCond->addThenRegion(thenBranch);
    newCond->addElseRegion(elseBranch);

    if (!cond.useThenBranchForCondition) {
      // Need to flip the negative status of the condition.
      newCond->loopCond_.negated_ ^= true;
    }

    if (!firstLoopCond) {
      firstLoopCond = newCond;
    }

    if (lastLoopCondThen) {
      // Add the next condition in the Then branch.
      lastLoopCondThen->addChildNode(newCond);
    }
    lastLoopCond = newCond;
    lastLoopCondThen = thenBranch;
  }

  // Detect allocate nodes of the transfer results.
  std::vector<dsc2::AllocateNode *> allocateNodesToMove;
  if (hoistingUp) {
    for (size_t i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e;
         ++i) {
      dsc2::AllocateNode *allocateNode = currDsc->getMutableAllocation(
          transferNode->dstLdsAndLoopOffsets_.at(i),
          transferNode->dstVias_.at(i).loc_.storage_);
      if (!allocateNode) {
        continue;
      }
      const dsc2::LoopNode *allocateOwnerLoop = allocateNode->getOwnerLoop();
      if (allocateOwnerLoop != newParentLoop) {
        for (auto loopIt : loopChain) {
          if (allocateOwnerLoop == loopIt) {
            if (isExternalNode(allocateNode)) {
              DT_ERROR(
                  "[moveTransferNode] Can not move transferNode " +
                  transferNode->name_ +
                  " as the transfer is associated with external allocateNode " +
                  allocateNode->name_);
            }
            allocateNodesToMove.push_back(allocateNode);
            break;
          }
        }
      }
    }
  }

  if (!allocateNodesToMove.empty() && dataStageExplorationDone_) {
    DT_ERROR(
        "MoveTransferNode: Can not move associated allocations "
        "after datastage exploration is complete.");
  }

  dsc2::BlockNode *effectiveNewParent = newParentLoop;
  bool insertBefore = true;
  // TO DO: compute insertBefore based on the original position of the Tranfer
  // node.

  dsc2::ScheduleNode *refNodeForInsertion = nullptr;
  if (hoistingUp) {
    // hoistingUp   : transferOwnerLoop, ... , newParentNode
    refNodeForInsertion = loopChain[loopChain.size() - 2];
    effectiveNewParent = refNodeForInsertion->getMutableParent();
  } else {
    // hoistingDown : newParentNode,     ... , transferOwnerLoop
    // TO DO: refNodeForInsertion = first or last child of newParentNode
    // TO DO: Set insertBefore accordingly.
  }

  if (!allocateNodesToMove.empty()) {
    for (auto *allocateNode : allocateNodesToMove) {
      allocateNode->moveNode(currDsc, effectiveNewParent, insertBefore,
                             refNodeForInsertion);
    }
    if (!insertBefore) {
      // Need to place the transfer node and the conditionals after the last
      // allocateNode.
      refNodeForInsertion = allocateNodesToMove.back();
    }
  }

  if (coreConditionToMove) {
    // Add coreConditionToMove as a child of newParentLoop.
    effectiveNewParent->addChildNode(coreConditionToMove, insertBefore,
                                     refNodeForInsertion);
    // The transferNode becomes
    //   - a direct child of coreConditionNode in case no additional
    //     loopConditions are needed, or
    //   - a descendent of coreConditionNode, nested in additional
    //     loopConditions that will be added later.

    // To materialize the above, update the effective new parent node of the
    // transferNode.
    dsc2::BlockNode *thenBranch = new dsc2::BlockNode();
    thenBranch->name_ = coreConditionToMove->name_ + "_then_region";
    dsc2::BlockNode *elseBranch = new dsc2::BlockNode();
    elseBranch->name_ = coreConditionToMove->name_ + "_else_region";
    coreConditionToMove->addThenRegion(thenBranch);
    coreConditionToMove->addElseRegion(elseBranch);
    effectiveNewParent = thenBranch;
    insertBefore = true;
    refNodeForInsertion = nullptr;
  }

  if (firstLoopCond) {
    // Add the loopCondition chain to the effective parent.
    effectiveNewParent->addChildNode(firstLoopCond, insertBefore,
                                     refNodeForInsertion);
    // Make the innermost new loopCondition the effective new parent.
    effectiveNewParent = lastLoopCondThen;
    insertBefore = true;
    refNodeForInsertion = nullptr;
  }

  // Move the node to the new effective parent loop.
  transferNode->moveNode(currDsc, effectiveNewParent, insertBefore,
                         refNodeForInsertion);
}

// ------------------------------------------------------------------------------------------------
// entry 250/382   level 1   scc 248   148 body lines
// unit: e250_convertResultFromFIFOtoReg
// authority: ddc/ddc_transformation_util.cpp:760
// original: bool Ddc::convertResultFromFIFOtoReg(dsc2::TransferNode *transferNode)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e250_convertResultFromFIFOtoReg(dsc2::TransferNode *transferNode)
{
  if (dataStageExplorationDone_) {
    DT_ERROR(
        "Conversion of FIFO result to register after datastage exploration is "
        "not supported.");
  }

  if (isExternalNode(transferNode)) {
    DT_ERROR(
        "[convertResultFromFIFOtoReg] Can not convert result of external "
        "transferNode " +
        transferNode->name_);
  }

  int fifoDestInd = -1;
  for (size_t i = 0, e = transferNode->dstVias_.size(); i < e; ++i) {
    if (dsc2::memories.count(transferNode->dstVias_[i].loc_.storage_) == 0) {
      // Check that the transferNode does not have more than one FIFO result.
      if (fifoDestInd != -1) {
        if (transformationReportLevel_ > 0) {
          std::cerr << "Convert FIFO to register: Transformation was not "
                       "applied as the transfer node "
                    << transferNode->name_ << " has more than one FIFO result.";
        }
        return false;
      }
      fifoDestInd = i;
    }
  }
  if (fifoDestInd == -1) {
    if (transformationReportLevel_ > 1) {
      std::cerr
          << "Convert FIFO to register: no FIFO result found for transfer "
          << transferNode->name_;
    }
    return true;
  }

  for (auto &fifoUser :
       metadata
           .dataConnects_[transferNode->dstLdsAndLoopOffsets_[fifoDestInd]
                              .dataConnect_]
           .consumers_) {
    if (fifoUser->nodeType_ == dsc2::ScheduleNode::COMPUTE &&
        static_cast<dsc2::ComputeNode *>(fifoUser)->isOpaqueOp_) {
      if (transformationReportLevel_ > 0) {
        std::cerr << "Convert FIFO to register: Transformation was not applied "
                     "as the FIFO result is used in opaque operation "
                  << fifoUser->name_;
      }
      return false;
    } else if (isExternalNode(fifoUser)) {
      if (transformationReportLevel_ > 0) {
        std::cerr << "Convert FIFO to register: Transformation was not applied "
                     "as the FIFO result is used in external operation "
                  << fifoUser->name_;
      }
      return false;
    }
  }

  if (transformationReportLevel_ > 1) {
    std::cerr << "\nConverting FIFO to register for transfer: "
              << transferNode->name_;
    // transferNode->print(std::cerr);
    std::cerr << "\n  Destination at index " << fifoDestInd << "\n";
  }

  auto &fifoDstVia = transferNode->dstVias_[fifoDestInd];
  const auto &fifoDstDataInfo =
      transferNode->dstLdsAndLoopOffsets_[fifoDestInd];
  dsc2::TransferNode::DstVia transformedDstVia = fifoDstVia;
  switch (transformedDstVia.loc_.unit_) {
    case SenComponents::SFP:
      transformedDstVia.loc_.storage_ = SenComponents::SFPLRF;
      break;
    case SenComponents::PE:
      transformedDstVia.loc_.storage_ = SenComponents::PELRF;
      break;
    default:
      if (transformationReportLevel_ > 0) {
        std::cerr << "Convert FIFO to register: Unsupported component.";
      }
      return false;
  }

  // Verify if this transformation is semantically correct.

  // Construct AllocateNode.
  dsc2::AllocateNode *allocNode = nullptr;

  if (!currDsc->allocationExists(fifoDstDataInfo,
                                 transformedDstVia.loc_.storage_)) {
    auto paddingPerdim = getPaddingPerDim(transferNode, /*forSrc*/ false);
    allocNode =
        constructAllocation(fifoDstDataInfo, transformedDstVia.loc_.storage_,
                            paddingPerdim, transferNode);
    allocNode->name_ =
        allocNode->name_ + "_" + fifoDstDataInfo.dataConnect_ + "_fifo_to_reg";

    // Add AllocateNode immediately before the transfer.
    transferNode->prev_->addChildNode(allocNode, true, transferNode);
  } else {
    // Analyze the usage of the existing allocation.
    // Check if the allocation is used in dummy computation to write to memory.
    // Otherwise (non-trivial usage), abandon the transformation.

    if (transformationReportLevel_ > 0) {
      std::cerr << "Convert FIFO to register: Usage of the same allocation in "
                   "multiple producers is not supported.";
    }
    return false;
  }

  // Modify destination of transfer from FIFO to register (allocation from
  // previous step).
  fifoDstVia.loc_.storage_ = allocNode->component_;
  // Replace all usage of the FIFO result with the new Register result.
  for (auto &fifoUser :
       metadata.dataConnects_[fifoDstDataInfo.dataConnect_].consumers_) {
    if (fifoUser->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      // Modify source
      dsc2::TransferNode *transferConsumer =
          static_cast<dsc2::TransferNode *>(fifoUser);
      transferConsumer->src_.storage_ = allocNode->component_;
      allocNode->addAllocUser(fifoUser);
    } else if (fifoUser->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      dsc2::ComputeNode *computeConsumer =
          static_cast<dsc2::ComputeNode *>(fifoUser);
      for (size_t i = 0, e = computeConsumer->inputsLdsAndLoopOffsets_.size();
           i < e; ++i) {
        if (computeConsumer->inputsLdsAndLoopOffsets_[i].dataConnect_ ==
            fifoDstDataInfo.dataConnect_) {
          computeConsumer->inputs_[i] = allocNode->component_;
          allocNode->addAllocUser(fifoUser);
        }
      }
    } else {
      DT_ERROR("Convert FIFO to register: Unsupported consumer type " +
               dsc2::ScheduleNode::nodeTypeToString.at(fifoUser->nodeType_) +
               ".");
    }
  }

  // No update is needed for metadata on data_connect as the pointers to the
  // producers and consumers have not changed.
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 251/382   level 1   scc 235   72 body lines
// unit: e251_unrollTransfer
// authority: ddc/ddc_transformation_util.cpp:1120
// original: bool Ddc::unrollTransfer(dsc2::TransferNode *transferNode)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e251_unrollTransfer(dsc2::TransferNode *transferNode)
{
  if (isExternalNode(transferNode)) {
    DT_ERROR("[unrollTransfer] Can not unroll external transfer " +
             transferNode->name_);
  }

  const dsc2::LoopNode *ownerLoop = transferNode->getMutableOwnerLoop();
  int ldsIdx = -1;
  if (transferNode->srcLdsAndLoopOffsets_.myLdsIdx_ >= 0) {
    ldsIdx = transferNode->srcLdsAndLoopOffsets_.myLdsIdx_;
  } else {
    for (const auto &dst : transferNode->dstLdsAndLoopOffsets_) {
      if (dst.myLdsIdx_ >= 0) {
        ldsIdx = dst.myLdsIdx_;
        break;
      }
    }
  }
  if (ldsIdx < 0) {
    // Neither source nor destination has ldsIdx set. The operation is likely
    // a transfer of a single constant element. No unrolling is needed in this
    // case.
    return true;
  }
  // TO DO: Do we need to include padding information (access-pattern style of
  // the transfer)?
  std::vector<PrimaryDimTypes> transferDims = currDsc->getLayoutDims(ldsIdx);
  std::unordered_set<PrimaryDimAndKind> remainingLoopDims(transferDims.begin(),
                                                          transferDims.end());

  // Determine numerator ids to be used for the dimensions in the new loops.
  std::map<PrimaryDimTypes, int> loopNumPerDim;
  const dsc2::LoopNode *currLoop = transferNode->getOwnerLoop();

  while (currLoop) {
    if (currLoop->isParametricLoop()) {
      if (transformationReportLevel_ > 0) {
        std::cerr << "\nCould not unroll transferNode " << transferNode->name_
                  << " as the transfer is enclosed in the parametric loop "
                  << currLoop->name_ << ".";
      }
      return false;
    }

    // Determine the numerator datastage for the new loop from the innermost
    // enclosing loops that is associated with currDim.
    int numId = currLoop->denId_;
    for (auto currDim : currLoop->dims_) {
      if (is_any_of(currDim, remainingLoopDims)) {
        loopNumPerDim[currDim.dim_] = numId;
        remainingLoopDims.erase(currDim);
      }
    }
    currLoop = currLoop->getOwnerLoop();
  }

  // Construct denominator datastage.
  int denId = constructDatastage();
  metadata.datastages_[denId].strategyMinimize_ = true;

  // Construct the new loops
  std::vector<PrimaryDimAndKind> newLoopDims;
  for (auto currDim : reverse(transferDims)) {
    newLoopDims.clear();
    newLoopDims.push_back(currDim);
    dsc2::LoopNode *newLoop =
        constructLoopNode(loopNumPerDim.at(currDim), denId, newLoopDims);
    transferNode->getMutableParent()->addChildNode(newLoop, true, transferNode);
    transferNode->moveNode(currDsc, newLoop);
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 252/382   level 1   scc 253   110 body lines
// unit: e252_unrollTransferForSymbolicDims
// authority: ddc/ddc_transformation_util.cpp:1193
// original: bool Ddc::unrollTransferForSymbolicDims( dsc2::TransferNode *transferNode, const std::map<PrimaryDimTypes, SymbolicDimInfo> &symbolicDims)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e252_unrollTransferForSymbolicDims(
    dsc2::TransferNode *transferNode,
    const std::map<PrimaryDimTypes, SymbolicDimInfo> &symbolicDims)
{
  if (isExternalNode(transferNode)) {
    DT_ERROR(
        "[unrollTransferForSymbolicDims] Can not unroll external transfer " +
        transferNode->name_);
  }
  if (symbolicDims.empty()) {
    return true;
  }

  const dsc2::LoopNode *ownerLoop = transferNode->getMutableOwnerLoop();
  int ldsIdx = -1;
  if (transferNode->srcLdsAndLoopOffsets_.myLdsIdx_ >= 0) {
    ldsIdx = transferNode->srcLdsAndLoopOffsets_.myLdsIdx_;
  } else {
    for (const auto &dst : transferNode->dstLdsAndLoopOffsets_) {
      if (dst.myLdsIdx_ >= 0) {
        ldsIdx = dst.myLdsIdx_;
        break;
      }
    }
  }
  if (ldsIdx < 0) {
    // Neither source nor destination has ldsIdx set. The operation is likely
    // a transfer of a single constant element. No unrolling is needed in this
    // case.
    return true;
  }
  // TO DO: Do we need to include padding information (access-pattern style of
  // the transfer)?
  std::vector<PrimaryDimTypes> transferDims;
  // Only keep symbolic dimesnsions.
  for (auto dim : currDsc->getNonBroadcastLdsDims(ldsIdx)) {
    if (symbolicDims.count(dim)) {
      transferDims.push_back(dim);
    }
  }

  std::unordered_set<PrimaryDimTypes> remainingLoopDims(transferDims.begin(),
                                                        transferDims.end());

  // Determine numerator ids to be used for the dimensions in the new loops.
  std::map<PrimaryDimTypes, int> loopNumPerDim;
  const dsc2::LoopNode *currLoop = transferNode->getOwnerLoop();

  while (currLoop && !remainingLoopDims.empty()) {
    if (currLoop->isParametricLoop()) {
      if (transformationReportLevel_ > 0) {
        std::cerr << "\n[unrollTransferForSymbolicDims] Could not unroll "
                     "transferNode "
                  << transferNode->name_
                  << " as the transfer is enclosed in the parametric loop "
                  << currLoop->name_ << ".";
      }
      return false;
    }

    // Determine the numerator datastage for the new loop from the innermost
    // enclosing loops that is associated with currDim.
    int numId = currLoop->denId_;
    for (const auto &[currDim, currKind] : currLoop->dims_) {
      if (remainingLoopDims.count(currDim)) {
        loopNumPerDim[currDim] = numId;
        remainingLoopDims.erase(currDim);
      }
    }
    currLoop = currLoop->getOwnerLoop();
  }

  // Construct denominator datastage.
  int denId = constructDatastage();
  auto &denDs = currDsc->dataStageParam_.at(denId);

  for (const auto &[dim, dsIdx] : loopNumPerDim) {
    const auto &refDs = currDsc->dataStageParam_.at(dsIdx);
    denDs.ss_.primaryDimToValHandler_st(dim) =
        refDs.ss_.primaryDimToVal_st(dim);
    denDs.el_.primaryDimToValHandler_st(dim) =
        refDs.el_.primaryDimToVal_st(dim);
    if (refDs.ss_.coreletSplit_.count(dim)) {
      denDs.ss_.coreletSplit_[dim] = refDs.ss_.coreletSplit_.at(dim);
      denDs.el_.coreletSplit_[dim] = refDs.el_.coreletSplit_.at(dim);
    }
    if (refDs.ss_.rowSplit_.count(dim)) {
      denDs.ss_.rowSplit_[dim] = refDs.ss_.rowSplit_.at(dim);
      denDs.el_.rowSplit_[dim] = refDs.el_.rowSplit_.at(dim);
    }
    if (refDs.ss_.peSfpSplit_.count(dim)) {
      denDs.ss_.peSfpSplit_[dim] = refDs.ss_.peSfpSplit_.at(dim);
      denDs.el_.peSfpSplit_[dim] = refDs.el_.peSfpSplit_.at(dim);
    }
    denDs.ss_.makeDimNotSymbolic(dim);
    denDs.el_.makeDimNotSymbolic(dim);

    if (refDs.ss_.paddingSizes_.count(dim)) {
      denDs.ss_.paddingSizes_[dim] = refDs.ss_.paddingSizes_.at(dim);
      denDs.el_.paddingSizes_[dim] = refDs.el_.paddingSizes_.at(dim);
    }
  }

  // Construct the new loops in layout order. Loops are added for the symbolic
  // dimensions only.
  for (auto currDim : reverse(transferDims)) {
    dsc2::LoopNode *newLoop =
        constructLoopNode(loopNumPerDim.at(currDim), denId, {currDim});
    transferNode->getMutableParent()->addChildNode(newLoop, true, transferNode);
    transferNode->moveNode(currDsc, newLoop);
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 253/382   level 1   scc 295   4 body lines
// unit: e253_srcRelatedToExternalNodes
// authority: ddc/ddc_transformation_util.cpp:1687
// original: bool Ddc::srcRelatedToExternalNodes( const dsc2::TransferNode *transferNode) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e253_srcRelatedToExternalNodes(
    const dsc2::TransferNode *transferNode) const
{
  return storageOrDatastreamIsExternal(transferNode->srcLdsAndLoopOffsets_,
                                       transferNode->src_.storage_, true);
}

// ------------------------------------------------------------------------------------------------
// entry 254/382   level 1   scc 224   5 body lines
// unit: e254_destRelatedToExternalNodes
// authority: ddc/ddc_transformation_util.cpp:1693
// original: bool Ddc::destRelatedToExternalNodes(const dsc2::TransferNode *transferNode, int dstIndex) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e254_destRelatedToExternalNodes(const dsc2::TransferNode *transferNode,
                                     int dstIndex) const
{
  return storageOrDatastreamIsExternal(
      transferNode->dstLdsAndLoopOffsets_.at(dstIndex),
      transferNode->dstVias_.at(dstIndex).loc_.storage_, false);
}

// ------------------------------------------------------------------------------------------------
// entry 255/382   level 1   scc 297   11 body lines
// unit: e255_inputRelatedToExternalNodes
// authority: ddc/ddc_transformation_util.cpp:1717
// original: bool Ddc::inputRelatedToExternalNodes( const dsc2::ComputeNode *computeNode) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e255_inputRelatedToExternalNodes(
    const dsc2::ComputeNode *computeNode) const
{
  for (size_t i = 0, e = computeNode->inputsLdsAndLoopOffsets_.size(); i < e;
       ++i) {
    if (storageOrDatastreamIsExternal(
            computeNode->inputsLdsAndLoopOffsets_.at(i),
            computeNode->inputs_.at(i), true)) {
      return true;
    }
  }
  return false;
}

// ------------------------------------------------------------------------------------------------
// entry 256/382   level 1   scc 298   11 body lines
// unit: e256_outputRelatedToExternalNodes
// authority: ddc/ddc_transformation_util.cpp:1730
// original: bool Ddc::outputRelatedToExternalNodes( const dsc2::ComputeNode *computeNode) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e256_outputRelatedToExternalNodes(
    const dsc2::ComputeNode *computeNode) const
{
  for (size_t i = 0, e = computeNode->outputsLdsAndLoopOffsets_.size(); i < e;
       ++i) {
    if (storageOrDatastreamIsExternal(
            computeNode->outputsLdsAndLoopOffsets_.at(i),
            computeNode->outputs_.at(i), false)) {
      return true;
    }
  }
  return false;
}

// ------------------------------------------------------------------------------------------------
// entry 257/382   level 1   scc 236   107 body lines
// unit: e257_addNewLds
// authority: ddc/ddc_transformation_util.cpp:1811
// original: int Ddc::addNewLds(LabeledDsInfo *refLds)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
int e257_addNewLds(LabeledDsInfo *refLds)
{
  std::vector<LabeledDsInfo *> oldLdsPtrs;
  for (auto &lds : currDsc->labeledDs_) {
    oldLdsPtrs.push_back(&lds);
  }
  LabeledDsInfo *lastLds =
      &currDsc->labeledDs_.at(currDsc->labeledDs_.size() - 1);
  LabeledDsInfo newLds = *refLds;
  int oldLastLdsIdx = lastLds->ldsIdx_;
  newLds.ldsIdx_ = currDsc->labeledDs_.size() - 1;
  // update idx of last lds
  currDsc->labeledDs_.back().ldsIdx_++;
  newLds.referenceLdsIdx_ = refLds->ldsIdx_;
  // insert just before the last lds
  auto *newLdsPtr = &*currDsc->labeledDs_.insert(currDsc->labeledDs_.end() - 1,
                                                 std::move(newLds));
  // fix pointers around the dsc
  std::unordered_map<LabeledDsInfo *, LabeledDsInfo *> oldNewLdsPtrs;
  for (int i = 0; i < oldLdsPtrs.size() - 1; i++) {
    oldNewLdsPtrs[oldLdsPtrs.at(i)] = &currDsc->labeledDs_.at(i);
  }
  oldNewLdsPtrs[oldLdsPtrs.back()] = &currDsc->labeledDs_.back();
  for (auto &lds : currDsc->labeledDs_) {
    for (auto &dt : lds.dataTransfers_) {
      dt.myDsInfo = oldNewLdsPtrs.at(dt.myDsInfo);
    }
  }
  for (auto &co : currDsc->computeOp_) {
    for (auto &lds : co.inputLabeledDs) {
      lds = oldNewLdsPtrs.at(lds);
    }
    for (auto &lds : co.interimLabeledDs) {
      lds = oldNewLdsPtrs.at(lds);
    }
    for (auto &lds : co.outputLabeledDs) {
      lds = oldNewLdsPtrs.at(lds);
    }
  }
  int newLastLdsIdx =
      currDsc->labeledDs_.at(currDsc->labeledDs_.size() - 1).ldsIdx_;
  auto &memorg = currDsc->labeledDs_.at(newLastLdsIdx).memOrg_;
  for (auto &compMemorg : memorg) {
    compMemorg.second.allocateNode_->ldsIdx_ = newLastLdsIdx;
  }
  currDsc->labeledDs_.at(newLds.ldsIdx_).memOrg_.clear();

  // Use newLds for all childs of start node
  for (auto &child : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr,
           {dsc2::ScheduleNode::ALLOCATE, dsc2::ScheduleNode::TRANSFER,
            dsc2::ScheduleNode::COMPUTE, dsc2::ScheduleNode::LOOP},
           ALL, -1, -1)) {
    if (child->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      auto cn = static_cast<dsc2::ComputeNode *>(child);
      if (cn->isOpaqueOp_) {
        auto &ldsIdx = metadata.opaqueOps_.at(cn).ldsIdx_;
        if (ldsIdx == oldLastLdsIdx) ldsIdx = newLastLdsIdx;
      }
      for (auto &in : cn->inputsLdsAndLoopOffsets_) {
        if (in.myLdsIdx_ == oldLastLdsIdx) in.myLdsIdx_ = newLastLdsIdx;
      }
      for (auto &out : cn->outputsLdsAndLoopOffsets_) {
        if (out.myLdsIdx_ == oldLastLdsIdx) out.myLdsIdx_ = newLastLdsIdx;
      }
    } else if (child->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      auto tn = static_cast<dsc2::TransferNode *>(child);
      if (tn->srcLdsAndLoopOffsets_.myLdsIdx_ == oldLastLdsIdx)
        tn->srcLdsAndLoopOffsets_.myLdsIdx_ = newLastLdsIdx;
      for (auto &out : tn->dstLdsAndLoopOffsets_) {
        if (out.myLdsIdx_ == oldLastLdsIdx) out.myLdsIdx_ = newLastLdsIdx;
      }
    } else if (child->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
      auto an = static_cast<dsc2::AllocateNode *>(child);
      if (an->ldsIdx_ == oldLastLdsIdx) an->ldsIdx_ = newLastLdsIdx;
    } else if (child->nodeType_ == dsc2::ScheduleNode::LOOP) {
      auto an = static_cast<dsc2::LoopNode *>(child);
      if (an->isParametricLoop()) {
        if (an->parametricLdsIdx_ == oldLastLdsIdx)
          an->parametricLdsIdx_ = newLastLdsIdx;
      }
    } else {
      DT_ERROR("Invalid block node.");
    }
  }
  for (auto &compAlloc : metadata.newAllocations_) {
    auto &ldsToAlloc = compAlloc.second.ldsIdxAndAllocNode;
    if (ldsToAlloc.count(oldLastLdsIdx)) {
      ldsToAlloc[newLastLdsIdx] = ldsToAlloc.at(oldLastLdsIdx);
      ldsToAlloc.erase(oldLastLdsIdx);
    }
    for (auto &compToAlloc : compAlloc.second.compAndAllocNode) {
      if (compToAlloc.second->ldsIdx_ == oldLastLdsIdx)
        compToAlloc.second->ldsIdx_ = newLastLdsIdx;
    }
  }

  // update DDL interface:
  if (metadata.ldsIdxAfterDdc.count(oldLastLdsIdx)) {
    metadata.ldsIdxAfterDdc.at(oldLastLdsIdx) = newLastLdsIdx;
  } else {
    for (auto &ldsIdxPair : metadata.ldsIdxAfterDdc) {
      if (ldsIdxPair.second == oldLastLdsIdx) ldsIdxPair.second = newLastLdsIdx;
    }
  }

  return newLds.ldsIdx_;
}

// ------------------------------------------------------------------------------------------------
// entry 258/382   level 1   scc 304   306 body lines
// unit: e258_allocAllMem
// authority: ddc/ddcv1.cpp:132
// original: bool Ddc::allocAllMem(bool commitIfValid)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// NOTE: SUPERSEDES HAND-TRANSCRIBED GUESSWORK: crates/compiler/deeptools/src/reginit.rs (1,569
//       lines, on the integ branch) hand-computes placement from ddc/ddcv1.cpp:132-360 -- THIS
//       function. Port what the authority does, not what reginit.rs guessed; the two will be
//       reconciled when this campaign and integ meet, and the authority wins.
// ------------------------------------------------------------------------------------------------
bool e258_allocAllMem(bool commitIfValid)
{
  std::vector<FailedAlloc> failedAllocs;
  std::map<dsc2::AllocateNode*, std::map<int, std::map<int, int64_t>>>
      startAddressCoreCorelet_;
  std::map<dsc2::AllocateNode*, std::map<int, std::map<int, int64_t>>>
      bufferOffsetCoreCorelet_;
  std::map<dsc2::AllocateNode*, std::pair<bool, bool>> copyToCoreCl;
  std::map<DsTrackInMem*, std::vector<DsTrackInMem::DsMemInfo>> trackerBackups;

  // store size of the allocate nodes in a list
  std::map<dsc2::ScheduleNode*, int64_t> allocate_size;
  for (auto& node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::ALLOCATE})) {
    auto* allocateNode = static_cast<dsc2::AllocateNode*>(node);
    if (allocateNode->ldsIdx_ == -1) {
      allocate_size[node] = -1;
    } else {
      allocate_size[node] =
          currDsc->getBufferCapacityForNode(allocateNode, allocateNode->ldsIdx_,
                                            allocateNode->component_, -1, -1);
    }
  }
  // set the first entry to be the largest allocate in the list
  // if we just need to allocate memory to the largest allocate
  for (auto& shadow_allocs : metadata.shadowAllocations_) {
    int max_size = 0;
    int switch_index = 0;
    for (int i = 0; i < shadow_allocs.size(); i++) {
      if (allocate_size.at(shadow_allocs.at(i)) > max_size) {
        max_size = allocate_size.at(shadow_allocs.at(i));
        switch_index = i;
      }
    }
    // switch the first entry with switch_index
    dsc2::AllocateNode* tmp = shadow_allocs.at(0);
    shadow_allocs.at(0) = shadow_allocs.at(switch_index);
    shadow_allocs.at(switch_index) = tmp;
  }
  /*
  std::cout << "------ rearranged: ------\n";
  for (auto& shadow_allocs : metadata.shadowAllocations_) {
    std::cout << "entry:\n";
    for (auto alloc : shadow_allocs) {
      std::cout << "  " << alloc << " -> " << alloc->name_ << "\n";
    }
  }
  std::cout << "end\n";
  for (auto [node, size] : allocate_size) {
    std::cout << node << " : " << size << " -> " << node->name_ << "\n";
  }
  */
  auto tryAlloc = [&]() {
    for (auto& [comp, allocMetadata] : metadata.newAllocations_) {
      DT_CHECK_MSG(
          trueLXTracker_ || comp != SenComponents::LX,
          "DDC is currently being passed ephemeral mem trackers, it can't use "
          "those for LX allocations");
      std::vector<int> cores;
      bool copyCore = false;
      if (comp == SenComponents::LX || comp == SenComponents::L0 ||
          comp == SenComponents::L0_SCALE) {
        cores = currDsc->coreIdsUsed_;
      } else {
        // for non-lx, we will use first core as proxy..
        cores.push_back(currDsc->coreIdsUsed_.front());
        copyCore = true;
      }
      // for corelets, rows use 0 as proxy
      bool copyCorelet =
          (dscGlobal.sysDef.coreArch > IsaCoreGen::RCUDD1A_ISA &&
           (comp == SenComponents::L0 || comp == SenComponents::L0_SCALE))
              ? false
              : true;
      std::vector<int> corelets(1, 0);
      if (!copyCorelet) {
        for (int i = 1; i < currDsc->numCoreletsUsed_DSC2_; i++) {
          corelets.push_back(i);
        }
      }
      std::vector<int> rows(1, 0);
      for (const auto& core : cores) {
        for (const auto& corelet : corelets) {
          for (const auto& row : rows) {
            auto myTracker = memTrackers->getTracker(comp, core, corelet, row);
            auto [it, didInsert] = trackerBackups.try_emplace(myTracker);
            if (didInsert) {
              it->second = myTracker->backupEps(exphase);
            }
            std::vector<std::pair<dsc2::AllocateNode*, int64_t>>
                nodeAndSize;  // in bytes..
            // currDsc->labeledDs_.at(acand.first).dsName_
            for (auto& [ldsIdx, allocNode] : allocMetadata.ldsIdxAndAllocNode) {
              int numBuffers = allocNode->numBuffers_;
              if (numBuffers == -1)
                numBuffers = 2;  // reserve at least 2 buffers
              // skip shadow allocates
              bool skip_allocate_mem = true;
              for (auto& shadowAllocs : metadata.shadowAllocations_) {
                if (shadowAllocs.size() == 0) continue;
                auto idx = std::find(shadowAllocs.begin(), shadowAllocs.end(),
                                     allocNode) -
                           shadowAllocs.begin();
                // only the allocate memory to the first node which is the
                // biggest in the size
                if (idx == 0) {
                  skip_allocate_mem = false;
                  break;
                }
              }
              if (!skip_allocate_mem) {
                nodeAndSize.emplace_back(
                    allocNode,
                    numBuffers * currDsc->getBufferCapacityForNode(
                                     allocNode, ldsIdx, allocNode->component_,
                                     corelet, row));
              }
            }
            for (auto& acand : allocMetadata.consIdAndAllocNode) {
              nodeAndSize.emplace_back(acand.second,
                                       dscGlobal.sysDef.bytesPerStick);
            }
            for (auto& acand : allocMetadata.compAndAllocNode) {
              DT_CHECK(acand.second->numBuffers_ == 1);
              const auto& opMetadata = metadata.opaqueOps_.at(acand.first);
              auto size = currDsc->getBufferCapacityForNode(
                  acand.second, acand.second->ldsIdx_, acand.second->component_,
                  corelet, row);
              auto unroll = size / dscGlobal.sysDef.bytesPerStick;
              // fail if more than the opaque op can handle or if not a power of
              // 2
              if (unroll > opMetadata.max_unroll_ || unroll == 0 ||
                  (unroll & (unroll - 1)))
                return false;
              nodeAndSize.emplace_back(
                  acand.second,
                  (opMetadata.internalRegs_.size() +
                   (unroll - 1) * opMetadata.internalRegsWithUnroll_) *
                      dscGlobal.sysDef.bytesPerStick);
            }
            // std::cout << "Comp: "
            //           << EnumsConversion::senComponentsToString.at(kv.first)
            //           << std::endl;
            // for (auto& kv1 : nodeAndSize) {
            //   std::cout << getLdsOrConstNameOfAllocNode(kv1.first) << " "
            //             << kv1.second << std::endl;
            // }
            std::vector<int> seps(1, exphase);
            for (auto& kv : nodeAndSize) {
              myTracker->removeDs(getLdsOrConstNameOfAllocNode(kv.first), seps);
            }
            // pre-fill PTARF tracker to avoid alloc in unaccessible registers
            if (dscGlobal.sysDef.coreArch <= MPW4_ISA && comp == PTARF &&
                is_any_of(currDsc->labeledDs_.at(0).dataFormat_,
                          DataFormats::SENINT8, DataFormats::SENINT4)) {
              // in dd1, PT has 12 registers, but only 6 are accessible in
              // int8/4 and only R4/R5 can be used for accumulation
              myTracker->addDsAtStartAddr(
                  "LRF-not-ARF", 4 * dscGlobal.sysDef.bytesPerStick, seps, 0);
              myTracker->addDsAtStartAddr(
                  "Reg-not-available-int", 6 * dscGlobal.sysDef.bytesPerStick,
                  seps, 6 * dscGlobal.sysDef.bytesPerStick);
            }
            if (dscGlobal.sysDef.coreArch > IsaCoreGen::RCUDD1A_ISA &&
                (comp == SenComponents::L0 ||
                 comp == SenComponents::L0_SCALE)) {
              // for sentient1.5, starting address of input tensor in l0
              // depends on coreletID and l0TetheredMode
              if (!currDsc->l0TetheredMode_) {
                // tethered=1, buffers shared between left&right corelets
                // tethered=0, L0 is half-half split between left&right
                // corelets
                auto corecoord =
                    dscGlobal.sysDef.coreIdToTetheredCoreCoord(core);
                int halfSize = myTracker->memCapacity / 2;
                if (corecoord.subcoreId == 0) {  // block top part
                  myTracker->addDsAtStartAddr("Reg-not-available-to-subcore0",
                                              halfSize, seps, halfSize);
                } else {  // block bottom part
                  myTracker->addDsAtStartAddr("Reg-not-available-to-subcore1",
                                              halfSize, seps, 0);
                }
              }
            }
            bool allDsFit = true;
            for (auto& kv : nodeAndSize) {
              auto mySize = kv.second;
              if (kv.first->numBuffers_ == -1) {
                // full capacity reserved for circular buffer..
                auto memCap =
                    (dscGlobal.sysDef.coreArch > IsaCoreGen::RCUDD1A_ISA &&
                     (comp == SenComponents::L0 ||
                      comp == SenComponents::L0_SCALE) &&
                     !currDsc->l0TetheredMode_)
                        ? myTracker->memCapacity / 2
                        : myTracker->memCapacity;
                mySize = std::max(mySize, memCap);
              }
              int64_t addr;
              if (kv.first->ldsIdx_ >= 0 &&
                  currDsc->labeledDs_.at(kv.first->ldsIdx_)
                          .scaledLdsCategory_ ==
                      LabeledDsInfo::ScaledLdsCategory::SCALE_TENSOR &&
                  kv.first->component_ == SenComponents::PTXRF) {
                addr = myTracker->checkAndAddDsAtAddr(
                    getLdsOrConstNameOfAllocNode(kv.first), mySize, seps,
                    dscGlobal.sysDef.xrfScaleStart *
                        dscGlobal.sysDef.bytesPerStick);
              } else {
                addr = myTracker->checkAndAddDs(
                    getLdsOrConstNameOfAllocNode(kv.first), mySize, seps);
              }
              DT_CHECK(addr != EXISTS);
              if (addr == DOESNT_FIT) {
                // std::cout << getLdsOrConstNameOfAllocNode(kv.first) << " "
                //           << mySize << std::endl;
                // myTracker->printExPhase(exphase);
                allDsFit = false;
                break;
              } else {
                if (commitIfValid) {
                  startAddressCoreCorelet_[kv.first][core][corelet] = addr;
                  int numBuffers = kv.first->numBuffers_;
                  if (numBuffers == -1) numBuffers = 2;
                  bufferOffsetCoreCorelet_[kv.first][core][corelet] =
                      kv.second / numBuffers;
                  copyToCoreCl[kv.first] =
                      std::make_pair(copyCore, copyCorelet);
                }
              }
            }
            if (!allDsFit) {
              FailedAlloc newFailedAlloc;
              newFailedAlloc.comp = comp;
              newFailedAlloc.core = core;
              newFailedAlloc.corelet = corelet;
              newFailedAlloc.row = row;
              failedAllocs.push_back(newFailedAlloc);
              return false;
            }
          }
        }
      }
    }
    return true;
  };
  bool success = tryAlloc();
  if (success && failedAllocs.size() == 0 && commitIfValid) {
    std::deque<const FoldDimProp*> sdscFoldProps{sdsc_->coreFoldProp_.get(),
                                                 sdsc_->coreletFoldProp_.get()};
    for (const auto& foldProp : sdsc_->sdscFoldProps_) {
      sdscFoldProps.push_back(foldProp.get());
    }
    std::deque<BaseFuncType> foldTypes(sdscFoldProps.size(),
                                       BaseFuncType::Constant);
    for (auto& [alloc, addrMap] : startAddressCoreCorelet_) {
      auto& addrFM = alloc->startAddressCoreCorelet_;
      DT_CHECK(addrFM.hasZeroFoldDim());
      // use const fold if address does not change across core and/or corelet
      const auto& [copyCore, copyCl] = copyToCoreCl.at(alloc);
      DT_CHECK(!copyCore || addrMap.size() == 1);
      foldTypes[0] = copyCore ? BaseFuncType::Constant : BaseFuncType::Map;
      DT_CHECK(!copyCl || addrMap.begin()->second.size() == 1);
      foldTypes[1] = copyCl ? BaseFuncType::Constant : BaseFuncType::Map;
      // build inner to outer
      addrFM.buildFoldSpace(sdscFoldProps, foldTypes);
      std::deque<int64_t> coord(sdscFoldProps.size(), 0);
      auto headCore = currDsc->coreIdsUsed_.front();
      for (auto& [core, clAddr] : addrMap) {
        for (auto& [cl, addr] : clAddr) {
          coord[0] = core;
          coord[1] = cl;
          addrFM.insertData(addr, coord);
        }
      }
    }
    for (auto& kv : bufferOffsetCoreCorelet_) {
      kv.first->bufferOffsetCoreCorelet_ = kv.second;
    }
    for (auto& [alloc, copyCoreCl] : copyToCoreCl) {
      if (copyCoreCl.second) {
        // copy to other corelets used..
        for (auto& corekv : alloc->bufferOffsetCoreCorelet_) {
          for (int cl = 1; cl < currDsc->numCoreletsUsed_; cl++) {
            corekv.second[cl] = corekv.second.at(0);
          }
        }
      }
      if (copyCoreCl.first) {
        // copy to other cores..
        const auto& headCore = currDsc->coreIdsUsed_.front();
        for (int cidx = 1; cidx < currDsc->coreIdsUsed_.size(); cidx++) {
          const auto& newCore = currDsc->coreIdsUsed_.at(cidx);
          if (alloc->numBuffers_ != 1) {
            alloc->bufferOffsetCoreCorelet_[newCore] =
                alloc->bufferOffsetCoreCorelet_.at(headCore);
          }
        }
      }
    }
  } else {
    for (const auto& [myTracker, backupInfo] : trackerBackups) {
      myTracker->restoreEps(exphase, backupInfo);
    }
  }
  return (success && failedAllocs.size() == 0);
}

// ------------------------------------------------------------------------------------------------
// entry 259/382   level 1   scc 312   123 body lines
// unit: e259_calculateClStartAddress
// authority: ddc/ddcv1.cpp:1895
// original: void Ddc::calculateClStartAddress(dsc2::AllocateNode* allocNode)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// NOTE: PLACES ADDRESSES. Part of the span reginit.rs hand-transcribes.
// ------------------------------------------------------------------------------------------------
void e259_calculateClStartAddress(dsc2::AllocateNode* allocNode)
{
  if (currDsc->numCoreletsUsed_DSC2_ < 2) return;
  DT_CHECK_MSG(allocNode, "Expect a valid allocate node.");
  DT_CHECK_MSG(allocNode->component_ == SenComponents::LX,
               "Expect LX in memOrg_.");

  const LabeledDsInfo& lds = currDsc->labeledDs_.at(allocNode->ldsIdx_);
  std::vector<PrimaryDimTypes> ldsCoreletSplitDim, ldsNonBroadcastDims;
  for (PrimaryDimTypes dim : allocNode->layoutDimOrder_) {
    if (lds.scale_.at(currDsc->getDimIndexInLayoutOrder(lds.dsType_, dim)) ==
        1) {
      ldsNonBroadcastDims.push_back(dim);
      if (is_any_of(dim, metadata.clSplitDims_)) {
        ldsCoreletSplitDim.push_back(dim);
      }
    }
  }
  DT_CHECK_MSG(ldsCoreletSplitDim.size() <= 1,
               "Support maximal one corelet split dimension for a tensor");

  // do not offset cl1 start addr if tensor is involved in cross-core reduction
  for (const auto& compOp : currDsc->computeOp_) {
    if (compOp.opFuncName == OpFuncs::GENERIC_PARTIAL_REDUCTION) {
      DT_CHECK(compOp.inputLabeledDs.size() == 1);
      if (compOp.inputLabeledDs.at(0) == &lds) ldsCoreletSplitDim.clear();
    }
  }

  const auto dsNode =
      currDsc->getSizeDataStageForNode(allocNode, allocNode).ss_;
  const auto stickSizePerDim = currDsc->getCumulativeStickSizes(lds.dsType_);
  const auto& dsChunk = currDsc->dataStageParam_.at(metadata.chunk_dstgid).ss_;

  int overallOffset = 0;
  for (int coreletId = 1; coreletId < currDsc->numCoreletsUsed_DSC2_;
       ++coreletId) {
    int coreletOffset = 0;
    if (!ldsCoreletSplitDim.empty()) {
      PrimaryDimTypes coreletSplitDim = ldsCoreletSplitDim.front();
      DT_CHECK_MSG(
          dsNode.coreletSplit_.count(coreletSplitDim),
          "Expect corelet split dimension in chunk data stage parameters.");
      // Compute corelet split offset by iterate from the innermost to outer
      // dimensions in layoutDimOrder_.
      coreletOffset = dscGlobal.sysDef.bytesPerStick;
      for (PrimaryDimTypes dim : ldsNonBroadcastDims) {
        double dimDensity = 1.0;
        if (lds.scaledLdsCategory_ ==
                LabeledDsInfo::ScaledLdsCategory::SCALE_TENSOR &&
            lds.mxInfo_.dim == dim) {
          dimDensity = 1.0 / lds.mxInfo_.blkSize;
        }
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
            DT_CHECK_MSG(dsChunk.paddingSizes_.count(dim),
                         "Expect padding sizes in chunk data stage params.");
            DT_CHECK_MSG(dim == PrimaryDimTypes::I,
                         "Support I dimension only.");
            DT_CHECK_MSG(dsChunk.paddingSizes_.at(dim).stride_ > 0,
                         "Expect a valid stride.");
            coreletOffset *= dsChunk.coreletSplit_.at(dim).at(coreletId - 1) *
                             dsChunk.paddingSizes_.at(dim).stride_ / stickSize;
          } else {
            // No padding on the corelet split dimension.
            coreletOffset *= dsChunk.primaryDimToVal_st(
                                 dim, allocNode->component_, -1, coreletId - 1,
                                 allocNode->padding_, dimDensity) /
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
          coreletOffset *= dsNode.primaryDimToVal_st(
                               dim, allocNode->component_, -1, coreletId - 1,
                               allocNode->padding_, dimDensity) /
                           stickSize;
        }
      }

      // Update offset
      overallOffset += coreletOffset;
    }
    DT_CHECK_MSG(overallOffset == 0 ||
                     allocNode->startAddressCoreCorelet_.getFuncType(1) ==
                         BaseFuncType::Map,
                 "AllocateNode start address cl fold must be map type if "
                 "corelets need to work on different data");
    for (auto& [coord, addr] :
         allocNode->startAddressCoreCorelet_.getDataAndFoldCoordinates(
             {{1, 0}})) {
      // Assign the values to startAddressCoreCorelet_ corelet1, because we only
      // have the values for corelet0 at this point.
      coord[1] = coreletId;
      allocNode->startAddressCoreCorelet_.insertData(addr + overallOffset,
                                                     coord);
    }
    if (allocNode->numBuffers_ != 1) {
      for (auto& [coreId, bufferOffset] : allocNode->bufferOffsetCoreCorelet_) {
        // Assign the values to startAddressCoreCorelet_ corelet1, because we
        // only have the values for corelet0 at this point.
        bufferOffset[coreletId] = bufferOffset.at(0);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 260/382   level 1   scc 316   844 body lines
// unit: e260_fillLoopOffsetsAndAddresses
// authority: ddc/ddcv1.cpp:2355
// original: void Ddc::fillLoopOffsetsAndAddresses( const bool allowUnpaddedIndexingAtPaddedNoZeroPad)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e260_fillLoopOffsetsAndAddresses(
    const bool allowUnpaddedIndexingAtPaddedNoZeroPad)
{
  auto scheduleTreeTraverse = currDsc->scheduleTree_.traverseTreeDFSMutable();

  auto fillDataInfo = [&](dsc2::DataInfo& di, DataLocation loc,
                          const dsc2::LoopNode* loopLocation,
                          dsc2::ScheduleNode* processorNode, bool isProducer,
                          const dsc2::CoordinateType<CoordinateBaseType>&
                              coordinates) {
    if (di.myLdsIdx_ < 0 && di.constantId_ < 0) return;
    if (dsc2::memories.count(loc.storage_) == 0) return;
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
    }

    const auto& allocation = di.myLdsIdx_ >= 0
                                 ? currDsc->labeledDs_.at(di.myLdsIdx_)
                                       .memOrg_.at(loc.storage_)
                                       .allocateNode_
                                 : currDsc->constantInfo_.at(di.constantId_)
                                       .allocations_.at(loc.storage_);
    di.startAddr_.clone(allocation->startAddressCoreCorelet_);
    auto genericUnit = EnumsConversion::senCompToGenericComp.at(loc.unit_);
    auto addrScale = dscGlobal.sysDef.addressGranularityScalePerUnit.at(
        {genericUnit, loc.storage_});
    if (genericUnit == L0LU) {
      addrScale *= dscGlobal.sysDef.numPTRows;
    }

    DT_CHECK(addrScale > 0);
    di.isStartAddrSymbolic_ = allocation->isStartAddrSymbolic_;
    if (addrScale != 1) {
      if (di.isStartAddrSymbolic_) {
        di.startAddr_.apply({}, [&](auto&& sym) {
          return sdsc_->symbolDefinitions_.addVar(
              VariableOperator::DIV, {{true, sym}, {false, addrScale}});
        });
      } else {  // non symbolic address
        di.startAddr_.apply({}, std::divides<int64_t>(), addrScale);
      }
    }

    // fill offsets
    if (di.myLdsIdx_ < 0) return;  // no offsets for constants
    DT_CHECK(di.loopEleOffsets_.empty());
    const auto& myLds = currDsc->labeledDs_.at(di.myLdsIdx_);
    auto loopPtr = loopLocation;
    const bool nonCoreletMemory =
        dsc2::nonCoreletMemories.count(allocation->component_);
    bool considerBothCorelets =
        nonCoreletMemory && !loopsBelowChunkBoundary.count(loopPtr);
    // const auto& dataConnectLoops =
    //     isProducer ? metadata.dataConnects_.at(di.dataConnect_).consumers_
    //                : metadata.dataConnects_.at(di.dataConnect_).producers_;
    auto* allocOwnerLoop = allocation->getOwnerLoop();
    while (loopPtr != allocOwnerLoop) {
      if (loopPtr->prev_ == nullptr) {  // this is the root node
        DT_ERROR("Cannot find allocation in parent loops: " + di.dataConnect_);
      }

      if (!considerBothCorelets && nonCoreletMemory &&
          !loopsBelowChunkBoundary.count(loopPtr)) {
        considerBothCorelets = true;
      }
      for (auto& [dim, kind] : loopPtr->dims_) {
        PadType allocPadding = allocation->padding_.getPadding(dim);
        bool relevant = is_any_of(dim, allocation->layoutDimOrder_);
        PrimaryDimTypes relatedPadDim = PrimaryDimTypesCount;
        if (!relevant && !loopPtr->isParametricLoop()) {
          for (const auto& [padDim, padInfo] :
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
              if (!datastageBasedElemOff) {
                di.loopEleOffsets_[clId][loopPtr][dim] =
                    loopDistributionParamInfo.at(processorNode)
                        .at(allocation)
                        .at(loopPtr)
                        .at(dim)
                        .temporalStridePostDistribution;
              } else {
                di.loopEleOffsets_[clId][loopPtr][dim] = 0;
              }

              if (verifyCoordinateBasedLoopElemOff || datastageBasedElemOff) {
                int loopEleOffs = -1;
                if (loopPtr->isParametricLoop()) {
                  loopEleOffs = loopPtr->parametricStride(currDsc);
                  if (datastageBasedElemOff) {
                    di.loopEleOffsets_[clId][loopPtr][dim] = loopEleOffs;
                  }
                  if (loopPtr->parametricIterCount(currDsc, clId, loc.unit_) >
                          1 &&
                      di.loopEleOffsets_[clId][loopPtr][dim] != loopEleOffs) {
                    std::cerr
                        << "\nMISMATCH: loopElemOffset[" << processorNode->name_
                        << "][alloc=" << allocation->name_ << "]["
                        << "corelet=" << clId
                        << "][loop(parametric)=" << loopPtr->name_ << "][dim="
                        << EnumsConversion::primaryDimToString.at(dim) << "]: {"
                        << "previous_method= " << loopEleOffs << ", "
                        << "coordinate_based= "
                        << di.loopEleOffsets_[clId][loopPtr][dim] << "}";
                    DT_ERROR(
                        "Coordinate-based loopElemOffset did not match the "
                        "result from current method.");
                  }
                  continue;
                }

                const auto& ds =
                    currDsc->dataStageParam_.at(loopPtr->denId_).ss_;
                loopEleOffs = ds.dataStageDimToVal_compView_st(
                    dim, loc.unit_, considerBothCorelets ? -1 : clId);
                const auto& numDs =
                    currDsc->dataStageParam_.at(loopPtr->numId_).ss_;
                int64_t iterCount =
                    numDs.dataStageDimToVal_compView_st(
                        dim, loc.unit_, considerBothCorelets ? -1 : clId) /
                    loopEleOffs;

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
                        DT_CHECK_MSG(ds.paddingSizes_.at(dim).windowDim_ !=
                                         PrimaryDimTypes::PrimaryDimTypesCount,
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
                        dim, loc.unit_, considerBothCorelets ? -1 : clId,
                        allocation->padding_);
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
                        dim, loc.unit_, considerBothCorelets ? -1 : clId,
                        allocation->padding_);
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
                  // No update is needed to loopEleOffs that was computed above.
                }
                if (datastageBasedElemOff) {
                  di.loopEleOffsets_[clId][loopPtr][dim] = loopEleOffs;
                }
                if (iterCount > 1 &&
                    di.loopEleOffsets_[clId][loopPtr][dim] != loopEleOffs) {
                  std::cerr
                      << "\nMISMATCH: loopElemOffset[" << processorNode->name_
                      << "][alloc=" << allocation->name_ << "]["
                      << "corelet=" << clId << "][loop=" << loopPtr->name_
                      << "][dim=" << EnumsConversion::primaryDimToString.at(dim)
                      << "]: {"
                      << "previous_method= " << loopEleOffs << ", "
                      << "coordinate_based= "
                      << di.loopEleOffsets_[clId][loopPtr][dim] << "}";
                  DT_ERROR(
                      "Coordinate-based loopElemOffset did not match the "
                      "result from current method.");
                }
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
      }
      for (auto& [dim, kind] : loopPtr->dims_) {
        if (is_any_of(dim, allocation->layoutDimOrder_)) {
          int dimIdx = currDsc->getDimIndexInLayoutOrder(myLds.dsType_, dim);
          int scale = dimIdx < 0 ? 1 : myLds.scale_.at(dimIdx);
          if (scale > 0) {
            PadType allocPadding = allocation->padding_.getPadding(dim);
            const auto& ds =
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
                  }
                }
              }
            }
          }
        }
      }
      loopPtr = loopPtr->getOwnerLoop();
    }

    if (!datastageBasedElemOff) {
      // use coordinates for constant offset
      bool anyConstOffset = !di.constEleOffsets_.empty();
      const auto& coreIdToWkSliceNode = coordinates.coreIdToWkSlice_.empty()
                                            ? sdsc_->coreIdToWkSlice_
                                            : coordinates.coreIdToWkSlice_;
      const auto& allocCoordinates =
          allocation->sliceViewCoordinates_.coordinates_.empty()
              ? allocation->allocateCoordinates_
              : allocation->sliceViewCoordinates_;
      const auto& coreIdToWkSliceAlloc =
          allocCoordinates.coreIdToWkSlice_.empty()
              ? sdsc_->coreIdToWkSlice_
              : allocCoordinates.coreIdToWkSlice_;
      const auto& relevantCoreCl = processorNode->getRelevantCoreCl();
      int rowId = 0;
      if (auto it = EnumsConversion::senCompToRowId.find(loc.unit_);
          it != EnumsConversion::senCompToRowId.end()) {
        rowId = std::max(0, it->second);
      }
      for (const auto& [dim, dimCoord] : coordinates.coordinates_) {
        if (!allocCoordinates.coordinates_.count(dim)) continue;
        const auto& allocDimCoord = allocCoordinates.coordinates_.at(dim);
        const bool useCoreIdNode = dimCoord.getFoldDimSize(static_cast<int>(
                                       dsc2::CoordinateFoldPosition::Core)) > 1;
        const bool useClIdNode =
            dimCoord.getFoldDimSize(
                static_cast<int>(dsc2::CoordinateFoldPosition::Corelet)) > 1;
        const int rowIdNode = dimCoord.getFoldDimSize(static_cast<int>(
                                  dsc2::CoordinateFoldPosition::RowSplit)) > 1
                                  ? rowId
                                  : 0;
        const bool useCoreIdAlloc =
            allocDimCoord.getFoldDimSize(
                static_cast<int>(dsc2::CoordinateFoldPosition::Core)) > 1;
        const bool useClIdAlloc =
            allocDimCoord.getFoldDimSize(
                static_cast<int>(dsc2::CoordinateFoldPosition::Corelet)) > 1;
        const int rowIdAlloc = allocDimCoord.getFoldDimSize(static_cast<int>(
                                   dsc2::CoordinateFoldPosition::RowSplit)) > 1
                                   ? rowId
                                   : 0;
        std::map<int64_t, int64_t> fixCoordAlloc = {
            {static_cast<int>(dsc2::CoordinateFoldPosition::RowSplit),
             rowIdAlloc}};
        // fix all temporal folds to 0, as we are only interested in the elem
        // arrangement folds
        for (int loopFoldStart = allocCoordinates.getNumOfSpatialFolds(dim),
                 i = loopFoldStart,
                 loopFoldEnd = loopFoldStart +
                               allocCoordinates.getNumOfTemporalFolds(dim);
             i < loopFoldEnd; i++) {
          fixCoordAlloc.emplace(i, 0);
        }
        for (auto coreId : currDsc->coreIdsUsed_) {
          for (int clId = 0; clId < currDsc->numCoreletsUsed_DSC2_; clId++) {
            // initializes to zero only if not present
            auto& constOffsetField = di.constEleOffsets_[coreId][clId][dim];
            if (!relevantCoreCl.count(coreId) ||
                !relevantCoreCl.at(coreId).count(clId) ||
                !coreIdToWkSliceNode.count(coreId) ||
                !coreIdToWkSliceAlloc.count(coreId)) {
              continue;
            }
            if (!useCoreIdNode && !useCoreIdAlloc &&
                coreId != currDsc->coreIdsUsed_[0]) {
              constOffsetField =
                  di.constEleOffsets_[currDsc->coreIdsUsed_[0]][clId][dim];
              continue;
            }
            if (!useClIdNode && !useClIdAlloc && clId != 0) {
              constOffsetField = di.constEleOffsets_[coreId][0][dim];
              continue;
            }
            const auto& sliceIdNode = coreIdToWkSliceNode.at(coreId).at(dim);
            const auto& nodeBeta = dimCoord.getSingleData(
                {{static_cast<int>(dsc2::CoordinateFoldPosition::Core),
                  useCoreIdNode ? sliceIdNode : 0},
                 {static_cast<int>(dsc2::CoordinateFoldPosition::Corelet),
                  useClIdNode ? clId : 0},
                 {static_cast<int>(dsc2::CoordinateFoldPosition::RowSplit),
                  rowIdNode}});
            const auto& sliceIdAlloc = coreIdToWkSliceAlloc.at(coreId).at(dim);
            fixCoordAlloc.insert_or_assign(
                static_cast<int>(dsc2::CoordinateFoldPosition::Core),
                useCoreIdAlloc ? sliceIdAlloc : 0);
            fixCoordAlloc.insert_or_assign(
                static_cast<int>(dsc2::CoordinateFoldPosition::Corelet),
                useClIdAlloc ? clId : 0);

            const auto myOffset =
                FoldInfraUtils::lexiAffineSolveDistanceInSteps(
                    allocDimCoord, nodeBeta, fixCoordAlloc);
            if (myOffset != 0) {
              anyConstOffset = true;
              DT_CHECK_MSG(
                  constOffsetField == 0 || constOffsetField == myOffset,
                  "Constant offset from coordinates in conflict with "
                  "existing constant offset");
              constOffsetField = myOffset;
            }
          }
        }
      }

      // remove constant offset if zero all the time
      if (!anyConstOffset) di.constEleOffsets_.clear();
    }

    if (allocation->numBuffers_ != 1) {
      DT_CHECK_MSG(loopPtr->getOwnerLoop() != nullptr,
                   "Do not expect the root node.");
      // fill buffer switching info
      di.bufferSwitchPosition_ = loopPtr;
      di.bufferAddrOffset_ = allocation->bufferOffsetCoreCorelet_;
      for (auto& corePair : di.bufferAddrOffset_) {
        for (auto& clPair : corePair.second) {
          clPair.second /= addrScale;
        }
      }
    }
  };

  for (auto* node : scheduleTreeTraverse) {
    if (metadata.externalNodes_.count(node)) continue;

    auto* ownerLoop = node->getOwnerLoop();
    if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      auto* transfer = static_cast<dsc2::TransferNode*>(node);
      auto findLastFusableLoop = [&](SenComponents unit) {
        const dsc2::LoopNode* lastFusableLoop = nullptr;
        if (transfer->isNodeRelevant(unit)) {
          auto* parent = transfer->getPrev();
          while (parent && parent->getPrev() &&
                 parent->getNextView(unit).size() == 1) {
            if (parent->nodeType_ == dsc2::ScheduleNode::CONDITION /*&&
                static_cast<const dsc2::ConditionNode*>(parent)
                        ->hasCoreClCond() == false*/)
              break;
            if (parent->nodeType_ == dsc2::ScheduleNode::LOOP) {
              auto tmpParentLoop = static_cast<const dsc2::LoopNode*>(parent);

              auto isSymbolicLoop = [&](const dsc2::LoopNode* tmpLoop) {
                if (tmpLoop->isParametricLoop())
                  return false;  // parametric loops cannot yet be symbolic
                // Couldn't use loopCountSymbolIds_ because it will be filled
                // later in the pipeline.
                const auto& numDs =
                    currDsc->dataStageParam_.at(tmpLoop->numId_).ss_;
                const auto& denDs =
                    currDsc->dataStageParam_.at(tmpLoop->denId_).ss_;
                for (const auto& [dim, kind] : tmpLoop->dims_) {
                  if (numDs.symbolicDimInfo_.count(dim) &&
                      !denDs.symbolicDimInfo_.count(dim)) {
                    // this means that loop involves a dimension that require
                    // correction. hence, cannot fuse the loops.
                    return true;
                  }
                }

                return false;
              };

              if (isSymbolicLoop(tmpParentLoop)) break;

              lastFusableLoop = tmpParentLoop;
            }
            parent = parent->getPrev();
          }
        }
        return lastFusableLoop;
      };
      if (transfer->src_.unit_ == NO_COMPONENT &&
          transfer->paddingInfo_.isEmpty())
        continue;

      fillDataInfo(transfer->srcLdsAndLoopOffsets_, transfer->src_, ownerLoop,
                   transfer, false, transfer->transferCoordinates_);
      transfer->lastFusableParentLoopSrc_ =
          findLastFusableLoop(transfer->src_.unit_);
      if (transfer->dstLdsAndLoopOffsets_.size() != transfer->dstVias_.size()) {
        DT_ERROR("Transfer node destinations missing information: " +
                 transfer->name_);
      }
      transfer->lastFusableParentLoopDst_.clear();
      for (int i = 0; i < transfer->dstVias_.size(); i++) {
        fillDataInfo(transfer->dstLdsAndLoopOffsets_[i],
                     transfer->dstVias_[i].loc_, ownerLoop, transfer, true,
                     transfer->transferCoordinates_);
        transfer->lastFusableParentLoopDst_.push_back(
            findLastFusableLoop(transfer->dstVias_[i].loc_.unit_));
      }
      if (auto tnMetaIt = metadata.datatransfers_.find(transfer);
          tnMetaIt != metadata.datatransfers_.end()) {
        if (tnMetaIt->second.apply_row_offset_src_ &&
            dsc2::memories.count(transfer->src_.storage_) &&
            transfer->srcLdsAndLoopOffsets_.constEleOffsets_.empty()) {
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
        } else if (tnMetaIt->second.apply_row_offset_dst_ &&
                   transfer->dstLdsAndLoopOffsets_.at(0)
                       .constEleOffsets_.empty()) {
          for (int i = 0; i < transfer->dstVias_.size(); i++) {
            if (!dsc2::memories.count(transfer->dstVias_.at(i).loc_.storage_))
              continue;
            for (auto coreId : currDsc->coreIdsUsed_) {
              for (int clId = 0; clId < currDsc->numCoreletsUsed_DSC2_;
                   clId++) {
                transfer->dstLdsAndLoopOffsets_.at(i)
                    .constEleOffsets_[coreId][clId][metadata.rowSplitDim] =
                    currDsc->getBlockTransferSizePerDim(
                        *transfer, transfer->dstVias_.at(i).loc_.unit_,
                        clId)[metadata.rowSplitDim] *
                    EnumsConversion::senCompToRowId.at(transfer->src_.unit_);
              }
            }
          }
        } else if (tnMetaIt->second.replicated_) {
          DT_CHECK_MSG(tnMetaIt->second.offset_src_ > 0 ||
                           tnMetaIt->second.offset_dest_.size() > 0,
                       "At least one of the src or dst should be replicated");

          dsc2::TransferNode* originalTransfer = nullptr;
          for (auto& [orig, clones] : metadata.nodeCloningMap_) {
            if (std::find(clones.begin(), clones.end(), transfer) !=
                clones.end()) {
              originalTransfer = static_cast<dsc2::TransferNode*>(orig);
            }
          }

          if (!originalTransfer) {
            throw std::runtime_error(
                "[fillLoopOffsetsAndAddresses] Could not find the original "
                "transferNode for cloned Transfer " +
                transfer->name_);
          }

          if (tnMetaIt->second.offset_src_ > 0) {
            for (auto coreId : currDsc->coreIdsUsed_) {
              for (int clId = 0; clId < currDsc->numCoreletsUsed_DSC2_;
                   clId++) {
                for (auto sizeAndIndex : transfer->unitTimeTransferChunkSize_) {
                  auto dim = sizeAndIndex.sizeDim_.dim_;
                  transfer->srcLdsAndLoopOffsets_
                      .constEleOffsets_[coreId][clId][dim] =
                      currDsc->getBlockTransferSizePerDim(
                          *originalTransfer, originalTransfer->src_.unit_,
                          clId)[dim];
                }
              }
            }
          }
          for (auto destIndRep : tnMetaIt->second.offset_dest_) {
            auto destInd = destIndRep.first;
            if (destIndRep.second > 0) {
              for (auto coreId : currDsc->coreIdsUsed_) {
                for (int clId = 0; clId < currDsc->numCoreletsUsed_DSC2_;
                     clId++) {
                  for (auto sizeAndIndex :
                       transfer->unitTimeTransferChunkSize_) {
                    auto dim = sizeAndIndex.sizeDim_.dim_;
                    transfer->dstLdsAndLoopOffsets_.at(destInd)
                        .constEleOffsets_[coreId][clId][dim] =
                        currDsc->getBlockTransferSizePerDim(
                            *originalTransfer, originalTransfer->src_.unit_,
                            clId)[dim];
                  }
                }
              }
            }
          }
        } else if (tnMetaIt->second.apply_pe_sfp_split_offset_src_ &&
                   transfer->srcLdsAndLoopOffsets_.constEleOffsets_.empty()) {
          // Use the original node for size calculation. The offset for the
          // cloned node is the transfer size of the original node.
          const dsc2::TransferNode* originalTransfer = nullptr;
          for (const auto& [orig, clones] : metadata.nodeCloningMap_) {
            DT_CHECK(clones.size() == 1);
            if (clones.at(0) == transfer) {
              originalTransfer = static_cast<dsc2::TransferNode*>(orig);
            }
          }

          if (!originalTransfer) {
            DT_ERROR(
                "[fillLoopOffsetsAndAddresses] Could not find the original "
                "transferNode for cloned Transfer " +
                transfer->name_);
          }

          const auto& peSfpSplitInfo =
              currDsc->dataStageParam_.at(metadata.core_dstgid).ss_.peSfpSplit_;
          for (auto coreId : currDsc->coreIdsUsed_) {
            for (int clId = 0; clId < currDsc->numCoreletsUsed_DSC2_; clId++) {
              for (const auto& [dim, splitInfo] : peSfpSplitInfo) {
                transfer->srcLdsAndLoopOffsets_
                    .constEleOffsets_[coreId][clId][dim] =
                    currDsc->getBlockTransferSizePerDim(
                        *originalTransfer, originalTransfer->src_.unit_,
                        clId)[dim];
              }
            }
          }
        } else if (!tnMetaIt->second.apply_pe_sfp_split_offset_dest_.empty() &&
                   transfer->dstLdsAndLoopOffsets_.at(0)
                       .constEleOffsets_.empty()) {
          // Use the original node for size calculation. The offset for the
          // cloned node is the transfer size of the original node.
          const dsc2::TransferNode* originalTransfer = nullptr;
          for (const auto& [orig, clones] : metadata.nodeCloningMap_) {
            DT_CHECK(clones.size() == 1);
            if (clones.at(0) == transfer) {
              originalTransfer = static_cast<dsc2::TransferNode*>(orig);
            }
          }

          if (!originalTransfer) {
            DT_ERROR(
                "[fillLoopOffsetsAndAddresses] Could not find the original "
                "transferNode for cloned Transfer " +
                transfer->name_);
          }
          for (auto destInd :
               tnMetaIt->second.apply_pe_sfp_split_offset_dest_) {
            const auto& peSfpSplitInfo =
                currDsc->dataStageParam_.at(metadata.core_dstgid)
                    .ss_.peSfpSplit_;
            for (auto coreId : currDsc->coreIdsUsed_) {
              for (int clId = 0; clId < currDsc->numCoreletsUsed_DSC2_;
                   clId++) {
                for (auto& [dim, splitInfo] : peSfpSplitInfo) {
                  transfer->dstLdsAndLoopOffsets_.at(destInd)
                      .constEleOffsets_[coreId][clId][dim] =
                      currDsc->getBlockTransferSizePerDim(
                          *originalTransfer, originalTransfer->src_.unit_,
                          clId)[dim];
                }
              }
            }
          }
        }
      }
    } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      auto* compute = static_cast<dsc2::ComputeNode*>(node);
      if (compute->inputsLdsAndLoopOffsets_.size() != compute->inputs_.size() ||
          compute->outputsLdsAndLoopOffsets_.size() !=
              compute->outputs_.size()) {
        DT_ERROR("Compute node input/output missing information");
      }
      for (int i = 0; i < compute->inputs_.size(); i++) {
        fillDataInfo(
            compute->inputsLdsAndLoopOffsets_[i],
            {compute->exUnit_, compute->inputs_[i]}, ownerLoop, compute, false,
            (datastageBasedElemOff ? compute->outputCoordinate_
                                   : compute->inputCoordinates_.at(i)));
      }
      for (int i = 0; i < compute->outputs_.size(); i++) {
        fillDataInfo(compute->outputsLdsAndLoopOffsets_[i],
                     {compute->exUnit_, compute->outputs_[i]}, ownerLoop,
                     compute, true, compute->outputCoordinate_);
      }
      if (metadata.nodeCloningMap_.count(node)) {
        for (int output_idx_ = 0;
             output_idx_ < compute->repetitionWithOffset_.forOutputs_.size();
             output_idx_++) {
          auto& myLds = currDsc->labeledDs_.at(
              compute->outputsLdsAndLoopOffsets_.at(output_idx_).myLdsIdx_);
          auto comp = compute->outputs_.at(output_idx_);
          auto alloc = myLds.memOrg_.at(comp == LXSU ? LX : comp).allocateNode_;
          auto dim = alloc->layoutDimOrder_.at(0);
          auto stickSizes = currDsc->getCumulativeStickSizes(myLds.dsType_);
          int size = 1;
          if (stickSizes.count(dim)) size = stickSizes.at(dim);
          int factor = metadata.nodeCloningMap_.at(node).size();
          for (auto& clonedNode : metadata.nodeCloningMap_.at(node)) {
            auto* clonedCompute = static_cast<dsc2::ComputeNode*>(clonedNode);
            for (auto coreId : currDsc->coreIdsUsed_) {
              for (int clId = 0; clId < currDsc->numCoreletsUsed_DSC2_;
                   clId++) {
                clonedCompute->outputsLdsAndLoopOffsets_.at(output_idx_)
                    .constEleOffsets_[coreId][clId][dim] = factor * size;
              }
            }
            factor--;
          }
        }
      }
    }
  }
  /*currDsc->exportJson("status_beforeloopOffset.json");
  if (metadata.TransferNodesInterSliceTranspose_.empty()==false){
    auto outputstickdimtrail = currDsc->getStickSizes(OUTPUT,false,true);
    auto dimForLoop = outputstickdimtrail.at(0).first;
    // auto inputstickdim = currDsc->primaryDsInfo_[INPUT].stickDimOrder_;
    // auto outputstickdim = currDsc->primaryDsInfo_[OUTPUT].stickDimOrder_;
    // std::vector<PrimaryDimTypes> diff;
    //
  std::set_difference(inputstickdim.begin(),inputstickdim.end(),outputstickdim.begin(),outputstickdim.end(),std::back_inserter(diff));
    //
  std::set_difference(outputstickdim.begin(),outputstickdim.end(),inputstickdim.begin(),inputstickdim.end(),std::back_inserter(diff));
    // auto dimForLoop = static_cast<PrimaryDimTypes>(diff.at(0)); //as we only
  one dimension changing wrt stick std::cout<<"dim for
  loop"<<dimForLoop<<std::endl;



    for (auto& node : currDsc->scheduleTree_.traverseTreeDFSMutable(
              nullptr, {dsc2::TransferNode::TRANSFER})) {
          auto* transferNode = static_cast<dsc2::TransferNode*>(node);


          if(metadata.TransferNodesInterSliceTranspose_.count(transferNode)){
              auto& constLoopOffset =
  transferNode->srcLdsAndLoopOffsets_.constEleOffsets_;
              std::cout<<transferNode->name_<<std::endl;
              //
  transferNode->srcLdsAndLoopOffsets_.constEleOffsets_[0][0][dimForLoop] = 8-1;
              //
  transferNode->srcLdsAndLoopOffsets_.constEleOffsets_[0][1][dimForLoop] = 8-1;
              // transferNode->srcLdsAndLoopOffsets_.constEleOffsets_[1] =
  transferNode->srcLdsAndLoopOffsets_.constEleOffsets_[0]; for (int c :
  currDsc->coreIdsUsed_){

                for (int clId=0;clId<currDsc->numCoreletsUsed_;clId++){
                  transferNode->srcLdsAndLoopOffsets_.constEleOffsets_[c][clId][dimForLoop]
  = 8-1;

                }

              }
              for (int clId=0;clId<currDsc->numCoreletsUsed_;clId++){
                for (auto& loopnode:
  transferNode->srcLdsAndLoopOffsets_.loopEleOffsets_[clId]){ for (auto&
  withinLoopNode: loopnode.second){ if (withinLoopNode.first == dimForLoop){
                      std::cout<<"Inside
  negation"<<withinLoopNode.second<<std::endl; withinLoopNode.second =
  -1*withinLoopNode.second;
                    }
                  }
                }
              }

              // for (auto& eachdst:
  transferNode->srcLdsAndLoopOffsets_.loopEleOffsets_){

              // }

              //
  transferNode->srcLdsAndLoopOffsets_.constEleOffsets_[1][0][dimForLoop] = 8-1;
              //
  transferNode->srcLdsAndLoopOffsets_.constEleOffsets_[1][1][dimForLoop] = 8-1;
              // transferNode->srcLdsAndLoopOffsets_.loopEleOffsets_[0]
          //     for (auto& sourceNode :
  currDsc->scheduleTree_.traverseTreeDFSMutable(
          //               nullptr, {dsc2::TransferNode::TRANSFER})) {
          //       auto* sourceTransferNode =
  static_cast<dsc2::TransferNode*>(sourceNode);
          //       for (auto& eachdst:
  sourceTransferNode->dstLdsAndLoopOffsets_){
          //         if
  (eachdst.dataConnect_==transferNode->srcLdsAndLoopOffsets_.dataConnect_){
          //           //for key in the
  sourcetransfernode.srclds.loopelementoffser, create equivalent one in
  constEleOffset
          //           for(auto& map_loop: eachdst.loopEleOffsets_[0]){
          //             auto& innerMap = map_loop.second;
          //             if (innerMap.find(dimForLoop)!=innerMap.end()){
          //               constLoopOffset[0][0][dimForLoop] =
  innerMap[dimForLoop]-1;
          //               constLoopOffset[0][1][dimForLoop] =
  innerMap[dimForLoop]-1;
          //
  transferNode->srcLdsAndLoopOffsets_.constEleOffsets_[0][0][dimForLoop] =
  innerMap[dimForLoop]-1;
          //
  transferNode->srcLdsAndLoopOffsets_.constEleOffsets_[0][1][dimForLoop] =
  innerMap[dimForLoop]-1;
          //             }
          //           }

          //         }
          //       }

          //     // std::unordered_map<PrimaryDimTypes, int> dsSizePerDim =
          //     //     currDsc->getBlockTransferSizePerDim(*transferNode,
  transferNode->src_.unit_, 0, false,
          //     //                               false, true);
          //     // std::cout<<dsSizePerDim[dimForLoop]<<std::endl;
          //     // for(auto& eachdim: transferNode->unitTimeTransfer_){
          //     //   std::cout<<eachdim.sizeDim_.size_<<"Outside"<<std::endl;
          //     //   if(eachdim.sizeDim_.dim_==dimForLoop){
          //     //     constLoopOffset = eachdim.sizeDim_.size_;
          //     //     std::cout<<"Inside "<<constLoopOffset <<std::endl;
          //     //     break;
          //     //   }
          //     // }

          // }

    }
    // currDsc->exportJson("test.json");
  }
  }*/
}

// ------------------------------------------------------------------------------------------------
// entry 261/382   level 1   scc 318   127 body lines
// unit: e261_finalizeOps
// authority: ddc/ddcv1.cpp:3329
// original: void Ddc::finalizeOps()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// NOTE: Binds register names to allocation start addresses (ddcv1.cpp:3345-3392) -- the `R +
//       std::to_string(startAddress)` convention our islands already read back.
// ------------------------------------------------------------------------------------------------
void e261_finalizeOps()
{
  // opaque ops
  for (const auto& [cn, opaqueProp] : metadata.opaqueOps_) {
    const int unrollFactor =
        opaqueProp.internalRegAlloc_
            ? currDsc->getBufferCapacityForNode(
                  opaqueProp.internalRegAlloc_,
                  opaqueProp.internalRegAlloc_->ldsIdx_,
                  opaqueProp.internalRegAlloc_->component_, 0, 0) /
                  dscGlobal.sysDef.bytesPerStick
            : 1;
    if ((unrollFactor == 0) || (unrollFactor & (unrollFactor - 1))) {
      DT_ERROR("Unroll factor in opaque op is not a power of two");
    }
    cn->instrAttribute_.param_map_["unroll"] = std::to_string(unrollFactor);
    auto insertReg = [&unrollFactor](const std::string& name,
                                     int64_t startAddress,
                                     std::map<std::string, std::string>& regMap,
                                     bool unrollIdx) {
      auto pos = name.find("_unroll");
      if (pos == std::string::npos) {
        regMap[name] = "R" + std::to_string(startAddress);
      } else {
        auto baseName = name.substr(0, pos + 1);  // base name including _
        for (int i = 0; i < unrollFactor; i++) {
          regMap[baseName + std::to_string(i)] =
              "R" + std::to_string(startAddress + (unrollIdx ? i : 0));
        }
      }
    };
    auto getStartAddress = [this](const FoldManager<int64_t>& addressFM) {
      auto allAddr = addressFM.getAllData();
      if (!std::equal(allAddr.begin() + 1, allAddr.end(), allAddr.begin())) {
        DT_ERROR(
            "Opaque op with different start addresses per core/corelet not "
            "supported");
      }
      int64_t startAddr = allAddr.at(0);
      return startAddr / dscGlobal.sysDef.bytesPerStick;
    };
    // handle internal registers
    if (!opaqueProp.internalRegs_.empty()) {
      DT_CHECK(opaqueProp.internalRegAlloc_ != nullptr);
      int64_t startAddress = getStartAddress(
          opaqueProp.internalRegAlloc_->startAddressCoreCorelet_);
      for (const auto& regName : opaqueProp.internalRegs_) {
        insertReg(regName,
                  startAddress + cn->instrAttribute_.read_write_reg_map_.size(),
                  cn->instrAttribute_.read_write_reg_map_, true);
      }
    }
    // handle input/output registers
    for (auto& [regName, regAlloc] : opaqueProp.inOutRegAllocs_) {
      int regUnrollFactor =
          regAlloc->ldsIdx_ >= 0
              ? (currDsc->getBufferCapacityForNode(regAlloc, regAlloc->ldsIdx_,
                                                   regAlloc->component_, 0, 0) /
                 dscGlobal.sysDef.bytesPerStick)
              : 1;
      DT_CHECK_MSG(regUnrollFactor == unrollFactor || regUnrollFactor == 1,
                   "Size of opaque op and register alloc are incompatible");
      insertReg(regName, getStartAddress(regAlloc->startAddressCoreCorelet_),
                cn->instrAttribute_.read_only_reg_map_, regUnrollFactor > 1);
    }
    // handle fp32 ops
    if (cn->dataFormat_ == DataFormats::IEEE_FP32)
      cn->instrAttribute_.param_map_["prec"] = "fp32";
    else
      cn->instrAttribute_.param_map_["prec"] = "fp16";
  }

  // implicit syncs
  std::unordered_set<SenComponents> memoriesWithImplicitSyncs;
  for (const auto& [syncNode, allocNode] : metadata.implicitSyncs_) {
    if (!memoriesWithImplicitSyncs.insert(allocNode->component_).second) {
      DT_ERROR("There can only be one implicit sync for each memory");
    }
    SenComponents destComp = SenComponents::NO_COMPONENT;
    if (allocNode->component_ == L0) {
      destComp = L0SU;
    } else {
      DT_ERROR("Implicit syncs not possible in units other than L0SU/LU");
    }
    if (dscGlobal.sysDef.coreArch < RCUDD1A_ISA) {
      DT_ERROR(
          "Implicit syncs not available for architectures prior to RCUDD1A");
    }
    auto transfers = currDsc->scheduleTree_.traverseTreeDFS(
        allocNode->getPrev(), {dsc2::ScheduleNode::TRANSFER}, destComp);
    auto refTransfer = transfers[0];
    if (transfers.size() != 1) {
      if (destComp == L0SU) {
        // For Mx-matmul, two LXLU->L0SU transfers are needed, one to storage L0
        // and the other to storage L0_SCALE. In the presence of implicit sync,
        // we need to ensure that multiple transfers are not writing to the same
        // destination storage (either L0 or L0_SCALE).
        std::set<SenComponents> transferStorages;
        for (auto& node : transfers) {
          SenComponents destStorage =
              static_cast<const dsc2::TransferNode*>(node)
                  ->dstVias_.at(0)
                  .loc_.storage_;
          if (transferStorages.count(destStorage)) {
            DT_ERROR("There should only be one transfer into L0 storage= " +
                     EnumsConversion::senComponentsToString.at(destStorage) +
                     " that has implicit "
                     "syncs");
          }
          transferStorages.insert(destStorage);
          if (destStorage == L0) {
            refTransfer = node;
          }
        }
      } else {
        DT_ERROR(
            "There should only be one transfer into memory " +
            EnumsConversion::senComponentsToString.at(allocNode->component_) +
            " that has implicit "
            "syncs");
      }
    }
    syncNode->implicitSyncRefTransfer_ =
        static_cast<const dsc2::TransferNode*>(refTransfer);
  }

  setPeFoldsIfPtInteraction(currDsc, dscGlobal);
}

// ------------------------------------------------------------------------------------------------
// entry 262/382   level 1   scc 320   181 body lines
// unit: e262_coordinateMasking
// authority: ddc/ddcv1.cpp:3485
// original: void Ddc::coordinateMasking()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e262_coordinateMasking()
{
  if (currDsc->coordinateMasking_.empty()) {
    if (currDsc->computeOp_.at(0).opConsts.count("samv-wsllen")) {
      DT_ERROR(
          "coordinateMasking_ field in DSC should have been filled by "
          "upstream "
          "tools");
    }
    return;
  }

  bool anyDimValidMasking = false;
  for (const auto& [dim, masking] : currDsc->coordinateMasking_) {
    if (masking.size() != 1) {
      DT_ERROR(
          "Cannot currently apply coordinate masking that is not at the end");
    }
    if (masking[0].second > 0) anyDimValidMasking = true;
  }

  // If there is not any dimension with masking elements, then don't construct
  // stickMask nodes.
  if (!anyDimValidMasking) return;

  for (const auto& [dim, masking] : currDsc->coordinateMasking_) {
    if (metadata.clSplitDims_.count(dim)) {
      DT_ERROR(
          "Cannot currently apply coordinate masking for dimensions split "
          "across corelets");
    }
    if (sdsc_->numWkSlicesPerDim_.at(dim) > 1) {
      DT_ERROR(
          "Cannot currently apply coordinate masking for dimensions split "
          "across cores");
    }
    if (currDsc->dimToSymbolMapping_.count(dim)) {
      DT_ERROR(
          "Cannot currently apply coordinate masking for symbolic dimensions");
    }
  }

  if (!currDsc->constantInfo_.count(currDsc->maskingConstId_)) {
    DT_ERROR("Coordinate masking requested but no masking constant provided");
  }
  dsc2::StickMaskNode refMaskNode;
  refMaskNode.name_ = "SAMV";
  refMaskNode.maskValConstId_ = currDsc->maskingConstId_;
  // find LXLU transfers and verify constraints
  auto transfers = currDsc->scheduleTree_.traverseTreeDFS(
      nullptr, {dsc2::ScheduleNode::TRANSFER});
  for (auto* node : transfers) {
    auto* tn = static_cast<const dsc2::TransferNode*>(node);
    if (tn->src_.unit_ != LXLU) continue;
    if (tn->srcLdsAndLoopOffsets_.myLdsIdx_ < 0) {
      DT_ERROR(
          "Cannot currently apply coordinate masking only on some transfers. "
          "Constant transfer detected from LXLU.");
    }
    const auto& lds =
        currDsc->labeledDs_.at(tn->srcLdsAndLoopOffsets_.myLdsIdx_);
    auto stickComp = currDsc->getStickSizes(lds.dsType_);
    std::vector<dsc2::ScheduleNode::Size> stickDimSizes(stickComp.begin(),
                                                        stickComp.end());
    auto noBcastDims = currDsc->getNonBroadcastLdsDimSet(lds.ldsIdx_);
    if (std::none_of(stickDimSizes.begin(), stickDimSizes.end(),
                     [&](dsc2::ScheduleNode::Size& x) {
                       return noBcastDims.count(x.dim_);
                     }))
      continue;

    if (!refMaskNode.stickLayout_.empty()) {
      if (refMaskNode.stickLayout_ != stickDimSizes) {
        DT_ERROR(
            "Cannot currently apply coordinate masking on multiple data "
            "structures with different stick layout");
      }
      refMaskNode.affectedTransfers_.push_back(tn);
      continue;
    }
    // save stick composition and check compatibility
    refMaskNode.stickLayout_ = stickDimSizes;
    bool relevant =
        std::any_of(stickDimSizes.begin(), stickDimSizes.end(),
                    [&](const dsc2::ScheduleNode::Size& x) {
                      return currDsc->coordinateMasking_.count(x.dim_);
                    });
    if (!relevant) {
      DT_ERROR(
          "Cannot currently apply coordinate masking only on some transfers. "
          "Transfer detected not affected by masking");
    }
    refMaskNode.affectedTransfers_.push_back(tn);
    refMaskNode.dataFormat_ = lds.dataFormat_;
    auto crossSlice = currDsc->getStickSizes(lds.dsType_, false, true);
    auto withinSlice = currDsc->getStickSizes(lds.dsType_, true, false);
    if (crossSlice.size() != 1 || withinSlice.size() > 2 ||
        (crossSlice[0].first != withinSlice[0].first &&
         (withinSlice.size() == 1 ||
          crossSlice[0].first != withinSlice[1].first))) {
      DT_ERROR("Too many dimensions in stick, not compatible with SAMV");
    }
    auto cumulativeStickComp = currDsc->getCumulativeStickSizes(lds.dsType_);
    for (const auto& [dim, size] : cumulativeStickComp) {
      if (!currDsc->coordinateMasking_.count(dim)) continue;
      int numMaskedElem = currDsc->coordinateMasking_.at(dim)[0].second;
      if (numMaskedElem > size) {
        DT_ERROR(
            "Cannot currently apply coordinate masking beyond a single "
            "stick");
      }
      refMaskNode.firstStickCoordToMaskPerDim_.emplace(dim,
                                                       size - numMaskedElem);
    }
  }

  // information collected and SAMV node ready, insert in schedule tree
  std::unordered_set<dsc2::BlockNode*> insertionNodes;
  for (auto* tn : refMaskNode.affectedTransfers_) {
    // find highest possible insertion point
    const dsc2::ScheduleNode* lastSeenNode = tn;
    while (lastSeenNode->prev_ &&
           lastSeenNode->prev_->prev_) {  // stop before root
      /* if (lastSeenNode->prev_->nodeType_ == dsc2::ScheduleNode::BLOCK &&
          lastSeenNode->prev_->prev_->nodeType_ ==
              dsc2::ScheduleNode::CONDITION) {
        break;
      } else */
      if (lastSeenNode->prev_->nodeType_ == dsc2::ScheduleNode::LOOP) {
        auto* loop = static_cast<dsc2::LoopNode*>(lastSeenNode->prev_);
        if (std::any_of(loop->dims_.begin(), loop->dims_.end(),
                        [&](const PrimaryDimAndKind& x) {
                          return refMaskNode.firstStickCoordToMaskPerDim_.count(
                              x.dim_);
                        })) {
          break;
        }
      }
      lastSeenNode = lastSeenNode->prev_;
    }
    if (!insertionNodes.insert(lastSeenNode->prev_).second) continue;
    DT_CHECK_MSG(insertionNodes.size() == 1,
                 "Multiple SAMV needed at different insertion points. Not "
                 "currently supported");
    // create condition node based on all loops related to dim
    if (refMaskNode.firstStickCoordToMaskPerDim_.size() != 1) {
      DT_ERROR("Cannot currently mask more than one dim at a time");
    }
    const auto& [dim, coord] =
        *refMaskNode.firstStickCoordToMaskPerDim_.begin();
    auto* cond = new dsc2::ConditionNode();
    lastSeenNode->prev_->addChildNode(cond, true, lastSeenNode);
    cond->name_ =
        "condition_SAMV_dim_" + EnumsConversion::primaryDimToString.at(dim);
    auto& loopCondAnd = cond->loopCond_.twoLevelOrOfAnds_.emplace_back();
    auto* currLoopParent = lastSeenNode->getOwnerLoop();
    while (currLoopParent->prev_) {  // root node
      for (auto& [loopDim, kind] : currLoopParent->dims_) {
        if (loopDim == dim) {
          loopCondAnd.emplace_back(currLoopParent, dim, CondOp::EQ,
                                   dsc2::LoopCond::LAST);
          break;
        }
      }
      currLoopParent = currLoopParent->getOwnerLoop();
    }
    // place SAMV node in then branch and SAMV reset in else
    auto* thenBlock = new dsc2::BlockNode();
    cond->addThenRegion(thenBlock);
    thenBlock->name_ =
        "block_SAMV_dim_" + EnumsConversion::primaryDimToString.at(dim);
    auto* samvNode = new dsc2::StickMaskNode(refMaskNode);
    thenBlock->addChildNode(samvNode);
    auto* elseBlock = new dsc2::BlockNode();
    cond->addElseRegion(elseBlock);
    elseBlock->name_ = "block_SAMV_reset";
    auto* samvReset = new dsc2::StickMaskNode(refMaskNode);
    elseBlock->addChildNode(samvReset);
    samvReset->name_ = "SAMV_reset";
    samvReset->firstStickCoordToMaskPerDim_.clear();
  }
}

// ------------------------------------------------------------------------------------------------
// entry 263/382   level 1   scc 323   8 body lines
// unit: e263_identifyBelowChunkBoundaryLoops
// authority: ddc/ddcv1.cpp:3683
// original: void Ddc::identifyBelowChunkBoundaryLoops()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e263_identifyBelowChunkBoundaryLoops()
{
  DT_CHECK(metadata.belowLxScheduleInsertBlock);
  loopsBelowChunkBoundary.clear();
  for (auto* node : currDsc->scheduleTree_.traverseTreeDFS(
           metadata.belowLxScheduleInsertBlock, {dsc2::ScheduleNode::LOOP})) {
    loopsBelowChunkBoundary.insert(static_cast<const dsc2::LoopNode*>(node));
  }
}

// ------------------------------------------------------------------------------------------------
// entry 264/382   level 1   scc 367   19 body lines
// unit: e264_bin_op
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:137
// original: ComputationOp bin_op(DimSymbol dim, const std::vector<int>& indices, bool expand_indices = true)
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
ComputationOp e264_bin_op(DimSymbol dim, const std::vector<int>& indices,
                     bool expand_indices = true)
{
  ComputationOp op;
  op.inputs.push_back({{dim, false}});
  op.inputs.push_back({{dim, true}});
  op.codegen = [indices, expand_indices](ComputationBuilder& builder,
                                         const std::vector<DataEdge>& input,
                                         DataEdge& output) {
    DT_CHECK(input.size() == 2);
    builder.insert_packmerge(input[0], input[1], output, indices,
                             expand_indices);
  };
  op.codegen_psuedocode = [indices](const std::vector<std::string>& in_regs,
                                    const std::string& out_reg) {
    DT_CHECK(in_regs.size() == 2);
    return std::vector<std::string>{
        packmerge_psuedostring(indices, in_regs[0], in_regs[1], out_reg)};
  };
  return op;
}

// ------------------------------------------------------------------------------------------------
// entry 265/382   level 1   scc 368   17 body lines
// unit: e265_unary_op
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:160
// original: ComputationOp unary_op(const std::vector<int>& indices)
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
ComputationOp e265_unary_op(const std::vector<int>& indices)
{
  ComputationOp op;
  op.inputs.emplace_back();  // Need a stick, no particular index.
  op.reuses_sticks = true;
  op.codegen = [indices](ComputationBuilder& builder,
                         const std::vector<DataEdge>& input, DataEdge& output) {
    DT_CHECK(input.size() == 1);
    builder.insert_packmerge(input[0], input[0], output, indices);
  };
  op.codegen_psuedocode = [indices](const std::vector<std::string>& in_regs,
                                    const std::string& out_reg) {
    DT_CHECK(in_regs.size() == 1);
    return std::vector<std::string>{
        packmerge_psuedostring(indices, in_regs[0], in_regs[0], out_reg)};
  };
  return op;
}

// ------------------------------------------------------------------------------------------------
// entry 266/382   level 1   scc 282   18 body lines
// unit: e266_canonicalize_layout
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:690
// original: AbstractLayout AutoShuffler::canonicalize_layout(const AbstractLayout& layout, const AbstractLayout& goal)
// class: AutoShuffler
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
AbstractLayout e266_canonicalize_layout(const AbstractLayout& layout,
                                                 const AbstractLayout& goal)
{
  DT_CHECK(layout.sliceDims().size() == dim_64bit + 1);
  DT_CHECK(goal.sliceDims().size() == dim_64bit + 1);
  AbstractLayout canonicalized({}, {}, layout.format);
  for (auto dim : layout.stickDims()) {
    if (goal.contains(dim)) {
      canonicalized.stick_dims.insert(dim);
    }
  }
  for (auto dim : layout.sliceDims()) {
    if (goal.contains(dim)) {
      canonicalized.slice_dims.push_back(dim);
    } else {
      canonicalized.slice_dims.push_back(DimSymbol::getDummy());
    }
  }
  return canonicalized;
}

// ------------------------------------------------------------------------------------------------
// entry 267/382   level 1   scc 280   6 body lines
// unit: e267_reset_graph
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:710
// original: void AutoShuffler::reset_graph()
// class: AutoShuffler
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e267_reset_graph()
{
  layout_to_nodes.clear();
  // clear priority queue by assigning a new one
  worklist = worklist_type();
  worklist_counter = 0;
}

// ------------------------------------------------------------------------------------------------
// entry 268/382   level 1   scc 377   15 body lines
// unit: e268_sort_by_stick_key
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:746
// original: template <bool sort_inputs> void sort_by_stick_key(const std::function<uint32_t(const StickIndex&)> key, std::vector<ComputationOp>& ops)
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
template <bool sort_inputs>
void e268_sort_by_stick_key(const std::function<uint32_t(const StickIndex&)> key,
                       std::vector<ComputationOp>& ops)
{
  if constexpr (sort_inputs) {
    std::sort(ops.begin(), ops.end(),
              [&key](const ComputationOp& a, const ComputationOp& b) {
                DT_CHECK(a.inputs.size() > 0);
                DT_CHECK(b.inputs.size() > 0);
                return key(a.inputs[0]) < key(b.inputs[0]);
              });
  } else {
    std::sort(ops.begin(), ops.end(),
              [&key](const ComputationOp& a, const ComputationOp& b) {
                return key(a.output) < key(b.output);
              });
  }
}

// ------------------------------------------------------------------------------------------------
// entry 269/382   level 1   scc 378   17 body lines
// unit: e269_check_single_inorder_accesses
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:767
// original: template <bool check_inputs> bool check_single_inorder_accesses( const std::function<uint32_t(const StickIndex&)> key, const std::vector<ComputationOp>& ops)
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
template <bool check_inputs>
bool e269_check_single_inorder_accesses(
    const std::function<uint32_t(const StickIndex&)> key,
    const std::vector<ComputationOp>& ops)
{
  int next_access = 0;
  bool flag = true;
  for (const auto& op : ops) {
    if constexpr (check_inputs) {
      if (op.reuses_sticks) flag = false;
      for (const auto& access : op.inputs) {
        if (key(access) != next_access) flag = false;
        next_access++;
      }
    } else {
      if (key(op.output) != next_access) flag = false;
      next_access++;
    }
  }
  return flag;
}

// ------------------------------------------------------------------------------------------------
// entry 270/382   level 1   scc 289   90 body lines
// unit: e270_inferLayouts
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:1047
// original: std::pair<ConcreteLayout, ConcreteLayout> AutoShuffler::inferLayouts( const PrimaryDsInfo& in, const PrimaryDsInfo& out)
// class: AutoShuffler
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::pair<ConcreteLayout, ConcreteLayout> e270_inferLayouts(
    const PrimaryDsInfo& in, const PrimaryDsInfo& out)
{
  DT_CHECK(all_one(out.stickRepl_));
  DT_CHECK(all_one(in.stickRepl_));

  std::pair<ConcreteLayout, ConcreteLayout> layouts;
  ConcreteLayout& layout_in = layouts.first;
  ConcreteLayout& layout_out = layouts.second;

  // We are given a layout with coarse-grain dimensions.
  // Break this down into subdimensions of size 2
  std::vector<PrimaryDimTypes> in_subdimensions;
  std::vector<PrimaryDimTypes> out_subdimensions;
  std::map<PrimaryDimTypes, int> in_dim_counts;
  std::map<PrimaryDimTypes, int> out_dim_counts;
  std::set<PrimaryDimTypes> involved_dimensions;
  // Do the same work for both in and out
  for (bool do_in : {true, false}) {
    const PrimaryDsInfo& ds_info = (do_in) ? in : out;
    auto& subdimensions = (do_in) ? in_subdimensions : out_subdimensions;
    auto& dim_counts = (do_in) ? in_dim_counts : out_dim_counts;

    for (int j = 0; j < ds_info.stickDimOrder_.size(); j++) {
      PrimaryDimTypes pdim = ds_info.stickDimOrder_[j];
      involved_dimensions.insert(pdim);
      int num_subdims = int_log2(ds_info.stickSize_[j]);
      DT_CHECK(ds_info.stickSize_[j] == (1 << num_subdims));
      dim_counts[pdim] += num_subdims;  // will default construct to 0 += x
      subdimensions.insert(subdimensions.end(), num_subdims, pdim);
    }
  }

  // We need to ignore dimensions that go across slices, and they need to be
  // the same for both input and output
  for (int i = 0; i < int_log2(num_slices); i++) {
    DT_CHECK(in_subdimensions.back() == out_subdimensions.back());
    in_dim_counts[in_subdimensions.back()]--;
    out_dim_counts[out_subdimensions.back()]--;
    in_subdimensions.pop_back();
    out_subdimensions.pop_back();
  }

  // Prepare IDs for each subdim that we can match across in/out.
  std::map<PrimaryDimTypes, std::vector<DimSymbol>> dim_mapping;
  DimSymbol next_symbol = DimSymbol::getDefaultSymbol();
  for (auto pdim : involved_dimensions) {
    int n_in = in_dim_counts[pdim];
    int n_out = out_dim_counts[pdim];
    for (int i = 0; i < std::max(n_in, n_out); i++) {
      dim_mapping[pdim].push_back(next_symbol);
      next_symbol = next_symbol.next();
    }
  }

  // Iterate over subdims an add their ids to the layout.
  // Ids not consumed in the slice must be located in the stick.
  for (bool do_in : {true, false}) {
    const PrimaryDsInfo& ds_info = (do_in) ? in : out;
    auto& subdimensions = (do_in) ? in_subdimensions : out_subdimensions;
    auto& other_dimcount = (do_in) ? out_dim_counts : in_dim_counts;
    auto& layout = (do_in) ? layout_in : layout_out;
    auto& other_layout = (do_in) ? layout_out : layout_in;

    std::map<PrimaryDimTypes, int> symbol_index;
    for (auto pdim : subdimensions) {
      int& idx = symbol_index[pdim];  // will emplace to 0 if needed
      auto sym = dim_mapping[pdim][idx];
      layout.slice_dims.push_back(sym);
      if (idx >= other_dimcount[pdim]) {
        // Not accounted for in other slice, so must come
        // from other stick (in order)
        other_layout.stick_dims.push_back(sym);
      }
      idx++;
    }
  }

  // Fill in symbols for the untouched low-precision dimensions
  for (bool do_in : {true, false}) {
    auto& slice = do_in ? layout_in.slice_dims : layout_out.slice_dims;
    DimSymbol low_dim_symbol = next_symbol;
    int insert_loc = 0;
    while (slice.size() < dim_64bit + 1) {
      slice.insert(slice.begin() + insert_loc, low_dim_symbol);
      low_dim_symbol = low_dim_symbol.next();
      insert_loc++;
    }
  }

  return layouts;
}

// ------------------------------------------------------------------------------------------------
// entry 296/382   level 2   scc 172   15 body lines
// unit: e296_rollBackNodesInBlock
// authority: ddc/ddc.h:511
// original: void rollBackNodesInBlock(DesignSpaceConfig* currDsc, dsc2::BlockNode* blockRoot)
// class: CoordPropTracker
// rust home: crates/compiler/deeptools/src/schedule/ddc/mod.rs
// ------------------------------------------------------------------------------------------------
void e296_rollBackNodesInBlock(DesignSpaceConfig* currDsc,
                              dsc2::BlockNode* blockRoot)
{
      std::vector<dsc2::ScheduleNode*> blockNodes =
          currDsc->scheduleTree_.traverseTreeDFSMutable(
              blockRoot,
              {dsc2::ScheduleNode::ALLOCATE, dsc2::ScheduleNode::COMPUTE,
               dsc2::ScheduleNode::TRANSFER});

      for (int i = 0; i < currItemToProcess_; ++i) {
        if (is_any_of(itemsToProcess_.at(i).refNode, blockNodes) ||
            is_any_of(itemsToProcess_.at(i).nodeToFold, blockNodes)) {
          rollbackToPos(i);
          return;
        }
      }
    }

// ------------------------------------------------------------------------------------------------
// entry 297/382   level 2   scc 195   217 body lines
// unit: e297_gatherRelatedPTRowsBase
// authority: ddc/ddc_fold.cpp:740
// original: bool Ddc::gatherRelatedPTRowsBase(const dsc2::CoordPropInfoType &coordPropInfo, RowGroupInfo &rowGroup)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
bool e297_gatherRelatedPTRowsBase(const dsc2::CoordPropInfoType &coordPropInfo,
                                  RowGroupInfo &rowGroup)
{
  rowGroup.nodeInfo.clear();
  rowGroup.commonGroupAncestor = nullptr;
  if (currDsc->dataStageParam_.at(metadata.core_dstgid).ss_.rowSplit_.empty()) {
    rowGroup.cat = RowGroupInfo::Category::NO_BUNDLING;
    return true;
  }
  PrimaryDimTypes rowSplitDim =
      currDsc->dataStageParam_.at(metadata.core_dstgid)
          .ss_.rowSplit_.begin()
          ->first;

  // Candidate info: <scheduleNode, row, labeledDsIndex, refCoordinate>
  std::vector<std::tuple<dsc2::ScheduleNode *, int, int,
                         dsc2::CoordinateType<CoordinateBaseType> *>>
      primaryCandidateNodes;
  std::unordered_set<dsc2::ScheduleNode::NodeType> nodeTypes;
  if (coordPropInfo.refNode->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    nodeTypes.insert(dsc2::ScheduleNode::COMPUTE);
  } else if (coordPropInfo.refNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
    nodeTypes.insert(dsc2::ScheduleNode::TRANSFER);
  }
  for (auto &node :
       currDsc->scheduleTree_.traverseTreeDFSMutable(nullptr, nodeTypes)) {
    if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      dsc2::TransferNode *transferNode =
          static_cast<dsc2::TransferNode *>(node);
      int srcRowId = getCompRowId(transferNode->src_.unit_);
      // Do we need to loop over all destinations?
      int destRowId = getCompRowId(transferNode->dstVias_.at(0).loc_.unit_);
      if (srcRowId == -1 && destRowId == -1) {
        // The transfer is not related to PT.
        continue;
      }
      bool isRelevant = false;
      int trLdsIdx = -1;
      if (coordPropInfo.refIsProducer) {
        for (int i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e;
             ++i) {
          if (matchDataStream(currDsc, coordPropInfo,
                              transferNode->dstLdsAndLoopOffsets_.at(i),
                              transferNode->dstVias_.at(i).loc_.storage_,
                              false /* !checkRefForAllocate */)) {
            if (!is_any_of(rowSplitDim, getLayoutDimsFromNode(
                                            currDsc, transferNode, -1, i))) {
              // Rowsplit dimension does not appear in the working node's
              // layout.
              rowGroup.cat = RowGroupInfo::Category::NO_BUNDLING;
              return true;
            }
            isRelevant = true;
            trLdsIdx = transferNode->dstLdsAndLoopOffsets_.at(i).myLdsIdx_;
            break;
          }
        }
      } else {
        if (matchDataStream(currDsc, coordPropInfo,
                            transferNode->srcLdsAndLoopOffsets_,
                            transferNode->src_.storage_,
                            false /* !checkRefForAllocate */)) {
          if (!is_any_of(rowSplitDim,
                         getLayoutDimsFromNode(currDsc, transferNode, 0, -1))) {
            // Rowsplit dimension does not appear in the working node's layout.
            rowGroup.cat = RowGroupInfo::Category::NO_BUNDLING;
            return true;
          }
          isRelevant = true;
          trLdsIdx = transferNode->srcLdsAndLoopOffsets_.myLdsIdx_;
        }
      }
      if (!isRelevant) {
        // Skip this scheduleNode.
        continue;
      }
      if (!transferNode->transferCoordinates_.coordinates_.count(
              rowSplitDim)) {  // foldConstructed()) {
        rowGroup.cat = RowGroupInfo::Category::NO_BUNDLING;
        primaryCandidateNodes.clear();
        return false;
      }
      // The transferNode's output is consistent with the target group. Record
      // the node for further processing.
      primaryCandidateNodes.push_back(
          {transferNode, (srcRowId != -1) ? srcRowId : destRowId, trLdsIdx,
           &(transferNode->transferCoordinates_)});
    } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      dsc2::ComputeNode *computeNode = static_cast<dsc2::ComputeNode *>(node);
      int rowId = getCompRowId(computeNode->exUnit_);
      if (rowId == -1) {
        // The computeNode does not execute on PT.
        continue;
      }
      if (coordPropInfo.refIsProducer) {
        for (int i = 0, e = computeNode->outputsLdsAndLoopOffsets_.size();
             i < e; ++i) {
          if (matchDataStream(currDsc, coordPropInfo,
                              computeNode->outputsLdsAndLoopOffsets_.at(i),
                              computeNode->outputs_.at(i),
                              false /* !checkRefForAllocate */)) {
            if (!computeNode->outputCoordinate_.coordinates_.count(
                    rowSplitDim)) {  //.foldConstructed()) {
              rowGroup.cat = RowGroupInfo::Category::NO_BUNDLING;
              primaryCandidateNodes.clear();
              return false;
            }
            if (!is_any_of(rowSplitDim, getLayoutDimsFromNode(
                                            currDsc, computeNode, -1, i))) {
              // Rowsplit dimension does not appear in the working node's
              // layout.
              rowGroup.cat = RowGroupInfo::Category::NO_BUNDLING;
              return true;
            }
            primaryCandidateNodes.push_back(
                {computeNode, rowId,
                 computeNode->outputsLdsAndLoopOffsets_.at(i).myLdsIdx_,
                 &(computeNode->outputCoordinate_)});
            break;
          }
        }
      } else {
        for (int i = 0, e = computeNode->inputsLdsAndLoopOffsets_.size(); i < e;
             ++i) {
          if (matchDataStream(currDsc, coordPropInfo,
                              computeNode->inputsLdsAndLoopOffsets_.at(i),
                              computeNode->inputs_.at(i),
                              false /* !checkRefForAllocate */)) {
            if (!computeNode->inputCoordinates_.at(i).coordinates_.count(
                    rowSplitDim)) {  //.foldConstructed()) {
              rowGroup.cat = RowGroupInfo::Category::NO_BUNDLING;
              primaryCandidateNodes.clear();
              return false;
            }
            if (!is_any_of(rowSplitDim, getLayoutDimsFromNode(
                                            currDsc, computeNode, i, -1))) {
              // Rowsplit dimension does not appear in the working node's
              // layout.
              rowGroup.cat = RowGroupInfo::Category::NO_BUNDLING;
              return true;
            }
            primaryCandidateNodes.push_back(
                {computeNode, rowId,
                 computeNode->inputsLdsAndLoopOffsets_.at(i).myLdsIdx_,
                 &(computeNode->inputCoordinates_.at(i))});
            break;
          }
        }
      }
    }
  }

  if (primaryCandidateNodes.empty()) {
    return true;
  }

  // Prune the candidate list to pick only the first node for each PT row.
  const dsc2::CoordinateType<CoordinateBaseType>
      *rowNode[dscGlobal.sysDef.numPTRows];
  for (int i = 0, e = dscGlobal.sysDef.numPTRows; i < e; ++i) {
    rowNode[i] = nullptr;
  }

  for (auto [node, rowId, ldsIdx, coord] : primaryCandidateNodes) {
    if (!rowNode[rowId]) {
      rowNode[rowId] = coord;
      auto beta =
          coord->coordinates_.at(rowSplitDim).getBeta(FOLD_POS_ROWSPLIT);
      if (coordPropInfo.scaleDown &&
          currDsc->labeledDs_.at(ldsIdx).scaledLdsCategory_ ==
              LabeledDsInfo::ScaledLdsCategory::VALUE_TENSOR &&
          currDsc->labeledDs_.at(ldsIdx).mxInfo_.dim == rowSplitDim) {
        beta /= currDsc->labeledDs_.at(ldsIdx).mxInfo_.blkSize;
      }
      rowGroup.nodeInfo.push_back({node, rowId, beta});
      continue;
    }
    if (!sameCoordinateRange(
            node, *rowNode[rowId], node, *coord, SenComponents::ALL,
            -1 /* No need to check for broadcast dims*/, true)) {
      primaryCandidateNodes.clear();
      std::cerr << "\n  LHS:\n";
      rowNode[rowId]->debugPrint(std::cerr, false, "    ");
      std::cerr << "\n  RHS:\n";
      coord->debugPrint(std::cerr, false, "    ");
      std::cerr << "\n  Working node= " << node->name_ << " row= " << rowId;
      DT_ERROR(
          "Mismatched coordinates found while constructing groups for PT "
          "rows.");
    }
  }

  // Find common ancestor
  std::vector<dsc2::ScheduleNode *> refPath;
  dsc2::BlockNode *currNode = rowGroup.nodeInfo.at(0).node->getMutableParent();
  while (currNode != nullptr) {
    refPath.push_back(currNode);
    currNode = currNode->getMutableParent();
  }

  currNode = rowGroup.nodeInfo.at(1).node->getMutableParent();
  while (currNode != nullptr) {
    if (is_any_of(currNode, refPath)) {
      break;
    }
    currNode = currNode->getMutableParent();
  }

  DT_CHECK_MSG(currNode,
               "Row scheduleNodes are not enclosed in a common ancestor Node.");
  rowGroup.commonGroupAncestor = currNode;
  int numChildren = rowGroup.nodeInfo.size();

  // Only allow either one PTrow or all PTrows, nothing in between.
  DT_CHECK_MSG(numChildren == 1 || numChildren == dscGlobal.sysDef.numPTRows,
               "Bundling of " + std::to_string(numChildren) +
                   " PT-rows is not supported.");
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 298/382   level 2   scc 204   245 body lines
// unit: e298_buildFoldFromAllocation
// authority: ddc/ddc_fold.cpp:3043
// original: void Ddc::buildFoldFromAllocation( dsc2::CoordPropInfoType &coordPropInfo, dsc2::ScheduleNode *node, dsc2::CoordinateType<CoordinateBaseType> &coordinate, SenComponents sizeRefComp, SenComponents propRefComp, dsc2::LoopDistributionParamPerNodeType &loopParamsAfterDistribution, RowGroupInfo &refRowGroup)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e298_buildFoldFromAllocation(
    dsc2::CoordPropInfoType &coordPropInfo, dsc2::ScheduleNode *node,
    dsc2::CoordinateType<CoordinateBaseType> &coordinate,
    SenComponents sizeRefComp, SenComponents propRefComp,
    dsc2::LoopDistributionParamPerNodeType &loopParamsAfterDistribution,
    RowGroupInfo &refRowGroup)
{
  DT_CHECK_MSG(coordPropInfo.refNode->nodeType_ == dsc2::ScheduleNode::ALLOCATE,
               "Reference node " + coordPropInfo.refNode->name_ +
                   " must be an allocateNode.");
  dsc2::AllocateNode *refAllocNode =
      static_cast<dsc2::AllocateNode *>(coordPropInfo.refNode);
  if (coordPropReportLevel_ > 0) {
    std::cout << "\n\n>>> buildFoldFromAllocation:"
              << "\n  node= ";
    dbgPrint(node);
    std::cout << "\n  refAllocNode= ";
    dbgPrint(refAllocNode);
    std::cout << "\n  dataConnect= " << coordPropInfo.dataConnect
              << "\n  refIsProducer= "
              << (coordPropInfo.refIsProducer ? "T" : "F")
              << ", scaleDown= " << (coordPropInfo.scaleDown ? "T" : "F")
              << std::endl;

    if (coordPropReportLevel_ > 2) {
      std::cout << "\n\n===================================================="
                << "\n\nReference fold: ";
      const_cast<dsc2::AllocateNode *>(refAllocNode)
          ->allocateCoordinates_.debugPrint(std::cout);
      if (refAllocNode->sliceViewCoordinates_.foldConstructed()) {
        std::cout << "\n\nslice-view coordinates:"
                  << "\n--------------------------";
        const_cast<dsc2::AllocateNode *>(refAllocNode)
            ->sliceViewCoordinates_.debugPrint(std::cout);
      }
    }
  }

  DT_CHECK_MSG(refAllocNode->ldsIdx_ >= 0,
               "Reference allocateNode " + refAllocNode->name_ +
                   " does not have a valid labeledDs index.");
  auto coreIdToWkSliceVariesOnDim =
      [](const std::map<int, std::map<PrimaryDimTypes, int>> &coreIdToWkSlice,
         PrimaryDimTypes dim) {
        bool hasRef = false;
        int refWkSlice = 0;
        for (const auto &[coreId, wkSlice] : coreIdToWkSlice) {
          auto it = wkSlice.find(dim);
          if (it == wkSlice.end()) {
            continue;
          }
          if (!hasRef) {
            refWkSlice = it->second;
            hasRef = true;
            continue;
          }
          if (it->second != refWkSlice) {
            return true;
          }
        }
        return false;
      };

  if (is_any_of(coreletSplitDim, coordPropInfo.dimsToPropagate) &&
      coreIdToWkSliceVariesOnDim(
          refAllocNode->allocateCoordinates_.coreIdToWkSlice_,
          coreletSplitDim)) {
    DT_ERROR(
        "[buildFoldFromAllocation] Can not propagate coordinates for "
        "coreletSplit dimension" +
        EnumsConversion::primaryDimToString.at(coreletSplitDim) +
        " from allocateNode " + refAllocNode->name_ +
        " with custom coreIdToWkSlice.");
  }

  // Find the enclosing loop chain. Collect the associated dimensions.
  dsc2::VectorOfLoopAndDim loopChain;
  bool buildCoreletFold =
      isExternalNode(refAllocNode) &&
      coreletSplitDim != PrimaryDimTypes::PrimaryDimTypesCount;
  getEnclosingLoopsAndRelatedDims(currDsc, node, loopChain,
                                  loopsBelowChunkBoundary, buildCoreletFold);

  int ptRowId = -1;
  if (EnumsConversion::senCompToRowId.count(sizeRefComp)) {
    ptRowId = EnumsConversion::senCompToRowId.at(sizeRefComp);
  }
  auto &effectiveRefCoord =
      (ptRowId != -1 && refAllocNode->sliceViewCoordinates_.foldConstructed()
           ? refAllocNode->sliceViewCoordinates_
           : refAllocNode->allocateCoordinates_);

  const auto &allocLds = currDsc->labeledDs_.at(refAllocNode->ldsIdx_);

  for (auto &[dim, cfm] : effectiveRefCoord.coordinates_) {
    if (!is_any_of(dim, coordPropInfo.dimsToPropagate)) {
      continue;
    }

    std::vector<dsc2::FoldParamInfoType> foldParams;
    gatherFoldParams(cfm, foldParams);
    // Propagation from allocation does not involve row-to-nonrow bundling. This
    // may involve nonrow-to-row UN-bundling. The nonrow-to-row unbundling can
    // be captured on the rowsplit fold directly.
    computeParamsForRowSplitFold(dim, foldParams.at(FOLD_POS_ROWSPLIT),
                                 refRowGroup);
    // Experimental, should be done inside the computeParamsForRowSplitFold
    // function and needs to be generalized.
    if (effectiveRefCoord.getPadding(dim) != PadType::NOPAD &&
        coordinate.getPadding(dim) == PadType::NOPAD) {
      // Divide by the stride.
      foldParams.at(FOLD_POS_ROWSPLIT).alpha /=
          currDsc->dataStageParam_.at(0).ss_.paddingSizes_.at(dim).stride_;
      foldParams.at(FOLD_POS_ROWSPLIT).beta /=
          currDsc->dataStageParam_.at(0).ss_.paddingSizes_.at(dim).stride_;
    }

    dsc2::VectorOfLoopAndDim relatedLoops;
    int dimIdx = currDsc->getDimIndexInLayoutOrder(allocLds.dsType_, dim);
    int scale = dimIdx < 0 ? 1 : allocLds.scale_.at(dimIdx);
    if (scale > 0) {
      // The dimension is a non-broadcast dimension.
      // Note: dim is of type PrimaryDimTypes, metaDimKind is missing here.
      collectRelatedLoops(currDsc, dim, loopChain, relatedLoops,
                          coordinate.getPadding(dim));
    }

    int spatialFoldEnds = effectiveRefCoord.getNumOfSpatialFolds(dim) - 1;
    int refTemporalCount = effectiveRefCoord.getNumOfTemporalFolds(dim);
    int temporalFoldEnds = spatialFoldEnds + refTemporalCount;
    if (refTemporalCount < relatedLoops.size()) {
      // The transferNode has more enclosing loops than the reference
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
      distributeElemArrToTemporalLoops(
          currDsc, dim, node, refAllocNode->ldsIdx_,
          effectiveRefCoord.getPadding(dim), coordinate.getPadding(dim),
          sizeRefComp, propRefComp, loopsToDistribute, elemArr,
          loopParamsAfterDistribution, elemArrParamsAfterDistribution,
          0 /* for one corelet */, coordPropReportLevel_);

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

      for (auto &[loop, loopDim, cat] : loopsToDistribute) {
        int iterationCount = -1;
        if (cat ==
            dsc2::LoopDistributionInfo::LoopDistributionCat::CORELET_SLICE) {
          // Corelet-slicing distribution result is placed under loop==nullptr
          // in loopParamsAfterDistribution.
          foldParams[FOLD_POS_CORELET] = {
              loopParamsAfterDistribution.at(nullptr).at(dim).alpha,
              loopParamsAfterDistribution.at(nullptr).at(dim).beta,
              currDsc->numCoreletsUsed_DSC2_, "corelet_fold"};
          --temporalDiff;
          continue;
        } else if (loop->isParametricLoop()) {
          iterationCount = loop->parametricIterCount(
              currDsc, 0 /* To generalize*/, SenComponents::NO_COMPONENT, -1);
        } else {
          auto &numDs = currDsc->dataStageParam_.at(loop->numId_).ss_;
          auto &denDs = currDsc->dataStageParam_.at(loop->denId_).ss_;
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

    if (coordinate.coordinates_.count(dim)) {
      if (coordPropReportLevel_ > 1) {
        std::cout << "\n[buildFoldFromAllocation] scheduleNode " << node->name_
                  << "'s coordinates already includes fold for dimension "
                  << EnumsConversion::primaryDimToString.at(dim) << std::endl;
      }
      continue;
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
  if (allDimsCovered(currDsc, coordinate, refAllocNode->ldsIdx_)) {
    coordinate.completeFoldConstruction();
  }
  if (coordPropReportLevel_ > 2) {
    std::cout << "\n\n===================================================="
              << "\n\n[buildFoldFromAllocation] fold at end:"
              << "\n  refAllocNode = " << refAllocNode->name_
              << "\n  node         = " << node->name_;
    coordinate.debugPrint(std::cout);
  }
}

// ------------------------------------------------------------------------------------------------
// entry 299/382   level 2   scc 203   355 body lines
// unit: e299_buildFoldFromNonAllocRef
// authority: ddc/ddc_fold.cpp:3294
// original: bool Ddc::buildFoldFromNonAllocRef( dsc2::CoordPropInfoType &coordPropInfo, const int refLdsIdx, const dsc2::CoordinateType<CoordinateBaseType> &refCoordinate, dsc2::CoordinateType<CoordinateBaseType> &coordinate, const SenComponents sizeRefComp, SenComponents propRefComp, dsc2::LoopDistributionParamPerNodeType &loopParamsAfterDistribution, RowGroupInfo &refRowGroup, PrimaryDimTypes foldSingleDim)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
bool e299_buildFoldFromNonAllocRef(
    dsc2::CoordPropInfoType &coordPropInfo, const int refLdsIdx,
    const dsc2::CoordinateType<CoordinateBaseType> &refCoordinate,
    dsc2::CoordinateType<CoordinateBaseType> &coordinate,
    const SenComponents sizeRefComp, SenComponents propRefComp,
    dsc2::LoopDistributionParamPerNodeType &loopParamsAfterDistribution,
    RowGroupInfo &refRowGroup, PrimaryDimTypes foldSingleDim)
{
  dsc2::ScheduleNode *refNode = coordPropInfo.refNode;
  dsc2::ScheduleNode *nodeForFold = coordPropInfo.nodeToFold;
  if (refNode->nodeType_ != dsc2::ScheduleNode::TRANSFER &&
      refNode->nodeType_ != dsc2::ScheduleNode::COMPUTE) {
    DT_ERROR("[buildFoldFromNonAllocRef] Reference node " + refNode->name_ +
             " must be a transfer or compute node, found node of type " +
             dsc2::ScheduleNode::nodeTypeToString.at(refNode->nodeType_));
  }

  if (nodeForFold->nodeType_ != dsc2::ScheduleNode::TRANSFER &&
      nodeForFold->nodeType_ != dsc2::ScheduleNode::COMPUTE) {
    DT_ERROR("[buildFoldFromNonAllocRef] Can not construct fold for node " +
             nodeForFold->name_ + " of type " +
             dsc2::ScheduleNode::nodeTypeToString.at(refNode->nodeType_) +
             ". Node must be a transfer or a compute node.");
  }

  if (coordinate.foldConstructed()) {
    return false;
  }
  if (coordPropReportLevel_ > 0) {
    std::cout << "\n\n>>> buildFoldFromNonAllocRef:"
              << "\n  node= " << nodeForFold->name_ << "(" << nodeForFold << ")"
              << "\n  refNode= " << refNode->name_ << "(" << refNode << ")";
    if (coordPropReportLevel_ > 2) {
      std::cout << "\n\n===================================================="
                << "\nReference fold: ";
      const_cast<dsc2::CoordinateType<CoordinateBaseType> &>(refCoordinate)
          .debugPrint(std::cout);
    }
  }

  std::vector<PrimaryDimTypes> dimList;
  if (foldSingleDim == PrimaryDimTypes::PrimaryDimTypesCount) {
    // Construct folds for all dimensions that are present in the reference
    // node's fold.
    for (auto &[dimFromRefFold, fm] : refCoordinate.coordinates_) {
      dimList.push_back(dimFromRefFold);
    }
  } else {
    // Construct fold only of the specified dimension.
    dimList = {foldSingleDim};
  }

  const auto &refLds = currDsc->labeledDs_.at(refLdsIdx);

  for (auto dim : dimList) {
    if (!is_any_of(dim, coordPropInfo.dimsToPropagate)) {
      continue;
    }
    if (coordinate.coordinates_.count(dim)) {
      if (coordPropReportLevel_ > 1) {
        std::cout << "\n[buildFoldFromNonAllocRef] scheduleNode "
                  << nodeForFold->name_
                  << "'s coordinates already includes fold for dimension "
                  << EnumsConversion::primaryDimToString.at(dim) << std::endl;
      }
      continue;
    }

    // Find common ancestor
    std::vector<dsc2::LoopNode *> loopsToRefNode, loopsToWorkingNode;
    int commonAncestorPosForRef = -1;

    int dimIdx = currDsc->getDimIndexInLayoutOrder(refLds.dsType_, dim);
    int scale = dimIdx < 0 ? 1 : refLds.scale_.at(dimIdx);
    if (scale > 0) {
      // The dimension is a non-broadcast dimension.
      dsc2::LoopNode *currNode = refNode->getMutableOwnerLoop();
      while (currNode != nullptr) {
        if (dsc2::loopRelevantForDim(currDsc, dim, currNode,
                                     refCoordinate.getPadding(dim))) {
          loopsToRefNode.push_back(currNode);
        }
        currNode = currNode->getMutableOwnerLoop();
      }

      currNode = nodeForFold->getMutableOwnerLoop();
      while (currNode != nullptr) {
        if (dsc2::loopRelevantForDim(currDsc, dim, currNode,
                                     coordinate.getPadding(dim))) {
          loopsToWorkingNode.push_back(currNode);
        }
        if (is_any_of(currNode, loopsToRefNode)) {
          break;
        }
        currNode = currNode->getMutableOwnerLoop();
      }

      // Find the position of the innermost common ancestor in the loop chain of
      // the reference node.
      currNode = loopsToWorkingNode.back();
      for (int i = 0, e = loopsToRefNode.size(); i < e; ++i) {
        if (loopsToRefNode.at(i) == currNode) {
          commonAncestorPosForRef = i;
          break;
        }
      }
    }

    // One or both loopchain may be empty.
    if (commonAncestorPosForRef >= 0 &&
        loopsToWorkingNode.back() !=
            loopsToRefNode.at(commonAncestorPosForRef)) {
      DT_ERROR(
          "\n[buildFoldFromNonAllocRef] Failed to find the common ancestor "
          "for " +
          nodeForFold->name_ + " and " + refNode->name_);
    }

    // Build the fold
    std::vector<dsc2::FoldParamInfoType> foldParams;
    gatherFoldParams(refCoordinate.coordinates_.at(dim), foldParams);

    int refSpatialFoldCount = refCoordinate.getNumOfSpatialFolds(dim);
    int refTemporalFoldCount = refCoordinate.getNumOfTemporalFolds(dim);
    int refElemArrFoldCount = refCoordinate.getNumOfElemArrFolds(dim);

    dsc2::FoldParamInfoType rowFoldParamInfo;
    computeParamsForRowSplitFold(dim, rowFoldParamInfo, refRowGroup);

    if (currDsc->dataStageParam_.at(metadata.core_dstgid)
            .ss_.rowSplit_.count(dim)) {
      if (refRowGroup.cat == RowGroupInfo::Category::ROW_TO_SAME_ROW) {
        foldParams.at(FOLD_POS_ROWSPLIT) = rowFoldParamInfo;
      } else if (refRowGroup.cat == RowGroupInfo::Category::ROW_TO_NONROW) {
        // Bundle coordinates from PT-rows.
        //   1. The spatial fold for PT-rows is set to default (noop).
        getDefaultRowSplitFold(foldParams.at(FOLD_POS_ROWSPLIT));
        //   2. Place the rowFoldParamInfo after the innermost common loop
        //      that contains the reference node and the row-siblings.
        dsc2::LoopNode *commonRefRowAncestorLoop = nullptr;
        dsc2::BlockNode *commonRefRowAncestor = refRowGroup.commonGroupAncestor;
        if (commonRefRowAncestor) {
          if (commonRefRowAncestor->nodeType_ == dsc2::ScheduleNode::LOOP) {
            commonRefRowAncestorLoop =
                static_cast<dsc2::LoopNode *>(commonRefRowAncestor);
          } else {
            commonRefRowAncestorLoop =
                commonRefRowAncestor->getMutableOwnerLoop();
          }
          while (commonRefRowAncestorLoop &&
                 !dsc2::loopRelevantForDim(currDsc, dim,
                                           commonRefRowAncestorLoop,
                                           coordinate.getPadding(dim))) {
            commonRefRowAncestorLoop =
                commonRefRowAncestorLoop->getMutableOwnerLoop();
          }
        }
        DT_CHECK_MSG(
            commonRefRowAncestorLoop,
            "[buildFoldFromNonAllocRef] (RefNode=" + refNode->name_ +
                ", workingNode=" + nodeForFold->name_ +
                "Failed to find common ancestor loop for PT-row bundling.");
        for (int i = 0, e = loopsToRefNode.size(); i < e; ++i) {
          if (loopsToRefNode.at(i) == commonRefRowAncestorLoop) {
            // Add a virtual temporal fold immediately inside the temporal fold
            // of the common ancestor of the the scheduleNodes to be bundled.
            foldParams.insert(foldParams.begin() + refSpatialFoldCount +
                                  refTemporalFoldCount - i,
                              rowFoldParamInfo);
            ++refElemArrFoldCount;
            // ++commonAncestorPosForRef;
            break;
          }
        }
      } else if (refRowGroup.cat == RowGroupInfo::Category::NONROW_TO_ROW) {
        // Consider all temporal and element arrangment folds from the refNode
        // as element arrangments. Distribute all loops enclosing the
        // nodeToFold.
        commonAncestorPosForRef = loopsToRefNode.size() - 1;
        loopsToWorkingNode.clear();
        auto currNode = nodeForFold->getMutableOwnerLoop();
        while (currNode != nullptr) {
          if (dsc2::loopRelevantForDim(currDsc, dim, currNode,
                                       coordinate.getPadding(dim))) {
            loopsToWorkingNode.push_back(currNode);
          }
          currNode = currNode->getMutableOwnerLoop();
        }
      }
    }

    if (commonAncestorPosForRef > 0) {
      // Some innermost loops enclosing the reference node do not appear in the
      // loop nest of the working node. Consider loop-diff from reference owner
      // to common ancestor as part of element arrangement of the reference
      // node.
      refElemArrFoldCount +=
          (commonAncestorPosForRef);  // Exclude the common ancestor loop
      refTemporalFoldCount -=
          (commonAncestorPosForRef);  // Exclude the common ancestor loop
    }

    int refTemporalFoldEnds = refSpatialFoldCount + refTemporalFoldCount - 1;
    int workingSpatialFoldEnds = refSpatialFoldCount - 1;
    int workingTemporalFoldEnds = workingSpatialFoldEnds + refTemporalFoldCount;
    if (scale > 0) {
      workingTemporalFoldEnds += (loopsToWorkingNode.size() - 1);
    }

    bool needsDistribution = false;
    dsc2::VectorOfLoopAndDim loopsToDistribute;
    std::vector<dsc2::FoldParamInfoType> elemArrParamsAfterDistribution;
    std::vector<dsc2::FoldParamInfoType> elemArr;

    if (loopsToWorkingNode.size() > 1) {
      // There are some additional loops from the common ancestor to the
      // working node. Distribute diff from working node's owner to common
      // ancestor over the reference element arrangement from previous step.
      needsDistribution = true;
      for (auto loop : loopsToWorkingNode) {
        loopsToDistribute.emplace_back(
            loop, dim,
            loopsBelowChunkBoundary.count(loop)
                ? dsc2::LoopDistributionInfo::LoopDistributionCat::BELOW_CHUNK
                : dsc2::LoopDistributionInfo::LoopDistributionCat::ABOVE_CHUNK);
      }
      // Exclude the common ancestor
      loopsToDistribute.pop_back();

      for (int i = foldParams.size() - 1; i > refTemporalFoldEnds; --i) {
        elemArr.push_back(foldParams.at(i));
      }
    } else if (nodeForFold->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      // ReferenceNode and the working node are at the same loop nest. Check if
      // overriding of transfer size requires re-distribution to some loops.
      auto transferNode = static_cast<dsc2::TransferNode *>(nodeForFold);
      if (transferNode->transferSize_.count(dim)) {
        int transferSize = transferNode->transferSize_.at(dim);
        int accumElemArrCard = 1;
        int accumulatedCard = 1;
        int transferSizeCardPos = -1;
        for (int i = foldParams.size() - 1; i >= refSpatialFoldCount; --i) {
          accumulatedCard *= foldParams.at(i).cardinality;
          if (i > refTemporalFoldEnds) {
            accumElemArrCard *= foldParams.at(i).cardinality;
          }
          if (accumulatedCard >= transferSize) {
            transferSizeCardPos = i;
            break;
          }
        }

        if (accumElemArrCard < transferSize) {
          needsDistribution = true;
          for (int i = 0; i < (refTemporalFoldEnds - transferSizeCardPos + 1);
               ++i) {
            loopsToDistribute.emplace_back(
                loopsToWorkingNode.at(i), dim,
                dsc2::LoopDistributionInfo::BELOW_CHUNK);
          }
          for (int i = foldParams.size() - 1; i >= transferSizeCardPos; --i) {
            elemArr.push_back(foldParams.at(i));
          }
          // Re-distribution happens from the level immediately after
          // refTemporalFoldEnds.
          refTemporalFoldEnds = transferSizeCardPos - 1;
        }
      }
    }

    if (needsDistribution) {
      // Distribute element arrangements to temporal loops and compute the
      // fold parameters for temporal and element arrangement folds.
      distributeElemArrToTemporalLoops(
          currDsc, dim, nodeForFold, refLdsIdx, refCoordinate.getPadding(dim),
          coordinate.getPadding(dim), sizeRefComp, propRefComp,
          loopsToDistribute, elemArr, loopParamsAfterDistribution,
          elemArrParamsAfterDistribution, 0 /* for one corelet */,
          coordPropReportLevel_);

      // Distribution may remove some original element arrangement levels. Clear
      // the old element arrangement folds from foldParams and add the element
      // arrangement levels from the result of distribution..
      foldParams.erase(foldParams.begin() + refTemporalFoldEnds + 1,
                       foldParams.end());
      // The innermost element arrangement is at the beginning
      // of elemArrParamsAfterDistribution.
      for (int j = elemArrParamsAfterDistribution.size() - 1; j >= 0; --j) {
        foldParams.push_back({elemArrParamsAfterDistribution.at(j).alpha,
                              elemArrParamsAfterDistribution.at(j).beta,
                              elemArrParamsAfterDistribution.at(j).cardinality,
                              "elem_arr_" + std::to_string(j)});
      }

      // Insert the additional temporal folds for the working node before the
      // (distributed) element arrangement folds.
      auto iter = foldParams.begin() + refTemporalFoldEnds + 1;
      std::string foldDimStr = EnumsConversion::primaryDimToString.at(dim);
      for (auto &[loopNode, dimAndKind, _] : loopsToDistribute) {
        int iterationCount = -1;
        if (loopNode->isParametricLoop()) {
          int ptRowId = -1;
          if (currDsc->dataStageParam_.at(metadata.core_dstgid)
                  .ss_.rowSplit_.count(dim) &&
              EnumsConversion::senCompToRowId.count(sizeRefComp)) {
            ptRowId = EnumsConversion::senCompToRowId.at(sizeRefComp);
          }
          iterationCount = loopNode->parametricIterCount(
              currDsc, 0 /* To generalize*/,
              SenComponents::NO_COMPONENT /* TO VERIFY */, ptRowId);
        } else {
          auto &numDs = currDsc->dataStageParam_.at(loopNode->numId_).ss_;
          auto &denDs = currDsc->dataStageParam_.at(loopNode->denId_).ss_;
          // Assumption: Spatial fold includes a level for corelets. Therefore,
          // get the datastage values per corelet.
          //
          // To do:
          //   Need to have special case when the corelets may have imbalanced
          //   distribution.
          auto denVal =
              denDs.dataStageDimToVal_compView_st(dim, propRefComp, 0);
          iterationCount =
              numDs.dataStageDimToVal_compView_st(dim, propRefComp, 0) / denVal;
        }
        iter = foldParams.insert(
            iter, {loopParamsAfterDistribution.at(loopNode).at(dim).alpha,
                   loopParamsAfterDistribution.at(loopNode).at(dim).beta,
                   iterationCount, loopNode->name_ + " " + foldDimStr});
      }
      // ----------- Candidate for refactoring (end)   ------------
    }

    // Construct folds after the distribution
    dsc2::CoordinateCategory coordCat = dsc2::CoordinateCategory::UNKNOWN_COORD;
    for (int i = foldParams.size() - 1; i >= 0; --i) {
      std::string foldDimLabel = foldParams.at(i).foldDimLabel;
      if (i > workingTemporalFoldEnds) {
        coordCat = dsc2::CoordinateCategory::ELEM_ARR_COORD;
        foldDimLabel = "elem_arr_" + std::to_string(foldParams.size() - 1 - i);
      } else if (i > workingSpatialFoldEnds) {
        coordCat = dsc2::CoordinateCategory::TEMPORAL_COORD;
      } else {
        coordCat = dsc2::CoordinateCategory::SPATIAL_COORD;
      }
      coordinate.addFold(dim, coordCat, foldParams.at(i).cardinality,
                         foldDimLabel, foldParams.at(i).alpha,
                         foldParams.at(i).beta, 0);
    }
  }
  if (allDimsCovered(currDsc, coordinate, refLdsIdx)) {
    coordinate.completeFoldConstruction();
  }
  if (coordPropReportLevel_ > 2) {
    std::cout << "\n\n===================================================="
              << "\n\n[buildFoldFromNonAllocRef] fold at end:"
              << "\n  ref  = " << refNode->name_
              << "\n  node = " << nodeForFold->name_;
    coordinate.debugPrint(std::cout);
    std::cout << std::endl;
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 300/382   level 2   scc 239   544 body lines
// unit: e300_packStickDim
// authority: ddc/ddc_transformation.cpp:811
// original: bool Ddc::packStickDim()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e300_packStickDim()
{
  dsc2::ComputeNode *releventComputeNode;
  auto [relevent, numberOfInputs] =
      isReleventComputeToPackStickDim(releventComputeNode);
  if (!relevent) return false;

  // iterate though transfer nodes
  std::vector<dsc2::TransferNode *> releventInputTransfers;
  dsc2::TransferNode *releventOutputTransfer;
  std::vector<int> releventTransferDstIdx;
  findReleventStickPackingTransfers(releventComputeNode, releventInputTransfers,
                                    releventOutputTransfer,
                                    releventTransferDstIdx);
  if (releventInputTransfers.size() != numberOfInputs) return false;
  DT_CHECK(releventInputTransfers.size() + 1 == releventTransferDstIdx.size());

  std::vector<int> ldsInputIdx;
  for (int i = 0; i < numberOfInputs; i++)
    ldsInputIdx.push_back(
        releventInputTransfers.at(i)->srcLdsAndLoopOffsets_.myLdsIdx_);
  if (!isEligibleCompute(releventInputTransfers, numberOfInputs, ldsInputIdx))
    return false;
  auto ldsType =
      currDsc->labeledDs_
          .at(releventInputTransfers.at(0)->srcLdsAndLoopOffsets_.myLdsIdx_)
          .dsType_;

  auto ldsOutputIdx = releventOutputTransfer->srcLdsAndLoopOffsets_.myLdsIdx_;
  dsc2::DataInfo releventFinalDataInfo;
  std::vector<dsc2::DataInfo> restOfFinalDataInfo;
  auto &dstVias = releventOutputTransfer->dstVias_;
  for (int i = 0; i < dstVias.size(); i++) {
    if (dstVias.at(i).loc_.storage_ !=
        releventInputTransfers.at(0)->src_.storage_) {
      restOfFinalDataInfo.push_back(
          releventOutputTransfer->dstLdsAndLoopOffsets_.at(i));
    } else {
      releventFinalDataInfo =
          releventOutputTransfer->dstLdsAndLoopOffsets_.at(i);
    }
  }
  std::vector<dsc2::AllocateNode *> allocateIn;
  for (int i = 0; i < numberOfInputs; i++)
    allocateIn.push_back(
        currDsc->labeledDs_.at(ldsInputIdx.at(i)).memOrg_.at(LX).allocateNode_);
  auto allocateOut =
      currDsc->labeledDs_.at(ldsOutputIdx).memOrg_.at(LX).allocateNode_;
  auto &primaryLds = currDsc->primaryDsInfo_.at(ldsType);
  auto &stickDims = primaryLds.stickDimOrder_;
  auto &stickSizes = primaryLds.stickSize_;
  int sticksize = 1;
  for (auto ss : stickSizes) sticksize *= ss;

  // all transfers has to be in the same parent loop
  if (releventInputTransfers.size() > 1) {
    for (auto &tr : releventInputTransfers) {
      if (releventInputTransfers.at(0)->getOwnerLoop() != tr->getOwnerLoop())
        return false;
    }
  }

  // check the size of the other dim if > 1
  // Since all input transfers lives in the same parent loop and LDS type and
  // scale are the same, it is enough to find the dim to form new stick using
  // only one of the transfers.
  std::vector<PrimaryDimTypes> lenderDims;
  auto dimSizesForTransfer = currDsc->getBufferCapacityForNodePerDim(
      releventInputTransfers.at(0), ldsInputIdx.at(0),
      releventInputTransfers.at(0)->src_.storage_, -1, -1);
  for (auto dimSize : dimSizesForTransfer) {
    if (std::find(stickDims.begin(), stickDims.end(), dimSize.first) !=
        stickDims.end())
      continue;
    lenderDims.push_back(dimSize.first);
  }
  // Lender dim need to exist to continue
  if (lenderDims.size() == 0) return false;
  auto parent_loop =
      releventComputeNode->getMutableParentDimLoop(lenderDims.at(0));

  auto dimSizesAboveLevel = currDsc->getBufferCapacityForNodePerDim(
      parent_loop, ldsInputIdx.at(0),
      releventInputTransfers.at(0)->src_.storage_, -1, -1);
  std::vector<std::pair<PrimaryDimTypes, int>> dimSizesNonStick;
  for (auto dimSize : dimSizesAboveLevel) {
    if (dimSize.second == -1) continue;
    bool stickDim = false;
    auto stickDimsIdx =
        std::find(stickDims.begin(), stickDims.end(), dimSize.first) -
        stickDims.begin();
    if (stickDimsIdx < stickDims.size()) {
      stickDim = true;
      int nonStickSize = dimSize.second;
      if (stickSizes.at(stickDimsIdx) > 0)
        nonStickSize = dimSize.second / stickSizes.at(stickDimsIdx);
      if (nonStickSize <= 1) continue;
      dimSizesNonStick.push_back({dimSize.first, nonStickSize});
    }
    if (!stickDim) {
      dimSizesNonStick.push_back(dimSize);
    }
  }

  auto &ldsInput0 = currDsc->labeledDs_.at(ldsInputIdx.at(0));
  auto &ldsOutput = currDsc->labeledDs_.at(ldsOutputIdx);
  auto &inputCoreletSplit =
      currDsc->dataStageParam_.at(parent_loop->numId_).ss_.coreletSplit_;
  std::vector<std::pair<PrimaryDimTypes, int>> newStickDimsSize;
  newStickDimsSize.push_back({stickDims.at(0), 1});
  auto remStickSize = sticksize;
  for (auto dimSize : dimSizesNonStick) {
    if (std::find(parent_loop->dims_.begin(), parent_loop->dims_.end(),
                  dimSize.first) == parent_loop->dims_.end())
      continue;
    if (remStickSize == 1) break;
    // if scale of dim from input and output lds are negative or not equal, the
    // dim is not eligible for new stick dim
    auto scaleDimInput = ldsInput0.scale_.at(
        currDsc->getDimIndexInLayoutOrder(ldsInput0.dsType_, dimSize.first));
    auto scaleDimOutput = ldsOutput.scale_.at(
        currDsc->getDimIndexInLayoutOrder(ldsOutput.dsType_, dimSize.first));
    if (scaleDimInput < 0 || scaleDimOutput < 0 ||
        scaleDimInput != scaleDimOutput)
      continue;
    // Consider corelet split, if the dim is in corelet split
    auto sizeWithCoreletSplit = dimSize.second;
    if (inputCoreletSplit.count(dimSize.first))
      sizeWithCoreletSplit =
          std::gcd(inputCoreletSplit.at(dimSize.first).at(0),
                   inputCoreletSplit.at(dimSize.first).at(1));
    auto largestSizeToGet = std::gcd(sizeWithCoreletSplit, remStickSize);
    if (largestSizeToGet == 1) continue;
    if (remStickSize > largestSizeToGet) {
      remStickSize /= largestSizeToGet;
      newStickDimsSize.push_back({dimSize.first, largestSizeToGet});
    } else {
      newStickDimsSize.push_back({dimSize.first, largestSizeToGet});
      remStickSize = 1;
    }
  }
  // There need to be some compression to continue
  if (remStickSize == sticksize) return false;
  // if there is still stick size left, then fill it with compressed dim
  if (remStickSize != 1) newStickDimsSize.at(0).second = remStickSize;

  // constains:
  // force newBotDs size to be what ever in original stick layout and other dim
  // sizes are 1. And ds_packed size to be what ever in newsticklayout except
  // dims in original stick layout where those are equal the original stick size
  // and other dim sizes are 1.
  int newBotDsId =
      constructDatastage(currDsc->dataStageParam_.at(parent_loop->denId_));
  auto &newBotDs = metadata.datastages_[newBotDsId];
  std::set<PrimaryDimTypes> loopDims(stickDims.begin(), stickDims.end());
  std::set<float> newBotDsSize(stickSizes.begin(), stickSizes.end());
  newBotDs.strategyMinimize_ = false;
  auto &newBotDsConstraints0 = newBotDs.constraints_[-1][loopDims];
  newBotDsConstraints0.updateValues(newBotDsSize);
  newBotDsConstraints0.loopDimKind_ = MetaDimKind::Unpadded;
  std::set<PrimaryDimTypes> minDims;
  for (auto dimAndKind : parent_loop->dims_) {
    if (std::find(loopDims.begin(), loopDims.end(), dimAndKind.dim_) ==
        loopDims.end()) {
      auto &newBotDsConstraints1 = newBotDs.constraints_[-1][{dimAndKind.dim_}];
      newBotDsConstraints1.updateValues({1});
      newBotDsConstraints1.loopDimKind_ = MetaDimKind::Unpadded;
    }
  }

  auto loop0 = constructLoopNode(parent_loop->numId_, newBotDsId,
                                 parent_loop->dims_, parent_loop);
  auto innerLoop0 = splitLoopBandOnDatastage(loop0);

  auto loop1 = static_cast<dsc2::LoopNode *>(loop0->clone());
  auto innerLoop1 = static_cast<dsc2::LoopNode *>(innerLoop0->clone());
  loop1->addChildNode(innerLoop1);

  auto &ds = metadata.datastages_[loop0->denId_];
  ds.strategyMinimize_ = false;
  for (auto dimAndKind : parent_loop->dims_) {
    if (std::find(stickDims.begin(), stickDims.end(), dimAndKind.dim_) !=
        stickDims.end())
      continue;
    bool memberOfNewStick = false;
    for (auto dimSize : newStickDimsSize) {
      if (dimSize.first == dimAndKind.dim_) {
        auto &dsConstraints1 = ds.constraints_[-1][{dimSize.first}];
        dsConstraints1.updateValues({float(dimSize.second)});
        dsConstraints1.loopDimKind_ = MetaDimKind::Unpadded;
        memberOfNewStick = true;
        break;
      }
    }
    if (!memberOfNewStick) {
      auto &dsConstraints2 = ds.constraints_[-1][{dimAndKind.dim_}];
      dsConstraints2.updateValues({1});
      dsConstraints2.loopDimKind_ = MetaDimKind::Unpadded;
    }
  }

  parent_loop->getMutableParent()->addChildNode(loop0, true, parent_loop);
  parent_loop->getMutableParent()->addChildNode(loop1, false, parent_loop);

  auto &chunkDs = currDsc->dataStageParam_.at(parent_loop->numId_);
  int chunkDsCompIdx = currDsc->dataStageParam_.size();
  currDsc->dataStageParam_[chunkDsCompIdx] = chunkDs;
  auto &chunkDsComp = currDsc->dataStageParam_.at(chunkDsCompIdx);
  chunkDsComp.ss_.name_ = std::to_string(chunkDsCompIdx);
  chunkDsComp.el_.name_ = chunkDsComp.ss_.name_ + "el";
  chunkDsComp.ss_.peSfpSplit_.clear();
  chunkDsComp.el_.peSfpSplit_.clear();
  auto &numDs = metadata.datastages_[chunkDsCompIdx];
  auto &denDs = metadata.datastages_[parent_loop->denId_];
  // newStickDimsSize = <out:16, MB:4>
  // oldsticklayout = <out:64>
  for (auto dimSize : newStickDimsSize) {
    if (std::find(stickDims.begin(), stickDims.end(), dimSize.first) !=
        stickDims.end()) {
      auto &dsConstraints = numDs.constraints_[-1][{dimSize.first}];
      dsConstraints.updateValues({float(dimSize.second)});
      dsConstraints.loopDimKind_ = MetaDimKind::Unpadded;
    }
    auto &dsConstraints = denDs.constraints_[-1][{dimSize.first}];
    dsConstraints.updateValues({float(dimSize.second)});
    dsConstraints.loopDimKind_ = MetaDimKind::Unpadded;
  }

  // chunkDsComp --> out: 16
  auto old_numId = parent_loop->numId_;
  parent_loop->numId_ = chunkDsCompIdx;

  // 1) insert a new_parent_loop containing parent_loop between loop0 and loop1
  // 2) insert a conditional in new_parent_loop where we execute parent_loop
  // only in the first iteration of new_parent_loop
  // 3) datastage numId_ is chunkDs and denId_ is chunkDsCompIdx

  auto new_parent_loop = constructLoopNode(old_numId, chunkDsCompIdx,
                                           parent_loop->dims_, parent_loop);
  parent_loop->getMutableParent()->addChildNode(new_parent_loop, false, loop0);

  dsc2::ConditionNode *newCond0 = new dsc2::ConditionNode();
  newCond0->name_ = "cond0_new_stick_layout";
  dsc2::BlockNode *thenBranch0 = new dsc2::BlockNode();
  thenBranch0->name_ = newCond0->name_ + "_then_region";
  newCond0->addThenRegion(thenBranch0);
  std::vector<dsc2::LoopCond> loop_conds0;
  for (auto dimSize : newStickDimsSize) {
    if (std::find(stickDims.begin(), stickDims.end(), dimSize.first) !=
        stickDims.end()) {
      loop_conds0.push_back(dsc2::LoopCond(new_parent_loop, dimSize.first,
                                           CondOp::EQ,
                                           dsc2::LoopCond::CondValType::FIRST));
    }
  }
  newCond0->loopCond_.twoLevelOrOfAnds_.push_back(loop_conds0);

  new_parent_loop->addChildNode(newCond0);
  parent_loop->moveNode(currDsc, thenBranch0);

  // new primaryDsInfo
  auto &compressedDsInfo = currDsc->primaryDsInfo_[DsTypes::INTERNAL];
  compressedDsInfo.layoutDimOrder_ = primaryLds.layoutDimOrder_;
  for (auto dimSize : newStickDimsSize) {
    compressedDsInfo.stickDimOrder_.push_back(dimSize.first);
    compressedDsInfo.stickSize_.push_back(dimSize.second);
    compressedDsInfo.stickRepl_.push_back(1);
  }

  // new internal labelDsInfos for input and output and fix myLdsOut pointer
  std::vector<int> newLdsInIdx;
  for (int i = 0; i < numberOfInputs; i++) {
    newLdsInIdx.push_back(
        addNewLds(&currDsc->labeledDs_.at(currDsc->labeledDs_.size() - 1)));
    ldsOutputIdx++;
    metadata.intermLdsIdxToExtLds[newLdsInIdx.back()] = ldsInputIdx.at(i);
  }
  int newLdsOutIdx =
      addNewLds(&currDsc->labeledDs_.at(currDsc->labeledDs_.size() - 1));
  ldsOutputIdx++;
  metadata.intermLdsIdxToExtLds[newLdsOutIdx] = ldsOutputIdx;

  for (int i = 0; i < numberOfInputs; i++)
    updateNodesWithNewLds(newLdsInIdx.at(i), ldsInputIdx.at(i), parent_loop);
  updateNodesWithNewLds(newLdsOutIdx, ldsOutputIdx, parent_loop);

  std::vector<LabeledDsInfo *> internalLdsIn;
  for (int i = 0; i < numberOfInputs; i++) {
    internalLdsIn.push_back(&currDsc->labeledDs_.at(newLdsInIdx.at(i)));
    internalLdsIn.at(i)->dsName_ =
        "Internal_compressed_input" + std::to_string(i);
    internalLdsIn.at(i)->dsType_ = DsTypes::INTERNAL;
  }
  auto internalLdsOut = &currDsc->labeledDs_.at(newLdsOutIdx);
  internalLdsOut->dsName_ = "Internal_compressed_output";
  internalLdsOut->dsType_ = DsTypes::INTERNAL;

  // internal alloc for input and output
  std::vector<dsc2::AllocateNode *> IntrInAlloc;
  for (int i = 0; i < numberOfInputs; i++) {
    IntrInAlloc.push_back(
        static_cast<dsc2::AllocateNode *>(allocateOut->clone()));
    allocateOut->getMutableParent()->addChildNode(IntrInAlloc.at(i), false,
                                                  allocateOut);
    internalLdsIn.at(i)->memOrg_[IntrInAlloc.at(i)->component_].allocateNode_ =
        IntrInAlloc.at(i);
    internalLdsIn.at(i)->memOrg_[IntrInAlloc.at(i)->component_].isPresent =
        false;
    IntrInAlloc.at(i)->ldsIdx_ = newLdsInIdx.at(i);
  }
  auto IntrOutAlloc = static_cast<dsc2::AllocateNode *>(allocateOut->clone());
  allocateOut->getMutableParent()->addChildNode(IntrOutAlloc, false,
                                                allocateOut);
  internalLdsOut->memOrg_[IntrOutAlloc->component_].allocateNode_ =
      IntrOutAlloc;
  internalLdsOut->memOrg_[IntrOutAlloc->component_].isPresent = false;
  IntrOutAlloc->ldsIdx_ = newLdsOutIdx;

  for (auto dimSize : newStickDimsSize) {
    if (std::find(stickDims.begin(), stickDims.end(), dimSize.first) ==
        stickDims.end()) {
      for (int i = 0; i < numberOfInputs; i++)
        IntrInAlloc.at(i)->gapStickSpread_[dimSize.first] = dimSize.second;
      IntrOutAlloc->gapStickSpread_[dimSize.first] = dimSize.second;
      metadata.opaqueOps_.at(releventComputeNode)
          .internalRegAlloc_->gapStickSpread_[dimSize.first] = dimSize.second;
    }
  }

  dsc2::DataInfo data_info_const;
  data_info_const.myLdsIdx_ = -1;
  dsc2::DataInfo data_info;
  std::vector<dsc2::TransferNode *> transfer_fromlx_comp, transfer_tolx_comp;
  for (int i = 0; i < numberOfInputs; i++) {
    data_info = releventInputTransfers.at(i)->srcLdsAndLoopOffsets_;
    data_info.myLdsIdx_ = ldsInputIdx.at(i);

    // transfer LXLU -> SFP
    auto transfer_fromlx =
        transfer_fromlx_comp.emplace_back(new dsc2::TransferNode());
    transfer_fromlx->src_ = {SenComponents::LXLU, SenComponents::LX};
    transfer_fromlx->dstVias_.push_back(
        {{SenComponents::SFP, SenComponents::SFP}, {}});
    transfer_fromlx->srcLdsAndLoopOffsets_ = data_info;
    data_info.dataConnect_ = "sfp_compress_input" + std::to_string(i) + "_comp";
    transfer_fromlx->dstLdsAndLoopOffsets_.push_back(data_info);
    transfer_fromlx->name_ = "lx_sfp_compress_comp" + std::to_string(i);
    allocateIn.at(i)->addAllocUser(transfer_fromlx);
    allocateIn.at(i)->removeAllocUser(releventInputTransfers.at(i));

    // FMA LXLU*1+0 -> LXSU
    auto compute_fma_comp = new dsc2::ComputeNode();
    compute_fma_comp->exUnit_ = SFP;
    compute_fma_comp->type_ = FMA16;
    compute_fma_comp->inputs_.push_back(LXLU);
    compute_fma_comp->inputs_.push_back(ONE);
    compute_fma_comp->inputs_.push_back(ZERO);
    compute_fma_comp->outputs_.push_back(LXSU);
    compute_fma_comp->inputsLdsAndLoopOffsets_.push_back(data_info);
    compute_fma_comp->inputsLdsAndLoopOffsets_.push_back(data_info_const);
    compute_fma_comp->inputsLdsAndLoopOffsets_.push_back(data_info_const);
    data_info.myLdsIdx_ = newLdsInIdx.at(i);
    data_info.dataConnect_ =
        "sfp_compress_output" + std::to_string(i) + "_comp";
    compute_fma_comp->outputsLdsAndLoopOffsets_.push_back(data_info);
    compute_fma_comp->name_ = "sfp_dummy_fma_comp" + std::to_string(i);

    // transfer SFP -> LXSU
    auto transfer_tolx =
        transfer_tolx_comp.emplace_back(new dsc2::TransferNode());
    transfer_tolx->src_ = {SenComponents::SFP, SenComponents::SFP};
    transfer_tolx->dstVias_.push_back(
        {{SenComponents::LXSU, SenComponents::LX}, {}});
    transfer_tolx->srcLdsAndLoopOffsets_ = data_info;
    data_info.dataConnect_ = "lx_compress_input" + std::to_string(i) + "_comp";
    transfer_tolx->dstLdsAndLoopOffsets_.push_back(data_info);
    transfer_tolx->name_ = "sfp_lx_compress_comp" + std::to_string(i);
    IntrInAlloc.at(i)->allocUsers_.clear();
    IntrInAlloc.at(i)->addAllocUser(transfer_tolx);
    IntrInAlloc.at(i)->addAllocUser(releventInputTransfers.at(i));

    // labeledDS idx is already updated
    releventInputTransfers.at(i)->srcLdsAndLoopOffsets_.dataConnect_ =
        data_info.dataConnect_;
    // Add transfers and compute node to loop0
    innerLoop0->addChildNode(transfer_fromlx);
    innerLoop0->addChildNode(compute_fma_comp);
    innerLoop0->addChildNode(transfer_tolx);
  }
  auto syncSend0 = new dsc2::SyncNode();
  syncSend0->name_ = "sync_lxsu_send_lxlu";
  syncSend0->isReceive_ = false;
  syncSend0->units_.insert(LXSU);
  auto syncRecv0 = new dsc2::SyncNode();
  syncRecv0->name_ = "sync_lxlu_recv_lxsu";
  syncRecv0->isReceive_ = true;
  syncRecv0->units_.insert(LXLU);
  syncSend0->otherEndOfTheSignals_.push_back(syncRecv0);
  syncRecv0->otherEndOfTheSignals_.push_back(syncSend0);
  // Add sync node at the end
  loop0->getMutableParent()->addChildNode(syncSend0, true, new_parent_loop);
  loop0->getMutableParent()->addChildNode(syncRecv0, true, new_parent_loop);

  // transfer LXLU -> SFP
  data_info.myLdsIdx_ = newLdsOutIdx;
  data_info.dataConnect_ = "lx_compress_output_exp";
  auto transfer_fromlx_exp = new dsc2::TransferNode();
  transfer_fromlx_exp->src_ = {SenComponents::LXLU, SenComponents::LX};
  transfer_fromlx_exp->dstVias_.push_back(
      {{SenComponents::SFP, SenComponents::SFP}, {}});
  transfer_fromlx_exp->srcLdsAndLoopOffsets_ = data_info;
  data_info.dataConnect_ = "sfp_compress_output_exp";
  transfer_fromlx_exp->dstLdsAndLoopOffsets_.push_back(data_info);
  transfer_fromlx_exp->name_ = "lx_sfp_compress_exp";
  IntrOutAlloc->allocUsers_.clear();
  IntrOutAlloc->addAllocUser(transfer_fromlx_exp);
  IntrOutAlloc->addAllocUser(releventOutputTransfer);

  // FMA LXLU*1+0 -> LXSU
  auto compute_fma_exp = new dsc2::ComputeNode();
  compute_fma_exp->exUnit_ = SFP;
  compute_fma_exp->type_ = FMA16;
  compute_fma_exp->inputs_.push_back(LXLU);
  compute_fma_exp->inputs_.push_back(ONE);
  compute_fma_exp->inputs_.push_back(ZERO);
  compute_fma_exp->outputs_.push_back(LXSU);
  compute_fma_exp->inputsLdsAndLoopOffsets_.push_back(data_info);
  compute_fma_exp->inputsLdsAndLoopOffsets_.push_back(data_info_const);
  compute_fma_exp->inputsLdsAndLoopOffsets_.push_back(data_info_const);
  data_info.myLdsIdx_ = ldsOutputIdx;
  data_info.dataConnect_ = "sfp_compress_output_exp";
  compute_fma_exp->outputsLdsAndLoopOffsets_.push_back(data_info);
  compute_fma_exp->name_ = "sfp_dummy_fma_exp";

  // transfer SFP -> LXSU
  auto transfer_tolx_exp = new dsc2::TransferNode();
  transfer_tolx_exp->src_ = {SenComponents::SFP, SenComponents::SFP};
  transfer_tolx_exp->dstVias_.push_back(
      {{SenComponents::LXSU, SenComponents::LX}, {}});
  transfer_tolx_exp->srcLdsAndLoopOffsets_ = data_info;
  releventFinalDataInfo.myLdsIdx_ = data_info.myLdsIdx_;
  transfer_tolx_exp->dstLdsAndLoopOffsets_.push_back(releventFinalDataInfo);
  for (auto &di : restOfFinalDataInfo)
    transfer_tolx_exp->dstLdsAndLoopOffsets_.push_back(di);
  transfer_tolx_exp->name_ = "sfp_lx_compress_exp";
  allocateOut->addAllocUser(transfer_tolx_exp);
  allocateOut->removeAllocUser(releventOutputTransfer);

  releventOutputTransfer->dstLdsAndLoopOffsets_.clear();
  releventOutputTransfer->dstLdsAndLoopOffsets_.push_back(
      transfer_fromlx_exp->srcLdsAndLoopOffsets_);
  // Add transfers and compute node to loop1
  innerLoop1->addChildNode(transfer_fromlx_exp);
  innerLoop1->addChildNode(compute_fma_exp);
  innerLoop1->addChildNode(transfer_tolx_exp);

  auto syncSend1 = new dsc2::SyncNode();
  syncSend1->name_ = "sync_lxsu_send_lxlu";
  syncSend1->isReceive_ = false;
  syncSend1->units_.insert(LXSU);
  auto syncRecv1 = new dsc2::SyncNode();
  syncRecv1->name_ = "sync_lxlu_recv_lxsu";
  syncRecv1->isReceive_ = true;
  syncRecv1->units_.insert(LXLU);
  syncSend1->otherEndOfTheSignals_.push_back(syncRecv1);
  syncRecv1->otherEndOfTheSignals_.push_back(syncSend1);
  // Add sync node at the end
  loop0->getMutableParent()->addChildNode(syncSend1, false, new_parent_loop);
  loop0->getMutableParent()->addChildNode(syncRecv1, false, new_parent_loop);

  // fix the address of internal tensors
  std::vector<std::pair<PrimaryDimTypes, int>> capacityLayoutInStick;
  for (auto dimSize : currDsc->getBufferCapacityForNodePerDim(
           allocateOut, allocateOut->ldsIdx_, allocateOut->component_, -1,
           -1)) {
    int idx = std::find(stickDims.begin(), stickDims.end(), dimSize.first) -
              stickDims.begin();
    if (idx < stickSizes.size()) {
      capacityLayoutInStick.push_back(
          {dimSize.first, int(dimSize.second / stickSizes.at(idx))});
    } else {
      capacityLayoutInStick.push_back(dimSize);
    }
  }
  std::vector<int> offset;
  for (int i = 0; i < numberOfInputs; i++) offset.push_back(0);
  for (auto newStickDimSize : newStickDimsSize) {
    if (std::find(stickDims.begin(), stickDims.end(), newStickDimSize.first) !=
        stickDims.end())
      continue;
    int offsetForDim = 128;
    for (auto dimSize : capacityLayoutInStick) {
      if (dimSize.first == newStickDimSize.first) break;
      offsetForDim *= dimSize.second;
    }
    for (int i = 0; i < numberOfInputs; i++)
      offset.at(i) += (newStickDimSize.second - (i + 1)) * offsetForDim;
  }

  for (int i = 0; i < numberOfInputs; i++) {
    for (auto coreIdx : currDsc->coreIdsUsed_) {
      for (int coreletIdx = 0; coreletIdx < currDsc->numCoreletsUsed_DSC2_;
           coreletIdx++) {
        std::map<int64_t, int64_t> indexPosToFix;
        indexPosToFix[0] = coreIdx;
        indexPosToFix[1] = coreletIdx;
        indexPosToFix[2] = 0;
        auto coreCoreletStartAddr =
            IntrInAlloc.at(i)->startAddressCoreCorelet_.getSingleData(
                indexPosToFix);
        std::deque<int64_t> insertIndices;
        insertIndices.push_back(coreIdx);
        insertIndices.push_back(coreletIdx);
        insertIndices.push_back(0);
        IntrInAlloc.at(i)->startAddressCoreCorelet_.insertData(
            coreCoreletStartAddr + offset.at(i), insertIndices);
      }
    }
  }
  for (auto coreIdx : currDsc->coreIdsUsed_) {
    for (int coreletIdx = 0; coreletIdx < currDsc->numCoreletsUsed_DSC2_;
         coreletIdx++) {
      std::map<int64_t, int64_t> indexPosToFix;
      indexPosToFix[0] = coreIdx;
      indexPosToFix[1] = coreletIdx;
      indexPosToFix[2] = 0;
      auto coreCoreletStartAddr =
          IntrOutAlloc->startAddressCoreCorelet_.getSingleData(indexPosToFix);
      std::deque<int64_t> insertIndices;
      insertIndices.push_back(coreIdx);
      insertIndices.push_back(coreletIdx);
      insertIndices.push_back(0);
      IntrOutAlloc->startAddressCoreCorelet_.insertData(
          coreCoreletStartAddr + offset.at(0), insertIndices);
    }
  }

  // unroll the transfers with spread
  std::vector<dsc2::TransferNode *> unroll_transfer;
  for (auto tr : releventInputTransfers) unroll_transfer.push_back(tr);
  unroll_transfer.push_back(releventOutputTransfer);
  for (auto tr : unroll_transfer) unrollTransfer(tr);

  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 301/382   level 2   scc 252   230 body lines
// unit: e301_hoistTransfersUpForReuse
// authority: ddc/ddc_transformation.cpp:1432
// original: bool Ddc::hoistTransfersUpForReuse()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e301_hoistTransfersUpForReuse()
{
  bool didTransformation = false;
  const auto &padInfoMap =
      currDsc->dataStageParam_.at(metadata.core_dstgid).ss_.paddingSizes_;

  for (auto *node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::TRANSFER})) {
    if (isExternalNode(node)) {
      // Skip external nodes.
      continue;
    }
    auto transferNode = static_cast<dsc2::TransferNode *>(node);
    std::set<SenComponents> relatedComponents;
    relatedComponents.insert(transferNode->src_.unit_);
    for (const auto &dstIt : transferNode->dstVias_) {
      relatedComponents.insert(dstIt.loc_.unit_);
      relatedComponents.insert(dstIt.via_.begin(), dstIt.via_.end());
    }
    DT_CHECK(!relatedComponents.count(SenComponents::NO_COMPONENT) &&
             !relatedComponents.count(SenComponents::HBM));
    if (relatedComponents.count(SenComponents::CONSTANT)) {
      // Do not promote external transfers and transfer of constants.
      continue;
    }
    // Note: Information on MetaDimKind is not included in transferDims.
    std::set<PrimaryDimTypes> transferDims = currDsc->getNonBroadcastLdsDimSet(
        transferNode->srcLdsAndLoopOffsets_.myLdsIdx_);
    std::vector<PrimaryDimTypes> transferWindowDims;
    auto paddingPerdim = getPaddingPerDim(transferNode, /*forSrc*/ true);
    for (auto trDim : transferDims) {
      if (paddingPerdim.getPadding(trDim) != PadType::NOPAD &&
          padInfoMap.count(trDim) &&
          padInfoMap.at(trDim).windowDim_ != PrimaryDimTypesCount) {
        transferWindowDims.push_back(padInfoMap.at(trDim).windowDim_);
      }
    }
    transferDims.insert(transferWindowDims.begin(), transferWindowDims.end());

    std::unordered_set<const dsc2::LoopNode *> producerLoops;
    if (!metadata.dataConnects_.count(
            transferNode->srcLdsAndLoopOffsets_.dataConnect_)) {
      std::cerr << transferNode << std::endl;
      DT_ERROR("Missing source data connect metadata for transfer");
    }
    producerLoops = metadata.dataConnects_
                        .at(transferNode->srcLdsAndLoopOffsets_.dataConnect_)
                        .getProducerLoops();

    // Traverse up the chain of parents to find a suitable location for moving
    // the transferNode to.
    auto transferOwnerLoop = transferNode->getOwnerLoop();
    std::set<const dsc2::LoopNode *> loopsLinkedWithConditions;
    std::unordered_set<const dsc2::ScheduleNode *> excludeList;
    std::set<const dsc2::LoopNode *> externalAllocOwnerLoops;
    for (int i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e;
         ++i) {
      const dsc2::AllocateNode *dstAllocNode = currDsc->getAllocation(
          transferNode->dstLdsAndLoopOffsets_.at(i),
          transferNode->dstVias_.at(i).loc_.storage_, true);
      if (dstAllocNode && isExternalNode(dstAllocNode)) {
        externalAllocOwnerLoops.insert(dstAllocNode->getOwnerLoop());
      }
    }
    for (dsc2::ScheduleNode *currParent = transferNode->getMutableParent();
         currParent; currParent = currParent->getMutableParent()) {
      if (currParent->nodeType_ == dsc2::ScheduleNode::CONDITION) {
        collectLoopReferences(currParent, loopsLinkedWithConditions);
        continue;
      }

      if (currParent->nodeType_ != dsc2::ScheduleNode::LOOP) {
        continue;
      }

      // Process the loop node.
      dsc2::LoopNode *loop = static_cast<dsc2::LoopNode *>(currParent);

      // The transfer can not cross the loop in any of the following cases.
      //   - The loop contains the producer of the transfer source.
      //   - The loop is a parametric loop.
      //   - Some loop-condition, enclosing the transferNode, refers to this
      //     loop.
      bool canNotCrossLoop = producerLoops.count(loop) ||
                             loop->isParametricLoop() ||
                             loopsLinkedWithConditions.count(loop) ||
                             externalAllocOwnerLoops.count(loop);
      if (loop == transferOwnerLoop && canNotCrossLoop) {
        break;
      }

      if (!canNotCrossLoop) {
        // Check if the current loop has any SYNC that is related to the
        // transfer.
        for (auto baseSyncNode : currDsc->scheduleTree_.traverseTreeDFSMutable(
                 loop, {dsc2::ScheduleNode::SYNC}, SenComponents::ALL, -1, -1,
                 -1, excludeList)) {
          // TO DO: Do we need to check at the granularity of core-corelet?
          const dsc2::SyncNode *syncNode =
              static_cast<dsc2::SyncNode *>(baseSyncNode);
          for (auto syncUnit : syncNode->units_) {
            if (is_any_of(syncUnit, relatedComponents)) {
              canNotCrossLoop = true;
            }
          }
        }
      }

      std::vector<PrimaryDimAndKind> dimOverlap;
      for (auto loopDim : loop->dims_) {
        // TO DO: Do we need to consider the metaDimKind as well?
        if (transferDims.count(loopDim.dim_)) {
          dimOverlap.push_back(loopDim);
        }
      }

      if (!dimOverlap.empty() || canNotCrossLoop) {
        // Either, some dimensions in the loop are related to the transfer, or
        // the transferNode can not move up beyond this loop due to other
        // constraints. Consider the loop as a candidate parent location for the
        // transfer node.
        if (loop == transferOwnerLoop &&
            loop->dims_.size() == dimOverlap.size()) {
          // No opportunity to split the transfer owner loop. This transfer
          // node can not be hoisted up.
          break;
        }

        std::string transferNodeDescription =
            transferNode->name_ + " source-data-connect= " +
            transferNode->srcLdsAndLoopOffsets_.dataConnect_ +
            " dstDataConnect(s)= [";
        for (const auto &dst : transferNode->dstLdsAndLoopOffsets_) {
          transferNodeDescription += (" " + dst.dataConnect_);
        }
        transferNodeDescription += " ] ";

        // FIFO results can not be converted to registers in case an opaque
        // operation consumes the FIFO result.
        int fifoInd = transferNode->getNonMemoryResultIndex();
        const dsc2::ComputeNode *fifoConsumer = nullptr;
        if (fifoInd != -1) {
          const auto &fifoConsumers =
              metadata.dataConnects_
                  .at(transferNode->dstLdsAndLoopOffsets_.at(fifoInd)
                          .dataConnect_)
                  .consumers_;
          for (auto &consumer : fifoConsumers) {
            if (consumer->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
              auto computeNode = static_cast<dsc2::ComputeNode *>(consumer);
              if (computeNode->isOpaqueOp_) {
                fifoConsumer = computeNode;
                break;
              }
            }
          }

          if (fifoConsumer) {
            if (transformationReportLevel_ > 0) {
              std::cerr
                  << "\n[Transformation: ReuseTransfer] Candidate transfer "
                  << transferNodeDescription
                  << " was not moved as the FIFO destination is used in opaque "
                     "operation "
                  << fifoConsumer->name_;
            }
            // Continue to the next transfeNode
            break;
          } else if (!convertResultFromFIFOtoReg(transferNode)) {
            if (transformationReportLevel_ > 0) {
              std::cerr
                  << "\n[Transformation: ReuseTransfer] Candidate transfer "
                  << transferNodeDescription
                  << " was not moved as the FIFO result could not be converted "
                     "to register.";
            }
            break;
          }
        }

        dsc2::LoopNode *effectiveParentLoop = loop;
        if (!canNotCrossLoop && (loop->dims_.size() > dimOverlap.size())) {
          // The loop is associated with some dimensions that are not relevant
          // for the transferNode.
          // Split the loop.
          std::vector<std::vector<PrimaryDimAndKind>> listOfDimLists;
          listOfDimLists.push_back(dimOverlap);
          effectiveParentLoop = splitLoopBandOnDim(loop, listOfDimLists, true)
                                    ->getMutableOwnerLoop();
        }

        if (transformationReportLevel_ > 0) {
          std::cerr << "\n[Transformation: ReuseTransfer] Moving transfer "
                    << transferNodeDescription << ", to loop " << loop->name_;
          std::cerr << "  OpFuncName = "
                    << EnumsConversion::opFuncsToString.at(
                           currDsc->computeOp_.at(0).opFuncName);
          if (transformationReportLevel_ > 1) {
            std::cerr << "\n  Current owner= "
                      << transferNode->getOwnerLoop()->name_;
            std::cerr << "\n  TransferDims= [";
            for (auto it : transferDims) {
              std::cerr << " " << EnumsConversion::primaryDimToString.at(it);
            }
            std::cerr << "]\n  Related dimensions= [";
            for (auto it : currDsc->getLayoutDims(
                     transferNode->srcLdsAndLoopOffsets_.myLdsIdx_)) {
              std::cerr << " " << EnumsConversion::primaryDimToString.at(it);
            }
            std::cerr << "]\n  Nonbroadcast dimensions= [";
            for (auto it : currDsc->getNonBroadcastLdsDims(
                     transferNode->srcLdsAndLoopOffsets_.myLdsIdx_)) {
              std::cerr << " " << EnumsConversion::primaryDimToString.at(it);
            }
            std::cerr << "]\n";
          }
        }

        // Move the transfer to the current loop.
        moveTransferNode(transferNode, effectiveParentLoop);
        didTransformation = true;
        break;
      }
      // Record current loop so that this can be excluded while scanning for
      // SYNC nodes in the next iteration.
      excludeList.clear();
      excludeList.insert(loop);
    }
  }
  return didTransformation;
}

// ------------------------------------------------------------------------------------------------
// entry 302/382   level 2   scc 254   42 body lines
// unit: e302_unrollSymbolicTransfers
// authority: ddc/ddc_transformation.cpp:1663
// original: bool Ddc::unrollSymbolicTransfers()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e302_unrollSymbolicTransfers()
{
  for (auto *node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::TRANSFER})) {
    if (isExternalNode(node)) {
      // Skip external nodes.
      continue;
    }
    auto transferNode = static_cast<dsc2::TransferNode *>(node);
    const dsc2::AllocateNode *refTensorAllocNode = nullptr;
    for (auto &di : transferNode->dstLdsAndLoopOffsets_) {
      if (di.myLdsIdx_ >= 0) {
        for (auto &memorg : currDsc->labeledDs_.at(di.myLdsIdx_).memOrg_) {
          if ((refTensorAllocNode = memorg.second.allocateNode_)) break;
        }
        if (refTensorAllocNode) break;
      }
    }

    if (!refTensorAllocNode) {
      int ldsIdx = transferNode->srcLdsAndLoopOffsets_.myLdsIdx_;
      if (ldsIdx >= 0) {
        for (auto &memorg : currDsc->labeledDs_.at(ldsIdx).memOrg_) {
          if ((refTensorAllocNode = memorg.second.allocateNode_)) break;
        }
      } else {
        // Most likely a transfer of constant.
        continue;
      }
    }
    if (!refTensorAllocNode) continue;  // fifo to fifo transfer

    auto sizeDatastage =
        currDsc->getSizeDataStageForNode(transferNode, refTensorAllocNode);
    if (!unrollTransferForSymbolicDims(transferNode,
                                       sizeDatastage.ss_.symbolicDimInfo_)) {
      DT_ERROR(
          "Unable to unroll transfer with symbolic dimensions, but that is "
          "needed for functionality");
    }
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 303/382   level 2   scc 255   24 body lines
// unit: e303_unrollSpreadTransfers
// authority: ddc/ddc_transformation.cpp:1706
// original: bool Ddc::unrollSpreadTransfers()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e303_unrollSpreadTransfers()
{
  // If the transfer is intracting with the allocate with non empty stickspread
  // then fully unroll the transfer
  for (auto *node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::TRANSFER})) {
    auto transferNode = static_cast<dsc2::TransferNode *>(node);
    auto *alloc_src = currDsc->getAllocation(
        transferNode->srcLdsAndLoopOffsets_, transferNode->src_.storage_,
        /* allowMissingAlloc */ true);
    bool alloc_src_has_gapStickSpread =
        alloc_src && !alloc_src->gapStickSpread_.empty();
    bool alloc_dst_has_gapStickSpread = false;
    for (int i = 0; i < transferNode->dstVias_.size(); i++) {
      auto *alloc =
          currDsc->getAllocation(transferNode->dstLdsAndLoopOffsets_.at(i),
                                 transferNode->dstVias_.at(i).loc_.storage_,
                                 /* allowMissingAlloc */ true);
      alloc_dst_has_gapStickSpread |= alloc && !alloc->gapStickSpread_.empty();
    }
    if (alloc_src_has_gapStickSpread || alloc_dst_has_gapStickSpread)
      unrollTransfer(transferNode);
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 304/382   level 2   scc 243   113 body lines
// unit: e304_cloneForPeSfpWorkSplit
// authority: ddc/ddc_transformation_util.cpp:1538
// original: dsc2::TransferNode *Ddc::cloneForPeSfpWorkSplit(dsc2::TransferNode *node)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
dsc2::TransferNode *e304_cloneForPeSfpWorkSplit(dsc2::TransferNode *node)
{
  bool performCloning = false;
  SenComponents existingComp = SenComponents::NO_COMPONENT;
  std::vector<SenComponents> comps = {SenComponents::PE, SenComponents::SFP};
  if (is_any_of(node->src_.unit_, comps)) {
    performCloning = true;
    existingComp = node->src_.unit_;
  }
  if (!performCloning) {
    for (int i = 0, e = node->dstVias_.size(); i < e; ++i) {
      if (!node->dstVias_.at(i).via_.empty()) {
        // A transfer to PE via other components can not be replicated for SFP
        // and vice versa.
        if (transformationReportLevel_ > 0) {
          DT_ERROR("Can not clone transferNode " + node->name_ +
                   " as dstVias.via(" + std::to_string(i) + ") is not empty");
        }
      }
      if (is_any_of(node->dstVias_.at(i).loc_.unit_, comps)) {
        performCloning = true;
        existingComp = node->dstVias_.at(i).loc_.unit_;
      }
    }
    if (performCloning && node->dstVias_.size() > 1) {
      DT_ERROR("Can not clone transferNode " + node->name_ +
               " as the transfer writes to multiple destinations.");
    }
  }

  if (!performCloning) {
    if (transformationReportLevel_ > 0) {
      std::cerr << "\nNo need to clone transferNode " << node->name_
                << " as the transfer does not involve PE or SFP.";
    }
    return nullptr;
  }

  if (metadata.nodeCloningMap_.count(node)) {
    DT_ERROR("[PeSfpWorkSplit] transferNode " + node->name_ +
             " has already been cloned.");
  }

  if (!unrollTransfer(node)) {
    DT_ERROR("[PeSfpWorkSplit] transferNode " + node->name_ +
             " could not be unrolled.");
  }

  dsc2::TransferNode *newNode =
      static_cast<dsc2::TransferNode *>(node->clone());
  std::string suffix =
      (existingComp == SenComponents::PE) ? "_sfp_parallel" : "_pe_parallel";
  newNode->name_ += suffix;

  if (toggleMap.count(newNode->src_.unit_)) {
    newNode->src_.unit_ = toggleMap.at(newNode->src_.unit_);
    if (!newNode->srcLdsAndLoopOffsets_.dataConnect_.empty()) {
      newNode->srcLdsAndLoopOffsets_.dataConnect_ += suffix;
    }
  }

  if (toggleMap.count(newNode->src_.storage_)) {
    newNode->src_.storage_ = toggleMap.at(newNode->src_.storage_);
  }

  // for (auto& dstVia : newNode->dstVias_) {
  for (int i = 0, e = newNode->dstVias_.size(); i < e; ++i) {
    if (toggleMap.count(newNode->dstVias_.at(i).loc_.unit_)) {
      newNode->dstVias_.at(i).loc_.unit_ =
          toggleMap.at(newNode->dstVias_.at(i).loc_.unit_);
      if (!newNode->dstLdsAndLoopOffsets_.at(i).dataConnect_.empty()) {
        newNode->dstLdsAndLoopOffsets_.at(i).dataConnect_ += suffix;
      }
    }
    if (toggleMap.count(newNode->dstVias_.at(i).loc_.storage_)) {
      newNode->dstVias_.at(i).loc_.storage_ =
          toggleMap.at(newNode->dstVias_.at(i).loc_.storage_);
    }
  }

  // Update allocation tracking info.
  dsc2::AllocateNode *allocNode = nullptr;
  if ((allocNode = currDsc->getMutableAllocation(
           newNode->srcLdsAndLoopOffsets_, newNode->src_.storage_, true))) {
    allocNode->addAllocUser(newNode);
  }

  for (int i = 0, e = newNode->dstLdsAndLoopOffsets_.size(); i < e; ++i) {
    if ((allocNode = currDsc->getMutableAllocation(
             newNode->dstLdsAndLoopOffsets_.at(i),
             newNode->dstVias_.at(i).loc_.storage_, true))) {
      allocNode->addAllocUser(newNode);
    }
  }

  // Determine if constant offset is needed for load/store from/to LX.
  if (!is_any_of(newNode->src_.unit_, SenComponents::PE, SenComponents::SFP) &&
      is_any_of(newNode->src_.storage_, dsc2::memories)) {
    metadata.datatransfers_[newNode].apply_pe_sfp_split_offset_src_ = true;
  }
  for (int i = 0, e = newNode->dstLdsAndLoopOffsets_.size(); i < e; ++i) {
    if (!is_any_of(newNode->dstVias_.at(i).loc_.unit_, SenComponents::PE,
                   SenComponents::SFP) &&
        is_any_of(newNode->dstVias_.at(i).loc_.storage_, dsc2::memories)) {
      metadata.datatransfers_[newNode]
          .apply_pe_sfp_split_offset_dest_.push_back(i);
    }
  }

  // Add the new node to the schedule tree immediately after node.
  node->getMutableParent()->addChildNode(newNode, false, node);
  metadata.nodeCloningMap_[node].push_back(newNode);
  return newNode;
}

// ------------------------------------------------------------------------------------------------
// entry 305/382   level 2   scc 225   10 body lines
// unit: e305_destRelatedToExternalNodes
// authority: ddc/ddc_transformation_util.cpp:1700
// original: bool Ddc::destRelatedToExternalNodes( const dsc2::TransferNode *transferNode) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e305_destRelatedToExternalNodes(
    const dsc2::TransferNode *transferNode) const
{
  const dsc2::AllocateNode *allocNode;
  for (size_t dstIndex = 0, e = transferNode->dstLdsAndLoopOffsets_.size();
       dstIndex < e; ++dstIndex) {
    if (destRelatedToExternalNodes(transferNode, dstIndex)) {
      return true;
    }
  }
  return false;
}

// ------------------------------------------------------------------------------------------------
// entry 306/382   level 2   scc 299   4 body lines
// unit: e306_relatedToExternalNodes
// authority: ddc/ddc_transformation_util.cpp:1743
// original: bool Ddc::relatedToExternalNodes(const dsc2::ComputeNode *computeNode) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e306_relatedToExternalNodes(const dsc2::ComputeNode *computeNode) const
{
  return inputRelatedToExternalNodes(computeNode) ||
         outputRelatedToExternalNodes(computeNode);
}

// ------------------------------------------------------------------------------------------------
// entry 307/382   level 2   scc 306   1126 body lines
// unit: e307_exploreAssignDataStages
// authority: ddc/ddcv1.cpp:555
// original: void Ddc::exploreAssignDataStages()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// NOTE: REUSE, DO NOT RE-IMPLEMENT: this body CONTAINS the `checkConstraints` lambda at
//       ddc/ddcv1.cpp:792 and its inner `checkConstraintsImpl` at :825, both ALREADY PORTED on
//       bridge1-campaign as `e001_checkConstraints` in
//       crates/compiler/deeptools/src/bridges/superdsc_to_dataflow_ir/shape_constraints.rs:439 (the
//       impl at that file's :518). CALL THOSE. A second copy of the constraint check is the
//       diverging-implementation failure this campaign's exclusion list exists to prevent.
// ------------------------------------------------------------------------------------------------
void e307_exploreAssignDataStages()
{
  dataStageExplorationDone_ = true;
  // collect all dims relevant to this dsc, in layout order
  std::vector<PrimaryDimTypes> relevantDims;
  {
    std::vector<std::vector<PrimaryDimTypes>> relevantDimsPerLds;
    for (const auto& lds : currDsc->labeledDs_) {
      relevantDimsPerLds.push_back(currDsc->getLayoutDims(lds.ldsIdx_));
    }

    for (int i = 0; i < PrimaryDimTypesCount - 2; i++) {
      for (const auto& ldsLayout : relevantDimsPerLds) {
        if (i >= ldsLayout.size()) continue;
        auto dim = ldsLayout[i];
        if (std::find(relevantDims.begin(), relevantDims.end(), dim) !=
            relevantDims.end())
          continue;
        relevantDims.push_back(dim);
      }
    }
  }
  const auto& coreDs = currDsc->dataStageParam_.at(metadata.core_dstgid);
  const auto& chunkDs = currDsc->dataStageParam_.at(metadata.chunk_dstgid);
  // consider window dims as relevant dims even if they do not appear in any
  // layout, because they can influence padded data structure sizes (pooling)
  for (auto& [dim, padInfo] : coreDs.ss_.paddingSizes_) {
    if (!is_any_of(dim, relevantDims)) continue;
    if (padInfo.windowDim_ != PrimaryDimTypesCount &&
        !is_any_of(padInfo.windowDim_, relevantDims))
      relevantDims.insert(
          std::find(relevantDims.begin(), relevantDims.end(), dim),
          padInfo.windowDim_);
  }
  std::unordered_map<PrimaryDimTypes, int> dimPositions;
  for (int i = 0; i < relevantDims.size(); i++) {
    dimPositions.emplace(relevantDims[i], i);
  }

  // collect all loops from schedule tree
  std::vector<dsc2::LoopNode*> allLoops;
  for (auto& node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::LOOP})) {
    auto* loop = static_cast<dsc2::LoopNode*>(node);
    allLoops.push_back(loop);
    if (loop->isParametricLoop() || metadata.externalNodes_.count(loop))
      continue;
    std::stable_sort(
        loop->dims_.begin(), loop->dims_.end(),
        [&dimPositions](PrimaryDimAndKind left, PrimaryDimAndKind right) {
          return dimPositions[left.dim_] < dimPositions[right.dim_];
        });
    // add dims to relevant dims for datastage, will use in exploration
    auto dsMetaIt = metadata.datastages_.find(loop->denId_);
    DT_CHECK(dsMetaIt != metadata.datastages_.end());
    dsMetaIt->second.nearestNumeratorIdx_ = loop->numId_;
    auto& constraints = dsMetaIt->second.constraints_[loop->numId_];
    for (const auto& dim : loop->dims_) {
      if (!is_any_of(dim.kind_, MetaDimKind::Unpadded, MetaDimKind::Padded,
                     MetaDimKind::WindowDim))
        continue;
      dsMetaIt->second.relevantDimsAndNumerator_.emplace(dim.dim_,
                                                         loop->numId_);
      auto& constraint = constraints[{dim.dim_}];
      constraint.mustBeMultiple_ = true;
      constraint.loopDimKind_ = dim.kind_;
      constraint.updateMax(1);
      // search for dim in loops above to check or enforce continuity
      std::unordered_set<PrimaryDimTypes> dimsToSearch = {dim.dim_};
      if (dim.kind_ == MetaDimKind::WindowDim) {
        for (const auto& [padDim, padInfo] : coreDs.ss_.paddingSizes_) {
          if (dim.dim_ == padInfo.windowDim_) dimsToSearch.insert(padDim);
        }
      }
      auto* parent = loop->getOwnerLoop();
      while (parent && !dimsToSearch.empty()) {
        for (auto& [pdim, pkind] : parent->dims_) {
          if (dimsToSearch.count(pdim)) {
            if (pdim == dim.dim_) {
              dimsToSearch.clear();
              if (parent->denId_ >= 0 && parent->denId_ != loop->numId_) {
                DT_ERROR(
                    "Continuity of datastage not respected between loop \"" +
                    parent->name_ + "\" and \"" + loop->name_ +
                    "\" for dimension " +
                    EnumsConversion::primaryDimToString.at(pdim));
              }
            } else {  // base dim of window dim
              dimsToSearch.erase(pdim);
              // create dependency between numerator of this loop and
              // denotimator of parent
              if (parent->denId_ >= 0) {
                if (metadata.datastages_.count(parent->denId_)) {
                  auto& pconstraint =
                      metadata.datastages_.at(parent->denId_)
                          .constraints_[loop->numId_][{dim.dim_}];
                  pconstraint.mustBeMultiple_ = true;
                  pconstraint.loopDimKind_ = dim.kind_;
                  pconstraint.updateMin(1);
                } else if (metadata.datastages_.count(loop->numId_)) {
                  auto& pconstraint =
                      metadata.datastages_.at(loop->numId_)
                          .constraints_[parent->denId_][{dim.dim_}];
                  pconstraint.mustBeMultiple_ = true;
                  pconstraint.loopDimKind_ = dim.kind_;
                  pconstraint.updateMax(1);
                }
              }
            }
          }
        }
        parent = parent->getOwnerLoop();
      }
    }
  }

  // avoid allocations with symbolic size in directly addressable memories (like
  // PE LRF) because they would require instruction elimination during
  // correction
  for (auto& node : currDsc->scheduleTree_.traverseTreeDFS(
           nullptr, {dsc2::ScheduleNode::ALLOCATE})) {
    if (metadata.externalNodes_.count(node)) continue;
    auto* an = static_cast<const dsc2::AllocateNode*>(node);
    if (!dsc2::directAddressableMemories.count(an->component_)) continue;
    auto targetDims = currDsc->getNonBroadcastLdsDimSet(an->ldsIdx_);
    for (auto& [dim, padInfo] : coreDs.ss_.paddingSizes_) {
      if (targetDims.count(dim) &&
          an->padding_.getPadding(dim) != PadType::NOPAD &&
          padInfo.windowDim_ != PrimaryDimTypesCount)
        targetDims.insert(padInfo.windowDim_);
    }
    const dsc2::LoopNode* myParentLoop = node->getOwnerLoop();
    auto disallowSymbolic = [&](int dsIdx, PrimaryDimTypes dim) {
      auto dsMetaIt = metadata.datastages_.find(dsIdx);
      if (dsMetaIt == metadata.datastages_.end()) return;  // external ds
      dsMetaIt->second.constraints_[-1][{dim}].cannotBeSymbolic_ = true;
    };
    while (!targetDims.empty()) {
      if (myParentLoop->getPrev() == nullptr) {  // root node
        for (auto& dim : targetDims) {
          disallowSymbolic(myParentLoop->denId_, dim);
        }
        targetDims.clear();
      } else {
        for (auto& [ldim, kind] : myParentLoop->dims_) {
          if (targetDims.count(ldim)) {
            if (!myParentLoop->isParametricLoop()) {
              disallowSymbolic(myParentLoop->denId_, ldim);
            }
            targetDims.erase(ldim);
          }
        }
      }
      myParentLoop = myParentLoop->getOwnerLoop();
    }
  }

  // rank data stages based on their relationship in loops
  std::vector<int> sortedDs;
  if (!allLoops.empty()) {
    sortedDs.push_back(allLoops.at(0)->numId_);

    // sort from smallest to largest
    for (auto& loop : allLoops) {
      if (loop->isParametricLoop())
        // Parametric loops are not associated with any datastage.
        continue;

      auto numDsIt = std::find(sortedDs.begin(), sortedDs.end(), loop->numId_);
      if (numDsIt == sortedDs.end()) {
        sortedDs.push_back(loop->numId_);
        numDsIt = sortedDs.end() - 1;
      }
      auto denDsIt = std::find(sortedDs.begin(), sortedDs.end(), loop->denId_);
      if (denDsIt == sortedDs.end()) {
        sortedDs.insert(numDsIt, loop->denId_);  // insert in front
      } else {
        /// TODO: verify robustness of this method
        if (std::distance(denDsIt, numDsIt) < 0)
          std::iter_swap(numDsIt, denDsIt);
      }
    }
  }

  // swap relative constraints so that there is never a constraint towards a
  // larger non-external constraint, to avoid complicating exploration
  for (auto dsIt = sortedDs.begin(); dsIt < sortedDs.end(); dsIt++) {
    auto dsMetadataIt = metadata.datastages_.find(*dsIt);
    if (dsMetadataIt == metadata.datastages_.end()) continue;  // external
    auto& dsConstraints = dsMetadataIt->second.constraints_;
    for (auto dsConstrIt = dsConstraints.begin(), nextIt = dsConstrIt;
         dsConstrIt != dsConstraints.end(); dsConstrIt = nextIt) {
      nextIt++;
      auto& [refDsId, constraints] = *dsConstrIt;
      if (refDsId >= 0 &&
          std::find(dsIt + 1, sortedDs.end(), refDsId) != sortedDs.end()) {
        auto refDsMetadataIt = metadata.datastages_.find(refDsId);
        if (refDsMetadataIt == metadata.datastages_.end())
          continue;  // external
        // move relative constraint from the smaller to the larger datastage
        auto& refConstraints = refDsMetadataIt->second.constraints_[*dsIt];
        for (auto& [dims, constraint] : constraints) {
          auto& refConstraint = refConstraints[dims];
          // min becomes max and values are reciprocal
          if (constraint.min_)
            refConstraint.updateMax(1.0 / (*constraint.min_));
          if (constraint.max_)
            refConstraint.updateMin(1.0 / (*constraint.max_));
          if (constraint.values_) {
            std::set<float> newValues;
            for (auto& val : *constraint.values_) newValues.insert(1.0 / val);
            refConstraint.updateValues(newValues);
          }
          refConstraint.mustBeMultiple_ |= constraint.mustBeMultiple_;
          if (refConstraint.loopDimKind_ == MetaDimKind::Count) {
            refConstraint.loopDimKind_ = constraint.loopDimKind_;
          } else {
            if (constraint.loopDimKind_ != MetaDimKind::Count &&
                refConstraint.loopDimKind_ != constraint.loopDimKind_) {
              DT_ERROR(
                  "Incompatible loop constraint between datastages during "
                  "swapping");
            }
          }
        }
        nextIt = dsConstraints.erase(dsConstrIt);
      }
    }
  }

  // calculate sizePerDim for stick of each dsType
  std::unordered_map<DsTypes, std::unordered_map<PrimaryDimTypes, int>>
      dsTypeStickSizePerDim;
  for (const auto& dsInfo : currDsc->primaryDsInfo_) {
    dsTypeStickSizePerDim[dsInfo.first] =
        currDsc->getCumulativeStickSizes(dsInfo.first);
  }

  auto checkConstraints =
      [&](const DataStructDims& ds, const Metadata::Datastage& dsMetadata,
          PrimaryDimTypes dimToCheck, bool allowEpilogue) -> bool {
    for (auto& [refDsId, constraints] : dsMetadata.constraints_) {
      const auto* refDs = refDsId < 0
                              ? nullptr
                              : &this->currDsc->dataStageParam_.at(refDsId).ss_;
      for (const auto& dimConstr : constraints) {
        auto& dims = dimConstr.first;
        auto& constraint = dimConstr.second;
        if (!dims.count(dimToCheck)) continue;
        if (constraint.cannotBeSymbolic_) {
          DT_CHECK(refDs == nullptr);  // only absolute constraint
          for (auto& dim : dims)
            if (ds.symbolicDimInfo_.count(dim)) return false;
        }
        int dimNumCl = 1, dimNumRow = 1;
        bool considerPeSfpSplit = false;
        std::vector<SenComponents> comps = {SenComponents::NO_COMPONENT};
        for (auto& dim : dims) {
          if (ds.coreletSplit_.count(dim)) {
            dimNumCl = dscGlobal.sysDef.numCoreletsPerCore;
          }
          if (ds.rowSplit_.count(dim)) {
            dimNumRow = dscGlobal.sysDef.numPTRows;
          }
          if (ds.peSfpSplit_.count(dim)) {
            considerPeSfpSplit = true;
            comps.clear();
            comps.push_back(SenComponents::PE);
            comps.push_back(SenComponents::SFP);
          }
        }
        auto checkConstraintsImpl = [&](int cl, int row, SenComponents comp) {
          // calculate size
          int size = 1;
          // In case refDs does not have row-interaction, perform the constraint
          // checking for the full dimension-size instead of per-row dimension
          // size.
          if (ds.rowSplit_.empty() || (refDs && refDs->rowSplit_.empty())) {
            // Clear row.
            row = -1;
          }
          for (auto& dim : dims)
            size *= ds.primaryDimToVal_st(dim, comp, row, cl);
          int refSize = 1;
          if (refDs != nullptr) {
            bool foundValidDim = false;
            for (auto& dim : dims) {
              int dimSize = refDs->primaryDimToVal_st(dim, comp, row, cl);
              if (dimSize > 0) {
                foundValidDim = true;
                refSize *= dimSize;
              }
            }
            if (!foundValidDim) return true;
          }
          if (constraint.mustBeMultiple_) {
            if (refDs == nullptr) {
              if (!constraint.min_.has_value()) {
                DT_ERROR("Must-be-multiple constraint but no min set");
              }
              if (fmodf(size, *constraint.min_ * refSize) != 0) return false;
            } else {
              if (!allowEpilogue) {
                if (constraint.loopDimKind_ == MetaDimKind::Count) {
                  DT_ERROR("Datastage no-epilogue constraint without dim kind");
                }
                if (dims.size() > 1) {
                  DT_ERROR(
                      "Cannot check datastage no-epilogue constraint on "
                      "multiple "
                      "dimensions");
                }
                int sizeLoop = size, refSizeLoop = refSize;
                if (constraint.loopDimKind_ == MetaDimKind::Padded) {
                  // calculate size considering padding
                  PrimaryDimTypes dim = *dims.begin();
                  PaddingFormType padding;
                  padding.setPadding(dim, PadType::PADDED_WZEROPAD);
                  sizeLoop = ds.primaryDimToVal_st(dim, comp, row, cl, padding);
                  refSizeLoop =
                      refDs->primaryDimToVal_st(dim, comp, row, cl, padding);
                  if (refSizeLoop <= 0) {
                    DT_ERROR(
                        "Cannot check datastage no-epilogue constraint if "
                        "dim "
                        "not relevant for reference data stage");
                  }
                } else if (!is_any_of(constraint.loopDimKind_,
                                      MetaDimKind::Unpadded,
                                      MetaDimKind::WindowDim)) {
                  DT_ERROR(
                      "Unhandled dim kind in no-epiloge datastage "
                      "constraint");
                }
                if (std::max(sizeLoop, refSizeLoop) %
                        std::min(sizeLoop, refSizeLoop) !=
                    0)
                  return false;
              }
            }
          }
          if (constraint.max_ && size > (*constraint.max_) * refSize) {
            return false;
          }
          if (constraint.min_ && size < (*constraint.min_) * refSize) {
            return false;
          }
          if (constraint.values_ &&
              !constraint.values_->count(float(size) / refSize)) {
            return false;
          }
          return true;
        };
        if (!considerPeSfpSplit) {
          for (int cl = 0; cl < dimNumCl; cl++) {
            for (int row = 0; row < dimNumRow; row++) {
              if (!checkConstraintsImpl(cl, row, NO_COMPONENT)) return false;
            }
          }
        } else {
          for (int cl = 0; cl < dimNumCl; cl++) {
            for (auto comp : comps) {
              if (!checkConstraintsImpl(cl, -1, comp)) return false;
            }
          }
        }
      }
    }
    return true;
  };

  auto propagateUpward = [&](int refDsPos, PrimaryDimTypes dimToPropagate =
                                               PrimaryDimTypesCount) -> bool {
    std::vector<PrimaryDimTypes> dims = {dimToPropagate};
    auto& dimsToCopy =
        dimToPropagate == PrimaryDimTypesCount ? relevantDims : dims;
    auto dsIdx = sortedDs.at(refDsPos);
    auto& refDs = currDsc->dataStageParam_.at(dsIdx);
    // look for the first external datastage
    DataStructDims* externalDs = nullptr;
    for (int j = refDsPos + 1; j < sortedDs.size(); j++) {
      if (metadata.datastages_.count(sortedDs[j])) continue;
      externalDs = &currDsc->dataStageParam_.at(sortedDs[j]).ss_;
      break;
    }
    // propagate
    dsIdx = metadata.datastages_.at(dsIdx).nearestNumeratorIdx_;
    while (metadata.datastages_.count(dsIdx)) {  // stop at external
      auto& ds = currDsc->dataStageParam_.at(dsIdx).ss_;
      const auto& dsMetadata = metadata.datastages_.at(dsIdx);
      for (auto dim : dimsToCopy) {
        const auto refDsDim = refDs.ss_.primaryDimToVal_st(dim);
        auto& dsDim = ds.primaryDimToValHandler_st(dim);
        int upperLimit = externalDs->primaryDimToVal_st(dim);
        dsDim = refDsDim;
        if (metadata.clSplitDims_.count(dim)) {
          ds.coreletSplit_[dim] = refDs.ss_.coreletSplit_[dim];
        }
        if (refDs.ss_.rowSplit_.count(dim)) {
          ds.rowSplit_[dim] = refDs.ss_.rowSplit_.at(dim);
        } else if (ds.rowSplit_.count(dim)) {
          // assuming equal split
          int val = std::ceil(dsDim / dscGlobal.sysDef.numPTRows /
                              (1 + ds.coreletSplit_.count(dim)));
          for (auto& rowClSplit : ds.rowSplit_.at(dim)) {
            rowClSplit.second.assign(dscGlobal.sysDef.numPTRows, val);
          }
          dsDim = val * dscGlobal.sysDef.numPTRows *
                  (1 + ds.coreletSplit_.count(dim));
        } else if (metadata.peSfpSplitDims_.count(dim)) {
          ds.peSfpSplit_[dim] = refDs.ss_.peSfpSplit_.at(dim);
        }

        while (!checkConstraints(ds, dsMetadata, dim, false)) {
          // try incrementing the value and see if it meets constraints
          int increment = 1;
          if (ds.rowSplit_.count(dim)) {
            for (auto& clRows : ds.rowSplit_.at(dim))
              for (auto& row : clRows.second) row += increment;
            increment *= dscGlobal.sysDef.numPTRows;
          } else if (metadata.peSfpSplitDims_.count(dim)) {
            for (auto& [cl, splitInfo] : ds.peSfpSplit_[dim])
              for (auto& [comp, compSize] : splitInfo) compSize += increment;
            increment *= 2;
          }
          if (metadata.clSplitDims_.count(dim)) {
            for (auto& cl : ds.coreletSplit_[dim]) cl += increment;
            increment *= dscGlobal.sysDef.numCoreletsPerCore;
          }
          dsDim += increment;
          if (dsDim > upperLimit) {
            return false;
          }
        }
      }
      ds.compound();
      dsIdx = metadata.datastages_.at(dsIdx).nearestNumeratorIdx_;
    }
    return true;
  };

  // first assign minimum size to all datastages based on transfers and computes
  for (int i = 0; i < sortedDs.size(); i++) {
    auto dsMetadataIt = metadata.datastages_.find(sortedDs[i]);
    if (dsMetadataIt == metadata.datastages_.end()) continue;  // external
    auto& [dsIdx, dsMetadata] = *dsMetadataIt;
    // for (auto& [dsIdx, dsMetadata] : metadata.datastages_) {
    std::unordered_map<PrimaryDimTypes, int> sizePerDim;
    bool rowUnitInteraction = false;
    auto updateSizePerDim =
        [rowDim = metadata.rowSplitDim, numRows = dscGlobal.sysDef.numPTRows,
         &baseMap = sizePerDim, &rowUnitInteraction,
         this](const std::unordered_map<PrimaryDimTypes, int>& otherMap,
               const std::vector<PrimaryDimAndKind>& dimList,
               const std::vector<SenComponents>& units, const int ldsIdx) {
          auto layoutDims = this->currDsc->getLayoutDimSet(ldsIdx);
          for (const auto& [dim, kind] : dimList) {
            bool dimInOtherMap = otherMap.count(dim);
            if (layoutDims.count(dim) == 0) {
              DT_CHECK(!dimInOtherMap);
              continue;
            }

            bool applyRowScaling = false;
            if (dim == rowDim) {
              for (auto& unit : units) {
                if (EnumsConversion::senCompToRowId.count(unit) ||
                    unit == SenComponents::L0SU) {
                  rowUnitInteraction = true;
                  applyRowScaling = (unit != SenComponents::L0SU);
                }
              }
            }

            if (!dimInOtherMap) continue;
            auto size = otherMap.at(dim);
            if (applyRowScaling) size *= numRows;
            auto& baseMapSize = baseMap.emplace(dim, 1).first->second;
            auto max = std::max(baseMapSize, size);
            if (max % std::min(baseMapSize, size) != 0) {
              DT_ERROR(
                  "Operations with granularity that is not multiple of each "
                  "other cannot coexist in the same loop");
            }
            baseMapSize = max;
          }
        };
    for (auto& loop : allLoops) {
      if (loop->denId_ != dsIdx) continue;  // loop not relevant
      for (auto& child : currDsc->scheduleTree_.traverseTreeDFSMutable(
               loop,
               {dsc2::ScheduleNode::TRANSFER, dsc2::ScheduleNode::COMPUTE}, ALL,
               -1, -1)) {
        if (child->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
          auto* tn = static_cast<const dsc2::TransferNode*>(child);
          const auto tnType = tn->getTransferType();
          if (tnType == dsc2::TransferNode::TransferType::CONSTANT_TO_CONSTANT)
            continue;
          std::unordered_map<PrimaryDimTypes, int> unitTimePerDim;
          PrimaryDimTypes stickBroadcastDim = PrimaryDimTypesCount;
          if (tn->srcLdsAndLoopOffsets_.myLdsIdx_ >= 0) {
            const auto& lds =
                currDsc->labeledDs_.at(tn->srcLdsAndLoopOffsets_.myLdsIdx_);
            for (int i = 0; i < lds.scale_.size(); i++) {
              if (lds.scale_[i] == -2) {
                stickBroadcastDim = currDsc->primaryDsInfo_.at(lds.dsType_)
                                        .layoutDimOrder_.at(i);
                break;
              }
            }
          }
          for (auto& dimSize : tn->unitTimeTransferChunkSize_) {
            if (unitTimePerDim.count(dimSize.sizeDim_.dim_)) {
              unitTimePerDim[dimSize.sizeDim_.dim_] *= dimSize.sizeDim_.size_;
            } else {
              unitTimePerDim[dimSize.sizeDim_.dim_] = dimSize.sizeDim_.size_;
              if (dimSize.sizeDim_.dim_ == stickBroadcastDim)
                unitTimePerDim[dimSize.sizeDim_.dim_] *= tn->replicationFactor_;
            }
          }
          // In case the transfer is for mx-scale, multiply by the scale-factor.
          auto ldsIdx = tn->isSrcLabeledDs()
                            ? tn->srcLdsAndLoopOffsets_.myLdsIdx_
                            : tn->dstLdsAndLoopOffsets_.at(0).myLdsIdx_;
          auto& trLds = currDsc->labeledDs_.at(ldsIdx);
          if (trLds.scaledLdsCategory_ ==
              LabeledDsInfo::ScaledLdsCategory::SCALE_TENSOR) {
            if (unitTimePerDim.count(trLds.mxInfo_.dim)) {
              unitTimePerDim.at(trLds.mxInfo_.dim) *= trLds.mxInfo_.blkSize;
            } else {
              // Create a new entry for the scaled dimension.
              unitTimePerDim[trLds.mxInfo_.dim] = trLds.mxInfo_.blkSize;
            }
          }

          updateSizePerDim(unitTimePerDim, loop->dims_,
                           {tn->src_.unit_, tn->dstVias_.at(0).loc_.unit_},
                           ldsIdx);
        } else if (child->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
          auto* compute = static_cast<dsc2::ComputeNode*>(child);
          // for (auto& inputInfo : compute->inputsLdsAndLoopOffsets_) {
          //   if (inputInfo.myLdsIdx_ < 0) continue;
          //   updateSizePerDim(
          //       sizePerDim,
          //       dsTypeStickSizePerDim.at(
          //           currDsc->labeledDs_.at(inputInfo.myLdsIdx_).dsType_));
          // }
          if (compute->isOpaqueOp_) {
            int ldsIdx = metadata.opaqueOps_.at(compute).ldsIdx_;
            updateSizePerDim(dsTypeStickSizePerDim.at(
                                 currDsc->labeledDs_.at(ldsIdx).dsType_),
                             loop->dims_, {compute->exUnit_}, ldsIdx);
          } else {
            for (const auto& outputInfo : compute->outputsLdsAndLoopOffsets_) {
              if (outputInfo.myLdsIdx_ < 0) continue;
              updateSizePerDim(
                  dsTypeStickSizePerDim.at(
                      currDsc->labeledDs_.at(outputInfo.myLdsIdx_).dsType_),
                  loop->dims_, {compute->exUnit_}, outputInfo.myLdsIdx_);
            }
          }
        }
      }
    }
    auto& ds = currDsc->dataStageParam_.at(dsIdx).ss_;
    auto& constraint = dsMetadata.constraints_[-1];
    auto setValueInDs = [&](int value, double& dsDim, PrimaryDimTypes dim) {
      dsDim = value;
      if (rowUnitInteraction && dim == metadata.rowSplitDim) {
        ds.rowSplit_[dim][0].assign(dscGlobal.sysDef.numPTRows, int(dsDim));
        if (dsDim >= 0) dsDim *= dscGlobal.sysDef.numPTRows;
        if (!metadata.clSplitDims_.empty()) {
          for (int cl = 1; cl < dscGlobal.sysDef.numCoreletsPerCore; cl++)
            ds.rowSplit_[dim][cl] = ds.rowSplit_[dim][0];
        }
      } else if (metadata.peSfpSplitDims_.count(dim)) {
        ds.peSfpSplit_[dim][0][SenComponents::PE] = dsDim;
        ds.peSfpSplit_[dim][0][SenComponents::SFP] = dsDim;
        if (dsDim >= 0) dsDim *= 2;
        if (!metadata.clSplitDims_.empty()) {
          for (int cl = 1; cl < dscGlobal.sysDef.numCoreletsPerCore; cl++)
            ds.peSfpSplit_[dim][cl] = ds.peSfpSplit_[dim][0];
        }
      }
      if (metadata.clSplitDims_.count(dim)) {
        ds.coreletSplit_[dim].assign(dscGlobal.sysDef.numCoreletsPerCore,
                                     int(dsDim));
        if (dsDim >= 0) dsDim *= dscGlobal.sysDef.numCoreletsPerCore;
      }
    };
    for (const auto& dim : relevantDims) {
      auto& dsDim = ds.primaryDimToValHandler_st(dim);
      int valToSet = -1;
      if (auto it = sizePerDim.find(dim); it != sizePerDim.end()) {
        valToSet = it->second;
        if (rowUnitInteraction && dim == metadata.rowSplitDim) {
          if (valToSet % dscGlobal.sysDef.numPTRows != 0) {
            DT_ERROR("Size is not multiple of row");
          }
          valToSet /= dscGlobal.sysDef.numPTRows;
        }
      } else if (dsMetadata.relevantDimsAndNumerator_.count(dim)) {
        valToSet = 1;
      }
      if (valToSet > 0) {
        for (const auto& [padDim, padInfo] : coreDs.ss_.paddingSizes_) {
          PrimaryDimTypes dimToInsert;
          if (dim == padDim) {
            dimToInsert = dim;
          } else if (dim == padInfo.windowDim_) {
            dimToInsert = padDim;
          } else {
            continue;
          }
          auto& newPadInfo =
              ds.paddingSizes_.emplace(dimToInsert, padInfo).first->second;
          newPadInfo.unneededPad_ = newPadInfo.unneededPadFront_ =
              newPadInfo.unneededPadBack_ = 0;
          newPadInfo.padBack_ = newPadInfo.padFront_ = -1;
          break;
        }
      }
      setValueInDs(valToSet, dsDim, dim);
    }
    // give a value to kernel whenever there is a padded dim
    for (const auto& [padDim, padInfo] : ds.paddingSizes_) {
      if (padInfo.windowDim_ == PrimaryDimTypesCount) continue;
      auto& wDim = ds.primaryDimToValHandler_st(padInfo.windowDim_);
      if (wDim < 0) setValueInDs(1, wDim, padInfo.windowDim_);
    }
    // check constraint in a separate loop to handle constraint on multiple dims
    for (const auto& dim : relevantDims) {
      auto& dsDim = ds.primaryDimToValHandler_st(dim);
      int origValue = dsDim;
      int factor = 1;
      int upperLimit = chunkDs.ss_.primaryDimToVal_st(dim, NO_COMPONENT, -1, -1,
                                                      {}, 1.0, true);
      if (!checkConstraints(ds, dsMetadata, dim, false)) {
        setValueInDs(1, dsDim, dim);  // drop to 1 and explore from there
      }
      while (!checkConstraints(ds, dsMetadata, dim, false)) {
        if (dsDim == origValue) {
          factor = 1;  // once above slice/stick, go in multiples
        }
        factor++;
        // try incrementing the value and see if it meets constraints
        if (ds.rowSplit_.count(dim)) {
          for (auto& clRows : ds.rowSplit_.at(dim))
            for (auto& row : clRows.second) row = (row / (factor - 1)) * factor;
        } else if (metadata.peSfpSplitDims_.count(dim)) {
          for (auto& clEntry : ds.peSfpSplit_.at(dim))
            for (auto& [comp, splitSize] : clEntry.second) {
              splitSize = (splitSize / (factor - 1)) * factor;
            }
        }
        if (metadata.clSplitDims_.count(dim)) {
          for (auto& cl : ds.coreletSplit_.at(dim))
            cl = (cl / (factor - 1)) * factor;
        }
        dsDim = (dsDim / (factor - 1)) * factor;
        if (dsDim > upperLimit) {
          DT_ERROR("Cannot find a valid minimum value for data stage " +
                   std::to_string(dsIdx));
        }
      }
    }
    ds.compound();
    for (const auto& [dim, size] : sizePerDim) {
      auto& dimConstraint = constraint[{dim}];
      // Assumption: Equal split across corelet, PT-rows, and PE-SFP.
      dimConstraint.updateMin(
          ds.primaryDimToVal_st(dim, SenComponents::PE, 0, 0));
      dimConstraint.mustBeMultiple_ = true;
    }
  }

  auto calculateEpilogues = [&]() -> bool {
    for (int i = sortedDs.size() - 1; i >= 0; i--) {
      auto metadataIt = metadata.datastages_.find(sortedDs[i]);
      if (metadataIt == metadata.datastages_.end()) continue;  // external
      const auto& dsMetadata = metadataIt->second;
      auto& denDs = currDsc->dataStageParam_.at(sortedDs[i]);
      denDs.el_ = denDs.ss_;
      denDs.el_.name_ += "el";
      for (const auto& [dim, numIdx] : dsMetadata.relevantDimsAndNumerator_) {
        if (denDs.ss_.symbolicDimInfo_.count(dim)) {
          denDs.el_.makeDimSymbolic(coreDs.ss_, dim);
          continue;
        }
        const auto& numDs = currDsc->dataStageParam_.at(numIdx);
        int dimNumCl = denDs.ss_.coreletSplit_.count(dim)
                           ? dscGlobal.sysDef.numCoreletsPerCore
                           : 1;
        int dimNumRows =
            denDs.ss_.rowSplit_.count(dim) ? dscGlobal.sysDef.numPTRows : 1;
        auto& denElTot = denDs.el_.primaryDimToValHandler_st(dim);
        denElTot = 0;
        for (int cl = 0; cl < dimNumCl; cl++) {
          if (!denDs.ss_.peSfpSplit_.count(dim)) {
            for (int row = 0; row < dimNumRows; row++) {
              auto rowcount = row;
              if (denDs.ss_.rowSplit_.count(dim) == 0 &&
                  numDs.ss_.rowSplit_.count(dim) !=
                      denDs.ss_.rowSplit_.count(dim)) {
                rowcount = -1;
              }
              auto numSs = numDs.ss_.primaryDimToVal_st(
                  dim, SenComponents::NO_COMPONENT, rowcount, cl);
              auto numEl = numDs.el_.primaryDimToVal_st(
                  dim, SenComponents::NO_COMPONENT, rowcount, cl);
              auto denEl = denDs.ss_.primaryDimToVal_st(
                  dim, SenComponents::NO_COMPONENT, rowcount, cl);
              if (numSs != numEl && numSs % denEl != 0) {
                return false;
              }
              auto modVal = numEl % denEl;
              if (modVal != 0) denEl = modVal;
              if (dimNumCl > 1) {
                if (row == 0)
                  denDs.el_.coreletSplit_.at(dim).at(cl) = denEl;
                else
                  denDs.el_.coreletSplit_.at(dim).at(cl) += denEl;
              }
              if (dimNumRows > 1) {
                if (dimNumCl > 1)
                  denDs.el_.rowSplit_.at(dim).at(cl).at(row) = denEl;
                else
                  for (auto& clRows : denDs.el_.rowSplit_.at(dim))
                    clRows.second.at(row) = denEl;
              }
              denElTot += denEl;
            }
          } else {
            // To Verify: peSfpWorkSplit
            std::vector<SenComponents> comps = {SenComponents::PE,
                                                SenComponents::SFP};
            std::map<SenComponents, int> numSs;
            std::map<SenComponents, int> numEl;
            std::map<SenComponents, int> denEl;
            for (auto comp : comps) {
              numSs[comp] = numDs.ss_.primaryDimToVal_st(dim, comp, -1, cl);
              numEl[comp] = numDs.el_.primaryDimToVal_st(dim, comp, -1, cl);
              denEl[comp] = denDs.ss_.primaryDimToVal_st(dim, comp, -1, cl);
              if (numSs.at(comp) != numEl.at(comp) &&
                  numSs.at(comp) % denEl.at(comp) != 0) {
                return false;
              }
              auto modVal = numEl.at(comp) % denEl.at(comp);
              if (modVal != 0) denEl.at(comp) = modVal;
            }

            if (dimNumCl > 1) {
              denDs.el_.coreletSplit_.at(dim).at(cl) = 0;
              for (auto comp : comps) {
                denDs.el_.coreletSplit_.at(dim).at(cl) += denEl.at(comp);
              }
            }

            for (auto comp : comps) {
              if (denDs.ss_.peSfpSplit_.count(dim)) {
                // To Verify: peSfpWorkSplit
                if (dimNumCl > 1)
                  denDs.el_.peSfpSplit_.at(dim).at(cl).at(comp) =
                      denEl.at(comp);
                else
                  for (auto& clEntry : denDs.el_.peSfpSplit_.at(dim))
                    clEntry.second.at(comp) = denEl.at(comp);
              }
              denElTot += denEl.at(comp);
            }
          }
        }
      }
      denDs.el_.compound();
    }
    return true;
  };

  if (!calculateEpilogues()) {
    DT_ERROR("Epilogue of epilogue needed in setting minimum datastage sizes");
  }
  if (!allocAllMem(false)) {
    DT_ERROR("Cannot allocate even the smallest size");
  }

  // explore data stage parameters to maximize
  for (int i = 0; i < sortedDs.size(); i++) {
    auto dsMetadataIt = metadata.datastages_.find(sortedDs[i]);
    if (dsMetadataIt == metadata.datastages_.end()) continue;  // external
    const auto& dsMetadata = dsMetadataIt->second;
    if (dsMetadata.strategyMinimize_) continue;
    // maximize this data stage, one dimension at a time
    auto& ds = currDsc->dataStageParam_.at(sortedDs[i]).ss_;
    auto exploreMaximizeDs = [&](bool allowEpilogue) {
      for (const auto& dim : relevantDims) {
        if (!dsMetadata.relevantDimsAndNumerator_.count(dim))
          continue;  // dim not relevant for datastage
        auto& dsDim = ds.primaryDimToValHandler_st(dim);
        const int origVal = dsDim;
        const bool origSymbolic = ds.symbolicDimInfo_.count(dim);
        // find upper limit
        for (int j = i + 1; j < sortedDs.size(); j++) {
          // look for the first external datastage
          if (metadata.datastages_.count(sortedDs[j])) continue;
          auto& refDs = currDsc->dataStageParam_.at(sortedDs[j]).ss_;
          dsDim = refDs.primaryDimToVal_st(dim);
          if (refDs.symbolicDimInfo_.count(dim)) {
            ds.symbolicDimInfo_.insert_or_assign(
                dim, refDs.symbolicDimInfo_.at(dim));
          }
          if (metadata.clSplitDims_.count(dim))
            ds.coreletSplit_[dim] = refDs.coreletSplit_[dim];
          if (ds.rowSplit_.count(dim)) {
            ds.rowSplit_[dim] = refDs.rowSplit_.at(dim);
          } else if (metadata.peSfpSplitDims_.count(dim)) {
            ds.peSfpSplit_[dim] = refDs.peSfpSplit_.at(dim);
          }
          break;
        }
        if (origVal == dsDim && origSymbolic == ds.symbolicDimInfo_.count(dim))
          continue;  // dim already maximized
        auto isValidAssignment = [&]() {
          ds.compound();
          if (!checkConstraints(ds, dsMetadata, dim, allowEpilogue))
            return false;
          if (!propagateUpward(i, dim)) return false;
          if (!calculateEpilogues()) return false;
          return allocAllMem(false);
        };
        while (!isValidAssignment()) {
          if (ds.symbolicDimInfo_.count(dim)) {
            ds.makeDimNotSymbolic(dim);
            continue;
          }
          // try decrementing the value and see if it fits
          int decrement = 1;
          if (ds.rowSplit_.count(dim)) {
            for (auto& clRows : ds.rowSplit_.at(dim))
              for (auto& row : clRows.second) row -= decrement;
            decrement *= dscGlobal.sysDef.numPTRows;
          } else if (metadata.peSfpSplitDims_.count(dim)) {
            for (auto& [cl, splitInfo] : ds.peSfpSplit_.at(dim))
              for (auto& [comp, splitSize] : splitInfo) splitSize -= decrement;
            decrement *= 2;
          }
          if (metadata.clSplitDims_.count(dim)) {
            for (auto& cl : ds.coreletSplit_[dim]) cl -= decrement;
            decrement *= dscGlobal.sysDef.numCoreletsPerCore;
          }
          dsDim -= decrement;
          if (dsDim < origVal) {
            DT_ERROR("Impossible to find a suitable value for " +
                     EnumsConversion::primaryDimToString.at(dim) +
                     " in datastage " + std::to_string(sortedDs[i]));
          }
        }
      }
    };
    // first maximize all dims without considering epilogues
    exploreMaximizeDs(false);
    // then repeat allowing them
    /// TEMP: disallow epilogue exploration if we have opaque ops because
    /// codegen for this scenario is not fully supported
    if (metadata.opaqueOps_.empty())
      if (dsMetadata.allowEpilogue_) exploreMaximizeDs(true);
  }

  for (auto& [dsIdx, ds] : currDsc->dataStageParam_) {
    auto dsMetadataIt = metadata.datastages_.find(dsIdx);
    if (dsMetadataIt == metadata.datastages_.end()) continue;  // external
    ds.ss_.pruneMaxSymbolicVolumes(coreDs.ss_);
    ds.el_.pruneMaxSymbolicVolumes(coreDs.ss_);
  }

  if (!allocAllMem(true)) {
    DT_ERROR("It should have been possible to allocate");
  }

  for (auto& node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::TRANSFER})) {
    auto* tn = static_cast<dsc2::TransferNode*>(node);
    if (metadata.externalNodes_.count(node) > 0) continue;
    if (is_any_of(tn->src_.unit_, NO_COMPONENT, CONSTANT)) continue;
    if (dsc2::memories.count(tn->src_.storage_) == 0 &&
        dsc2::memories.count(tn->dstVias_.at(0).loc_.storage_) == 0)
      continue;
    if (tn->srcLdsAndLoopOffsets_.myLdsIdx_ == -1 &&
        tn->srcLdsAndLoopOffsets_.constantId_ != -1) {
      continue;
    }
    // reduce unit time transfers to fit data stage if unitTimeTransfer >
    // its block transfer size
    // unit time transfer is only applicable to src unit.
    // TODO: expand this to cover dst units.
    std::unordered_map<PrimaryDimTypes, int> dsSizePerDim =
        currDsc->getBlockTransferSizePerDim(*tn, tn->src_.unit_, 0, false,
                                            false, true);
    auto& srcLds = currDsc->labeledDs_.at(tn->srcLdsAndLoopOffsets_.myLdsIdx_);
    bool doSplat =
        (EnumsConversion::senCompToGenericComp.at(tn->src_.unit_) == LXLU ||
         (EnumsConversion::senCompToGenericComp.at(tn->src_.unit_) == L0LU &&
          srcLds.scaledLdsCategory_ ==
              LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR &&
          is_any_of(srcLds.dataFormat_, DataFormats::SEN169_FP16,
                    DataFormats::SEN143_FP8)));

    // do not perform shrinking if we have already manipulated the transfer to
    // achieve 16b splat
    if (doSplat &&
        currDsc->labeledDs_.at(tn->srcLdsAndLoopOffsets_.myLdsIdx_)
                .dataFormat_ == DataFormats::IEEE_FP32 &&
        tn->replicationFactor_ == 8)
      continue;

    bool isAllowedUnit =
        is_any_of(EnumsConversion::senCompToGenericComp.at(tn->src_.unit_),
                  LXLU, L0LU) ||
        tn->dstVias_.back().loc_.unit_ == LXSU;
    if (!isAllowedUnit) continue;

    bool disableSmallerByteStoresLXSU =
        dscGlobal.sysDef.coreArch < RCUDD1A_ISA &&
        tn->dstVias_.back().loc_.unit_ == LXSU &&
        tn->dstVias_.back().loc_.storage_ == LX;

    // Disabling smaller Byte stores in LXSU
    // For now, we are restricting to single dimension in
    // unitTimeTransferChunkSize_
    // TODO: can be relaxed later.
    disableSmallerByteStoresLXSU &= tn->unitTimeTransferChunkSize_.size() == 1;
    // No longer supporting this
    bool isChunkStridedLoad = false;

    const auto& lds =
        currDsc->labeledDs_.at(tn->srcLdsAndLoopOffsets_.myLdsIdx_);

    bool reduced = false;
    std::unordered_set<PrimaryDimTypes> maximizedDims;
    bool reduce2B = true;
    // reduce unit transfer size if it exceeds data stage size
    for (int i = 0; i < tn->unitTimeTransferChunkSize_.size(); i++) {
      auto& [dim, size] = tn->unitTimeTransferChunkSize_[i].sizeDim_;
      auto& dsDim = dsSizePerDim.at(dim);

      if (reduced && dsDim != 1 && !isChunkStridedLoad) {
        DT_ERROR("Datastage size requiring unit time transfer with holes");
      }
      auto dimScale =
          lds.scale_.at(currDsc->getDimIndexInLayoutOrder(lds.dsType_, dim));
      if (dsDim > 1 &&
          (dimScale > 0 ||
           (dimScale == -2 && tn->dstVias_.back().loc_.unit_ == LXSU)))
        reduce2B = false;
      if (isChunkStridedLoad && dim != IN) {
        // chunkstrided load is to fetch inputs to fill subsimds at IN dim.
        // reduce non-IN dim sizes to one.
        size = 1;
        continue;
      }

      // Don't shrink if it is meant for single element stores in LXSU for DD1.
      // E.g., 2B stores in MAX/SUM/EXX2 for fp16.
      // TODO: single element thing can be relaxed later.
      bool disableShrinking = disableSmallerByteStoresLXSU && dsDim == 1;
      if (disableShrinking && verbose_ > 0)
        std::cout << "Disabling single element stores for " << currDsc->name_
                  << std::endl;

      // if unit transfer size > data stage dim size, shrink unit transfer size
      // to fit in dsDim
      /*
      for (auto stickDim :
           currDsc->primaryDsInfo_.at(lds.dsType_).stickDimOrder_) {
        if (lds.scale_.at(
                currDsc->getDimIndexInLayoutOrder(lds.dsType_, stickDim)) == -2)
          continue;
        splat2B = false;
        disableShrinking = true;
      }
      */
      if (dsDim < size && !disableShrinking) {
        int dsDimPad = coreDs.ss_.primaryDimToVal_st(
            dim, SenComponents::NO_COMPONENT, -1, -1,
            {dim, coreDs.ss_.paddingSizes_.count(dim)
                      ? PadType::PADDED_FULLSPAN_WUNNEEDED
                      : PadType::NOPAD});
        if (i != tn->unitTimeTransferChunkSize_.size() - 1 &&
            coreDs.ss_.primaryDimToVal_st(dim) == dsDim && size == dsDimPad) {
          if (!maximizedDims.insert(dim).second) {
            DT_ERROR("Cannot maximize a dimension more than once");
          }
        } else {
          if (size % int(dsDim) != 0) {
            DT_ERROR("Datastage size not divisor of unit time trasfer");
          }
          if (doSplat) {
            tn->replicationFactor_ *= size / dsDim;
          }
          size = dsDim;
          reduced = true;
        }
      }
      if (tn->src_.unit_ == L0LU) DT_CHECK(dsDim % size == 0);
      dsDim /= size;
    }

    auto checkAndResetUnitTimeTransfer = [&](dsc2::TransferNode* tn) {
      int ldsIdx = tn->srcLdsAndLoopOffsets_.myLdsIdx_;
      const auto& lds = currDsc->labeledDs_.at(ldsIdx);
      bool l0SliceOnly =
          EnumsConversion::senCompToGenericComp.at(tn->src_.unit_) == L0LU;
      auto tensorSizes = currDsc->getStickSizes(
          lds.dsType_, false, false, l0SliceOnly, dscGlobal.sysDef.numPTRows);
      auto& uttChunkSize = tn->unitTimeTransferChunkSize_;
      DT_CHECK(uttChunkSize.size() == tensorSizes.size());

      // unitTimeTransferChunkSize_ is derived from stickSizes initially.
      // check unitTimeTransfer complies with subsimd constrain
      // Only IN is mapped on subsimds. Kij are not considered.
      int numIN = 1;
      for (auto sizeIndex : uttChunkSize) {
        if (sizeIndex.sizeDim_.dim_ == IN) numIN *= sizeIndex.sizeDim_.size_;
      }
      const static std::unordered_map<DataFormats, std::string> precToStr = {
          {DataFormats::SEN169_FP16, "fp16"}, {DataFormats::BFLOAT16, "fp16"},
          {DataFormats::SEN143_FP8, "fp8"},   {DataFormats::SENINT8, "int8"},
          {DataFormats::SENINT4, "int4"},
      };
      const auto numSubsimdPerPT =
          dscGlobal.sysDef.numSubSimdPerPT.at(precToStr.at(lds.dataFormat_));
      // DT_CHECK(numIN <= numSubsimdPerPT && numSubsimdPerPT % numIN == 0);
      if (numIN != numSubsimdPerPT) {
        // if not complied with subsimds, add additional IN dims outside of
        // stick
        int sizeIdx = currDsc->getDimIndexInLayoutOrder(
            currDsc->labeledDs_[tn->srcLdsAndLoopOffsets_.myLdsIdx_].dsType_,
            IN);
        sizeIdx += tensorSizes.size();  // include stick dims
        const auto& ds =
            currDsc->dataStageParam_.at(tn->getOwnerLoop()->denId_).ss_;
        auto dimSize = ds.dataStageDimToVal_compView_st(IN, tn->src_.unit_, -1);
        // assume all chunks are fetched within its ownerLoop's tensor space
        DT_CHECK(dimSize >= numSubsimdPerPT / numIN);
        tn->unitTimeTransferChunkSize_.push_back(
            {{IN, numSubsimdPerPT / numIN}, sizeIdx, sizeIdx});
        tensorSizes.push_back({IN, dimSize});
      }

      // find out holes in load and split it to make unitTimetransferChunkSize_
      // contiguous.
      int numHoles = 0;
      bool holeStarts = false;
      int holeEndPos = -1;  // exclusive
      for (int i = 0; i < tensorSizes.size(); i++) {
        DT_CHECK(uttChunkSize[i].sizeDim_.dim_ == tensorSizes[i].first);
        if (uttChunkSize[i].sizeDim_.size_ < tensorSizes[i].second) {
          // stride at this dim
          if (holeStarts && uttChunkSize[i].sizeDim_.size_ > 1) {
            // if a hole already starts and current dim chunk size > 1, start a
            // new hole
            numHoles++;
            holeEndPos = i;
          } else if (!holeStarts) {
            // a new hole starts
            holeStarts = true;
          }
        } else {
          // no stride at this dim
          if (holeStarts) {
            // if hole starts already, hole ends here.
            numHoles++;
            holeEndPos = i;
            holeStarts = false;
          }
        }
        if (numHoles > 1) {
          throw std::runtime_error("More than 1 holes in unitTimeTransfer.");
        }
      }
      tn->unitTimeTransferNumChunks_ = 1;
      if (numHoles == 1) {
        // split unitTimeTransferChunkSize_
        for (int i = holeEndPos; i < uttChunkSize.size(); i++) {
          auto sizeDim = uttChunkSize[i];
          tn->unitTimeTransferNumChunks_ *= sizeDim.sizeDim_.size_;
          sizeDim.sizeDim_.size_ = 1;
          tn->unitTimeTransferChunkStride_.push_back(sizeDim);
        }
        uttChunkSize.erase(uttChunkSize.begin() + holeEndPos,
                           uttChunkSize.end());
      }
    };
    // check whether unitTimeTransferChunkSize is compliant with subsimd
    // requirement. if not, fill in unitTimeTransferNumChunks_ and
    // unitTimeTransferChunkStride_ for chunk-strided load or emit error.
    // if (EnumsConversion::senCompToGenericComp.at(tn->src_.unit_) == L0LU)
    //   checkAndResetUnitTimeTransfer(tn);

    // fill replicationFactor. Splat is required for sen1.5 because PT cols read
    // different 16B from w-fifo
    if (dscGlobal.sysDef.coreArch >= SEN1P5_ISA &&
        srcLds.scaledLdsCategory_ ==
            LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR &&
        EnumsConversion::senCompToGenericComp.at(tn->src_.unit_) == L0LU) {
      auto& dataFormat = srcLds.dataFormat_;
      int loadSize = 1;
      for (auto kv : tn->unitTimeTransferChunkSize_) {
        loadSize *= kv.sizeDim_.size_;
      }
      loadSize *= tn->unitTimeTransferNumChunks_;
      tn->replicationFactor_ =
          dscGlobal.sysDef.l0PtBwPerSlice /
          ((float)EnumsConversion::dataFormatsToBitWidth.at(dataFormat) / 8) /
          loadSize;
    }
    if (reduce2B && !disableSmallerByteStoresLXSU) {
      // reduce all remaining dimensions in unitTimeTransfer
      for (int i = 0; i < tn->unitTimeTransferChunkSize_.size(); i++) {
        auto& [dim, size] = tn->unitTimeTransferChunkSize_[i].sizeDim_;
        if (doSplat) {
          tn->replicationFactor_ *= size;
        }
        size = 1;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 308/382   level 2   scc 313   252 body lines
// unit: e308_prepDsc
// authority: ddc/ddcv1.cpp:2019
// original: void Ddc::prepDsc()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e308_prepDsc()
{
  // Set of actions to prepare the DSC to be used by DDC.
  // Most of these actions should be taken in other tools before DDC, but they
  // are temporarily here until other tools are updated

  currDsc->numCoreletsUsed_DSC2_ =
      currDsc->numCoreletsUsed_;  // do imbalanced corelet split in the future

  bool usePt = currDsc->computeOp_.at(0).exUnit == PT ||
               currDsc->computeOp_.at(0).opFuncName == OpFuncs::ReStickifyOpHBM;
  if (usePt) {
    const auto& stickWithoutSliceLayout = currDsc->getStickSizes(
        currDsc->computeOp_.at(0).inputLabeledDs.at(0)->dsType_, false, true);
    DT_CHECK(stickWithoutSliceLayout.size() == 1);
    metadata.rowSplitDim = stickWithoutSliceLayout.at(0).first;
  }
  metadata.peSfpSplitDims_ = getPeSfpSplitDim(*currDsc, dscGlobal);
  // detect need for core to core reduction and insert in computeOp
  for (int i = 0; i < currDsc->computeOp_.size(); i++) {
    const auto& compOp = currDsc->computeOp_.at(i);
    std::set<PrimaryDimTypes> reducedDims;
    for (const auto& lds : compOp.inputLabeledDs) {
      reducedDims.merge(currDsc->getNonBroadcastLdsDimSet(lds->ldsIdx_));
    }
    for (const auto& lds : compOp.outputLabeledDs) {
      reducedDims = set_diff(reducedDims,
                             currDsc->getNonBroadcastLdsDimSet(lds->ldsIdx_));
    }
    bool reductionOpNeeded = std::any_of(
        reducedDims.begin(), reducedDims.end(), [&](PrimaryDimTypes dim) {
          return sdsc_->numWkSlicesPerDim_.at(dim) > 1;
        });
    if (!reductionOpNeeded) continue;
    DT_CHECK(compOp.outputLabeledDs.size() == 1);
    auto* ldsPtr = compOp.outputLabeledDs.at(0);
    auto& newOp =
        *currDsc->computeOp_.emplace(currDsc->computeOp_.begin() + i + 1);
    // compOp invalidated after this point
    newOp.opFuncName = OpFuncs::GENERIC_PARTIAL_REDUCTION;
    newOp.attributes_.dataFormat_ =
        currDsc->computeOp_.at(i).attributes_.dataFormat_;
    newOp.inputLabeledDs.push_back(ldsPtr);
    newOp.outputLabeledDs.push_back(ldsPtr);
    i++;  // skip newly inserted op
  }
  if (auto& cOp = currDsc->computeOp_.at(0); cOp.opFuncName == OpFuncs::EXX2) {
    for (auto& [id, constinfo] : currDsc->constantInfo_) {
      if (constinfo.name_ == "useZeroMean" &&
          FoldInfraUtils::getSingleDataStrict(constinfo.data_).at(0) == 1) {
        metadata.opFuncBackup_ = cOp.opFuncName;
        cOp.opFuncName = OpFuncs::EXX2_ZEROMEAN;
        break;
      }
    }
    auto constIt = cOp.opConsts.find("useZeroMean");
    if (constIt != cOp.opConsts.end() && constIt->second[0] == 1) {
      metadata.opFuncBackup_ = cOp.opFuncName;
      cOp.opFuncName = OpFuncs::EXX2_ZEROMEAN;
    }
  }

  // finalize external datastages
  auto clearDeprecatedFields = [](DataStructDims& ds) {
    ds.r_ = ds.c_ = ds.rc_ = ds.si_ = ds.sj_ = ds.sij_ = ds.zi_ = ds.zj_ =
        ds.zij_ = -1;
  };
  for (auto& [dsId, ds] : currDsc->dataStageParam_) {
    finalizeExternalDataStage(*currDsc, dsId, metadata.clSplitDims_,
                              dscGlobal.sysDef.numPTRows, usePt,
                              metadata.rowSplitDim, metadata.peSfpSplitDims_);
    clearDeprecatedFields(ds.ss_);
    clearDeprecatedFields(ds.el_);
  }

  for (const auto& [dim, split] :
       currDsc->dataStageParam_.at(metadata.core_dstgid).ss_.coreletSplit_) {
    metadata.clSplitDims_.insert(dim);
  }

  for (auto& computeOp : currDsc->computeOp_) {
    if (computeOp.opFuncName != OpFuncs::ReStickifyOpLx &&
        computeOp.opFuncName != OpFuncs::ReStickifyOpHBM)
      continue;

    int newLdsIdx = addNewLds(computeOp.inputLabeledDs.at(0));
    auto newLds = &currDsc->labeledDs_.at(newLdsIdx);
    computeOp.inputLabeledDs.push_back(newLds);
    newLds->dsName_ += "_internalInput";
    newLds->dsType_ = DsTypes::INTERNAL;

    auto& inputLds = computeOp.inputLabeledDs.at(0);
    auto& outputLds = computeOp.outputLabeledDs.at(0);
    auto& inputDsInfo = currDsc->primaryDsInfo_.at(inputLds->dsType_);
    auto& outputDsInfo = currDsc->primaryDsInfo_.at(outputLds->dsType_);
    auto& internalDsInfo = currDsc->primaryDsInfo_[DsTypes::INTERNAL];

    internalDsInfo.layoutDimOrder_ = inputDsInfo.layoutDimOrder_;
    for (auto dim : outputDsInfo.stickDimOrder_) {
      internalDsInfo.stickDimOrder_.push_back(dim);
      internalDsInfo.stickRepl_.push_back(1);
    }
    for (auto dim : inputDsInfo.stickDimOrder_) {
      internalDsInfo.stickDimOrder_.push_back(dim);
      internalDsInfo.stickRepl_.push_back(1);
    }
    bool first = true;
    for (auto size : outputDsInfo.stickSize_) {
      if (first) size /= 8;
      internalDsInfo.stickSize_.push_back(size);
      first = false;
    }
    first = true;
    for (auto size : inputDsInfo.stickSize_) {
      if (first) size /= 8;
      internalDsInfo.stickSize_.push_back(size);
      first = false;
    }

    newLds->memOrg_[LX] = inputLds->memOrg_.at(LX);
    auto allocInput0 = newLds->memOrg_.at(LX).allocateNode_;
    auto transferInput0 = static_cast<const dsc2::TransferNode*>(
        allocInput0->allocUsers_.begin()->first);
    auto newAlloc = static_cast<dsc2::AllocateNode*>(allocInput0->clone());
    newAlloc->name_ = allocInput0->name_ + "_internalInput";
    newAlloc->allocUsers_.clear();
    newAlloc->ldsIdx_ = newLdsIdx;
    auto newTransfer = new dsc2::TransferNode();
    newTransfer->name_ = "dummy_transfer_to_lx_internalInput";
    newTransfer->src_.unit_ = newTransfer->src_.storage_ = NO_COMPONENT;
    dsc2::TransferNode::DstVia dstVia;
    dstVia.loc_ = {NO_COMPONENT, LX};
    newTransfer->dstVias_.push_back(dstVia);
    dsc2::DataInfo dataInfo;
    dataInfo.myLdsIdx_ = newLdsIdx;
    newTransfer->dstLdsAndLoopOffsets_.push_back(dataInfo);

    newAlloc->addAllocUser(newTransfer);
    newLds->memOrg_.at(LX).allocateNode_ = newAlloc;
    newLds->memOrg_.at(LX).isPresent = false;
    allocInput0->getMutableParent()->addChildNode(newAlloc, true, allocInput0);
    const_cast<dsc2::TransferNode*>(transferInput0)
        ->getMutableParent()
        ->addChildNode(newTransfer, false, transferInput0);
  }

  for (auto& computeOp : currDsc->computeOp_) {
    if (computeOp.opFuncName != OpFuncs::BATCHMATMUL_MXFP4W_FWD) continue;

    int newLdsIdx = addNewLds(computeOp.inputLabeledDs.at(1));
    auto newLds = &currDsc->labeledDs_.at(newLdsIdx);
    computeOp.inputLabeledDs.push_back(newLds);
    newLds->dsName_ += "_internalKernel";
    newLds->dsType_ = DsTypes::INTERNAL;
    newLds->wordLength = 2;
    newLds->dataFormat_ = DataFormats::BFLOAT16;
    newLds->scaledLdsCategory_ =
        LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR;
    newLds->mxInfo_ = computeOp.inputLabeledDs.at(0)->mxInfo_;

    auto& kerLds = computeOp.inputLabeledDs.at(1);
    auto& outputLds = computeOp.outputLabeledDs.at(0);
    auto& kerDsInfo = currDsc->primaryDsInfo_.at(kerLds->dsType_);
    auto& outputDsInfo = currDsc->primaryDsInfo_.at(outputLds->dsType_);
    auto& internalDsInfo = currDsc->primaryDsInfo_[DsTypes::INTERNAL];

    internalDsInfo.layoutDimOrder_ = kerDsInfo.layoutDimOrder_;
    for (auto dim : outputDsInfo.stickDimOrder_) {
      internalDsInfo.stickDimOrder_.push_back(dim);
      internalDsInfo.stickRepl_.push_back(1);
    }
    for (auto size : outputDsInfo.stickSize_)
      internalDsInfo.stickSize_.push_back(size);

    newLds->memOrg_[LX] = kerLds->memOrg_.at(LX);
    auto allocInput0 = newLds->memOrg_.at(LX).allocateNode_;
    auto transferInput0 = static_cast<const dsc2::TransferNode*>(
        allocInput0->allocUsers_.begin()->first);
    auto newAlloc = static_cast<dsc2::AllocateNode*>(allocInput0->clone());
    newAlloc->name_ = allocInput0->name_ + "_internalKernel";
    newAlloc->allocUsers_.clear();
    newAlloc->ldsIdx_ = newLdsIdx;
    auto newTransfer = new dsc2::TransferNode();
    newTransfer->name_ = "dummy_transfer_to_lx_internalKernel";
    newTransfer->src_.unit_ = newTransfer->src_.storage_ = NO_COMPONENT;
    dsc2::TransferNode::DstVia dstVia;
    dstVia.loc_ = {NO_COMPONENT, LX};
    newTransfer->dstVias_.push_back(dstVia);
    dsc2::DataInfo dataInfo;
    dataInfo.myLdsIdx_ = newLdsIdx;
    newTransfer->dstLdsAndLoopOffsets_.push_back(dataInfo);

    newAlloc->addAllocUser(newTransfer);
    newLds->memOrg_.at(LX).allocateNode_ = newAlloc;
    newLds->memOrg_.at(LX).isPresent = false;
    allocInput0->getMutableParent()->addChildNode(newAlloc, true, allocInput0);
    allocInput0->getMutableParent()->addChildNode(newTransfer, false,
                                                  transferInput0);
  }

// TEMP: disabling stickpacking opt
#if 0
  for (auto& computeOp : currDsc->computeOp_) {
    if (computeOp.opFuncName != OpFuncs::LAYERNORM_SCALE) continue;
    int newLdsIdx = addNewLds(computeOp.inputLabeledDs.at(0));
    auto newLds = &currDsc->labeledDs_.at(newLdsIdx);
    computeOp.inputLabeledDs.push_back(newLds);
    newLds->dsName_ += "_2";
    newLds->memOrg_[LX] = computeOp.inputLabeledDs.at(0)->memOrg_.at(LX);
    auto allocInput0 = newLds->memOrg_.at(LX).allocateNode_;
    auto transferInput0 = static_cast<const dsc2::TransferNode*>(
        allocInput0->allocUsers_.begin()->first);
    auto newAlloc = static_cast<dsc2::AllocateNode*>(allocInput0->clone());
    newAlloc->name_ = allocInput0->name_ + "_2";
    newAlloc->allocUsers_.clear();
    for (auto coreIdx : currDsc->coreIdsUsed_) {
      for (int coreletIdx = 0; coreletIdx < currDsc->numCoreletsUsed_DSC2_;
           coreletIdx++) {
        std::map<int64_t, int64_t> indexPosToFix;
        indexPosToFix[0] = coreIdx;
        indexPosToFix[1] = coreletIdx;
        indexPosToFix[2] = 0;
        auto coreCoreletStartAddr =
            newAlloc->startAddressCoreCorelet_.getSingleData(indexPosToFix);
        std::deque<int64_t> insertIndices;
        insertIndices.push_back(coreIdx);
        insertIndices.push_back(coreletIdx);
        insertIndices.push_back(0);
        // shift 8 FP16 elements = 16 bytes
        newAlloc->startAddressCoreCorelet_.insertData(coreCoreletStartAddr + 16,
                                                      insertIndices);
      }
    }
    newAlloc->ldsIdx_ = newLdsIdx;
    auto newTransfer = new dsc2::TransferNode();
    newTransfer->name_ = "dummy_transfer_to_lx_input2";
    newTransfer->src_.unit_ = newTransfer->src_.storage_ = NO_COMPONENT;
    dsc2::TransferNode::DstVia dstVia;
    dstVia.loc_ = {NO_COMPONENT, LX};
    newTransfer->dstVias_.push_back(dstVia);
    dsc2::DataInfo dataInfo;
    dataInfo.myLdsIdx_ = newLdsIdx;
    newTransfer->dstLdsAndLoopOffsets_.push_back(dataInfo);

    newAlloc->addAllocUser(newTransfer);
    newLds->memOrg_.at(LX).allocateNode_ = newAlloc;
    newLds->memOrg_.at(LX).isPresent = false;
    allocInput0->getMutableParent()->addChildNode(newAlloc, true, allocInput0);
    allocInput0->getMutableParent()->addChildNode(newTransfer, false,
                                                  transferInput0);
  }
#endif
}

// ------------------------------------------------------------------------------------------------
// entry 309/382   level 2   scc 315   71 body lines
// unit: e309_attachToPrefilledSchedule
// authority: ddc/ddcv1.cpp:2280
// original: void Ddc::attachToPrefilledSchedule()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e309_attachToPrefilledSchedule()
{
  if (currDsc->dataStageParam_.size() < 2 ||
      !currDsc->dataStageParam_.count(metadata.core_dstgid) ||
      currDsc->dataStageParam_.at(metadata.core_dstgid).name() != "core" ||
      !currDsc->dataStageParam_.count(metadata.chunk_dstgid) ||
      currDsc->dataStageParam_.at(metadata.chunk_dstgid).name() != "chunk") {
    DT_ERROR("Expected atleast core and chunk data stage parameters");
  }
  metadata.belowLxScheduleInsertBlock = nullptr;
  for (auto* node : currDsc->scheduleTree_.traverseTreeDFSMutable()) {
    metadata.externalNodes_.insert(node);

    if (node->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
      auto* an = static_cast<dsc2::AllocateNode*>(node);
      if (an->ldsIdx_ < 0 || an->ldsIdx_ >= currDsc->labeledDs_.size() ||
          dsc2::memories.count(an->component_) == 0) {
        DT_ERROR("External allocate node improperly set: " + an->name_);
      }
      const auto& lds = currDsc->labeledDs_.at(an->ldsIdx_);
      auto memorgIt = lds.memOrg_.find(an->component_);
      if (memorgIt == lds.memOrg_.end()) {
        DT_ERROR("Missing memorg for external allocate node: " + an->name_);
      }
      if (memorgIt->second.allocateNode_ != an) {
        DT_ERROR("External allocate node not connected to memorg: " +
                 an->name_);
      }
      // memorg.isPresent = true;
      if (datastageBasedElemOff && an->component_ == SenComponents::LX)
        calculateClStartAddress(an);
    } else if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      auto* tn = static_cast<dsc2::TransferNode*>(node);
      auto* dataConnectPtr = &tn->srcLdsAndLoopOffsets_.dataConnect_;
      auto ldsIdx = tn->srcLdsAndLoopOffsets_.myLdsIdx_;
      auto storage = tn->src_.storage_;
      if (is_any_of(storage, NO_COMPONENT, HBM, CONSTANT) &&
          !tn->dstVias_.empty()) {
        if (tn->dstLdsAndLoopOffsets_.empty()) {
          DT_ERROR("External transfer node improperly set: " + tn->name_);
        }
        storage = tn->dstVias_[0].loc_.storage_;
        ldsIdx = tn->dstLdsAndLoopOffsets_[0].myLdsIdx_;
        dataConnectPtr = &tn->dstLdsAndLoopOffsets_[0].dataConnect_;
      }
      const auto tnType = tn->getTransferType();
      if ((tnType != dsc2::TransferNode::TransferType::CONSTANT_TO_TENSOR &&
           tnType != dsc2::TransferNode::TransferType::TENSOR_TO_TENSOR &&
           tnType != dsc2::TransferNode::TransferType::NO_TRANSFER_TO_TENSOR &&
           tnType !=
               dsc2::TransferNode::TransferType::NO_TRANSFER_FROM_TENSOR) ||
          ldsIdx >= currDsc->labeledDs_.size() ||
          !is_any_of(storage, LX, PTXRF, L3LUIBR)) {
        DT_ERROR("External transfer node improperly set: " + tn->name_);
      }
      metadata
          .prefilledExternalTransferToDataConnectToFill_[{ldsIdx, storage}] =
          dataConnectPtr;
    } else if (node->nodeType_ == dsc2::ScheduleNode::LOOP) {
      auto* ln = static_cast<dsc2::LoopNode*>(node);
      for (auto& [dim, kind] : ln->dims_) {
        metadata.dimToCoreChunkLoops_[dim].push_back(ln);
      }
    } else if (node->nodeType_ == dsc2::ScheduleNode::BLOCK) {
      if (node->name_ == "lx_below_schedule")
        metadata.belowLxScheduleInsertBlock =
            static_cast<dsc2::BlockNode*>(node);
    }
  }
  DT_CHECK_MSG(metadata.belowLxScheduleInsertBlock,
               "Missing below-lx schedule insert block");
}

// ------------------------------------------------------------------------------------------------
// entry 310/382   level 2   scc 270   11 body lines
// unit: e310_add_valid_actions
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:226
// original: static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
// class: MergeAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e310_add_valid_actions(ActionList& actions,
                                const AbstractLayout& input,
                                const AbstractLayout& goal)
{
    int bw = EnumsConversion::dataFormatsToBitWidth.at(input.format);
    int min_dim = std::max(dim_8bit, bitwidth_to_idx(bw));
    for (auto stick_dim : input.stickDims()) {
      for (int i = min_dim; i <= dim_64bit; i++) {
        bool swap = !input.sliceDims()[i].is_dummy();
        actions.emplace_back().reset(
            new MergeAction(stick_dim, input.sliceDims()[i], i, swap));
      }
    }
  }

// ------------------------------------------------------------------------------------------------
// entry 311/382   level 2   scc 371   15 body lines
// unit: e311_enumerate_stick_computations
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:305
// original: std::vector<ComputationOp> enumerate_stick_computations() override
// class: MergeAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::vector<ComputationOp> e311_enumerate_stick_computations()
{
    // Don't need to condition on keep_input/output_order
    // since we do it anyway.
    std::vector<ComputationOp> out_sticks;
    int outs = swap ? 2 : 1;
    for (int i = 0; i < outs; i++) {
      bool high = (i == 1);
      auto indices = get_shuffle_indices(high);
      out_sticks.push_back(bin_op(stick_dim, indices));
      if (swap) {
        out_sticks.back().output.insert({slice_dim, high});
      }
    }
    return out_sticks;
  }

// ------------------------------------------------------------------------------------------------
// entry 312/382   level 2   scc 268   12 body lines
// unit: e312_add_valid_actions
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:332
// original: static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
// class: PackAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e312_add_valid_actions(ActionList& actions,
                                const AbstractLayout& input,
                                const AbstractLayout& goal)
{
    // can't chop up 16, 32 bits in half- even if its reassembled later somehow
    // it can mess with quantization happening in between
    int bw = EnumsConversion::dataFormatsToBitWidth.at(input.format);
    if (bw > 8) return;
    if (!input.sliceDims()[dim_8bit].is_dummy()) return;
    for (auto stick_dim : input.stickDims()) {
      for (int i = dim_16bit; i <= dim_64bit; i++) {
        actions.emplace_back().reset(new PackAction(stick_dim, i));
      }
    }
  }

// ------------------------------------------------------------------------------------------------
// entry 313/382   level 2   scc 373   4 body lines
// unit: e313_enumerate_stick_computations
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:379
// original: std::vector<ComputationOp> enumerate_stick_computations() override
// class: PackAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::vector<ComputationOp> e313_enumerate_stick_computations()
{
    auto indices = get_indices();
    return {bin_op(stick_dim, indices)};
  }

// ------------------------------------------------------------------------------------------------
// entry 314/382   level 2   scc 276   6 body lines
// unit: e314_add_valid_actions
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:398
// original: static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
// class: ShiftLeftAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e314_add_valid_actions(ActionList& actions,
                                const AbstractLayout& input,
                                const AbstractLayout& goal)
{
    int bw = EnumsConversion::dataFormatsToBitWidth.at(input.format);
    if (bw > 8) return;
    actions.emplace_back().reset(
        new ShiftLeftAction(input.sliceDims()[dim_64bit]));
  }

// ------------------------------------------------------------------------------------------------
// entry 315/382   level 2   scc 374   15 body lines
// unit: e315_enumerate_stick_computations
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:428
// original: std::vector<ComputationOp> enumerate_stick_computations() override
// class: ShiftLeftAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::vector<ComputationOp> e315_enumerate_stick_computations()
{
    // Don't need to condition on keep_input/output_order
    // since we do it anyway.
    std::vector<ComputationOp> out_sticks;
    // Don't need to condition on keep_input/output_order
    // since we do it anyway.
    std::vector<int> indices[2] = {pack12, pack13};

    int outs = !extract_dim.is_dummy() ? 2 : 1;
    for (int i = 0; i < outs; i++) {
      out_sticks.push_back(unary_op(indices[i]));
      out_sticks.back().output.insert({extract_dim, (i == 1)});
    }
    return out_sticks;
  }

// ------------------------------------------------------------------------------------------------
// entry 316/382   level 2   scc 269   20 body lines
// unit: e316_add_valid_actions
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:451
// original: static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
// class: Pack8Action
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e316_add_valid_actions(ActionList& actions,
                                const AbstractLayout& input,
                                const AbstractLayout& goal)
{
    int bw = EnumsConversion::dataFormatsToBitWidth.at(input.format);
    if (bw > 4) return;
    if (input.sliceDims().size() <= dim_4bit) return;
    // TODO: this logic asserts that the only dimension we
    // can ever move into the 4 bit place is its goal dimension
    // or dummy, on the basis that our current instruction set
    // cannot recover 4-bit dimensions. If that changes, revisit.
    bool can_pack = true;
    can_pack = can_pack && input.sliceDims()[dim_4bit].is_dummy();
    can_pack = can_pack && input.sliceDims()[dim_8bit].is_dummy();
    can_pack = can_pack &&
               (input.sliceDims()[dim_16bit] == goal.sliceDims()[dim_4bit] ||
                input.sliceDims()[dim_16bit].is_dummy());
    if (can_pack) {
      for (auto stick_dim : input.stickDims()) {
        actions.emplace_back().reset(new Pack8Action(stick_dim));
      }
    }
  }

// ------------------------------------------------------------------------------------------------
// entry 317/382   level 2   scc 275   17 body lines
// unit: e317_add_valid_actions
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:501
// original: static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
// class: Pack9Action
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e317_add_valid_actions(ActionList& actions,
                                const AbstractLayout& input,
                                const AbstractLayout& goal)
{
    int bw = EnumsConversion::dataFormatsToBitWidth.at(input.format);
    if (bw > 4) return;
    // TODO: this logic asserts that the only dimension we
    // can ever move into the 4 bit place is its goal dimension
    // or dummy, on the basis that our current instruction set
    // cannot recover 4-bit dimensions. If that changes, revisit.
    bool stick_contains_goal =
        (input.stickDims().find(goal.sliceDims()[dim_4bit]) !=
         input.stickDims().end());
    bool can_pack = stick_contains_goal;
    can_pack = can_pack && input.sliceDims()[dim_4bit].is_dummy();
    can_pack = can_pack && input.sliceDims()[dim_8bit].is_dummy();
    if (can_pack) {
      actions.emplace_back().reset(new Pack9Action(goal.sliceDims()[dim_4bit]));
    }
  }

// ------------------------------------------------------------------------------------------------
// entry 318/382   level 2   scc 267   20 body lines
// unit: e318_add_valid_actions
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:548
// original: static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
// class: Pack24Action
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e318_add_valid_actions(ActionList& actions,
                                const AbstractLayout& input,
                                const AbstractLayout& goal)
{
    // TODO: this logic asserts that the only dimension we
    // can ever move into the 2 or 4 bit place is its goal dimension
    // or dummy, on the basis that our current instruction set
    // cannot recover 2 or 4-bit dimensions. If that changes, revisit.
    bool can_pack = true;
    int bw = EnumsConversion::dataFormatsToBitWidth.at(input.format);
    can_pack = can_pack && (bw > 2);
    for (auto must_be_dummy : {dim_2bit, dim_4bit, dim_8bit}) {
      can_pack = can_pack && input.sliceDims()[must_be_dummy].is_dummy();
    }
    can_pack = can_pack &&
               (input.sliceDims()[dim_16bit] == goal.sliceDims()[dim_2bit] ||
                input.sliceDims()[dim_16bit].is_dummy());
    if (can_pack) {
      for (auto stick : input.stick_dims) {
        actions.emplace_back().reset(new Pack24Action(stick));
      }
    }
  }

// ------------------------------------------------------------------------------------------------
// entry 319/382   level 2   scc 274   8 body lines
// unit: e319_add_valid_actions
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:607
// original: static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
// class: GCVTF16F8PackAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e319_add_valid_actions(ActionList& actions,
                                const AbstractLayout& input,
                                const AbstractLayout& goal)
{
    if (_out_format(input.format, goal).has_value() &&
        input.slice_dims[dim_8bit].is_dummy()) {
      for (auto stick_dim : input.stick_dims) {
        actions.emplace_back().reset(new GCVTF16F8PackAction(stick_dim));
      }
    }
  }

// ------------------------------------------------------------------------------------------------
// entry 320/382   level 2   scc 273   8 body lines
// unit: e320_add_valid_actions
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:653
// original: static void add_valid_actions(ActionList& actions, const AbstractLayout& input, const AbstractLayout& goal)
// class: GCVTF16F8MergeAction
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
void e320_add_valid_actions(ActionList& actions,
                                const AbstractLayout& input,
                                const AbstractLayout& goal)
{
    if (_out_format(input.format, goal).has_value() &&
        input.slice_dims[dim_8bit].is_dummy()) {
      for (auto stick_dim : input.stick_dims) {
        actions.emplace_back().reset(new GCVTF16F8MergeAction(stick_dim));
      }
    }
  }

// ------------------------------------------------------------------------------------------------
// entry 337/382   level 3   scc 196   298 body lines
// unit: e337_gatherRelatedPTRows
// authority: ddc/ddc_fold.cpp:959
// original: bool Ddc::gatherRelatedPTRows( RowGroupInfo &refRowGroup, const dsc2::CoordPropInfoType &coordPropInfo, const dsc2::CoordinateType<CoordinateBaseType> &refCoordinate)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
bool e337_gatherRelatedPTRows(
    RowGroupInfo &refRowGroup, const dsc2::CoordPropInfoType &coordPropInfo,
    const dsc2::CoordinateType<CoordinateBaseType> &refCoordinate)
{
  dsc2::ScheduleNode *refNode = coordPropInfo.refNode;
  dsc2::ScheduleNode *nodeForFold = coordPropInfo.nodeToFold;
  SenComponents propSrcUnit = SenComponents::NO_COMPONENT;
  SenComponents propDestUnit = SenComponents::NO_COMPONENT;

  if (nodeForFold->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
    const dsc2::TransferNode *transferNode =
        static_cast<const dsc2::TransferNode *>(nodeForFold);
    // Destination unit for propagation is for the transferNode itself. If any
    // of the source or destination units is a row unit, use the row unit as
    // propDestUnit.
    propDestUnit = transferNode->src_.unit_;
    if (getCompRowId(propDestUnit) == -1) {
      for (int i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e;
           ++i) {
        if (getCompRowId(transferNode->dstVias_.at(i).loc_.unit_) != -1) {
          propDestUnit = transferNode->dstVias_.at(i).loc_.unit_;
          break;
        }
      }
    }

    // Determine propSrcUnit.
    if (coordPropInfo.refIsProducer) {
      propSrcUnit = transferNode->src_.unit_;
    } else {
      propSrcUnit = SenComponents::NO_COMPONENT;
      for (int i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e;
           ++i) {
        if (matchDataStream(currDsc, coordPropInfo,
                            transferNode->dstLdsAndLoopOffsets_.at(i),
                            transferNode->dstVias_.at(i).loc_.storage_,
                            true /* checkRefForAllocate */)) {
          propSrcUnit = transferNode->dstVias_.at(i).loc_.unit_;
          break;
        }
      }
      DT_CHECK(propSrcUnit != SenComponents::NO_COMPONENT);
    }
  } else if (nodeForFold->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    const dsc2::ComputeNode *computeNode =
        static_cast<const dsc2::ComputeNode *>(nodeForFold);
    propDestUnit = computeNode->exUnit_;
    if (coordPropInfo.refIsProducer) {
      propSrcUnit = SenComponents::NO_COMPONENT;
      if (refNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
        // When a computeNode gets input for one row (say l0lurow3), the unit in
        // computeNode's inputs_ specifies the generic unit (like l0lu). Get the
        // rowSpecific unit from the reference transferNode.
        const dsc2::TransferNode *transferNode =
            static_cast<const dsc2::TransferNode *>(refNode);
        for (int i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e;
             ++i) {
          if (matchDataStream(currDsc, coordPropInfo,
                              transferNode->dstLdsAndLoopOffsets_.at(i),
                              transferNode->dstVias_.at(i).loc_.storage_,
                              true /* checkRefForAllocate */)) {
            propSrcUnit = transferNode->dstVias_.at(i).loc_.unit_;
            break;
          }
        }
      } else {
        for (int i = 0, e = computeNode->inputsLdsAndLoopOffsets_.size(); i < e;
             ++i) {
          if (matchDataStream(currDsc, coordPropInfo,
                              computeNode->inputsLdsAndLoopOffsets_.at(i),
                              computeNode->inputs_.at(i),
                              true /* checkRefForAllocate */)) {
            propSrcUnit = computeNode->inputs_.at(i);
            break;
          }
        }
      }
      DT_CHECK(propSrcUnit != SenComponents::NO_COMPONENT);
    } else {
      propSrcUnit = SenComponents::NO_COMPONENT;
      for (int i = 0, e = computeNode->outputsLdsAndLoopOffsets_.size(); i < e;
           ++i) {
        if (matchDataStream(currDsc, coordPropInfo,
                            computeNode->outputsLdsAndLoopOffsets_.at(i),
                            computeNode->outputs_.at(i),
                            true /* checkRefForAllocate */)) {
          propSrcUnit = computeNode->outputs_.at(i);
          break;
        }
      }

      if (refNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
        const dsc2::TransferNode *transferNode =
            static_cast<const dsc2::TransferNode *>(refNode);
        // Override the comp in case transfer is for a single PT row.
        // Example: transfer from LXLU to PTRow4.
        //          - Src unit is lxlu.
        //          - Dest unit is ptrow4.
        //          In this case, coordinate should be for the PT row 4 only.
        if (!EnumsConversion::senCompToRowId.count(propSrcUnit)) {
          if (EnumsConversion::senCompToRowId.count(transferNode->src_.unit_)) {
            propSrcUnit = transferNode->src_.unit_;
          } else if (EnumsConversion::senCompToRowId.count(
                         transferNode->dstVias_.at(0).loc_.unit_)) {
            propSrcUnit = transferNode->dstVias_.at(0).loc_.unit_;
          }
        }
      }
      DT_CHECK(propSrcUnit != SenComponents::NO_COMPONENT);
    }
  } else if (nodeForFold->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
    if (refNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      const dsc2::TransferNode *transferNode =
          static_cast<const dsc2::TransferNode *>(refNode);
      if (coordPropInfo.refIsProducer) {
        propDestUnit = SenComponents::NO_COMPONENT;
        for (int i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e;
             ++i) {
          if (matchDataStream(currDsc, coordPropInfo,
                              transferNode->dstLdsAndLoopOffsets_.at(i),
                              transferNode->dstVias_.at(i).loc_.storage_,
                              false /* !checkRefForAllocate */)) {
            propDestUnit = transferNode->dstVias_.at(i).loc_.unit_;
            break;
          }
        }
        DT_CHECK(propDestUnit != SenComponents::NO_COMPONENT);
        propSrcUnit = propDestUnit;
        if (getCompRowId(propSrcUnit) == -1 &&
            getCompRowId(transferNode->src_.unit_) != -1) {
          propSrcUnit = transferNode->src_.unit_;
        }
        const dsc2::AllocateNode *allocNode =
            static_cast<const dsc2::AllocateNode *>(nodeForFold);
        if (is_any_of(allocNode->component_, SenComponents::PTARF,
                      SenComponents::PTXRF)) {
          propDestUnit = allocNode->component_;
        }
      } else {
        propSrcUnit = transferNode->dstVias_.at(0).loc_.unit_;
        if (getCompRowId(propSrcUnit) == -1) {
          // Try to see if any of the remaining transfer destinations is a row
          // unit. In case a row unit is found, use the row unit as the source
          // unit for propagation.
          for (int i = 1, e = transferNode->dstVias_.size(); i < e; ++i) {
            if (getCompRowId(transferNode->dstVias_.at(i).loc_.unit_) != -1) {
              propSrcUnit = transferNode->dstVias_.at(i).loc_.unit_;
              break;
            }
          }
        }
        if (getCompRowId(propSrcUnit) == -1 &&
            getCompRowId(transferNode->src_.unit_) != -1) {
          propSrcUnit = transferNode->src_.unit_;
        }
        propDestUnit = transferNode->src_.unit_;
        const dsc2::AllocateNode *allocNode =
            static_cast<const dsc2::AllocateNode *>(nodeForFold);
        if (is_any_of(allocNode->component_, SenComponents::PTARF,
                      SenComponents::PTXRF)) {
          propDestUnit = allocNode->component_;
        }
      }
    } else if (refNode->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      const dsc2::ComputeNode *computeNode =
          static_cast<const dsc2::ComputeNode *>(refNode);
      if (coordPropInfo.refIsProducer) {
        propSrcUnit = computeNode->exUnit_;
        propDestUnit = SenComponents::NO_COMPONENT;
        for (int i = 0, e = computeNode->outputsLdsAndLoopOffsets_.size();
             i < e; ++i) {
          if (matchDataStream(currDsc, coordPropInfo,
                              computeNode->outputsLdsAndLoopOffsets_.at(i),
                              computeNode->outputs_.at(i),
                              false /* checkRefForAllocate */)) {
            propDestUnit = computeNode->outputs_.at(i);
            break;
          }
        }
        DT_CHECK(propDestUnit != SenComponents::NO_COMPONENT);
      } else {
        propSrcUnit = computeNode->exUnit_;
        propDestUnit = SenComponents::NO_COMPONENT;
        for (int i = 0, e = computeNode->inputsLdsAndLoopOffsets_.size(); i < e;
             ++i) {
          if (matchDataStream(currDsc, coordPropInfo,
                              computeNode->inputsLdsAndLoopOffsets_.at(i),
                              computeNode->inputs_.at(i),
                              false /* checkRefForAllocate */)) {
            propDestUnit = computeNode->inputs_.at(i);
            break;
          }
        }
        DT_CHECK(propDestUnit != SenComponents::NO_COMPONENT);
      }
    } else {
      DT_ERROR(
          "[gatherRelatedPTRows] Unsupported reference node for propagation: " +
          refNode->name_ + ".");
    }
  }

  if (getCompRowId(propSrcUnit) == -1 && getCompRowId(propDestUnit) == -1) {
    // Neither reference nor working node corresponds to a PT row.
    refRowGroup.cat = RowGroupInfo::Category::NO_BUNDLING;
    return true;
  }

  if (currDsc->dataStageParam_.at(metadata.core_dstgid).ss_.rowSplit_.empty()) {
    // No dimension is split across PT rows.
    refRowGroup.cat = RowGroupInfo::Category::NO_BUNDLING;
    return true;
  }
  auto rowSplitDim = currDsc->dataStageParam_.at(metadata.core_dstgid)
                         .ss_.rowSplit_.begin()
                         ->first;

  if (getCompRowId(propSrcUnit) != -1 && getCompRowId(propDestUnit) == -1) {
    // Category can be overwritten by the base function.
    refRowGroup.cat = RowGroupInfo::Category::ROW_TO_NONROW;
    if (!gatherRelatedPTRowsBase(coordPropInfo, refRowGroup)) {
      return false;
    }
  } else if (getCompRowId(propSrcUnit) != -1 &&
             getCompRowId(propDestUnit) != -1) {
    int refRowId = getCompRowId(propSrcUnit);
    DT_CHECK(abs(refRowId - getCompRowId(propDestUnit)) <= 1);
    // Category can be overwritten by the base function.
    if (refRowId == getCompRowId(propDestUnit)) {
      // No need to collect nodes for all rows. In rowGroup, just add the
      // reference node's <node_ptr, row no, row offset>.
      refRowGroup.cat = RowGroupInfo::Category::ROW_TO_SAME_ROW;
      refRowGroup.activeRow = refRowId;
      CoordinateBaseType beta = (refCoordinate.coordinates_.at(rowSplitDim)
                                     .getFoldDimSize(FOLD_POS_ROWSPLIT) > 1)
                                    ? refCoordinate.coordinates_.at(rowSplitDim)
                                              .getAlpha(FOLD_POS_ROWSPLIT) *
                                          refRowId
                                    : refCoordinate.coordinates_.at(rowSplitDim)
                                          .getBeta(FOLD_POS_ROWSPLIT);

      refRowGroup.nodeInfo.push_back({coordPropInfo.refNode, refRowId, beta});
    } else {
      refRowGroup.cat = RowGroupInfo::Category::ROW_NORTH_SOUTH;
      if (!gatherRelatedPTRowsBase(coordPropInfo, refRowGroup)) {
        return false;
      }
    }
  } else if ((propSrcUnit == SenComponents::PTXRF ||
              propSrcUnit == SenComponents::PTARF) &&
             getCompRowId(propDestUnit) != -1) {
    // Consider propagation from PTXRF/PTARF to a row unit as ROW_TO_SAME_ROW.
    int refRowId = getCompRowId(propDestUnit);
    refRowGroup.cat = RowGroupInfo::Category::ROW_TO_SAME_ROW;
    refRowGroup.activeRow = refRowId;
    CoordinateBaseType beta =
        refCoordinate.coordinates_.at(rowSplitDim).getAlpha(FOLD_POS_ROWSPLIT) *
        refRowId;
    refRowGroup.nodeInfo.push_back({coordPropInfo.refNode, refRowId, beta});
  } else if (getCompRowId(propSrcUnit) == -1 &&
             getCompRowId(propDestUnit) != -1) {
    // UN-bundling is needed in this scenario.
    // Category can be overwritten by the base function.
    refRowGroup.cat = RowGroupInfo::Category::NONROW_TO_ROW;
    dsc2::CoordPropInfoType reverseCoordPropInfo;
    reverseCoordPropInfo.refNode = coordPropInfo.nodeToFold;
    reverseCoordPropInfo.nodeToFold = coordPropInfo.refNode;
    reverseCoordPropInfo.dataConnect = coordPropInfo.dataConnect;
    reverseCoordPropInfo.refIsProducer = !coordPropInfo.refIsProducer;
    if (!gatherRelatedPTRowsBase(reverseCoordPropInfo, refRowGroup)) {
      return false;
    }
  }

  if (coordPropReportLevel_ > 1) {
    refRowGroup.print(std::cout);
    if (refRowGroup.cat == RowGroupInfo::Category::ROW_TO_SAME_ROW) {
      std::cout << "\nRefNode " << refNode->name_ << "(" << refNode
                << "), workingNode= " << nodeForFold->name_ << ")"
                << " requires single row " << refRowGroup.activeRow << " :";

      std::cout << " (" << refRowGroup.nodeInfo.at(0).node->name_
                << ", row= " << refRowGroup.nodeInfo.at(0).row
                << ", beta=" << refRowGroup.nodeInfo.at(0).beta << ")";
    } else if (refRowGroup.cat == RowGroupInfo::Category::ROW_TO_NONROW) {
      std::cout << "\nRefNode " << refNode->name_ << "(" << refNode
                << "), workingNode= " << nodeForFold->name_ << ")"
                << " needs bundling of rows.";
    } else if (refRowGroup.cat == RowGroupInfo::Category::NONROW_TO_ROW) {
      std::cout << "\nRefNode " << refNode->name_ << "(" << refNode
                << "), workingNode= " << nodeForFold->name_ << ")"
                << " needs unbundling to a single row.";
    } else if (refRowGroup.cat == RowGroupInfo::Category::NO_BUNDLING) {
      std::cout << "\nRefNode " << refNode->name_ << "(" << refNode
                << "), workingNode= " << nodeForFold->name_ << ")"
                << " no bundling needed.";
    }
    std::cout << std::endl;
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 338/382   level 3   scc 245   36 body lines
// unit: e338_performPeSfpWorkSplit
// authority: ddc/ddc_transformation.cpp:1395
// original: bool Ddc::performPeSfpWorkSplit()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e338_performPeSfpWorkSplit()
{
  if (currDsc->dataStageParam_.at(metadata.core_dstgid)
          .ss_.peSfpSplit_.empty()) {
    if (transformationReportLevel_ > 0) {
      if (currDsc->computeOp_.size() > 0) {
        std::cerr << "\nOpFunc= "
                  << EnumsConversion::opFuncsToString.at(
                         currDsc->computeOp_.at(0).opFuncName)
                  << ": ";
      }
      std::cerr << "PE-SFP worksplit was not performed as input SDSC does not "
                   "have any worksplit information.";
    }
    return true;
  }

  for (auto &child : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr,
           {dsc2::ScheduleNode::ALLOCATE, dsc2::ScheduleNode::TRANSFER,
            dsc2::ScheduleNode::COMPUTE},
           ALL, -1, -1)) {
    if (child->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
      dsc2::AllocateNode *allocNode = static_cast<dsc2::AllocateNode *>(child);
      cloneForPeSfpWorkSplit(allocNode, allocNode->tempStorageForCompute_);
    } else if (child->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      dsc2::TransferNode *transferNode =
          static_cast<dsc2::TransferNode *>(child);
      // TO VERIFY: Can unrolling invalidate the traversal result (i.e. child)?
      cloneForPeSfpWorkSplit(transferNode);
    } else if (child->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      dsc2::ComputeNode *computeNode = static_cast<dsc2::ComputeNode *>(child);
      cloneForPeSfpWorkSplit(computeNode);
    }
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 339/382   level 3   scc 227   110 body lines
// unit: e339_convertResultToSkipReg
// authority: ddc/ddc_transformation_util.cpp:909
// original: bool Ddc::convertResultToSkipReg(dsc2::TransferNode *transferNode, int destIndex, bool useLatch)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e339_convertResultToSkipReg(dsc2::TransferNode *transferNode,
                                 int destIndex, bool useLatch)
{
  bool isDstLxluValue =
      transferNode->dstVias_.at(destIndex).loc_.unit_ == LXLUVALUE;
  if (isExternalNode(transferNode)) {
    DT_ERROR(
        "[convertResultToSkipReg] Can not convert result of external "
        "transferNode " +
        transferNode->name_);
  }

  DT_CHECK(destIndex < transferNode->dstLdsAndLoopOffsets_.size());

  auto resultStorage = transferNode->dstVias_.at(destIndex).loc_.storage_;
  if (!is_any_of(resultStorage, registerComponents) && !isDstLxluValue) {
    if (useLatch && resultStorage == SenComponents::LATCH) {
      // Result is already a latch. The caller transformation can continue.
      return true;
    } else if (!useLatch && resultStorage != SenComponents::LATCH &&
               dsc2::memories.count(resultStorage) == 0) {
      // Result is already a FIFO. The caller transformation can continue.
      return true;
    }
    if (transformationReportLevel_ > 1) {
      std::cerr << "[convertResultToSkipReg]: Unsupported destination storage "
                << EnumsConversion::senComponentsToString.at(resultStorage);
    }
    return false;
  }
  const std::string &resultDC =
      transferNode->dstLdsAndLoopOffsets_.at(destIndex).dataConnect_;

  if (!isDstLxluValue) {
    auto allocNode = currDsc->getMutableAllocation(
        transferNode->dstLdsAndLoopOffsets_.at(destIndex),
        transferNode->dstVias_.at(destIndex).loc_.storage_);
    if (!allocNode) {
      if (transformationReportLevel_ > 1) {
        std::cerr << "[convertResultToSkipReg]: Missing allocation in labelled "
                     "datastage. Result for data_connect "
                  << resultDC << " is not converted to "
                  << (useLatch ? "Latch." : "FIFO.");
      }
      return false;
    }

    if (destRelatedToExternalNodes(transferNode, destIndex)) {
      return false;
    }

    // Remove allocation from labelled DS
    reduceUsersOrDeleteAllocationAndMetadata(
        transferNode->dstLdsAndLoopOffsets_.at(destIndex),
        transferNode->dstVias_.at(destIndex).loc_.storage_, transferNode);
  }
  SenComponents newComponent = SenComponents::NO_COMPONENT;
  int currLatchDataId = -1;
  if (useLatch) {
    currLatchDataId = latchDataIdCounter_++;
    newComponent = SenComponents::LATCH;
    transferNode->dstLdsAndLoopOffsets_.at(destIndex).latchDataId_ =
        currLatchDataId;
  } else {
    newComponent = transferNode->src_.unit_;
    if (!transferNode->dstVias_.at(destIndex).via_.empty()) {
      newComponent = transferNode->dstVias_.at(destIndex).via_.back();
    }
  }
  transferNode->dstVias_.at(destIndex).loc_.storage_ = newComponent;

  // Update all consumers
  for (auto &consumer : metadata.dataConnects_.at(resultDC).consumers_) {
    if (consumer->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      // Modify source
      dsc2::TransferNode *transferConsumer =
          static_cast<dsc2::TransferNode *>(consumer);
      reduceUsersOrDeleteAllocationAndMetadata(
          transferConsumer->srcLdsAndLoopOffsets_,
          transferConsumer->src_.storage_, transferConsumer);
      transferConsumer->src_.storage_ = newComponent;
      if (useLatch) {
        transferConsumer->srcLdsAndLoopOffsets_.latchDataId_ = currLatchDataId;
      }
    } else if (consumer->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      dsc2::ComputeNode *computeConsumer =
          static_cast<dsc2::ComputeNode *>(consumer);
      for (size_t i = 0, e = computeConsumer->inputsLdsAndLoopOffsets_.size();
           i < e; ++i) {
        if (computeConsumer->inputsLdsAndLoopOffsets_.at(i).dataConnect_ ==
            resultDC) {
          if (!isDstLxluValue) {
            // Modify input to read from FIFO or latch.
            reduceUsersOrDeleteAllocationAndMetadata(
                computeConsumer->inputsLdsAndLoopOffsets_.at(i),
                computeConsumer->inputs_.at(i), computeConsumer);
          }
          computeConsumer->inputs_.at(i) = newComponent;
          if (useLatch) {
            computeConsumer->inputsLdsAndLoopOffsets_.at(i).latchDataId_ =
                currLatchDataId;
          }
        }
      }
    } else {
      DT_ERROR("[convertResultToSkipReg]: Unsupported consumer type " +
               dsc2::ScheduleNode::nodeTypeToString.at(consumer->nodeType_) +
               ".");
    }
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 340/382   level 3   scc 257   69 body lines
// unit: e340_insertComputeBetweenTransferAndReg
// authority: ddc/ddc_transformation_util.cpp:1021
// original: bool Ddc::insertComputeBetweenTransferAndReg(dsc2::TransferNode *transferNode, int transferDestIndex, dsc2::ComputeNode *computeNode, int computeInputIndex)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e340_insertComputeBetweenTransferAndReg(dsc2::TransferNode *transferNode,
                                             int transferDestIndex,
                                             dsc2::ComputeNode *computeNode,
                                             int computeInputIndex)
{
  if (isExternalNode(transferNode)) {
    DT_ERROR(
        "[convertResultToSkipReg] Can not convert result of external "
        "transferNode " +
        transferNode->name_);
  }

  DT_CHECK(transferDestIndex < transferNode->dstLdsAndLoopOffsets_.size());

  // insert node in schedule tree
  DT_CHECK(computeNode->getPrev() == nullptr);
  transferNode->getMutableParent()->addChildNode(computeNode, false,
                                                 transferNode);

  auto resultStorage =
      transferNode->dstVias_.at(transferDestIndex).loc_.storage_;
  if (!is_any_of(resultStorage, registerComponents)) {
    if (transformationReportLevel_ > 1) {
      std::cerr
          << "[insertComputeBetweenTransferAndReg]: transfer destination is "
             "not a register in "
          << transferNode->name_;
    }
    return false;
  }

  const std::string &resultDC =
      transferNode->dstLdsAndLoopOffsets_.at(transferDestIndex).dataConnect_;

  auto allocNode = currDsc->getMutableAllocation(
      transferNode->dstLdsAndLoopOffsets_.at(transferDestIndex),
      transferNode->dstVias_.at(transferDestIndex).loc_.storage_);
  if (!allocNode) {
    if (transformationReportLevel_ > 1) {
      std::cerr << "[insertComputeBetweenTransferAndReg]: Missing allocation "
                   "in labeledDs. Result for data_connect "
                << resultDC << " is not converted";
    }
    return false;
  }

  if (destRelatedToExternalNodes(transferNode, transferDestIndex)) {
    return false;
  }

  // change allocation users
  allocNode->addAllocUser(computeNode);
  allocNode->removeAllocUser(transferNode);

  // change dest/input component
  computeNode->outputs_.resize(1);
  computeNode->outputs_.at(0) =
      transferNode->dstVias_.at(transferDestIndex).loc_.storage_;
  SenComponents newComponent = transferNode->src_.unit_;
  if (!transferNode->dstVias_.at(transferDestIndex).via_.empty()) {
    newComponent = transferNode->dstVias_.at(transferDestIndex).via_.back();
  }
  transferNode->dstVias_.at(transferDestIndex).loc_.storage_ = newComponent;
  computeNode->inputs_.resize(computeInputIndex + 1);
  computeNode->inputs_.at(computeInputIndex) = newComponent;
  computeNode->inputsLdsAndLoopOffsets_.resize(computeInputIndex + 1);
  computeNode->outputsLdsAndLoopOffsets_.resize(1);
  computeNode->inputsLdsAndLoopOffsets_.at(computeInputIndex) =
      computeNode->outputsLdsAndLoopOffsets_.at(0) =
          transferNode->dstLdsAndLoopOffsets_.at(transferDestIndex);

  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 341/382   level 3   scc 296   4 body lines
// unit: e341_relatedToExternalNodes
// authority: ddc/ddc_transformation_util.cpp:1712
// original: bool Ddc::relatedToExternalNodes(const dsc2::TransferNode *transferNode) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e341_relatedToExternalNodes(const dsc2::TransferNode *transferNode) const
{
  return srcRelatedToExternalNodes(transferNode) ||
         destRelatedToExternalNodes(transferNode);
}

// ------------------------------------------------------------------------------------------------
// entry 342/382   level 3   scc 379   123 body lines
// unit: e342_codegen_generic
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:910
// original: template <typename EdgeType> void AutoShuffler::codegen_generic( const ConcreteLayout& input_layout, const ConcreteLayout& output_layout, const std::vector<std::shared_ptr<GraphNode>>& shuffle, const std::vector<EdgeType>& input_edges, const std::function<std::vector<EdgeType>(std::shared_ptr<GraphNode> node)>& get_edges_for_node, const std::function<void(ComputationOp&, std::vector<EdgeType>&, Ed
// class: AutoShuffler
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
template <typename EdgeType>
void e342_codegen_generic(
    const ConcreteLayout& input_layout, const ConcreteLayout& output_layout,
    const std::vector<std::shared_ptr<GraphNode>>& shuffle,
    const std::vector<EdgeType>& input_edges,
    const std::function<std::vector<EdgeType>(std::shared_ptr<GraphNode> node)>&
        get_edges_for_node,
    const std::function<void(ComputationOp&, std::vector<EdgeType>&, EdgeType&,
                             std::pair<std::vector<int>, int>&, bool)>&
        do_op_codegen)
{
  // Go through the shuffle list and do codegen. Reading from the
  // input tensor and writing to the output tensor must be in order,
  // but anything we put in registers can be done in any order.
  SwapBuffer<std::function<uint32_t(StickIndex)>> stick_to_int;
  stick_to_int.first() = make_stick_number_key(input_layout.stick_dims);
  SwapBuffer<std::vector<DimSymbol>> stick_ordering;
  stick_ordering.first() = input_layout.stick_dims;

  SwapBuffer<std::map<uint32_t, EdgeType>> edges;
  int num_input_sticks = int_pow2(input_layout.stick_dims.size());
  DT_CHECK(num_input_sticks == input_edges.size());
  for (int i = 0; i < num_input_sticks; i++) {
    edges.first().insert(std::make_pair(i, input_edges[i]));
  }

  for (int i = 0; i < shuffle.size(); i++) {
    bool read_in_order = (i == 0);
    bool write_in_order = (i == shuffle.size() - 1);
    auto node = shuffle[i];
    auto action = node->prev_action;
    DT_CHECK(action != nullptr);

    // Housekeeping: set up output stick order, data edges, and int key.
    stick_ordering.second().clear();
    edges.second().clear();
    if (write_in_order) {
      stick_ordering.second() = output_layout.stick_dims;
      int num_output_sticks = int_pow2(stick_ordering.second().size());
      for (int i = 0; i < num_input_sticks; i++) {
        edges.second().insert(std::make_pair(i, input_edges[i]));
      }
    } else {
      auto dim_set = node->layout.stickDims();
      // Any order works here, but heuristically try to keep active sticks in
      // low dims.
      for (auto dim : stick_ordering.first()) {
        if (dim_set.count(dim)) {
          stick_ordering.second().push_back(dim);
          dim_set.erase(dim);
        }
      }
      stick_ordering.second().insert(stick_ordering.second().begin(),
                                     dim_set.begin(), dim_set.end());
    }
    stick_to_int.second() = make_stick_number_key(stick_ordering.second());

    // Get computation list for this action.
    // These computations are implicitly repeated across absent stick dims.
    auto partial_stick_computations = action->enumerate_stick_computations();
    std::vector<ComputationOp> total_stick_computations;
    for (auto comp : partial_stick_computations) {
      comp.repeat_over_dims(stick_ordering.first(), total_stick_computations);
    }

    // We can check if in-order reading/writing is possible by sorting
    // the computations accordingly. If reading/writing is not possible,
    // insert identity ops to load into registers first.
    sort_by_stick_key<true>(stick_to_int.first(), total_stick_computations);
    if (read_in_order) {
      bool satisfied = check_single_inorder_accesses<true>(
          stick_to_int.first(), total_stick_computations);
      if (!satisfied) {
        // DT_ERROR("TODO: read into registers");
        read_in_order = false;
      }
    }
    if (write_in_order) {
      if (!read_in_order) {
        // we can only perturb the order if it isn't needed for reading.
        // If it is needed for reading, then we can only hope the write
        // accesses are already in the right order.
        sort_by_stick_key<false>(stick_to_int.second(),
                                 total_stick_computations);
      }
      bool satisfied = check_single_inorder_accesses<false>(
          stick_to_int.second(), total_stick_computations);
      if (!satisfied) {
        DT_ERROR("TODO: insert identity op at end to write in correct order");
        write_in_order = false;
      }
    }

    std::vector<EdgeType> outputs = get_edges_for_node(node);

    bool output_added = false;
    std::pair<std::vector<int>, int> input_output_stick_id;
    // We have our computation ordering, execute it.
    for (auto comp : total_stick_computations) {
      std::vector<EdgeType> inputs;
      input_output_stick_id.first.clear();
      for (auto idx : comp.inputs) {
        int key = stick_to_int.first()(idx);
        // key is the stick index for input
        inputs.push_back(edges.first().at(key));
        input_output_stick_id.first.push_back(key);
      }
      // out_key is the stick index for output
      int out_key = stick_to_int.second()(comp.output);
      input_output_stick_id.second = out_key;
      // either writing to output or need to allocate a register.
      EdgeType output = outputs.back();
      if (outputs.size() != 1) outputs.pop_back();
      edges.second()[out_key] = output;
      bool out_added = (outputs.size() == 1 && output_added);
      do_op_codegen(comp, inputs, output, input_output_stick_id, out_added);
      output_added = true;
      /*
      for (int i = 0; i < input_output_stick_id.first.size(); i++)
        std::cout << "input = " << i
                  << " --> stick: " << input_output_stick_id.first.at(i)
                  << "\n";
      std::cout << "output --> stick: " << input_output_stick_id.second << "\n";
      std::cout << " -- \n";
      */
    }

    // Housekeeping: route the out edges back to the in edges
    edges.swap();
    stick_to_int.swap();
    stick_ordering.swap();
  }
}

// ------------------------------------------------------------------------------------------------
// entry 343/382   level 3   scc 277   12 body lines
// unit: e343_get_legal_transforms
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:1147
// original: std::vector<std::shared_ptr<ShuffleAction>> AutoShuffler::get_legal_transforms( const AbstractLayout& layout, const AbstractLayout& goal)
// class: AutoShuffler
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::vector<std::shared_ptr<ShuffleAction>> e343_get_legal_transforms(
    const AbstractLayout& layout, const AbstractLayout& goal)
{
  std::vector<std::shared_ptr<ShuffleAction>> actions;
  MergeAction::add_valid_actions(actions, layout, goal);
  PackAction::add_valid_actions(actions, layout, goal);
  ShiftLeftAction::add_valid_actions(actions, layout, goal);
  Pack8Action::add_valid_actions(actions, layout, goal);
  Pack9Action::add_valid_actions(actions, layout, goal);
  Pack24Action::add_valid_actions(actions, layout, goal);
  GCVTF16F8PackAction::add_valid_actions(actions, layout, goal);
  GCVTF16F8MergeAction::add_valid_actions(actions, layout, goal);
  return actions;
}

// ------------------------------------------------------------------------------------------------
// entry 356/382   level 4   scc 201   473 body lines
// unit: e356_buildFoldForAllocation
// authority: ddc/ddc_fold.cpp:2395
// original: bool Ddc::buildFoldForAllocation( dsc2::CoordPropInfoType &coordPropInfo, const dsc2::CoordinateType<CoordinateBaseType> &inputRefCoord, dsc2::AllocateNode *allocNode)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
bool e356_buildFoldForAllocation(
    dsc2::CoordPropInfoType &coordPropInfo,
    const dsc2::CoordinateType<CoordinateBaseType> &inputRefCoord,
    dsc2::AllocateNode *allocNode)
{
  dsc2::ScheduleNode *refNode = coordPropInfo.refNode;
  dsc2::CoordinateType<CoordinateBaseType> refNodeCoordinates;
  refNodeCoordinates = inputRefCoord;
  if (coordPropInfo.scaleDown) {
    scaleDownCoord(
        refNodeCoordinates,
        const_cast<dsc2::CoordinateType<CoordinateBaseType> &>(inputRefCoord),
        allocNode->ldsIdx_);
  }
  if (coordPropReportLevel_ > 0) {
    std::cout << "\n\n>>> buildFoldForAllocation:"
              << "\n  allocNode= ";
    dbgPrint(allocNode);
    std::cout << "\n  refNode= ";
    dbgPrint(refNode);
    std::cout << "\n  dataConnect= " << coordPropInfo.dataConnect
              << "\n  refIsProducer= "
              << (coordPropInfo.refIsProducer ? "T" : "F")
              << ", scaleDown= " << (coordPropInfo.scaleDown ? "T" : "F")
              << std::endl;

    if (coordPropReportLevel_ > 2) {
      std::cout << "\n\n===================================================="
                << "\nReference fold: ";
      const_cast<dsc2::CoordinateType<CoordinateBaseType> &>(refNodeCoordinates)
          .debugPrint(std::cout);
    }
  }

  RowGroupInfo refRowGroup;
  if (needToConsiderRowBundling(
          currDsc->dataStageParam_.at(metadata.core_dstgid), refNodeCoordinates,
          coordPropInfo.dimsToPropagate) &&
      !gatherRelatedPTRows(refRowGroup, coordPropInfo, refNodeCoordinates)) {
    if (coordPropReportLevel_ > 2) {
      std::cout << "\n    Could not collect PT-row group. Skipping this round.";
    }
    // Schedule the propagation to retry later.
    coordPropTracker.retry(coordPropInfo, refNodeCoordinates.getTensorDims());
    return false;
  }

  if (allocNode->ldsIdx_ < 0) {
    return false;
  }
  const auto &allocLds = currDsc->labeledDs_.at(allocNode->ldsIdx_);
  if (allocNode->component_ == SenComponents::L0 &&
      allocLds.scaledLdsCategory_ ==
          LabeledDsInfo::ScaledLdsCategory::VALUE_TENSOR &&
      !allocNode->allocateCoordinates_.hasCoordForDim(allocLds.mxInfo_.dim)) {
    // The coordinates of the value tensor on L0 needs to be constructed from
    // the coordinates of the corresponding scale tensor allocation on L0_SCALE.
    auto scaleAlloc = dsc2::getScaleAllocation(currDsc, allocNode);
    if (!scaleAlloc->allocateCoordinates_.hasCoordForDim(
            allocLds.mxInfo_.dim)) {
      if (coordPropReportLevel_ > 2) {
        std::cout << "\n    Coordinates for " << scaleAlloc->name_
                  << " are not constructed yet, skipping this round.";
      }
      // The scale coordinates are not constructed yet. Schedule the propagation
      // to retry later.
      coordPropTracker.retry(coordPropInfo, refNodeCoordinates.getTensorDims());
      return false;
    }
    // Construct coordinates for allocNode from the coordinates of the scale
    // allocation.
    scaleUpCoord(allocNode, allocNode->allocateCoordinates_,
                 scaleAlloc->allocateCoordinates_, allocNode->ldsIdx_,
                 allocNode->component_);
    scaleUpCoord(allocNode, allocNode->sliceViewCoordinates_,
                 scaleAlloc->sliceViewCoordinates_, allocNode->ldsIdx_,
                 allocNode->component_);
  }

  auto &allocNodeCoordinates = allocNode->allocateCoordinates_;

  dsc2::VectorOfLoopAndDim loopChain, refLoopChain;
  getEnclosingLoopsAndRelatedDims(currDsc, allocNode, loopChain,
                                  loopsBelowChunkBoundary);
  bool lowerCoreletLoop =
      isExternalNode(allocNode) &&
      (coreletSplitDim != PrimaryDimTypes::PrimaryDimTypesCount) &&
      allocNode->allocateCoordinates_.coreIdToWkSlice_.empty();
  getEnclosingLoopsAndRelatedDims(currDsc, refNode, refLoopChain,
                                  loopsBelowChunkBoundary, lowerCoreletLoop);

  if (allocNode->padding_.hasPaddingInfo()) {
    allocNode->allocateCoordinates_.setPadding(allocNode->padding_);
  } else {
    allocNode->allocateCoordinates_.setPadding(refNodeCoordinates.getPadding());
  }
  // Record loop distribution parameters for common loops. Normally the
  // distribution function would record such information. However, no
  // explicit distribution is done for propagating coordinates from
  // transfer/compute to allocateNodes.
  loopDistributionParamInfo[refNode][allocNode];

  if (coordPropReportLevel_ > 0) {
    std::cout << "\n\n>>> buildFoldForAllocation:"
              << "\n  allocNode= " << allocNode->name_ << "(" << allocNode
              << ")"
              << "\n  refNode= " << refNode->name_ << "(" << refNode << ")";

    if (coordPropReportLevel_ > 2) {
      std::cout << "\n\n===================================================="
                << "\nReference fold: ";
      const_cast<dsc2::CoordinateType<CoordinateBaseType> &>(refNodeCoordinates)
          .debugPrint(std::cout);
    }
  }

  std::set<PrimaryDimTypes> dimsAlreadyProcessed;
  for (auto dim : allocNode->layoutDimOrder_) {
    // TODO: add merging of two transfers into one allocation such as subcore
    // split. the dim where merge happens is specified in subcoreSplit fields.
    if (!is_any_of(dim, coordPropInfo.dimsToPropagate)) {
      continue;
    }

    if (!refNodeCoordinates.coordinates_.count(dim)) {
      continue;
    }

    if (dimsAlreadyProcessed.count(dim)) {
      // A given dimension may appear multiple times in the allocation's layout.
      continue;
    }
    dimsAlreadyProcessed.insert(dim);
    bool allocFoldAlreadyExists = allocNodeCoordinates.coordinates_.count(dim);
    const FoldManager<CoordinateBaseType> &fm =
        refNodeCoordinates.coordinates_.at(dim);
    std::vector<dsc2::FoldParamInfoType> foldParams;
    gatherFoldParams(fm, foldParams);

    // transfer temporalFolds to elemArrFolds
    dsc2::VectorOfLoopAndDim relatedLoopsForAlloc, relatedLoopsForRefNode;
    int dimIdx = currDsc->getDimIndexInLayoutOrder(allocLds.dsType_, dim);
    int scale = dimIdx < 0 ? 1 : allocLds.scale_.at(dimIdx);
    if (scale > 0) {
      collectRelatedLoops(currDsc, dim, loopChain, relatedLoopsForAlloc,
                          allocNode->padding_.getPadding(dim));
      collectRelatedLoops(currDsc, dim, refLoopChain, relatedLoopsForRefNode,
                          refNodeCoordinates.getPadding(dim));

      if (lowerCoreletLoop && dim == coreletSplitDim) {
        int coreletLoopPos = -1;
        for (int i = 0, e = relatedLoopsForRefNode.size(); i < e; ++i) {
          if (relatedLoopsForRefNode.at(i).cat ==
              LoopDistributionInfo::LoopDistributionCat::CORELET_SLICE) {
            coreletLoopPos = i;
            break;
          }
        }
        if (coreletLoopPos == -1) {
          DT_ERROR(
              "[buildFoldForAllocation] Could not find corelet_slice loop "
              "in the loop-chain of " +
              refNode->name_ + ".");
        }
        // Move corelet fold parameters to a new temporal position immediately
        // below the chunk loops.
        foldParams.insert(foldParams.begin() +
                              refNodeCoordinates.getNumOfSpatialFolds(dim) +
                              coreletLoopPos,
                          foldParams.at(FOLD_POS_CORELET));
        // Make the corelet fold a no-op.
        foldParams.at(FOLD_POS_CORELET) = {0, 0, 1, "corelet_fold"};
      }
    }

    int numSpatialFoldsOfRefNode = refNodeCoordinates.getNumOfSpatialFolds(dim);
    int numTemporalFoldsOfRefNode =
        refNodeCoordinates.getNumOfTemporalFolds(dim) +
        (lowerCoreletLoop ? 1 : 0);
    int origNumElemArrFoldsOfRefNode =
        refNodeCoordinates.getNumOfElemArrFolds(dim);
    // allocNode sits in the same loop of refNode or above.
    DT_CHECK(numTemporalFoldsOfRefNode >= relatedLoopsForAlloc.size());
    int numTemporalFoldsOfAllocNode = numTemporalFoldsOfRefNode;
    int numElemArrFoldsOfAllocNode = origNumElemArrFoldsOfRefNode;

    dsc2::FoldParamInfoType rowFoldParamInfo;
    computeParamsForRowSplitFold(dim, rowFoldParamInfo, refRowGroup);
    bool rowBundlingNeeded =
        currDsc->dataStageParam_.at(metadata.core_dstgid)
            .ss_.rowSplit_.count(dim) &&
        refRowGroup.nodeInfo.size() == dscGlobal.sysDef.numPTRows;
    if (!rowBundlingNeeded ||
        is_any_of(allocNode->component_, SenComponents::PTXRF,
                  SenComponents::PTARF)) {
      foldParams.at(FOLD_POS_ROWSPLIT) = rowFoldParamInfo;
    } else {
      // Bundle coordinates from PT-rows.
      //   1. The spatial fold for PT-rows is set to default (noop).
      getDefaultRowSplitFold(foldParams.at(FOLD_POS_ROWSPLIT));
      //   2. Place the rowFoldParamInfo after the innermost common loop
      //      that contains the reference node and the row-siblings.
      dsc2::LoopNode *commonRefRowAncestorLoop = nullptr;
      dsc2::BlockNode *commonRefRowAncestor = refRowGroup.commonGroupAncestor;
      if (commonRefRowAncestor) {
        if (commonRefRowAncestor->nodeType_ == dsc2::ScheduleNode::LOOP) {
          commonRefRowAncestorLoop =
              static_cast<dsc2::LoopNode *>(commonRefRowAncestor);
        } else {
          commonRefRowAncestorLoop =
              commonRefRowAncestor->getMutableOwnerLoop();
        }
        while (commonRefRowAncestorLoop &&
               !dsc2::loopRelevantForDim(currDsc, dim, commonRefRowAncestorLoop,
                                         refNodeCoordinates.getPadding(dim))) {
          commonRefRowAncestorLoop =
              commonRefRowAncestorLoop->getMutableOwnerLoop();
        }
      }
      DT_CHECK_MSG(
          commonRefRowAncestorLoop,
          "[buildFoldForAllocation] (RefNode=" + refNode->name_ +
              ", workingNode=" + allocNode->name_ +
              "Failed to find common ancestor loop for PT-row bundling.");
      for (int i = 0, e = relatedLoopsForRefNode.size(); i < e; ++i) {
        if (relatedLoopsForRefNode.at(i).loopNode == commonRefRowAncestorLoop) {
          // Add a virtual fold immediately inside the temporal fold of
          // the common ancestor of the the scheduleNodes to be bundled.
          foldParams.insert(foldParams.begin() + numSpatialFoldsOfRefNode +
                                numTemporalFoldsOfRefNode - i,
                            rowFoldParamInfo);
          ++origNumElemArrFoldsOfRefNode;
          break;
        }
      }
    }

    // if allocNode is above refNode, then change temporalFolds between
    // refNode and allocaNode to elemArrFolds. This assumes that stores
    // at destination are in linear fashion and no shuffling which is true for
    // all memories below lx.
    if (numTemporalFoldsOfRefNode > relatedLoopsForAlloc.size()) {
      numTemporalFoldsOfAllocNode = relatedLoopsForAlloc.size();
      numElemArrFoldsOfAllocNode = origNumElemArrFoldsOfRefNode +
                                   numTemporalFoldsOfRefNode -
                                   numTemporalFoldsOfAllocNode;

      if (allocFoldAlreadyExists) {
        // We can not change the existing element arrangment of the allocateNode
        // as other nodes' loop-levels may already have been associated with
        // specific levels of the element arrangment. Use the existing element
        // arrangment of the allocateNode to simulate a distribution of the
        // reference node's loops. This process relates the loops enclosing the
        // current reference node with the existing element arrangment levels in
        // the allocateNode's coordinates.
        relateLoopsToAllocElemArr(
            coordPropInfo, dim, refNodeCoordinates, refLoopChain,
            loopDistributionParamInfo.at(coordPropInfo.refNode).at(allocNode));
      } else {
        // Setup the link between reference temporal folds and the allocNode's
        // element arrangement folds.
        //
        //   Ref Fold  :  0 1 2 3 4 5 6
        //                S S T T T T E
        //   Alloc Fold:  S S T T E E E
        //     For loops related to positions [4-5]
        //       loop_elem_arr_level = foldNumDims - i - 1

        // First-time construction of coordinates for the allocateNode.
        // Determine relation between loops enclosing the reference node and the
        // element arrangement levels of the allocate node.

        // Absolute level of element arrangement (i.e. not fold index).
        int currElemArrLevel = origNumElemArrFoldsOfRefNode;
        auto &loopParamInfo = loopDistributionParamInfo[refNode][allocNode];
        int foldDimCount = foldParams.size();

        for (int i = foldDimCount - origNumElemArrFoldsOfRefNode - 1,
                 e = foldDimCount - numElemArrFoldsOfAllocNode;
             i >= e; --i) {
          //  Indexing into different data structures:
          //  i : absolute fold index for a temporal loop
          //  i - numSpatialFoldsOfRefNode
          //    : index of a temporal loop within the ordered list
          //      (outer to inner) of temporal folds of refNode
          //  relatedLoopsForRefNode.size() - (i - numSpatialFoldsOfRefNode) - 1
          //    : index of a temporal loop within the ordered list
          //      (inner to outer) of related enclosing loops
          int relatedLoopIndex = relatedLoopsForRefNode.size() -
                                 (i - numSpatialFoldsOfRefNode) - 1;
          auto &currLoopParam =
              loopParamInfo[relatedLoopsForRefNode.at(relatedLoopIndex)
                                .loopNode][dim];
          currLoopParam.alpha = foldParams.at(i).alpha;
          currLoopParam.beta = foldParams.at(i).beta;
          currLoopParam.relatedElemArrLevel = currElemArrLevel;
          ++currElemArrLevel;
        }
        constructAllocElemArrLayout(currDsc, dscGlobal, allocNode, dim,
                                    foldParams, numElemArrFoldsOfAllocNode,
                                    loopParamInfo);
        // Update the fold labels
        for (int i = foldParams.size() - 1, c = 0;
             c < numElemArrFoldsOfAllocNode; --i, ++c) {
          foldParams.at(i).foldDimLabel = "elem_arr_" + std::to_string(c);
        }
      }
    } else if (scale > 0) {
      const dsc2::LoopNode *innermostRelevantRefLoop = nullptr;
      const dsc2::LoopNode *innermostRelevantAllocLoop = nullptr;
      auto currNode = refNode->getMutableOwnerLoop();
      while (currNode) {
        if (dsc2::loopRelevantForDim(currDsc, dim, currNode,
                                     refNodeCoordinates.getPadding(dim))) {
          innermostRelevantRefLoop = currNode;
          break;
        }
        currNode = currNode->getMutableOwnerLoop();
      }

      currNode = allocNode->getMutableOwnerLoop();
      while (currNode) {
        if (dsc2::loopRelevantForDim(
                currDsc, dim, currNode,
                allocNode->allocateCoordinates_.getPadding(dim))) {
          innermostRelevantAllocLoop = currNode;
          break;
        }
        currNode = currNode->getMutableOwnerLoop();
      }
      if (innermostRelevantAllocLoop != innermostRelevantRefLoop) {
        std::cout << "\nRefLoopChain("
                  << EnumsConversion::primaryDimToString.at(dim) << "): ";
        currNode = refNode->getMutableOwnerLoop();
        while (currNode) {
          if (dsc2::loopRelevantForDim(currDsc, dim, currNode,
                                       refNodeCoordinates.getPadding(dim))) {
            std::cout << " " << currNode->name_;
          }
          currNode = currNode->getMutableOwnerLoop();
        }
        std::cout << "\nAllocNodeLoopChain("
                  << EnumsConversion::primaryDimToString.at(dim) << "): ";
        currNode = allocNode->getMutableOwnerLoop();
        while (currNode) {
          if (dsc2::loopRelevantForDim(currDsc, dim, currNode,
                                       refNodeCoordinates.getPadding(dim))) {
            std::cout << " " << currNode->name_;
          }
          currNode = currNode->getMutableOwnerLoop();
        }
        std::cout << std::endl;
      }
      DT_CHECK(innermostRelevantAllocLoop == innermostRelevantRefLoop);
    }

    if (!allocNodeCoordinates.coordinates_.count(dim)) {
      // Construct the folds for the allocateNode.
      int spatialFoldEnds = refNodeCoordinates.getNumOfSpatialFolds(dim) - 1;
      int temporalFoldEnds = spatialFoldEnds + numTemporalFoldsOfAllocNode;
      dsc2::CoordinateCategory coordCat =
          dsc2::CoordinateCategory::UNKNOWN_COORD;
      for (int i = foldParams.size() - 1; i >= 0; --i) {
        std::string foldDimLabel = foldParams.at(i).foldDimLabel;
        if (i > temporalFoldEnds) {
          coordCat = dsc2::CoordinateCategory::ELEM_ARR_COORD;
        } else if (i > spatialFoldEnds) {
          coordCat = dsc2::CoordinateCategory::TEMPORAL_COORD;
        } else {
          coordCat = dsc2::CoordinateCategory::SPATIAL_COORD;
        }
        allocNodeCoordinates.addFold(
            dim, coordCat, foldParams.at(i).cardinality, foldDimLabel,
            foldParams.at(i).alpha, foldParams.at(i).beta, 0);
      }

      // In case the allocation requires a slicedView, construct a separate fold
      // for the sliced arrangement.
      if (allocNode->component_ == SenComponents::L0 ||
          allocNode->component_ == SenComponents::L0_SCALE) {
        if (currDsc->dataStageParam_.at(metadata.core_dstgid)
                .ss_.rowSplit_.count(dim)) {
          int elemArrIndex;
          int innerCard = 1;
          int e = foldParams.size() - 1 -
                  allocNodeCoordinates.getNumOfElemArrFolds(dim);
          int targetSliceChunkSize =
              getNumElementsInPTSlice(allocNode->ldsIdx_, dim);

          for (elemArrIndex = foldParams.size() - 1; elemArrIndex > e;
               --elemArrIndex) {
            if (innerCard * foldParams.at(elemArrIndex).cardinality >
                targetSliceChunkSize) {
              break;
            }
            innerCard *= foldParams.at(elemArrIndex).cardinality;
          }

          CoordinateBaseType sliceAlpha = targetSliceChunkSize *
                                          foldParams.at(elemArrIndex).alpha /
                                          innerCard;

          // Setup the rowsplit fold.
          foldParams.at(FOLD_POS_ROWSPLIT).alpha = sliceAlpha;
          foldParams.at(FOLD_POS_ROWSPLIT).beta = 0;
          foldParams.at(FOLD_POS_ROWSPLIT).cardinality =
              dscGlobal.sysDef.numPTRows;

          foldParams.at(elemArrIndex).cardinality =
              std::ceil((float)foldParams.at(elemArrIndex).cardinality /
                        dscGlobal.sysDef.numPTRows);
        }
        // Construct the folds for the sliceview.
        for (int i = foldParams.size() - 1; i >= 0; --i) {
          std::string foldDimLabel = foldParams.at(i).foldDimLabel;
          if (i > temporalFoldEnds) {
            coordCat = dsc2::CoordinateCategory::ELEM_ARR_COORD;
          } else if (i > spatialFoldEnds) {
            coordCat = dsc2::CoordinateCategory::TEMPORAL_COORD;
          } else {
            coordCat = dsc2::CoordinateCategory::SPATIAL_COORD;
          }
          allocNode->sliceViewCoordinates_.addFold(
              dim, coordCat, foldParams.at(i).cardinality, foldDimLabel,
              foldParams.at(i).alpha, foldParams.at(i).beta, 0);
        }
        allocNode->sliceViewCoordinates_.setPadding(
            dim, allocNode->allocateCoordinates_.getPadding(dim));
      }
    } else {
      // Verification.
      if (!allocNode->allocateCoordinates_.coreIdToWkSlice_.empty() &&
          refNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
        dsc2::TransferNode *transferNode =
            static_cast<dsc2::TransferNode *>(refNode);
        if (transferNode->transferCoordinates_.foldConstructed() &&
            !sameCoordinateRange(allocNode, allocNode->allocateCoordinates_,
                                 transferNode,
                                 transferNode->transferCoordinates_,
                                 (coordPropInfo.refIsProducer
                                      ? transferNode->dstVias_.at(0).loc_.unit_
                                      : transferNode->src_.unit_),
                                 allocNode->ldsIdx_)) {
          std::cerr << "\nCoordinate of allocateNode " << allocNode->name_
                    << ":\n";
          allocNode->allocateCoordinates_.debugPrint(std::cerr, false, "  ");
          std::cerr << "\nCoordinate of transferNode " << refNode->name_
                    << ":\n";
          transferNode->transferCoordinates_.debugPrint(std::cerr, false, "  ");
          DT_ERROR("Coordinates of transfer " + refNode->name_ +
                   " and allocateNode " + allocNode->name_ +
                   " are not consistent.");
        }
      }
    }
  }
  if (allDimsCovered(currDsc, allocNodeCoordinates, allocNode->ldsIdx_)) {
    allocNodeCoordinates.completeFoldConstruction();
    if (!allocNode->sliceViewCoordinates_.coordinates_.empty()) {
      allocNode->sliceViewCoordinates_.completeFoldConstruction();
    }
  }
  if (coordPropReportLevel_ > 2) {
    std::cout << "\n\n===================================================="
              << "\n\n[buildFoldForAllocation] fold at end:"
              << "\n  ref       = " << refNode->name_
              << "\n  allocNode = " << allocNode->name_;
    allocNodeCoordinates.debugPrint(std::cout);
    if (allocNode->sliceViewCoordinates_.foldConstructed()) {
      std::cout << "\n\nslice-view coordinates:"
                << "\n--------------------------";
      allocNode->sliceViewCoordinates_.debugPrint(std::cout);
    }
    std::cout << std::endl;
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 357/382   level 4   scc 208   773 body lines
// unit: e357_buildFoldForCompute
// authority: ddc/ddc_fold.cpp:3656
// original: bool Ddc::buildFoldForCompute(dsc2::ComputeNode *computeNode, dsc2::CoordPropInfoType &coordPropInfo, std::vector<int> &constructedInputCoords, std::vector<int> &constructedOutputCoords)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
bool e357_buildFoldForCompute(dsc2::ComputeNode *computeNode,
                              dsc2::CoordPropInfoType &coordPropInfo,
                              std::vector<int> &constructedInputCoords,
                              std::vector<int> &constructedOutputCoords)
{
  auto clearWorkDone = [&]() {
    constructedInputCoords.clear();
    constructedOutputCoords.clear();
  };
  dsc2::ScheduleNode *refNode = coordPropInfo.refNode;
  if (refNode == computeNode) {
    // A computeNode can be both a consumer and a producer of the same
    // data_connect. Example:
    //
    //   ddl.compute([%pe_fma_src00, %one_const, %pe_fma_lrf], [%pe_fma_lrf]) {
    //     computetype="FMA16", unit="pe"}
    return false;
  }
  if (coordPropReportLevel_ > 0) {
    std::cout << "\n\n>>> buildFoldForCompute:\n";
    dbgPrint(computeNode);
    std::cout << "\n  refNode= ";
    dbgPrint(refNode);
    std::cout << "\n  dataconnect= " << coordPropInfo.dataConnect
              << "\n  refIsProducer= "
              << (coordPropInfo.refIsProducer ? "T" : "F")
              << ", scaleDown= " << (coordPropInfo.scaleDown ? "T" : "F")
              << std::endl;
  }
  clearWorkDone();

  int computeLdsIdx = -1;
  auto &computeCoord =
      getRelatedComputeCoord(computeNode, coordPropInfo, constructedInputCoords,
                             constructedOutputCoords, computeLdsIdx);

  if (computeCoord.foldConstructed()) {
    clearWorkDone();
    if (coordPropReportLevel_ > 1) {
      std::cout
          << "\n[buildFoldForCompute] Fold is already constructed, returning";
    }
    return false;
  }

  if (!computeNode->isOpaqueOp_ && !constructedInputCoords.empty() &&
      (refNode->nodeType_ != dsc2::ScheduleNode::ALLOCATE)) {
    if (needNonRowBundling(computeNode->inputsLdsAndLoopOffsets_
                               .at(constructedInputCoords.at(0))
                               .dataConnect_,
                           true /* checkProducers*/)) {
      clearWorkDone();
      if (coordPropReportLevel_ > 1) {
        std::cout << "\n[buildFoldForCompute] Non-row bundling is not "
                     "supported, returning";
      }
      return false;
    }
  }

  if (coordPropReportLevel_ > 0) {
    std::cout << "\n  Building folds for inputs: [";
    for (auto i : constructedInputCoords) {
      std::cout << " " << i;
    }
    std::cout << "]\n  Building folds for outputs: [";
    for (auto i : constructedOutputCoords) {
      std::cout << " " << i;
    }
    std::cout << "]";
  }

  if (computeLdsIdx < 0) {
    DT_ERROR(
        "[buildFoldForCompute] Can not propagate coordinate to computeNode" +
        computeNode->name_ + " as input ldsIdx is not set.");
  }

  PrimaryDimTypes rowSplitDim = PrimaryDimTypes::PrimaryDimTypesCount;
  if (!currDsc->dataStageParam_.at(metadata.core_dstgid)
           .ss_.rowSplit_.empty()) {
    rowSplitDim = currDsc->dataStageParam_.at(metadata.core_dstgid)
                      .ss_.rowSplit_.begin()
                      ->first;
  }
  auto computeDims = currDsc->getLayoutDims(computeLdsIdx);
  auto &coreDs = currDsc->dataStageParam_.at(metadata.core_dstgid);
  bool canProceed = true;
  if (refNode->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
    auto refAllocNode = static_cast<dsc2::AllocateNode *>(refNode);
    auto &effectiveRefCoord =
        (getCompRowId(computeNode->exUnit_) != -1 &&
                 refAllocNode->sliceViewCoordinates_.foldConstructed()
             ? refAllocNode->sliceViewCoordinates_
             : refAllocNode->allocateCoordinates_);
    RowGroupInfo refRowGroup;
    if (needToConsiderRowBundling(coreDs, refAllocNode->allocateCoordinates_,
                                  coordPropInfo.dimsToPropagate) &&
        !gatherRelatedPTRows(refRowGroup, coordPropInfo, effectiveRefCoord)) {
      if (coordPropReportLevel_ > 2) {
        std::cout
            << "\n    Could not collect PT-row group. Skipping this round.";
      }
      // Schedule the propagation to retry later.
      coordPropTracker.retry(
          coordPropInfo, refAllocNode->allocateCoordinates_.getTensorDims());
      clearWorkDone();
      return false;
    }

    SenComponents propRefComp = refAllocNode->component_;
    if (propRefComp == SenComponents::PTXRF ||
        propRefComp == SenComponents::PTARF) {
      // *** This is a kludge. FIND A GENERAL SOLUTION ***
      //  PTXRF and PTARF represents allocation for all PT rows. On the other
      //  hand, a computeNode on the PT represents computation on an individual
      //  PT row.
      propRefComp = computeNode->exUnit_;
    }
    computeCoord.setPadding(refAllocNode->allocateCoordinates_.getPadding());
    buildFoldFromAllocation(
        coordPropInfo, computeNode, computeCoord, computeNode->exUnit_,
        propRefComp, loopDistributionParamInfo[computeNode][refAllocNode],
        refRowGroup);
    dsc2::computeLoopElemOffsetsFromCoordinates(
        currDsc, computeNode, computeCoord, refAllocNode,
        loopDistributionParamInfo.at(computeNode).at(refAllocNode),
        coordPropInfo.dimsToPropagate, coordPropReportLevel_);
  } else if (refNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
    // FIFO communication between reference transferNode and computeNode.
    const dsc2::TransferNode *transferNode =
        static_cast<const dsc2::TransferNode *>(refNode);
    const dsc2::CoordinateType<CoordinateBaseType> &refCoordinate =
        transferNode->transferCoordinates_;
    RowGroupInfo refRowGroup;
    if (needToConsiderRowBundling(coreDs, transferNode->transferCoordinates_,
                                  coordPropInfo.dimsToPropagate) &&
        !gatherRelatedPTRows(refRowGroup, coordPropInfo, refCoordinate)) {
      if (coordPropReportLevel_ > 2) {
        std::cout
            << "\n    Could not collect PT-row group. Skipping this round.";
      }
      // Schedule the propagation to retry later.
      coordPropTracker.retry(coordPropInfo, refCoordinate.getTensorDims());
      clearWorkDone();
      return false;
    }
    computeCoord.setPadding(refCoordinate.getPadding());
    SenComponents sizeRefComp = computeNode->exUnit_;
    SenComponents propRefComp = coordPropInfo.refIsProducer
                                    ? transferNode->dstVias_.at(0).loc_.unit_
                                    : transferNode->src_.unit_;
    if (refRowGroup.cat != RowGroupInfo::Category::ROW_TO_NONROW) {
      // In case one end of the transfer corresponds to a PT row, use the
      // row unit for size and propagation.
      if (EnumsConversion::senCompToRowId.count(transferNode->src_.unit_)) {
        sizeRefComp = transferNode->src_.unit_;
        propRefComp = transferNode->src_.unit_;
      } else if (EnumsConversion::senCompToRowId.count(
                     transferNode->dstVias_.at(0).loc_.unit_)) {
        sizeRefComp = transferNode->dstVias_.at(0).loc_.unit_;
        propRefComp = transferNode->dstVias_.at(0).loc_.unit_;
      }
    }
    canProceed = buildFoldFromNonAllocRef(
        coordPropInfo, computeLdsIdx, refCoordinate, computeCoord, sizeRefComp,
        propRefComp, loopDistributionParamInfo[computeNode][refNode],
        refRowGroup);
  } else if (refNode->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    // FIFO communication between reference computeNode and current computeNode.
    if (coordPropInfo.dataConnect == "") {
      DT_ERROR("[buildFoldForCompute] RefNode= " + refNode->name_ +
               ", computeNode= " + computeNode->name_ +
               ": missing data_connect.");
    }
    dsc2::ComputeNode *refComputeNode =
        static_cast<dsc2::ComputeNode *>(refNode);
    std::vector<int> refInputPos, refOutputPos;
    int selectedLdsIdx = -1;
    const dsc2::CoordinateType<CoordinateBaseType> &refCoordinate =
        getRelatedComputeCoord(refComputeNode, coordPropInfo, refInputPos,
                               refOutputPos, selectedLdsIdx);
    RowGroupInfo refRowGroup;
    if (needToConsiderRowBundling(coreDs, refCoordinate,
                                  coordPropInfo.dimsToPropagate) &&
        !gatherRelatedPTRows(refRowGroup, coordPropInfo, refCoordinate)) {
      if (coordPropReportLevel_ > 2) {
        std::cout
            << "\n    Could not collect PT-row group. Skipping this round.";
      }
      // Schedule the propagation to retry later.
      coordPropTracker.retry(coordPropInfo, refCoordinate.getTensorDims());
      clearWorkDone();
      return false;
    }
    computeCoord.setPadding(refCoordinate.getPadding());
    // In case a computeNode corresponds to an individual PT row, the node's
    // unit is expected to be set to the row unit (as opposed to the whole PT
    // array, like PTXRF).
    canProceed = buildFoldFromNonAllocRef(
        coordPropInfo, computeLdsIdx, refCoordinate, computeCoord,
        computeNode->exUnit_, refComputeNode->exUnit_,
        loopDistributionParamInfo[computeNode][refNode], refRowGroup);
  }
  if (!canProceed) {
    clearWorkDone();
    return false;
  }
  if (allDimsCovered(currDsc, computeCoord, computeLdsIdx)) {
    computeCoord.completeFoldConstruction();
  }

  auto propagateToDataStream = [&](const dsc2::AllocateNode *workingAllocNode,
                                   std::string &workingDc, int rhsLdsIdx,
                                   dsc2::CoordinateType<CoordinateBaseType>
                                       &rhsCoord,
                                   bool forInput) {
    if (coordPropReportLevel_ > 2) {
      std::cout << "\n  propagateToDataStream: "
                << (forInput ? "(inputs)" : "(outputs)")
                << " rhsDataConnect = " << workingDc
                << " rhsLdsIdx= " << rhsLdsIdx << " rhsAllocNode= "
                << (workingAllocNode ? workingAllocNode->name_ : "<none>")
                << ".";
    }
    auto &dataInfoList = (forInput ? computeNode->inputsLdsAndLoopOffsets_
                                   : computeNode->outputsLdsAndLoopOffsets_);
    auto &dataLocList =
        (forInput ? computeNode->inputs_ : computeNode->outputs_);
    auto &coordList =
        (forInput ? constructedInputCoords : constructedOutputCoords);

    for (int i = 0, e = dataInfoList.size(); i < e; ++i) {
      if (!coordList.empty() && (i == coordList.at(0))) {
        continue;
      }
      if (dataInfoList.at(i).myLdsIdx_ < 0) {
        continue;
      }
      const dsc2::AllocateNode *currAllocNode =
          currDsc->getAllocation(dataInfoList.at(i), dataLocList.at(i), true);
      if ((dataInfoList.at(i).myLdsIdx_ == rhsLdsIdx) ||
          (workingAllocNode && currAllocNode == workingAllocNode) ||
          (workingDc == dataInfoList.at(i).dataConnect_)) {
        std::string reason;
        if (dataInfoList.at(i).myLdsIdx_ == rhsLdsIdx) {
          reason = " due to matching ldsIdx " + std::to_string(rhsLdsIdx);
        } else if (workingAllocNode && currAllocNode == workingAllocNode) {
          reason = " due to matching alloc " + workingAllocNode->name_;
        } else if (workingDc == dataInfoList.at(i).dataConnect_) {
          reason = " due to matching data_connect " + workingDc;
        }
        if (forInput && !computeNode->inputCoordinates_[i].foldConstructed()) {
          if (coordPropReportLevel_ > 2) {
            std::cout << "\n    Copying coordinate to input " << i << " "
                      << EnumsConversion::senComponentsToString.at(
                             dataLocList.at(i))
                      << reason << ".";
          }
          computeNode->inputCoordinates_[i] = rhsCoord;
          if (allDimsCovered(currDsc, computeNode->inputCoordinates_[i],
                             dataInfoList.at(i).myLdsIdx_)) {
            computeNode->inputCoordinates_[i].completeFoldConstruction();
          }
          if (!is_any_of(i, constructedInputCoords)) {
            constructedInputCoords.push_back(i);
          }
        } else if (!forInput) {
          if (coordPropReportLevel_ > 2) {
            std::cout << "\n    Copying coordinate to output "
                      << EnumsConversion::senComponentsToString.at(
                             dataLocList.at(i))
                      << reason << ".";
          }

          computeNode->outputCoordinate_ = rhsCoord;
          if (allDimsCovered(currDsc, computeNode->outputCoordinate_,
                             dataInfoList.at(i).myLdsIdx_)) {
            computeNode->outputCoordinate_.completeFoldConstruction();
          }
          constructedOutputCoords.push_back(0);
        }
      } else if (dataInfoList.at(i).myLdsIdx_ >= 0) {
        // Try to propagate coordinates for individual dimensions.
        auto &rhsLds = currDsc->labeledDs_.at(rhsLdsIdx);
        auto &lhsCoord = (forInput ? computeNode->inputCoordinates_[i]
                                   : computeNode->outputCoordinate_);
        auto &lhsLds = currDsc->labeledDs_.at(dataInfoList.at(i).myLdsIdx_);
        PaddingFormType lhsPad;
        if (!lhsLds.memOrg_.empty()) {
          lhsPad = lhsLds.memOrg_.begin()->second.allocateNode_->padding_;
        } else {
          // TO DO: Any action needed?
        }
        bool addedNewDim = false;
        bool rowSplitDimPropagated = false;
        auto lhsDims = currDsc->getLayoutDims(lhsLds.ldsIdx_);
        for (auto lhsDim : lhsDims) {
          if (lhsCoord.coordinates_.count(lhsDim)) {
            continue;
          }
          if (!rhsCoord.coordinates_.count(lhsDim)) {
            continue;
          }

          if (rhsCoord.getPadding(lhsDim) != lhsPad.getPadding(lhsDim)) {
            continue;
          }
          int dimIdx =
              currDsc->getDimIndexInLayoutOrder(lhsLds.dsType_, lhsDim);
          int lhsScale = dimIdx < 0 ? 1 : lhsLds.scale_.at(dimIdx);
          if (lhsScale < 0) {
            if (coordPropReportLevel_ > 2) {
              std::cout << "\n    Constructing broadcast coordinate("
                        << EnumsConversion::primaryDimToString.at(lhsDim)
                        << ") for "
                        << (forInput ? " input " + std::to_string(i) : "output")
                        << " due to scale= " << lhsScale
                        << ", lhsLdsIdx= " << dataInfoList.at(i).myLdsIdx_;
            }
            buildFoldForBroadcastDim(currDsc, lhsCoord, lhsLds, lhsDim,
                                     lhsScale);
            continue;
          }

          dimIdx = currDsc->getDimIndexInLayoutOrder(rhsLds.dsType_, lhsDim);
          int rhsScale = dimIdx < 0 ? 1 : rhsLds.scale_.at(dimIdx);
          if (lhsScale != rhsScale) {
            continue;
          }
          if (lhsDim == rowSplitDim) {
            rowSplitDimPropagated = true;
          }

          if (lhsLds.scaledLdsCategory_ != rhsLds.scaledLdsCategory_) {
            if (lhsLds.mxInfo_.dim == lhsDim) {
              // The labeledDs must be related in the context of mxScale.
              if (lhsLds.mxInfo_.relatedLdsIdx != rhsLdsIdx) {
                // The LHS and RHS are not related through the mxInfo.
                continue;
              }
              // Assumption: verifying one direction of the relation is
              // sufficient (i.e. input is valid).
            } else if (rhsLds.mxInfo_.dim == lhsDim) {
              continue;
            }
          }

          lhsCoord.setPadding(lhsDim, lhsPad.getPadding(lhsDim));
          if (lhsLds.mxInfo_.dim != lhsDim ||
              lhsLds.scaledLdsCategory_ == rhsLds.scaledLdsCategory_) {
            if (coordPropReportLevel_ > 2) {
              std::cout << "\n    Copying coordinate("
                        << EnumsConversion::primaryDimToString.at(lhsDim)
                        << ") to "
                        << (forInput ? " input " + std::to_string(i) : "output")
                        << " due to matching <scale=" << lhsScale
                        << ", pad>, lhsLdsIdx= " << dataInfoList.at(i).myLdsIdx_
                        << ", rhsLdsIdx= " << rhsLdsIdx << ".";
            }

            auto &rhsFm = rhsCoord.coordinates_.at(lhsDim);
            for (int i = rhsFm.getNumDims() - 1; i >= 0; --i) {
              dsc2::CoordinateCategory coordCat =
                  computeCoord.getFoldCategory(lhsDim, i);
              CoordinateBaseType alpha, beta;
              rhsFm.getAlphaBeta(alpha, beta, i);
              lhsCoord.addFold(lhsDim, coordCat, rhsFm.getFoldDimSize(i),
                               rhsFm.getFoldDimProp(i)->Label(), alpha, beta,
                               0);
            }
          } else if (lhsLds.scaledLdsCategory_ ==
                     LabeledDsInfo::ScaledLdsCategory::VALUE_TENSOR) {
            if (coordPropReportLevel_ > 2) {
              std::cout << "\n    Scaling-up coordinate("
                        << EnumsConversion::primaryDimToString.at(lhsDim)
                        << ") to "
                        << (forInput ? " input " + std::to_string(i) : "output")
                        << " due to matching <scale=" << lhsScale
                        << ", pad>, lhsLdsIdx(VALUE)= "
                        << dataInfoList.at(i).myLdsIdx_
                        << ", rhsLdsIdx(SCALE)= " << rhsLdsIdx << ".";
            }
            // Scale-tensor to value-tensor.
            scaleUpCoord(computeNode, lhsCoord, rhsCoord, lhsLds.ldsIdx_,
                         computeNode->exUnit_);
          } else if (lhsLds.scaledLdsCategory_ ==
                     LabeledDsInfo::ScaledLdsCategory::SCALE_TENSOR) {
            if (coordPropReportLevel_ > 2) {
              std::cout << "\n    Scaling-down coordinate("
                        << EnumsConversion::primaryDimToString.at(lhsDim)
                        << ") to "
                        << (forInput ? " input " + std::to_string(i) : "output")
                        << " due to matching <scale=" << lhsScale
                        << ", pad>, lhsLdsIdx(SCALE)= "
                        << dataInfoList.at(i).myLdsIdx_
                        << ", rhsLdsIdx(VALUE)= " << rhsLdsIdx << ".";
            }
            // Scale-tensor to value-tensor.
            scaleDownCoord(lhsCoord, rhsCoord, lhsLds.ldsIdx_);
          } else {
            DT_ERROR(
                "[buildfoldForCompute] Unhandled ScaledLdsCategory found for "
                "labeledDs " +
                std::to_string(lhsLds.ldsIdx_) + ".");
          }

          addedNewDim = true;
        }
        bool allDimsCovered = true;
        for (auto &lhsDim : lhsDims) {
          if (!lhsCoord.coordinates_.count(lhsDim)) {
            allDimsCovered = false;
            break;
          }
        }
        if (addedNewDim) {
          if (forInput) {
            if (allDimsCovered) {
              computeNode->inputCoordinates_[i].completeFoldConstruction();
            }
            if (!is_any_of(i, constructedInputCoords)) {
              constructedInputCoords.push_back(i);
            }
          } else {
            constructedOutputCoords.push_back(0);
          }
        }
      }
    }
  };

  if (!constructedInputCoords.empty()) {
    std::string workingDc = "";
    const dsc2::AllocateNode *workingAllocNode = nullptr;
    if (computeNode->isOpaqueOp_) {
      workingDc = computeNode->instrAttribute_.input_data_connects_.at(
          constructedInputCoords.at(0));
      for (int inpIndex = 0;
           inpIndex < computeNode->instrAttribute_.input_data_connects_.size();
           ++inpIndex) {
        if (computeNode->inputCoordinates_.at(inpIndex).foldConstructed()) {
          continue;
        }
        if (computeNode->instrAttribute_.input_data_connects_.at(inpIndex) ==
            workingDc) {
          computeNode->inputCoordinates_[inpIndex] = computeCoord;
          if (computeCoord.foldConstructed()) {
            computeNode->inputCoordinates_.at(inpIndex)
                .completeFoldConstruction();
          }
          if (!is_any_of(inpIndex, constructedInputCoords)) {
            constructedInputCoords.push_back(inpIndex);
          }
        }
      }
    } else {
      workingDc =
          computeNode->inputsLdsAndLoopOffsets_.at(constructedInputCoords.at(0))
              .dataConnect_;
      workingAllocNode = currDsc->getAllocation(
          computeNode->inputsLdsAndLoopOffsets_.at(
              constructedInputCoords.at(0)),
          computeNode->inputs_.at(constructedInputCoords.at(0)), true);
      propagateToDataStream(workingAllocNode, workingDc, computeLdsIdx,
                            computeCoord, true);
      propagateToDataStream(workingAllocNode, workingDc, computeLdsIdx,
                            computeCoord, false);
    }
  }

  // Propagate coordinates to the output(s) of the compute node.
  // Ouput may not have all dimensions from the inputs. Example is pooling.
  // getNonbroadcastLayoutDims.
  // First input -> output -> Back prop to other inputs
  //            |-> Prop to other inputs
  //
  //   If coordinates do not match, we may need additional resolution.
  //   Example: Extra data stages added to make things work. For example, extra
  //   loops added because of
  //            hardware requirements, not for the dataflow. BMM.
  // Input_2 -> Verify existing coordinates to see if they are in agreement.
  //            If not, try to resolve.

  if (!constructedOutputCoords.empty()) {
    // Above fold construction is for the output. No further processing is
    // needed. TO DO: Copy to other inputs that use the same allocation as the
    // output.
    std::string workingDc = "";
    const dsc2::AllocateNode *workingAllocNode = nullptr;
    if (computeNode->isOpaqueOp_) {
      workingDc = computeNode->instrAttribute_.output_data_connects_.at(
          constructedOutputCoords.at(0));
      for (int inpIndex = 0;
           inpIndex < computeNode->instrAttribute_.input_data_connects_.size();
           ++inpIndex) {
        if (computeNode->inputCoordinates_.at(inpIndex).foldConstructed()) {
          continue;
        }
        if (computeNode->instrAttribute_.input_data_connects_.at(inpIndex) ==
            workingDc) {
          computeNode->inputCoordinates_[inpIndex] =
              computeNode->outputCoordinate_;
          if (computeNode->outputCoordinate_.foldConstructed()) {
            computeNode->inputCoordinates_.at(inpIndex)
                .completeFoldConstruction();
          }
          if (!is_any_of(inpIndex, constructedInputCoords)) {
            constructedInputCoords.push_back(inpIndex);
          }
        }
      }
    } else {
      workingDc = computeNode->outputsLdsAndLoopOffsets_
                      .at(constructedOutputCoords.at(0))
                      .dataConnect_;
      workingAllocNode = currDsc->getAllocation(
          computeNode->outputsLdsAndLoopOffsets_.at(
              constructedOutputCoords.at(0)),
          computeNode->outputs_.at(constructedOutputCoords.at(0)), true);
      propagateToDataStream(workingAllocNode, workingDc, computeLdsIdx,
                            computeNode->outputCoordinate_, true);
    }
    if (coordPropReportLevel_ > 2) {
      std::cout << "\nOutput Coordinate(" << computeNode->name_ << "): ";
      computeNode->outputCoordinate_.debugPrint(std::cout);
    }
  }

  if (/*(computeNode->outputsLdsAndLoopOffsets_.at(0).myLdsIdx_ < 0) ||*/
      computeNode->outputCoordinate_.foldConstructed()) {
    return true;
  }

  // Find an output LdsIdx.
  int outputLdsIdx = -1;
  for (int i = 0, e = computeNode->outputsLdsAndLoopOffsets_.size(); i < e;
       ++i) {
    if (computeNode->outputsLdsAndLoopOffsets_.at(i).myLdsIdx_ >= 0) {
      outputLdsIdx = computeNode->outputsLdsAndLoopOffsets_.at(i).myLdsIdx_;
      break;
    }
  }

  bool outputPaddingAvailable = false;
  const dsc2::ScheduleNode *outputNode = nullptr;
  if (computeNode->isOpaqueOp_) {
    if (!computeNode->instrAttribute_.output_data_connects_.empty()) {
      for (auto &consumer :
           metadata.dataConnects_
               .at(computeNode->instrAttribute_.output_data_connects_.at(0))
               .consumers_) {
        if (consumer->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
          outputNode = static_cast<const dsc2::AllocateNode *>(consumer);
          break;
        }
      }
    }
  } else {
    outputNode =
        currDsc->getAllocation(computeNode->outputsLdsAndLoopOffsets_.at(0),
                               computeNode->outputs_.at(0), true);
  }

  if (outputNode) {
    computeNode->outputCoordinate_.setPadding(
        static_cast<const dsc2::AllocateNode *>(outputNode)->padding_);
    outputPaddingAvailable = true;
  } else {
    std::string dc = "";
    if (computeNode->isOpaqueOp_) {
      dc = computeNode->instrAttribute_.output_data_connects_.at(0);
    } else {
      dc = computeNode->outputsLdsAndLoopOffsets_.at(0).dataConnect_;
    }
    DT_CHECK(dc != "");
    outputNode = *metadata.dataConnects_.at(dc).consumers_.begin();
    if (outputNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      auto transferNode = static_cast<const dsc2::TransferNode *>(outputNode);
      if (transferNode->transferCoordinates_.foldConstructed()) {
        computeNode->outputCoordinate_.setPadding(
            transferNode->transferCoordinates_.getPadding());
        outputPaddingAvailable = true;
      }
    }
  }
  if (!outputPaddingAvailable) {
    return true;
  }
  DT_CHECK(outputNode);

  auto &loopParamsAfterDistribution =
      loopDistributionParamInfo[computeNode][outputNode];

  std::vector<PrimaryDimTypes> outputDims;

  if (computeNode->isOpaqueOp_) {
    outputDims =
        currDsc->getLayoutDims(metadata.opaqueOps_.at(computeNode).ldsIdx_);
  } else {
    if (outputLdsIdx != -1) {
      outputDims = currDsc->getLayoutDims(outputLdsIdx);
    }
  }

  const LabeledDsInfo *outputLds = nullptr;
  if (outputLdsIdx != -1) {
    outputLds = &(currDsc->labeledDs_.at(outputLdsIdx));
  }
  const LabeledDsInfo *computeLds = &(currDsc->labeledDs_.at(computeLdsIdx));

  // Propagate the newly constructed input-folds to the output.
  for (auto dim : outputDims) {
    if (computeNode->outputCoordinate_.coordinates_.count(dim)) {
      continue;
    }
    // Coordinates for all dimensions may not be available in the base
    // coordinate.
    if (!computeCoord.coordinates_.count(dim)) {
      continue;
    }
    // Indicates broadcast property of the dimension.
    int dimScale = 1;
    if (outputLds) {
      int dimIdx = currDsc->getDimIndexInLayoutOrder(outputLds->dsType_, dim);
      dimScale = dimIdx < 0 ? 1 : outputLds->scale_.at(dimIdx);
    }

    if (dimScale < 0) {
      buildFoldForBroadcastDim(currDsc, computeNode->outputCoordinate_,
                               *outputLds, dim, dimScale);
      continue;
    }

    int refDimIdx = currDsc->getDimIndexInLayoutOrder(computeLds->dsType_, dim);
    int refDimScale = refDimIdx < 0 ? 1 : computeLds->scale_.at(refDimIdx);
    if (dimScale > 0 && refDimScale < 0) {
      // Reference dim is a broadcast dim while the current dim is not a
      // broadcast dim.
      continue;
    }

    PadType inputPadding = computeCoord.getPadding(dim);
    PadType outputPadding = computeNode->outputCoordinate_.getPadding(dim);
    if (inputPadding == outputPadding) {
      auto &inputFm = computeCoord.coordinates_.at(dim);
      for (int i = inputFm.getNumDims() - 1; i >= 0; --i) {
        dsc2::CoordinateCategory coordCat =
            computeCoord.getFoldCategory(dim, i);
        CoordinateBaseType alpha, beta;
        inputFm.getAlphaBeta(alpha, beta, i);
        computeNode->outputCoordinate_.addFold(
            dim, coordCat, inputFm.getFoldDimSize(i),
            inputFm.getFoldDimProp(i)->Label(), alpha, beta, 0);
      }
    } else if (outputPadding == PadType::NOPAD) {
      if (inputPadding != PadType::NOPAD) {
        std::vector<dsc2::LoopNode *> currLoopChain;
        dsc2::LoopNode *currNode = computeNode->getMutableOwnerLoop();
        while (currNode != nullptr) {
          if (dsc2::loopRelevantForDim(currDsc, dim, currNode, inputPadding)) {
            currLoopChain.push_back(currNode);
          }
          currNode = currNode->getMutableOwnerLoop();
        }
        auto &inputFm = computeCoord.coordinates_.at(dim);
        int numOfElemArrFolds = computeCoord.getNumOfElemArrFolds(dim);
        for (int i = inputFm.getNumDims() - 1; i >= 0; --i) {
          dsc2::CoordinateCategory coordCat =
              computeCoord.getFoldCategory(dim, i);
          int foldParamFactor = 1;
          if (coordCat == dsc2::CoordinateCategory::TEMPORAL_COORD ||
              coordCat == dsc2::CoordinateCategory::SPATIAL_COORD) {
            // Do we need to get the stride from a loop denom?
            // For now, using stride from core datastage.
            foldParamFactor = currDsc->dataStageParam_.at(0)
                                  .ss_.paddingSizes_.at(dim)
                                  .stride_;
            if (foldParamFactor == -1) {
              foldParamFactor = 1;
            }
          }
          // TO DO: We need to reduce the cardinality of window loop to 1.
          //    What is the best way to identify the fold for the window loop?
          int cardinality = inputFm.getFoldDimSize(i);
          CoordinateBaseType alpha, beta;
          inputFm.getAlphaBeta(alpha, beta, i);
          // Remove temporal folds for window loops as they are not relevant for
          // unpadded access.
          if (coordCat == dsc2::CoordinateCategory::TEMPORAL_COORD) {
            auto loop = currLoopChain.at(inputFm.getNumDims() - 1 - i -
                                         numOfElemArrFolds);
            bool isWindowLoop = false;
            for (auto &[loopDim, loopDimKind] : loop->dims_) {
              auto &ds = currDsc->dataStageParam_.at(loop->denId_).ss_;
              if (loopDimKind == MetaDimKind::WindowDim &&
                  ds.paddingSizes_.count(dim) &&
                  ds.paddingSizes_.at(dim).windowDim_ == loopDim) {
                isWindowLoop = true;
                break;
              }
            }
            if (isWindowLoop) {
              // No fold is needed for window loop when accessing unpadded data
              // structure.
              continue;
            }
          }
          alpha = alpha / foldParamFactor;
          if (coordCat == dsc2::CoordinateCategory::SPATIAL_COORD) {
            // Should we scale beta for temporal folds as well?
            beta = beta / foldParamFactor;
          }
          computeNode->outputCoordinate_.addFold(
              dim, coordCat, cardinality, inputFm.getFoldDimProp(i)->Label(),
              alpha, beta, 0);
        }
      } else {
        DT_ERROR("[buildFoldForCompute] ComputeNode= " + computeNode->name_ +
                 " Unsupported padtype conversion.");
      }
    }
  }

  if (coordPropReportLevel_ > 2) {
    std::cout << "\nOutput Coordinate(" << computeNode->name_ << "): ";
    computeNode->outputCoordinate_.debugPrint(std::cout);
  }

  bool allDimsCovered = true;
  for (auto &dim : outputDims) {
    if (!computeNode->outputCoordinate_.coordinates_.count(dim)) {
      allDimsCovered = false;
      if (coordPropReportLevel_ > 1) {
        std::cout << "\n  dim " << EnumsConversion::primaryDimToString.at(dim)
                  << " was not found in the output coordinate fold.";
      }
    }
  }
  // Propagate/copy to inputs that are associated with the same
  // allocation or data-connect as any of the outputs.
  if (computeNode->isOpaqueOp_) {
    std::string workingDc =
        computeNode->instrAttribute_.output_data_connects_.at(0);
    for (int inpIndex = 0;
         inpIndex < computeNode->instrAttribute_.input_data_connects_.size();
         ++inpIndex) {
      if (computeNode->inputCoordinates_.at(inpIndex).foldConstructed()) {
        continue;
      }
      if (computeNode->instrAttribute_.input_data_connects_.at(inpIndex) ==
          workingDc) {
        computeNode->inputCoordinates_[inpIndex] =
            computeNode->outputCoordinate_;
        computeNode->inputCoordinates_.at(inpIndex).completeFoldConstruction();
        constructedInputCoords.push_back(inpIndex);
      }
    }
  } else if (outputLdsIdx != -1) {
    for (int i = 0, e = computeNode->outputsLdsAndLoopOffsets_.size(); i < e;
         ++i) {
      std::string workingDc =
          computeNode->outputsLdsAndLoopOffsets_.at(i).dataConnect_;
      const dsc2::AllocateNode *workingAllocNode =
          currDsc->getAllocation(computeNode->outputsLdsAndLoopOffsets_.at(i),
                                 computeNode->outputs_.at(i), true);
      propagateToDataStream(workingAllocNode, workingDc, outputLdsIdx,
                            computeNode->outputCoordinate_, true);
    }
  }

  constructedOutputCoords.push_back(0);
  if (allDimsCovered) {
    computeNode->outputCoordinate_.completeFoldConstruction();
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 358/382   level 4   scc 207   289 body lines
// unit: e358_buildFoldForTransfer
// authority: ddc/ddc_fold.cpp:4433
// original: bool Ddc::buildFoldForTransfer(dsc2::TransferNode *transferNode, dsc2::CoordPropInfoType &coordPropInfo)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
bool e358_buildFoldForTransfer(dsc2::TransferNode *transferNode,
                               dsc2::CoordPropInfoType &coordPropInfo)
{
  if (coordPropReportLevel_ > 0) {
    std::cout << "\n\n>>> buildFoldForTransfer:\n";
    dbgPrint(transferNode);
    std::cout << "\n  refNode= ";
    dbgPrint(coordPropInfo.refNode);
    std::cout << "\n  dataconnect= " << coordPropInfo.dataConnect
              << "\n  refIsProducer= "
              << (coordPropInfo.refIsProducer ? "T" : "F")
              << ", scaleDown= " << (coordPropInfo.scaleDown ? "T" : "F")
              << std::endl;
  }

  if (isExternalNode(transferNode)) {
    if (coordPropReportLevel_ > 0) {
      std::cout << "\n  [buildFoldForTransfer]: Skipping external transfer.";
    }
    return false;
  }

  if (transferNode->src_.storage_ == SenComponents::HBM) {
    if (coordPropReportLevel_ > 0) {
      std::cout << "\n  [buildFoldForTransfer]: Skipping transfer from HBM.";
    }
    return false;
  }
  for (auto &dst : transferNode->dstVias_) {
    if (dst.loc_.storage_ == SenComponents::HBM) {
      if (coordPropReportLevel_ > 0) {
        std::cout << "\n  [buildFoldForTransfer]: Skipping transfer to HBM.";
      }
      return false;
    }
  }

  bool refIsSrc = true;
  int refPosInDest = -1;

  PaddingFormType srcPad, destPad;
  auto transferDims = currDsc->getNonBroadcastLdsDims(
      transferNode->srcLdsAndLoopOffsets_.myLdsIdx_);
  Metadata::DataTransfer *transferMetaInfo =
      (metadata.datatransfers_.count(transferNode)
           ? &metadata.datatransfers_.at(transferNode)
           : nullptr);
  const dsc2::AllocateNode *allocNode = currDsc->getAllocation(
      transferNode->srcLdsAndLoopOffsets_, transferNode->src_.storage_, true);

  if (allocNode) {
    srcPad = allocNode->padding_;
  } else if (transferMetaInfo) {
    for (auto dim : transferDims) {
      if (transferMetaInfo->hasAccessPattern(dim)) {
        srcPad.setPadding(dim, transferMetaInfo->getAccessPattern(dim).first);
      }
    }
  }

  // Determine if reference is source or destination.
  if (allocNode == coordPropInfo.refNode ||
      transferNode->srcLdsAndLoopOffsets_.dataConnect_ ==
          coordPropInfo.dataConnect) {
    refIsSrc = true;
    refPosInDest = -1;
  } else {
    refIsSrc = false;
    for (int i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e;
         ++i) {
      allocNode = currDsc->getAllocation(
          transferNode->dstLdsAndLoopOffsets_.at(i),
          transferNode->dstVias_.at(i).loc_.storage_, true);
      if (allocNode == coordPropInfo.refNode ||
          transferNode->dstLdsAndLoopOffsets_.at(i).dataConnect_ ==
              coordPropInfo.dataConnect) {
        refPosInDest = i;
        break;
      }
    }
  }

  bool destPadComputed = false;
  for (int i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e; ++i) {
    allocNode = currDsc->getAllocation(
        transferNode->dstLdsAndLoopOffsets_.at(i),
        transferNode->dstVias_.at(i).loc_.storage_, true);
    if (allocNode && allocNode->padding_.hasPaddingInfo()) {
      destPad = allocNode->padding_;
      destPadComputed = true;
      break;
    }
  }

  if (!destPadComputed && transferMetaInfo) {
    destPadComputed = true;
    for (auto dim : transferDims) {
      if (transferMetaInfo->hasAccessPattern(dim)) {
        destPad.setPadding(dim, transferMetaInfo->getAccessPattern(dim).second);
      } else {
        destPad.setPadding(dim, srcPad.getPadding(dim));
      }
    }
  }

  transferNode->transferCoordinates_.setPadding(destPad);

  // Record whether coordinates for the pe-spf split dimensions are already
  // constructed.
  std::vector<PrimaryDimTypes> existingPeSfpCoords;
  for (auto &peSfpDim : metadata.peSfpSplitDims_) {
    if (transferNode->transferCoordinates_.coordinates_.count(peSfpDim)) {
      existingPeSfpCoords.push_back(peSfpDim);
    }
  }

  int ldsIdxForProp = -1;
  if (transferNode->dstLdsAndLoopOffsets_.at(0).myLdsIdx_ >= 0) {
    ldsIdxForProp = transferNode->dstLdsAndLoopOffsets_.at(0).myLdsIdx_;
  } else if (transferNode->srcLdsAndLoopOffsets_.myLdsIdx_ >= 0 &&
             transferMetaInfo && !transferMetaInfo->hasAccessPattern()) {
    ldsIdxForProp = transferNode->srcLdsAndLoopOffsets_.myLdsIdx_;
  } else {
    // Most likely a transfer of constant. No coordinate is needed in that case.
    if (coordPropReportLevel_ > 1) {
      std::cout << "\n[buildFoldForTransfer] Can not propagate coordinate to "
                   "transferNode"
                << transferNode->name_ << " as destination ldsIdx is not set.";
    }
    return false;
  }
  DT_CHECK_MSG(
      ldsIdxForProp != -1,
      "[buildFoldForTransfer] Could not find a valid index to labeledDs.");

  SenComponents sizeRefComp =
      (refIsSrc ? transferNode->src_.unit_
                : transferNode->dstVias_.at(refPosInDest).loc_.unit_);
  // Override the comp in case transfer is for a single PT row.
  // Example: transfer from LXLU to PTRow4.
  //          - Src unit is lxlu.
  //          - Dest unit is ptrow4.
  //          In this case, coordinate should be for the PT row 4 only.
  if (!EnumsConversion::senCompToRowId.count(sizeRefComp)) {
    if (EnumsConversion::senCompToRowId.count(transferNode->src_.unit_)) {
      sizeRefComp = transferNode->src_.unit_;
    } else if (EnumsConversion::senCompToRowId.count(
                   transferNode->dstVias_.at(0).loc_.unit_)) {
      sizeRefComp = transferNode->dstVias_.at(0).loc_.unit_;
    }
  }

  // Override the comp in case transfer involves PE or SFP. This is needed to
  // account for PE-SFP splitting.
  //   In case the component already represents an individual PT row, that
  //   setting has higher priority than PE or SFP.
  //   Example: transfer_lds4_src:ptrow6_dst:pe in Conv_0 in resnet. The size
  //   should correspond to row6 only.
  if (!EnumsConversion::senCompToRowId.count(sizeRefComp) &&
      !is_any_of(sizeRefComp, SenComponents::PE, SenComponents::SFP)) {
    if (is_any_of(transferNode->src_.unit_, SenComponents::PE,
                  SenComponents::SFP)) {
      sizeRefComp = transferNode->src_.unit_;
    } else if (is_any_of(transferNode->dstVias_.at(0).loc_.unit_,
                         SenComponents::PE, SenComponents::SFP)) {
      sizeRefComp = transferNode->dstVias_.at(0).loc_.unit_;
    }
  }

  auto &coreDs = currDsc->dataStageParam_.at(metadata.core_dstgid);
  if (coordPropInfo.refNode->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
    dsc2::AllocateNode *refAllocNode =
        static_cast<dsc2::AllocateNode *>(coordPropInfo.refNode);
    auto &effectiveRefCoord =
        (getCompRowId(sizeRefComp) != -1 &&
                 refAllocNode->sliceViewCoordinates_.foldConstructed()
             ? refAllocNode->sliceViewCoordinates_
             : refAllocNode->allocateCoordinates_);
    RowGroupInfo refRowGroup;
    if (needToConsiderRowBundling(coreDs, refAllocNode->allocateCoordinates_,
                                  coordPropInfo.dimsToPropagate) &&
        !gatherRelatedPTRows(refRowGroup, coordPropInfo, effectiveRefCoord)) {
      if (coordPropReportLevel_ > 2) {
        std::cout
            << "\n    Could not collect PT-row group. Skipping this round.";
      }
      // Schedule the propagation to retry later.
      coordPropTracker.retry(
          coordPropInfo, refAllocNode->allocateCoordinates_.getTensorDims());
      return false;
    }

    // TO DO: Try to find a more general checking.
    if (transferNode->dstVias_.at(0).loc_.unit_ == SenComponents::LXLUVALUE &&
        coordPropInfo.refIsProducer) {
      if (coordPropReportLevel_ > 2) {
        std::cout << "\n    Skipping forward coordinate propagation for "
                     "transfer to lxluScaledValue.";
      }
      return false;
    }

    buildFoldFromAllocation(
        coordPropInfo, transferNode, transferNode->transferCoordinates_,
        sizeRefComp,
        coordPropInfo.refIsProducer ? transferNode->src_.unit_
                                    : transferNode->dstVias_.at(0).loc_.unit_,
        loopDistributionParamInfo[transferNode][refAllocNode], refRowGroup);
    dsc2::computeLoopElemOffsetsFromCoordinates(
        currDsc, transferNode, transferNode->transferCoordinates_, refAllocNode,
        loopDistributionParamInfo.at(transferNode).at(refAllocNode),
        coordPropInfo.dimsToPropagate, coordPropReportLevel_);
  } else if (coordPropInfo.refNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
    const dsc2::TransferNode *refTransferNode =
        static_cast<const dsc2::TransferNode *>(coordPropInfo.refNode);
    RowGroupInfo refRowGroup;
    if (needToConsiderRowBundling(coreDs, refTransferNode->transferCoordinates_,
                                  coordPropInfo.dimsToPropagate) &&
        !gatherRelatedPTRows(refRowGroup, coordPropInfo,
                             refTransferNode->transferCoordinates_)) {
      if (coordPropReportLevel_ > 2) {
        std::cout
            << "\n    Could not collect PT-row group. Skipping this round.";
      }
      // Schedule the propagation to retry later.
      coordPropTracker.retry(
          coordPropInfo, refTransferNode->transferCoordinates_.getTensorDims());
      return false;
    }
    buildFoldFromNonAllocRef(
        coordPropInfo, ldsIdxForProp, refTransferNode->transferCoordinates_,
        transferNode->transferCoordinates_, sizeRefComp,
        coordPropInfo.refIsProducer ? transferNode->src_.unit_
                                    : refTransferNode->src_.unit_,
        loopDistributionParamInfo[transferNode][coordPropInfo.refNode],
        refRowGroup);
  } else if (coordPropInfo.refNode->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
    auto refComputeNode =
        static_cast<dsc2::ComputeNode *>(coordPropInfo.refNode);
    std::vector<int> refInputPos, refOutputPos;
    int selectedLdsIdx = -1;
    dsc2::CoordinateType<CoordinateBaseType> &refCoordinate =
        getRelatedComputeCoord(refComputeNode, coordPropInfo, refInputPos,
                               refOutputPos, selectedLdsIdx);
    RowGroupInfo refRowGroup;
    if (needToConsiderRowBundling(coreDs, refCoordinate,
                                  coordPropInfo.dimsToPropagate) &&
        !gatherRelatedPTRows(refRowGroup, coordPropInfo, refCoordinate)) {
      if (coordPropReportLevel_ > 2) {
        std::cout
            << "\n    Could not collect PT-row group. Skipping this round.";
      }
      // Schedule the propagation to retry later.
      coordPropTracker.retry(coordPropInfo, refCoordinate.getTensorDims());
      return false;
    }
    buildFoldFromNonAllocRef(
        coordPropInfo, ldsIdxForProp, refCoordinate,
        transferNode->transferCoordinates_, sizeRefComp,
        refComputeNode->exUnit_,
        loopDistributionParamInfo[transferNode][coordPropInfo.refNode],
        refRowGroup);
  }
  // Add offset for PE-SFP split to the innermost element-arrangement.
  //   The fold parameter for PE-SFP split is different from the rowsplit as
  //   the rowsplit is represented by a dedicated fold level. The rowsplit is
  //   processed in a different general path inside the lower level
  //   fold-constructing functions.
  if (metadata.datatransfers_.count(transferNode) &&
      (metadata.datatransfers_.at(transferNode)
           .apply_pe_sfp_split_offset_src_ ||
       !metadata.datatransfers_.at(transferNode)
            .apply_pe_sfp_split_offset_dest_.empty())) {
    SenComponents transferComp = SenComponents::NO_COMPONENT;
    if (isNodeRelatedToComps(transferNode,
                             {SenComponents::PE, SenComponents::SFP},
                             transferComp)) {
      auto transferSizePerDim = currDsc->getBlockTransferSizePerDim(
          *transferNode, transferComp, 0 /* TO DO: Generalize */);
      for (auto &[dim, fm] : transferNode->transferCoordinates_.coordinates_) {
        if (metadata.peSfpSplitDims_.count(dim) &&
            !is_any_of(dim, existingPeSfpCoords)) {
          CoordinateBaseType beta =
              transferSizePerDim.at(dim) + fm.getBeta(fm.getNumDims() - 1);
          fm.insertBeta(beta, fm.getNumDims() - 1);
        }
      }
    }
  }
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 359/382   level 4   scc 228   82 body lines
// unit: e359_transformRegToFifoOrLatch
// authority: ddc/ddc_transformation.cpp:610
// original: bool Ddc::transformRegToFifoOrLatch()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e359_transformRegToFifoOrLatch()
{
  bool didTransformation = false;
  for (auto *node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::TRANSFER})) {
    if (isExternalNode(node)) {
      // Skip external nodes.
      continue;
    }

    auto transferNode = static_cast<dsc2::TransferNode *>(node);
    std::string transferNodeDescription = getNodeDescription(transferNode);

    for (size_t dstIndex = 0, e = transferNode->dstLdsAndLoopOffsets_.size();
         dstIndex < e; ++dstIndex) {
      if (!registerComponents.count(
              transferNode->dstVias_.at(dstIndex).loc_.storage_) &&
          transferNode->dstVias_.at(dstIndex).loc_.unit_ != LXLUVALUE) {
        continue;
      }
      const auto &resultDC =
          transferNode->dstLdsAndLoopOffsets_.at(dstIndex).dataConnect_;
      const auto &consumers = metadata.dataConnects_.at(resultDC).consumers_;

      if (consumers.size() == 0) {
        if (transformationReportLevel_ > 1) {
          std::cerr << "\nWarning: Result with data_connect= " << resultDC
                    << " does not have any consumer."
                    << "\n         TransferNode: " << transferNodeDescription;
        }
        continue;
      }

      if (destRelatedToExternalNodes(transferNode, dstIndex)) {
        continue;
      }

      bool canPerformTransformation = true;
      for (auto node : consumers) {
        if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE &&
            (static_cast<dsc2::ComputeNode *>(node))->isOpaqueOp_) {
          if (transformationReportLevel_ > 1) {
            std::cerr << "\n  Transfer destination [data_connect=" << resultDC
                      << "] is used in opaque operation " << node->name_;
          }
          canPerformTransformation = false;
          break;
        }
      }

      if (!canPerformTransformation) {
        continue;
      }

      if (consumers.size() == 1 &&
          canUseFifo(transferNode, dstIndex, *consumers.begin())) {
        if (convertResultToSkipReg(transferNode, dstIndex, false)) {
          didTransformation = true;
          if (transformationReportLevel_ > 0) {
            std::cerr << "\n[Skip-register] Converted destination " << dstIndex
                      << " of transfer " << transferNodeDescription
                      << " from register to FIFO." << " opFunc= "
                      << EnumsConversion::opFuncsToString.at(
                             currDsc->computeOp_.at(0).opFuncName);
          }
        }
      } else if (consumers.size() == 1 && dataStageExplorationDone_ &&
                 canUseLatch(transferNode, dstIndex, consumers)) {
        if (convertResultToSkipReg(transferNode, dstIndex, true)) {
          didTransformation = true;
          if (transformationReportLevel_ > 0) {
            std::cerr << "\n[Use-latch] Converted destination " << dstIndex
                      << " of transfer " << transferNodeDescription
                      << " from register to latch." << " opFunc= "
                      << EnumsConversion::opFuncsToString.at(
                             currDsc->computeOp_.at(0).opFuncName);
          }
        }
      }
    }
  }
  return didTransformation;
}

// ------------------------------------------------------------------------------------------------
// entry 360/382   level 4   scc 258   87 body lines
// unit: e360_transformFor4BsplatRead
// authority: ddc/ddc_transformation.cpp:1766
// original: bool Ddc::transformFor4BsplatRead()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e360_transformFor4BsplatRead()
{
  bool didTransformation = false;

  for (auto *node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::TRANSFER})) {
    if (isExternalNode(node)) continue;  // Skip external nodes

    auto transferNode = static_cast<dsc2::TransferNode *>(node);
    if (transferNode->replicationFactor_ != 8)
      continue;  // only works in combination with 16Bsplat read

    auto &myLds =
        currDsc->labeledDs_.at(transferNode->srcLdsAndLoopOffsets_.myLdsIdx_);
    DT_CHECK(myLds.dataFormat_ == DataFormats::IEEE_FP32 ||
             myLds.dataFormat_ == DataFormats::IEEE_INT32 ||
             myLds.dataFormat_ == DataFormats::SENUINT32);
    DT_CHECK(is_any_of(-2, myLds.scale_));
    DT_CHECK(currDsc->primaryDsInfo_.at(myLds.dsType_).stickDimOrder_.size() ==
             1);
    auto bcastDim =
        currDsc->primaryDsInfo_.at(myLds.dsType_).stickDimOrder_.at(0);
    // if all lds have dim with scale=-2, no need to perform splat
    // transformation
    bool broadcastNeeded = false;
    for (const auto &lds : currDsc->labeledDs_) {
      const auto dimIdx =
          currDsc->getDimIndexInLayoutOrder(lds.dsType_, bcastDim);
      if (dimIdx < 0 || lds.scale_.at(dimIdx) != -2) {
        broadcastNeeded = true;
        break;
      }
    }

    if (!broadcastNeeded) continue;

    std::string transferNodeDescription =
        transferNode->name_ + " source-data-connect= " +
        transferNode->srcLdsAndLoopOffsets_.dataConnect_ +
        " dstDataConnect(s)= [";
    for (const auto &dst : transferNode->dstLdsAndLoopOffsets_) {
      transferNodeDescription += (" " + dst.dataConnect_);
    }
    transferNodeDescription += " ] ";
    DT_CHECK_MSG(
        transferNode->src_.unit_ == SenComponents::LXLU &&
            transferNode->src_.storage_ == SenComponents::LX &&
            transferNode->dstVias_.at(0).loc_.unit_ == SenComponents::SFP,
        "[Transformation: 4Bsplat] can only be done when sending data from "
        "LXLU to SFP. Transfer not compliant: " +
            transferNodeDescription);

    // if transfer writes to FIFO only, convert to register
    int fifoInd = transferNode->getNonMemoryResultIndex();
    if (fifoInd != -1) {
      if (!convertResultFromFIFOtoReg(transferNode)) {
        DT_ERROR("[Transformation: 4Bsplat] transfer " +
                 transferNodeDescription +
                 " was not modified as the FIFO result could not be converted "
                 "to register");
      }
    }

    // Transfer reads from LXLU and writes to SFP LRF. Change to write to
    // fifo and introduce explicit SPLAT operation that writes into LRF
    for (int i = 0; i < transferNode->dstVias_.size(); i++) {
      DT_CHECK(transferNode->dstVias_.at(i).loc_.storage_ ==
               SenComponents::SFPLRF);
      auto computeNode = new dsc2::ComputeNode();
      computeNode->exUnit_ = transferNode->dstVias_.at(i).loc_.unit_;
      computeNode->name_ =
          "compute_splat_4B_lxlu_" +
          EnumsConversion::senComponentsToString.at(computeNode->exUnit_);
      computeNode->type_ = ComputeOpType::SPLAT;
      computeNode->dataFormat_ = DataFormats::IEEE_FP32;
      if (!insertComputeBetweenTransferAndReg(transferNode, i,
                                              std::move(computeNode), 0)) {
        DT_ERROR(
            "[Transformation: 4Bsplat] insertion of SPLAT compute node for "
            "transfer " +
            transferNodeDescription + " failed");
      }
    }

    didTransformation = true;
  }
  return didTransformation;
}

// ------------------------------------------------------------------------------------------------
// entry 361/382   level 4   scc 301   20 body lines
// unit: e361_relatedToExternalNodes
// authority: ddc/ddc_transformation_util.cpp:1757
// original: bool Ddc::relatedToExternalNodes(const dsc2::BlockNode *root) const
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation_util.rs
// ------------------------------------------------------------------------------------------------
bool e361_relatedToExternalNodes(const dsc2::BlockNode *root) const
{
  for (auto *node : currDsc->scheduleTree_.traverseTreeDFS(
           root, {dsc2::ScheduleNode::ALLOCATE, dsc2::ScheduleNode::COMPUTE,
                  dsc2::ScheduleNode::TRANSFER})) {
    if (isExternalNode(node)) {
      return true;
    } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      if (relatedToExternalNodes(
              static_cast<const dsc2::ComputeNode *>(node))) {
        return true;
      }
    } else if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      if (relatedToExternalNodes(
              static_cast<const dsc2::TransferNode *>(node))) {
        return true;
      }
    }
  }
  return false;
}

// ------------------------------------------------------------------------------------------------
// entry 362/382   level 4   scc 286   61 body lines
// unit: e362_get_shuffle
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:1161
// original: std::vector<std::shared_ptr<GraphNode>> AutoShuffler::get_shuffle( const AbstractLayout& input, const AbstractLayout& output)
// class: AutoShuffler
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
std::vector<std::shared_ptr<GraphNode>> e362_get_shuffle(
    const AbstractLayout& input, const AbstractLayout& output)
{
  reset_graph();

  DT_CHECK(input.slice_dims.size() == dims_per_slice);
  DT_CHECK(output.slice_dims.size() == dims_per_slice);

  AbstractLayout simplified_input = canonicalize_layout(input, output);
  // std::cout << "Canonicalized input/output: " << input << " " << output
  //           << std::endl;

  auto origin_node = get_node(simplified_input);
  origin_node->cost_origin_to_here = 0;

  update_worklist(origin_node);

  // Dijkstras
  while (!worklist.empty()) {
    auto node = std::get<2>(worklist.top());
    worklist.pop();

    if (node->finalized) continue;
    node->finalized = true;

    // If this node implements our goal, we are done. Due to canonicalization
    // we shouldn't have to worry about inexact matches (i.e. [][abcd] is a
    // valid implementation of [][aXcX])
    if (node->layout == output) {
      break;
    };

    for (auto action : get_legal_transforms(node->layout, output)) {
      AbstractLayout neighbor_layout = action->act(node->layout);
      auto out_format = action->out_format(node->layout.format, output);
      DT_CHECK_MSG(out_format.has_value(), "Illegal format applied to action");
      neighbor_layout.format = *out_format;
      double cost_to_neighbor =
          node->cost_origin_to_here + action->cost(node->layout);
      auto neighbor_node = get_node(neighbor_layout);

      if (cost_to_neighbor < neighbor_node->cost_origin_to_here) {
        neighbor_node->cost_origin_to_here = cost_to_neighbor;
        neighbor_node->previous_node = node;
        neighbor_node->prev_action = action;
        update_worklist(neighbor_node);
      }
    }
  }

  auto node = get_node(output);
  if (!node->finalized) {
    DT_ERROR("Requested layout not reachable.");
  }

  std::vector<std::shared_ptr<GraphNode>> shuffle;
  while (node->previous_node != nullptr) {
    shuffle.push_back(node);
    node = node->previous_node;
  }
  std::reverse(shuffle.begin(), shuffle.end());
  return shuffle;
}

// ------------------------------------------------------------------------------------------------
// entry 370/382   level 5   scc 211   350 body lines
// unit: e370_buildAndPropagateFold
// authority: ddc/ddc_fold.cpp:1625
// original: void Ddc::buildAndPropagateFold()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e370_buildAndPropagateFold()
{
  // Reserve placeholders for coordinates in computeNodes.
  for (auto *node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::COMPUTE})) {
    auto computeNode = static_cast<dsc2::ComputeNode *>(node);
    if (computeNode->isOpaqueOp_) {
      computeNode->inputCoordinates_.resize(
          computeNode->instrAttribute_.input_data_connects_.size());
    } else {
      computeNode->inputCoordinates_.resize(
          computeNode->inputsLdsAndLoopOffsets_.size());
    }
  }
  coordPropTracker.reset();
  // TO DO: Consider moving the external allocations to prepDsc to simulate the
  // computation in upstream components. First, construct folds for the external
  // allocateNodes.
  for (auto &lds : currDsc->labeledDs_) {
    dsc2::AllocateNode *extAllocNode = nullptr;

    if (lds.memOrg_.count(SenComponents::LX)) {
      extAllocNode = lds.memOrg_.at(SenComponents::LX).allocateNode_;
    }

    if (coordPropReportLevel_ > 2) {
      std::cout << std::endl << "LDS: ";
      currDsc->printLabeledDs(std::cout, lds, "  ");
      std::cout << std::endl
                << "  extAllocNode: "
                << (extAllocNode ? extAllocNode->name_ : "<not found>");
    }

    if (lds.dsType_ == DsTypes::INTERNAL || !extAllocNode ||
        !isExternalNode(extAllocNode)) {
      continue;
    }

    buildFoldForExternalAllocation(extAllocNode);
    std::vector<PrimaryDimTypes> refDims;

    if (!extAllocNode->allocateCoordinates_.coreIdToWkSlice_.empty()) {
      for (auto &refDim : extAllocNode->allocateCoordinates_.getTensorDims()) {
        bool workSliceMatches = true;
        for (auto &[customCore, customCoreInfo] :
             extAllocNode->allocateCoordinates_.coreIdToWkSlice_) {
          if (!customCoreInfo.count(refDim)) {
            continue;
          }
          if (sdsc_->coreIdToWkSlice_.at(customCore).at(refDim) !=
              customCoreInfo.at(refDim)) {
            workSliceMatches = false;
            break;
          }
        }
        if (workSliceMatches) {
          refDims.push_back(refDim);
        }
      }
    } else {
      refDims = extAllocNode->allocateCoordinates_.getTensorDims();
    }

    if (!refDims.empty()) {
      for (auto &[userNode, refCount] : extAllocNode->allocUsers_) {
        if (!is_any_of(userNode->nodeType_, dsc2::ScheduleNode::COMPUTE,
                       dsc2::ScheduleNode::TRANSFER)) {
          continue;
        }
        // Opaque operations do not maintain dataInfos. Currently, it is not
        // possible to determine which input or output of an opaque op
        // corresponds to an allocateNode unless the associated data_connect is
        // also specified.
        if (userNode->nodeType_ != dsc2::ScheduleNode::COMPUTE ||
            !(static_cast<const dsc2::ComputeNode *>(userNode)->isOpaqueOp_)) {
          coordPropTracker.addPropInfo(
              extAllocNode, const_cast<dsc2::ScheduleNode *>(userNode), refDims,
              "", isAllocateIncoming(currDsc, extAllocNode, userNode));
        }
      }
    }
  }

  auto collectAndProcessDatastreamNodes =
      [&](dsc2::ScheduleNode *currNode,
          const dsc2::ScheduleNode *refForCurrNode,
          const dsc2::DataInfo &dataInfo, const SenComponents &comp,
          const dsc2::CoordinateType<CoordinateBaseType> &refCoordinate,
          int ptRowId, bool dataInfoIsInput) -> void {
    dsc2::AllocateNode *allocNode =
        currDsc->getMutableAllocation(dataInfo, comp, true);
    if (allocNode) {
      if (allocNode != refForCurrNode) {
        dsc2::CoordPropInfoType coordPropInfo = {currNode, allocNode, "",
                                                 !dataInfoIsInput};
        coordPropTracker.addPropInfo(coordPropInfo,
                                     refCoordinate.getTensorDims());
        if (currNode->nodeType_ == dsc2::ScheduleNode::COMPUTE &&
            allocNode->ldsIdx_ != -1) {
          // In case a computeNode consumes from or produces to a value tensor
          // allocation in MX-scale mode, include the corresponding scale tensor
          // allocation in the propagation queue.
          auto &allocLds = currDsc->labeledDs_.at(allocNode->ldsIdx_);
          if (allocLds.scaledLdsCategory_ ==
              LabeledDsInfo::ScaledLdsCategory::VALUE_TENSOR) {
            auto &scaleLds =
                currDsc->labeledDs_.at(allocLds.mxInfo_.relatedLdsIdx);
            SenComponents scaleComp =
                (allocNode->component_ == SenComponents::L0)
                    ? SenComponents::L0_SCALE
                    : ((allocNode->component_ == SenComponents::PTXRF)
                           ? SenComponents::PTXRF
                           : SenComponents::NO_COMPONENT);
            if (scaleComp == SenComponents::NO_COMPONENT) {
              DT_ERROR("Unsupported component " +
                       EnumsConversion::senComponentsToString.at(
                           allocNode->component_) +
                       " for value tensor allocation " + allocNode->name_ +
                       ".");
            }
            coordPropInfo.nodeToFold =
                scaleLds.memOrg_.at(scaleComp).allocateNode_;
            coordPropInfo.scaleDown = true;
            coordPropTracker.addPropInfo(coordPropInfo,
                                         refCoordinate.getTensorDims());
          }
        }
      }
    } else if (!is_any_of(comp, memories) &&
               comp != SenComponents::NO_COMPONENT &&
               dataInfo.dataConnect_ != "") {
      // The datastream is a FIFO.
      auto &nodeList =
          dataInfoIsInput
              ? metadata.dataConnects_.at(dataInfo.dataConnect_).producers_
              : metadata.dataConnects_.at(dataInfo.dataConnect_).consumers_;
      for (auto dcNode : nodeList) {
        if (dcNode->nodeType_ == dsc2::ScheduleNode::COMPUTE ||
            dcNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
          if (ptRowId != -1) {
            if (dcNode->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
              SenComponents dcNodeComp =
                  static_cast<const dsc2::ComputeNode *>(dcNode)->exUnit_;
              int dcNodePtRowId =
                  (EnumsConversion::senCompToRowId.count(dcNodeComp)
                       ? EnumsConversion::senCompToRowId.at(dcNodeComp)
                       : -1);
              if (currNode->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
                // Allow ComputeNode -> ComputeNode with row difference of 1 to
                // cover for PT_NORTH and PT_SOUTH dataflow.
                if (!(comp == SenComponents::PTNORTH &&
                      dcNodePtRowId == ptRowId - 1) &&
                    !(comp == SenComponents::PTSOUTH &&
                      dcNodePtRowId == ptRowId + 1) &&
                    dcNodePtRowId != ptRowId) {
                  continue;
                }
              } else if (dcNodePtRowId != ptRowId) {
                continue;
              }
            } else if (dcNode->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
              const dsc2::TransferNode *dcTransferNode =
                  static_cast<const dsc2::TransferNode *>(dcNode);
              int dcTransferRowId = -1;
              if (dataInfoIsInput) {
                // Transfer's destination corresponds to the input dataInfo.
                for (int i = 0,
                         e = dcTransferNode->dstLdsAndLoopOffsets_.size();
                     i < e; ++i) {
                  if (dcTransferNode->dstLdsAndLoopOffsets_.at(i)
                              .dataConnect_ == dataInfo.dataConnect_ &&
                      EnumsConversion::senCompToRowId.count(
                          dcTransferNode->dstVias_.at(i).loc_.unit_)) {
                    dcTransferRowId = EnumsConversion::senCompToRowId.at(
                        dcTransferNode->dstVias_.at(i).loc_.unit_);
                    break;
                  }
                }
              } else {
                // Transfer's source corresponds to the input dataInfo.
                if (EnumsConversion::senCompToRowId.count(
                        dcTransferNode->src_.unit_)) {
                  dcTransferRowId = EnumsConversion::senCompToRowId.at(
                      dcTransferNode->src_.unit_);
                }
              }
              if (dcTransferRowId != ptRowId) {
                continue;
              }
            }
          }

          coordPropTracker.addPropInfo(currNode, dcNode,
                                       refCoordinate.getTensorDims(),
                                       dataInfo.dataConnect_, !dataInfoIsInput);
        }
      }
    }
  };

  dsc2::CoordPropInfoType coordPropInfo;
  // Construct folds for below-LX scheduleNodes based on the folds of the
  // external allocateNodes.
  while (coordPropTracker.getCurrItem(coordPropInfo)) {
    if (coordPropReportLevel_ > 0) {
      std::cout << "\n>>> CoordinatePropagationInfo:";
      std::cout << "\n      refNode=";
      dbgPrint(coordPropInfo.refNode);
      std::cout << "\n      nodeTofold= ";
      dbgPrint(coordPropInfo.nodeToFold);
      std::cout << "\n      data_connect= " << coordPropInfo.dataConnect
                << ", refIsProducer= "
                << (coordPropInfo.refIsProducer ? "T" : "F") << "\n      Dims:";
      for (auto &dim : coordPropInfo.dimsToPropagate) {
        std::cout << " " << EnumsConversion::primaryDimToString.at(dim);
      }
      std::cout << std::endl;
    }
    // Check if a fold is constructed already.
    if (coordPropInfo.nodeToFold->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      dsc2::ComputeNode *computeNode =
          static_cast<dsc2::ComputeNode *>(coordPropInfo.nodeToFold);
      std::vector<int> constructedInputCoords, constructedOutputCoords;
      buildFoldForCompute(computeNode, coordPropInfo, constructedInputCoords,
                          constructedOutputCoords);
      int ptRowId = -1;
      if (EnumsConversion::senCompToRowId.count(computeNode->exUnit_)) {
        ptRowId = EnumsConversion::senCompToRowId.at(computeNode->exUnit_);
      }
      if (computeNode->isOpaqueOp_) {
        for (auto i : constructedInputCoords) {
          std::string workingDc =
              computeNode->instrAttribute_.input_data_connects_.at(i);
          for (auto producer :
               metadata.dataConnects_.at(workingDc).producers_) {
            coordPropTracker.addPropInfo(
                computeNode, producer,
                computeNode->inputCoordinates_.at(i).getTensorDims(), workingDc,
                false);
          }
        }
        if (!constructedOutputCoords.empty()) {
          // Include all outputs for propagation.
          for (auto &workingDc :
               computeNode->instrAttribute_.output_data_connects_) {
            for (auto consumer :
                 metadata.dataConnects_.at(workingDc).consumers_) {
              coordPropTracker.addPropInfo(
                  computeNode, consumer,
                  computeNode->outputCoordinate_.getTensorDims(), workingDc,
                  true);
            }
          }
        }
      } else {
        for (auto i : constructedInputCoords) {
          collectAndProcessDatastreamNodes(
              computeNode, coordPropInfo.refNode,
              computeNode->inputsLdsAndLoopOffsets_.at(i),
              computeNode->inputs_.at(i), computeNode->inputCoordinates_.at(i),
              ptRowId, true);
        }
        for (auto i : constructedOutputCoords) {
          collectAndProcessDatastreamNodes(
              computeNode, coordPropInfo.refNode,
              computeNode->outputsLdsAndLoopOffsets_.at(i),
              computeNode->outputs_.at(i), computeNode->outputCoordinate_,
              ptRowId, false);
        }
      }
    } else if (coordPropInfo.nodeToFold->nodeType_ ==
               dsc2::ScheduleNode::TRANSFER) {
      dsc2::TransferNode *transferNode =
          static_cast<dsc2::TransferNode *>(coordPropInfo.nodeToFold);
      if (!buildFoldForTransfer(transferNode, coordPropInfo)) {
        continue;
      }

      if (!transferNode->transferCoordinates_.coordinates_.empty()) {
        collectAndProcessDatastreamNodes(
            transferNode, coordPropInfo.refNode,
            transferNode->srcLdsAndLoopOffsets_, transferNode->src_.storage_,
            transferNode->transferCoordinates_,
            getCompRowId(transferNode->src_.unit_), true);
        for (int i = 0, e = transferNode->dstLdsAndLoopOffsets_.size(); i < e;
             ++i) {
          collectAndProcessDatastreamNodes(
              transferNode, coordPropInfo.refNode,
              transferNode->dstLdsAndLoopOffsets_.at(i),
              transferNode->dstVias_.at(i).loc_.storage_,
              transferNode->transferCoordinates_,
              getCompRowId(transferNode->dstVias_.at(i).loc_.unit_), false);
        }
      }
    } else if (coordPropInfo.nodeToFold->nodeType_ ==
               dsc2::ScheduleNode::ALLOCATE) {
      dsc2::AllocateNode *allocNode =
          static_cast<dsc2::AllocateNode *>(coordPropInfo.nodeToFold);
      std::vector<int> inputCoords, outputCoords;
      int ldsIdx = -1;
      dsc2::CoordPropInfoType reverseCoordPropInfo;
      reverseCoordPropInfo.refNode = coordPropInfo.nodeToFold;
      reverseCoordPropInfo.nodeToFold = coordPropInfo.refNode;
      reverseCoordPropInfo.dataConnect = coordPropInfo.dataConnect;
      reverseCoordPropInfo.refIsProducer = !coordPropInfo.refIsProducer;
      dsc2::CoordinateType<CoordinateBaseType> &refCoordinate =
          (coordPropInfo.refNode->nodeType_ == dsc2::ScheduleNode::TRANSFER)
              ? static_cast<dsc2::TransferNode *>(coordPropInfo.refNode)
                    ->transferCoordinates_
              : getRelatedComputeCoord(
                    static_cast<dsc2::ComputeNode *>(coordPropInfo.refNode),
                    reverseCoordPropInfo, inputCoords, outputCoords, ldsIdx);
      if (buildFoldForAllocation(coordPropInfo, refCoordinate, allocNode)) {
        dsc2::computeLoopElemOffsetsFromCoordinates(
            currDsc, coordPropInfo.refNode,
            const_cast<dsc2::CoordinateType<CoordinateBaseType> &>(
                refCoordinate),
            allocNode,
            loopDistributionParamInfo.at(coordPropInfo.refNode).at(allocNode),
            coordPropInfo.dimsToPropagate, coordPropReportLevel_);
        if (allocNode->allocateCoordinates_.coreIdToWkSlice_.empty()) {
          for (auto &[userNode, refCount] : allocNode->allocUsers_) {
            if (!is_any_of(userNode->nodeType_, dsc2::ScheduleNode::COMPUTE,
                           dsc2::ScheduleNode::TRANSFER)) {
              continue;
            }
            // Opaque operations do not maintain dataInfos. Currently, it is not
            // possible to determine which input or output of an opaque op
            // corresponds to an allocateNode unless the associated data_connect
            // is also specified.
            if (userNode->nodeType_ != dsc2::ScheduleNode::COMPUTE ||
                !(static_cast<const dsc2::ComputeNode *>(userNode)
                      ->isOpaqueOp_)) {
              coordPropTracker.addPropInfo(
                  allocNode, const_cast<dsc2::ScheduleNode *>(userNode),
                  allocNode->allocateCoordinates_.getTensorDims(), "",
                  isAllocateIncoming(currDsc, allocNode, userNode));
            }
          }
        }
      }
    } else {
      if (coordPropReportLevel_ > 2) {
        std::cout << "[buildAndPropagateFold] Unsupported user node type "
                  << dsc2::ScheduleNode::nodeTypeToString.at(
                         coordPropInfo.nodeToFold->nodeType_);
      }
    }
  }
  return;
}

// ------------------------------------------------------------------------------------------------
// entry 371/382   level 5   scc 290   119 body lines
// unit: e371_replace_assign
// authority: ddc/transformations/automatic_shuffle/shuffle.cpp:788
// original: bool AutoShuffler::replace_assign(DesignSpaceConfig* dsc, ComputationBuilder& builder, ComputeNode* assign)
// class: AutoShuffler
// rust home: crates/compiler/deeptools/src/schedule/ddc/shuffle.rs
// ------------------------------------------------------------------------------------------------
bool e371_replace_assign(DesignSpaceConfig* dsc,
                                  ComputationBuilder& builder,
                                  ComputeNode* assign)
{
  DT_CHECK(assign != nullptr);
  DT_CHECK(assign->type_ == ComputeOpType::ASSIGN);
  DT_CHECK(is_any_of(assign->exUnit_, PE, SFP));
  DT_CHECK(assign->inputs_.size() == 1);
  DT_CHECK(assign->outputs_.size() == 1);
  // TODO: how to check if assign is a node in our current dsc?

  // Infer layout information
  auto parent = assign->getMutableParent();
  const auto& i_dinfo = assign->inputsLdsAndLoopOffsets_[0];
  const auto& o_dinfo = assign->outputsLdsAndLoopOffsets_[0];
  const auto& i_ds_info = dsc->labeledDs_[i_dinfo.myLdsIdx_];
  const auto& o_ds_info = dsc->labeledDs_[o_dinfo.myLdsIdx_];
  const PrimaryDsInfo& i_primary_ds_info =
      dsc->primaryDsInfo_.at(i_ds_info.dsType_);
  const PrimaryDsInfo& o_primary_ds_info =
      dsc->primaryDsInfo_.at(o_ds_info.dsType_);

  auto input_output_layouts =
      inferLayouts(i_primary_ds_info, o_primary_ds_info);

  AbstractLayout in_layout(input_output_layouts.first, i_ds_info.dataFormat_);
  AbstractLayout out_layout(input_output_layouts.second, o_ds_info.dataFormat_);
  auto shuffle = get_shuffle(in_layout, out_layout);

  DataEdge input_edge;
  input_edge.component = assign->inputs_[0];
  input_edge.dinfo = assign->inputsLdsAndLoopOffsets_[0];
  // check the other logics
  auto alloc = const_cast<dsc2::AllocateNode*>(
      dsc->getAllocation(input_edge.dinfo, input_edge.component, true));
  if (alloc) input_edge.allocation = alloc;
  std::vector<DataEdge> input_edges;
  input_edges.insert(input_edges.end(), in_layout.numSticks(), input_edge);

  const std::function<std::vector<DataEdge>(std::shared_ptr<GraphNode> node)>
      create_node_allocations = [&](std::shared_ptr<GraphNode> node) {
        std::vector<DataEdge> outputs;
        if (node->layout == out_layout) {
          DataEdge edge_from_output;
          edge_from_output.component = assign->outputs_[0];
          edge_from_output.dinfo = assign->outputsLdsAndLoopOffsets_[0];
          // check the other logics
          auto alloc = const_cast<dsc2::AllocateNode*>(dsc->getAllocation(
              edge_from_output.dinfo, edge_from_output.component, true));
          if (alloc) edge_from_output.allocation = alloc;
          edge_from_output.alloc_added = true;
          outputs.insert(outputs.begin(), node->layout.numSticks(),
                         edge_from_output);
        } else {
          // Word length is partially independent to bit size- e.g. a freshly
          // quantized f16->int4 now has type int4 but keeps word length of 2
          int wl =
              EnumsConversion::dataFormatsToBitWidth.at(node->layout.format);
          for (auto& dim : node->layout.slice_dims) {
            if (dim.is_dummy()) wl *= 2;
          }
          outputs = builder.allocate_sticks(node->layout.format, wl / 8.0,
                                            node->layout.numSticks());
        }
        return outputs;
      };

  std::vector<int> added_alloc_lds;
  std::function<void(ComputationOp&, std::vector<DataEdge>&, DataEdge&,
                     std::pair<std::vector<int>, int>&, bool)>
      do_codegen = [&builder, &dsc, &added_alloc_lds](
                       ComputationOp& op, std::vector<DataEdge>& in,
                       DataEdge& out,
                       std::pair<std::vector<int>, int>& input_output_stick_id,
                       bool out_added) {
        if (std::find(added_alloc_lds.begin(), added_alloc_lds.end(),
                      out.dinfo.myLdsIdx_) != added_alloc_lds.end()) {
          out.alloc_added = true;
        } else {
          added_alloc_lds.push_back(out.dinfo.myLdsIdx_);
          if (out.allocation.has_value() && !out.allocation.value()->getPrev())
            out.alloc_added = false;
        }

        for (int i = 0; i < in.size(); i++) {
          auto& myLds = dsc->labeledDs_.at(in.at(i).dinfo.myLdsIdx_);
          auto first_dim = dsc->getLayoutDims(myLds.ldsIdx_).at(0);
          int factor = 1;
          for (auto [stickDim, stickSize] :
               dsc->getCumulativeStickSizes(myLds.dsType_)) {
            if (stickDim == first_dim) factor = stickSize;
          }
          std::unordered_map<PrimaryDimTypes, int> jump_to_stick;
          jump_to_stick[first_dim] = factor * input_output_stick_id.first.at(i);
          for (auto core : dsc->coreIdsUsed_)
            for (int corelet = 0; corelet < dsc->numCoreletsUsed_; corelet++)
              in.at(i).dinfo.constEleOffsets_[core][corelet] = jump_to_stick;
        }
        auto& myLds = dsc->labeledDs_.at(out.dinfo.myLdsIdx_);
        auto first_dim = dsc->getLayoutDims(myLds.ldsIdx_).at(0);
        int factor = 1;
        for (auto [stickDim, stickSize] :
             dsc->getCumulativeStickSizes(myLds.dsType_)) {
          if (stickDim == first_dim) factor = stickSize;
        }
        std::unordered_map<PrimaryDimTypes, int> jump_to_stick;
        jump_to_stick[first_dim] = factor * input_output_stick_id.second;
        for (auto core : dsc->coreIdsUsed_)
          for (int corelet = 0; corelet < dsc->numCoreletsUsed_; corelet++)
            out.dinfo.constEleOffsets_[core][corelet] = jump_to_stick;

        op.codegen(builder, in, out);
      };

  codegen_generic<DataEdge>(input_output_layouts.first,
                            input_output_layouts.second, shuffle, input_edges,
                            create_node_allocations, do_codegen);

  builder.delete_node(assign);

  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 375/382   level 6   scc 212   86 body lines
// unit: e375_coordinateCapture
// authority: ddc/ddc_fold.cpp:1538
// original: void Ddc::coordinateCapture()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/fold.rs
// ------------------------------------------------------------------------------------------------
void e375_coordinateCapture()
{
  // TO DO: Split loops as needed to maintain affine properties.

  buildAndPropagateFold();

  if (coordFoldReportLevel_ > 0) {
    for (auto *node : currDsc->scheduleTree_.traverseTreeDFSMutable(
             nullptr,
             {dsc2::ScheduleNode::ALLOCATE, dsc2::ScheduleNode::COMPUTE,
              dsc2::ScheduleNode::TRANSFER})) {
      if (node->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
        std::cout << "\n================================";
        std::cout << "\nAllocateNode: " << node->name_;
        dsc2::AllocateNode *allocNode = static_cast<dsc2::AllocateNode *>(node);
        std::cout << "\nMemory: "
                  << EnumsConversion::senComponentsToString.at(
                         allocNode->component_);
        if (coordFoldReportLevel_ > 2) {
          allocNode->print(std::cout);
        }
        allocNode->allocateCoordinates_.debugPrint(std::cout,
                                                   coordFoldReportLevel_ > 1);
      } else if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
        std::cout << "\n================================";
        std::cout << "\nTransferNode: " << node->name_;
        dsc2::TransferNode *transferNode =
            static_cast<dsc2::TransferNode *>(node);
        std::cout << "\nsrc storage: "
                  << EnumsConversion::senComponentsToString.at(
                         transferNode->src_.storage_)
                  << ", src unit : "
                  << EnumsConversion::senComponentsToString.at(
                         transferNode->src_.unit_)
                  << ", dst storage: "
                  << EnumsConversion::senComponentsToString.at(
                         transferNode->dstVias_.front().loc_.storage_)
                  << ", dst unit: "
                  << EnumsConversion::senComponentsToString.at(
                         transferNode->dstVias_.front().loc_.unit_);
        if (coordFoldReportLevel_ > 2) {
          transferNode->print(std::cout);
        }
        transferNode->transferCoordinates_.debugPrint(
            std::cout, coordFoldReportLevel_ > 1);
      } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
        std::cout << "\n================================";
        std::cout << "\nComputeNode: " << node->name_;
        dsc2::ComputeNode *computeNode = static_cast<dsc2::ComputeNode *>(node);
        std::cout << "\nOp: "
                  << EnumsConversion::computeTypeToString.at(
                         computeNode->type_);
        std::cout << ", inputs:[ ";
        for (int i = 0, e = computeNode->inputsLdsAndLoopOffsets_.size(); i < e;
             ++i) {
          std::cout << "(storage="
                    << EnumsConversion::senComponentsToString.at(
                           computeNode->inputs_.at(i))
                    << ") ";
        }
        std::cout << "], outputs:[ ";
        for (int i = 0, e = computeNode->outputsLdsAndLoopOffsets_.size();
             i < e; ++i) {
          std::cout << "(storage="
                    << EnumsConversion::senComponentsToString.at(
                           computeNode->outputs_.at(i))
                    << ") ";
        }
        std::cout << "]";
        if (coordFoldReportLevel_ > 2) {
          computeNode->print(std::cout);
        }
        std::cout << "\n\nInput coordinates:";
        int i = 0;
        for (auto &inpCoord : computeNode->inputCoordinates_) {
          std::cout << "\n--------------------------"
                    << "\n  input= " << i++ << "\n--------------------------"
                    << std::endl;
          inpCoord.debugPrint(std::cout, coordFoldReportLevel_ > 1);
        }
        std::cout << "\n\nOutput coordinates:";
        computeNode->outputCoordinate_.debugPrint(std::cout,
                                                  coordFoldReportLevel_ > 1);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 376/382   level 6   scc 292   167 body lines
// unit: e376_performAutomaticShuffling
// authority: ddc/ddc_transformation.cpp:1855
// original: bool Ddc::performAutomaticShuffling()
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/transformation.rs
// ------------------------------------------------------------------------------------------------
bool e376_performAutomaticShuffling()
{
  using namespace shuffle;

  struct BuilderImpl : public shuffle::ComputationBuilder {
    DesignSpaceConfig *currDsc;
    dsc2::ComputeNode *assign;
    dsc2::BlockNode *assign_parent;
    SenComponents storage;
    dsc2::DataInfo dinfo_template;
    LabeledDsInfo dsinfo_template;
    Ddc *ddc_parent;
    int op_name_counter = 0;

    BuilderImpl(Ddc *parent, DesignSpaceConfig *currDsc,
                dsc2::ComputeNode *assign)
        : currDsc(currDsc),
          assign(assign),
          ddc_parent(parent),
          assign_parent(assign->getMutableParent()) {
      DT_CHECK(assign->type_ == ComputeOpType::ASSIGN);
      if (assign->exUnit_ == PE) {
        storage = SenComponents::PELRF;
      } else if (assign->exUnit_ == SFP) {
        storage = SenComponents::SFPLRF;
      } else {
        DT_ERROR("Unrecognized shuffle storage location");
      }
      dinfo_template = assign->inputsLdsAndLoopOffsets_[0];
      dsinfo_template = currDsc->labeledDs_[dinfo_template.myLdsIdx_];
    }

    virtual void delete_node(dsc2::ScheduleNode *node) override {
      node->getMutableParent()->deleteChildNode(currDsc, node);
    }

    std::vector<DataEdge> allocate_sticks(DataFormats format,
                                          double word_length,
                                          size_t n) override {
      std::vector<DataEdge> outputs;
      int old_last_idx = currDsc->labeledDs_.size() - 1;
      int num_intermidate_reg = n;
      auto input_comp = assign->inputs_.at(0);
      if (input_comp == SFPLRF || input_comp == PELRF) num_intermidate_reg = 1;
      for (int i = 0; i < num_intermidate_reg; i++) {
        auto name_suffix = std::to_string(name_counter);
        name_counter++;

        // use copy constructor to retain fields we don't overwrite
        LabeledDsInfo ds_info = dsinfo_template;
        ds_info.dsName_ = std::string("autoshuffle_reg_") + name_suffix;
        ds_info.dataFormat_ = format;
        ds_info.wordLength = word_length;
        int lds_idx = ddc_parent->addNewLds(&ds_info);
        DT_CHECK(currDsc->computeOp_.size() == 1);
        currDsc->computeOp_.back().interimLabeledDs.push_back(
            &currDsc->labeledDs_[lds_idx]);

        dsc2::DataInfo data_info = dinfo_template;  // copy constructor
        data_info.dataConnect_ = "autoshuffle_edge_" + name_suffix;
        data_info.myLdsIdx_ = lds_idx;

        PaddingFormType padding;  // TODO
        auto alloc = ddc_parent->constructAllocation(data_info, storage,
                                                     padding, assign);
        alloc->numBuffers_ = 1;
        alloc->removeAllocUser(assign);

        shuffle::DataEdge &edge = outputs.emplace_back();
        edge.allocation = alloc;
        edge.component = storage;
        edge.dinfo = std::move(data_info);
      }

      // TODO: Masoud's PR on stick packing included the following additional
      // code when inserting a new lds into labeledDS. But, looking at the
      // function definition, this seems redundant to our scenario.
      // However, please check with ALberto/Masoud. TODO: PC.
      // int new_last_idx = currDsc->labeledDs_.size() - 1;
      // ddc_parent->updateNodesWithNewLds(
      //     new_last_idx, old_last_idx,
      //     currDsc->scheduleTree_.getHeadMutable());
      return outputs;
    }

    const std::vector<int> expand_indices(std::vector<int> compact_indices,
                                          unsigned element_bit_width) {
      const int slice_bits = 128;
      int compact_indices_bit_width = slice_bits / compact_indices.size();

      // The downstream indices to instruction translations is not set up
      // to deal with, say, 8-bit pack on 16-bit values.
      DT_CHECK(element_bit_width <= compact_indices_bit_width);
      int scale = compact_indices_bit_width / element_bit_width;

      std::vector<int> out_indices;
      for (int i = 0; i < compact_indices.size(); i++) {
        for (int j = 0; j < scale; j++) {
          out_indices.push_back(compact_indices[i] * scale + j);
        }
      }

      return out_indices;
    }

    dsc2::ComputeNode *insert_packmerge(const DataEdge &in1,
                                        const DataEdge &in2, DataEdge &out,
                                        std::vector<int> indices,
                                        bool expand) override {
      auto packmerge_node = new dsc2::ComputeNode(*assign);
      packmerge_node->type_ = ComputeOpType::PACKMERGE;
      if (expand) {
        DataFormats in_type =
            currDsc->labeledDs_[in1.dinfo.myLdsIdx_].dataFormat_;
        int elem_bitwidth = EnumsConversion::dataFormatsToBitWidth.at(in_type);
        packmerge_node->instrAttribute_.indices_ =
            expand_indices(indices, elem_bitwidth);
      } else {
        packmerge_node->instrAttribute_.indices_ = indices;
      }

      // going from two inputs to one input
      packmerge_node->inputs_ = {in1.component, in2.component};
      packmerge_node->outputs_ = {out.component};
      packmerge_node->inputsLdsAndLoopOffsets_ = {in1.dinfo, in2.dinfo};
      packmerge_node->outputsLdsAndLoopOffsets_ = {out.dinfo};
      packmerge_node->name_ =
          assign->name_ + "_autoshuffle_" + std::to_string(op_name_counter++);

      if (out.allocation.has_value()) {
        out.allocation.value()->addAllocUser(packmerge_node);
      }
      for (auto &in : {in1, in2}) {
        if (in.allocation) {
          in.allocation.value()->addAllocUser(packmerge_node);
        }
      }

      assign_parent->addChildNode(packmerge_node, true, assign);
      out.insert_before(assign_parent, packmerge_node);

      return packmerge_node;
    }
  };

  AutoShuffler shuffler;
  bool success = false;
  for (auto &child : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::COMPUTE}, ALL, -1, -1)) {
    auto *computeNode = static_cast<dsc2::ComputeNode *>(child);

    // Return if exUnit is not PE/SFP.
    if (!is_any_of(computeNode->exUnit_, PE, SFP)) {
      continue;
    }
    if (!(computeNode->type_ == ComputeOpType::ASSIGN)) {
      continue;
    }

    BuilderImpl builder(this, currDsc, computeNode);

    // Query the shuffle sequence
    shuffler.replace_assign(currDsc, builder, computeNode);
    success = true;
  }

  return success;
}

// ------------------------------------------------------------------------------------------------
// entry 379/382   level 7   scc 352   109 body lines
// unit: e379_run_v1
// authority: ddc/ddcv1.cpp:3692
// original: bool Ddc::run_v1(SuperDsc& sdsc)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
bool e379_run_v1(SuperDsc& sdsc)
{
  sdsc_ = &sdsc;

  // map ddl to dsc2
  for (auto& dsc : sdsc.dscs_) {
    if (dsc.computeOp_.empty()) {
      continue;
    }
    currDsc = &dsc;

    if (verbose_ > 0) {
      std::cout << "[DDC] start working on DSC: " << dsc.name_ << std::endl;
    }

    metadata.clear();
    dataStageExplorationDone_ = false;
    latchDataIdCounter_ = 0;
    // temporary disable computing elemOffset from coordinate
    for (auto& computeOp : currDsc->computeOp_) {
      if (computeOp.opFuncName == OpFuncs::ReStickifyOpLx ||
          computeOp.opFuncName == OpFuncs::ReStickifyOpHBM) {
        datastageBasedElemOff = true;
        sdsc.datastageBasedElemOff = true;
      }
    }
    prepDsc();
    initGlobalData();

    attachToPrefilledSchedule();

    // get my ddl file..
    DdlConvertInterface ddlConvTnterface(dscGlobal, sdsc, dsc, metadata,
                                         verbose_);
    bool parseDdl = ddlConvTnterface.selectAndParseDdlTemplate();
    if (!parseDdl) {  // no suitable DDL found
      restoreDsc();
      return false;
    }
    bool has_auto_shuffling = performAutomaticShuffling();
    updateLdsIdxMetadata(dsc);
    cloneForOffsetAdjustment();

// TEMP: disabling stickpacking opt
#if 0
    // start DSC2 optimization...
    // packStickDim run successfully, avoid running performPeSfpWorkSplit
    if (dscGlobal.sysDef.coreArch >= RCUDD1A_ISA && packStickDim()) {
      // PeSfpWorkSplit is diabaled becouse the compression and expanssion part
      // makes at least two problems:
      // (1) the compression and expanssion part doesn't consider the
      // peSfpSplit_ feild in datastage.
      // (2) PeSfpWorkSplit adds new bottom datastage to the bottom datastge of
      // compression and expanssion part which is manually set, so the datastage
      // exploration cannot assign value for manually set bottom datastge.
      for (auto& ds : currDsc->dataStageParam_) {
        ds.second.ss_.peSfpSplit_.clear();
        ds.second.el_.peSfpSplit_.clear();
      }
      metadata.peSfpSplitDims_.clear();
    }
#endif
    performPeSfpWorkSplit();
    populateUnitTimeTransfers();
    createDataConnectMetadata();

    // Pre-exploration transformations
    if (metadata.transformationConfig_.enableMovingDataTransfer) {
      hoistTransfersUpForReuse();
    }

    transformFor4BsplatRead();
    transformRegToFifoOrLatch();
    transformForInterSliceRestickify();
    createDataConnectMetadata();
    minimizeAllocations(has_auto_shuffling);
    spreadDataInAllocate();
    unrollSpreadTransfers();
    setSizeForFixedSizeTransfers();
    exploreAssignDataStages();
    finalizeAllocateLayouts();

    coordinateMasking();

    // Apply post datastage-exploration transformations.
    transformRegToFifoOrLatch();
    unrollSymbolicTransfers();
    // finalize ScheduleTree
    dsc.setRelevantCompCoreCl();
    simplifyScheduleTree();
    createDataConnectMetadata();

    identifyBelowChunkBoundaryLoops();

    if (!datastageBasedElemOff) coordinateCapture();
    fillLoopOffsetsAndAddresses();
    adjustLoopOffsetsAndAddresses();

    finalizeOps();
    dsc.finalizeScheduleTree(sdsc, dscGlobal.sysDef);
    restoreDsc();

    if (verbose_ > 0) {
      std::cout << "\n[DDC] DSC2 successfully filled" << std::endl;
    }
    if (dscToDdl_) ddlConvTnterface.exportToDdl(std::cout);
  }

  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 381/382   level 8   scc 353   15 body lines
// unit: e381_run
// authority: ddc/ddcv1.cpp:3802
// original: void Ddc::run(SuperDsc& sdsc)
// class: Ddc
// rust home: crates/compiler/deeptools/src/schedule/ddc/v1.rs
// ------------------------------------------------------------------------------------------------
void e381_run(SuperDsc& sdsc)
{
  // If we only want to run dataOps through DCC, don't do any further work
  // here, return.
  if (dscGlobal.doDataOpTesting) return;

  bool useDdc = true;

  if (dscGlobal.ddcVersion > 0 && useDdc) {
    bool dscFilled = run_v1(sdsc);
    if (dscGlobal.ddcVersion == 2 && !dscFilled) {
      DT_ERROR("DDCv1 force-requested but DSC2 not filled for node: " +
               sdsc.name_);
    }
  }
}

