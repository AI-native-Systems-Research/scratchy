/************************************************************
 * IBM Confidential
 * (C) Copyright IBM Corp. 2023, 2025
 ************************************************************/

/*
 * Description:
 *
 */

#ifndef DDC_METADATA_H_
#define DDC_METADATA_H_

#include <dsc/designSpaceConfig.h>

#include <optional>
#include <unordered_map>
namespace ddc {

const std::unordered_set<SenComponents> memories = {LX,    L0,    PELRF, SFPLRF,
                                                    PTARF, PTXRF, PTIRF, HBM};
const std::unordered_set<SenComponents> aboveLxMemories = {L3, HBM};

struct FailedAlloc {
  SenComponents comp;
  int core;
  int corelet;
  int row;
};

struct Metadata {
  struct Datastage {
    struct Constraints {
      bool mustBeMultiple_ = false;  // multiple of ref datastage if present,
                                     // otherwise multiple of min
      MetaDimKind loopDimKind_ = MetaDimKind::Count;
      std::optional<float> min_, max_;
      std::optional<std::set<float>> values_;
      bool cannotBeSymbolic_ = false;
      inline void updateMin(float newVal) {
        min_ = min_ ? std::max(*min_, newVal) : newVal;
      }
      inline void updateMax(float newVal) {
        max_ = max_ ? std::min(*max_, newVal) : newVal;
      }
      inline void updateValues(std::set<float> newVals) {
        values_ = values_ ? set_intersect(*values_, newVals) : newVals;
      }
      void dump() const {
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
    };
    // int key is reference datastage, -1 for absolute constraints
    // set key contains the dimensions the constraints apply to
    std::unordered_map<int, std::map<std::set<PrimaryDimTypes>, Constraints>>
        constraints_;
    bool strategyMinimize_ = true;  // false == maximize
    bool allowEpilogue_ = false;
    std::unordered_map<PrimaryDimTypes, int> relevantDimsAndNumerator_;
    int nearestNumeratorIdx_ = -1;
  };
  std::unordered_map<int, Datastage> datastages_;

  typedef std::pair<PadType, PadType> TransferAccessPatternType;
  typedef std::map<PrimaryDimTypes, TransferAccessPatternType>
      TransferAccessPatternPerDimType;

  struct DataTransfer {
   public:
    bool apply_row_offset_src_ = false;
    bool apply_row_offset_dst_ = false;
    bool apply_pe_sfp_split_offset_src_ = false;
    std::vector<int> apply_pe_sfp_split_offset_dest_;
    bool replicated_ = false;
    int offset_src_ = 0;
    std::map<int, int> offset_dest_;  // (dst -> offset)
    int force_num_elements_ = -1;
    void setAccessPattern(PrimaryDimTypes dimVal,
                          TransferAccessPatternType accessPattern) {
      accessPatternPerDim_[dimVal] = accessPattern;
    }
    bool hasAccessPattern(PrimaryDimTypes dimVal) {
      return accessPatternPerDim_.count(dimVal);
    }
    bool hasAccessPattern() { return !accessPatternPerDim_.empty(); }
    TransferAccessPatternType getAccessPattern(PrimaryDimTypes dimVal);
    std::string getAccessPatternAsStr(PrimaryDimTypes dimVal);
    TransferAccessPatternPerDimType& getMutableAccessPatternList() {
      return accessPatternPerDim_;
    };
    const TransferAccessPatternPerDimType& getAccessPatternList() const {
      return accessPatternPerDim_;
    };
    void dump();

   private:
    TransferAccessPatternPerDimType accessPatternPerDim_;
  };
  std::unordered_map<const dsc2::TransferNode*, DataTransfer> datatransfers_;

  struct Allocation {
    std::map<int, dsc2::AllocateNode*> ldsIdxAndAllocNode;
    std::map<int, dsc2::AllocateNode*> consIdAndAllocNode;
    std::unordered_map<dsc2::ComputeNode*, dsc2::AllocateNode*>
        compAndAllocNode;
  };
  std::unordered_map<SenComponents, Allocation> newAllocations_;
  std::vector<std::vector<dsc2::AllocateNode*>> shadowAllocations_;

  struct ExternalTransfer {
    std::unique_ptr<dsc2::TransferNode> transfer_;
    std::unique_ptr<dsc2::AllocateNode> allocate_;
    ExternalTransfer(dsc2::TransferNode* transferNode,
                     dsc2::AllocateNode* allocateNode)
        : transfer_(transferNode), allocate_(allocateNode) {}
  };
  std::vector<ExternalTransfer> externalTransfers_;
  std::map<std::pair<int, SenComponents>, std::string*>
      prefilledExternalTransferToDataConnectToFill_;
  std::set<const dsc2::ScheduleNode*> externalNodes_;
  std::set<dsc2::TransferNode*> TransferNodesInterSliceTranspose_;

  struct DataConnect {
    std::vector<dsc2::ScheduleNode*> producers_;
    std::vector<dsc2::ScheduleNode*> consumers_;

    void insertProducer(dsc2::ScheduleNode* node) {
      if (!is_any_of(node, producers_)) {
        producers_.push_back(node);
      }
    }

    void insertConsumer(dsc2::ScheduleNode* node) {
      if (!is_any_of(node, consumers_)) {
        consumers_.push_back(node);
      }
    }

    std::unordered_set<const dsc2::LoopNode*> getProducerLoops() const {
      return getLoops(producers_);
    }
    std::unordered_set<const dsc2::LoopNode*> getConsumerLoops() const {
      return getLoops(consumers_);
    }

    void print(std::ostream& outs) const {
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

   private:
    std::unordered_set<const dsc2::LoopNode*> getLoops(
        const std::vector<dsc2::ScheduleNode*>& baseNodes) const {
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
  };
  std::unordered_map<std::string, DataConnect> dataConnects_;

  struct OpaqueOp {
    std::unordered_map<std::string, dsc2::AllocateNode*> inOutRegAllocs_;
    std::vector<std::string> internalRegs_;
    int internalRegsWithUnroll_ = 0;
    dsc2::AllocateNode* internalRegAlloc_ = nullptr;
    int max_unroll_ = 1;
    int ldsIdx_ = -1;
  };
  std::unordered_map<dsc2::ComputeNode*, OpaqueOp> opaqueOps_;

  std::unordered_map<dsc2::SyncNode*, const dsc2::AllocateNode*> implicitSyncs_;

  std::unordered_map<PrimaryDimTypes, std::vector<const dsc2::LoopNode*>>
      dimToCoreChunkLoops_;

  const int core_dstgid = 0;
  const int chunk_dstgid = 1;
  PrimaryDimTypes rowSplitDim = PrimaryDimTypesCount;
  std::set<PrimaryDimTypes> clSplitDims_;
  std::set<PrimaryDimTypes> peSfpSplitDims_;
  std::unordered_map<dsc2::ScheduleNode*, std::vector<dsc2::ScheduleNode*>>
      nodeCloningMap_;
  bool discardAboveLxSchedule_ = false;
  dsc2::BlockNode* belowLxScheduleInsertBlock = nullptr;

  // dsc backup for fields to restore
  OpFuncs opFuncBackup_ = OpFuncs::NONE;

  // reinitialize structure
  void clear() {
    this->~Metadata();
    new (this) Metadata();
  }

  struct DDCTransformationConfigT {
    bool enableMovingDataTransfer = true;
  } transformationConfig_;

  // map index of labeledDs before DDC to after DDC optimizations
  std::map<int, int> ldsIdxAfterDdc;

  // maps index of interm tensor to the reference external tensor
  std::map<int, int> intermLdsIdxToExtLds;
};
}  // namespace ddc
#endif
