/************************************************************
 * IBM Confidential
 * (C) Copyright IBM Corp. 2018, 2025
 ************************************************************/

/*
 * Description:
 *
 */

#ifndef DSC2_DEFN_
#define DSC2_DEFN_

#include "dscdefn.h"
#include "util/foldManager/foldInfrastructure.h"
#include "util/foldManager/mapWithFMHelper.h"
#include "sys-arch-spec/sysdef.h"
#include "util/variabledefinition/VariableDefinition.h"
#include "util/dt_exception.hpp"
#include "util/utils.h"

class DesignSpaceConfig;
namespace ddc {
class Ddc;
class DdlConversion;
}  // namespace ddc
class L3DlOpsScheduler;
class PerfDSCToSDSC;
class SuperDsc;

namespace dsc2 {

// GTR (group tag register) info
struct GroupTagRegInfo {
  int groupId_ = -1;
  int numSharers_ = -1;
};

// group steady-state and epilogue data staging params
struct DataStage {
  DataStructDims ss_;
  DataStructDims el_;
  inline std::string name() const { return ss_.name_; };
};

struct ConstantInfo {
  DataFormats dataFormat_ = DataFormats::INVALID;
  std::string name_;
  FoldManager<std::vector<int64_t>>
      data_;  // core/corelet/sdsc folds, values encoded in the specified format
  bool isDataSymbolic_ = false;
  std::map<SenComponents, dsc2::AllocateNode*> allocations_;

  ConstantInfo& operator=(const ConstantInfo& rhs) {
    dataFormat_ = rhs.dataFormat_;
    name_ = rhs.name_;
    allocations_ = rhs.allocations_;
    data_.clone(rhs.data_);
    return *this;
  }
};

class LoopNode;
class BlockNode;

enum CoordinateCategory {
  UNKNOWN_COORD = 0,
  SPATIAL_COORD = 1,
  TEMPORAL_COORD = 2,
  ELEM_ARR_COORD = 3
};

enum class CoordinateFoldPosition : int { Core = 0, Corelet = 1, RowSplit = 2 };

template <typename Dtype>
class CoordinateType {
 public:
  CoordinateType() {}
  ~CoordinateType() { clear(); }

  void clear() {
    for (auto& [dim, cfm] : coordinates_) {
      // The tree of fold functions will be cleared automatically when
      // coordinates_ is cleared. Only the dimProperties part needs special
      // clearing.
      fm_dim_prop dimProps;
      cfm.getAllDimProFromPos(0, dimProps);
      for (auto& [dimPtr, funcType] : dimProps) {
        delete dimPtr;
      }
    }
    coordinates_.clear();
    numOfSpatialFolds_.clear();
    numOfTemporalFolds_.clear();
    numOfElemArrFolds_.clear();
    padding_.clear();
    foldConstructed_ = false;
  }

  // Clears only the fold part for a given dimension.
  //
  // Note: Padding is not cleared.
  void clearFoldForDim(PrimaryDimTypes dim) {
    if (numOfSpatialFolds_.count(dim)) {
      numOfSpatialFolds_.at(dim) = 0;
    }
    if (numOfTemporalFolds_.count(dim)) {
      numOfTemporalFolds_.at(dim) = 0;
    }
    if (numOfElemArrFolds_.count(dim)) {
      numOfElemArrFolds_.at(dim) = 0;
    }
    if (coordinates_.count(dim)) {
      coordinates_.at(dim).reset();
    }
  }

  void completeFoldConstruction() { foldConstructed_ = true; }
  bool foldConstructed() const { return foldConstructed_; }
  void addFold(PrimaryDimTypes dim, CoordinateCategory coordCat,
               int foldCardinality, std::string foldLabel, Dtype alpha,
               Dtype beta, int pos) {
    FoldDimProp* foldDim = new FoldDimProp(foldCardinality, foldLabel);

    auto& foldForCurrDim = coordinates_[dim];
    foldForCurrDim.buildAffineDim(foldDim, pos);
    foldForCurrDim.insertAlphaBeta(alpha, beta, pos);

    switch (coordCat) {
      case SPATIAL_COORD:
        ++numOfSpatialFolds_[dim];
        break;
      case TEMPORAL_COORD:
        ++numOfTemporalFolds_[dim];
        break;
      case ELEM_ARR_COORD:
        ++numOfElemArrFolds_[dim];
        break;
      default:
        DT_ERROR("[CoordinateType::addFold] Unsupported coordinate category.");
    }
  }

  int getNumOfSpatialFolds(PrimaryDimTypes dim) const {
    return numOfSpatialFolds_.count(dim) ? numOfSpatialFolds_.at(dim) : 0;
  }
  int getNumOfTemporalFolds(PrimaryDimTypes dim) const {
    return numOfTemporalFolds_.count(dim) ? numOfTemporalFolds_.at(dim) : 0;
  }
  int getNumOfElemArrFolds(PrimaryDimTypes dim) const {
    return numOfElemArrFolds_.count(dim) ? numOfElemArrFolds_.at(dim) : 0;
  }
  CoordinateCategory getCoordinateCategoryOfPos(PrimaryDimTypes dim,
                                                int pos) const {
    auto coord = coordinates_.at(dim);
    DT_CHECK(pos < coord.getNumDims());
    if (pos < getNumOfSpatialFolds(dim)) {
      return CoordinateCategory::SPATIAL_COORD;
    } else if (pos < getNumOfSpatialFolds(dim) + getNumOfTemporalFolds(dim)) {
      return CoordinateCategory::TEMPORAL_COORD;
    } else {
      return CoordinateCategory::ELEM_ARR_COORD;
    }
  }

  CoordinateType<Dtype>& operator=(const CoordinateType<Dtype>& rhs) {
    if (this == &rhs) return *this;

    this->clear();
    for (auto& [dim, cfm] : rhs.coordinates_) {
      FoldManager<Dtype>& fm = const_cast<FoldManager<Dtype>&>(cfm);
      for (int i = fm.getNumDims() - 1; i >= 0; --i) {
        auto coordCat = rhs.getCoordinateCategoryOfPos(dim, i);
        Dtype alpha, beta;
        fm.getAlphaBeta(alpha, beta, i);
        addFold(dim, coordCat, fm.getFoldDimSize(i),
                fm.getFoldDimProp(i)->Label(), alpha, beta, 0);
      }
    }
    this->padding_ = rhs.padding_;
    this->coreIdToWkSlice_ = rhs.coreIdToWkSlice_;
    this->foldConstructed_ = rhs.foldConstructed_;
    return *this;
  }

  CoordinateType(const CoordinateType<Dtype>& rhs) { *this = rhs; }

  bool operator==(const CoordinateType<Dtype>& rhs) const {
    if (this->coordinates_.size() != rhs.coordinates_.size()) {
      return false;
    }
    for (auto& [coordDim, lhsFm] : this->coordinates_) {
      if (!rhs.coordinates_.count(coordDim)) {
        return false;
      }
      if (getNumOfSpatialFolds(coordDim) !=
              rhs.getNumOfSpatialFolds(coordDim) ||
          getNumOfTemporalFolds(coordDim) !=
              rhs.getNumOfTemporalFolds(coordDim) ||
          getNumOfElemArrFolds(coordDim) !=
              rhs.getNumOfElemArrFolds(coordDim) ||
          getPadding(coordDim) != rhs.getPadding(coordDim)) {
        return false;
      }
      if (!(lhsFm == rhs.coordinates_.at(coordDim))) {
        return false;
      }
    }
    if (this->coreIdToWkSlice_ != rhs.coreIdToWkSlice_) {
      return false;
    }
    return true;
  }

  // TEMP. remove these two after fold type vector is used.
  void setNumOfTemporalFoldPerDim(PrimaryDimTypes dim, int num) {
    numOfTemporalFolds_.at(dim) = num;
  }
  void setNumOfElemArrFoldPerDim(PrimaryDimTypes dim, int num) {
    numOfElemArrFolds_.at(dim) = num;
  }

  CoordinateCategory getFoldCategory(PrimaryDimTypes dim, int pos) {
    if (pos < 0 || pos >= coordinates_.at(dim).getNumDims()) {
      return CoordinateCategory::UNKNOWN_COORD;
    }
    if (pos < getNumOfSpatialFolds(dim)) {
      return CoordinateCategory::SPATIAL_COORD;
    }
    if (pos < (getNumOfSpatialFolds(dim) + getNumOfTemporalFolds(dim))) {
      return CoordinateCategory::TEMPORAL_COORD;
    }
    return CoordinateCategory::ELEM_ARR_COORD;
  }

  void setPadding(const PrimaryDimTypes dim, const PadType pad) {
    padding_.setPadding(dim, pad);
  }

  void setPadding(const PaddingFormType padding) { padding_ = padding; }

  PadType getPadding(PrimaryDimTypes dim) const {
    return padding_.getPadding(dim);
  }
  PaddingFormType getPadding() const { return padding_; }
  std::vector<PrimaryDimTypes> getTensorDims() const {
    std::vector<PrimaryDimTypes> coordDims;
    for (auto& [coordDim, _] : coordinates_) {
      coordDims.push_back(coordDim);
    }
    return coordDims;
  }

  bool hasCoordForDim(const PrimaryDimTypes dim) {
    return coordinates_.count(dim) != 0;
  }

  void printCoordinates(std::ostream& out, bool printContent = false,
                        std::string ps = "") const {
    auto QUOTE = [](std::string str) { return "\"" + str + "\""; };
    const std::string indent = "  ", ps1 = ps + indent, ps2 = ps1 + indent,
                      ps3 = ps2 + indent;
    out << "\n" << ps << QUOTE("coordinates_") << " : {";
    out << "\n" << ps1 << QUOTE("coordInfo") << " : {\n";
    int c = coordinates_.size();
    for (auto& [currDim, foldManager] : coordinates_) {
      int foldLevelcount = foldManager.getNumDims();
      out << ps2 << QUOTE(EnumsConversion::primaryDimToString.at(currDim))
          << " : {\n";
      out << ps3 << QUOTE("spatial") << " : " << getNumOfSpatialFolds(currDim)
          << ",\n"
          << ps3 << QUOTE("temporal") << " : " << getNumOfTemporalFolds(currDim)
          << ",\n"
          << ps3 << QUOTE("elemArr") << " : " << getNumOfElemArrFolds(currDim)
          << ",\n";

      out << ps3 << QUOTE("padding") << " : "
          << QUOTE(EnumsConversion::padTypeToString.at(getPadding(currDim)))
          << ",\n";
      out << ps3 << QUOTE("folds") << " : ";
      foldManager.print(out, ps3, printContent);
      out << "\n" << ps2 << "}";
      if (--c > 0) {
        out << ", ";
      }
      out << "\n";
    }
    // End of coordInfo
    out << ps1 << "},\n";

    out << ps1 << QUOTE("coreIdToWkSlice_") << " : { \n";
    int i = 1;
    for (const auto& kv : coreIdToWkSlice_) {
      out << ps2 << QUOTE(std::to_string(kv.first)) << " : { ";
      int j = 1;
      for (auto kv2 : kv.second) {
        out << QUOTE(EnumsConversion::primaryDimToString.at(kv2.first)) << " : "
            << kv2.second;
        if (j < kv.second.size()) {
          out << ", ";
        }
        j++;
      }
      out << " }";
      if (i < coreIdToWkSlice_.size()) {
        out << ", ";
      }
      out << "\n";
      i++;
    }
    out << ps1 << "} \n";

    // End of coordinates_
    out << ps << "}\n";
    out.flush();
  }

  /**
   * @brief This method imports coordinates for scheduleNodes exported with
   * above print method.
   *
   * @param json
   */
  void dsc_import_json(const json11::Json& json) {
    auto& jsonMap = json.object_items();
    for (auto& [dimStr, coordJson] : jsonMap.at("coordInfo").object_items()) {
      PrimaryDimTypes dim = EnumsConversion::stringToPrimaryDim.at(dimStr);
      auto& coordMap = coordJson.object_items();
      numOfSpatialFolds_[dim] = coordMap.at("spatial").int_value();
      numOfTemporalFolds_[dim] = coordMap.at("temporal").int_value();
      numOfElemArrFolds_[dim] = coordMap.at("elemArr").int_value();
      setPadding(dim, EnumsConversion::stringToPadType.at(
                          coordMap.at("padding").string_value()));

      auto& props =
          coordMap.at("folds").object_items().at("dim_prop_attr").array_items();
      std::deque<const FoldDimProp*> foldDimProps;
      for (int i = 0; i < props.size(); i++) {
        // Construct an empty placeholder.
        FoldDimProp* foldDimProp = new FoldDimProp();
        // Import from json.
        foldDimProp->importFromJson(props.at(i));
        foldDimProps.push_back(foldDimProp);
      }
      coordinates_[dim].importFromJson(coordMap.at("folds"), foldDimProps);
    }

    coreIdToWkSlice_.clear();
    for (const auto& map2 : jsonMap.at("coreIdToWkSlice_").object_items()) {
      int coreId = std::stoi(map2.first);
      coreIdToWkSlice_[coreId];
      auto& myWkSlice = coreIdToWkSlice_.at(coreId);
      for (const auto& map3 : map2.second.object_items()) {
        PrimaryDimTypes dim =
            EnumsConversion::stringToPrimaryDim.at(map3.first);
        myWkSlice[dim] = map3.second.int_value();
      }
    }
  }

  void debugPrint(std::ostream& out, bool printContent = false,
                  std::string ps = "") const {
    out << "\nDDC Coordinates<int64_t>: ";
    out << coordinates_.size() << " coordinate entries";
    for (auto& [currDim, foldManager] : coordinates_) {
      int foldLevelcount = foldManager.getNumDims();
      out << "\n\nPrimary Dim= "
          << EnumsConversion::primaryDimToString.at(currDim);
      auto foldDims = foldManager.getFoldDimProp();
      for (int i = 0; i < foldLevelcount; ++i) {
        out << "\n  Fold dimension= ";
        foldDims.at(i)->print(out);
        std::vector<FoldFunction<Dtype>*> foldFunctions;
        foldManager.collectFoldFunctionAtLevel(i, foldFunctions);
        for (auto& ff : foldFunctions) {
          if (ff->Type() == FoldFunction<Dtype>::FuncType::WkSplit_leaf) {
            ff->printMetaData(out);
          } else if (ff->Type() ==
                     FoldFunction<Dtype>::FuncType::Affine_nonleaf) {
            AffineFoldFunction_NonLeaf<Dtype>* affineFf =
                static_cast<AffineFoldFunction_NonLeaf<Dtype>*>(ff);
            out << "\n    Affine:";
            affineFf->printMetaData(out);
          } else if (ff->Type() == FoldFunction<Dtype>::FuncType::Affine_leaf) {
            AffineFoldFunction_Leaf<Dtype>* affineFf =
                static_cast<AffineFoldFunction_Leaf<Dtype>*>(ff);
            out << "\n    Affine: ";
            affineFf->printMetaData(out);
          } else if (ff->Type() ==
                     FoldFunction<Dtype>::FuncType::Constant_nonleaf) {
            out << "\n    Constant ";
          }
        }
      }

      out << "\n  #Spatial  = " << getNumOfSpatialFolds(currDim)
          << "\n  #Temporal = " << getNumOfTemporalFolds(currDim)
          << "\n  #ElemArr  = " << getNumOfElemArrFolds(currDim);

      out << "\n  Padding: { ("
          << EnumsConversion::primaryDimToString.at(currDim) << ", "
          << EnumsConversion::padTypeToString.at(getPadding(currDim)) << ") }";
    }

    if (!coreIdToWkSlice_.empty()) {
      out << "\n  coreIdToWkSlice_ : { \n";
      int i = 1;
      for (const auto& kv : coreIdToWkSlice_) {
        out << "    " << std::to_string(kv.first) << " : { ";
        int j = 1;
        for (auto kv2 : kv.second) {
          out << EnumsConversion::primaryDimToString.at(kv2.first) << " : "
              << kv2.second;
          if (j < kv.second.size()) {
            out << ", ";
          }
          j++;
        }
        out << " }";
        if (i < coreIdToWkSlice_.size()) {
          out << ", ";
        }
        out << "\n";
        i++;
      }
      out << "  } \n";
    }
    out.flush();
  }

  std::map<PrimaryDimTypes, FoldManager<Dtype>> coordinates_;
  std::map<int, std::map<PrimaryDimTypes, int>> coreIdToWkSlice_;

 private:
  bool foldConstructed_ = false;
  std::map<PrimaryDimTypes, int> numOfSpatialFolds_;
  std::map<PrimaryDimTypes, int> numOfTemporalFolds_;
  std::map<PrimaryDimTypes, int> numOfElemArrFolds_;
  PaddingFormType padding_;
};

#define CoordinateBaseType int64_t

class ScheduleNode {
 public:
  enum NodeType {
    INVALID,
    BLOCK,
    LOOP,
    TRANSFER,
    COMPUTE,
    SYNC,
    CONDITION,
    ALLOCATE,
    STICKMASK,
  };
  static const std::map<NodeType, std::string> nodeTypeToString;
  static const std::map<std::string, NodeType> stringToNodeType;

  const NodeType nodeType_ = INVALID;
  std::string name_;

  inline const BlockNode* getPrev() const { return prev_; };
  inline BlockNode* getMutableParent() { return prev_; };
  const LoopNode* getOwnerLoop() const;
  LoopNode* getMutableOwnerLoop();
  const LoopNode* getParentDimLoop(PrimaryDimTypes dim) const;
  LoopNode* getMutableParentDimLoop(PrimaryDimTypes dim);
  void insertLoopAbove(LoopNode* nodeToAdd);
  bool isNodeRelevant(SenComponents comp, int clId = -1, int coreId = -1) const;
  std::map<int, std::set<int>> getRelevantCoreCl(
      SenComponents comp = SenComponents::ALL) const;
  std::set<SenComponents> getRelevantComps(int coreId = -1,
                                           int clId = -1) const;

  void moveNode(DesignSpaceConfig* ownerDsc, BlockNode* newParent,
                bool addBefore = false,
                const ScheduleNode* siblingRefNode = nullptr);
  bool isBlockNode() {
    return nodeType_ == BLOCK || nodeType_ == LOOP || nodeType_ == CONDITION;
  }
  ScheduleNode(NodeType nodeType) : nodeType_(nodeType) {}
  virtual ~ScheduleNode() {}
  virtual ScheduleNode* clone() const = 0;

  struct Size {
    PrimaryDimTypes dim_;
    int size_ = -1;

    Size() = default;
    Size(PrimaryDimTypes dim, int size) : dim_(dim), size_(size) {};
    Size(const std::pair<PrimaryDimTypes, int>& dimSize)
        : Size(dimSize.first, dimSize.second) {};
    bool operator==(const Size& x) const {
      return dim_ == x.dim_ && size_ == x.size_;
    };
    bool operator!=(const Size& x) const { return !(*this == x); };
  };
  struct UnitView {
    struct LoopInfo {
      const LoopNode* loop_ = nullptr;
      PrimaryDimTypes dim_ = PrimaryDimTypesCount;
      int sizeIdx_ = -1;
      int elemOffset_ = -1;
    };
    std::vector<Size> sizesNoGaps_;
    std::vector<LoopInfo> compositeLoops_;
    std::vector<LoopInfo> outerLoops_;
    std::map<int, std::vector<Size>> sizesWithGaps_;  // per core
    const std::vector<Size>& getSizesForCoreId(int coreId) const;
    void print(std::ostream& out, int level) const;
  };

 protected:
  BlockNode* prev_ = nullptr;
  std::map<SenComponents, std::map<int, std::set<int>>> relevantComps_;

  friend class ::DesignSpaceConfig;
  friend class ::ddc::Ddc;
  friend class ::L3DlOpsScheduler;
  friend class ScheduleTree;
  friend class BlockNode;
  friend void transformLxZeroPadInfoInScheduleTree(SuperDsc& mySDsc);
};

class BlockNode : public InheritWithClone<ScheduleNode, BlockNode> {
 protected:
  using BaseClass::BaseClass;
  struct VectorOfChildren : public std::vector<std::unique_ptr<ScheduleNode>> {
    VectorOfChildren() = default;
    VectorOfChildren(VectorOfChildren&&) = default;
    VectorOfChildren& operator=(VectorOfChildren&&) = default;
    VectorOfChildren(const VectorOfChildren&) {}  // do nothing on purpose
    // when copying, it is up to the caller to manually insert copies of the
    // children
    VectorOfChildren& operator=(const VectorOfChildren&) = delete;
  };
  VectorOfChildren next_;

 public:
  std::vector<const ScheduleNode*> getNextView(SenComponents comp,
                                               int clId = -1,
                                               int coreId = -1) const;
  virtual void addChildNode(ScheduleNode* nodeToAdd, bool addBefore = false,
                            const ScheduleNode* siblingRefNode = nullptr);
  virtual void moveChildNode(DesignSpaceConfig* ownerDsc,
                             ScheduleNode* nodeToMove, BlockNode* newParent,
                             bool addBefore = false,
                             const ScheduleNode* siblingRefNode = nullptr);
  virtual void insertPerfectlyNestedBlockNode(BlockNode* nodeToAdd);
  virtual void moveChildren(BlockNode* toNode);
  void deleteChildNode(DesignSpaceConfig* ownerDsc, ScheduleNode* nodeToDelete,
                       bool nonDestructive = false);
  BlockNode() : BaseClass(BLOCK) {}
  virtual ~BlockNode() {}
  friend class ::DesignSpaceConfig;
  friend class ::ddc::Ddc;
  friend class ::L3DlOpsScheduler;
  friend class ScheduleTree;
  friend class ScheduleNode;
};

struct LoopNode : public InheritWithClone<BlockNode, LoopNode> {
  // struct PrimaryDimAndKind {
  //   PrimaryDimTypes dim_ = PrimaryDimTypes::PrimaryDimTypesCount;
  //   MetaDimKind kind_ = MetaDimKind::Unpadded;
  //   PrimaryDimAndKind(
  //       PrimaryDimTypes dim = PrimaryDimTypes::PrimaryDimTypesCount,
  //       MetaDimKind kind = MetaDimKind::Unpadded)
  //       : dim_(dim), kind_(kind) {}
  // };

  int numId_ = -1;
  int denId_ = -1;
  std::vector<PrimaryDimAndKind> dims_;  // ordered inner to outer
  std::map<PrimaryDimTypes, std::vector<VariableSymbol>>
      loopCountSymbolIds_;  // single value for pure symbolic and pivot dims,
                            // multiple entries (max-pivot) for irregular dims

  LoopNode() : BaseClass(LOOP) {}

  // Adding a new constructor to facilate DSC2.1 to Dataflow xlator to create
  // new loop nodes (dummy) for implicit loops involved in data transfer. Using
  // the default constructor requires updating dims_ in a later step, and const
  // qualifier doesn't allow it.
  LoopNode(int numId, int denId, const std::vector<PrimaryDimAndKind>& dims,
           bool isParametricLoop = false)
      : BaseClass(LOOP),
        numId_(numId),
        denId_(denId),
        isParametricLoop_(isParametricLoop) {
    dims_.insert(dims_.begin(), dims.begin(), dims.end());
  }

  bool isDimSymbolic(PrimaryDimTypes dim) const {
    return loopCountSymbolIds_.count(dim) != 0;
  }

  bool isParametricLoop() const { return isParametricLoop_; }
  void markAsParametricLoop() { isParametricLoop_ = true; }
  int parametricIterCount(const DesignSpaceConfig* currDsc, int clId,
                          SenComponents comp, int rowId = -1) const;
  int parametricLdsIdx() const { return parametricLdsIdx_; }
  void setParametricLdsIdx(int idx) { parametricLdsIdx_ = idx; }
  int parametricStride(const DesignSpaceConfig* currDsc) const;

  bool hasLoopDim(PrimaryDimTypes dim) const;

  void print(std::ostream& out, int level = 0) const;

  friend class ::DesignSpaceConfig;
  friend class ::ddc::Ddc;
  friend class ScheduleTree;
  friend class ScheduleNode;

 private:
  bool isParametricLoop_ = false;
  int parametricLdsIdx_ = -1;
};

class ScheduleTree {
 private:
  LoopNode head_;

 public:
  inline void clear() { head_.next_.clear(); };
  inline bool empty() const { return head_.next_.empty(); };

  ScheduleTree() { head_.denId_ = 0; };  // core datastage
  // move constructor
  ScheduleTree(ScheduleTree&& old) = default;
  // copy constructor (uses copyFrom)
  ScheduleTree(const ScheduleTree& old) { copyFrom(old); }
  ScheduleTree& copyFrom(const ScheduleTree& old);
  // delete operator= so that copy needs to be more voluntary
  ScheduleTree& operator=(const ScheduleTree& old) = delete;
  const LoopNode* getHead() const { return &head_; }
  LoopNode* getHeadMutable() { return &head_; }
  std::vector<const ScheduleNode*> traverseTreeDFS(
      const ScheduleNode* startNode = nullptr,
      const std::unordered_set<dsc2::ScheduleNode::NodeType>& nodeTypes = {},
      SenComponents comp = ALL, int clId = -1, int coreId = -1,
      int maxLoopDepth = -1,
      const std::unordered_set<const dsc2::ScheduleNode*>& excludeList = {})
      const;
  std::vector<ScheduleNode*> traverseTreeDFSMutable(
      ScheduleNode* startNode = nullptr,
      const std::unordered_set<dsc2::ScheduleNode::NodeType>& nodeTypes = {},
      SenComponents comp = ALL, int clId = -1, int coreId = -1,
      int maxLoopDepth = -1,
      const std::unordered_set<const dsc2::ScheduleNode*>& excludeList = {});
};

struct LoopCond {
  enum CondValType { INT, FIRST, LAST };
  static const std::unordered_map<CondValType, std::string> condValTypeToString;
  static const std::unordered_map<std::string, CondValType> stringToCondValType;

  const LoopNode* loopComp_ = nullptr;
  PrimaryDimTypes dim_ = PrimaryDimTypesCount;
  CondOp condOp_ = CondOp::DEFAULT;
  CondValType condValType_ = CondValType::INT;
  int condValInt_ = -1;

  LoopCond(const LoopNode* loopComp, PrimaryDimTypes dim, CondOp condOp,
           LoopCond::CondValType condValType, int condValInt = -1)
      : loopComp_(loopComp),
        dim_(dim),
        condOp_(condOp),
        condValType_(condValType),
        condValInt_(condValInt) {};
  LoopCond() = default;
};

struct LoopCondComposite {
  std::vector<std::vector<LoopCond>> twoLevelOrOfAnds_;
  bool negated_ = false;
  // adjust condition in case of loop splitting. If origLoop is still present,
  // include it in newLoops
  void adjustConditionForSplitLoop(
      const dsc2::LoopNode* origLoop,
      const std::vector<const dsc2::LoopNode*>& newLoops);
};

struct ConditionNode : public InheritWithClone<BlockNode, ConditionNode> {
  ConditionNode() : BaseClass(CONDITION) {}
  // assumptions:
  // - max 2 children in next_, of type BLOCK: the "then" and "else" regions
  // - only loopCond_ or coreClCond_ is filled, not both
  LoopCondComposite loopCond_;
  std::map<int, std::set<int>>
      coreClCond_;  // list of core/corelets the "then" region applies to
  inline bool hasCoreClCond() const {
    return loopCond_.twoLevelOrOfAnds_.empty();
  }
  void addChildNode(ScheduleNode* nodeToAdd, bool addBefore = false,
                    const ScheduleNode* siblingRefNode = nullptr) override;
  void addThenRegion(BlockNode* nodeToAdd);
  void addElseRegion(BlockNode* nodeToAdd);
  std::vector<const ScheduleNode*> getNextView(SenComponents comp,
                                               int clId = -1,
                                               int coreId = -1) const;
  std::map<int, std::set<int>> getThenCoreCl(
      SenComponents comp = SenComponents::ALL) const;
  std::map<int, std::set<int>> getElseCoreCl(
      SenComponents comp = SenComponents::ALL) const;
  const ScheduleNode* getThenBranchNode() const {
    if (next_.empty()) {
      return nullptr;
    }
    return next_[0].get();
  }
  const ScheduleNode* getElseBranchNode() const {
    if (next_.size() < 2) {
      return nullptr;
    }
    return next_[1].get();
  }
};

struct DataInfo {
  int myLdsIdx_ = -1;
  FoldManager<int64_t> startAddr_;  // per core, corelet, and sdsc folds
  bool isStartAddrSymbolic_ = false;
  int latchDataId_ = -1;  // used to link producer and consumer when using LATCH
  int constantId_ = -1;   // index of the constant container when using CONSTANT
  std::map<int, std::map<int, std::unordered_map<PrimaryDimTypes, int>>>
      constEleOffsets_;  // constant offset on top of the start address. Per
                         // core and corelet
  std::map<int, std::unordered_map<const LoopNode*,
                                   std::unordered_map<PrimaryDimTypes, int>>>
      loopEleOffsets_;  // if memory, put offset (in number of elements in that
                        // dim (e.g. 4 mb)) or if fifo put 1 if popping next
                        // element, or 0 if "reuse" is expected. Per corelet
  std::map<int, std::map<int, int64_t>>
      bufferAddrOffset_;  // address offset to move to the next buffer (per core
                          // and corelet)
  const LoopNode* bufferSwitchPosition_ = nullptr;
  std::string dataConnect_;

  bool isLabeledDs() const {
    DT_CHECK_MSG(!(myLdsIdx_ >= 0 && constantId_ >= 0),
                 "Cannot be both labeledDs and constant.");
    return myLdsIdx_ >= 0;
  }
  bool isConstant() const {
    DT_CHECK_MSG(!(myLdsIdx_ >= 0 && constantId_ >= 0),
                 "Cannot be both labeledDs and constant.");
    return constantId_ >= 0;
  }

  void print(std::ostream& out, int level = 0) const;
};

class TransferPadInfo {
 public:
  TransferPadInfo()
      : transferPadFrontSizeHelper(transferPadFrontSize_),
        transferPadBackSizeHelper(transferPadBackSize_) {}
  TransferPadInfo(TransferPadInfo&&) = default;
  // Do nothing on purpose for copy constructor.
  TransferPadInfo(const TransferPadInfo&)
      : transferPadFrontSizeHelper(transferPadFrontSize_),
        transferPadBackSizeHelper(transferPadBackSize_) {}
  // It is up to the caller to manually insert copies.
  TransferPadInfo& operator=(const TransferPadInfo&) = delete;

  enum FoldDimPosition {
    WORK_SLICE_FOLDDIM = 0,
    CHUNK_FOLDDIM,
    TOTAL_FOLDDIM_NUM
  };

  bool isEmpty() const {
    return transferPadFrontSize_.empty() && transferPadBackSize_.empty();
  }
  void buildPadFrontSizes(PrimaryDimTypes dim, const std::vector<int>& sizes,
                          const std::vector<int>& alphas,
                          const std::vector<int>& betas);
  void buildPadBackSizes(PrimaryDimTypes dim, const std::vector<int>& sizes,
                         const std::vector<int>& alphas,
                         const std::vector<int>& betas);
  std::set<PrimaryDimTypes> getPadFrontOrBackDimsSet(
      const bool isPadFront = true) const {
    const auto& fmHelper =
        isPadFront ? transferPadFrontSizeHelper : transferPadBackSizeHelper;
    return fmHelper.getAllKeys();
  }
  int getWkSlicePadSizeFrontOrBack(PrimaryDimTypes dim, const int wkSliceIdx,
                                   const int numChunks, const int chunkOffset,
                                   const int chunkSizePadded,
                                   const bool isPadFront) const;
  int getTransferPadSizeFrontOrBack(PrimaryDimTypes dim, const int wkSliceIdx,
                                    const int chunkIdx,
                                    const int chunkSizePadded,
                                    const bool isPadFront) const;

 private:
  void buildTransferFoldDim(PrimaryDimTypes dim, const std::vector<int>& sizes,
                            const bool isPadFront = true);
  void buildPadSizes(PrimaryDimTypes dim, const std::vector<int>& sizes,
                     const std::vector<int>& alphas,
                     const std::vector<int>& betas,
                     const bool isPadFront = true);

  std::map<PrimaryDimTypes, std::vector<FoldDimProp>> transferPadFrontFoldProps;
  std::map<PrimaryDimTypes, std::vector<FoldDimProp>> transferPadBackFoldProps;
  MapWithFMHelper<PrimaryDimTypes, int> transferPadFrontSizeHelper;
  MapWithFMHelper<PrimaryDimTypes, int> transferPadBackSizeHelper;
  std::map<PrimaryDimTypes, FoldManager<int>> transferPadFrontSize_;
  std::map<PrimaryDimTypes, FoldManager<int>> transferPadBackSize_;
};

struct TransferNode : public InheritWithClone<ScheduleNode, TransferNode> {
  TransferNode() : BaseClass(TRANSFER) {}
  struct DstVia {
    DataLocation loc_, locIndirect_;
    std::vector<SenComponents> via_;
  };
  struct SizeAndIndex {
    Size sizeDim_;
    int srcSizeIdx_ = -1, dstSizeIdx_ = -1;
  };
  DataLocation src_, srcIndirect_;
  std::vector<DstVia> dstVias_;
  struct {
    int srcRep_;
    std::vector<int> dstReps_;
  } repetition_;
  const LoopNode* lastFusableParentLoopSrc_ = nullptr;
  std::vector<const LoopNode*> lastFusableParentLoopDst_;
  DataInfo srcLdsAndLoopOffsets_, srcIndirectLdsAndLoopOffsets_;
  std::vector<DataInfo> dstLdsAndLoopOffsets_, dstIndirectLdsAndLoopOffsets_;
  int replicationFactor_ = 1;
  // continuous elements within a stick
  std::vector<SizeAndIndex> unitTimeTransferChunkSize_;
  int unitTimeTransferNumChunks_ = 1;
  std::vector<SizeAndIndex> unitTimeTransferChunkStride_;
  int rotateNumElements_ = 0;
  std::map<int, dsc2::GroupTagRegInfo> coreIdToGTRInfo_;  // L3
  // Explicit transfer size. If filled, use this size rather than
  // derived from data stage.
  std::map<PrimaryDimTypes, int> transferSize_;
  // Zero padding sizes in L3 transfers.
  TransferPadInfo paddingInfo_;

  struct CoreletView {
    UnitView srcLoopsAndSize_, srcIndirectLoopsAndSize_;
    std::vector<UnitView> dstLoopsAndSizes_, dstIndirectLoopsAndSizes_;
  };
  std::map<int, CoreletView> coreletViews_;
  CoordinateType<CoordinateBaseType> transferCoordinates_;

  enum TransferType {
    CONSTANT_TO_CONSTANT,
    CONSTANT_TO_TENSOR,
    TENSOR_TO_TENSOR,
    NO_TRANSFER_TO_TENSOR,
    NO_TRANSFER_FROM_TENSOR,
    INVALID_TRANSFER_TYPE
  };

  bool hasNonMemorySource() const;
  bool hasNonMemoryResult() const;
  bool checkNonMemoryResultIndex(int i) const;
  int getNonMemoryResultIndex() const;
  bool isSrcLabeledDs() const { return srcLdsAndLoopOffsets_.isLabeledDs(); }
  bool isDstLabeledDs() const {
    return !dstLdsAndLoopOffsets_.empty() &&
           dstLdsAndLoopOffsets_.front().isLabeledDs();
  }
  bool isSrcConstant() const { return srcLdsAndLoopOffsets_.isConstant(); }
  bool isDstConstant() const {
    return !dstLdsAndLoopOffsets_.empty() &&
           dstLdsAndLoopOffsets_.front().isConstant();
  }
  bool isSrcIndirect() const { return (srcIndirect_.unit_ != NO_COMPONENT); }
  bool isDstIndirect() const { return !dstIndirectLdsAndLoopOffsets_.empty(); }
  bool isDstIndirectAtIndex(const int dstViasIdx) const {
    DT_CHECK_MSG(dstViasIdx >= 0 && dstViasIdx < dstVias_.size(),
                 "Invalid dstVias_ index.");
    return (dstVias_[dstViasIdx].locIndirect_.unit_ != NO_COMPONENT);
  }
  TransferType getTransferType() const {
    if (isSrcConstant() && isDstConstant())
      return TransferType::CONSTANT_TO_CONSTANT;
    if (isSrcConstant() && isDstLabeledDs())
      return TransferType::CONSTANT_TO_TENSOR;
    if (isSrcLabeledDs() && isDstLabeledDs())
      return TransferType::TENSOR_TO_TENSOR;
    if ((!isSrcLabeledDs() && !isSrcConstant()) && isDstLabeledDs())
      return TransferType::NO_TRANSFER_TO_TENSOR;
    if (isSrcLabeledDs() && (!isDstLabeledDs() && !isDstConstant()))
      return TransferType::NO_TRANSFER_FROM_TENSOR;
    return TransferType::INVALID_TRANSFER_TYPE;
  }
  void print(std::ostream& out, int level = 0) const;
};

struct ComputeNode : public InheritWithClone<ScheduleNode, ComputeNode> {
  ComputeNode() : BaseClass(COMPUTE) {}

  // indices = [<array of indices, -1 for zero/sign extend>], repetition=8,
  // sign_extend=false
  struct InstrAttribute {
    std::vector<int> indices_;  // PACK/MERGE Mapping
    int repetition_ = 8;        // default 8 slices works the same.
    bool sign_extend_ = false;  // PACK/MERGE mapping
    std::map<std::string, std::string>
        read_write_reg_map_;  // OPAQUE register alias map
    std::map<std::string, std::string>
        read_only_reg_map_;  // OPAQUE read-only register alias map
    std::map<std::string, std::string>
        param_map_;              // OPAQUE parameter alias map
    int mode_ = -1;              // General SRC1/IMM field
    size_t compute_mask_ = 255;  // Mask of the compute
    // std::unordered_map<const LoopNode*,
    //                                std::unordered_map<PrimaryDimTypes, int>>
    //   loopEleOffsets_;  // if memory, put offset (in number of elements in
    //   that
    // dim (e.g. 4 mb)) or if fifo put 1 if popping next
    // element, or 0 if "reuse" is expected. Per corelet
    std::map<int, std::unordered_map<const LoopNode*,
                                     std::unordered_map<PrimaryDimTypes, int>>>
        computeMaskLoopOffsets_;
    std::vector<std::string>
        input_data_connects_;  // OPAQUE input data_connects
    std::vector<std::string>
        output_data_connects_;  // OPAQUE output data_connects
  };

  SenComponents exUnit_ = SenComponents::NO_COMPONENT;
  ComputeOpType type_ = ComputeOpType::COUNT;
  DataFormats dataFormat_ = DataFormats::SEN169_FP16;
  std::vector<SenComponents> inputs_;
  std::vector<SenComponents> outputs_;
  std::vector<DataInfo> inputsLdsAndLoopOffsets_;
  std::vector<DataInfo> outputsLdsAndLoopOffsets_;
  InstrAttribute instrAttribute_;  // Additional attributes of the instruction.
  int numFoldsEngaged = 1;
  bool isOpaqueOp_ = false;

  struct CoreletView {
    std::vector<ScheduleNode::UnitView> inputsLoopsAndSizes_;
    std::vector<ScheduleNode::UnitView> outputsLoopsAndSizes_;
  };
  std::map<int, CoreletView> coreletViews_;
  std::vector<CoordinateType<CoordinateBaseType>> inputCoordinates_;
  CoordinateType<CoordinateBaseType> outputCoordinate_;
  struct RepetitionWithOffset {
    std::vector<int> forInputs_ = {};
    std::vector<int> forOutputs_ = {};
  };
  RepetitionWithOffset repetitionWithOffset_;

  // number of elements read or written by compute node.
  // All inputs, output last
  std::vector<int> getComputeOperandSizes(const SenSystemDef& sysDef) const;
  std::vector<DataFormats> getComputeOperandFormats(
      const DesignSpaceConfig& dsc) const;
  void print(std::ostream& out, int level = 0) const;
};

struct SyncNode : public InheritWithClone<ScheduleNode, SyncNode> {
  SyncNode() : BaseClass(SYNC) {}
  std::unordered_set<SenComponents> units_;  // all to all signals
  bool isReceive_ = false, isSoft_ = false;
  const TransferNode* implicitSyncRefTransfer_ = nullptr;
  std::vector<const SyncNode*> otherEndOfTheSignals_;
  std::map<SenComponents, std::map<int, std::set<int>>>
  getComponentsFromOtherEnds(int coreId = -1) const;
};

struct AllocateNode : public InheritWithClone<ScheduleNode, AllocateNode> {
  AllocateNode() : BaseClass(ALLOCATE) {}
  int ldsIdx_ = -1;
  int constIdx_ = -1;
  const ComputeNode* tempStorageForCompute_ = nullptr;
  SenComponents component_ = SenComponents::NO_COMPONENT;
  // Padding padding_;
  PaddingFormType padding_;
  std::vector<PrimaryDimTypes> layoutDimOrder_;
  std::vector<int> maxDimSizes_;
  int numBuffers_ = 1;  // 1:no buffering, 2:double-buffer, -1:streaming buffer
  FoldManager<int64_t>
      startAddressCoreCorelet_;  // per core, corelet, and sdsc folds
  bool isStartAddrSymbolic_ = false;
  std::map<int, std::map<int, int64_t>> bufferOffsetCoreCorelet_;
  std::map<PrimaryDimTypes, std::map<int, int>> backGapCore_;  // HBM is -1
  enum class IndirectAllocType {
    NO_INDIRECTION,
    VALUE_TENSOR,
    INDEX_TENSOR
  } indirectAllocType_ = IndirectAllocType::NO_INDIRECTION;
  enum class IndexTensorType {
    ADDRESS,
    INDEX
  } indexTensorType_ = IndexTensorType::ADDRESS;
  AllocateNode* relatedIndirectAccessAlloc_ =
      nullptr;  // in case of indirect access, link to either value or index
                // tensor alloc
  bool ignoreSymbolicVolumeLimits_ =
      false;  // force this allocation to be "ghost rectangular"
  bool nonUnifiedAllocInHBM_ =
      false;  // HBM allocation for each core is residing in different locations
  std::map<PrimaryDimTypes, int> gapStickSpread_;
  std::vector<std::pair<const ScheduleNode*, int>> allocUsers_;
  CoordinateType<CoordinateBaseType> allocateCoordinates_;
  CoordinateType<CoordinateBaseType> sliceViewCoordinates_;

  std::map<PrimaryDimTypes, int> getPageSize() const;
  void addAllocUser(const ScheduleNode* userNode) {
    for (auto& [node, refCount] : allocUsers_) {
      if (node == userNode) {
        refCount++;
        return;
      }
    }
    allocUsers_.push_back({userNode, 1});
  }

  void removeAllocUser(const ScheduleNode* userNode) {
    for (auto it = allocUsers_.begin(), e = allocUsers_.end(); it != e; ++it) {
      if (it->first == userNode) {
        --(it->second);
        if (it->second == 0) {
          allocUsers_.erase(it);
        }
        return;
      }
    }
    DT_ERROR("RemoveAllocUser: Schedule node " + userNode->name_ +
             " is not in the user list of allocate node " + name_);
  }

  bool hasAllocUsers() const { return !allocUsers_.empty(); }
  bool hasAllocUser(const ScheduleNode* userNode) const {
    for (auto& [node, refCount] : allocUsers_) {
      if (node == userNode) {
        return true;
      }
    }
    return false;
  }

  void clearAllocUsers() { allocUsers_.clear(); }

  void print(std::ostream& out, int level = 0) const;
  static const std::unordered_map<IndirectAllocType, std::string>
      indirectAllocTypeToString;
  static const std::unordered_map<std::string, IndirectAllocType>
      stringToIndirectAllocType;
  static const std::unordered_map<IndexTensorType, std::string>
      indexTensorTypeToString;
  static const std::unordered_map<std::string, IndexTensorType>
      stringToIndexTensorType;
};

struct StickMaskNode : public InheritWithClone<ScheduleNode, StickMaskNode> {
  StickMaskNode() : BaseClass(STICKMASK) {}
  int maskValConstId_ = -1;
  DataFormats dataFormat_ = DataFormats::INVALID;
  std::vector<Size> stickLayout_;
  std::map<PrimaryDimTypes, int> firstStickCoordToMaskPerDim_;
  std::vector<const dsc2::TransferNode*> affectedTransfers_;

  struct View {
    std::pair<int, int> maskA_, maskB_;  // <unmasked, masked>
    int transitionSliceId_;
  };
  View getView() const;
};

void transformLxZeroPadInfoInScheduleTree(SuperDsc& mySDsc);

AllocateNode* getValueAllocation(const DesignSpaceConfig* dsc,
                                 const AllocateNode* allocNode);
AllocateNode* getScaleAllocation(const DesignSpaceConfig* dsc,
                                 const AllocateNode* allocNode);

struct FoldParamInfoType {
  CoordinateBaseType alpha = 1, beta = 0;
  int64_t cardinality = 0;
  std::string foldDimLabel = "";
};

struct CoordPropInfoType {
  enum PropStateType { NOT_PROCESSED, ROLLED_BACK, OVERRIDDEN, COMPLETE };
  dsc2::ScheduleNode* refNode;
  dsc2::ScheduleNode* nodeToFold;
  std::string dataConnect = "";
  bool refIsProducer = true;
  PropStateType propState = NOT_PROCESSED;
  std::vector<PrimaryDimTypes> dimsToPropagate;
  bool scaleDown = false;

 public:
  void print(std::ostream& out) const {
    out << "\n  CoordinatePropagationInfo: refNode=" << refNode->name_
        << ", nodeTofold= " << nodeToFold->name_
        << ", data_connect= " << dataConnect
        << ", refIsProducer= " << (refIsProducer ? "T" : "F")
        << ", scaleDown= " << (scaleDown ? "T" : "F") << "\n    Dims:";
    for (auto& dim : dimsToPropagate) {
      out << " " << EnumsConversion::primaryDimToString.at(dim);
    }
  }
};

struct LoopDistributionParamType {
  CoordinateBaseType alpha = 1, beta = 0;
  int64_t temporalStridePostDistribution = -1;
  int relatedElemArrLevel = -1;
};
typedef std::map<const PrimaryDimTypes, LoopDistributionParamType>
    LoopDistributionParamPerLoopType;

typedef std::map<const dsc2::LoopNode*, LoopDistributionParamPerLoopType>
    LoopDistributionParamPerNodeType;

// Methods related to coordinate capturing
// ----------------------------------------------

// extern std::map<const ScheduleNode*,
//                 std::map<const ScheduleNode*,
//                 LoopDistributionParamPerNodeType>>
//     loopDistributionParamInfo;

enum DistributionStatusType { SUCCESS = 0, NEED_LOOP_SPLIT = 1 };

struct DistributionStatusInfo {
  LoopNode* loopToSplit = nullptr;
  PrimaryDimTypes loopSplitDim = PrimaryDimTypes::PrimaryDimTypesCount;
  std::vector<int> loopSplitDimSizes;
};

struct LoopDistributionInfo {
  enum LoopDistributionCat { UNKNOWN, ABOVE_CHUNK, BELOW_CHUNK, CORELET_SLICE };
  LoopDistributionInfo(LoopNode* pLoopNode, PrimaryDimAndKind pDimAndKind,
                       LoopDistributionCat pCat)
      : loopNode(pLoopNode), dimAndKind(pDimAndKind), cat(pCat) {}
  LoopNode* loopNode;
  PrimaryDimAndKind dimAndKind;
  LoopDistributionCat cat = LoopDistributionCat::UNKNOWN;

  void print(std::ostream& out) const {
    out << " (" << loopNode->name_ << ": dim= ("
        << EnumsConversion::primaryDimToString.at(dimAndKind.dim_) << ", "
        << EnumsConversion::metaDimKindToString.at(dimAndKind.kind_) << ") "
        << ", cat= ";
    switch (cat) {
      case UNKNOWN:
        out << "Unknwon";
        break;
      case ABOVE_CHUNK:
        out << "Above_chunk";
        break;
      case BELOW_CHUNK:
        out << "Below_chunk";
        break;
      case CORELET_SLICE:
        out << "Corelet_slice";
        break;
      default:
        out << "ERROR";
        break;
    }
    out << ")";
  }
};

typedef std::vector<LoopDistributionInfo> VectorOfLoopAndDim;

void distributeElemArrToTemporalLoops(
    const DesignSpaceConfig* currDsc, const PrimaryDimAndKind& currDim,
    const dsc2::ScheduleNode* foldOwnerNode, const int targetLdxIdx,
    const PadType& refPadType, const PadType& targetPadType,
    const SenComponents sizeRefComp, const SenComponents propRefComp,
    const VectorOfLoopAndDim& loopChain,
    const std::vector<FoldParamInfoType>& elemArr,
    LoopDistributionParamPerNodeType& loopParamsAfterDistribution,
    std::vector<FoldParamInfoType>& elemArrParamsAfterDistribution,
    const int targetCoreletId = 0, const int coordReportLevel = 0);

// Determines whether a given dimension of a loop is relevant for a specified
// dimension. Considers cases where a primary dimension (say I) is accessed
// through a corresponding meta dimension (say KI).
bool isLoopDimRelated(const DesignSpaceConfig* currDsc,
                      const PrimaryDimAndKind& dimAndKind,
                      const dsc2::LoopNode* loop,
                      const PrimaryDimAndKind& loopDim, PadType accessPadType);

// Determines whether a given loop is relevant for a specified dimension.
// Considers cases where a primary dimension (say I) is accessed through a
// corresponding meta dimension (say KI).
bool loopRelevantForDim(const DesignSpaceConfig* currDsc,
                        const PrimaryDimAndKind& dimAndKind,
                        dsc2::LoopNode* loop, PadType accessPadType);

// Given a collection of enclosing loops for all dimensions, filters loops that
// contain certain dim or its related dim like ki/kj if dim=i/j
void collectRelatedLoops(const DesignSpaceConfig* currDsc,
                         const PrimaryDimAndKind dimToFind,
                         const dsc2::VectorOfLoopAndDim& allEnclosingLoops,
                         dsc2::VectorOfLoopAndDim& relatedLoops,
                         PadType accessPadType);

// Collects all loops that enclose a given scheduleNode. The loops are ordered
// from innermost to outermost. For each collected loop, the result includes the
// loop dimension as well as the category of the loop according to
// LoopDistributionCat enumeration. Each entry in the resulting collection
// corresponds to a single dimension. Therefore, a loopband with 3 dimensions
// will result in 3 entries, one for each of the 3 dimensions.
//
// The includeCoreletSplit determines whether an artificial loop needs to be
// inserted immediately below the innermost chunk loop to represent
// corelet-slicing.
void getEnclosingLoopsAndRelatedDims(
    const DesignSpaceConfig* currDsc, dsc2::ScheduleNode* node,
    dsc2::VectorOfLoopAndDim& loopChain,
    const std::unordered_set<const dsc2::LoopNode*>& loopsBelowChunkBoundary =
        {},
    bool includeCoreletSplit = false);

void computeLoopElemOffsetsFromCoordinates(
    const DesignSpaceConfig* currDsc, dsc2::ScheduleNode* node,
    dsc2::CoordinateType<CoordinateBaseType>& nodeCoordinate,
    const dsc2::AllocateNode* allocNode,
    dsc2::LoopDistributionParamPerNodeType& loopDistributionParams,
    std::vector<PrimaryDimTypes> workingDims, int coordReportLevel = 0);

void computeLoopElemOffsetsFromCoordinates(
    const DesignSpaceConfig* currDsc, const PrimaryDimTypes dim,
    const int ptRowId, std::vector<dsc2::LoopNode*>& loopChain,
    dsc2::CoordinateType<CoordinateBaseType>& nodeCoordinate,
    const dsc2::AllocateNode* allocNode,
    dsc2::LoopDistributionParamPerNodeType& loopDistributionParamInfo,
    int coordReportLevel = 0);
}  // namespace dsc2

#endif
