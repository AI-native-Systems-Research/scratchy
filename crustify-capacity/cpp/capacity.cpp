// capacity.cpp — the buffer-capacity closure of IBM Spyre deeptools, extracted VERBATIM.
//
// EVERY BODY BELOW IS A SLICE OF THE AUTHORITY TREE AT
//   /Users/nickm/git/deeptools-src/<file>:<line>
// and that tree, NOT this extract, is what you port from.  This file exists so the
// translation unit stands alone and so the schedule can name and order the units.
//
// ⛔ 19 DEFINITIONS, 761 LINES — the transitive callee closure of
// `DesignSpaceConfig::getBufferCapacityForNode`, closed to fixpoint.  A brief that named
// only the nine outermost bodies (533 lines) had closed the graph one level deep.
//
// ⛔ `calculate_padded` (e009) and `primaryDimToVal_base_st` (e010) are MUTUALLY RECURSIVE
// (dims.cpp:560 calls calculate_padded; dims.cpp:594 calls primaryDimToVal_base_st).  They
// are one batch, not two waves.
//
// ⛔ ALREADY PORTED, DO NOT RE-PORT: `getStickSizes` (dsc/dsc2.cpp:4066) and
// `getCumulativeStickSizes` (dsc/dsc2.cpp:4108) landed as bridge-1's e041/e071 in
// crates/compiler/deeptools/src/bridges/superdsc_to_dataflow_ir/shape_constraints.rs.
// e006/e014/e015/e017 CALL them; call the Rust port.

// ===== e001_compound  L0  27L  dsc/dims.cpp:84-110 =====
void DataStructDims::compound() {
  if (i_ >= 0 && j_ >= 0) {
    ij_ = i_ * j_;
  } else {
    ij_ = -1;
  }
  if (ki_ >= 0 && kj_ >= 0) {
    kij_ = ki_ * kj_;
  } else {
    kij_ = -1;
  }
  if (zi_ >= 0 && zj_ >= 0) {
    zij_ = zi_ * zj_;
  } else {
    zij_ = -1;
  }
  if (si_ >= 0 && sj_ >= 0) {
    sij_ = si_ * sj_;
  } else {
    sij_ = -1;
  }
  if (r_ >= 0 && c_ >= 0) {
    rc_ = r_ * c_;
  } else {
    rc_ = -1;
  }
}

// ===== e002_primaryDimToValHandler_st  L0  30L  dsc/dims.cpp:485-514 =====
double &DataStructDims::primaryDimToValHandler_st(PrimaryDimTypes d) {
  if (d == PrimaryDimTypes::IN) {
    return in_;
  } else if (d == PrimaryDimTypes::OUT) {
    return out_;
  } else if (d == PrimaryDimTypes::MB) {
    return mb_;
  } else if (d == PrimaryDimTypes::I) {
    return i_;
  } else if (d == PrimaryDimTypes::J) {
    return j_;
  } else if (d == PrimaryDimTypes::IJ) {
    return ij_;
  } else if (d == PrimaryDimTypes::KI) {
    return ki_;
  } else if (d == PrimaryDimTypes::KJ) {
    return kj_;
  } else if (d == PrimaryDimTypes::KIJ) {
    return kij_;
  } else if (d == PrimaryDimTypes::X) {
    return x_;
  } else if (d == PrimaryDimTypes::X1) {
    return x1_;
  } else if (d == PrimaryDimTypes::Y) {
    return y_;
  } else {
    DT_ERROR("Invalid PrimaryDim: " +
             EnumsConversion::primaryDimToString.at(d));
  }
}

// ===== e003_scaleFromMaxToGranularity  L0  12L  dsc/dims.cpp:618-629 =====
int DataStructDims::scaleFromMaxToGranularity(PrimaryDimTypes d,
                                              int val) const {
  auto symIt = symbolicDimInfo_.find(d);
  if (symIt != symbolicDimInfo_.end()) {
    // assume cl size is filled based on max value
    DT_CHECK(symIt->second.maxSize_ % symIt->second.granularity_ == 0);
    auto factor = symIt->second.maxSize_ / symIt->second.granularity_;
    DT_CHECK(factor != 0 && val % factor == 0);
    val /= factor;
  }
  return val;
}

// ===== e004_getDimIndexInLayoutOrder  L0  10L  dsc/designSpaceConfig.cpp:429-438 =====
int DesignSpaceConfig::getDimIndexInLayoutOrder(DsTypes dstype,
                                                PrimaryDimTypes dim) const {
  auto& layout_dim_order = primaryDsInfo_.at(dstype).layoutDimOrder_;
  for (int i = 0; i < layout_dim_order.size(); i++) {
    if (dim == layout_dim_order[i]) {
      return i;
    }
  }
  return -1;
}

// ===== e005_getLayoutDimSet  L0  5L  dsc/dsc2.cpp:4027-4031 =====
std::set<PrimaryDimTypes> DesignSpaceConfig::getLayoutDimSet(
    DsTypes dsType) const {
  auto& ldov = primaryDsInfo_.at(dsType).layoutDimOrder_;
  return std::set<PrimaryDimTypes>(ldov.begin(), ldov.end());
}

// ===== e006_getLayoutDims  L0  19L  dsc/dsc2.cpp:4007-4025 =====
std::vector<PrimaryDimTypes> DesignSpaceConfig::getLayoutDims(
    int ldsIdx) const {
  DT_CHECK(ldsIdx >= 0 && ldsIdx < labeledDs_.size());
  dsc2::AllocateNode* allocNode = nullptr;
  while (!allocNode && ldsIdx >= 0) {
    const auto& lds = labeledDs_.at(ldsIdx);
    for (auto& [comp, memInfo] : lds.memOrg_) {
      if (memInfo.allocateNode_) {
        allocNode = memInfo.allocateNode_;
        if (is_any_of(comp, LX, HBM)) break;
      }
    }
    ldsIdx = lds.referenceLdsIdx_;
  }

  DT_CHECK(allocNode);
  DT_CHECK(!allocNode->layoutDimOrder_.empty());
  return allocNode->layoutDimOrder_;
}

// ===== e007_getPageSize  L0  34L  dsc/dsc2.cpp:4480-4513 =====
std::map<PrimaryDimTypes, int> dsc2::AllocateNode::getPageSize() const {
  std::map<PrimaryDimTypes, int> pageSize;
  const dsc2::AllocateNode* refAlloc = nullptr;
  if (indirectAllocType_ ==
      dsc2::AllocateNode::IndirectAllocType::NO_INDIRECTION) {
    return pageSize;
  } else if (indirectAllocType_ ==
             dsc2::AllocateNode::IndirectAllocType::VALUE_TENSOR) {
    refAlloc = this;
  } else if (indirectAllocType_ ==
             dsc2::AllocateNode::IndirectAllocType::INDEX_TENSOR) {
    DT_CHECK(relatedIndirectAccessAlloc_);
    refAlloc = relatedIndirectAccessAlloc_;
  } else {
    DT_ERROR("Unhandled indirect alloc type");
  }
  DT_CHECK(refAlloc->layoutDimOrder_.size() == refAlloc->maxDimSizes_.size());
  std::unordered_set<PrimaryDimTypes> unboundedDims;
  for (int i = 0; i < refAlloc->layoutDimOrder_.size(); i++) {
    auto& dim = refAlloc->layoutDimOrder_[i];
    auto& maxSize = refAlloc->maxDimSizes_[i];
    if (maxSize < 0) {
      unboundedDims.insert(dim);
      pageSize.erase(dim);  // safe even if key not present
    } else {
      if (!unboundedDims.count(dim)) {
        // it will insert 1 only if dim did not exist in the map already
        auto& dimPageSize = pageSize.try_emplace(dim, 1).first->second;
        dimPageSize *= maxSize;
      }
    }
  }
  return pageSize;
}

// ===== e008_pruneMaxSymbolicVolumes  L0  34L  dsc/dims.cpp:729-762 =====
void DataStructDims::pruneMaxSymbolicVolumes(const DataStructDims &refDstg) {
  for (auto it = maxSymbolicVolume_.begin(); it != maxSymbolicVolume_.end();) {
    const auto &[symDims, volumeLimit] = *it;
    bool needPruning = std::any_of(
        symDims.begin(), symDims.end(),
        [&](PrimaryDimTypes dim) { return !symbolicDimInfo_.count(dim); });
    if (!needPruning) {
      it++;
      continue;
    }
    std::set<PrimaryDimTypes> mySymDims;
    auto myVolumeLimit = volumeLimit, mulOfMaxes = 1;
    for (const auto &symDim : symDims) {
      if (symbolicDimInfo_.count(symDim)) {
        mySymDims.insert(symDim);
        mulOfMaxes *= symbolicDimInfo_.at(symDim).maxSize_;
      } else {  // reduce volume limit
        DT_CHECK(refDstg.symbolicDimInfo_.count(symDim));
        auto dimGranularity = refDstg.symbolicDimInfo_.at(symDim).granularity_;
        DT_CHECK(myVolumeLimit % dimGranularity == 0);
        myVolumeLimit /= dimGranularity;
      }
    }
    myVolumeLimit = std::min(myVolumeLimit, mulOfMaxes);
    if (!mySymDims.empty()) {
      auto maxSymVolIt = maxSymbolicVolume_.find(mySymDims);
      if (maxSymVolIt == maxSymbolicVolume_.end() ||
          maxSymVolIt->second > myVolumeLimit) {
        maxSymbolicVolume_.insert_or_assign(mySymDims, myVolumeLimit);
      }
    }
    it = maxSymbolicVolume_.erase(it);
  }
}

// ===== e009_calculate_padded  L1  54L  dsc/dims.cpp:563-616 =====
int DataStructDims::calculate_padded(PrimaryDimTypes d, int val,
                                     const PaddingFormType &padded,
                                     bool getSymbolicGranularity) const {
  auto padType = padded.getPadding(d);
  if (val < 0)
    return -1;
  else if (padType == PadType::NOPAD)
    return val;
  else {
    if (is_any_of(d, IJ, KIJ)) {
      DT_ERROR("Cannot calculate padded version of compound dim");
    }
    if (!paddingSizes_.count(d)) {
      DT_ERROR(
          "Padded dimension without padding sizes information in datastage");
    }
    auto &padInfo = paddingSizes_.at(d);
    if (padInfo.windowDim_ == PrimaryDimTypesCount) {  // no kernel, like csq
      if (padInfo.padFront_ < 0 || padInfo.padBack_ < 0) {
        DT_ERROR("Padded access is not valid in datastage " + name_);
      }
      if (padType == PadType::PADDED_FULLSPAN_WUNNEEDED)
        return val + padInfo.padFront_ + padInfo.padBack_ +
               padInfo.unneededPad_;
      else if (padType == PadType::PADDED_FULLSPAN)
        return val + padInfo.padFront_ + padInfo.padBack_;
      else
        DT_ERROR(
            "Unsupported padding type requested for padded non-window "
            "operation");
    } else {  // regular windowed operation
      int wSize = primaryDimToVal_base_st(padInfo.windowDim_, {}, 1.0,
                                          getSymbolicGranularity);
      if (wSize < 1) {
        DT_ERROR("Missing window size for padded size calculation");
      }
      if (padType == PadType::PADDED_FULLSPAN_WUNNEEDED)
        return wSize + (val - 1) * padInfo.stride_ + padInfo.unneededPad_;
      else if (padType == PadType::PADDED_WZEROPAD)
        return wSize + (val - 1) * padInfo.stride_;
      else if (padType == PadType::PADDED_NOZEROPAD) {
        if (padInfo.padFront_ < 0 || padInfo.padBack_ < 0) {
          DT_ERROR("Valid access is not allowed in datastage " + name_);
        }
        return wSize + (val - 1) * padInfo.stride_ + padInfo.unneededPad_ -
               padInfo.unneededPadFront_ - padInfo.unneededPadBack_ -
               padInfo.padFront_ - padInfo.padBack_;
      } else if (padType == PadType::LOWERED_PADDED)
        return wSize * val;
      else
        DT_ERROR("Unsupported padding type requested for window operation");
    }
  }
}

// ===== e010_primaryDimToVal_base_st  L1  46L  dsc/dims.cpp:516-561 =====
int DataStructDims::primaryDimToVal_base_st(PrimaryDimTypes d,
                                            const PaddingFormType &padded,
                                            double dimDensity,
                                            bool getSymbolicGranularity) const {
  int val;
  auto symIt = symbolicDimInfo_.find(d);
  if (symIt != symbolicDimInfo_.end()) {
    if (getSymbolicGranularity) {
      val = symIt->second.granularity_;
    } else {
      val = symIt->second.maxSize_;
    }
  } else {
    if (d == PrimaryDimTypes::IN) {
      val = in_;
    } else if (d == PrimaryDimTypes::OUT) {
      val = out_;
    } else if (d == PrimaryDimTypes::MB) {
      val = mb_;
    } else if (d == PrimaryDimTypes::I) {
      val = i_;
    } else if (d == PrimaryDimTypes::J) {
      val = j_;
    } else if (d == PrimaryDimTypes::IJ) {
      val = ij_;
    } else if (d == PrimaryDimTypes::KI) {
      val = ki_;
    } else if (d == PrimaryDimTypes::KJ) {
      val = kj_;
    } else if (d == PrimaryDimTypes::KIJ) {
      val = kij_;
    } else if (d == PrimaryDimTypes::X) {
      val = x_;
    } else if (d == PrimaryDimTypes::X1) {
      val = x1_;
    } else if (d == PrimaryDimTypes::Y) {
      val = y_;
    } else {
      DT_ERROR("Invalid PrimaryDim: " +
               EnumsConversion::primaryDimToString.at(d));
    }
  }
  DT_CHECK(dimDensity > 0.0 && dimDensity <= 1.0);
  val *= dimDensity;
  return calculate_padded(d, val, padded, getSymbolicGranularity);
}

// ===== e011_primaryDimToVal_clView_st  L1  15L  dsc/dims.cpp:631-645 =====
int DataStructDims::primaryDimToVal_clView_st(
    PrimaryDimTypes d, int clId, const PaddingFormType &padded,
    double dimDensity, bool getSymbolicGranularity) const {
  if (clId >= 0 && coreletSplit_.count(d) > 0) {
    int size = coreletSplit_.at(d).at(clId);
    if (getSymbolicGranularity) {
      size = scaleFromMaxToGranularity(d, size);
    }
    DT_CHECK(dimDensity > 0.0 && dimDensity <= 1.0);
    size *= dimDensity;
    size = calculate_padded(d, size, padded, getSymbolicGranularity);
    return size;
  }
  return primaryDimToVal_base_st(d, padded, dimDensity, getSymbolicGranularity);
}

// ===== e012_primaryDimToVal_st  L2  56L  dsc/dims.cpp:651-706 =====
int DataStructDims::primaryDimToVal_st(
    PrimaryDimTypes d, SenComponents peOrSfp, int ptrowId /*=-1*/,
    int clId /*=-1*/, const PaddingFormType &padded /*= {}*/,
    double dimDensity /*= 1.0*/,
    bool getSymbolicGranularity /*= false*/) const {
  DT_CHECK(dimDensity > 0.0 && dimDensity <= 1.0);
  // TO DO: check component is consistent with other arguments.
  // Map memory components to compute components.
  if (peOrSfp == SenComponents::PELRF) {
    peOrSfp = SenComponents::PE;
  } else if (peOrSfp == SenComponents::SFPLRF) {
    peOrSfp = SenComponents::SFP;
  }
  if (ptrowId >= 0 && rowSplit_.count(d) > 0) {
    int val = 0;
    if (clId >= 0) {
      val = rowSplit_.at(d).at(clId).at(ptrowId);
    } else {
      if (coreletSplit_.count(d)) {
        for (auto &clSplit : rowSplit_.at(d)) {
          val += clSplit.second.at(ptrowId);
        }
      } else {
        val = rowSplit_.at(d).begin()->second.at(ptrowId);
      }
    }
    if (getSymbolicGranularity) {
      val = scaleFromMaxToGranularity(d, val);
    }
    val *= dimDensity;
    val = calculate_padded(d, val, padded, getSymbolicGranularity);
    return val;
  } else if ((peOrSfp == SenComponents::PE || peOrSfp == SenComponents::SFP) &&
             peSfpSplit_.count(d) > 0) {
    int val = 0;
    if (clId >= 0) {
      val = peSfpSplit_.at(d).at(clId).at(peOrSfp);
    } else {
      if (coreletSplit_.count(d)) {
        for (auto &[dim, splitInfo] : peSfpSplit_.at(d)) {
          val += splitInfo.at(peOrSfp);
        }
      } else {
        val = peSfpSplit_.at(d).begin()->second.at(peOrSfp);
      }
    }
    if (getSymbolicGranularity) {
      val = scaleFromMaxToGranularity(d, val);
    }
    val *= dimDensity;
    val = calculate_padded(d, val, padded, getSymbolicGranularity);
    return val;
  }
  return primaryDimToVal_clView_st(d, clId, padded, dimDensity,
                                   getSymbolicGranularity);
}

// ===== e013_primaryDimToVal_st_1arg  L2  3L  dsc/dims.cpp:647-649 =====
int DataStructDims::primaryDimToVal_st(PrimaryDimTypes d) const {
  return primaryDimToVal_st(d, SenComponents::NO_COMPONENT);
}

// ===== e014_parametricStride  L3  25L  dsc/dsc2.cpp:4197-4221 =====
int dsc2::LoopNode::parametricStride(const DesignSpaceConfig* currDsc) const {
  const auto ldsIdx = parametricLdsIdx();
  DT_CHECK_MSG(
      ldsIdx != -1,
      "Cannot calculate stride of parametric loop as parametricLdsIdx_ is "
      "not set");
  const auto& lds = currDsc->labeledDs_.at(ldsIdx);
  auto stickSizes = currDsc->getCumulativeStickSizes(lds.dsType_);

  PrimaryDimTypes loopDim = dims_[0].dim_;
  int loopStride = 1;
  if (stickSizes.count(loopDim)) {
    loopStride = stickSizes[loopDim];
  } else {
    // Check if the dim at least exists in case this is not in stick dims.
    if (!is_any_of(loopDim, currDsc->getLayoutDims(ldsIdx))) {
      DT_ERROR("Specified dimension " +
               EnumsConversion::primaryDimToString.at(loopDim) +
               " of parametric loop " + name_ +
               " is not included in the associated primaryDs.");
    }
  }

  return loopStride;
}

// ===== e015_getSizeDataStageForNode  L3  137L  dsc/dsc2.cpp:3616-3752 =====
dsc2::DataStage DesignSpaceConfig::getSizeDataStageForNode(
    const dsc2::ScheduleNode* node, const int ldsIdx,
    const PaddingFormType& padding) const {
  dsc2::DataStage newDstg;
  if (ldsIdx < 0) {
    DT_ERROR("Cannot get datastage for node without an allocation for lds");
  }

  if (node->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
    auto* an = static_cast<const dsc2::AllocateNode*>(node);
    if (an->component_ == SenComponents::HBM && !an->nonUnifiedAllocInHBM_) {
      // HBM allocations always have size N_, even if
      DT_CHECK_MSG(node->getOwnerLoop() == nullptr ||
                       node->getOwnerLoop()->prev_ == nullptr,
                   "HBM allocation should be at root of schedule tree");
      newDstg.ss_ = newDstg.el_ = N_;
      return newDstg;
    }
  }

  auto stickSizes = getCumulativeStickSizes(labeledDs_.at(ldsIdx).dsType_);
  std::set<PrimaryDimTypes> targetDims = getLayoutDimSet(ldsIdx);
  auto& coreDs = dataStageParam_.at(0).ss_;
  DT_CHECK(coreDs.name_ == "core");
  for (auto& [dim, padInfo] : coreDs.paddingSizes_) {
    if (targetDims.count(dim) && padding.getPadding(dim) != PadType::NOPAD &&
        padInfo.windowDim_ != PrimaryDimTypesCount)
      targetDims.insert(padInfo.windowDim_);
  }

  // get the denominator datastage for each relevant dimension
  std::map<PrimaryDimTypes, int> denDsForDim;
  const dsc2::LoopNode* myParentLoop = node->getOwnerLoop();
  while (!targetDims.empty()) {
    if (myParentLoop->getPrev() == nullptr) {  // root node
      for (auto& dim : targetDims) {
        denDsForDim[dim] = myParentLoop->denId_;
      }
      targetDims.clear();
    } else {
      if (myParentLoop->isParametricLoop()) {
        // Parametric loops are not associated with any datastages.
        auto loopDim = myParentLoop->dims_[0].dim_;
        if (targetDims.count(loopDim)) {
          int loopStride = myParentLoop->parametricStride(this);
          newDstg.ss_.primaryDimToValHandler_st(loopDim) = loopStride;
          newDstg.el_.primaryDimToValHandler_st(loopDim) = loopStride;
          if (coreDs.paddingSizes_.count(loopDim)) {
            // use default values so that padded and unpadded values are same
            newDstg.ss_.paddingSizes_[loopDim];
            newDstg.el_.paddingSizes_[loopDim];
          }
          targetDims.erase(loopDim);
        }
      } else {
        for (auto& [ldim, kind] : myParentLoop->dims_) {
          if (targetDims.count(ldim)) {
            denDsForDim[ldim] = myParentLoop->denId_;
            targetDims.erase(ldim);
          }
        }
      }
    }
    myParentLoop = myParentLoop->getOwnerLoop();
  }

  DT_CHECK(targetDims.size() == 0);
  for (auto& [dim, dsIdx] : denDsForDim) {
    auto& myDenDstg = dataStageParam_.at(dsIdx);
    newDstg.ss_.primaryDimToValHandler_st(dim) =
        myDenDstg.ss_.primaryDimToVal_st(dim);
    newDstg.el_.primaryDimToValHandler_st(dim) =
        myDenDstg.el_.primaryDimToVal_st(dim);
    if (myDenDstg.ss_.coreletSplit_.count(dim)) {
      newDstg.ss_.coreletSplit_[dim] = myDenDstg.ss_.coreletSplit_.at(dim);
      newDstg.el_.coreletSplit_[dim] = myDenDstg.el_.coreletSplit_.at(dim);
    }
    if (myDenDstg.ss_.rowSplit_.count(dim)) {
      newDstg.ss_.rowSplit_[dim] = myDenDstg.ss_.rowSplit_.at(dim);
      newDstg.el_.rowSplit_[dim] = myDenDstg.el_.rowSplit_.at(dim);
    }
    if (myDenDstg.ss_.peSfpSplit_.count(dim)) {
      newDstg.ss_.peSfpSplit_[dim] = myDenDstg.ss_.peSfpSplit_.at(dim);
      newDstg.el_.peSfpSplit_[dim] = myDenDstg.el_.peSfpSplit_.at(dim);
    }
    if (myDenDstg.ss_.symbolicDimInfo_.count(dim)) {
      newDstg.ss_.symbolicDimInfo_[dim] =
          myDenDstg.ss_.symbolicDimInfo_.at(dim);
      newDstg.el_.symbolicDimInfo_[dim] =
          myDenDstg.el_.symbolicDimInfo_.at(dim);
    }
    for (const auto& [symDims, volumeLimit] :
         myDenDstg.ss_.maxSymbolicVolume_) {
      if (!symDims.count(dim)) continue;
      auto maxSymVolIt = newDstg.ss_.maxSymbolicVolume_.find(symDims);
      if (maxSymVolIt == newDstg.ss_.maxSymbolicVolume_.end() ||
          maxSymVolIt->second > volumeLimit) {
        newDstg.ss_.maxSymbolicVolume_.insert_or_assign(symDims, volumeLimit);
        // will copy into el dstg later
      }
    }
  }

  newDstg.ss_.pruneMaxSymbolicVolumes(coreDs);
  newDstg.el_.maxSymbolicVolume_ = newDstg.ss_.maxSymbolicVolume_;

  // run in a separate loop so that all symbolic, unpadded, window dims are set
  for (auto& [dim, dsIdx] : denDsForDim) {
    auto& myDenDstg = dataStageParam_.at(dsIdx);
    // padding info
    if (padding.getPadding(dim) != PadType::NOPAD) {
      if (!myDenDstg.ss_.paddingSizes_.count(dim)) {
        DT_ERROR("Dim with padding is missing padding info");
      }
      auto& padInfo = newDstg.ss_.paddingSizes_
                          .emplace(dim, myDenDstg.ss_.paddingSizes_.at(dim))
                          .first->second;
      if (node->nodeType_ != dsc2::ScheduleNode::ALLOCATE) {
        // we never want to transfer unneeded padding
        padInfo.unneededPad_ = padInfo.unneededPadFront_ =
            padInfo.unneededPadBack_ = 0;
        int totalPadSize = newDstg.ss_.primaryDimToVal_st(
            dim, SenComponents::NO_COMPONENT, -1, -1,
            {dim, PadType::PADDED_FULLSPAN_WUNNEEDED});
        if (stickSizes.count(dim) && totalPadSize < stickSizes.at(dim)) {
          // in stick unneeded padding needs to be preserved
          padInfo.unneededPad_ = stickSizes.at(dim) - totalPadSize;
        }
      }
      newDstg.el_.paddingSizes_.emplace(dim, padInfo);
    }
  }

  newDstg.ss_.compound();
  newDstg.el_.compound();
  return newDstg;
}

// ===== e016_getSizeDataStageForNode_2arg  L3  4L  dsc/dsc2.cpp:3611-3614 =====
dsc2::DataStage DesignSpaceConfig::getSizeDataStageForNode(
    const dsc2::ScheduleNode* node, const dsc2::AllocateNode* alloc) const {
  return getSizeDataStageForNode(node, alloc->ldsIdx_, alloc->padding_);
}

// ===== e017_getBufferCapacityForNodePerDimCustomLocation  L4  211L  dsc/dsc2.cpp:3754-3964 =====
std::vector<std::pair<PrimaryDimTypes, int>>
DesignSpaceConfig::getBufferCapacityForNodePerDimCustomLocation(
    const dsc2::ScheduleNode* nodeForLocation, const dsc2::ScheduleNode* node,
    int ldsIdx, SenComponents comp, int corelet, int row,
    const bool doNotRound /*= false*/, const bool includeGaps /*= true*/,
    bool allowSymbolicVolumeLimit /*= false*/) const {
  std::vector<PrimaryDimTypes> ldims = getLayoutDims(ldsIdx);
  auto& myLds = labeledDs_.at(ldsIdx);
  auto* myAllocNode = node->nodeType_ == dsc2::ScheduleNode::ALLOCATE
                          ? static_cast<const dsc2::AllocateNode*>(node)
                          : myLds.memOrg_.at(comp).allocateNode_;
  auto myDstg = getSizeDataStageForNode(nodeForLocation, myAllocNode);
  bool noRowView = false;
  bool noCoreletView = false;
  if (!is_any_of(comp, PTARF, PTIRF, PTXRF)) {
    row = -1;
    if (!is_any_of(comp, SFPLRF, PELRF, L0, L0_SCALE, SFPSTATE, PESTATE))
      corelet = -1;
  }

  const auto& effectiveCoord =
      (row != -1 && myAllocNode->sliceViewCoordinates_.foldConstructed())
          ? myAllocNode->sliceViewCoordinates_
          : myAllocNode->allocateCoordinates_;
  const bool hasCoordinate = effectiveCoord.foldConstructed();

  // collect symbolic volume limit
  std::pair<std::set<PrimaryDimTypes>, int> currentVolumeLimit;
  if (!myAllocNode->ignoreSymbolicVolumeLimits_) {
    for (const auto& volLimPair : myDstg.ss_.maxSymbolicVolume_) {
      bool matchingDim = false, nonMatchingDim = false;
      for (auto& dim : volLimPair.first) {
        bool match = is_any_of(dim, ldims);
        matchingDim |= match;
        nonMatchingDim |= !match;
      }
      if (matchingDim) {
        DT_CHECK_MSG(
            allowSymbolicVolumeLimit,
            "Symbolic volume limit detected but function call disallows it");
        DT_CHECK_MSG(!nonMatchingDim,
                     "Tensor partially matching symbolic volume limit");
        DT_CHECK_MSG(
            currentVolumeLimit.first.empty(),
            "Only one applicalbe symbolic volume limit per tensor allowed");
        currentVolumeLimit = volLimPair;
      }
    }
  }

  auto stickSizePerDim = getCumulativeStickSizes(myLds.dsType_);
  std::vector<std::pair<PrimaryDimTypes, int>> sizePerDim;
  const auto pageSize = myAllocNode->getPageSize();
  bool symVolumeInProgress = false;
  for (auto& entry : ldims) {
    bool dimIsSymbolic = myDstg.ss_.symbolicDimInfo_.count(entry);
    bool dimInSymVolume = currentVolumeLimit.first.count(entry);
    DT_CHECK_MSG(!(symVolumeInProgress && !dimInSymVolume),
                 "Non symbolic dim in layout between symbolic dims with sym "
                 "volume limit");
    symVolumeInProgress |= dimInSymVolume;
    currentVolumeLimit.first.erase(entry);
    bool lastInSymVolume = dimInSymVolume && currentVolumeLimit.first.empty();
    if (lastInSymVolume) symVolumeInProgress = false;

    int dimSize = -1;
    auto dimIdx = getDimIndexInLayoutOrder(myLds.dsType_, entry);
    const auto scale = myLds.scale_.at(dimIdx);
    DT_CHECK_MSG(!(dimInSymVolume && scale < 0),
                 "Cannot handle symbolic volume limit and negative scale");
    if (scale == -1) {
      dimSize = 1;
      DT_CHECK(!pageSize.count(entry));
    } else if (scale == -2) {
      dimSize = stickSizePerDim.at(entry);
      DT_CHECK(!pageSize.count(entry));
    } else {
      // FIXME: It is a temporaty solution to use the coordinate only when it
      // has the custom coreIdToWkSlice_. Revisit this.
      const bool useCoordinate =
          hasCoordinate && !effectiveCoord.coreIdToWkSlice_.empty();
      if (useCoordinate) {
        // Use explicit coordinate to compute.
        constexpr int coreFoldPos =
            static_cast<int>(dsc2::CoordinateFoldPosition::Core);
        constexpr int coreletFoldPos =
            static_cast<int>(dsc2::CoordinateFoldPosition::Corelet);
        constexpr int rowSplitFoldPos =
            static_cast<int>(dsc2::CoordinateFoldPosition::RowSplit);
        DT_CHECK_MSG(effectiveCoord.coordinates_.count(entry),
                     "Expect dimension in coordinate.");
        auto& dimCoord = const_cast<FoldManager<CoordinateBaseType>&>(
            effectiveCoord.coordinates_.at(entry));
        const int numFolds = dimCoord.getNumDims();
        // FIXME: A temporary solution: dimSize considers cardinality of all
        // elem arrangement folds (+ corelet and "dummy" rowsplit folds in LX)
        // (+ core and "dummy" rowsplit folds in HBM). With proper coordinate
        // setup, dimSize should be the product of the cardinalities of only the
        // element arrangement folds.
        dimSize = 1;
        for (int foldPos = 0; foldPos < numFolds; ++foldPos) {
          auto includeFold = [&effectiveCoord, &foldPos, &comp, &entry,
                              coreFoldPos, coreletFoldPos, rowSplitFoldPos]() {
            if ((effectiveCoord.getCoordinateCategoryOfPos(entry, foldPos) ==
                 dsc2::CoordinateCategory::ELEM_ARR_COORD) ||
                (comp == SenComponents::HBM &&
                 (foldPos == coreFoldPos || foldPos == rowSplitFoldPos)) ||
                (comp == SenComponents::LX &&
                 (foldPos == coreletFoldPos || foldPos == rowSplitFoldPos)))
              return true;
            else
              return false;
          };
          if (includeFold()) {
            const auto cardinality = dimCoord.getFoldDimSize(foldPos);
            DT_CHECK_MSG(
                !(foldPos == rowSplitFoldPos &&
                  (comp == SenComponents::HBM || comp == SenComponents::LX)) ||
                    cardinality == 1,
                "Expect cardinality of 1 for row split fold in HBM or LX "
                "allocation.");
            dimSize *= cardinality;
          }
        }
      } else {
        // Use data stage to compute.
        if (noRowView) row = -1;
        if (noCoreletView) corelet = -1;
        if (lastInSymVolume) {
          dimSize = currentVolumeLimit.second;
        } else if (symVolumeInProgress) {
          dimSize = -1;
        } else {
          const auto ssVal = myDstg.ss_.primaryDimToVal_st(
              entry, comp, row, corelet, myAllocNode->padding_);
          const auto elVal = myDstg.el_.primaryDimToVal_st(
              entry, comp, row, corelet, myAllocNode->padding_);
          dimSize = std::max(ssVal, elVal);
        }
        if (pageSize.count(entry)) {
          const auto dimPageSize = pageSize.at(entry);
          if (myAllocNode->indirectAllocType_ ==
              dsc2::AllocateNode::IndirectAllocType::INDEX_TENSOR) {
            DT_CHECK_MSG(!dimInSymVolume,
                         "Index tensor with symbolic volume not supported");
            dimSize = std::ceil(float(dimSize) / dimPageSize);
          } else if (myAllocNode->indirectAllocType_ ==
                     dsc2::AllocateNode::IndirectAllocType::VALUE_TENSOR) {
            if (!dimIsSymbolic) {
              dimSize = std::min(dimSize, dimPageSize);
            } else {
              // recalculate size using granularity
              const auto granularity = myDstg.ss_.primaryDimToVal_st(
                  entry, comp, row, corelet, myAllocNode->padding_, 1.0, true);
              DT_CHECK_MSG(granularity >= dimPageSize,
                           "Symbolic dim cannot have granularity smaller than "
                           "page size");
              dimSize = dimPageSize;
              if (dimInSymVolume) {
                currentVolumeLimit.second /= granularity;
              }
              if (lastInSymVolume) {
                dimSize *= currentVolumeLimit.second;
              }
            }
          } else {
            DT_ERROR("Unhandled indirect alloc type");
          }
        }
      }
      if (myLds.scaledLdsCategory_ ==
              LabeledDsInfo::ScaledLdsCategory::SCALE_TENSOR &&
          entry == myLds.mxInfo_.dim) {
        dimSize = dimSize / myLds.mxInfo_.blkSize;
        DT_CHECK(dimSize != 0);
      }
      // round to full sticks
      if (!doNotRound)
        if (auto it = stickSizePerDim.find(entry);
            it != stickSizePerDim.end()) {
          dimSize = std::ceil(float(dimSize) / it->second) * it->second;
        }
    }
    int64_t gap = 0;
    if (includeGaps && myAllocNode->backGapCore_.count(entry)) {
      DT_CHECK_MSG(!symVolumeInProgress,
                   "Gaps on dims with symbolic volume limit not handled");
      auto& dimGap = myAllocNode->backGapCore_.at(entry);
      DT_CHECK(!dimGap.empty());
      if (comp == SenComponents::HBM) {
        DT_CHECK(dimGap.count(-1));
        gap = dimGap.at(-1);
      } else {
        gap = dimGap.begin()->second;
        for (const auto& coreId : coreIdsUsed_) {
          auto it = dimGap.find(coreId);
          DT_CHECK_MSG(it != dimGap.end() && it->second == gap,
                       "At the moment only uniform LX back gap across cores is "
                       "supported");
        }
      }
    }
    sizePerDim.emplace_back(entry, dimSize + gap);
  }
  for (auto gapStickSpread : myAllocNode->gapStickSpread_)
    for (auto& sizeDim : sizePerDim)
      if (gapStickSpread.first == sizeDim.first)
        sizeDim.second /= gapStickSpread.second;

  return sizePerDim;
}

// ===== e018_getBufferCapacityForNodePerDim  L5  10L  dsc/dsc2.cpp:3966-3975 =====
std::vector<std::pair<PrimaryDimTypes, int>>
DesignSpaceConfig::getBufferCapacityForNodePerDim(
    const dsc2::ScheduleNode* node, int ldsIdx, SenComponents comp, int corelet,
    int row, const bool doNotRound /*= false*/,
    const bool includeGaps /*= true*/,
    bool allowSymbolicVolumeLimit /*= false*/) const {
  return getBufferCapacityForNodePerDimCustomLocation(
      node, node, ldsIdx, comp, corelet, row, doNotRound, includeGaps,
      allowSymbolicVolumeLimit);
}

// ===== e019_getBufferCapacityForNode  L6  29L  dsc/dsc2.cpp:3977-4005 =====
int64_t DesignSpaceConfig::getBufferCapacityForNode(
    const dsc2::ScheduleNode* node, int ldsIdx, SenComponents comp, int corelet,
    int row, const uint64_t bytesPerStick /*= 0*/,
    const bool forceEvenNumSticks /*= false*/,
    const bool doNotRound /*= false*/,
    const bool includeGaps /*= true*/) const {
  auto& myLds = labeledDs_.at(ldsIdx);
  auto* myAllocNode = node->nodeType_ == dsc2::ScheduleNode::ALLOCATE
                          ? static_cast<const dsc2::AllocateNode*>(node)
                          : myLds.memOrg_.at(comp).allocateNode_;

  // Size per dimension
  auto sizePerDim = getBufferCapacityForNodePerDim(
      myAllocNode, ldsIdx, comp, corelet, row, doNotRound, includeGaps, true);

  // Accumulate
  int64_t cap = 1;
  for (auto [_, size] : sizePerDim) cap *= std::max(size, 1);
  cap *= myLds.wordLength;

  if (myAllocNode->numBuffers_ >= 1 && comp == SenComponents::LX &&
      forceEvenNumSticks) {
    // If it is double buffering on LX, the size of each buffer must be an even
    // number of the multiple of the stick size.
    DT_CHECK_MSG(bytesPerStick > 0, "Invalid bytes per stick.");
    cap = ((cap / bytesPerStick) % 2) ? cap + bytesPerStick : cap;
  }
  return cap;
}

// TOTAL: 19 definitions, 761 lines.
