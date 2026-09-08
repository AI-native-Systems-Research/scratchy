// bridge1.cpp -- IBM Spyre deeptools bridge 1: SuperDSC -> DataflowIR.
//
// Every body below is VERBATIM from the authority tree
//   /Users/nickm/git/deeptools-src   (repo_info.txt: deeptools|master|a0d29abbed...)
// and each banner gives that body's ORIGINAL <file>:<line>. Port from the authority
// file at the cited line; this unit tells you WHICH function and in WHAT ORDER.
//
// 110 units. Member definitions are rewritten as free functions
// (Class::foo -> eNNN_foo) so the unit needs no class declarations; bodies are untouched.
#include "prelude.inc"


// ================================================================================================
// LEVEL 0
// ================================================================================================

// ---- 1/110  checkConstraints  --  ddc/ddcv1.cpp:792  (132L)   [lambda nested in Ddc::exploreAssignDataStages]
static auto e001_checkConstraints = [&](const DataStructDims& ds, const Metadata::Datastage& dsMetadata,
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

// ---- 2/110  createDataConnectMetadata  --  ddc/ddcv1.cpp:3283  (45L)
void e002_createDataConnectMetadata() {
  auto& dcMap = metadata.dataConnects_;
  dcMap.clear();
  for (auto* node : currDsc->scheduleTree_.traverseTreeDFSMutable(
           nullptr,
           {dsc2::ScheduleNode::COMPUTE, dsc2::ScheduleNode::TRANSFER})) {
    // collect all data_connects from node
    if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      const auto* transfer = static_cast<dsc2::TransferNode*>(node);
      if (transfer->src_.unit_ != SenComponents::CONSTANT) {
        dcMap[transfer->srcLdsAndLoopOffsets_.dataConnect_].insertConsumer(
            node);
      }
      for (const auto& di : transfer->dstLdsAndLoopOffsets_)
        dcMap[di.dataConnect_].insertProducer(node);
    } else if (node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      auto* compute = static_cast<dsc2::ComputeNode*>(node);
      for (size_t ind = 0, e = compute->inputsLdsAndLoopOffsets_.size();
           ind < e; ++ind) {
        if (compute->inputs_[ind] != SenComponents::CONSTANT) {
          dcMap[compute->inputsLdsAndLoopOffsets_[ind].dataConnect_]
              .insertConsumer(node);
        }
      }
      for (const auto& di : compute->outputsLdsAndLoopOffsets_)
        dcMap[di.dataConnect_].insertProducer(node);
      // Process data_connetcs for opaque operations.
      for (const auto& dcIt : compute->instrAttribute_.input_data_connects_) {
        dcMap[dcIt].insertConsumer(node);
      }
      for (const auto& dcIt : compute->instrAttribute_.output_data_connects_) {
        dcMap[dcIt].insertProducer(node);
      }
    }
  }

  // For each label of a dataconnect, check that the corresponding producer is
  // not empty.
  for (const auto& [label, dc] : dcMap) {
    if (dc.producers_.empty()) {
      DT_ERROR("Illegal DDL: data_connect " + label +
               " does not have any producer.");
    }
  }
}

// ---- 3/110  startDataflowIRGeneration  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:20  (18L)
void e003_startDataflowIRGeneration() {
  if (module_op_) {
    module_op_.emitRemark("Module op is already created");
    return;
  }

  OpBuilder builder(&this->context_);
  auto loc = builder.getUnknownLoc();

  module_op_ = mlir::ModuleOp::create(loc);
  auto funcType = builder.getFunctionType(mlir::TypeRange{}, mlir::TypeRange{});
  dataflow_func_op_ =
      mlir::func::FuncOp::create(loc, "dataflowProgram", funcType);
  module_op_.push_back(dataflow_func_op_);

  // add entry block to the function (only once for the function)
  auto &entry_block = *dataflow_func_op_.addEntryBlock();
}

// ---- 4/110  stopDataflowIRGeneration  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:39  (15L)
void e004_stopDataflowIRGeneration() {
  OpBuilder builder(dataflow_func_op_);
  if (dataflow_func_op_.front().empty()) {
    builder.setInsertionPointToStart(&dataflow_func_op_.front());
  } else {
    builder.setInsertionPointAfter(&dataflow_func_op_.front().back());
  }

  auto return_op =
      mlir::func::ReturnOp::create(builder, dataflow_func_op_.getLoc());

  if (failed(mlir::verify(module_op_))) {
    DT_ERROR("module verification error");
  }
}

// ---- 5/110  areFoldedAddressesSameAcrossFoldsAndCoresForGivenCorelet  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:226  (21L)
bool e005_areFoldedAddressesSameAcrossFoldsAndCoresForGivenCorelet(
    std::vector<int> &core_ids_used, std::vector<int> &corelet_ids_used,
    const FoldManager<int64_t> &addresses) {
  DT_CHECK(!core_ids_used.empty());
  DT_CHECK(!corelet_ids_used.empty());

  for (auto &core_id : core_ids_used) {
    for (auto &corelet_id : corelet_ids_used) {
      auto foldedAddresses =
          addresses.getAllDataWithMapUnrolled({{0, core_id}, {1, corelet_id}});
      DT_CHECK_MSG(is_any_of(foldedAddresses.size(), 1, num_folds_),
                   "Fold addresses can either be constant or should be "
                   "available for each fold");
      if (foldedAddresses.size() != 1) {
        return false;
      }
    }
  }

  return true;
}

// ---- 6/110  getTranslatorVersion  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:22  (15L)
inline mlir::LogicalResult e006_getTranslatorVersion(const SuperDsc &sdsc,
                                                int &version) {
  version = 3;
  for (const auto &dsc : sdsc.dscs_) {
    if (dsc.computeOp_.empty()) {
      version = 1;
      return mlir::failure();
    }
    if (!dsc.isDSC2()) {
      version = 1;
      return mlir::success();
    }
  }
  return mlir::success();
}

// ---- 7/110  createGetLocalUnitOp  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:44  (13L)
Value e007_createGetLocalUnitOp(OpBuilder &builder,
                                             SenComponents comp,
                                             Value unit_op) {
  if (component_to_handler_.count(comp) == 0) {
    auto unit_name = EnumsConversion::senComponentsToString.at(comp);
    auto local_unit_op = dataflow::GetLocalUnitOp::create(
        builder, builder.getUnknownLoc(), builder.getIndexType(), unit_op,
        unit_name);
    return local_unit_op.getResult();
  } else {
    return component_to_handler_[comp];
  }
}

// ---- 8/110  setPrecisionInUnitOp  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:143  (12L)
void e008_setPrecisionInUnitOp(const SenComponents &comp,
                                            dataflow::ProgramUnitOp &unit_op,
                                            std::string &precision) const {
  auto record = EnumsConversion::senCompToGenericComp.find(comp);
  if (record != EnumsConversion::senCompToGenericComp.end()) {
    if (is_any_of(record->second, PT, PE, SFP)) {
      OpBuilder builder(unit_op);
      DT_CHECK_MSG(precision != "", "Invalid compute precision");
      unit_op->setAttr("precision", builder.getStringAttr(precision));
    }
  }
}

// ---- 9/110  constructAVectorOfIndexType  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:711  (5L)
void e009_constructAVectorOfIndexType(
    int size, OpBuilder &builder, SmallVectorImpl<mlir::Type> &type_vector) {
  for (int i = 0; i < size; i++)
    type_vector.emplace_back(builder.getIndexType());
}

// ---- 10/110  emitError  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:718  (4L)
void e010_emitError(std::string message) {
  module_op_->emitError("[DSC2.0 to Dataflow IR]: " + message);
  return;
}

// ---- 11/110  constructLogicalMemoryViewOp  --  dsc-based-utils/DSC2ToDataflowIR/DataflowIRConstructionUtils.hpp:33  (22L)
static LogicalResult e011_constructLogicalMemoryViewOp(
    OpBuilder &builder, Value storage_unit_op, Value start_address,
    dataflow::GetLogicalMemoryViewOp &view, Type element_type) {
  llvm::SmallVector<int64_t, 8> extents;
  auto context = element_type.getContext();
  AffineExpr layout_expr = getAffineConstantExpr(0, context);
  auto dim = getAffineDimExpr(0, context);
  auto mul = getAffineConstantExpr(1, context);
  auto mul_expr = getAffineBinaryOpExpr(AffineExprKind::Mul, mul, dim);
  layout_expr =
      getAffineBinaryOpExpr(AffineExprKind::Add, mul_expr, layout_expr);
  extents.insert(extents.end(), 1);

  AffineMap layout_map =
      AffineMap::getMultiDimIdentityMap(1, builder.getContext());

  auto memref_type = MemRefType::get(extents, element_type);
  view = dataflow::GetLogicalMemoryViewOp::create(
      builder, builder.getUnknownLoc(), memref_type, storage_unit_op,
      start_address, layout_map);
  return LogicalResult::success();
}

// ---- 12/110  convertPrecisionToType  --  dsc-based-utils/DSC2ToDataflowIR/DataflowIRConstructionUtils.hpp:130  (20L)
static Type e012_convertPrecisionToType(OpBuilder &builder,
                                   const std::string type_string) {
  if (type_string == "fp16")
    return builder.getF16Type();
  else if (type_string == "bf16")
    return builder.getBF16Type();
  else if (type_string == "fp32")
    return builder.getF32Type();
  else if (type_string == "f8E4M3FN")
    return mlir::Float8E4M3FNType::get(builder.getContext());
  else if (type_string == "f8E5M2")
    return mlir::Float8E5M2Type::get(builder.getContext());
  else if (type_string == "int16")
    return builder.getIntegerType(16);
  else if (type_string == "int8")
    return builder.getIntegerType(8);
  else if (type_string == "int4")
    return builder.getIntegerType(4);
  llvm_unreachable("unknown type string");
}

// ---- 13/110  convertTypeToString  --  dsc-based-utils/DSC2ToDataflowIR/DataflowIRConstructionUtils.hpp:151  (19L)
static std::string e013_convertTypeToString(OpBuilder &builder, Type type) {
  if (type == builder.getF16Type())
    return "fp16";
  else if (type == builder.getBF16Type())
    return "bf16";
  else if (type == builder.getF32Type())
    return "fp32";
  else if (type == mlir::Float8E4M3FNType::get(builder.getContext()))
    return "f8E4M3FN";
  else if (type == mlir::Float8E5M2Type::get(builder.getContext()))
    return "f8E5M2";
  else if (type == builder.getIntegerType(16))
    return "i16";
  else if (type == builder.getIntegerType(8))
    return "i8";
  else if (type == builder.getIntegerType(4))
    return "i4";
  llvm_unreachable("unknown type");
}

// ---- 14/110  constructTypeFromFormat  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:278  (91L)
LogicalResult e014_constructTypeFromFormat(
    OpBuilder &builder, DataFormats format,
    LabeledDsInfo::ScaledLdsCategory category, SenComponents comp,
    int num_elements, Type &result_type, Type &element_type, bool &is_integer) {
  if (format == DataFormats::SENINT8) {
    is_integer = true;
    element_type = builder.getIntegerType(8);
  } else if (format == DataFormats::SENINT4) {
    is_integer = true;
    element_type = builder.getIntegerType(4);
  } else if (format == DataFormats::SENINT2) {
    is_integer = true;
    element_type = builder.getIntegerType(2);
  } else if (format == DataFormats::SENINT24) {
    is_integer = true;
    if (EnumsConversion::senCompToGenericComp.at(comp) == PT) {
      element_type = builder.getIntegerType(24);
    } else {
      element_type = builder.getIntegerType(16);
    }
  } else if (format == DataFormats::SEN143_FP8) {
    is_integer = false;
    if (category == LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR) {
      element_type = mlir::Float8E4M3FNType::get(builder.getContext());
    } else {
      element_type = dataflow::CustomMXFloatType::get(builder.getContext(), 8);
    }
  } else if (format == DataFormats::SEN080_FP8) {
    is_integer = false;
    if (category == LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR) {
      element_type = mlir::Float8E8M0FNUType::get(builder.getContext());
    } else {
      element_type = dataflow::CustomMXFloatType::get(builder.getContext(), 8);
    }
  } else if (format == DataFormats::SEN169_FP16) {
    is_integer = false;
    element_type = builder.getF16Type();
  } else if (format == DataFormats::SEN121_FP4) {
    is_integer = false;
    if (category == LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR) {
      element_type = mlir::Float4E2M1FNType::get(builder.getContext());
    } else {
      element_type = dataflow::CustomMXFloatType::get(builder.getContext(), 4);
    }
  } else if (format == DataFormats::SEN053_FP8) {
    is_integer = false;
    if (category == LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR) {
      // currently, using E4M3 instead of E5M3 because of lack of that
      // availability in MLIR.
      element_type = mlir::Float8E4M3FNType::get(builder.getContext());
    } else {
      element_type = dataflow::CustomMXFloatType::get(builder.getContext(), 8);
    }
  } else if (format == DataFormats::BFLOAT16) {
    is_integer = false;
    element_type = builder.getBF16Type();
  } else if (format == DataFormats::IEEE_FP32) {
    is_integer = false;
    element_type = builder.getF32Type();
  } else if (format == DataFormats::IEEE_INT32) {
    is_integer = true;
    element_type = builder.getIntegerType(32);
  } else if (format == DataFormats::IEEE_INT64) {
    is_integer = true;
    element_type = builder.getIntegerType(64);
  } else if (format == DataFormats::BOOL) {
    is_integer = true;
    element_type = builder.getIntegerType(1);
  } else if (format == DataFormats::SENUINT32) {
    is_integer = true;
    element_type = builder.getIntegerType(32);
  } else {
    DT_ERROR("Unknown conversion from DT data format to MLIR data format");
    return LogicalResult::failure();
  }

  if (num_elements == -1) {
    // TODO: Reset the width in case of boolean tensors assuming SFP/PE
    // statereg.
    int width =
        format != DataFormats::BOOL ? element_type.getIntOrFloatBitWidth() : 16;
    DT_CHECK_MSG((128 * 8) % (width) == 0,
                 ""
                 "number of elements in a stick multiplied by precision should"
                 "form a stick");
    num_elements = (128 * 8) / (width);
  }

  result_type = dataflow::utils::constructVectorType(element_type, num_elements);
  return LogicalResult::success();
}

// ---- 15/110  constructSingleValCustomVector  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:435  (15L)
Value e015_constructSingleValCustomVector(
    OpBuilder &builder, Location &loc, const dsc2::ComputeNode &compute_op,
    int64_t val, dataflow::CustomVectorType custom_vtype) {
  llvm::SmallVector<Attribute> val_attrs = {builder.getI64IntegerAttr(val)};
  auto bitstream_op = vectorchain::ConstantBitstreamOp::create(
      builder, builder.getUnknownLoc(), custom_vtype,
      builder.getArrayAttr(val_attrs));
  int index_array[1] = {0};
  auto shuffle_op = vectorchain::ShuffleOp::create(
      builder, builder.getUnknownLoc(), custom_vtype, bitstream_op.getResult(),
      nullptr, builder.getI32ArrayAttr(index_array),
      builder.getI32IntegerAttr(custom_vtype.getNumElements()),
      builder.getStringAttr(compute_op.name_));
  return shuffle_op.getResult();
}

// ---- 16/110  mapToDicAttr  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1524  (9L)
static mlir::DictionaryAttr e016_mapToDicAttr(
    OpBuilder &builder, std::map<std::string, std::string> str_str_map) {
  std::vector<NamedAttribute> vec_named_attr;
  for (auto itr : str_str_map) {
    vec_named_attr.push_back(builder.getNamedAttr(
        StringRef(itr.first), builder.getStringAttr(itr.second)));
  }
  return builder.getDictionaryAttr(vec_named_attr);
}

// ---- 17/110  getCmpIPredicate_dup  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:21  (25L)
static LogicalResult e017_getCmpIPredicate_dup(mlir::arith::CmpIPredicate &pred,
                                          CondOp condOp) {
  switch (condOp) {
    case CondOp::EQ:
      pred = mlir::arith::CmpIPredicate::eq;
      return LogicalResult::success();
    case CondOp::NE:
      pred = mlir::arith::CmpIPredicate::ne;
      return LogicalResult::success();
    case CondOp::LT:
      pred = mlir::arith::CmpIPredicate::slt;
      return LogicalResult::success();
    case CondOp::LE:
      pred = mlir::arith::CmpIPredicate::sle;
      return LogicalResult::success();
    case CondOp::GT:
      pred = mlir::arith::CmpIPredicate::sgt;
      return LogicalResult::success();
    case CondOp::GE:
      pred = mlir::arith::CmpIPredicate::sge;
      return LogicalResult::success();
    default:
      return LogicalResult::failure();
  }
}

// ---- 18/110  getMLIRLoopFromSNLoopNode  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:49  (16L)
mlir::Operation *e018_getMLIRLoopFromSNLoopNode(
    const dsc2::LoopNode *loop_node, PrimaryDimTypes dim,
    std::map<const dsc2::LoopNode *, std::vector<Operation *>>
        &dsc_loops_to_mlir_loops_map) {
  auto record = dsc_loops_to_mlir_loops_map.find(loop_node);
  if (record != dsc_loops_to_mlir_loops_map.end()) {
    for (int i = 0; i < loop_node->dims_.size(); i++) {
      if (loop_node->dims_[i].dim_ == dim) {
        int size = record->second.size();
        return record->second[size - i - 1];
      }
    }
  }

  return nullptr;
}

// ---- 19/110  getParentLoop  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:381  (26L)
Operation *e019_getParentLoop(const dsc2::LoopNode *node,
                                                PrimaryDimTypes dim) {
  if (node == nullptr || node->getPrev() == nullptr) {
    return nullptr;
  }

  const dsc2::LoopNode *parent = node->getOwnerLoop();
  while (parent != nullptr) {
    for (int i = 0; i < parent->dims_.size(); i++) {
      if (parent->dims_.at(i).dim_ == dim) {
        int p_ss_val = dsc_->dataStageParam_.at(parent->numId_)
                           .ss_.dataStageDimToVal_compView_st(dim, comp_);
        int p_el_val = dsc_->dataStageParam_.at(parent->numId_)
                           .el_.dataStageDimToVal_compView_st(dim, comp_);
        if (p_ss_val == p_el_val) {
          int size = (*dsc_loops_to_mlir_loops_map_).at(parent).size();
          return (*dsc_loops_to_mlir_loops_map_).at(parent).at(size - i - 1);
        }
      }
    }

    parent = parent->getOwnerLoop();
  }

  return nullptr;
}

// ---- 20/110  propagateBufferSwitchLoopsToRootRecursively  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:519  (50L)
LogicalResult
e020_propagateBufferSwitchLoopsToRootRecursively(
    const dsc2::BlockNode *node,
    std::vector<const dsc2::TransferNode *> &node_contrib) {
  auto children = node->getNextView(comp_, corelet_id_, core_id_);
  for (auto &child : children) {
    std::vector<const dsc2::TransferNode *> child_contrib;

    // TODO: the code for all the bodies are same.
    // Operating over BlockNode causes getNextView segmentation faults.
    // Operating over ScheduleNode doesn't have getNextView method.
    if (const auto *loop = dynamic_cast<const dsc2::LoopNode *>(child)) {
      if (failed(propagateBufferSwitchLoopsToRootRecursively(loop,
                                                             child_contrib))) {
        return LogicalResult::failure();
      }
    } else if (const auto *cond_node =
                   dynamic_cast<const dsc2::ConditionNode *>(child)) {
      if (failed(propagateBufferSwitchLoopsToRootRecursively(cond_node,
                                                             child_contrib))) {
        return LogicalResult::failure();
      }
    } else if (const auto *block_node =
                   dynamic_cast<const dsc2::BlockNode *>(child)) {
      if (failed(propagateBufferSwitchLoopsToRootRecursively(block_node,
                                                             child_contrib))) {
        return LogicalResult::failure();
      }
    }

    for (auto &transfer : child_contrib) {
      node_contrib.push_back(transfer);
      (*dsc_all_parent_loops_to_buffers_switch_map_)[node].push_back(transfer);
    }
  }

  if (const auto *loop = dynamic_cast<const dsc2::LoopNode *>(node)) {
    auto record = dsc_loops_to_buffers_switch_map_.find(loop);
    if (record != dsc_loops_to_buffers_switch_map_.end()) {
      auto transfers = (dsc_loops_to_buffers_switch_map_)[loop];
      for (auto &transfer : transfers) {
        node_contrib.push_back(transfer);
        (*dsc_all_parent_loops_to_buffers_switch_map_)[node].push_back(
            transfer);
      }
    }
  }

  return LogicalResult::success();
}

// ---- 21/110  resetIterArguments  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:738  (15L)
void e021_resetIterArguments(
    mlir::Operation *loop, llvm::SmallVectorImpl<Value> &iter_args) {
  iter_args.clear();
  if (auto for_loop = llvm::dyn_cast<mlir::affine::AffineForOp>(loop)) {
    for (auto &arg : for_loop.getRegionIterArgs()) {
      iter_args.push_back(arg);
    }
  } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop)) {
    for (auto &arg : scf_for.getRegionIterArgs()) {
      iter_args.push_back(arg);
    }
  } else {
    llvm_unreachable("Unknown loop");
  }
}

// ---- 22/110  retrieveGetUnitOpInSameCore  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:32  (35L)
Value e022_retrieveGetUnitOpInSameCore(OpBuilder &builder,
                                                 SenComponents comp,
                                                 int core_id, int corelet_id) {
  // TODO: Currently, PC forcing component to be LRFREG since DCC backend
  // doesn't understand PELRF or SFPLRF yet.
  if (is_any_of(comp, PTARF, PELRF, SFPLRF)) comp = LRFREG;

  DT_CHECK(core_id == core_id_ || this->uniformization_enabled_);
  if (component_to_handler_->find(comp) != component_to_handler_->end()) {
    auto comp_handler = (*component_to_handler_)[comp];
    auto comp_def = comp_handler.getDefiningOp();

    // corelet attribute present in only GetUnitOp.
    if (isa<dataflow::GetUnitOp>(comp_def) && is_any_of(comp, L3LU, L3SU))
      return comp_handler;
    else if (isa<dataflow::GetUnitOp>(comp_def) &&
             comp_def->hasAttr("corelet")) {
      if (comp_def->getAttr("corelet") ==
          builder.getI32IntegerAttr(corelet_id)) {
        return comp_handler;
      }
    } else {
      return comp_handler;
    }
  }

  // DT_CHECK(false && "GetUnitOp must have been created already");
  auto unit_name = EnumsConversion::senComponentsToString.at(comp);
  auto unit_op = dataflow::GetUnitOp::create(
      builder, builder.getUnknownLoc(), builder.getIndexType(),
      builder.getStringAttr(unit_name), builder.getStringAttr(unit_name));
  unit_op->setAttr("core", builder.getI32IntegerAttr(core_id));
  unit_op->setAttr("corelet", builder.getI32IntegerAttr(corelet_id));
  return unit_op.getResult(0);
}

// ---- 23/110  createGetUnitOpInDifferentCore  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:73  (18L)
dataflow::GetUnitOp e023_createGetUnitOpInDifferentCore(
    OpBuilder &builder, SenComponents comp, int core_id, int corelet_id,
    int num_folds) {
  DT_CHECK(core_id != core_id_);
  auto unit_name = EnumsConversion::senComponentsToString.at(comp);

  SmallVector<Type> get_unit_type;
  for (int i = 0; i < num_folds_; i++)
    get_unit_type.push_back(builder.getIndexType());

  auto unit_op = dataflow::GetUnitOp::create(
      builder, builder.getUnknownLoc(), get_unit_type,
      builder.getStringAttr(unit_name), builder.getStringAttr(unit_name));
  unit_op->setAttr("core", builder.getI32IntegerAttr(core_id));
  unit_op->setAttr("corelet", builder.getI32IntegerAttr(corelet_id));
  unit_op->setAttr("num_folds", builder.getI32IntegerAttr(num_folds));
  return unit_op;
}

// ---- 24/110  getMLIRTypeFromDSCDataFormat  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:97  (53L)
Type e024_getMLIRTypeFromDSCDataFormat(
    DataFormats format, LabeledDsInfo::ScaledLdsCategory category,
    OpBuilder &builder) {
  if (format == DataFormats::SENINT2) {
    return builder.getIntegerType(2);
  } else if (format == DataFormats::SENINT4) {
    return builder.getIntegerType(4);
  } else if (format == DataFormats::SENINT8) {
    return builder.getIntegerType(8);
  } else if (format == DataFormats::SENINT24) {
    return builder.getIntegerType(24);
  } else if (format == DataFormats::SENUINT32) {
    return builder.getI32Type();
  } else if (format == DataFormats::SEN143_FP8) {
    if (category == LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR) {
      return mlir::Float8E4M3FNType::get(builder.getContext());
    } else {
      return dataflow::CustomMXFloatType::get(builder.getContext(), 8);
    }
  } else if (format == DataFormats::SEN080_FP8) {
    if (category == LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR) {
      return mlir::Float8E8M0FNUType::get(builder.getContext());
    } else {
      return dataflow::CustomMXFloatType::get(builder.getContext(), 8);
    }
  } else if (format == DataFormats::SEN169_FP16) {
    return builder.getF16Type();
  } else if (format == DataFormats::BFLOAT16) {
    return builder.getBF16Type();
  } else if (format == DataFormats::IEEE_FP32) {
    return builder.getF32Type();
  } else if (format == DataFormats::SEN121_FP4) {
    if (category == LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR) {
      return mlir::Float4E2M1FNType::get(builder.getContext());
    } else {
      return dataflow::CustomMXFloatType::get(builder.getContext(), 4);
    }
  } else if (format == DataFormats::SEN053_FP8) {
    if (category == LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR) {
      // temporarily using E4M3 since E5M3 don't exist in MLIR.
      return mlir::Float8E4M3FNType::get(builder.getContext());
    } else {
      return dataflow::CustomMXFloatType::get(builder.getContext(), 8);
    }
  } else if (format == DataFormats::IEEE_INT32) {
    return builder.getIntegerType(32);
  } else if (format == DataFormats::IEEE_INT64) {
    return builder.getIntegerType(64);
  }

  DT_ERROR("Unknown data format");
  return builder.getNoneType();
}

// ---- 25/110  getAddressGranularityMultiplyFactor  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:156  (17L)
double e025_getAddressGranularityMultiplyFactor(SenComponents unit,
                                                          SenComponents storage,
                                                          Type precision) {
  DataLocation loc;
  loc.unit_ = EnumsConversion::senCompToGenericComp.at(unit);
  loc.storage_ = storage;
  auto factor = (double)dscGlobal().sysDef.addressGranularityScalePerUnit.at(loc);
  factor = factor * 8;
  unsigned bitwidth = dataflow::utils::getIntOrFloatBitWidth(precision);
  if (bitwidth == 24) {
    factor = factor / 16.0;
  } else {
    factor = factor / bitwidth;
  }

  return factor;
}

// ---- 26/110  emitError  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:175  (5L)
void e026_emitError(std::string message) {
  module_op_->emitError("[DSC2.0 to Dataflow IR]: " + message);
  DT_ERROR("Error encountered during DSC2 to DFIR translation for the node: " +
           this->dsc_->name_);
}

// ---- 27/110  setBuilderForDataTransfer  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:185  (32L)
LogicalResult e027_setBuilderForDataTransfer(OpBuilder &builder,
                                                       Operation *mlir_loop,
                                                       bool is_outer,
                                                       bool is_before) {
  if (is_outer && is_before) {
    builder.setInsertionPoint(mlir_loop);
  } else if (is_outer && !is_before) {
    builder.setInsertionPointAfter(mlir_loop);
  } else if (!is_outer && is_before) {
    if (auto affine_for =
            llvm::dyn_cast<mlir::affine::AffineForOp>(mlir_loop)) {
      builder.setInsertionPointToStart(affine_for.getBody());
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(mlir_loop)) {
      builder.setInsertionPointToStart(scf_for.getBody());
    } else {
      // By construction, it shouldn't happen.
      return LogicalResult::failure();
    }
  } else {
    if (auto affine_for =
            llvm::dyn_cast<mlir::affine::AffineForOp>(mlir_loop)) {
      builder.setInsertionPoint(affine_for.getBody()->getTerminator());
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(mlir_loop)) {
      builder.setInsertionPoint(scf_for.getBody()->getTerminator());
    } else {
      // By construction, it shouldn't happen.
      return LogicalResult::failure();
    }
  }

  return LogicalResult::success();
}

// ---- 28/110  getMLIRLoopFromLoopNode  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:220  (14L)
mlir::Operation *e028_getMLIRLoopFromLoopNode(
    const dsc2::LoopNode *loop_node, PrimaryDimTypes dim) {
  auto record = dsc_loops_to_mlir_loops_map_->find(loop_node);
  if (record != dsc_loops_to_mlir_loops_map_->end()) {
    for (int i = 0; i < loop_node->dims_.size(); i++) {
      if (loop_node->dims_[i].dim_ == dim) {
        int size = record->second.size();
        return record->second[size - i - 1];
      }
    }
  }

  return nullptr;
}

// ---- 29/110  constructUniformizedAddress  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:236  (47L)
Value e029_constructUniformizedAddress(
    OpBuilder &builder, const std::map<int, std::map<int, int64_t>> &addresses,
    double factor) {
  // check if all addresses are same. if so, don't create uniform operations.
  bool is_all_addresses_same = true;
  DT_CHECK(!this->core_ids_used_.empty());
  DT_CHECK(!addresses.empty() && !addresses.begin()->second.empty());
  int first_val = addresses.begin()->second.begin()->second;
  for (auto &core_id : this->core_ids_used_) {
    for (auto &corelet_id : this->corelet_ids_used_) {
      if (addresses.at(core_id).at(corelet_id) != first_val) {
        is_all_addresses_same = false;
      }
    }
  }

  if (is_all_addresses_same) {
    auto start_addr = int(first_val * factor);
    auto const_op = mlir::arith::ConstantIndexOp::create(
        builder, builder.getUnknownLoc(), start_addr);
    return const_op.getResult();
  } else {
    SmallVector<mlir::Value> values;
    for (auto &core_id : this->core_ids_used_) {
      for (auto &corelet_id : this->corelet_ids_used_) {
        auto start_addr = int(addresses.at(core_id).at(corelet_id) * factor);
        auto const_op = mlir::arith::ConstantIndexOp::create(
            builder, builder.getUnknownLoc(), start_addr);
        for (int fold = 0; fold < num_folds_; fold++)
          values.push_back(const_op.getResult());
      }
    }

    // create a immutable map
    DT_CHECK((this->units_involved_)->size() == values.size());
    auto map = uniform::DefImmutableMappingOp::create(
        builder, builder.getUnknownLoc(), builder.getIndexType(),
        *(this->units_involved_), values);

    // create a query op
    auto result = uniform::QueryMapOp::create(
        builder, map.getLoc(), builder.getIndexType(), map.getResult(),
        this->program_unit_iterator_);

    return result.getResult();
  }
}

// ---- 30/110  constructUniformizedFoldedAddress  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:285  (51L)
Value e030_constructUniformizedFoldedAddress(
    OpBuilder &builder, const FoldManager<int64_t> &addresses, double factor) {
  // check if all addresses are same. if so, don't create uniform operations.
  DT_CHECK(!this->core_ids_used_.empty());
  auto allAddresses = addresses.getAllData();
  const bool is_all_addresses_same = std::equal(
      allAddresses.begin() + 1, allAddresses.end(), allAddresses.begin());

  if (is_all_addresses_same) {
    auto start_addr = int(allAddresses.at(0) * factor);
    auto const_op = mlir::arith::ConstantIndexOp::create(
        builder, builder.getUnknownLoc(), start_addr);
    return const_op.getResult();
  } else {
    SmallVector<mlir::Value> values;
    for (auto &core_id : this->core_ids_used_) {
      for (auto &corelet_id : this->corelet_ids_used_) {
        auto foldedAddresses = addresses.getAllDataWithMapUnrolled(
            {{0, core_id}, {1, corelet_id}});
        DT_CHECK_MSG(is_any_of(foldedAddresses.size(), 1, num_folds_),
                     "Fold addresses can either be constant or should be "
                     "available for each fold");
        for (int fold_id = 0; fold_id < num_folds_; fold_id++) {
          int start_addr = 0;
          if (foldedAddresses.size() != 1) {
            start_addr = int(foldedAddresses[fold_id] * factor);
          } else {
            start_addr = int(foldedAddresses[0] * factor);
          }

          auto const_op = mlir::arith::ConstantIndexOp::create(
              builder, builder.getUnknownLoc(), start_addr);
          values.push_back(const_op.getResult());
        }
      }
    }

    // create a immutable map
    DT_CHECK((this->units_involved_)->size() == values.size());
    auto map = uniform::DefImmutableMappingOp::create(
        builder, builder.getUnknownLoc(), builder.getIndexType(),
        *(this->units_involved_), values);

    // create a query op
    auto result = uniform::QueryMapOp::create(
        builder, map.getLoc(), builder.getIndexType(), map.getResult(),
        this->program_unit_iterator_);

    return result.getResult();
  }
}

// ---- 31/110  constructUniformizedFoldedDoubleBufferToggling  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:339  (83L)
Value e031_constructUniformizedFoldedDoubleBufferToggling(
    OpBuilder &builder,
    const std::map<int, std::map<int, int64_t>> &buffer_addresses,
    const FoldManager<int64_t> &start_addresses, double factor) {
  // check if all addresses are same. if so, don't create uniform operations.
  bool is_all_addresses_same = true;
  DT_CHECK(!this->core_ids_used_.empty());
  DT_CHECK(!buffer_addresses.empty() &&
           !buffer_addresses.begin()->second.empty());
  int first_val = buffer_addresses.begin()->second.begin()->second;
  first_val += 2 * start_addresses.getSingleData();
  first_val = int(first_val * factor);

  for (auto &core_id : this->core_ids_used_) {
    for (auto &corelet_id : this->corelet_ids_used_) {
      auto foldedAddresses = start_addresses.getAllDataWithMapUnrolled(
          {{0, core_id}, {1, corelet_id}});
      DT_CHECK_MSG(is_any_of(foldedAddresses.size(), 1, num_folds_),
                   "Fold addresses can either be constant or should be "
                   "available for each fold");
      for (int fold_id = 0; fold_id < num_folds_; fold_id++) {
        long long foldedAddr = 0;
        if (foldedAddresses.size() != 1) {
          foldedAddr = foldedAddresses[fold_id];
        } else {
          foldedAddr = foldedAddresses[0];
        }

        int buff_addr_offset =
            int((buffer_addresses.at(core_id).at(corelet_id) + 2 * foldedAddr) *
                factor);
        if (buff_addr_offset != first_val) {
          is_all_addresses_same = false;
        }
      }
    }
  }

  if (is_all_addresses_same) {
    auto const_op = mlir::arith::ConstantIndexOp::create(
        builder, builder.getUnknownLoc(), first_val);
    return const_op.getResult();
  }

  SmallVector<mlir::Value> values;
  for (auto &core_id : this->core_ids_used_) {
    for (auto &corelet_id : this->corelet_ids_used_) {
      auto foldedAddresses = start_addresses.getAllDataWithMapUnrolled(
          {{0, core_id}, {1, corelet_id}});
      DT_CHECK_MSG(is_any_of(foldedAddresses.size(), 1, num_folds_),
                   "Fold addresses can either be constant or should be "
                   "available for each fold");
      for (int fold_id = 0; fold_id < num_folds_; fold_id++) {
        long long foldedAddr = 0;
        if (foldedAddresses.size() != 1) {
          foldedAddr = foldedAddresses[fold_id];
        } else {
          foldedAddr = foldedAddresses[0];
        }

        int buff_addr_offset =
            int((buffer_addresses.at(core_id).at(corelet_id) + 2 * foldedAddr) *
                factor);
        auto const_op = mlir::arith::ConstantIndexOp::create(
            builder, builder.getUnknownLoc(), buff_addr_offset);
        values.push_back(const_op.getResult());
      }
    }
  }

  // create a immutable map
  DT_CHECK((this->units_involved_)->size() == values.size());
  auto map = uniform::DefImmutableMappingOp::create(
      builder, builder.getUnknownLoc(), builder.getIndexType(),
      *(this->units_involved_), values);

  // create a query op
  auto result = uniform::QueryMapOp::create(
      builder, map.getLoc(), builder.getIndexType(), map.getResult(),
      this->program_unit_iterator_);

  return result.getResult();
}

// ---- 32/110  constructUniformizedFoldedConstantBitStream  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:462  (75L)
Value e032_constructUniformizedFoldedConstantBitStream(
    OpBuilder &builder, const FoldManager<std::vector<int64_t>> &cst_data,
    Type &bitstream_vector_type, bool is_symbolic) {
  auto all_fold_info = cst_data.getAllData();
  if (all_fold_info.size() == 1) {
    llvm::SmallVector<Attribute> val_attrs;
    for (auto &val : all_fold_info.front()) {
      // Vectorchain constant bitstream assume it as signed.
      // hence converting from uint to int. TODO: need to double check.
      auto attr = builder.getI64IntegerAttr(val);
      val_attrs.push_back(attr);
    }

    auto bitstream_op = vectorchain::ConstantBitstreamOp::create(
        builder, builder.getUnknownLoc(), bitstream_vector_type,
        builder.getArrayAttr(val_attrs));

    if (is_symbolic) {
      bitstream_op->setAttr("is_symbol", builder.getBoolAttr(true));
    }

    return bitstream_op.getResult();
  }

  // When fold is not of size 1
  SmallVector<mlir::Value> values;
  for (auto &core_id : this->core_ids_used_) {
    for (auto &corelet_id : this->corelet_ids_used_) {
      auto foldedAddresses =
          cst_data.getAllDataWithMapUnrolled({{0, core_id}, {1, corelet_id}});
      DT_CHECK_MSG(is_any_of(foldedAddresses.size(), 1, num_folds_),
                   "Fold addresses can either be constant or should be "
                   "available for each fold");
      for (int fold_id = 0; fold_id < num_folds_; fold_id++) {
        std::vector<int64_t> cst_data_per_fold;
        if (foldedAddresses.size() != 1) {
          cst_data_per_fold = foldedAddresses[fold_id];
        } else {
          cst_data_per_fold = foldedAddresses[0];
        }

        llvm::SmallVector<Attribute> val_attrs;
        for (auto &val : cst_data_per_fold) {
          // Vectorchain constant bitstream assume it as signed.
          // hence converting from uint to int. TODO: need to double check.
          auto attr = builder.getI64IntegerAttr(val);
          val_attrs.push_back(attr);
        }

        auto bitstream_op = vectorchain::ConstantBitstreamOp::create(
            builder, builder.getUnknownLoc(), bitstream_vector_type,
            builder.getArrayAttr(val_attrs));

        if (is_symbolic) {
          bitstream_op->setAttr("is_symbol", builder.getBoolAttr(true));
        }

        values.push_back(bitstream_op.getResult());
      }
    }
  }

  // create a immutable map
  DT_CHECK((this->units_involved_)->size() == values.size());
  auto map = uniform::DefImmutableMappingOp::create(
      builder, builder.getUnknownLoc(), builder.getIndexType(),
      *(this->units_involved_), values);

  // create a query op
  auto result = uniform::QueryMapOp::create(
      builder, map.getLoc(), bitstream_vector_type, map.getResult(),
      this->program_unit_iterator_);

  return result.getResult();
}

// ---- 33/110  constructUnitsForUniformization  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNSyncLowering.cpp:47  (90L)
void e033_constructUnitsForUniformization(
    OpBuilder &builder, std::vector<mlir::Value> &units) {
  // core --> list of units
  // TODO: We can get this as a utility function from dsc2.1
  std::unordered_map<int, std::unordered_map<int, std::vector<mlir::Value>>>
      core_fold_specific_units;
  auto list = sync->getComponentsFromOtherEnds(core_id_);
  for (auto &record : list) {
    auto unit = record.first;
    // skip the signals from the l0lorow1-7.
    if (!is_any_of(unit, L0LUROW1, L0LUROW2, L0LUROW3, L0LUROW4, L0LUROW5,
                   L0LUROW6, L0LUROW7)) {
      for (auto &inner_record : record.second) {
        auto core_id = inner_record.first;
        if (!is_any_of(unit, L3LU, L3SU)) {
          for (auto &corelet : inner_record.second) {
            Value to_unit =
                this->unit_to_value_map_->at(core_id).at(corelet).at(unit);
            auto to_unit_def_op = to_unit.getDefiningOp();
            for (unsigned idx = 0; idx < to_unit_def_op->getNumResults();
                 idx++) {
              core_fold_specific_units[core_id][idx].push_back(
                  to_unit_def_op->getResult(idx));
            }
          }
        } else {
          Value to_unit = this->unit_to_value_map_->at(core_id).at(-1).at(unit);
          auto to_unit_def_op = to_unit.getDefiningOp();
          for (unsigned idx = 0; idx < to_unit_def_op->getNumResults(); idx++) {
            core_fold_specific_units[core_id][idx].push_back(
                to_unit_def_op->getResult(idx));
          }
        }
      }
    }
  }

  // create groups for each core
  std::unordered_map<int, std::map<int, mlir::Value>> core_fold_specific_groups;
  for (auto &record : core_fold_specific_units) {
    for (auto &inner_record : record.second) {
      auto vals =
          core_fold_specific_units.at(record.first).at(inner_record.first);
      if (vals.size() == 1) {
        core_fold_specific_groups[record.first][inner_record.first] =
            vals.front();
      } else {
        auto group_op = dataflow::CreateGroupOp::create(
            builder, builder.getUnknownLoc(), builder.getIndexType(),
            core_fold_specific_units.at(record.first).at(inner_record.first));
        core_fold_specific_groups[record.first][inner_record.first] =
            group_op.getResult();
      }
    }
  }

  // iterate through each core and corelet and see if its relevant
  SmallVector<mlir::Value> keys;
  SmallVector<mlir::Value> values;
  for (auto &core_id : this->getDSC()->coreIdsUsed_) {
    for (int corelet_id = 0; corelet_id < this->getDSC()->numCoreletsUsed_DSC2_;
         corelet_id++) {
      if (sync->isNodeRelevant(this->comp_, corelet_id, core_id)) {
        Value to_unit =
            this->unit_to_value_map_->at(core_id).at(corelet_id).at(comp_);
        auto to_unit_def_op = to_unit.getDefiningOp();
        for (unsigned idx = 0; idx < to_unit_def_op->getNumResults(); idx++) {
          keys.push_back(to_unit_def_op->getResult(idx));
          values.push_back(core_fold_specific_groups.at(core_id).at(idx));
        }
      }
    }
  }

  DT_CHECK(keys.size() == values.size());

  // create a immutable map
  auto map = uniform::DefImmutableMappingOp::create(
      builder, builder.getUnknownLoc(), builder.getIndexType(), keys, values);

  // create a query op
  auto result = uniform::QueryMapOp::create(
      builder, map.getLoc(), builder.getIndexType(), map.getResult(),
      this->uniform_region_iterator_ ? this->uniform_region_iterator_
                                     : this->program_unit_iterator_);

  units.push_back(result.getResult());

  return;
}

// ---- 34/110  constructLogicalMemoryViewOp  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:94  (32L)
LogicalResult e034_constructLogicalMemoryViewOp(
    OpBuilder &builder, const std::vector<dsc2::ScheduleNode::Size> &view_sizes,
    Value &storage_unit_op, Value &start_address,
    dataflow::GetLogicalMemoryViewOp &view, bool bypass_viewsizes) {
  int size = 1;
  llvm::SmallVector<int64_t, 8> extents;
  auto context = result_type.getContext();
  AffineExpr layout_expr = getAffineConstantExpr(0, context);
  for (int i = 0; i < view_sizes.size(); i++) {
    auto dim = getAffineDimExpr(i, context);
    auto mul = getAffineConstantExpr(size, context);
    auto mul_expr = getAffineBinaryOpExpr(AffineExprKind::Mul, mul, dim);
    layout_expr =
        getAffineBinaryOpExpr(AffineExprKind::Add, mul_expr, layout_expr);
    size *= view_sizes[i].size_;
    extents.insert(extents.end(), view_sizes[i].size_);
  }

  AffineMap layout_map;
  if (bypass_viewsizes) {
    layout_map = AffineMap::getMultiDimIdentityMap(1, builder.getContext());
  } else {
    layout_map = AffineMap::get(view_sizes.size(), 0, layout_expr);
  }

  auto memref_type =
      MemRefType::get(extents, dataflow::utils::getElementType(result_type));
  view = dataflow::GetLogicalMemoryViewOp::create(
      builder, builder.getUnknownLoc(), memref_type, storage_unit_op,
      start_address, layout_map);
  return LogicalResult::success();
}

// ---- 35/110  constructTimeOrder  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:206  (12L)
LogicalResult e035_constructTimeOrder(
    const std::vector<dsc2::TransferNode::UnitView::LoopInfo> &composite_loops,
    AffineMap &time_order_map) {
  MLIRContext *context = unit_op_->getContext();
  llvm::SmallVector<unsigned> permutation_vector;
  for (int i = composite_loops.size() - 1; i >= 0; i--)
    permutation_vector.push_back(i);

  time_order_map = AffineMap::getPermutationMap(permutation_vector, context);

  return LogicalResult::failure();
}

// ---- 36/110  constructTimeAddressMap  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:224  (29L)
LogicalResult e036_constructTimeAddressMap(
    OpBuilder &builder, int input_dims, int output_dims,
    AffineMap &time_address_map,
    const std::vector<dsc2::TransferNode::UnitView::LoopInfo>
        &composite_loops) {
  auto context = builder.getContext();
  SmallVector<AffineExpr, 8> dims;
  for (int dim = 0; dim < input_dims; dim++) {
    auto dim_var = getAffineDimExpr(dim, context);
    dims.push_back(dim_var);
  }

  SmallVector<AffineExpr, 8> base_address_exprs;
  for (int dim = 0; dim < output_dims; dim++) {
    base_address_exprs.push_back(getAffineConstantExpr(0, context));
  }

  std::map<int, int> visited;
  for (int i = 0; i < composite_loops.size(); i++) {
    auto idx = composite_loops[i].sizeIdx_;
    if (idx != -1) {
      base_address_exprs[idx] =
          base_address_exprs[idx] + composite_loops[i].elemOffset_ * dims[i];
    }
  }

  time_address_map = AffineMap::get(input_dims, 0, base_address_exprs, context);
  return LogicalResult::success();
}

// ---- 37/110  getImmediateParentWithMatchingDim  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:670  (33L)
const dsc2::LoopNode *e037_getImmediateParentWithMatchingDim(
    const dsc2::LoopNode *node, PrimaryDimTypes dim) {
  if (node == nullptr) {
    return nullptr;
  }

  int ss_val = dsc_->dataStageParam_.at(node->denId_)
                   .ss_.dataStageDimToVal_compView_st(dim, comp_);
  int el_val = dsc_->dataStageParam_.at(node->denId_)
                   .el_.dataStageDimToVal_compView_st(dim, comp_);
  auto *parent = node;
  while (parent != nullptr) {
    if (std::find(parent->dims_.begin(), parent->dims_.end(), dim) !=
        parent->dims_.end()) {
      int p_den_ss_val = dsc_->dataStageParam_.at(parent->denId_)
                             .ss_.dataStageDimToVal_compView_st(dim, comp_);
      int p_den_el_val = dsc_->dataStageParam_.at(parent->denId_)
                             .el_.dataStageDimToVal_compView_st(dim, comp_);
      int p_num_ss_val = dsc_->dataStageParam_.at(parent->numId_)
                             .ss_.dataStageDimToVal_compView_st(dim, comp_);
      int p_num_el_val = dsc_->dataStageParam_.at(parent->numId_)
                             .el_.dataStageDimToVal_compView_st(dim, comp_);
      if (is_any_of(ss_val, -1, p_den_ss_val) &&
          is_any_of(el_val, -1, p_den_el_val) && p_num_ss_val == p_num_el_val) {
        return parent;
      }
    }

    parent = parent->getOwnerLoop();
  }

  return nullptr;
}

// ---- 38/110  areEpiloguesInTransferSizes  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1491  (12L)
bool e038_areEpiloguesInTransferSizes() {
  int ndims = src_sticks_ss_per_dim.size();
  for (auto &record : src_sticks_ss_per_dim) {
    int ss_val = record.second;
    int el_val = src_sticks_el_per_dim.at(record.first);
    if (ss_val > 1 || el_val > 1) {
      if (ss_val != el_val) return true;
    }
  }

  return false;
}

// ---- 39/110  construct2B16BLoadShuffle  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1778  (86L)
mlir::vectorchain::ShuffleOp e039_construct2B16BLoadShuffle(
    OpBuilder loop_builder, mlir::Value load_op_result, Type result_type,
    int replicationFactor) {
  vectorchain::ShuffleOp shuffle_op;
  auto element_type = dataflow::utils::getElementType(result_type);
  auto elements_total = dataflow::utils::getNumElements(result_type);
  auto new_vec_type = mlir::dataflow::utils::constructVectorType(
      element_type, replicationFactor * elements_total);
  if ((element_type.isF16() || element_type.isInteger(16))) {
    if (elements_total == 1 && replicationFactor == 64) {
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type,
          load_op_result, nullptr, loop_builder.getI32ArrayAttr({0}),
          loop_builder.getI32IntegerAttr(64),
          loop_builder.getStringAttr(transfer_->name_));
    } else if (elements_total == 16 && replicationFactor == 8) {
      int index_arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type,
          load_op_result, nullptr, loop_builder.getI32ArrayAttr(index_arr),
          loop_builder.getI32IntegerAttr(8),
          loop_builder.getStringAttr(transfer_->name_));
    } else if (elements_total == 8 && replicationFactor == 8) {
      int index_arr[] = {0, 1, 2, 3, 4, 5, 6, 7};
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type,
          load_op_result, nullptr, loop_builder.getI32ArrayAttr(index_arr),
          loop_builder.getI32IntegerAttr(8),
          loop_builder.getStringAttr(transfer_->name_));
    }
  } else if (element_type.isInteger(8)) {
    if (elements_total == 2 && replicationFactor == 64) {
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type,
          load_op_result, nullptr, loop_builder.getI32ArrayAttr({0, 1}),
          loop_builder.getI32IntegerAttr(64),
          loop_builder.getStringAttr(transfer_->name_));
    } else if (elements_total == 32 && replicationFactor == 8) {
      int index_arr[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type,
          load_op_result, nullptr, loop_builder.getI32ArrayAttr(index_arr),
          loop_builder.getI32IntegerAttr(8),
          loop_builder.getStringAttr(transfer_->name_));
    }
  } else if (element_type.isInteger(4)) {
    if (elements_total == 4 && replicationFactor == 64) {
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type,
          load_op_result, nullptr, loop_builder.getI32ArrayAttr({0, 1, 2, 3}),
          loop_builder.getI32IntegerAttr(64),
          loop_builder.getStringAttr(transfer_->name_));
    } else if (elements_total == 64 && replicationFactor == 8) {
      int index_arr[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                         39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                         52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type,
          load_op_result, nullptr, loop_builder.getI32ArrayAttr(index_arr),
          loop_builder.getI32IntegerAttr(8),
          loop_builder.getStringAttr(transfer_->name_));
    }
  } else if (element_type.isF32()) {
    if (elements_total == 4 && replicationFactor == 8) {
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type,
          load_op_result, nullptr, loop_builder.getI32ArrayAttr({0, 1, 2, 3}),
          loop_builder.getI32IntegerAttr(8),
          loop_builder.getStringAttr(transfer_->name_));
    } else if (elements_total == 32 && replicationFactor == 1) {
      int index_arr[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type,
          load_op_result, nullptr, loop_builder.getI32ArrayAttr(index_arr),
          loop_builder.getI32IntegerAttr(1),
          loop_builder.getStringAttr(transfer_->name_));
    }
  }
  return shuffle_op;
}

// ---- 40/110  construct2B16BStoreShuffle  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1865  (87L)
mlir::vectorchain::ShuffleOp e040_construct2B16BStoreShuffle(
    OpBuilder loop_builder, Value &data, Type result_type,
    int replicationFactor) {
  vectorchain::ShuffleOp shuffle_op;
  auto element_type = dataflow::utils::getElementType(result_type);
  auto element_size =
      dataflow::utils::getNumElements(result_type) / replicationFactor;
  auto new_vec_type =
      mlir::dataflow::utils::constructVectorType(element_type, element_size);
  if ((element_type.isF16() || element_type.isInteger(16))) {
    if (element_size == 1 && replicationFactor == 64) {
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type, data,
          nullptr, loop_builder.getI32ArrayAttr({0}),
          loop_builder.getI32IntegerAttr(1),
          loop_builder.getStringAttr(transfer_->name_));
    } else if (element_size == 16 && replicationFactor == 8) {
      int index_arr[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15};
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type, data,
          nullptr, loop_builder.getI32ArrayAttr(index_arr),
          loop_builder.getI32IntegerAttr(1),
          loop_builder.getStringAttr(transfer_->name_));
    } else if (element_size == 8 && replicationFactor == 8) {
      int index_arr[] = {0, 1, 2, 3, 4, 5, 6, 7};
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type, data,
          nullptr, loop_builder.getI32ArrayAttr(index_arr),
          loop_builder.getI32IntegerAttr(1),
          loop_builder.getStringAttr(transfer_->name_));
    }
  } else if (element_type.isInteger(8)) {
    if (element_size == 2 && replicationFactor == 64) {
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type, data,
          nullptr, loop_builder.getI32ArrayAttr({0, 1}),
          loop_builder.getI32IntegerAttr(1),
          loop_builder.getStringAttr(transfer_->name_));
    } else if (element_size == 32 && replicationFactor == 8) {
      int index_arr[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type, data,
          nullptr, loop_builder.getI32ArrayAttr(index_arr),
          loop_builder.getI32IntegerAttr(1),
          loop_builder.getStringAttr(transfer_->name_));
    }
  } else if (element_type.isInteger(4)) {
    if (element_size == 4 && replicationFactor == 64) {
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type, data,
          nullptr, loop_builder.getI32ArrayAttr({0, 1, 2, 3}),
          loop_builder.getI32IntegerAttr(1),
          loop_builder.getStringAttr(transfer_->name_));
    } else if (element_size == 64 && replicationFactor == 8) {
      int index_arr[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
                         13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
                         26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38,
                         39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
                         52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63};
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type, data,
          nullptr, loop_builder.getI32ArrayAttr(index_arr),
          loop_builder.getI32IntegerAttr(1),
          loop_builder.getStringAttr(transfer_->name_));
    }
  } else if (element_type.isF32()) {
    if (element_size == 4 && replicationFactor == 8) {
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type, data,
          nullptr, loop_builder.getI32ArrayAttr({0, 1, 2, 3}),
          loop_builder.getI32IntegerAttr(8),
          loop_builder.getStringAttr(transfer_->name_));
    } else if (element_size == 32 && replicationFactor == 1) {
      int index_arr[] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10,
                         11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                         22, 23, 24, 25, 26, 27, 28, 29, 30, 31};
      shuffle_op = vectorchain::ShuffleOp::create(
          loop_builder, loop_builder.getUnknownLoc(), new_vec_type, data,
          nullptr, loop_builder.getI32ArrayAttr(index_arr),
          loop_builder.getI32IntegerAttr(1),
          loop_builder.getStringAttr(transfer_->name_));
    }
  }
  return shuffle_op;
}

// ---- 41/110  getStickSizes  --  dsc/dsc2.cpp:4066  (41L)
std::vector<std::pair<PrimaryDimTypes, int>> e041_getStickSizes(
    DsTypes dsType, bool stickSliceOnly /*= false*/,
    bool stickWithoutSlice /*= false*/, bool l0SliceOnly /*= false*/,
    int numL0Slices /*= -1*/) const {
  DT_CHECK_MSG(
      (stickSliceOnly + stickWithoutSlice + l0SliceOnly) < 2,
      "stickSliceOnly, l0SliceOnly and stickWithoutSlice can not be true at "
      "same time");
  DT_CHECK_MSG(!l0SliceOnly || numL0Slices > 0,
               "If l0SliceOnly requested, numL0Sclides must be provided");
  std::vector<std::pair<PrimaryDimTypes, int>> result;
  auto& pDsI = primaryDsInfo_.at(dsType);
  int elemInSlice = 1;
  for (auto& size : pDsI.stickSize_) elemInSlice *= size;
  const int numSlices = l0SliceOnly ? numL0Slices : 8;
  // elemInSlice should never be zero, and want to make sure it's at least 1
  // after the division
  DT_CHECK(elemInSlice > 0 && elemInSlice % numSlices == 0);
  elemInSlice /= numSlices;

  int elemSoFar = 1;
  for (int i = 0; i < pDsI.stickDimOrder_.size(); i++) {
    int size = pDsI.stickSize_.at(i);
    if (stickSliceOnly || l0SliceOnly) {
      // Can not fit more elements into the slice.
      if (elemSoFar >= elemInSlice) break;
      elemSoFar *= size;
      if (elemSoFar > elemInSlice) size /= (elemSoFar / elemInSlice);
    } else if (stickWithoutSlice) {
      // Calculation for stick without slice
      int newElemSoFar = elemSoFar * size;
      if (elemSoFar < elemInSlice && newElemSoFar > elemInSlice)
        size /= (elemInSlice / elemSoFar);
      elemSoFar = newElemSoFar;
      if (elemSoFar <= elemInSlice) continue;  // dim included in slice
    }
    auto dim = pDsI.stickDimOrder_[i];
    result.emplace_back(dim, size);
  }
  return result;
}


// ================================================================================================
// LEVEL 1
// ================================================================================================

// ---- 42/110  terminate  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:221  (3L)
void e042_terminate() {
  module_op_->emitError("Unable to translate DSC2.0 to the Dataflow IR");
}

// ---- 43/110  areFoldsNeeded  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:249  (40L)
bool e043_areFoldsNeeded(DesignSpaceConfig &dsc,
                                      SenComponents &comp) {
  std::vector<int> corelet_ids_used(1, 0);
  if (dsc.numCoreletsUsed_ == 2) corelet_ids_used.emplace_back(1);

  for (auto &[idx, cst_info] : dsc.constantInfo_) {
    for (int dim_idx = 0; dim_idx < cst_info.data_.getNumDims(); dim_idx++) {
      if (cst_info.data_.getFuncType(dim_idx) != BaseFuncType::Constant) {
        return true;
      }
    }
  }

  auto nodes = dsc.scheduleTree_.traverseTreeDFS(
      nullptr, {dsc2::ScheduleNode::NodeType::TRANSFER}, comp, -1, -1);
  for (auto &node : nodes) {
    const auto *transfer = dynamic_cast<const dsc2::TransferNode *>(node);
    if (transfer->src_.unit_ == comp) {
      bool check = areFoldedAddressesSameAcrossFoldsAndCoresForGivenCorelet(
          dsc.coreIdsUsed_, corelet_ids_used,
          transfer->srcLdsAndLoopOffsets_.startAddr_);
      if (!check) {
        return true;
      }
    } else {
      for (int dst_idx = 0; dst_idx < transfer->dstVias_.size(); ++dst_idx) {
        auto &dst = transfer->dstVias_[dst_idx];
        if (dst.loc_.unit_ == comp) {
          bool check = areFoldedAddressesSameAcrossFoldsAndCoresForGivenCorelet(
              dsc.coreIdsUsed_, corelet_ids_used,
              transfer->dstLdsAndLoopOffsets_[dst_idx].startAddr_);
          if (!check) {
            return true;
          }
        }
      }
    }
  }
  return false;
}

// ---- 44/110  createGetUnitOp  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:64  (24L)
dataflow::GetUnitOp e044_createGetUnitOp(OpBuilder &builder,
                                                      SenComponents comp,
                                                      int core_id,
                                                      int corelet_id) {
  if (component_to_handler_.count(comp) == 0) {
    auto unit_name = EnumsConversion::senComponentsToString.at(comp);

    SmallVector<Type> get_unit_type;
    constructAVectorOfIndexType(num_folds_, builder, get_unit_type);

    auto unit_op = dataflow::GetUnitOp::create(
        builder, builder.getUnknownLoc(), get_unit_type,
        builder.getStringAttr(unit_name), builder.getStringAttr(unit_name));
    unit_op->setAttr("core", builder.getI32IntegerAttr(core_id));
    unit_op->setAttr("num_folds", builder.getI32IntegerAttr(num_folds_));
    if (corelet_id != -1)
      unit_op->setAttr("corelet", builder.getI32IntegerAttr(corelet_id));

    this->unit_to_value_map_[core_id][corelet_id][comp] = unit_op.getResult(0);
    return unit_op;
  } else {
    return component_to_handler_[comp].getDefiningOp<dataflow::GetUnitOp>();
  }
}

// ---- 45/110  getReductionMapForMACOperation  --  dsc-based-utils/DSC2ToDataflowIR/DataflowIRConstructionUtils.hpp:61  (40L)
static LogicalResult e045_getReductionMapForMACOperation(OpBuilder &builder,
                                                    ComputeOpType type,
                                                    AffineMap &reduction_map) {
  auto *context = builder.getContext();
  auto i = getAffineDimExpr(0, builder.getContext());
  if (is_any_of(type, ComputeOpType::FMA16, ComputeOpType::FNMS,
                ComputeOpType::FMA32)) {
    // (i) : e.g. 0 -> 0
    reduction_map = builder.getDimIdentityMap();
  } else if (type == ComputeOpType::FMA8) {
    // (floordiv 2) : e.g. 0, 1 -> 0
    auto affineConst2 = builder.getAffineConstantExpr(2);
    auto affineFloorDivExpr =
        getAffineBinaryOpExpr(mlir::AffineExprKind::FloorDiv, i, affineConst2);
    reduction_map = AffineMap::get(1, 0, affineFloorDivExpr, context);
  } else if (type == ComputeOpType::IMA8) {
    // ((i mod 128) floordiv 2) : e.g. 0, 1, 128, 129 -> 0
    auto affineConst2 = builder.getAffineConstantExpr(2);
    auto affineConst128 = builder.getAffineConstantExpr(128);
    auto affineModExpr =
        getAffineBinaryOpExpr(mlir::AffineExprKind::Mod, i, affineConst128);
    auto affineFloorDivExpr = getAffineBinaryOpExpr(
        mlir::AffineExprKind::FloorDiv, affineModExpr, affineConst2);
    reduction_map = AffineMap::get(1, 0, affineFloorDivExpr, context);
  } else if (type == ComputeOpType::IMA4) {
    // ((i mod 128) floordiv 4) : e.g. 0, 1, 2, 3, 128, 129, 130, 131 -> 0
    auto affineConst4 = builder.getAffineConstantExpr(4);
    auto affineConst128 = builder.getAffineConstantExpr(128);
    auto affineModExpr =
        getAffineBinaryOpExpr(mlir::AffineExprKind::Mod, i, affineConst128);
    auto affineFloorDivExpr = getAffineBinaryOpExpr(
        mlir::AffineExprKind::FloorDiv, affineModExpr, affineConst4);
    reduction_map = AffineMap::get(1, 0, affineFloorDivExpr, context);
  } else {
    DT_ERROR("Unknown MAC operation encountered");
    return LogicalResult::failure();
  }

  return LogicalResult::success();
}

// ---- 46/110  getStaticContinuousMaskValue  --  dsc-based-utils/DSC2ToDataflowIR/DataflowIRConstructionUtils.hpp:105  (24L)
static vectorchain::CreateAffineMaskOp e046_getStaticContinuousMaskValue(
    mlir::OpBuilder builder, mlir::Location loc, int vector_dim,
    int mask_value = 255) {
  std::map<int, int> static_mask_value = {{255, 0}, {127, 1}, {63, 2}, {31, 3},
                                          {15, 4},  {7, 5},   {3, 6},  {1, 7}};
  if (static_mask_value.count(mask_value) < 1) {
    DT_ERROR("Mask value is not supported.");
  }
  SmallVector<AffineExpr, 2> exprs;
  SmallVector<bool, 2> eq_flags(2);
  std::fill(eq_flags.begin(), eq_flags.end(), false);
  auto id = getAffineDimExpr(0, builder.getContext());
  exprs.push_back(id - vector_dim +
                  static_mask_value[mask_value] * vector_dim / 8);
  exprs.push_back(-id + vector_dim - 1);
  auto mask_set = IntegerSet::get(1, 0, exprs, eq_flags);
  auto zero = mlir::arith::ConstantOp::create(
      builder, loc, builder.getIndexType(),
      builder.getZeroAttr(builder.getIndexType()));
  auto mask_result_type = VectorType::get(vector_dim, builder.getI1Type());
  auto mask_op = vectorchain::CreateAffineMaskOp::create(
      builder, loc, mask_result_type, nullptr, IntegerSetAttr::get(mask_set));
  return mask_op;
}

// ---- 47/110  getStaticContinuousMaskValue  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:33  (23L)
vectorchain::CreateAffineMaskOp e047_getStaticContinuousMaskValue(
    mlir::OpBuilder builder, mlir::Location loc, mlir::Type result_type,
    llvm::SmallVectorImpl<Value> &inputs, int mask_value) {
  std::map<int, int> static_mask_value = {{255, 0}, {127, 1}, {63, 2}, {31, 3},
                                          {15, 4},  {7, 5},   {3, 6},  {1, 7}};
  if (static_mask_value.count(mask_value) < 1) {
    inputs[0].getDefiningOp()->emitError("Mask value is not supported.");
    return nullptr;
  }
  SmallVector<AffineExpr, 2> exprs;
  SmallVector<bool, 2> eq_flags(2);
  std::fill(eq_flags.begin(), eq_flags.end(), false);
  auto vector_dim = dataflow::utils::getDimSize(result_type, 0);
  auto id = getAffineDimExpr(0, inputs[0].getContext());
  exprs.push_back(id - vector_dim +
                  static_mask_value[mask_value] * vector_dim / 8);
  exprs.push_back(-id + vector_dim - 1);
  auto mask_set = IntegerSet::get(1, 0, exprs, eq_flags);
  auto mask_result_type = VectorType::get(vector_dim, builder.getI1Type());
  auto mask_op = vectorchain::CreateAffineMaskOp::create(
      builder, loc, mask_result_type, nullptr, IntegerSetAttr::get(mask_set));
  return mask_op;
}

// ---- 48/110  constructDynamicMasking  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:60  (43L)
vectorchain::CreateAffineMaskOp e048_constructDynamicMasking(
    mlir::OpBuilder builder, mlir::Location loc, mlir::Type result_type,
    llvm::SmallVectorImpl<Value> &inputs,
    const std::unordered_map<const dsc2::LoopNode *,
                             std::unordered_map<PrimaryDimTypes, int>>
        &computeMaskLoopOffsets_) {
  DT_CHECK_MSG(
      computeMaskLoopOffsets_.size() == 1,
      "Translator currently supports translating only 1-D dynamic masking");

  auto entry = computeMaskLoopOffsets_.begin();
  const dsc2::LoopNode *loop_node = entry->first;
  DT_CHECK_MSG(
      entry->second.size() == 1,
      "Translator currently supports translating only 1-D dynamic masking");
  auto record = entry->second.begin();
  PrimaryDimTypes mask_dim = record->first;
  int mask_offset = record->second;

  Operation *loop = getMLIRLoopFromLoopNode(loop_node, mask_dim);
  mlir::Value iv;
  if (auto affine_for = llvm::dyn_cast<affine::AffineForOp>(loop)) {
    iv = affine_for.getInductionVar();
  } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop)) {
    iv = scf_for.getInductionVar();
  }

  //"loop_ds2_ds3_mb -> s0" : {"mb" : 1}
  // #set = affine_set<(d0)[s0] : (d0 + s0 * 8 - 64 >= 0, -d0 + 63 >= 0)>
  SmallVector<AffineExpr, 2> exprs;
  SmallVector<bool, 2> eq_flags(2);
  std::fill(eq_flags.begin(), eq_flags.end(), false);
  auto vector_dim = dataflow::utils::getDimSize(result_type, 0);
  auto id = getAffineDimExpr(0, inputs[0].getContext());
  auto symbol = getAffineSymbolExpr(0, inputs[0].getContext());
  exprs.push_back(id - vector_dim + symbol * (vector_dim / 8) * mask_offset);
  exprs.push_back(-id + vector_dim - 1);
  auto mask_set = IntegerSet::get(1, 1, exprs, eq_flags);
  auto mask_result_type = VectorType::get(vector_dim, builder.getI1Type());
  auto mask_op = vectorchain::CreateAffineMaskOp::create(
      builder, loc, mask_result_type, iv, IntegerSetAttr::get(mask_set));
  return mask_op;
}

// ---- 49/110  getTypeBasedOnComputeType  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:108  (12L)
LogicalResult e049_getTypeBasedOnComputeType(
    OpBuilder &builder, DataFormats original_format, Type &result_type,
    Type &element_type) {
  bool is_result_integer;
  if (failed(constructTypeFromFormat(
          builder, original_format,
          LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR, comp_, -1,
          result_type, element_type, is_result_integer))) {
    return LogicalResult::failure();
  }
  return LogicalResult::success();
}

// ---- 50/110  getReductionMapForMACOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:126  (70L)
LogicalResult e050_getReductionMapForMACOperation(
    OpBuilder &builder, ComputeOpType type, AffineMap &reduction_map) {
  auto *context = builder.getContext();
  auto arch = dscGlobal().sysDef.coreArch;
  auto i = getAffineDimExpr(0, builder.getContext());
  if (is_any_of(type, ComputeOpType::FMA16, ComputeOpType::FNMS)) {
    int factor = arch == SEN1P5_ISA ? 4 : 1;
    // (floordiv factor)
    auto affineConst = builder.getAffineConstantExpr(factor);
    auto affineFloorDivExpr =
        getAffineBinaryOpExpr(mlir::AffineExprKind::FloorDiv, i, affineConst);
    reduction_map = AffineMap::get(1, 0, affineFloorDivExpr, context);
  } else if (type == ComputeOpType::FMA32) {
    auto affineConst = builder.getAffineConstantExpr(1);
    auto affineFloorDivExpr =
        getAffineBinaryOpExpr(mlir::AffineExprKind::FloorDiv, i, affineConst);
    reduction_map = AffineMap::get(1, 0, affineFloorDivExpr, context);
  } else if (type == ComputeOpType::FMA8) {
    int factor = arch == SEN1P5_ISA ? 16 : 2;
    // (floordiv 2) : e.g. 0, 1 -> 0
    auto affineConst = builder.getAffineConstantExpr(factor);
    auto affineFloorDivExpr =
        getAffineBinaryOpExpr(mlir::AffineExprKind::FloorDiv, i, affineConst);
    reduction_map = AffineMap::get(1, 0, affineFloorDivExpr, context);
  } else if (type == ComputeOpType::FMA4) {
    DT_CHECK(arch == SEN1P5_ISA);
    // (floordiv 32) : e.g. 0, 2 -> 0
    auto affineConst = builder.getAffineConstantExpr(32);
    auto affineFloorDivExpr =
        getAffineBinaryOpExpr(mlir::AffineExprKind::FloorDiv, i, affineConst);
    reduction_map = AffineMap::get(1, 0, affineFloorDivExpr, context);
  } else if (type == ComputeOpType::IMA8) {
    if (arch == SEN1P5_ISA) {
      auto affineConst = builder.getAffineConstantExpr(16);
      auto affineFloorDivExpr =
          getAffineBinaryOpExpr(mlir::AffineExprKind::FloorDiv, i, affineConst);
      reduction_map = AffineMap::get(1, 0, affineFloorDivExpr, context);
    } else {
      // ((i mod 128) floordiv 2) : e.g. 0, 1, 128, 129 -> 0
      auto affineConst2 = builder.getAffineConstantExpr(2);
      auto affineConst128 = builder.getAffineConstantExpr(128);
      auto affineModExpr =
          getAffineBinaryOpExpr(mlir::AffineExprKind::Mod, i, affineConst128);
      auto affineFloorDivExpr = getAffineBinaryOpExpr(
          mlir::AffineExprKind::FloorDiv, affineModExpr, affineConst2);
      reduction_map = AffineMap::get(1, 0, affineFloorDivExpr, context);
    }
  } else if (type == ComputeOpType::IMA4) {
    if (arch == SEN1P5_ISA) {
      auto affineConst = builder.getAffineConstantExpr(32);
      auto affineFloorDivExpr =
          getAffineBinaryOpExpr(mlir::AffineExprKind::FloorDiv, i, affineConst);
      reduction_map = AffineMap::get(1, 0, affineFloorDivExpr, context);
    } else {
      // ((i mod 128) floordiv 4) : e.g. 0, 1, 2, 3, 128, 129, 130, 131 -> 0
      auto affineConst4 = builder.getAffineConstantExpr(4);
      auto affineConst128 = builder.getAffineConstantExpr(128);
      auto affineModExpr =
          getAffineBinaryOpExpr(mlir::AffineExprKind::Mod, i, affineConst128);
      auto affineFloorDivExpr = getAffineBinaryOpExpr(
          mlir::AffineExprKind::FloorDiv, affineModExpr, affineConst4);
      reduction_map = AffineMap::get(1, 0, affineFloorDivExpr, context);
    }
  } else {
    emitError("Unknown MAC operation encountered");
    return LogicalResult::failure();
  }

  return LogicalResult::success();
}

// ---- 51/110  getSelectionMapForMACOperandFromL0  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:200  (74L)
LogicalResult e051_getSelectionMapForMACOperandFromL0(
    OpBuilder &builder, ComputeOpType type, DataFormats format,
    Type original_data_type, AffineMap &selection_map, Type &splat_data_type) {
  auto *context = builder.getContext();
  auto arch = dscGlobal().sysDef.coreArch;
  auto i = getAffineDimExpr(0, builder.getContext());
  if (type == ComputeOpType::FMA16) {
    int factor = arch == SEN1P5_ISA ? 4 : 1;
    //(i) -> (i mod 4)
    auto affineConst = builder.getAffineConstantExpr(factor);
    auto affineModExpr =
        getAffineBinaryOpExpr(mlir::AffineExprKind::Mod, i, affineConst);
    selection_map = AffineMap::get(1, 0, affineModExpr, context);
    auto element_type = (format == DataFormats::BFLOAT16)
                            ? builder.getBF16Type()
                            : builder.getF16Type();
    splat_data_type = VectorType::get(64 * factor, element_type);
  } else if (type == ComputeOpType::FMA8) {
    //(i) -> (i mod 2)
    int factor = arch == SEN1P5_ISA ? 16 : 2;
    auto affineConst2 = builder.getAffineConstantExpr(factor);
    auto affineModExpr =
        getAffineBinaryOpExpr(mlir::AffineExprKind::Mod, i, affineConst2);
    selection_map = AffineMap::get(1, 0, affineModExpr, context);
    if (mlir::isa<VectorType>(original_data_type)) {
      auto element_type = mlir::Float8E4M3FNType::get(context);
      splat_data_type = VectorType::get(64 * factor, element_type);
    } else {
      auto element_type =
          dataflow::CustomMXFloatType::get(builder.getContext(), 8);
      splat_data_type =
          dataflow::utils::constructVectorType(element_type, 64 * factor);
    }
  } else if (type == ComputeOpType::FMA4) {
    DT_CHECK(arch == SEN1P5_ISA);
    int factor = 32;
    auto affineConst = builder.getAffineConstantExpr(factor);
    auto affineModExpr =
        getAffineBinaryOpExpr(mlir::AffineExprKind::Mod, i, affineConst);
    selection_map = AffineMap::get(1, 0, affineModExpr, context);
    if (mlir::isa<VectorType>(original_data_type)) {
      auto element_type = mlir::Float4E2M1FNType::get(context);
      splat_data_type = VectorType::get(64 * factor, element_type);
    } else {
      auto element_type =
          dataflow::CustomMXFloatType::get(builder.getContext(), 4);
      splat_data_type =
          dataflow::utils::constructVectorType(element_type, 64 * factor);
    }
  } else if (type == ComputeOpType::IMA8) {
    //(i) -> (i mod 16)
    int factor = arch == SEN1P5_ISA ? 16 : 4;
    auto affineConst16 = builder.getAffineConstantExpr(factor);
    auto affineModExpr =
        getAffineBinaryOpExpr(mlir::AffineExprKind::Mod, i, affineConst16);
    selection_map = AffineMap::get(1, 0, affineModExpr, context);
    auto element_type = builder.getIntegerType(8);
    splat_data_type = VectorType::get(64 * factor, element_type);
  } else if (type == ComputeOpType::IMA4) {
    //(i) -> (i mod 32)
    int factor = arch == SEN1P5_ISA ? 32 : 8;
    auto affineConst32 = builder.getAffineConstantExpr(factor);
    auto affineModExpr =
        getAffineBinaryOpExpr(mlir::AffineExprKind::Mod, i, affineConst32);
    selection_map = AffineMap::get(1, 0, affineModExpr, context);
    auto element_type = builder.getIntegerType(4);
    splat_data_type = VectorType::get(64 * factor, element_type);
  } else {
    emitError("Unknown MAC operation encountered");
    return LogicalResult::failure();
  }

  return LogicalResult::success();
}

// ---- 52/110  constructPrecisionConversionOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:375  (59L)
LogicalResult e052_constructPrecisionConversionOperation(
    OpBuilder &builder, Value &src, DataFormats result_format, Value &result,
    SenComponents comp) {
  Type element_type;
  VectorType result_type;
  bool is_result_integer = false;
  bool is_src_integer = false;
  int num_elements = 0;
  if (mlir::isa<dataflow::CustomVectorType>(src.getType())) {
    auto src_type = mlir::cast<dataflow::CustomVectorType>(src.getType());

    // MX format is a logical format for simplifying PT and L0LU DSC2 and DFIR.
    // no precision conversion on those units.
    if (mlir::isa<dataflow::CustomMXFloatType>(src_type.getElementType()))
      return LogicalResult::success();

    is_src_integer = mlir::cast<dataflow::CustomVectorType>(src.getType())
                         .getElementType()
                         .isIntOrIndex();
    num_elements =
        mlir::cast<dataflow::CustomVectorType>(src.getType()).getNumElements();
  } else {
    DT_CHECK(mlir::isa<VectorType>(src.getType()));
    is_src_integer =
        mlir::cast<VectorType>(src.getType()).getElementType().isIntOrIndex();
    num_elements = mlir::cast<VectorType>(src.getType()).getNumElements();
  }

  if (failed(constructTypeFromFormat(
          builder, result_format,
          LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR, comp, num_elements,
          result_type, element_type, is_result_integer))) {
    return LogicalResult::failure();
  }

  if (src.getType() == result_type) {
    result = src;
    return LogicalResult::success();
  } else {
    if (is_src_integer && !is_result_integer) {
      auto op = mlir::arith::SIToFPOp::create(builder, builder.getUnknownLoc(),
                                              result_type, src);
      result = op.getResult();
      return LogicalResult::success();
    } else if (!is_src_integer && is_result_integer) {
      auto op = mlir::arith::FPToSIOp::create(builder, builder.getUnknownLoc(),
                                              result_type, src);
      result = op.getResult();
      return LogicalResult::success();
    } else if (!is_src_integer && !is_result_integer) {
      auto op = mlir::vectorchain::CastOp::create(
          builder, builder.getUnknownLoc(), result_type, src);
      result = op.getResult();
      return LogicalResult::success();
    }

    return LogicalResult::failure();
  }
}

// ---- 53/110  constructOpaqueOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1534  (25L)
LogicalResult e053_constructOpaqueOperation(
    OpBuilder &builder, const dsc2::ComputeNode &compute_op) {
  StringAttr func_name_attr = builder.getStringAttr(
      EnumsConversion::computeTypeToString.at(compute_op.type_));
  auto loc = builder.getUnknownLoc();

  std::vector<NamedAttribute> read_only_reg_vec;
  auto read_only_reg_dic_attr =
      mapToDicAttr(builder, compute_op.instrAttribute_.read_only_reg_map_);

  std::vector<NamedAttribute> read_write_reg_vec;
  auto read_write_reg_dic_attr =
      mapToDicAttr(builder, compute_op.instrAttribute_.read_write_reg_map_);

  std::vector<NamedAttribute> param_vec;
  auto param_dic_attr =
      mapToDicAttr(builder, compute_op.instrAttribute_.param_map_);

  auto opaque_op = dataflow::OpaqueOp::create(
      builder, loc, builder.getStringAttr(compute_op.name_), func_name_attr,
      read_write_reg_dic_attr, read_only_reg_dic_attr, param_dic_attr);

  if (!opaque_op) return LogicalResult::failure();
  return LogicalResult::success();
}

// ---- 54/110  constructConditionalOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:66  (211L)
LogicalResult e054_constructConditionalOperation(
    std::map<const dsc2::LoopNode *, std::vector<Operation *>>
        &dsc_loops_to_mlir_loops_map,
    bool has_else_branch, OpBuilder &builder, OpBuilder &else_builder,
    OpBuilder &endif_builder, const std::string name,
    const dsc2::LoopCondComposite &cond, mlir::scf::IfOp &if_op) {
  int counter = 0;
  std::vector<mlir::scf::IfOp> cmp_list;
  mlir::Value val_true = mlir::arith::ConstantIntOp::create(
      builder, builder.getUnknownLoc(), 1, 1);
  mlir::Value val_false = mlir::arith::ConstantIntOp::create(
      builder, builder.getUnknownLoc(), 0, 1);
  auto then_builder = builder;
  bool first_and = true;
  // Going though conditions
  if (cond.twoLevelOrOfAnds_.size() > 1 || cond.negated_ || has_else_branch) {
    // Either with AND and OR statements
    // or only AND with negation
    // or has else branch.
    for (const auto &cond_and_sets : cond.twoLevelOrOfAnds_) {
      mlir::scf::IfOp scf_ifop_tmp;
      endif_builder = then_builder = else_builder = builder;
      counter = 0;
      for (auto cond_and_set : cond_and_sets) {
        Value loop_val;
        Value loop_itr;
        auto *loop =
            getMLIRLoopFromSNLoopNode(cond_and_set.loopComp_, cond_and_set.dim_,
                                      dsc_loops_to_mlir_loops_map);
        if (auto affine_for = llvm::dyn_cast<mlir::affine::AffineForOp>(loop)) {
          if (affine_for
                  .hasConstantBounds()) {  // FIXME: Is it always constant?
            loop_itr = affine_for.getInductionVar();
            if (cond_and_set.condValType_ ==
                dsc2::LoopCond::CondValType::LAST) {
              loop_val = mlir::arith::ConstantIndexOp::create(
                  then_builder, then_builder.getUnknownLoc(),
                  affine_for.getConstantUpperBound() - 1);
            } else if (cond_and_set.condValType_ ==
                       dsc2::LoopCond::CondValType::FIRST) {
              loop_val = mlir::arith::ConstantIndexOp::create(
                  then_builder, then_builder.getUnknownLoc(),
                  affine_for.getConstantLowerBound());
            } else {
              DT_CHECK(cond_and_set.condValType_ == dsc2::LoopCond::INT);
              loop_val = mlir::arith::ConstantIndexOp::create(
                  then_builder, then_builder.getUnknownLoc(),
                  cond_and_set.condValInt_);
            }
          } else {
            emitError("Unsupported dynamic affine loop!");
            return LogicalResult::failure();
          }
        } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop)) {
          loop_itr = scf_for.getInductionVar();
          if (cond_and_set.condValType_ == dsc2::LoopCond::CondValType::LAST) {
            auto val_one = mlir::arith::ConstantIndexOp::create(
                then_builder, then_builder.getUnknownLoc(), 1);
            loop_val = mlir::arith::SubIOp::create(
                then_builder, then_builder.getUnknownLoc(),
                scf_for.getUpperBound(), val_one);
          } else if (cond_and_set.condValType_ ==
                     dsc2::LoopCond::CondValType::FIRST) {
            loop_val = scf_for.getLowerBound();
          } else {
            DT_CHECK(cond_and_set.condValType_ == dsc2::LoopCond::INT);
            loop_val = mlir::arith::ConstantIndexOp::create(
                then_builder, then_builder.getUnknownLoc(),
                cond_and_set.condValInt_);
          }
        } else {
          return LogicalResult::failure();
        }

        mlir::arith::CmpIPredicate pred;
        if (getCmpIPredicate_dup(pred, cond_and_set.condOp_).failed()) {
          emitError("Unsupported CondOp!");
          return LogicalResult::failure();
        }
        auto cmp = mlir::arith::CmpIOp::create(then_builder,
                                               then_builder.getUnknownLoc(),
                                               pred, loop_itr, loop_val);
        scf_ifop_tmp =
            mlir::scf::IfOp::create(then_builder, then_builder.getUnknownLoc(),
                                    then_builder.getI1Type(), cmp, true);
        dataflow::setDbgName(scf_ifop_tmp, name);

        if (counter != 0) {
          mlir::scf::YieldOp::create(then_builder, then_builder.getUnknownLoc(),
                                     scf_ifop_tmp.getResults()[0]);
        } else {
          cmp_list.push_back(scf_ifop_tmp);
        }
        then_builder = scf_ifop_tmp.getThenBodyBuilder();
        if (counter + 1 == cond_and_sets.size()) {
          mlir::scf::YieldOp::create(then_builder, then_builder.getUnknownLoc(),
                                     val_true);
        }
        else_builder = scf_ifop_tmp.getElseBodyBuilder();
        if (first_and) {
          mlir::scf::YieldOp::create(else_builder, else_builder.getUnknownLoc(),
                                     val_false);
        } else {
          mlir::scf::YieldOp::create(
              else_builder, else_builder.getUnknownLoc(),
              cmp_list[cmp_list.size() - 2].getResults()[0]);
        }
        counter++;
      }
      first_and = false;
      builder = endif_builder;
    }

    mlir::scf::IfOp scf_ifop;
    mlir::scf::IfOp scf_ifop_cmp_top = cmp_list[cmp_list.size() - 1];
    // Negation statments
    if (cond.negated_) {
      auto cmp_negate = mlir::arith::CmpIOp::create(
          builder, builder.getUnknownLoc(), mlir::arith::CmpIPredicate::eq,
          scf_ifop_cmp_top.getResults()[0], val_false);
      scf_ifop = mlir::scf::IfOp::create(builder, builder.getUnknownLoc(),
                                         cmp_negate, has_else_branch);
    } else {
      scf_ifop = mlir::scf::IfOp::create(builder, builder.getUnknownLoc(),
                                         scf_ifop_cmp_top.getResults()[0],
                                         has_else_branch);
    }
    dataflow::setDbgName(scf_ifop, name);
    endif_builder = builder;
    builder = scf_ifop.getThenBodyBuilder();
    if (has_else_branch) {
      else_builder = scf_ifop.getElseBodyBuilder();
    }

    if_op = scf_ifop;
  } else {
    // Only AND statement without negation and has only then region.
    auto cond_and = cond.twoLevelOrOfAnds_[0];
    mlir::scf::IfOp scf_ifop_tmp;
    counter = 0;
    for (auto cond_eq : cond_and) {
      Value loop_val;
      Value loop_itr;
      auto *loop = getMLIRLoopFromSNLoopNode(cond_eq.loopComp_, cond_eq.dim_,
                                             dsc_loops_to_mlir_loops_map);
      if (auto affine_for = llvm::dyn_cast<mlir::affine::AffineForOp>(loop)) {
        if (affine_for.hasConstantBounds()) {  // FIXME: Is it always constant?
          loop_itr = affine_for.getInductionVar();
          if (cond_eq.condValType_ == dsc2::LoopCond::CondValType::LAST) {
            loop_val = mlir::arith::ConstantIndexOp::create(
                builder, builder.getUnknownLoc(),
                affine_for.getConstantUpperBound() - 1);
          } else if (cond_eq.condValType_ ==
                     dsc2::LoopCond::CondValType::FIRST) {
            loop_val = mlir::arith::ConstantIndexOp::create(
                builder, builder.getUnknownLoc(),
                affine_for.getConstantLowerBound());
          } else {
            DT_CHECK(cond_eq.condValType_ == dsc2::LoopCond::INT);
            loop_val = mlir::arith::ConstantIndexOp::create(
                builder, builder.getUnknownLoc(), cond_eq.condValInt_);
          }
        } else {
          emitError("Unsupported dynamic affine loop!");
          return LogicalResult::failure();
        }
      } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop)) {
        loop_itr = scf_for.getInductionVar();
        if (cond_eq.condValType_ == dsc2::LoopCond::CondValType::LAST) {
          auto val_one = mlir::arith::ConstantIndexOp::create(
              builder, builder.getUnknownLoc(), 1);
          loop_val =
              mlir::arith::SubIOp::create(builder, builder.getUnknownLoc(),
                                          scf_for.getUpperBound(), val_one);
        } else if (cond_eq.condValType_ == dsc2::LoopCond::CondValType::FIRST) {
          loop_val = scf_for.getLowerBound();
        } else {
          DT_CHECK(cond_eq.condValType_ == dsc2::LoopCond::INT);
          loop_val = mlir::arith::ConstantIndexOp::create(
              builder, builder.getUnknownLoc(), cond_eq.condValInt_);
        }
      } else {
        return LogicalResult::failure();
      }
      mlir::arith::CmpIPredicate pred;
      if (getCmpIPredicate_dup(pred, cond_eq.condOp_).failed()) {
        emitError("Unsupported CondOp!");
        return LogicalResult::failure();
      }

      auto cmp = mlir::arith::CmpIOp::create(builder, builder.getUnknownLoc(),
                                             pred, loop_itr, loop_val);
      scf_ifop_tmp =
          mlir::scf::IfOp::create(builder, builder.getUnknownLoc(), cmp, false);
      dataflow::setDbgName(scf_ifop_tmp, name);

      // set end_if builder.. no need for else_if_builder because there
      // is no else branch.
      // builder keeps getting updating and becomes the then_region_builder.
      if (counter == 0) {
        endif_builder.setInsertionPointAfter(scf_ifop_tmp);
      }

      if_op = scf_ifop_tmp;
      builder = scf_ifop_tmp.getThenBodyBuilder();
      counter++;
    }
  }

  return LogicalResult::success();
}

// ---- 55/110  constructConditionalsForSAMV  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:284  (91L)
LogicalResult e055_constructConditionalsForSAMV(
    OpBuilder builder,
    std::map<const dsc2::LoopNode *, std::vector<Operation *>>
        *dsc_loops_to_mlir_loops_map,
    std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> &loops,
    mlir::scf::IfOp &if_op) {
  // collect loops pertaining to OUT dimension.
  std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> out_loops;
  for (auto &loop : loops) {
    if (loop.dim_ == OUT) {
      if (!(loop.loop_->numId_ == 0 && loop.loop_->denId_ == 1)) {
        out_loops.push_back(loop);
      }
    }
  }

  if (out_loops.empty()) {
    if_op = nullptr;
    return LogicalResult::success();
  }

  // create AND condition covering for all the last iterations
  int counter = 0;
  mlir::scf::IfOp scf_ifop_tmp;
  mlir::Value val_true = mlir::arith::ConstantIntOp::create(
      builder, builder.getUnknownLoc(), 1, 1);
  mlir::Value val_false = mlir::arith::ConstantIntOp::create(
      builder, builder.getUnknownLoc(), 0, 1);

  auto then_builder = builder;
  auto else_builder = builder;
  for (auto &loop : out_loops) {
    Value loop_val;
    Value loop_itr;
    auto *mlir_loop = getMLIRLoopFromSNLoopNode(loop.loop_, OUT,
                                                *dsc_loops_to_mlir_loops_map);
    if (auto affine_for =
            llvm::dyn_cast<mlir::affine::AffineForOp>(mlir_loop)) {
      if (affine_for.hasConstantBounds()) {  // FIXME: Is it always constant?
        loop_itr = affine_for.getInductionVar();
        loop_val = mlir::arith::ConstantIndexOp::create(
            then_builder, then_builder.getUnknownLoc(),
            affine_for.getConstantUpperBound() - 1);
      } else {
        llvm_unreachable("Unsupported dynamic affine loop!");
      }
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(mlir_loop)) {
      loop_itr = scf_for.getInductionVar();
      auto val_one = mlir::arith::ConstantIndexOp::create(
          then_builder, then_builder.getUnknownLoc(), 1);
      loop_val = mlir::arith::SubIOp::create(then_builder,
                                             then_builder.getUnknownLoc(),
                                             scf_for.getUpperBound(), val_one);
    } else {
      return LogicalResult::failure();
    }

    // create a condition.
    auto cmp = mlir::arith::CmpIOp::create(
        then_builder, then_builder.getUnknownLoc(),
        mlir::arith::CmpIPredicate::eq, loop_itr, loop_val);

    scf_ifop_tmp = mlir::scf::IfOp::create(then_builder, cmp.getLoc(),
                                           then_builder.getI1Type(), cmp, true);

    if (counter != 0) {
      // for inner if condition, return the output of the if condition inside.
      mlir::scf::YieldOp::create(then_builder, then_builder.getUnknownLoc(),
                                 scf_ifop_tmp.getResults()[0]);
    } else {
      // outermost if condition.
      if_op = scf_ifop_tmp;
    }

    // Returning true in the inner-most condition.
    then_builder = scf_ifop_tmp.getThenBodyBuilder();
    if (counter + 1 == out_loops.size()) {
      mlir::scf::YieldOp::create(then_builder, then_builder.getUnknownLoc(),
                                 val_true);
    }

    // Always returning false in the else branch.
    else_builder = scf_ifop_tmp.getElseBodyBuilder();
    mlir::scf::YieldOp::create(else_builder, else_builder.getUnknownLoc(),
                               val_false);

    counter++;
  }

  return LogicalResult::success();
}

// ---- 56/110  getBufferingOrStreamingMode  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:408  (54L)
LogicalResult e056_getBufferingOrStreamingMode(
    const dsc2::TransferNode *transfer, int &mode,
    const FoldManager<int64_t> *&start_address_map, double &factor) {
  mode = -1;  // Neither buffering or streaming
  const auto &src_ld_idx = transfer->srcLdsAndLoopOffsets_.myLdsIdx_;
  auto src_prec = dsc_->labeledDs_[src_ld_idx].dataFormat_;
  auto src_category = dsc_->labeledDs_[src_ld_idx].scaledLdsCategory_;

  OpBuilder builder(*this->unit_op_);
  auto element_type =
      getMLIRTypeFromDSCDataFormat(src_prec, src_category, builder);
  if (transfer->src_.unit_ == comp_) {
    if (transfer->srcLdsAndLoopOffsets_.bufferSwitchPosition_ != nullptr) {
      auto num_buffers =
          dsc_->labeledDs_.at(transfer->srcLdsAndLoopOffsets_.myLdsIdx_)
              .memOrg_.at(transfer->src_.storage_)
              .allocateNode_->numBuffers_;
      factor = getAddressGranularityMultiplyFactor(
          comp_, transfer->src_.storage_, element_type);
      start_address_map = &transfer->srcLdsAndLoopOffsets_.startAddr_;
      if (num_buffers == -1) {
        mode = 2;  // Streaming
        return LogicalResult::success();
      } else {
        mode = 1;  // Buffering
        return LogicalResult::success();
      }
    }
  }

  for (int i = 0; i < transfer->dstVias_.size(); i++) {
    const auto &via = transfer->dstVias_[i];
    if (via.loc_.unit_ == comp_) {
      const auto &lds = transfer->dstLdsAndLoopOffsets_[i];
      if (lds.bufferSwitchPosition_ != nullptr) {
        auto num_buffers = dsc_->labeledDs_.at(lds.myLdsIdx_)
                               .memOrg_.at(via.loc_.storage_)
                               .allocateNode_->numBuffers_;
        factor = getAddressGranularityMultiplyFactor(comp_, via.loc_.storage_,
                                                     element_type);
        start_address_map = &transfer->dstLdsAndLoopOffsets_[i].startAddr_;
        if (num_buffers == -1) {
          mode = 2;  // streaming
          return LogicalResult::success();
        } else {
          mode = 1;  // Buffering
          return LogicalResult::success();
        }
      }
    }
  }

  return LogicalResult::success();
}

// ---- 57/110  propagateBufferSwitchLoopsToRoot  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:575  (15L)
LogicalResult e057_propagateBufferSwitchLoopsToRoot() {
  auto children =
      dsc_->scheduleTree_.getHead()->getNextView(comp_, corelet_id_, core_id_);
  for (auto &child : children) {
    if (const auto *loop = dynamic_cast<const dsc2::LoopNode *>(child)) {
      std::vector<const dsc2::TransferNode *> child_contrib;
      if (failed(propagateBufferSwitchLoopsToRootRecursively(loop,
                                                             child_contrib))) {
        return LogicalResult::failure();
      }
    }
  }

  return LogicalResult::success();
}

// ---- 58/110  constructLoopForADim  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:754  (101L)
Operation *e058_constructLoopForADim(
    OpBuilder &builder, const dsc2::LoopNode *node,
    SmallVectorImpl<Value> &iter_args, PrimaryDimTypes dim,
    const PaddingFormType &padInfo) {
  Operation *loop = nullptr;
  auto loc = builder.getUnknownLoc();
  auto num_ss_staging = dsc_->dataStageParam_.at(node->numId_).ss_;
  auto num_el_staging = dsc_->dataStageParam_.at(node->numId_).el_;
  auto den_ss_staging = dsc_->dataStageParam_.at(node->denId_).ss_;
  auto den_el_staging = dsc_->dataStageParam_.at(node->denId_).el_;
  auto num_ss = num_ss_staging.dataStageDimToVal_compView_st(
      dim, comp_, corelet_id_, padInfo);
  auto num_el = num_el_staging.dataStageDimToVal_compView_st(
      dim, comp_, corelet_id_, padInfo);
  auto den_ss = den_ss_staging.dataStageDimToVal_compView_st(
      dim, comp_, corelet_id_, padInfo);
  auto den_el = den_el_staging.dataStageDimToVal_compView_st(
      dim, comp_, corelet_id_, padInfo);

  loop = nullptr;
  if (num_ss == num_el) {
    if (den_ss == den_el) {
      int size = static_cast<int>(std::ceil(num_ss / den_ss));
      loop = mlir::affine::AffineForOp::create(builder, loc, 0, size, 1,
                                               iter_args);
      builder.setInsertionPointToStart(
          dyn_cast<mlir::affine::AffineForOp>(loop).getBody());
      (*dsc_loops_to_mlir_loops_map_)[node].push_back(loop);
      resetIterArguments(loop, iter_args);
    } else {
      int size = static_cast<int>(std::ceil((num_ss - den_el) / den_ss)) + 1;
      loop = mlir::affine::AffineForOp::create(builder, loc, 0, size, 1,
                                               iter_args);
      builder.setInsertionPointToStart(
          dyn_cast<mlir::affine::AffineForOp>(loop).getBody());
      (*dsc_loops_to_mlir_loops_map_)[node].push_back(loop);
      resetIterArguments(loop, iter_args);
    }
  } else {
    int ss_iters = static_cast<int>(std::ceil(num_ss / den_ss));  // SS + SS
    int el_iters = 0;
    if (den_ss != den_el) {
      el_iters = static_cast<int>(std::ceil((num_el - den_el) / den_ss)) +
                 1;  // EL + EL
    } else {
      el_iters = static_cast<int>(std::ceil(num_el / den_ss));  // EL + SS
    }

    Value parent_loop_last_val;
    Value parent_loop_iv;
    auto *parent_loop = this->getParentLoop(node, dim);
    if (auto affine_for =
            llvm::dyn_cast<mlir::affine::AffineForOp>(parent_loop)) {
      if (affine_for.affine::AffineForOp::hasConstantBounds()) {
        auto ub = affine_for.affine::AffineForOp::getConstantUpperBound();
        parent_loop_iv = affine_for.affine::AffineForOp::getInductionVar();
        parent_loop_last_val =
            mlir::arith::ConstantIndexOp::create(builder, loc, ub - 1);
      } else {
        DT_CHECK("Unexpected branch");
      }
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(parent_loop)) {
      parent_loop_iv = scf_for.scf::ForOp::getInductionVar();
      auto val_one = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
      parent_loop_last_val = mlir::arith::SubIOp::create(
          builder, loc, scf_for.scf::ForOp::getUpperBound(), val_one);
      // Logically speaking based on the construction, we shouldn't
      // encounter this branch.
    } else {
      DT_CHECK("Unexpected branch");
    }

    // Construct cmpi conditions to selective choose the loop size
    if (ss_iters != el_iters) {
      auto cond = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::slt, parent_loop_iv,
          parent_loop_last_val);
      auto ss_val =
          mlir::arith::ConstantIndexOp::create(builder, loc, ss_iters);
      auto el_val =
          mlir::arith::ConstantIndexOp::create(builder, loc, el_iters);
      auto ub =
          mlir::arith::SelectOp::create(builder, loc, cond, ss_val, el_val);
      auto lb = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
      auto step = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
      loop = scf::ForOp::create(builder, loc, lb, ub, step, iter_args);
      builder.setInsertionPointToStart(dyn_cast<scf::ForOp>(loop).getBody());
      (*dsc_loops_to_mlir_loops_map_)[node].push_back(loop);
      resetIterArguments(loop, iter_args);
    } else {
      loop = mlir::affine::AffineForOp::create(builder, loc, 0, ss_iters, 1,
                                               iter_args);
      builder.setInsertionPointToStart(
          dyn_cast<mlir::affine::AffineForOp>(loop).getBody());
      (*dsc_loops_to_mlir_loops_map_)[node].push_back(loop);
      resetIterArguments(loop, iter_args);
    }
  }

  return loop;
}

// ---- 59/110  constructUniformizedFoldedDestinationCore  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNDSCLowering.cpp:425  (36L)
Value e059_constructUniformizedFoldedDestinationCore(
    OpBuilder &builder, const FoldManager<int64_t> &addresses) {
  SmallVector<mlir::Value> keys;
  SmallVector<mlir::Value> values;
  for (auto &core_id : this->core_ids_used_) {
    for (auto &corelet_id : this->corelet_ids_used_) {
      auto unit_def_op = this->unit_to_value_map_->at(core_id)
                             .at(corelet_id)
                             .at(comp_)
                             .getDefiningOp();
      // expecting constant value across sdsc folds
      int dst_core_id = int(FoldInfraUtils::getSingleDataStrict(
          addresses, {{0, core_id}, {1, corelet_id}}));
      auto dst_unit_op = createGetUnitOpInDifferentCore(
          builder, comp_, dst_core_id, corelet_id, num_folds_);
      for (int fold_id = 0; fold_id < num_folds_; fold_id++) {
        keys.push_back(unit_def_op->getResult(fold_id));
        values.push_back(dst_unit_op.getResult(fold_id));
      }
    }
  }

  DT_CHECK(keys.size() == values.size());

  // create a immutable map
  auto map = uniform::DefImmutableMappingOp::create(
      builder, builder.getUnknownLoc(), builder.getIndexType(), keys, values);

  // create a query op
  auto query_op = uniform::QueryMapOp::create(
      builder, map.getLoc(), builder.getIndexType(), map.getResult(),
      this->uniform_region_iterator_ ? this->uniform_region_iterator_
                                     : this->program_unit_iterator_);

  return query_op.getResult();
}

// ---- 60/110  constructStickMaskOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNStickMaskLowering.cpp:21  (58L)
LogicalResult e060_constructStickMaskOperation(
    OpBuilder &builder) {
  auto view = this->stick_mask_->getView();

  // construct masked/unmasked offsets.
  std::vector<int> unmasked_offsets, masked_offsets;
  unmasked_offsets.push_back(view.maskA_.first);
  unmasked_offsets.push_back(view.maskB_.first);
  masked_offsets.push_back(view.maskA_.second);
  masked_offsets.push_back(view.maskB_.second);

  int cst_idx = this->stick_mask_->maskValConstId_;
  DT_CHECK_MSG(
      cst_idx >= 0,
      "constant id should be non-negative when using with CONST container");
  auto &cst_info = dsc_->constantInfo_.at(cst_idx);

  // construct type
  auto element_type = getMLIRTypeFromDSCDataFormat(
      cst_info.dataFormat_, LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR,
      builder);
  int num_elements =
      (128 * 8) / dataflow::utils::getIntOrFloatBitWidth(
                      element_type);  // 128B/size
  Type result_type =
      dataflow::utils::constructVectorType(element_type, num_elements);

  // constructing slice mask
  int num_slices = 8;
  std::string slice_mask_map;
  for (int i = 0; i < num_slices; i++) {
    if (i < view.transitionSliceId_)
      slice_mask_map += "(A)";
    else if (i == view.transitionSliceId_)
      slice_mask_map += "(A|B)";
    else if (i > view.transitionSliceId_)
      slice_mask_map += "(1)";
  }

  // create mask value
  auto all_data = cst_info.data_.getAllData();
  DT_CHECK_MSG(all_data.size() == 1, "No folding over SAMV values");
  DT_CHECK_MSG(all_data.front().size() == 1,
               "Translator expects mask value to be a single element");
  mlir::Value mask_val = mlir::arith::ConstantIndexOp::create(
      builder, builder.getUnknownLoc(), all_data.front()[0]);

  // construct samv operation.
  auto samv_op = agen::SetTransferMaskStateOp::create(
      builder, builder.getUnknownLoc(), result_type, mask_val,
      builder.getStringAttr(this->stick_mask_->name_),
      builder.getI32IntegerAttr(num_slices),
      builder.getStringAttr(slice_mask_map),
      builder.getI32ArrayAttr(unmasked_offsets),
      builder.getI32ArrayAttr(masked_offsets));

  return LogicalResult::success();
}

// ---- 61/110  constructUnits  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNSyncLowering.cpp:20  (25L)
void e061_constructUnits(OpBuilder &builder,
                                    std::vector<mlir::Value> &units) {
  auto list = sync->getComponentsFromOtherEnds(core_id_);
  for (auto &record : list) {
    auto unit = record.first;
    // skip the signals from the l0lorow1-7.
    if (!is_any_of(unit, L0LUROW1, L0LUROW2, L0LUROW3, L0LUROW4, L0LUROW5,
                   L0LUROW6, L0LUROW7)) {
      for (auto &inner_record : record.second) {
        auto core_id = inner_record.first;
        if (!is_any_of(unit, L3LU, L3SU)) {
          for (auto &corelet : inner_record.second) {
            auto to_unit =
                retrieveGetUnitOpInSameCore(builder, unit, core_id, corelet);
            units.push_back(to_unit);
          }
        } else {
          auto to_unit =
              retrieveGetUnitOpInSameCore(builder, unit, core_id, -1);
          units.push_back(to_unit);
        }
      }
    }
  }
}

// ---- 62/110  constructImplicitSyncOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNSyncLowering.cpp:138  (66L)
LogicalResult e062_constructImplicitSyncOperation(
    OpBuilder &builder) {
  DT_CHECK_MSG(is_any_of(this->comp_, L0SU, L0LUROW0),
               "implicit sync is supported only in L0SU/L0LUROW0");

  // step-1: constructing destination unit and memory
  SenComponents dst_comp = this->comp_ == L0SU ? L0LUROW0 : L0SU;
  mlir::Value dst_unit = this->getComponentHandler(dst_comp);
  auto l0_memory = this->getComponentHandler(L0);

  // step-2: get tile sizes
  // Temporary fix to get tile sizes in case of uniformization.
  // TODO: Needs to be fixed for imbalanced work split
  int corelet_id = corelet_id_;
  if (this->uniformization_enabled_ && corelet_id == -1) {
    corelet_id = 0;
  }
  auto tilesizes_per_dim_ss =
      this->dsc_->getImplicitSyncTileSizePerDim(*this->sync, corelet_id, false);
  auto tilesizes_per_dim_el =
      this->dsc_->getImplicitSyncTileSizePerDim(*this->sync, corelet_id, true);

  int tilesize_ss = 1, tilesize_el = 1;
  for (auto &record : tilesizes_per_dim_ss) {
    tilesize_ss *= record.second;
  }

  for (auto &record : tilesizes_per_dim_el) {
    tilesize_el *= record.second;
  }

  DT_CHECK_MSG(tilesize_ss >= 1, "implicit tile size sync has to be >= 1");
  DT_CHECK_MSG(tilesize_el == tilesize_ss,
               "translator currently doesn't epilogues in implicit sync");

  // step-3: construct tile size variable
  mlir::Value tile_size = mlir::arith::ConstantIndexOp::create(
      builder, builder.getUnknownLoc(), tilesize_ss);

  // step-4: construct memory view
  // Get precision
  const auto &src_ld_idx =
      sync->implicitSyncRefTransfer_->srcLdsAndLoopOffsets_.myLdsIdx_;
  auto src_prec = dsc_->labeledDs_[src_ld_idx].dataFormat_;
  auto src_category = dsc_->labeledDs_[src_ld_idx].scaledLdsCategory_;
  auto element_type =
      getMLIRTypeFromDSCDataFormat(src_prec, src_category, builder);

  // construct layout map
  auto layout_map =
      AffineMap::get(1, 0, getAffineConstantExpr(0, builder.getContext()));
  auto memref_type = MemRefType::get(1, element_type);
  mlir::Value start_address =
      mlir::arith::ConstantIndexOp::create(builder, builder.getUnknownLoc(), 0);
  auto view = dataflow::GetLogicalMemoryViewOp::create(
      builder, builder.getUnknownLoc(), memref_type, l0_memory, start_address,
      layout_map);

  // step-5: construct sync operation.
  if (dataflow::ImplicitSyncOnStreamingBufferOp::create(
          builder, builder.getUnknownLoc(), view.getResult(), dst_unit,
          tile_size, builder.getStringAttr(sync->name_)))
    return LogicalResult::success();

  return LogicalResult::failure();
}

// ---- 63/110  getLabeledDsType  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:28  (21L)
LogicalResult e063_getLabeledDsType(
    DataFormats src_prec, LabeledDsInfo::ScaledLdsCategory category,
    OpBuilder &builder, Type &type) {
  // Construct result type
  int elements = 1;
  for (const auto &dim_size : transfer_->unitTimeTransferChunkSize_) {
    elements *= dim_size.sizeDim_.size_;
  }

  if (transfer_->unitTimeTransferNumChunks_ > 0)
    elements *= transfer_->unitTimeTransferNumChunks_;

  // auto src_prec = dsc.labeledDs_[ld_idx].dataFormat_;
  auto element_type = getMLIRTypeFromDSCDataFormat(src_prec, category, builder);
  if (mlir::isa<NoneType>(element_type)) {
    return LogicalResult::failure();
  }

  type = mlir::dataflow::utils::constructVectorType(element_type, elements);
  return LogicalResult::success();
}

// ---- 64/110  getBufferingOrStreamingMode  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:50  (39L)
LogicalResult e064_getBufferingOrStreamingMode(int &mode) {
  mode = -1;  // Neither buffering or streaming
  if (transfer_->src_.unit_ == comp_) {
    if (transfer_->srcLdsAndLoopOffsets_.bufferSwitchPosition_ != nullptr) {
      auto num_buffers =
          dsc_->labeledDs_.at(transfer_->srcLdsAndLoopOffsets_.myLdsIdx_)
              .memOrg_.at(transfer_->src_.storage_)
              .allocateNode_->numBuffers_;
      if (num_buffers == -1) {
        mode = 2;  // Streaming
        return LogicalResult::success();
      } else {
        mode = 1;  // Buffering
        return LogicalResult::success();
      }
    }
  }

  for (int i = 0; i < transfer_->dstVias_.size(); i++) {
    const auto &via = transfer_->dstVias_[i];
    if (via.loc_.unit_ == comp_) {
      const auto &lds = transfer_->dstLdsAndLoopOffsets_[i];
      if (lds.bufferSwitchPosition_ != nullptr) {
        auto num_buffers = dsc_->labeledDs_.at(lds.myLdsIdx_)
                               .memOrg_.at(via.loc_.storage_)
                               .allocateNode_->numBuffers_;
        if (num_buffers == -1) {
          mode = 2;  // streaming
          return LogicalResult::success();
        } else {
          mode = 1;  // Buffering
          return LogicalResult::success();
        }
      }
    }
  }

  return LogicalResult::success();
}

// ---- 65/110  constructBaseAddress  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:132  (73L)
LogicalResult e065_constructBaseAddress(
    OpBuilder &builder, int ndims,
    llvm::SmallVectorImpl<Value> &base_address_args,
    AffineMap &base_address_map,
    const std::vector<dsc2::TransferNode::UnitView::LoopInfo> &loop_strides,
    bool bypass_strides, bool is_scalereg) {
  auto context = builder.getContext();
  SmallVector<AffineExpr, 8> base_address_exprs;
  SmallVector<AffineExpr, 8> iv_exprs;

  //  for (int dim = 0; dim < ndims; dim++) {
  //    iv_exprs.push_back(getAffineDimExpr(dim, context));
  //  }

  if (!bypass_strides && !is_scalereg) {
    int unique_variables = 0;
    for (int dim = 0; dim < ndims; dim++) {
      bool modified = false;
      AffineExpr base_address_expr = getAffineConstantExpr(0, context);
      for (int i = 0; i < loop_strides.size(); i++) {
        if (dim == loop_strides[i].sizeIdx_) {
          modified = true;
          AffineExpr mul_expr;
          if (loop_strides[i].loop_ != nullptr) {
            auto iv_expr = getAffineDimExpr(unique_variables++, context);
            mul_expr = iv_expr * loop_strides[i].elemOffset_;
            auto *loop = this->getMLIRLoopFromLoopNode(loop_strides[i].loop_,
                                                       loop_strides[i].dim_);
            if (auto affine_for =
                    llvm::dyn_cast<mlir::affine::AffineForOp>(loop)) {
              base_address_args.push_back(affine_for.getInductionVar());
            } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop)) {
              base_address_args.push_back(scf_for.getInductionVar());
            }
          } else {
            mul_expr =
                getAffineConstantExpr(loop_strides[i].elemOffset_, context);
          }

          base_address_expr = base_address_expr + mul_expr;
        }
      }

      if (!modified) {
        unique_variables++;
        auto const_op = mlir::arith::ConstantIndexOp::create(
            builder, builder.getUnknownLoc(), 0);
        base_address_args.push_back(const_op.getResult());
      }

      base_address_exprs.push_back(base_address_expr);
    }

    base_address_map =
        AffineMap::get(unique_variables, 0, base_address_exprs, context);
  } else {
    if (is_scalereg) {
      SmallVector<AffineExpr> zero_exprs(ndims,
                                         getAffineConstantExpr(0, context));
      base_address_map = AffineMap::get(ndims, 0, zero_exprs, context);
      for (int i = 0; i < ndims; i++) {
        auto const_op = mlir::arith::ConstantIndexOp::create(
            builder, builder.getUnknownLoc(), 0);
        base_address_args.push_back(const_op.getResult());
      }
    } else {
      base_address_map = AffineMap::getConstantMap(0, builder.getContext());
    }
  }

  DEBUG_WITH_TYPE(verbose_debug_, base_address_map.dump());
  return LogicalResult::success();
}

// ---- 66/110  constructTimeSet  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:260  (47L)
LogicalResult e066_constructTimeSet(
    OpBuilder &builder, llvm::SmallVectorImpl<Value> &args,
    const std::vector<dsc2::TransferNode::UnitView::LoopInfo> &loops,
    IntegerSet &time_set) {
  int ndims = loops.size();
  int num_eq = 0;
  int num_symbols = 0;
  int num_ineq = 2 * ndims;
  int total_constraints = num_eq + num_ineq;
  auto context = builder.getContext();

  SmallVector<bool, 16> eq_flags(total_constraints);
  std::fill(eq_flags.begin(), eq_flags.begin() + num_ineq, false);
  std::fill(eq_flags.begin() + num_ineq, eq_flags.end(), true);

  SmallVector<AffineExpr, 8> exprs;

  // Construct in_equalities
  for (int i = 0; i < num_ineq / 2; i++) {
    auto id = getAffineDimExpr(i, context);
    exprs.push_back(id);  // id >= 0

    // The other constraint: id < x
    //  x - id > 0 --> x - id - 1 >= 0
    auto *loop = this->getMLIRLoopFromLoopNode(loops[i].loop_, loops[i].dim_);
    if (auto affine_for = llvm::dyn_cast<mlir::affine::AffineForOp>(loop)) {
      if (affine_for.hasConstantUpperBound()) {
        exprs.push_back(affine_for.getConstantUpperBound() - id - 1);
      } else {
        num_symbols++;
        // args.push_back(affine_for.getUpperBound().getOperand(0));
        return LogicalResult::failure();
      }
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop)) {
      if (auto const_bound = llvm::dyn_cast<mlir::arith::ConstantIndexOp>(
              scf_for->getOperand(1).getDefiningOp())) {
        exprs.push_back(const_bound.value() - id - 1);
      } else {
        return LogicalResult::failure();
      }
    }
  }

  time_set = IntegerSet::get(ndims, 0, exprs, eq_flags);
  DEBUG_WITH_TYPE(verbose_debug_, time_set.dump());
  return LogicalResult::success();
}

// ---- 67/110  constructLoadOrStoreSet  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:314  (90L)
LogicalResult e067_constructLoadOrStoreSet(
    OpBuilder &builder,
    const std::vector<dsc2::TransferNode::SizeAndIndex>
        &unit_time_transfer_chunk_size,
    const std::vector<dsc2::TransferNode::SizeAndIndex>
        &unit_time_transfer_chunk_stride,
    const std::vector<dsc2::ScheduleNode::Size> &view_sizes, bool is_load,
    IntegerSet &load_or_store_set, int num_chunk_strides) {
  auto context = builder.getContext();
  SmallVector<AffineExpr, 8> exprs, eq_exprs, ineq_exprs;

  if (!unit_time_transfer_chunk_stride.empty() &&
      unit_time_transfer_chunk_stride.size() > 1) {
    emitError(
        "Currently supports chunk strides with striding over a single "
        "dimension");
    return LogicalResult::failure();
  }

  for (int dim_id = 0; dim_id < view_sizes.size(); dim_id++) {
    auto id = getAffineDimExpr(dim_id, context);

    // find index of dim_id in unit time transfer chunk size
    int idx_in_chunk_size = -1;
    for (int i = 0; i < unit_time_transfer_chunk_size.size(); i++) {
      if (is_load) {
        if (unit_time_transfer_chunk_size[i].srcSizeIdx_ == dim_id) {
          idx_in_chunk_size = i;
          break;
        }
      } else {
        if (unit_time_transfer_chunk_size[i].dstSizeIdx_ == dim_id) {
          idx_in_chunk_size = i;
          break;
        }
      }
    }

    // find index of dim_id in unit time transfer chunk stride
    int idx_in_chunk_stride = -1;
    for (int i = 0; i < unit_time_transfer_chunk_stride.size(); i++) {
      if (is_load) {
        if (unit_time_transfer_chunk_stride[i].srcSizeIdx_ == dim_id) {
          idx_in_chunk_stride = i;
          break;
        }
      } else {
        if (unit_time_transfer_chunk_stride[i].dstSizeIdx_ == dim_id) {
          idx_in_chunk_stride = i;
          break;
        }
      }
    }

    if (idx_in_chunk_size != -1 && idx_in_chunk_stride != -1) {
      emitError(
          "A dimension cannot be present in both chunk size and stride\n");
      return LogicalResult::failure();
    }

    if (idx_in_chunk_size != -1 || idx_in_chunk_stride != -1) {
      int size =
          (idx_in_chunk_stride != -1)
              ? num_chunk_strides
              : unit_time_transfer_chunk_size[idx_in_chunk_size].sizeDim_.size_;
      if (size > 1) {
        ineq_exprs.push_back(id);  // id >= 0

        // The other constraint: id < x
        //  x - id > 0 --> x - id - 1 >= 0
        ineq_exprs.push_back(size - id - 1);
      } else {
        eq_exprs.push_back(id);  // id == 0
      }
    } else {
      eq_exprs.push_back(id);  // id == 0
    }
  }

  exprs.insert(exprs.end(), ineq_exprs.begin(), ineq_exprs.end());
  exprs.insert(exprs.end(), eq_exprs.begin(), eq_exprs.end());

  SmallVector<bool, 16> eq_flags(eq_exprs.size() + ineq_exprs.size());
  std::fill(eq_flags.begin(), eq_flags.begin() + ineq_exprs.size(), false);
  std::fill(eq_flags.begin() + ineq_exprs.size(), eq_flags.end(), true);

  load_or_store_set = IntegerSet::get(view_sizes.size(), 0, exprs, eq_flags);
  DEBUG_WITH_TYPE(verbose_debug_, load_or_store_set.dump());
  return LogicalResult::success();
}

// ---- 68/110  constructImplicitLoopsForContiguousTransfer  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:707  (137L)
LogicalResult e068_constructImplicitLoopsForContiguousTransfer(
    OpBuilder &builder, const std::vector<dsc2::ScheduleNode::Size> &view_sizes,
    const std::vector<dsc2::TransferNode::SizeAndIndex> &unit_time_transfer,
    std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> &implicit_loops,
    std::unordered_map<PrimaryDimTypes, int> &ctgs_transfer_sizes) {
  // Construct outer-loops
  std::vector<std::pair<int, int>> outer_loop_sizes;

  // references to parent loops in case ss != el
  std::vector<const dsc2::LoopNode *> parent_loops;

  // the indexing into this vector is based on parent_loops.
  std::vector<PrimaryDimTypes> parent_loop_to_dim;

  if (!view_sizes.empty()) {
    int unit_time_dims = unit_time_transfer.size();
    for (int i = unit_time_dims; i < view_sizes.size(); i++) {
      auto record_ss = src_sticks_ss_per_dim.find(view_sizes[i].dim_);
      if (record_ss != src_sticks_ss_per_dim.end()) {
        auto record_el = src_sticks_el_per_dim.find(view_sizes[i].dim_);
        if (view_sizes[i].dim_ == PrimaryDimTypes::OUT) {
          record_ss->second = record_ss->second / transfer_->replicationFactor_;
          record_el->second = record_el->second / transfer_->replicationFactor_;
        }
        // avoid unit size loops or zero loops after substituition
        if (record_ss->second > 1 || record_el->second > 1) {
          dsc2::ScheduleNode::UnitView::LoopInfo loop_info;
          loop_info.loop_ = new dsc2::LoopNode(-1, -1, {view_sizes[i].dim_});
          loop_info.sizeIdx_ = i;
          loop_info.elemOffset_ = 1;
          loop_info.dim_ = view_sizes[i].dim_;
          implicit_loops.push_back(loop_info);
          outer_loop_sizes.emplace_back(record_ss->second, record_el->second);

          // store parent information.
          if (record_ss->second == record_el->second) {
            parent_loops.push_back(nullptr);
            parent_loop_to_dim.push_back(PrimaryDimTypes::PrimaryDimTypesCount);
          } else {
            const auto *parent_loop = getImmediateParentWithMatchingDim(
                transfer_->getOwnerLoop(), view_sizes[i].dim_);
            parent_loops.push_back(parent_loop);
            parent_loop_to_dim.push_back(view_sizes[i].dim_);
          }

          record_ss->second -= view_sizes[i].size_;
          record_el->second -= view_sizes[i].size_;
        }
      } else {
        DT_ERROR("view dims and contigous transfer dim didn't match");
      }
    }
  } else {
    DT_CHECK_MSG(sticks_src_ss == sticks_src_el,
                 "Number of contiguous "
                 "transfers should be same in steady state/epilouge state");
    if (sticks_src_ss > 1) {
      // create outer dummy loop
      dsc2::ScheduleNode::UnitView::LoopInfo loop_info;
      loop_info.loop_ = new dsc2::LoopNode();
      loop_info.sizeIdx_ = -1;
      loop_info.elemOffset_ = 1;
      implicit_loops.push_back(loop_info);
      outer_loop_sizes.emplace_back(sticks_src_ss, sticks_src_el);
    }
  }

  for (int i = outer_loop_sizes.size() - 1; i >= 0; i--) {
    if (outer_loop_sizes[i].first == outer_loop_sizes[i].second) {
      auto loop = mlir::affine::AffineForOp::create(
          builder, builder.getUnknownLoc(), 0, outer_loop_sizes[i].first, 1);
      dataflow::setDbgName(
          loop, "ImplicitLoopForContiguousTransfer(" + transfer_->name_ + ")");
      builder.setInsertionPointToStart(loop.getBody());
      (*dsc_loops_to_mlir_loops_map_)[implicit_loops[i].loop_].push_back(loop);
    } else {
      // create cmpi operation, scf loops.
      // TODO: this code needs to be refactored with SS/EL loops creation.
      Location loc = builder.getUnknownLoc();
      Value parent_loop_iv;
      Value parent_loop_last_val;
      DT_CHECK_MSG(parent_loops[i] != nullptr,
                   "parent loop in "
                   "ss/el data transfer shouldn't be empty");
      Operation *parent_loop = nullptr;
      int ndims = parent_loops[i]->dims_.size();
      for (int j = 0; j < ndims; j++) {
        if (parent_loops[i]->dims_[j] == parent_loop_to_dim[i]) {
          parent_loop = (*dsc_loops_to_mlir_loops_map_)
                            .at(parent_loops[i])
                            .at(ndims - j - 1);
        }
      }

      DT_CHECK("parent loop in implicit loops cannot be empty" && parent_loop);
      if (auto affine_for =
              llvm::dyn_cast<mlir::affine::AffineForOp>(parent_loop)) {
        if (affine_for.hasConstantBounds()) {
          auto ub = affine_for.getConstantUpperBound();
          parent_loop_iv = affine_for.getInductionVar();
          parent_loop_last_val =
              mlir::arith::ConstantIndexOp::create(builder, loc, ub - 1);
        } else {
          return LogicalResult::failure();
        }
      } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(parent_loop)) {
        parent_loop_iv = scf_for.getInductionVar();
        auto val_one = mlir::arith::ConstantIndexOp::create(
            builder, builder.getUnknownLoc(), 1);
        parent_loop_last_val = mlir::arith::SubIOp::create(
            builder, loc, scf_for.getUpperBound(), val_one);
      } else {
        // Logically speaking based on the construction, we shouldn't encounter
        // this branch.
        return LogicalResult::failure();
      }

      // create cmpi operation
      auto cond = mlir::arith::CmpIOp::create(
          builder, loc, mlir::arith::CmpIPredicate::slt, parent_loop_iv,
          parent_loop_last_val);
      auto ss_val = mlir::arith::ConstantIndexOp::create(
          builder, loc, outer_loop_sizes[i].first);
      auto el_val = mlir::arith::ConstantIndexOp::create(
          builder, loc, outer_loop_sizes[i].second);
      auto ub =
          mlir::arith::SelectOp::create(builder, loc, cond, ss_val, el_val);
      auto lb = mlir::arith::ConstantIndexOp::create(builder, loc, 0);
      auto step = mlir::arith::ConstantIndexOp::create(builder, loc, 1);
      auto loop = scf::ForOp::create(builder, loc, lb, ub, step);
      builder.setInsertionPointToStart(loop.getBody());
      (*dsc_loops_to_mlir_loops_map_)[implicit_loops[i].loop_].push_back(loop);
    }
  }

  return LogicalResult::success();
}

// ---- 69/110  areEpiloguesInLoops  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1464  (24L)
bool e069_areEpiloguesInLoops(
    std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> &loops) {
  for (auto &cloop : loops) {
    auto *loop = this->getMLIRLoopFromLoopNode(cloop.loop_, cloop.dim_);
    if (auto affine_for = llvm::dyn_cast<mlir::affine::AffineForOp>(loop)) {
      if (affine_for.hasConstantUpperBound()) {
        continue;
      } else {
        return true;
      }
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop)) {
      if (auto const_bound = llvm::dyn_cast<mlir::arith::ConstantIndexOp>(
              scf_for->getOperand(1).getDefiningOp())) {
        continue;
      } else {
        return true;
      }
    } else {
      llvm_unreachable("Unknown for-loops");
    }
  }

  return false;
}

// ---- 70/110  GenerateConstantBitStreamAndShuffle  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2475  (41L)
LogicalResult e070_GenerateConstantBitStreamAndShuffle(
    const dsc2::ConstantInfo &cst_info, OpBuilder &loop_builder,
    Value &input) const {  // create constant bit stream operation.
  bool bistream_element_integer;
  Type bitstream_element_type;
  Type bitstream_vector_type;

  auto all_data = cst_info.data_.getAllData();
  if (SNComputeLowering::constructTypeFromFormat(
          loop_builder, cst_info.dataFormat_,
          LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR, comp_,
          all_data.front().size(), bitstream_vector_type,
          bitstream_element_type, bistream_element_integer)
          .failed()) {
    return LogicalResult::failure();
  }

  auto current_lowering = (SNTransferLowering *)this;
  auto bit_stream_op_result =
      current_lowering->constructUniformizedFoldedConstantBitStream(
          loop_builder, cst_info.data_, bitstream_vector_type,
          cst_info.isDataSymbolic_);

  // create a splat operation
  llvm::SmallVector<Attribute> index_attrs;
  for (int i = 0; i < all_data.front().size(); i++) {
    auto attr = loop_builder.getI32IntegerAttr(i);
    index_attrs.push_back(attr);
  }

  int repetition = dataflow::utils::getNumElements(result_type) /
                   dataflow::utils::getNumElements(bitstream_vector_type);
  auto shuffle_op = vectorchain::ShuffleOp::create(
      loop_builder, loop_builder.getUnknownLoc(), result_type,
      bit_stream_op_result, nullptr, loop_builder.getArrayAttr(index_attrs),
      loop_builder.getI32IntegerAttr(repetition),
      loop_builder.getStringAttr(transfer_->name_));
  input = shuffle_op.getResult();

  return LogicalResult::success();
}

// ---- 71/110  getCumulativeStickSizes  --  dsc/dsc2.cpp:4108  (17L)
std::unordered_map<PrimaryDimTypes, int>
e071_getCumulativeStickSizes(DsTypes dsType,
                                           bool stickSliceOnly /*= false*/,
                                           bool stickWithoutSlice /*= false*/,
                                           bool l0SliceOnly /*= false*/,
                                           int numL0Slices /*= -1*/) const {
  std::unordered_map<PrimaryDimTypes, int> result;
  auto stickSizes = getStickSizes(dsType, stickSliceOnly, stickWithoutSlice,
                                  l0SliceOnly, numL0Slices);
  for (auto& dimSize : stickSizes) {
    if (result.count(dimSize.first))
      result[dimSize.first] *= dimSize.second;
    else
      result[dimSize.first] = dimSize.second;
  }
  return result;
}


// ================================================================================================
// LEVEL 2
// ================================================================================================

// ---- 72/110  createUniformizedGetUnitOp  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:95  (42L)
Value e072_createUniformizedGetUnitOp(
    OpBuilder &builder, SenComponents comp,
    mlir::Value program_unit_op_iterator,
    SmallVectorImpl<mlir::Value> &get_unit_ops, std::vector<int> &cores,
    std::vector<int> &corelets) {
  if (component_to_handler_.count(comp) == 0) {
    SmallVector<mlir::Value> units;
    for (auto &core_id : cores) {
      if (is_any_of(comp, L3LU, L3SU)) {
        auto get_unit = createGetUnitOp(builder, comp, core_id);
        // get_unit_ops could correspond to multiple corelets, for, e.g., in
        // LXLU. we need to have two entries in the units collection so that
        // LXLU0 and LXLU1 point to the same L3 thing.
        for (auto &corelet_id : corelets) {
          for (int i = 0; i < num_folds_; i++)
            units.emplace_back(get_unit.getResult(i));
        }
      } else {
        for (auto &corelet_id : corelets) {
          auto get_unit = createGetUnitOp(builder, comp, core_id, corelet_id);
          for (int i = 0; i < num_folds_; i++)
            units.emplace_back(get_unit.getResult(i));
        }
      }
    }

    // create a uniform immutable mapping
    DT_CHECK(get_unit_ops.size() == units.size());
    auto map = uniform::DefImmutableMappingOp::create(
        builder, builder.getUnknownLoc(), builder.getIndexType(), get_unit_ops,
        units);

    // create a query op
    auto result = uniform::QueryMapOp::create(
        builder, map.getLoc(), builder.getIndexType(), map.getResult(),
        program_unit_op_iterator);

    component_to_handler_[comp] = result.getResult();
  }

  return component_to_handler_[comp];
}

// ---- 73/110  buildNeighborUnits  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:161  (213L)
void e073_buildNeighborUnits(
    int core_id, int corelet_id, SenComponents comp, OpBuilder &builder,
    const DesignSpaceConfigGlobal &dsc_global) {
  auto unit_op =
      createGetUnitOp(builder, comp, core_id, corelet_id).getResult(0);

  if (comp == PTROW0) {
    component_to_handler_[PTROW1] = component_to_handler_[PTSOUTH] =
        createGetUnitOp(builder, PTROW1, core_id, corelet_id).getResult(0);
    component_to_handler_[SFP] = component_to_handler_[PTNORTH] =
        createGetUnitOp(builder, SFP, core_id, corelet_id).getResult(0);
    component_to_handler_[LXLU] =
        createGetUnitOp(builder, LXLU, core_id, corelet_id).getResult(0);
    component_to_handler_[PTWEST] =
        createGetUnitOp(builder, L0LU, core_id, corelet_id).getResult(0);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW1) {
    component_to_handler_[PTROW2] = component_to_handler_[PTSOUTH] =
        createGetUnitOp(builder, PTROW2, core_id, corelet_id).getResult(0);
    component_to_handler_[PTROW0] = component_to_handler_[PTNORTH] =
        createGetUnitOp(builder, PTROW0, core_id, corelet_id).getResult(0);
    component_to_handler_[PTWEST] =
        createGetUnitOp(builder, L0LU, core_id, corelet_id).getResult(0);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW2) {
    component_to_handler_[PTROW3] = component_to_handler_[PTSOUTH] =
        createGetUnitOp(builder, PTROW3, core_id, corelet_id).getResult(0);
    component_to_handler_[PTROW1] = component_to_handler_[PTNORTH] =
        createGetUnitOp(builder, PTROW1, core_id, corelet_id).getResult(0);
    component_to_handler_[PTWEST] =
        createGetUnitOp(builder, L0LU, core_id, corelet_id).getResult(0);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW3) {
    if (dsc_global.sysDef.coreArch <= RCUDD1A_ISA)
      component_to_handler_[PTROW4] = component_to_handler_[PTSOUTH] =
          createGetUnitOp(builder, PTROW4, core_id, corelet_id).getResult(0);
    else
      component_to_handler_[PE] = component_to_handler_[PTSOUTH] =
          createGetUnitOp(builder, PE, core_id, corelet_id).getResult(0);
    component_to_handler_[PTROW2] = component_to_handler_[PTNORTH] =
        createGetUnitOp(builder, PTROW2, core_id, corelet_id).getResult(0);
    component_to_handler_[PTWEST] =
        createGetUnitOp(builder, L0LU, core_id, corelet_id).getResult(0);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW4) {
    component_to_handler_[PTROW5] = component_to_handler_[PTSOUTH] =
        createGetUnitOp(builder, PTROW5, core_id, corelet_id).getResult(0);
    component_to_handler_[PTROW3] = component_to_handler_[PTNORTH] =
        createGetUnitOp(builder, PTROW3, core_id, corelet_id).getResult(0);
    component_to_handler_[PTWEST] =
        createGetUnitOp(builder, L0LU, core_id, corelet_id).getResult(0);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW5) {
    component_to_handler_[PTROW6] = component_to_handler_[PTSOUTH] =
        createGetUnitOp(builder, PTROW6, core_id, corelet_id).getResult(0);
    component_to_handler_[PTROW4] = component_to_handler_[PTNORTH] =
        createGetUnitOp(builder, PTROW4, core_id, corelet_id).getResult(0);
    component_to_handler_[PTWEST] =
        createGetUnitOp(builder, L0LU, core_id, corelet_id).getResult(0);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW6) {
    component_to_handler_[PTROW7] = component_to_handler_[PTSOUTH] =
        createGetUnitOp(builder, PTROW7, core_id, corelet_id).getResult(0);
    component_to_handler_[PTROW5] = component_to_handler_[PTNORTH] =
        createGetUnitOp(builder, PTROW5, core_id, corelet_id).getResult(0);
    component_to_handler_[PTWEST] =
        createGetUnitOp(builder, L0LU, core_id, corelet_id).getResult(0);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW7) {
    component_to_handler_[PE] = component_to_handler_[PTSOUTH] =
        createGetUnitOp(builder, PE, core_id, corelet_id).getResult(0);
    component_to_handler_[PTROW6] = component_to_handler_[PTNORTH] =
        createGetUnitOp(builder, PTROW6, core_id, corelet_id).getResult(0);
    component_to_handler_[PTWEST] =
        createGetUnitOp(builder, L0LU, core_id, corelet_id).getResult(0);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == L0LUROW0) {
    component_to_handler_[L0SU] =
        createGetUnitOp(builder, L0SU, core_id, corelet_id).getResult(0);
    component_to_handler_[PTROW0] =
        createGetUnitOp(builder, PTROW0, core_id, corelet_id).getResult(0);
    component_to_handler_[L0] =
        createGetUnitOp(builder, L0, core_id, corelet_id).getResult(0);
  } else if (comp == L0SU) {
    component_to_handler_[SFP] =
        createGetUnitOp(builder, SFP, core_id, corelet_id).getResult(0);
    component_to_handler_[L0LUROW0] =
        createGetUnitOp(builder, L0LUROW0, core_id, corelet_id).getResult(0);
    component_to_handler_[L0] =
        createGetUnitOp(builder, L0, core_id, corelet_id).getResult(0);
    component_to_handler_[LXLU] =
        createGetUnitOp(builder, LXLU, core_id, corelet_id).getResult(0);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == LXLU) {
    component_to_handler_[LXSU] =
        createGetUnitOp(builder, LXSU, core_id, corelet_id).getResult(0);
    component_to_handler_[SFP] =
        createGetUnitOp(builder, SFP, core_id, corelet_id).getResult(0);
    component_to_handler_[LX] =
        createGetUnitOp(builder, LX, core_id, corelet_id).getResult(0);
    component_to_handler_[PE] =
        createGetUnitOp(builder, PE, core_id, corelet_id).getResult(0);
    component_to_handler_[L3LU] =
        createGetUnitOp(builder, L3LU, core_id, corelet_id).getResult(0);
    component_to_handler_[L3SU] =
        createGetUnitOp(builder, L3SU, core_id, corelet_id).getResult(0);
    component_to_handler_[PTROW0] =
        createGetUnitOp(builder, PTROW0, core_id, corelet_id).getResult(0);
    component_to_handler_[L0SU] =
        createGetUnitOp(builder, L0SU, core_id, corelet_id).getResult(0);
  } else if (comp == LXSU) {
    component_to_handler_[LXLU] =
        createGetUnitOp(builder, LXLU, core_id, corelet_id).getResult(0);
    component_to_handler_[SFP] =
        createGetUnitOp(builder, SFP, core_id, corelet_id).getResult(0);
    component_to_handler_[LX] =
        createGetUnitOp(builder, LX, core_id, corelet_id).getResult(0);
    component_to_handler_[PE] =
        createGetUnitOp(builder, PE, core_id, corelet_id).getResult(0);
    component_to_handler_[L3LU] =
        createGetUnitOp(builder, L3LU, core_id, corelet_id).getResult(0);
    component_to_handler_[L3SU] =
        createGetUnitOp(builder, L3SU, core_id, corelet_id).getResult(0);
  } else if (comp == PE) {
    component_to_handler_[LXLU] =
        createGetUnitOp(builder, LXLU, core_id, corelet_id).getResult(0);
    component_to_handler_[LXSU] =
        createGetUnitOp(builder, LXSU, core_id, corelet_id).getResult(0);
    if (dsc_global.sysDef.coreArch <= RCUDD1A_ISA)
      component_to_handler_[PTROW7] =
          createGetUnitOp(builder, PTROW7, core_id, corelet_id).getResult(0);
    else
      component_to_handler_[PTROW3] =
          createGetUnitOp(builder, PTROW3, core_id, corelet_id).getResult(0);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PE_LRFREG, unit_op);
    component_to_handler_[SFP] =
        createGetUnitOp(builder, SFP, core_id, corelet_id).getResult(0);
    component_to_handler_[CONSTANT] =
        createGetUnitOp(builder, CONSTANT, core_id, corelet_id).getResult(0);
    component_to_handler_[PESTATE] =
        createGetUnitOp(builder, PESTATE, core_id, corelet_id).getResult(0);

  } else if (comp == SFP) {
    component_to_handler_[LXLU] =
        createGetUnitOp(builder, LXLU, core_id, corelet_id).getResult(0);
    component_to_handler_[LXSU] =
        createGetUnitOp(builder, LXSU, core_id, corelet_id).getResult(0);
    component_to_handler_[L0SU] =
        createGetUnitOp(builder, L0SU, core_id, corelet_id).getResult(0);
    component_to_handler_[PTROW0] =
        createGetUnitOp(builder, PTROW0, core_id, corelet_id).getResult(0);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, SFP_LRFREG, unit_op);
    component_to_handler_[PE] =
        createGetUnitOp(builder, PE, core_id, corelet_id).getResult(0);
    component_to_handler_[CONSTANT] =
        createGetUnitOp(builder, CONSTANT, core_id, corelet_id).getResult(0);
    component_to_handler_[SFPSTATE] =
        createGetUnitOp(builder, SFPSTATE, core_id, corelet_id).getResult(0);
  } else {
  }
}

// ---- 74/110  getBlockingOrStreamingBufferLoopLocations  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:467  (46L)
LogicalResult
e074_getBlockingOrStreamingBufferLoopLocations() {
  // TODO: corelet_id_);
  auto nodes = dsc_->scheduleTree_.traverseTreeDFS(
      nullptr, {dsc2::ScheduleNode::NodeType::TRANSFER}, comp_, corelet_id_,
      core_id_);
  for (auto &node : nodes) {
    int mode = -1;
    int offset = 0;
    double factor = 0;
    const FoldManager<int64_t> *start_address_map;
    const auto *transfer = dynamic_cast<const dsc2::TransferNode *>(node);
    if (failed(this->getBufferingOrStreamingMode(transfer, mode,
                                                 start_address_map, factor))) {
      emitError("Unable to get buffering or streaming mode");
      return LogicalResult::failure();
    }

    if (mode == 1 || mode == 2) {
      bool consider_transfer = false;
      const dsc2::LoopNode *buffer_position = nullptr;
      if (transfer->src_.unit_ == comp_) {
        consider_transfer = true;
        buffer_position = transfer->srcLdsAndLoopOffsets_.bufferSwitchPosition_;
      } else {
        int dst_idx = 0;
        for (const auto &dst : transfer->dstVias_) {
          if (dst.loc_.unit_ == comp_) {
            consider_transfer = true;
            buffer_position =
                transfer->dstLdsAndLoopOffsets_[0].bufferSwitchPosition_;
            break;
          }
        }

        dst_idx++;
      }

      if (consider_transfer && (buffer_position != nullptr)) {
        (dsc_loops_to_buffers_switch_map_)[buffer_position].push_back(transfer);
      }
    }
  }

  return LogicalResult::success();
}

// ---- 75/110  constructLoopIterArgs  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:595  (91L)
LogicalResult e075_constructLoopIterArgs(
    OpBuilder &builder, const dsc2::LoopNode *node,
    const dsc2::BlockNode *parent_node, Location &loc,
    llvm::SmallVectorImpl<Value> &iter_args, int dim_idx_band) {
  if (parent_node == nullptr && dim_idx_band == 0) {
    for (auto &transfer :
         (*dsc_all_parent_loops_to_buffers_switch_map_)[node]) {
      // Get the starting offset.
      int mode = -1;
      int offset = 0;
      double factor = 0;
      const FoldManager<int64_t> *start_address_map;
      if (failed(this->getBufferingOrStreamingMode(
              transfer, mode, start_address_map, factor))) {
        emitError("Unable to get buffering or streaming mode");
        return LogicalResult::failure();
      }

      if (mode == 2 || mode == 1) {
        mlir::Value address;
        if (needsUniform()) {
          address = constructUniformizedFoldedAddress(
              builder, *start_address_map, factor);
        } else {
          // expecting constant value across sdsc folds
          int start_addr =
              int(FoldInfraUtils::getSingleDataStrict(
                      *start_address_map, {{0, core_id_}, {1, corelet_id_}}) *
                  factor);
          auto const_op = mlir::arith::ConstantIndexOp::create(
              builder, builder.getUnknownLoc(), start_addr);
          address = const_op.getResult();
        }

        iter_args.push_back(address);
      } else {
        llvm_unreachable("Unknown buffering mode\n");
      }
    }

    return LogicalResult::success();
  }

  bool already_part_of_band = dim_idx_band > 0;
  const dsc2::LoopNode *parent_node_loop =
      parent_node == nullptr ? node : parent_node->getOwnerLoop();

  // if parent node is already a loop, then don't get owner loop.
  if (const auto *tmp = dynamic_cast<const dsc2::LoopNode *>(parent_node))
    parent_node_loop = tmp;

  // If dim_idx_band > 0, parent node is self.
  if (already_part_of_band) parent_node_loop = node;

  if (parent_node_loop) {
    for (auto &transfer :
         (*dsc_all_parent_loops_to_buffers_switch_map_)[node]) {
      int index = -1;
      auto &parent_transfers =
          (*dsc_all_parent_loops_to_buffers_switch_map_)[parent_node_loop];
      for (int i = 0; i < parent_transfers.size(); i++) {
        if (transfer == parent_transfers[i]) {
          index = i;
          break;
        }
      }

      if (index == -1) {
        return LogicalResult::failure();
      }

      auto *parent_for =
          (*dsc_loops_to_mlir_loops_map_)[parent_node_loop].back();

      // if the current node itself is parent, then dim_idx_band-1 is parent
      // loop.
      if (already_part_of_band)
        parent_for =
            (*dsc_loops_to_mlir_loops_map_)[parent_node_loop][dim_idx_band - 1];

      if (auto affine_for =
              llvm::dyn_cast<mlir::affine::AffineForOp>(parent_for)) {
        iter_args.push_back(affine_for.getRegionIterArgs()[index]);
      } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(parent_for)) {
        iter_args.push_back(scf_for.getRegionIterArgs()[index]);
      }
    }
  }

  return LogicalResult::success();
}

// ---- 76/110  constructSyncOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNSyncLowering.cpp:205  (69L)
LogicalResult e076_constructSyncOperation(OpBuilder &builder) {
  // Creating sync send operation
  // (sync->producers_).find(comp_) != (sync->producers_).end()
  bool is_implicit_sync = sync->implicitSyncRefTransfer_ != nullptr;
  if (!sync->isReceive_ && !is_implicit_sync) {  // producer
    bool wait_immediately_for_async_transfers =
        (sync->isSoft_ == 0 ? true : false);
    std::vector<mlir::Value> to_units;
    if (needsUniform()) {
      constructUnitsForUniformization(builder, to_units);
    } else {
      constructUnits(builder, to_units);
    }

    if (to_units.size() > 1) {
      auto group_op = dataflow::CreateGroupOp::create(
          builder, builder.getUnknownLoc(), builder.getIndexType(), to_units);
      if (dataflow::SyncSendOp::create(
              builder, builder.getUnknownLoc(), group_op,
              builder.getStringAttr(sync->name_),
              builder.getBoolAttr(wait_immediately_for_async_transfers)))
        return LogicalResult::success();
    } else if (to_units.size() == 1) {
      if (dataflow::SyncSendOp::create(
              builder, builder.getUnknownLoc(), to_units[0],
              builder.getStringAttr(sync->name_),
              builder.getBoolAttr(wait_immediately_for_async_transfers)))
        return LogicalResult::success();
    }
    return LogicalResult::failure();
  }

  // Creating sync recv operation
  // (sync->consumers_).find(comp_) != (sync->consumers_).end()
  if (sync->isReceive_ && !is_implicit_sync) {
    std::vector<mlir::Value> from_units;
    if (needsUniform()) {
      constructUnitsForUniformization(builder, from_units);
    } else {
      constructUnits(builder, from_units);
    }

    if (from_units.size() > 1) {
      auto group_op = dataflow::CreateGroupOp::create(
          builder, builder.getUnknownLoc(), builder.getIndexType(), from_units);
      if (dataflow::SyncRecvOp::create(builder, builder.getUnknownLoc(),
                                       group_op,
                                       builder.getStringAttr(sync->name_)))
        return LogicalResult::success();
    } else if (from_units.size() == 1) {
      if (dataflow::SyncRecvOp::create(builder, builder.getUnknownLoc(),
                                       from_units[0],
                                       builder.getStringAttr(sync->name_)))
        return LogicalResult::success();
    }
    return LogicalResult::failure();
  }

  // creating sync send/recv implicit sync operation
  if (is_implicit_sync) {
    DT_CHECK_MSG(
        dscGlobal().sysDef.coreArch >= RCUDD1A_ISA,
        "implicit sync is supported only on RCUDD1a and later version");
    return constructImplicitSyncOperation(builder);
  }

  // Neither sync send or recv operations created
  return LogicalResult::failure();
}

// ---- 77/110  constructElementsOfAgenDataTransfer  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:412  (68L)
LogicalResult e077_constructElementsOfAgenDataTransfer(
    OpBuilder &builder, SenComponents storage, bool is_load,
    Value start_address,
    const std::vector<dsc2::ScheduleNode::Size> &view_sizes,
    const std::vector<dsc2::TransferNode::SizeAndIndex>
        &unit_time_transfer_chunk_size,
    const std::vector<dsc2::TransferNode::SizeAndIndex>
        &unit_time_transfer_chunk_stride,
    const std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> &outer_loops,
    dataflow::GetLogicalMemoryViewOp &view, AffineMap &base_address_map,
    llvm::SmallVectorImpl<Value> &base_address_args, IntegerSet &transfer_set,
    AffineMap &transfer_order, bool is_constant_read_write) {
  // Get Storage unit op
  auto storage_unit_op =
      retrieveGetUnitOpInSameCore(builder, storage, core_id_, corelet_id_);

  // Construct logical memory view for the corresponding transfer operation
  if (failed(constructLogicalMemoryViewOp(builder, view_sizes, storage_unit_op,
                                          start_address, view,
                                          is_constant_read_write))) {
    return LogicalResult::failure();
  } else {
    DEBUG_WITH_TYPE(verbose_debug_, view->dump());
  }

  if (failed(constructBaseAddress(builder, view.getLayoutMap().getNumDims(),
                                  base_address_args, base_address_map,
                                  outer_loops, is_constant_read_write))) {
    return LogicalResult::failure();
  } else {
    DEBUG_WITH_TYPE(verbose_debug_, base_address_map.dump());
  }

  // Construct store set
  if (!is_constant_read_write) {
    if (failed(constructLoadOrStoreSet(builder, unit_time_transfer_chunk_size,
                                       unit_time_transfer_chunk_stride,
                                       view_sizes, is_load, transfer_set))) {
      return LogicalResult::failure();
    } else {
      DEBUG_WITH_TYPE(verbose_debug_, transfer_set.dump());
    }
  } else {
    int num_elements = 1;
    for (auto &entry : unit_time_transfer_chunk_size) {
      num_elements *= entry.sizeDim_.size_;
    }

    SmallVector<AffineExpr, 8> exprs, eq_exprs, ineq_exprs;
    auto id = getAffineDimExpr(0, builder.getContext());
    ineq_exprs.push_back(id);  // id >= 0
    ineq_exprs.push_back(num_elements - id - 1);

    exprs.insert(exprs.end(), ineq_exprs.begin(), ineq_exprs.end());
    exprs.insert(exprs.end(), eq_exprs.begin(), eq_exprs.end());

    SmallVector<bool, 16> eq_flags(eq_exprs.size() + ineq_exprs.size());
    std::fill(eq_flags.begin(), eq_flags.begin() + ineq_exprs.size(), false);
    std::fill(eq_flags.begin() + ineq_exprs.size(), eq_flags.end(), true);
    transfer_set = IntegerSet::get(1, 0, exprs, eq_flags);
    DEBUG_WITH_TYPE(verbose_debug_, transfer_set.dump());
  }

  // Construct store order
  transfer_order = AffineMap::getMultiDimIdentityMap(
      view.getLayoutMap().getNumDims(), view.getContext());
  return LogicalResult::success();
}

// ---- 78/110  constructElementsOfAgenCompositeDataTransfer  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:488  (94L)
LogicalResult e078_constructElementsOfAgenCompositeDataTransfer(
    OpBuilder &builder, SenComponents storage, bool is_load,
    Value start_address,
    const std::vector<dsc2::ScheduleNode::Size> &view_sizes,
    const std::vector<dsc2::TransferNode::SizeAndIndex>
        &unit_time_transfer_chunk_size,
    const std::vector<dsc2::TransferNode::SizeAndIndex>
        &unit_time_transfer_chunk_stride,
    const std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> &outer_loops,
    dataflow::GetLogicalMemoryViewOp &view, AffineMap &base_address_map,
    llvm::SmallVectorImpl<Value> &base_address_args, IntegerSet &transfer_set,
    AffineMap &transfer_order,
    const std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> &composite_loops,
    IntegerSet &time_set, AffineMap &time_order_map,
    AffineMap &time_address_map, bool is_constant_read_write) {
  // Get Storage unit op
  auto storage_unit_op =
      retrieveGetUnitOpInSameCore(builder, storage, core_id_, corelet_id_);

  // Construct logical memory view for the corresponding transfer operation
  if (failed(constructLogicalMemoryViewOp(builder, view_sizes, storage_unit_op,
                                          start_address, view,
                                          is_constant_read_write))) {
    return LogicalResult::failure();
  } else {
    DEBUG_WITH_TYPE(verbose_debug_, view->dump());
  }

  if (failed(constructBaseAddress(builder, view.getLayoutMap().getNumDims(),
                                  base_address_args, base_address_map,
                                  outer_loops, is_constant_read_write))) {
    return LogicalResult::failure();
  } else {
    DEBUG_WITH_TYPE(verbose_debug_, base_address_map.dump());
  }

  // Construct store set
  if (!is_constant_read_write) {
    if (failed(constructLoadOrStoreSet(builder, unit_time_transfer_chunk_size,
                                       unit_time_transfer_chunk_stride,
                                       view_sizes, is_load, transfer_set))) {
      return LogicalResult::failure();
    } else {
      DEBUG_WITH_TYPE(verbose_debug_, transfer_set.dump());
    }
  } else {
    int num_elements = 1;
    for (auto &entry : unit_time_transfer_chunk_size) {
      num_elements *= entry.sizeDim_.size_;
    }

    SmallVector<AffineExpr, 8> exprs, eq_exprs, ineq_exprs;
    auto id = getAffineDimExpr(0, builder.getContext());
    ineq_exprs.push_back(id);  // id >= 0
    ineq_exprs.push_back(num_elements - id - 1);

    exprs.insert(exprs.end(), ineq_exprs.begin(), ineq_exprs.end());
    exprs.insert(exprs.end(), eq_exprs.begin(), eq_exprs.end());

    SmallVector<bool, 16> eq_flags(eq_exprs.size() + ineq_exprs.size());
    std::fill(eq_flags.begin(), eq_flags.begin() + ineq_exprs.size(), false);
    std::fill(eq_flags.begin() + ineq_exprs.size(), eq_flags.end(), true);
    transfer_set = IntegerSet::get(1, 0, exprs, eq_flags);
    DEBUG_WITH_TYPE(verbose_debug_, transfer_set.dump());
  }

  // Construct store order
  transfer_order = AffineMap::getMultiDimIdentityMap(
      view.getLayoutMap().getNumDims(), view.getContext());

  llvm::SmallVector<unsigned> permutation_vector;
  for (int i = composite_loops.size() - 1; i >= 0; i--)
    permutation_vector.push_back(i);

  time_order_map =
      AffineMap::getPermutationMap(permutation_vector, builder.getContext());

  if (failed(constructTimeSet(builder, base_address_args, composite_loops,
                              time_set))) {
    return LogicalResult::failure();
  } else {
    DEBUG_WITH_TYPE(verbose_debug_, time_set.dump());
  }

  if (failed(constructTimeAddressMap(builder, time_set.getNumDims(),
                                     view.getLayoutMap().getNumDims(),
                                     time_address_map, composite_loops))) {
    return LogicalResult::failure();
  } else {
    DEBUG_WITH_TYPE(verbose_debug_, time_address_map.dump());
  }

  return LogicalResult::success();
}

// ---- 79/110  constructElementsOfAffineDataTransferViaAgenTransfer  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:590  (77L)
LogicalResult
e079_constructElementsOfAffineDataTransferViaAgenTransfer(
    OpBuilder &builder, SenComponents storage, bool is_load,
    Value start_address,
    const std::vector<dsc2::ScheduleNode::Size> &view_sizes,
    const std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> &outer_loops,
    dataflow::GetLogicalMemoryViewOp &view, AffineMap &base_address_map,
    llvm::SmallVectorImpl<Value> &base_address_args, IntegerSet &transfer_set,
    AffineMap &transfer_order, bool is_constant_read_write) {
  // Get Storage unit op
  auto storage_unit_op =
      retrieveGetUnitOpInSameCore(builder, storage, core_id_, corelet_id_);

  // Construct logical memory view for the corresponding transfer operation
  if (failed(constructLogicalMemoryViewOp(builder, view_sizes, storage_unit_op,
                                          start_address, view,
                                          is_constant_read_write))) {
    return LogicalResult::failure();
  } else {
    DEBUG_WITH_TYPE(verbose_debug_, view->dump());
  }

  if (failed(constructBaseAddress(builder, view.getLayoutMap().getNumDims(),
                                  base_address_args, base_address_map,
                                  outer_loops, is_constant_read_write,
                                  storage == LXLUSCALEREG))) {
    return LogicalResult::failure();
  } else {
    DEBUG_WITH_TYPE(verbose_debug_, base_address_map.dump());
  }

  if (!is_constant_read_write) {
    std::vector<dsc2::TransferNode::SizeAndIndex> unit_time_transfer;
    int min_dim = outer_loops.size() + 1;
    for (auto &outer_loop : outer_loops) {
      if (outer_loop.sizeIdx_ < min_dim) min_dim = outer_loop.sizeIdx_;
    }

    if (storage == LXLUSCALEREG) min_dim = 2;
    for (int i = 0; i < min_dim; i++) {
      dsc2::TransferNode::SizeAndIndex dim;
      dim.srcSizeIdx_ = dim.dstSizeIdx_ = i;
      dim.sizeDim_.dim_ = view_sizes[i].dim_;
      dim.sizeDim_.size_ = view_sizes[i].size_;
      unit_time_transfer.push_back(dim);
    }

    // Construct store set
    if (failed(constructLoadOrStoreSet(builder, unit_time_transfer, {},
                                       view_sizes, is_load, transfer_set))) {
      return LogicalResult::failure();
    } else {
      DEBUG_WITH_TYPE(verbose_debug_, transfer_set.dump());
    }
  } else {
    int num_elements = 64;  // TODO: Change it to reflect from result_type
    SmallVector<AffineExpr, 8> exprs, eq_exprs, ineq_exprs;
    auto id = getAffineDimExpr(0, builder.getContext());
    ineq_exprs.push_back(id);  // id >= 0
    ineq_exprs.push_back(num_elements - id - 1);

    exprs.insert(exprs.end(), ineq_exprs.begin(), ineq_exprs.end());
    exprs.insert(exprs.end(), eq_exprs.begin(), eq_exprs.end());

    SmallVector<bool, 16> eq_flags(eq_exprs.size() + ineq_exprs.size());
    std::fill(eq_flags.begin(), eq_flags.begin() + ineq_exprs.size(), false);
    std::fill(eq_flags.begin() + ineq_exprs.size(), eq_flags.end(), true);
    transfer_set = IntegerSet::get(1, 0, exprs, eq_flags);
    DEBUG_WITH_TYPE(verbose_debug_, transfer_set.dump());
  }

  // Construct store order
  transfer_order = AffineMap::getMultiDimIdentityMap(
      view.getLayoutMap().getNumDims(), view.getContext());

  return LogicalResult::success();
}

// ---- 80/110  constructStreamingOrDoubleBufferingLoad  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:851  (347L)
LogicalResult e080_constructStreamingOrDoubleBufferingLoad(
    OpBuilder &builder, SenComponents src_storage, Value to, Operation *load_op,
    int mode,
    const std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> &outer_loops,
    bool perform_composite_load,
    std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> &composite_loops) {
  // Create streaming address modifications
  const auto &lds_and_loop_offsets = transfer_->srcLdsAndLoopOffsets_;
  const dsc2::TransferNode::CoreletView *cl_view;
  if (this->uniformization_enabled_) {
    cl_view = &transfer_->coreletViews_.begin()->second;
  } else {
    cl_view = &transfer_->coreletViews_.at(corelet_id_);
  }

  const auto &view_sizes = cl_view->srcLoopsAndSize_;

  const std::vector<dsc2::ScheduleNode::Size> *view_sizes_core_specific;
  if (this->uniformization_enabled_) {
    view_sizes_core_specific = &view_sizes.sizesNoGaps_;
  } else {
    view_sizes_core_specific = &view_sizes.getSizesForCoreId(core_id_);
  }

  auto storage_unit_op =
      retrieveGetUnitOpInSameCore(builder, src_storage, core_id_, corelet_id_);

  int iter_arg_index = -1;
  auto &transfers_with_loop = (*dsc_all_parent_loops_to_buffers_switch_map_)
      [lds_and_loop_offsets.bufferSwitchPosition_];
  for (int i = 0; i < transfers_with_loop.size(); i++) {
    if (transfers_with_loop[i] == transfer_) {
      iter_arg_index = i;
    }
  }

  Value start_address;
  Operation *buffer_switch_loop_terminator = nullptr;
  if (iter_arg_index == -1) {
    return LogicalResult::failure();
  } else {
    auto *loop = (*dsc_loops_to_mlir_loops_map_)[lds_and_loop_offsets
                                                     .bufferSwitchPosition_]
                     .back();
    if (auto affine_for = llvm::dyn_cast<mlir::affine::AffineForOp>(loop)) {
      start_address = affine_for.getRegionIterArgs()[iter_arg_index];
      buffer_switch_loop_terminator = affine_for.getBody(0)->getTerminator();
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop)) {
      start_address = scf_for.getRegionIterArgs()[iter_arg_index];
      buffer_switch_loop_terminator = scf_for.getBody(0)->getTerminator();
    } else {
      return LogicalResult::failure();
    }
  }

  // Construct logical memory view for the corresponding store operation
  dataflow::GetLogicalMemoryViewOp view;
  if (failed(constructLogicalMemoryViewOp(builder, *view_sizes_core_specific,
                                          storage_unit_op, start_address,
                                          view))) {
    return LogicalResult::failure();
  } else {
    DEBUG_WITH_TYPE(verbose_debug_, view->dump());
  }

  AffineMap base_address_map;
  llvm::SmallVector<Value, 8> base_address_args;
  if (failed(constructBaseAddress(builder, view.getLayoutMap().getNumDims(),
                                  base_address_args, base_address_map,
                                  outer_loops))) {
    return LogicalResult::failure();
  } else {
    DEBUG_WITH_TYPE(verbose_debug_, base_address_map.dump());
  }

  // Construct store set
  IntegerSet load_set;
  if (failed(constructLoadOrStoreSet(
          builder, transfer_->unitTimeTransferChunkSize_,
          transfer_->unitTimeTransferChunkStride_, *view_sizes_core_specific,
          true, load_set, transfer_->unitTimeTransferNumChunks_))) {
    emitError("Unable to construct load set");
    return LogicalResult::failure();
  } else {
    DEBUG_WITH_TYPE(verbose_debug_, load_set.dump());
  }

  // Construct store order
  AffineMap load_order_map = AffineMap::getMultiDimIdentityMap(
      view.getLayoutMap().getNumDims(), view.getContext());

  if (!perform_composite_load) {
    // Create agen.load operation
    auto load_op_wo_repl = agen::VectorLoadOp::create(
        builder, builder.getUnknownLoc(), result_type, view.getResult(),
        builder.getStringAttr(transfer_->name_), base_address_map,
        base_address_args, load_set, load_order_map);
    DEBUG_WITH_TYPE(verbose_debug_, load_op_wo_repl->dump());

    bool use_latch = false;
    for (int dst_idx = 0; dst_idx < transfer_->dstVias_.size(); dst_idx++) {
      auto &dst_via = transfer_->dstVias_.at(dst_idx);
      if (dst_via.loc_.unit_ == SenComponents::LXLUVALUE) {
        DT_CHECK(dst_via.loc_.storage_ == SenComponents::LATCH);
        int latch_id =
            transfer_->dstLdsAndLoopOffsets_.at(dst_idx).latchDataId_;
        DT_CHECK_MSG(latch_id != -1, "latch id cannot be negative");
        addToLatchMap(latch_id, load_op_wo_repl->getResult(0));
        use_latch = true;
      }
    }

    if (transfer_->replicationFactor_ != 1) {
      int total_elements = dataflow::utils::getNumElements(result_type) *
                           transfer_->replicationFactor_;
      auto new_vec_type = mlir::dataflow::utils::constructVectorType(
          dataflow::utils::getElementType(result_type), total_elements);
      if (comp_ == SenComponents::LXLU && transfer_->replicationFactor_ > 1) {
        if (auto shuffle_op = construct2B16BLoadShuffle(
                builder, load_op_wo_repl.getResult(), result_type,
                transfer_->replicationFactor_)) {
          load_op = shuffle_op;
        } else {
          emitError("Unsupported load type.");
          return LogicalResult::failure();
        }
      } else {
        // TODO: will have to be deprecated.
        auto dim = getAffineDimExpr(0, builder.getContext());
        auto selection_map = AffineMap::get(
            1, 0, dim % transfer_->replicationFactor_, builder.getContext());
        load_op = mlir::vectorchain::SelectOp::create(
            builder, builder.getUnknownLoc(), new_vec_type,
            load_op_wo_repl.getResult(), selection_map);
      }
    } else {
      load_op = load_op_wo_repl;
    }

    Value load_op_result;
    if (transfer_->rotateNumElements_ > 0) {
      DT_CHECK_MSG(comp_ == LXLU, "Rotation is allowed only in LXLU");
      auto rot_element = mlir::arith::ConstantIndexOp::create(
          builder, builder.getUnknownLoc(), transfer_->rotateNumElements_);
      auto rot_op = vectorchain::RotateOp::create(
          builder, rot_element.getLoc(), result_type, load_op->getResult(0),
          rot_element, builder.getStringAttr(transfer_->name_));
      load_op_result = rot_op.getResult();
    } else {
      load_op_result = load_op->getResult(0);
    }

    // Convert data if src precision and dst precision don't match.
    // example1: src_result_type (lxlu): <64xfp16>, dst_result_type (pe):
    // <64xfp32> for the above one, don't generate cast for lxlu since its
    // already a stick worth example2: src_result_type (pe): <64xfp32>,
    // dst_result_type (sfp): <64xfp16> for the above one, generate cast for pe
    // since its not fitting into a stick.
    auto stick_size = 1024;  // bits
    auto input_size = mlir::dataflow::utils::getDimSize(result_type, 0) *
                      dataflow::utils::getElementTypeBitWidth(result_type);
    if (dst_result_type != result_type && input_size != stick_size) {
      auto tmp_data = load_op_result;
      if (SNComputeLowering::constructPrecisionConversionOperation(
              builder, tmp_data, dst_prec_, load_op_result, comp_)
              .failed()) {
        emitError(
            "Unable to construct precision conversion in a transfer operation");
      }
    }

    if (!use_latch)
      auto send_op = dataflow::SendOp::create(
          builder, builder.getUnknownLoc(), to, load_op_result, /*dir*/ nullptr,
          builder.getStringAttr(transfer_->name_));
  } else {
    // Create agen.composite_load operation
    //    AffineMap time_order_map = AffineMap::getMultiDimIdentityMap(
    //        view_sizes.compositeLoops_.size(),
    //        builder.getContext());

    llvm::SmallVector<unsigned> permutation_vector;
    for (int i = composite_loops.size() - 1; i >= 0; i--)
      permutation_vector.push_back(i);

    AffineMap time_order_map =
        AffineMap::getPermutationMap(permutation_vector, builder.getContext());

    IntegerSet time_set;
    if (failed(constructTimeSet(builder, base_address_args, composite_loops,
                                time_set))) {
      return LogicalResult::failure();
    } else {
      DEBUG_WITH_TYPE(verbose_debug_, time_set.dump());
    }

    AffineMap time_address_map;
    if (failed(constructTimeAddressMap(builder, time_set.getNumDims(),
                                       view.getLayoutMap().getNumDims(),
                                       time_address_map, composite_loops))) {
      return LogicalResult::failure();
    } else {
      DEBUG_WITH_TYPE(verbose_debug_, time_address_map.dump());
    }

    auto map_operands =
        ArrayRef<Value>(base_address_args).drop_back(time_set.getNumSymbols());
    auto time_symbols =
        ArrayRef<Value>(base_address_args).take_back(time_set.getNumSymbols());
    auto composite_load = agen::CompositeLoadOp::create(
        builder, builder.getUnknownLoc(), view.getResult(),
        builder.getStringAttr(transfer_->name_), base_address_map, map_operands,
        result_type, load_set, load_order_map, time_symbols, time_set,
        time_order_map, time_address_map);
    OpBuilder local_builder(composite_load);
    local_builder.setInsertionPointToStart(composite_load.getBody());

    Value load;
    if (transfer_->replicationFactor_ != 1) {
      int total_elements = dataflow::utils::getNumElements(result_type) *
                           transfer_->replicationFactor_;
      auto new_vec_type = mlir::dataflow::utils::constructVectorType(
          dataflow::utils::getElementType(result_type), total_elements);
      if (comp_ == SenComponents::LXLU && transfer_->replicationFactor_ > 1) {
        if (auto shuffle_op = construct2B16BLoadShuffle(
                local_builder, composite_load.getLoadInductionVar(),
                result_type, transfer_->replicationFactor_)) {
          load_op = shuffle_op;
        } else {
          emitError("Unsupported load type.");
          return LogicalResult::failure();
        }
      } else {
        auto dim = getAffineDimExpr(0, local_builder.getContext());
        auto selection_map =
            AffineMap::get(1, 0, dim % transfer_->replicationFactor_,
                           local_builder.getContext());
        load_op = mlir::vectorchain::SelectOp::create(
            local_builder, local_builder.getUnknownLoc(), new_vec_type,
            composite_load.getLoadInductionVar(), selection_map);
      }
      load = load_op->getResult(0);
    } else {
      load = composite_load.getLoadInductionVar();
    }

    if (transfer_->rotateNumElements_ > 0) {
      DT_CHECK_MSG(comp_ == LXLU, "Rotation is allowed only in LXLU");
      auto rot_element = mlir::arith::ConstantIndexOp::create(
          local_builder, local_builder.getUnknownLoc(),
          transfer_->rotateNumElements_);
      auto rot_op = vectorchain::RotateOp::create(
          local_builder, rot_element.getLoc(), result_type, load, rot_element,
          local_builder.getStringAttr(transfer_->name_));
      load = rot_op.getResult();
    }

    // Convert data if src precision and dst precision don't match.
    // example1: src_result_type (lxlu): <64xfp16>, dst_result_type (pe):
    // <64xfp32> for the above one, don't generate cast for lxlu since its
    // already a stick worth example2: src_result_type (pe): <64xfp32>,
    // dst_result_type (sfp): <64xfp16> for the above one, generate cast for pe
    // since its not fitting into a stick.
    auto stick_size = 1024;  // bits
    auto input_size = mlir::dataflow::utils::getDimSize(result_type, 0) *
                      dataflow::utils::getElementTypeBitWidth(result_type);
    if (dst_result_type != result_type && input_size != stick_size) {
      auto tmp_data = load;
      if (SNComputeLowering::constructPrecisionConversionOperation(
              builder, tmp_data, dst_prec_, load, comp_)
              .failed()) {
        emitError(
            "Unable to construct precision conversion in a transfer operation");
      }
    }

    auto send_op = dataflow::SendOp::create(
        local_builder, local_builder.getUnknownLoc(), to, load, /*dir*/ nullptr,
        local_builder.getStringAttr(transfer_->name_));
    DEBUG_WITH_TYPE(verbose_debug_, composite_load->dump());
  }

  OpBuilder factor_builder(*this->unit_op_);
  mlir::Type factor_element_type =
      dataflow::utils::getElementType(result_type);
  auto factor = getAddressGranularityMultiplyFactor(comp_, src_storage,
                                                    factor_element_type);

  // Update the yield argument with increment to reflect streaming behavior.
  OpBuilder local_builder(buffer_switch_loop_terminator);

  // streaming mode
  if (mode == 2) {
    mlir::Value buffer_increment;
    if (this->uniformization_enabled_) {
      buffer_increment = constructUniformizedAddress(
          local_builder, transfer_->srcLdsAndLoopOffsets_.bufferAddrOffset_,
          factor);
    } else {
      int buff_addr_offset = int(
          transfer_->srcLdsAndLoopOffsets_.bufferAddrOffset_.at(core_id_).at(
              corelet_id_) *
          factor);
      auto const_op = mlir::arith::ConstantIndexOp::create(
          local_builder, local_builder.getUnknownLoc(), buff_addr_offset);
      buffer_increment = const_op.getResult();
    }

    auto buffer_index = mlir::arith::AddIOp::create(
        local_builder, local_builder.getUnknownLoc(),
        local_builder.getIndexType(),
        buffer_switch_loop_terminator->getOperand(iter_arg_index),
        buffer_increment);
    buffer_switch_loop_terminator->setOperand(iter_arg_index,
                                              buffer_index.getResult());
  } else if (mode == 1) {  // double buffering
    mlir::Value buffer_increment;
    if (needsUniform()) {
      buffer_increment = constructUniformizedFoldedDoubleBufferToggling(
          local_builder, transfer_->srcLdsAndLoopOffsets_.bufferAddrOffset_,
          transfer_->srcLdsAndLoopOffsets_.startAddr_, factor);
    } else {
      // expecting constant value across sdsc folds
      int buff_addr_offset = int(
          (transfer_->srcLdsAndLoopOffsets_.bufferAddrOffset_.at(core_id_).at(
               corelet_id_) +
           2 * FoldInfraUtils::getSingleDataStrict(
                   transfer_->srcLdsAndLoopOffsets_.startAddr_,
                   {{0, core_id_}, {1, corelet_id_}})) *
          factor);
      auto const_op = mlir::arith::ConstantIndexOp::create(
          local_builder, local_builder.getUnknownLoc(), buff_addr_offset);
      buffer_increment = const_op.getResult();
    }

    auto buffer_index = mlir::arith::SubIOp::create(
        local_builder, local_builder.getUnknownLoc(),
        local_builder.getIndexType(), buffer_increment,
        buffer_switch_loop_terminator->getOperand(iter_arg_index));
    buffer_switch_loop_terminator->setOperand(iter_arg_index,
                                              buffer_index.getResult());
  } else {
    llvm_unreachable("Unknown buffering mode");
  }

  return LogicalResult::success();
}

// ---- 81/110  GenerateReceiveAndSendFromDataTransferNode  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1395  (66L)
LogicalResult e081_GenerateReceiveAndSendFromDataTransferNode(
    OpBuilder &builder, Value from, Value to, int count) {
  OpBuilder loop_builder(builder);

  const dsc2::TransferNode::CoreletView *view;
  if (this->uniformization_enabled_) {
    view = &transfer_->coreletViews_.begin()->second;
  } else {
    view = &transfer_->coreletViews_.at(corelet_id_);
  }

  const auto &view_sizes = view->srcLoopsAndSize_.sizesNoGaps_.empty()
                               ? view->dstLoopsAndSizes_.front()
                               : view->srcLoopsAndSize_;
  const std::vector<dsc2::ScheduleNode::Size> *view_sizes_core_specific;
  if (this->uniformization_enabled_) {
    view_sizes_core_specific = &view_sizes.sizesNoGaps_;
  } else {
    view_sizes_core_specific = &view_sizes.getSizesForCoreId(core_id_);
  }

  std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> implicit_loops;
  std::unordered_map<PrimaryDimTypes, int> ctgs_transfer_sizes;
  if (failed(constructImplicitLoopsForContiguousTransfer(
          loop_builder, *view_sizes_core_specific,
          transfer_->unitTimeTransferChunkSize_, implicit_loops,
          ctgs_transfer_sizes))) {
    emitError("Unable to construct implicit loops for contiguous transfer");
    return LogicalResult::failure();
  }

  if (!implicit_loops.empty()) {
    DT_CHECK(
        "check" &&
        (*dsc_loops_to_mlir_loops_map_)[implicit_loops.front().loop_].size() ==
            1);
    auto *outer_loop =
        (*dsc_loops_to_mlir_loops_map_)[implicit_loops.front().loop_].front();
    builder.setInsertionPointAfter(outer_loop);

    auto *inner_loop =
        (*dsc_loops_to_mlir_loops_map_)[implicit_loops.front().loop_].front();
    if (auto affine_for =
            llvm::dyn_cast<mlir::affine::AffineForOp>(inner_loop)) {
      loop_builder.setInsertionPointToStart(affine_for.getBody());
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(inner_loop)) {
      loop_builder.setInsertionPointToStart(scf_for.getBody());
    }
  } else {
    auto loop = mlir::affine::AffineForOp::create(
        loop_builder, builder.getUnknownLoc(), 0, count, 1);
    dataflow::setDbgName(
        loop, "SingleImplicitLoopForTransfer(" + transfer_->name_ + ")");
    loop_builder.setInsertionPointToStart(loop.getBody());
    builder.setInsertionPointAfter(loop);
  }

  auto receive_op = dataflow::ReceiveOp::create(
      loop_builder, loop_builder.getUnknownLoc(), result_type, from,
      loop_builder.getStringAttr(transfer_->name_));
  auto send_op = dataflow::SendOp::create(
      loop_builder, loop_builder.getUnknownLoc(), to, receive_op.getResult(),
      /*dir*/ nullptr, loop_builder.getStringAttr(transfer_->name_));

  return LogicalResult::success();
}

// ---- 82/110  ConstructSAMVOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1957  (96L)
LogicalResult e082_ConstructSAMVOperation(
    OpBuilder &loop_builder,
    std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> &outer_loops) {
  // create conditionals
  mlir::scf::IfOp if_op, samv_if_op;
  if (SNControlFlowLowering::constructConditionalsForSAMV(
          loop_builder, this->dsc_loops_to_mlir_loops_map_, outer_loops, if_op)
          .failed()) {
    emitError("Unable to construct conditionals for SAMV operation");
    return LogicalResult::failure();
  }

  // create samv operation in the then branch, and reset in else branch
  OpBuilder samv_builder(if_op);
  samv_builder.setInsertionPointAfter(if_op);
  mlir::Value val_false = mlir::arith::ConstantIndexOp::create(
      samv_builder, samv_builder.getUnknownLoc(), 0);

  bool yield_samv_results = false;
  if (yield_samv_results) {
    samv_if_op = mlir::scf::IfOp::create(
        samv_builder, if_op.getLoc(), result_type, if_op->getResult(0), true);
  } else {
    samv_if_op = mlir::scf::IfOp::create(samv_builder, if_op.getLoc(),
                                         if_op->getResult(0), true);
  }

  // create samv mask.
  std::vector<int> unmasked_offsets, masked_offsets;
  int entries_per_slice = 8;  // (16B/FP16)
  int num_slices = 8;         // 2B
  int mask_all =
      false;  // dsc_->computeOp_.at(0).opConsts.at("samv-maskall")[0];
  int slice_idx_xsl = ceil(((float)this->dsc_->computeOp_.at(0).opConsts.at(
                                "samv-numvalidentry")[0] /
                            entries_per_slice)) -
                      1;
  int wsllen = dsc_->computeOp_.at(0).opConsts.at("samv-wsllen")[0];
  DT_CHECK_MSG(wsllen == 0, "WSL length should be zero");

  int num_valid_entries =
      dsc_->computeOp_.at(0).opConsts.at("samv-numvalidentry")[0] %
      entries_per_slice;

  // constructing slice mask
  std::string slice_mask_map;
  if (mask_all) {
    slice_mask_map = "(1)(1)(1)(1)(1)(1)(1)(1)";
  } else {
    for (int i = 0; i < entries_per_slice; i++) {
      if (i < slice_idx_xsl)
        slice_mask_map += "(A)";
      else if (i == slice_idx_xsl)
        slice_mask_map += "(A|B)";
      else if (i > slice_idx_xsl)
        slice_mask_map += "(1)";
    }

    // unmask-0, mask-1 for MaskA (WSL masking)
    unmasked_offsets.push_back(wsllen);
    masked_offsets.push_back(1);

    // unmask-valid entries, mask-(entries-num valid entries) (XSL masking)
    unmasked_offsets.push_back(num_valid_entries);
    masked_offsets.push_back(entries_per_slice - num_valid_entries);
  }

  samv_builder = samv_if_op.getThenBodyBuilder();
  auto samv_op = agen::SetTransferMaskStateOp::create(
      samv_builder, samv_builder.getUnknownLoc(), result_type, val_false,
      samv_builder.getStringAttr(transfer_->name_),
      samv_builder.getI32IntegerAttr(num_slices),
      samv_builder.getStringAttr(slice_mask_map),
      samv_builder.getI32ArrayAttr(unmasked_offsets),
      samv_builder.getI32ArrayAttr(masked_offsets));

  if (yield_samv_results) {
    mlir::scf::YieldOp::create(samv_builder, samv_op.getLoc(),
                               samv_op.getResult());
  }

  // create reset mask operation
  samv_builder = samv_if_op.getElseBodyBuilder();
  auto samv_reset = agen::SetTransferMaskStateOp::create(
      samv_builder, samv_builder.getUnknownLoc(), result_type, val_false,
      samv_builder.getStringAttr(transfer_->name_),
      samv_builder.getI32IntegerAttr(num_slices),
      samv_builder.getStringAttr("(0)(0)(0)(0)(0)(0)(0)(0)"), nullptr, nullptr);

  if (yield_samv_results) {
    mlir::scf::YieldOp::create(samv_builder, samv_reset.getLoc(),
                               samv_reset.getResult());
  }

  return LogicalResult::success();
}


// ================================================================================================
// LEVEL 3
// ================================================================================================

// ---- 83/110  buildUniformizedNeighborUnits  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:380  (239L)
void e083_buildUniformizedNeighborUnits(
    SenComponents comp, OpBuilder &builder,
    const DesignSpaceConfigGlobal &dsc_global, mlir::Value program_unit_op_iterator,
    SmallVectorImpl<mlir::Value> &get_unit_ops, std::vector<int> &cores,
    std::vector<int> &corelets) {
  auto unit_op = program_unit_op_iterator;
  if (comp == PTROW0) {
    component_to_handler_[PTROW1] = component_to_handler_[PTSOUTH] =
        createUniformizedGetUnitOp(builder, PTROW1, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[SFP] = component_to_handler_[PTNORTH] =
        createUniformizedGetUnitOp(builder, SFP, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[LXLU] = createUniformizedGetUnitOp(
        builder, LXLU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[PTWEST] = createUniformizedGetUnitOp(
        builder, L0LU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW1) {
    component_to_handler_[PTROW2] = component_to_handler_[PTSOUTH] =
        createUniformizedGetUnitOp(builder, PTROW2, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTROW0] = component_to_handler_[PTNORTH] =
        createUniformizedGetUnitOp(builder, PTROW0, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTWEST] = createUniformizedGetUnitOp(
        builder, L0LU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW2) {
    component_to_handler_[PTROW3] = component_to_handler_[PTSOUTH] =
        createUniformizedGetUnitOp(builder, PTROW3, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTROW1] = component_to_handler_[PTNORTH] =
        createUniformizedGetUnitOp(builder, PTROW1, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTWEST] = createUniformizedGetUnitOp(
        builder, L0LU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW3) {
    if (dsc_global.sysDef.coreArch <= RCUDD1A_ISA)
      component_to_handler_[PTROW4] = component_to_handler_[PTSOUTH] =
          createUniformizedGetUnitOp(builder, PTROW4, program_unit_op_iterator,
                                     get_unit_ops, cores, corelets);
    else
      component_to_handler_[PE] = component_to_handler_[PTSOUTH] =
          createUniformizedGetUnitOp(builder, PE, program_unit_op_iterator,
                                     get_unit_ops, cores, corelets);
    component_to_handler_[PTROW2] = component_to_handler_[PTNORTH] =
        createUniformizedGetUnitOp(builder, PTROW2, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTWEST] = createUniformizedGetUnitOp(
        builder, L0LU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW4) {
    component_to_handler_[PTROW5] = component_to_handler_[PTSOUTH] =
        createUniformizedGetUnitOp(builder, PTROW5, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTROW3] = component_to_handler_[PTNORTH] =
        createUniformizedGetUnitOp(builder, PTROW3, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTWEST] = createUniformizedGetUnitOp(
        builder, L0LU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW5) {
    component_to_handler_[PTROW6] = component_to_handler_[PTSOUTH] =
        createUniformizedGetUnitOp(builder, PTROW6, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTROW4] = component_to_handler_[PTNORTH] =
        createUniformizedGetUnitOp(builder, PTROW4, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTWEST] = createUniformizedGetUnitOp(
        builder, L0LU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW6) {
    component_to_handler_[PTROW7] = component_to_handler_[PTSOUTH] =
        createUniformizedGetUnitOp(builder, PTROW7, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTROW5] = component_to_handler_[PTNORTH] =
        createUniformizedGetUnitOp(builder, PTROW5, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTWEST] = createUniformizedGetUnitOp(
        builder, L0LU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == PTROW7) {
    component_to_handler_[PE] = component_to_handler_[PTSOUTH] =
        createUniformizedGetUnitOp(builder, PE, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTROW6] = component_to_handler_[PTNORTH] =
        createUniformizedGetUnitOp(builder, PTROW6, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PTWEST] = createUniformizedGetUnitOp(
        builder, L0LU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PT_LRFREG, unit_op);
    component_to_handler_[PTXRF] =
        createGetLocalUnitOp(builder, PTXRF, unit_op);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == L0LUROW0) {
    component_to_handler_[L0SU] = createUniformizedGetUnitOp(
        builder, L0SU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[PTROW0] =
        createUniformizedGetUnitOp(builder, PTROW0, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[L0] = createUniformizedGetUnitOp(
        builder, L0, program_unit_op_iterator, get_unit_ops, cores, corelets);
  } else if (comp == L0SU) {
    component_to_handler_[SFP] = createUniformizedGetUnitOp(
        builder, SFP, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[L0LUROW0] =
        createUniformizedGetUnitOp(builder, L0LUROW0, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[L0] = createUniformizedGetUnitOp(
        builder, L0, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LXLU] = createUniformizedGetUnitOp(
        builder, LXLU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    if (dsc_global_.sysDef.coreArch >= SEN1P5_ISA)
      component_to_handler_[L0_SCALE] =
          createGetLocalUnitOp(builder, L0_SCALE, unit_op);
  } else if (comp == LXLU) {
    component_to_handler_[LXSU] = createUniformizedGetUnitOp(
        builder, LXSU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[SFP] = createUniformizedGetUnitOp(
        builder, SFP, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LX] = createUniformizedGetUnitOp(
        builder, LX, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[PE] = createUniformizedGetUnitOp(
        builder, PE, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[L3LU] = createUniformizedGetUnitOp(
        builder, L3LU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[L3SU] = createUniformizedGetUnitOp(
        builder, L3SU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[PTROW0] =
        createUniformizedGetUnitOp(builder, PTROW0, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[L0SU] = createUniformizedGetUnitOp(
        builder, L0SU, program_unit_op_iterator, get_unit_ops, cores, corelets);
  } else if (comp == LXSU) {
    component_to_handler_[LXLU] = createUniformizedGetUnitOp(
        builder, LXLU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[SFP] = createUniformizedGetUnitOp(
        builder, SFP, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LX] = createUniformizedGetUnitOp(
        builder, LX, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[PE] = createUniformizedGetUnitOp(
        builder, PE, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[L3LU] = createUniformizedGetUnitOp(
        builder, L3LU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[L3SU] = createUniformizedGetUnitOp(
        builder, L3SU, program_unit_op_iterator, get_unit_ops, cores, corelets);
  } else if (comp == PE) {
    component_to_handler_[LXLU] = createUniformizedGetUnitOp(
        builder, LXLU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LXSU] = createUniformizedGetUnitOp(
        builder, LXSU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    if (dsc_global.sysDef.coreArch <= RCUDD1A_ISA)
      component_to_handler_[PTROW7] =
          createUniformizedGetUnitOp(builder, PTROW7, program_unit_op_iterator,
                                     get_unit_ops, cores, corelets);
    else
      component_to_handler_[PTROW3] =
          createUniformizedGetUnitOp(builder, PTROW3, program_unit_op_iterator,
                                     get_unit_ops, cores, corelets);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, PE_LRFREG, unit_op);
    component_to_handler_[SFP] = createUniformizedGetUnitOp(
        builder, SFP, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[CONSTANT] =
        createUniformizedGetUnitOp(builder, CONSTANT, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[PESTATE] =
        createUniformizedGetUnitOp(builder, PESTATE, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
  } else if (comp == SFP) {
    component_to_handler_[LXLU] = createUniformizedGetUnitOp(
        builder, LXLU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[LXSU] = createUniformizedGetUnitOp(
        builder, LXSU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[L0SU] = createUniformizedGetUnitOp(
        builder, L0SU, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[PTROW0] =
        createUniformizedGetUnitOp(builder, PTROW0, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[LRFREG] =
        createGetLocalUnitOp(builder, SFP_LRFREG, unit_op);
    component_to_handler_[PE] = createUniformizedGetUnitOp(
        builder, PE, program_unit_op_iterator, get_unit_ops, cores, corelets);
    component_to_handler_[CONSTANT] =
        createUniformizedGetUnitOp(builder, CONSTANT, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
    component_to_handler_[SFPSTATE] =
        createUniformizedGetUnitOp(builder, SFPSTATE, program_unit_op_iterator,
                                   get_unit_ops, cores, corelets);
  } else {
  }
}

// ---- 84/110  initializeUnit  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:624  (33L)
dataflow::ProgramUnitOp e084_initializeUnit(
    int core_id, int corelet_id, SenComponents comp,
    const DesignSpaceConfigGlobal &dsc_global) {
  OpBuilder builder(dataflow_func_op_);
  builder.setInsertionPointToEnd(&dataflow_func_op_.back());

  std::string type = EnumsConversion::senComponentsToString.at(comp);
  std::string name = type + "-CL" + std::to_string(corelet_id);

  // Create GetUnit op
  auto get_unit_type = builder.getIndexType();
  auto get_unit_op = mlir::dataflow::GetUnitOp::create(
      builder, builder.getUnknownLoc(), get_unit_type,
      builder.getStringAttr(name), builder.getStringAttr(type));
  get_unit_op->setAttr("core", builder.getI32IntegerAttr(core_id));
  get_unit_op->setAttr("corelet", builder.getI32IntegerAttr(corelet_id));

  DT_CHECK(num_folds_ == 1 &&
           "this initalization is for only non-uniform and non-fold mode");
  get_unit_op->setAttr("num_folds", builder.getI32IntegerAttr(num_folds_));

  // Create ProgramUnit op
  auto unit_op = mlir::dataflow::ProgramUnitOp::create(
      builder, get_unit_op.getLoc(), get_unit_op.getResult(0));

  this->component_to_handler_.clear();
  this->component_to_handler_[comp] = get_unit_op.getResult(0);
  OpBuilder unit_builder(unit_op);
  unit_builder.setInsertionPointToStart(&(unit_op.getRegion().front()));
  this->buildNeighborUnits(core_id, corelet_id, comp, unit_builder,
                           dsc_global);
  return unit_op;
}

// ---- 85/110  initializeUnit  --  dsc-based-utils/DSC2ToDataflowIR/DataflowIRConstructionUtils.hpp:171  (22L)
static dataflow::ProgramUnitOp e085_initializeUnit(OpBuilder &builder, int core_id,
                                              int corelet_id,
                                              SenComponents comp) {
  std::string type = EnumsConversion::senComponentsToString.at(comp);
  std::string name = type + "-CL" + std::to_string(corelet_id);

  // Create GetUnit op
  auto get_unit_type = builder.getIndexType();
  auto get_unit_op = mlir::dataflow::GetUnitOp::create(
      builder, builder.getUnknownLoc(), get_unit_type,
      builder.getStringAttr(name), builder.getStringAttr(type));
  get_unit_op->setAttr("core", builder.getI32IntegerAttr(core_id));
  get_unit_op->setAttr("corelet", builder.getI32IntegerAttr(corelet_id));

  // This initalization is for only non-uniform and non-fold mode.
  get_unit_op->setAttr("num_folds", builder.getI32IntegerAttr(1));

  std::vector<mlir::Value> unit_list = {get_unit_op.getResult(0)};
  // Create ProgramUnit op
  return mlir::dataflow::ProgramUnitOp::create(builder, get_unit_op.getLoc(),
                                               unit_list);
}

// ---- 86/110  constructComputeInputOperandAndAddToList  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:456  (320L)
LogicalResult e086_constructComputeInputOperandAndAddToList(
    OpBuilder &builder, Location &loc, Type &element_type, Type &result_type,
    const dsc2::ComputeNode &compute_op, llvm::SmallVectorImpl<Value> &inputs,
    int index) {
  Value input_data;
  if (compute_op.inputs_[index] == ZERO) {
    if (auto result_vtype = dyn_cast<VectorType>(result_type)) {
      auto zero = mlir::arith::ConstantOp::create(
          builder, loc, result_vtype, builder.getZeroAttr(result_vtype));
      input_data = zero.getResult();
    } else if (auto result_custom_vtype =
                   dyn_cast<dataflow::CustomVectorType>(result_type)) {
      DT_CHECK(isa<dataflow::CustomMXFloatType>(element_type));
      input_data = constructSingleValCustomVector(
          builder, loc, compute_op, /*val*/ 0, result_custom_vtype);
    } else {
      DT_ERROR("Result type is expected to be a vector type");
    }
  } else if (compute_op.inputs_[index] == ONE) {
    TypedAttr one_attr;

    if (auto result_vtype = dyn_cast<VectorType>(result_type)) {
      if (element_type.isIntOrIndex()) {
        one_attr = DenseElementsAttr::get(
            result_vtype, builder.getIntegerAttr(element_type, 1));
      } else {
        one_attr = DenseElementsAttr::get(
            result_vtype, builder.getFloatAttr(element_type, 1.0));
      }
      auto one =
          mlir::arith::ConstantOp::create(builder, loc, result_type, one_attr);
      input_data = one.getResult();
    } else if (auto result_custom_vtype =
                   dyn_cast<dataflow::CustomVectorType>(result_type)) {
      DT_CHECK(isa<dataflow::CustomMXFloatType>(element_type));
      input_data = constructSingleValCustomVector(
          builder, loc, compute_op, /*val*/ 1, result_custom_vtype);
    } else {
      DT_ERROR("Result type is expected to be a vector type");
    }
  } else if (is_any_of(compute_op.inputs_[index], PTWEST, L0LU, L0LUROW0)) {
    auto west =
        retrieveGetUnitOpInSameCore(builder, PTWEST, core_id_, corelet_id_);
    auto data =
        dataflow::ReceiveOp::create(builder, loc, result_type, west,
                                    builder.getStringAttr(compute_op.name_));
    AffineMap selection_map;
    Type splat_data_type;
    if (failed(getSelectionMapForMACOperandFromL0(
            builder, compute_op.type_, compute_op.dataFormat_, result_type,
            selection_map, splat_data_type))) {
      return LogicalResult::failure();
    } else {
      auto splat_data = mlir::vectorchain::SelectOp::create(
          builder, loc, splat_data_type, data.getData(), selection_map);
      input_data = splat_data.getResult();
    }
  } else if (compute_op.inputs_[index] == PTNORTH) {
    auto north =
        retrieveGetUnitOpInSameCore(builder, PTNORTH, core_id_, corelet_id_);
    auto data =
        dataflow::ReceiveOp::create(builder, loc, result_type, north,
                                    builder.getStringAttr(compute_op.name_));
    input_data = data.getResult();
  } else if (is_any_of(compute_op.inputs_[index], LRFREG, PTXRF, PTARF, NFWD0,
                       NFWD2, PELRF, SFPLRF, PESTATE, SFPSTATE, LXLUSCALEREG)) {
    dataflow::GetLogicalMemoryViewOp view;
    AffineMap base_address_map;
    llvm::SmallVector<Value, 8> base_address_args;
    IntegerSet transfer_set;
    AffineMap transfer_order;

    SenComponents storage = compute_op.inputs_[index];
    const auto *data_info = &compute_op.inputsLdsAndLoopOffsets_[index];

    if (compute_op.inputs_[index] == NFWD0 ||
        compute_op.inputs_[index] == NFWD2) {
      storage = compute_op.outputs_.front();
      data_info = &compute_op.outputsLdsAndLoopOffsets_.front();
    }

    double factor = getAddressGranularityMultiplyFactor(
        comp_, storage, dataflow::utils::getElementType(result_type));

    mlir::Value address;
    const dsc2::ScheduleNode::UnitView *unit_view;
    const std::vector<dsc2::ScheduleNode::Size> *unit_view_sizes;
    if (needsUniform()) {
      address = constructUniformizedFoldedAddress(
          builder, data_info->startAddr_, factor);
      // TODO: For now, we assume that unit_views are same for all
      // cores/corelets of a DSC.
      if (is_any_of(compute_op.inputs_[index], NFWD0, NFWD2)) {
        unit_view = &(compute_op.coreletViews_.begin())
                         ->second.outputsLoopsAndSizes_.front();
      } else {
        unit_view = &(compute_op.coreletViews_.begin())
                         ->second.inputsLoopsAndSizes_[index];
      }

      // TODO: modify this later.
      unit_view_sizes = &unit_view->sizesNoGaps_;
    } else {
      // expecting constant value across sdsc folds
      auto start_addr = int64_t(
          FoldInfraUtils::getSingleDataStrict(
              data_info->startAddr_, {{0, core_id_}, {1, corelet_id_}}) *
          factor);
      auto const_op = mlir::arith::ConstantIndexOp::create(
          builder, builder.getUnknownLoc(), start_addr);
      address = const_op.getResult();
      if (is_any_of(compute_op.inputs_[index], NFWD0, NFWD2)) {
        unit_view = &compute_op.coreletViews_.at(corelet_id_)
                         .outputsLoopsAndSizes_.front();
      } else {
        unit_view = &compute_op.coreletViews_.at(corelet_id_)
                         .inputsLoopsAndSizes_[index];
      }

      unit_view_sizes = &unit_view->getSizesForCoreId(core_id_);
    }

    bool is_constant_read =
        data_info->myLdsIdx_ == -1 && data_info->constantId_ >= 0;
    SNDSCLowering dsc_lowering = this->cloneSNDSCLoweringObject();
    SNTransferLowering transfer_lowering(dsc_lowering, result_type);
    if (failed(transfer_lowering
                   .constructElementsOfAffineDataTransferViaAgenTransfer(
                       builder, storage, true, address, *unit_view_sizes,
                       unit_view->outerLoops_, view, base_address_map,
                       base_address_args, transfer_set, transfer_order,
                       is_constant_read))) {
      return LogicalResult::failure();
    } else {
      auto load_op = agen::VectorLoadOp::create(
          builder, builder.getUnknownLoc(), result_type, view.getResult(),
          builder.getStringAttr(compute_op.name_), base_address_map,
          base_address_args, transfer_set, transfer_order);
      if (compute_op.inputs_[index] == NFWD0) {
        if (element_type.isF16()) {
          int index_array[8] = {2, 3, 0, 1, 6, 7, 6, 7};
          auto nfwd0 = vectorchain::ShuffleOp::create(
              builder, builder.getUnknownLoc(), result_type,
              load_op.getResult(), nullptr,
              builder.getI32ArrayAttr(index_array),
              builder.getI32IntegerAttr(8),
              builder.getStringAttr(compute_op.name_));
          input_data = nfwd0.getResult();
        } else if (element_type.isF32()) {
          int index_array[4] = {1, 0, 3, 3};
          auto nfwd0 = vectorchain::ShuffleOp::create(
              builder, builder.getUnknownLoc(), result_type,
              load_op.getResult(), nullptr,
              builder.getI32ArrayAttr(index_array),
              builder.getI32IntegerAttr(8),
              builder.getStringAttr(compute_op.name_));
          input_data = nfwd0.getResult();
        }
      } else if (compute_op.inputs_[index] == NFWD2) {
        if (element_type.isF16()) {
          int index_array[8] = {4, 5, 3, 3, 5, 5, 6, 7};
          auto nfwd2 = vectorchain::ShuffleOp::create(
              builder, builder.getUnknownLoc(), result_type,
              load_op.getResult(), nullptr,
              builder.getI32ArrayAttr(index_array),
              builder.getI32IntegerAttr(8),
              builder.getStringAttr(compute_op.name_));
          input_data = nfwd2.getResult();
        } else if (element_type.isF32()) {
          int index_array[4] = {2, 0, 1, 3};
          auto nfwd2 = vectorchain::ShuffleOp::create(
              builder, builder.getUnknownLoc(), result_type,
              load_op.getResult(), nullptr,
              builder.getI32ArrayAttr(index_array),
              builder.getI32IntegerAttr(8),
              builder.getStringAttr(compute_op.name_));
          input_data = nfwd2.getResult();
        }
      } else if (compute_op.inputs_[index] == LXLUSCALEREG) {
        // generate shuffle op
        auto clId = (corelet_id_ == -1 ? 0 : corelet_id_);
        auto &loop_offset =
            compute_op.inputsLdsAndLoopOffsets_.at(index).loopEleOffsets_.at(
                clId);
        // inner most loop
        auto loop_offset_it = std::find_if(
            loop_offset.begin(), loop_offset.end(), [](const auto &pair) {
              const auto &dim_offset = pair.second;
              DT_CHECK(dim_offset.size() == 1);
              return dim_offset.begin()->second != 0;
            });
        DT_CHECK(loop_offset_it != loop_offset.end());
        auto [curr_loop, dim_offset] = *loop_offset_it;
        auto [dim, offset] = *dim_offset.begin();
        auto *loop = this->getMLIRLoopFromLoopNode(curr_loop, dim);
        mlir::Value loop_iter;
        if (auto affine_for = llvm::dyn_cast<mlir::affine::AffineForOp>(loop)) {
          loop_iter = affine_for.getInductionVar();
        } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop)) {
          loop_iter = scf_for.getInductionVar();
        }
        auto element_type =
            dataflow::utils::getElementType(result_type);
        auto num_elements =
            dataflow::utils::getNumElements(result_type);
        auto reduced_num_elements = num_elements / 2;
        auto reduced_result_type =
            dataflow::utils::constructVectorType(element_type, reduced_num_elements);
        auto data = vectorchain::ShuffleOp::create(
            builder, loc, reduced_result_type, load_op, ValueRange{loop_iter},
            ValueRange{}, nullptr, builder.getI32ArrayAttr({-1}),
            builder.getI32IntegerAttr(reduced_num_elements),
            builder.getStringAttr(compute_op.name_));
        input_data = data.getResult();
      } else {
        input_data = load_op.getResult();
      }
    }
  } else if (compute_op.inputs_[index] == PT) {
    auto src_unit =
        retrieveGetUnitOpInSameCore(builder, PTROW7, core_id_, corelet_id_);
    auto data =
        dataflow::ReceiveOp::create(builder, loc, result_type, src_unit,
                                    builder.getStringAttr(compute_op.name_));
    input_data = data.getResult();
  } else if (compute_op.inputs_[index] == LXLU) {
    auto src_unit =
        retrieveGetUnitOpInSameCore(builder, LXLU, core_id_, corelet_id_);
    auto data =
        dataflow::ReceiveOp::create(builder, loc, result_type, src_unit,
                                    builder.getStringAttr(compute_op.name_));
    input_data = data.getResult();
  } else if (compute_op.inputs_[index] == PE) {
    auto src_unit =
        retrieveGetUnitOpInSameCore(builder, PE, core_id_, corelet_id_);
    auto data =
        dataflow::ReceiveOp::create(builder, loc, result_type, src_unit,
                                    builder.getStringAttr(compute_op.name_));
    input_data = data.getResult();
  } else if (compute_op.inputs_[index] == SFP) {
    auto src_unit =
        retrieveGetUnitOpInSameCore(builder, SFP, core_id_, corelet_id_);
    auto data =
        dataflow::ReceiveOp::create(builder, loc, result_type, src_unit,
                                    builder.getStringAttr(compute_op.name_));
    input_data = data.getResult();
  } else if (compute_op.inputs_[index] == SFPRING) {
    DT_CHECK(comp_ == SFP);
    mlir::Value get_unit_val;
    if (needsUniform()) {
      get_unit_val = constructUniformizedFoldedDestinationCore(
          builder, compute_op.inputsLdsAndLoopOffsets_[index].startAddr_);
    } else {
      // expecting constant value across sdsc folds
      int src_core_id = int(FoldInfraUtils::getSingleDataStrict(
          compute_op.inputsLdsAndLoopOffsets_[index].startAddr_,
          {{0, core_id_}, {1, corelet_id_}}));
      get_unit_val = createGetUnitOpInDifferentCore(builder, comp_, src_core_id,
                                                    corelet_id_, num_folds_)
                         .getResult(0);
    }

    auto data =
        dataflow::ReceiveOp::create(builder, loc, result_type, get_unit_val,
                                    builder.getStringAttr(compute_op.name_));
    input_data = data.getResult();
  } else if (compute_op.inputs_[index] == LATCH) {
    int latch_id = compute_op.inputsLdsAndLoopOffsets_[index].latchDataId_;
    DT_CHECK_MSG(latch_id != -1, "latch id cannot be negative");
    auto latch_data = this->getFromLatchMap(latch_id);
    if (compute_op.exUnit_ == LXLU) {
      auto clId = (corelet_id_ == -1 ? 0 : corelet_id_);
      auto &loop_offset =
          compute_op.inputsLdsAndLoopOffsets_.at(index).loopEleOffsets_.at(
              clId);
      DT_CHECK(loop_offset.size() == 1);
      auto [curr_loop, dim_offset] = *loop_offset.begin();
      DT_CHECK(dim_offset.size() == 1);
      auto [dim, offset] = *dim_offset.begin();
      auto *loop = this->getMLIRLoopFromLoopNode(curr_loop, dim);
      mlir::Value loop_iter;
      if (auto affine_for = llvm::dyn_cast<mlir::affine::AffineForOp>(loop)) {
        loop_iter = affine_for.getInductionVar();
      } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop)) {
        loop_iter = scf_for.getInductionVar();
      }
      auto element_type =
          dataflow::utils::getElementType(result_type);
      auto num_elements =
          dataflow::utils::getNumElements(result_type);
      auto reduced_num_elements = num_elements / 4;
      auto reduced_result_type =
          dataflow::utils::constructVectorType(element_type, reduced_num_elements);
      auto data = vectorchain::ShuffleOp::create(
          builder, loc, reduced_result_type, latch_data, ValueRange{loop_iter},
          ValueRange{}, nullptr, builder.getI32ArrayAttr({-1}),
          builder.getI32IntegerAttr(reduced_num_elements),
          builder.getStringAttr(compute_op.name_));
      input_data = data.getResult();
    } else {
      input_data = latch_data;
    }
    DT_CHECK_MSG(input_data, "latch value cannot be empty");
  } else {
    return LogicalResult::failure();
  }

  Value final_data = input_data;
  if (!is_any_of(compute_op.inputs_[index], PESTATE, SFPSTATE) &&
      compute_op.exUnit_ != LXLU) {
    auto compute_format = compute_op.getComputeOperandFormats(*dsc_)[index];
    if (failed(constructPrecisionConversionOperation(
            builder, input_data, compute_format, final_data, comp_))) {
      return LogicalResult::failure();
    }
  }

  inputs.push_back(final_data);
  return LogicalResult::success();
}

// ---- 87/110  constructComputeOutputOperand  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:777  (139L)
LogicalResult e087_constructComputeOutputOperand(
    OpBuilder &builder, Location &loc, Type &result_type,
    const dsc2::ComputeNode &compute_op, Value result, int i) {
  Value compute_result = result;

  if (!is_any_of(compute_op.outputs_[i], PESTATE, SFPSTATE) &&
      compute_op.exUnit_ != LXLU) {
    DataFormats format;
    int lds_idx = compute_op.outputsLdsAndLoopOffsets_[i].myLdsIdx_;
    if (lds_idx != -1) {
      format = dsc_->labeledDs_[lds_idx].dataFormat_;
    } else {
      format = compute_op.getComputeOperandFormats(*dsc_)[i];
    }

    if (failed(constructPrecisionConversionOperation(builder, result, format,
                                                     compute_result, comp_))) {
      return LogicalResult::failure();
    }
  }

  if (compute_op.outputs_[i] == PTSOUTH) {
    auto south =
        retrieveGetUnitOpInSameCore(builder, PTSOUTH, core_id_, corelet_id_);
    auto data = dataflow::SendOp::create(
        builder, loc, south, compute_result,
        /*dir*/ nullptr, builder.getStringAttr(compute_op.name_));
  } else if (is_any_of(compute_op.outputs_[i], LRFREG, PTXRF, PTARF, PELRF,
                       SFPLRF, SFPSTATE, PESTATE)) {
    Value store_op_data;
    dataflow::GetLogicalMemoryViewOp view;
    AffineMap base_address_map;
    llvm::SmallVector<Value, 8> base_address_args;
    IntegerSet transfer_set;
    AffineMap transfer_order;

    auto storage = compute_op.outputs_[i];
    auto factor = getAddressGranularityMultiplyFactor(
        comp_, storage, dataflow::utils::getElementType(result_type));

    mlir::Value address;
    const dsc2::ScheduleNode::UnitView *unit_view;
    const std::vector<dsc2::ScheduleNode::Size> *unit_view_sizes;
    if (needsUniform()) {
      address = constructUniformizedFoldedAddress(
          builder, compute_op.outputsLdsAndLoopOffsets_[i].startAddr_, factor);

      // TODO: For now, we assume that unit_views are same for all
      // cores/corelets of a DSC.
      unit_view =
          &(compute_op.coreletViews_.begin())->second.outputsLoopsAndSizes_[i];
      // TODO: modify this later.
      unit_view_sizes = &unit_view->sizesNoGaps_;
    } else {
      // expecting constant value across sdsc folds
      auto start_addr =
          int64_t(FoldInfraUtils::getSingleDataStrict(
                      compute_op.outputsLdsAndLoopOffsets_[i].startAddr_,
                      {{0, core_id_}, {1, corelet_id_}}) *
                  factor);
      auto const_op = mlir::arith::ConstantIndexOp::create(
          builder, builder.getUnknownLoc(), start_addr);
      address = const_op.getResult();

      unit_view =
          &compute_op.coreletViews_.at(corelet_id_).outputsLoopsAndSizes_[i];
      unit_view_sizes = &unit_view->getSizesForCoreId(core_id_);
    }

    SNDSCLowering dsc_lowering = this->cloneSNDSCLoweringObject();
    SNTransferLowering transfer_lowering(dsc_lowering, result_type);
    const auto *data_info = &compute_op.outputsLdsAndLoopOffsets_[i];
    bool is_constant_write =
        data_info->myLdsIdx_ == -1 && data_info->constantId_ >= 0;
    if (failed(transfer_lowering
                   .constructElementsOfAffineDataTransferViaAgenTransfer(
                       builder, storage, false, address, *unit_view_sizes,
                       unit_view->outerLoops_, view, base_address_map,
                       base_address_args, transfer_set, transfer_order,
                       is_constant_write))) {
      return LogicalResult::failure();
    } else {
      auto store_op = agen::VectorStoreOp::create(
          builder, builder.getUnknownLoc(), compute_result, view.getResult(),
          builder.getStringAttr(compute_op.name_), base_address_map,
          base_address_args, transfer_set, transfer_order);
    }
  } else if (compute_op.outputs_[i] == LXSU) {
    auto dst_unit =
        retrieveGetUnitOpInSameCore(builder, LXSU, core_id_, corelet_id_);
    auto data = dataflow::SendOp::create(
        builder, loc, dst_unit, compute_result,
        /*dir*/ nullptr, builder.getStringAttr(compute_op.name_));
  } else if (compute_op.outputs_[i] == SFP) {
    auto dst_unit =
        retrieveGetUnitOpInSameCore(builder, SFP, core_id_, corelet_id_);
    auto data = dataflow::SendOp::create(
        builder, loc, dst_unit, compute_result,
        /*dir*/ nullptr, builder.getStringAttr(compute_op.name_));
  } else if (compute_op.outputs_[i] == PE) {
    auto dst_unit =
        retrieveGetUnitOpInSameCore(builder, PE, core_id_, corelet_id_);
    auto data = dataflow::SendOp::create(
        builder, loc, dst_unit, compute_result,
        /*dir*/ nullptr, builder.getStringAttr(compute_op.name_));
  } else if (compute_op.outputs_[i] == SFPRING) {
    DT_CHECK(comp_ == SFP);
    mlir::Value get_unit_val;
    if (needsUniform()) {
      get_unit_val = constructUniformizedFoldedDestinationCore(
          builder, compute_op.outputsLdsAndLoopOffsets_[i].startAddr_);
    } else {
      // expecting constant value across sdsc folds
      int dst_core_id = int(FoldInfraUtils::getSingleDataStrict(
          compute_op.outputsLdsAndLoopOffsets_[i].startAddr_,
          {{0, core_id_}, {1, corelet_id_}}));
      get_unit_val = createGetUnitOpInDifferentCore(builder, comp_, dst_core_id,
                                                    corelet_id_, num_folds_)
                         .getResult(0);
    }

    auto data = dataflow::SendOp::create(
        builder, loc, get_unit_val, compute_result, /*dir*/ nullptr,
        builder.getStringAttr(compute_op.name_));
  } else if (compute_op.outputs_[i] == LATCH) {
    int latch_id = compute_op.outputsLdsAndLoopOffsets_[i].latchDataId_;
    DT_CHECK_MSG(latch_id != -1, "latch id cannot be negative");
    addToLatchMap(latch_id, compute_result);
  } else if (compute_op.outputs_[i] == PT) {
    auto pt = retrieveGetUnitOpInSameCore(builder, PT, core_id_, corelet_id_);
    auto data = dataflow::SendOp::create(
        builder, loc, pt, compute_result,
        /*dir*/ nullptr, builder.getStringAttr(compute_op.name_));
  } else {
    return LogicalResult::failure();
  }

  return LogicalResult::success();
}

// ---- 88/110  constructLoopsRecursive  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:862  (324L)
LogicalResult e088_constructLoopsRecursive(
    const std::vector<const dsc2::ScheduleNode *> &sched_nodes,
    const dsc2::BlockNode *parent_node, OpBuilder &builder,
    SmallVectorImpl<mlir::Value> &sched_nodes_results) {
  auto loc = builder.getUnknownLoc();

  // Assumptions: nodes are sorted in the tie-order.
  for (const auto *sched_node : sched_nodes) {
    const auto *loopNode = dynamic_cast<const dsc2::LoopNode *>(sched_node);
    if (loopNode && loopNode->isParametricLoop()) {
      // We cannot pass corelet_id_ because when uniformization is enabled it is
      // set to -1
      int size = loopNode->parametricIterCount(dsc_, 0, comp_);
      DT_CHECK((dsc_->numCoreletsUsed_DSC2_ == 1 ||
                loopNode->parametricIterCount(dsc_, 1, comp_) == size) &&
               "DCC does not currently support imbalanced corelet split");
      SmallVector<Value, 8> iter_args;  // TO DO:
      mlir::Operation *loop = mlir::affine::AffineForOp::create(
          builder, loc, 0, size, 1, iter_args);
      builder.setInsertionPointToStart(
          dyn_cast<mlir::affine::AffineForOp>(loop).getBody());
      (*dsc_loops_to_mlir_loops_map_)[loopNode].push_back(loop);
      resetIterArguments(loop, iter_args);
      SmallVector<mlir::Value, 2> node_results;
      auto children = loopNode->getNextView(comp_, corelet_id_, core_id_);
      if (failed(constructLoopsRecursive(children, loopNode, builder,
                                         node_results))) {
        return LogicalResult::failure();
      }
      sched_nodes_results.insert(sched_nodes_results.end(),
                                 node_results.begin(), node_results.end());
      auto *outer_most_loop = (*dsc_loops_to_mlir_loops_map_)[loopNode].at(0);
      builder.setInsertionPointAfter(outer_most_loop);
    } else if (const auto *node =
                   dynamic_cast<const dsc2::LoopNode *>(sched_node)) {
      bool iterator_args_present = false;
      for (int dim_idx = node->dims_.size() - 1; dim_idx >= 0; dim_idx--) {
        SmallVector<Value, 8> iter_args;
        int dim_idx_front = node->dims_.size() - 1 - dim_idx;
        if (failed(constructLoopIterArgs(builder, node, parent_node, loc,
                                         iter_args, dim_idx_front))) {
          return LogicalResult::failure();
        }

        iterator_args_present =
            !iterator_args_present ? !iter_args.empty() : true;

        /// TODO: logic to be moved to dsc2 and improved to handle all
        /// MetaDimKinds
        const auto &[dim, kind] = node->dims_[dim_idx];
        PaddingFormType padInfo;
        if (kind == MetaDimKind::Padded)
          padInfo.setPadding(dim, PadType::PADDED_WZEROPAD);

        mlir::Operation *loop;

        if (!node->isDimSymbolic(dim)) {
          loop = constructLoopForADim(builder, node, iter_args, dim, padInfo);
        } else {
          auto &dimLoopCountSymbols = node->loopCountSymbolIds_.at(dim);
          DT_CHECK(dimLoopCountSymbols.size() == 1 &&
                   "Only one symbol for a loop bound");

          auto root_loop = (*dsc_loops_to_mlir_loops_map_)[nullptr].front();
          OpBuilder global_builder(root_loop);

          auto symbol_op = mlir::symbol::CreateSymbolOp::create(
              global_builder, loc, global_builder.getIndexType());
          symbol_op->setAttr("SymbolId", global_builder.getI64IntegerAttr(
                                             dimLoopCountSymbols.front()));
          auto sym_max = sym_info_.getMax(dimLoopCountSymbols.front());
          symbol_op->setAttr("maxValue",
                             global_builder.getI64IntegerAttr(sym_max));
          auto sym_granularity =
              sym_info_.getGranularity(dimLoopCountSymbols.front());
          symbol_op->setAttr("granularity",
                             global_builder.getI64IntegerAttr(sym_granularity));

          auto lb =
              mlir::arith::ConstantIndexOp::create(global_builder, loc, 0);
          auto step =
              mlir::arith::ConstantIndexOp::create(global_builder, loc, 1);
          loop = scf::ForOp::create(builder, loc, lb, symbol_op.getResult(),
                                    step, iter_args);
          builder.setInsertionPointToStart(
              dyn_cast<scf::ForOp>(loop).getBody());
          (*dsc_loops_to_mlir_loops_map_)[node].push_back(loop);
          resetIterArguments(loop, iter_args);
        }

        auto loop_name = EnumsConversion::primaryDimToString.at(dim) + "-" +
                         dsc_->dataStageParam_.at(node->numId_).name() + "/" +
                         dsc_->dataStageParam_.at(node->denId_).name();
        dataflow::setDbgName(loop, loop_name);
      }

      SmallVector<mlir::Value, 2> node_results;
      auto children = node->getNextView(comp_, corelet_id_, core_id_);
      if (failed(
              constructLoopsRecursive(children, node, builder, node_results))) {
        return LogicalResult::failure();
      }

      // setting iterator arguments
      if (iterator_args_present) {
        for (int dim_idx = node->dims_.size() - 1; dim_idx >= 0; dim_idx--) {
          const auto &dim = node->dims_[dim_idx].dim_;

          // COnfirm to assertions on loops within a band
          auto *loop = this->getMLIRLoopFromLoopNode(node, dim);
          DT_CHECK(loop->getNumRegions() == 1);
          auto &block = loop->getRegion(0).front();

          // Get child loop
          // Accounting dummy loop created for block nodes
          // TODO: it looks like creation of loop iterator arguments and yield
          // logic needs to be better worked out.
          Operation *loop_for_args = loop;
          bool use_return_values =
              is_any_of(block.getOperations().size(), 1, 2) &&
              isa<mlir::affine::AffineForOp, mlir::scf::ForOp>(block.front()) &&
              block.front().getNumResults() > 0;

          // Set yield builders
          // The yield operation should be after the above use_return_values
          // computation because it would influence the size of operations
          // in the block.
          mlir::Operation *yield_op;
          OpBuilder yield_builder(loop);
          yield_builder.setInsertionPointToEnd(&block);

          if (isa<mlir::affine::AffineForOp>(loop)) {
            yield_op = mlir::affine::AffineYieldOp::create(yield_builder, loc);
          } else {
            yield_op = mlir::scf::YieldOp::create(yield_builder, loc);
          }

          if (use_return_values) {
            SmallVector<mlir::Value, 10> yield_operands;

            loop_for_args = &block.front();
            auto loop_for_args_results = loop_for_args->getResults();
            yield_operands.insert(yield_operands.begin(),
                                  loop_for_args_results.begin(),
                                  loop_for_args_results.end());

            // happens when multiple transfers are at different locations.
            int extra_args_count =
                loop->getNumResults() - loop_for_args_results.size();
            for (int idx = loop->getNumResults() - extra_args_count;
                 idx < loop->getNumResults(); idx++) {
              yield_operands.push_back(block.getArgument(idx + 1));
            }

            yield_op->insertOperands(0, yield_operands);
          } else {
            DT_CHECK_MSG(
                node_results.empty(),
                "translator still needs support in scenarios of a loop "
                "with streaming/double buffer "
                "transfer with siblings of loops having again transfers");

            ValueRange iter_args;
            if (auto affine_for =
                    llvm::dyn_cast<mlir::affine::AffineForOp>(loop_for_args)) {
              iter_args = affine_for.getRegionIterArgs();

            } else if (auto scf_for =
                           llvm::dyn_cast<scf::ForOp>(loop_for_args)) {
              iter_args = scf_for.getRegionIterArgs();
            }

            yield_op->insertOperands(0, iter_args);
          }
        }
      }

      sched_nodes_results.insert(sched_nodes_results.end(),
                                 node_results.begin(), node_results.end());
      auto *outer_most_loop = (*dsc_loops_to_mlir_loops_map_)[node].at(0);
      builder.setInsertionPointAfter(outer_most_loop);
    } else if (const auto *cond_node =
                   dynamic_cast<const dsc2::ConditionNode *>(sched_node)) {
      auto children = cond_node->getNextView(comp_, corelet_id_, core_id_);
      DT_CHECK_MSG(!children.empty(), "if-regions should not be empty");

      SmallVector<mlir::Value, 2> node_results;
      if (!cond_node->hasCoreClCond()) {  // condition is not on core/corelet.
        OpBuilder else_builder = builder;
        OpBuilder endif_builder = builder;
        bool has_else_branch = children.size() == 2;
        dsc2::LoopCondComposite condition = cond_node->loopCond_;

        mlir::scf::IfOp if_op;
        if (failed(constructConditionalOperation(
                (*dsc_loops_to_mlir_loops_map_), has_else_branch, builder,
                else_builder, endif_builder, sched_node->name_, condition,
                if_op))) {
          return LogicalResult::failure();
        }

        (*this->dsc_conds_to_mlir_conds_map_)[cond_node] = if_op;
        // builder points to then region of the if operation.
        if (failed(constructLoopsRecursive({children[0]}, cond_node, builder,
                                           node_results))) {
          return LogicalResult::failure();
        }

        if (has_else_branch) {
          if (failed(constructLoopsRecursive({children[1]}, cond_node,
                                             else_builder, node_results))) {
            return LogicalResult::failure();
          }
        }

        builder = endif_builder;
      } else {
        DT_CHECK(cond_node->loopCond_.twoLevelOrOfAnds_.empty());
        if (!this->uniformization_enabled_) {  // no uniformization
          DT_CHECK(children.size() == 1);
          if (failed(constructLoopsRecursive({children[0]}, node, builder,
                                             node_results))) {
            emitError("Unable to construct conditionals on conditionals.");
            return LogicalResult::failure();
          }
        } else {
          int max_num_regions = children.size();
          DT_CHECK_MSG(1 <= max_num_regions && max_num_regions <= 2,
                       "either true/false on coreClConditions");

          SmallVector<mlir::Value> region_units;
          llvm::SmallVector<Attribute, 2> list_sizes;

          // fill in the then region units
          for (const auto &core_pair : cond_node->getThenCoreCl(comp_)) {
            for (const auto &corelet : core_pair.second) {
              auto unit_def_op = this->unit_to_value_map_->at(core_pair.first)
                                     .at(corelet)
                                     .at(comp_)
                                     .getDefiningOp();
              for (auto result : unit_def_op->getResults()) {
                region_units.push_back(result);
              }
            }
          }

          int num_then_region_units = region_units.size();
          bool has_then_region = false, has_else_region = false;
          if (num_then_region_units > 0) {
            list_sizes.push_back(
                builder.getI32IntegerAttr(num_then_region_units));
            has_then_region = true;
          }

          if (max_num_regions == 2) {
            // fill in the else region units
            for (const auto &core_pair : cond_node->getElseCoreCl(comp_)) {
              for (const auto &corelet : core_pair.second) {
                auto unit_def_op = this->unit_to_value_map_->at(core_pair.first)
                                       .at(corelet)
                                       .at(comp_)
                                       .getDefiningOp();
                for (auto result : unit_def_op->getResults()) {
                  region_units.push_back(result);
                }
              }
            }

            if (region_units.size() > num_then_region_units) {
              list_sizes.push_back(builder.getI32IntegerAttr(
                  region_units.size() - num_then_region_units));
              has_else_region = true;
            }
          }

          int num_regions = list_sizes.size();
          auto uniform_region = mlir::uniform::UniformizeRegionsOp::create(
              builder, builder.getUnknownLoc(), mlir::TypeRange(), region_units,
              ArrayAttr::get(builder.getContext(), list_sizes),
              /*regIndices*/ nullptr, /*regLocales*/ nullptr, num_regions);

          (*this->dsc_conds_to_mlir_conds_map_)[cond_node] = uniform_region;
          for (int i = 0; i < num_regions; i++) {
            auto &block = uniform_region.getRegion(i).emplaceBlock();
            block.addArgument(builder.getIndexType(), builder.getUnknownLoc());
            OpBuilder builder_region(uniform_region.getRegion(i));

            int child_index = i;
            if (!has_then_region && has_else_region) child_index = i + 1;
            if (failed(constructLoopsRecursive({children[child_index]}, node,
                                               builder_region, node_results))) {
              emitError("Unable to construct conditionals on conditionals.");
              return LogicalResult::failure();
            }

            uniform::YieldOp::create(builder_region,
                                     builder_region.getUnknownLoc());
          }
        }
      }
    } else if (const auto *block_node =
                   dynamic_cast<const dsc2::BlockNode *>(sched_node)) {
      SmallVector<mlir::Value, 2> node_results;
      auto children = block_node->getNextView(comp_, corelet_id_, core_id_);

      // creating a dummy loop to hold for a reference while inserting
      // operations
      auto dummy_loop =
          mlir::affine::AffineForOp::create(builder, loc, 0, 1, 1);
      dataflow::setDbgName(dummy_loop, "block node");
      (*dsc_blocks_to_mlir_loops_map_)[block_node] = dummy_loop;

      builder.setInsertionPoint(dummy_loop);
      if (failed(constructLoopsRecursive(children, block_node, builder,
                                         node_results))) {
        return LogicalResult::failure();
      }

      builder.setInsertionPointAfter(dummy_loop);
    }
  }

  return LogicalResult::success();
}

// ---- 89/110  constructStreamingOrDoubleBufferingStore  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1205  (185L)
LogicalResult e089_constructStreamingOrDoubleBufferingStore(
    OpBuilder &builder, SenComponents dst_storage, Value result_to_store,
    std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> &outer_loops,
    bool perform_composite_store,
    std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> &composite_loops,
    Operation *store_op, int mode) {
  // Create streaming address modifications
  const auto &lds_and_loop_offsets = transfer_->dstLdsAndLoopOffsets_[0];

  const dsc2::TransferNode::CoreletView *cl_view;
  if (this->uniformization_enabled_) {
    cl_view = &transfer_->coreletViews_.begin()->second;
  } else {
    cl_view = &transfer_->coreletViews_.at(corelet_id_);
  }

  const auto &view_sizes = cl_view->dstLoopsAndSizes_[0];

  const std::vector<dsc2::ScheduleNode::Size> *view_sizes_core_specific;
  if (this->uniformization_enabled_) {
    view_sizes_core_specific = &view_sizes.sizesNoGaps_;
  } else {
    view_sizes_core_specific = &view_sizes.getSizesForCoreId(core_id_);
  }

  auto storage_unit_op =
      retrieveGetUnitOpInSameCore(builder, dst_storage, core_id_, corelet_id_);

  int iter_arg_index = -1;
  auto &transfers_with_loop = (*dsc_all_parent_loops_to_buffers_switch_map_)
      [lds_and_loop_offsets.bufferSwitchPosition_];
  for (int i = 0; i < transfers_with_loop.size(); i++) {
    if (transfers_with_loop[i] == transfer_) {
      iter_arg_index = i;
    }
  }

  Value start_address;
  Operation *buffer_switch_loop_terminator = nullptr;
  if (iter_arg_index == -1) {
    return LogicalResult::failure();
  } else {
    auto loop = (*dsc_loops_to_mlir_loops_map_)[lds_and_loop_offsets
                                                    .bufferSwitchPosition_]
                    .back();
    if (auto affine_for = llvm::dyn_cast<mlir::affine::AffineForOp>(loop)) {
      start_address = affine_for.getRegionIterArgs()[iter_arg_index];
      buffer_switch_loop_terminator = affine_for.getBody(0)->getTerminator();
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop)) {
      start_address = scf_for.getRegionIterArgs()[iter_arg_index];
      buffer_switch_loop_terminator = scf_for.getBody(0)->getTerminator();
    } else {
      return LogicalResult::failure();
    }
  }

  dataflow::GetLogicalMemoryViewOp view;
  AffineMap base_address_map;
  llvm::SmallVector<Value, 8> base_address_args;
  IntegerSet transfer_set;
  AffineMap transfer_order;

  if (failed(constructElementsOfAgenDataTransfer(
          builder, dst_storage, true, start_address, *view_sizes_core_specific,
          transfer_->unitTimeTransferChunkSize_,
          transfer_->unitTimeTransferChunkStride_, outer_loops, view,
          base_address_map, base_address_args, transfer_set, transfer_order))) {
    emitError("Unable to construct elements of agen data transfer");
    return LogicalResult::failure();
  }

  if (!perform_composite_store) {
    auto agen_store_op = agen::VectorStoreOp::create(
        builder, builder.getUnknownLoc(), result_to_store, view.getResult(),
        builder.getStringAttr(transfer_->name_), base_address_map,
        base_address_args, transfer_set, transfer_order);
  } else {
    // Create agen.composite_load operation
    //    AffineMap time_order_map = AffineMap::getMultiDimIdentityMap(
    //        view_sizes.compositeLoops_.size(),
    //        builder.getContext());

    llvm::SmallVector<unsigned> permutation_vector;
    for (int i = composite_loops.size() - 1; i >= 0; i--)
      permutation_vector.push_back(i);

    AffineMap time_order_map =
        AffineMap::getPermutationMap(permutation_vector, builder.getContext());

    IntegerSet time_set;
    if (failed(constructTimeSet(builder, base_address_args, composite_loops,
                                time_set))) {
      return LogicalResult::failure();
    } else {
      DEBUG_WITH_TYPE(verbose_debug_, time_set.dump());
    }

    AffineMap time_address_map;
    if (failed(constructTimeAddressMap(builder, time_set.getNumDims(),
                                       view.getLayoutMap().getNumDims(),
                                       time_address_map, composite_loops))) {
      return LogicalResult::failure();
    } else {
      DEBUG_WITH_TYPE(verbose_debug_, time_address_map.dump());
    }

    auto composite_store = agen::CompositeStoreOp::create(
        builder, builder.getUnknownLoc(), view.getResult(),
        builder.getStringAttr(transfer_->name_), base_address_map,
        base_address_args, transfer_set, transfer_order, {}, time_set,
        time_order_map, time_address_map);

    OpBuilder local_builder(composite_store);
    local_builder.setInsertionPointToStart(composite_store.getBody());

    auto *data_produce_op = result_to_store.getDefiningOp();
    auto *cloned_op = local_builder.clone(*data_produce_op);
    local_builder.getInsertionPoint()->insertOperands(
        0, {cloned_op->getResult(0)});
    data_produce_op->erase();
    DEBUG_WITH_TYPE(verbose_debug_, composite_store->dump());
  }

  auto factor = getAddressGranularityMultiplyFactor(
      comp_, dst_storage, dataflow::utils::getElementType(result_type));

  // Update the yield argument with increment to reflect streaming behavior.
  OpBuilder local_builder(buffer_switch_loop_terminator);

  if (mode == 2) {  // streaming
    mlir::Value buffer_increment;
    if (this->uniformization_enabled_) {
      buffer_increment = constructUniformizedAddress(
          local_builder, transfer_->dstLdsAndLoopOffsets_[0].bufferAddrOffset_,
          factor);
    } else {
      int buffer_addr_offset = int(
          transfer_->dstLdsAndLoopOffsets_[0].bufferAddrOffset_.at(core_id_).at(
              corelet_id_) *
          factor);
      auto const_op = mlir::arith::ConstantIndexOp::create(
          local_builder, local_builder.getUnknownLoc(), buffer_addr_offset);
      buffer_increment = const_op.getResult();
    }

    auto buffer_index = mlir::arith::AddIOp::create(
        local_builder, local_builder.getUnknownLoc(),
        local_builder.getIndexType(),
        buffer_switch_loop_terminator->getOperand(iter_arg_index),
        buffer_increment);
    buffer_switch_loop_terminator->setOperand(iter_arg_index,
                                              buffer_index.getResult());
  } else if (mode == 1) {  // double buffering
    mlir::Value buffer_increment;
    if (needsUniform()) {
      buffer_increment = constructUniformizedFoldedDoubleBufferToggling(
          local_builder, transfer_->dstLdsAndLoopOffsets_[0].bufferAddrOffset_,
          transfer_->dstLdsAndLoopOffsets_[0].startAddr_, factor);
    } else {
      int buff_addr_offset =
          int((transfer_->dstLdsAndLoopOffsets_[0]
                   .bufferAddrOffset_.at(core_id_)
                   .at(corelet_id_) +
               2 * FoldInfraUtils::getSingleDataStrict(
                       transfer_->dstLdsAndLoopOffsets_[0].startAddr_,
                       {{0, core_id_}, {1, corelet_id_}})) *
              factor);
      auto const_op = mlir::arith::ConstantIndexOp::create(
          local_builder, local_builder.getUnknownLoc(), buff_addr_offset);
      buffer_increment = const_op.getResult();
    }

    auto buffer_index = mlir::arith::SubIOp::create(
        local_builder, local_builder.getUnknownLoc(),
        local_builder.getIndexType(), buffer_increment,
        buffer_switch_loop_terminator->getOperand(iter_arg_index));
    buffer_switch_loop_terminator->setOperand(iter_arg_index,
                                              buffer_index.getResult());
  } else {
    return LogicalResult::failure();
    llvm_unreachable("Unknown buffering mode");
  }

  return LogicalResult::success();
}

// ---- 90/110  GenerateLoadAndSendFromDataTransferNode  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2058  (274L)
LogicalResult e090_GenerateLoadAndSendFromDataTransferNode(
    OpBuilder &builder, Value to, SenComponents src, SenComponents src_storage,
    std::unordered_map<PrimaryDimTypes, int> &ctgs_transfer_sizes) {
  const auto &lds_and_loop_offsets = transfer_->srcLdsAndLoopOffsets_;

  const dsc2::TransferNode::CoreletView *view;
  if (this->uniformization_enabled_) {
    view = &transfer_->coreletViews_.begin()->second;
  } else {
    view = &transfer_->coreletViews_.at(corelet_id_);
  }

  const auto &view_sizes = view->srcLoopsAndSize_;

  const std::vector<dsc2::ScheduleNode::Size> *view_sizes_core_specific;
  if (this->uniformization_enabled_) {
    view_sizes_core_specific = &view_sizes.sizesNoGaps_;
  } else {
    view_sizes_core_specific = &view_sizes.getSizesForCoreId(core_id_);
  }

  // Construct implicit loops via the count instead of single for-loop.
  OpBuilder loop_builder(builder.getContext());
  loop_builder.setInsertionPoint(builder.getBlock(),
                                 builder.getInsertionPoint());

  std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> outer_loops;
  outer_loops.insert(outer_loops.begin(), view_sizes.outerLoops_.begin(),
                     view_sizes.outerLoops_.end());

  bool is_memory = is_any_of(comp_, LXLU, LXSU, L0LUROW0, L0SU);
  auto composite_loops = view_sizes.compositeLoops_;
  bool perform_composite_load =
      is_memory && !view_sizes.compositeLoops_.empty() &&
      !areEpiloguesInLoops(composite_loops) && !areEpiloguesInTransferSizes();

  // Switching builder to innermost composite loop for the worst case of
  // epiloues. in case of composite loops, creation of implicit loops have no
  // meaning.
  if (perform_composite_load) {
    auto inner_loop =
        (*dsc_loops_to_mlir_loops_map_)[composite_loops.front().loop_].front();
    if (auto affine_for =
            llvm::dyn_cast<mlir::affine::AffineForOp>(inner_loop)) {
      builder.setInsertionPointToStart(affine_for.getBody());
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(inner_loop)) {
      builder.setInsertionPointToStart(scf_for.getBody());
    }
  }

  std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> implicit_loops;
  if (!ctgs_transfer_sizes.empty()) {
    if (failed(constructImplicitLoopsForContiguousTransfer(
            builder, *view_sizes_core_specific,
            transfer_->unitTimeTransferChunkSize_, implicit_loops,
            ctgs_transfer_sizes))) {
      emitError("Unable to construct implicit loops for contiguous transfer");
      return LogicalResult::failure();
    } else {
      perform_composite_load =
          perform_composite_load && !areEpiloguesInLoops(implicit_loops);
      if (perform_composite_load) {
        composite_loops.insert(composite_loops.begin(), implicit_loops.begin(),
                               implicit_loops.end());
      } else {
        outer_loops.insert(outer_loops.begin(),
                           view_sizes.compositeLoops_.begin(),
                           view_sizes.compositeLoops_.end());
        outer_loops.insert(outer_loops.begin(), implicit_loops.begin(),
                           implicit_loops.end());
      }
    }
  } else {
    if (!perform_composite_load) {
      outer_loops.insert(outer_loops.begin(),
                         view_sizes.compositeLoops_.begin(),
                         view_sizes.compositeLoops_.end());
    }
  }

  bool update_insertion_loc =
      !perform_composite_load && !outer_loops.empty() &&
      (!implicit_loops.empty() || !composite_loops.empty());
  if (update_insertion_loc) {
    auto inner_loop =
        (*dsc_loops_to_mlir_loops_map_)[outer_loops.front().loop_].back();
    if (auto affine_for =
            llvm::dyn_cast<mlir::affine::AffineForOp>(inner_loop)) {
      loop_builder.setInsertionPointToStart(affine_for.getBody());
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(inner_loop)) {
      loop_builder.setInsertionPointToStart(scf_for.getBody());
    }
  }

  if (src_storage == LATCH) {
    int latch_id = transfer_->srcLdsAndLoopOffsets_.latchDataId_;
    DT_CHECK_MSG(latch_id != -1, "latch id cannot be negative");
    auto data = this->getFromLatchMap(latch_id);
    DT_CHECK_MSG(data, "latch value cannot be empty");
    auto send_op = dataflow::SendOp::create(
        loop_builder, loop_builder.getUnknownLoc(), to, data, /*dir*/ nullptr,
        loop_builder.getStringAttr(transfer_->name_));
    return LogicalResult::success();
  }

  int mode = -1;
  if (failed(this->getBufferingOrStreamingMode(mode))) {
    emitError("Unable to get buffering or streaming mode");
    return LogicalResult::failure();
  }

  // streaming or double buffering
  if (mode == 2 || mode == 1) {
    Operation *load_op = nullptr;
    if (failed(constructStreamingOrDoubleBufferingLoad(
            loop_builder, src_storage, to, load_op, mode, outer_loops,
            perform_composite_load, composite_loops))) {
      emitError("Unable to construct streaming load");
      return LogicalResult::failure();
    }
  } else {
    Value data;
    if (src_storage == ZERO) {
      auto zero = mlir::arith::ConstantOp::create(
          loop_builder, loop_builder.getUnknownLoc(), result_type,
          builder.getZeroAttr(result_type));
      data = zero.getResult();
    } else if (src_storage == CONSTANT) {
      int cst_idx = transfer_->srcLdsAndLoopOffsets_.constantId_;
      auto &cst_info = dsc_->constantInfo_.at(cst_idx);
      DT_CHECK(
          cst_idx >= 0 &&
          "constant id should be non-negative when using with CONST container");
      if (GenerateConstantBitStreamAndShuffle(cst_info, loop_builder, data)
              .failed()) {
        emitError("Unable to create constant bitstream and shuffle");
        return LogicalResult::failure();
      }
    } else {
      dataflow::GetLogicalMemoryViewOp view;
      AffineMap base_address_map;
      llvm::SmallVector<Value, 8> base_address_args;
      IntegerSet transfer_set, time_set;
      AffineMap transfer_order, time_order, time_addr_map;

      auto factor = getAddressGranularityMultiplyFactor(
          src, src_storage, dataflow::utils::getElementType(result_type));

      mlir::Value address;
      if (needsUniform()) {
        address = constructUniformizedFoldedAddress(
            loop_builder, transfer_->srcLdsAndLoopOffsets_.startAddr_, factor);
      } else {
        // expecting constant value across sdsc folds
        int start_addr = int(FoldInfraUtils::getSingleDataStrict(
                                 transfer_->srcLdsAndLoopOffsets_.startAddr_,
                                 {{0, core_id_}, {1, corelet_id_}}) *
                             factor);
        auto const_op = mlir::arith::ConstantIndexOp::create(
            loop_builder, builder.getUnknownLoc(), start_addr);
        address = const_op.getResult();
      }

      Value load_op_result;
      if (!perform_composite_load) {
        if (failed(constructElementsOfAgenDataTransfer(
                loop_builder, src_storage, true, address,
                *view_sizes_core_specific,
                transfer_->unitTimeTransferChunkSize_,
                transfer_->unitTimeTransferChunkStride_, outer_loops, view,
                base_address_map, base_address_args, transfer_set,
                transfer_order,
                transfer_->srcLdsAndLoopOffsets_.constantId_ != -1))) {
          emitError("Unable to construct elements of agen data transfer");
          return LogicalResult::failure();
        }

        auto load_op = agen::VectorLoadOp::create(
            loop_builder, loop_builder.getUnknownLoc(), result_type,
            view.getResult(), loop_builder.getStringAttr(transfer_->name_),
            base_address_map, base_address_args, transfer_set, transfer_order);
        if (transfer_->rotateNumElements_ > 0) {
          DT_CHECK_MSG(comp_ == LXLU, "Rotation is allowed only in LXLU");
          auto rot_element = mlir::arith::ConstantIndexOp::create(
              loop_builder, load_op.getLoc(), transfer_->rotateNumElements_);
          auto rot_op = vectorchain::RotateOp::create(
              loop_builder, rot_element.getLoc(), result_type,
              load_op.getResult(), rot_element,
              loop_builder.getStringAttr(transfer_->name_));
          load_op_result = rot_op.getResult();
        } else {
          load_op_result = load_op.getResult();
        }
      } else {
        if (failed(constructElementsOfAgenCompositeDataTransfer(
                loop_builder, src_storage, true, address,
                *view_sizes_core_specific,
                transfer_->unitTimeTransferChunkSize_,
                transfer_->unitTimeTransferChunkStride_, outer_loops, view,
                base_address_map, base_address_args, transfer_set,
                transfer_order, composite_loops, time_set, time_order,
                time_addr_map,
                transfer_->srcLdsAndLoopOffsets_.constantId_ != -1))) {
          emitError("Unable to construct elements of agen data transfer");
          return LogicalResult::failure();
        }

        auto map_operands = ArrayRef<Value>(base_address_args)
                                .drop_back(time_set.getNumSymbols());
        auto time_symbols = ArrayRef<Value>(base_address_args)
                                .take_back(time_set.getNumSymbols());
        auto composite_load = agen::CompositeLoadOp::create(
            loop_builder, builder.getUnknownLoc(), view.getResult(),
            loop_builder.getStringAttr(transfer_->name_), base_address_map,
            map_operands, result_type, transfer_set, transfer_order,
            time_symbols, time_set, time_order, time_addr_map);
        OpBuilder local_builder(composite_load);
        loop_builder.setInsertionPointToStart(composite_load.getBody());

        if (transfer_->rotateNumElements_ > 0) {
          DT_CHECK_MSG(comp_ == LXLU, "Rotation is allowed only in LXLU");
          auto rot_element = mlir::arith::ConstantIndexOp::create(
              loop_builder, composite_load.getLoc(),
              transfer_->rotateNumElements_);
          auto rot_op = vectorchain::RotateOp::create(
              loop_builder, rot_element.getLoc(), result_type,
              composite_load.getLoadInductionVar(), rot_element,
              loop_builder.getStringAttr(transfer_->name_));
          load_op_result = rot_op.getResult();
        } else {
          load_op_result = composite_load.getLoadInductionVar();
        }
      }

      if (src == SenComponents::LXLU && transfer_->replicationFactor_ > 1) {
        if (auto shuffle_op = construct2B16BLoadShuffle(
                loop_builder, load_op_result, result_type,
                transfer_->replicationFactor_)) {
          data = shuffle_op.getResult();
        } else {
          emitError("Unsupported load type.");
          return LogicalResult::failure();
        }
      } else {
        data = load_op_result;
      }
    }

    // Convert data if src precision and dst precision don't match.
    // example1: src_result_type (lxlu): <64xfp16>, dst_result_type (pe):
    // <64xfp32> for the above one, don't generate cast for lxlu since its
    // already a stick worth example2: src_result_type (pe): <64xfp32>,
    // dst_result_type (sfp): <64xfp16> for the above one, generate cast for pe
    // since its not fitting into a stick.
    auto stick_size = 1024;  // bits
    auto input_size = mlir::dataflow::utils::getDimSize(result_type, 0) *
                      dataflow::utils::getElementTypeBitWidth(result_type);
    if (dst_result_type != result_type && input_size != stick_size) {
      auto tmp_data = data;
      if (SNComputeLowering::constructPrecisionConversionOperation(
              loop_builder, tmp_data, dst_prec_, data, comp_)
              .failed()) {
        emitError(
            "Unable to construct precision conversion in a transfer operation");
      }
    }

    auto send_op = dataflow::SendOp::create(
        loop_builder, loop_builder.getUnknownLoc(), to, data, /*dir*/ nullptr,
        loop_builder.getStringAttr(transfer_->name_));
  }

  return LogicalResult::success();
}

// ---- 91/110  GenerateLoadAndStoreFromDataTransferNode  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2337  (132L)
LogicalResult e091_GenerateLoadAndStoreFromDataTransferNode(
    OpBuilder &builder, SenComponents src_storage, SenComponents dst_storage,
    int dst_idx,
    std::unordered_map<PrimaryDimTypes, int> &ctgs_transfer_sizes) {
  const auto &lds_and_loop_offsets = transfer_->srcLdsAndLoopOffsets_;

  const dsc2::TransferNode::CoreletView *view;
  if (this->uniformization_enabled_) {
    view = &transfer_->coreletViews_.begin()->second;
  } else {
    view = &transfer_->coreletViews_.at(corelet_id_);
  }

  const auto &view_sizes = view->srcLoopsAndSize_;

  const std::vector<dsc2::ScheduleNode::Size> *view_sizes_core_specific;
  if (this->uniformization_enabled_) {
    view_sizes_core_specific = &view_sizes.sizesNoGaps_;
  } else {
    view_sizes_core_specific = &view_sizes.getSizesForCoreId(core_id_);
  }

  // Construct implicit loops via the count instead of single for-loop.
  OpBuilder loop_builder(builder.getContext());
  loop_builder.setInsertionPoint(builder.getBlock(),
                                 builder.getInsertionPoint());

  std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> outer_loops;
  outer_loops.insert(outer_loops.begin(), view_sizes.outerLoops_.begin(),
                     view_sizes.outerLoops_.end());

  std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> implicit_loops;
  if (!ctgs_transfer_sizes.empty() && !implicit_loops.empty()) {
    if (failed(constructImplicitLoopsForContiguousTransfer(
            loop_builder, *view_sizes_core_specific,
            transfer_->unitTimeTransferChunkSize_, implicit_loops,
            ctgs_transfer_sizes))) {
      emitError("Unable to construct implicit loops for contiguous transfer");
      return LogicalResult::failure();
    } else {
      outer_loops.insert(outer_loops.begin(), implicit_loops.begin(),
                         implicit_loops.end());
    }
  }

  if (!outer_loops.empty()) {
    auto *inner_loop =
        (*dsc_loops_to_mlir_loops_map_)[outer_loops.front().loop_].front();
    if (auto affine_for =
            llvm::dyn_cast<mlir::affine::AffineForOp>(inner_loop)) {
      loop_builder.setInsertionPointToStart(affine_for.getBody());
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(inner_loop)) {
      loop_builder.setInsertionPointToStart(scf_for.getBody());
    }
  }

  Value input;
  if (src_storage == ZERO) {
    auto zero = mlir::arith::ConstantOp::create(
        loop_builder, loop_builder.getUnknownLoc(), result_type,
        builder.getZeroAttr(result_type));
    input = zero.getResult();
  } else if (src_storage == CONSTANT) {
    int cst_idx = transfer_->srcLdsAndLoopOffsets_.constantId_;
    auto &cst_info = dsc_->constantInfo_.at(cst_idx);
    DT_CHECK(
        cst_idx >= 0 &&
        "constant id should be non-negative when using with CONST container");
    if (GenerateConstantBitStreamAndShuffle(cst_info, loop_builder, input)
            .failed()) {
      emitError("Unable to create constant bitstream and shuffle");
      return LogicalResult::failure();
    }
  } else {
    emitError("Unknown source storage for load_and_store operation");
    return LogicalResult::failure();
  }

  // Destination storage
  {
    dataflow::GetLogicalMemoryViewOp view;
    AffineMap base_address_map;
    llvm::SmallVector<Value, 8> base_address_args;
    IntegerSet transfer_set;
    AffineMap transfer_order;

    auto factor = getAddressGranularityMultiplyFactor(
        comp_, dst_storage, dataflow::utils::getElementType(result_type));

    mlir::Value address;
    const std::vector<dsc2::ScheduleNode::Size> *dst_view_sizes_core_specific;
    if (needsUniform()) {
      address = constructUniformizedFoldedAddress(
          loop_builder, transfer_->dstLdsAndLoopOffsets_[dst_idx].startAddr_,
          factor);
      dst_view_sizes_core_specific = &transfer_->coreletViews_.begin()
                                          ->second.dstLoopsAndSizes_[dst_idx]
                                          .sizesNoGaps_;
    } else {
      // expecting constant value across sdsc folds
      int start_addr = static_cast<int>(
          FoldInfraUtils::getSingleDataStrict(
              transfer_->dstLdsAndLoopOffsets_[dst_idx].startAddr_,
              {{0, core_id_}, {1, corelet_id_}}) *
          factor);
      auto const_op = mlir::arith::ConstantIndexOp::create(
          loop_builder, builder.getUnknownLoc(), start_addr);
      address = const_op.getResult();
      dst_view_sizes_core_specific = &transfer_->coreletViews_.at(corelet_id_)
                                          .dstLoopsAndSizes_[dst_idx]
                                          .getSizesForCoreId(core_id_);
    }

    if (failed(constructElementsOfAgenDataTransfer(
            loop_builder, dst_storage, true, address,
            *dst_view_sizes_core_specific,
            transfer_->unitTimeTransferChunkSize_,
            transfer_->unitTimeTransferChunkStride_, outer_loops, view,
            base_address_map, base_address_args, transfer_set,
            transfer_order))) {
      emitError("Unable to construct elements of agen data transfer");
      return LogicalResult::failure();
    } else {
      auto store_op = agen::VectorStoreOp::create(
          loop_builder, loop_builder.getUnknownLoc(), input, view.getResult(),
          loop_builder.getStringAttr(transfer_->name_), base_address_map,
          base_address_args, transfer_set, transfer_order);
    }
  }

  return LogicalResult::success();
}

// ---- 92/110  GenerateDataTransfersForViaIfSo  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2617  (52L)
LogicalResult e092_GenerateDataTransfersForViaIfSo(
    const DataLocation &src, const dsc2::TransferNode::DstVia &dst_via,
    std::set<std::pair<SenComponents, SenComponents>> &explored_pairs,
    OpBuilder &builder) {
  // If comp is either src or dst, then its not via. So, simply return;
  if (src.unit_ == comp_) return LogicalResult::success();

  if (dst_via.loc_.unit_ == comp_) return LogicalResult::success();

  SenComponents to = dst_via.loc_.unit_;  // default
  SenComponents from = src.unit_;

  // Check if comp is part of vias, if so, pick the next one as to.
  bool part_of_via = false;
  for (int i = 0; i < dst_via.via_.size(); i++) {
    if (dst_via.via_[i] == comp_) {
      if (i < dst_via.via_.size() - 1) {
        to = dst_via.via_[i + 1];
      }

      if (i > 0) {
        from = dst_via.via_[i - 1];
      }

      part_of_via = true;
      break;
    }
  }

  if (!part_of_via) return LogicalResult::success();

  std::pair<SenComponents, SenComponents> from_to_pair =
      std::make_pair(from, to);
  if (explored_pairs.find(from_to_pair) == explored_pairs.end()) {
    explored_pairs.emplace(from_to_pair);
  } else {
    return LogicalResult::success();
  }

  // Create src, dst GetUnit
  auto src_unit_op =
      retrieveGetUnitOpInSameCore(builder, from, core_id_, corelet_id_);
  auto dst_unit_op =
      retrieveGetUnitOpInSameCore(builder, to, core_id_, corelet_id_);

  if (failed(GenerateReceiveAndSendFromDataTransferNode(
          builder, src_unit_op, dst_unit_op, sticks_src_ss))) {
    emitError("Unable to generate receive and send\n");
    return LogicalResult::failure();
  }
  return LogicalResult::success();
}


// ================================================================================================
// LEVEL 4
// ================================================================================================

// ---- 93/110  initializeUniformizedUnit  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIRUtils.hpp:663  (47L)
dataflow::ProgramUnitOp e093_initializeUniformizedUnit(
    SenComponents comp, std::vector<int> &cores, std::vector<int> &corelets,
    mlir::Value &program_unit_iterator, SmallVectorImpl<mlir::Value> *units,
    const DesignSpaceConfigGlobal &dsc_global) {
  OpBuilder builder(dataflow_func_op_);
  builder.setInsertionPointToEnd(&dataflow_func_op_.back());

  std::string type = EnumsConversion::senComponentsToString.at(comp);

  mlir::dataflow::GetUnitOp get_unit_op;
  for (auto &core_id : cores) {
    for (auto &corelet_id : corelets) {
      std::string name = type + "-CL" + std::to_string(corelet_id);

      // Create GetUnit op
      SmallVector<Type> get_unit_type;
      constructAVectorOfIndexType(num_folds_, builder, get_unit_type);

      get_unit_op = mlir::dataflow::GetUnitOp::create(
          builder, builder.getUnknownLoc(), get_unit_type,
          builder.getStringAttr(name), builder.getStringAttr(type));
      get_unit_op->setAttr("core", builder.getI32IntegerAttr(core_id));
      get_unit_op->setAttr("corelet", builder.getI32IntegerAttr(corelet_id));
      get_unit_op->setAttr("num_folds", builder.getI32IntegerAttr(num_folds_));
      for (int i = 0; i < num_folds_; i++)
        units->emplace_back(get_unit_op.getResult(i));

      this->unit_to_value_map_[core_id][corelet_id][comp] =
          get_unit_op.getResult(0);
    }
  }

  // Create ProgramUnit op
  auto unit_op = mlir::dataflow::ProgramUnitOp::create(
      builder, get_unit_op.getLoc(), *units);

  this->component_to_handler_.clear();
  this->component_to_handler_[comp] = program_unit_iterator =
      unit_op.getRegion().getArguments().front();

  OpBuilder unit_builder(unit_op);
  unit_builder.setInsertionPointToStart(&(unit_op.getRegion().front()));
  this->buildUniformizedNeighborUnits(comp, unit_builder, dsc_global,
                                      this->component_to_handler_[comp], *units,
                                      cores, corelets);
  return unit_op;
}

// ---- 94/110  constructComputeOutputOperands  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:921  (14L)
LogicalResult e094_constructComputeOutputOperands(
    OpBuilder &builder, Location &loc, Type &result_type,
    const dsc2::ComputeNode &compute_op, Value result) {
  Value compute_result = result;
  for (int i = 0; i < compute_op.outputs_.size(); i++) {
    auto status = constructComputeOutputOperand(builder, loc, result_type,
                                                compute_op, compute_result, i);
    if (failed(status)) {
      return LogicalResult::failure();
    }
  }

  return LogicalResult::success();
}

// ---- 95/110  constructFMINorFMAXOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1189  (86L)
LogicalResult e095_constructFMINorFMAXOperation(
    OpBuilder &builder, const dsc2::ComputeNode &compute_op) {
  auto loc = builder.getUnknownLoc();

  // Assumption: TODO: All the inputs are worth of 1 stick (128 bytes).
  // Otherwise, we have to generate more than one compute operation.

  // Construct inputs
  llvm::SmallVector<mlir::Value> inputs;
  for (int i = 0; i < compute_op.inputs_.size(); i++) {
    // Get the expected input type from compute operation
    Type element_type;
    Type result_type;
    DataFormats input_format = DataFormats::SEN169_FP16;
    int lds_idx = compute_op.inputsLdsAndLoopOffsets_[i].myLdsIdx_;
    if (lds_idx != -1) {
      input_format = dsc_->labeledDs_[lds_idx].dataFormat_;
    } else {
      input_format = compute_op.getComputeOperandFormats(*dsc_)[i];
    }

    if (failed(getTypeBasedOnComputeType(builder, input_format, result_type,
                                         element_type))) {
      return LogicalResult::failure();
    }

    // Construct input operand
    if (failed(constructComputeInputOperandAndAddToList(
            builder, loc, element_type, result_type, compute_op, inputs, i))) {
      emitError("Unable to construct Binary operation input operand");
      return LogicalResult::failure();
    }
  }

  // Construct expected result type from the contract.
  Type result_element_type;
  VectorType result_type;  // = inputs[0].getType();
  auto result_format = compute_op.getComputeOperandFormats(*dsc_).back();
  if (failed(getTypeBasedOnComputeType(builder, result_format, result_type,
                                       result_element_type))) {
    return LogicalResult::failure();
  }

  auto bool_type = VectorType::get(dataflow::utils::getNumElements(result_type),
                                   builder.getI1Type());

  // Construct mask operation
  auto mask_op =
      getStaticContinuousMaskValue(builder, loc, result_type, inputs,
                                   compute_op.instrAttribute_.compute_mask_);
  DT_CHECK_MSG(mask_op, "Could not create valid mask");

  vectorchain::VectorChainElementWiseCompareOperator op_name =
      compute_op.type_ == FMAX
          ? vectorchain::VectorChainElementWiseCompareOperator::compare_gt
          : vectorchain::VectorChainElementWiseCompareOperator::compare_le;

  auto compare_op = vectorchain::ElementWiseCompareOp::create(
      builder, loc, bool_type, inputs[0], inputs[1], mask_op,
      builder.getStringAttr(compute_op.name_), op_name);

  auto select_op = vectorchain::ElementWiseSelectionOp::create(
      builder, loc, result_type, compare_op.getResult(), inputs[0], inputs[1],
      mask_op, builder.getStringAttr(compute_op.name_));

  for (int i = 0; i < compute_op.outputs_.size(); i++) {
    // Construct output operands
    if (is_any_of(compute_op.outputs_[i], PESTATE, SFPSTATE)) {
      if (failed(constructComputeOutputOperand(builder, loc, bool_type,
                                               compute_op,
                                               compare_op.getResult(), i))) {
        emitError("Unable to construct FMIN/FMAX operation output operand");
        return LogicalResult::failure();
      }
    } else {
      if (failed(constructComputeOutputOperand(builder, loc, result_type,
                                               compute_op,
                                               select_op.getResult(), i))) {
        emitError("Unable to construct FMIN/FMAX operation output operand");
        return LogicalResult::failure();
      }
    }
  }

  return LogicalResult::success();
}

// ---- 96/110  constructLoops  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNControlFlowLowering.cpp:1191  (27L)
LogicalResult e096_constructLoops() {
  OpBuilder builder(*unit_op_);
  builder.setInsertionPoint(&(unit_op_->getRegion().front().back()));

  // Create the synthetic outermost loop with count being 1.
  auto synthetic_root = mlir::affine::AffineForOp::create(
      builder, builder.getUnknownLoc(), 0, 1, 1);
  dataflow::setDbgName(synthetic_root, "synthetic_root");
  builder.setInsertionPointToStart(synthetic_root.getBody(0));
  (*dsc_loops_to_mlir_loops_map_)[nullptr].push_back(synthetic_root);

  if (is_any_of(comp_, L0LUROW0, L0SU, LXLU, LXSU)) {
    if (failed(getBlockingOrStreamingBufferLoopLocations())) {
      return LogicalResult::failure();
    }

    if (failed(propagateBufferSwitchLoopsToRoot())) {
      return LogicalResult::failure();
    }
  }

  auto roots =
      dsc_->scheduleTree_.getHead()->getNextView(comp_, corelet_id_, core_id_);

  SmallVector<mlir::Value, 2> results;
  return constructLoopsRecursive({roots}, nullptr, builder, results);
}

// ---- 97/110  GenerateReceiveAndStoreFromDataTransferNode  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:1508  (269L)
LogicalResult e097_GenerateReceiveAndStoreFromDataTransferNode(
    int dst_idx, OpBuilder &builder, Value from, SenComponents dst,
    SenComponents dst_storage,
    std::unordered_map<PrimaryDimTypes, int> &ctgs_transfer_sizes,
    mlir::Value &final_store_value) {
  const auto &lds_and_loop_offsets = transfer_->dstLdsAndLoopOffsets_[dst_idx];

  const dsc2::TransferNode::CoreletView *view;
  if (this->uniformization_enabled_) {
    view = &transfer_->coreletViews_.begin()->second;
  } else {
    view = &transfer_->coreletViews_.at(corelet_id_);
  }

  const auto &view_sizes = view->dstLoopsAndSizes_[dst_idx];

  const std::vector<dsc2::ScheduleNode::Size> *view_sizes_core_specific;
  if (this->uniformization_enabled_) {
    view_sizes_core_specific = &view_sizes.sizesNoGaps_;
  } else {
    view_sizes_core_specific = &view_sizes.getSizesForCoreId(core_id_);
  }

  // Construct implicit loops via the count instead of single for-loop.
  OpBuilder loop_builder(builder.getContext());
  loop_builder.setInsertionPoint(builder.getBlock(),
                                 builder.getInsertionPoint());

  std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> outer_loops;
  outer_loops.insert(outer_loops.begin(), view_sizes.outerLoops_.begin(),
                     view_sizes.outerLoops_.end());

  bool is_memory = is_any_of(comp_, LXLU, LXSU, L0LUROW0, L0SU);
  auto composite_loops = view_sizes.compositeLoops_;
  bool perform_composite_store =
      is_memory && !view_sizes.compositeLoops_.empty() &&
      !areEpiloguesInLoops(composite_loops) && !areEpiloguesInTransferSizes();

  // Switching builder to innermost composite loop for the worst case of
  // epiloues. in case of composite loops, creation of implicit loops have no
  // meaning.
  if (perform_composite_store) {
    auto inner_loop =
        (*dsc_loops_to_mlir_loops_map_)[composite_loops.front().loop_].front();
    if (auto affine_for =
            llvm::dyn_cast<mlir::affine::AffineForOp>(inner_loop)) {
      builder.setInsertionPointToStart(affine_for.getBody());
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(inner_loop)) {
      builder.setInsertionPointToStart(scf_for.getBody());
    }
  }

  std::vector<dsc2::ScheduleNode::UnitView::LoopInfo> implicit_loops;
  if (!ctgs_transfer_sizes.empty()) {
    if (failed(constructImplicitLoopsForContiguousTransfer(
            builder, *view_sizes_core_specific,
            transfer_->unitTimeTransferChunkSize_, implicit_loops,
            ctgs_transfer_sizes))) {
      emitError("Unable to construct implicit loops for contiguous transfer");
      return LogicalResult::failure();
    } else {
      perform_composite_store =
          perform_composite_store && !areEpiloguesInLoops(implicit_loops);
      if (perform_composite_store) {
        composite_loops.insert(composite_loops.begin(), implicit_loops.begin(),
                               implicit_loops.end());
      } else {
        outer_loops.insert(outer_loops.begin(),
                           view_sizes.compositeLoops_.begin(),
                           view_sizes.compositeLoops_.end());
        outer_loops.insert(outer_loops.begin(), implicit_loops.begin(),
                           implicit_loops.end());
      }
    }
  } else {
    if (!perform_composite_store) {
      outer_loops.insert(outer_loops.begin(),
                         view_sizes.compositeLoops_.begin(),
                         view_sizes.compositeLoops_.end());
    }
  }

  bool update_insertion_loc =
      !perform_composite_store && !outer_loops.empty() &&
      (!implicit_loops.empty() || !composite_loops.empty());
  if (update_insertion_loc) {
    auto inner_loop =
        (*dsc_loops_to_mlir_loops_map_)[outer_loops.front().loop_].back();
    if (auto affine_for =
            llvm::dyn_cast<mlir::affine::AffineForOp>(inner_loop)) {
      loop_builder.setInsertionPointToStart(affine_for.getBody());
    } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(inner_loop)) {
      loop_builder.setInsertionPointToStart(scf_for.getBody());
    }
  }

  if (transfer_->replicationFactor_ > 1) {
    result_type = mlir::dataflow::utils::constructVectorType(
        dataflow::utils::getElementType(result_type),
        dataflow::utils::getNumElements(result_type) *
            transfer_->replicationFactor_);
    src_result_type = mlir::dataflow::utils::constructVectorType(
        dataflow::utils::getElementType(src_result_type),
        dataflow::utils::getNumElements(src_result_type) *
            transfer_->replicationFactor_);
  }

  Value data;
  bool is_constant_read = transfer_->src_.unit_ == CONSTANT;
  if (is_constant_read) {
    int cst_idx = transfer_->srcLdsAndLoopOffsets_.constantId_;
    auto &cst_info = dsc_->constantInfo_.at(cst_idx);
    DT_CHECK(
        cst_idx >= 0 &&
        "constant id should be non-negative when using with CONST container");
    if (GenerateConstantBitStreamAndShuffle(cst_info, loop_builder, data)
            .failed()) {
      emitError("Unable to create constant bitstream and shuffle");
      return LogicalResult::failure();
    }
  } else {
    auto receive_op = dataflow::ReceiveOp::create(
        loop_builder, loop_builder.getUnknownLoc(), src_result_type, from,
        loop_builder.getStringAttr(transfer_->name_));

    // Convert data if src precision and dst precision don't match.
    // example1: src_result_type (lxlu): <64xfp16>, dst_result_type (pe):
    // <64xfp32> for the above one, generate cast for pe since its
    // greater than a stick worth
    // example2: src_result_type (pe): <64xfp32>,
    // dst_result_type (sfp): <64xfp16> for the above one, don't generate cast
    // for sfp since its a worth of stick already.
    auto stick_size = 1024;  // bits
    unsigned dim_size_0 = 0;
    auto input_size = mlir::dataflow::utils::getDimSize(result_type, 0) *
                      dataflow::utils::getElementTypeBitWidth(result_type);
    if (src_result_type != result_type && input_size != stick_size) {
      auto tmp_data = receive_op.getResult();
      if (SNComputeLowering::constructPrecisionConversionOperation(
              loop_builder, tmp_data, dst_prec_, data, comp_)
              .failed()) {
        emitError(
            "Unable to construct precision conversion in a transfer operation");
      }
    } else {
      data = receive_op.getResult();
    }
  }

  // If destination is latch, hold the result of receive_op to a later use.
  if (dst_storage == LATCH) {
    int latch_id = transfer_->dstLdsAndLoopOffsets_[dst_idx].latchDataId_;
    DT_CHECK_MSG(latch_id != -1, "latch id cannot be negative");
    addToLatchMap(latch_id, data);
    return LogicalResult::success();
  }

  int mode = -1;
  if (failed(this->getBufferingOrStreamingMode(mode))) {
    emitError("Unable to get buffering or streaming mode");
    return LogicalResult::failure();
  }

  final_store_value = data;
  if (mode == 2 || mode == 1) {
    Operation *store_op = nullptr;
    if (failed(constructStreamingOrDoubleBufferingStore(
            loop_builder, dst_storage, data, outer_loops,
            perform_composite_store, composite_loops, store_op, mode))) {
      emitError("Unable to construct streaming store");
      return LogicalResult::failure();
    }
  } else {
    dataflow::GetLogicalMemoryViewOp view;
    AffineMap base_address_map;
    llvm::SmallVector<Value, 8> base_address_args;
    IntegerSet transfer_set, time_set;
    AffineMap transfer_order, time_order, time_addr_map;

    auto factor = getAddressGranularityMultiplyFactor(
        dst, dst_storage, dataflow::utils::getElementType(result_type));

    mlir::Value address;
    if (needsUniform()) {
      address = constructUniformizedFoldedAddress(
          loop_builder, transfer_->dstLdsAndLoopOffsets_[dst_idx].startAddr_,
          factor);
    } else {
      // expecting constant value across sdsc folds
      int start_addr =
          (int)(FoldInfraUtils::getSingleDataStrict(
                    transfer_->dstLdsAndLoopOffsets_[dst_idx].startAddr_,
                    {{0, core_id_}, {1, corelet_id_}}) *
                factor);
      auto const_op = mlir::arith::ConstantIndexOp::create(
          loop_builder, builder.getUnknownLoc(), start_addr);
      address = const_op.getResult();
    }

    mlir::Value store_val;
    OpBuilder composite_store_builder = loop_builder;
    if (!perform_composite_store) {
      if (failed(constructElementsOfAgenDataTransfer(
              loop_builder, dst_storage, true, address,
              *view_sizes_core_specific, transfer_->unitTimeTransferChunkSize_,
              transfer_->unitTimeTransferChunkStride_, outer_loops, view,
              base_address_map, base_address_args, transfer_set, transfer_order,
              is_constant_read))) {
        emitError("Unable to construct elements of agen data transfer");
        return LogicalResult::failure();
      }
    } else {
      if (failed(constructElementsOfAgenCompositeDataTransfer(
              loop_builder, dst_storage, true, address,
              *view_sizes_core_specific, transfer_->unitTimeTransferChunkSize_,
              transfer_->unitTimeTransferChunkStride_, outer_loops, view,
              base_address_map, base_address_args, transfer_set, transfer_order,
              composite_loops, time_set, time_order, time_addr_map,
              is_constant_read))) {
        emitError("Unable to construct elements of agen data transfer");
        return LogicalResult::failure();
      }

      auto composite_store = agen::CompositeStoreOp::create(
          loop_builder, builder.getUnknownLoc(), view.getResult(),
          loop_builder.getStringAttr(transfer_->name_), base_address_map,
          base_address_args, transfer_set, transfer_order, {}, time_set,
          time_order, time_addr_map);
      composite_store_builder.setInsertionPointToStart(
          composite_store.getBody());
    }

    if (dst == SenComponents::LXSU && transfer_->replicationFactor_ > 1) {
      if (auto shuffle_op = construct2B16BStoreShuffle(
              perform_composite_store ? composite_store_builder : loop_builder,
              data, result_type, transfer_->replicationFactor_)) {
        if (!perform_composite_store) {
          auto store_op = agen::VectorStoreOp::create(
              loop_builder, loop_builder.getUnknownLoc(),
              shuffle_op.getResult(), view.getResult(),
              loop_builder.getStringAttr(transfer_->name_), base_address_map,
              base_address_args, transfer_set, transfer_order);
        } else {
          auto yield_op = agen::YieldOp::create(
              composite_store_builder, composite_store_builder.getUnknownLoc(),
              shuffle_op.getResult());
        }
      } else {
        emitError("Unsupported store type.");
        return LogicalResult::failure();
      }
    } else {
      if (!perform_composite_store) {
        auto store_op = agen::VectorStoreOp::create(
            loop_builder, loop_builder.getUnknownLoc(), data, view.getResult(),
            loop_builder.getStringAttr(transfer_->name_), base_address_map,
            base_address_args, transfer_set, transfer_order);
      } else {
        auto *data_produce_op = data.getDefiningOp();
        auto *cloned_op = composite_store_builder.clone(*data_produce_op);
        composite_store_builder.getInsertionPoint()->insertOperands(
            0, {cloned_op->getResult(0)});
        data_produce_op->erase();
      }
    }
  }

  return LogicalResult::success();
}

// ---- 98/110  GenerateDataTranferForSrc  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2522  (34L)
LogicalResult e098_GenerateDataTranferForSrc(
    OpBuilder &builder, const dsc2::TransferNode::DstVia &dst) {
  SenComponents to_unit;
  SenComponents to_storage = SenComponents::NO_COMPONENT;
  if (dst.via_.empty()) {
    to_unit = dst.loc_.unit_;
    to_storage = dst.loc_.storage_;
  } else {
    to_unit = dst.via_.front();
  }

  if (comp_ != to_unit || to_unit == LXLU) {
    auto unit = (comp_ == to_unit && to_unit == LXLU ? LXLUSCALEREG : to_unit);
    auto dst_unit_op =
        retrieveGetUnitOpInSameCore(builder, unit, core_id_, corelet_id_);
    if (failed(GenerateLoadAndSendFromDataTransferNode(
            builder, dst_unit_op, comp_, transfer_->src_.storage_,
            src_sticks_ss_per_dim))) {
      emitError("Unable to generate load and send operations");
      return LogicalResult::failure();
    }
  } else {
    DT_CHECK(to_storage != SenComponents::NO_COMPONENT);
    int dst_idx = 0;
    if (failed(GenerateLoadAndStoreFromDataTransferNode(
            builder, transfer_->src_.storage_, to_storage, 0,
            src_sticks_ss_per_dim))) {
      emitError("Unable to generate load and send operations");
      return LogicalResult::failure();
    }
  }

  return LogicalResult::success();
}


// ================================================================================================
// LEVEL 5
// ================================================================================================

// ---- 99/110  constructMACOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:942  (87L)
LogicalResult e099_constructMACOperation(
    OpBuilder &builder, const dsc2::ComputeNode &compute_op) {
  if (compute_op.inputs_.size() != 3) {
    return LogicalResult::failure();
  }

  auto loc = builder.getUnknownLoc();

  // Assumption: TODO: All the inputs are worth of 1 stick (128 bytes).
  // Otherwise, we have to generate more than one compute operation.
  // Construct inputs
  llvm::SmallVector<mlir::Value, 1> inputs;
  auto formats = compute_op.getComputeOperandFormats(*dsc_);
  auto elements =
      compute_op.getComputeOperandSizes(dsc_global_.sysDef);
  Type element_type;
  for (int i = 0; i < compute_op.inputs_.size(); i++) {
    bool is_integer;
    Type result_type;
    DataFormats format = formats[i];
    LabeledDsInfo::ScaledLdsCategory category =
        LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR;
    if (compute_op.inputsLdsAndLoopOffsets_[i].myLdsIdx_ != -1) {
      format =
          dsc_->labeledDs_[compute_op.inputsLdsAndLoopOffsets_[i].myLdsIdx_]
              .dataFormat_;
      category =
          dsc_->labeledDs_[compute_op.inputsLdsAndLoopOffsets_[i].myLdsIdx_]
              .scaledLdsCategory_;
    }

    if (failed(constructTypeFromFormat(builder, format, category, comp_,
                                       elements[i], result_type, element_type,
                                       is_integer))) {
      emitError("Unable to construct MAC operation type");
      return LogicalResult::failure();
    }

    if (failed(constructComputeInputOperandAndAddToList(
            builder, loc, element_type, result_type, compute_op, inputs, i))) {
      emitError("Unable to construct MAC operation input operand");
      return LogicalResult::failure();
    }
  }

  // Construct FMA operation
  AffineMap reduction_map;
  if (failed(getReductionMapForMACOperation(builder, compute_op.type_,
                                            reduction_map))) {
    return LogicalResult::failure();
  }

  auto result_type = inputs[2].getType();
  mlir::vectorchain::CreateAffineMaskOp mask_op;

  if (compute_op.instrAttribute_.computeMaskLoopOffsets_.empty()) {
    mask_op =
        getStaticContinuousMaskValue(builder, loc, result_type, inputs,
                                     compute_op.instrAttribute_.compute_mask_);
  } else {
    auto gen_comp = EnumsConversion::senCompToGenericComp.at(comp_);
    DT_CHECK_MSG(gen_comp == PT, "Dynamic masking allowed only in PT units");
    int corelet_id = corelet_id_ == -1 ? 0 : corelet_id_;
    mask_op = constructDynamicMasking(
        builder, loc, result_type, inputs,
        compute_op.instrAttribute_.computeMaskLoopOffsets_.at(corelet_id));
  }
  auto input0 = inputs[0];
  if (compute_op.type_ == ComputeOpType::FNMS) {
    auto neg_input0 =
        vectorchain::NegOp::create(builder, loc, result_type, input0);
    input0 = neg_input0.getResult();
  }
  auto mac_op = vectorchain::MultiplyAndAccumulateOp::create(
      builder, loc, result_type, input0, inputs[1], inputs[2], mask_op,
      builder.getStringAttr(compute_op.name_), reduction_map);
  auto result = mac_op.getResult();

  // Construct output
  if (failed(constructComputeOutputOperands(builder, loc, result_type,
                                            compute_op, result))) {
    emitError("Unable to construct MAC operation output operand");
    return LogicalResult::failure();
  }

  return LogicalResult::success();
}

// ---- 100/110  constructBinaryOrTernaryOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1036  (152L)
LogicalResult e100_constructBinaryOrTernaryOperation(
    OpBuilder &builder, const dsc2::ComputeNode &compute_op) {
  if (!is_any_of(compute_op.inputs_.size(), 2, 3)) {
    return LogicalResult::failure();
  }

  auto loc = builder.getUnknownLoc();

  // Assumption: TODO: All the inputs are worth of 1 stick (128 bytes).
  // Otherwise, we have to generate more than one compute operation.

  // Construct inputs
  llvm::SmallVector<mlir::Value> inputs;
  for (int i = 0; i < compute_op.inputs_.size(); i++) {
    // Get the expected input type from compute operation
    Type element_type;
    Type result_type;
    DataFormats input_format = DataFormats::SEN169_FP16;
    int lds_idx = compute_op.inputsLdsAndLoopOffsets_[i].myLdsIdx_;
    if (lds_idx != -1) {
      input_format = dsc_->labeledDs_[lds_idx].dataFormat_;
    } else {
      input_format = compute_op.getComputeOperandFormats(*dsc_)[i];
    }

    if (failed(getTypeBasedOnComputeType(builder, input_format, result_type,
                                         element_type))) {
      return LogicalResult::failure();
    }

    // Construct input operand
    if (failed(constructComputeInputOperandAndAddToList(
            builder, loc, element_type, result_type, compute_op, inputs, i))) {
      emitError("Unable to construct Binary operation input operand");
      return LogicalResult::failure();
    }
  }

  // Construct expected result type from the contract.
  Value op_result;
  Type result_element_type;
  Type result_type;  // = inputs[0].getType();
  auto result_format = compute_op.getComputeOperandFormats(*dsc_).back();
  if (failed(getTypeBasedOnComputeType(builder, result_format, result_type,
                                       result_element_type))) {
    return LogicalResult::failure();
  }

  // Construct mask operation
  auto mask_op =
      getStaticContinuousMaskValue(builder, loc, result_type, inputs,
                                   compute_op.instrAttribute_.compute_mask_);
  DT_CHECK_MSG(mask_op, "Could not create valid mask");
  // Construct compute operation
  if (is_any_of(compute_op.type_, ComputeOpType::FMAX, ComputeOpType::FABSMAX,
                ComputeOpType::FMIN, ComputeOpType::FMUL, ComputeOpType::FSUB,
                ComputeOpType::OR, ComputeOpType::AND)) {
    // Construct binary operation
    mlir::vectorchain::VectorChainBinaryOperator op_name;
    if (compute_op.type_ == ComputeOpType::FMAX) {
      op_name = mlir::vectorchain::VectorChainBinaryOperator::max;
    } else if (compute_op.type_ == ComputeOpType::FMIN) {
      op_name = mlir::vectorchain::VectorChainBinaryOperator::min;
    } else if (compute_op.type_ == ComputeOpType::FABSMAX) {
      op_name = mlir::vectorchain::VectorChainBinaryOperator::abs_max;
    } else if (compute_op.type_ == ComputeOpType::FMUL) {
      if (compute_op.instrAttribute_.mode_ == 11) {
        // The operation is X*Y/2
        op_name = mlir::vectorchain::VectorChainBinaryOperator::mul_div2;
      } else {
        // The operation is X*Y
        op_name = mlir::vectorchain::VectorChainBinaryOperator::mul;
      }
    } else if (compute_op.type_ == ComputeOpType::FSUB) {
      op_name = mlir::vectorchain::VectorChainBinaryOperator::sub;
    } else if (compute_op.type_ == ComputeOpType::OR) {
      op_name = mlir::vectorchain::VectorChainBinaryOperator::or0;
    } else if (compute_op.type_ == ComputeOpType::AND) {
      op_name = mlir::vectorchain::VectorChainBinaryOperator::and0;
    }

    AffineMap op_specific_map = builder.getDimIdentityMap();
    auto binary_op = vectorchain::BinaryOp::create(
        builder, loc, result_type, inputs[0], inputs[1], mask_op,
        builder.getStringAttr(compute_op.name_), op_name, op_specific_map);
    op_result = binary_op.getResult();
  } else if (compute_op.type_ == ComputeOpType::PACKMERGE) {
    auto extend_attr =
        builder.getBoolAttr(compute_op.instrAttribute_.sign_extend_);
    auto indices_array_attr =
        builder.getI32ArrayAttr(compute_op.instrAttribute_.indices_);
    auto repetition_attr =
        builder.getIndexAttr(compute_op.instrAttribute_.repetition_);
    auto pack_merge_op = vectorchain::PackOp::create(
        builder, loc, result_type, inputs[0], inputs[1], mask_op,
        builder.getStringAttr(compute_op.name_), indices_array_attr,
        repetition_attr, extend_attr);
    op_result = pack_merge_op.getResult();
  } else if (compute_op.type_ == ComputeOpType::GREATERTHAN) {
    auto gt_operation = vectorchain::ElementWiseCompareOp::create(
        builder, loc, result_type, inputs[0], inputs[1], mask_op,
        builder.getStringAttr(compute_op.name_),
        vectorchain::VectorChainElementWiseCompareOperator::compare_gt);
    op_result = gt_operation.getResult();
  } else if (compute_op.type_ == ComputeOpType::GREATEREQUAL) {
    auto ge_operation = vectorchain::ElementWiseCompareOp::create(
        builder, loc, result_type, inputs[0], inputs[1], mask_op,
        builder.getStringAttr(compute_op.name_),
        vectorchain::VectorChainElementWiseCompareOperator::compare_ge);
    op_result = ge_operation.getResult();
  } else if (compute_op.type_ == ComputeOpType::LESSERTHAN) {
    auto le_operation = vectorchain::ElementWiseCompareOp::create(
        builder, loc, result_type, inputs[0], inputs[1], mask_op,
        builder.getStringAttr(compute_op.name_),
        vectorchain::VectorChainElementWiseCompareOperator::compare_lt);
    op_result = le_operation.getResult();
  } else if (compute_op.type_ == ComputeOpType::LESSEREQUAL) {
    auto lt_operation = vectorchain::ElementWiseCompareOp::create(
        builder, loc, result_type, inputs[0], inputs[1], mask_op,
        builder.getStringAttr(compute_op.name_),
        vectorchain::VectorChainElementWiseCompareOperator::compare_le);
    op_result = lt_operation.getResult();
  } else if (compute_op.type_ == ComputeOpType::EQUALTO) {
    auto eq_operation = vectorchain::ElementWiseCompareOp::create(
        builder, loc, result_type, inputs[0], inputs[1], mask_op,
        builder.getStringAttr(compute_op.name_),
        vectorchain::VectorChainElementWiseCompareOperator::compare_eq);
    op_result = eq_operation.getResult();
  } else if (compute_op.type_ == ComputeOpType::NOTEQUAL) {
    auto neq_operation = vectorchain::ElementWiseCompareOp::create(
        builder, loc, result_type, inputs[0], inputs[1], mask_op,
        builder.getStringAttr(compute_op.name_),
        vectorchain::VectorChainElementWiseCompareOperator::compare_neq);
    op_result = neq_operation.getResult();
  } else if (compute_op.type_ == ComputeOpType::SELECT) {
    auto select_operation = vectorchain::ElementWiseSelectionOp::create(
        builder, loc, result_type, inputs[0], inputs[1], inputs[2], mask_op,
        builder.getStringAttr(compute_op.name_));
    op_result = select_operation.getResult();
  } else {
    return LogicalResult::failure();
  }

  // Construct output operands
  if (failed(constructComputeOutputOperands(builder, loc, result_type,
                                            compute_op, op_result))) {
    emitError("Unable to construct Binary operation output operand");
    return LogicalResult::failure();
  }

  return LogicalResult::success();
}

// ---- 101/110  constructUnaryOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1276  (247L)
LogicalResult e101_constructUnaryOperation(
    OpBuilder &builder, const dsc2::ComputeNode &compute_op) {
  if (compute_op.inputs_.size() != 1) {
    return LogicalResult::failure();
  }

  auto loc = builder.getUnknownLoc();

  Type input_type;
  Type input_element_type;
  DataFormats input_format = DataFormats::SEN169_FP16;
  int input_lds_idx = compute_op.inputsLdsAndLoopOffsets_[0].myLdsIdx_;
  if (input_lds_idx != -1) {
    input_format = dsc_->labeledDs_[input_lds_idx].dataFormat_;
  } else {
    input_format = compute_op.getComputeOperandFormats(*dsc_)[0];
  }

  if (failed(getTypeBasedOnComputeType(builder, input_format, input_type,
                                       input_element_type))) {
    emitError("Unable to get type");
    return LogicalResult::failure();
  }

  llvm::SmallVector<mlir::Value> inputs;
  if (failed(constructComputeInputOperandAndAddToList(
          builder, loc, input_element_type, input_type, compute_op, inputs,
          0))) {
    emitError("Unable to construct unary operation input operand");
    return LogicalResult::failure();
  }

  Value unary_op_result;
  Type unary_op_result_type = input_type;
  // VectorType mask_result_type = mlir::cast<VectorType>(inputs[0].getType());
  auto mask_op =
      getStaticContinuousMaskValue(builder, loc, unary_op_result_type, inputs,
                                   compute_op.instrAttribute_.compute_mask_);
  DT_CHECK_MSG(mask_op, "Could not create valid mask");
  auto context = builder.getContext();
  if (compute_op.type_ == ComputeOpType::FEST) {
    if (compute_op.instrAttribute_.mode_ == 0) {
      // Construct exp_a estimate operation
      auto unary_op = vectorchain::ExpEstimateOp::create(
          builder, loc, unary_op_result_type, inputs[0], mask_op,
          builder.getStringAttr(compute_op.name_),
          mlir::vectorchain::VectorChainExpEstimate::a);
      unary_op_result = unary_op.getResult();
    } else if (compute_op.instrAttribute_.mode_ == 1) {
      // Construct exp_b estimate operation
      auto unary_op = vectorchain::ExpEstimateOp::create(
          builder, loc, unary_op_result_type, inputs[0], mask_op,
          builder.getStringAttr(compute_op.name_),
          mlir::vectorchain::VectorChainExpEstimate::b);
      unary_op_result = unary_op.getResult();
    } else if (compute_op.instrAttribute_.mode_ == 2) {
      // Construct rec estimate operation
      auto unary_op = vectorchain::RecEstimateOp::create(
          builder, loc, unary_op_result_type, inputs[0], mask_op,
          builder.getStringAttr(compute_op.name_));
      unary_op_result = unary_op.getResult();
    } else if (compute_op.instrAttribute_.mode_ == 3) {
      // Construct ln estimate operation
      auto unary_op = vectorchain::LnEstimateOp::create(
          builder, loc, unary_op_result_type, inputs[0], mask_op,
          builder.getStringAttr(compute_op.name_));
      unary_op_result = unary_op.getResult();
    } else if (compute_op.instrAttribute_.mode_ == 5) {
      // Construct rsqrt estimate operation
      auto unary_op = vectorchain::RsqrtEstimateOp::create(
          builder, loc, unary_op_result_type, inputs[0], mask_op,
          builder.getStringAttr(compute_op.name_));
      unary_op_result = unary_op.getResult();
    } else if (compute_op.instrAttribute_.mode_ == 6) {
      auto slope_attr = vectorchain::VectorChainEstimateVersionsAttr::get(
          builder.getContext(),
          mlir::vectorchain::VectorChainEstimateVersions::slope);
      // Construct sigmoid_slope estimate operation
      auto unary_op = vectorchain::SigmoidEstimateOp::create(
          builder, loc, unary_op_result_type, inputs[0], mask_op,
          builder.getStringAttr(compute_op.name_), slope_attr);
      unary_op_result = unary_op.getResult();
    } else if (compute_op.instrAttribute_.mode_ == 7) {
      auto offset_attr = vectorchain::VectorChainEstimateVersionsAttr::get(
          builder.getContext(),
          mlir::vectorchain::VectorChainEstimateVersions::offset);
      // Construct sigmoid_offset estimate operation
      auto unary_op = vectorchain::SigmoidEstimateOp::create(
          builder, loc, unary_op_result_type, inputs[0], mask_op,
          builder.getStringAttr(compute_op.name_), offset_attr);
      unary_op_result = unary_op.getResult();
    } else if (compute_op.instrAttribute_.mode_ == 8) {
      auto slope_attr = vectorchain::VectorChainEstimateVersionsAttr::get(
          builder.getContext(),
          mlir::vectorchain::VectorChainEstimateVersions::slope);
      // Construct tanh_slope estimate operation
      auto unary_op = vectorchain::TanhEstimateOp::create(
          builder, loc, unary_op_result_type, inputs[0], mask_op,
          builder.getStringAttr(compute_op.name_), slope_attr);
      unary_op_result = unary_op.getResult();
    } else if (compute_op.instrAttribute_.mode_ == 9) {
      auto offset_attr = vectorchain::VectorChainEstimateVersionsAttr::get(
          builder.getContext(),
          mlir::vectorchain::VectorChainEstimateVersions::offset);
      // Construct tanh_offset estimate operation
      auto unary_op = vectorchain::TanhEstimateOp::create(
          builder, loc, unary_op_result_type, inputs[0], mask_op,
          builder.getStringAttr(compute_op.name_), offset_attr);
      unary_op_result = unary_op.getResult();
    } else {
      emitError("Unknown estimate instruction.");
      return LogicalResult::failure();
    }
  } else if (compute_op.type_ == ComputeOpType::ICVT) {
    if (compute_op.instrAttribute_.mode_ == 7) {
      // Construct fast exp operation
      auto unary_op = vectorchain::FastExpOp::create(
          builder, loc, unary_op_result_type, inputs[0], mask_op,
          builder.getStringAttr(compute_op.name_));
      unary_op_result = unary_op.getResult();
    } else {
      emitError("Unknown ICVT instruction.");
      return LogicalResult::failure();
    }
  } else if (compute_op.type_ == ComputeOpType::FLOOR) {
    // Construct floor
    auto unary_op = vectorchain::FloorOp::create(
        builder, loc, unary_op_result_type, inputs[0], mask_op,
        builder.getStringAttr(compute_op.name_));
    unary_op_result = unary_op.getResult();
  } else if (compute_op.type_ == ComputeOpType::SPLAT) {
    // Construct shuffle operation
    mlir::vectorchain::ShuffleOp unary_op;
    int repetition = compute_op.instrAttribute_.repetition_;
    if (compute_op.instrAttribute_.sign_extend_ == 0) {
      if (input_element_type.isF16()) {
        int index_array[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        unary_op = vectorchain::ShuffleOp::create(
            builder, builder.getUnknownLoc(), unary_op_result_type, inputs[0],
            nullptr, builder.getI32ArrayAttr(index_array),
            builder.getI32IntegerAttr(repetition),
            builder.getStringAttr(compute_op.name_));
      } else if (input_element_type.isF32()) {
        int index_array[4] = {0, 0, 0, 0};
        unary_op = vectorchain::ShuffleOp::create(
            builder, builder.getUnknownLoc(), unary_op_result_type, inputs[0],
            nullptr, builder.getI32ArrayAttr(index_array),
            builder.getI32IntegerAttr(repetition),
            builder.getStringAttr(compute_op.name_));
      }
    } else if (compute_op.instrAttribute_.sign_extend_ == 1) {
      auto zero = mlir::arith::ConstantOp::create(
          builder, builder.getUnknownLoc(),
          builder.getIntegerAttr(builder.getIndexType(), 0));
      if (input_element_type.isF16()) {
        int index_array[8] = {0, -1, -1, -1, -1, -1, -1, -1};
        unary_op = vectorchain::ShuffleOp::create(
            builder, builder.getUnknownLoc(), unary_op_result_type, inputs[0],
            ValueRange{}, ValueRange{zero.getResult()},
            builder.getI32ArrayAttr(index_array),
            builder.getI32IntegerAttr(repetition),
            builder.getStringAttr(compute_op.name_));
      } else if (input_element_type.isF32()) {
        int index_array[4] = {0, -1, -1, -1};
        unary_op = vectorchain::ShuffleOp::create(
            builder, builder.getUnknownLoc(), unary_op_result_type, inputs[0],
            ValueRange{}, ValueRange{zero.getResult()},
            builder.getI32ArrayAttr(index_array),
            builder.getI32IntegerAttr(repetition),
            builder.getStringAttr(compute_op.name_));
      }
    } else {
      emitError("sign_extend has to be either 0 or 1.");
      return LogicalResult::failure();
    }
    unary_op_result = unary_op.getResult();
  } else if (compute_op.type_ == ComputeOpType::REDUCE) {
    mlir::vectorchain::VectorChainBinaryOperator bin_operator;
    if (compute_op.instrAttribute_.mode_ == 1) {
      bin_operator = mlir::vectorchain::VectorChainBinaryOperator::add;
    } else if (compute_op.instrAttribute_.mode_ == 8) {
      bin_operator = mlir::vectorchain::VectorChainBinaryOperator::max;
    } else if (compute_op.instrAttribute_.mode_ == 10) {
      bin_operator = mlir::vectorchain::VectorChainBinaryOperator::abs_max;
    } else if (compute_op.instrAttribute_.mode_ == 12) {
      bin_operator = mlir::vectorchain::VectorChainBinaryOperator::min;
    } else if (compute_op.instrAttribute_.mode_ == 14) {
      bin_operator = mlir::vectorchain::VectorChainBinaryOperator::abs_min;
    } else {
      emitError("Unknown binary operation for reduction.");
      return LogicalResult::failure();
    }
    llvm::APInt gap(64, 8);  // gap = 8 : i64
    auto eval_order =
        mlir::vectorchain::VectorChainScanOpEvalOrders::left_to_right;
    // Construct scan_with_gap operation
    auto unary_op = vectorchain::ScanWithGapOp::create(
        builder, loc, unary_op_result_type, inputs[0],
        builder.getStringAttr(compute_op.name_), bin_operator, gap, eval_order);
    unary_op_result = unary_op.getResult();
  } else if (compute_op.type_ == ComputeOpType::SHUFFLE) {
    VectorType output_type;
    Type output_element_type;
    DataFormats output_format = DataFormats::SEN169_FP16;
    int output_lds_idx = compute_op.outputsLdsAndLoopOffsets_[0].myLdsIdx_;
    if (output_lds_idx != -1) {
      output_format = dsc_->labeledDs_[output_lds_idx].dataFormat_;
    } else {
      output_format = compute_op.getComputeOperandFormats(*dsc_)[1];
    }

    if (failed(getTypeBasedOnComputeType(builder, output_format, output_type,
                                         output_element_type))) {
      emitError("Unable to get type");
      return LogicalResult::failure();
    }

    unary_op_result_type = output_type;
    auto indices_array_attr =
        builder.getI32ArrayAttr(compute_op.instrAttribute_.indices_);
    // In case there is zero padding in the pattern, pass in zero constant to
    // pad operand.
    // TODO: Confirm this covers all padding types and operand combinations we
    // will produce.
    auto zero = mlir::arith::ConstantOp::create(
        builder, builder.getUnknownLoc(),
        builder.getIntegerAttr(builder.getIndexType(), 0));
    auto unary_op = vectorchain::ShuffleOp::create(
        builder, builder.getUnknownLoc(), unary_op_result_type, inputs[0],
        ValueRange{}, ValueRange{zero.getResult()}, indices_array_attr,
        builder.getI32IntegerAttr(compute_op.instrAttribute_.repetition_),
        builder.getStringAttr(compute_op.name_));
    unary_op_result = unary_op.getResult();
  } else {
    emitError("Unknown unary instruction.");
    return LogicalResult::failure();
  }

  // Construct output
  if (failed(constructComputeOutputOperands(builder, loc, unary_op_result_type,
                                            compute_op, unary_op_result))) {
    emitError("Unable to construct Unary operation output operand");
    return LogicalResult::failure();
  }

  return LogicalResult::success();
}

// ---- 102/110  GenerateDataTranferForDst  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2563  (47L)
LogicalResult e102_GenerateDataTranferForDst(
    int dst_idx, OpBuilder &builder, const dsc2::TransferNode::DstVia &dst,
    std::map<SenComponents, std::set<SenComponents>> &dst_forward_map,
    std::set<std::pair<SenComponents, SenComponents>> &explored_pairs_for_via,
    bool perform_el_codegen) {
  SenComponents from;
  if (dst.via_.empty()) {
    from = transfer_->src_.unit_;
  } else {
    from = dst.via_.back();
  }

  if (from != comp_) {
    auto src_unit_op =
        retrieveGetUnitOpInSameCore(builder, from, core_id_, corelet_id_);

    mlir::Value final_store_data;
    if (failed(GenerateReceiveAndStoreFromDataTransferNode(
            dst_idx, builder, src_unit_op, comp_, dst.loc_.storage_,
            src_sticks_ss_per_dim, final_store_data))) {
      emitError("Unable to generate receive and store operations");
      return LogicalResult::failure();
    }

    auto forward_record = dst_forward_map.find(comp_);
    if (forward_record != dst_forward_map.end()) {
      mlir::Operation *final_store_operation = final_store_data.getDefiningOp();
      DT_CHECK_MSG(
          final_store_operation,
          "store operation should be visible for forwarding to other units");

      OpBuilder forward_builder(final_store_operation);
      forward_builder.setInsertionPointAfter(final_store_operation);
      for (auto &dst_fwd : forward_record->second) {
        auto dst_unit_op = retrieveGetUnitOpInSameCore(builder, dst_fwd,
                                                       core_id_, corelet_id_);
        auto send_op = dataflow::SendOp::create(
            forward_builder, final_store_operation->getLoc(), dst_unit_op,
            final_store_data,
            /*dir*/ nullptr, forward_builder.getStringAttr(transfer_->name_));
        explored_pairs_for_via.emplace(from, dst_fwd);
      }
    }
  }

  return LogicalResult::success();
}


// ================================================================================================
// LEVEL 6
// ================================================================================================

// ---- 103/110  constructComputeOperation  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNComputeLowering.cpp:1567  (83L)
LogicalResult e103_constructComputeOperation(OpBuilder &builder) {
  if (comp_ == compute->exUnit_) {
    auto &compute_node = compute;
    if (is_any_of(compute_node->type_, ComputeOpType::IMA8, ComputeOpType::IMA4,
                  ComputeOpType::FMA4, ComputeOpType::FMA8,
                  ComputeOpType::FMA16, ComputeOpType::FMA32,
                  ComputeOpType::FNMS)) {
      if (failed(constructMACOperation(builder, *compute_node))) {
        emitError("Unable to construct MAC operation");
        return LogicalResult::failure();
      }

      precision =
          DSC2ToDataflowIR::stringifyComputePrecision(compute_node->type_);
      for (int i = 0; i < compute_node->inputs_.size(); i++) {
        LabeledDsInfo::ScaledLdsCategory category =
            LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR;
        if (compute_node->inputsLdsAndLoopOffsets_[i].myLdsIdx_ != -1) {
          category = dsc_->labeledDs_[compute_node->inputsLdsAndLoopOffsets_[i]
                                          .myLdsIdx_]
                         .scaledLdsCategory_;
          if (category != LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR) {
            precision = "mx" + precision;
            break;
          }
        }
      }
    } else if (is_any_of(compute_node->type_, ComputeOpType::FMUL,
                         ComputeOpType::FSUB, ComputeOpType::PACKMERGE,
                         ComputeOpType::FABSMAX, ComputeOpType::GREATERTHAN,
                         ComputeOpType::GREATEREQUAL, ComputeOpType::LESSERTHAN,
                         ComputeOpType::LESSEREQUAL, ComputeOpType::EQUALTO,
                         ComputeOpType::NOTEQUAL, ComputeOpType::SELECT,
                         ComputeOpType::OR, ComputeOpType::AND)) {
      if (failed(constructBinaryOrTernaryOperation(builder, *compute_node))) {
        emitError("Unable to construct binary operation");
        return LogicalResult::failure();
      }
    } else if (is_any_of(compute_node->type_, ComputeOpType::FMAX,
                         ComputeOpType::FMIN)) {
      if (failed(constructFMINorFMAXOperation(builder, *compute_node))) {
        emitError("Unable to construct binary operation");
        return LogicalResult::failure();
      }
    } else if (is_any_of(compute_node->type_, ComputeOpType::FEST,
                         ComputeOpType::ICVT, ComputeOpType::SPLAT,
                         ComputeOpType::REDUCE, ComputeOpType::SHUFFLE,
                         ComputeOpType::FLOOR)) {
      if (failed(constructUnaryOperation(builder, *compute_node))) {
        emitError("Unable to construct unary operation");
        return LogicalResult::failure();
      }
    } else if (is_any_of(
                   compute_node->type_, ComputeOpType::RECIPROCAL,
                   ComputeOpType::LAYERNORMSCALE, ComputeOpType::GELU,
                   ComputeOpType::SIGMOID, ComputeOpType::EXP_P1,
                   ComputeOpType::EXP_P2, ComputeOpType::EXP,
                   ComputeOpType::LOG_P1, ComputeOpType::LOG_P2,
                   ComputeOpType::REALDIV, ComputeOpType::GELU_BWD_P1,
                   ComputeOpType::GELU_BWD_P2, ComputeOpType::SQRT,
                   ComputeOpType::RSQRT, ComputeOpType::MISH_P1,
                   ComputeOpType::MISH_P2, ComputeOpType::LAYERNORMSCALE32,
                   ComputeOpType::EXX2_32_P1, ComputeOpType::EXX2_32_P2,
                   ComputeOpType::EXX2_32_P3, ComputeOpType::DL16TOFP32,
                   ComputeOpType::FP32TODL16, ComputeOpType::DL16TOBF16,
                   ComputeOpType::SOFTPLUS_P1, ComputeOpType::SOFTPLUS_P2,
                   ComputeOpType::IDX32TOADDR, ComputeOpType::MUL_I32_TO_I32,
                   ComputeOpType::ADD_I32_TO_I32,
                   ComputeOpType::ADD_I64_TO_I64)) {
      if (failed(constructOpaqueOperation(builder, *compute_node))) {
        emitError(
            "Unable to construct operation for Opaque "
            "function");
        return LogicalResult::failure();
      }
    } else {
      emitError("Unknown compute operation in constructComputeOperation");
      return LogicalResult::failure();
    }
  }

  return LogicalResult::success();
}

// ---- 104/110  constructDataTransfer  --  dsc-based-utils/DSC2ToDataflowIR/V3/SNTransferLowering.cpp:2676  (164L)
LogicalResult e104_constructDataTransfer(
    mlir::OpBuilder &builder) {
  const auto &src_ld_idx = transfer_->srcLdsAndLoopOffsets_.myLdsIdx_;
  LabeledDsInfo::ScaledLdsCategory src_category =
      LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR;

  if (src_ld_idx == -1) {
    DT_CHECK_MSG(transfer_->srcLdsAndLoopOffsets_.constantId_ >= 0,
                 "transfer src should either have labeled ds or it has to be a "
                 "constant");
    src_prec_ =
        dsc_->constantInfo_.at(transfer_->srcLdsAndLoopOffsets_.constantId_)
            .dataFormat_;
  } else {
    src_prec_ = dsc_->labeledDs_[src_ld_idx].dataFormat_;
    src_category = dsc_->labeledDs_[src_ld_idx].scaledLdsCategory_;

    // Use specific scalared lds category only for L0LU.
    if (transfer_->src_.unit_ != L0LUROW0) {
      src_category = LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR;
    }
  }

  if (failed(getLabeledDsType(src_prec_, src_category, builder, result_type))) {
    return LogicalResult::failure();
  }

  src_result_type = result_type;

  for (auto &dst_lds : transfer_->dstLdsAndLoopOffsets_) {
    int dst_ld_idx = dst_lds.myLdsIdx_;
    if (dst_ld_idx == -1) {
      DT_CHECK_MSG(
          dst_lds.constantId_ >= 0,
          "transfer src should either have labeled ds or it has to be a "
          "constant");
      dst_prec_ = dsc_->constantInfo_.at(dst_lds.constantId_).dataFormat_;
    } else
      dst_prec_ = dsc_->labeledDs_[dst_ld_idx].dataFormat_;

    if (failed(getLabeledDsType(
            dst_prec_, LabeledDsInfo::ScaledLdsCategory::REGULAR_TENSOR,
            builder, dst_result_type))) {
      return LogicalResult::failure();
    }
  }

  // Temporary fix to get blocks.
  // TODO: Needs to be fixed for imbalanced work split
  int corelet_id = corelet_id_;
  if (this->uniformization_enabled_ && corelet_id == -1) {
    corelet_id = 0;
  }

  src_sticks_ss_per_dim = dsc_->getBlockTransferSizePerDim(
      *transfer_, comp_, corelet_id, false, true);
  src_sticks_el_per_dim = dsc_->getBlockTransferSizePerDim(
      *transfer_, comp_, corelet_id, true, true);

  for (auto &it : src_sticks_ss_per_dim) {
    sticks_src_ss *= it.second;
  }

  for (auto &it : src_sticks_el_per_dim) {
    sticks_src_el *= it.second;
  }

  if (sticks_src_ss <= 0 || sticks_src_el <= 0) {
    emitError("Negative number of block transfers");
    return LogicalResult::failure();
  }

  if (transfer_->src_.unit_ == comp_) {
    std::set<SenComponents> explored_dst_units;
    for (auto &dst : transfer_->dstVias_) {
      OpBuilder tmp_builder = builder;

      // update the tmp_builder to lastFusableParentSrc_ if exists.
      bool is_memory = is_any_of(comp_, LXLU, LXSU, L0LUROW0, L0SU);
      if (transfer_->lastFusableParentLoopSrc_ != nullptr && is_memory &&
          !areEpiloguesInTransferSizes()) {
        auto *loop_op = this->dsc_loops_to_mlir_loops_map_
                            ->at(transfer_->lastFusableParentLoopSrc_)
                            .front();
        tmp_builder.setInsertionPoint(loop_op);
      }

      SenComponents to_unit =
          dst.via_.empty() ? dst.loc_.unit_ : dst.via_.front();
      if (explored_dst_units.find(to_unit) == explored_dst_units.end()) {
        explored_dst_units.emplace(to_unit);
      } else {
        continue;
      }

      if (failed(GenerateDataTranferForSrc(tmp_builder, dst))) {
        emitError("Unable to generate data transfer operations");
        return LogicalResult::failure();
      }
    }
  }

  // result_type is the major type for construction of loads/stores, hence
  // it needs to be updated correctly.
  result_type = dst_result_type;
  std::map<SenComponents, std::set<SenComponents>> dst_forward_map;
  for (const auto &dst : transfer_->dstVias_) {
    if (dst.loc_.unit_ == comp_) {
      for (const auto &dst_fwd : transfer_->dstVias_) {
        // TODO: What if the src is destination comp_ itself?
        DT_CHECK(transfer_->src_.unit_ != comp_ ||
                 dst.loc_.storage_ == LXLUSCALEREG);
        for (int i = 0; i < dst_fwd.via_.size(); i++) {
          if (dst_fwd.via_[i] == comp_) {
            if (i < dst_fwd.via_.size() - 1) {
              dst_forward_map[comp_].emplace(dst_fwd.via_[i + 1]);
            } else {
              dst_forward_map[comp_].emplace(dst_fwd.loc_.unit_);
            }
          }
        }
      }
    }
  }

  // Iterate through each destination, if the same data is sent
  // We assume these data transfers are already sorted
  std::set<std::pair<SenComponents, SenComponents>> explored_pairs_for_via;
  for (int dst_idx = 0; dst_idx < transfer_->dstVias_.size(); dst_idx++) {
    const auto &dst = transfer_->dstVias_[dst_idx];
    if (dst.loc_.unit_ == comp_) {
      OpBuilder tmp_builder = builder;

      // update the tmp_builder to lastFusableParentSrc_ if exists.
      bool is_memory = is_any_of(comp_, LXLU, LXSU, L0LUROW0, L0SU);
      if (!transfer_->lastFusableParentLoopDst_.empty() &&
          transfer_->lastFusableParentLoopDst_[dst_idx] != nullptr &&
          is_memory && !areEpiloguesInTransferSizes()) {
        auto *loop_op = this->dsc_loops_to_mlir_loops_map_
                            ->at(transfer_->lastFusableParentLoopDst_[dst_idx])
                            .front();
        tmp_builder.setInsertionPoint(loop_op);
      }

      if (failed(GenerateDataTranferForDst(
              dst_idx, tmp_builder, dst, dst_forward_map,
              explored_pairs_for_via, sticks_src_ss != sticks_src_el))) {
        emitError("Unable to generate data transfer operations");
        return LogicalResult::failure();
      }

      break;
    }
  }

  for (const auto &dst : transfer_->dstVias_) {
    OpBuilder tmp_builder = builder;
    if (failed(GenerateDataTransfersForViaIfSo(
            transfer_->src_, dst, explored_pairs_for_via, tmp_builder)))
      return LogicalResult::failure();
  }

  return LogicalResult::success();
}


// ================================================================================================
// LEVEL 7
// ================================================================================================

// ---- 105/110  constructOperationsRecursively  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:55  (165L)
LogicalResult e105_constructOperationsRecursively(
    OpBuilder builder, SNDSCLowering &dsc_lowering,
    std::vector<const dsc2::ScheduleNode *> sched_nodes,
    std::string &precision) {
  // Iterate through each operation
  // We assume the operations within a record (location) are already sorted.
  for (const auto *sched_node : sched_nodes) {
    // Already constructed loops
    if (sched_node->nodeType_ == dsc2::ScheduleNode::LOOP) {
      OpBuilder tmp_builder(builder);
      const auto *node = static_cast<const dsc2::LoopNode *>(sched_node);
      auto *mlir_bottom_loop =
          (*dsc_lowering.getDSCLoopsToMLIRloops())[node].back();
      if (auto affine_loop =
              llvm::dyn_cast<mlir::affine::AffineForOp>(mlir_bottom_loop)) {
        tmp_builder.setInsertionPointToStart(affine_loop.getBody());
      } else if (auto scf_loop = llvm::dyn_cast<scf::ForOp>(mlir_bottom_loop)) {
        tmp_builder.setInsertionPointToStart(scf_loop.getBody());
      } else {
        llvm_unreachable("unknown for-op");
      }

      auto children = node->getNextView(dsc_lowering.getComponent(),
                                        dsc_lowering.getCoreletId(),
                                        dsc_lowering.getCoreId());
      auto result = constructOperationsRecursively(tmp_builder, dsc_lowering,
                                                   children, precision);
      auto *mlir_top_loop =
          (*dsc_lowering.getDSCLoopsToMLIRloops())[node].front();
      builder.setInsertionPointAfter(mlir_top_loop);
    } else if (sched_node->nodeType_ == dsc2::ScheduleNode::CONDITION) {
      const auto *node = static_cast<const dsc2::ConditionNode *>(sched_node);
      auto children = node->getNextView(dsc_lowering.getComponent(),
                                        dsc_lowering.getCoreletId(),
                                        dsc_lowering.getCoreId());
      DT_CHECK_MSG(!children.empty(), "if-regions should not be empty");
      if (!node->hasCoreClCond()) {  // condition is not on core/corelet.
        mlir::scf::IfOp if_op = llvm::dyn_cast<mlir::scf::IfOp>(
            *(dsc_lowering.getDSCConddsToMLIRConds())->at(node));

        builder.setInsertionPointToStart(&if_op.getThenRegion().front());
        OpBuilder endif_builder(if_op->getNextNode());
        bool has_else_branch = children.size() == 2;
        dsc2::LoopCondComposite condition = node->loopCond_;

        // builder points to then region of the if operation.
        auto then_result = constructOperationsRecursively(
            builder, dsc_lowering, {children[0]}, precision);

        if (has_else_branch) {
          OpBuilder else_builder(if_op);
          else_builder.setInsertionPointToStart(&if_op.getElseRegion().front());
          auto else_result = constructOperationsRecursively(
              else_builder, dsc_lowering, {children[1]}, precision);
        }

        builder = endif_builder;
      } else {
        DT_CHECK(node->loopCond_.twoLevelOrOfAnds_.empty());
        if (!this->uniformization_) {
          // no uniformization
          DT_CHECK(children.size() == 1);

          // creating a dummy op to reset the builder location for subsequent
          // siblings
          auto dummy_op = mlir::arith::ConstantIntOp::create(
              builder, builder.getUnknownLoc(), 1, 1);

          OpBuilder tmp_builder(dummy_op);
          if (failed(constructOperationsRecursively(
                  tmp_builder, dsc_lowering, {children[0]}, precision))) {
            emitError("Unable to construct conditionals on conditionals.");
            return LogicalResult::failure();
          }

          builder.setInsertionPointAfter(dummy_op);
          dummy_op.erase();
        } else {
          int num_regions = children.size();
          DT_CHECK_MSG(1 <= num_regions && num_regions <= 2,
                       "either true/false on coreClConditions");

          mlir::uniform::UniformizeRegionsOp uniform_region =
              llvm::dyn_cast<mlir::uniform::UniformizeRegionsOp>(
                  *(dsc_lowering.getDSCConddsToMLIRConds())->at(node));

          int index_offset = 0;
          if (uniform_region.getNumRegions() < children.size()) {
            bool has_then_region =
                !node->getThenCoreCl(dsc_lowering.getComponent()).empty();
            bool has_else_region =
                !node->getElseCoreCl(dsc_lowering.getComponent()).empty();

            if (!has_then_region && has_else_region) index_offset = 1;
          }
          for (int i = 0; i < uniform_region.getNumRegions(); i++) {
            OpBuilder builder_region(uniform_region.getRegion(i));

            auto region_arg = uniform_region.getRegionArg(i);
            dsc_lowering.setUniformRegionArg(region_arg);

            if (failed(constructOperationsRecursively(
                    builder_region, dsc_lowering, {children[i + index_offset]},
                    precision))) {
              emitError("Unable to construct conditionals on conditionals.");
              return LogicalResult::failure();
            }

            dsc_lowering.resetUniformRegionArg();
          }

          builder.setInsertionPointAfter(uniform_region);
        }
      }
    } else if (sched_node->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      const auto *node = static_cast<const dsc2::ComputeNode *>(sched_node);
      SNComputeLowering compute_lowering(dsc_lowering, node, precision);
      if (failed(compute_lowering.constructComputeOperation(builder))) {
        emitError("Unable to construct a compute operation.");
        return LogicalResult::failure();
      }

      precision = compute_lowering.getPrecision();

    } else if (sched_node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      const auto *node = static_cast<const dsc2::TransferNode *>(sched_node);
      SNTransferLowering transfer_lowering(dsc_lowering, node);
      if (failed(transfer_lowering.constructDataTransfer(builder))) {
        emitError("Unable to construct a data transfer operation.");
        return LogicalResult::failure();
      }
    } else if (sched_node->nodeType_ == dsc2::ScheduleNode::SYNC) {
      const auto *node = static_cast<const dsc2::SyncNode *>(sched_node);
      SNSyncLowering sync_lowering(dsc_lowering, node);
      if (failed(sync_lowering.constructSyncOperation(builder))) {
        emitError("Unable to construct a sync operation.");
        return LogicalResult::failure();
      }
    } else if (sched_node->nodeType_ == dsc2::ScheduleNode::STICKMASK) {
      const auto *node = static_cast<const dsc2::StickMaskNode *>(sched_node);
      SNStickMaskLowering stick_mask_lowering(dsc_lowering, node);
      if (failed(stick_mask_lowering.constructStickMaskOperation(builder))) {
        emitError("Unable to construct a stick mask operation.");
        return LogicalResult::failure();
      }
    } else if (sched_node->nodeType_ == dsc2::ScheduleNode::BLOCK) {
      const auto *node = static_cast<const dsc2::BlockNode *>(sched_node);
      auto children = node->getNextView(dsc_lowering.getComponent(),
                                        dsc_lowering.getCoreletId(),
                                        dsc_lowering.getCoreId());
      auto mlir_dummy_loop = (*dsc_lowering.getDSCBlocksToMLIRloops())[node];
      if (failed(constructOperationsRecursively(builder, dsc_lowering, children,
                                                precision))) {
        emitError("Unable to construct a block of operations.");
        return LogicalResult::failure();
      }

      builder.setInsertionPointAfter(mlir_dummy_loop);
      (*dsc_lowering.getDSCBlocksToMLIRloops()).erase(node);
      mlir_dummy_loop.erase();
    }
  }

  return LogicalResult::success();
}


// ================================================================================================
// LEVEL 8
// ================================================================================================

// ---- 106/110  ConstructAProgramUnit  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:293  (81L)
void e106_ConstructAProgramUnit(DesignSpaceConfig &dsc,
                                             int core_idx, int core_id,
                                             int corelet_id,
                                             SenComponents &comp,
                                             dsc_based_utils::DSCSymbolInfo &sym_info) {
  auto roots =
      dsc.scheduleTree_.getHead()->getNextView(comp, corelet_id, core_id);
  if (roots.empty()) return;

  // Build Dataflow.GetUnit and ProgramUnit corresponding to the comp
  dataflow::ProgramUnitOp unit_op;
  Value program_unit_iterator;
  llvm::SmallVector<Value> units_involved;

  std::vector<int> core_ids_used(1, core_id);
  std::vector<int> corelet_ids_used(1, corelet_id);

  // compute number of folds by multiplying size of each fold dimension.
  this->num_folds_ = 1;
  for (int i = 0; i < sdsc_->sdscFolds_.getNumDims(); i++) {
    this->num_folds_ *= sdsc_->sdscFolds_.getFoldDimSize(i);
  }

  if (num_folds_ == 1) {
    unit_op = initializeUnit(core_id, corelet_id, comp, dsc_global_);
    units_involved.emplace_back(unit_op.getUnits()[0]);
    program_unit_iterator = unit_op.getRegion().getArgument(0);
  } else {
    unit_op = initializeUniformizedUnit(comp, core_ids_used, corelet_ids_used,
                                        program_unit_iterator, &units_involved,
                                        dsc_global_);
  }

  // Construct loops
  std::map<const dsc2::LoopNode *, std::vector<Operation *>>
      dsc_loops_to_mlir_loops_map;
  std::map<const dsc2::BlockNode *, mlir::affine::AffineForOp>
      dsc_blocks_to_mlir_loops_map;
  std::map<const dsc2::BlockNode *, std::vector<const dsc2::TransferNode *>>
      dsc_all_parent_loops_to_buffers_switch_map;
  std::map<const dsc2::ConditionNode *, Operation *>
      dsc_conds_to_mlir_conds_map;

  std::map<int, Value> global_latch_map;
  SNDSCLowering dsc_lowering(
      dsc_global_, &dsc, &module_op_, false, core_ids_used,
      corelet_ids_used, program_unit_iterator, &units_involved,
      &sen_components_, &component_to_handler_, &unit_to_value_map_, &unit_op,
      core_id, corelet_id, num_folds_, comp, &global_latch_map,
      &dsc_loops_to_mlir_loops_map, &dsc_blocks_to_mlir_loops_map,
      &dsc_all_parent_loops_to_buffers_switch_map,
      &dsc_conds_to_mlir_conds_map);

  SNControlFlowLowering loop_lowering(dsc_lowering, sym_info);
  if (failed(loop_lowering.constructLoops())) {
    module_op_->emitError("Unable to construct loops.");
    terminate();
  }

  // If there are no loops, please return to the next unit
  if (dsc_loops_to_mlir_loops_map.empty()) return;

  // Set the builder location
  OpBuilder builder(unit_op);
  builder.setInsertionPoint(dsc_loops_to_mlir_loops_map[nullptr].front());

  // Construct transfers, computations, and sync operations together.
  std::string precision = "";
  if (failed(constructOperationsRecursively(builder, dsc_lowering, {roots},
                                            precision))) {
    terminate();
  }

  // Set the unit precision attribute for only PT units.
  auto record = EnumsConversion::senCompToGenericComp.find(comp);
  if (record != EnumsConversion::senCompToGenericComp.end()) {
    if (is_any_of(record->second, PT)) {
      setPrecisionInUnitOp(comp, unit_op, precision);
    }
  }
}

// ---- 107/110  ConstructAUniformizedProgramUnit  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:378  (84L)
void e107_ConstructAUniformizedProgramUnit(
    DesignSpaceConfig &dsc, SenComponents &comp, dsc_based_utils::DSCSymbolInfo &sym_info) {
  auto roots = dsc.scheduleTree_.getHead()->getNextView(comp);

  if (roots.empty()) return;

  // compute number of folds by multiplying size of each fold dimension.
  this->num_folds_ = 1;
  for (int i = 0; i < sdsc_->sdscFolds_.getNumDims(); i++) {
    this->num_folds_ *= sdsc_->sdscFolds_.getFoldDimSize(i);
  }

  // Skip folding if each fold is identical.
  if (this->num_folds_ > 1) {
    if (!areFoldsNeeded(dsc, comp)) {
      this->num_folds_ = 1;
      // std::cout << "Setting folds to = 1 for " <<
      // EnumsConversion::senComponentsToString.at(comp) << std::endl;
    } else {
      // std::cout << dsc.name_ << " : Setting folds to > 1 for "
      //           << EnumsConversion::senComponentsToString.at(comp)
      //           << std::endl;
    }
  }

  // Build Dataflow.GetUnit ops and ProgramUnit corresponding to the comp
  Value program_unit_iterator;
  llvm::SmallVector<Value> units_involved;

  std::vector<int> corelet_ids_used(1, 0);
  if (dsc.numCoreletsUsed_ == 2) corelet_ids_used.emplace_back(1);

  auto unit_op = initializeUniformizedUnit(
      comp, dsc.coreIdsUsed_, corelet_ids_used, program_unit_iterator,
      &units_involved, dsc_global_);

  // Construct loops
  std::map<const dsc2::LoopNode *, std::vector<Operation *>>
      dsc_loops_to_mlir_loops_map;
  std::map<const dsc2::BlockNode *, mlir::affine::AffineForOp>
      dsc_blocks_to_mlir_loops_map;
  std::map<const dsc2::BlockNode *, std::vector<const dsc2::TransferNode *>>
      dsc_all_parent_loops_to_buffers_switch_map;
  std::map<const dsc2::ConditionNode *, Operation *>
      dsc_conds_to_mlir_conds_map;

  std::map<int, mlir::Value> global_latch_map;
  SNDSCLowering dsc_lowering(
      dsc_global_, &dsc, &module_op_, true, dsc.coreIdsUsed_,
      corelet_ids_used, program_unit_iterator, &units_involved,
      &sen_components_, &component_to_handler_, &unit_to_value_map_, &unit_op,
      -1, -1, num_folds_, comp, &global_latch_map, &dsc_loops_to_mlir_loops_map,
      &dsc_blocks_to_mlir_loops_map,
      &dsc_all_parent_loops_to_buffers_switch_map,
      &dsc_conds_to_mlir_conds_map);

  SNControlFlowLowering loop_lowering(dsc_lowering, sym_info);
  if (failed(loop_lowering.constructLoops())) {
    module_op_->emitError("Unable to construct loops.");
    terminate();
  }

  // If there are no loops, please return to the next unit
  if (dsc_loops_to_mlir_loops_map.empty()) return;

  // Set the builder location
  OpBuilder builder(unit_op);
  builder.setInsertionPoint(dsc_loops_to_mlir_loops_map[nullptr].front());

  // Construct transfers, computations, and sync operations together.
  std::string precision = "";
  if (failed(constructOperationsRecursively(builder, dsc_lowering, {roots},
                                            precision))) {
    terminate();
  }

  // Set the unit precision attribute for only PT units.
  auto record = EnumsConversion::senCompToGenericComp.find(comp);
  if (record != EnumsConversion::senCompToGenericComp.end()) {
    if (is_any_of(record->second, PT)) {
      setPrecisionInUnitOp(comp, unit_op, precision);
    }
  }
}


// ================================================================================================
// LEVEL 9
// ================================================================================================

// ---- 108/110  convertV3  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:466  (29L)
void e108_convertV3() {
  // Setup the dataflow IR
  this->startDataflowIRGeneration();

  int num_corelets = 2;

  // Iterate through each DSC2.0
  DT_CHECK(!sen_components_.empty());
  for (auto &dsc : sdsc_->dscs_) {
    // Generate the symbol information for the DSC once.
    dsc_based_utils::DSCSymbolInfo sym_info(&dsc, &sdsc_->symbolDefinitions_);

    // Iterate through each Core
    for (int core_idx = 0; core_idx < dsc.coreIdsUsed_.size(); core_idx++) {
      int core_id = dsc.coreIdsUsed_[core_idx];

      // Iterate through each corelet
      for (int corelet_id = 0; corelet_id < num_corelets; corelet_id++) {
        // Iterate through each component
        for (auto comp : sen_components_) {
          ConstructAProgramUnit(dsc, core_idx, core_id, corelet_id, comp,
                                sym_info);
        }
      }
    }
  }

  this->stopDataflowIRGeneration();
}

// ---- 109/110  convertV4  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:499  (33L)
void e109_convertV4() {
  // Setup the dataflow IR
  this->startDataflowIRGeneration();

  int num_corelets = 2;

  // Iterate through each DSC2.0
  DT_CHECK(!sen_components_.empty());
  for (auto &dsc : sdsc_->dscs_) {
    // Generate the symbol information for the DSC once.
    dsc_based_utils::DSCSymbolInfo sym_info(&dsc, &sdsc_->symbolDefinitions_);

    // Iterate through each component
    for (auto comp : sen_components_) {
      ConstructAUniformizedProgramUnit(dsc, comp, sym_info);
    }

    // TODO: Temporary fix in presence of DCG generated L3 code syncing
    // with both corelets, but dsc2.0 mentioning use of a single corelet.
    if (dsc.numCoreletsUsed_DSC2_ != num_corelets) {
      // Iterate through each Core
      for (int core_id : dsc.coreIdsUsed_) {
        // Iterate through each component
        for (auto comp : sen_components_) {
          ConstructAProgramUnit(dsc, core_id, core_id, 1 /*corelet_id*/, comp,
                                sym_info);
        }
      }
    }
  }

  this->stopDataflowIRGeneration();
}


// ================================================================================================
// LEVEL 10
// ================================================================================================

// ---- 110/110  runTranslator  --  dsc-based-utils/DSC2ToDataflowIR/DSC2ToDataflowIR.cpp:534  (32L)
LogicalResult e110_runTranslator() {
  // Initialize the sen_components_ vector according to senarch.
  // TODO: Is there some global config we can grab this list from?
  if (dsc_global_.sysDef.coreArch >= IsaCoreGen::SEN1P5_ISA) {
    sen_components_ = {PTROW0, PTROW1,   PTROW2, PTROW3, PE,
                       SFP,    L0LUROW0, L0SU,   LXLU,   LXSU};
  } else {
    sen_components_ = {PTROW0, PTROW1, PTROW2, PTROW3,   PTROW4, PTROW5, PTROW6,
                       PTROW7, PE,     SFP,    L0LUROW0, L0SU,   LXLU,   LXSU};
  }

  int translator_version = -1;
  if (failed(getTranslatorVersion(*sdsc_, translator_version))) {
    return LogicalResult::failure();
  }

  if (translator_version == 2) {
    DT_ERROR("Translator version 2 is deprecated and removed");
  } else if (translator_version == 3) {
    if (this->uniformization_) {
      // std::cout << "uniformization is enabled." << std::endl;
      this->convertV4();
    } else {
      // std::cout << "uniformization is disabled." << std::endl;
      this->convertV3();
    }
  } else {
    return LogicalResult::failure();
  }

  return LogicalResult::success();
}

