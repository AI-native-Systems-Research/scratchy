// ================================================================================================
// DDC / L3-SCHEDULER CAMPAIGN - consolidated translation unit: ddl
//
// STAGE 2's DDL conversion sub-surface, reached from Ddc
//   dbo/src/Utils/sdsc_bundle/SchedulerStages.cpp:29-41 is the call site.
//
// 44 of the campaign's 382 function bodies, from ddc/ddl/, extracted VERBATIM
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
// entry 165/382   level 0   scc 354   10 body lines
// unit: e165_initialize
// authority: ddc/ddl/Dialect/DdlOps.cpp:33
// original: void DdlDialect::initialize()
// class: DdlDialect
// rust home: crates/compiler/deeptools/src/schedule/ddl/ops.rs
// ------------------------------------------------------------------------------------------------
void e165_initialize()
{
  addOperations<
#define GET_OP_LIST
#include "Dialect/DdlOps.cpp.inc"
      >();
  addAttributes<
#define GET_ATTRDEF_LIST
#include "Dialect/DdlOpsAttributes.cpp.inc"
      >();
}

// ------------------------------------------------------------------------------------------------
// entry 166/382   level 0   scc 355   30 body lines
// unit: e166_verify
// authority: ddc/ddl/Dialect/DdlOps.cpp:59
// original: LogicalResult ForceInnermostDimensionsOp::verify()
// class: ForceInnermostDimensionsOp
// rust home: crates/compiler/deeptools/src/schedule/ddl/ops.rs
// ------------------------------------------------------------------------------------------------
LogicalResult e166_verify()
{
  if (!isa<AllocateOp>(getAllocate().getDefiningOp())) {
    this->emitError("Input is not an allocation we can control");
    return failure();
  }
  if (!isa<DatastageOp, GetExternalDatastageOp>(
          getDatastage().getDefiningOp())) {
    this->emitError("Input is not a datastage");
    return failure();
  }
  auto dims = getDims();
  if (dims.empty()) {
    this->emitError("Provide at least one dimension");
    return failure();
  }
  std::unordered_set<mlir::Value> uniqueDims;
  for (auto dim : getDims()) {
    if (!isa<DimensionOp, PaddedDimensionOp>(dim.getDefiningOp())) {
      this->emitError("Input is not a dimension");
      return failure();
    }
    if (!uniqueDims.insert(dim).second) {
      this->emitError(
          "Multiple instances of the same dimension are not currently "
          "supported");
      return failure();
    }
  }
  return success();
}

// ------------------------------------------------------------------------------------------------
// entry 167/382   level 0   scc 356   25 body lines
// unit: e167_verify
// authority: ddc/ddl/Dialect/DdlOps.cpp:109
// original: LogicalResult ComputeOp::verify()
// class: ComputeOp
// rust home: crates/compiler/deeptools/src/schedule/ddl/ops.rs
// ------------------------------------------------------------------------------------------------
LogicalResult e167_verify()
{
  auto opFunc = this->getComputetype().upper();
  auto compute_op_checks = [&](size_t inputs) {
    if (this->getInputs().size() != inputs) {
      this->emitError(opFunc + " requires exactly " + std::to_string(inputs) +
                      " inputs");
      return failure();
    }
    return success();
  };
  if (is_any_of(opFunc, "MACC", "FMA16", "FMA8", "FMA4", "IMA8", "IMA4", "FNMS",
                "SELECT", "FMA32")) {
    return compute_op_checks(3);
  } else if (is_any_of(opFunc, "REDUCE", "SPLAT", "ICVT", "FEST", "SHR",
                       "SHUFFLE", "ASSIGN", "FLOOR")) {
    return compute_op_checks({1});
  } else if (is_any_of(opFunc, "PACKMERGE", "FMUL", "FSUB", "FMAX", "FMIN",
                       "FABSMAX", "GCVT", "IME", "GREATERTHAN", "LESSERTHAN",
                       "EQUAL", "NOTEQUAL", "GREATEREQUAL", "LESSEREQUAL", "OR",
                       "AND")) {
    return compute_op_checks(2);
  }
  this->emitError("Unrecognized compute op " + opFunc);
  return failure();
}

// ------------------------------------------------------------------------------------------------
// entry 168/382   level 0   scc 357   16 body lines
// unit: e168_verify
// authority: ddc/ddl/Dialect/DdlOps.cpp:149
// original: LogicalResult DatatypeOp::verify()
// class: DatatypeOp
// rust home: crates/compiler/deeptools/src/schedule/ddl/ops.rs
// ------------------------------------------------------------------------------------------------
LogicalResult e168_verify()
{
  auto data_type = FromString<DataFormats>(this->getDataType().str());
  auto type_size_it = EnumsConversion::dataFormatsToBitWidth.find(data_type);
  if (type_size_it == EnumsConversion::dataFormatsToBitWidth.end()) {
    this->emitError("can not find this type: " + this->getDataType().str());
    return failure();
  }
  auto type_size = type_size_it->second;
  if (type_size > this->getBitWidth().value_or(100)) {
    this->emitError("bit_width(" + std::to_string(this->getBitWidth().value()) +
                    ") should be greater or equal than size(" +
                    std::to_string(type_size) + ") of type");
    return failure();
  }
  return success();
}

// ------------------------------------------------------------------------------------------------
// entry 169/382   level 0   scc 359   7 body lines
// unit: e169_verify
// authority: ddc/ddl/Dialect/DdlOps.cpp:195
// original: LogicalResult ImplicitSyncOp::verify()
// class: ImplicitSyncOp
// rust home: crates/compiler/deeptools/src/schedule/ddl/ops.rs
// ------------------------------------------------------------------------------------------------
LogicalResult e169_verify()
{
  if (!isa<AllocateOp>(getAllocate().getDefiningOp())) {
    emitError("Input should be an allocate op");
    return failure();
  }
  return success();
}

// ------------------------------------------------------------------------------------------------
// entry 170/382   level 0   scc 360   72 body lines
// unit: e170_parse
// authority: ddc/ddl/Dialect/DdlOps.cpp:220
// original: ::mlir::ParseResult AllocateOp::parse(::mlir::OpAsmParser& parser, ::mlir::OperationState& result)
// class: AllocateOp
// rust home: crates/compiler/deeptools/src/schedule/ddl/ops.rs
// ------------------------------------------------------------------------------------------------
::mlir::ParseResult e170_parse(::mlir::OpAsmParser& parser,
                                      ::mlir::OperationState& result)
{
  ::mlir::OpAsmParser::UnresolvedOperand tensorRawOperands[1];
  ::llvm::ArrayRef<::mlir::OpAsmParser::UnresolvedOperand> tensorOperands(
      tensorRawOperands);
  ::llvm::SMLoc tensorOperandsLoc;
  (void)tensorOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4>
      padding_dimOperands;
  ::llvm::SMLoc padding_dimOperandsLoc;
  (void)padding_dimOperandsLoc;
  ::llvm::SmallVector<::mlir::OpAsmParser::UnresolvedOperand, 4>
      external_allocateOperands;
  ::llvm::SMLoc external_allocateOperandsLoc;
  (void)external_allocateOperandsLoc;
  if (parser.parseLParen()) return ::mlir::failure();

  bool commaNeedsFurtherParsing = false;
  tensorOperandsLoc = parser.getCurrentLocation();
  if (parser.parseOperand(tensorRawOperands[0])) return ::mlir::failure();
  if (::mlir::succeeded(parser.parseOptionalComma())) {
    if (::mlir::succeeded(parser.parseOptionalLSquare())) {
      padding_dimOperandsLoc = parser.getCurrentLocation();
      if (parser.parseOperandList(padding_dimOperands))
        return ::mlir::failure();
      if (parser.parseRSquare()) return ::mlir::failure();
    } else {
      commaNeedsFurtherParsing = true;
    }
  }
  // In case, we have seen a `,` but no `[$padding_dim]`, try to parse the rest
  // as the `external_allocate` argument. No need to look for another `,`. In
  // case, the parsed `,` was followed by `[$padding_dim]`, look for a new `,`
  // that might start the `external_allocate` argument.
  if (commaNeedsFurtherParsing ||
      ::mlir::succeeded(parser.parseOptionalComma())) {
    {
      external_allocateOperandsLoc = parser.getCurrentLocation();
      ::mlir::OpAsmParser::UnresolvedOperand operand;
      ::mlir::OptionalParseResult parseResult =
          parser.parseOptionalOperand(operand);
      if (parseResult.has_value()) {
        if (failed(*parseResult)) return ::mlir::failure();
        external_allocateOperands.push_back(operand);
      }
    }
  }

  if (parser.parseRParen()) return ::mlir::failure();
  {
    auto loc = parser.getCurrentLocation();
    (void)loc;
    if (parser.parseOptionalAttrDict(result.attributes))
      return ::mlir::failure();
  }
  result.addAttribute(
      "operandSegmentSizes",
      parser.getBuilder().getDenseI32ArrayAttr(
          {1, static_cast<int32_t>(padding_dimOperands.size()),
           static_cast<int32_t>(external_allocateOperands.size())}));
  ::mlir::Type odsBuildableType0 = parser.getBuilder().getIndexType();
  result.addTypes(odsBuildableType0);
  if (parser.resolveOperands(tensorOperands, odsBuildableType0,
                             tensorOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(padding_dimOperands, odsBuildableType0,
                             padding_dimOperandsLoc, result.operands))
    return ::mlir::failure();
  if (parser.resolveOperands(external_allocateOperands, odsBuildableType0,
                             external_allocateOperandsLoc, result.operands))
    return ::mlir::failure();
  return ::mlir::success();
}

// ------------------------------------------------------------------------------------------------
// entry 171/382   level 0   scc 341   22 body lines
// unit: e171_performActions
// authority: ddc/ddl/ddl.cpp:36
// original: OwningOpRef<Operation*> performActions( const std::shared_ptr<llvm::SourceMgr>& sourceMgr, MLIRContext* context)
// rust home: crates/compiler/deeptools/src/schedule/ddl/mod.rs
// ------------------------------------------------------------------------------------------------
OwningOpRef<Operation*> e171_performActions(
    const std::shared_ptr<llvm::SourceMgr>& sourceMgr, MLIRContext* context)
{
  // Disable multi-threading when parsing the input file. This removes the
  // unnecessary/costly context synchronization when parsing.
  bool wasThreadingEnabled = context->isMultithreadingEnabled();
  context->disableMultithreading();

  // Prepare the parser config, and attach any useful/necessary resource
  // handlers. Unhandled external resources are treated as passthrough, i.e.
  // they are not processed and will be emitted directly to the output
  // untouched.
  PassReproducerOptions reproOptions;
  FallbackAsmResourceMap fallbackResourceMap;
  ParserConfig parseConfig(context, /*verifyAfterParse=*/true,
                           &fallbackResourceMap);
  reproOptions.attachResourceParser(parseConfig);

  // Parse the input file and reset the context threading state.
  OwningOpRef<Operation*> op =
      parseSourceFileForTool(sourceMgr, parseConfig, true);
  context->enableMultithreading(wasThreadingEnabled);
  return op;
}

// ------------------------------------------------------------------------------------------------
// entry 172/382   level 0   scc 339   24 body lines
// unit: e172_processTypes
// authority: ddc/ddl/ddl_conversion.cpp:447
// original: std::vector<const DdlInterface::TypeDefinition*> DdlConversion::processTypes( mlir::OperandRange supportedTypes, mlir::Operation* userOp)
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
std::vector<const DdlInterface::TypeDefinition*> e172_processTypes(
    mlir::OperandRange supportedTypes, mlir::Operation* userOp)
{
  std::vector<const DdlInterface::TypeDefinition*> types;
  for (const auto& typeValue : supportedTypes) {
    auto& type = ddlInterface.type_definition_[typeValue];
    if (type.dataFormat_ == DataFormats::INVALID) {  // not yet parsed
      auto type_op = dyn_cast<DatatypeOp>(typeValue.getDefiningOp());
      if (!type_op) {
        type_op.emitError("This op should be a type op but is not");
        userOp->emitError("Used by this op");
        DT_ERROR("Illegal ddl");
      }
      type.dataFormat_ = FromString<DataFormats>(type_op.getDataType().str());
      if (type.dataFormat_ == DataFormats::INVALID) {
        type_op.emitError("Invalid type name:");
        DT_ERROR("Illegal ddl");
      }

      type.bitSize_ = type_op.getBitWidth().value_or(
          EnumsConversion::dataFormatsToBitWidth.at(type.dataFormat_));
    }
    types.push_back(&type);
  }
  return types;
}

// ------------------------------------------------------------------------------------------------
// entry 173/382   level 0   scc 327   107 body lines
// unit: e173_addInternalTensor
// authority: ddc/ddl/ddl_conversion.cpp:473
// original: LabeledDsInfo& DdlConversion::addInternalTensor(const LabeledDsInfo& refLds, int computeOpIdx)
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
LabeledDsInfo& e173_addInternalTensor(const LabeledDsInfo& refLds,
                                                int computeOpIdx)
{
  // add new lds for local tensor
  // keep track of current pointers to fix them later
  std::vector<LabeledDsInfo*> oldLdsPtrs;
  for (auto& lds : dsc.labeledDs_) {
    oldLdsPtrs.push_back(&lds);
  }
  // create new lds
  LabeledDsInfo newLds;
  newLds.ldsIdx_ = dsc.labeledDs_.size() - 1;
  newLds.dsName_ = "internal_tensor_lds" + std::to_string(newLds.ldsIdx_);
  newLds.dsType_ = refLds.dsType_;
  newLds.segment_ = LdsSegment::STACK;
  newLds.isFirstUse_ = true;
  newLds.scale_ = refLds.scale_;
  newLds.density_ = refLds.density_;
  newLds.wordLength = refLds.wordLength;
  newLds.dataFormat_ = refLds.dataFormat_;
  newLds.hbmSize_ = 0;
  newLds.lxSize_ = 0;
  newLds.lxBufferSize_ = 0;
  // update idx of last lds
  int oldLastLdsIdx = dsc.labeledDs_.back().ldsIdx_;
  int newLastLdsIdx = ++(dsc.labeledDs_.back().ldsIdx_);
  newLds.referenceLdsIdx_ = refLds.ldsIdx_;
  for (auto& [ssa, tensorDef] : ddlInterface.tensor_definition_) {
    if (tensorDef.ldsIdx_ == oldLastLdsIdx) {
      tensorDef.ldsIdx_++;
      break;
    }
  }
  // insert just before the last lds
  auto* newLdsPtr =
      &*dsc.labeledDs_.insert(dsc.labeledDs_.end() - 1, std::move(newLds));
  // fix pointers around the dsc
  std::unordered_map<LabeledDsInfo*, LabeledDsInfo*> oldNewLdsPtrs;
  for (int i = 0; i < oldLdsPtrs.size() - 1; i++) {
    oldNewLdsPtrs[oldLdsPtrs.at(i)] = &dsc.labeledDs_.at(i);
  }
  oldNewLdsPtrs[oldLdsPtrs.back()] = &dsc.labeledDs_.back();
  for (auto& lds : dsc.labeledDs_) {
    for (auto& dt : lds.dataTransfers_) {
      dt.myDsInfo = oldNewLdsPtrs.at(dt.myDsInfo);
    }
  }
  for (auto& co : dsc.computeOp_) {
    for (auto& lds : co.inputLabeledDs) {
      lds = oldNewLdsPtrs.at(lds);
    }
    for (auto& lds : co.interimLabeledDs) {
      lds = oldNewLdsPtrs.at(lds);
    }
    for (auto& lds : co.outputLabeledDs) {
      lds = oldNewLdsPtrs.at(lds);
    }
  }
  dsc.computeOp_.at(computeOpIdx).interimLabeledDs.push_back(newLdsPtr);

  // Use newLds for all childs of start node
  for (auto& child : dsc.scheduleTree_.traverseTreeDFSMutable(
           nullptr, {dsc2::ScheduleNode::ALLOCATE, dsc2::ScheduleNode::TRANSFER,
                     dsc2::ScheduleNode::COMPUTE})) {
    if (child->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      auto cn = static_cast<dsc2::ComputeNode*>(child);
      if (cn->isOpaqueOp_) {
        auto& ldsIdx = metadata_.opaqueOps_.at(cn).ldsIdx_;
        if (ldsIdx == oldLastLdsIdx) ldsIdx = newLastLdsIdx;
      }
      for (auto& in : cn->inputsLdsAndLoopOffsets_) {
        if (in.myLdsIdx_ == oldLastLdsIdx) in.myLdsIdx_ = newLastLdsIdx;
      }
      for (auto& out : cn->outputsLdsAndLoopOffsets_) {
        if (out.myLdsIdx_ == oldLastLdsIdx) out.myLdsIdx_ = newLastLdsIdx;
      }
    } else if (child->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      auto tn = static_cast<dsc2::TransferNode*>(child);
      if (tn->srcLdsAndLoopOffsets_.myLdsIdx_ == oldLastLdsIdx)
        tn->srcLdsAndLoopOffsets_.myLdsIdx_ = newLastLdsIdx;
      for (auto& out : tn->dstLdsAndLoopOffsets_) {
        if (out.myLdsIdx_ == oldLastLdsIdx) out.myLdsIdx_ = newLastLdsIdx;
      }
    } else if (child->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
      auto an = static_cast<dsc2::AllocateNode*>(child);
      if (an->ldsIdx_ == oldLastLdsIdx) an->ldsIdx_ = newLastLdsIdx;
    } else {
      DT_ERROR("the node has to be either compute, transfer or allocate.");
    }
  }
  for (auto& compAlloc : metadata_.newAllocations_) {
    auto& ldsToAlloc = compAlloc.second.ldsIdxAndAllocNode;
    if (auto kv = ldsToAlloc.extract(oldLastLdsIdx)) {
      kv.key() = newLastLdsIdx;
      ldsToAlloc.insert(std::move(kv));
    }
    for (auto& compToAlloc : compAlloc.second.compAndAllocNode) {
      if (compToAlloc.second->ldsIdx_ == oldLastLdsIdx)
        compToAlloc.second->ldsIdx_ = newLastLdsIdx;
    }
  }
  if (auto kv = metadata_.prefilledExternalTransferToDataConnectToFill_.extract(
          {oldLastLdsIdx, LX})) {
    kv.key().first = newLastLdsIdx;
    metadata_.prefilledExternalTransferToDataConnectToFill_.insert(
        std::move(kv));
  }
  return *newLdsPtr;
}

// ------------------------------------------------------------------------------------------------
// entry 174/382   level 0   scc 324   10 body lines
// unit: e174_processExpression
// authority: ddc/ddl/ddl_conversion.cpp:2072
// original: float DdlConversion::processExpression(std::string expr)
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
float e174_processExpression(std::string expr)
{
  char* remainingString;
  float val = std::strtof(expr.c_str(), &remainingString);
  if (remainingString != expr.c_str()) {
    // check remaining
    while (std::isspace(*remainingString)) remainingString++;
    if (*remainingString == '\0') return val;
  }
  DT_ERROR("Unable to convert expression to number: " + expr);
}

// ------------------------------------------------------------------------------------------------
// entry 175/382   level 0   scc 325   26 body lines
// unit: e175_getTensorProp
// authority: ddc/ddl/ddl_conversion.cpp:2083
// original: const DdlInterface::TensorProp& DdlInterface::getTensorProp(Value tensorSSA)
// class: DdlInterface
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
const DdlInterface::TensorProp& e175_getTensorProp(Value tensorSSA)
{
  if (!tensor_definition_.count(tensorSSA)) {
    auto alias_op = dyn_cast<AliasOneTensorOfOp>(tensorSSA.getDefiningOp());
    if (!alias_op) {
      tensorSSA.getDefiningOp()->emitError("Not a valid tensor");
      DT_ERROR("Illegal ddl");
    }
    bool found = false;
    for (auto tensor : alias_op.getTensors()) {
      auto refTensorDefIt = tensor_definition_.find(tensor);
      if (refTensorDefIt != tensor_definition_.end()) {
        if (found) {
          alias_op.emitError("Multiple tensors are active");
          DT_ERROR("Illegal ddl");
        }
        found = true;
        tensor_definition_[tensorSSA] = refTensorDefIt->second;
      }
    }
    if (!found) {
      alias_op.emitError("No tensor is active");
      DT_ERROR("Illegal ddl");
    }
  }
  return tensor_definition_.at(tensorSSA);
}

// ------------------------------------------------------------------------------------------------
// entry 176/382   level 0   scc 115   20 body lines
// unit: e176_getTensor
// authority: ddc/ddl/ddl_conversion.cpp:2825
// original: Value DdlConversion::getTensor(SenComponents unit, const dsc2::DataInfo dtinfo) const
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
Value e176_getTensor(SenComponents unit,
                               const dsc2::DataInfo dtinfo) const
{
  Value tensor0 = nullptr;
  if (dtinfo.myLdsIdx_ > -1) {
    for (auto kv : ddlInterface.tensor_definition_) {
      if (kv.second.ldsIdx_ == dtinfo.myLdsIdx_) {
        tensor0 = kv.first;
      }
    }
  } else if (dtinfo.constantId_ > -1) {
    for (auto kv : ddlInterface.ext_constant_definition_) {
      if (kv.second == dtinfo.constantId_) {
        tensor0 = kv.first;
      }
    }
  } else {
    for (auto kv : ddlInterface.operand_constant_tensor_)
      if (kv.second == unit) tensor0 = kv.first;
  }
  return tensor0;
}

// ------------------------------------------------------------------------------------------------
// entry 177/382   level 0   scc 331   23 body lines
// unit: e177_dump
// authority: ddc/ddl/ddl_conversion.cpp:3509
// original: void DdlInterface::DimProp::dump(std::string msg) const
// class: DimProp
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
void e177_dump(std::string msg) const
{
  if (!msg.empty()) {
    llvm::errs() << "\n" << msg;
    std::string headerLine(msg.length(), '-');
    llvm::errs() << "\n" << headerLine;
  }
  llvm::errs() << "\nDimProp: " << "\n  PrimaryDimTypes = ";
  if (dim_ == PrimaryDimTypes::PrimaryDimTypesCount) {
    llvm::errs() << "Invalid";
  } else {
    llvm::errs() << EnumsConversion::primaryDimToString.at(dim_);
  }
  llvm::errs() << "\n  DropDim         = " << (dropDim_ ? "T" : "F")
               << "\n  NonPaddedDim    = " << nonPaddedDim
               << "\n  MetaDimKind     = "
               << EnumsConversion::metaDimKindToString.at(metaDimKind_);
  llvm::errs() << "\n  dimCandidates   = [";
  for (auto dc : dimCandidates_) {
    llvm::errs() << " " << EnumsConversion::primaryDimToString.at(dc);
  }
  llvm::errs() << "]\n  NumRefsInGlobalLayouts = " << numRefsInGlobalLayouts_
               << "\n";
}

// ------------------------------------------------------------------------------------------------
// entry 178/382   level 0   scc 202   8 body lines
// unit: e178_getAccessPattern
// authority: ddc/ddl/ddl_conversion.cpp:3619
// original: ddc::Metadata::TransferAccessPatternType ddc::Metadata::DataTransfer::getAccessPattern(PrimaryDimTypes dimVal)
// class: DataTransfer
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
ddc::Metadata::TransferAccessPatternType
e178_getAccessPattern(PrimaryDimTypes dimVal)
{
  if (accessPatternPerDim_.count(dimVal)) {
    return accessPatternPerDim_.at(dimVal);
  }
  DT_ERROR(
      "[DataTransfer::getAccessPattern] No access pattern is specified for the "
      "given dimension.");
}

// ------------------------------------------------------------------------------------------------
// entry 179/382   level 0   scc 349   14 body lines
// unit: e179_getAccessPatternAsStr
// authority: ddc/ddl/ddl_conversion.cpp:3629
// original: std::string ddc::Metadata::DataTransfer::getAccessPatternAsStr( PrimaryDimTypes dimVal)
// class: DataTransfer
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
std::string e179_getAccessPatternAsStr(
    PrimaryDimTypes dimVal)
{
  if (accessPatternPerDim_.count(dimVal)) {
    std::string res = "";
    res += EnumsConversion::padTypeToString.at(
        accessPatternPerDim_.at(dimVal).first);
    res += "-to-";
    res += EnumsConversion::padTypeToString.at(
        accessPatternPerDim_.at(dimVal).second);
    return res;
  }
  DT_ERROR(
      "[DataTransfer::getAccessPattern] No access pattern is specified for the "
      "given dimension.");
}

// ------------------------------------------------------------------------------------------------
// entry 180/382   level 0   scc 361   34 body lines
// unit: e180_checkAccessPattern
// authority: ddc/ddl/ddl_conversion.cpp:3648
// original: template <typename T> bool DdlConversion::checkAccessPattern( T& op, std::string dimArgName, mlir::Operation::operand_range dims, std::string accessPatternAttrName, std::optional<mlir::ArrayAttr> opAccessPatternStyles)
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
template <typename T>
bool e180_checkAccessPattern(
    T& op, std::string dimArgName, mlir::Operation::operand_range dims,
    std::string accessPatternAttrName,
    std::optional<mlir::ArrayAttr> opAccessPatternStyles)
{
  bool hasAccessPatternStyle = bool(opAccessPatternStyles);
  bool hasAccessPatternDim = !dims.empty();
  size_t sizeofAccessPatternDim = dims.size();

  if (!hasAccessPatternStyle || opAccessPatternStyles.value().empty()) {
    if (hasAccessPatternDim) {
      op.emitError(dimArgName + " is specified but " + accessPatternAttrName +
                   " is missing.");
      DT_ERROR("Illegal ddl");
    }
    // Nothing to process
    return false;
  }

  mlir::ArrayAttr accessPatternStyles = opAccessPatternStyles.value();
  size_t sizeofAccessPatternStyle = accessPatternStyles.size();

  if (!hasAccessPatternDim) {
    // access_pattern_dim is not specified but access_pattern_style is specified
    op.emitError(dimArgName + " is not specified but " + accessPatternAttrName +
                 " is specified.");
    DT_ERROR("Illegal ddl");
  }

  if ((sizeofAccessPatternStyle != 1) &&
      (sizeofAccessPatternStyle != sizeofAccessPatternDim)) {
    op.emitError("Inconsistent number of elements found in " + dimArgName +
                 " and " + accessPatternAttrName + ".");
    DT_ERROR("Illegal ddl");
  }
  // Access pattern list can be processed
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 181/382   level 0   scc 362   29 body lines
// unit: e181_convertAccessPatternStrToDdcType
// authority: ddc/ddl/ddl_conversion.cpp:3687
// original: ddc::Metadata::TransferAccessPatternType convertAccessPatternStrToDdcType( DataTransferOp& op, std::string const& accessPattern)
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
ddc::Metadata::TransferAccessPatternType e181_convertAccessPatternStrToDdcType(
    DataTransferOp& op, std::string const& accessPattern)
{
  size_t delimPos = accessPattern.find("-to-");
  if (delimPos == std::string::npos) {
    op.emitError("Unknown access pattern format specified in " + accessPattern +
                 ". Expected <src_pad_type>-to-<dst_pad_type>.");
    DT_ERROR("Illegal ddl");
  }
  std::string srcAccessPatternStr = accessPattern.substr(0, delimPos);
  std::string destAccessPatternStr =
      accessPattern.substr(delimPos + (sizeof("-to-") - 1));

  if (!EnumsConversion::stringToPadType.count(srcAccessPatternStr)) {
    op.emitError("Unknown source access pattern specified in " + accessPattern);
    DT_ERROR("Illegal ddl");
  }

  if (!EnumsConversion::stringToPadType.count(destAccessPatternStr)) {
    op.emitError("Unknown destination access pattern specified in " +
                 accessPattern);
    DT_ERROR("Illegal ddl");
  }

  PadType srcAccessPattern =
      EnumsConversion::stringToPadType.at(srcAccessPatternStr);
  PadType destAccessPattern =
      EnumsConversion::stringToPadType.at(destAccessPatternStr);
  return ddc::Metadata::TransferAccessPatternType(srcAccessPattern,
                                                  destAccessPattern);
}

// ------------------------------------------------------------------------------------------------
// entry 182/382   level 0   scc 363   7 body lines
// unit: e182_convertAccessPatternStrToDdcType
// authority: ddc/ddl/ddl_conversion.cpp:3718
// original: PadType convertAccessPatternStrToDdcType(AllocateOp& op, std::string const& paddingType)
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
PadType e182_convertAccessPatternStrToDdcType(AllocateOp& op,
                                         std::string const& paddingType)
{
  if (!EnumsConversion::stringToPadType.count(paddingType)) {
    op.emitError("Unknown memory access pattern specified in " + paddingType);
    DT_ERROR("Illegal ddl");
  }
  return EnumsConversion::stringToPadType.at(paddingType);
}

// ------------------------------------------------------------------------------------------------
// entry 183/382   level 0   scc 330   13 body lines
// unit: e183_dump
// authority: ddc/ddl/ddl_conversion.cpp:3756
// original: void Metadata::DataTransfer::dump()
// class: DataTransfer
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
void e183_dump()
{
  std::cerr << "\n[Ddl::DataTransfer]" << "\n--------------------------"
            << "\n  access-patterns per dimension: ";

  for (auto it : accessPatternPerDim_) {
    std::cerr << "\n    Dim "
              << EnumsConversion::primaryDimToString.at(it.first)
              << " access-pattern ("
              << EnumsConversion::padTypeToString.at(it.second.first) << " to "
              << EnumsConversion::padTypeToString.at(it.second.second) << ")";
  }
  std::cerr << std::endl;
}

// ------------------------------------------------------------------------------------------------
// entry 184/382   level 0   scc 334   7 body lines
// unit: e184_setMetaDimKind
// authority: ddc/ddl/ddl_conversion.h:317
// original: bool setMetaDimKind(llvm::StringRef inDimKind)
// class: DimProp
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
bool e184_setMetaDimKind(llvm::StringRef inDimKind)
{
      if (!EnumsConversion::stringToMetaDimKind.count(inDimKind.str())) {
        return false;
      }
      metaDimKind_ = EnumsConversion::stringToMetaDimKind.at(inDimKind.str());
      return true;
    }

// ------------------------------------------------------------------------------------------------
// entry 185/382   level 0   scc 337   5 body lines
// unit: e185_isMetaDim
// authority: ddc/ddl/ddl_conversion.h:329
// original: bool isMetaDim() const
// class: DimProp
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
bool e185_isMetaDim() const
{
      return metaDimKind_ != MetaDimKind::Unpadded &&
             metaDimKind_ != MetaDimKind::Padded &&
             metaDimKind_ != MetaDimKind::Count;
    }

// ------------------------------------------------------------------------------------------------
// entry 186/382   level 0   scc 365   5 body lines
// unit: e186_getNonPaddedDimProp
// authority: ddc/ddl/ddl_conversion.h:351
// original: DimProp& getNonPaddedDimProp(Value ddlDim)
// class: DdlInterface
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
DimProp& e186_getNonPaddedDimProp(Value ddlDim)
{
    auto& dimProp = dim_association_[ddlDim];
    if (!dimProp.isPadded()) return dimProp;
    return dim_association_[dimProp.nonPaddedDim];
  }

// ------------------------------------------------------------------------------------------------
// entry 187/382   level 0   scc 5   4 body lines
// unit: e187_clear
// authority: ddc/ddl/ddl_conversion.h:458
// original: void clear()
// class: DdlInterface
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
void e187_clear()
{
    this->~DdlInterface();
    new (this) DdlInterface();
  }

// ------------------------------------------------------------------------------------------------
// entry 271/382   level 1   scc 358   7 body lines
// unit: e271_verify
// authority: ddc/ddl/Dialect/DdlOps.cpp:178
// original: LogicalResult OperationBindOp::verify()
// class: OperationBindOp
// rust home: crates/compiler/deeptools/src/schedule/ddl/ops.rs
// ------------------------------------------------------------------------------------------------
LogicalResult e271_verify()
{
  if (!EnumsConversion::stringToOpFuncs.count(this->getOpFuncName().str())) {
    this->emitError("Can not find this OpFunc: " + this->getOpFuncName());
    return failure();
  }
  return success();
}

// ------------------------------------------------------------------------------------------------
// entry 272/382   level 1   scc 116   19 body lines
// unit: e272_print
// authority: ddc/ddl/Dialect/DdlOps.cpp:305
// original: void AllocateOp::print(::mlir::OpAsmPrinter& _odsPrinter)
// class: AllocateOp
// rust home: crates/compiler/deeptools/src/schedule/ddl/ops.rs
// ------------------------------------------------------------------------------------------------
void e272_print(::mlir::OpAsmPrinter& _odsPrinter)
{
  _odsPrinter << "(";
  _odsPrinter << getTensor();
  if (!getPaddingDim().empty()) {
    _odsPrinter << ",";
    _odsPrinter << ' ' << "[";
    _odsPrinter << getPaddingDim();
    _odsPrinter << "]";
  }
  if (getExternalAllocate()) {
    _odsPrinter << ",";
    _odsPrinter << ' ';
    if (::mlir::Value value = getExternalAllocate()) _odsPrinter << value;
  }
  _odsPrinter << ")";
  ::llvm::SmallVector<::llvm::StringRef, 2> elidedAttrs;
  elidedAttrs.push_back("operandSegmentSizes");
  _odsPrinter.printOptionalAttrDict((*this)->getAttrs(), elidedAttrs);
}

// ------------------------------------------------------------------------------------------------
// entry 273/382   level 1   scc 342   13 body lines
// unit: e273_processBuffer
// authority: ddc/ddl/ddl.cpp:62
// original: OwningOpRef<Operation*> processBuffer(std::unique_ptr<MemoryBuffer> ownedBuffer, ThreadPoolInterface* threadPool, MLIRContext* context)
// rust home: crates/compiler/deeptools/src/schedule/ddl/mod.rs
// ------------------------------------------------------------------------------------------------
OwningOpRef<Operation*> e273_processBuffer(std::unique_ptr<MemoryBuffer> ownedBuffer,
                                      ThreadPoolInterface* threadPool,
                                      MLIRContext* context)
{
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  auto sourceMgr = std::make_shared<SourceMgr>();
  sourceMgr->AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

  if (threadPool) context->setThreadPool(*threadPool);

  // Parse the input file.
  context->loadAllAvailableDialects();
  context->allowUnregisteredDialects(false);
  SourceMgrDiagnosticHandler sourceMgrHandler(*sourceMgr, context);
  return performActions(sourceMgr, context);
}

// ------------------------------------------------------------------------------------------------
// entry 274/382   level 1   scc 335   22 body lines
// unit: e274_processDimensionOp
// authority: ddc/ddl/ddl_conversion.cpp:187
// original: DdlInterface::DimProp& DdlConversion::processDimensionOp( const mlir::Value& dimVal)
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
DdlInterface::DimProp& e274_processDimensionOp(
    const mlir::Value& dimVal)
{
  // Check if the dim is already processed
  if (ddlInterface.dim_association_.count(dimVal)) {
    return ddlInterface.dim_association_[dimVal];
  }

  DimensionOp dimensionOp = dyn_cast<DimensionOp>(dimVal.getDefiningOp());
  if (!dimensionOp) {
    dimVal.getDefiningOp()->emitOpError(
        "is supposed to be a dimension but is not");
    DT_ERROR("Illegal ddl file");
  }

  // Setup the dimension properties
  DdlInterface::DimProp& dimProp = ddlInterface.dim_association_[dimVal];
  if (!dimProp.setMetaDimKind(dimensionOp.getDimProperty())) {
    dimVal.getDefiningOp()->emitOpError(
        "specifies an unsupported dimension property");
    DT_ERROR("Illegal ddl file");
  }
  return dimProp;
}

// ------------------------------------------------------------------------------------------------
// entry 275/382   level 1   scc 326   234 body lines
// unit: e275_processCondition
// authority: ddc/ddl/ddl_conversion.cpp:211
// original: const DdlInterface::CondProp& DdlConversion::processCondition( mlir::Value cond)
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
const DdlInterface::CondProp& e275_processCondition(
    mlir::Value cond)
{
  if (!ddlInterface.resolvedConditions_.count(cond)) {
    // process new condition
    auto op = cond.getDefiningOp();
    auto& myCp = ddlInterface.resolvedConditions_[cond];
    if (auto bind_op = dyn_cast<OperationBindOp>(op)) {
      auto opIt = ddlInterface.operation_definition_.find(cond);
      if (opIt == ddlInterface.operation_definition_.end()) {
        myCp.isResolvedToBool_ = true;
        myCp.resolvedValue_ = false;
      } else {
        if (opIt->second.coreClCond_.empty()) {
          myCp.isResolvedToBool_ = true;
          myCp.resolvedValue_ = true;
        } else {
          myCp.coreClCond_ = opIt->second.coreClCond_;
        }
      }
    } else if (auto extAlloc_op =
                   dyn_cast<GetExternalDataTransferAllocationOp>(op)) {
      const auto& myLds = dsc.labeledDs_.at(
          ddlInterface.getTensorProp(extAlloc_op.getTensor()).ldsIdx_);
      auto compIt = EnumsConversion::stringToSenComponents.find(
          extAlloc_op.getMemory().str());
      if (compIt == EnumsConversion::stringToSenComponents.end() ||
          !dsc2::memories.count(compIt->second)) {
        extAlloc_op.emitError("Unrecognized memory");
        DT_ERROR("Illegal ddl");
      }
      myCp.resolvedValue_ = myLds.memOrg_.count(compIt->second);
      myCp.isResolvedToBool_ = true;
    } else if (auto cond_op = dyn_cast<ConditionOp>(op)) {
      auto dimIt = ddlInterface.dim_association_.find(cond_op.getDimension());
      if (dimIt == ddlInterface.dim_association_.end()) {
        cond_op.emitError("Dimension not mapped");
        DT_ERROR("Illegal ddl");
      }
      if (cond_op.getLoopLabel().empty()) {
        cond_op.emitError("Loop label cannot be empty");
        DT_ERROR("Illegal ddl");
      }
      auto condOperIt =
          EnumsConversion::stringToCondOp.find(cond_op.getCondition().str());
      if (condOperIt == EnumsConversion::stringToCondOp.end()) {
        cond_op.emitError("Condition operator not recognized");
        DT_ERROR("Illegal ddl");
      }
      auto condOper = condOperIt->second;
      if (!is_any_of(condOper, CondOp::EQ, CondOp::NE, CondOp::GE, CondOp::LE,
                     CondOp::GT, CondOp::LT)) {
        cond_op.emitError("Condition operator not supported");
        DT_ERROR("Illegal ddl");
      }
      auto valueExpr = cond_op.getValueExpr();
      auto condValType = dsc2::LoopCond::CondValType::INT;
      int condVal = -1;
      auto condValTypeIt =
          dsc2::LoopCond::stringToCondValType.find(valueExpr.str());
      if (condValTypeIt != dsc2::LoopCond::stringToCondValType.end()) {
        condValType = condValTypeIt->second;
      } else {
        condVal = processExpression(valueExpr.str());
      }
      if (dimIt->second.dropDim_) {  // dim does not exist
        // treat as loop of size one and resolve to bool
        myCp.isResolvedToBool_ = true;
        if (is_any_of(condValType, dsc2::LoopCond::CondValType::FIRST,
                      dsc2::LoopCond::CondValType::LAST)) {
          condVal = 0;
        }
        if (condOper == CondOp::NE) {
          myCp.resolvedValue_ = condVal != 0;
        } else if (condOper == CondOp::EQ) {
          myCp.resolvedValue_ = condVal == 0;
        } else if (condOper == CondOp::LE) {
          myCp.resolvedValue_ = condVal <= 0;
        } else if (condOper == CondOp::LT) {
          myCp.resolvedValue_ = condVal < 0;
        } else if (condOper == CondOp::GE) {
          myCp.resolvedValue_ = condVal >= 0;
        } else if (condOper == CondOp::GT) {
          myCp.resolvedValue_ = condVal > 0;
        }
      } else {  // real loop conditional
        const dsc2::LoopNode* refLoop = nullptr;
        if (cond_op.getLoopLabel().str() ==
            ddlInterface.core_chunk_loop_label_) {
          auto loopIt = metadata_.dimToCoreChunkLoops_.find(dimIt->second.dim_);
          if (loopIt == metadata_.dimToCoreChunkLoops_.end() ||
              loopIt->second.empty()) {
            cond_op.emitError(
                "Requested dim does not exist in pre-filled core/chunk "
                "loops");
            DT_ERROR("Illegal ddl");
          }
          refLoop = loopIt->second.at(0);
        } else {
          auto loopIt =
              ddlInterface.loop_labels_.find(cond_op.getLoopLabel().str());
          if (loopIt == ddlInterface.loop_labels_.end()) {
            cond_op.emitError("Loop label not found");
            DT_ERROR("Illegal ddl");
          }
          refLoop = loopIt->second;
        }
        myCp.loopCond_.twoLevelOrOfAnds_.emplace_back().emplace_back(
            refLoop, dimIt->second.dim_, condOperIt->second, condValType,
            condVal);
      }
    } else if (auto condNot_op = dyn_cast<ConditionNotOp>(op)) {
      myCp = processCondition(condNot_op.getOperand());
      if (myCp.isResolvedToBool_) {
        myCp.resolvedValue_ = !myCp.resolvedValue_;
      } else if (!myCp.loopCond_.twoLevelOrOfAnds_.empty()) {
        myCp.loopCond_.negated_ = !myCp.loopCond_.negated_;
      } else {
        for (const auto& coreId : dsc.coreIdsUsed_) {
          if (auto coreClIt = myCp.coreClCond_.find(coreId);
              coreClIt != myCp.coreClCond_.end()) {
            for (int cl = 0; cl < dsc.numCoreletsUsed_DSC2_; cl++) {
              if (coreClIt->second.count(cl))
                coreClIt->second.erase(cl);
              else
                coreClIt->second.insert(cl);
            }
            if (coreClIt->second.empty()) myCp.coreClCond_.erase(coreClIt);
          } else {
            myCp.coreClCond_[coreId].insert(0);
            if (dsc.numCoreletsUsed_DSC2_ > 1)
              myCp.coreClCond_[coreId].insert(1);
          }
        }
      }
    } else if (isa<ConditionAndOp, ConditionOrOp>(op)) {
      bool initialized = false, isAnd = isa<ConditionAndOp>(op);
      for (auto operand : op->getOperands()) {
        auto& operandCp = processCondition(operand);
        if (operandCp.isResolvedToBool_) {
          if (isAnd && !operandCp.resolvedValue_) {
            myCp.isResolvedToBool_ = true;
            myCp.resolvedValue_ = false;
            myCp.loopCond_ = {};
            myCp.coreClCond_.clear();
            break;
          } else if (!isAnd && operandCp.resolvedValue_) {
            myCp.isResolvedToBool_ = true;
            myCp.resolvedValue_ = true;
            myCp.loopCond_ = {};
            myCp.coreClCond_.clear();
            break;
          }
        }
        if (!initialized) {
          myCp = operandCp;
          initialized = true;
          continue;
        }
        if (!operandCp.loopCond_.twoLevelOrOfAnds_.empty()) {
          if (!myCp.coreClCond_.empty()) {
            op->emitError("And/or op is mixing incompatible types");
            DT_ERROR("Illegal ddl");
          }
          if (myCp.isResolvedToBool_) {
            myCp.isResolvedToBool_ = false;
            myCp = operandCp;
            continue;
          }
          if (isAnd) {
            if (myCp.loopCond_.twoLevelOrOfAnds_.size() != 1 ||
                myCp.loopCond_.negated_ ||
                operandCp.loopCond_.twoLevelOrOfAnds_.size() != 1 ||
                operandCp.loopCond_.negated_) {
              op->emitError(
                  "Loop conditionals can only be composed to create a "
                  "two-level OR of ANDs, with optionally an "
                  "overall negation");
              DT_ERROR("Illegal ddl");
            }
            auto& origVect = operandCp.loopCond_.twoLevelOrOfAnds_[0];
            auto& destVect = myCp.loopCond_.twoLevelOrOfAnds_[0];
            destVect.insert(destVect.end(), origVect.begin(), origVect.end());
          } else {  // OR
            if (myCp.loopCond_.negated_ || operandCp.loopCond_.negated_) {
              op->emitError(
                  "Loop conditionals can only be composed to create a "
                  "two-level OR of ANDs, with optionally an "
                  "overall negation");
              DT_ERROR("Illegal ddl");
            }
            auto& origVect = operandCp.loopCond_.twoLevelOrOfAnds_;
            auto& destVect = myCp.loopCond_.twoLevelOrOfAnds_;
            destVect.insert(destVect.end(), origVect.begin(), origVect.end());
          }
        } else if (!operandCp.coreClCond_.empty()) {
          if (!myCp.loopCond_.twoLevelOrOfAnds_.empty()) {
            op->emitError("And/or op is mixing incompatible types");
            DT_ERROR("Illegal ddl");
          }
          if (myCp.isResolvedToBool_) {
            myCp.isResolvedToBool_ = false;
            myCp = operandCp;
            continue;
          }
          if (isAnd) {
            for (auto& corePair : myCp.coreClCond_) {
              auto clsIt = operandCp.coreClCond_.find(corePair.first);
              if (clsIt == operandCp.coreClCond_.end()) {
                myCp.coreClCond_.erase(corePair.first);
                continue;
              }
              corePair.second = set_intersect(corePair.second, clsIt->second);
              if (corePair.second.empty())
                myCp.coreClCond_.erase(corePair.first);
            }
          } else {  // OR
            for (auto& corePair : operandCp.coreClCond_) {
              auto& cls = myCp.coreClCond_[corePair.first];
              cls.insert(corePair.second.begin(), corePair.second.end());
            }
          }
        }
      }
    } else {
      op->emitError("Op type not supported in conditionals");
      DT_ERROR("Illegal ddl");
    }
    if (!myCp.isResolvedToBool_ && myCp.loopCond_.twoLevelOrOfAnds_.empty() &&
        myCp.coreClCond_.empty()) {
      myCp.isResolvedToBool_ = true;
      myCp.resolvedValue_ = false;
    }
  }
  return ddlInterface.resolvedConditions_.at(cond);
}

// ------------------------------------------------------------------------------------------------
// entry 276/382   level 1   scc 346   216 body lines
// unit: e276_verifyDdlConstraints
// authority: ddc/ddl/ddl_conversion.cpp:2553
// original: void DdlConversion::verifyDdlConstraints()
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
void e276_verifyDdlConstraints()
{
  auto ddlMlirRoot = ddlParser_.ddl_module_op_.get();
  ddlMlirRoot->walk<WalkOrder::PreOrder>([&](ConstraintOp constraint_op) {
    // ddl.constraint() {min_num_cores = 1}

    if (constraint_op.getMinNumCores().has_value()) {
      if (dsc.numCoresUsed_ < constraint_op.getMinNumCores().value()) {
        constraint_op.emitError(
            "Minimum number of cores greater than total core used");
        llvm_unreachable("Ddl constraints not met");
      }
      return;
    }

    /* ddl.constraint(%bmm_int8_mbkg3_op, %bmm_int8_op, %bmm_fp16_op,
       %bmm_fp8_op, %bmm_int4_op, %bmmxrf_int8_op, %bmmxrf_fp16_op,
       %bmmxrf_fp8_op, %bmmxrf_int4_op, %mm_int8_op, %mm_fp16_op, %mm_fp8_op,
       %mm_int4_op) {min_num_valid = 1, max_num_valid = 1}
    */
    if (constraint_op.getMinNumValid().has_value() ||
        constraint_op.getMaxNumValid().has_value()) {
      int num_valid = 0;
      for (auto input : constraint_op.getOperands()) {
        auto inputOp = input.getDefiningOp();
        if (isa<OperationBindOp>(inputOp)) {
          if (ddlInterface.operation_definition_.count(input)) num_valid++;
        } else if (auto extTrOp =
                       dyn_cast<GetExternalDataTransferAllocationOp>(inputOp)) {
          const auto& tensProp =
              ddlInterface.getTensorProp(extTrOp.getTensor());
          const auto& myLds = dsc.labeledDs_.at(tensProp.ldsIdx_);
          auto strit = EnumsConversion::stringToSenComponents.find(
              extTrOp.getMemory().str());
          if (strit == EnumsConversion::stringToSenComponents.end() ||
              !dsc2::memories.count(strit->second)) {
            extTrOp.emitError("Unrecognised memory");
            DT_ERROR("Illegal ddl");
          }
          auto storage = strit->second == PELRF      ? PE
                         : strit->second == SFPLRF   ? SFP
                         : strit->second == PTXRF    ? PT
                         : strit->second == PESTATE  ? PE
                         : strit->second == SFPSTATE ? SFP
                                                     : strit->second;
          if (storage == myLds.pinnedComponent()) num_valid++;
        } else {
          constraint_op.emitError(
              "Variable type not yet supported in this check");
          llvm_unreachable("Illegal ddl");
        }
      }
      int min_num_valid = constraint_op.getMinNumValid().value_or(0);
      int max_num_valid = constraint_op.getMaxNumValid().value_or(100);
      if (num_valid < min_num_valid || num_valid > max_num_valid) {
        constraint_op.emitError("Constraint violated, got " +
                                std::to_string(num_valid) +
                                " active variables");
        llvm_unreachable("Ddl constraints not met");
      }
      return;
    }

    /*

    ddl.constraint(%bmm_int8_mbkg3_op, %bmm_int8_op, %bmm_fp16_op, %bmm_fp8_op,
    %bmm_int4_op, %bmmxrf_int8_op, %bmmxrf_fp16_op, %bmmxrf_fp8_op,
    %bmmxrf_int4_op, %mm_int8_op, %mm_fp16_op, %mm_fp8_op, %mm_int4_op,
                  %psum_op, %bn_op, %bias_op, %relu_op, %stradd_op)
    {relative_op_order=true}
    */
    int cur_max_index = INT_MIN;
    if (constraint_op.getRelativeOpOrder().has_value() &&
        constraint_op.getRelativeOpOrder().value()) {
      int index = 0;
      for (auto input : constraint_op.getOperands()) {
        auto opDefIt = ddlInterface.operation_definition_.find(input);
        if (opDefIt == ddlInterface.operation_definition_.end()) continue;
        index = opDefIt->second.computeOpIdx_;
        if (index > cur_max_index) {
          cur_max_index = index;
        } else {
          constraint_op.emitError("Relative operation order not maintained");
          llvm_unreachable("Ddl constraints not met");
        }
      }
      return;
    }

    // ddl.constraint(%inptensor_int8, %inptensor_fp8) {property = "slice",
    // dim_idx = 0, cmp = "equal", value = 8}
    if (constraint_op.getCmp().has_value()) {
      const StringRef cmp = constraint_op.getCmp().value();
      if (constraint_op.getProperty().has_value()) {
        bool is_slice =
            constraint_op.getProperty().value() == "slice" ? true : false;
        if (constraint_op.getDimIdx().has_value()) {
          if (!constraint_op.getValue().has_value()) {
            constraint_op.emitError("Missing \"value\" attribute");
            llvm_unreachable("Illegal ddl");
          }
          for (auto input : constraint_op.getOperands()) {
            DsTypes dt;
            auto opDefIt = ddlInterface.tensor_definition_.find(input);
            if (opDefIt == ddlInterface.tensor_definition_.end()) continue;
            const auto& lds = dsc.labeledDs_.at(opDefIt->second.ldsIdx_);
            dt = lds.dsType_;
            if (cmp == "equal") {
              auto stick_sizes = dsc.getStickSizes(dt, is_slice, !is_slice);
              if (stick_sizes.size() <= constraint_op.getDimIdx().value()) {
                constraint_op.emitError("Index out of bound");
                llvm_unreachable("Ddl constraints not met");
              }
              int size = stick_sizes[constraint_op.getDimIdx().value()].second;
              if (size != constraint_op.getValue().value()) {
                constraint_op.emitError(
                    constraint_op.getProperty().value().str() +
                    " size does not match");
                llvm_unreachable("Ddl constraints not met");
              }
            } else {
              constraint_op.emitError("\"cmp\" type not yet supported");
              llvm_unreachable("Illegal ddl");
            }
          }
        } else {
          // ddl.constraint(%inptensor, %outtensor) {property = "slice", cmp =
          // "equal"}

          std::vector<std::pair<PrimaryDimTypes, int>> previous_sizes;
          for (auto input : constraint_op.getOperands()) {
            auto opDefIt = ddlInterface.tensor_definition_.find(input);
            if (opDefIt == ddlInterface.tensor_definition_.end()) continue;
            const auto& lds = dsc.labeledDs_.at(opDefIt->second.ldsIdx_);
            DsTypes dt = lds.dsType_;
            if (cmp == "equal") {
              auto sizes = dsc.getStickSizes(dt, is_slice, !is_slice);
              if (previous_sizes.empty()) previous_sizes = sizes;
              if (previous_sizes != sizes) {
                constraint_op.emitError(
                    constraint_op.getProperty().value().str() +
                    " size does not match");
                llvm_unreachable("Ddl constraints not met");
              }
            } else {
              constraint_op.emitError("\"cmp\" type not yet supported");
              llvm_unreachable("Illegal ddl");
            }
          }
        }
      } else {  // no property
        if (!constraint_op.getValue().has_value()) {
          constraint_op.emitError("Missing \"value\" attribute");
          llvm_unreachable("Illegal ddl");
        }
        auto compVal = constraint_op.getValue().value();
        for (auto input : constraint_op.getOperands()) {
          auto dimInfoIt = ddlInterface.dim_association_.find(input);
          if (dimInfoIt == ddlInterface.dim_association_.end()) {
            constraint_op.emitError(
                "Constraint of this type not supported on this variable");
            llvm_unreachable("Illegal ddl");
          }
          if (dimInfoIt->second.dropDim_) continue;
          auto dim = dimInfoIt->second.getPrimaryDim();
          auto dimKind = dimInfoIt->second.getMetaDimKind();
          const auto& refDs = dsc.dataStageParam_.at(metadata_.core_dstgid).ss_;
          int size;
          if (is_any_of(dimKind, MetaDimKind::Unpadded,
                        MetaDimKind::WindowDim)) {
            size = refDs.primaryDimToVal_st(dim);
          } else {
            if (!refDs.paddingSizes_.count(dim)) continue;
            const auto& padInfo = refDs.paddingSizes_.at(dim);
            if (dimKind == MetaDimKind::PadFront) {
              size = padInfo.padFront_;
            } else if (dimKind == MetaDimKind::PadBack) {
              size = padInfo.padBack_;
            } else if (dimKind == MetaDimKind::Stride) {
              size = padInfo.stride_;
            } else if (dimKind == MetaDimKind::Dilation) {
              size = padInfo.dilation_;
            } else if (dimKind == MetaDimKind::Padded) {
              size = refDs.primaryDimToVal_st(
                  dim, SenComponents::NO_COMPONENT, -1, -1,
                  {dim, PadType::PADDED_FULLSPAN_WUNNEEDED});
            } else if (dimKind == MetaDimKind::PadValid) {
              size = refDs.primaryDimToVal_st(dim, SenComponents::NO_COMPONENT,
                                              -1, -1,
                                              {dim, PadType::PADDED_NOZEROPAD});
            } else {
              constraint_op.emitError("Constraint on unsupported dim kind");
              llvm_unreachable("Illegal ddl");
            }
          }
          bool ok = false;
          if (cmp == "equal") {
            ok = (size == compVal);
          } else if (cmp == "less") {
            ok = (size < compVal);
          } else {
            constraint_op.emitError("\"cmp\" type not yet supported");
            llvm_unreachable("Illegal ddl");
          }
          if (!ok) {
            constraint_op.emitError("Dim constraint not met");
            llvm_unreachable("Ddl constraints not met");
          }
        }
      }
      return;
    }

    constraint_op.emitError("Unsupported constraint type/format");
    llvm_unreachable("Illegal ddl");
  });
}

// ------------------------------------------------------------------------------------------------
// entry 277/382   level 1   scc 348   31 body lines
// unit: e277_getTensorAndAllocation
// authority: ddc/ddl/ddl_conversion.cpp:2847
// original: std::pair<Value, Value> DdlConversion::getTensorAndAllocation( llvm::DenseMap<AllocateOp, const dsc2::AllocateNode*>& allocations, SenComponents unit, const dsc2::DataInfo dtinfo) const
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
std::pair<Value, Value> e277_getTensorAndAllocation(
    llvm::DenseMap<AllocateOp, const dsc2::AllocateNode*>& allocations,
    SenComponents unit, const dsc2::DataInfo dtinfo) const
{
  Value tensor = nullptr;
  Value allocate = nullptr;
  AllocateOp allocateop = nullptr;
  if (unit == SenComponents::NO_COMPONENT) return {nullptr, nullptr};

  const dsc2::AllocateNode* allocatenode = nullptr;
  if (dtinfo.myLdsIdx_ >= 0) {
    allocatenode =
        dsc.labeledDs_.at(dtinfo.myLdsIdx_).memOrg_.at(unit).allocateNode_;
  } else if (dtinfo.constantId_ >= 0) {
    allocatenode =
        dsc.constantInfo_.at(dtinfo.constantId_).allocations_.at(unit);
  } else {
    for (auto kv : ddlInterface.operand_constant_tensor_)
      if (kv.second == unit) {
        tensor = kv.first;
      }
    return {tensor, allocate};
  }

  for (auto kv : allocations) {
    if (kv.second == allocatenode) {
      allocateop = kv.first;
      allocate = allocateop.getResult();
      tensor = allocateop.getTensor();
      break;
    }
  }
  return {tensor, allocate};
}

// ------------------------------------------------------------------------------------------------
// entry 278/382   level 1   scc 338   81 body lines
// unit: e278_checkMetaDimensions
// authority: ddc/ddl/ddl_conversion.cpp:3537
// original: void DdlConversion::checkMetaDimensions()
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
void e278_checkMetaDimensions()
{
  // DSC: Presence status of meta dimensions for each primary dimension
  std::map<PrimaryDimTypes, bool[MetaDimKind::Count]> metaDimsDsc;

  // Construct the primarydim -> [ metadims] mapping to indicate which
  // meta dimensions (for each primary dimension of interest) are
  // present in the input DSC.
  for (const auto& [dim, padInfo] :
       dsc.dataStageParam_.at(metadata_.core_dstgid).ss_.paddingSizes_) {
    metaDimsDsc[dim][MetaDimKind::Padded] = true;
    metaDimsDsc[dim][MetaDimKind::PadFront] = true;
    metaDimsDsc[dim][MetaDimKind::PadBack] = true;
    metaDimsDsc[dim][MetaDimKind::PadValid] = true;
    if (padInfo.windowDim_ != PrimaryDimTypesCount) {
      metaDimsDsc[dim][MetaDimKind::WindowDim] = true;
      metaDimsDsc[dim][MetaDimKind::Stride] = true;
      metaDimsDsc[dim][MetaDimKind::Dilation] = true;
    }
  }

  bool metaDimKindsSeen[MetaDimKind::Count];

  // For each padded-dimension, perform constraint checking on the
  // defining operation.
  for (auto it : ddlInterface.dim_association_) {
    if (!it.second.isPadded() || it.second.dropDim_) {
      continue;
    }
    // Clear the seen status.
    for (int i = 0; i < MetaDimKind::Count; ++i) {
      metaDimKindsSeen[i] = false;
    }
    PrimaryDimTypes dimType =
        ddlInterface.dim_association_[it.second.nonPaddedDim].dim_;
    PaddedDimensionOp op =
        dyn_cast<PaddedDimensionOp>(it.first.getDefiningOp());
    if (!op) {
      op.emitError("Could not find padded_dimension operation.");
      DT_ERROR("Internal error in PaddedDimensionOp verification.");
    }
    // Iterate over the operands.
    //   - Check if specified metadim is also present in input DSC.
    for (int i = 0; i < op.getNumOperands(); ++i) {
      auto operand = op.getOperand(i);
      const DdlInterface::DimProp& dimProp =
          ddlInterface.dim_association_[operand];
      if (!dimProp.isMetaDim()) {
        continue;
      }
      MetaDimKind opDimKind = dimProp.getMetaDimKind();
      metaDimKindsSeen[opDimKind] = true;
      if (!metaDimsDsc.count(dimType) || !metaDimsDsc[dimType][opDimKind]) {
        op.emitError(std::string("Specified meta dimension of kind ") +
                     EnumsConversion::metaDimKindToString.at(opDimKind) +
                     " for primary dimension of kind " +
                     EnumsConversion::primaryDimToString.at(dimType) +
                     " does not exist in the input DSC.");
        DT_ERROR("Illegal ddl.");
      }
    }
    // Check if all metadims from input DSC are specified in the DDL.
    for (int i = 0; i < MetaDimKind::Count; ++i) {
      if (i == MetaDimKind::Unpadded || i == MetaDimKind::Padded) {
        continue;
      }
      if (i == MetaDimKind::Dilation) {
        // For now, skip Dilation as DSC does not define a corresponding
        // dimension
        continue;
      }
      if (metaDimsDsc.count(dimType) && metaDimsDsc[dimType][i] &&
          !metaDimKindsSeen[i]) {
        op.emitError(
            std::string("DSC includes meta dimension ") +
            EnumsConversion::metaDimKindToString.at(MetaDimKind(i)) +
            ", but the ddl operation does not specify the meta dimension.");
        DT_ERROR("Illegal ddl.");
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 279/382   level 1   scc 364   24 body lines
// unit: e279_processAccessPatterns
// authority: ddc/ddl/ddl_conversion.cpp:3727
// original: template <typename OpT, typename AccPatT> void DdlConversion::processAccessPatterns( OpT& op, mlir::Operation::operand_range dims, mlir::ArrayAttr inOpAccessPatternStyles, std::map<PrimaryDimTypes, AccPatT>& result)
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
template <typename OpT, typename AccPatT>
void e279_processAccessPatterns(
    OpT& op, mlir::Operation::operand_range dims,
    mlir::ArrayAttr inOpAccessPatternStyles,
    std::map<PrimaryDimTypes, AccPatT>& result)
{
  PrimaryDimTypes dimKey = PrimaryDimTypes::PrimaryDimTypesCount;
  auto opAccessPatternStyles =
      inOpAccessPatternStyles.getAsRange<mlir::StringAttr>();
  size_t patternListSize = inOpAccessPatternStyles.size();
  auto it = opAccessPatternStyles.begin();
  AccPatT ddcPat;

  if (patternListSize == 1) {
    ddcPat = convertAccessPatternStrToDdcType(op, (*it).str());
  }

  for (size_t i = 0; i < dims.size(); ++i, ++it) {
    dimKey = ddlInterface.dim_association_[dims[i]].getPrimaryDim();
    if (patternListSize == 1) {
      result[dimKey] = ddcPat;
    } else {
      result[dimKey] = convertAccessPatternStrToDdcType(op, (*it).str());
    }
  }

  // TO DO: Do we need to check if the access-pattern is consistent with the
  //        tensor dimension(s) (both source and destination, if applicable)
}

// ------------------------------------------------------------------------------------------------
// entry 321/382   level 2   scc 343   16 body lines
// unit: e321_DdlMain
// authority: ddc/ddl/ddl.cpp:78
// original: OwningOpRef<Operation*> DdlMain(std::unique_ptr<llvm::MemoryBuffer> buffer, MLIRContext* context)
// rust home: crates/compiler/deeptools/src/schedule/ddl/mod.rs
// ------------------------------------------------------------------------------------------------
OwningOpRef<Operation*> e321_DdlMain(std::unique_ptr<llvm::MemoryBuffer> buffer,
                                MLIRContext* context)
{
  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  // We use an explicit threadpool to avoid creating and joining/destroying
  // threads for each of the split.
  ThreadPoolInterface* threadPool = nullptr;

  // Create a temporary context for the sake of checking if
  // --mlir-disable-threading was passed on the command line.
  // We use the thread-pool this context is creating, and avoid
  // creating any thread when disabled.
  MLIRContext threadPoolCtx;
  if (threadPoolCtx.isMultithreadingEnabled())
    threadPool = &threadPoolCtx.getThreadPool();
  return processBuffer(std::move(buffer), threadPool, context);
}

// ------------------------------------------------------------------------------------------------
// entry 322/382   level 2   scc 336   84 body lines
// unit: e322_processPaddedDimensionOp
// authority: ddc/ddl/ddl_conversion.cpp:102
// original: void DdlConversion::processPaddedDimensionOp(const mlir::Value& dimVal)
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
void e322_processPaddedDimensionOp(const mlir::Value& dimVal)
{
#define META_DIM_START_IN_PADDED_DIM_OP 1

  // Check if the dim is already processed
  if (ddlInterface.dim_association_.count(dimVal)) return;

  auto paddedDimOp = dyn_cast<PaddedDimensionOp>(dimVal.getDefiningOp());
  if (!paddedDimOp) {
    dimVal.getDefiningOp()->emitError(
        "Expected a padded_dimension op but got a different operation.");
    DT_ERROR("Illegal ddl");
  }
  mlir::Value unpaddedDim = paddedDimOp.getPrimaryDim();
  processDimensionOp(unpaddedDim);
  if (ddlInterface.dim_association_[unpaddedDim].getMetaDimKind() !=
      MetaDimKind::Unpadded) {
    paddedDimOp.emitError(
        "First input of the padded_dimension op must be a primary unpadded "
        "dimension.");
    DT_ERROR("Illegal ddl");
  }
  // The primary, unpadded dimension has been found.
  // Setup the result padded dimension.
  DdlInterface::DimProp& paddedDimProp = ddlInterface.dim_association_[dimVal];
  paddedDimProp.setMetaDimKind(MetaDimKind::Padded);
  paddedDimProp.nonPaddedDim = unpaddedDim;

  // Process the remaining inputs.
  std::map<MetaDimKind, mlir::Value> dimsSeen;
  for (auto operand : paddedDimOp.getPaddingDim()) {
    DdlInterface::DimProp& dimProp = processDimensionOp(operand);
    MetaDimKind dimKind = dimProp.getMetaDimKind();
    if (!is_any_of(dimKind, MetaDimKind::PadFront, MetaDimKind::PadValid,
                   MetaDimKind::PadBack)) {
      paddedDimOp.emitError(std::string("A dimension of kind ") +
                            EnumsConversion::metaDimKindToString.at(dimKind) +
                            " is not allowed in the PaddingDim operand.");
      DT_ERROR("Illegal ddl");
    }
    // Check if a dimension kind is repeated.
    if (dimsSeen.count(dimKind)) {
      paddedDimOp.emitError(
          std::string("Multiple input-dimensions of the same kind (") +
          EnumsConversion::metaDimKindToString.at(dimKind) +
          ") can not be specified for "
          "the padded_dimension op.");
      DT_ERROR("Illegal ddl");
    }
    // Set up the relation between a padded dimension and the corresponding
    // non_padded dimension.
    dimProp.nonPaddedDim = unpaddedDim;
    // Record the processed input dimension for verification.
    dimsSeen[dimProp.getMetaDimKind()] = operand;
  }

  for (auto operand : paddedDimOp.getWindowDim()) {
    DdlInterface::DimProp& dimProp = processDimensionOp(operand);
    MetaDimKind dimKind = dimProp.getMetaDimKind();
    if (!is_any_of(dimKind, MetaDimKind::WindowDim, MetaDimKind::Stride,
                   MetaDimKind::Dilation)) {
      paddedDimOp.emitError(std::string("A dimension of kind ") +
                            EnumsConversion::metaDimKindToString.at(dimKind) +
                            " is not allowed in the WindowDim operand.");
      DT_ERROR("Illegal ddl");
    }
    // Check if a dimension kind is repeated.
    if (dimsSeen.count(dimKind)) {
      paddedDimOp.emitError(
          std::string("Multiple input-dimensions of the same kind (") +
          EnumsConversion::metaDimKindToString.at(dimKind) +
          ") can not be specified for "
          "the padded_dimension op.");
      DT_ERROR("Illegal ddl");
    }
    // Set up the relation between a padded dimension and the corresponding
    // non_padded dimension.
    dimProp.nonPaddedDim = unpaddedDim;
    // Record the processed input dimension for verification.
    dimsSeen[dimProp.getMetaDimKind()] = operand;
  }

  // Error checking for presence or absence of a meta dimension is done after
  // the mapping of the primary dimensions.
}

// ------------------------------------------------------------------------------------------------
// entry 323/382   level 2   scc 328   1424 body lines
// unit: e323_processOp
// authority: ddc/ddl/ddl_conversion.cpp:582
// original: std::pair<dsc2::BlockNode*, std::vector<int>> DdlConversion::processOp( Operation& op, dsc2::BlockNode* currParent)
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
std::pair<dsc2::BlockNode*, std::vector<int>> e323_processOp(
    Operation& op, dsc2::BlockNode* currParent)
{
  dsc2::BlockNode* blockNodeForInsertion = nullptr;
  std::vector<int> regionIndecesToProcess;
  auto getGenericCompIfAvailable =
      [](const SenComponents& comp) -> SenComponents {
    auto it = EnumsConversion::senCompToGenericComp.find(comp);
    if (it != EnumsConversion::senCompToGenericComp.end()) return it->second;
    return comp;
  };

  auto getTensorOrExtConstIdx = [&](mlir::Value mySsa) {
    std::pair<int, int> idx = {-1, -1};
    if (isa<TensorOp, InternalTensorOp, AliasOneTensorOfOp>(
            mySsa.getDefiningOp())) {
      idx.first = ddlInterface.getTensorProp(mySsa).ldsIdx_;
      return idx;
    }
    // check if constant already parsed
    auto ecit = ddlInterface.ext_constant_definition_.find(mySsa);
    if (ecit != ddlInterface.ext_constant_definition_.end()) {
      idx.second = ecit->second;
      return idx;
    }
    auto myExtConst = dyn_cast<GetExternalConstantOp>(mySsa.getDefiningOp());
    auto myDefConst = dyn_cast<DefineConstantOp>(mySsa.getDefiningOp());
    auto myDefConstAlias =
        dyn_cast<AliasOneConstantOfOp>(mySsa.getDefiningOp());
    dsc2::ConstantInfo myConstInfo;
    mlir::ddl::DatatypeOp myDataTypeOp;

    if (myExtConst) {
      auto datatypesRange = myExtConst.getDatatype();
      DT_CHECK(!datatypesRange.empty());
      for (auto dataTypeValue : datatypesRange) {
        auto typeOp = dyn_cast<DatatypeOp>(dataTypeValue.getDefiningOp());
        if (dsc.computeOp_.front().attributes_.dataFormat_ ==
            FromString<DataFormats>(typeOp.getDataType().str())) {
          myDataTypeOp = typeOp;
          break;
        }
      }
    } else if (myDefConst) {
      myDataTypeOp =
          dyn_cast<DatatypeOp>(myDefConst.getDatatype().getDefiningOp());
    } else if (myDefConstAlias) {
      for (auto constDefValue : myDefConstAlias.getConstants()) {
        auto constDefOp =
            dyn_cast<DefineConstantOp>(constDefValue.getDefiningOp());
        if (!constDefOp) {
          myDefConstAlias.emitError(
              "Expected only constant definitions in here");
          DT_ERROR("Illegal ddl");
        }
        auto typeOp =
            dyn_cast<DatatypeOp>(constDefOp.getDatatype().getDefiningOp());
        if (dsc.computeOp_.front().attributes_.dataFormat_ ==
            FromString<DataFormats>(typeOp.getDataType().str())) {
          myDataTypeOp = typeOp;
          myDefConst = constDefOp;
          break;
        }
      }
    }

    if (!myDataTypeOp) {
      mySsa.getDefiningOp()->emitError(
          "Could not find a suitable match as a constant");
      DT_ERROR("Illegal ddl");
    }
    myConstInfo.dataFormat_ =
        FromString<DataFormats>(myDataTypeOp.getDataType().str());
    if (myConstInfo.dataFormat_ == DataFormats::INVALID) {
      myDataTypeOp.emitError("Invalid type name");
      DT_ERROR("Illegal ddl");
    }
    if (myDefConst) {
      std::vector<int64_t> data;
      for (const auto& val : myDefConst.getValue()) {
        auto valInt = mlir::dyn_cast<IntegerAttr>(val);
        if (!valInt) {
          myDefConst.emitError(
              "Constant value must be an integer encoding in sentient format");
          DT_ERROR("Illegal ddl");
        }
        if (dsc.computeOp_.front().attributes_.dataFormat_ ==
                DataFormats::IEEE_FP32 &&
            myConstInfo.dataFormat_ == DataFormats::SEN169_FP16) {
          auto valConvert = deeptools::BinaryConvert<uint32_t>(
              deeptools::Fp16BinToFloat(valInt.getInt()));
          myConstInfo.dataFormat_ = DataFormats::IEEE_FP32;
          data.push_back(valConvert);
        } else {
          data.push_back(valInt.getInt());
        }
      }
      std::deque<const FoldDimProp*> sdscFoldProps;
      if (sdsc.coreFoldProp_) {
        DT_CHECK_MSG(sdsc.coreletFoldProp_,
                     "sdsc coreFoldProp set but no coreletFoldProp");
        sdscFoldProps.push_back(sdsc.coreFoldProp_.get());
        sdscFoldProps.push_back(sdsc.coreletFoldProp_.get());
        for (auto& fold : sdsc.sdscFoldProps_) {
          sdscFoldProps.push_back(fold.get());
        }
        myConstInfo.data_.buildAllConstantFoldSpace(sdscFoldProps);
      } else {
        DT_CHECK_MSG(!sdsc.coreletFoldProp_ && sdsc.sdscFoldProps_.empty(),
                     "sdsc coreFoldProp not set but other folds present");
      }
      const std::deque<int64_t> coord(sdscFoldProps.size(), 0);
      myConstInfo.data_.insertData(data, coord);
      if (myDefConst.getName().has_value())
        myConstInfo.name_ = myDefConst.getName()->str();
    } else {  // external constant
      const int bitsPerElem =
          EnumsConversion::dataFormatsToBitWidth.at(myConstInfo.dataFormat_);

      myConstInfo.name_ = myExtConst.getName().str();
      const int numElems = myExtConst.getNumElements();
      if (bitsPerElem * numElems > 4 * 32) {
        myExtConst.emitError("NumElements larger than opconst size");
        DT_ERROR("Illegal ddl");
      }
      // check if external constant already present in constant container
      for (const auto& [id, constProp] : dsc.constantInfo_) {
        if (constProp.name_ != myConstInfo.name_) continue;
        // check if other properties are matching
        if (constProp.dataFormat_ != myConstInfo.dataFormat_ ||
            constProp.data_.getSingleData().size() != numElems) {
          myExtConst.emitError(
              "Constant found in DSC but properties do not match");
          DT_ERROR("Illegal ddl");
        }
        idx.second = id;
        ddlInterface.ext_constant_definition_[mySsa] = id;
        return idx;
      }
      DT_ERROR("Missing external constant in DSC: " + myConstInfo.name_);
    }
    // insert newly created constant in constant container
    int myId = dsc.constantInfo_.size();
    while (dsc.constantInfo_.count(myId)) myId++;
    dsc.constantInfo_.emplace(myId, std::move(myConstInfo));
    ddlInterface.ext_constant_definition_[mySsa] = myId;
    idx.second = myId;
    return idx;
  };

  auto unrollRowUnits = [this](std::string unitStr, mlir::Operation* op) {
    std::vector<SenComponents> units;
    auto pushInExUnits = [&](const std::string& unitName) {
      auto exuit = EnumsConversion::stringToSenComponents.find(unitName);
      if (exuit == EnumsConversion::stringToSenComponents.end()) {
        op->emitError("Unknown unit");
        DT_ERROR("Illegal ddl");
      }
      units.push_back(exuit->second);
    };
    if (unitStr == "pt")
      unitStr = "ptrow0-" + std::to_string(dscGlobal_.sysDef.numPTRows - 1);
    else if (unitStr == "l0lu")
      unitStr = "l0lurow0-" + std::to_string(dscGlobal_.sysDef.numPTRows - 1);
    if (int dashPos = unitStr.find("-"); dashPos != std::string::npos) {
      char startRowChar = unitStr[dashPos - 1];
      char endRowChar = unitStr[dashPos + 1];
      if (unitStr.size() != dashPos + 2 ||
          unitStr.find("row") == std::string::npos || !isdigit(startRowChar) ||
          !isdigit(endRowChar) || endRowChar < startRowChar) {
        op->emitError("Incorrect format for multiple row units");
        DT_ERROR("Illegal ddl");
      }
      auto baseName = unitStr.substr(0, dashPos - 1);
      for (char i = startRowChar; i <= endRowChar; i++) {
        pushInExUnits(baseName + i);
      }
    } else {
      pushInExUnits(unitStr);
    }
    return units;
  };

  auto processAllocation =
      [&](AllocateOp myAlloc,
          const dsc2::ScheduleNode* userNode) -> dsc2::AllocateNode* {
    auto myAllocNode = new dsc2::AllocateNode();
    auto strit =
        EnumsConversion::stringToSenComponents.find(myAlloc.getMemory().str());
    if (strit == EnumsConversion::stringToSenComponents.end() ||
        !dsc2::memories.count(strit->second)) {
      myAlloc.emitError("Unrecognised memory");
      DT_ERROR("Illegal ddl");
    }
    auto storage = strit->second;
    auto allocTensorOrCosntIdx = getTensorOrExtConstIdx(myAlloc.getTensor());
    if (allocTensorOrCosntIdx.first >= 0) {
      auto& myLds = dsc.labeledDs_.at(allocTensorOrCosntIdx.first);
      // create an allocate node..
      auto allocParent =
          ddlInterface.region2blocks_.at(myAlloc->getParentRegion());
      allocParent->addChildNode(myAllocNode, allocParent != currParent);

      std::map<PrimaryDimTypes, PadType> result;
      if (checkAccessPattern<AllocateOp>(
              myAlloc, "padding_dim", myAlloc.getPaddingDim(), "padding_type",
              myAlloc.getPaddingType())) {
        processAccessPatterns<AllocateOp, PadType>(
            myAlloc, myAlloc.getPaddingDim(), myAlloc.getPaddingTypeAttr(),
            result);
        // Store the padding info in the DSC node
        for (auto it : result) {
          myAllocNode->padding_.setPadding(it.first, it.second);
        }
      }

      myAllocNode->layoutDimOrder_ = dsc.getLayoutDims(myLds.ldsIdx_);
      DT_CHECK_MSG(std::set(myAllocNode->layoutDimOrder_.begin(),
                            myAllocNode->layoutDimOrder_.end())
                           .size() == myAllocNode->layoutDimOrder_.size(),
                   "Handling of external allocations with repeated "
                   "dimensions is not yet implemented");
      myAllocNode->maxDimSizes_.resize(myAllocNode->layoutDimOrder_.size(), -1);
      myAllocNode->numBuffers_ = myAlloc.getNumBuffers();
      myAllocNode->ldsIdx_ = allocTensorOrCosntIdx.first;
      myAllocNode->component_ = storage;
      myAllocNode->name_ =
          "allocate_lds" + std::to_string(myAllocNode->ldsIdx_) + "_" +
          EnumsConversion::senComponentsToString.at(myAllocNode->component_);
      if (userNode) myAllocNode->addAllocUser(userNode);
      auto& memorg = myLds.memOrg_[storage];
      if (memorg.allocateNode_ != nullptr) {
        myAlloc.emitError("An allocation for this tensor is already present");
        DT_ERROR("Illegal ddl");
      }
      memorg.isPresent = true;
      memorg.allocateNode_ = myAllocNode;
      if (metadata_.newAllocations_.find(myAllocNode->component_) ==
          metadata_.newAllocations_.end()) {
        metadata_.newAllocations_[myAllocNode->component_];
      }
      auto& allocMeta = metadata_.newAllocations_.at(myAllocNode->component_)
                            .ldsIdxAndAllocNode;
      DT_CHECK(allocMeta.find(myAllocNode->ldsIdx_) == allocMeta.end());
      allocMeta[myAllocNode->ldsIdx_] = myAllocNode;
    } else if (allocTensorOrCosntIdx.second >= 0) {
      auto& myConst = dsc.constantInfo_.at(allocTensorOrCosntIdx.second);
      // create an allocate node..
      auto allocParent =
          ddlInterface.region2blocks_.at(myAlloc->getParentRegion());
      allocParent->addChildNode(myAllocNode, allocParent != currParent);
      myConst.allocations_[storage] = myAllocNode;
      myAllocNode->numBuffers_ = myAlloc.getNumBuffers();
      myAllocNode->constIdx_ = allocTensorOrCosntIdx.second;
      myAllocNode->component_ = storage;
      myAllocNode->name_ =
          "allocate_const" + std::to_string(myAllocNode->constIdx_) + "_" +
          EnumsConversion::senComponentsToString.at(myAllocNode->component_);
      if (userNode) {
        myAllocNode->addAllocUser(userNode);
      }
      if (metadata_.newAllocations_.find(myAllocNode->component_) ==
          metadata_.newAllocations_.end()) {
        metadata_.newAllocations_[myAllocNode->component_];
      }
      auto& allocMeta = metadata_.newAllocations_.at(myAllocNode->component_)
                            .consIdAndAllocNode;
      DT_CHECK(allocMeta.find(myAllocNode->constIdx_) == allocMeta.end());
      allocMeta[myAllocNode->constIdx_] = myAllocNode;
    } else {
      myAlloc.emitError("Allocation for no tensor and no constant");
      DT_ERROR("Illegal ddl");
    }
    ddlInterface.alloc_storage_[myAlloc.getResult()] = myAllocNode;
    return myAllocNode;
  };

  auto getRepetitionIfExists = [&](mlir::Value myUnitSsa) -> int {
    auto myUnitOp = dyn_cast<UnitOp>(myUnitSsa.getDefiningOp());
    if (!myUnitOp) return 1;
    if (!llvm::detail::isPresent(myUnitOp.getAllocation())) return 1;
    auto alloc = llvm::dyn_cast<ddl::AllocateOp>(
        myUnitOp.getAllocation().getDefiningOp());
    if (!alloc) return 1;
    int rep = 1;
    auto repetition = alloc.getReplication();
    if (repetition.has_value()) rep = repetition.value();
    return rep;
  };

  auto setDataLocAndInfo = [&](mlir::Value myUnitSsa,
                               const dsc2::ScheduleNode* userNode,
                               DataLocation& dataLoc, dsc2::DataInfo& dataInfo,
                               std::vector<SenComponents>* vias =
                                   nullptr) -> std::vector<SenComponents> {
    std::vector<SenComponents> units;
    auto myUnitOp = dyn_cast<UnitOp>(myUnitSsa.getDefiningOp());
    auto myOprConst = dyn_cast<OperandConstantOp>(myUnitSsa.getDefiningOp());
    if (!myUnitOp && !myOprConst) {
      myUnitSsa.getDefiningOp()->emitError(
          "Src/Dest/I/O is not a ddl.unit or ddl.operand_constant op");
      DT_ERROR("Illegal ddl");
    }
    if (myUnitOp) {
      bool allocAvail = llvm::detail::isPresent(myUnitOp.getAllocation());
      SenComponents storage = SenComponents::NO_COMPONENT;
      // std::cout << "AllocAvail : " << allocAvail << std::endl;
      if (allocAvail) {
        auto alit = ddlInterface.alloc_storage_.find(myUnitOp.getAllocation());
        if (alit != ddlInterface.alloc_storage_.end()) {
          storage = alit->second->component_;
          if (userNode) {
            alit->second->addAllocUser(userNode);
          }
        } else {
          if (auto myAlloc = dyn_cast<AllocateOp>(
                  myUnitOp.getAllocation().getDefiningOp())) {
            auto myAllocNode = processAllocation(myAlloc, userNode);
            storage = myAllocNode->component_;
          } else if (auto myExtAlloc =
                         dyn_cast<GetExternalDataTransferAllocationOp>(
                             myUnitOp.getAllocation().getDefiningOp())) {
            auto strit = EnumsConversion::stringToSenComponents.find(
                myExtAlloc.getMemory().str());
            if (strit == EnumsConversion::stringToSenComponents.end() ||
                !dsc2::memories.count(strit->second)) {
              myExtAlloc.emitError("Unrecognised memory");
              DT_ERROR("Illegal ddl");
            }
            storage = strit->second;
            auto allocTensorOrCosntIdx =
                getTensorOrExtConstIdx(myExtAlloc.getTensor());
            if (allocTensorOrCosntIdx.first < 0) {
              myExtAlloc.emitError("External allocation for no tensor");
              DT_ERROR("Illegal ddl");
            }
            auto& myLds = dsc.labeledDs_.at(allocTensorOrCosntIdx.first);
            auto& newMemOrg = myLds.memOrg_[storage];
            if (!newMemOrg.allocateNode_) {
              myExtAlloc.emitError(
                  "Missing pre-filled allocation in schedule tree for this "
                  "tensor-memory combination");
              DT_ERROR("Illegal ddl");
            }
            auto* myAllocNode = newMemOrg.allocateNode_;
            if (userNode) {
              myAllocNode->addAllocUser(userNode);
            }
            auto prefilledIt =
                metadata_.prefilledExternalTransferToDataConnectToFill_.find(
                    {allocTensorOrCosntIdx.first, storage});
            if (prefilledIt ==
                metadata_.prefilledExternalTransferToDataConnectToFill_.end()) {
              myExtAlloc.emitError(
                  "Missing pre-filled transfer in schedule tree for this "
                  "tensor-memory combination");
              DT_ERROR("Illegal ddl");
            }
            *prefilledIt->second = myExtAlloc.getDataConnect();
            ddlInterface.alloc_storage_[myUnitOp.getAllocation()] = myAllocNode;
          } else if (auto myC2C = dyn_cast<CoreToCoreCommunicationOp>(
                         myUnitOp.getAllocation().getDefiningOp())) {
            if (!myUnitOp.getUnit().equals_insensitive(
                    EnumsConversion::senComponentsToString.at(SFPRING))) {
              myUnitOp.emitError(
                  "Core to core communication is only allowed through sfpring");
              DT_ERROR("Illegal ddl");
            }
            // set destination cores of sfpring using start address field
            DT_CHECK(dataInfo.startAddr_.hasZeroFoldDim());
            if (myUnitOp.getAllocation() == myC2C.getNextCoreInChain()) {
              dataInfo.startAddr_.clone(
                  ddlInterface.coreToCore_definitions_.at(myC2C).nextCore_);
            } else if (myUnitOp.getAllocation() == myC2C.getPrevCoreInChain()) {
              dataInfo.startAddr_.clone(
                  ddlInterface.coreToCore_definitions_.at(myC2C).prevCore_);
            } else {
              myUnitOp.emitError(
                  "Wrong return value used from core_to_core_communication op");
              DT_ERROR("Illegal ddl");
            }
          } else {
            myUnitOp.emitError("allocation input is not a valid op");
            DT_ERROR("Illegal ddl");
          }
        }
        if (dsc2::memories.count(storage) == 0 &&
            !isa<CoreToCoreCommunicationOp>(
                myUnitOp.getAllocation().getDefiningOp())) {
          myUnitOp.emitError("Storage component is not a memory");
          DT_ERROR("Illegal ddl");
        }
      }
      units = unrollRowUnits(myUnitOp.getUnit().str(), myUnitOp);
      dataLoc.unit_ = units.front();
      if (dataLoc.unit_ == SFPRING && dataInfo.startAddr_.hasZeroFoldDim()) {
        myUnitOp.emitError(
            "SFPRING unit requested, but no core_to_core_communication pattern "
            "speficied in unit op");
        DT_ERROR("Illegal ddl");
      }
      units.erase(units.begin());
      dataLoc.storage_ = storage;

      if (dsc2::memories.count(dataLoc.storage_) &&
          !dscGlobal_.sysDef.addressGranularityScalePerUnit.count(
              {EnumsConversion::senCompToGenericComp.at(dataLoc.unit_),
               dataLoc.storage_})) {
        myUnitOp.emitError("Unit not connected to allocation component");
        DT_ERROR("Illegal ddl");
      }

      // get tensor..
      auto tensorOrCosntIdx = getTensorOrExtConstIdx(myUnitOp.getTensor());
      if (tensorOrCosntIdx.first == -1 && tensorOrCosntIdx.second == -1) {
        myUnitOp.emitError(
            "getTensor() is not a ddl.tensor / ddl.get_external_constant op OR "
            "the referenced tensor/const is not available");
        DT_ERROR("Illegal ddl");
      }
      if (tensorOrCosntIdx.first >= 0) {
        dataInfo.myLdsIdx_ = tensorOrCosntIdx.first;
      } else if (tensorOrCosntIdx.second >= 0) {
        dataInfo.constantId_ = tensorOrCosntIdx.second;
      }
      dataInfo.dataConnect_ = myUnitOp.getDataConnect().str();
      // start offset
      auto offsetElems = myUnitOp.getStickReplicatedDimOffsetElements();
      if (offsetElems.has_value()) {
        if (dataInfo.myLdsIdx_ < 0) {
          myUnitOp.emitError("Start offset can only be given for real tensors");
          DT_ERROR("Illegal ddl");
        }
        const auto& myLds = dsc.labeledDs_.at(tensorOrCosntIdx.first);
        auto& pdi = dsc.primaryDsInfo_.at(myLds.dsType_);
        if (pdi.stickDimOrder_.size() != 1 || !is_any_of(-2, myLds.scale_)) {
          myUnitOp.emitError(
              "Start offset can only be given for tensors with a reduced stick "
              "dimension");
          DT_ERROR("Illegal ddl");
        }
        auto dim = pdi.stickDimOrder_[0];
        if (offsetElems.value() >= pdi.stickSize_[0]) {
          myUnitOp.emitError(
              "Start offset can only be smaller than the stick size");
          DT_ERROR("Illegal ddl");
        }
        for (auto core : dsc.coreIdsUsed_) {
          for (int cl = 0; cl < dsc.numCoreletsUsed_DSC2_; cl++) {
            dataInfo.constEleOffsets_[core][cl].emplace(dim,
                                                        offsetElems.value());
          }
        }
      }
      // vias
      auto opVias = myUnitOp.getVias();
      if (!opVias.empty() && vias == nullptr) {
        myUnitOp.emitError(
            "Vias can only be speficied for data transfer destinations");
        DT_ERROR("Illegal ddl");
      }
      for (auto& viaAttr : myUnitOp.getVias()) {
        // cast guaranteed because of StrArrayAttr
        auto units = unrollRowUnits(cast<StringAttr>(viaAttr).str(), myUnitOp);
        if (units.size() != 1) {
          myUnitOp.emitError(
              "Vias cannot be automatically unpacked in row units");
          DT_ERROR("Illegal ddl");
        }
        vias->push_back(units[0]);
      }
    } else if (myOprConst) {
      auto oprName = myOprConst.getName().str();
      if (oprName == "0.0") {
        dataLoc.unit_ = SenComponents::ZERO;
        dataLoc.storage_ = SenComponents::ZERO;
      } else if (oprName == "1.0") {
        dataLoc.unit_ = SenComponents::ONE;
        dataLoc.storage_ = SenComponents::ONE;
      } else if (oprName == "nfwd0") {
        dataLoc.unit_ = SenComponents::NFWD0;
        dataLoc.storage_ = SenComponents::NFWD0;
      } else if (oprName == "nfwd2") {
        dataLoc.unit_ = SenComponents::NFWD2;
        dataLoc.storage_ = SenComponents::NFWD2;
      } else {
        myOprConst.emitError("Unknown name in operand const");
        DT_ERROR("Illegal ddl");
      }
      ddlInterface.operand_constant_tensor_[myUnitSsa] = dataLoc.unit_;
    }
    return units;
  };

  if (auto loop_op = dyn_cast<LoopOp>(op)) {
    auto newLoopNode = new dsc2::LoopNode();
    regionIndecesToProcess = {0};
    auto ddlDims = loop_op.getDimensions();
    for (auto ddlDim : ddlDims) {
      auto it = ddlInterface.dim_association_.find(ddlDim);
      if (it == ddlInterface.dim_association_.end()) {
        loop_op.emitError("Unknown dimension in loop operation");
        DT_ERROR("Illegal ddl");
      }
      if (!it->second.dropDim_) {
        newLoopNode->dims_.push_back(
            PrimaryDimAndKind(it->second.dim_, it->second.getMetaDimKind()));
      }
    }
    auto numIt =
        ddlInterface.datastage_definition_.find(loop_op.getNumeratorDs());
    if (numIt == ddlInterface.datastage_definition_.end()) {
      loop_op.emitError("Loop numerator datastage undefined");
      DT_ERROR("Illegal ddl");
    }
    newLoopNode->numId_ = numIt->second;
    auto denIt =
        ddlInterface.datastage_definition_.find(loop_op.getDenominatorDs());
    if (denIt == ddlInterface.datastage_definition_.end()) {
      loop_op.emitError("Loop denominator datastage undefined");
      DT_ERROR("Illegal ddl");
    }
    newLoopNode->denId_ = denIt->second;
    newLoopNode->name_ = "loop_ds" + std::to_string(newLoopNode->numId_) +
                         "_ds" + std::to_string(newLoopNode->denId_);
    for (const auto& entry : newLoopNode->dims_) {
      newLoopNode->name_ +=
          "_" + EnumsConversion::primaryDimToString.at(entry.dim_);
    }
    auto label = loop_op.getLabel();
    if (newLoopNode->dims_.empty()) {
      blockNodeForInsertion = currParent;
      delete newLoopNode;
    } else if (newLoopNode->numId_ == metadata_.core_dstgid &&
               newLoopNode->denId_ == metadata_.chunk_dstgid) {
      blockNodeForInsertion = metadata_.belowLxScheduleInsertBlock;
      delete newLoopNode;
      if (label.has_value())
        ddlInterface.core_chunk_loop_label_ = label.value().str();
    } else {
      currParent->addChildNode(newLoopNode);
      blockNodeForInsertion = newLoopNode;
      if (label.has_value())
        ddlInterface.loop_labels_[label.value().str()] = newLoopNode;
    }
  } else if (auto parametric_loop_op = dyn_cast<ParametricLoopOp>(op)) {
    auto newLoopNode = new dsc2::LoopNode();
    newLoopNode->markAsParametricLoop();
    newLoopNode->numId_ = -1;
    newLoopNode->denId_ = -1;
    blockNodeForInsertion = newLoopNode;
    regionIndecesToProcess = {0};
    auto ddlDim = parametric_loop_op.getDimension();
    auto it = ddlInterface.dim_association_.find(ddlDim);
    if (it == ddlInterface.dim_association_.end()) {
      parametric_loop_op.emitError(
          "Unknown dimension in parametric loop operation");
      DT_ERROR("Illegal ddl");
    }
    if (it->second.dropDim_) {
      parametric_loop_op.emitError(
          "Specified dimension of parametric loop operation is marked to be "
          "dropped");
      DT_ERROR("Illegal ddl");
    }
    newLoopNode->dims_.emplace_back(it->second.dim_,
                                    it->second.getMetaDimKind());
    newLoopNode->name_ =
        "parametric_loop_" +
        EnumsConversion::primaryDimToString.at(it->second.dim_) + "(" +
        EnumsConversion::metaDimKindToString.at(it->second.getMetaDimKind()) +
        ")";
    int ldsIdx =
        ddlInterface.getTensorProp(parametric_loop_op.getReferenceTensor())
            .ldsIdx_;
    if (ldsIdx == -1) {
      parametric_loop_op.emitError(
          "Specified tensor of parametric loop does not have ldsIdx_");
      DT_ERROR("Illegal ddl");
    }
    newLoopNode->setParametricLdsIdx(ldsIdx);
    currParent->addChildNode(newLoopNode);
    auto label = parametric_loop_op.getLabel();
    if (label.has_value())
      ddlInterface.loop_labels_[label.value().str()] = newLoopNode;
  } else if (auto transfer_op = dyn_cast<DataTransferOp>(op)) {
    auto newNode = new dsc2::TransferNode();
    auto remainingUnitsSrc =
        setDataLocAndInfo(transfer_op.getSource(), newNode, newNode->src_,
                          newNode->srcLdsAndLoopOffsets_);
    newNode->repetition_.srcRep_ =
        getRepetitionIfExists(transfer_op.getSource());
    auto myDstsDdl = transfer_op.getDestinations();
    std::vector<SenComponents> remainingUnitsDst;
    for (auto entry : myDstsDdl) {
      auto& newDstVia = newNode->dstVias_.emplace_back();
      auto& newDataInfo = newNode->dstLdsAndLoopOffsets_.emplace_back();
      auto remainingUnits = setDataLocAndInfo(entry, newNode, newDstVia.loc_,
                                              newDataInfo, &newDstVia.via_);
      if (remainingUnitsDst.empty()) {
        remainingUnitsDst = std::move(remainingUnits);
      } else {  // remainingUnitsDst not empty
        if (!remainingUnits.empty()) {
          transfer_op.emitError(
              "Multiple destinations with row expansion not supported");
          DT_ERROR("Illegal ddl");
        }
      }
      newNode->repetition_.dstReps_.push_back(getRepetitionIfExists(entry));
    }
    if (!remainingUnitsSrc.empty() && !remainingUnitsDst.empty() &&
        remainingUnitsSrc.size() != remainingUnitsDst.size()) {
      transfer_op.emitError(
          "Row unit expansion in source and destination has different size");
      DT_ERROR("Illegal ddl");
    }
    // update storage when coming from fifo (no matching allocate operation)..
    if (dsc2::memories.count(newNode->src_.storage_) == 0) {
      newNode->src_.storage_ =
          getGenericCompIfAvailable(newNode->dstVias_.front().loc_.unit_);
    }
    for (auto& entry : newNode->dstVias_) {
      if (dsc2::memories.count(entry.loc_.storage_) == 0) {
        entry.loc_.storage_ = getGenericCompIfAvailable(newNode->src_.unit_);
      }
    }
    auto nameTransferNode = [](dsc2::TransferNode* tn) {
      tn->name_ =
          "transfer_lds" + std::to_string(tn->srcLdsAndLoopOffsets_.myLdsIdx_) +
          "_src:" + EnumsConversion::senComponentsToString.at(tn->src_.unit_) +
          "_dst";
      for (const auto& entry : tn->dstVias_) {
        tn->name_ +=
            ":" + EnumsConversion::senComponentsToString.at(entry.loc_.unit_);
      }
    };
    auto& dtMeta = metadata_.datatransfers_[newNode];
    if (checkAccessPattern<DataTransferOp>(
            transfer_op, "access_pattern_dim",
            transfer_op.getAccessPatternDim(), "access_pattern_style",
            transfer_op.getAccessPatternStyle())) {
      processAccessPatterns<DataTransferOp,
                            ddc::Metadata::TransferAccessPatternType>(
          transfer_op, transfer_op.getAccessPatternDim(),
          transfer_op.getAccessPatternStyleAttr(),
          metadata_.datatransfers_[newNode].getMutableAccessPatternList());
    }

    // Record the SSA values in access_pattern_dims for DSC2DDL translation.
    ddlInterface.transfer_acc_pat_dims_[newNode].clear();
    for (auto dim : transfer_op.getAccessPatternDim()) {
      ddlInterface.transfer_acc_pat_dims_[newNode].push_back(dim);
    }
    // forced num_elements
    auto numElements = transfer_op.getLimitNumElementsStickReplicatedDim();
    if (numElements.has_value()) {
      dtMeta.force_num_elements_ = numElements.value();
    }

    auto rotateTypeAttr = transfer_op.getRotateNumElements();
    if (rotateTypeAttr.has_value()) {
      int64_t rotateNumElements = rotateTypeAttr.value();
      if (newNode->src_.unit_ != SenComponents::LXLU) {
        transfer_op.emitError(
            "Attribute rotate_num_elements was specified on the data_transfer "
            "operation. The source must be LXLU.");
        DT_ERROR("Illegal ddl");
      }

      // Verify the value of the rotate_num_elements attribute.
      auto unitOp = dyn_cast<UnitOp>(transfer_op.getSource().getDefiningOp());
      auto tensorIdx = ddlInterface.getTensorProp(unitOp.getTensor()).ldsIdx_;

      if (tensorIdx != -1) {
        auto tensorLds = dsc.labeledDs_.at(tensorIdx);
        int numElementsInStick = 1;
        for (auto numElements :
             dsc.primaryDsInfo_.at(tensorLds.dsType_).stickSize_) {
          numElementsInStick *= numElements;
        }

        if (rotateNumElements >= numElementsInStick) {
          transfer_op.emitError(
              "The value of rotate_num_elements must be less than " +
              std::to_string(numElementsInStick) + ".");
          DT_ERROR("Illegal ddl");
        }
        // Lx LU ISA:
        // Rotate Type (Right):
        // 000 - 0 B
        // 001 - 16B
        // 010 - 32B
        // 011 - 48B
        // 100 - 64B
        // 101 - 80B
        // 110 - 96B
        // 111 - 112B
        int bytePerElement = 128 / numElementsInStick;
        int rotateBytes = bytePerElement * rotateNumElements;
        if (rotateBytes % 16) {
          transfer_op.emitError(
              "The value of rotate_num_elements must be a multiple of " +
              std::to_string(16 / bytePerElement) + ".");
          DT_ERROR("Illegal ddl");
        }
      }

      newNode->rotateNumElements_ = rotateNumElements;
    }
    currParent->addChildNode(newNode);
    bool splitInMultipleTransfers = false;
    const auto origVias = newNode->dstVias_.at(0).via_;
    if (!remainingUnitsSrc.empty()) {
      splitInMultipleTransfers = true;
      if (remainingUnitsDst.empty()) {
        auto& vias = newNode->dstVias_.at(0).via_;
        vias.insert(vias.begin(), remainingUnitsSrc.begin(),
                    remainingUnitsSrc.end());
      }
    } else if (!remainingUnitsDst.empty()) {
      if (newNode->srcLdsAndLoopOffsets_.constantId_ >= 0 ||
          dsc.getDimIndexInLayoutOrder(
              dsc.labeledDs_.at(newNode->srcLdsAndLoopOffsets_.myLdsIdx_)
                  .dsType_,
              metadata_.rowSplitDim) >= 0)
        splitInMultipleTransfers = true;
    }
    auto* copyNode = newNode;
    for (int i = 0;
         i < std::max(remainingUnitsSrc.size(), remainingUnitsDst.size());
         i++) {
      if (splitInMultipleTransfers) {
        copyNode = new dsc2::TransferNode(*copyNode);
        auto& vias = copyNode->dstVias_.at(0).via_;
        if (!remainingUnitsSrc.empty()) {
          if (remainingUnitsDst.empty()) {
            vias.clear();
            for (int j = i + 1; j < remainingUnitsSrc.size(); j++)
              vias.push_back(remainingUnitsSrc.at(j));
            vias.insert(vias.end(), origVias.begin(), origVias.end());
          }
          copyNode->src_.unit_ = remainingUnitsSrc.at(i);
        }
        if (!remainingUnitsDst.empty()) {
          if (remainingUnitsSrc.empty())
            vias.push_back(copyNode->dstVias_.at(0).loc_.unit_);
          copyNode->dstVias_.at(0).loc_.unit_ = remainingUnitsDst.at(i);
        }
        nameTransferNode(copyNode);
        currParent->addChildNode(copyNode);
        auto& copyDtMeta =
            metadata_.datatransfers_.emplace(copyNode, dtMeta).first->second;
        if (remainingUnitsSrc.empty()) copyDtMeta.apply_row_offset_src_ = true;
        if (remainingUnitsDst.empty()) copyDtMeta.apply_row_offset_dst_ = true;
        copyNode->rotateNumElements_ = newNode->rotateNumElements_;

        // Update allocation tracker.
        dsc2::AllocateNode* allocNode = dsc.getMutableAllocation(
            copyNode->srcLdsAndLoopOffsets_, copyNode->src_.storage_, true);
        if (allocNode) {
          allocNode->addAllocUser(copyNode);
        }
        for (int i = 0, e = copyNode->dstLdsAndLoopOffsets_.size(); i < e;
             ++i) {
          allocNode = dsc.getMutableAllocation(
              copyNode->dstLdsAndLoopOffsets_.at(i),
              copyNode->dstVias_.at(i).loc_.storage_, true);
          if (allocNode) {
            allocNode->addAllocUser(copyNode);
          }
        }
      } else {
        // add to existing node
        auto& newDst = newNode->dstVias_.emplace_back(newNode->dstVias_.back());
        newDst.via_.push_back(newDst.loc_.unit_);
        newDst.loc_.unit_ = remainingUnitsDst.at(i);
        newNode->dstLdsAndLoopOffsets_.emplace_back(
            newNode->dstLdsAndLoopOffsets_.back());
      }
    }
    nameTransferNode(newNode);
  } else if (auto compute_op = dyn_cast<ComputeOp>(op)) {
    auto newNode = new dsc2::ComputeNode();
    auto myMode = compute_op.getMode();
    auto myMask = compute_op.getMask();
    auto myRepetition = compute_op.getRepetition();
    auto myIndices = compute_op.getIndices();
    if (myMode.has_value()) {
      newNode->instrAttribute_.mode_ = myMode.value();
    }
    if (myMask.has_value()) {
      newNode->instrAttribute_.compute_mask_ = myMask.value();
    }
    if (myRepetition.has_value()) {
      newNode->instrAttribute_.repetition_ = myRepetition.value();
    }
    if (myIndices.has_value()) {
      newNode->instrAttribute_.indices_.clear();
      for (int idx = 0; idx < myIndices.value().size(); idx++) {
        newNode->instrAttribute_.indices_.push_back(
            mlir::cast<IntegerAttr>(myIndices.value()[idx]).getInt());
      }
    }
    auto myInpsDdl = compute_op.getInputs();
    for (auto entry : myInpsDdl) {
      auto& newDataInfo = newNode->inputsLdsAndLoopOffsets_.emplace_back();
      DataLocation newDataLoc;
      setDataLocAndInfo(entry, newNode, newDataLoc, newDataInfo);
      if (newDataLoc.storage_ == SenComponents::NO_COMPONENT) {
        newDataLoc.storage_ = getGenericCompIfAvailable(newDataLoc.unit_);
      }
      newNode->inputs_.push_back(newDataLoc.storage_);
      newNode->repetitionWithOffset_.forInputs_.push_back(
          getRepetitionIfExists(entry));
    }
    auto myOutsDdl = compute_op.getOutputs();
    for (auto entry : myOutsDdl) {
      auto& newDataInfo = newNode->outputsLdsAndLoopOffsets_.emplace_back();
      DataLocation newDataLoc;
      setDataLocAndInfo(entry, newNode, newDataLoc, newDataInfo);
      if (newDataLoc.storage_ == SenComponents::NO_COMPONENT) {
        newDataLoc.storage_ = getGenericCompIfAvailable(newDataLoc.unit_);
      }
      newNode->outputs_.push_back(newDataLoc.storage_);
      newNode->repetitionWithOffset_.forOutputs_.push_back(
          getRepetitionIfExists(entry));
    }
    auto computeTypeStr = compute_op.getComputetype().lower();

    if (computeTypeStr == "macc") {  // auto assign FMA/IMA type based on prec
      auto& prec = dsc.computeOp_.front().attributes_.dataFormat_;
      if (prec == DataFormats::SENINT4) {
        newNode->type_ = ComputeOpType::IMA4;
      } else if (prec == DataFormats::SENINT8) {
        newNode->type_ = ComputeOpType::IMA8;
      } else if (is_any_of(prec, DataFormats::SEN143_FP8,
                           DataFormats::SEN152_FP8)) {
        newNode->type_ = ComputeOpType::FMA8;
      } else if (is_any_of(prec, DataFormats::SEN169_FP16,
                           DataFormats::BFLOAT16)) {
        newNode->type_ = ComputeOpType::FMA16;
      } else if (prec == DataFormats::IEEE_FP32) {
        newNode->type_ = ComputeOpType::FMA32;
      } else if (prec == DataFormats::SEN121_FP4) {
        newNode->type_ = ComputeOpType::FMA4;
      } else {
        compute_op.emitError("Unexpected input precision");
        DT_ERROR("Illegal ddl");
      }
      newNode->dataFormat_ = prec;
    } else {
      auto oprit = EnumsConversion::stringToComputeType.find(computeTypeStr);
      if (oprit == EnumsConversion::stringToComputeType.end()) {
        compute_op.emitError("Unknown operation in ddl.compute");
        DT_ERROR("Illegal ddl");
      }
      newNode->type_ = oprit->second;
      newNode->dataFormat_ = DataFormats::INVALID;
      if (newNode->type_ == ComputeOpType::FMA32) {
        newNode->dataFormat_ = DataFormats::IEEE_FP32;
      }
      if (newNode->dataFormat_ == DataFormats::INVALID) {
        for (auto& inputInfo : newNode->inputsLdsAndLoopOffsets_) {
          if (inputInfo.myLdsIdx_ >= 0 &&
              dsc.labeledDs_.at(inputInfo.myLdsIdx_).dataFormat_ !=
                  DataFormats::BOOL) {
            newNode->dataFormat_ =
                dsc.labeledDs_.at(inputInfo.myLdsIdx_).dataFormat_;
            break;
          }
        }
      }
      if (newNode->dataFormat_ == DataFormats::INVALID) {
        for (auto& outputInfo : newNode->outputsLdsAndLoopOffsets_) {
          if (outputInfo.myLdsIdx_ >= 0 &&
              dsc.labeledDs_.at(outputInfo.myLdsIdx_).dataFormat_ !=
                  DataFormats::BOOL) {
            newNode->dataFormat_ =
                dsc.labeledDs_.at(outputInfo.myLdsIdx_).dataFormat_;
            break;
          }
        }
      }
      auto& prec = dsc.computeOp_.front().attributes_.dataFormat_;
      if (newNode->dataFormat_ == DataFormats::IEEE_FP32) {
        // DT_CHECK(newNode->dataFormat_ == prec); // not true in sen1p5 bmm
      } else {
        newNode->dataFormat_ = DataFormats::SEN169_FP16;
      }
    }
    auto exUnitStr = compute_op.getUnit();
    auto exUnits = unrollRowUnits(exUnitStr.str(), compute_op);
    for (int i = 0; i < exUnits.size(); i++) {
      auto* nodeToInsert = i == 0 ? newNode : new dsc2::ComputeNode(*newNode);
      nodeToInsert->exUnit_ = exUnits[i];
      if (is_any_of(nodeToInsert->exUnit_, nodeToInsert->inputs_) ||
          is_any_of(nodeToInsert->exUnit_, nodeToInsert->outputs_)) {
        compute_op.emitError(
            "Input or output of compute cannot be fifo to itself");
        DT_ERROR("Illegal ddl");
      }
      nodeToInsert->name_ =
          "compute_" +
          EnumsConversion::senComponentsToString.at(nodeToInsert->exUnit_) +
          "_" + EnumsConversion::computeTypeToString.at(nodeToInsert->type_);
      currParent->addChildNode(nodeToInsert);
      if (nodeToInsert != newNode) {
        dsc2::AllocateNode* allocNode = nullptr;
        for (int i = 0, e = nodeToInsert->inputsLdsAndLoopOffsets_.size();
             i < e; ++i) {
          allocNode = dsc.getMutableAllocation(
              nodeToInsert->inputsLdsAndLoopOffsets_.at(i),
              nodeToInsert->inputs_.at(i), true);
          if (allocNode) {
            allocNode->addAllocUser(nodeToInsert);
          }
        }
        for (int i = 0, e = nodeToInsert->outputsLdsAndLoopOffsets_.size();
             i < e; ++i) {
          allocNode = dsc.getMutableAllocation(
              nodeToInsert->outputsLdsAndLoopOffsets_.at(i),
              nodeToInsert->outputs_.at(i), true);
          if (allocNode) {
            allocNode->addAllocUser(nodeToInsert);
          }
        }
      }
    }
  } else if (auto dstg_op = dyn_cast<DatastageOp>(op)) {
    auto id = dsc.dataStageParam_.size();
    while (dsc.dataStageParam_.count(id)) id++;
    auto& newDs = dsc.dataStageParam_[id];
    newDs.ss_.maxSymbolicVolume_ = newDs.el_.maxSymbolicVolume_ =
        dsc.dataStageParam_.at(metadata_.core_dstgid).ss_.maxSymbolicVolume_;
    newDs.ss_.name_ = std::to_string(id);
    ddlInterface.datastage_definition_[dstg_op.getResult()] = id;
    if (dstg_op.getStrategy().equals_insensitive("minimize")) {
      metadata_.datastages_[id].strategyMinimize_ = true;
    } else if (dstg_op.getStrategy().equals_insensitive("maximize")) {
      metadata_.datastages_[id].strategyMinimize_ = false;
    } else {
      dstg_op.emitError("Unknown strategy");
      DT_ERROR("Illegal ddl");
    }
    metadata_.datastages_[id].allowEpilogue_ = dstg_op.getAllowEpilogue();
  } else if (auto edstg_op = dyn_cast<GetExternalDatastageOp>(op)) {
    auto myProperty = edstg_op.getProperty();
    if (myProperty.equals_insensitive("chunk")) {
      DT_CHECK(metadata_.chunk_dstgid >= 0);
      ddlInterface.datastage_definition_[edstg_op.getResult()] =
          metadata_.chunk_dstgid;
    } else if (myProperty.equals_insensitive("core")) {
      DT_CHECK(metadata_.core_dstgid >= 0);
      ddlInterface.datastage_definition_[edstg_op.getResult()] =
          metadata_.core_dstgid;
    } else {
      edstg_op.emitError("External datastage property is either chunk or core");
      DT_ERROR("Illegal ddl");
    }
  } else if (auto if_op = dyn_cast<IfOp>(op)) {
    auto& condProp = processCondition(if_op.getCondition());
    if (condProp.isResolvedToBool_) {
      // only need to process one region, no need to create ConditionNode
      blockNodeForInsertion = currParent;
      if (condProp.resolvedValue_) {
        regionIndecesToProcess = {0};
      } else {
        regionIndecesToProcess = {1};
      }
    } else {
      // create ConditionNode and process both regions
      regionIndecesToProcess = {0, 1};
      auto newNode = new dsc2::ConditionNode();
      currParent->addChildNode(newNode);
      blockNodeForInsertion = newNode;
      newNode->name_ = "condition";
      newNode->loopCond_ = condProp.loopCond_;
      newNode->coreClCond_ = condProp.coreClCond_;
      // handle multiple chunk loops per dim
      for (const auto& [dim, loops] : metadata_.dimToCoreChunkLoops_) {
        if (loops.size() > 1) {
          newNode->loopCond_.adjustConditionForSplitLoop(loops.at(0), loops);
        }
      }
    }
  } else if (auto opaque_op = dyn_cast<OpaqueOp>(op)) {
    int ldsIdx =
        ddlInterface.getTensorProp(opaque_op.getReferenceTensor()).ldsIdx_;
    if (dsc.computeOp_.size() != 1) {
      opaque_op.emitError("Opaque supported only without opfusion");
      DT_ERROR("Illegal ddl");
    }
    auto& newLds = addInternalTensor(dsc.labeledDs_.at(ldsIdx), 0);
    auto newNode = new dsc2::ComputeNode();
    newNode->isOpaqueOp_ = true;
    ldsIdx = newLds.ldsIdx_;
    auto& opMetadata = metadata_.opaqueOps_[newNode];
    opMetadata.ldsIdx_ = ldsIdx;
    auto exuit =
        EnumsConversion::stringToSenComponents.find(opaque_op.getUnit().str());
    if (exuit == EnumsConversion::stringToSenComponents.end()) {
      opaque_op.emitError("Unknown unit");
      DT_ERROR("Illegal ddl");
    }
    newNode->exUnit_ = exuit->second;
    auto oprit =
        EnumsConversion::stringToComputeType.find(opaque_op.getOp().lower());
    if (oprit == EnumsConversion::stringToComputeType.end()) {
      opaque_op.emitError("Unknown operation");
      DT_ERROR("Illegal ddl");
    }

    newNode->type_ = oprit->second;
    newNode->name_ =
        "compute_opaque_" +
        EnumsConversion::senComponentsToString.at(newNode->exUnit_) + "_" +
        opaque_op.getOp().lower();
    opMetadata.max_unroll_ = opaque_op.getMaxUnrollFactor();
    for (auto& param : opaque_op.getParams()) {
      auto val = dyn_cast<StringAttr>(param.getValue());
      if (!val) {
        opaque_op.emitError("Params need to be strings");
        DT_ERROR("Illegal ddl");
      }
      auto name = param.getName().str();
      auto pos = name.find("_unroll");
      if (pos == std::string::npos) {
        newNode->instrAttribute_.param_map_[name] = val.str();
      } else {
        auto baseName = name.substr(0, pos + 1);  // base name including _
        for (int i = 0; i < opMetadata.max_unroll_; i++) {
          newNode->instrAttribute_.param_map_[baseName + std::to_string(i)] =
              val.str();
        }
      }
    }

    auto internalRegs = opaque_op.getInternalRegisters();
    if (internalRegs.empty()) {
      if (opaque_op.getMaxUnrollFactor() != 1) {
        opaque_op.emitError("No need unrolling without internal registers");
        DT_ERROR("Illegal ddl");
      }
    } else {
      auto newAlloc = new dsc2::AllocateNode();
      currParent->addChildNode(newAlloc);
      newAlloc->name_ = "allocate_" + newNode->name_;
      newAlloc->tempStorageForCompute_ = newNode;
      newAlloc->ldsIdx_ = ldsIdx;
      if (newNode->exUnit_ == PE) {
        newAlloc->component_ = PELRF;
      } else if (newNode->exUnit_ == SFP) {
        newAlloc->component_ = SFPLRF;
      } else if (newNode->exUnit_ == PT) {
        newAlloc->component_ = PTARF;
      } else {
        opaque_op.emitError("Execution unit is not a compute unit");
        DT_ERROR("Illegal ddl");
      }
      newAlloc->layoutDimOrder_ = dsc.getLayoutDims(ldsIdx);
      newAlloc->maxDimSizes_.resize(newAlloc->layoutDimOrder_.size(), -1);
      metadata_.newAllocations_[newAlloc->component_]
          .compAndAllocNode[newNode] = opMetadata.internalRegAlloc_ = newAlloc;
      auto& memOrg =
          dsc.labeledDs_.at(newAlloc->ldsIdx_).memOrg_[newAlloc->component_];
      if (memOrg.allocateNode_ != nullptr) {
        opaque_op.emitError(
            "Opaque op is trying to add an internal register allocation for a "
            "tensor for which an allocation is already present");
        DT_ERROR("Illegal ddl");
      }
      memOrg.isPresent = true;
      memOrg.allocateNode_ = newAlloc;
      newAlloc->addAllocUser(newNode);
    }
    auto& internalRegsMetadata = opMetadata.internalRegs_;
    for (auto& reg : internalRegs) {
      auto val = dyn_cast<StringAttr>(reg);
      if (!val) {
        opaque_op.emitError("internal_registers need to be strings");
        DT_ERROR("Illegal ddl");
      }
      if (val.getValue().ends_with("_unroll")) {
        opMetadata.internalRegsWithUnroll_++;
      }
      internalRegsMetadata.push_back(val.str());
    }
    auto inOutRegs = opaque_op.getInputOutputRegisters();
    auto inOutRegAllocs = opaque_op.getInputOutputAllocations();
    if (inOutRegs.size() != inOutRegAllocs.size()) {
      opaque_op.emitError(
          "Number of input/output registers and allocations should be the "
          "same");
      DT_ERROR("Illegal ddl");
    }
    auto& inOutRegsMetadata = opMetadata.inOutRegAllocs_;
    for (int i = 0; i < inOutRegs.size(); i++) {
      auto name = dyn_cast<StringAttr>(inOutRegs[i]);
      if (!name) {
        opaque_op.emitError("input_output_registers need to be strings");
        DT_ERROR("Illegal ddl");
      }
      auto allocIt = ddlInterface.alloc_storage_.find(inOutRegAllocs[i]);
      if (allocIt != ddlInterface.alloc_storage_.end()) {
        inOutRegsMetadata[name.str()] = allocIt->second;
      } else {
        // process allocate
        if (auto allocOp =
                dyn_cast<AllocateOp>(inOutRegAllocs[i].getDefiningOp())) {
          inOutRegsMetadata[name.str()] = processAllocation(allocOp, newNode);
        } else {
          opaque_op.emitError(
              "input_output_registers allocation not an allocation");
          DT_ERROR("Illegal ddl");
        }
      }
    }

    for (auto inputDcIt :
         opaque_op.getInputDataConnects().getAsRange<mlir::StringAttr>()) {
      newNode->instrAttribute_.input_data_connects_.push_back(inputDcIt.str());
    }
    for (auto outputDcIt :
         opaque_op.getOutputDataConnects().getAsRange<mlir::StringAttr>()) {
      newNode->instrAttribute_.output_data_connects_.push_back(
          outputDcIt.str());
    }
    newNode->dataFormat_ = dsc.computeOp_.front().attributes_.dataFormat_;
    currParent->addChildNode(newNode);
  } else if (auto sync_op = dyn_cast<SyncOp>(op)) {
    auto newNode = new dsc2::SyncNode();
    newNode->isReceive_ = sync_op.getIsReceive();
    auto label = sync_op.getSignalName().str();
    auto& syncProp = ddlInterface.sync_definitions_[label];
    if (sync_op.getSeparateCorelets() != syncProp.separateCorelets_) {
      if (syncProp.separateCorelets_ || !syncProp.syncsPerCl_.empty()) {
        sync_op.emitError(
            "separate_corelets not matching other syncs with same signal_name");
        DT_ERROR("Illegal ddl");
      }
      syncProp.separateCorelets_ = sync_op.getSeparateCorelets();
    }
    newNode->name_ = "sync_" + label;
    for (auto& unitName : sync_op.getUnits()) {
      // cast guaranteed because of StrArrayAttr
      auto units = unrollRowUnits(cast<StringAttr>(unitName).str(), sync_op);
      for (const auto& unit : units) {
        newNode->units_.insert(unit);
        newNode->name_ += "_" + EnumsConversion::senComponentsToString.at(unit);
      }
    }
    newNode->name_ += newNode->isReceive_ ? "_receive" : "_send";
    if (syncProp.separateCorelets_ && dsc.numCoreletsUsed_DSC2_ == 2) {
      // create if/else structure based on cl0/cl1
      auto ifNode = new dsc2::ConditionNode();
      currParent->addChildNode(ifNode);
      ifNode->name_ = "condition_separate_corelets_" + newNode->name_;
      for (const auto& core : dsc.coreIdsUsed_) ifNode->coreClCond_[core] = {0};
      auto thenBlockNode = new dsc2::BlockNode();
      thenBlockNode->name_ = ifNode->name_ + "_then_region";
      ifNode->addThenRegion(thenBlockNode);
      thenBlockNode->addChildNode(newNode);
      auto elseBlockNode = new dsc2::BlockNode();
      elseBlockNode->name_ = ifNode->name_ + "_else_region";
      ifNode->addElseRegion(elseBlockNode);
      auto newNodeCl1 = static_cast<dsc2::SyncNode*>(newNode->clone());
      elseBlockNode->addChildNode(newNodeCl1);
      newNode->name_ += "_cl0";
      newNodeCl1->name_ += "_cl1";
      auto& destVect0 = newNode->isReceive_ ? syncProp.syncsPerCl_[0].receivers_
                                            : syncProp.syncsPerCl_[0].senders_;
      destVect0.push_back(newNode);
      auto& destVect1 = newNode->isReceive_ ? syncProp.syncsPerCl_[1].receivers_
                                            : syncProp.syncsPerCl_[1].senders_;
      destVect1.push_back(newNodeCl1);
    } else {
      currParent->addChildNode(newNode);
      auto& destVect = newNode->isReceive_ ? syncProp.syncsPerCl_[-1].receivers_
                                           : syncProp.syncsPerCl_[-1].senders_;
      destVect.push_back(newNode);
    }
  } else if (auto implSync_op = dyn_cast<ImplicitSyncOp>(op)) {
    auto newNode = new dsc2::SyncNode();
    const dsc2::AllocateNode* alloc = nullptr;
    auto alit = ddlInterface.alloc_storage_.find(implSync_op.getAllocate());
    if (alit != ddlInterface.alloc_storage_.end()) {
      alloc = alit->second;
    } else {
      alloc = processAllocation(
          cast<AllocateOp>(implSync_op.getAllocate().getDefiningOp()), newNode);
    }
    if (!is_any_of(alloc->component_, L0, L0_SCALE)) {
      implSync_op.emitError(
          "Implicit syncs are only possible between L0 LU/SU units");
      DT_ERROR("Illegal ddl");
    }
    if (alloc->numBuffers_ != -1) {
      implSync_op.emitError(
          "Implicit syncs are only possible on circular buffers "
          "(num_buffers=-1)");
      DT_ERROR("Illegal ddl");
    }

    currParent->addChildNode(newNode);
    newNode->units_ = {L0SU};
    // insert L0LUrows
    for (auto i = 0; i < dscGlobal_.sysDef.numPTRows; i++) {
      newNode->units_.insert(SenComponents(L0LUROW0 + i));
    }
    newNode->name_ = "sync_implicit_L0";
    metadata_.implicitSyncs_[newNode] = alloc;
  } else if (auto dsConstr_op = dyn_cast<DatastageConstraintOp>(op)) {
    auto dsIt =
        ddlInterface.datastage_definition_.find(dsConstr_op.getDatastage());
    if (dsIt == ddlInterface.datastage_definition_.end()) {
      dsConstr_op.emitError("Datastage to constraint not processed before");
      DT_ERROR("Illegal ddl");
    }
    auto reference = dsConstr_op.getReference();
    int refDsId = -1;  //-1 means reference is tensor, not a datastage
    std::unordered_map<PrimaryDimTypes, int> stickSizes;
    if (auto refDsIt = ddlInterface.datastage_definition_.find(reference);
        refDsIt != ddlInterface.datastage_definition_.end()) {
      refDsId = refDsIt->second;
    } else {
      try {
        stickSizes = dsc.getCumulativeStickSizes(
            dsc.labeledDs_.at(ddlInterface.getTensorProp(reference).ldsIdx_)
                .dsType_);
      } catch (const DtException& e) {
        dsConstr_op.emitError(
            "Reference is neither a valid datastage nor a valid tensor");
        throw;
      }
    }
    std::set<PrimaryDimTypes> dims;
    float scale = 1;
    for (auto dim : dsConstr_op.getDimensions()) {
      auto dimIt = ddlInterface.dim_association_.find(dim);
      if (dimIt == ddlInterface.dim_association_.end()) {
        dsConstr_op.emitError("Dimension not valid");
        DT_ERROR("Illegal ddl");
      }
      if (!dimIt->second.dropDim_) {
        if (is_any_of(dimIt->second.getMetaDimKind(), MetaDimKind::Padded,
                      MetaDimKind::Unpadded, MetaDimKind::WindowDim)) {
          dims.insert(dimIt->second.dim_);
        } else {
          scale /= dsc.dataStageParam_.at(metadata_.core_dstgid)
                       .ss_.paddingSizes_.at(dimIt->second.dim_)
                       .getMetaDimVal(dimIt->second.getMetaDimKind());
        }
      }
    }
    for (const auto& [dim, size] : stickSizes)
      if (dims.count(dim)) scale *= size;
    auto& dsConstraints =
        metadata_.datastages_.at(dsIt->second).constraints_[refDsId][dims];
    if (auto min = dsConstr_op.getMin(); min.has_value()) {
      dsConstraints.updateMin(processExpression(min.value().str()) * scale);
    }
    if (auto max = dsConstr_op.getMax(); max.has_value()) {
      dsConstraints.updateMax(processExpression(max.value().str()) * scale);
    }
    std::set<float> valConstr;
    for (auto value : dsConstr_op.getValues()) {
      valConstr.insert(processExpression(cast<StringAttr>(value).str()) *
                       scale);
    }
    if (!valConstr.empty()) {
      dsConstraints.updateValues(valConstr);
      if (dsConstraints.values_->empty()) {
        dsConstr_op.emitError(
            "Values constraint, combined with existing values constraint, "
            "create an empty set");
        DT_ERROR("Illegal ddl");
      }
    }
  } else if (auto innerDims_op = dyn_cast<ForceInnermostDimensionsOp>(op)) {
    dsc2::AllocateNode* allocNode = nullptr;
    auto alit = ddlInterface.alloc_storage_.find(innerDims_op.getAllocate());
    if (alit != ddlInterface.alloc_storage_.end()) {
      allocNode = alit->second;
    } else {
      allocNode = processAllocation(
          cast<AllocateOp>(innerDims_op.getAllocate().getDefiningOp()),
          nullptr);
    }
    if (allocNode->ldsIdx_ < 0) {
      innerDims_op.emitError("Op can be applied only to tensor allocations");
      DT_ERROR("Illegal ddl");
    }
    auto dsIt =
        ddlInterface.datastage_definition_.find(innerDims_op.getDatastage());
    if (dsIt == ddlInterface.datastage_definition_.end()) {
      innerDims_op.emitError("Datastage not processed before");
      DT_ERROR("Illegal ddl");
    }
    if (std::any_of(allocNode->maxDimSizes_.begin(),
                    allocNode->maxDimSizes_.end(),
                    [](int x) { return x >= 0; })) {
      innerDims_op.emitError(
          "Cannot apply this type of op to an allocation more than once");
      DT_ERROR("Illegal ddl");
    }
    auto dims = innerDims_op.getDims();
    for (int i = dims.size() - 1; i >= 0; i--) {
      auto dim = dims[i];
      auto dimIt = ddlInterface.dim_association_.find(dim);
      if (dimIt == ddlInterface.dim_association_.end()) {
        innerDims_op.emitError("Dimension cannot be used");
        dim.getDefiningOp()->emitError("Dimension not processed before");
        DT_ERROR("Illegal ddl");
      }
      if (dimIt->second.dropDim_) {
        innerDims_op.emitError("Dimension cannot be used");
        dim.getDefiningOp()->emitError("Dimension marked as dropped");
        DT_ERROR("Illegal ddl");
      }
      allocNode->layoutDimOrder_.insert(allocNode->layoutDimOrder_.begin(),
                                        dimIt->second.dim_);
      // store the Datastage index in the maxDimSizes vector. It will be later
      // converted into an actual size
      allocNode->maxDimSizes_.insert(allocNode->maxDimSizes_.begin(),
                                     dsIt->second);
    }
  } else if (auto c2c_op = dyn_cast<CoreToCoreCommunicationOp>(op)) {
    auto& c2cProp = ddlInterface.coreToCore_definitions_[c2c_op];
    bool dimSplitFound = false;
    for (const auto& dimValue : c2c_op.getDimensions()) {
      auto dimIt = ddlInterface.dim_association_.find(dimValue);
      if (dimIt == ddlInterface.dim_association_.end() ||
          dimIt->second.dropDim_)
        continue;
      if (c2cProp.dim_ == PrimaryDimTypesCount)  // initialize it
        c2cProp.dim_ = dimIt->second.dim_;
      if (sdsc.numWkSlicesPerDim_.at(dimIt->second.dim_) > 1) {
        if (dimSplitFound) {
          c2c_op.emitError(
              "More than one reduction dim is split across cores: not "
              "currently supported");
          DT_ERROR("Illegal ddl");
        }
        c2cProp.dim_ = dimIt->second.dim_;
        dimSplitFound = true;
      }
    }

    if (c2cProp.dim_ == PrimaryDimTypesCount) {
      c2c_op.emitError("None of the dimensions is mapped");
      DT_ERROR("Illegal ddl");
    }

    int numSlices = sdsc.numWkSlicesPerDim_.at(c2cProp.dim_);
    auto& startCond =
        ddlInterface.resolvedConditions_[c2c_op.getChainStartCore()];
    auto& endCond = ddlInterface.resolvedConditions_[c2c_op.getChainEndCore()];
    // initialize fold managers
    DT_CHECK(c2cProp.nextCore_.hasZeroFoldDim() &&
             c2cProp.prevCore_.hasZeroFoldDim());
    std::deque<const FoldDimProp*> sdscFoldProps{sdsc.coreFoldProp_.get(),
                                                 sdsc.coreletFoldProp_.get()};
    std::deque<BaseFuncType> foldTypes(sdscFoldProps.size(), BaseFuncType::Map);
    for (const auto& foldProp : sdsc.sdscFoldProps_) {
      sdscFoldProps.push_back(foldProp.get());
      foldTypes.push_back(BaseFuncType::Constant);
    }
    c2cProp.nextCore_.buildFoldSpace(sdscFoldProps, foldTypes);
    c2cProp.prevCore_.buildFoldSpace(sdscFoldProps, foldTypes);
    if (numSlices == 1) {
      startCond.resolvedValue_ = startCond.isResolvedToBool_ =
          endCond.resolvedValue_ = endCond.isResolvedToBool_ = true;
    } else {
      auto& startCoreCl = startCond.coreClCond_;
      auto& endCoreCl = endCond.coreClCond_;
      std::deque<int64_t> coordCl0(sdscFoldProps.size(), 0),
          coordCl1 = coordCl0;
      coordCl1[1] = 1;
      // cl0 sfp ring goes in increasing coreId direction, cl1 the other way
      for (auto& [coreId, wkSlice] : sdsc.coreIdToWkSlice_) {
        if (std::find(dsc.coreIdsUsed_.begin(), dsc.coreIdsUsed_.end(),
                      coreId) == dsc.coreIdsUsed_.end())
          continue;
        auto& dimSlice = wkSlice.at(c2cProp.dim_);
        auto wkSliceCopy = wkSlice;
        auto findCoreBySlice = [&](int sliceId) {
          wkSliceCopy.at(c2cProp.dim_) = sliceId;
          for (auto& [coreId_, wkSlice_] : sdsc.coreIdToWkSlice_) {
            if (wkSlice_ == wkSliceCopy) return coreId_;
          }
          DT_ERROR("Slice not found");
        };
        coordCl0[0] = coordCl1[0] = coreId;
        if (dimSlice == 0) {
          startCoreCl[coreId].insert(0);
          auto tgtCore = findCoreBySlice(dimSlice + 1);
          c2cProp.nextCore_.insertData(tgtCore, coordCl0);
          if (dsc.numCoreletsUsed_DSC2_ == 2) {
            endCoreCl[coreId].insert(1);
            c2cProp.prevCore_.insertData(tgtCore, coordCl1);
          }
        } else if (dimSlice == numSlices - 1) {
          endCoreCl[coreId].insert(0);
          auto tgtCore = findCoreBySlice(dimSlice - 1);
          c2cProp.prevCore_.insertData(tgtCore, coordCl0);
          if (dsc.numCoreletsUsed_DSC2_ == 2) {
            startCoreCl[coreId].insert(1);
            c2cProp.nextCore_.insertData(tgtCore, coordCl1);
          }
        } else {
          auto tgtCorep1 = findCoreBySlice(dimSlice + 1);
          auto tgtCorem1 = findCoreBySlice(dimSlice - 1);
          c2cProp.nextCore_.insertData(tgtCorep1, coordCl0);
          c2cProp.prevCore_.insertData(tgtCorem1, coordCl0);
          if (dsc.numCoreletsUsed_DSC2_ == 2) {
            c2cProp.prevCore_.insertData(tgtCorep1, coordCl1);
            c2cProp.nextCore_.insertData(tgtCorem1, coordCl1);
          }
        }
      }
    }
  } else if (dyn_cast<CoreCoreletCondOp>(op)) {
    op.emitError("Uses of this op is Prohibited in ddl");
  }
  return {blockNodeForInsertion, regionIndecesToProcess};
}

// ------------------------------------------------------------------------------------------------
// entry 324/382   level 2   scc 332   35 body lines
// unit: e324_processTransformations
// authority: ddc/ddl/ddl_conversion.cpp:2036
// original: void DdlConversion::processTransformations(::mlir::Region& myRegion)
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
void e324_processTransformations(::mlir::Region& myRegion)
{
  for (Block& block : myRegion.getBlocks()) {
    for (Operation& op : block.getOperations()) {
      if (auto disableTransferPromotionOp =
              dyn_cast<DisableTransferPromotion>(op)) {
        if (verbose_ > 0) {
          std::cerr << "\nDDL specification disables promotion of transfer.";
        }
        metadata_.transformationConfig_.enableMovingDataTransfer = false;
      } else if (auto ifOp = dyn_cast<IfOp>(op)) {
        auto& condProp = processCondition(ifOp.getCondition());
        if (condProp.isResolvedToBool_) {
          auto ifRegions = ifOp.getRegions();
          if (condProp.resolvedValue_) {
            processTransformations(*ifRegions[0]);
          } else {
            processTransformations(*ifRegions[1]);
          }
        } else {
          op.dump();
          DT_ERROR(
              "Illegal ddl: An if-condition in the transformations section "
              "must be known at compile-time.");
        }
      } else if (auto yieldOp = dyn_cast<YieldOp>(op)) {
        // No processing is needed.
      } else {
        op.dump();
        DT_ERROR(
            "Illegal ddl: Unexpected operation found in the transformations "
            "section.");
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 325/382   level 2   scc 350   627 body lines
// unit: e325_convertDsc2Ddl
// authority: ddc/ddl/ddl_conversion.cpp:2881
// original: void DdlConversion::convertDsc2Ddl(std::ostream& outputDdl)
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
void e325_convertDsc2Ddl(std::ostream& outputDdl)
{
  std::unordered_map<const dsc2::SyncNode*, std::string> external_syncs;
  auto ddlMlirRoot = ddlParser_.ddl_module_op_.get();
  ddlMlirRoot->walk<WalkOrder::PreOrder>([&](DimensionOp dop) {
    std::vector<mlir::Attribute> dim_mappings;
    for (auto res : dop->getResults()) {
      if (ddlInterface.dim_association_.count(res)) {
        auto dimprop = ddlInterface.dim_association_.at(res);
        if (EnumsConversion::primaryDimToString.count(dimprop.dim_))
          dim_mappings.push_back(StringAttr::get(
              ddlMlirRoot->getContext(),
              EnumsConversion::primaryDimToString.at(dimprop.dim_)));
        else {
          dim_mappings.push_back(
              StringAttr::get(ddlMlirRoot->getContext(), "ignored"));
        }
      } else {
        dim_mappings.push_back(
            StringAttr::get(ddlMlirRoot->getContext(), "ignored"));
      }
    }
    std::vector<mlir::Attribute> dim_mapping(dim_mappings.begin(),
                                             dim_mappings.end());
    dop->setAttr("dim_mapping", ArrayAttr::get(dop->getContext(), dim_mapping));
  });
  ddlMlirRoot->walk<WalkOrder::PreOrder>([&](TensorOp top) {
    std::vector<mlir::Attribute> lds_mappings;
    for (auto res : top->getResults()) {
      if (ddlInterface.tensor_definition_.count(res))
        lds_mappings.push_back(IntegerAttr::get(
            IntegerType::get(top->getContext(), 64),
            APInt(64, ddlInterface.tensor_definition_.at(res).ldsIdx_)));
      else
        lds_mappings.push_back(IntegerAttr::get(
            IntegerType::get(top->getContext(), 64), APInt(64, -1)));
    }
    top->setAttr("lds_mapping",
                 ArrayAttr::get(top->getContext(), lds_mappings));
  });
  // Now lets do the dataflow section..
  DataflowOp original_dataflow_op = nullptr;
  ddlMlirRoot->walk<WalkOrder::PreOrder>(
      [&](DataflowOp dfOp) { original_dataflow_op = dfOp; });
  OpBuilder builder(ddlMlirRoot);
  original_dataflow_op->setAttr("original_dataflow",
                                BoolAttr::get(builder.getContext(), true));
  builder.setInsertionPoint(original_dataflow_op);

  std::stack<const dsc2::ScheduleNode*> stack;
  std::unordered_map<const dsc2::ScheduleNode*, bool> visited;
  const dsc2::ScheduleNode* currParent = dsc.scheduleTree_.getHead();
  auto next_view = dynamic_cast<const dsc2::BlockNode*>(currParent)
                       ->getNextView(SenComponents::ALL);
  std::reverse(next_view.begin(), next_view.end());
  for (auto next : next_view) {
    stack.push(next);
  }
  auto new_dataflow_op =
      DataflowOp::create(builder, original_dataflow_op->getLoc());
  new_dataflow_op->setAttr("original_dataflow",
                           BoolAttr::get(builder.getContext(), false));
  builder.createBlock(&new_dataflow_op.getBody());
  YieldOp::create(builder, builder.getUnknownLoc());
  builder.setInsertionPointToStart(&new_dataflow_op.getBody().front());

  std::stack<mlir::OpBuilder::InsertPoint> insPts;
  llvm::DenseMap<AllocateOp, const dsc2::AllocateNode*> allocations;
  std::map<std::string, GetExternalDatastageOp> name_to_external_datastage_op;
  std::map<std::string, DatastageOp> name_to_datastage_op;
  IfOp ifop_then = nullptr, ifop_else = nullptr;
  while (!stack.empty()) {
    auto* curHead = stack.top();
    if (visited[curHead]) {
      stack.pop();
      if (curHead->nodeType_ == dsc2::ScheduleNode::LOOP ||
          curHead->nodeType_ == dsc2::ScheduleNode::CONDITION ||
          curHead->nodeType_ == dsc2::ScheduleNode::BLOCK) {
        builder.restoreInsertionPoint(insPts.top());
        insPts.pop();
      }
      continue;
    }
    visited[curHead] = true;

    // use these allocations in ops which follow it.
    if (curHead->nodeType_ == dsc2::ScheduleNode::LOOP) {
      auto loopnode = static_cast<const dsc2::LoopNode*>(curHead);
      std::string label = "";
      for (auto kv : ddlInterface.loop_labels_) {
        if (kv.second == loopnode) label = kv.first;
      }
      GetExternalDatastageOp num_ds = nullptr, den_ds = nullptr;
      DatastageOp den_ds0 = nullptr, num_ds0 = nullptr;
      if (!dsc.dataStageParam_[loopnode->numId_].name().empty()) {
        std::string core_or_chunk =
            loopnode->numId_ == metadata_.chunk_dstgid ? "chunk" : "core";
        if (name_to_external_datastage_op.count(core_or_chunk)) {
          num_ds = name_to_external_datastage_op.at(core_or_chunk);
        } else {
          num_ds = GetExternalDatastageOp::create(
              builder, builder.getUnknownLoc(), builder.getIndexType(),
              core_or_chunk);
          name_to_external_datastage_op[core_or_chunk] = num_ds;
        }
      } else {
        std::string strategy = "maximize";
        if (metadata_.datastages_[loopnode->numId_].strategyMinimize_)
          strategy = "minimize";
        if (name_to_datastage_op.count(strategy)) {
          num_ds0 = name_to_datastage_op.at(strategy);
        } else {
          num_ds0 =
              DatastageOp::create(builder, builder.getUnknownLoc(),
                                  builder.getIndexType(), StringRef(strategy));
          name_to_datastage_op[strategy] = num_ds0;
        }
      }
      if (!dsc.dataStageParam_[loopnode->denId_].name().empty()) {
        std::string core_or_chunk =
            loopnode->denId_ == metadata_.chunk_dstgid ? "chunk" : "core";
        if (name_to_external_datastage_op.count(core_or_chunk)) {
          den_ds = name_to_external_datastage_op.at(core_or_chunk);
        } else {
          den_ds = GetExternalDatastageOp::create(
              builder, builder.getUnknownLoc(), builder.getIndexType(),
              core_or_chunk);
          name_to_external_datastage_op[core_or_chunk] = den_ds;
        }
      } else {
        std::string strategy = "maximize";
        if (metadata_.datastages_[loopnode->denId_].strategyMinimize_)
          strategy = "minimize";
        if (name_to_datastage_op.count(strategy)) {
          den_ds0 = name_to_datastage_op.at(strategy);
        } else {
          den_ds0 =
              DatastageOp::create(builder, builder.getUnknownLoc(),
                                  builder.getIndexType(), StringRef(strategy));
          name_to_datastage_op[strategy] = den_ds0;
        }
      }
      LoopOp newloopop = nullptr;
      std::vector<Value> dims;
      for (auto kv_dim : loopnode->dims_) {
        for (auto kv_dim0 : ddlInterface.dim_association_) {
          if (kv_dim0.second.dim_ == kv_dim.dim_ &&
              (kv_dim0.second.getMetaDimKind() == kv_dim.kind_ ||
               (kv_dim.kind_ == MetaDimKind::WindowDim &&
                kv_dim0.second.isUnpadded()))) {
            dims.push_back(kv_dim0.first);
            break;
          }
        }
      }
      DT_CHECK(loopnode->dims_.size() == dims.size());
      if (den_ds) {
        if (num_ds)
          newloopop =
              LoopOp::create(builder, builder.getUnknownLoc(),
                             StringAttr::get(builder.getContext(), label),
                             num_ds->getResult(0), den_ds->getResult(0),
                             ValueRange{ArrayRef<Value>(dims)});
        else
          newloopop =
              LoopOp::create(builder, builder.getUnknownLoc(),
                             StringAttr::get(builder.getContext(), label),
                             num_ds0->getResult(0), den_ds->getResult(0),
                             ValueRange{ArrayRef<Value>(dims)});
      } else {
        if (num_ds)
          newloopop =
              LoopOp::create(builder, builder.getUnknownLoc(),
                             StringAttr::get(builder.getContext(), label),
                             num_ds->getResult(0), den_ds0->getResult(0),
                             ValueRange{ArrayRef<Value>(dims)});
        else
          newloopop =
              LoopOp::create(builder, builder.getUnknownLoc(),
                             StringAttr::get(builder.getContext(), label),
                             num_ds0->getResult(0), den_ds0->getResult(0),
                             ValueRange{ArrayRef<Value>(dims)});
      }
      newloopop->setAttr("name",
                         StringAttr::get(builder.getContext(),
                                         mlir::StringRef(loopnode->name_)));
      auto denstg = dsc.dataStageParam_.at(loopnode->denId_);
      auto numstg = dsc.dataStageParam_.at(loopnode->numId_);
      llvm::SmallVector<NamedAttribute> dim_to_count;
      for (auto dim : loopnode->dims_) {
        int num_count = numstg.ss_.primaryDimToVal_st(dim.dim_);
        int den_count = denstg.ss_.primaryDimToVal_st(dim.dim_);
        size_t count = std::ceil(num_count * 1.0 / den_count);
        dim_to_count.push_back(builder.getNamedAttr(
            EnumsConversion::primaryDimToString.at(dim.dim_),
            builder.getStringAttr(std::to_string(count))));
      }
      newloopop->setAttr("ss_loop_count",
                         builder.getDictionaryAttr(dim_to_count));
      insPts.push(builder.saveInsertionPoint());
      builder.createBlock(&newloopop.getBody());
      YieldOp::create(builder, builder.getUnknownLoc());
      builder.setInsertionPointToStart(&newloopop.getBody().front());
    } else if (curHead->nodeType_ == dsc2::ScheduleNode::COMPUTE) {
      auto computenode = static_cast<const dsc2::ComputeNode*>(curHead);
      std::vector<Value> inputs, outputs;
      auto createUnitOp = [&](SenComponents sencomp,
                              const dsc2::DataInfo& dtinfo,
                              std::vector<Value>& unitsVector) {
        SenComponents sencomp0;
        Value tensor = nullptr, allocation = nullptr;
        if (dsc2::memories.count(sencomp) > 0) {
          std::tie(tensor, allocation) =
              getTensorAndAllocation(allocations, sencomp, dtinfo);
          sencomp0 = computenode->exUnit_;
        } else {
          tensor = getTensor(sencomp, dtinfo);
          sencomp0 = sencomp;
        }

        if (tensor && tensor.getDefiningOp<OperandConstantOp>()) {
          unitsVector.push_back(tensor);
          return;
        }
        IntegerAttr stickOffset(nullptr);
        if (!dtinfo.constEleOffsets_.empty()) {
          stickOffset =
              IntegerAttr::get(IntegerType::get(builder.getContext(), 64),
                               APInt(64, dtinfo.constEleOffsets_.begin()
                                             ->second.begin()
                                             ->second.begin()
                                             ->second));
        }
        auto unit = UnitOp::create(
            builder, builder.getUnknownLoc(), builder.getIndexType(), tensor,
            allocation,
            StringAttr::get(
                builder.getContext(),
                mlir::StringRef(
                    EnumsConversion::senComponentsToString.at(sencomp0))),
            StringAttr::get(builder.getContext(),
                            mlir::StringRef(dtinfo.dataConnect_)),
            stickOffset, builder.getStrArrayAttr(ArrayRef<StringRef>()));
        unitsVector.push_back(unit);
      };
      for (auto [sencomp, dtinfo] : llvm::zip(
               computenode->inputs_, computenode->inputsLdsAndLoopOffsets_)) {
        createUnitOp(sencomp, dtinfo, inputs);
      }
      for (auto [sencomp, dtinfo] : llvm::zip(
               computenode->outputs_, computenode->outputsLdsAndLoopOffsets_)) {
        createUnitOp(sencomp, dtinfo, outputs);
      }

      std::vector<Attribute> indices;
      for (auto& index : computenode->instrAttribute_.indices_) {
        indices.push_back(builder.getI64IntegerAttr(index));
      }
      auto newcomputeop = ComputeOp::create(
          builder, builder.getUnknownLoc(),
          StringRef(
              EnumsConversion::computeTypeToString.at(computenode->type_)),
          StringRef(
              EnumsConversion::senComponentsToString.at(computenode->exUnit_)),
          IntegerAttr::get(IntegerType::get(builder.getContext(), 64),
                           APInt(64, computenode->instrAttribute_.mode_)),
          IntegerAttr::get(
              IntegerType::get(builder.getContext(), 64),
              APInt(64, computenode->instrAttribute_.compute_mask_)),
          IntegerAttr::get(IntegerType::get(builder.getContext(), 64),
                           APInt(64, computenode->instrAttribute_.repetition_)),
          builder.getArrayAttr(indices), ValueRange(ArrayRef<Value>(inputs)),
          ValueRange(ArrayRef<Value>(outputs)));
      newcomputeop->setAttr(
          "name", StringAttr::get(builder.getContext(),
                                  mlir::StringRef(computenode->name_)));
    } else if (curHead->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
      auto transfernode = static_cast<const dsc2::TransferNode*>(curHead);
      auto createUnitOp =
          [&](const dsc2::DataInfo& dtinfo, const DataLocation& loc,
              const std::vector<SenComponents>* vias) -> UnitOp {
        Value allocation = nullptr, tensor = nullptr;
        if (dsc2::memories.count(loc.storage_) > 0)
          std::tie(tensor, allocation) =
              getTensorAndAllocation(allocations, loc.storage_, dtinfo);
        else
          tensor = getTensor(loc.storage_, dtinfo);
        IntegerAttr stickOffset(nullptr);
        if (!dtinfo.constEleOffsets_.empty()) {
          stickOffset =
              IntegerAttr::get(IntegerType::get(builder.getContext(), 64),
                               APInt(64, dtinfo.constEleOffsets_.begin()
                                             ->second.begin()
                                             ->second.begin()
                                             ->second));
        }
        std::vector<StringRef> viasVect;
        if (vias) {
          for (auto& via : *vias) {
            viasVect.emplace_back(
                EnumsConversion::senComponentsToString.at(via));
          }
        }
        return UnitOp::create(
            builder, builder.getUnknownLoc(), builder.getIndexType(), tensor,
            allocation,
            StringAttr::get(
                builder.getContext(),
                mlir::StringRef(
                    EnumsConversion::senComponentsToString.at(loc.unit_))),
            StringAttr::get(builder.getContext(),
                            mlir::StringRef(dtinfo.dataConnect_)),
            stickOffset,
            builder.getStrArrayAttr(ArrayRef<StringRef>(viasVect)));
      };

      UnitOp source = createUnitOp(transfernode->srcLdsAndLoopOffsets_,
                                   transfernode->src_, nullptr);
      std::vector<Value> destinations;
      for (auto [dstdatainfo, dstvia] : llvm::zip(
               transfernode->dstLdsAndLoopOffsets_, transfernode->dstVias_)) {
        UnitOp dst = createUnitOp(dstdatainfo, dstvia.loc_, &dstvia.via_);
        destinations.push_back(dst->getResult(0));
      }

      // In preparation for generating the access_pattern_styles attribute
      // construct a vector of pattern strings.
      std::vector<Attribute> transferAccPatternStyles;
      for (auto it : ddlInterface.transfer_acc_pat_dims_[transfernode]) {
        PrimaryDimTypes dim = ddlInterface.dim_association_[it].dim_;
        transferAccPatternStyles.push_back(builder.getStringAttr(
            metadata_.datatransfers_[transfernode].getAccessPatternAsStr(dim)));
      }

      IntegerAttr limitNumElem(nullptr);
      if (auto dtMetaIt = metadata_.datatransfers_.find(transfernode);
          dtMetaIt != metadata_.datatransfers_.end() &&
          dtMetaIt->second.force_num_elements_ > 0) {
        limitNumElem =
            IntegerAttr::get(IntegerType::get(builder.getContext(), 64),
                             APInt(64, dtMetaIt->second.force_num_elements_));
      }
      auto new_data_transfer_op = DataTransferOp::create(
          builder, builder.getUnknownLoc(),
          builder.getArrayAttr(transferAccPatternStyles), limitNumElem,
          IntegerAttr::get(IntegerType::get(builder.getContext(), 64),
                           transfernode->rotateNumElements_),
          source, ValueRange(ArrayRef<Value>(destinations)),
          ValueRange(ArrayRef<Value>(
              ddlInterface.transfer_acc_pat_dims_.at(transfernode))));
      new_data_transfer_op->setAttr(
          "name", StringAttr::get(builder.getContext(),
                                  mlir::StringRef(transfernode->name_)));
      if (!transfernode->getRelevantCoreCl().empty())
        new_data_transfer_op->setAttr(
            "transfer_size",
            builder.getStringAttr(std::to_string(dsc.getBlockTransferSize(
                *transfernode, transfernode->src_.unit_, 0, false, true))));
    } else if (curHead->nodeType_ == dsc2::ScheduleNode::CONDITION) {
      auto conditionnode = static_cast<const dsc2::ConditionNode*>(curHead);
      ConditionOp condition = nullptr;
      Value dim = nullptr;
      auto getDim = [&](PrimaryDimTypes& dim) -> Value {
        for (auto kv : ddlInterface.dim_association_) {
          if (kv.second.dim_ == dim) return kv.first;
        }
        return nullptr;
      };
      if (conditionnode->hasCoreClCond()) {
        llvm::SmallVector<NamedAttribute> corecls;
        for (auto& kv : conditionnode->coreClCond_) {
          std::string key = std::to_string(kv.first);  // core
          llvm::SmallVector<mlir::Attribute> corelets;
          for (auto& cl : kv.second) {
            corelets.push_back(builder.getStringAttr(std::to_string(cl)));
          }
          corecls.push_back(
              builder.getNamedAttr(key, builder.getArrayAttr(corelets)));
        }

        auto cond = CoreCoreletCondOp::create(
            builder, builder.getUnknownLoc(), builder.getIndexType(),
            builder.getDictionaryAttr(corecls));
        ifop_then =
            IfOp::create(builder, builder.getUnknownLoc(), cond->getResult(0));
      } else {
        std::vector<Value> conditions_ors;
        for (auto ors : conditionnode->loopCond_.twoLevelOrOfAnds_) {
          std::vector<Value> conditions_ands;
          for (auto ands : ors) {
            auto cond = ConditionOp::create(
                builder, builder.getUnknownLoc(), builder.getIndexType(),
                getDim(ands.dim_), StringRef("label"),
                StringRef(EnumsConversion::condOpToString.at(ands.condOp_)),
                StringRef(
                    dsc2::LoopCond::condValTypeToString.at(ands.condValType_)));
            cond->setAttr(
                "condValInt_",
                IntegerAttr::get(IntegerType::get(builder.getContext(), 64),
                                 APInt(64, ands.condValInt_)));
            conditions_ands.push_back(cond->getResult(0));
          }
          auto conditions_and = ConditionAndOp::create(
              builder, builder.getUnknownLoc(), builder.getIndexType(),
              ValueRange(ArrayRef<Value>(conditions_ands)));
          conditions_ors.push_back(conditions_and->getResult(0));
        }
        auto condition_ored = ConditionOrOp::create(
            builder, builder.getUnknownLoc(), builder.getIndexType(),
            ValueRange(ArrayRef<Value>(conditions_ors)));
        if (conditionnode->loopCond_.negated_) {
          auto negated_cond =
              ConditionNotOp::create(builder, builder.getUnknownLoc(),
                                     builder.getIndexType(), condition_ored);
          ifop_then = IfOp::create(builder, builder.getUnknownLoc(),
                                   negated_cond->getResult(0));

        } else {
          ifop_then = IfOp::create(builder, builder.getUnknownLoc(),
                                   condition_ored->getResult(0));
        }
      }
      ifop_then->setAttr(
          "name", StringAttr::get(builder.getContext(),
                                  mlir::StringRef(conditionnode->name_)));
      insPts.push(builder.saveInsertionPoint());
    } else if (curHead->nodeType_ == dsc2::ScheduleNode::SYNC) {
      auto syncnode = static_cast<const dsc2::SyncNode*>(curHead);
      mlir::Operation* syncop;
      if (auto* tn = syncnode->implicitSyncRefTransfer_) {
        // implicit sync
        Value allocation = getTensorAndAllocation(
                               allocations, tn->dstVias_.at(0).loc_.storage_,
                               tn->dstLdsAndLoopOffsets_.at(0))
                               .second;
        syncop = ImplicitSyncOp::create(builder, builder.getUnknownLoc(),
                                        allocation);
      } else {
        // explicit sync
        std::vector<Attribute> units;
        for (auto unit : syncnode->units_) {
          units.push_back(
              StringAttr::get(builder.getContext(),
                              EnumsConversion::senComponentsToString.at(unit)));
        }
        std::string label;
        bool seperate_corelets = false;
        for (auto& kv : ddlInterface.sync_definitions_) {
          if (syncnode->name_.find(kv.first) != std::string::npos) {
            label = kv.first;
            seperate_corelets = kv.second.separateCorelets_;
            break;
          }
        }
        if (label.empty()) {
          if (!external_syncs.count(syncnode)) {
            // create entry for L3 syncs
            external_syncs.emplace(syncnode, syncnode->name_);
            for (auto* otherSn : syncnode->otherEndOfTheSignals_)
              external_syncs.emplace(otherSn, syncnode->name_);
          }
          label = external_syncs.at(syncnode);
        }
        syncop = SyncOp::create(
            builder, builder.getUnknownLoc(),
            ArrayAttr::get(builder.getContext(), units),
            BoolAttr::get(builder.getContext(), syncnode->isReceive_),
            StringAttr::get(builder.getContext(), label),
            BoolAttr::get(builder.getContext(), seperate_corelets));
      }
      syncop->setAttr("name",
                      StringAttr::get(builder.getContext(),
                                      mlir::StringRef(syncnode->name_)));
    } else if (curHead->nodeType_ == dsc2::ScheduleNode::BLOCK) {
      insPts.push(builder.saveInsertionPoint());
      // then block
      if (ifop_then) {
        builder.createBlock(&ifop_then.getThenRegion());
        YieldOp::create(builder, builder.getUnknownLoc());
        builder.setInsertionPointToStart(&ifop_then.getThenRegion().front());
        ifop_then = nullptr;
      } else {
        if (ifop_else) {
          // else block
          builder.createBlock(&ifop_else.getElseRegion());
          YieldOp::create(builder, builder.getUnknownLoc());
          builder.setInsertionPointToStart(&ifop_else.getElseRegion().front());
          ifop_else = nullptr;
        }
      }
    } else if (curHead->nodeType_ == dsc2::ScheduleNode::ALLOCATE) {
      auto allocatenode = static_cast<const dsc2::AllocateNode*>(curHead);
      if (!allocatenode->tempStorageForCompute_) {
        Value tensor = nullptr;
        Value allocation = nullptr;
        auto comp = allocatenode->component_;
        if (allocatenode->ldsIdx_ > -1) {
          for (auto kv : ddlInterface.tensor_definition_) {
            if (metadata_.ldsIdxAfterDdc.count(kv.second.ldsIdx_) &&
                metadata_.ldsIdxAfterDdc.at(kv.second.ldsIdx_) ==
                    allocatenode->ldsIdx_) {
              tensor = kv.first;
              break;
            }
          }
          if (!tensor) {
            mlir::Value ref_tensor;
            for (auto kv : ddlInterface.tensor_definition_) {
              if (isa<ddl::TensorOp>(kv.first.getDefiningOp()) &&
                  metadata_.intermLdsIdxToExtLds.count(allocatenode->ldsIdx_) &&
                  metadata_.ldsIdxAfterDdc.at(kv.second.ldsIdx_) ==
                      metadata_.intermLdsIdxToExtLds.at(
                          allocatenode->ldsIdx_)) {
                ref_tensor = kv.first;
                break;
              }
            }
            if (ref_tensor) {
              auto tensor_builder(builder);
              tensor_builder.setInsertionPoint(new_dataflow_op);
              auto ref_tensor_op = ref_tensor.getDefiningOp<ddl::TensorOp>();
              auto interm_tensor = ddl::InternalTensorOp::create(
                  tensor_builder, tensor_builder.getUnknownLoc(),
                  tensor_builder.getIndexType(), ref_tensor,
                  ref_tensor_op.getType());
              tensor = interm_tensor.getResult(0);
              ddlInterface.tensor_definition_[tensor].ldsIdx_ =
                  allocatenode->ldsIdx_;
            }
          }
        } else if (allocatenode->constIdx_ > -1) {
          for (auto kv : ddlInterface.ext_constant_definition_) {
            if (kv.second == allocatenode->constIdx_) {
              tensor = kv.first;
              break;
            }
          }
        } else if (allocatenode->constIdx_ > -1) {
          for (auto kv : ddlInterface.ext_constant_definition_) {
            if (kv.second == allocatenode->constIdx_) {
              tensor = kv.first;
              break;
            }
          }
        } else {
          llvm_unreachable("invalid idx");
        }
        // In preparation for generating the padding_style attribute
        // construct a vector of pattern strings.
        std::vector<Attribute> allocateAccPatternStyles;
        std::vector<Value> allocate_pat_dims;
        std::vector<Attribute> layoutDimOrder;
        for (auto dim : allocatenode->layoutDimOrder_) {
          layoutDimOrder.push_back(builder.getStringAttr(
              EnumsConversion::primaryDimToString.at(dim)));
        }

        for (auto kv0 : allocatenode->padding_) {
          for (auto kv1 : ddlInterface.dim_association_) {
            if (kv0.first == kv1.second.dim_ && kv1.second.isPadded()) {
              allocate_pat_dims.push_back(kv1.first);
              allocateAccPatternStyles.push_back(builder.getStringAttr(
                  allocatenode->padding_.getPaddingAsStr(kv0.first)));
            }
          }
        }
        if (tensor) {
          auto allocop = AllocateOp::create(
              builder, builder.getUnknownLoc(), builder.getIndexType(), tensor,
              ValueRange(ArrayRef<Value>(allocate_pat_dims)), nullptr,
              StringAttr::get(
                  builder.getContext(),
                  mlir::StringRef(
                      EnumsConversion::senComponentsToString.at(comp))),
              allocatenode->numBuffers_,
              builder.getArrayAttr(allocateAccPatternStyles),
              builder.getIntegerAttr(builder.getI64Type(), 1));
          allocop->setAttr("name", builder.getStringAttr(
                                       mlir::StringRef(allocatenode->name_)));

          allocop->setAttr(
              "layout_dim_order",
              ArrayAttr::get(builder.getContext(), layoutDimOrder));
          if (allocatenode->ldsIdx_ >= 0) {
            auto size = dsc.getBufferCapacityForNode(
                allocatenode, allocatenode->ldsIdx_, comp, 0, 0);
            allocop->setAttr(
                "allocation_size",
                builder.getIntegerAttr(builder.getIndexType(), size));
          }

          allocations[allocop] = allocatenode;
        }
      }
    } else if (curHead->nodeType_ == dsc2::ScheduleNode::STICKMASK) {
      auto stickmasknode = static_cast<const dsc2::StickMaskNode*>(curHead);
      auto genericop = GenericOp::create(builder, builder.getUnknownLoc());
      genericop->setAttr(
          "name", StringAttr::get(builder.getContext(),
                                  mlir::StringRef(stickmasknode->name_)));
      genericop->setAttr(
          "nodetype",
          StringAttr::get(
              builder.getContext(),
              mlir::StringRef(dsc2::ScheduleNode::nodeTypeToString.at(
                  stickmasknode->nodeType_))));
    } else {
      llvm_unreachable("Invalid node type");
    }
    // DFS
    if (curHead->nodeType_ == dsc2::ScheduleNode::LOOP ||
        curHead->nodeType_ == dsc2::ScheduleNode::CONDITION ||
        curHead->nodeType_ == dsc2::ScheduleNode::BLOCK) {
      auto next = dynamic_cast<const dsc2::BlockNode*>(curHead)->getNextView(
          SenComponents::ALL);
      if (curHead->nodeType_ == dsc2::ScheduleNode::CONDITION &&
          next.size() > 1) {
        ifop_else = ifop_then;
      }
      std::reverse(next.begin(), next.end());
      for (auto next : next) {
        if (!visited[next]) stack.push(next);
      }
    }
  }

  ddlMlirRoot->dump();
}

// ------------------------------------------------------------------------------------------------
// entry 344/382   level 3   scc 344   12 body lines
// unit: e344_DdlMain
// authority: ddc/ddl/ddl.cpp:96
// original: OwningOpRef<Operation*> DdlMain(const char* input_filename, MLIRContext* context)
// rust home: crates/compiler/deeptools/src/schedule/ddl/mod.rs
// ------------------------------------------------------------------------------------------------
OwningOpRef<Operation*> e344_DdlMain(const char* input_filename,
                                MLIRContext* context)
{
  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(input_filename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return nullptr;
  }
  auto op = DdlMain(std::move(file), context);
  DT_CHECK(op);
  return op;
}

// ------------------------------------------------------------------------------------------------
// entry 345/382   level 3   scc 351   3 body lines
// unit: e345_exportToDdl
// authority: ddc/ddl/ddl_conversion.cpp:38
// original: void DdlConvertInterface::exportToDdl(std::ostream& outputDdl)
// class: DdlConvertInterface
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
void e345_exportToDdl(std::ostream& outputDdl)
{
  ddlConv_->convertDsc2Ddl(outputDdl);
}

// ------------------------------------------------------------------------------------------------
// entry 346/382   level 3   scc 329   26 body lines
// unit: e346_processRegion
// authority: ddc/ddl/ddl_conversion.cpp:2008
// original: void DdlConversion::processRegion(::mlir::Region& myRegion, dsc2::BlockNode* currParent)
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
void e346_processRegion(::mlir::Region& myRegion,
                                  dsc2::BlockNode* currParent)
{
  ddlInterface.region2blocks_[&myRegion] = currParent;
  for (Block& block : myRegion.getBlocks()) {
    for (Operation& op : block.getOperations()) {
      // std::cout << "-------" << std::endl;
      // op.dump();
      // std::cout << "=======" << std::endl;
      auto [insertionPoint, regionIndeces] = processOp(op, currParent);
      auto newInsertionPoint = insertionPoint;
      int numRegions = regionIndeces.size();
      DT_CHECK(numRegions == 0 || insertionPoint != nullptr);
      // std::cout << "Op Regions: " << numRegions << std::endl;
      int c = 0;
      for (const auto& i : regionIndeces) {
        auto& opRegion = op.getRegion(i);
        if (numRegions > 1) {  // e.g. ddl.if
          auto bn = new dsc2::BlockNode();
          insertionPoint->addChildNode(bn);
          bn->name_ = insertionPoint->name_ + "_region" + std::to_string(c++);
          newInsertionPoint = bn;
        }
        processRegion(opRegion, newInsertionPoint);
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 347/382   level 3   scc 340   442 body lines
// unit: e347_matchDdl2Dsc
// authority: ddc/ddl/ddl_conversion.cpp:2110
// original: bool DdlConversion::matchDdl2Dsc()
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
bool e347_matchDdl2Dsc()
{
  auto ddlMlirRoot = ddlParser_.ddl_module_op_.get();

  bool failed = false;
  std::vector<bool> computeOpMapped(dsc.computeOp_.size(), false);
  std::vector<mlir::Value> layout_ops;  // vector to guarantee order
  ddlMlirRoot->walk<WalkOrder::PreOrder>([&](OperationBindOp opBind_op) {
    auto opFuncStr = opBind_op.getOpFuncName().str();
    if (!EnumsConversion::stringToOpFuncs.count(opFuncStr)) {
      DT_ERROR("Unrecognized opFuncName " + opFuncStr);
    }
    // Check supported data formats for op.
    auto opFunc = EnumsConversion::stringToOpFuncs.at(opFuncStr);
    int computeOpIdx = -1;
    auto opBindInputs = opBind_op.getInputs();
    auto opBindInterim = opBind_op.getInterim();
    auto opBindOutputs = opBind_op.getOutputs();
    auto opBindDataFormats = opBind_op.getDataFormats();

    auto types = processTypes(opBindDataFormats, opBind_op);
    // search for a suitable computeOp to map to
    for (int i = 0; i < dsc.computeOp_.size(); i++) {
      if (computeOpMapped[i]) continue;
      const auto& compOp = dsc.computeOp_[i];
      if (compOp.opFuncName != opFunc) continue;
      if (compOp.inputLabeledDs.size() != opBindInputs.size()) continue;
      if (compOp.outputLabeledDs.size() != opBindOutputs.size()) continue;
      if (!types.empty()) {
        if (std::none_of(types.begin(), types.end(),
                         [&](const DdlInterface::TypeDefinition* x) {
                           return x->dataFormat_ ==
                                  compOp.attributes_.dataFormat_;
                         })) {
          continue;
        }
      }

      computeOpIdx = i;
      computeOpMapped[i] = true;
      break;
    }
    if (computeOpIdx < 0) {
      failed |= opBind_op.getRequired();
      return;
    }
    // op match found
    auto& opProp = ddlInterface.operation_definition_[opBind_op.getResult()];
    opProp.computeOpIdx_ = computeOpIdx;
    const auto& compOp = dsc.computeOp_[computeOpIdx];
    if (!compOp.coreClExclude.empty() || !compOp.coreExclude.empty()) {
      // transform coreExclude and coreClExclude into coreCl include to use in
      // ConditionNodes
      for (const auto& coreId : dsc.coreIdsUsed_) {
        if (!compOp.coreExclude.count(coreId)) {
          for (int cl = 0; cl < dsc.numCoreletsUsed_DSC2_; cl++) {
            if (!compOp.coreClExclude.count({coreId, cl}))
              opProp.coreClCond_[coreId].insert(cl);
          }
        }
      }
    }

    // match tensors
    auto matchTensor = [&](mlir::Value opBindTensor, int ldsIdx = -1) {
      auto* opBindTensorDefOp = opBindTensor.getDefiningOp();
      if (isa<AliasOneTensorOfOp>(opBindTensorDefOp)) {
        ddlInterface.getTensorProp(opBindTensor);
      }
      auto [tensorDefIt, tensorDefIsNew] =
          ddlInterface.tensor_definition_.try_emplace(opBindTensor);
      if (!tensorDefIsNew) {  // entry existed already
        if (tensorDefIt->second.ldsIdx_ != ldsIdx) {
          opBindTensorDefOp->emitError(
              "Multiple operation_bind ops use above tensor with different "
              "meaning");
          DT_ERROR("Illegal ddl");
        }
        return;
      }

      if (auto tensor_op = dyn_cast<TensorOp>(opBindTensorDefOp)) {
        if (ldsIdx < 0) {
          tensor_op.emitError(
              "Regular tensor op cannot be used as interim tensor in "
              "operation_bind. Use internal_tensor op instead");
          DT_ERROR("Illegal ddl");
        }
        tensorDefIt->second.ldsIdx_ = ldsIdx;

        auto types = processTypes(tensor_op.getType(), tensor_op);

        auto typeIt =
            std::find_if(types.begin(), types.end(),
                         [&ldsDf = dsc.labeledDs_.at(ldsIdx).dataFormat_](
                             const DdlInterface::TypeDefinition* x) {
                           return x->dataFormat_ == ldsDf;
                         });
        if (typeIt == types.end()) {
          tensor_op.emitError("Problem with type");
          DT_ERROR("Type incompatible with tensor role in operation");
        }

        layout_ops.emplace_back(tensor_op.getSliceLayout());
        layout_ops.emplace_back(tensor_op.getStickLayout());
        layout_ops.emplace_back(tensor_op.getGlobalLayout());
      } else if (auto internal_tensor_op =
                     dyn_cast<InternalTensorOp>(opBindTensorDefOp)) {
        // get reference tensor
        const auto& refTensorProp =
            ddlInterface.getTensorProp(internal_tensor_op.getReferenceTensor());
        auto& newLds = addInternalTensor(
            dsc.labeledDs_.at(refTensorProp.ldsIdx_), computeOpIdx);
        const ddc::DdlInterface::TypeDefinition* type = [&] {
          auto types =
              processTypes(internal_tensor_op.getType(), internal_tensor_op);
          if (types.size() == 1) return types[0];
          for (const auto& type : types) {
            if (type->dataFormat_ == compOp.attributes_.dataFormat_)
              return type;
          }
          internal_tensor_op.emitError("Problem with type");
          DT_ERROR("Type incompatible with tensor role in operation");
        }();
        newLds.wordLength = type->bitSize_ / 8;
        newLds.dataFormat_ = type->dataFormat_;
        tensorDefIt->second.ldsIdx_ = newLds.ldsIdx_;
      } else {
        opBindTensorDefOp->emitError("Should be a tensor op");
        opBind_op.emitError("Use of non-tensor op when expected");
        DT_ERROR("Illegal ddl");
      }
    };
    for (int i = 0; i < opBindOutputs.size(); i++) {
      matchTensor(opBindOutputs[i], compOp.outputLabeledDs[i]->ldsIdx_);
    }
    for (int i = 0; i < opBindInputs.size(); i++) {
      matchTensor(opBindInputs[i], compOp.inputLabeledDs[i]->ldsIdx_);
    }
    for (int i = 0; i < opBindInterim.size(); i++) {
      matchTensor(opBindInterim[i]);
    }
    if (opBindInterim.size() > 0) {
      // fix ldsIdx in external transfer and allocate nodes
      int oldLdsIdx = dsc.labeledDs_.size() - 1 - opBindInterim.size();
      for (auto* node : dsc.scheduleTree_.traverseTreeDFSMutable(
               nullptr,
               {dsc2::ScheduleNode::TRANSFER, dsc2::ScheduleNode::ALLOCATE})) {
        if (node->nodeType_ == dsc2::ScheduleNode::TRANSFER) {
          auto* tn = static_cast<dsc2::TransferNode*>(node);
          if (tn->srcLdsAndLoopOffsets_.myLdsIdx_ == oldLdsIdx)
            tn->srcLdsAndLoopOffsets_.myLdsIdx_ += opBindInterim.size();
          for (auto& dstDi : tn->dstLdsAndLoopOffsets_)
            if (dstDi.myLdsIdx_ == oldLdsIdx)
              dstDi.myLdsIdx_ += opBindInterim.size();
        } else {
          auto* an = static_cast<dsc2::AllocateNode*>(node);
          if (an->ldsIdx_ == oldLdsIdx) an->ldsIdx_ += opBindInterim.size();
        }
      }
      for (auto it =
               metadata_.prefilledExternalTransferToDataConnectToFill_.begin();
           it != metadata_.prefilledExternalTransferToDataConnectToFill_.end();
           it++) {
        if (it->first.first != oldLdsIdx) continue;
        auto toMove =
            metadata_.prefilledExternalTransferToDataConnectToFill_.extract(it);
        toMove.key().first += opBindInterim.size();
        metadata_.prefilledExternalTransferToDataConnectToFill_.insert(
            std::move(toMove));
      }
    }
  });

  if (is_any_of(false, computeOpMapped)) failed = true;
  if (failed) {
    // std::cout << "Template not suitable for DSC " << dsc.name_ << std::endl;
    return false;
  }

  // check if tensors in DSC meet the constraints from ddl.constraint
  /// TODO: implement checks

  std::vector<DdlInterface::DimProp*> ddlDimsToMap;  // vector to ensure order
  // add dimensions
  for (const auto& layout_ssa : layout_ops) {
    auto layout_op = dyn_cast<LayoutOp>(layout_ssa.getDefiningOp());
    if (!layout_op) {
      layout_op.emitError("This op is used as a layout, but it is not");
      DT_ERROR("Illegal ddl");
    }
    for (auto ddlDim : layout_op.getDimensions()) {
      if (!ddlInterface.dim_association_.count(ddlDim)) {
        if (auto paddedDim_op =
                dyn_cast<PaddedDimensionOp>(ddlDim.getDefiningOp())) {
          processPaddedDimensionOp(ddlDim);
        } else if (!isa<DimensionOp>(ddlDim.getDefiningOp())) {
          ddlDim.getDefiningOp()->emitOpError(
              "is supposed to be a dimension but is not");
          DT_ERROR("Illegal ddl file");
        } else {
          processDimensionOp(ddlDim);
          // TO DO: Constraint checking to see if the dimension makes sense for
          // a layout. Acceptable ones are Unpadded and Window.
        }
      }
      auto& dimProp = ddlInterface.dim_association_.at(ddlDim);
      if (dimProp.isUnpadded() ||
          dimProp.getMetaDimKind() == MetaDimKind::WindowDim) {
        if (std::find(ddlDimsToMap.begin(), ddlDimsToMap.end(), &dimProp) ==
            ddlDimsToMap.end()) {
          ddlDimsToMap.push_back(&dimProp);
        }
      } else {
        auto* unpadDimProp =
            &ddlInterface.dim_association_.at(dimProp.nonPaddedDim);
        if (std::find(ddlDimsToMap.begin(), ddlDimsToMap.end(), unpadDimProp) ==
            ddlDimsToMap.end()) {
          ddlDimsToMap.push_back(unpadDimProp);
        }
      }
    }
  }
  // prune candidate mappings
  std::unordered_map<PrimaryDimTypes, bool> dscDimsMapped;
  std::set<int> processedLds;
  std::unordered_set<mlir::Value> ddlPaddedDims;
  for (const auto& tensor : ddlInterface.tensor_definition_) {
    auto tensor_op = dyn_cast<TensorOp>(tensor.first.getDefiningOp());
    if (!tensor_op) continue;  // skip aliases
    // skip if lds already processed
    if (!processedLds.insert(tensor.second.ldsIdx_).second) continue;
    auto& lds = dsc.labeledDs_.at(tensor.second.ldsIdx_);
    std::vector<PrimaryDimTypes> sliceDims, stickDims,
        globalDims = dsc.getLayoutDims(lds.ldsIdx_);
    std::set<PrimaryDimTypes> bcastedDims,
        nonBcastedDims = dsc.getNonBroadcastLdsDimSet(lds.ldsIdx_);
    for (auto& dimToInsert : globalDims) {
      if (nonBcastedDims.count(dimToInsert))
        dscDimsMapped.emplace(dimToInsert, false);
      else
        bcastedDims.insert(dimToInsert);
    }
    int elemInSlice =
        128 / dsc.labeledDs_.at(tensor.second.ldsIdx_).wordLength / 8;
    const auto& stickInfo = dsc.primaryDsInfo_.at(lds.dsType_);
    for (int i = 0, numElem = 1; i < stickInfo.stickDimOrder_.size(); i++) {
      auto dim = stickInfo.stickDimOrder_[i];
      if (numElem < elemInSlice) sliceDims.push_back(dim);
      numElem *= stickInfo.stickSize_[i];
      if (numElem > elemInSlice) stickDims.push_back(dim);
    }
    // cast legality verified earlier
    auto sliceLayout_op =
        cast<LayoutOp>(tensor_op.getSliceLayout().getDefiningOp());
    auto stickLayout_op =
        cast<LayoutOp>(tensor_op.getStickLayout().getDefiningOp());
    auto globalLayout_op =
        cast<LayoutOp>(tensor_op.getGlobalLayout().getDefiningOp());
    auto addDimConstraints = [this, &bcastedDims, &ddlPaddedDims](
                                 LayoutOp layout_op,
                                 const std::vector<PrimaryDimTypes>& layoutDims,
                                 bool isGlobal) {
      auto layout_op_dims = layout_op.getDimensions();
      if (layout_op_dims.empty()) return;  // no constraints
      std::set<PrimaryDimTypes> layoutDimsSet(layoutDims.begin(),
                                              layoutDims.end());
      auto layoutDimsSetNoBcast = set_diff(layoutDimsSet, bcastedDims);
      if (layout_op_dims.size() < layoutDimsSetNoBcast.size()) {
        layout_op.emitError("Not enough dimensions");
        DT_ERROR("Impossible to match for sdsc" + sdsc.name_);
      }
      bool isOrederFixed = layout_op.getIsOrderFixed();
      if (isOrederFixed && layout_op_dims.size() > layoutDims.size()) {
        layout_op.emitError("Fixed layout with too many dimensions");
        DT_ERROR("Impossible to match for sdsc" + sdsc.name_);
      }
      std::vector<mlir::Value> layout_op_dims_noPad;
      for (auto op_dim : layout_op_dims) {
        if (ddlInterface.dim_association_[op_dim].isPadded()) {
          ddlPaddedDims.insert(op_dim);
          op_dim = ddlInterface.dim_association_[op_dim].nonPaddedDim;
          ddlPaddedDims.insert(op_dim);
        }
        layout_op_dims_noPad.push_back(op_dim);
        if (isGlobal)
          ddlInterface.dim_association_[op_dim].numRefsInGlobalLayouts_++;
      }
      // for each ddl dim, first intersect candidates with possible dsc
      // dims then remove these dsc dims from candidates of all other ddl
      // dims
      for (int i = 0; i < layout_op_dims_noPad.size(); i++) {
        mlir::Value layout_op_dim = layout_op_dims_noPad[i];
        auto& dimCandidates =
            ddlInterface.dim_association_[layout_op_dim].dimCandidates_;
        if (isOrederFixed) {
          if (i >= layoutDims.size()) {
            dimCandidates.clear();
          } else {
            dimCandidates = set_intersect(dimCandidates, {layoutDims[i]});
            for (auto& ddlDim : ddlInterface.dim_association_) {
              if (ddlDim.second.isPadded()) continue;
              if (ddlDim.first != layout_op_dim)
                ddlDim.second.dimCandidates_.erase(layoutDims[i]);
            }
          }
        } else {  // flexible order
          if (isGlobal || layout_op_dims.size() <= layoutDims.size()) {
            dimCandidates = set_intersect(dimCandidates, layoutDimsSet);
          }
        }
      }
      if (!isOrederFixed) {
        for (auto& ddlDim : ddlInterface.dim_association_) {
          if (ddlDim.second.isPadded()) continue;
          if (std::find(layout_op_dims_noPad.begin(),
                        layout_op_dims_noPad.end(),
                        ddlDim.first) == layout_op_dims_noPad.end())
            ddlDim.second.dimCandidates_ =
                set_diff(ddlDim.second.dimCandidates_, layoutDimsSetNoBcast);
        }
      }
    };
    addDimConstraints(sliceLayout_op, sliceDims, false);
    addDimConstraints(stickLayout_op, stickDims, false);
    addDimConstraints(globalLayout_op, globalDims, true);
  }

  // remove dsc dims w/ padding from the candidates of ddl dims w/o padding
  // remove window dsc dims from the candidates of ddl dims that are not window
  std::set<PrimaryDimTypes> dscPaddedDims, dscWindowDims;
  for (const auto& [dim, padInfo] :
       dsc.dataStageParam_.at(metadata_.core_dstgid).ss_.paddingSizes_) {
    dscPaddedDims.insert(dim);
    dscWindowDims.insert(padInfo.windowDim_);
  }
  for (auto& [op_dim, op_prop] : ddlInterface.dim_association_) {
    auto& dimCandidates = op_prop.dimCandidates_;
    if (!ddlPaddedDims.count(op_dim))
      dimCandidates = set_diff(dimCandidates, dscPaddedDims);
    if (op_prop.getMetaDimKind() != MetaDimKind::WindowDim)
      dimCandidates = set_diff(dimCandidates, dscWindowDims);
  }

  // map dimensions
  std::stable_sort(
      ddlDimsToMap.begin(), ddlDimsToMap.end(),
      [](ddc::DdlInterface::DimProp* a, ddc::DdlInterface::DimProp* b) {
        if (a->numRefsInGlobalLayouts_ != b->numRefsInGlobalLayouts_)
          return a->numRefsInGlobalLayouts_ < b->numRefsInGlobalLayouts_;
        else
          return a->dimCandidates_.size() < b->dimCandidates_.size();
      });  // sort ddl dims: those appearing in the fewest tensors first; in
           // case of tie, those with less candidates first

  if (ddlDimsToMap.size() < dscDimsMapped.size()) {
    DT_ERROR("Not enough ddl dimensions for this DSC");
  }

  // recursively try to find a mapping of dsc dims with ddl dims
  std::function<bool(int)> tryDimMapping = [&tryDimMapping, &dscDimsMapped,
                                            &ddlDimsToMap](int i) -> bool {
    bool allDone = std::all_of(dscDimsMapped.begin(), dscDimsMapped.end(),
                               [](auto& x) { return x.second; });
    if (i >= ddlDimsToMap.size()) return allDone;  // no more ddl dims
    auto* dimProp = ddlDimsToMap[i];
    if (allDone) {  // all dsc dims already mapped
      dimProp->dropDim_ = true;
      tryDimMapping(i + 1);
      return true;
    }
    dimProp->dropDim_ = false;
    for (auto candidateDim : dimProp->dimCandidates_) {
      if (dscDimsMapped[candidateDim]) continue;
      dimProp->dim_ = candidateDim;
      dscDimsMapped[candidateDim] = true;
      // try to map the next ddl dim
      bool mapped = tryDimMapping(i + 1);
      if (mapped) return true;
      // could not find a mapping for remaining dims, try another
      dscDimsMapped[candidateDim] = false;  // "free" this dimension
    }
    // no candidate work, try dropping this dim and continue
    dimProp->dropDim_ = true;
    bool mapped = tryDimMapping(i + 1);
    return mapped;
  };

  if (!tryDimMapping(0)) {
    for (int i = 0; i < ddlDimsToMap.size(); ++i) {
      llvm::errs() << "\n----";
      llvm::errs() << "\nDimProp[" << i << "]:";
      ddlDimsToMap[i]->dump();
      llvm::errs() << "\n----";
    }
    llvm::errs() << "\n";
    DT_ERROR("Could not find any suitable dimension mapping");
  }

  // fill padded dims too
  const auto& padInfoMap =
      dsc.dataStageParam_.at(metadata_.core_dstgid).ss_.paddingSizes_;
  for (auto& kv : ddlInterface.dim_association_) {
    auto& dimProp = kv.second;
    if (!dimProp.isUnpadded()) {
      dimProp.dropDim_ =
          ddlInterface.dim_association_[dimProp.nonPaddedDim].dropDim_;
      if (dimProp.dropDim_) continue;
      if (dimProp.getMetaDimKind() != MetaDimKind::WindowDim) {
        dimProp.dim_ = ddlInterface.dim_association_[dimProp.nonPaddedDim].dim_;
        // make sure that the padding info is present in dsc for this dim
        dsc.N_.paddingSizes_[dimProp.dim_];
        for (auto& [dsIdx, ds] : dsc.dataStageParam_) {
          ds.ss_.paddingSizes_[dimProp.dim_];
          ds.el_.paddingSizes_[dimProp.dim_];
        }
      } else {
        PrimaryDimTypes unpaddedDim =
            ddlInterface.dim_association_[dimProp.nonPaddedDim].dim_;
        auto piIt = padInfoMap.find(unpaddedDim);
        if (piIt == padInfoMap.end() ||
            piIt->second.windowDim_ == PrimaryDimTypesCount) {
          dimProp.nonPaddedDim.print(llvm::errs());
          DT_ERROR(
              "[Error in dimension-mapping] Unknown primary dimension kind "
              "found for a window dimension.");
        }
        dimProp.dim_ = padInfoMap.at(unpaddedDim).windowDim_;
      }
    }
  }

  // Check presence/absence of meta-dimensions.
  checkMetaDimensions();

  // for (auto it : ddlInterface.dim_association_) {
  //   it.second.dump("After dimension-mapping");
  // }

  // mapping successful
  // std::cout << "Success!" << std::endl;
  return true;
}

// ------------------------------------------------------------------------------------------------
// entry 363/382   level 4   scc 345   9 body lines
// unit: e363_parseDdl
// authority: ddc/ddl/ddl.cpp:121
// original: void DdlModuleOp::parseDdl(const char* input_file)
// class: DdlModuleOp
// rust home: crates/compiler/deeptools/src/schedule/ddl/mod.rs
// ------------------------------------------------------------------------------------------------
void e363_parseDdl(const char* input_file)
{
  mlir::DialectRegistry registry;
  registry.insert<mlir::ddl::DdlDialect>();
  // Create a context just for the current buffer. Disable threading on creation
  // since we'll inject the thread-pool separately.
  mlir_context_ =
      std::make_unique<MLIRContext>(registry, MLIRContext::Threading::DISABLED);
  ddl_module_op_ = DdlMain(input_file, mlir_context_.get());
}

// ------------------------------------------------------------------------------------------------
// entry 364/382   level 4   scc 333   54 body lines
// unit: e364_parseDdl2Dsc
// authority: ddc/ddl/ddl_conversion.cpp:2770
// original: void DdlConversion::parseDdl2Dsc()
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
void e364_parseDdl2Dsc()
{
  auto ddlMlirRoot = ddlParser_.ddl_module_op_.get();
  // ddlMlirRoot->dump();

  // Now lets do the dataflow section..
  DT_CHECK(metadata_.belowLxScheduleInsertBlock);
  dsc2::BlockNode* initialInsertionBlock = dsc.scheduleTree_.getHeadMutable();
  if (metadata_.belowLxScheduleInsertBlock != initialInsertionBlock) {
    // create new block at the beginning of the tree
    initialInsertionBlock = new dsc2::BlockNode();
    dsc.scheduleTree_.getHeadMutable()->addChildNode(initialInsertionBlock,
                                                     true);
    initialInsertionBlock->name_ = "root_level_operations";
  }
  ddlMlirRoot->walk<WalkOrder::PreOrder>([&](DataflowOp dfOp) {
    // dfOp.dump();
    auto& myBody = dfOp.getBody();
    processRegion(myBody, initialInsertionBlock);
  });

  ddlMlirRoot->walk<WalkOrder::PreOrder>([&](TransformationsOp trOp) {
    // trOp.dump();
    auto& myBody = trOp.getBody();
    processTransformations(myBody);
  });

  // connect sync nodes
  for (auto& [label, syncProp] : ddlInterface.sync_definitions_) {
    auto insertOtherEnds =
        [&label = label](std::vector<dsc2::SyncNode*>& baseVect,
                         const std::vector<dsc2::SyncNode*>& otherEndVect) {
          if (otherEndVect.empty()) {
            DT_ERROR("Label \"" + label + "\" does not have matching syncs");
          }
          std::unordered_set<SenComponents> units;
          for (auto& sn : baseVect) {
            for (auto& unit : sn->units_) {
              if (!units.insert(unit).second) {
                DT_ERROR("Multiple sync ops with label \"" + label +
                         "\" have same unit " +
                         EnumsConversion::senComponentsToString.at(unit));
              }
            }
            sn->otherEndOfTheSignals_.insert(sn->otherEndOfTheSignals_.end(),
                                             otherEndVect.begin(),
                                             otherEndVect.end());
          }
        };
    for (auto& [cl, syncsCl] : syncProp.syncsPerCl_) {
      insertOtherEnds(syncsCl.senders_, syncsCl.receivers_);
      insertOtherEnds(syncsCl.receivers_, syncsCl.senders_);
    }
  }
}

// ------------------------------------------------------------------------------------------------
// entry 372/382   level 5   scc 347   59 body lines
// unit: e372_selectAndParseDdlTemplate
// authority: ddc/ddl/ddl_conversion.cpp:42
// original: bool DdlConversion::selectAndParseDdlTemplate()
// class: DdlConversion
// rust home: crates/compiler/deeptools/src/schedule/ddl/conversion.rs
// ------------------------------------------------------------------------------------------------
bool e372_selectAndParseDdlTemplate()
{
  if (dsc.computeOp_.empty()) {
    return false;
  }

  auto dtDir = dtGetEnv<std::string>("DEEPTOOLS_PATH");
  if (!dtDir.has_value()) {
    DT_ERROR("Please specify DEEPTOOLS_PATH (i.e. ~/WORK/deeptools/)");
  }
  std::string ddlTemplateDir = dtDir.value() + "/ddc/ddl_templates/";

  const auto& myFirstOpFunc = dsc.computeOp_.front().opFuncName;
  auto fileit = opFuncToDdlTemplate.find(myFirstOpFunc);
  if (fileit == opFuncToDdlTemplate.end()) {
    if (verbose_ >= 0) {
      std::cout << "[DDC] no DDL available for op "
                << EnumsConversion::opFuncsToString.at(myFirstOpFunc)
                << std::endl;
    }
    return false;
  }
  bool enableLn32 = false;
  std::string matchedDdlFile;
  auto& ddlFiles = fileit->second;
  for (auto& [ddlFile, arch] : ddlFiles) {
    if (arch.has_value() && arch.value() != dscGlobal_.sysDef.coreArch)
      continue;
    if (!enableLn32 && (myFirstOpFunc == OpFuncs::EXX2 ||
                        myFirstOpFunc == OpFuncs::LAYERNORM_SCALE)) {
      auto ln_prec_env = dtGetEnv<std::string>("ENABLE_LN32");
      if (ln_prec_env.has_value() && std::stoi(ln_prec_env.value()) > 0) {
        DT_ERROR(
            "ENABLE_LN32 not currently supported, ddl enhancements needed");
        enableLn32 = true;  // use 32-bit exx2
        continue;
      }
    }
    std::string currDdlFile = ddlTemplateDir + ddlFile;
    ddlInterface.clear();  // clear in case of residuals from previous try
    ddlParser_.parseDdl(currDdlFile.c_str());
    if (matchDdl2Dsc()) {
      matchedDdlFile = ddlFile;
      verifyDdlConstraints();
      parseDdl2Dsc();
      break;
    } else {
      ddlParser_.ddl_module_op_.release();  // clear for next try
    }
  }
  if (matchedDdlFile.empty()) {
    if (verbose_ >= 0) {
      std::cout << "[DDC] DDL found but not suitable for op "
                << EnumsConversion::opFuncsToString.at(myFirstOpFunc)
                << std::endl;
    }
    return false;
  }
  return true;
}

