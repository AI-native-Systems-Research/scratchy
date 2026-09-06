// SPDX-License-Identifier: Apache-2.0
// BRIDGE 2 — DataflowIR -> SentientIR, the D1-D28 span of IBM Spyre deeptools (dcc).
//
// THE 384 FUNCTIONS TO PORT, consolidated. Bodies are VERBATIM from the reference; each is
// preceded by its entry number and original file:line so any port can be audited against the
// authority (pod /project_src/deeptools at a0d29abbed).
//
// Excluded from this file, and listed with reasons in docs/bridge2-porting-order.md: 81 C++ field
// accessors and data members (a struct field in Rust), 19 MLIR pass-machinery functions (there is no
// pass manager, pattern driver or rewriter at run time — #[forward] runs the pipeline at macro
// expansion), and 5 printing/parsing/verification functions (this crate emits and never reads back).
//
// ⛔ NO MLIR, NO LLVM, NO dcc HEADERS. Everything these bodies call is declared in prelude.inc as
// plain C++, so this is one self-contained translation unit.

#include "prelude.inc"


// ==================================================================================================
// LEVEL 0
// ==================================================================================================

// ---- 001/384  matchAndRewrite  —  dcc/src/Conversion/AffineToStandard/AffineToStandard.cpp:41  (8L)
  LogicalResult matchAndRewrite(AffineYieldOp op,
                                PatternRewriter &rewriter) {
    if (isa<scf::ParallelOp>(op->getParentOp())) {
      // Terminator is rewritten as part of the "affine.parallel" lowering
      // pattern.
      return failure();
    }
    rewriter.replaceOpWithNewOp<scf::YieldOp>(op, op.getOperands());

}
// ---- 002/384  setCoalescedBoundValues  —  dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:672  (5L)
void e002_setCoalescedBoundValues(
    SmallVectorImpl<int64_t>& time_bounds, int outer_dim, int inner_dim,
    int64_t coalesced_bound) {
  time_bounds[inner_dim] = coalesced_bound;
  for (int i = outer_dim + 1; i < inner_dim; i++) {

}}
// ---- 003/384  setIndices  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:77  (2L)
  void setIndices(const SmallVectorImpl<Value>& indices) {
    indices_.assign(indices.begin(), indices.end());

}
// ---- 004/384  setMemViewStartAddr  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:81  (2L)
  void setMemViewStartAddr(Value mem_view_start_addr) {
    mem_view_start_addr_ = mem_view_start_addr;

}
// ---- 005/384  setMemoryIndex  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:93  (2L)
  void setMemoryIndex(MemoryOperandIndex memory_index) {
    memory_index_ = memory_index;

}
// ---- 006/384  setLayoutCoeffs  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:96  (2L)
  void setLayoutCoeffs(const SmallVectorImpl<int64_t>& layout_coeffs) {
    layout_coeffs_.assign(layout_coeffs.begin(), layout_coeffs.end());

}
// ---- 007/384  setMemViewLayoutMap  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:99  (2L)
  void setMemViewLayoutMap(AffineMap mem_view_layout_map) {
    mem_view_layout_map_ = mem_view_layout_map;

}
// ---- 008/384  setShuffleMode  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:104  (2L)
  void setShuffleMode(std::string shuffle_mode) {
    shuffle_mode_ = shuffle_mode;

}
// ---- 009/384  setRotationPosition  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:107  (2L)
  void setRotationPosition(int rotation_position) {
    rotation_position_ = rotation_position;

}
// ---- 010/384  setExpectedTotalElements  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:110  (2L)
  void setExpectedTotalElements(int expected_total_elements) {
    expected_total_elements_ = expected_total_elements;

}
// ---- 011/384  setExtents  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:113  (2L)
  void setExtents(const SmallVectorImpl<int>& extents) {
    extents_.assign(extents.begin(), extents.end());

}
// ---- 012/384  setTotalElements  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:116  (2L)
  void setTotalElements(int total_elements) {
    total_elements_ = total_elements;

}
// ---- 013/384  setElementWidth  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:119  (2L)
  void setElementWidth(unsigned element_width) {
    element_width_ = element_width;

}
// ---- 014/384  setTransferSet  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:122  (2L)
  void setTransferSet(IntegerSetAttr transfer_set) {
    transfer_set_ = transfer_set;

}
// ---- 015/384  setTransferOrder  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:125  (2L)
  void setTransferOrder(AffineMap transfer_order) {
    transfer_order_ = transfer_order;

}
// ---- 016/384  AccessDetailsBase  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:218  (0L)
      : AccessDetailsBase(op, comp) {}
// ---- 017/384  setSubscriptsMap  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:228  (2L)
  void setSubscriptsMap(AffineMap subscripts_map) {
    subscripts_map_ = subscripts_map;

}
// ---- 018/384  setIndicesCoeffDict  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:231  (2L)
  void setIndicesCoeffDict(DenseMap<Value, int64_t>& indices_coeff_dict) {
    indices_coeff_dict_ = indices_coeff_dict;

}
// ---- 019/384  AccessDetailsAffine  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:259  (0L)
      : AccessDetailsAffine(op, comp) {}
// ---- 020/384  setTimeAddrMap  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:286  (2L)
  void setTimeAddrMap(AffineMap time_addr_map) {
    time_addr_map_ = time_addr_map;

}
// ---- 021/384  setTimeSymbols  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:289  (2L)
  void setTimeSymbols(const Operation::operand_range time_symbols) {
    time_symbols_.assign(time_symbols.begin(), time_symbols.end());

}
// ---- 022/384  setTimeBounds  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:292  (2L)
  void setTimeBounds(const SmallVectorImpl<int64_t>& time_bounds) {
    time_bounds_.assign(time_bounds.begin(), time_bounds.end());

}
// ---- 023/384  setTimeOffsets  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:295  (2L)
  void setTimeOffsets(const SmallVectorImpl<int64_t>& time_offsets) {
    time_offsets_.assign(time_offsets.begin(), time_offsets.end());

}
// ---- 024/384  setInterleaveGroupIndex  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:299  (2L)
  void setInterleaveGroupIndex(int interleave_group_index) {
    interleave_group_index_ = interleave_group_index;

}
// ---- 025/384  setStrides  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:355  (2L)
  void setStrides(const SmallVectorImpl<Value>& strides) {
    strides_.assign(strides.begin(), strides.end());

}
// ---- 026/384  has  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:397  (2L)
  bool has(MemoryOperandIndex moi) {
    return index_mapping_[(int)moi] != -1;

}
// ---- 027/384  constructLoadAndSendStmt  —  dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:230  (4L)
  LogicalResult constructLoadAndSendStmt(
      OpBuilder* builder, dataflow::ProgramUnitOp unit_op, Operation* op,
      AccessDetailsTy& access_details, Value& mutable_addr,
      Value& immutable_addr, Operation* extract_op) {

}
// ---- 028/384  constructReceiveAndStoreStmt  —  dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:248  (4L)
  LogicalResult constructReceiveAndStoreStmt(
      OpBuilder* builder, dataflow::ProgramUnitOp unit_op, Operation* op,
      Type data_elem_type, AccessDetailsTy& access_details, Value& mutable_addr,
      Value& immutable_addr, Operation* extract_op) {

}
// ---- 029/384  insertCopyAndAddStmtsHelper  —  dcc/src/Conversion/AgenToSentient/AgenToSentient.hpp:502  (17L)
  static Value insertCopyAndAddStmtsHelper(T loop_op, int index, int imm_val) {
    unsigned num_iter_args = loop_op.getNumRegionIterArgs();
    auto iter_arg = loop_op.getRegionIterArgs()[num_iter_args - index - 1];
    OpBuilder builder(loop_op.getBody()->getTerminator());
    auto const_op = mlir::arith::ConstantIndexOp::create(
        builder, loop_op.getBody()->getTerminator()->getLoc(), imm_val);

    // Create add operation reflecting address arithmetic addition
    auto add_op = mlir::arith::AddIOp::create(builder, const_op.getLoc(),
                                              iter_arg.getType(), iter_arg,
                                              const_op.getResult());

    // update yield operand
    auto* yield_op = loop_op.getBody()->getTerminator();
    yield_op->setOperand(yield_op->getNumOperands() - index - 1,
                         add_op.getResult());
    return loop_op.getRegionIterArgs()[num_iter_args - index - 1];

}
// ---- 030/384  getLoopNestLevel  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:43  (6L)
static inline int getLoopNestLevel(Operation* op) {
  if (isa<affine::AffineForOp>(op))
    return dcc::utils::getLoopNestLevel<affine::AffineForOp>(op);
  if (isa<scf::ForOp>(op)) return dcc::utils::getLoopNestLevel<scf::ForOp>(op);
  op->emitError("Expected a loop");
  return -1;

}
// ---- 031/384  checkIndirectMemViewForExtractOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:388  (42L)
bool e031_checkIndirectMemViewForExtractOp(
    dataflow::GetLogicalMemoryViewOp& ind_mem_view) {
  // The mem_view should be a 1D space with start address of 0 operating on a
  // virtual IBR unit.
  auto ind_mem_view_unit = dyn_cast_or_null<dataflow::GetUnitOp>(
      ind_mem_view.getFromUnit().getDefiningOp());
  if (!ind_mem_view_unit) {
    ind_mem_view->emitError(
        "from unit of indirect memory view is not a get_unit operation");
    return false;
  }
  SenComponents ind_mem_view_unit_comp =
      EnumsConversion::stringToSenComponents
          .find(ind_mem_view_unit.getType().str())
          ->second;
  if (ind_mem_view_unit_comp != SenComponents::LXVIRTUALIBR) {
    ind_mem_view->emitError(
        "indirect memory view is not operating on a virtual IBR");
    return false;
  }

  auto ind_mem_view_start_addr = dyn_cast_or_null<arith::ConstantOp>(
      ind_mem_view.getStartAddress().getDefiningOp());
  if (!ind_mem_view_start_addr) {
    ind_mem_view->emitError(
        "indirect memory view does not have a constant start address");
    return false;
  }
  auto ind_mem_view_start_addr_int =
      dyn_cast<IntegerAttr>(ind_mem_view_start_addr.getValue());
  if (!ind_mem_view_start_addr_int ||
      ind_mem_view_start_addr_int.getInt() != 0) {
    ind_mem_view->emitError("indirect memory view start address is not 0");
    return false;
  }

  auto ind_layout_map = ind_mem_view.getLayoutMap();
  if (ind_layout_map.getNumDims() != 1 || !ind_layout_map.isIdentity()) {
    ind_mem_view->emitError("expecting 1D identity map for layout_map");
    return false;
  }


}
// ---- 032/384  findExtractScalarOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:514  (20L)
Operation* e032_findExtractScalarOp(
    Operation* op, dataflow::ProgramUnitOp& unit) {
  DT_CHECK_MSG(op->hasAttr("extract_idx"),
               "input operation does not have extract_idx attribute");
  auto extract_idx = cast<IntegerAttr>(op->getAttr("extract_idx")).getInt();

  Operation* extract_op = nullptr;
  unit.walk<WalkOrder::PreOrder>([&](ExtractOpTy curr_op) {
    DT_CHECK_MSG(curr_op->hasAttr("extract_idx"),
                 "load_and_extract_scalar_op found without extract_idx");
    auto curr_extract_idx =
        cast<IntegerAttr>(curr_op->getAttr("extract_idx")).getInt();
    if (curr_extract_idx == extract_idx) {
      extract_op = curr_op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  DT_CHECK(extract_op);


}
// ---- 033/384  getLoadConsumer  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:1242  (36L)
std::pair<Operation*, Operation*> e033_getLoadConsumer(
    Operation* load_op) {
  Value consumer_root = nullptr;
  if (isa<VectorLoadOp, IndirectVectorLoadOp, SymbolicVectorLoadOp>(load_op))
    consumer_root = load_op->getResult(0);
  else if (auto comp_load_op = dyn_cast<CompositeLoadOp>(load_op))
    consumer_root = comp_load_op.getLoadInductionVar();
  else if (auto comp_ind_load_op = dyn_cast<CompositeIndirectLoadOp>(load_op))
    consumer_root = comp_ind_load_op.getLoadInductionVar();
  DT_CHECK_MSG(consumer_root,
               "could not determine root of the loadOp consumer");

  // Assume single consumer per loadOp.
  if (!consumer_root.hasOneUse()) {
    load_op->emitError("loadOp should have one consumer");
    return std::make_pair(nullptr, nullptr);
  }

  // Assume loadOp is followed by only:
  //   - a sendOp, or
  //   - a selectOp/shuffleOp followed by a sendOp, or
  //   - a storeOp
  auto* user = *consumer_root.getUsers().begin();
  if (auto send_op = dyn_cast<dataflow::SendOp>(user)) {
    Operation* consumer = send_op.getToUnit().getDefiningOp();
    return std::make_pair(send_op, consumer);
  } else if (isa<vectorchain::SelectOp, vectorchain::ShuffleOp,
                 vectorchain::RotateOp>(user) &&
             user->hasOneUse()) {
    if (auto send_op = dyn_cast<dataflow::SendOp>(*user->getUsers().begin())) {
      Operation* consumer = send_op.getToUnit().getDefiningOp();
      return std::make_pair(send_op, consumer);
    }
  }

  load_op->emitError("unsupported loadOp consumer!");

}
// ---- 034/384  setldtype  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:1647  (60L)
LogicalResult e034_setldtype(SenComponents comp,
                                                    Operation* consumer,
                                                    int& total_elements,
                                                    unsigned& element_width,
                                                    std::string& shuffle_mode) {
  // non-default ldtypes currently support for LX only
  if (!is_any_of(comp, LXLU, LXSU)) return LogicalResult::success();

  DT_CHECK_MSG(isa<dataflow::SendOp>(consumer), "consumer must be SendOp");
  auto send_op = cast<dataflow::SendOp>(consumer);

  SenSystemDef sysDef;
  // ShuffleOp feeding a consumer indicates an explicit, non-default ldtype
  // setting.
  if (auto shuffle_op = dyn_cast_or_null<vectorchain::ShuffleOp>(
          send_op.getSendData().getDefiningOp())) {
    // total_elements should reflect full stick. element_width is in bits.
    total_elements = sysDef.bytesPerStick * 8 / element_width;

    int repetitions = shuffle_op.getRepetition();
    if (isSplatFromFirstElem(shuffle_op, 2) && repetitions == 64) {
      // 2B 64way splat ldtype (mode 1)
      shuffle_mode = "splat2b";
    } else if (isRightZeroPadFromFirstElem(shuffle_op, 16) &&
               repetitions == 1) {
      // Return type of shuffle should reflect a stick
      int num_elems =
          dataflow::utils::getNumElements(shuffle_op.getResult().getType());
      unsigned result_bitwidth = dataflow::utils::getElementTypeBitWidth(
          shuffle_op.getResult().getType());
      if (result_bitwidth * num_elems / 8 != sysDef.bytesPerStick) {
        return shuffle_op->emitOpError(
            "LX loads involving explicit padding should "
            "be at stick granularity");
      }
      // 16B 0 pad ldtype (mode 2)
      shuffle_mode = "zpad16b";
    } else if (isSplatFromFirstElem(shuffle_op, 16) && repetitions == 8) {
      // 16B 8way splat ldtype (mode 3)
      shuffle_mode = "splat16b";
    } else {
      return shuffle_op->emitOpError("unsupported ldtype");
    }
  } else if (element_width * total_elements / 8 != sysDef.bytesPerStick) {
    // If sendOp isn't at stick granularity, set ldtype appropriately. If a
    // shuffleOp wasn't used to explicitly state a non-default ldtype, assume
    // user doesn't care how data is arranged (pad versus splat). In this
    // case for 16B data transfers, use zpad16b (mode 2). 2B data transfers
    // only support splat.
    int num_bytes = element_width * total_elements / 8;
    if (num_bytes == 2) {
      // 2B 64way splat ldtype (mode 1)
      shuffle_mode = "splat2b";
    } else if (num_bytes == 16) {
      // 16B zero pad ldtype (mode 2)
      shuffle_mode = "zpad16b";
    } else {
      return consumer->emitOpError("unsupported ldtype");
    }
    // total_elements should reflect full stick. element_width is in bits.

}}
// ---- 035/384  generateSetSendDestinationStmts  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2731  (49L)
LogicalResult e035_generateSetSendDestinationStmts(
    OpBuilder& builder, dataflow::ProgramUnitOp unit_op, Operation* op,
    Operation* consumer_op) {
  auto comp = dcc::getUnitType(
      unit_op.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
  if (comp != LXLU) {
    return LogicalResult::success();
  }

  auto areAnyOf = [](std::vector<SenComponents>& consumer_comps,
                     std::vector<SenComponents> reference) {
    for (auto consumer_comp : consumer_comps) {
      if (is_any_of(consumer_comp, reference)) return true;
    }
    return false;
  };

  std::vector<SenComponents> consumer_comps;
  if (auto consumer_get_unit = dyn_cast<dataflow::GetUnitOp>(consumer_op)) {
    consumer_comps.push_back(dcc::getUnitType(consumer_get_unit));
  } else if (auto query_op = dyn_cast<uniform::QueryMapOp>(consumer_op)) {
    auto map_op =
        query_op.getMap().getDefiningOp<uniform::DefImmutableMappingOp>();
    llvm::SmallVector<Value> units;
    query_op.getAllQueriedValues(units);
    for (auto unit : units) {
      if (auto consumer_get_unit = unit.getDefiningOp<dataflow::GetUnitOp>()) {
        consumer_comps.push_back(dcc::getUnitType(consumer_get_unit));
      } else {
        return op->emitError("vector_loadOp's consumer is not a getUnitOp.");
      }
    }
  }
  // If consumer is bypassing SFP, we need to insert set_send_dst operations
  // to make the routing explicit. Redundant set_send_dst operations will be
  // eliminated by a later transform.
  if (areAnyOf(consumer_comps,
               {SenComponents::PT, SenComponents::SFP, SenComponents::L0SU,
                SenComponents::CROSSPTNLINK})) {
    if (dccExtContext().getArch() >= RCUDD1A_ISA) {
      (void)sentient::SetSendDestinationOp::create(builder, op->getLoc(),
                                                   consumer_op->getResult(0));
    } else if (areAnyOf(consumer_comps, {SenComponents::PT, SenComponents::L0SU,
                                         SenComponents::CROSSPTNLINK})) {
      return op->emitError(
          "load consumer requires generating SETDSTMASK instruction which "
          "is not supported at the current arch level");
    }
  }

}
// ---- 036/384  getStoreOpFromLoadStorePattern  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2872  (8L)
Operation* e036_getStoreOpFromLoadStorePattern(
    Operation* op) {
  Operation* store_op = nullptr;
  DT_CHECK(op->getNumResults() == 1);
  auto result = op->getResult(0);
  if (result.hasOneUse())
    store_op = dyn_cast<VectorStoreTy>(*result.user_begin());


}
// ---- 037/384  findCandidateForLowering  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2884  (12L)
OpTy e037_findCandidateForLowering(ProgramUnitOp& unit) {
  OpTy candidate_op = nullptr;
  unit.walk<WalkOrder::PreOrder>([&](OpTy op) {
    if (op->hasAttr("marked")) {
      candidate_op = op;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  DT_CHECK(candidate_op);

  return candidate_op;

}
// ---- 038/384  addLoadChainToDeleteList  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2975  (9L)
void e038_addLoadChainToDeleteList(
    Operation* op, SmallVectorImpl<Operation*>& to_be_deleted) {
  // TODO: Proper model the handling of select op, for, e.g., sizes.
  for (auto* user : op->getUsers()) {
    if (isa<vectorchain::SelectOp, vectorchain::ShuffleOp,
            vectorchain::RotateOp>(user))
      to_be_deleted.push_back(*user->getUsers().begin());

    to_be_deleted.push_back(user);

}}
// ---- 039/384  isSenComponentL0LU  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:96  (2L)
static inline bool isSenComponentL0LU(SenComponents comp) {
  return EnumsConversion::senCompToGenericComp.at(comp) == L0LU;

}
// ---- 040/384  isSenComponentL0SU  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:100  (2L)
static inline bool isSenComponentL0SU(SenComponents comp) {
  return EnumsConversion::senCompToGenericComp.at(comp) == L0SU;

}
// ---- 041/384  ExtendUnitNameToCorelet  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:104  (11L)
static inline LogicalResult ExtendUnitNameToCorelet(std::string &name,
                                                    dataflow::GetUnitOp unit,
                                                    OpBuilder builder) {
  if (!unit->hasAttr("corelet")) {
    unit->emitError("Unknown corelet information for sentient");
    return LogicalResult::failure();
  }
  if (unit->getAttr("corelet") == builder.getI32IntegerAttr(0)) {
    name += "0";
  } else {
    name += "1";

}}
// ---- 042/384  isSameListOfUnits  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:175  (10L)
static bool isSameListOfUnits(
    std::vector<mlir::Operation *> key_units,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops) {
  std::vector<mlir::dataflow::GetUnitOp> key_unit_ops;
  for (auto key : key_units) {
    if (auto unit = llvm::dyn_cast<dataflow::GetUnitOp>(key)) {
      key_unit_ops.push_back(unit);
    } else {
      return false;
    }

}}
// ---- 043/384  isTargetL3  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1720  (6L)
static bool isTargetL3(uniform::QueryMapOp query_map) {
  auto unit_type =
      dcc::uniform::utils::getUnitTypeFromUniformMappingAsString(query_map);
  if (unit_type.has_value())
    return (unit_type.value().substr(0, 2) == "l3" ? true : false);
  return true;

}
// ---- 044/384  lowerOpaqueOperation  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1984  (27L)
LogicalResult e044_lowerOpaqueOperation(
    dataflow::OpaqueOp opaque_op) {
  for (auto itr : opaque_op.getReadWriteRegisterDictionary()) {
    if (!mlir::dyn_cast<StringAttr>(itr.getValue())) {
      opaque_op->emitError("Registers parameters should have values.");
      return LogicalResult::failure();
    }
  }
  for (auto itr : opaque_op.getReadOnlyRegisterDictionary()) {
    if (!mlir::dyn_cast<StringAttr>(itr.getValue())) {
      opaque_op->emitError("Registers parameters should have values.");
      return LogicalResult::failure();
    }
  }
  for (auto itr : opaque_op.getParameterDictionary()) {
    if (!mlir::dyn_cast<StringAttr>(itr.getValue())) {
      opaque_op->emitError("Parameters should have values.");
      return LogicalResult::failure();
    }
  }
  OpBuilder builder(opaque_op);
  auto dofunc = opaque_op.getFuncName();
  StringAttr dbg_name_attr = getDbgNameAttr(opaque_op);
  sentient::OpaqueOp::create(builder, opaque_op->getLoc(), dbg_name_attr,
                             dofunc, opaque_op.getReadWriteRegisterDictionary(),
                             opaque_op.getReadOnlyRegisterDictionary(),
                             opaque_op.getParameterDictionary());

}
// ---- 045/384  getSentientCmpIPredicate  —  dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:33  (16L)
static CmpIPredicate getSentientCmpIPredicate(
    mlir::arith::CmpIPredicate condop) {
  if (condop == mlir::arith::CmpIPredicate::eq) {
    return CmpIPredicate::eq;
  } else if (condop == mlir::arith::CmpIPredicate::ne) {
    return CmpIPredicate::ne;
  } else if (condop == mlir::arith::CmpIPredicate::slt) {
    return CmpIPredicate::slt;
  } else if (condop == mlir::arith::CmpIPredicate::sle) {
    return CmpIPredicate::sle;
  } else if (condop == mlir::arith::CmpIPredicate::sgt) {
    return CmpIPredicate::sgt;
  } else if (condop == mlir::arith::CmpIPredicate::sge) {
    return CmpIPredicate::sge;
  } else {
    llvm_unreachable("invalid predicate");

}}
// ---- 046/384  ConversionPattern  —  dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:70  (0L)
      : mlir::ConversionPattern(mlir::scf::ForOp::getOperationName(), 1, ctx) {}
// ---- 047/384  runOnOperation  —  dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:251  (30L)
void e047_runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext *context = &getContext();

  // The first thing to define is the conversion target. This will define the
  // final target for this lowering.
  ConversionTarget target(getContext());

  // We define the specific operations, or dialects, that are legal targets for
  // this lowering.
  target.addLegalDialect<
      arith::ArithDialect, mlir::vectorchain::VectorChainDialect,
      mlir::sentient::SentientDialect, mlir::memref::MemRefDialect,
      mlir::func::FuncDialect>();

  target.addIllegalDialect<scf::SCFDialect>();

  // Now that the conversion target has been defined, we just need to provide
  // the set of patterns that will lower the Toy operations.
  mlir::RewritePatternSet patterns(context);
  patterns.insert<ForOpLowering>(context);
  patterns.insert<IfOpLowering>(context);
  patterns.insert<YieldOpLowering>(context);

  // With the target and rewrite patterns defined, we can now attempt the
  // conversion. The conversion will signal failure if any of our `illegal`
  // operations were not converted successfully.
  if (failed(applyPartialConversion(module_op, target, std::move(patterns)))) {
    signalPassFailure();
  }

}
// ---- 048/384  getSentientCmpIPredicate  —  dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:36  (18L)
static CmpIPredicate getSentientCmpIPredicate(
    mlir::arith::CmpIPredicate condop) {
  if (condop == mlir::arith::CmpIPredicate::eq) {
    return CmpIPredicate::eq;
  } else if (condop == mlir::arith::CmpIPredicate::ne) {
    return CmpIPredicate::ne;
  } else if (condop == mlir::arith::CmpIPredicate::slt) {
    return CmpIPredicate::slt;
  } else if (condop == mlir::arith::CmpIPredicate::sle) {
    return CmpIPredicate::sle;
  } else if (condop == mlir::arith::CmpIPredicate::sgt) {
    return CmpIPredicate::sgt;
  } else if (condop == mlir::arith::CmpIPredicate::sge) {
    return CmpIPredicate::sge;
  } else {
    DT_CHECK(0);
  }
  // to silence to warning

}
// ---- 049/384  LowerAddIOpToSentient  —  dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:79  (9L)
void e049_LowerAddIOpToSentient(Operation *op) {
  auto addi_op = llvm::dyn_cast<mlir::arith::AddIOp>(op);
  OpBuilder builder(addi_op);
  auto sentient_add_op = sentient::AddOp::create(
      builder, addi_op->getLoc(), addi_op.getLhs().getType(), addi_op.getLhs(),
      addi_op.getRhs());
  // sentient_add_op->setAttrs(addi_op->getAttrDictionary());
  addi_op->replaceAllUsesWith(sentient_add_op);
  addi_op->erase();

}
// ---- 050/384  LowerSubIOpToSentient  —  dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:90  (10L)
void e050_LowerSubIOpToSentient(Operation *op) {
  auto subi_op = llvm::dyn_cast<mlir::arith::SubIOp>(op);
  OpBuilder builder(subi_op);
  auto sentient_subi_op = sentient::SubOp::create(
      builder, subi_op->getLoc(), subi_op.getLhs().getType(), subi_op.getLhs(),
      subi_op.getRhs());
  // We do not use Arith Attributes
  // sentient_subi_op->setAttrs(subi_op->getAttrDictionary());
  subi_op->replaceAllUsesWith(sentient_subi_op);
  subi_op->erase();

}
// ---- 051/384  LowerMulIOpToSentient  —  dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:102  (9L)
void e051_LowerMulIOpToSentient(Operation *op) {
  auto muli_op = llvm::dyn_cast<mlir::arith::MulIOp>(op);
  OpBuilder builder(muli_op);
  auto sentient_muli_op = sentient::MulOp::create(
      builder, muli_op->getLoc(), muli_op.getLhs().getType(), muli_op.getLhs(),
      muli_op.getRhs());
  sentient_muli_op->setAttrs(muli_op->getAttrDictionary());
  muli_op->replaceAllUsesWith(sentient_muli_op);
  muli_op->erase();

}
// ---- 052/384  If  —  dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:159  (0L)
    // return If(lhs) {If(rhs) true_val; else false_val} else false_val;
// ---- 053/384  LowerConstantIndexToSentient  —  dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:347  (8L)
void e053_LowerConstantIndexToSentient(
    Operation *op) {
  auto const_index_op = llvm::dyn_cast<mlir::arith::ConstantIndexOp>(op);
  OpBuilder builder(const_index_op);
  auto sentient_const_op = sentient::ConstantOp::create(
      builder, const_index_op.getLoc(), const_index_op.getResult().getType(),
      const_index_op.value());
  const_index_op->replaceAllUsesWith(sentient_const_op);

}
// ---- 054/384  LowerConstantIntToSentient  —  dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:358  (15L)
void e054_LowerConstantIntToSentient(Operation *op) {
  auto const_int_op = llvm::dyn_cast<mlir::arith::ConstantIntOp>(op);
  OpBuilder builder(const_int_op);

  int val = -1;
  auto bool_attr = mlir::cast<mlir::BoolAttr>(const_int_op.getValue());
  if (bool_attr) {
    val = bool_attr.getValue();
  } else {
    val = const_int_op.value();
  }
  auto sentient_const_op = sentient::ConstantOp::create(
      builder, const_int_op.getLoc(), const_int_op.getResult().getType(), val);
  const_int_op->replaceAllUsesWith(sentient_const_op);
  const_int_op->erase();

}
// ---- 055/384  getId  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:65  (6L)
std::optional<int> e055_getId(Operation *op) {
  if (data_origins_.count(op) != 0) {
    return data_origins_[op].id_;
  } else {
    return -1;
  }

}
// ---- 056/384  getAbsorbtionFlag  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:73  (6L)
std::optional<bool> e056_getAbsorbtionFlag(Operation *op) {
  if (data_origins_.count(op) != 0) {
    return data_origins_[op].absorbed_;
  } else {
    return std::nullopt;
  }

}
// ---- 057/384  setReuseFlag  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:91  (2L)
void e057_setReuseFlag(Operation *op) {
  data_origins_[op].absorbed_ = true;

}
// ---- 058/384  dominates  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:38  (2L)
  bool dominates(Operation *op1, Operation *op2) {
    return dominance_info_.dominates(op1, op2);

}
// ---- 059/384  isSentientBinaryLogicalOp  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:30  (4L)
bool e059_isSentientBinaryLogicalOp(SentientBinaryOperator sb) {
  return is_any_of(sb, SentientBinaryOperator::and0,
                   SentientBinaryOperator::or0, SentientBinaryOperator::xnor,
                   SentientBinaryOperator::and_not);

}
// ---- 060/384  getInputPrecisionFromOperand  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:36  (6L)
std::string e060_getInputPrecisionFromOperand(VectorOperand& operand) {
  std::string precision = operand.orig_precision_;

  // Currently, we use fp80 type in MLIR to represent fp8.
  if (precision == "fp80") precision = "fp8";
  return precision;

}
// ---- 061/384  getResultPrecisionFromOperands  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:52  (10L)
std::string e061_getResultPrecisionFromOperands(
    SmallVectorImpl<std::optional<VectorOperand>>& operands) {
  for (auto opr_ : operands)
    if (opr_.has_value()) {
      std::string precision = opr_.value().on_the_fly_conv_precision_;

      // Currently, we use fp80 type in MLIR to represent fp8.
      if (precision == "fp80") precision = "fp8";
      return precision;
    }

}
// ---- 062/384  getComputePrecisionOfOp  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:65  (13L)
std::string e062_getComputePrecisionOfOp(Operation* op) {
  DT_CHECK(op);
  if (isa<vectorchain::PackOp, vectorchain::MergeOp>(op)) return "fp16";

  auto elem_type = dcc::utils::getElementType(op);
  std::string precision = dataflow::utils::getPrecisionInString(elem_type);
  // Currently, we use fp80 type in MLIR to represent fp8.
  // bf16, dlpfp16 correspond to fp16 compute precision
  if (precision == "fp80")
    precision = "fp8";
  else if (is_any_of(precision, "bf16", "dlfp16"))
    precision = "fp16";
  return precision;

}
// ---- 063/384  hasConstantBounds  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:80  (10L)
bool e063_hasConstantBounds(
    affine::FlatAffineValueConstraints& mask_set_flat) {
  if (!mask_set_flat.getConstantBound(mlir::presburger::BoundType::LB, 0)
           .has_value())
    return false;

  if (!mask_set_flat.getConstantBound(mlir::presburger::BoundType::UB, 0)
           .has_value())
    return false;


}
// ---- 064/384  size  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:319  (6L)
          size(vec.size()) {
      sum = [](std::vector<int> v) {
        int sum = 0;
        for (auto x : v) sum += x;
        return sum;
      }(vec);

}
// ---- 065/384  fuseCompareAndSelectIntoMinOrMax  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:415  (49L)
void e065_fuseCompareAndSelectIntoMinOrMax(
    ElementWiseSelectionOp selection_op, bool& fusion_to_min,
    bool& fusion_to_max) {
  fusion_to_min = false;
  fusion_to_max = false;
  if (auto compare_op = llvm::dyn_cast<ElementWiseCompareOp>(
          selection_op.getCond().getDefiningOp())) {
    bool same_pair_matching = false, opposite_pair_matching = false;

    // Operation equivalence is used since some times constant operands
    // are duplicated, and direct match may result in spurious mismatches.
    dcc::OperationEquivalence oe;

    auto lhs_def = selection_op.getLhs().getDefiningOp();
    auto rhs_def = selection_op.getRhs().getDefiningOp();
    auto op1_def = compare_op.getOp1().getDefiningOp();
    auto op2_def = compare_op.getOp2().getDefiningOp();
    if (!lhs_def || !rhs_def || !op1_def || !op2_def) return;

    if (oe.operationsAreEquivalent(*lhs_def, *op1_def, nullptr) &&
        oe.operationsAreEquivalent(*rhs_def, *op2_def, nullptr)) {
      same_pair_matching = true;
    } else if (oe.operationsAreEquivalent(*lhs_def, *op2_def, nullptr) &&
               oe.operationsAreEquivalent(*rhs_def, *op1_def, nullptr)) {
      opposite_pair_matching = true;
    }

    if (!same_pair_matching && !opposite_pair_matching) return;

    if (is_any_of(compare_op.getCompareOp(),
                  VectorChainElementWiseCompareOperator::compare_gt,
                  VectorChainElementWiseCompareOperator::compare_ge)) {
      if (same_pair_matching) {
        fusion_to_max = true;
      } else if (opposite_pair_matching) {
        fusion_to_min = true;
      }
    } else if (is_any_of(compare_op.getCompareOp(),
                         VectorChainElementWiseCompareOperator::compare_lt,
                         VectorChainElementWiseCompareOperator::compare_le)) {
      if (same_pair_matching) {
        fusion_to_min = true;
      } else if (opposite_pair_matching) {
        fusion_to_max = true;
      }
    }

    if (!fusion_to_min && !fusion_to_max) {
      DT_ERROR("Unable to lower compare and select into max/min operation");

}}}
// ---- 066/384  resetSentientFMAsIfExists  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:468  (13L)
void e066_resetSentientFMAsIfExists(dataflow::ProgramUnitOp unit) {
  Builder builder(unit);
  auto attr = builder.getSI32IntegerAttr(-1);
  unit.walk<WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (auto mac_op = llvm::dyn_cast<sentient::MacOp>(op)) {
      mac_op.setOpADataIDAttr(attr);
      mac_op.setOpBDataIDAttr(attr);
      mac_op.setOpCDataIDAttr(attr);
    } else if (auto bin_op = llvm::dyn_cast<sentient::BinaryOp>(op)) {
      bin_op.setOpADataIDAttr(attr);
      bin_op.setOpBDataIDAttr(attr);
    }
  });

}
// ---- 067/384  redefineConstantVectors  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:571  (37L)
void e067_redefineConstantVectors(Operation* op) {
  // This structure used to hold the information of the operation
  // `op_to_update_` whose operand at location_ is updated with
  // `update_with_` value.
  struct updateInfo {
    Operation* op_to_update_;
    Value update_with_;
    unsigned int location_;
  };

  std::vector<updateInfo> info;
  std::vector<Operation*> to_be_deleted;
  op->walk<WalkOrder::PreOrder>([&](arith::ConstantOp const_op) {
    auto is_vec = mlir::isa<VectorType, dataflow::CustomVectorType>(
        const_op.getResult().getType());
    if (is_vec && !const_op->use_empty()) {
      // MLIR doesn't allow updating uses while operating on getUses
      // If try instead, we are getting incorrect set of uses.
      for (auto& use : const_op->getUses()) {
        auto* owner = use.getOwner();
        OpBuilder builder(owner);
        auto* new_op = builder.clone(*const_op);
        info.push_back({owner, new_op->getResult(0), use.getOperandNumber()});
      }

      to_be_deleted.push_back(const_op);
    }
  });

  for (auto& group : info) {
    group.op_to_update_->setOperand(group.location_, group.update_with_);
  }

  for (auto* tmp_op : to_be_deleted) {
    tmp_op->dropAllUses();
    tmp_op->erase();
  }

}
// ---- 068/384  getVectorBinaryToSentientBinary  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:32  (17L)
static SentientBinaryOperator getVectorBinaryToSentientBinary(
    VectorChainBinaryOperator vb) {
  // clang-format off
  if (vb == mlir::vectorchain::VectorChainBinaryOperator::and0)    return SentientBinaryOperator::and0;
  if (vb == mlir::vectorchain::VectorChainBinaryOperator::or0)     return SentientBinaryOperator::or0;
  if (vb == mlir::vectorchain::VectorChainBinaryOperator::xnor)    return SentientBinaryOperator::xnor;
  if (vb == mlir::vectorchain::VectorChainBinaryOperator::and_not) return SentientBinaryOperator::and_not;
  if (vb == mlir::vectorchain::VectorChainBinaryOperator::min)     return SentientBinaryOperator::min;
  if (vb == mlir::vectorchain::VectorChainBinaryOperator::max)     return SentientBinaryOperator::max;
  if (vb == mlir::vectorchain::VectorChainBinaryOperator::abs_min) return SentientBinaryOperator::abs_min;
  if (vb == mlir::vectorchain::VectorChainBinaryOperator::abs_max) return SentientBinaryOperator::abs_max;
  if (vb == mlir::vectorchain::VectorChainBinaryOperator::add)     return SentientBinaryOperator::add;
  if (vb == mlir::vectorchain::VectorChainBinaryOperator::mul)     return SentientBinaryOperator::mul;
  if (vb == mlir::vectorchain::VectorChainBinaryOperator::mul_div2)     return SentientBinaryOperator::mul_div2;
  if (vb == mlir::vectorchain::VectorChainBinaryOperator::sub)     return SentientBinaryOperator::sub;

  // clang-format on

}
// ---- 069/384  getVectorElementWiseCompareOperatorToSentientBinaryOperator  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:55  (9L)
getVectorElementWiseCompareOperatorToSentientBinaryOperator(
    VectorChainElementWiseCompareOperator vb) {
  // clang-format off
  if (vb == VectorChainElementWiseCompareOperator::compare_eq) return SentientBinaryOperator::fcmp_eq;
  if (vb == VectorChainElementWiseCompareOperator::compare_neq) return SentientBinaryOperator::fcmp_neq;
  if (vb == VectorChainElementWiseCompareOperator::compare_lt) return SentientBinaryOperator::fcmp_lt;
  if (vb == VectorChainElementWiseCompareOperator::compare_le) return SentientBinaryOperator::fcmp_le;

  // clang-format on

}
// ---- 070/384  getVectorTernaryToSentientTernary  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:247  (7L)
static SentientTernaryOperator getVectorTernaryToSentientTernary(
    Operation* op) {
  if (isa<vectorchain::ElementWiseSelectionOp>(op)) {
    return SentientTernaryOperator::select;
  }

  // clang-format on

}
// ---- 071/384  getOperandFromReceiveOp  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:34  (54L)
std::optional<VectorOperand> e071_getOperandFromReceiveOp(
    const dcc::DccExtContext &dcc_ext_ctx, dataflow::ReceiveOp &receive_op,
    const SenComponents &comp) {
  std::string link;
  std::string unit_str;
  std::optional<std::string> unit_str_optional =
      dcc::uniform::utils::findUnitType(receive_op.getFromUnit());
  if (unit_str_optional.has_value()) {
    unit_str = unit_str_optional.value();
  } else {
    receive_op.emitOpError("Unit type is inconsistent in ReceiveOp.");
    return std::nullopt;
  }
  auto record = EnumsConversion::stringToSenComponents.find(unit_str);

  if (record != EnumsConversion::stringToSenComponents.end()) {
    auto generic = EnumsConversion::senCompToGenericComp.at(record->second);
    if (comp == PT) {
      if (record->second == L0LU) {
        link = "west";
      } else if (generic == CROSSPTNLINK) {
        link = "crossptnlink";
      } else if (generic == PT || record->second == SFP ||
                 record->second == LXLU) {
        link = "north";
      } else {
        receive_op->emitError(
            "PT cannot expect data "
            "other than L0-LU, N-link, CROSS-PT-N-LINK");
        return std::nullopt;
      }
    } else {  // PE/SFP
      if (record->second == LXLU || record->second == LXSU) {
        link = "lx";
      } else if (record->second == PE) {
        link = "pe";
      } else if (record->second == SFP) {
        link = "sfp";
        if (dcc_ext_ctx.getArch() >= RCUDD1A_ISA && comp == SFP) {
          if (EnumsConversion::stringToSenComponents.at(unit_str) == SFP) {
            link += "ring";
          }
        }
      } else if (generic == PT) {
        link = "pt";
      } else {
        receive_op->emitError("Unsupported receive unit for PE/SFP");
        return std::nullopt;
      }
    }

    return VectorOperand(Link, link, receive_op.getOperation());
  } else {
    receive_op->emitError("Unknown receiver");

}}
// ---- 072/384  getOperandFromSendOp  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:95  (50L)
std::optional<VectorOperand> e072_getOperandFromSendOp(
    const dcc::DccExtContext &dcc_ext_ctx, dataflow::SendOp &send_op,
    const SenComponents &comp) {
  std::string link;
  std::string unit_str;
  std::optional<std::string> unit_str_optional =
      dcc::uniform::utils::findUnitType(send_op.getToUnit());
  if (unit_str_optional.has_value()) {
    unit_str = unit_str_optional.value();
  } else {
    send_op.emitOpError("Unit type is inconsistent in SendOp.");
    return std::nullopt;
  }
  auto record = EnumsConversion::stringToSenComponents.find(unit_str);

  if (record != EnumsConversion::stringToSenComponents.end()) {
    auto generic = EnumsConversion::senCompToGenericComp.at(record->second);
    if (comp == PT) {
      if (generic == PT || record->second == PE) {
        link = "south";
      }
    } else {  // PE/SFP
      if (generic == PT) {
        link = "pt";
      } else if (record->second == PE) {
        link = "pe";
      } else if (record->second == SFP) {
        link = "sfp";
        if (comp == SFP) {
          if (dcc_ext_ctx.getArch() >= RCUDD1A_ISA)
            link += "ring";
          else
            send_op.emitWarning(
                "SFP to SFP communication requires target arch DD1 and above");
        }
      } else if (record->second == L0LU || record->second == L0SU) {
        link = "l0";
      } else if (record->second == LXLU || record->second == LXSU) {
        link = "lx";
      }
    }

    if (link.empty()) {
      send_op->emitError("Unsupported destination for PE/SFP FMA: " + unit_str);
      return std::nullopt;
    }

    return VectorOperand(Link, link, send_op.getOperation());
  } else {
    send_op->emitError("Unknown destination");

}}
// ---- 073/384  constValToField  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:250  (12L)
static std::string constValToField(double const_val) {
  if (const_val == 0) {
    return "zero";
  } else if (const_val == 1) {
    return "one";
  } else if (const_val == 2) {
    return "two";
  } else if (const_val == 3) {
    return "three";
  }
  DT_ERROR("Only 0, 1, 2, or 3 are supported values");
  return "";

}
// ---- 074/384  sameBlock  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:652  (11L)
LogicalResult e074_sameBlock(Operation *this_op,
                                       std::optional<VectorOperand> &operand) {
  if (operand.has_value()) {
    if (!isa<mlir::arith::ConstantOp>(operand.value().op_) &&
        operand.value().op_->getBlock() != this_op->getBlock()) {
      return LogicalResult::failure();
    }

    return LogicalResult::success();
  } else {
    return LogicalResult::failure();

}}
// ---- 075/384  eraseOp  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:806  (16L)
void e075_eraseOp(Operation *op) {
  bool all_uses_deleted = true;
  std::vector<mlir::Operation *> to_be_erased;
  for (auto &use : op->getUses()) {
    Operation *user = use.getOwner();
    if (user && user->getUses().empty()) {
      to_be_erased.push_back(user);
    } else {
      all_uses_deleted = false;
    }
  }
  for (auto e : to_be_erased) {
    e->erase();
  }

  if (all_uses_deleted) op->erase();

}
// ---- 076/384  fuseComputeOps  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1243  (26L)
void e076_fuseComputeOps(
    MLIRContext *context, dataflow::ProgramUnitOp unit_op,
    OperandReuse &reuse_info) {
  RewritePatternSet compute_ops_patterns(context);
  compute_ops_patterns
      .insert<BinaryOpLowering, ElementWiseCompareOpLowering,
              ElementWiseSelectionOpLowering, MultiplyAndAccumulateOpLowering,
              MultiplyOpLowering, ShuffleOpLowering, PackOpLowering,
              ScanWithGapOpLowering, FastExpOpLowering, ExpEstimateOpLowering,
              FloorOpLowering, RecEstimateOpLowering, LnEstimateOpLowering,
              RsqrtEstimateOpLowering, SigmoidEstimateOpLowering,
              TanhEstimateOpLowering>(context, dccExtContext(), unit_op,
                                      reuse_info);

  ConversionTarget target(*context);
  target.addLegalDialect<
      arith::ArithDialect, mlir::sentient::SentientDialect,
      mlir::dataflow::DataflowDialect, mlir::memref::MemRefDialect,
      mlir::uniform::UniformDialect, mlir::symbol::SymbolDialect>();
  target.addIllegalOp<BinaryOp, MultiplyAndAccumulateOp, MultiplyOp, PackOp,
                      ScanWithGapOp, FastExpOp, ExpEstimateOp, FloorOp,
                      RecEstimateOp, LnEstimateOp, RsqrtEstimateOp,
                      SigmoidEstimateOp, TanhEstimateOp>();
  if (failed(applyPartialConversion(unit_op, target,
                                    std::move(compute_ops_patterns)))) {
    signalPassFailure();

}}
// ---- 077/384  walk  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:132  (2L)
void e077_walk(LoopMaskNode::ActionFuncTy action) {
  LoopMaskNode::walk<OperationNode::WalkOrder::kBFS>(getRoot(), action);

}
// ---- 078/384  findNodeFromOp  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:167  (4L)
LoopMaskNode *LoopMaskTree::findNodeFromOp(Operation *op) {
  DT_CHECK_MSG(op, "valid op expected");
  if (op_to_node_.find(op) == op_to_node_.end()) return nullptr;
  return op_to_node_[op];

}
// ---- 079/384  OperationNode  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:32  (0L)
  LoopMaskNode(Operation *op) : OperationNode(op) {};
// ---- 080/384  LoopMaskNode  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:34  (0L)
  virtual ~LoopMaskNode() {}
// ---- 081/384  getParentNode  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:36  (2L)
  LoopMaskNode *getParentNode() {
    return static_cast<LoopMaskNode *>(OperationNode::getParentNode());

}
// ---- 082/384  getFirstChild  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:39  (2L)
  LoopMaskNode *getFirstChild() {
    return static_cast<LoopMaskNode *>(OperationNode::getFirstChild());

}
// ---- 083/384  getNextSibling  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:42  (2L)
  LoopMaskNode *getNextSibling() {
    return static_cast<LoopMaskNode *>(OperationNode::getNextSibling());

}
// ---- 084/384  getPrevSibling  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:45  (2L)
  LoopMaskNode *getPrevSibling() {
    return static_cast<LoopMaskNode *>(OperationNode::getPrevSibling());

}
// ---- 085/384  getLastChild  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:48  (2L)
  LoopMaskNode *getLastChild() {
    return static_cast<LoopMaskNode *>(OperationNode::getLastChild());

}
// ---- 086/384  MaskNode  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:74  (0L)
  ~MaskNode() {}
// ---- 087/384  LMTLoopNode  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:94  (0L)
  ~LMTLoopNode() {}
// ---- 088/384  getRoot  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:109  (2L)
  const LoopMaskNode *getRoot() {
    return static_cast<const LoopMaskNode *>(OperationTreeBase::getRoot());

}
// ---- 089/384  getMaskValueForPT  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Helper.cpp:17  (196L)
std::optional<Value> e089_getMaskValueForPT(
    Operation* op, Operation* operand, const SenComponents comp,
    const SenSystemDef& sys_def) {
  DT_CHECK_MSG(comp == PT, "function only checks mask value for PT unit");

  // For PT, masking can only be specified with a create_affine_mask
  // operation.
  auto cam_op = dyn_cast<vectorchain::CreateAffineMaskOp>(operand);
  if (!cam_op) {
    operand->emitOpError("PT mask should come from a CreateAffineMaskOp");
    return std::nullopt;
  }

  auto mask_set_attr = operand->getAttr("mask_set");
  auto mask_set = cast<IntegerSetAttr>(mask_set_attr).getValue();
  if (mask_set.getNumDims() != 1) {
    operand->emitOpError("Mask affine set has to have one dimension.");
    return std::nullopt;
  }

  // The result of the operation being masked and the mask operation will
  // reflect how many lanes in the PT row. These should match.
  DT_CHECK_MSG(op->getNumResults() == 1, "expecting one result");
  auto op_result_type = dyn_cast<VectorType>(op->getResult(0).getType());
  DT_CHECK_MSG(op_result_type, "expecting op to be VectorType");
  unsigned op_num_elems = op_result_type.getNumElements();

  DT_CHECK_MSG(operand->getNumResults() == 1, "expecting one result");
  auto mask_result_type = dyn_cast<VectorType>(operand->getResult(0).getType());
  DT_CHECK_MSG(mask_result_type, "expecting operand to be VectorType");
  if (mask_result_type.getNumElements() != op_num_elems) {
    operand->emitOpError(
        "number of PT mask elements should match number of op elements");
    return std::nullopt;
  }

  // The number of lanes should be equally divisible into slices.
  DT_CHECK_MSG(op_num_elems % sys_def.numSlicesPerStick == 0,
               "expecting equal elements per slice from op result");
  int num_lanes_in_slice = op_num_elems / sys_def.numSlicesPerStick;

  // Masks have different constraints depending on if they are constant or
  // not. A mask is considered a constant variant if it doesn't have a
  // mask_parameter operand. If it does, it is a dynamic mask where the operand
  // is used as the symbol for the affine set describing the mask.
  auto mask_parameter = cam_op.getMaskParameter();

  bool static_mask = !mask_parameter;

  // If it has a mask parameter, see if it's a constant parameter and the mask
  // set can be turned into a static mask.
  if (!static_mask) {
    if (auto const_op =
            llvm::dyn_cast<arith::ConstantOp>(mask_parameter.getDefiningOp())) {
      static_mask = true;
      auto affine_const =
          getAffineConstantExpr(cast<IntegerAttr>(const_op.getValue()).getInt(),
                                const_op.getContext());
      mask_set = mask_set.replaceDimsAndSymbols({}, {affine_const},
                                                mask_set.getNumDims(), 0);
    }
  }

  if (static_mask) {
    // CONSTANT MASK
    if (mask_set.getNumSymbols() != 0) {
      operand->emitOpError("Mask affine set should not have any symbols");
      return std::nullopt;
    }

    // For constant masks, the affine set fully describes which lanes should be
    // masked.
    // The upper bound should be:
    //   <number of lanes> - 1.
    affine::FlatAffineValueConstraints mask_set_flat(mask_set);
    auto upper_bound = int64_t(
        mask_set_flat.getConstantBound(mlir::presburger::BoundType::UB, 0)
            .value());
    if (upper_bound != op_num_elems - 1) {
      operand->emitOpError(
          "Constant mask upper bound does not match expected value");
      return std::nullopt;
    }

    // The lower bound should be the first lane to be masked.
    auto lower_bound = int64_t(
        mask_set_flat.getConstantBound(mlir::presburger::BoundType::LB, 0)
            .value());
    auto num_masked_elems = op_num_elems - lower_bound;
    auto masked_columns = num_masked_elems / sys_def.numSlicesPerStick;
    if (num_masked_elems < 0 ||
        num_masked_elems % sys_def.numSlicesPerStick != 0) {
      operand->emitOpError(
          "Constant mask bounds do not describe a valid mask set");
      return std::nullopt;
    }
    // Create a constant op for the mask value. This is a dummy Value.
    OpBuilder builder(op);
    return sentient::ConstantOp::create(builder, op->getLoc(),
                                        builder.getIndexType(), masked_columns);
  } else {
    // DYNAMIC MASK
    if (mask_set.getNumSymbols() != 1) {
      operand->emitOpError("Mask affine set should have 1 symbol");
      return std::nullopt;
    }

    auto mask_def_op = mask_parameter.getDefiningOp();
    DT_CHECK_MSG(mask_def_op,
                 "expecting mask_val to have a defining operation");
    auto non_const_mask = dyn_cast<arith::SubIOp>(mask_def_op);
    if (!non_const_mask) {
      operand->emitOpError(
          "Non-constant PT mask parameter is an unexpected operation");
      return std::nullopt;
    }

    if (mask_set.getNumSymbols() != 1) {
      operand->emitOpError("dynamic PT mask set should contain one symbol");
      return std::nullopt;
    }

    // Must be a sub of the <loop upper bound> - <loop iterator>.
    // Note: At this point in the pipeline all loops have been lowered to
    //       sentient.for operations.
    // Note: Sentient loops count down, not up, so to retain this difference
    //       in behaviour during lowering, loop iterators are converted to
    //       subs before use in loops. At vectorchain lowering, these subs
    //       exist.
    auto lhs = non_const_mask.getLhs();
    auto rhs = non_const_mask.getRhs();
    auto rhs_block_arg = dyn_cast<BlockArgument>(rhs);
    if (!rhs_block_arg) {
      operand->emitOpError(
          "non-constant PT mask value should be loop iterator");
      return std::nullopt;
    }
    auto rhs_parent =
        dyn_cast<sentient::ForOp>(rhs_block_arg.getOwner()->getParentOp());
    DT_CHECK_MSG(
        rhs_parent,
        "Parent operation of RHS of operation making up mask should be a "
        "sentient::ForOp");
    if (lhs != rhs_parent.getBound() || rhs != rhs_parent.getInductionVar()) {
      operand->emitOpError(
          "non-constant PT mask value should be loop iterator (LHS == loop "
          "upper bound && RHS == loop induction var)");
      return std::nullopt;
    }

    // Currently, only the following mask set is supported for non-constant
    // masks:
    // #set = affine_set<(d0)[s0] :
    //            (d0 + s0 * slice_elems - op_num_elems >= 0,
    //             -d0 + (op_result_elems - 1) >= 0)>
    // where op_num_elems is the number of elements in op result and
    // slice_elems is the number of result elements in a slice.
    // op_num_elems represents the number of lanes in the PT row.
    // slice_elems is how many lanes are used for a slice.
    // Check the affine sets conform.
    auto checkAffineSetConstraints =
        [](IntegerSet& mask_set, AffineExpr& constraintA,
           AffineExpr& constraintB, Operation* mask_operand) -> LogicalResult {
      auto num_constraints = mask_set.getNumConstraints();
      if (num_constraints != 2)
        return mask_operand->emitOpError(
            "PT mask set has incorrect number of constraints");

      bool found_constraintA = false, found_constraintB = false;
      for (int i = 0; i < num_constraints; ++i) {
        auto constraint = mask_set.getConstraint(i);
        if (constraint == constraintA)
          found_constraintA = true;
        else if (constraint == constraintB)
          found_constraintB = true;
        else
          return mask_operand->emitOpError(
              "unexpected constraint detected in PT mask_set");
      }
      if (!found_constraintA || !found_constraintB)
        return mask_operand->emitOpError(
            "expected constraints for PT mask_set not found");
      return LogicalResult::success();
    };

    auto context = op->getContext();
    AffineExpr d0, s0;
    bindDims(context, d0);
    bindSymbols(context, s0);
    AffineExpr symbol_constraint = d0 + s0 * num_lanes_in_slice - op_num_elems;
    AffineExpr non_symbol_constraint = -d0 + (op_num_elems - 1);
    if (checkAffineSetConstraints(mask_set, symbol_constraint,
                                  non_symbol_constraint, operand)
            .failed())
      return std::nullopt;
    // Return the loop iterator used for the mask.

}}
// ---- 090/384  getXrfValue  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:248  (11L)
Value e090_getXrfValue(Operation *xrf_ptr, int idx = 0) {
  Value xrf_ptr_val;
  if (isa<sentient::YieldOp>(xrf_ptr)) {
    xrf_ptr_val = xrf_ptr->getParentOp()->getResult(0 + idx);
  } else if (isa<sentient::ForOp>(xrf_ptr)) {
    xrf_ptr_val =
        cast<sentient::ForOp>(xrf_ptr).getBody()->getArgument(1 + idx);
  } else {
    xrf_ptr_val = xrf_ptr->getResult(0);
  }
  return xrf_ptr_val;

}
// ---- 091/384  getForOpBound  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:262  (31L)
int64_t e091_getForOpBound(Operation *op) {
  auto for_op = dyn_cast<sentient::ForOp>(op);
  if (for_op) {
    auto bound_op = for_op.getBound().getDefiningOp();
    if (isa<mlir::arith::ConstantIndexOp>(bound_op)) {
      auto const_op = cast<mlir::arith::ConstantIndexOp>(bound_op);
      return const_op.value();
    } else if (isa<mlir::arith::DivSIOp>(bound_op)) {
      auto div_op = cast<mlir::arith::DivSIOp>(bound_op);
      auto sub_op = div_op.getLhs().getDefiningOp<mlir::arith::SubIOp>();
      if (auto const_op =
              sub_op.getLhs().getDefiningOp<mlir::arith::ConstantIndexOp>()) {
        return const_op.value();
      } else if (auto symbol_op =
                     sub_op.getLhs()
                         .getDefiningOp<mlir::symbol::CreateSymbolOp>()) {
        // We know that XRF read/write accesses don't involve loops with
        // symbolic bounds. So, the caller of this function which is computing
        // the movement, it can be safe to treat as zero.
        // The movement within a loop = its loop iter_arg init + bound * stride
        // Since symbolic loops are not involved in array subscripts, the stride
        // is zero, and hence movement is simply same as loop iter_arg init.
        // So, bound doesn't play role and it safe to consider as zero.
        return 0;
      }
    } else {
      op->emitError("unsupported op for getForOpBound()");
      DT_ERROR("Could not get valid ForOp bound");
    }
  }
  return 0;

}
// ---- 092/384  setSentientMacXrfRegIncrAttr  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:665  (11L)
void e092_setSentientMacXrfRegIncrAttr(
    sentient::MacOp *mac_op, OpBuilder *builder, std::string precision,
    const dcc::DccExtContext &dcc_ext_ctx) {
  int xrf_read_incr_value = 0;
  if (mac_op->isXrfRdRelated()) {
    xrf_read_incr_value = dcc_ext_ctx.getXrfRdPtrIncrValAfterMAC(precision);
  }

  int xrf_write_incr_value = mac_op->isXrfWtRelated() ? 1 : 0;
  auto int_attr_rd = builder->getI32IntegerAttr(xrf_read_incr_value);
  auto int_attr_wt = builder->getI32IntegerAttr(xrf_write_incr_value);

}
// ---- 093/384  replaceAndEraseDummyMacOps  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:681  (7L)
void e093_replaceAndEraseDummyMacOps(XrfPtrMap &mac_op_to_xrfptr_map) {
  for (auto item : mac_op_to_xrfptr_map) {
    item.second.at(1).at(0).replaceAllUsesWith(item.first->getResult(0));
    item.second.at(1).at(1).replaceAllUsesWith(item.first->getResult(1));
    item.second.at(1).at(0).getDefiningOp()->erase();
    item.second.at(1).at(1).getDefiningOp()->erase();
  }

}
// ---- 094/384  computeUnitPrecision  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:31  (9L)
std::string e094_computeUnitPrecision(
    dataflow::ProgramUnitOp &unit, const SenComponents &comp) {
  DT_CHECK(comp == PT);
  DT_CHECK_MSG(unit.getPrecision().has_value(),
               "Precision attribute for PT is expected");
  std::string precision = unit.getPrecision().value().str();
  // Currently, we use fp80 type in MLIR to represent fp8.
  if (precision == "fp80") return "fp8";


}
// ---- 095/384  isOperationSelected  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:34  (2L)
bool e095_isOperationSelected(const Operation &op) {
  return isa<mlir::affine::AffineIfOp, scf::IfOp>(&op);

}
// ---- 096/384  createDummyYieldInElseReg  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:383  (13L)
static void createDummyYieldInElseReg(Operation *if_op) {
  DT_CHECK_MSG(if_op, "Expect valid op.");
  DT_CHECK_MSG(
      if_op->getNumResults() == 0 && if_op->getRegions()[1].empty(),
      "Expect conditionals with empty else regions to yield no values.");
  if_op->getRegions()[1].push_back(new Block);
  OpBuilder builder(if_op->getRegions()[1]);
  if (isa<scf::IfOp>(if_op))
    scf::YieldOp::create(builder, if_op->getLoc());
  else if (isa<affine::AffineIfOp>(if_op))
    affine::AffineYieldOp::create(builder, if_op->getLoc());
  else
    llvm_unreachable("unexpected IfOp type");

}
// ---- 097/384  getNewDbgNameFromList  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:456  (2L)
          dataflow::utils::getNewDbgNameFromList("CFGSM(", {src, dst})) {
    dataflow::setDbgNameAttr(dst, new_dbg_name_attr);

}
// ---- 098/384  getLhsRhsOfEQPredicate  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:519  (11L)
bool e098_getLhsRhsOfEQPredicate(Operation *op,
                                                              Value &lhs,
                                                              Value &rhs) {
  DT_CHECK_MSG(op, "Expect valid Operation.");
  auto if_op = llvm::dyn_cast<mlir::scf::IfOp>(op);
  if (!if_op) return false;
  auto cond = if_op.getCondition();
  auto cmpi_op = cond.getDefiningOp<mlir::arith::CmpIOp>();
  if (!cmpi_op || cmpi_op.getPredicate() != mlir::arith::CmpIPredicate::eq)
    return false;
  lhs = cmpi_op.getLhs();

}
// ---- 099/384  ConditionalSimplificationManager  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:78  (2L)
  ~ConditionalSimplificationManager() {
    if (val_array_) delete[] val_array_;

}
// ---- 100/384  TransformationConditionalTree  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.hpp:111  (5L)
      : TransformationConditionalTree(unit) {
    setOE(dcc::OperationEquivalence(nullptr, nullptr,
                                    "cfg-merging-and-hoisting-cond-tree",
                                    /*do_recursive_compare*/ true,
                                    /*all_block_args_are_equiv*/ false));

}
// ---- 101/384  OperationNode  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:51  (0L)
  LocalOpNode(Operation *op) : OperationNode(op) {};
// ---- 102/384  getParentNode  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:53  (2L)
  LocalOpNode *getParentNode() {
    return static_cast<LocalOpNode *>(OperationNode::getParentNode());

}
// ---- 103/384  getFirstChild  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:56  (2L)
  LocalOpNode *getFirstChild() {
    return static_cast<LocalOpNode *>(OperationNode::getFirstChild());

}
// ---- 104/384  getNextSibling  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:59  (2L)
  LocalOpNode *getNextSibling() {
    return static_cast<LocalOpNode *>(OperationNode::getNextSibling());

}
// ---- 105/384  getPrevSibling  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:62  (2L)
  LocalOpNode *getPrevSibling() {
    return static_cast<LocalOpNode *>(OperationNode::getPrevSibling());

}
// ---- 106/384  OperationTreeBase  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:78  (0L)
  FlatteningLocalRegionsTree() : OperationTreeBase() {}
// ---- 107/384  getRoot  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:81  (2L)
  const LocalOpNode *getRoot() {
    return static_cast<const LocalOpNode *>(OperationTreeBase::getRoot());

}
// ---- 108/384  partitionUnits  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:168  (17L)
void e108_partitionUnits(
    llvm::MapVector<mlir::Value, std::vector<mlir::Value>>
        &equivalence_classes) {
  // if the target operations the same, puts the units into the same bucket.
  std::vector<mlir::Value> visited;
  for (auto unit_to_op0 : unit_to_ops) {
    if (std::find(visited.begin(), visited.end(), unit_to_op0.first) !=
        visited.end())
      continue;
    for (auto unit_to_op1 : unit_to_ops) {
      if (std::find(visited.begin(), visited.end(), unit_to_op1.first) !=
          visited.end())
        continue;
      if (unit_to_op0.second == unit_to_op1.second) {
        equivalence_classes[unit_to_op0.first].push_back(unit_to_op1.first);
        visited.push_back(unit_to_op1.first);
      }

}}}
// ---- 109/384  performFullUnroll  —  dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:141  (7L)
LogicalResult e109_performFullUnroll(
    Operation *loop_op) {
  if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop_op))
    return performFullUnroll(scf_for);
  else if (auto affine_for = llvm::dyn_cast<affine::AffineForOp>(loop_op))
    return performFullUnroll(affine_for);
  else

}
// ---- 110/384  getConstantTripCount  —  dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:167  (14L)
std::optional<int64_t> e110_getConstantTripCount(
    scf::ForOp for_op) {
  auto lb_const = for_op.getLowerBound().getDefiningOp<arith::ConstantIntOp>();
  auto ub_const = for_op.getUpperBound().getDefiningOp<arith::ConstantIntOp>();
  auto step_const = for_op.getStep().getDefiningOp<arith::ConstantIntOp>();

  if (!lb_const || !ub_const || !step_const) return std::nullopt;

  auto lb = lb_const.value();
  auto ub = ub_const.value();
  auto step = step_const.value();

  if (step <= 0) return std::nullopt;


}
// ---- 111/384  getMaxMutableRange  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:673  (8L)
int64_t e111_getMaxMutableRange(SenComponents comp) {
  DT_CHECK(is_any_of(comp, SenComponents::L3LU, SenComponents::L3SU));
  auto &sys_def = dcc_ext_ctx_.dsc_global_->sysDef;
  return MaxMutableSize < 0
             ? pow(2,
                   sys_def.regInfoPerUnit.at(comp).at(RegType::EAR).bitSize) *
                   sys_def.bytesPerStick * 8
             : MaxMutableSize;

}
// ---- 112/384  getMaxImmutableRange  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:683  (8L)
int64_t e112_getMaxImmutableRange(
    SenComponents comp) {
  DT_CHECK(is_any_of(comp, SenComponents::L3LU, SenComponents::L3SU));
  auto &sys_def = dcc_ext_ctx_.dsc_global_->sysDef;
  return MaxImmutableSize < 0
             ? pow(2,
                   sys_def.regInfoPerUnit.at(comp).at(RegType::EBR).bitSize) *
                   sys_def.bytesPerStick * 8

}
// ---- 113/384  isEligibleForSplitting  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:832  (20L)
bool e113_isEligibleForSplitting(
    const SmallVectorImpl<Value> &all_mem_views) {
  // Other passes are responsible to remove variability from immutable
  // addresses. Currently, AddressPinningAndToggle is one such pass. However, it
  // expects a toggle or conditional immutable address to be used only in one
  // memory operation and a yield operation. At this time, MutableAddrSplitting
  // will not support any memory view used in the load/store chain where
  // splitting is required that does not have a constant start address.
  for (auto &mem_view : all_mem_views) {
    // Ineligible for splitting if any of the mem views are not a
    // GetLogicalMemoryViewOp.
    auto mem_view_op =
        dyn_cast<dataflow::GetLogicalMemoryViewOp>(mem_view.getDefiningOp());
    if (!mem_view_op) return false;

    // Start address must be a constant.
    if (!dcc::utils::isConstant<arith::ConstantOp>(
            mem_view_op.getStartAddress()))
      return false;
  }

}
// ---- 114/384  sortDataBasedOnWeight  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:855  (4L)
void e114_sortDataBasedOnWeight(
    SmallVectorImpl<MASData> &mas_data) {
  llvm::sort(mas_data, [](MASData &a, MASData &b) -> bool {
    return a.weight_ > b.weight_;

}}
// ---- 115/384  createNewMemViewWithMod  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:966  (12L)
MutableAddrSplittingPass::createNewMemViewWithMod(
    ExpressionEvaluator &evaluator, OpBuilder &builder,
    dataflow::ProgramUnitOp unit, Operation *op,
    dataflow::GetLogicalMemoryViewOp &mem_view_op, int64_t modifier) {
  auto start_addr_op = mem_view_op.getStartAddress().getDefiningOp();
  DT_CHECK(start_addr_op);

  auto new_mem_view_op =
      cast<dataflow::GetLogicalMemoryViewOp>(builder.clone(*mem_view_op));
  OpBuilder const_builder(unit);
  OpBuilder query_map_builder(dcc::uniform::utils::getLocalOrGlobalRegion(op));
  dcc::agen::utils::updateMemViewStartAddress(

}
// ---- 116/384  getMaxImmutableRange  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:355  (8L)
int64_t e116_getMaxImmutableRange(
    SenComponents comp) {
  DT_CHECK(is_any_of(comp, SenComponents::L3LU, SenComponents::L3SU));
  auto &sys_def = dcc_ext_ctx_.dsc_global_->sysDef;
  return MaxImmutableSize < 0
             ? pow(2,
                   sys_def.regInfoPerUnit.at(comp).at(RegType::EBR).bitSize) *
                   sys_def.bytesPerStick * 8

}
// ---- 117/384  transformSCFToAffineLoop  —  dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:102  (44L)
TransformLoopToLegalizeForSentientLowering::transformSCFToAffineLoop(
    OpBuilder builder, scf::ForOp scf_for, int lbound, int ubound, int step,
    affine::AffineForOp& affine_for) {
  // we currently support lbound being 0, step being 1, non-constant ubound.
  if (lbound == 0 && step == 1) {
    affine_for = affine::AffineForOp::create(builder, builder.getUnknownLoc(),
                                             0, ubound, 1, scf_for.getInits());

    if (auto dbg_name_attr = getDbgNameAttr(scf_for))
      setDbgNameAttr(affine_for, dbg_name_attr);

    // Map induction var, region iterator arguments to affine loop variables.
    IRMapping bv_map;
    bv_map.map(scf_for.getInductionVar(), affine_for.getInductionVar());
    for (unsigned i = 0; i < scf_for.getNumRegionIterArgs(); i++) {
      bv_map.map(scf_for.getRegionIterArgs()[i],
                 affine_for.getRegionIterArgs()[i]);
    }

    // Set insertion point to the beginning of the loop.
    builder.setInsertionPointToStart(&affine_for.getRegion().front());

    // Clone each operation within the body except the terminator.
    for (auto& op : scf_for.getRegion().front().without_terminator()) {
      builder.clone(op, bv_map);
    }

    // affine_for already has an implicit affine.yield. We need to update it if
    // there are results.
    if (scf_for.getNumResults() > 0) {
      // Has results, need to update the yield with mapped operands
      auto* scf_yield = scf_for.getRegion().front().getTerminator();
      SmallVector<Value> mapped_operands;
      for (auto operand : scf_yield->getOperands()) {
        mapped_operands.push_back(bv_map.lookup(operand));
      }

      builder.setInsertionPointToEnd(&affine_for.getRegion().front());
      builder.create<affine::AffineYieldOp>(affine_for->getLoc(),
                                            mapped_operands);
    }

    return LogicalResult::success();
  }

}
// ---- 118/384  removeValuesFromIndices  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:36  (7L)
void e118_removeValuesFromIndices(
    SmallVectorImpl<Value> &indices,
    SmallVectorImpl<Value> &indices_to_delete) {
  for (auto &index : indices_to_delete) {
    auto it = std::find(indices.begin(), indices.end(), index);
    DT_CHECK_MSG(it != indices.end(),
                 "could not find value to delete in indices vector");

}}
// ---- 119/384  replaceDimsInMapWithSyms  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:47  (7L)
AffineMap e119_replaceDimsInMapWithSyms(AffineMap &map) {
  SmallVector<AffineExpr, 16> sym_exprs;
  int num_args = 0;
  for (int dim = 0; dim < map.getNumDims(); ++dim)
    sym_exprs.emplace_back(getAffineSymbolExpr(num_args++, context_));

  return map.replaceDimsAndSymbols(sym_exprs, {}, 0, num_args);

}
// ---- 120/384  createEqualityCondition  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:253  (7L)
Operation *TPMVBase::createEqualityCondition(OpBuilder &builder, Value &lhs,
                                             int64_t rhs) {
  auto rhs_const =
      mlir::arith::ConstantIndexOp::create(builder, lhs.getLoc(), rhs);
  auto cond = mlir::arith::CmpIOp::create(
      builder, lhs.getLoc(), mlir::arith::CmpIPredicate::eq, lhs, rhs_const);
  return mlir::scf::IfOp::create(builder, lhs.getLoc(), cond.getResult(),

}
// ---- 121/384  createInequalityCondition  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:263  (14L)
Operation *TPMVBase::createInequalityCondition(OpBuilder &builder, Value &lhs,
                                               int64_t rhs_lb, int64_t rhs_ub) {
  auto lb_const = arith::ConstantIndexOp::create(builder, lhs.getLoc(), rhs_lb);
  auto lb_cond = mlir::arith::CmpIOp::create(
      builder, lhs.getLoc(), mlir::arith::CmpIPredicate::sge, lhs, lb_const);
  auto lb_if_op = mlir::scf::IfOp::create(builder, lhs.getLoc(),
                                          lb_cond.getResult(), false);
  builder = lb_if_op.getThenBodyBuilder();

  auto ub_const =
      mlir::arith::ConstantIndexOp::create(builder, lhs.getLoc(), rhs_ub);
  auto ub_cond = mlir::arith::CmpIOp::create(
      builder, lhs.getLoc(), mlir::arith::CmpIPredicate::sle, lhs, ub_const);
  return mlir::scf::IfOp::create(builder, lhs.getLoc(), ub_cond.getResult(),

}
// ---- 122/384  setBuilderToInsertRef  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:280  (6L)
void e122_setBuilderToInsertRef(OpBuilder &builder,
                                     Operation *insert_ref) {
  DT_CHECK(insert_ref);
  if (auto if_op = dyn_cast<scf::IfOp>(insert_ref))
    builder = if_op.getThenBodyBuilder();
  else

}
// ---- 123/384  calculateStartElementsForPage  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:532  (10L)
SmallVector<int64_t, 16> e123_calculateStartElementsForPage(
    FlatLinearValueConstraints &page_sel_constraints) {
  SmallVector<int64_t, 16> start_elements;
  for (int dim = 0; dim < page_sel_constraints.getNumDimVars(); ++dim) {
    auto lb = page_sel_constraints.getConstantBound(
        mlir::presburger::BoundType::LB, dim);
    DT_CHECK_MSG(lb.has_value(), "expected constant lower bound");
    start_elements.emplace_back((int64_t)lb.value());
  }


}
// ---- 124/384  createNonPagedMemView  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:545  (10L)
dataflow::GetLogicalMemoryViewOp e124_createNonPagedMemView(
    OpBuilder &builder, dataflow::GetPagedLogicalMemoryViewOp &paged_mem_view,
    int page_idx) {
  auto new_start_addr = mlir::arith::AddIOp::create(
      builder, paged_mem_view->getLoc(), paged_mem_view.getStartAddr(),
      paged_mem_view.getPageStartAddrs()[page_idx]);

  return mlir::dataflow::GetLogicalMemoryViewOp::create(
      builder, new_start_addr->getLoc(),
      cast<MemRefType>(paged_mem_view.getResult().getType()),

}
// ---- 125/384  cloneMemViewIfNonPaged  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:633  (9L)
Value e125_cloneMemViewIfNonPaged(OpBuilder &builder, Value mem_view) {
  if (auto non_paged_mem_view =
          dyn_cast_or_null<dataflow::GetLogicalMemoryViewOp>(
              mem_view.getDefiningOp())) {
    auto new_mem_view = builder.clone(*non_paged_mem_view);
    return new_mem_view->getResult(0);
  }

  return mem_view;

}
// ---- 126/384  getUseChain  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:673  (3L)
SmallVector<Operation *> e126_getUseChain(Operation *mem_op) {
  auto load_op = cast<agen::VectorLoadOp>(mem_op);
  return load_op.getUseChain();

}
// ---- 127/384  cloneUseChain  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:678  (3L)
void e127_cloneUseChain(OpBuilder &builder, Operation *mem_op,
                                   Operation *new_mem_op) {
  auto load_op = cast<agen::VectorLoadOp>(mem_op);

}
// ---- 128/384  createNewMemOp  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:684  (9L)
Operation *TPMVVectorLoad::createNewMemOp(OpBuilder &builder, Operation *mem_op,
                                          TPMVInfo &info, Value &mem_view,
                                          AffineMap &subscripts_map,
                                          SmallVectorImpl<Value> &indices) {
  auto load_op = cast<agen::VectorLoadOp>(mem_op);
  auto new_load_op = load_op.cloneWithNewAccessInfo(builder, mem_view,
                                                    subscripts_map, indices);

  builder.setInsertionPointAfter(new_load_op);

}
// ---- 129/384  eraseMemOpAndUseChain  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:698  (3L)
void e129_eraseMemOpAndUseChain(Operation *mem_op) {
  auto load_op = cast<agen::VectorLoadOp>(mem_op);
  load_op.eraseOpAndUseChain();

}
// ---- 130/384  getStoreOp  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:843  (6L)
agen::VectorStoreOp e130_getStoreOp(
    agen::VectorLoadOp &load_op) {
  DT_CHECK(load_op.getResult().hasOneUse());
  auto store_op =
      dyn_cast<agen::VectorStoreOp>(*load_op.getResult().user_begin());
  DT_CHECK(store_op);

}
// ---- 131/384  addTimeDimIndicesRanges  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:876  (5L)
void e131_addTimeDimIndicesRanges(
    const SmallVectorImpl<int64_t> &time_bounds,
    SmallVectorImpl<IVRange> &indices_ranges) {
  for (auto &b : time_bounds) {
    DT_CHECK_MSG(b - 1 >= 0, "no special time bound values should exist");

}}
// ---- 132/384  identifyTimeDimForExplicitLoops  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:963  (10L)
int e132_identifyTimeDimForExplicitLoops(int num_non_time_dims) {
  // Traverse innermost to outermost loops. Once a page_dependent time dim is
  // found, break. Dims below this dim may be preserved.
  for (int i = time_set_.getNumDims() - 1; i >= 0; --i) {
    if (auto it = page_dependent_time_syms_.find(i + num_non_time_dims) !=
                  page_dependent_time_syms_.end())
      return i;
  }

  return -1;

}
// ---- 133/384  getUseChain  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:328  (2L)
  virtual SmallVector<Operation *> getUseChain(Operation *mem_op) {
    return {};

}
// ---- 134/384  cloneUseChain  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:341  (0L)
  virtual void cloneUseChain(OpBuilder &builder, Operation *mem_op,
// ---- 135/384  eraseMemOpAndUseChain  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:364  (0L)
  virtual void eraseMemOpAndUseChain(Operation *mem_op) { mem_op->erase(); }
// ---- 136/384  TPMVBase  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:389  (0L)
  TPMVVector(Operation *mem_op, SenComponents comp) : TPMVBase(mem_op, comp) {}
// ---- 137/384  TPMVVector  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:397  (0L)
      : TPMVVector(mem_op, comp) {}
// ---- 138/384  TPMVComposite  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.hpp:519  (0L)
      : TPMVComposite(mem_op, comp) {}
// ---- 139/384  run  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewManager.cpp:21  (48L)
LogicalResult e139_run() {
  Operation *op = nullptr;
  DT_CHECK(paged_mem_view_->hasOneUse() &&
           "expecting paged memory view to have one user");
  op = *paged_mem_view_->getUsers().begin();
  DT_CHECK(op);

  if (auto load_op = dyn_cast<agen::VectorLoadOp>(op)) {
    //// agen::VectorLoadOp + agen::VectorStoreOp ////
    auto result = load_op.getResult();
    if (result.hasOneUse()) {
      if (auto store_op = dyn_cast<agen::VectorStoreOp>(*result.user_begin())) {
        TPMVVectorLoadStore tpmv(load_op, store_op, comp_);
        return tpmv.run();
      }
    }
    //// agen::VectorLoadOp ////
    TPMVVectorLoad tpmv(op, comp_);
    return tpmv.run();
  } else if (auto store_op = dyn_cast<agen::VectorStoreOp>(op)) {
    //// agen::VectorLoadOp + agen::VectorStoreOp ////
    auto val = store_op.getValueToStore();
    if (auto load_op =
            dyn_cast_or_null<agen::VectorLoadOp>(val.getDefiningOp())) {
      TPMVVectorLoadStore tpmv(load_op, store_op, comp_);
      return tpmv.run();
    }
    //// agen::VectorStoreOp ////
    TPMVVectorStore tpmv(op, comp_);
    return tpmv.run();
  } else if (auto comp_load_op = dyn_cast<agen::CompositeLoadOp>(op)) {
    //// agen::CompositeLoadOp ////
    TPMVCompositeLoad tpmv(op, comp_);
    return tpmv.run();
  } else if (auto comp_store_op = dyn_cast<agen::CompositeStoreOp>(op)) {
    //// agen::CompositeStoreOp ////
    TPMVCompositeStore tpmv(op, comp_);
    return tpmv.run();
  } else if (auto comp_load_store_op =
                 dyn_cast<agen::CompositeLoadAndStoreOp>(op)) {
    //// agen::CompositeLoadAndStoreOp ////
    TPMVCompositeLoadStore tpmv(op, comp_);
    return tpmv.run();
  } else {
    llvm_unreachable(
        "memory operation is not supported for static paged tensors");
  }
  return LogicalResult::failure();

}
// ---- 140/384  removeCoresCoreletsFoldsFromProgramUnit  —  dcc/src/Transform/Dataflow/UnitFiltering.cpp:240  (20L)
void e140_removeCoresCoreletsFoldsFromProgramUnit(
    dataflow::ProgramUnitOp unit_op) {
  for (int i = unit_op.getNumOperands() - 1; i >= 0; --i) {
    Value unit = unit_op.getUnits()[i];
    auto get_unit_op_result = llvm::dyn_cast<OpResult>(unit);
    unsigned fold_id = get_unit_op_result.getResultNumber();
    auto get_unit_op =
        llvm::dyn_cast<mlir::dataflow::GetUnitOp>(unit.getDefiningOp());
    unsigned core_id = dcc::getCoreId(get_unit_op);
    int corelet_id = dcc::getCoreletId(get_unit_op);
    if ((!filter_folds_except_.empty() &&
         filter_folds_except_.count(fold_id) == 0) ||
        (!filter_cores_except_.empty() &&
         filter_cores_except_.count(core_id) == 0) ||
        (!filter_corelets_except_.empty() && corelet_id != -1 &&
         filter_corelets_except_.count(corelet_id) == 0)) {
      unit_op->eraseOperand(i);
    }
  }
  DT_CHECK_MSG(unit_op.getNumOperands() >= 1,

}
// ---- 141/384  isDataTransfer  —  dcc/src/Transform/Dataflow/UnitFiltering.cpp:314  (11L)
static bool isDataTransfer(Operation *op) {
  return isa<dataflow::SyncSendOp, dataflow::SyncRecvOp,
             dataflow::ImplicitSyncOnStreamingBufferOp, dataflow::SendOp,
             dataflow::ReceiveOp, dataflow::OpaqueOp, agen::VectorLoadOp,
             agen::VectorStoreOp, agen::CompositeLoadOp, agen::CompositeStoreOp,
             agen::CompositeLoadAndStoreOp,
             agen::CompositeIndirectLoadAndStoreOp,
             agen::CompositeIndirectLoadOp, agen::CompositeIndirectStoreOp,
             agen::IndirectVectorLoadOp, agen::IndirectVectorStoreOp,
             agen::CompositeMemoryInterleaveOp, agen::SymbolicVectorLoadOp,
             agen::SymbolicVectorStoreOp>(op);

}
// ---- 142/384  getDataflowForLoopInfoIfIV  —  dcc/src/Transform/Dataflow/Utils.cpp:99  (31L)
getDataflowForLoopInfoIfIV(Value& val) {
  if (isa<BlockArgument>(val)) {
    Operation* for_op = cast<BlockArgument>(val).getOwner()->getParentOp();
    if (auto scf_for = llvm::dyn_cast<mlir::scf::ForOp>(for_op)) {
      auto lb_const =
          scf_for.getLowerBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
      auto ub_const =
          scf_for.getUpperBound().getDefiningOp<mlir::arith::ConstantIndexOp>();
      auto step_const =
          scf_for.getStep().getDefiningOp<mlir::arith::ConstantIndexOp>();
      if (lb_const && ub_const && step_const &&
          scf_for.getInductionVar() == val) {
        auto num_iterations =
            (ub_const.value() - lb_const.value()) / step_const.value();
        return std::make_tuple(for_op, lb_const.value(), ub_const.value(),
                               step_const.value(), num_iterations);
      }
    } else if (auto affine_for =
                   llvm::dyn_cast<mlir::affine::AffineForOp>(for_op)) {
      if (affine_for.hasConstantLowerBound() &&
          affine_for.hasConstantUpperBound() &&
          affine_for.getInductionVar() == val) {
        auto lb = affine_for.getConstantLowerBound();
        auto ub = affine_for.getConstantUpperBound();
        auto step = affine_for.getStepAsInt();
        auto num_iterations = (ub - lb) / step;
        return std::make_tuple(for_op, lb, ub, step, num_iterations);
      }
    }
  }
  return std::make_tuple(nullptr, 0, 0, 0, 0);


}
// ==================================================================================================
// LEVEL 1
// ==================================================================================================

// ---- 143/384  constructExtentAndTotalElements  —  dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:32  (64L)
LogicalResult e143_constructExtentAndTotalElements() {
  affine::FlatAffineValueConstraints transfer_set_csts(
      getTransferSet().getValue());
  auto transfer_order = getTransferOrder();
  if (transfer_set_csts.composeMatchingMap(transfer_order).failed())
    return LogicalResult::failure();

  transfer_set_csts.projectOut(transfer_order.getNumDims(),
                               transfer_order.getNumDims());
  transfer_set_csts.removeRedundantConstraints();

  // TODO: currently, we do support only row major and column major in lowering.
  SmallVector<int64_t, 8> layout_coeffs;
  agen::utils::getMapCoefficients(layout_coeffs, getMemViewLayoutMap(), 0);
  setLayoutCoeffs(layout_coeffs);
  auto layout_size = layout_coeffs.size();
  auto last_index = layout_size != transfer_set_csts.getNumDimVars()
                        ? layout_size - 2
                        : layout_size - 1;
  bool is_row_major = (layout_coeffs[0] > layout_coeffs[last_index]);

  auto op = getOp();
  auto total_elements = getTotalElements();
  SmallVector<int> extents;
  for (int i = 0; i < transfer_set_csts.getNumDimVars(); i++) {
    int width = 0;
    if (agen::utils::isDimValueZero(transfer_set_csts, i)) {
      extents.push_back(1);
      total_elements = (total_elements == 0) ? 1 : total_elements * 1;
    } else if (agen::utils::isDimAConstantRange(transfer_set_csts, i, width)) {
      if (width > 0) {
        extents.push_back(width);
        total_elements = (total_elements == 0) ? width : total_elements * width;

        int size = 0;
        if (is_row_major) {
          size = (i == 0) ? (width + 1)
                          : (layout_coeffs[i - 1] / layout_coeffs[i]);
        } else {
          size = (i == last_index) ? (width + 1)
                                   : (layout_coeffs[i + 1] / layout_coeffs[i]);
        }

        if (width > size) {
          return op->emitError(
              "Extent requsted along a dimension is "
              "more than its limit");
        }
      } else {
        return op->emitError("Extent along a dimension is negative");
      }
    } else {
      return op->emitError("Not in a hyper-rectangular form");
    }
  }
  setTotalElements(total_elements);
  setExtents(extents);

  if (this->getExpectedTotalElements() != total_elements)
    return op->emitError(
        "Number of elements in return type not matching "
        "with load_set/store_set elements");

  return LogicalResult::success();

}
// ---- 144/384  constructLdOrStType  —  dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:267  (5L)
void e144_constructLdOrStType() {
  if (is_any_of(getComp(), L0LU, L0SU))
    setLdOrStSize(getChunkSize());
  else
    setLdOrStSize(getExpectedTotalElements());

}
// ---- 145/384  initializeMemViewInfo  —  dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:274  (16L)
LogicalResult e145_initializeMemViewInfo() {
  auto mem_ref = getMemRef().getDefiningOp();
  if (auto mem_view = dyn_cast<dataflow::GetLogicalMemoryViewOp>(mem_ref)) {
    setMemViewLayoutMap(mem_view.getLayoutMap());
    setMemViewStartAddr(mem_view.getStartAddress());
    setMemory(mem_view.getFromUnit());
  } else if (auto paged_mem_view =
                 dyn_cast<dataflow::GetPagedLogicalMemoryViewOp>(mem_ref)) {
    setMemViewLayoutMap(paged_mem_view.getLayoutMap());
    setMemViewStartAddr(paged_mem_view.getStartAddr());
    setMemory(paged_mem_view.getUnit());
  } else {
    llvm_unreachable("unsupported mem_ref type");
  }

  return LogicalResult::success();

}
// ---- 146/384  constructIndices  —  dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:354  (50L)
LogicalResult e146_constructIndices() {
  // Step-1: Construct indices
  std::map<int, int> dim_to_cst_val_map;
  SmallVector<Value> new_indices;
  auto indices = getIndices();
  auto op = getOp();
  for (int dim = 0; dim < indices.size(); dim++) {
    auto& index = indices[dim];
    if (mlir::isa<BlockArgument>(index)) {
      auto* loop_op =
          mlir::cast<BlockArgument>(index).getOwner()->getParentOp();
      if (isa<affine::AffineForOp, scf::ForOp>(loop_op))
        new_indices.push_back(index);
      else
        return op->emitError(
            "The loop iterators involved in the agen memory "
            "operation subscripts have to be affine loops");
    } else if (auto const_op = index.getDefiningOp<mlir::arith::ConstantOp>()) {
      auto val = mlir::cast<IntegerAttr>(const_op.getValue()).getInt();
      dim_to_cst_val_map[dim] = val;
    } else {
      return op->emitError(
          "All the map operands need to be either loop"
          "iterators or constant values");
    }
  }
  setIndices(new_indices);

  // substitute constant indices into the subscripts map itself
  auto subscripts_map = getSubscriptsMap();
  if (!dim_to_cst_val_map.empty()) {
    int ndims = 0;
    llvm::SmallVector<AffineExpr, 2> operand_exprs;
    llvm::SmallVector<AffineExpr, 1> symbol_exprs;
    for (int dim = 0; dim < indices.size(); dim++) {
      auto record = dim_to_cst_val_map.find(dim);
      if (record == dim_to_cst_val_map.end()) {
        operand_exprs.push_back(getAffineDimExpr(ndims++, op->getContext()));
      } else {
        auto cst_expr = getAffineConstantExpr(record->second, op->getContext());
        operand_exprs.push_back(cst_expr);
      }
    }

    subscripts_map = subscripts_map.replaceDimsAndSymbols(
        operand_exprs, symbol_exprs, ndims, 0);
  }
  setSubscriptsMap(subscripts_map);

  return LogicalResult::success();

}
// ---- 147/384  constructIteratorCoefficients  —  dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:406  (10L)
LogicalResult e147_constructIteratorCoefficients() {
  auto subscripts_map = getSubscriptsMap();
  auto transfer_order = getTransferOrder();
  auto mem_view_layout_map = getMemViewLayoutMap();
  auto indices = getIndices();
  auto indices_coeff_dict = agen::utils::constructIteratorCoeffDict(
      subscripts_map, transfer_order, mem_view_layout_map, indices);
  setIndicesCoeffDict(indices_coeff_dict);

  return success();

}
// ---- 148/384  computeBurstAndGroup  —  dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:796  (36L)
void e148_computeBurstAndGroup() {
  auto time_bounds = getTimeBounds();
  auto time_offsets = getTimeOffsets();
  for (int i = time_bounds.size() - 1; i >= 0; --i) {
    // llvm::outs() << "time bound idx: " << i << "\n";
    auto curr_bound = time_bounds[i];
    // llvm::outs() << "check bound:: " << curr_bound << "\n";
    if (curr_bound == kCoalesced) {
      // ignore coalesced time dim
      continue;
    } else if (curr_bound < 0) {
      // if forOp bound is variable, terminate search
      return;
    } else if (curr_bound == 0) {
    } else {
      // if current bound is valid value, find a field to fit
      auto burst_index = getBurstIndex();
      if (burst_index < 0) {
        // always fill in burst field first before group.
        setBurstIndex(i);
        // llvm::outs() << "set burst: " << i << "\n";
      } else if (!is_any_of(getComp(), L3LU, L3SU)) {
        // if burst field has already been used, check if group field can be
        // used.
        if ((time_bounds[burst_index] == 2 || time_bounds[burst_index] == 4) &&
            time_offsets[i] == getLdOrStSize() &&
            time_offsets[burst_index] != 0) {
          setInterleaveGroupIndex(burst_index);
          setBurstIndex(i);
        }
        // llvm::outs() << "group  idx: " << burst_index << ", burst idx: " <<
        // i << "\n";
        return;
      }
    }
  }

}
// ---- 149/384  insert  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:388  (7L)
  void insert(MemoryOperandIndex moi, T access) {
    DT_CHECK_MSG(!has(moi),
                 "trying to insert into a slot that is already filled");
    DT_CHECK_MSG(this->size() <= MemoryOperandIndex::kMax,
                 "no more than kMax entries are allowed");
    index_mapping_[moi] = this->size();
    this->push_back(access);

}
// ---- 150/384  get  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:401  (4L)
  const T& get(MemoryOperandIndex moi) {
    DT_CHECK_MSG(has(moi),
                 "no entry exists for the requested memory operand index");
    return this->at(index_mapping_[(int)moi]);

}
// ---- 151/384  checkCompositeRegion  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:206  (89L)
LogicalResult e151_checkCompositeRegion(Operation* op) {
  if (isa<agen::CompositeLoadOp, agen::CompositeIndirectLoadOp>(op)) {
    Region* region;
    if (auto load_op = dyn_cast<agen::CompositeLoadOp>(op)) {
      region = &load_op.getRegion();
    } else {
      auto ind_load_op = cast<agen::CompositeIndirectLoadOp>(op);
      region = &ind_load_op.getRegion();
    }
    auto op_list = &region->front().getOperations();

    // check region's operation number
    auto region_size = op_list->size();
    if (region_size != 3 && region_size != 2) {
      return op->emitError(
          "composite_load's region size must be 2 or 3 and only "
          "dataflow.send, vectorchain.select + dataflow.send, or "
          "vectorchain.shuffle + dataflow.send are allowed!\n");
    }

    // check the last operation before yieldOp is sendOp
    auto curr_op = op_list->rbegin();
    curr_op++;  // skip yieldOp
    auto send_op = dyn_cast<dataflow::SendOp>(*curr_op);
    if (!send_op) return op->emitError("A sendOp must precede yieldOp.\n");

    // check send_data's def op
    auto send_data_def_op = send_op.getSendData().getDefiningOp();
    curr_op++;
    if ((curr_op == op_list->rend() && send_data_def_op != op) &&
        (curr_op != op_list->rend() &&
         (curr_op.getNodePtr() != send_data_def_op ||
          !isa<vectorchain::SelectOp, vectorchain::ShuffleOp>(
              send_data_def_op))))
      return op->emitError("Wrong send_data's def op.\n");
  } else if (isa<agen::CompositeStoreOp, agen::CompositeIndirectStoreOp>(op)) {
    Region* region;
    if (auto store_op = dyn_cast<agen::CompositeStoreOp>(op)) {
      region = &store_op.getRegion();
    } else {
      auto ind_store_op = cast<agen::CompositeIndirectStoreOp>(op);
      region = &ind_store_op.getRegion();
    }
    auto op_list = &region->front().getOperations();

    // check the region conforms to one of the allowed formats
    auto region_size = op_list->size();
    if (region_size == 2) {
      auto& curr_op = op_list->front();
      // The YieldOp should be preceded by a ReceiveOp.
      if (!isa<dataflow::ReceiveOp>(curr_op))
        return op->emitError("A ReceiveOp must precede yieldOp.\n");

      // check the op result is returned by yieldOp
      if (curr_op.user_begin() == curr_op.user_end())
        return op->emitError(
            "composite_store region yielding incorrect value!");
    } else if (region_size == 3) {
      // The YieldOp should be preceded by a ReceiveOp + ShuffleOp or a
      // ConstantBitstreamOp + ShuffleOp that feeds the YieldOp.
      Operation& curr_op = op_list->front();
      if (!isa<vectorchain::ConstantBitstreamOp>(curr_op) &&
          !isa<dataflow::ReceiveOp>(curr_op))
        return op->emitError(
            "composite_store with region size 3 must contain only "
            "dataflow.receive and one vectorchain.shuffle or "
            "vectorchain.constant_bitstream and one vectorchain.shuffle!\n");

      auto shuffle_op = dyn_cast<vectorchain::ShuffleOp>(curr_op.getNextNode());
      if (!shuffle_op)
        return op->emitError(
            "composite_store with region size 3 must contain a "
            "vectorchain.shuffle!\n");

      auto yield_op = cast<agen::YieldOp>(region->front().getTerminator());
      if (!curr_op.hasOneUse() || !shuffle_op->hasOneUse() ||
          *shuffle_op->getUsers().begin() != yield_op)
        return op->emitError(
            "composite_store region is not a linear chain to the "
            "terminator!\n");
    } else {
      return op->emitError(
          "composite_store's region size must be 2 or 3 and only 1 "
          "dataflow.receive OR 1 "
          "dataflow.receive/vectorchain.constant_bitstream and 1 "
          "vectorchain.shuffle is allowed!\n");
    }
  }
  return LogicalResult::success();

}
// ---- 152/384  checkStoreOpFromExtractPattern  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:433  (27L)
bool e152_checkStoreOpFromExtractPattern(
    VectorStoreOp& store_op) {
  // StoreOp should be for a 1d space with 0 start address.
  if (store_op.getMapOperands().size() != 1) {
    store_op->emitError("expecting indices size 1");
    return false;
  }
  auto store_start_addr = dyn_cast<arith::ConstantOp>(
      store_op.getMapOperands().front().getDefiningOp());
  auto store_start_addr_int =
      dyn_cast<IntegerAttr>(store_start_addr.getValue());
  if (!store_start_addr_int || store_start_addr_int.getInt() != 0) {
    store_op->emitError("expecting start offset of 0 from vector_store");
    return false;
  }

  if (store_op.getStoreSet().getValue().getNumDims() != 1) {
    store_op->emitError("expecting 1D store_set");
    return false;
  }

  auto store_map = store_op.getStoreOrder();
  if (store_map.getNumDims() != 1 || !store_map.isIdentity()) {
    store_op->emitError("expecting 1D identity map for store_map");
    return false;
  }


}
// ---- 153/384  isLoadAndExtractScalarPattern  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:463  (27L)
bool e153_isLoadAndExtractScalarPattern(
    VectorLoadOp& load_op) {
  auto load_mem_view = dyn_cast<dataflow::GetLogicalMemoryViewOp>(
      load_op.getMemRef().getDefiningOp());
  if (!load_mem_view) return false;
  auto load_mem_view_unit = dyn_cast_or_null<dataflow::GetUnitOp>(
      load_mem_view.getFromUnit().getDefiningOp());
  if (!load_mem_view_unit) return false;
  SenComponents load_comp = EnumsConversion::stringToSenComponents
                                .find(load_mem_view_unit.getType().str())
                                ->second;
  if (load_comp != SenComponents::LX) return false;

  if (!load_op->hasOneUse()) return false;
  auto store_op = dyn_cast<VectorStoreOp>(*load_op->getUsers().begin());
  if (!store_op) return false;
  auto store_mem_view = dyn_cast<dataflow::GetLogicalMemoryViewOp>(
      store_op.getMemRef().getDefiningOp());
  if (!store_mem_view) return false;
  auto store_mem_view_unit = dyn_cast_or_null<dataflow::GetUnitOp>(
      store_mem_view.getFromUnit().getDefiningOp());
  if (!store_mem_view_unit) return false;
  SenComponents store_comp = EnumsConversion::stringToSenComponents
                                 .find(store_mem_view_unit.getType().str())
                                 ->second;
  if (store_comp != SenComponents::LXVIRTUALIBR) return false;


}
// ---- 154/384  isReceiveAndExtractScalarPattern  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:493  (17L)
bool e154_isReceiveAndExtractScalarPattern(
    VectorStoreOp& store_op) {
  DT_CHECK(store_op);
  auto store_mem_view = dyn_cast<dataflow::GetLogicalMemoryViewOp>(
      store_op.getMemRef().getDefiningOp());
  if (!store_mem_view) return false;
  auto store_mem_view_unit = dyn_cast_or_null<dataflow::GetUnitOp>(
      store_mem_view.getFromUnit().getDefiningOp());
  if (!store_mem_view_unit) return false;
  // Note: Can't use dcc::getUnitType() here because the memory views use
  //       components in the form L0/LX/PE etc. and these don't exist in
  //       the EnumsConversion::senCompToGenericComp enum.
  SenComponents comp = EnumsConversion::stringToSenComponents
                           .find(store_mem_view_unit.getType().str())
                           ->second;
  if (comp != SenComponents::LXVIRTUALIBR) return false;


}
// ---- 155/384  updateSymbolicAccessDetails  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:1013  (35L)
void e155_updateSymbolicAccessDetails(
    AccessContainer<AccessDetailsSymbolic>& access_details, IRMapping& ir_map) {
  for (auto& ad : access_details) {
    // Update the operation. This will always be in the innermost loop
    // so it will always be cloned.
    auto new_op = ir_map.lookupOrDefault(ad.getOp());
    ad.setOp(new_op);

    // Update the indices.
    SmallVector<Value> indices = ad.getIndices(), new_indices;
    for (auto& index : indices) {
      auto new_index = ir_map.lookupOrDefault(index);
      new_indices.push_back(new_index);
    }
    ad.setIndices(new_indices);

    // Update the strides.
    SmallVector<Value> strides = ad.getStrides(), new_strides;
    for (auto& stride : strides) {
      auto new_stride = ir_map.lookupOrDefault(stride);
      new_strides.push_back(new_stride);
    }
    ad.setStrides(new_strides);

    // Update the mem_ref.
    auto new_mem_ref = ir_map.lookupOrDefault(ad.getMemRef());
    ad.setMemRef(new_mem_ref);

    // Update the mem_view_start_addr.
    auto new_mem_start = ir_map.lookupOrDefault(ad.getMemViewStartAddr());
    ad.setMemViewStartAddr(new_mem_start);

    // Update the memory operation.
    auto new_memory = ir_map.lookupOrDefault(ad.getMemory());
    ad.setMemory(new_memory);

}}
// ---- 156/384  getStoreProducer  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:1285  (159L)
std::pair<Operation*, Operation*> e156_getStoreProducer(
    Operation* op) {
  Operation* inp_op = nullptr;
  if (isa<VectorStoreOp, IndirectVectorStoreOp, SymbolicVectorStoreOp>(op)) {
    if (auto store_op = dyn_cast<VectorStoreOp>(op)) {
      inp_op = store_op.getValueToStore().getDefiningOp();
    } else if (auto ind_store_op = dyn_cast<IndirectVectorStoreOp>(op)) {
      inp_op = ind_store_op.getValue().getDefiningOp();
    } else {
      auto sym_store_op = cast<SymbolicVectorStoreOp>(op);
      inp_op = sym_store_op.getValueToStore().getDefiningOp();
    }

    DT_CHECK(
        isa<dataflow::ReceiveOp>(inp_op) ||
        isa<vectorchain::ShuffleOp>(inp_op) &&
            "expecting a ReceiveOp or a ShuffleOp as input to VectorStoreOp");
  } else if (auto comp_store_op = dyn_cast<CompositeStoreOp>(op)) {
    if (const auto input_vector = comp_store_op.getInputVector();
        input_vector) {
      // coalesce store
      Operation* def_op = input_vector.getDefiningOp();
      if (auto prev_op = dyn_cast_or_null<dataflow::ReceiveOp>(def_op)) {
        inp_op = prev_op;
#ifdef COMMENT_OUT_COALESCE_STORE
#else
      } else if (auto coalesce_op =
                     dyn_cast_or_null<vectorchain::CoalesceOp>(def_op)) {
        if (auto prev_op =
                coalesce_op.data().getDefiningOp<dataflow::ReceiveOp>()) {
          inp_op = prev_op;
        } else {
          op->emitOpError("must be preceded by dataflow.receiveOp.");
          return std::make_pair(nullptr, nullptr);
        }
#endif
      } else if (auto prev_op =
                     dyn_cast_or_null<vectorchain::ShuffleOp>(def_op)) {
        inp_op = prev_op;
      } else {
        op->emitOpError(
            "must be preceded by vectorchain.coalesce, vectorchain.shuffle, "
            "or dataflow.receive.");
        return std::make_pair(nullptr, nullptr);
      }
    } else {
      // CompositeStore region contains either:
      //   - a ReceiveOp followed by a YieldOp
      //   - a ReceiveOp followed by a ShuffleOp
      //   - a ConstantBitstreamOp followed by a ShuffleOp
      auto& op_list = comp_store_op.getRegion().front().getOperations();
      Operation& front_op = op_list.front();
      if (auto recv_op = dyn_cast<dataflow::ReceiveOp>(front_op)) {
        Operation* next_op = recv_op->getNextNode();
        if (isa<agen::YieldOp>(next_op))
          inp_op = recv_op;
        else if (isa<vectorchain::ShuffleOp>(next_op))
          inp_op = next_op;
      } else if (auto bs_op =
                     dyn_cast<vectorchain::ConstantBitstreamOp>(front_op)) {
        Operation* next_op = bs_op->getNextNode();
        if (isa<vectorchain::ShuffleOp>(next_op)) inp_op = next_op;
      }
    }
  } else if (auto ind_comp_store_op = dyn_cast<CompositeIndirectStoreOp>(op)) {
    // CompositeIndirectStore region contains either:
    //   - a ReceiveOp followed by a YieldOp
    //   - a ReceiveOp followed by a ShuffleOp
    //   - a ConstantBitstreamOp followed by a ShuffleOp
    auto& op_list = ind_comp_store_op.getRegion().front().getOperations();
    Operation& front_op = op_list.front();
    if (auto recv_op = dyn_cast<dataflow::ReceiveOp>(front_op)) {
      Operation* next_op = recv_op->getNextNode();
      if (isa<agen::YieldOp>(next_op))
        inp_op = recv_op;
      else if (isa<vectorchain::ShuffleOp>(next_op))
        inp_op = next_op;
    } else if (auto bs_op =
                   dyn_cast<vectorchain::ConstantBitstreamOp>(front_op)) {
      Operation* next_op = bs_op->getNextNode();
      if (isa<vectorchain::ShuffleOp>(next_op)) inp_op = next_op;
    }
  }

  if (!inp_op) {
    op->emitError("unsupported storeOp producer! (1)");
    return std::make_pair(nullptr, nullptr);
  }

  Operation* producer = nullptr;
  if (auto recv_op = dyn_cast<dataflow::ReceiveOp>(inp_op)) {
    producer = recv_op.getFromUnit().getDefiningOp();
    DT_CHECK(
        producer &&
        (isa<dataflow::GetUnitOp>(producer) ||
         isa<dataflow::CreateMulticastGroupOp>(producer) ||
         isa<uniform::QueryMapOp>(producer) ||
         (isa<scf::IfOp>(producer) &&
          (dcc::utils::findYieldsResolvingTo<dataflow::CreateMulticastGroupOp,
                                             scf::IfOp>(producer) ||
           dcc::utils::findYieldsResolvingTo<dataflow::GetUnitOp, scf::IfOp>(
               producer)))) &&
        "expected a get_unit, create_multicast_group, or ifOp producer for "
        "receive inputs");
  } else if (auto shuf_op = dyn_cast<vectorchain::ShuffleOp>(inp_op)) {
    Operation* shuffle_input = shuf_op.getInput().getDefiningOp();
    if (auto recv_op = dyn_cast<dataflow::ReceiveOp>(shuffle_input)) {
      producer = recv_op.getFromUnit().getDefiningOp();
      DT_CHECK(
          producer && isa<dataflow::GetUnitOp>(producer) &&
          "producers for ReceiveOps feeding ShuffleOps should be a GetUnitOp");
    } else if (auto bs_op = dyn_cast_or_null<vectorchain::ConstantBitstreamOp>(
                   shuffle_input)) {
      producer = shuffle_input;
      // ConstantBitstreamOp producers should have:
      //   - one value (TODO: Expand this to any number of values)
      //   - the result should be an 8 or 16 bit input
      auto bs_value = bs_op.getValue().getValue();
      if (bs_value.size() != 1) {
        op->emitOpError("ConstantBitstreamOp producers should contain 1 value");
        return std::make_pair(nullptr, nullptr);
      }
      unsigned bs_result_bitwidth =
          dataflow::utils::getElementTypeBitWidth(bs_op.getResult().getType());
      if (bs_result_bitwidth != 4 && bs_result_bitwidth != 8 &&
          bs_result_bitwidth != 16) {
        op->emitOpError("ConstantBitstreamOp producers should be 8 or 16 bits");
        return std::make_pair(nullptr, nullptr);
      }

      // ShuffleOp inputs with constant bitstreams should have:
      //   - indices matching a NoPaddingFirstElemSplat pattern
      //   - the result should be a vector of 128 elements of 8 bit type or a
      //     vector of 64 elements of 16 bit type
      if (!isFirstElemSplat(shuf_op)) {
        op->emitOpError("unsupported splat mode for ShuffleOp input");
        return std::make_pair(nullptr, nullptr);
      }
      int num_elems =
          dataflow::utils::getNumElements(shuf_op.getResult().getType());
      unsigned bitwidth = dataflow::utils::getElementTypeBitWidth(
          shuf_op.getResult().getType());
      if (!(num_elems == 256 && bitwidth == 4) &&
          !(num_elems == 128 && bitwidth == 8) &&
          !(num_elems == 64 && bitwidth == 16)) {
        op->emitOpError("unsupported splat result for ShuffleOp input");
        return std::make_pair(nullptr, nullptr);
      }
    } else {
      op->emitError(
          "ShuffleOp inp_op producers can only be a ReceiveOp or "
          "a ConstantBitstreamOp");
      return std::make_pair(nullptr, nullptr);
    }
  } else {
    op->emitError("unsupported storeOp producer! (2)");
    return std::make_pair(nullptr, nullptr);
  }


}
// ---- 157/384  constructSetActiveMaskValueOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2567  (161L)
AgenToSentientLoweringPass::constructSetActiveMaskValueOp(
    dataflow::ProgramUnitOp unit, SetTransferMaskStateOp mask_op) {
  // Only valid in LXLU.
  auto comp =
      dcc::getUnitType(unit.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
  DT_CHECK_MSG(comp == LXLU, "only supported in LXLU unit");

  // Get num slices
  int num_slices = mask_op.getNumSlices();
  DT_CHECK_MSG(num_slices == 8, "SAMV requires 8 slices");

  // Get number of elements per slice:
  // - get num elements of the result vector, divide by num_slices
  auto result_type = mlir::cast<VectorType>(mask_op.getResult().getType());
  int num_elems = result_type.getNumElements();
  int elems_per_slice = num_elems / num_slices;

  DT_CHECK(isa<IntegerType>(result_type.getElementType()) ||
           isa<FloatType>(result_type.getElementType()) &&
               "expecting int or float element type");
  int precision = result_type.getElementType().getIntOrFloatBitWidth();
  SenSystemDef sysDef;
  DT_CHECK_MSG(((num_elems * precision) == (sysDef.bytesPerStick * 8)),
               "SAMV requires masks of stick length");

  // Parse slice_mask_map for the SAMV pattern
  auto slice_mask_map = mask_op.getSliceMaskMap();
  auto dbg_name_attr = getDbgNameAttr(mask_op);
  if (mask_op.isUnmask()) {
    // SAMV pattern for resetting detected
    //   (0)(0)(0)(0)(0)(0)(0)(0)
    DT_CHECK_MSG(!mask_op.getNumUnmaskedElements().has_value(),
                 "SAMVs resetting the mask should not have mask attributes");
    OpBuilder builder(mask_op);
    return sentient::SetActiveMaskValueOp::create(
        builder, mask_op.getLoc(), mask_op.getMaskValue(),
        /*maskall*/ false,
        /*numvalidentry*/ 0, /*sliceid-xsl*/ 7, /*xslinner*/ false,
        /*wsllen*/ 1, result_type.getElementType().getIntOrFloatBitWidth(),
        dbg_name_attr);
  } else if (mask_op.isFullMask()) {
    // SAMV pattern for full masking detected
    //   (1)(1)(1)(1)(1)(1)(1)(1)
    OpBuilder builder(mask_op);
    DT_CHECK_MSG(!mask_op.getNumUnmaskedElements().has_value(),
                 "SAMVs fully masking should not contain a mask");
    return sentient::SetActiveMaskValueOp::create(
        builder, mask_op.getLoc(), mask_op.getMaskValue(),
        /*maskall*/ true,
        /*numvalidentry*/ 0, /*sliceid-xsl*/ 0, /*xslinner*/ false,
        /*wsllen*/ 1, result_type.getElementType().getIntOrFloatBitWidth(),
        dbg_name_attr);
  }
  // SAMVs not matching a pattern above must match the generic SAMV pattern:
  //   - 8 slices total consisting of 0-7 (A), followed by (A|B),
  //     followed by 0-7 (1)
  // Valid pattern examples:
  // (A|B)(1)(1)(1)(1)(1)(1)(1)
  // (A)(A)(A)(A)(A)(A|B)(1)(1)
  // (A)(A)(A)(A)(A)(A)(A)(A|B)
  DT_CHECK_MSG(agen::utils::isGenericSAMV(mask_op),
               "invalid slice_mask_map for SAMV");

  int sliceid_xsl = agen::utils::getSliceIDXsl(mask_op);

  // Verify there are two masks: maskA for Wsl and maskB for Xsl.
  // Note: Verifier verifies unmasked and masked are same length.
  DT_CHECK_MSG(mask_op.getNumUnmaskedElements().has_value(),
               "generic SAMVs should have mask attributes");
  auto unmasked_elems = mask_op.getNumUnmaskedElements().value();
  auto masked_elems = mask_op.getNumMaskedElements().value();
  DT_CHECK_MSG(unmasked_elems.size() == 2,
               "SAMV should only contain two masks");

  // Determine xslinner
  // maskA (first mask) is MaskWsl, maskB (second mask) is MaskXsl
  // if (maskB unmasked + masked elements == num elements per slice)
  //   xslinner = true
  // else
  //   xslinner = false
  int mask_wsl_elems = mlir::cast<IntegerAttr>(unmasked_elems[0]).getInt() +
                       mlir::cast<IntegerAttr>(masked_elems[0]).getInt();
  int mask_xsl_elems = mlir::cast<IntegerAttr>(unmasked_elems[1]).getInt() +
                       mlir::cast<IntegerAttr>(masked_elems[1]).getInt();
  bool xslinner = mask_xsl_elems != elems_per_slice;

  // Determine numvalidentry.
  // 1. Determine inner dim length:
  //      inner dim length = (xslinner == 0) ? maskA unmasked + masked :
  //                                           maskB unmasked + masked
  int inner_dim_idx, outer_dim_idx;
  if (xslinner) {
    inner_dim_idx = 1;
    outer_dim_idx = 0;
  } else {
    inner_dim_idx = 0;
    outer_dim_idx = 1;
  }
  int inner_dim_unmasked =
      mlir::cast<IntegerAttr>(unmasked_elems[inner_dim_idx]).getInt();
  int inner_dim_masked =
      mlir::cast<IntegerAttr>(masked_elems[inner_dim_idx]).getInt();
  int inner_dim_len = inner_dim_unmasked + inner_dim_masked;
  DT_CHECK_MSG(elems_per_slice % inner_dim_len == 0,
               "inner dim mask should be evenly divisible into the slice");

  // 2. Determine inner dim num valid:
  //      inner dim num valid = xslinner ? maskB unmasked :
  //                                       maskA unmasked
  //
  //      if (inner dim num valid == inner dim length)
  //        inner dim num valid = 0
  int inner_dim_valid = inner_dim_unmasked;
  if (inner_dim_valid == inner_dim_len) inner_dim_valid = 0;

  // 3. Determine outer dim len:
  //      outer dim len = num elements per slice / inner dim len
  int outer_dim_len = elems_per_slice / inner_dim_len;

  // 4. Determine outer dim num valid:
  //      outer dim num valid = xslinner ?
  //                                maskB unmasked / inner dim len :
  //                                maskA unmasked / inner dim len
  //
  //      if (outer dim num valid == outer dim length)
  //        outer dim num valid = 0
  int outer_dim_unmasked =
      mlir::cast<IntegerAttr>(unmasked_elems[outer_dim_idx]).getInt();
  DT_CHECK((outer_dim_unmasked +
                mlir::cast<IntegerAttr>(masked_elems[outer_dim_idx]).getInt() ==
            elems_per_slice) &&
           "outer dim masked and unmasked should span a whole slice");
  int outer_dim_valid = outer_dim_unmasked / inner_dim_len;
  if (outer_dim_valid == outer_dim_len) outer_dim_valid = 0;

  // 5. Form encoding
  //      if (xslinner) {
  //        - num bits for outer dim = log2(outer_dim_len)
  //        - shift inner dim num valid bits required to represent outer dim
  //          valid
  //        - numvalidentries = shifted inner dim num valid +
  //                            outer dim num valid
  //      } else {
  //        - num bits for inner dim = log2(inner_dim_len)
  //        - shift outer dim num valid bits required to represent inner dim
  //          valid
  //        - numvalidentries = shifted outer dim num valid +
  //                            inner dim num valid
  //      }
  int numvalidentry =
      xslinner
          ? (inner_dim_valid << (int)std::log2(outer_dim_len)) + outer_dim_valid
          : (outer_dim_valid << (int)std::log2(inner_dim_len)) +
                inner_dim_valid;

  // Create the sentient.samv operation
  // Note: maskall is always false unless the Full Mask pattern was detected.
  OpBuilder builder(mask_op);
  return sentient::SetActiveMaskValueOp::create(
      builder, mask_op.getLoc(), mask_op.getMaskValue(),
      /*maskall*/ false, numvalidentry, sliceid_xsl, xslinner,

}
// ---- 158/384  createUniformizeRegionsOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3880  (57L)
AgenToSentientLoweringPass::createUniformizeRegionsOp(OpBuilder* builder,
                                                      Operation* region_op,
                                                      Operation* loop_op,
                                                      unsigned region_idx) {
  uniform::UniformizeRegionsOp new_uniformize_op;
  SmallVector<Type, 1> result_types = {builder->getIndexType()};
  SmallVector<Value> units;
  ArrayAttr list_sizes;
  unsigned num_of_regions;
  if (auto uniformize_op = dyn_cast<uniform::UniformizeRegionsOp>(region_op)) {
    units = uniformize_op.getUnits();
    list_sizes = uniformize_op.getListSizes();
    num_of_regions = uniformize_op.getNumRegions();
  } else if (auto equalize_op =
                 dyn_cast<uniform::EqualizePatternOp>(region_op)) {
    units = equalize_op.getUnits();
    list_sizes = equalize_op.getListSizes();
    num_of_regions = equalize_op.getNumRegions();
  } else {
    DT_ERROR("num_of_regions was not set");
  }
  if (region_idx == 0) {
    // make sure units are outside of loop_op
    for (int i = 0; i < units.size(); i++) {
      auto unit_op = units[i].getDefiningOp();
      if (loop_op->isProperAncestor(unit_op)) {
        if (isa<dataflow::CreateGroupOp, dataflow::GetUnitOp>(unit_op)) {
          auto new_op = builder->clone(*unit_op);
          units[i] = new_op->getResult(0);
        }
      }
    }
    // for first region, create new uniformizeRegions op.
    new_uniformize_op = uniform::UniformizeRegionsOp::create(
        *builder, region_op->getLoc(), result_types, units, list_sizes,
        /*regIndices*/ nullptr, /*regLocales*/ nullptr, num_of_regions);
    for (int i = 0; i < num_of_regions; i++) {
      auto& block = new_uniformize_op.getRegion(i).emplaceBlock();
      block.addArgument(builder->getIndexType(), region_op->getLoc());
      builder->setInsertionPointToStart(&block);
      SmallVector<Value, 1> operands = {{new_uniformize_op.getRegionArg(i)}};
      uniform::YieldOp::create(*builder, region_op->getLoc(), operands);
    }
    if (num_of_regions > 1) {
      new_uniformize_op->setAttr("active", builder->getI8IntegerAttr(1));
    }
  } else {
    // for the rest regions, find the newly created uniformize_regions op.
    // the assumption here is that regions of same op are processed back to
    // back.
    auto prev_node = loop_op->getPrevNode();
    while (!prev_node->hasAttr("active")) {
      prev_node = prev_node->getPrevNode();
    }
    new_uniformize_op = cast<uniform::UniformizeRegionsOp>(prev_node);
    if (region_idx == num_of_regions - 1) {
      prev_node->removeAttr("active");

}}}
// ---- 159/384  getUnitNameFromAListOfGetUnitOp  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:119  (10L)
static std::string getUnitNameFromAListOfGetUnitOp(
    std::vector<mlir::dataflow::GetUnitOp> &units) {
  DT_CHECK(units.size() > 0);
  std::string unit_name = units[0].getType().str();
  for (auto src_unit : units) {
    if (src_unit.getType().str() != unit_name) {
      units[0]->emitError("Src unit types has to be the same.");
      return "";
    }
  }

}
// ---- 160/384  areCoreletsDifferent  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:132  (6L)
static bool areCoreletsDifferent(
    OpBuilder builder, dataflow::GetUnitOp unit,
    std::vector<dataflow::GetUnitOp> units_with_corelet_0,
    std::vector<dataflow::GetUnitOp> units_with_corelet_1) {
  DT_CHECK(units_with_corelet_0.size() > 0 || units_with_corelet_1.size() > 0);
  return (unit->getAttr("corelet") == builder.getI32IntegerAttr(0) &&

}
// ---- 161/384  separateBasedOnDestinationUnits  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:761  (18L)
static void separateBasedOnDestinationUnits(
    mlir::OpBuilder builder, std::vector<mlir::Value> src_vs,
    std::vector<mlir::Value> dst_vs,
    std::vector<std::pair<mlir::Value, mlir::Value>> &src_dst_lx_corelet0,
    std::vector<std::pair<mlir::Value, mlir::Value>> &src_dst_lx_corelet1,
    std::vector<std::pair<mlir::Value, mlir::Value>> &src_dst_l3,
    std::vector<std::pair<mlir::Value, mlir::Value>> &src_dst_group) {
  for (int i = 0; i < dst_vs.size(); i++) {
    if (auto dst_unit =
            llvm::dyn_cast<dataflow::GetUnitOp>(dst_vs[i].getDefiningOp())) {
      if (dst_unit.getType().str().substr(0, 2) != "l3") {  // LX
        if (dst_unit->getAttr("corelet") == builder.getI32IntegerAttr(0)) {
          src_dst_lx_corelet0.push_back(std::make_pair(src_vs[i], dst_vs[i]));
        } else {  // corelet = 1
          src_dst_lx_corelet1.push_back(std::make_pair(src_vs[i], dst_vs[i]));
        }
      } else {  // L3
        src_dst_l3.push_back(std::make_pair(src_vs[i], dst_vs[i]));

}}}}
// ---- 162/384  insertIfNotExists  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:81  (8L)
bool e162_insertIfNotExists(Operation *op) {
  if (data_origins_.count(op) == 0) {
    int new_id = data_origins_.size();
    data_origins_[op] = {new_id, false};
    return true;
  }

  return false;

}
// ---- 163/384  getMaskValueConstantForNonPT  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:93  (40L)
std::optional<unsigned> e163_getMaskValueConstantForNonPT(
    Operation* op, Operation* operand, const SenSystemDef& sys_def) {
  if (!operand->hasAttr("mask_set")) {
    operand->emitOpError("Mask operation doesn't have mask_set attribute.");
    return std::nullopt;
  }
  auto mask_set_attr = operand->getAttr("mask_set");
  auto mask_set = cast<IntegerSetAttr>(mask_set_attr).getValue();
  if (mask_set.getNumDims() != 1) {
    operand->emitOpError("Mask affine set has to have one dimension.");
    return std::nullopt;
  }

  if (mask_set.getNumSymbols() > 0) {
    operand->emitOpError("Mask affine set should not have any symbols");
    return std::nullopt;
  }

  affine::FlatAffineValueConstraints mask_set_flat(mask_set);
  if (!hasConstantBounds(mask_set_flat)) {
    operand->emitOpError("Mask affine set has to have constant bounds.");
    return std::nullopt;
  }

  auto lowerbound_elm =
      int64_t(mask_set_flat.getConstantBound(mlir::presburger::BoundType::LB, 0)
                  .value());
  auto upperbound_elm =
      int64_t(mask_set_flat.getConstantBound(mlir::presburger::BoundType::UB, 0)
                  .value());
  DT_CHECK_MSG(isa<VectorType>(operand->getOpResult(0).getType()),
               "expecting operand to be VectorType");
  auto dim = cast<VectorType>(operand->getOpResult(0).getType()).getDimSize(0);

  DT_CHECK(dim / sys_def.numSlicesPerStick > 0);
  unsigned from_slice = lowerbound_elm / (dim / sys_def.numSlicesPerStick) + 1;
  unsigned to_slice = upperbound_elm / (dim / sys_def.numSlicesPerStick);
  unsigned mask_val = 0;
  for (unsigned i = from_slice - 1; i <= to_slice; i++) mask_val += 1 << i;


}
// ---- 164/384  checkValidityOfPackAndShuffleLowering  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:140  (45L)
mlir::LogicalResult e164_checkValidityOfPackAndShuffleLowering(
    Operation* op, std::vector<int>& indices, int repetition) {
  if (!is_any_of(repetition, 4, 8)) {
    op->emitError(
        "The cases with !is_any_of(repetition, 4, 8) are not supported yet");
    return failure();
  }

  auto pack_op = llvm::dyn_cast<vectorchain::PackOp>(op);
  auto shuffle_op = llvm::dyn_cast<vectorchain::ShuffleOp>(op);
  DT_CHECK_MSG((pack_op || shuffle_op), "Expect a pack or shuffle op.");
  auto op_indices = pack_op ? pack_op.getIndices().getValue()
                            : shuffle_op.getIndices().getValue();
  Type type = pack_op ? pack_op.getType() : shuffle_op.getType();
  int vec_size = dcc::utils::getTotalVectorSize(type);
  for (auto index_iter : op_indices) {
    if (auto index_attr = mlir::dyn_cast<IntegerAttr>(index_iter)) {
      int index = index_attr.getInt();
      if (index < -1) {
        op->emitError("Indices should be integers >= -1");
        return failure();
      }
      indices.push_back(index);
    } else {
      op->emitError("Indices should be integers");
      return failure();
    }
  }

  if (repetition * indices.size() != vec_size) {
    op->emitError(
        "Size of array indices multiply to repetition should be equal to "
        "the size of vector");
    return failure();
  }

  for (auto index : indices) {
    if (index > 2 * (int)indices.size()) {
      op->emitError(
          "None of the elements of the indices array can be greater than 2 "
          "times the size of the array");
      op->dump();
      return failure();
    }
  }

}
// ---- 165/384  getMergeTypeFromIndices  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:301  (109L)
std::string e165_getMergeTypeFromIndices(std::vector<int> indices,
                                                 int repetition,
                                                 bool sign_extend,
                                                 unsigned element_bit_width) {
  struct merge_and_pack_type {
    const std::string name;
    unsigned element_bit_width;
    std::vector<int> vec;
    bool sign_extend = false;
    int repetition = 8;
    int sum = 0;
    int size = 0;
    merge_and_pack_type(std::string name, unsigned element_bit_width,
                        std::vector<int> vec, bool sign_extend = false)
        : name(name),
          element_bit_width(element_bit_width),
          vec(vec),
          sign_extend(sign_extend),
          size(vec.size()) {
      sum = [](std::vector<int> v) {
        int sum = 0;
        for (auto x : v) sum += x;
        return sum;
      }(vec);
    }
  };

  std::vector<merge_and_pack_type> merge_and_pack_insts = {
      {"pack24", 2, {0,   8,   16,  24, 32, 40, 48, 56, 64, 72, 80, 88, 96,
                     104, 112, 120, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
                     -1,  -1,  -1,  -1, -1, -1, -1, -1, -1, -1, -1, -1}},
      {"pack8", 4, {0,  4,  8,  12, 16, 20, 24, 28, 32, 36, 40,
                    44, 48, 52, 56, 60, -1, -1, -1, -1, -1, -1,
                    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1}},
      {"pack9", 4, {0,  32, -1, -1, 4,  36, -1, -1, 8,  40, -1,
                    -1, 12, 44, -1, -1, 16, 48, -1, -1, 20, 52,
                    -1, -1, 24, 56, -1, -1, 28, 60, -1, -1}},
      {"merge8h",
       8,
       {1, 17, 3, 19, 5, 21, 7, 23, 9, 25, 11, 27, 13, 29, 15, 31}},
      {"merge8l",
       8,
       {0, 16, 2, 18, 4, 20, 6, 22, 8, 24, 10, 26, 12, 28, 14, 30}},
      {"pack12", 8, {0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1}},
      {"pack13",
       8,
       {8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1}},
      {"pack14",
       8,
       {0, -1, 1, -1, 2, -1, 3, -1, 4, -1, 5, -1, 6, -1, 7, -1},
       /*sign_extend=*/true},
      {"pack15",
       8,
       {8, -1, 9, -1, 10, -1, 11, -1, 12, -1, 13, -1, 14, -1, 15, -1},
       /*sign_extend=*/true},
      {"pack25",
       8,
       {0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30}},
      {"pack26",
       8,
       {0, 2, 4, 6, 16, 18, 20, 22, 8, 10, 12, 14, 24, 26, 28, 30}},
      {"pack27",
       8,
       {0, 2, 16, 18, 4, 6, 20, 22, 8, 10, 24, 26, 12, 14, 28, 30}},

      {"merge16h", 16, {1, 9, 3, 11, 5, 13, 7, 15}},
      {"merge16l", 16, {0, 8, 2, 10, 4, 12, 6, 14}},
      {"merge32h", 16, {2, 3, 10, 11, 6, 7, 14, 15}},
      {"merge32l", 16, {0, 1, 8, 9, 4, 5, 12, 13}},
      {"merge64h", 16, {4, 5, 6, 7, 12, 13, 14, 15}},
      {"merge64l", 16, {0, 1, 2, 3, 8, 9, 10, 11}},
      {"pack0", 16, {0, 1, 2, 3, 4, 5, 6, 7}},
      {"pack1", 16, {15, 1, 2, 3, 4, 5, 6, 7}},
      {"pack2", 16, {14, 15, 2, 3, 4, 5, 6, 7}},
      {"pack3", 16, {13, 14, 15, 3, 4, 5, 6, 7}},
      {"pack4", 16, {12, 13, 14, 15, 4, 5, 6, 7}},
      {"pack5", 16, {11, 12, 13, 14, 15, 5, 6, 7}},
      {"pack6", 16, {10, 11, 12, 13, 14, 15, 6, 7}},
      {"pack7", 16, {9, 10, 11, 12, 13, 14, 15, 7}},
      {"pack16", 16, {0, 1, 2, 3, 4, 5, 6, 7}},
      {"pack17", 16, {8, 0, 1, 2, 3, 4, 5, 6}},
      {"pack18", 16, {8, 9, 0, 1, 2, 3, 4, 5}},
      {"pack19", 16, {8, 9, 10, 0, 1, 2, 3, 4}},
      {"pack20", 16, {8, 9, 10, 11, 0, 1, 2, 3}},
      {"pack21", 16, {8, 9, 10, 11, 12, 0, 1, 2}},
      {"pack22", 16, {8, 9, 10, 11, 12, 13, 0, 1}},
      {"pack23", 16, {8, 9, 10, 11, 12, 13, 14, 0}}};

  int indices_sum = 0;
  for (auto index : indices) indices_sum += index;

  for (auto merge_and_pack_inst : merge_and_pack_insts) {
    if (element_bit_width > merge_and_pack_inst.element_bit_width) continue;
    int scale = merge_and_pack_inst.element_bit_width / element_bit_width;
    bool found = true;
    if (indices.size() != merge_and_pack_inst.size * scale) continue;
    if (repetition != merge_and_pack_inst.repetition) continue;
    if (scale == 1 && indices_sum != merge_and_pack_inst.sum) continue;

    for (int i = 0; i < merge_and_pack_inst.size && found; i++) {
      for (int j = 0; j < scale && found; j++) {
        if (indices[i * scale + j] != merge_and_pack_inst.vec[i] * scale + j)
          found = false;
      }
    }

    if (found && merge_and_pack_inst.sign_extend == sign_extend)

}}
// ---- 166/384  getOperandFromConstantOp  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:267  (26L)
std::optional<VectorOperand> e166_getOperandFromConstantOp(
    mlir::arith::ConstantOp &op) {
  std::string value;
  auto splat_attr = mlir::cast<SplatElementsAttr>(op.getValue());
  if (splat_attr) {
    double const_val;
    auto splat_value = splat_attr.getSplatValue<Attribute>();
    if (mlir::isa<IntegerAttr>(splat_value)) {
      const_val = mlir::cast<IntegerAttr>(splat_value).getInt();
    } else if (mlir::isa<FloatAttr>(splat_value)) {
      const_val = mlir::cast<FloatAttr>(splat_value).getValueAsDouble();
    } else {
      op->emitError("Only integer or float vectors are supported");
      return std::nullopt;
    }

    value = constValToField(const_val);
    if (value == "") {
      op->emitError("Only 0, 1, 2, or 3 are supported values");
      return std::nullopt;
    }
  } else {
    op->emitError("Only constant splatted vectors are supported");
    return std::nullopt;
  }


}
// ---- 167/384  getOperandFromConstantBitstreamOp  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:299  (5L)
std::optional<VectorOperand> e167_getOperandFromConstantBitstreamOp(
    vectorchain::ConstantBitstreamOp &op,
    bool is_constant_splatted_vector = false) {
  auto const_val = mlir::cast<IntegerAttr>(op.getValue()[0]).getInt();
  std::string value = is_constant_splatted_vector ? constValToField(const_val)

}
// ---- 168/384  getOperandFromNegOp  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:366  (5L)
std::optional<VectorOperand> e168_getOperandFromNegOp(
    const dcc::DccExtContext &dcc_ext_ctx, vectorchain::NegOp &op,
    const SenComponents comp) {
  DT_CHECK(isa<vectorchain::NegOp>(op));
  auto *parent = op.getOperand(0).getDefiningOp();

}
// ---- 169/384  getName  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:866  (11L)
std::string e169_getName() {
  if (this->type_ == LRF)
    return "lrf" + this->getFirstValue();
  else if (this->type_ == IRF)
    return "irf" + this->getFirstValue();
  else if (this->type_ == XRF)
    return "xrf";
  else if (this->type_ == ISTATE)
    return "istate" + this->getFirstValue();
  else
    return this->getFirstValue();

}
// ---- 170/384  getLayoutMapAndIndices  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:879  (40L)
LogicalResult e170_getLayoutMapAndIndices(
    Operation *op, AffineMap &layout_map, SmallVector<Value> &operands,
    dataflow::GetLogicalMemoryViewOp &logical_view_op) {
  if (auto tmp_op = dyn_cast<agen::VectorStoreOp>(op)) {
    auto indices = tmp_op.getMapOperands();
    operands = {indices.begin(), indices.end()};
    logical_view_op =
        tmp_op.getMemRef().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
    layout_map = logical_view_op.getLayoutMap();
    auto indices_map = tmp_op.getAffineMap();
    auto order_map = tmp_op.getStoreOrder();
    indices_map = order_map.compose(indices_map);
    layout_map = layout_map.compose(indices_map);
    layout_map = compressUnusedSymbols(layout_map);
  } else if (auto tmp_op = dyn_cast<vector::StoreOp>(op)) {
    auto indices = tmp_op.getIndices();
    operands = {indices.begin(), indices.end()};
    logical_view_op =
        tmp_op.getBase().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
    layout_map = logical_view_op.getLayoutMap();
  } else if (auto tmp_op = dyn_cast<agen::VectorLoadOp>(op)) {
    auto indices = tmp_op.getMapIndices();
    operands = {indices.begin(), indices.end()};
    logical_view_op =
        tmp_op.getMemRef().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
    layout_map = logical_view_op.getLayoutMap();
    auto indices_map = tmp_op.getAffineMap();
    auto order_map = tmp_op.getLoadOrder();
    indices_map = order_map.compose(indices_map);
    layout_map = layout_map.compose(indices_map);
    layout_map = compressUnusedSymbols(layout_map);
  } else if (auto tmp_op = dyn_cast<vector::LoadOp>(op)) {
    auto indices = tmp_op.getIndices();
    operands = {indices.begin(), indices.end()};
    logical_view_op =
        tmp_op.getBase().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
    layout_map = logical_view_op.getLayoutMap();
  } else {
    op->emitOpError("can't extract memory layout map or indices.");
    return failure();

}}
// ---- 171/384  isMaskEquivalentToNode  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:123  (4L)
bool e171_isMaskEquivalentToNode(MaskNode *n) {
  return (n->getParentNode() == getParentNode() &&
          n->getStartVal() == getStartVal() &&
          n->getIncrement() == getIncrement());

}
// ---- 172/384  areXrfAccessesLegal  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:98  (28L)
bool e172_areXrfAccessesLegal(LayoutExprMap expr_maps[]) {
  for (int i = 0; i < 2; i++) {
    auto expr_map = expr_maps[i];
    if (expr_map.size() <= 1) continue;
    std::vector<Operation *> keys;

    // create a key list
    for (auto pair : expr_map) {
      keys.push_back(pair.first);
    }

    for (int i = 0; i < keys.size() - 1; i++) {
      for (int j = i + 1; j < keys.size(); j++) {
        auto expr_a = expr_map[keys[i]];
        auto expr_b = expr_map[keys[j]];
        // only need to check union of expr_a and expr_b
        for (auto item : expr_a.layout_map) {
          // because there is only one reg for xrf read or write, all xrf
          // accesses have to have the same expr coeffients except constant expr
          if (item.first != nullptr && expr_b.layout_map.count(item.first) &&
              expr_b.layout_map[item.first] != item.second)
            return false;
        }
      }
    }
  }

  return true;

}
// ---- 173/384  insertConstAndAddOps  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:296  (10L)
Value e173_insertConstAndAddOps(OpBuilder *builder, Location &loc,
                                        Type &xrf_reg_type, Value xrf_ptr_val,
                                        int64_t val, std::string name) {
  if (val == 0) return xrf_ptr_val;

  auto const_offset =
      sentient::ConstantOp::create(*builder, loc, xrf_reg_type, val);

  Operation *add_op = sentient::AddOp::create(
      *builder, loc, xrf_reg_type, const_offset.getOut(), xrf_ptr_val);

}
// ---- 174/384  isXrfRelated  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:531  (31L)
bool e174_isXrfRelated(Operation *op) {
  dataflow::GetLogicalMemoryViewOp memory_view_op;
  if (auto load_op = dyn_cast<vector::LoadOp>(op)) {
    memory_view_op =
        load_op.getBase().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
  } else if (auto store_op = dyn_cast<vector::StoreOp>(op)) {
    memory_view_op =
        store_op.getBase().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
  } else if (auto load_op = dyn_cast<agen::VectorLoadOp>(op)) {
    memory_view_op =
        load_op.getMemRef().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
  } else if (auto store_op = dyn_cast<agen::VectorStoreOp>(op)) {
    memory_view_op =
        store_op.getMemRef().getDefiningOp<dataflow::GetLogicalMemoryViewOp>();
  } else if (auto tmp_op = dyn_cast<dataflow::GetLogicalMemoryViewOp>(op)) {
    memory_view_op = tmp_op;
  } else {
    op->emitError("unsupported in isXrfRelated()!");
    DT_ERROR("Could not determine if op is XRF-related");
  }
  std::string unit_name;
  std::optional<std::string> unit_str_optional =
      dcc::uniform::utils::findUnitType(memory_view_op.getFromUnit());
  if (unit_str_optional.has_value()) {
    unit_name = unit_str_optional.value();
  } else {
    memory_view_op.emitOpError("Unit type is inconsistent in memory view.");
    DT_ERROR("Could not determine if op is XRF-related");
  }
  if (unit_name.find("xrf") != std::string::npos) return true;
  return false;

}
// ---- 175/384  updateYieldArgs  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:690  (8L)
Value e175_updateYieldArgs(sentient::YieldOp &yield_op,
                                   Value &xrf_ptr_val, int idx) {
  SmallVector<Value, 2> yield_args;
  for (auto it : yield_op.getOperands()) {
    yield_args.push_back(it);
  }
  yield_args.push_back(xrf_ptr_val);
  yield_op->setOperands(yield_args);

}
// ---- 176/384  opHasSideEffect  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:38  (13L)
bool e176_opHasSideEffect(Operation &op) {
  bool result = false;
  op.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isOperationSelected(*op) &&
        !isa<arith::ConstantOp, arith::ConstantIndexOp, arith::ConstantIntOp,
             arith::CmpIOp, scf::YieldOp, scf::ForOp, affine::AffineForOp,
             affine::AffineYieldOp, mlir::symbol::CreateSymbolOp>(op)) {
      result = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return result;

}
// ---- 177/384  mergeShallow  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:398  (60L)
void e177_mergeShallow(Operation *src, Operation *dst,
                                               bool dst_before_src) {
  for (unsigned i = 0; i < 2; ++i) {
    bool src_region_empty = src->getRegions()[i].empty();
    bool dst_region_empty = dst->getRegions()[i].empty();
    // No need to merge regions if both are empty.
    if (src_region_empty && dst_region_empty) continue;

    // If the src/dst region is empty, create a dummy yield to help with the
    // merging.
    if (src_region_empty)
      createDummyYieldInElseReg(src);
    else if (dst_region_empty)
      createDummyYieldInElseReg(dst);

    Block &src_bb = src->getRegions()[i].front();
    Block &dst_bb = dst->getRegions()[i].front();
    if (dst_before_src) {
      // Move all of src's ops to the end of dst_bb.
      // If dst yields results, its new terminator will simply be its old one.
      // Otherwise the new terminator will be src's terminator.
      auto *dst_terminator = dst_bb.getTerminator();
      bool dst_has_no_results = (dst->getNumResults() == 0);
      if (dst_has_no_results)
        dst_terminator->erase();
      else {
        auto *src_terminator = src_bb.getTerminator();
        DT_CHECK_MSG(src_terminator->getNumResults() == 0,
                     "Expect src to not yield any results if dst does.");
        src_terminator->erase();
      }
      while (!src_bb.getOperations().empty()) {
        Operation &op = src_bb.getOperations().front();
        op.moveBefore(&dst_bb, dst_bb.end());
      }
      if (!dst_has_no_results)
        dst_terminator->moveBefore(&dst_bb, dst_bb.end());
    } else {
      // Move all of src's ops except its terminator to the start of dst_bb.
      // If src does not yield results, dst's new terminator will simply be its
      // old one. Otherwise the new terminator will be src's terminator.
      auto *src_terminator = src_bb.getTerminator();
      if (src->getNumResults() == 0)
        src_terminator->erase();
      else {
        auto *dst_terminator = dst_bb.getTerminator();
        DT_CHECK_MSG(dst_terminator->getNumResults() == 0,
                     "Expect dst to not yield any results if src does.");
        src_terminator->moveBefore(dst_terminator);
        dst_terminator->erase();
      }
      while (!src_bb.getOperations().empty()) {
        Operation &op = src_bb.getOperations().back();
        op.moveBefore(&dst_bb, dst_bb.begin());
      }
    }
  }
  if (StringAttr new_dbg_name_attr =
          dataflow::utils::getNewDbgNameFromList("CFGSM(", {src, dst})) {
    dataflow::setDbgNameAttr(dst, new_dbg_name_attr);

}}
// ---- 178/384  runOnOperation  —  dcc/src/Transform/Dataflow/CanonicalizeToggle.cpp:49  (42L)
  void runOnOperation() {
    if (DisableThisPass) return;

    // Collect the list of candidate units to analyze. Execution of the
    // canonicalizations will lead to operation deletion so we collect the
    // candidate units first. Only operations in the program unit operation
    // regions will change.
    ModuleOp module_op = getOperation();
    std::vector<dataflow::ProgramUnitOp> candidate_units;
    module_op.walk([&](dataflow::ProgramUnitOp unit_op) {
      auto comp = dcc::getUnitType(
          unit_op.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
      if (is_any_of(comp, SenComponents::L3LU, SenComponents::L3SU))
        candidate_units.push_back(unit_op);
    });

    // For each candidate unit, execute the pattern until we hit IR convergence.
    // The IR needs to be reanalyzed after every applied transformation as the
    // pattern application deletes operations.
    // TODO: Using the RewritePattern drivers leads to running all basic
    // canonicalizations (such as DCE) and will have unintended side effects if
    // run during the DFIR pipeline. There may be a better way than manually
    // calling the pattern on every candidate operation.
    for (auto unit : candidate_units) {
      bool has_changed = true;
      while (has_changed) {
        has_changed = false;
        unit.walk([&](Operation *op) {
          if (auto candidate = dyn_cast<dataflow::GetLogicalMemoryViewOp>(op)) {
            PatternRewriter rewriter(&getContext());
            rewriter.setInsertionPoint(op);

            dcc::dataflow::DuplicateReusedTogglePattern pattern(&getContext());
            if (succeeded(pattern.matchAndRewrite(candidate, rewriter))) {
              has_changed = true;
              return WalkResult::interrupt();
            }
          }
          return WalkResult::advance();
        });
      }
    }

}
// ---- 179/384  clear  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:112  (15L)
void e179_clear() {
  if (!empty()) {
    SmallVector<LocalOpNode *> to_be_deleted;
    LocalOpNode::walk<OperationNode::WalkOrder::kPostOrder>(
        const_cast<LocalOpNode *>(getRoot()),
        [&](LocalOpNode *n) -> LocalOpNode * {
          to_be_deleted.push_back(n);
          return nullptr;
        });
    for (LocalOpNode *n : to_be_deleted) delete n;
  } else if (root_)
    delete root_;
  root_ = nullptr;

  unit_to_ops.clear();

}
// ---- 180/384  traverseRegion  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:129  (17L)
void e180_traverseRegion(mlir::Region &region,
                                                LocalOpNode *parent_node,
                                                std::vector<mlir::Value> &units,
                                                int is_in_region_num) {
  for (auto &op : region.getOps()) {
    auto new_node = new LocalOpNode(&op);
    new_node->is_in_region_num = is_in_region_num;
    parent_node->insertChildNode(new_node);
    if (isa<uniform::UniformizeRegionsOp>(op)) {
      compute(new_node);
    } else {
      for (auto u : units) {
        new_node->units.push_back(u);
        unit_to_ops[u].push_back(&op);
      }
      for (int region_num = 0; region_num < op.getNumRegions(); region_num++) {
        traverseRegion(op.getRegion(region_num), new_node, units, region_num);

}}}}
// ---- 181/384  inRegionEmpty  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:214  (6L)
bool e181_inRegionEmpty(int region_num,
                                               LocalOpNode *node) {
  while (node) {
    if (node->is_in_region_num == region_num) return false;
    node = node->getNextSibling();
  }

}
// ---- 182/384  cloneOpsForRegions  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:223  (60L)
void e182_cloneOpsForRegions(
    LocalOpNode *node, const std::vector<mlir::Operation *> &op_list,
    mlir::OpBuilder builder_region, int region_num, IRMapping &arg_map,
    mlir::BlockArgument &block_arg) {
  std::map<mlir::Operation *, mlir::Operation *> old_to_new_op_map;
  while (node) {
    if (isa<mlir::uniform::YieldOp>(node->getOperation())) {
      node = node->getNextSibling();
      continue;
    }
    if (std::find(op_list.begin(), op_list.end(), node->getOperation()) ==
        op_list.end()) {
      if (isa<uniform::UniformizeRegionsOp>(node->getOperation())) {
        cloneOpsForRegions(node->getFirstChild(), op_list, builder_region,
                           region_num, arg_map, block_arg);
      }
      node = node->getNextSibling();
      continue;
    }
    if (node->is_in_region_num != region_num) {
      node = node->getNextSibling();
      continue;
    }
    if (auto parent_uniform_op = llvm::dyn_cast<uniform::UniformizeRegionsOp>(
            node->getOperation()->getParentOp())) {
      for (int region_idx = 0; region_idx < parent_uniform_op.getNumRegions();
           region_idx++) {
        if (&parent_uniform_op.getRegion(region_idx) ==
            node->getOperation()->getParentRegion()) {
          arg_map.map(parent_uniform_op.getRegion(region_idx).getArgument(0),
                      block_arg);
        }
      }
    }
    auto new_op_ =
        builder_region.cloneWithoutRegions(*node->getOperation(), arg_map);
    old_to_new_op_map[node->getOperation()] = new_op_;
    if (new_op_->getNumRegions() > 0) {
      DT_CHECK(new_op_->getNumRegions() ==
               node->getOperation()->getNumRegions());
      for (int rn = 0; rn < new_op_->getNumRegions(); rn++) {
        OpBuilder builder_inner_region(new_op_->getRegion(rn));
        auto &block = new_op_->getRegion(rn).emplaceBlock();
        builder_inner_region.setInsertionPointToStart(&block);
        cloneOpsForRegions(node->getFirstChild(), op_list, builder_inner_region,
                           rn, arg_map, block_arg);
        if (block.empty()) block.erase();
      }
    }
    node = node->getNextSibling();
  }
  for (auto old_new_op : old_to_new_op_map) {
    auto parent_region = old_new_op.second->getParentRegion();
    old_new_op.first->replaceUsesWithIf(
        old_new_op.second, [&](mlir::OpOperand &use) {
          auto curr_region = use.getOwner()->getParentRegion();
          while (!isa<dataflow::ProgramUnitOp>(curr_region->getParentOp())) {
            if (curr_region == parent_region) return true;
            curr_region = curr_region->getParentRegion();
          }

}}}
// ---- 183/384  expandAffineApplyOps  —  dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:183  (53L)
void e183_expandAffineApplyOps(
    dataflow::ProgramUnitOp unit) {
  // Collect all affine.apply operations in the unit
  SmallVector<affine::AffineApplyOp> apply_ops;
  unit.walk(
      [&](affine::AffineApplyOp apply_op) { apply_ops.push_back(apply_op); });

  // Expand each affine.apply operation to arithmetic operations
  for (auto apply_op : apply_ops) {
    OpBuilder builder(apply_op);

    // Use MLIR's expandAffineMap utility to convert affine.apply to arithmetic
    // ops
    auto expanded_result = mlir::affine::expandAffineMap(
        builder, apply_op.getLoc(), apply_op.getAffineMap(),
        llvm::to_vector<8>(apply_op.getOperands()));

    if (!expanded_result.has_value() || expanded_result->empty()) {
      apply_op.emitError("Failed to expand affine.apply operation");
      signalPassFailure();
      return;
    }

    Value expanded_value = (*expanded_result)[0];

    // Try to fold the expanded value to ensure it resolves to a constant
    // This is important because after unrolling, all affine expressions should
    // be foldable to constants
    if (auto defining_op = expanded_value.getDefiningOp()) {
      SmallVector<OpFoldResult> fold_results;
      if (succeeded(defining_op->fold(fold_results)) && !fold_results.empty()) {
        // If folding succeeded and produced a constant attribute, create a
        // constant op
        if (auto const_attr = fold_results[0].dyn_cast<Attribute>()) {
          Value const_op = arith::ConstantIndexOp::create(
              builder, defining_op->getLoc(),
              cast<IntegerAttr>(const_attr).getInt());
          expanded_value = const_op;
        }
      }
    }

    // Validate that the final expanded value is a constant or query-based value
    if (!dcc::utils::isConstant<arith::ConstantOp>(expanded_value)) {
      apply_op.emitOpError(
          "Expanded affine.apply operation does not resolve to a constant");
      signalPassFailure();
      return;
    }

    // Replace the affine.apply with the expanded (and possibly folded) value
    apply_op.getResult().replaceAllUsesWith(expanded_value);
    apply_op.erase();

}}
// ---- 184/384  getLoopTripCount  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:743  (56L)
int64_t e184_getLoopTripCount(Operation *op) {
  DT_CHECK(op);
  int step = 1;
  int64_t lb = 0, ub = 0;
  if (auto affine_for = dyn_cast<affine::AffineForOp>(op)) {
    // Calculate step
    step = affine_for.getStepAsInt();

    // Calculate lower and upper bounds
    if (!affine_for.hasConstantBounds()) return -1;

    lb = affine_for.getConstantLowerBound();
    ub = affine_for.getConstantUpperBound();
  } else if (auto scf_for = dyn_cast<scf::ForOp>(op)) {
    // Calculate step
    auto const_step = scf_for.getConstantStep();
    if (!const_step.has_value()) return -1;

    step = const_step.value().getSExtValue();

    // Calculate lower and upper bounds
    auto lb_op = dyn_cast_or_null<arith::ConstantOp>(
        scf_for.getLowerBound().getDefiningOp());
    DT_CHECK(lb_op);
    lb = cast<IntegerAttr>(lb_op.getValue()).getInt();

    auto ub_op = scf_for.getUpperBound().getDefiningOp();
    DT_CHECK(ub_op);
    if (auto ub_const_op = dyn_cast<arith::ConstantOp>(ub_op)) {
      ub = cast<IntegerAttr>(ub_const_op.getValue()).getInt();
    } else if (auto ub_sym_op = dyn_cast<symbol::CreateSymbolOp>(ub_op)) {
      if (ub_sym_op->hasAttr("maxValue"))
        ub = cast<IntegerAttr>(ub_sym_op->getAttr("maxValue")).getInt();
      else
        return false;
    } else if (auto select_ub_op = dyn_cast<arith::SelectOp>(ub_op)) {
      auto true_op = dyn_cast_or_null<arith::ConstantOp>(
          select_ub_op.getTrueValue().getDefiningOp());
      auto false_op = dyn_cast_or_null<arith::ConstantOp>(
          select_ub_op.getFalseValue().getDefiningOp());
      DT_CHECK_MSG(
          true_op && false_op,
          "Expecting SelectOp upper bound to contain constant values.");
      auto true_val = cast<IntegerAttr>(true_op.getValue()).getInt();
      auto false_val = cast<IntegerAttr>(false_op.getValue()).getInt();
      ub = true_val > false_val ? true_val : false_val;
    } else {
      llvm_unreachable("unsupported upper loop bound operation");
    }
  } else {
    llvm_unreachable("Unsupported operation.");
  }
  DT_CHECK_MSG(lb == 0 || step == 1,
               "Expecting a lower bound of 0 and a step of 1 for the loop.");

  return (ub - lb) / step;

}
// ---- 185/384  hasMutableAddrOverflow  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:801  (14L)
bool e185_hasMutableAddrOverflow(
    const SmallVectorImpl<MASData> &mas_data, const SenComponents comp,
    const int64_t max_mutable, const int elem_size_in_bits) {
  if (max_mutable > (getMaxMutableRange(comp) / elem_size_in_bits)) {
    // Currently preventing transformation of any EAR overflow cases and
    // addressing them upstream unless manual override specified.
    if (!dcc_ext_ctx_.dsc_global_->dcc_correct_ear_overflow)
      DT_CHECK_MSG(false, "EAR overflow detected");

    DT_CHECK_MSG(
        !mas_data.empty(),
        "Mutable address overflow detected, no loops involved in address "
        "calculation - cannot split data transfer.");
    return true;

}}
// ---- 186/384  calculatePartitionSizes  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:862  (97L)
void e186_calculatePartitionSizes(
    ExpressionEvaluator &evaluator, SmallVectorImpl<MASData> &mas_data,
    SmallVectorImpl<int64_t> &partition_sizes, const SenComponents comp,
    dataflow::GetLogicalMemoryViewOp mem_view_op, const int64_t &max_mutable,
    const int elem_size_in_bits) {
  // Note: This is not optimal at all but a more optimal solution requires a
  // much more complex algorithm or an enumeration method. Use of external
  // math solvers could help.

  // For each dim, check if the weight of the dim is enough to bring the mutable
  // back in range. If it is, calculate the split. If it is not, fully split the
  // current dim and go to the next one.
  int64_t max_mutable_range = getMaxMutableRange(comp);
  int64_t mutable_overflow =
      max_mutable - (max_mutable_range / elem_size_in_bits);

  int64_t shifted_mutable = 0;
  int64_t total_partitions = 1;
  for (int i = 0, e = mas_data.size(); i < e; ++i) {
    // The following code is responsible for two calculations:
    // 1. The calculation of how many partitions we need for this dim.
    // 2. The calculation of how much shifting this dim impacts the immutable
    // (shifted_mutable).
    //
    // When splitting a dim into x partitions, (x-1) partitions will need
    // adjustment. Each adjusted partition will shift the mutable over by
    // <partition size> * <coeff>. For each dim, the amount of immutable shifted
    // is:
    //   (<num partitions> - 1) * <partition size> * <coeff>
    // The sum of these calculations for each dim is the total amount shifted to
    // the immutable.
    if (mas_data[i].weight_ <= mutable_overflow) {
      // This dim doesn't have enough weight to fully resolve the required
      // shift. Set partition size 1 for this dim and continue splitting.
      partition_sizes.push_back(1);
      mutable_overflow -= mas_data[i].weight_;
      // Entire weight is shifted.
      shifted_mutable += mas_data[i].weight_;
      // Each iteration is a partition.
      total_partitions *= mas_data[i].num_iters_;
    } else {
      // This dim has enough weight to fully resolve the required shift.
      // Calculate how many partitions are required and then stop splitting.
      int64_t extra_iters = mutable_overflow / mas_data[i].composed_coeff_;
      if (mutable_overflow % mas_data[i].composed_coeff_ != 0) extra_iters += 1;
      partition_sizes.push_back(mas_data[i].num_iters_ - extra_iters);
      mutable_overflow -= extra_iters * mas_data[i].composed_coeff_;
      int64_t num_partitions =
          mas_data[i].num_iters_ % partition_sizes.back() == 0
              ? mas_data[i].num_iters_ / partition_sizes.back()
              : mas_data[i].num_iters_ / partition_sizes.back() + 1;
      shifted_mutable += (num_partitions - 1) * partition_sizes.back() *
                         mas_data[i].composed_coeff_;
      total_partitions *= num_partitions;
      break;
    }
  }

  DT_CHECK_MSG(mutable_overflow <= 0,
               "Cannot split enough to bring mutable address back in range.");
  DT_CHECK_MSG(!partition_sizes.empty(), "No partitions were identified.");

  auto max_immutable =
      dcc::agen::utils::getMaxImmutableAddress(evaluator, mem_view_op);
  auto immutable_space =
      (getMaxImmutableRange(comp) / elem_size_in_bits) - max_immutable;
  DT_CHECK_MSG(shifted_mutable <= immutable_space,
               "Shifting the required mutable would exceed the maximum "
               "immutable range.");

  // For sen1p5, immutable addresses need to be in an even number of sticks
  // only. If the immutable would become an odd number of sticks as result of
  // splitting, there needs to be enough mutable space remaining to shift a
  // stick back when filling the partitions.
  if (dcc_ext_ctx_.getArch() >= IsaCoreGen::SEN1P5_ISA) {
    int64_t num_elems_in_stick =
        dcc_ext_ctx_.getBytesPerStick() * 8 / elem_size_in_bits;
    bool is_even = dcc::agen::utils::isL3ImmutableAddrEven(
        evaluator, mem_view_op.getStartAddress(), num_elems_in_stick);
    bool is_shift_even = shifted_mutable % 2 == 0;
    if (is_even != is_shift_even) {
      DT_CHECK_MSG(is_even ? true
                           : dcc::agen::utils::isL3ImmutableAddrAllOdd(
                                 evaluator, mem_view_op.getStartAddress(),
                                 num_elems_in_stick),
                   "All immutable addrs must be all even or all odd to execute "
                   "even shift.");
      DT_CHECK_MSG(immutable_space >= num_elems_in_stick,
                   "No mutable space to shift back one stick to maintain even "
                   "immutable address.");
    }
  }

  // The number of conditionals required will be one less than the total number
  // of partitions. A MaxNumConditionals value of -1 indicates no maximum.
  int req_conditionals = total_partitions - 1;
  num_conditionals_ += req_conditionals;

}
// ---- 187/384  constructConditionals  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1000  (59L)
scf::IfOp e187_constructConditionals(
    const SmallVectorImpl<MASData> &mas_data,
    const SmallVectorImpl<int64_t> &partition_sizes, Operation *op) {
  SmallVector<scf::IfOp, 32> previous_leaves;
  scf::IfOp root_op = nullptr;
  OpBuilder cond_builder(op);

  auto createConditionalsForPartition =
      [&](int partition_dim) -> SmallVector<scf::IfOp, 32> {
    SmallVector<scf::IfOp, 32> if_ops;
    int64_t ub = partition_sizes[partition_dim];
    while (ub < mas_data[partition_dim].num_iters_) {
      // Create the cmpi operation for this partition.
      auto ub_const =
          arith::ConstantIndexOp::create(cond_builder, op->getLoc(), ub);
      DT_CHECK(mas_data[partition_dim].iter_arg_ != nullptr);
      auto ub_cond = arith::CmpIOp::create(
          cond_builder, op->getLoc(), arith::CmpIPredicate::slt,
          mas_data[partition_dim].iter_arg_, ub_const);

      // Create the if operation for this partition.
      if_ops.emplace_back(scf::IfOp::create(cond_builder, op->getLoc(),
                                            ub_cond.getResult(), true));

      // Set the builder to the else block of the new ifOp.
      cond_builder = if_ops.back().getElseBodyBuilder();

      // Set the ub for the next iteration.
      ub += partition_sizes[partition_dim];
    }
    return if_ops;
  };

  // Create the partitions for the first dim.
  auto prev_partitions = createConditionalsForPartition(0);

  // Store the root operation for the conditional tree. It will be used to
  // generate a ConditionalTree later to insert the new operations for each
  // partition.
  auto root_if_op = prev_partitions.front();

  // Create the partitions for the remaining dims in each previous partition.
  for (int d = 1, e = partition_sizes.size(); d < e; ++d) {
    SmallVector<scf::IfOp, 32> curr_partitions;
    for (int p = 0, num_prev_partitions = prev_partitions.size();
         p < num_prev_partitions; ++p) {
      auto curr_partition = prev_partitions[p];
      cond_builder = curr_partition.getThenBodyBuilder();
      curr_partitions = createConditionalsForPartition(d);
    }
    // The last partition requires conditionals in both the then and else paths
    // to cover every combination.
    cond_builder = prev_partitions.back().getElseBodyBuilder();
    auto last_partition = createConditionalsForPartition(d);
    // Reset previous partitions to the partitions just created to prepare for
    // next dim.
    prev_partitions = curr_partitions;
    for (auto &o : last_partition) prev_partitions.push_back(o);
  }

}
// ---- 188/384  calculateSubscriptsCoefficients  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1188  (11L)
MutableAddrSplittingPass::calculateSubscriptsCoefficients(
    const SmallVectorImpl<MASData> &mas_data,
    const SmallVectorImpl<int64_t> &partition_sizes,
    const AffineMap &subscripts_map) {
  SmallVector<SmallVector<int64_t>> subscripts_coeffs;
  for (int p = 0, e = partition_sizes.size(); p < e; ++p) {
    SmallVector<int64_t> dim_coeffs =
        dcc::agen::utils::extractConstantOffsetsFromMapForDim(subscripts_map,
                                                              mas_data[p].dim_);
    // Add the vector to the map.
    subscripts_coeffs.push_back(dim_coeffs);

}}
// ---- 189/384  synthesizeTimeInfo  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1256  (22L)
void e189_synthesizeTimeInfo(
    SmallVectorImpl<MASData> &mas_data, agen::AccessDetailsAffineComposite &ad,
    int64_t &max_mutable) {
  auto layout_map = ad.getMemViewLayoutMap();
  auto time_offsets = ad.getTimeOffsets();
  auto time_bounds = ad.getTimeBounds();
  DT_CHECK(time_offsets.size() >= time_bounds.size());

  // Note: time_offsets may contain one entry more than time_bounds to
  //       represent the constant coefficient. However, that constant
  //       coefficient isn't used for anything currently so it is
  //       ignored.
  int num_non_time_dims = ad.getIndices().size();
  for (int i = 0, e = time_bounds.size(); i < e; ++i) {
    // The iter_arg will be the time_bounds - 1 at maximum and the
    // last iteration of every loop can overflow the mutable as no data transfer
    // will occur after it. So really, the time_bounds - 2 is the last
    // utilized mutable address.
    int64_t weight =
        time_bounds[i] < 2 ? 0 : (time_bounds[i] - 2) * time_offsets[i];
    mas_data.emplace_back(num_non_time_dims + i, time_offsets[i],
                          time_bounds[i], weight);

}}
// ---- 190/384  createExplicitTimeLoops  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1282  (57L)
void e190_createExplicitTimeLoops(
    SmallVectorImpl<MASData> &mas_data,
    const SmallVectorImpl<int64_t> &partition_sizes, Operation *op,
    agen::AccessDetailsAffineComposite &ad, AffineMap &subscripts_map,
    SmallVector<Value> &indices, IntegerSet &time_set) {
  // If any of the dims to split are time dims, the time loops from the
  // outermost time dim being split to the innermost time dim need to be
  // explicity created.
  //
  // MASData objects are sorted by weight and the partition_sizes entries
  // correspond to the equivalent entry in mas_data. Find the MASData object
  // without a set iter_arg with the highest dim value.
  int innermost_time_dim = -1;
  for (int i = 0, e = partition_sizes.size(); i < e; ++i) {
    if (mas_data[i].iter_arg_ != nullptr) continue;
    if (mas_data[i].dim_ > innermost_time_dim)
      innermost_time_dim = mas_data[i].dim_;
  }

  if (innermost_time_dim < 0) return;

  // Store the number of non-time dims so the appropriate for op can be
  // referenced later when updating mas_data_.
  int num_non_time_dims = indices.size();

  // Create explicit loops, update the subscripts map and the indices.
  int time_dim_idx = innermost_time_dim - num_non_time_dims;
  auto time_bounds = ad.getTimeBounds();
  auto for_ops = dcc::agen::utils::constructExplicitTimeLoops(op, time_bounds,
                                                              time_dim_idx);

  auto time_addr_map = ad.getTimeAddrMap();
  AffineMap subscripts_map_time =
      agen::utils::concatenateMaps(subscripts_map, time_addr_map);

  auto time_order = ad.getTimeOrder();
  dcc::agen::utils::updateTimeSetForExplicitDims(time_dim_idx, time_order,
                                                 time_set);
  dcc::agen::utils::updateSubscriptsAndIndicesForExplicitTimeLoops(
      for_ops, time_dim_idx, subscripts_map, subscripts_map_time, indices);

  // Move the operation into the innermost loop. When partitioning, the
  // maps and sets altered in this function will be used as a base for
  // further calculations.
  auto for_body = cast<affine::AffineForOp>(for_ops.back()).getBody();
  op->moveBefore(for_body, for_body->begin());

  // Now that some of the time loops are explicit, the associated mas_data
  // entries should be updated to include the explicit loop iv.
  // Note: Time loops are created outermost to innermost, preserving the
  //       innermost time loops as much as possible. Some time loops may
  //       have been created that weren't in the list of dims to split
  //       if they were inner to the outermost split time dim.
  for (int i = 0, e = mas_data.size(); i < e; ++i) {
    if (mas_data[i].iter_arg_ != nullptr) continue;
    int actual_time_dim = mas_data[i].dim_ - num_non_time_dims;
    if (actual_time_dim > time_dim_idx) continue;

}}
// ---- 191/384  calculateFullShift  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:462  (25L)
int64_t e191_calculateFullShift(
    SmallVectorImpl<int64_t> &shifts, agen::AccessDetailsAffine &ad) {
  DT_CHECK(shifts.empty());
  auto layout_coeffs = ad.getLayoutCoeffs();

  auto ordered_subscripts_map =
      ad.getTransferOrder().compose(ad.getSubscriptsMap());
  int num_dims = ordered_subscripts_map.getNumDims();
  int64_t total_shift = 0;
  for (int i = 0, num_res = ordered_subscripts_map.getNumResults(); i < num_res;
       ++i) {
    AffineExpr expr = ordered_subscripts_map.getResult(i);
    SmallVector<int64_t> coeffs;
    affine::FlatAffineValueConstraints constraints;
    auto flat_result =
        getFlattenedAffineExpr(expr, num_dims, 0, &coeffs, &constraints);
    int constant_offset = coeffs.size() != num_dims ? coeffs.back() : 0;
    shifts.push_back(constant_offset);
    total_shift += constant_offset * layout_coeffs[i];
  }
  // Ensure shifts are completed in terms of sticks. The mutable address start
  // should already be in terms of sticks.
  int num_elems_in_stick =
      dcc_ext_ctx_.getBytesPerStick() * 8 / ad.getElementWidth();
  DT_CHECK(total_shift % num_elems_in_stick == 0);

}
// ---- 192/384  calculateDimWeights  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:560  (26L)
void e192_calculateDimWeights(
    SmallVectorImpl<DimWeight> &dim_weights,
    agen::AccessDetailsAffine &ad) {
  DT_CHECK(dim_weights.empty());
  auto layout_coeffs = ad.getLayoutCoeffs();

  auto ordered_subscripts_map =
      ad.getTransferOrder().compose(ad.getSubscriptsMap());
  SmallVector<int64_t> ordered_constant_offsets;
  int num_dims = ordered_subscripts_map.getNumDims();
  DT_CHECK(layout_coeffs.size() > num_dims);
  for (int i = 0, num_res = ordered_subscripts_map.getNumResults(); i < num_res;
       ++i) {
    AffineExpr expr = ordered_subscripts_map.getResult(i);
    SmallVector<int64_t> coeffs;
    affine::FlatAffineValueConstraints constraints;
    auto flat_result =
        getFlattenedAffineExpr(expr, num_dims, 0, &coeffs, &constraints);
    DT_CHECK(!coeffs.empty());
    int constant_offset = coeffs.size() != num_dims ? coeffs.back() : 0;
    ordered_constant_offsets.push_back(constant_offset);
    dim_weights.emplace_back(i, constant_offset,
                             layout_coeffs[i] * constant_offset);
  }

  llvm::sort(dim_weights, [](DimWeight &a, DimWeight &b) -> bool {

}}
// ---- 193/384  matchUnits  —  dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:69  (77L)
LogicalResult e193_matchUnits(
    ReducibleProgramUnits &group, ProgramUnitOp &curr_unit) {
  auto base_get_unit =
      group.base_unit_program_.getUnits()[0].getDefiningOp<GetUnitOp>();
  auto curr_get_unit = curr_unit.getUnits()[0].getDefiningOp<GetUnitOp>();
  int base_unit_core_id = sentient::getCoreOrCoreletID(base_get_unit, "core");
  int base_unit_corelet_id =
      sentient::getCoreOrCoreletID(base_get_unit, "corelet");
  int curr_unit_core_id = sentient::getCoreOrCoreletID(curr_get_unit, "core");
  int curr_unit_corelet_id =
      sentient::getCoreOrCoreletID(curr_get_unit, "corelet");

  // Check if units have same type (e.g., ptrow0)
  if (base_get_unit.getType() != curr_get_unit.getType())
    return LogicalResult::failure();

  struct unit_matching_info {
    unit_matching_info(int base_unit_core_id, int base_unit_corelet_id,
                       int curr_unit_core_id, int curr_unit_corelet_id)
        : base_unit_core_id_(base_unit_core_id),
          base_unit_corelet_id_(base_unit_corelet_id),
          curr_unit_core_id_(curr_unit_core_id),
          curr_unit_corelet_id_(curr_unit_corelet_id) {}

    int base_unit_core_id_;
    int base_unit_corelet_id_;
    int curr_unit_core_id_;
    int curr_unit_corelet_id_;
  };

  unit_matching_info *info =
      new unit_matching_info(base_unit_core_id, base_unit_corelet_id,
                             curr_unit_core_id, curr_unit_corelet_id);

  dcc::OperationEquivalence equivalence_analysis(

      // The first argument is lambda functor that acts as a high preference
      // in matching two operations. In our case of matching, we need
      // to repect the use of core_id, corelet_id across the program, i.e.,
      // If base_unit uses its default core_id in statement S, then the
      // curr_unit should use its core_id in the statement S. It makes
      // the core_id and corelet_id as parametric and keeps the semantics
      // after merging the curr_unit in the group.
      [](Operation &op_a, Operation &op_b, void *context) -> bool {
        unit_matching_info *info = (unit_matching_info *)context;
        if (isa<GetUnitOp>(op_a) && isa<GetUnitOp>(op_b)) {
          auto operation_a = llvm::dyn_cast<GetUnitOp>(op_a);
          auto operation_b = llvm::dyn_cast<GetUnitOp>(op_b);
          int a_core_id = sentient::getCoreOrCoreletID(operation_a, "core");
          int a_corelet_id =
              sentient::getCoreOrCoreletID(operation_a, "corelet");
          if (a_core_id == info->base_unit_core_id_ &&
              a_corelet_id == info->base_unit_corelet_id_) {
            int b_core_id = sentient::getCoreOrCoreletID(operation_b, "core");
            int b_corelet_id =
                sentient::getCoreOrCoreletID(operation_b, "corelet");
            if (b_core_id == info->curr_unit_core_id_ &&
                b_corelet_id == info->curr_unit_corelet_id_) {
              if (operation_a.getType() == operation_b.getType()) {
                return true;
              }
            }
          }
        }

        return false;
      },
      (void *)info, "program-units-reduction");

  // Check if regions of base unit and curr unit are same
  if (!equivalence_analysis.regionsAreEquivalent(
          group.base_unit_program_.getRegion(), curr_unit.getRegion())) {
    delete info;
    return LogicalResult::failure();
  }

  delete info;

}
// ---- 194/384  analyzeLoop  —  dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:268  (127L)
TransformLoopToLegalizeForSentientLowering::analyzeLoop(
    LoopLikeOpInterface loop) {
  Value induc_var = nullptr;
  bool constant_bounds = false;
  bool affine_non_const_maps = true;
  if (auto affine_for =
          llvm::dyn_cast<affine::AffineForOp>(loop.getOperation())) {
    induc_var = affine_for.getInductionVar();
    constant_bounds = affine_for.hasConstantBounds();
    affine_non_const_maps = !affine_for.hasConstantBounds();
    // TODO: Sometimes, loop bounds are constant SSA variables rather than
    // constant maps. Hence, we may have to check about them.
    // this leads to different values of affine_const_maps and constant_bounds
  } else if (auto scf_for = llvm::dyn_cast<scf::ForOp>(loop.getOperation())) {
    // if a bound is defined for uniformation purpose, don't convert
    if (dyn_cast<uniform::QueryMapOp>(
            scf_for.getLowerBound().getDefiningOp()) ||
        dyn_cast<uniform::QueryMapOp>(
            scf_for.getUpperBound().getDefiningOp())) {
      return TransformLoopToLegalizeForSentientLowering::kNone;
    }
    affine_non_const_maps = false;
    induc_var = scf_for.getInductionVar();
    auto lb_const =
        isa<arith::ConstantIndexOp>(scf_for.getLowerBound().getDefiningOp());
    bool ub_const =
        isa<arith::ConstantIndexOp>(scf_for.getUpperBound().getDefiningOp());
    auto step_const = scf_for.getStep().getDefiningOp<arith::ConstantIndexOp>();
    DT_CHECK_MSG(step_const.value() == 1,
                 "We expect the SCF-for loop to have step size of 1");

    // Allow for symbolic upper bounds in case of load/store units since
    // AgenToSentient is already enhanced to support scf.for loops natively.
    if (!ub_const &&
        is_any_of(curr_unit_, L3LU, L3SU, LXLU, LXSU, L0LU, L0SU)) {
      if (isa<mlir::symbol::CreateSymbolOp>(
              scf_for.getUpperBound().getDefiningOp())) {
        return TransformLoopToLegalizeForSentientLowering::kNone;
      }
    }

    constant_bounds = lb_const && ub_const && step_const;
  } else if (auto sentient_for =
                 llvm::dyn_cast<sentient::ForOp>(loop.getOperation())) {
    affine_non_const_maps = false;
    return TransformLoopToLegalizeForSentientLowering::kNone;
  } else {
    llvm_unreachable("Unknown loop operation");
  }

  // if loop has constant bounds in load units or store units --> don't perform
  // conversion
  // TODO: may be in future, we want to translate scf.for to affine.for
  // for constant loops because we want subscripts to be coming from
  // affine loops.
  if (constant_bounds &&
      is_any_of(curr_unit_, L3LU, L3SU, LXLU, LXSU, L0LU, L0SU)) {
    return TransformLoopToLegalizeForSentientLowering::kNone;
  }

  for (auto* use : induc_var.getUsers()) {
    bool is_memory_op =
        isa<agen::VectorLoadOp, agen::VectorStoreOp, agen::CompositeLoadOp,
            agen::CompositeStoreOp, agen::CompositeLoadAndStoreOp>(use);

    // if it's not a memory operation, --> don't perform conversion
    if (!is_memory_op) {
      continue;
    }

    // if it's not constant, then unrolling doesn't work.
    // we currently support bound being part of conditional on the parent loop.
    if (!constant_bounds && !affine_non_const_maps) {
      // TODO: enhance condition to do splitting only for PT.
      if (!is_any_of(curr_unit_, L3LU, L3SU))
        return TransformLoopToLegalizeForSentientLowering::KSplitParent;
    }

    // in the constant bounds and in PT LRF's --> perform unrolling
    if (is_any_of(curr_unit_, PT, SFP, PE)) {
      Value memref = nullptr;
      if (auto agen_load = llvm::dyn_cast<agen::VectorLoadOp>(use)) {
        memref = agen_load.getMemRef();
      } else if (auto agen_store = llvm::dyn_cast<agen::VectorStoreOp>(use)) {
        memref = agen_store.getMemRef();
      } else if (auto vec_load =
                     llvm::dyn_cast<affine::AffineVectorLoadOp>(use)) {
        memref = vec_load.getMemref();
      } else if (auto vec_store = llvm::dyn_cast<affine::AffineStoreOp>(use)) {
        memref = vec_store.getMemref();
      } else {
        // note that there is no composite load/store and load_store for PT, PE,
        // SFP
        llvm_unreachable("Memory operation not supported");
      }

      auto view_op = memref.getDefiningOp<dataflow::GetLogicalMemoryViewOp>();

      SenComponents mem_unit;
      auto* mem_unit_op = view_op.getFromUnit().getDefiningOp();
      if (isa<GetUnitOp>(mem_unit_op)) {
        mem_unit = dcc::getUnitType(llvm::dyn_cast<GetUnitOp>(mem_unit_op));
      } else if (isa<GetLocalUnitOp>(mem_unit_op)) {
        mem_unit =
            dcc::getUnitType(llvm::dyn_cast<GetLocalUnitOp>(mem_unit_op));
      } else if (auto mapping_unit_op =
                     llvm::dyn_cast<uniform::QueryMapOp>(mem_unit_op)) {
        auto unit_type =
            dcc::uniform::utils::getUnitTypeFromUniformMappingAsString(
                mapping_unit_op);
        if (unit_type.has_value()) {
          auto record =
              EnumsConversion::stringToSenComponents.find(unit_type.value());
          mem_unit = EnumsConversion::senCompToGenericComp.at(record->second);
        } else {
          mem_unit_op->emitOpError("Unit type is inconsistent.");
        }
      } else {
        llvm_unreachable("Unknown unit");
      }

      if (is_any_of(mem_unit, LRFREG)) {
        return TransformLoopToLegalizeForSentientLowering::KUnroll;
      }
    }
  }


}
// ---- 195/384  transformLoop  —  dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:399  (38L)
void e195_transformLoop(
    mlir::LoopLikeOpInterface loop_op, TransformType type) {
  // if no action required, return;
  if (type == TransformType::kNone) {
    return;
  }

  if (auto affine_for =
          llvm::dyn_cast<affine::AffineForOp>(loop_op.getOperation())) {
    if (type == TransformType::KUnroll) {
      if (failed(loopUnrollFull(affine_for))) {
        affine_for->emitError("Unable to unroll loop");
        signalPassFailure();
        return;
      }
    }
  } else if (auto scf_for =
                 llvm::dyn_cast<scf::ForOp>(loop_op.getOperation())) {
    auto lb_const =
        scf_for.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
    auto ub_const =
        scf_for.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
    auto step_const = scf_for.getStep().getDefiningOp<arith::ConstantIndexOp>();
    if (type == TransformType::KUnroll) {
      auto unroll_size =
          (ub_const.value() - lb_const.value()) / step_const.value();
      if (failed(loopUnrollByFactor(scf_for, unroll_size))) {
        scf_for->emitError("Unable to unroll the scf loop");
        signalPassFailure();
        return;
      }
    } else {  // kParentSplit
      if (failed(transformSCFLoopWithNonConstantUpperBound(scf_for))) {
        scf_for->emitError("Doesn't support this loops for transformation");
        signalPassFailure();
        return;
      }
    }

}}
// ---- 196/384  runOnOperation  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemView.cpp:41  (23L)
void e196_runOnOperation() {
  ModuleOp module_op = getOperation();
  module_op.walk([&](dataflow::ProgramUnitOp unit_op) {
    auto comp = dcc::getUnitType(
        unit_op.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
    if (is_any_of(comp, SenComponents::LXLU, SenComponents::LXSU,
                  SenComponents::L3LU, SenComponents::L3SU)) {
      while (true) {
        dataflow::GetPagedLogicalMemoryViewOp candidate = nullptr;
        unit_op.walk(
            [&](dataflow::GetPagedLogicalMemoryViewOp paged_mem_view_op) {
              candidate = paged_mem_view_op;
              return WalkResult::interrupt();
            });
        if (!candidate) break;
        TPMVManager tpmv_manager(candidate, comp);
        if (tpmv_manager.run().failed()) {
          signalPassFailure();
          return;
        }
      }
    }
  });

}
// ---- 197/384  calculateIndicesRanges  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:56  (25L)
void e197_calculateIndicesRanges(
    SmallVectorImpl<Value> &indices, SmallVectorImpl<IVRange> &indices_ranges) {
  for (int dim = 0; dim < indices.size(); ++dim) {
    auto index = indices[dim];
    auto block_arg = dyn_cast<BlockArgument>(index);
    DT_CHECK_MSG(block_arg, "expecting only BlockArguments in indices");

    auto *loop_op = block_arg.getOwner()->getParentOp();
    auto affine_for = dyn_cast<affine::AffineForOp>(loop_op);
    DT_CHECK_MSG(
        affine_for,
        "agen memory operations involving loop iterators can only have loop "
        "iterators from affine::AffineForOps in the subscripts");

    auto lb_map = affine_for.getLowerBoundMap();
    auto ub_map = affine_for.getUpperBoundMap();
    int lb, ub;
    if (lb_map.isSingleConstant() && ub_map.isSingleConstant()) {
      lb = lb_map.getSingleConstantResult();
      ub = ub_map.getSingleConstantResult() - 1;
    } else {
      llvm_unreachable("only loops with constant bounds are supported");
    }

    indices_ranges.emplace_back(lb, ub);

}}
// ---- 198/384  createConditionsForHyperRectSubscripts  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:289  (48L)
void e198_createConditionsForHyperRectSubscripts(
    SmallVectorImpl<Operation *> &insert_refs,
    FlatLinearValueConstraints &page_sel_constraints, AffineMap &subscripts_map,
    SmallVectorImpl<Value> &indices, SmallVectorImpl<IVRange> &indices_ranges) {
  int num_dim_vars = 0;
  SmallVector<AffineExpr, 16> new_dim_exprs;
  SmallVector<Value> indices_to_delete;

  OpBuilder builder(context_);
  int num_dims = subscripts_map.getNumDims();
  for (unsigned dim = 0; dim < num_dims; ++dim) {
    auto lb = page_sel_constraints.getConstantBound(
        mlir::presburger::BoundType::LB, dim);
    auto ub = page_sel_constraints.getConstantBound(
        mlir::presburger::BoundType::UB, dim);
    DT_CHECK_MSG(lb.has_value() && ub.has_value(),
                 "expected constant lower and upper bounds");
    auto lb_val = (int64_t)lb.value();
    auto ub_val = (int64_t)ub.value();

    if (lb_val == ub_val) {
      // If LB == UB, we only need one equality condition.
      for (int i = 0, e = insert_refs.size(); i < e; ++i) {
        setBuilderToInsertRef(builder, insert_refs[i]);
        insert_refs[i] = createEqualityCondition(builder, indices[dim], lb_val);
      }

      // If LB == UB, the loop iterator can be replaced by a constant in the
      // subscripts.
      new_dim_exprs.emplace_back(getAffineConstantExpr(lb_val, context_));
      indices_to_delete.push_back(indices[dim]);
    } else {
      new_dim_exprs.emplace_back(getAffineDimExpr(num_dim_vars++, context_));
      // If the lower and upper bounds span the whole loop iteration space,
      // the loop iterator does not aid in identifying a unique page. It can
      // be skipped.
      if (lb_val == indices_ranges[dim].first &&
          ub_val == indices_ranges[dim].second)
        continue;

      for (int i = 0, e = insert_refs.size(); i < e; ++i) {
        setBuilderToInsertRef(builder, insert_refs[i]);
        insert_refs[i] =
            createInequalityCondition(builder, indices[dim], lb_val, ub_val);
      }
    }
  }
  subscripts_map = subscripts_map.replaceDimsAndSymbols({new_dim_exprs}, {},

}
// ---- 199/384  createConditionsForNonHyperRectSubscripts  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:342  (35L)
void e199_createConditionsForNonHyperRectSubscripts(
    SmallVectorImpl<Operation *> &insert_refs, AffineMap &subscripts_map,
    SmallVectorImpl<Value> &iter_args,
    FlatLinearValueConstraints &page_set_constraints) {
  DT_CHECK(page_set_constraints.getNumDimVars() ==
           subscripts_map.getNumResults());

  OpBuilder builder(context_);
  int arg_idx = 0;
  for (int res = 0, e = subscripts_map.getNumResults(); res < e; ++res) {
    if (isa<AffineConstantExpr>(subscripts_map.getResult(res))) continue;

    auto lb = page_set_constraints.getConstantBound(
        mlir::presburger::BoundType::LB, res);
    auto ub = page_set_constraints.getConstantBound(
        mlir::presburger::BoundType::UB, res);
    DT_CHECK_MSG(lb.has_value() && ub.has_value(),
                 "expected constant lower and upper bounds");
    auto lb_val = (int64_t)lb.value();
    auto ub_val = (int64_t)ub.value();

    if (lb_val == ub_val) {
      // Equality
      for (int i = 0, e = insert_refs.size(); i < e; ++i) {
        setBuilderToInsertRef(builder, insert_refs[i]);
        insert_refs[i] =
            createEqualityCondition(builder, iter_args[arg_idx], lb_val);
      }
    } else {
      // Inequality - one condition for each bound
      for (int i = 0, e = insert_refs.size(); i < e; ++i) {
        setBuilderToInsertRef(builder, insert_refs[i]);
        insert_refs[i] = createInequalityCondition(builder, iter_args[arg_idx],
                                                   lb_val, ub_val);
      }

}}}
// ---- 200/384  updateTPMVInfo  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:382  (17L)
void e200_updateTPMVInfo(TPMVInfo &info, IRMapping &ir_map) {
  // Update the indices to keep them in sync as we clone the loops.
  for (int i = 0, e = info.indices_.size(); i < e; ++i) {
    auto new_index = ir_map.lookupOrNull(info.indices_[i]);
    // indices may contain iterators that weren't re-cloned.
    if (new_index) info.indices_[i] = new_index;
  }

  // Update paged_mem_view, if it exists, since it may have changed.
  // The mem_view may not be in the innermost loop.
  if (info.paged_mem_view_) {
    auto new_mem_view = ir_map.lookupOrDefault(info.paged_mem_view_);
    // mem_view may be outside re-cloned loop.
    if (new_mem_view)
      info.paged_mem_view_ = cast<dataflow::GetPagedLogicalMemoryViewOp>(
          new_mem_view.getDefiningOp());
  }

}
// ---- 201/384  setLoopIteratorOrder  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:575  (13L)
SmallVector<int> e201_setLoopIteratorOrder(
    SmallVectorImpl<Value> &indices) {
  // Initialize the ordered_indices_idxs vector to prepare for sorting.
  SmallVector<int> ordered_indices_idxs;
  for (int i = 0, e = indices.size(); i < e; ++i)
    ordered_indices_idxs.push_back(i);

  llvm::sort(ordered_indices_idxs, [&](int a, int b) -> bool {
    auto *a_loop = cast<BlockArgument>(indices[a]).getOwner()->getParentOp();
    auto *b_loop = cast<BlockArgument>(indices[b]).getOwner()->getParentOp();
    return a_loop->isProperAncestor(b_loop);
  });


}
// ---- 202/384  initialize  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:658  (13L)
LogicalResult e202_initialize() {
  DT_CHECK(mem_ops_.size() == 1);
  auto op = dyn_cast<agen::VectorLoadOp>(mem_ops_[0]);
  DT_CHECK(op);

  auto paged_mem_view = cast<dataflow::GetPagedLogicalMemoryViewOp>(
      op.getMemRef().getDefiningOp());
  tpmv_info_.emplace_back(paged_mem_view, op.getAffineMapAttr().getValue());
  for (auto index : op.getMapIndices()) tpmv_info_[0].indices_.push_back(index);

  context_ = op->getContext();

  return LogicalResult::success();

}
// ---- 203/384  removeCoresCoreletsFoldsFromDefImmutMap  —  dcc/src/Transform/Dataflow/UnitFiltering.cpp:99  (37L)
void e203_removeCoresCoreletsFoldsFromDefImmutMap(
    uniform::DefImmutableMappingOp immutable_map) {
  auto keys = immutable_map.getKeys();
  auto values = immutable_map.getValues();
  std::vector<mlir::Value> new_keys;
  std::vector<mlir::Value> new_values;
  DT_CHECK(!values.empty());
  bool unit_list_is_reduced = false;
  for (int i = 0; i < values.size(); ++i) {
    auto key = keys[i];
    auto get_unit_op_result = llvm::dyn_cast<OpResult>(key);
    unsigned fold_id = get_unit_op_result.getResultNumber();
    auto get_unit_op =
        llvm::dyn_cast<mlir::dataflow::GetUnitOp>(key.getDefiningOp());
    unsigned core_id = dcc::getCoreId(get_unit_op);
    int corelet_id = dcc::getCoreletId(get_unit_op);
    if ((!filter_folds_except_.empty() &&
         filter_folds_except_.count(fold_id) == 0) ||
        (!filter_cores_except_.empty() &&
         filter_cores_except_.count(core_id) == 0) ||
        (!filter_corelets_except_.empty() && corelet_id != -1 &&
         filter_corelets_except_.count(corelet_id) == 0)) {
      unit_list_is_reduced = true;
      continue;
    }
    new_keys.push_back(key);
    new_values.push_back(values[i]);
  }
  // NOTE: If all keys are to be filtered out, do not modify the map.
  // Instead, the entire region should be deleted at the last step of this pass.
  if (!unit_list_is_reduced || new_keys.empty()) return;

  OpBuilder builder(immutable_map);
  Value new_immutable_map = mlir::uniform::DefImmutableMappingOp::create(
      builder, immutable_map->getLoc(), builder.getIndexType(), new_keys,
      new_values);
  immutable_map.replaceAllUsesWith(new_immutable_map);

}
// ---- 204/384  cleanup  —  dcc/src/Transform/Dataflow/UnitFiltering.cpp:263  (49L)
void e204_cleanup(ModuleOp module_op) {
  module_op.walk([&](Operation *op) {
    if (isa<dataflow::CreateGroupOp, dataflow::CreateMulticastGroupOp>(op) &&
        op->use_empty())
      op->erase();
    else if (auto get_unit_op = dyn_cast<dataflow::GetUnitOp>(op)) {
      if (get_unit_op.use_empty()) {
        get_unit_op.erase();
        return WalkResult::advance();
      }

      if (filter_folds_except_.empty()) return WalkResult::advance();
      unsigned core_id = dcc::getCoreId(get_unit_op);
      int corelet_id = dcc::getCoreletId(get_unit_op);

      OpBuilder builder(get_unit_op);
      int orig_num_folds = get_unit_op.getNumResults();
      // Determine the new number of folds. Cannot assume
      // all elements of filter_folds_except_ are existing fold IDs.
      int new_num_folds = 0;
      for (auto fold_id : filter_folds_except_) {
        if (fold_id < orig_num_folds) ++new_num_folds;
      }

      SmallVector<Type> result_types;
      for (int i = 0; i < new_num_folds; i++)
        result_types.push_back(builder.getIndexType());
      auto new_unit_op = dataflow::GetUnitOp::create(
          builder, get_unit_op.getLoc(), result_types,
          get_unit_op.getNameAttr(), get_unit_op.getTypeAttr());
      new_unit_op->setAttr("core", builder.getI32IntegerAttr(core_id));
      if (corelet_id != -1)
        new_unit_op->setAttr("corelet", builder.getI32IntegerAttr(corelet_id));
      new_unit_op->setAttr("num_folds",
                           builder.getI32IntegerAttr(new_num_folds));
      // Map the original fold IDs into the new, filtered fold IDs.
      for (int orig_fold_id = 0, new_fold_id = 0; orig_fold_id < orig_num_folds;
           ++orig_fold_id) {
        if ((filter_folds_except_.count(orig_fold_id) != 0)) {
          get_unit_op->getResults()[orig_fold_id].replaceAllUsesWith(
              new_unit_op->getResults()[new_fold_id]);
          ++new_fold_id;
        }
      }
      get_unit_op->dropAllUses();
      get_unit_op->erase();
    }
    return WalkResult::advance();
  });

}
// ---- 205/384  isDataTransferToKeep  —  dcc/src/Transform/Dataflow/UnitFiltering.cpp:327  (10L)
bool e205_isDataTransferToKeep(Operation *op) {
  if (isDataTransfer(op)) {
    if (auto dbg_name = getDbgNameAttr(op)) {
      if (std::find(filter_transfers_except_.begin(),
                    filter_transfers_except_.end(),
                    dbg_name.str()) != filter_transfers_except_.end())
        return true;
    }
  }
  return false;


}
// ==================================================================================================
// LEVEL 2
// ==================================================================================================

// ---- 206/384  constructChunkAndShuffleInfo  —  dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:98  (167L)
LogicalResult e206_constructChunkAndShuffleInfo() {
  SmallVector<int64_t> tmp_layout_coeffs;
  SmallVector<int> tmp_extents;

  // Find out the order
  // We expect the layout coefficients to be in decreasing order.
  // i.e., row-major order;
  // However, layout map can express both row-major and column-major.
  // Hence, we convert them into row-major momentarily to compute chunk sizes
  // and chunk strides.
  auto layout_coeffs = getLayoutCoeffs();
  auto extents = getExtents();
  DT_CHECK(layout_coeffs.size() >= extents.size());
  if (layout_coeffs[extents.size() - 1] > layout_coeffs[0]) {
    tmp_layout_coeffs.resize(layout_coeffs.size() - 1);
    std::reverse_copy(layout_coeffs.begin(), layout_coeffs.end() - 1,
                      tmp_layout_coeffs.begin());
    tmp_layout_coeffs.push_back(layout_coeffs.back());

    tmp_extents.resize(extents.size());
    std::reverse_copy(extents.begin(), extents.end(), tmp_extents.begin());
  } else {
    tmp_layout_coeffs.insert(tmp_layout_coeffs.begin(), layout_coeffs.begin(),
                             layout_coeffs.end());
    tmp_extents.insert(tmp_extents.begin(), extents.begin(), extents.end());
  }

  // Layout is constructed from the memory view. It has additional dimension
  // of constant, hence its size is 1 greater than extents.
  if (tmp_layout_coeffs.size() != tmp_extents.size() + 1 ||
      tmp_layout_coeffs[tmp_extents.size() - 1] != 1) {
    return LogicalResult::failure();
  }

  // calculate chunk_size
  // Chunk size is calculated as the contiguous set of elements from the
  // innermost dimension (dim = 0).
  // layouts: [1][4][2][3], extent: [4][1][1] --> chunk size = 4
  // layouts: [1][6][2][3], extent: [3][1][1] --> chunk size = 3
  // layouts: [1][6][2][3], extent: [1][2][1] --> chunk size = illegal

  // first dimension at which layout and extent don't match.
  int chunk_dim_idx = -1;

  // tmp_layout_extent_multiplier captures overall size of dimensions encounterd
  // will be used later to compute chunk stride by subtracting with chunk size.
  int tmp_layout_extent_multiplier = 1;

  auto op = getOp();
  int chunk_size = 0;
  for (int dim = tmp_extents.size() - 1; dim >= 0; dim--) {
    int multiplier = dim == 0
                         ? INT32_MAX
                         : tmp_layout_coeffs[dim - 1] / tmp_layout_coeffs[dim];

    if (multiplier < tmp_extents[dim])
      return op->emitError(
          "Extent in load/store set is larger than from the layout");

    // Break loop when extent is different from layout coeff
    if (multiplier != tmp_extents[dim]) {
      chunk_dim_idx = dim;
      chunk_size = tmp_layout_extent_multiplier * tmp_extents[dim];
      tmp_layout_extent_multiplier *= multiplier;
      break;
    }

    tmp_layout_extent_multiplier *= multiplier;
  }
  setChunkSize(chunk_size);

  // Find chunk_stride
  //  Chunk_stride dimension is equal to first non-unit extent accessed.
  int chunk_stride = 0;
  int chunk_stride_dim_idx = -1;
  for (int dim = chunk_dim_idx - 1; dim >= 0; dim--) {
    int multiplier = dim == 0
                         ? INT32_MAX
                         : tmp_layout_coeffs[dim - 1] / tmp_layout_coeffs[dim];

    if (multiplier < tmp_extents[dim])
      return op->emitError(
          "Extent in load/store set is larger than from the layout");

    if (tmp_extents[dim] != 1) {
      chunk_stride = tmp_layout_extent_multiplier;
      chunk_stride_dim_idx = dim;
      break;
    }
    tmp_layout_extent_multiplier *= multiplier;
  }
  setChunkStride(chunk_stride);

  // Legality conditions for lowering
  //  extent size at chunk_dim_idx == 1
  //  and chunk_stride != 0
  if (tmp_layout_coeffs[chunk_dim_idx] == 1 && chunk_stride != 0)
    return op->emitError(
        "Chunk starting offset "
        "from load/store set are not supported in lowering");

  // Check for presence of variable strides
  for (int dim = chunk_stride_dim_idx - 1; dim >= 0; dim--) {
    if (tmp_extents[dim] != 1)
      return op->emitError("Variable chunk strides cannot be lowered");
  }

  // find shuffle info for vector_load op
  // TODO need to enable rotation
  if (isa<VectorLoadOp, IndirectVectorLoadOp, CompositeLoadOp,
          CompositeIndirectLoadOp>(op)) {
    SmallVector<Operation*> users;
    if (isa<VectorLoadOp, IndirectVectorLoadOp>(op)) {
      for (auto* user : op->getResult(0).getUsers()) users.push_back(user);
    } else if (auto load_op = dyn_cast<CompositeLoadOp>(op)) {
      for (auto* user : load_op.getLoadInductionVar().getUsers())
        users.push_back(user);
    } else if (auto ind_load_op = dyn_cast<CompositeIndirectLoadOp>(op)) {
      for (auto* user : ind_load_op.getLoadInductionVar().getUsers())
        users.push_back(user);
    }

    for (auto* user : users) {
      // Assume each vector-loaded data can be used only once.
      if (isa<vectorchain::SelectOp>(user)) {
        auto select_op = cast<vectorchain::SelectOp>(user);
        auto input_size =
            dataflow::utils::getNumElements(select_op.getData().getType());
        auto selection_map = select_op.getSelectionMap();
        bool valid_select_map = true;
        // TODO need a better algorithm to determine shuffle mode
        selection_map.walkExprs([&](AffineExpr expr) {
          if (expr.getKind() == AffineExprKind::Mod) {
            setShuffleMode("splat");
          } else if (expr.getKind() == AffineExprKind::Constant) {
            int64_t modulo = llvm::cast<AffineConstantExpr>(expr).getValue();
            // DT_CHECK(input_size == modulo);
          } else if (expr.getKind() != AffineExprKind::DimId) {
            op->emitError("Non-supported select map for VectorLoadOp!");
            valid_select_map = false;
          }
        });
        if (!valid_select_map) {
          return op->emitError("Abort due to invalid select map");
        }
      } else if (isa<vectorchain::RotateOp>(user)) {
        auto rotate_op = dyn_cast<vectorchain::RotateOp>(user);
        if (auto const_op =
                rotate_op.getPosition()
                    .getDefiningOp<mlir::arith::ConstantIndexOp>()) {
          setRotationPosition(const_op.value());
          DT_CHECK_MSG(
              getRotationPosition() <= getTotalElements(),
              "Rotation position has to be less than or equal to total "
              "elements");
          DT_CHECK_MSG(getRotationPosition() >= 0,
                       "right rotation amount has to be non-negative");
          DT_CHECK_MSG(rotate_op.getRightShift(),
                       "only right rotation supported");
        } else {
          DT_ERROR("index position to rotation op has to be a constant");
        }
      }
    }
  }

  return LogicalResult::success();

}
// ---- 207/384  initialize  —  dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:295  (57L)
LogicalResult e207_initialize() {
  DT_CHECK_MSG(getMemoryIndex() != MemoryOperandIndex::kMax,
               "uninitialized memory_index_ detected");

  auto op = getOp();
  SmallVector<Value> indices;
  if (auto load_op = dyn_cast<VectorLoadOp>(op)) {
    setMemRef(load_op.getMemRef());
    setTransferSet(load_op.getLoadSet());
    setTransferOrder(load_op.getLoadOrder());
    setSubscriptsMap(load_op.getAffineMap());
    setElementWidth(
        dataflow::utils::getElementTypeBitWidth(load_op.getResult().getType()));
    setExpectedTotalElements(
        dataflow::utils::getNumElements(load_op.getResult().getType()));
    for (auto index : load_op.getMapIndices()) indices.push_back(index);
  } else if (auto store_op = dyn_cast<VectorStoreOp>(op)) {
    setMemRef(store_op.getMemRef());
    setTransferSet(store_op.getStoreSet());
    setTransferOrder(store_op.getStoreOrder());
    setSubscriptsMap(store_op.getAffineMap());
    setElementWidth(dataflow::utils::getElementTypeBitWidth(
        store_op.getValueToStore().getType()));
    setExpectedTotalElements(
        dataflow::utils::getNumElements(store_op.getValueToStore().getType()));
    for (auto index : store_op.getMapOperands()) indices.push_back(index);
  } else if (auto ind_load_op = dyn_cast<IndirectVectorLoadOp>(op)) {
    setMemRef(ind_load_op.getDirectMemref());
    setTransferSet(ind_load_op.getLoadSet());
    setTransferOrder(ind_load_op.getLoadOrder());
    setSubscriptsMap(ind_load_op.getDirectAffineMapAttr().getValue());
    setElementWidth(dataflow::utils::getElementTypeBitWidth(
        ind_load_op.getResult().getType()));
    setExpectedTotalElements(
        dataflow::utils::getNumElements(ind_load_op.getResult().getType()));
    for (auto index : ind_load_op.getDirectMapIndices())
      indices.push_back(index);
  } else if (auto ind_store_op = dyn_cast<IndirectVectorStoreOp>(op)) {
    setMemRef(ind_store_op.getDirectMemref());
    setTransferSet(ind_store_op.getStoreSet());
    setTransferOrder(ind_store_op.getStoreOrder());
    setSubscriptsMap(ind_store_op.getDirectAffineMapAttr().getValue());
    setElementWidth(dataflow::utils::getElementTypeBitWidth(
        ind_store_op.getValue().getType()));
    setExpectedTotalElements(
        dataflow::utils::getNumElements(ind_store_op.getValue().getType()));
    for (auto index : ind_store_op.getDirectMapIndices())
      indices.push_back(index);
  } else {
    return op->emitError("unsupported operation");
  }

  setIndices(indices);

  if (initializeMemViewInfo().failed()) return LogicalResult::failure();

  return LogicalResult::success();

}
// ---- 208/384  emplace_insert  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:378  (8L)
  T& emplace_insert(MemoryOperandIndex moi, Args&&... args) {
    DT_CHECK_MSG(!has(moi),
                 "trying to insert into a slot that is already filled");
    DT_CHECK_MSG(this->size() <= MemoryOperandIndex::kMax,
                 "no more than kMax entries are allowed");
    index_mapping_[moi] = this->size();
    this->emplace_back(std::forward<Args>(args)...);
    return get(moi);

}
// ---- 209/384  getFirst  —  dcc/src/Conversion/AgenToSentient/AccessDetails.hpp:411  (7L)
  const T& getFirst() {
    if (has(MemoryOperandIndex::kDirSrc))
      return get(MemoryOperandIndex::kDirSrc);
    else if (has(MemoryOperandIndex::kDirDst))
      return get(MemoryOperandIndex::kDirDst);
    else
      llvm_unreachable("expected at least one of kDirSrc or kDirDst operands");

}
// ---- 210/384  checkBasicConditions  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:58  (133L)
LogicalResult e210_checkBasicConditions(Operation* op) {
  IntegerSet time_set;
  AffineMap time_order;
  SmallVector<IntegerSet, 2> data_sets;
  SmallVector<AffineMap, 2> data_orders;
  if (auto load_op = dyn_cast<agen::VectorLoadOp>(op)) {
    data_sets.push_back(load_op.getLoadSet().getValue());
    data_orders.push_back(load_op.getLoadOrder());
  } else if (auto ind_load_op = dyn_cast<agen::IndirectVectorLoadOp>(op)) {
    data_sets.push_back(ind_load_op.getLoadSet().getValue());
    data_orders.push_back(ind_load_op.getLoadOrder());
  } else if (auto store_op = dyn_cast<agen::VectorStoreOp>(op)) {
    data_sets.push_back(store_op.getStoreSet().getValue());
    data_orders.push_back(store_op.getStoreOrder());
  } else if (auto sym_load_op = dyn_cast<SymbolicVectorLoadOp>(op)) {
    data_sets.push_back(sym_load_op.getLoadSet().getValue());
    data_orders.push_back(sym_load_op.getLoadOrder());
  } else if (auto sym_store_op = dyn_cast<SymbolicVectorStoreOp>(op)) {
    data_sets.push_back(sym_store_op.getStoreSet().getValue());
    data_orders.push_back(sym_store_op.getStoreOrder());
  } else if (auto ind_store_op = dyn_cast<agen::IndirectVectorStoreOp>(op)) {
    data_sets.push_back(ind_store_op.getStoreSet().getValue());
    data_orders.push_back(ind_store_op.getStoreOrder());
  } else if (auto composite_load_op = dyn_cast<agen::CompositeLoadOp>(op)) {
    data_sets.push_back(composite_load_op.getLoadSet().getValue());
    data_orders.push_back(composite_load_op.getLoadOrder());
    time_set = composite_load_op.getTimeSet().getValue();
    time_order = composite_load_op.getTimeOrder();
    if (checkCompositeRegion(op).failed()) {
      signalPassFailure();
      return LogicalResult::failure();
    }
  } else if (auto comp_ind_load_op =
                 dyn_cast<agen::CompositeIndirectLoadOp>(op)) {
    data_sets.push_back(comp_ind_load_op.getLoadSet().getValue());
    data_orders.push_back(comp_ind_load_op.getLoadOrder());
    time_set = comp_ind_load_op.getTimeSet().getValue();
    time_order = comp_ind_load_op.getTimeOrder();
    if (checkCompositeRegion(op).failed()) {
      signalPassFailure();
      return LogicalResult::failure();
    }
  } else if (auto composite_store_op = dyn_cast<agen::CompositeStoreOp>(op)) {
    data_orders.push_back(composite_store_op.getStoreOrder().value());
    time_set = composite_store_op.getTimeSet().getValue();
    time_order = composite_store_op.getTimeOrder();
    if (composite_store_op.getInputVector() == nullptr) {
      data_sets.push_back(composite_store_op.getStoreSet().value().getValue());
    }
    if (checkCompositeRegion(op).failed()) {
      signalPassFailure();
      return LogicalResult::failure();
    }
  } else if (auto comp_ind_store_op =
                 dyn_cast<agen::CompositeIndirectStoreOp>(op)) {
    data_sets.push_back(comp_ind_store_op.getStoreSet().getValue());
    data_orders.push_back(comp_ind_store_op.getStoreOrder());
    time_set = comp_ind_store_op.getTimeSet().getValue();
    time_order = comp_ind_store_op.getTimeOrder();
    if (checkCompositeRegion(op).failed()) {
      signalPassFailure();
      return LogicalResult::failure();
    }
  } else if (auto composite_load_store_op =
                 dyn_cast<agen::CompositeLoadAndStoreOp>(op)) {
    data_sets.push_back(composite_load_store_op.getLoadSet().getValue());
    data_sets.push_back(composite_load_store_op.getStoreSet().getValue());
    data_orders.push_back(composite_load_store_op.getLoadOrder());
    data_orders.push_back(composite_load_store_op.getStoreOrder());
    time_set = composite_load_store_op.getTimeSet().getValue();
    time_order = composite_load_store_op.getTimeOrder();
  } else if (auto composite_ind_load_store_op =
                 dyn_cast<agen::CompositeIndirectLoadAndStoreOp>(op)) {
    data_sets.push_back(composite_ind_load_store_op.getLoadSet().getValue());
    data_sets.push_back(composite_ind_load_store_op.getStoreSet().getValue());
    data_orders.push_back(composite_ind_load_store_op.getLoadOrder());
    data_orders.push_back(composite_ind_load_store_op.getStoreOrder());
    time_set = composite_ind_load_store_op.getTimeSet().getValue();
    time_order = composite_ind_load_store_op.getTimeOrder();
  } else if (auto recv_and_send_op =
                 dyn_cast<sentient::ReceiveAndStoreOp>(op)) {
    return recv_and_send_op->hasAttr("marked") ? LogicalResult::success()
                                               : LogicalResult::failure();
  }

  // Allowing only permutation maps in the data_order
  for (auto& data_order : data_orders) {
    auto data_order_inverse = inversePermutation(data_order);
    if (!data_order_inverse) {
      signalPassFailure();
      return op->emitError(
          "Only permutations are allowed in the load "
          "order for lowering into sentient");
    }
  }

  for (auto& data_set : data_sets) {
    affine::FlatAffineValueConstraints constraints(data_set);
    auto is_hyper_rectangle =
        constraints.isHyperRectangular(0, constraints.getNumCols() - 1);
    if (!is_hyper_rectangle) {
      signalPassFailure();
      return op->emitError(
          "The load set needs to be hyper rectangular for "
          "lowering into sentient");
    }
  }

  if (isa<CompositeLoadOp, CompositeStoreOp, CompositeLoadAndStoreOp,
          CompositeIndirectLoadAndStoreOp>(op)) {
    // Allowing only permutation maps in the time_order
    auto time_order_inverse = inversePermutation(time_order);
    if (!time_order_inverse) {
      signalPassFailure();
      return op->emitError(
          "Only permutations are allowed in the time "
          "order for lowering into sentient");
    }

    // TODO find out why time_set is not rectangle.
    // FlatAffineValueConstraints constraints(time_set);
    // auto is_hyper_rectangle =
    //     constraints.isHyperRectangular(0, constraints.getNumCols() - 1);
    // if (!is_hyper_rectangle) {
    //   op->emitError(
    //       "The time set needs to be hyper rectangular for "
    //       "lowering into sentient");
    //   signalPassFailure();
    //   return LogicalResult::failure();
    // }
  }

  return LogicalResult::success();

}
// ---- 211/384  processInterleaveOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:305  (80L)
LogicalResult e211_processInterleaveOp(
    dataflow::ProgramUnitOp& unit, Operation* op) {
  auto interleave_op = dyn_cast_or_null<CompositeMemoryInterleaveOp>(op);
  DT_CHECK_MSG(interleave_op, "expecting a CompositeMemoryInterleaveOp");

  // Currently only supported in L3 units.
  auto comp =
      dcc::getUnitType(unit.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
  if (!is_any_of(comp, L3LU, L3SU)) {
    signalPassFailure();
    return interleave_op->emitOpError("only supported in L3 units");
  }

  // If the granularity attribute is specified, it must not exceed the maximum
  // burst supported for the unit.
  SenSystemDef sysDef;
  auto max_burst = sysDef.l3BurstSize;
  auto granularity = max_burst;
  if (interleave_op->hasAttr("granularity")) {
    granularity = interleave_op.getGranularity().value();
    if (granularity == 0 || granularity > max_burst) {
      signalPassFailure();
      return interleave_op->emitOpError("illegal granularity setting");
    }
  }

  // The region should contain at least two operations plus the YieldOp.
  Region& region = interleave_op.getRegion();

  auto& region_ops = region.front().getOperations();
  if (region_ops.size() < 3) {
    signalPassFailure();
    return interleave_op->emitOpError(
        "region does not contain enough operations");
  }

  // The region must only contain composite memory operations, and they must
  // have the same burst value and total_elements.
  auto first_op = &region_ops.front();
  if (!first_op->hasAttr("burst_size")) {
    signalPassFailure();
    return op->emitOpError(
        "invalid operation in region (no burst_size attribute)");
  }
  int burst =
      mlir::cast<mlir::IntegerAttr>(first_op->getAttr("burst_size")).getInt();

  if (!first_op->hasAttr("total_elements")) {
    signalPassFailure();
    return op->emitOpError(
        "invalid operation in region (no total_elements attribute)");
  }
  int total_elements =
      mlir::cast<mlir::IntegerAttr>(first_op->getAttr("total_elements"))
          .getInt();

  auto op_type = first_op->getName().getStringRef();
  for (auto& region_op : region_ops) {
    if (isa<agen::YieldOp>(region_op)) continue;
    if (!isa<sentient::LoadAndSendOp, sentient::ReceiveAndStoreOp,
             sentient::LoadAndStoreOp>(region_op)) {
      signalPassFailure();
      return op->emitOpError("invalid operation found in op region");
    }
    if (mlir::cast<mlir::IntegerAttr>(region_op.getAttr("burst_size"))
            .getInt() != burst) {
      signalPassFailure();
      return op->emitOpError("operation with different burst found");
    }
    if (mlir::cast<mlir::IntegerAttr>(region_op.getAttr("total_elements"))
            .getInt() != total_elements) {
      signalPassFailure();
      return op->emitOpError("operation with different total_elements found");
    }
    if (region_op.getName().getStringRef() != op_type) {
      signalPassFailure();
      return op->emitOpError("operation with unexpected operation type found");
    }
  }


}
// ---- 212/384  gatherAffineLoadStoreDetails  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:538  (74L)
LogicalResult e212_gatherAffineLoadStoreDetails(
    Operation* op, ProgramUnitOp& unit, SenComponents comp,
    AccessContainer<AccessDetailsTy>& access_details,
    AccessContainer<Value>& mutable_addrs,
    AccessContainer<Value>& immutable_addrs) {
  // Mark the op and its uses to recover later
  {
    OpBuilder builder(op);
    op->setAttr("marked", builder.getI8IntegerAttr(1));
  }

  AccessContainer<Value> mem_view_start_addrs;
  // Each unique index, in the set of all access_detail entries, has an
  // entry in this map.

  llvm::DenseMap<Value, std::vector<int64_t>> indices_coeff_dict;
  for (int i = 0, e = access_details.size(); i < e; ++i) {
    auto& record = access_details[i];
    // Initialize the indices_coeff_dict to map each unique loop iterator in the
    // set of all access_details entries to a vector of the same size to make
    // lowering simple.
    for (auto& inner_record : record.getIndicesCoeffDict()) {
      auto& list = indices_coeff_dict[inner_record.first];
      if (list.empty()) {
        for (auto j = 0; j < e; ++j) list.push_back(0);
      }
    }

    // Gather information from multiple access details for simultaneous
    // modification of loops.
    mutable_addrs.insert(record.getMemoryIndex(), record.getMemViewStartAddr());
    mem_view_start_addrs.insert(record.getMemoryIndex(),
                                record.getMemViewStartAddr());
    for (auto& inner_record : record.getIndicesCoeffDict())
      indices_coeff_dict[inner_record.first][i] = inner_record.second;

    LLVM_DEBUG({
      llvm::dbgs() << "access_detail[" << i << "]'s coeff_dict_:\n";
      for (auto& inner_record : record.getIndicesCoeffDict()) {
        if (inner_record.first && isa<BlockArgument>(inner_record.first))
          llvm::dbgs() << "inner_record.first:" << inner_record.first
                       << " from level:"
                       << getLoopNestLevel(
                              cast<BlockArgument>(inner_record.first)
                                  .getOwner()
                                  ->getParentOp())
                       << "\n";
        else
          llvm::dbgs() << "inner_record.first:" << inner_record.first << "\n";
        llvm::dbgs() << "inner_record.second:" << inner_record.second << "\n";
      }
    });
  }

  // Generate address manipulation statements and return the outermost loop
  // and also update the address with initial value coming from the
  // constant of the layout coefficients.
  {
    OpBuilder builder(&unit.getRegion().front().front());
    if (generateAffineAddressManipulationStmts<AccessDetailsTy>(
            builder, comp, access_details, mem_view_start_addrs, mutable_addrs,
            indices_coeff_dict)
            .failed()) {
      return op->emitError(
          "Unable to generate address manipulation statements");
    }
  }

  // If the unit is an L3 load or store unit, the immutable address of the
  // lowered operation should be set to the actual memory view start
  // address. It is expected these types of offsets will be handled in an
  // LBR/EBR, not LAR/EAR.
  if (failed(constructImmutableAddress<AccessDetailsTy>(
          comp, access_details, mem_view_start_addrs, immutable_addrs))) {

}}
// ---- 213/384  constructImmutableAddress  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:1217  (16L)
LogicalResult e213_constructImmutableAddress(
    SenComponents comp, AccessContainer<AccessDetailsTy>& access_details,
    const AccessContainer<Value>& updated_mem_view_start_addrs,
    AccessContainer<Value>& immutable_addrs) {
  if (access_details.size() != updated_mem_view_start_addrs.size()) {
    return LogicalResult::failure();
  }

  for (int i = 0; i < access_details.size(); i++) {
    if (is_any_of(comp, L3LU, L3SU)) {
      immutable_addrs.insert(access_details[i].getMemoryIndex(),
                             access_details[i].getMemViewStartAddr());
    } else {
      immutable_addrs.insert(access_details[i].getMemoryIndex(),
                             updated_mem_view_start_addrs[i]);
    }

}}
// ---- 214/384  setImmutableAddrAndIncrements  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:1581  (43L)
LogicalResult e214_setImmutableAddrAndIncrements(
    OpBuilder& builder, SenComponents comp, Operation* op,
    bool perform_burst_or_groups, int stride_size, int burst_size,
    int total_elements, Value memory_start_addr, Value& immutable_addr,
    Value& increment, dataflow::ProgramUnitOp unit_op) {
  // In case of L3, immutable_addr is always mem_view_start_addr.
  if (is_any_of(comp, L3LU, L3SU)) {
    immutable_addr = memory_start_addr;
  }

  // Set the insertion point to before the ProgramUnitOp for insertion of the
  // ConstantOps.
  auto insert_pt = builder.saveInsertionPoint();
  builder.setInsertionPoint(unit_op);

  if (perform_burst_or_groups) {
    if (is_any_of(comp, L3LU, L3SU)) {
      // TODO: assert that stride_size x precision = stick size
      auto increment_op = sentient::ConstantOp::create(
          builder, op->getLoc(), IndexType::get(op->getContext()),
          total_elements * burst_size);
      increment = increment_op.getOut();
    } else {
      auto immutable_op = sentient::ConstantOp::create(
          builder, op->getLoc(), IndexType::get(op->getContext()), stride_size);
      immutable_addr = immutable_op.getOut();

      auto increment_op = sentient::ConstantOp::create(
          builder, op->getLoc(), IndexType::get(op->getContext()), stride_size);
      increment = increment_op.getOut();
    }
  } else {
    // If there is no burst size or IL groups, the increment is always
    // zero in the AgenToSentient lowering. The later passes will modify
    // increment value to non-zero values depending on possibilities.
    // This is a statement irrespective of L3/LX/L0 LU/SU units.
    if (!is_any_of(comp, L3LU, L3SU)) {
      auto immutable_op = sentient::ConstantOp::create(
          builder, op->getLoc(), IndexType::get(op->getContext()), 0);
      immutable_addr = immutable_op.getOut();
    }
    auto increment_op = sentient::ConstantOp::create(
        builder, op->getLoc(), IndexType::get(op->getContext()), 0);

}}
// ---- 215/384  setsttype  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:1731  (50L)
LogicalResult e215_setsttype(
    OpBuilder* builder, SenComponents comp, Operation* producer_input,
    int& total_elements, unsigned& element_width,
    SentientShuffleModeAttr& shuffle_mode) {
  // non-default ldtypes currently support for LX only
  if (!is_any_of(comp, LXLU, LXSU)) return LogicalResult::success();

  SenSystemDef sysDef;
  // ShuffleOp producer input indicates an explicit, non-default sttype
  // setting.
  if (auto shuffle_op = dyn_cast<vectorchain::ShuffleOp>(producer_input)) {
    // total_elements should reflect full stick. element_width is in bits.
    total_elements = sysDef.bytesPerStick * 8 / element_width;

    int repetitions = shuffle_op.getRepetition();
    if (isSplatFromFirstElem(shuffle_op, 2) && repetitions == 1) {
      // 2B sttype (mode 1)
      shuffle_mode = SentientShuffleModeAttr::get(
          builder->getContext(), SentientShuffleMode::masked2b);
    } else if (isSplatFromFirstElem(shuffle_op, 16) && repetitions == 1) {
      // 16B sttype (mode 2)
      shuffle_mode = SentientShuffleModeAttr::get(
          builder->getContext(), SentientShuffleMode::masked16b);
    } else {
      return shuffle_op->emitOpError("unsupported sttype");
    }
  } else if (auto receive_op = dyn_cast<dataflow::ReceiveOp>(producer_input)) {
    Type elem_type =
        dataflow::utils::getElementType(receive_op->getResultTypes()[0]);
    int result_size =
        dataflow::utils::getNumElements(receive_op->getResultTypes()[0]);
    DT_CHECK_MSG(dataflow::utils::isIntOrFloatType(elem_type),
                 "producer_input should return an int or float vector");
    unsigned sttype =
        dataflow::utils::getIntOrFloatBitWidth(elem_type) * result_size / 8;

    if (sttype == 2) {
      shuffle_mode = SentientShuffleModeAttr::get(
          builder->getContext(), SentientShuffleMode::masked2b);
      ;
    } else if (sttype == 16) {
      shuffle_mode = SentientShuffleModeAttr::get(
          builder->getContext(), SentientShuffleMode::masked16b);
      ;
    } else if (sttype != 128) {
      return receive_op->emitOpError("unsupported sttype");
    }

    // total_elements should reflect full stick. element_width is in bits.
    total_elements = sysDef.bytesPerStick * 8 / element_width;

}}
// ---- 216/384  constructReceiveAndExtractScalarOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2471  (91L)
LogicalResult e216_constructReceiveAndExtractScalarOp(
    VectorStoreOp store_op, unsigned& extract_idx,
    SmallVectorImpl<Operation*>& ops_to_be_deleted) {
  // Pattern to match (op should be %1 here):
  //   #map = <a one dimensional, identity map>
  //   #set = <a one dimensional set>
  //   %0 = dataflow.receive %unit : vector<??x??>
  //   %1 = dataflow.get_logical_memory_view %virtual_ibr, 0
  //            {layout_map = #map} : index, index, memref<??x??>
  //   agen.vector_store %0, %1[0] {store_set = #set, store_order = #map}
  //       : memref<??x??>, vector<??x??>
  //
  // Each indirect store op will have its own ReceiveAndExtractScalarOp
  // pattern.

  // Get the indirect memory view from the store op.
  auto ind_mem_view = cast<dataflow::GetLogicalMemoryViewOp>(
      store_op.getMemRef().getDefiningOp());

  // Check the memory view matches expected pattern
  if (!checkIndirectMemViewForExtractOp(ind_mem_view))
    return ind_mem_view->emitError(
        "indirect memory view does not match expected extract pattern");

  // Check the users of the indirect mem view and identify the indirect
  // memory op user that matches the pattern.
  Operation* ind_store_op = nullptr;
  unsigned num_users = 0;
  for (auto user : ind_mem_view->getUsers()) {
    ++num_users;
    if (isa<IndirectVectorStoreOp, CompositeIndirectStoreOp>(user))
      ind_store_op = user;
    else if (!isa<VectorStoreOp>(user))
      return ind_mem_view->emitError("invalid user of indirect memory view");
  }
  if (!ind_store_op)
    return ind_mem_view->emitError(
        "indirect memory view should have an "
        "IndirectVectorStoreOp/CompositeIndirectStoreOp user");

  if (num_users != 2)
    return ind_mem_view->emitError(
        "indirect memory view should only have 2 users");

  if (!checkStoreOpFromExtractPattern(store_op))
    return store_op->emitError("store_op failed checks");

  // The store op should operate on a receive op. Check the receive op has one
  // use and operates on the correct unit type.
  auto receive_op = dyn_cast_or_null<dataflow::ReceiveOp>(
      store_op.getValueToStore().getDefiningOp());
  DT_CHECK_MSG(receive_op, "store op should operate on a receive op");

  if (!receive_op->hasOneUse())
    return store_op->emitError(
        "receive op should only be used by the store_op");

  auto receive_from_unit =
      dyn_cast<dataflow::GetUnitOp>(receive_op.getFromUnit().getDefiningOp());
  DT_CHECK_MSG(receive_from_unit, "from_unit of receiveOp should be GetUnitOp");
  SenComponents receive_from_unit_comp =
      EnumsConversion::stringToSenComponents
          .find(receive_from_unit.getType().str())
          ->second;
  auto receive_from_gen_comp =
      EnumsConversion::senCompToGenericComp.at(receive_from_unit_comp);
  if (!is_any_of(receive_from_gen_comp, SenComponents::PE, SenComponents::PT,
                 SenComponents::LXLU))
    return receive_op->emitError(
        "from unit of receive op should be PE/PT/LXLU");

  // Create the receive_and_extract_scalar_op.
  OpBuilder builder(ind_mem_view);
  auto zero_constant = sentient::ConstantOp::create(
      builder, ind_mem_view->getLoc(),
      IndexType::get(ind_mem_view->getContext()), 0);
  auto dbg_name_attr = getDbgNameAttr(ind_mem_view);
  auto extract_op = sentient::ReceiveAndExtractScalarOp::create(
      builder, ind_mem_view->getLoc(), receive_op.getFromUnit(), zero_constant,
      dbg_name_attr);

  // Mark both the receive_and_extract_scalar op and the indirect store op
  // associated to it with the extract_idx. This allows the indirect store op to
  // find the correct receive_and_extract op to use when it is being lowered.
  // Increment the extract_idx for the next operation to use it.
  extract_op->setAttr("extract_idx", builder.getI8IntegerAttr(extract_idx));
  ind_store_op->setAttr("extract_idx", builder.getI8IntegerAttr(extract_idx));
  ++extract_idx;

  ops_to_be_deleted.push_back(store_op);
  ops_to_be_deleted.push_back(receive_op);

}
// ---- 217/384  lowerVectorLoadHelper  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2899  (46L)
LogicalResult e217_lowerVectorLoadHelper(
    VectorLoadTy load_op, Operation* store_op, ProgramUnitOp& unit,
    AccessContainer<AccessDetailsTy>& access_details,
    AccessContainer<Value>& mutable_addrs,
    AccessContainer<Value>& immutable_addrs,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  OpBuilder builder(load_op);
  if (!store_op) {
    DT_CHECK_MSG(access_details.size() == 1 && mutable_addrs.size() == 1 &&
                     immutable_addrs.size() == 1,
                 "single access info needed");
    if (constructLoadAndSendStmt<AccessDetailsTy>(
            &builder, unit, load_op, access_details[0], mutable_addrs[0],
            immutable_addrs[0])
            .failed())
      return load_op->emitError(
          "Unable to generate load_and_send '"
          "statement for the agen.vector_load operation");
    addLoadChainToDeleteList(load_op, to_be_deleted);
  } else {
    DT_CHECK_MSG(access_details.size() == 2 && mutable_addrs.size() == 2 &&
                     immutable_addrs.size() == 2,
                 "double access info needed");
    bool cond =
        (access_details.get(MemoryOperandIndex::kDirSrc).getTotalElements() ==
             access_details.get(MemoryOperandIndex::kDirDst)
                 .getTotalElements() &&
         access_details.get(MemoryOperandIndex::kDirSrc).getElementWidth() ==
             access_details.get(MemoryOperandIndex::kDirDst)
                 .getElementWidth() &&
         access_details.get(MemoryOperandIndex::kDirSrc).getChunkSize() ==
             access_details.get(MemoryOperandIndex::kDirDst).getChunkSize() &&
         access_details.get(MemoryOperandIndex::kDirSrc).getChunkStride() ==
             access_details.get(MemoryOperandIndex::kDirDst).getChunkStride());
    if (!cond)
      return load_op->emitError(
          "Access details of load and store operations are different");

    if (constructLoadAndStoreStmt<AccessDetailsTy>(
            &builder, unit, load_op, access_details, mutable_addrs,
            immutable_addrs)
            .failed())
      return load_op->emitError(
          "Unable to generate load_and_store '"
          "statement for the agen.vector_load operation");


}}
// ---- 218/384  lowerSetTransferMaskStateOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3815  (9L)
LogicalResult e218_lowerSetTransferMaskStateOp(
    dataflow::ProgramUnitOp unit, Operation* op) {
  auto mask_op = dyn_cast_or_null<SetTransferMaskStateOp>(op);
  DT_CHECK_MSG(mask_op, "expecting a SetTransferMaskStateOp");

  // Current support is only for SAMV.
  auto samv_op = constructSetActiveMaskValueOp(unit, mask_op);
  DT_CHECK_MSG(mask_op.use_empty(),
               "SetTransferMaskStateOps should not have users");

}
// ---- 219/384  cloneStartAddrOutsideLoop  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3942  (79L)
Value e219_cloneStartAddrOutsideLoop(
    Operation* loop_op, Value& memory_view_start_addr) {
  if (dyn_cast<BlockArgument>(memory_view_start_addr)) {
    auto parent_op = memory_view_start_addr.getParentRegion()->getParentOp();
    DT_CHECK(parent_op->isProperAncestor(loop_op));
    return memory_view_start_addr;
  }
  OpBuilder builder(loop_op);
  auto start_addr_parent_op = memory_view_start_addr.getDefiningOp();
  Operation* new_start_addr_op;
  if (loop_op->isProperAncestor(start_addr_parent_op)) {
    builder.setInsertionPoint(loop_op);
    if (isa<arith::ConstantOp>(start_addr_parent_op)) {
      new_start_addr_op = builder.clone(*start_addr_parent_op);
    } else if (auto query_map_op =
                   dyn_cast<uniform::QueryMapOp>(start_addr_parent_op)) {
      Value key;
      uniform::UniformizeRegionsOp new_uniformize_op;
      bool insert_uniformize_op = false;
      unsigned region_idx;
      if (mlir::dyn_cast<BlockArgument>(query_map_op.getKey())) {
        auto key_parent_op =
            mlir::dyn_cast<BlockArgument>(query_map_op.getKey())
                .getParentRegion()
                ->getParentOp();
        if (isa<uniform::UniformizeRegionsOp, uniform::EqualizePatternOp>(
                key_parent_op) &&
            !key_parent_op->isProperAncestor(loop_op)) {
          // if key is iter_arg defined in uniformize_regions and
          // equalize_pattern, need to copy it for the starting
          // address outside the loop
          insert_uniformize_op = true;
          region_idx = mlir::dyn_cast<BlockArgument>(query_map_op.getKey())
                           .getParentRegion()
                           ->getRegionNumber();
          new_uniformize_op = createUniformizeRegionsOp(&builder, key_parent_op,
                                                        loop_op, region_idx);
          key = new_uniformize_op.getRegionArg(region_idx);
          // set insertion point inside the region
          auto& block =
              new_uniformize_op.getRegion(region_idx).getBlocks().front();
          builder.setInsertionPointToStart(&block);
        } else {
          // key is program_unit's iter_arg
          key = query_map_op.getKey();
        }
      } else {
        query_map_op->emitError("unsupported key type");
        signalPassFailure();
        return nullptr;
      }
      // clone def_immutable_mapping if needed
      auto def_mapping_op = query_map_op.getMap().getDefiningOp();
      if (loop_op->isProperAncestor(def_mapping_op)) {
        def_mapping_op = builder.clone(*def_mapping_op);
      }
      // create new queryMapOp
      new_start_addr_op = uniform::QueryMapOp::create(
          builder, query_map_op->getLoc(), query_map_op.getType(),
          def_mapping_op->getResult(0), key);
      // set yieldOp if needed
      if (insert_uniformize_op) {
        SmallVector<Value, 1> operands = {{new_start_addr_op->getResult(0)}};
        auto yield_op = new_uniformize_op.getRegion(region_idx)
                            .getBlocks()
                            .front()
                            .getTerminator();
        yield_op->setOperands(operands);
        new_start_addr_op = new_uniformize_op;
      }
    } else {
      start_addr_parent_op->emitOpError(
          "unsupported operation for start address.");
      signalPassFailure();
      return nullptr;
    }
  } else {
    new_start_addr_op = start_addr_parent_op;
  }

}
// ---- 220/384  cleanupTriviallyRedundantSetSendDestination  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:4084  (38L)
void e220_cleanupTriviallyRedundantSetSendDestination(
    dataflow::ProgramUnitOp unit) {
  // We do not generate set_send_dst operations at arch levels that don't
  // support it.
  if (dccExtContext().getArch() < RCUDD1A_ISA) return;

  // Remove all of set_send_dst operations from the unit if no SFP bypass is
  // required.
  SmallVector<sentient::SetSendDestinationOp, 16> set_send_dst_ops;
  unit.walk([&](sentient::SetSendDestinationOp op) {
    set_send_dst_ops.push_back(op);
  });

  // If all set_send_dst operations set the same mode, we can replace them
  // all with one such operation at the start of the unit.
  if (set_send_dst_ops.empty()) return;
  sentient::SetSendDestinationOp lead_op = *set_send_dst_ops.begin();
  dataflow::GetUnitOp lead_unit =
      lead_op.getUnits().getDefiningOp<dataflow::GetUnitOp>();
  if (!lead_unit) return;
  if (llvm::all_of(set_send_dst_ops, [&](sentient::SetSendDestinationOp op) {
        dataflow::GetUnitOp op_unit =
            op.getUnits().getDefiningOp<dataflow::GetUnitOp>();
        if (!op_unit) return false;
        return op_unit->getAttrs() == lead_unit->getAttrs();
      })) {
    // If the mode is only ever set to the default (sfp), then we don't even
    // need to generate any setdstmask instructions.
    if (lead_unit.getType() != "sfp") {
      OpBuilder builder(unit);
      builder.setInsertionPointToStart(&unit.getRegion().front());
      auto cloned_getUnit = builder.clone(*lead_unit);
      builder.setInsertionPointAfter(cloned_getUnit);
      (void)sentient::SetSendDestinationOp::create(
          builder, unit->getLoc(), cloned_getUnit->getResult(0));
    }
    for (auto op : set_send_dst_ops) op.erase();
  }

}
// ---- 221/384  pushBackTheUnitToListIfDoesnotExist  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:143  (5L)
static void pushBackTheUnitToListIfDoesnotExist(std::string unit_name,
                                                SmallVector<Attribute, 1> &list,
                                                mlir::MLIRContext *context) {
  auto unit_attr = SentientLoadConsumerAttr::get(
      context, symbolizeSentientLoadConsumer(unit_name).value());

}
// ---- 222/384  createUniformRegionsWithTwoRegionsNoResult  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:153  (18L)
createUniformRegionsWithTwoRegionsNoResult(
    OpBuilder builder, Location loc, std::vector<mlir::Value> src_unit_res,
    llvm::SmallVector<Attribute, 2> list_sizes) {
  auto uniform_region = mlir::uniform::UniformizeRegionsOp::create(
      builder, loc, mlir::TypeRange(), src_unit_res,
      ArrayAttr::get(builder.getContext(), list_sizes), /*regIndices*/ nullptr,
      /*regLocales*/ nullptr, 2);
  auto &block0 = uniform_region.getRegion(0).emplaceBlock();
  block0.addArgument(builder.getIndexType(), loc);
  auto &block1 = uniform_region.getRegion(1).emplaceBlock();
  block1.addArgument(builder.getIndexType(), loc);
  OpBuilder builder_region0(uniform_region.getRegion(0));
  OpBuilder builder_region1(uniform_region.getRegion(1));
  uniform::YieldOp::create(builder_region0, loc);
  uniform::YieldOp::create(builder_region1, loc);
  builder_region0.setInsertionPointToStart(
      &uniform_region.getRegion(0).front());
  builder_region1.setInsertionPointToStart(

}
// ---- 223/384  lowerL3SyncOperationForAUnit  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:375  (57L)
LogicalResult e223_lowerL3SyncOperationForAUnit(
    mlir::Operation *op, OpBuilder builder,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops,
    mlir::dataflow::GetUnitOp dst_unit) {
  auto send_op = llvm::dyn_cast<dataflow::SyncSendOp>(op);
  auto recv_op = llvm::dyn_cast<dataflow::SyncRecvOp>(op);
  DT_CHECK(send_op || recv_op);
  SentientSyncMode sync_tag = SentientSyncMode::send;
  if (recv_op) sync_tag = SentientSyncMode::recv;

  if (send_op && !send_op.getWaitImmediatelyForAsyncTransfers().has_value()) {
    op->emitError("Unknown async transfers modes for sentient");
    return LogicalResult::failure();
  }

  auto dbg_name_attr = getDbgNameAttr(op);

  std::string dst_unit_name = dst_unit.getType().str();
  SenComponents dst_comp =
      EnumsConversion::stringToSenComponents.find(dst_unit_name)->second;
  std::string src_unit_name = getUnitNameFromAListOfGetUnitOp(src_unit_ops);

  SenComponents src_comp =
      EnumsConversion::stringToSenComponents.find(src_unit_name)->second;
  auto gen_comp = EnumsConversion::senCompToGenericComp.at(src_comp);
  if (gen_comp == SenComponents::L3LU || gen_comp == SenComponents::L3SU) {
    bool soft = false;
    if (send_op && !send_op.getWaitImmediatelyForAsyncTransfers().value())
      soft = true;

    if (is_any_of(dst_comp, SenComponents::L3LU, SenComponents::L3SU,
                  SenComponents::LXLU, SenComponents::LXSU,
                  SenComponents::LXLU0, SenComponents::LXSU0,
                  SenComponents::LXLU1, SenComponents::LXSU1)) {
      if ((dst_comp == SenComponents::LXLU ||
           dst_comp == SenComponents::LXSU) &&
          ExtendUnitNameToCorelet(dst_unit_name, dst_unit, builder).failed())
        return LogicalResult::failure();

      if (sentient::SyncOp::create(
              builder, op->getLoc(), dbg_name_attr,
              SentientSyncModeAttr::get(builder.getContext(), sync_tag),
              ArrayAttr::get(
                  builder.getContext(),
                  {SentientLoadConsumerAttr::get(
                      op->getContext(),
                      symbolizeSentientLoadConsumer(dst_unit_name).value())}),
              BoolAttr::get(op->getContext(), soft),
              builder.getSI32IntegerAttr(-1)))
        return LogicalResult::success();
      else {
        op->emitError("Cannot create sync operation");
        return LogicalResult::failure();
      }
    }
    op->emitError("Unknown lowering of the L3 sync operation");
    return LogicalResult::failure();

}}
// ---- 224/384  lowerL3SyncOperationForAGroupOfUnits  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:667  (61L)
DataflowToSentientLoweringPass::lowerL3SyncOperationForAGroupOfUnits(
    mlir::Operation *op, OpBuilder builder,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops,
    dataflow::CreateGroupOp dst_unit_group) {
  auto send_op = llvm::dyn_cast<dataflow::SyncSendOp>(op);
  auto recv_op = llvm::dyn_cast<dataflow::SyncRecvOp>(op);
  DT_CHECK(send_op || recv_op);
  SentientSyncMode sync_tag = SentientSyncMode::send;
  if (recv_op) sync_tag = SentientSyncMode::recv;

  if (send_op && !send_op.getWaitImmediatelyForAsyncTransfers().has_value()) {
    op->emitError("Unknown async transfers modes for sentient");
    return LogicalResult::failure();
  }

  auto dbg_name_attr = getDbgNameAttr(op);

  std::string src_unit_name = getUnitNameFromAListOfGetUnitOp(src_unit_ops);
  SenComponents src_comp =
      EnumsConversion::stringToSenComponents.find(src_unit_name)->second;
  auto gen_comp = EnumsConversion::senCompToGenericComp.at(src_comp);
  SmallVector<Attribute, 1> dst_unit_names;
  if (gen_comp == SenComponents::L3LU || gen_comp == SenComponents::L3SU) {
    bool soft = false;
    if (send_op && !send_op.getWaitImmediatelyForAsyncTransfers().value())
      soft = true;
    for (auto dst_unit_itr : dst_unit_group.getUnitIds()) {
      auto dst_unit =
          dcc::getDirectUnitOpOrgetFirstIndirectUnitOpViaQueryMap(dst_unit_itr);
      SenComponents dst_comp =
          EnumsConversion::stringToSenComponents.find(dst_unit.getType().str())
              ->second;
      if (dst_comp != SenComponents::L3LU && dst_comp != SenComponents::L3SU &&
          dst_comp != SenComponents::LXLU && dst_comp != SenComponents::LXSU &&
          dst_comp != SenComponents::LXLU0 &&
          dst_comp != SenComponents::LXSU0 &&
          dst_comp != SenComponents::LXLU1 &&
          dst_comp != SenComponents::LXSU1) {
        op->emitError("Unknown lowering of the L3 sync send operation");
        break;
      }
      std::string dst_unit_type_name = dst_unit.getType().str();
      if ((dst_comp == SenComponents::LXLU ||
           dst_comp == SenComponents::LXSU) &&
          ExtendUnitNameToCorelet(dst_unit_type_name, dst_unit, builder)
              .failed())
        break;

      auto dst_unit_attr = SentientLoadConsumerAttr::get(
          op->getContext(),
          symbolizeSentientLoadConsumer(dst_unit_type_name).value());
      if (std::find(dst_unit_names.begin(), dst_unit_names.end(),
                    dst_unit_attr) == dst_unit_names.end())
        dst_unit_names.push_back(dst_unit_attr);
    }
    if (sentient::SyncOp::create(
            builder, op->getLoc(), dbg_name_attr,
            SentientSyncModeAttr::get(builder.getContext(), sync_tag),
            ArrayAttr::get(op->getContext(), dst_unit_names),
            BoolAttr::get(op->getContext(), soft),
            builder.getSI32IntegerAttr(-1)))

}}
// ---- 225/384  matchAndRewrite  —  dcc/src/Conversion/SCFToSentient/SCFToSentient.cpp:72  (69L)
  mlir::LogicalResult matchAndRewrite(
      mlir::Operation *op, ArrayRef<mlir::Value> operands,
      mlir::ConversionPatternRewriter &rewriter) const final {
    SmallVector<Attribute, 4> reg_locales_vector;
    MLIRContext *context = op->getContext();
    auto scf_for_loop = llvm::dyn_cast<mlir::scf::ForOp>(op);
    auto lb = scf_for_loop.getLowerBound();
    auto ub = scf_for_loop.getUpperBound();
    auto step = scf_for_loop.getStep();
    auto range =
        mlir::arith::SubIOp::create(rewriter, scf_for_loop.getLoc(), ub, lb);
    auto nIterations =
        mlir::arith::DivSIOp::create(rewriter, range.getLoc(), range, step);

    //{} is required to maintain the scope and also reset the insertion point
    // after it exits the scope.
    // Replace SCF yield with Sentient yield.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToEnd(scf_for_loop.getBody());
      if (!llvm::hasSingleElement(scf_for_loop.getRegion())) {
        return scf_for_loop.emitError("expected scf.forop to have one block");
      }
      rewriter.replaceOpWithNewOp<sentient::YieldOp>(
          scf_for_loop.getBody()->getTerminator(),
          scf_for_loop.getBody()->getTerminator()->getOperands());
    }

    // Set the locale information
    /* Avoiding setting the LCCR information in the lowering.
     * It should be set in the registerTypeAssignment */
    /*    reg_locales_vector.push_back(StringAttr::get(context, "lccr"));*/

    for (int i = 0; i < op->getNumResults(); i++)
      reg_locales_vector.push_back(
          SentientRegTypeAttr::get(context, SentientRegType::unknown));
    ArrayAttr regLocales = ArrayAttr::get(context, reg_locales_vector);
    // Create Sentient for loop operation
    auto scf_dbg_name_attr = dataflow::getDbgNameAttr(op);
    auto loop = sentient::ForOp::create(
        rewriter, scf_for_loop.getLoc(), nIterations.getResult(),
        scf_dbg_name_attr, regLocales, ArrayRef<int32_t>(), ArrayRef<bool>(),
        scf_for_loop.getInitArgs(), nullptr);

    rewriter.inlineRegionBefore(scf_for_loop.getRegion(), loop.getRegion(),
                                loop.getRegion().begin());

    // Update the results of scf_for_loop
    scf_for_loop->replaceAllUsesWith(loop);

    rewriter.eraseOp(scf_for_loop);

    //{} is required to maintain the scope and also reset the insertion point
    // after it exits the scope.
    // Create a new iterator i' = Bound - i and substitute i' in place of i.
    {
      OpBuilder::InsertionGuard guard(rewriter);
      rewriter.setInsertionPointToStart(&loop.getRegion().front());
      auto new_iterator = mlir::arith::SubIOp::create(
          rewriter, loop.getRegion().getLoc(), nIterations.getResult(),
          loop.getInductionVar());

      llvm::SmallPtrSet<Operation *, 1> exception_list;
      exception_list.insert(exception_list.begin(), new_iterator);
      rewriter.modifyOpInPlace(loop, [&]() {
        loop.getInductionVar().replaceAllUsesExcept(new_iterator,
                                                    exception_list);
      });
    }

}
// ---- 226/384  createIfOpFromMapping  —  dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:123  (61L)
sentient::IfOp e226_createIfOpFromMapping(
    symbol::SymbolQueryMapOp query_map, int num_results) {
  // Set the builder location
  OpBuilder builder(query_map);

  // immutable_mapping's keys will be the RHS values in the comparisons, the
  // immutable_mapping's values will be the yielded values in the IfOp.
  auto immutable_mapping =
      query_map.getMap().getDefiningOp<symbol::SymbolImmutableMappingOp>();
  auto key_list = immutable_mapping.getKeys();
  auto val_list = immutable_mapping.getValues();
  DT_CHECK_MSG(key_list.size() >= 2,
               "Expect query_map with at least 2 mappings");

  sentient::IfOp sentient_ifop_tmp, new_if_op;
  auto then_builder = builder;
  auto else_builder = builder;
  bool is_first_pair = true;
  Value last_map_key = key_list.back();

  auto unknown_locale = mlir::sentient::SentientRegTypeAttr::get(
      builder.getContext(), SentientRegType::unknown);
  std::vector<Attribute> locale_list(num_results, unknown_locale);
  ArrayAttr locale_attr = builder.getArrayAttr(locale_list);
  llvm::SmallVector<Type> types(num_results, query_map.getResult().getType());

  for (auto key_val_pair : llvm::zip(key_list, val_list)) {
    Value rhs_val = std::get<0>(key_val_pair);
    // Handle the last pair separately, as it will correspond to the "else"
    // (unless there is exactly one pair).
    if (rhs_val == last_map_key) break;

    Value yielded_val = std::get<1>(key_val_pair);

    sentient_ifop_tmp = sentient::IfOp::create(
        else_builder, query_map->getLoc(), types,
        mlir::sentient::CmpIPredicateAttr::get(else_builder.getContext(),
                                               CmpIPredicate::eq),
        query_map.getKey(), rhs_val, locale_attr, ArrayRef<int32_t>(),
        /*has else branch*/ true);
    if (is_first_pair) {
      is_first_pair = false;
      new_if_op = sentient_ifop_tmp;
    } else
      sentient::YieldOp::create(else_builder, sentient_ifop_tmp.getLoc(),
                                sentient_ifop_tmp.getResults());

    else_builder.createBlock(&sentient_ifop_tmp.getThenRegion());
    then_builder = sentient_ifop_tmp.getThenBodyBuilder();
    // The then-branch will yield num_result of yielded_val
    std::vector<Value> yielded_vals(num_results, yielded_val);
    sentient::YieldOp::create(then_builder, sentient_ifop_tmp.getLoc(),
                              yielded_vals);
    else_builder.createBlock(&sentient_ifop_tmp.getElseRegion());
    else_builder = sentient_ifop_tmp.getElseBodyBuilder();
  }
  // The innermost else-branch will yield num_result of the last value in the
  // list.
  std::vector<Value> yielded_vals(num_results, val_list.back());
  sentient::YieldOp::create(else_builder, sentient_ifop_tmp.getLoc(),
                            yielded_vals);

}
// ---- 227/384  OperandReuse  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.hpp:30  (0L)
  ~OperandReuse() { data_origins_.clear(); }
// ---- 228/384  validateLoweringAndSetMissingParameters  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:483  (49L)
mlir::LogicalResult e228_validateLoweringAndSetMissingParameters(
    dataflow::ProgramUnitOp& unit, OperandReuse& reuse_info) {
  Builder builder(unit);
  int total_reuse_ids = reuse_info.getTotalDataOriginsCount();
  mlir::LogicalResult lowering_status = success();
  unit.walk<WalkOrder::PreOrder>([&](mlir::Operation* op) {
    if (isa<dataflow::ReceiveOp, dataflow::SendOp, vector::LoadOp,
            agen::VectorLoadOp, vector::StoreOp, agen::VectorStoreOp,
            vectorchain::MultiplyAndAccumulateOp, vectorchain::MultiplyOp,
            vectorchain::BinaryOp>(op)) {
      lowering_status = op->emitError(
          "Unable to lower the op into Sentient "
          "as part of the vectorchain lowering.\n");
      return WalkResult::interrupt();
    }

    if (auto mac_op = llvm::dyn_cast<sentient::MacOp>(op)) {
      if (mac_op.getOpADataID() == -1) {
        auto attr = builder.getSI32IntegerAttr(total_reuse_ids);
        mac_op.setOpADataIDAttr(attr);
        total_reuse_ids++;
      }

      if (mac_op.getOpBDataID() == -1) {
        auto attr = builder.getSI32IntegerAttr(total_reuse_ids);
        mac_op.setOpBDataIDAttr(attr);
        total_reuse_ids++;
      }

      if (mac_op.getOpCDataID() == -1) {
        auto attr = builder.getSI32IntegerAttr(total_reuse_ids);
        mac_op.setOpCDataIDAttr(attr);
        total_reuse_ids++;
      }
    } else if (auto bin_op = llvm::dyn_cast<sentient::BinaryOp>(op)) {
      if (bin_op.getOpADataID() == -1) {
        auto attr = builder.getSI32IntegerAttr(total_reuse_ids);
        bin_op.setOpADataIDAttr(attr);
        total_reuse_ids++;
      }

      if (bin_op.getOpBDataID() == -1) {
        auto attr = builder.getSI32IntegerAttr(total_reuse_ids);
        bin_op.setOpBDataIDAttr(attr);
        total_reuse_ids++;
      }
    }
    return WalkResult::advance();
  });

}
// ---- 229/384  getMaskValueForNonPT  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:114  (9L)
std::optional<Value> getMaskValueForNonPT(Operation* op, Operation* operand,
                                          const SenSystemDef& sys_def,
                                          BuilderType& builder) {
  std::optional<unsigned> mask_val =
      getMaskValueConstantForNonPT(op, operand, sys_def);
  if (mask_val.has_value()) {
    builder.setInsertionPoint(op);
    return sentient::ConstantOp::create(
        builder, op->getLoc(), builder.getIndexType(), mask_val.value());

}}
// ---- 230/384  convertStringToType  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:196  (24L)
static Type convertStringToType(OpBuilder& builder,
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
  else if (type_string == "mxfp4")
    return dataflow::CustomMXFloatType::get(builder.getContext(), 4);
  else if (type_string == "mxfp8")
    return dataflow::CustomMXFloatType::get(builder.getContext(), 8);
  else if (type_string == "mxint4")
    return dataflow::CustomMXIntType::get(builder.getContext(), 4);
  else if (type_string == "i16")
    return builder.getIntegerType(16);
  else if (type_string == "i8")
    return builder.getIntegerType(8);
  else if (type_string == "i4")
    return builder.getIntegerType(4);

}
// ---- 231/384  convertTypeToString  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:223  (22L)
static std::string convertTypeToString(OpBuilder& builder, Type type) {
  if (type == builder.getF16Type())
    return "fp16";
  else if (type == builder.getBF16Type())
    return "bf16";
  else if (type == builder.getF32Type())
    return "fp32";
  else if (llvm::isa<mlir::Float8E4M3FNType>(type))
    return "f8E4M3FN";
  else if (llvm::isa<mlir::Float8E5M2Type>(type))
    return "f8E5M2";
  else if (type == dataflow::CustomMXFloatType::get(builder.getContext(), 4))
    return "mxfp4";
  else if (type == dataflow::CustomMXFloatType::get(builder.getContext(), 8))
    return "mxfp8";
  else if (type == builder.getIntegerType(16))
    return "i16";
  else if (type == builder.getIntegerType(8))
    return "i8";
  else if (type == builder.getIntegerType(4))
    return "i4";
  llvm_unreachable("unknown type");

}
// ---- 232/384  getOperandFromLoadOrStoreOp  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:153  (94L)
std::optional<VectorOperand> e232_getOperandFromLoadOrStoreOp(
    Operation *op) {
  std::string result;
  llvm::SmallVector<Value> indices;
  dataflow::GetLogicalMemoryViewOp logical_view_op;
  AffineMap layout_map;
  if (failed(
          getLayoutMapAndIndices(op, layout_map, indices, logical_view_op))) {
    op->emitError("op needs to be either vector load or store");
    return std::nullopt;
  }

  if (layout_map.getNumResults() != 1) {
    logical_view_op->emitError(
        "Layout map of logical memory view "
        "should map to a 1D view of memory for LRF/XRF/IRF.");
    return std::nullopt;
  }

  // find regfile type
  std::string memory_unit_name;
  std::optional<std::string> unit_str_optional =
      dcc::uniform::utils::findUnitType(logical_view_op.getFromUnit());
  if (unit_str_optional.has_value()) {
    memory_unit_name = unit_str_optional.value();
  } else {
    logical_view_op.emitOpError("Unit type is inconsistent in memory view.");
    return std::nullopt;
  }
  affine::fullyComposeAffineMapAndOperands(&layout_map, &indices);
  if (layout_map.getNumInputs() == indices.size()) {
    affine::canonicalizeMapAndOperands(&layout_map, &indices);
  }

  VectorOperandType operand_type;
  auto record = EnumsConversion::stringToSenComponents.find(memory_unit_name);
  if (record != EnumsConversion::stringToSenComponents.end()) {
    if (record->second == PTXRF) {
      operand_type = XRF;
    } else if (record->second == PTIRF) {
      operand_type = IRF;
    } else if (dcc::utils::isLRFReg(memory_unit_name)) {
      operand_type = LRF;
    } else if (is_any_of(record->second, PESTATE, SFPSTATE)) {
      operand_type = ISTATE;
    } else {
      std::cout << "Type: " << memory_unit_name << std::endl;
      op->emitError("Unknown memory type");
      return std::nullopt;
    }
  } else {
    std::cout << "Type: " << memory_unit_name << std::endl;
    op->emitError("Unknown memory type");
    return std::nullopt;
  }

  if (operand_type != XRF && !layout_map.isSingleConstant()) {
    op->dump();
    layout_map.dump();
    logical_view_op->dump();
    op->emitError(
        "Store indices should lead to "
        "constant indices in the 1D memory for LRF/IRF");
    return std::nullopt;
  }

  if (operand_type == XRF) {
    return VectorOperand(operand_type, "xrf", op);
  }

  int offset_per_slice = 1024;  // bits
  auto bit_width = dataflow::utils::getElementTypeBitWidth(
      logical_view_op.getResult().getType());

  // TODO: this if statement must be dropped due to F80Type workaround
  if (bit_width == 80) {
    bit_width = 8;
  } else if (bit_width == 24) {
    bit_width = 16;
  }

  // Start address is in always Bytes
  if (auto const_start_address = llvm::dyn_cast<mlir::arith::ConstantIndexOp>(
          logical_view_op.getStartAddress().getDefiningOp())) {
    auto start_address = const_start_address.value();
    auto total_address =
        (start_address + layout_map.getSingleConstantResult()) * bit_width;
    auto access = (total_address) / offset_per_slice;
    return VectorOperand(operand_type, std::to_string(access), op);
  } else {
    logical_view_op->emitError(
        "Only constant start address "
        "are supported for LRF/XRF/IRF/ISTATE in PT");
    return std::nullopt;

}}
// ---- 233/384  eraseOperands  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:690  (46L)
void e233_eraseOperands(
    llvm::SmallVectorImpl<std::optional<VectorOperand>> &operands) {
  std::vector<mlir::Operation *> erased_list;
  for (auto &operand : operands) {
    if (std::find(erased_list.begin(), erased_list.end(),
                  operand.value().op_) == erased_list.end()) {
      if (!operand.has_value()) continue;
      bool all_uses_deleted = true;
      llvm::SmallVector<mlir::Operation *> to_be_erased;
      llvm::SmallVector<mlir::Operation *> intermediate_ops;
      for (auto user : operand.value().op_->getUsers()) {
        // There could be an intermediate operations between the operand and the
        // op feeding the compute op (e.g. NegOp, cast, SelectOp). Keep track of
        // them.
        while (user &&
               isa<arith::SIToFPOp, arith::FPToSIOp, vectorchain::CastOp,
                   vectorchain::NegOp, vectorchain::SelectOp>(user) &&
               user->hasOneUse()) {
          intermediate_ops.push_back(user);
          user = *user->getUsers().begin();
        }

        if (user && user->getUses().empty()) {
          user->dropAllUses();
          to_be_erased.push_back(user);
          for (auto op : intermediate_ops) {
            op->dropAllUses();
            to_be_erased.push_back(op);
          }
        } else {
          all_uses_deleted = false;
        }
      }

      for (auto e : to_be_erased) {
        erased_list.push_back(e);
        e->erase();
      }

      if (all_uses_deleted) {
        if (operand.value().getName() != "latch") {
          erased_list.push_back(operand.value().op_);
          operand.value().op_->erase();
        }
      }
    }

}}
// ---- 234/384  setValue  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.hpp:72  (3L)
  void setValue(std::string val) {
    values_.clear();
    values_.emplace_back(val);

}
// ---- 235/384  createSentientConstants  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:34  (31L)
mlir::Value createSentientConstants(
    mlir::ConversionPatternRewriter &rewriter,
    vectorchain::ConstantBitstreamOp const_bit_op,
    vectorchain::ShuffleOp shuffle_op) {
  mlir::Type elements_type =
      dataflow::utils::getElementType(const_bit_op.getType());
  llvm::SmallVector<int, 8> vals;
  for (auto val_attr : const_bit_op.getValue())
    vals.push_back(mlir::cast<IntegerAttr>(val_attr).getInt());

  // DT_CHECK(vals.size() > 0);
  //  if the constant bitstream has one value use scalar constant otherwise
  //  vector constant to represent in Sentient IR.
  if (vals.size() == 1) {
    auto sentient_const_op = sentient::ConstantOp::create(
        rewriter, const_bit_op->getLoc(), elements_type, vals[0]);
    if (const_bit_op->hasAttr("is_symbol"))
      sentient_const_op->setAttr("is_symbol", rewriter.getBoolAttr(true));
    return sentient_const_op.getResult();
  } else {
    auto total_elements =
        128 / dataflow::utils::getIntOrFloatBitWidth(elements_type);
    auto vec_type = VectorType::get({total_elements}, elements_type);
    llvm::SmallVector<int, 8U> extended_vals;
    DT_CHECK(total_elements == shuffle_op.getIndices().size());
    for (auto idx_attr : shuffle_op.getIndices()) {
      auto idx = mlir::cast<IntegerAttr>(idx_attr).getInt();
      extended_vals.push_back(vals[idx]);
    }
    auto sentient_vconst_op = sentient::VectorConstantOp::create(
        rewriter, const_bit_op->getLoc(), vec_type,

}}
// ---- 236/384  addMaskNode  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:136  (16L)
void e236_addMaskNode(Operation *loop_op, Operation *mask_related_op,
                               int start_val, int increment) {
  MaskNode *mask_node = new MaskNode(mask_related_op, start_val, increment);
  if (loop_op) {
    DT_CHECK_MSG(isa<sentient::ForOp>(loop_op), "expecting a sentient::ForOp");

    LoopMaskNode *n = getRoot()->getFirstChild();
    LoopMaskNode *found_node = findNodeFromOp(loop_op);
    DT_CHECK_MSG(found_node, "could not locate loop_op in tree");
    found_node->insertChildNode(mask_node);
  } else {
    LoopMaskNode *n = getRoot();
    n->insertChildNode(mask_node);
  }
  DT_CHECK_MSG(op_to_node_.find(mask_related_op) == op_to_node_.end(),
               "op already in map - should not happen");

}
// ---- 237/384  updateNode  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:155  (10L)
void e237_updateNode(Operation *from, Operation *to) {
  DT_CHECK_MSG(from && to, "expecting valid from and to operations");
  DT_CHECK_MSG(op_to_node_.find(from) != op_to_node_.end(),
               "could not find from op in LoopMaskTree");
  auto from_node = op_to_node_[from];
  from_node->setOperation(to);

  // Update op_to_node_ map.
  op_to_node_.insert(std::make_pair(to, from_node));
  op_to_node_.erase(from);

}
// ---- 238/384  computeLoops  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.cpp:173  (18L)
void e238_computeLoops(dataflow::ProgramUnitOp &unit) {
  DT_CHECK_MSG(!root_ && op_to_node_.empty(), "expecting tree to be empty");
  root_ = new LoopMaskNode(nullptr);
  unit.walk<WalkOrder::PreOrder>([&](Operation *op) {
    if (!isa<sentient::ForOp>(op)) return;
    LMTLoopNode *new_node = new LMTLoopNode(op);
    op_to_node_.insert(std::make_pair(op, new_node));
    Operation *parent_op = op->getParentOfType<sentient::ForOp>();
    LoopMaskNode *parent_node = getRoot();
    if (parent_op) {
      // Since we walk in pre-order, we should have created LoopMaskNode objects
      // for the parents already.
      DT_CHECK_MSG(op_to_node_.find(parent_op) != op_to_node_.end(),
                   "expected to find parent_op in the map");
      parent_node = op_to_node_[parent_op];
    }
    parent_node->insertChildNode(new_node);
  });

}
// ---- 239/384  insertPTMaskOps  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:41  (164L)
void e239_insertPTMaskOps(
    LoopMaskTree *pt_masking_tree) {
  // Using the information of the given mask node, insert sentient.set_mask and
  // sentient.incrmask operations to the program.
  auto insertMaskOps = [&](MaskNode *n) {
    auto op = n->getOperation();
    auto start_val = n->getStartVal();
    auto increment = n->getIncrement();

    if (increment == 0) {
      // CONSTANT MASK
      // Insert "sentient.set_mask <start_val>" before the op.
      OpBuilder builder(op);
      auto mask_val = sentient::ConstantOp::create(
          builder, op->getLoc(), builder.getIndexType(), start_val);
      (void)sentient::SetMaskOp::create(
          builder, op->getLoc(), mask_val,
          dcc::utils::getNewDbgNameFromOpOrCount(
              /*prefix*/ "PTSetMask", op, ++set_incr_mask_count_));

      // Insert "sentient.set_mask 0" after the op to reset the mask.
      builder.setInsertionPointAfter(op);
      auto reset_mask_val = sentient::ConstantOp::create(
          builder, op->getLoc(), builder.getIndexType(), 0);
      (void)sentient::SetMaskOp::create(
          builder, op->getLoc(), reset_mask_val,
          dcc::utils::getNewDbgNameFromOpOrCount(
              /*prefix*/ "PTSetMask", op, ++set_incr_mask_count_));
    } else {
      // NON-CONST MASK
      DT_CHECK_MSG(increment == 1, "increments > 1 not currently supported");

      // Insert "sentient.set_mask <start_val>" before the loop controlling the
      // mask.
      auto parent_loop = n->getParentNode()->getOperation();
      OpBuilder builder(parent_loop);

      auto mask_val = sentient::ConstantOp::create(
          builder, op->getLoc(), builder.getIndexType(), start_val);
      (void)sentient::SetMaskOp::create(
          builder, op->getLoc(), mask_val,
          dcc::utils::getNewDbgNameFromOpOrCount(
              /*prefix*/ "PTSetMask", parent_loop, ++set_incr_mask_count_));

      // Insert "sentient.incrmask" after the loop controlling the mask.
      // TODO: When we support increments greater than 1, we will need to wrap
      // the incrmask op with a loop to increment the correct number of times.
      auto for_op = cast<sentient::ForOp>(parent_loop);
      builder.setInsertionPoint(for_op.getRegion().back().getTerminator());
      (void)sentient::IncrMaskOp::create(
          builder, op->getLoc(),
          dcc::utils::getNewDbgNameFromOpOrCount(
              /*prefix*/ "PTIncrMask", parent_loop, ++set_incr_mask_count_));

      // Insert "sentient.set_mask 0" after the loop controlling the mask to
      // reset the mask.
      builder.setInsertionPointAfter(parent_loop);
      auto reset_mask_val = sentient::ConstantOp::create(
          builder, op->getLoc(), builder.getIndexType(), 0);
      (void)sentient::SetMaskOp::create(
          builder, op->getLoc(), reset_mask_val,
          dcc::utils::getNewDbgNameFromOpOrCount(
              /*prefix*/ "PTSetMask", parent_loop, ++set_incr_mask_count_));
    }
  };

  // Track the visited_nodes so we don't reanalyze entire subtrees multiple
  // times.
  std::unordered_set<LoopMaskNode *> visited_nodes;

  // Verify the loop trees from the loop children of the given node do not
  // contain any other mask nodes.
  // Add nodes encountered to the visited_nodes set.
  auto verifyLoopNest = [&](LoopMaskNode *node, MaskNode *mask_node) -> bool {
    DT_CHECK_MSG(node->isLoopNode(), "expecting a loop node");
    bool found_mask_node = false;
    auto curr_node = node->getFirstChild();
    while (curr_node) {
      if (curr_node->isLoopNode()) {
        LoopMaskNode::walk<OperationNode::WalkOrder::kBFS>(
            node, [&](LoopMaskNode *n) -> LoopMaskNode * {
              visited_nodes.emplace(n);
              if (n->isMaskNode()) {
                // If the mask is equivalent, it is allowed.
                auto m = static_cast<MaskNode *>(n);
                if (!m->isMaskEquivalentToNode(mask_node)) {
                  found_mask_node = true;
                  WalkResult::interrupt();
                }
              }
              return n;
            });
      }
      if (found_mask_node) break;
      curr_node = curr_node->getNextSibling();
    }
    return !found_mask_node;
  };

  // Given node n, analyze the children of n looking for mask nodes.
  // If a mask node indicates a non-default constant mask, insert the
  // appropriate operations. If a mask node indicates a non-constant mask,
  // verify the loop nests of the sibling loop nodes do not contain other masks
  // before inserting the appropriate operations.
  //
  // Example:
  // In the following example, {x,y} describes a mask where x is the start value
  // and y is the increment. Non-zero increment indicates a dynamic mask
  // dependent on the parent of the node.
  //
  //
  //             A:[root]
  //                /  \
  //       B:[loop i]  C:[mask {2,0}]
  //          /    \
  //  D:[loop j]   E:[mask {0,1}]
  //       |
  //  F:[mask {0,1}]
  //
  // When analyzing loop node B, we would see child mask node E. Determining
  // that node E is a dynamic mask, we check the loop nest rooted at loop node
  // D. We find another mask node so the program is not supported and we signal
  // pass failure.
  auto analyzeAndInsertMaskOps = [&](LoopMaskNode *n) -> LoopMaskNode * {
    if (!n) return nullptr;
    if (!n->isLoopNode() && n != pt_masking_tree->getRoot()) return n;

    // Look at the children of the current node.
    // Either all the masks directly under this node are constant masks or
    // they are all identical non-constant masks.
    auto curr_child = n->getFirstChild();
    while (curr_child) {
      if (visited_nodes.find(curr_child) != visited_nodes.end()) return n;
      if (!curr_child->isMaskNode()) {
        curr_child = curr_child->getNextSibling();
        continue;
      }
      auto *mask_node = static_cast<MaskNode *>(curr_child);
      auto start_val = mask_node->getStartVal();
      auto increment = mask_node->getIncrement();
      // If increment is 0, mask is constant. Add the sentient.set_mask
      // operation and set const_mask.
      if (increment == 0) {
        //  Only insert mask ops for non-default masks.
        if (start_val != 0) insertMaskOps(mask_node);
      } else {
        // If increment > 0, mask is dynamic. Add the sentient.set_mask and
        // sentient.incrmask operations.
        DT_CHECK_MSG(increment == 1, "increments > 1 not currently supported");
        if (verifyLoopNest(n, mask_node))
          insertMaskOps(mask_node);
        else {
          signalPassFailure();
          mask_node->getOperation()->emitOpError(
              "Unsupported: Multiple MACs generate different PT masks for "
              "the same loop nest");
          return nullptr;
        }
      }
      curr_child = curr_child->getNextSibling();
    }
    return n;
  };


}
// ---- 240/384  getLayoutExpr  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:28  (67L)
LayoutExpr e240_getLayoutExpr(Operation *op, int stick_elem_num) {
  LayoutExpr layout_expr;

  // extract layout expression from memref
  mlir::AffineMap logical_view_map;
  SmallVector<Value> operands;
  dataflow::GetLogicalMemoryViewOp logical_view_op;
  auto result =
      getLayoutMapAndIndices(op, logical_view_map, operands, logical_view_op);
  affine::fullyComposeAffineMapAndOperands(&logical_view_map, &operands);
  if (logical_view_map.getNumInputs() == operands.size()) {
    logical_view_map = simplifyAffineMap(logical_view_map);
    affine::canonicalizeMapAndOperands(&logical_view_map, &operands);
  }

  DT_CHECK(logical_view_map.getNumResults() == 1);
  auto expr = logical_view_map.getResult(0);
  SmallVector<int64_t, 8> layout_coeffs;
  affine::FlatAffineValueConstraints *layout_flat_csts = nullptr;
  auto flat_result = getFlattenedAffineExpr(expr, operands.size(),
                                            logical_view_map.getNumSymbols(),
                                            &layout_coeffs, layout_flat_csts);
  DT_CHECK(layout_coeffs.size() == operands.size() + 1);

  for (int i = 0; i < operands.size(); i++) {
    // TODO need to reorder passes to avoid sub_op
    auto sub_op = operands[i].getDefiningOp<mlir::arith::SubIOp>();
    if (sub_op) {
      auto for_op = sub_op.getRhs().getParentRegion()->getParentOp();
      DT_CHECK_MSG(isa<sentient::ForOp>(for_op),
                   "indices must be constant or for loop iterator - could not "
                   "extract layout info");
      layout_expr.layout_map[for_op] =
          layout_coeffs[i] / stick_elem_num;  // converted to stick number
    }
  }

  int const_val = layout_coeffs.back();
  if (auto const_base_op = llvm::dyn_cast<mlir::arith::ConstantIndexOp>(
          logical_view_op.getStartAddress().getDefiningOp())) {
    const_val += const_base_op.value();
  } else {
    DT_CHECK_MSG(false,
                 "the base address for xrf logical memory view "
                 "should be constants for lowering");
  }

  layout_expr.constant_val += const_val / stick_elem_num;

  // identity the nested loops where op sits starting from the innermost loop
  // and add them to layout_expr table if not existing
  auto curr_region = op->getParentRegion();
  while (curr_region) {
    auto parent_op = curr_region->getParentOp();
    if (parent_op && isa<sentient::ForOp>(parent_op)) {
      auto for_iterator =
          dyn_cast<sentient::ForOp>(parent_op).getInductionVar();
      // loop iterator var that is not in layout expression has zero for
      // coefficient.
      if (layout_expr.layout_map.count(parent_op) == 0) {
        layout_expr.layout_map[parent_op] = 0;
      }
    }
    curr_region = parent_op->getParentRegion();
  }

  return layout_expr;

}
// ---- 241/384  createForOpWithReturnValue  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:129  (56L)
sentient::ForOp e241_createForOpWithReturnValue(
    sentient::ForOp for_op) {
  OpBuilder builder(for_op);

  // create a dummy constant op as input operand
  Type xrf_reg_type = IndexType::get(for_op->getContext());
  SentientRegType xrf_wr_ptr_name = SentientRegType::xrfwrptr;
  auto xrf_wr_const_op = sentient::ConstantOp::create(
      builder, for_op.getLoc(), xrf_reg_type, 0, xrf_wr_ptr_name);
  SentientRegType xrf_rd_ptr_name = SentientRegType::xrfrdptr;
  auto xrf_rd_const_op = sentient::ConstantOp::create(
      builder, for_op.getLoc(), xrf_reg_type, 0, xrf_rd_ptr_name);
  SmallVector<Value, 8> iter_args;
  for (auto operand : for_op.getIterOperands()) {
    iter_args.push_back(operand);
  }
  iter_args.push_back(xrf_wr_const_op.getOut());
  iter_args.push_back(xrf_rd_const_op.getOut());

  StringAttr dbg_name = nullptr;
  if (for_op.getDbgName().has_value())
    dbg_name = builder.getStringAttr(for_op.getDbgName().value());
  auto new_for_op = sentient::ForOp::create(
      builder, for_op->getLoc(), for_op.getBound(), dbg_name,
      for_op.getRegLocales(), ArrayRef<int32_t>(), ArrayRef<bool>(), iter_args);

  // temporary fix to add loop body to overcome sentient::forop build issue.
  // ****************************************************
  // auto bodyRegion = new_for_op.getRegion();
  new_for_op.getRegion().push_back(new Block);
  Block &bodyBlock = new_for_op.getRegion().front();
  bodyBlock.addArgument(builder.getIndexType(), new_for_op.getLoc());
  for (Value v : iter_args) bodyBlock.addArgument(v.getType(), v.getLoc());
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPointToStart(&bodyBlock);
  sentient::YieldOp::create(builder, for_op->getLoc());
  // ****************************************************

  builder.setInsertionPointToStart(new_for_op.getBody());

  // Clone the body
  IRMapping bv_map;
  bv_map.map(for_op.getBody()->getArguments(),
             new_for_op.getBody()->getArguments());

  for (auto &it : for_op.getBody()->getOperations()) {
    if (!isa<sentient::YieldOp>(&it)) auto new_op = builder.clone(it, bv_map);
  }

  // Replace uses if exists
  for (int i = 0; i < for_op.getNumResults(); i++) {
    for_op.getResult(i).replaceAllUsesWith(new_for_op.getResult(i));
  }

  // Erase the loop
  for_op.erase();

}
// ---- 242/384  createIfOpWithReturnValue  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:189  (56L)
sentient::IfOp e242_createIfOpWithReturnValue(sentient::IfOp if_op) {
  OpBuilder builder(if_op);

  auto new_if_op = sentient::IfOp::create(
      builder, if_op->getLoc(),
      TypeRange({IndexType::get(if_op->getContext()),
                 IndexType::get(if_op->getContext())}),
      if_op.getPredicateAttr(), if_op.getLhs(), if_op.getRhs(),
      if_op.getRegLocales(), ArrayRef<int32_t>(), true);
  if (auto orig_dbg_name = dataflow::getDbgNameAttr(if_op))
    dataflow::setDbgNameAttr(new_if_op, orig_dbg_name);

  // temporary fix to add then/else body to overcome sentient::ifop build issue.
  // ****************************************************
  // then body
  new_if_op.getThenRegion().push_back(new Block);
  Block &bodyBlock = new_if_op.getRegion(0).front();
  builder.setInsertionPointToStart(&bodyBlock);
  sentient::YieldOp::create(builder, if_op->getLoc());
  // else body
  new_if_op.getElseRegion().push_back(new Block);
  Block &elseBlock = new_if_op.getRegion(1).front();
  builder.setInsertionPointToStart(&elseBlock);
  sentient::YieldOp::create(builder, if_op->getLoc());
  // ****************************************************

  // Clone the then body
  builder.setInsertionPointToStart(&new_if_op.getThenRegion().front());
  IRMapping bv_map;
  bv_map.map(if_op.getBody(0)->getArguments(),
             new_if_op.getBody(0)->getArguments());

  for (auto &it : if_op.getBody()->getOperations()) {
    if (!isa<sentient::YieldOp>(&it)) auto new_op = builder.clone(it, bv_map);
  }

  // Don't consider the else region if it is empty.
  if (!if_op.getElseRegion().empty()) {
    // Clone the else body
    builder.setInsertionPointToStart(&new_if_op.getElseRegion().front());
    IRMapping bv_map_else;
    bv_map_else.map(if_op.getBody(1)->getArguments(),
                    new_if_op.getBody(1)->getArguments());

    for (auto &it : if_op.getBody(1)->getOperations()) {
      if (!isa<sentient::YieldOp>(&it))
        auto new_op = builder.clone(it, bv_map_else);
    }
  }
  // Replace uses if exists
  for (int i = 0; i < if_op.getNumResults(); i++) {
    if_op.getResult(i).replaceAllUsesWith(new_if_op.getResult(i));
  }
  // Erase the loop
  if_op.erase();
  return new_if_op;

}
// ---- 243/384  insertDummyMacOp  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:312  (15L)
Value e243_insertDummyMacOp(OpBuilder *builder, MLIRContext *context,
                                    Location &loc) {
  auto mask_const_op =
      sentient::ConstantOp::create(*builder, loc, builder->getIndexType(), 0);
  auto ret = sentient::MacOp::create(
      *builder, loc, TypeRange(IndexType::get(context)), mask_const_op,
      ValueRange(),
      StringAttr::get(context, StringRef("LoweringXRF dummy Mac")),
      symbolizeSentientComputePort("lrf0").value(),
      symbolizeSentientComputePort("lrf0").value(),
      symbolizeSentientComputePort("lrf0").value(),
      ArrayAttr::get(context, SmallVector<Attribute, 4>()),
      ArrayAttr::get(context, SmallVector<Attribute, 4>()),
      ArrayAttr::get(context, SmallVector<Attribute, 4>()),
      ArrayAttr::get(context, SmallVector<Attribute, 4>()), nullptr);

}
// ---- 244/384  isHoistable  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:82  (9L)
bool e244_isHoistable(Operation *to_hoist,
                                              Block *then_or_else_block) {
  if (to_hoist->getBlock() != then_or_else_block ||
      CFGSDataflowConditionalTree::opHasSideEffect(*to_hoist))
    return false;
  for (auto &op : then_or_else_block->getOperations()) {
    if (&op == to_hoist) break;
    if (CFGSDataflowConditionalTree::opHasSideEffect(op)) return false;
  }

}
// ---- 245/384  enumerateCollectionUnit  —  dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:34  (57L)
void e245_enumerateCollectionUnit(
    ModuleOp &moduleOp, dataflow::ProgramCollectionOp &collectionDefinitionOp) {
  auto type =
      mlir::dyn_cast<VectorType>(collectionDefinitionOp.unit().getType());
  if (!type) {
    return;
  }

  int nUnits = type.getShape()[0];

  // No body present inside the op
  if (collectionDefinitionOp.getBody()->empty() ||
      collectionDefinitionOp.getBody()->begin() ==
          std::prev(collectionDefinitionOp.getBody()->end())) {
    return;
  }

  // Builder to insert single units
  OpBuilder builder(collectionDefinitionOp);
  auto intType = IntegerType::get(builder.getContext(), 32);
  const char *collectionName = collectionDefinitionOp.unit()
                                   .getDefiningOp<GetUnitCollectionOp>()
                                   .nameAttr()
                                   .getValue()
                                   .data();

  // 1. Replace GetMyUnit to appropriate number
  // 2. Clone the bodies
  // 3. Do constant propagation and dead code elimination if possible
  for (int32_t i = 0; i < nUnits; i++) {
    std::string name = std::string(collectionName) + "-" + std::to_string(i);
    StringRef unitName = StringRef(name);
    auto unitDeclarationOp = dataflow::GetUnitOp::create(
        builder, collectionDefinitionOp.getLoc(), intType, unitName);

    auto constIntOp = ConstantIntOp::create(
        builder, collectionDefinitionOp.getLoc(), i, intType);

    auto programUnitOp = ProgramUnitOp::create(
        builder, unitDeclarationOp.getLoc(), unitDeclarationOp,
        [&](OpBuilder &nestedBuilder, Location /*loc*/,
            ValueRange /*valueRange*/) {
          BlockAndValueMapping operandMap;
          for (auto &it : *collectionDefinitionOp.getBody()) {
            auto *newOp = it.clone(operandMap);
            nestedBuilder.insert(newOp);
          }
        });

    SmallVector<dataflow::GetMyUnitInCollectionOp, 16> workList;
    programUnitOp.walk(
        [&](dataflow::GetMyUnitInCollectionOp op) { workList.push_back(op); });

    for (auto op : workList) {
      op.replaceAllUsesWith(constIntOp.getResult());
      op->erase();
    }

}}
// ---- 246/384  FlatteningLocalRegionsTree  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:79  (0L)
  ~FlatteningLocalRegionsTree() { clear(); }
// ---- 247/384  compute  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:151  (15L)
void e247_compute(LocalOpNode *node) {
  auto uniform_op =
      llvm::dyn_cast<uniform::UniformizeRegionsOp>(node->getOperation());
  if (!uniform_op) return;
  // get the list of units per region
  std::vector<std::vector<mlir::Value>> units_for_each_region;
  dcc::uniform::utils::getUnitsPerRegionsAsVectorOfVector(
      uniform_op, units_for_each_region);

  // for each region, call traverseRegion with its unit list
  DT_CHECK(units_for_each_region.size() == uniform_op.getNumRegions());
  for (int idx = 0; idx < uniform_op.getNumRegions(); idx++) {
    traverseRegion(uniform_op.getRegion(idx), node,
                   units_for_each_region.at(idx), false);
  }

}
// ---- 248/384  runOnOperation  —  dcc/src/Transform/Dataflow/LoopUnrollForShuffleOp.cpp:70  (68L)
void e248_runOnOperation() {
  if (DisableThisPass) return;

  ModuleOp module_op = getOperation();

  module_op.walk([&](dataflow::ProgramUnitOp unit) {
    auto comp = dcc::getUnitType(
        unit.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
    if (comp != LXLU) return;

    // Repeatedly search for and unroll loops that need unrolling.
    // This approach avoids iterator invalidation by doing a fresh walk
    // after each unroll operation.
    while (true) {
      Operation *candidate = nullptr;

      // Look for a loop that needs unrolling.
      // Walk the unit looking for ShuffleOps with loop iterators used in
      // their variable operand.
      unit.walk<WalkOrder::PreOrder>([&](vectorchain::ShuffleOp shuffle_op) {
        // Look at the variable operand of the shuffle. Check if any loop
        // iterator is used.
        for (auto variable : shuffle_op.getVariable()) {
          if (auto block_arg = dyn_cast<BlockArgument>(variable)) {
            auto for_op = block_arg.getParentRegion()->getParentOp();

            if (auto scf_for = dyn_cast<scf::ForOp>(for_op)) {
              if (block_arg == scf_for.getInductionVar()) {
                candidate = scf_for.getOperation();
                return WalkResult::interrupt();
              }
            } else if (auto affine_for =
                           dyn_cast<affine::AffineForOp>(for_op)) {
              if (block_arg == affine_for.getInductionVar()) {
                candidate = affine_for.getOperation();
                return WalkResult::interrupt();
              }
            } else {
              llvm_unreachable("unsupported loop type");
            }
          }
        }
        return WalkResult::advance();
      });

      // If no candidate found, nothing to do
      if (!candidate) break;

      // Unroll the candidate fully. Candidates must be unrolled fully to
      // produce constant variable entries. If this would lead to an IBUFF
      // overflow, it will be detected later in the pipeline.
      if (performFullUnroll(candidate).failed()) {
        candidate->emitError("Cannot unroll candidate");
        signalPassFailure();
        return;
      }
    }
  });

  // After unrolling, expand and fold affine.apply operations to arithmetic
  // operations to avoid having affine.apply ops in the IR.
  module_op.walk([&](dataflow::ProgramUnitOp unit) {
    auto comp = dcc::getUnitType(
        unit.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
    if (comp != LXLU) return;

    expandAffineApplyOps(unit);
  });

}
// ---- 249/384  processComputeUnit  —  dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:37  (91L)
void e249_processComputeUnit(
    dataflow::ProgramUnitOp &unit_op) {
  // Collect loops meant for unrolling
  unit_op.walk([&](dataflow::GetLogicalMemoryViewOp view_op) {
    auto memory_unit =
        view_op.getFromUnit().getDefiningOp<dataflow::GetLocalUnitOp>();
    if (dcc::utils::isLRFReg(memory_unit.getName())) {
      for (auto use : view_op->getUsers()) {
        ValueRange indices;
        if (auto affine_load = llvm::dyn_cast<AffineVectorLoadOp>(use)) {
          indices = affine_load.getIndices();
        } else if (auto affine_store =
                       llvm::dyn_cast<AffineVectorStoreOp>(use)) {
          indices = affine_store.getIndices();
        } else if (auto agen_load = llvm::dyn_cast<VectorLoadOp>(use)) {
          indices = agen_load.getMapIndices();
        } else if (auto agen_store = llvm::dyn_cast<VectorStoreOp>(use)) {
          indices = agen_store.getIndices();
        } else {
          use->emitError("Encountered an unknown use of memory view\n");
          signalPassFailure();
          return;
        }

        for (auto index : indices) {
          if (mlir::isa<BlockArgument>(index)) {
            auto for_op = index.getParentRegion()->getParentOp();
            if (isa<LoopLikeOpInterface>(for_op)) {
              if (!for_op->hasAttr("unroll")) {
                OpBuilder builder(for_op);
                for_op->setAttr("unroll", builder.getI32IntegerAttr(1));
              }
            } else {
              for_op->emitError(
                  "Support for unrolling "
                  "loop-like op interfaces only");
              signalPassFailure();
              return;
            }
          } else if (!isa<mlir::arith::ConstantIntOp,
                          mlir::arith::ConstantIndexOp>(
                         index.getDefiningOp())) {
            index.getDefiningOp()->emitError(
                "Support for unrolling "
                "of loop iterators only");
            signalPassFailure();
            return;
          }
        }
      }
    }
  });

  // Step-2: Start unrolling the marked loops in reverse order
  unit_op.walk<WalkOrder::PostOrder>([&](mlir::LoopLikeOpInterface loop_op) {
    if (loop_op->hasAttr("unroll")) {
      loop_op->removeAttr("unroll");
      if (auto affine_for =
              llvm::dyn_cast<AffineForOp>(loop_op.getOperation())) {
        if (failed(loopUnrollFull(affine_for))) {
          affine_for->emitError("Non-constant trip bound for unrolling");
          signalPassFailure();
          return;
        }
      } else if (auto scf_for =
                     llvm::dyn_cast<scf::ForOp>(loop_op.getOperation())) {
        auto lb_const =
            scf_for.getLowerBound().getDefiningOp<mlir::arith::ConstantIntOp>();
        auto ub_const =
            scf_for.getUpperBound().getDefiningOp<mlir::arith::ConstantIntOp>();
        auto step_const =
            scf_for.getStep().getDefiningOp<mlir::arith::ConstantIntOp>();
        if (!lb_const || !ub_const || !step_const) {
          auto unroll_size =
              (ub_const.value() - lb_const.value()) / step_const.value();
          if (failed(loopUnrollByFactor(scf_for, unroll_size))) {
            scf_for->emitError("Unable to unroll the scf loop");
            signalPassFailure();
            return;
          }
        } else {
          scf_for->emitError("Non-Constant trip bound for unrolling");
          signalPassFailure();
          return;
        }
      } else {
        loop_op->emitError("Unknown for-loop for unrolling");
        signalPassFailure();
        return;
      }
    }

}}
// ---- 250/384  initMASData  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:707  (31L)
void e250_initMASData(SmallVectorImpl<MASData> &mas_data,
                                           agen::AccessDetailsAffine &ad,
                                           int64_t &max_mutable) {
  // If there are no indices involved in the subscripts, there is nothing to
  // initialize.
  auto indices = ad.getIndices();
  if (indices.size() == 0) return;

  // Construct the coefficients for the non-time part of the transfer.
  auto iter_coeff_dict = ad.getIndicesCoeffDict();
  DT_CHECK(iter_coeff_dict.size() == indices.size() + 1);

  // Calculate the max mutable encountered and fill MASData objects for each
  // iterator in the subscripts.
  //
  // The max mutable will start at the initial mutable value which is
  // determined by the constant offset of the subscripts.
  max_mutable = iter_coeff_dict[nullptr];
  for (int i = 0, e = indices.size(); i < e; ++i) {
    auto index = indices[i];
    auto loop = cast<BlockArgument>(index).getOwner()->getParentOp();
    DT_CHECK(loop);
    auto num_iters = getLoopTripCount(loop);
    DT_CHECK(num_iters >= 0);
    auto composed_coeff = iter_coeff_dict[index];
    // The iter_arg will be the number of iterations - 1 at maximum and the
    // last iteration of every loop can overflow the mutable as no data transfer
    // will occur after it. So really, the number of iterations - 2 is the last
    // utilized mutable address.
    int64_t weight = num_iters < 2 ? 0 : (num_iters - 2) * composed_coeff;
    mas_data.emplace_back(index, i, composed_coeff, num_iters, weight);

}}
// ---- 251/384  setupForPartitioning  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:819  (6L)
void e251_setupForPartitioning(
    ExpressionEvaluator &evaluator, SmallVectorImpl<MASData> &mas_data,
    SmallVectorImpl<int64_t> &partition_sizes,
    const SmallVectorImpl<Value> &all_mem_views, const SenComponents comp,
    dataflow::GetLogicalMemoryViewOp mem_view_op, const int64_t &max_mutable,
    const int elem_size_in_bits) {

}
// ---- 252/384  fillPartitions  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1063  (117L)
void e252_fillPartitions(
    dcc::ConditionalTree &cond_tree, const SmallVectorImpl<MASData> &mas_data,
    const SmallVectorImpl<int64_t> &partition_sizes,
    const AffineMap &subscripts_map,
    const std::function<void(OpBuilder &, AffineMap &, int64_t start_addr_mod)>
        &op_creator) {
  // Get the coefficients for each partitioned dimension for each subscript
  // result. For example if the subscripts are:
  //   <0, arg1 * 3 + arg2 * 2, arg3 + arg2 * 4>
  // and arg1, arg2, and arg3 are partitioned in that order, the
  // subscripts_coeffs would be:
  //   [ [0, 3, 0], [0, 2, 4], [0, 0, 1] ]
  auto subscripts_coeffs = calculateSubscriptsCoefficients(
      mas_data, partition_sizes, subscripts_map);

  auto addOpsToPartitions = [&](dcc::CondNode *n) -> dcc::CondNode * {
    if (!n->isLeafNode()) return nullptr;

    // Leaves represent partitions. Fill this partition.
    AffineMap new_subscripts_map;
    SmallVector<AffineExpr> new_exprs;
    for (auto &res : subscripts_map.getResults()) new_exprs.push_back(res);

    int64_t start_addr_mod = 0;
    dcc::CondNode *curr_node = n;
    scf::IfOp if_op = cast<scf::IfOp>(n->getParentNode()->getOperation());

    // The tree was built so every partition is a nested conditional tree in the
    // form below. This example assumes arg1 and arg2 are split into three
    // partitions each. This leads to 9 partitions to fill (3 x 3):
    //                    <root>
    //                       |
    //                if <arg1 ub1>
    //                /             \
    //             then             else
    //             /                   \
    //       if <arg2 ub1>               if <arg1 ub2>
    //       /          \              /          \
    //     then        else          then         else
    //       |          |             |              \
    //      P1     if <arg2 ub2>    if <arg2 ub1>     if <arg2 ub1>
    //             /       \          /      \         /       \
    //          then      else     then      else     then    else
    //            |         |        |          |        |      |
    //           P2        P3        P4   if <arg2 ub2>   P7   if <arg2 ub2>
    //                                      /      \          /      \
    //                                   then      else     then    else
    //                                     |         |       |        |
    //                                    P5         P6      P8      P9
    //
    // Traversing the tree a number of conditionals equivalent to the number of
    // dimensions that were partitioned provides all the info required to
    // calculate the immutable and subscripts required for the partition.
    //
    // Since the tree is traversed in reverse, the partitions need to be
    // accessed in reverse as well.
    for (int p = partition_sizes.size() - 1; p >= 0; --p) {
      // Only leaves are analyzed and every conditional was created with a then
      // and else body. This means every node being analyzed will have a parent
      // CondNode containing an scf.if operation.
      // Calculations are slightly different for then and else nodes.
      bool is_then_node = curr_node->isThenNode();
      auto curr_if_op =
          cast<scf::IfOp>(curr_node->getParentNode()->getOperation());
      auto cond =
          cast<arith::CmpIOp>(curr_if_op.getCondition().getDefiningOp());

      // The LHS of the condition is the iterator.
      auto iter = cond.getLhs();

      // The RHS of the condition is the partition demarkation point.
      // The demarkation point denotes where the last partition ended.
      auto rhs_const = cast<arith::ConstantOp>(cond.getRhs().getDefiningOp());
      auto demarkation = cast<IntegerAttr>(rhs_const.getValue()).getInt();

      // If the current node is a then node:
      //   <prev iterations> = <demarkation> - <partition size>
      // If the current node is an else node:
      //   <prev iterations> = <demarkation>
      int64_t prev_iters =
          is_then_node ? demarkation - partition_sizes[p] : demarkation;

      // Calculate the amount to add to the start address.
      // The start address will be modified by:
      //   <prev iterations> * <composed coefficient>
      start_addr_mod += prev_iters * mas_data[p].composed_coeff_;

      // Calculate the amount to add to the new subscripts.
      auto coeffs = subscripts_coeffs[p];
      DT_CHECK(coeffs.size() == subscripts_map.getNumResults());
      for (int r = 0, num_res = subscripts_map.getNumResults(); r < num_res;
           ++r) {
        // The current result for the subscripts will be modified by:
        //   -(<subscripts coeff> * <prev iterations>)
        new_exprs[r] = new_exprs[r] - (coeffs[r] * prev_iters);
      }

      // If this is not the last iteration, set the curr_node to the next
      // CondNode to analyze the next partitioned dimension.
      //
      // Note: The parent of this node will be the conditional itself. Two
      //       traversals are required to get to the next Then or Else node
      //       if this isn't the last dim. Traversing up after the last dim
      //       risks attempting to get the parent of the root.
      if (p != 0) curr_node = curr_node->getParentNode()->getParentNode();
    }

    // Update the subscripts map with the new expressions
    new_subscripts_map = AffineMap::get(subscripts_map.getNumDims(), 0,
                                        new_exprs, subscripts_map.getContext());

    OpBuilder partition_builder(if_op);
    partition_builder = n->isThenNode() ? if_op.getThenBodyBuilder()
                                        : if_op.getElseBodyBuilder();

    op_creator(partition_builder, new_subscripts_map, start_addr_mod);


}}
// ---- 253/384  adjustForEvenImmutableAddr  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:1204  (47L)
void e253_adjustForEvenImmutableAddr(
    ExpressionEvaluator &evaluator, Value immutable_addr,
    AffineMap &subscripts_map, agen::AccessDetailsAffine &ad,
    int64_t &immutable_addr_mod, const int elem_size_in_bits) {
  if (dcc_ext_ctx_.getArch() < IsaCoreGen::SEN1P5_ISA) return;
  int num_elems_in_stick =
      dcc_ext_ctx_.getBytesPerStick() * 8 / elem_size_in_bits;
  bool is_even = dcc::agen::utils::isL3ImmutableAddrEven(
      evaluator, immutable_addr, num_elems_in_stick);
  bool is_even_mod = immutable_addr_mod % 2 == 0;
  if (is_even != is_even_mod) {
    DT_CHECK_MSG(is_even ? true
                         : dcc::agen::utils::isL3ImmutableAddrAllOdd(
                               evaluator, immutable_addr, num_elems_in_stick),
                 "All immutable addrs must be all even or all odd to execute "
                 "even shift.");

    // Move a stick out from the immutable addr.
    immutable_addr_mod -= num_elems_in_stick;

    // Update the subscripts map to add a stick to d0.
    // d0 coefficient will always be 1 so it is simple to just
    // add a stick to this dimension.
    auto transfer_order = ad.getTransferOrder();
    unsigned res = 0, num_res = transfer_order.getNumResults();
    for (; res < num_res; ++res) {
      if (transfer_order.getResult(res).isFunctionOfDim(0)) break;
    }
    DT_CHECK(res < num_res);

    // TODO: We should also be looking at the transfer set to ensure
    // we don't add too many elements to a dimension in the case the
    // stick is split into more than one dimension. For now we just
    // check we have the space, but we should add logic to calculate
    // an appropriate offset amount if one exists.
    // Note: The extents are ordered already in
    //       AccessDetailsBase::constructExtentAndTotalElements().
    DT_CHECK_MSG(ad.getExtents()[res] >= num_elems_in_stick,
                 "Innermost dimension extend must fit a full stick.");

    SmallVector<AffineExpr> new_exprs;
    for (int i = 0, e = subscripts_map.getNumResults(); i < e; ++i) {
      AffineExpr expr = subscripts_map.getResult(i);
      if (i == res) expr = expr + num_elems_in_stick;
      new_exprs.push_back(expr);
    }
    subscripts_map = AffineMap::get(subscripts_map.getNumDims(),

}}
// ---- 254/384  offsetShifts  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:590  (22L)
void e254_offsetShifts(
    SmallVectorImpl<int64_t> &shifts, agen::AccessDetailsAffine &ad,
    const int64_t offset) {
  if (offset == 0) return;
  auto transfer_order = ad.getTransferOrder();
  unsigned res = 0, num_res = transfer_order.getNumResults();
  for (; res < num_res; ++res) {
    if (transfer_order.getResult(res).isFunctionOfDim(0)) break;
  }
  DT_CHECK(res < num_res && shifts.size() == num_res);

  // TODO: We should also be looking at the transfer set to ensure
  // we don't add too many elements to a dimension in the case the
  // stick is split into more than one dimension. For now we just
  // check we have the space, but we should add logic to calculate
  // an appropriate offset amount if one exists.
  // Note: The extents are ordered already in
  //       AccessDetailsBase::constructExtentAndTotalElements().
  //       Only negative shifts add elements to the subscripts map
  //       so only negative values need to check the transfer set.
  DT_CHECK_MSG(offset < 0 ? ad.getExtents()[res] >= offset : true,
               "Innermost dimension extend must fit the full offset amount.");

}
// ---- 255/384  applyShifts  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:616  (27L)
AffineMap e255_applyShifts(
    ExpressionEvaluator &evaluator, const SmallVectorImpl<int64_t> &shifts,
    dataflow::ProgramUnitOp unit, Operation *op,
    dataflow::GetLogicalMemoryViewOp mem_view_op,
    agen::AccessDetailsAffine &ad) {
  // The shifts were calculated with respect to transfer order already, so
  // the shifts can be directly applied to the results.
  auto subscripts_map = ad.getSubscriptsMap();
  auto num_dims = subscripts_map.getNumDims();
  SmallVector<AffineExpr> new_exprs;
  DT_CHECK(shifts.size() == subscripts_map.getNumResults());
  for (auto [shift, res] : llvm::zip(shifts, subscripts_map.getResults()))
    new_exprs.emplace_back(res - shift);

  subscripts_map =
      AffineMap::get(num_dims, 0, new_exprs, subscripts_map.getContext());

  // Calculate how many elements in total the shifts imply.
  auto layout_coeffs = ad.getLayoutCoeffs();
  int64_t total_shift = 0;
  DT_CHECK(layout_coeffs.size() >= shifts.size());
  for (int i = 0, e = shifts.size(); i < e; ++i)
    total_shift += shifts[i] * layout_coeffs[i];

  OpBuilder const_builder(unit);
  OpBuilder query_map_builder(dcc::uniform::utils::getLocalOrGlobalRegion(op));
  DT_CHECK(mem_view_op->hasOneUse());

}
// ---- 256/384  runOnOperation  —  dcc/src/Transform/Dataflow/ProgramUnitsReduction.cpp:152  (76L)
void e256_runOnOperation() {
  if (DisableThisPass) return;

  // Disable program units reduction in case of folding.
  if (dcc_ext_ctx_.getFolding()) return;

  // Collect all programmable units
  ModuleOp module_op = getOperation();
  module_op.walk([&](dataflow::ProgramUnitOp unit) {
    // Check for presence of folds --> this could happen in standalone testing.
    bool folding_exists = false;
    std::set<mlir::Operation *> operations;
    for (auto unit_arg : unit.getUnits()) {
      auto unit_arg_parent = unit_arg.getDefiningOp();
      if (operations.find(unit_arg_parent) == operations.end()) {
        operations.insert(unit_arg_parent);
      } else {
        folding_exists = true;
        break;
      }
    }

    if (!folding_exists) {
      auto unit_type = dcc::getUnitType(unit.getUnits()[0].getDefiningOp());
      if (!is_any_of(unit_type, LXLU, LXSU)) {
        program_units_.push_back(unit);
      }
    }
  });

  // Explore in reverse direction because dataflow.get_units would have been
  // defined before and this avoids recreation of those operations.
  for (int i = program_units_.size() - 1; i >= 0; i--) {
    bool matched = false;
    auto &unit = program_units_[i];

    // Iterate through each reducible group to see if it matches.
    for (auto &group : reducible_groups_) {
      if (succeeded(matchUnits(group, unit))) {
        // If matching is successful, add the list of units to the group base.
        for (auto tmp_unit : unit.getUnits()) {
          group.units_list_.push_back(tmp_unit.getDefiningOp<GetUnitOp>());
        }

        matched = true;
        break;
      }
    }

    // If it's matched, delete the existing unit.
    // The appending to group was already taken care.
    if (matched) {
      unit.erase();
    } else {
      // If it's not matched, create a new group.
      ReducibleProgramUnits group(unit);
      reducible_groups_.push_back(unit);
    }
  }

  // After matching process, iterate through each reducible group,
  // and update the base unit program operands.
  for (auto &group : reducible_groups_) {
    llvm::SmallVector<Value> units_list;
    for (auto tmp_unit : group.units_list_) {
      for (auto fold : tmp_unit.getResults()) {
        units_list.push_back(fold);
      }
    }

    group.base_unit_program_->setOperands(units_list);
    Builder builder(group.base_unit_program_.getOperation());
    if (group.base_unit_program_.getRegion().getNumArguments() == 0)
      group.base_unit_program_.getRegion().addArgument(
          builder.getIndexType(), group.base_unit_program_.getLoc());
  }

}
// ---- 257/384  analyzeAndTransform  —  dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:441  (3L)
void e257_analyzeAndTransform(
    LoopLikeOpInterface& loop_op) {
  auto type = analyzeLoop(loop_op);

}
// ---- 258/384  addConstraintsForIVRanges  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:86  (22L)
void e258_addConstraintsForIVRanges(
    FlatLinearValueConstraints &page_sel_constraints,
    SmallVectorImpl<IVRange> &indices_ranges) {
  int num_syms = page_sel_constraints.getNumSymbolVars();
  if (num_syms == 0) return;

  SmallVector<AffineExpr, 16> range_ineq_constraints;
  for (unsigned sym_idx = 0; sym_idx < num_syms; ++sym_idx) {
    AffineExpr sym_expr = getAffineSymbolExpr(sym_idx, context_);
    // The first element of the IVRange pair is the lower bound. The lower bound
    // is represented by the expression: <sym> - <lb> >= 0.
    range_ineq_constraints.emplace_back(sym_expr -
                                        indices_ranges[sym_idx].first);
    // The second element of the IVRange pair is the upper bound. The upper
    // bound is represented by the expression: -<sym> + <ub> >= 0.
    range_ineq_constraints.emplace_back(-sym_expr +
                                        indices_ranges[sym_idx].second);
  }

  SmallVector<bool, 16> range_eq_constraints(num_syms * 2, false);
  IntegerSet range_set = IntegerSet::get(0, num_syms, range_ineq_constraints,
                                         range_eq_constraints);

}
// ---- 259/384  createNewSubscriptsFromStartElements  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:562  (10L)
AffineMap e259_createNewSubscriptsFromStartElements(
    AffineMap &subscripts_map, SmallVectorImpl<int64_t> &start_elements) {
  DT_CHECK(start_elements.size() >= subscripts_map.getNumResults());

  SmallVector<AffineExpr, 16> new_subscripts_exprs;
  for (int dim = 0; dim < subscripts_map.getNumResults(); ++dim)
    new_subscripts_exprs.emplace_back(subscripts_map.getResult(dim) -
                                      start_elements[dim]);

  return AffineMap::get(subscripts_map.getNumDims(), 0, new_subscripts_exprs,

}
// ---- 260/384  gatherPageDependentDimsForPage  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:927  (32L)
void e260_gatherPageDependentDimsForPage(
    FlatLinearValueConstraints &page_sel_constraints,
    FlatLinearValueConstraints &compare_constraints) {
  DT_CHECK(compare_constraints.getNumInequalities() ==
               page_sel_constraints.getNumInequalities() &&
           compare_constraints.getNumEqualities() ==
               page_sel_constraints.getNumEqualities());

  // The coefficients of each page constraints will be the same across
  // pages but the const values may differ. If they differ from page to
  // page, the dims in that constraint with non-zero coefficients have a
  // bearing on page selection.
  int const_col = page_sel_constraints.getNumCols() - 1;
  int num_syms = page_sel_constraints.getNumSymbolVars();
  for (int i = 0; i < page_sel_constraints.getNumInequalities(); ++i) {
    auto compare_ineq = compare_constraints.getInequality(i);
    auto ineq = page_sel_constraints.getInequality(i);
    if (compare_ineq[const_col] != ineq[const_col]) {
      for (int sym = 0; sym < num_syms; ++sym) {
        DT_CHECK(compare_ineq[sym] == ineq[sym]);
        if (ineq[sym] != 0) page_dependent_time_syms_.insert(sym);
      }
    }
  }
  for (int i = 0; i < page_sel_constraints.getNumEqualities(); ++i) {
    auto compare_eq = compare_constraints.getEquality(i);
    auto eq = page_sel_constraints.getEquality(i);
    if (compare_eq[const_col] != eq[const_col]) {
      for (int sym = 0; sym < num_syms; ++sym) {
        DT_CHECK(compare_eq[sym] == eq[sym]);
        if (eq[sym] != 0) page_dependent_time_syms_.insert(sym);
      }

}}}
// ---- 261/384  runOnOperation  —  dcc/src/Transform/Dataflow/UniformQueryMapsCanonicalization.cpp:55  (41L)
void e261_runOnOperation() {
  ModuleOp module_op = getOperation();

  SmallVector<uniform::QueryMapOp> query_maps_to_be_deleted;
  module_op.walk([&](uniform::QueryMapOp query_map_op) {
    dcc::uniform::utils::simplifyQueryMapWithSameTarget(query_map_op);
    if (query_map_op.getResult().getUses().empty()) {
      query_maps_to_be_deleted.push_back(query_map_op);
    }
  });

  for (auto it = query_maps_to_be_deleted.rbegin();
       it != query_maps_to_be_deleted.rend(); ++it) {
    auto def_map_op =
        it->getMap().getDefiningOp<uniform::DefImmutableMappingOp>();
    LLVM_DEBUG(llvm::dbgs() << "Deleting QueryMap operation: " << *it << "\n");
    it->erase();

    if (!def_map_op.getResult().getUses().empty()) continue;
    SmallPtrSet<Operation *, 32> dead_ops;
    for (Value v : def_map_op.getValues()) {
      // A value in the map that has a single use would become dead as a
      // result of erasing the map (because the map is the only user of that
      // value). Erase such values since mlir's canonicalizer is slow in
      // detecting and erasing them later.
      Operation *op = v.getDefiningOp();
      if (op && v.hasOneUse()) dead_ops.insert(op);
    }

    LLVM_DEBUG(llvm::dbgs() << "Deleting DefImmutableMapping operation:  "
                            << def_map_op << "\n");
    def_map_op.erase();
    if (EnableDeadMapVarDeletion) {
      for (Operation *op : dead_ops) {
        DT_CHECK(op->getUses().empty());
        LLVM_DEBUG(llvm::dbgs()
                   << "Deleting dead map value ops: " << *op << "\n");
        op->erase();
      }
    }
  }

}
// ---- 262/384  removeCoresCoreletsFoldsFromUniformizeRegion  —  dcc/src/Transform/Dataflow/UnitFiltering.cpp:139  (98L)
void e262_removeCoresCoreletsFoldsFromUniformizeRegion(
    uniform::UniformizeRegionsOp uniformize_op) {
  OpBuilder builder(uniformize_op);
  bool unit_list_is_reduced = false;
  int num_regions = uniformize_op.getNumRegions();
  int new_num_regions = num_regions;
  std::vector<mlir::Value> new_unit_list;
  std::vector<unsigned> empty_region_indices;
  std::vector<Attribute> new_list_sizes;
  for (auto unit : uniformize_op.getUnits()) {
    auto get_unit_op_result = llvm::dyn_cast<OpResult>(unit);
    unsigned fold_id = get_unit_op_result.getResultNumber();
    auto get_unit_op =
        llvm::dyn_cast<mlir::dataflow::GetUnitOp>(unit.getDefiningOp());
    unsigned core_id = dcc::getCoreId(get_unit_op);
    int corelet_id = dcc::getCoreletId(get_unit_op);
    if ((!filter_folds_except_.empty() &&
         filter_folds_except_.count(fold_id) == 0) ||
        (!filter_cores_except_.empty() &&
         filter_cores_except_.count(core_id) == 0) ||
        (!filter_corelets_except_.empty() && corelet_id != -1 &&
         filter_corelets_except_.count(corelet_id) == 0)) {
      unit_list_is_reduced = true;
      continue;
    }
    new_unit_list.push_back(unit);
  }
  if (!unit_list_is_reduced) return;

  if (new_unit_list.empty()) {
    for (int i = 0; i < uniformize_op.getNumResults(); ++i)
      DT_CHECK_MSG(uniformize_op.getResult(i).use_empty(),
                   "Cannot filter out uniformize region with uses");
    to_delete_.push_back(uniformize_op);
    return;
  }

  for (int i = 0; i < num_regions; ++i) {
    auto unit_list_i = uniformize_op.getRegionUnitList(i);
    std::vector<mlir::Value> new_unit_list_i;
    for (auto unit : unit_list_i) {
      auto get_unit_op_result = llvm::dyn_cast<OpResult>(unit);
      unsigned fold_id = get_unit_op_result.getResultNumber();
      auto get_unit_op =
          llvm::dyn_cast<mlir::dataflow::GetUnitOp>(unit.getDefiningOp());
      unsigned core_id = dcc::getCoreId(get_unit_op);
      int corelet_id = dcc::getCoreletId(get_unit_op);
      if ((!filter_folds_except_.empty() &&
           filter_folds_except_.count(fold_id) == 0) ||
          (!filter_cores_except_.empty() &&
           filter_cores_except_.count(core_id) == 0) ||
          (!filter_corelets_except_.empty() && corelet_id != -1 &&
           filter_corelets_except_.count(corelet_id) == 0))
        continue;
      new_unit_list_i.push_back(unit);
    }
    // Skip regions which are filtered out
    if (new_unit_list_i.empty()) {
      empty_region_indices.push_back(i);
      --new_num_regions;
    } else {
      new_list_sizes.push_back(
          builder.getI32IntegerAttr(new_unit_list_i.size()));
    }
  }

  auto new_uniformize_op = mlir::uniform::UniformizeRegionsOp::create(
      builder, uniformize_op.getLoc(), uniformize_op.getResultTypes(),
      new_unit_list, ArrayAttr::get(builder.getContext(), new_list_sizes),
      uniformize_op.getRegIndicesIfExist(),
      uniformize_op.getRegLocalesIfExist(), new_num_regions);

  // Walk through the old and new lists of regions. Use empty_region_indices to
  // skip over any regions which have been filtered out.
  for (int new_reg_idx = 0, old_reg_idx = 0; old_reg_idx < num_regions;
       ++old_reg_idx) {
    if (std::find(empty_region_indices.begin(), empty_region_indices.end(),
                  old_reg_idx) != empty_region_indices.end())
      continue;
    new_uniformize_op.getRegion(new_reg_idx).push_back(new Block);
    auto &curr_block = new_uniformize_op.getRegion(new_reg_idx).front();
    builder.setInsertionPointToStart(&curr_block);

    // insert block argument as unit list iterator
    curr_block.addArgument(builder.getIndexType(), new_uniformize_op.getLoc());

    // Clone the body
    auto &orig_block = uniformize_op.getRegion(old_reg_idx).front();
    IRMapping bv_map;
    bv_map.map(orig_block.getArguments(), curr_block.getArguments());
    for (auto &it : orig_block.getOperations()) {
      builder.clone(it, bv_map);
    }
    ++new_reg_idx;
  }
  for (int i = 0; i < uniformize_op.getNumResults(); ++i)
    uniformize_op.getResult(i).replaceAllUsesWith(
        new_uniformize_op.getResult(i));

}
// ---- 263/384  removeAncestors  —  dcc/src/Transform/Dataflow/UnitFiltering.cpp:339  (18L)
void e263_removeAncestors(Operation *op) {
  llvm::SmallVector<Operation *> worklist = {op};
  while (!worklist.empty()) {
    Operation *oper = worklist.back();
    worklist.pop_back();
    for (auto operand : oper->getOperands()) {
      Operation *operand_op = operand.getDefiningOp();
      if (operand_op && !isDataTransferToKeep(operand_op)) {
        if (std::find(worklist.begin(), worklist.end(), operand_op) ==
            worklist.end())
          worklist.push_back(operand_op);
      }
    }
    bool has_uses = false;
    for (int i = 0; i < oper->getNumResults(); ++i)
      if (!oper->getResult(i).use_empty()) has_uses = true;
    if (!has_uses) oper->erase();
  }

}
// ---- 264/384  createForOpWithAdditionalReturnValue  —  dcc/src/Transform/Dataflow/Utils.cpp:28  (66L)
Operation* createForOpWithAdditionalReturnValue(Operation* loop_op,
                                                int n_values, IRMapping& ir_map,
                                                bool delete_op) {
  Operation* ret_op = nullptr;
  // Adding exiting iterator arguments
  SmallVector<Value, 8> iter_args;
  OpBuilder builder(loop_op);

  if (auto affine_for = dyn_cast<affine::AffineForOp>(loop_op)) {
    for (auto operand : affine_for.getInits()) {
      iter_args.push_back(operand);
    }
    // Adding new iterator arguments
    for (int i = 0; i < n_values; i++) {
      auto const_op =
          mlir::arith::ConstantIndexOp::create(builder, loop_op->getLoc(), 0);
      iter_args.push_back(const_op);
    }

    auto new_loop = affine::AffineForOp::create(
        builder, loop_op->getLoc(), affine_for.getLowerBoundOperands(),
        affine_for.getLowerBoundMap(), affine_for.getUpperBoundOperands(),
        affine_for.getUpperBoundMap(), affine_for.getStepAsInt(), iter_args);
    if (auto dbg_name_attr = dataflow::getDbgNameAttr(loop_op))
      dataflow::setDbgNameAttr(new_loop, dbg_name_attr);
    builder.setInsertionPointToStart(new_loop.getBody());

    ir_map = copyLoopBody(affine_for, new_loop, builder, n_values);

    ret_op = static_cast<Operation*>(new_loop);
  } else if (auto scf_for = dyn_cast<scf::ForOp>(loop_op)) {
    for (auto operand : scf_for.getInits()) {
      iter_args.push_back(operand);
    }
    // Adding new iterator arguments
    for (int i = 0; i < n_values; i++) {
      auto const_op =
          mlir::arith::ConstantIndexOp::create(builder, loop_op->getLoc(), 1);
      iter_args.push_back(const_op);
    }

    auto new_loop = scf::ForOp::create(
        builder, loop_op->getLoc(), scf_for.getLowerBound(),
        scf_for.getUpperBound(), scf_for.getStep(), iter_args);
    if (auto dbg_name_attr = dataflow::getDbgNameAttr(loop_op))
      dataflow::setDbgNameAttr(new_loop, dbg_name_attr);
    builder.setInsertionPointToStart(new_loop.getBody());

    ir_map = copyLoopBody(scf_for, new_loop, builder, n_values);

    ret_op = static_cast<Operation*>(new_loop);
  }

  // Replace uses if exists
  for (int i = 0; i < loop_op->getNumResults(); i++) {
    loop_op->getResult(i).replaceAllUsesWith(ret_op->getResult(i));
  }

  for (auto attr : loop_op->getAttrs()) {
    if (attr.getName() != "operandSegmentSizes") {
      DT_CHECK(ret_op != nullptr);
      ret_op->setAttr(attr.getName(), attr.getValue());
    }
  }

  // Erase the loop


}
// ==================================================================================================
// LEVEL 3
// ==================================================================================================

// ---- 265/384  constructDetails  —  dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:418  (18L)
LogicalResult e265_constructDetails(
    MemoryOperandIndex memory_index) {
  // Memory index is used to identify src/dest in composite_load_and_store
  setMemoryIndex(memory_index);

  if (initialize().failed()) return LogicalResult::failure();

  if (constructIndices().failed()) return LogicalResult::failure();

  if (constructExtentAndTotalElements().failed())
    return LogicalResult::failure();

  if (constructChunkAndShuffleInfo().failed()) return LogicalResult::failure();

  constructLdOrStType();

  if (constructIteratorCoefficients().failed()) return LogicalResult::failure();


}
// ---- 266/384  coalesceTimeDimensions  —  dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:681  (112L)
void e266_coalesceTimeDimensions(
    AccessContainer<AccessDetailsAffineComposite>& access_details) {
  DT_CHECK(access_details.size() > 0 && access_details.size() <= 4);
  DT_CHECK(access_details.getFirst().getTimeBounds().size() ==
           access_details.getFirst().getTimeOffsets().size() - 1);

  SmallVector<int64_t> time_bounds_remaining;

  // time_bounds and time_offsets are already ordered based on time_order.
  auto time_bounds = access_details.getFirst().getTimeBounds();
  auto time_offsets = access_details.getFirst().getTimeOffsets();

  // When we have more than one access detail, only the manadatory operands (ie
  // kDirSrc and kDirDst) are used for the actual time steps.
  if (access_details.size() >= 2) {
    DT_CHECK(access_details.get(MemoryOperandIndex::kDirDst)
                 .getTimeBounds()
                 .size() == access_details.get(MemoryOperandIndex::kDirDst)
                                    .getTimeOffsets()
                                    .size() -
                                1);
    // time bounds must be the same, but time offsets may be different
    DT_CHECK(llvm::all_of(
        llvm::zip(
            access_details.get(MemoryOperandIndex::kDirSrc).getTimeBounds(),
            access_details.get(MemoryOperandIndex::kDirDst).getTimeBounds()),
        [&](std::tuple<int64_t, int64_t> pair) {
          return std::get<0>(pair) == std::get<1>(pair);
        }));

    // time_bounds and time_offsets are already ordered based on time_order.
    auto time_bounds_src =
        access_details.get(MemoryOperandIndex::kDirSrc).getTimeBounds();
    auto time_offsets_src =
        access_details.get(MemoryOperandIndex::kDirSrc).getTimeOffsets();
    auto time_offsets_dst =
        access_details.get(MemoryOperandIndex::kDirDst).getTimeOffsets();

    bool skip = false;
    time_bounds.clear();
    time_offsets.clear();
    time_offsets.emplace_back(time_offsets_src.back());

    // time_bounds (/src), time_offsets (/src/dst) --> ordering --> outer to
    // innermost time dimensions.
    //  Hence, we are iterating in a reverse order to find the cut from the
    //  inner dimension where the offsets start differing. While filling
    //  time_bounds, the ordering has to be again outer to inner.
    for (int i = time_bounds_src.size() - 1; i >= 0; i--) {
      if (!skip && (time_bounds_src[i] == 1 ||
                    time_offsets_src[i] == time_offsets_dst[i])) {
        time_offsets.insert(time_offsets.begin(), time_offsets_src[i]);
        time_bounds.insert(time_bounds.begin(), time_bounds_src[i]);
      } else {
        skip = true;
        time_bounds_remaining.insert(time_bounds_remaining.begin(),
                                     time_bounds_src[i]);
      }
    }
  }

  DT_CHECK_MSG(
      time_bounds.size() + 1 == time_offsets.size(),
      "expected same number of dimensions for time_addr map and time_set");
  //  scan from innermost loop
  bool time_bound_coalesced = false;
  int num_of_dim = time_bounds.size();
  int time_index_inner = num_of_dim - 1;
  auto total_dist_inner =
      time_bounds[time_index_inner] * time_offsets[time_index_inner];
  int64_t coalesced_bound = time_bounds[time_index_inner];
  // TODO double check time_index_outer value..
  // time_index_outer goes down to -1 to catch the outermost time dim
  for (int time_index_outer = num_of_dim - 2; time_index_outer >= -1;
       --time_index_outer) {
    // llvm::outs() << "count: " << time_index_outer << "\n";
    if (time_index_outer == -1 ||
        time_offsets[time_index_outer] != total_dist_inner) {
      if ((time_index_inner - time_index_outer) > 1) {
        // llvm::outs() << "time_index_outer: " << time_index_outer
        //              << "time_index_inner: " << time_index_inner
        //              << "coalesced_bound: " << coalesced_bound << "\n";
        setCoalescedBoundValues(time_bounds, time_index_outer, time_index_inner,
                                coalesced_bound);
        time_bound_coalesced = true;
        // for (auto a : time_bound) {
        //   llvm::outs() << "modified time_bound: " << a << "\n";
        // }
      }
      if (time_index_outer > -1) {
        time_index_inner = time_index_outer;
        total_dist_inner =
            time_bounds[time_index_outer] * time_offsets[time_index_outer];
        coalesced_bound = time_bounds[time_index_outer];
      }
    } else {
      coalesced_bound *= time_bounds[time_index_outer];
      total_dist_inner =
          time_bounds[time_index_outer] * time_offsets[time_index_outer];
    }
  }

  // if the cut has happened, then those remaining time dimensions from the cut
  // to the outermost has to be added back.
  time_bounds.insert(time_bounds.begin(), time_bounds_remaining.begin(),
                     time_bounds_remaining.end());
  access_details.getFirst().setTimeBounds(time_bounds);

  if (time_bound_coalesced && access_details.size() >= 2) {
    // use copy assignment to make sure time_bounds of both access details
    // objects are the same
    access_details.get(MemoryOperandIndex::kDirDst).setTimeBounds(time_bounds);

}}
// ---- 267/384  constructTimeLoopsAndVectorOperations  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:1789  (109L)
LogicalResult e267_constructTimeLoopsAndVectorOperations(
    dataflow::ProgramUnitOp unit_op, Operation* op,
    AccessContainer<Value>& mutable_addrs,
    AccessContainer<Value>& immutable_addrs,
    AccessContainer<AccessDetailsAffineComposite>& access_details,
    Operation* extract_op) {
  DT_CHECK(mutable_addrs.size() == immutable_addrs.size());
  DT_CHECK(mutable_addrs.size() == access_details.size());
  auto access_details_front = access_details.getFirst();
  auto time_bounds = access_details_front.getTimeBounds();
  auto time_offsets = access_details_front.getTimeOffsets();
  auto burst_index = access_details_front.getBurstIndex();
  auto group_index = access_details_front.getInterleaveGroupIndex();

  OpBuilder builder(op);

  // generate nested forOp for time dims.
  llvm::SmallVector<Operation*, 8> for_ops;
  auto loop_num = burst_index > -1 ? burst_index : time_bounds.size();
  for (int idx = 0; idx < loop_num; idx++) {
    // Add mutable addr to the loop iterators corresponding to time
    // dimensions.
    SmallVector<Value, 2> iter_args;
    for (const auto& addr : mutable_addrs) {
      iter_args.push_back(addr);
    }
    size_t loop_bound =
        (time_bounds[idx] == AccessDetailsAffineComposite::
                                 SpecialTimeBoundValues::kCoalesced
             ? 1
             : time_bounds[idx]);
    DT_CHECK(loop_bound > 0);
    auto for_op = affine::AffineForOp::create(builder, op->getLoc(), 0,
                                              loop_bound, 1, iter_args);
    if (auto new_dbg_name_attr = dcc::utils::getNewDbgNameFromOp(
            /*prefix*/ "Time-Loop(", op,
            /*suffix*/ ", t-dim " + std::to_string(idx) + ")"))
      setDbgNameAttr(for_op, new_dbg_name_attr);

    builder.setInsertionPointToStart(for_op.getBody());

    // reset mutable_addr with for_op iterator arguments.
    mutable_addrs.clear();
    for (auto& arg : for_op.getRegionIterArgs()) {
      mutable_addrs.push_back(arg);
    }

    SmallVector<Value, 2> yield_args;
    for (int i = 0; i < access_details.size(); i++) {
      // int64_t time_offset = access_details[i].getTimeOffsets()[idx];
      int64_t time_offset = access_details[i].getTimeOffsets()[idx];

      // Indirect src/dst operands don't really have their own time dimensions.
      if (access_details[i].getMemoryIndex() != MemoryOperandIndex::kDirSrc &&
          access_details[i].getMemoryIndex() != MemoryOperandIndex::kDirDst)
        time_offset = 0;

      auto const_op = mlir::arith::ConstantIndexOp::create(
          builder, op->getLoc(), time_offset);

      auto add_op = mlir::arith::AddIOp::create(
          builder, const_op.getLoc(), mutable_addrs[i], const_op.getResult());
      yield_args.push_back(add_op.getResult());
    }

    affine::AffineYieldOp::create(builder, op->getLoc(), yield_args);
    builder.setInsertionPointToStart(for_op.getBody());
    for_ops.push_back(for_op);
  }

  // generate LoadAndSendOp
  auto burst_size = burst_index >= 0 ? time_bounds[burst_index] : 0;
  auto group_size = group_index >= 0 ? time_bounds[group_index] : 0;
  auto stride_step = group_index >= 0
                         ? time_offsets[group_index]
                         : (burst_index >= 0 ? time_offsets[burst_index]
                                             : time_offsets.front());

  LogicalResult result = LogicalResult::failure();
  if (isa<CompositeLoadOp>(op)) {
    result = constructLoadAndSendStmt<AccessDetailsAffineComposite>(
        &builder, unit_op, op, access_details[0], mutable_addrs[0],
        immutable_addrs[0], burst_size, group_size, stride_step,
        for_ops.empty() ? nullptr : for_ops.front());
  } else if (auto composite_store_op = dyn_cast<CompositeStoreOp>(op)) {
    Type element_type =
        composite_store_op.getMemRef().getType().getElementType();
    result = constructReceiveAndStoreStmt<AccessDetailsAffineComposite>(
        &builder, unit_op, op, element_type, access_details[0],
        mutable_addrs[0], immutable_addrs[0], burst_size, group_size,
        stride_step, for_ops.empty() ? nullptr : for_ops.front());
  } else if (isa<CompositeLoadAndStoreOp, CompositeIndirectLoadAndStoreOp>(
                 op)) {
    result = constructLoadAndStoreStmt<AccessDetailsAffineComposite>(
        &builder, unit_op, op, access_details, mutable_addrs, immutable_addrs,
        burst_size, group_size, stride_step);
  } else if (isa<CompositeIndirectLoadOp>(op)) {
    result = constructLoadAndSendStmt<AccessDetailsAffineComposite>(
        &builder, unit_op, op, access_details[0], mutable_addrs[0],
        immutable_addrs[0], burst_size, group_size, stride_step,
        for_ops.empty() ? nullptr : for_ops.front(), extract_op);
  } else if (auto composite_ind_store_op =
                 dyn_cast<CompositeIndirectStoreOp>(op)) {
    Type element_type =
        composite_ind_store_op.getDirectMemrefType().getElementType();
    result = constructReceiveAndStoreStmt<AccessDetailsAffineComposite>(
        &builder, unit_op, op, element_type, access_details[0],
        mutable_addrs[0], immutable_addrs[0], burst_size, group_size,
        stride_step, for_ops.empty() ? nullptr : for_ops.front(), extract_op);

}}
// ---- 268/384  constructLoadAndStoreStmt  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2167  (177L)
LogicalResult e268_constructLoadAndStoreStmt(
    OpBuilder* builder, dataflow::ProgramUnitOp unit_op, Operation* op,
    AccessContainer<AccessDetailsTy>& access_details,
    AccessContainer<Value>& mutable_addrs,
    AccessContainer<Value>& immutable_addrs, unsigned burst_size,
    unsigned group_size, int stride_step) {
  // It is guaranteed from the upstream passes that all units of a unit
  // operation will have same types.
  auto comp = dcc::getUnitType(
      unit_op.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
  if (!is_any_of(comp, L3LU, L3SU)) {
    return LogicalResult::failure();
  }

  // In the case of indirect load/store, we expect the immutable_addr of
  // kDirSrc and/or kDirDst to be 0. For example in an indirect load, the
  // kDirSrc mutable_addr is the EAR and kIndSrc mutable_addr is the IBR. The
  // IBR acts as the immutable_addr (base addr) for the load, so no need for the
  // kDirSrc immutable_addr.
  Value indirect_src_offset, indirect_dst_offset;
  if (access_details.has(MemoryOperandIndex::kIndSrc)) {
    Value ind_immutable_addr = immutable_addrs.get(MemoryOperandIndex::kIndSrc);
    auto cv = llvm::dyn_cast_or_null<arith::ConstantOp>(
        ind_immutable_addr.getDefiningOp());
    if (!(cv && mlir::dyn_cast<IntegerAttr>(cv.getValue()).getInt() == 0))
      indirect_src_offset = ind_immutable_addr;

    Value ignore = immutable_addrs.get(MemoryOperandIndex::kDirSrc);
    cv = llvm::dyn_cast_or_null<arith::ConstantOp>(ignore.getDefiningOp());
    if (!(cv && mlir::dyn_cast<IntegerAttr>(cv.getValue()).getInt() == 0)) {
      return op->emitError("Memory view index of direct src must be zero.");
    }
  }
  if (access_details.has(MemoryOperandIndex::kIndDst)) {
    Value ind_immutable_addr = immutable_addrs.get(MemoryOperandIndex::kIndDst);
    auto cv = llvm::dyn_cast_or_null<arith::ConstantOp>(
        ind_immutable_addr.getDefiningOp());
    if (!(cv && dyn_cast<IntegerAttr>(cv.getValue()).getInt() == 0))
      indirect_dst_offset = ind_immutable_addr;
    Value ignore = immutable_addrs.get(MemoryOperandIndex::kDirDst);
    cv = llvm::dyn_cast_or_null<arith::ConstantOp>(ignore.getDefiningOp());
    if (!(cv && dyn_cast<IntegerAttr>(cv.getValue()).getInt() == 0)) {
      return op->emitError("Memory view index of direct dst must be zero.");
    }
  }
  const auto& load_memory =
      access_details.has(MemoryOperandIndex::kIndSrc)
          ? access_details.get(MemoryOperandIndex::kIndSrc).getMemory()
          : access_details.get(MemoryOperandIndex::kDirSrc).getMemory();
  auto& load_mutable_addr = mutable_addrs.get(MemoryOperandIndex::kDirSrc);
  auto& load_mem_view_start_addr =
      immutable_addrs.get(MemoryOperandIndex::kDirSrc);
  const auto& store_memory =
      access_details.has(MemoryOperandIndex::kIndDst)
          ? access_details.get(MemoryOperandIndex::kIndDst).getMemory()
          : access_details.get(MemoryOperandIndex::kDirDst).getMemory();
  auto& store_mutable_addr = mutable_addrs.get(MemoryOperandIndex::kDirDst);
  auto& store_mem_view_start_addr =
      immutable_addrs.get(MemoryOperandIndex::kDirDst);
  auto total_elements = access_details.getFirst().getTotalElements();
  auto element_width = access_details.getFirst().getElementWidth();
  auto chunk_size = access_details.getFirst().getChunkSize();
  auto chunk_stride = access_details.getFirst().getChunkStride();
  auto shuffle_mode = access_details.getFirst().getShuffleMode();

  // Set Immutable address and increments for load related information
  Value load_increment;
  Value load_immutable_addr;
  bool perform_burst_or_group = burst_size > 0 || group_size > 0;
  if (failed(setImmutableAddrAndIncrements(
          *builder, comp, op, perform_burst_or_group, stride_step, burst_size,
          total_elements, load_mem_view_start_addr, load_immutable_addr,
          load_increment, unit_op))) {
    return op->emitOpError("could not set offset and load_increment.");
  }

  // Set Immutable address and increments for store related information
  Value store_increment;
  Value store_immutable_addr;
  if (failed(setImmutableAddrAndIncrements(
          *builder, comp, op, perform_burst_or_group, stride_step, burst_size,
          total_elements, store_mem_view_start_addr, store_immutable_addr,
          store_increment, unit_op))) {
    return op->emitOpError("could not set offset and load_increment.");
  }

  // In case of indirect load/store we want the immutable (base) addr to be the
  // indirectly loaded address (ie IBR). Since the immutable addresses are
  // expected to be zero for the indirect operands, we use the mutable addresses
  // of the *indirect* operands as the base addr.
  if (access_details.has(MemoryOperandIndex::kIndSrc))
    load_immutable_addr = mutable_addrs.get(MemoryOperandIndex::kIndSrc);
  if (access_details.has(MemoryOperandIndex::kIndDst))
    store_immutable_addr = mutable_addrs.get(MemoryOperandIndex::kIndDst);

  // If the pattern is a vector_load + vector_store, set the builder insertion
  // point to the store op to prevent any potential dominance issues.
  auto insert_pt = builder->saveInsertionPoint();
  if (auto load_op = dyn_cast<agen::VectorLoadOp>(op)) {
    auto result = load_op.getResult();
    if (auto store_op = dyn_cast<agen::VectorStoreOp>(*result.user_begin())) {
      builder->setInsertionPoint(store_op);
    }
  }

#if !defined(TOGGLE_INDIRECT_IMPL1)
  // This solution depends on enhancements tracked in issue 2013. It would
  // create a functional dependency on an optimization pass, unless we treat the
  // generic hoisting as canonicalization and optionally pull it out into its
  // own pass.
  if (indirect_src_offset)
    load_immutable_addr = sentient::AddOp::create(
        builder, op->getLoc(), load_immutable_addr.getType(),
        load_immutable_addr, indirect_src_offset);
  if (indirect_dst_offset)
    store_immutable_addr = sentient::AddOp::create(
        builder, op->getLoc(), store_immutable_addr.getType(),
        store_immutable_addr, indirect_dst_offset);
#endif

  // Collect the multicast info from the vector_load/composite_load_and_store
  // operation.
  Value multicast_info = nullptr;
  if (auto load_op = dyn_cast<VectorLoadOp>(op)) {
    auto mc_info = load_op.getMulticastInfo();
    if (mc_info) multicast_info = mc_info;
  } else if (auto load_op = dyn_cast<CompositeLoadAndStoreOp>(op)) {
    auto mc_info = load_op.getMulticastInfo();
    if (mc_info) multicast_info = mc_info;
  } else if (auto load_op = dyn_cast<CompositeIndirectLoadAndStoreOp>(op)) {
    multicast_info = load_op.getMulticastInfo();
  } else if (auto load_op = dyn_cast<SymbolicVectorLoadOp>(op)) {
    auto mc_info = load_op.getMulticastInfo();
    if (mc_info.has_value()) multicast_info = mc_info.value();
  } else
    llvm_unreachable(
        "Expecting a VectorLoadOp or Composite[Indirect]LoadAndStoreOp!");

  auto dbg_name_attr = getDbgNameAttr(op);
  sentient::SentientRoutingDirectionAttr routing_dir = nullptr;
  if (auto loadOp = dyn_cast<CompositeLoadAndStoreOp>(op)) {
    if (loadOp.getDir().has_value()) {
      auto agenDir = loadOp.getDir().value();
      SentientRoutingDirection srd;
      if (agenDir == AgenRoutingDirection::BothWays)
        srd = SentientRoutingDirection::BothWays;
      else if (agenDir == AgenRoutingDirection::Clockwise)
        srd = SentientRoutingDirection::Clockwise;
      else if (agenDir == AgenRoutingDirection::CounterClockwise)
        srd = SentientRoutingDirection::CounterClockwise;
      else
        srd = SentientRoutingDirection::PseudoRandom;
      routing_dir =
          SentientRoutingDirectionAttr::get(builder->getContext(), srd);
    }
  }
  auto load_and_store_op = sentient::LoadAndStoreOp::create(
      *builder, op->getLoc(), IndexType::get(op->getContext()),
      IndexType::get(op->getContext()), load_memory, store_memory,
      load_mutable_addr, load_immutable_addr, load_increment,
      store_mutable_addr, store_immutable_addr, store_increment, multicast_info,
      dbg_name_attr, total_elements, element_width, nullptr, chunk_size,
      chunk_stride, burst_size,
      symbolizeSentientShuffleMode(shuffle_mode).value(), stride_step,
      routing_dir);

  // In the case of scatter, distinguishing ibr-read from ibr-write is not
  // possible solely based on memory type of src vs dst. In such cases we add an
  // attribute to indicate ibr-write. In the case of gather, there is no
  // ambiguity, but we set the attribute for consistency.
  if (!access_details.has(MemoryOperandIndex::kIndDst) &&
      dcc::getUnitType(store_memory.getDefiningOp()) == L3IBR) {
    load_and_store_op->setAttr(
        sentient::LoadAndStoreOp::getIsIBRWriteAttrStrName(),
        builder->getI8IntegerAttr(1));
  }


}
// ---- 269/384  constructLoadAndExtractScalarOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2351  (114L)
LogicalResult e269_constructLoadAndExtractScalarOp(
    VectorLoadOp& load_op, dataflow::ProgramUnitOp& unit, SenComponents comp,
    AccessDetailsAffine& access_details, Value& mutable_addr,
    Value& immutable_addr, unsigned& extract_idx,
    SmallVectorImpl<Operation*>& ops_to_be_deleted) {
  // Pattern to match (load_op is %1):
  //   #map = <a one dimensional, identity map>
  //   #set = <a one dimensional set>
  //   #set2 = <a one dimensional set>
  //   %0 = dataflow.get_logical_memory_view %lx, <start_addr>
  //            {layout_map = #map} : index, index, memref<??x??>
  //   %1 = agen.vector_load %0[<start_offset>]
  //            {load_set = #set, load_order = #map}
  //            : memref<??x??>, vector<??x??>
  //   %2 = dataflow.get_logical_memory_view %virtual_ibr, 0
  //            {layout_map = #map} : index, index, memref<??x??>
  //   agen.vector_store %1, %2[0] {store_set = #set2, store_order = #map}
  //       : memref<??x??>, vector<??x??>
  //
  // Each indirect store op will have its own LoadAndExtractScalarOp
  // pattern.
  //
  // Note: It is expected isLoadAndExtractScalarPattern() was called on load_op
  // prior to this function which determines that the load_op is a part of the
  // pattern.

  // load_op should operate in a 1D space
  if (load_op.getLoadSet().getValue().getNumDims() != 1)
    return load_op->emitError("expecting 1D load_set");

  auto load_map = load_op.getLoadOrder();
  if (load_map.getNumDims() != 1 || !load_map.isIdentity())
    return load_op->emitError("expecting 1D identity map for load_map");

  if (load_op.getMapOperands().size() != 1)
    return load_op->emitError("expecting indices size 1");

  // The memory view feeding the load_op should also operate in a 1D space.
  auto load_mem_view = cast<dataflow::GetLogicalMemoryViewOp>(
      load_op.getMemRef().getDefiningOp());
  auto layout_map = load_mem_view.getLayoutMap();
  if (layout_map.getNumDims() != 1 || !layout_map.isIdentity())
    return load_op->emitError("expecting 1D identity map for layout_map");

  // The store_op using the load_op should operate on a 1D space with a start
  // address 0.
  auto store_op = cast<VectorStoreOp>(*load_op->getUsers().begin());
  if (!checkStoreOpFromExtractPattern(store_op))
    return store_op->emitError("store_op failed checks");

  // The indirect memory view used by store_op should operate in a 1D space with
  // a start address of 0.
  auto ind_mem_view = cast<dataflow::GetLogicalMemoryViewOp>(
      store_op.getMemRef().getDefiningOp());
  if (!checkIndirectMemViewForExtractOp(ind_mem_view))
    return ind_mem_view->emitError("indirect memory view does not pass checks");

  // ind_mem_view should only have 2 users:
  //   - store_op
  //   - an indirect load operation
  unsigned num_users = 0;
  Operation* ind_load_op = nullptr;
  for (auto user : ind_mem_view->getUsers()) {
    ++num_users;
    if (isa<IndirectVectorLoadOp, CompositeIndirectLoadOp>(user))
      ind_load_op = user;
    else if (user != store_op)
      return ind_mem_view->emitError("invalid user of indirect memory view");
  }
  if (num_users != 2)
    return ind_mem_view->emitError(
        "indirect memory view should only have 2 users");

  if (!ind_load_op)
    return ind_mem_view->emitError(
        "indirect memory view is not feeding an indirect load operation");

  // Set Immutable address and increments
  auto total_elements = access_details.getTotalElements();
  Value actual_immutable_addr, increment;
  OpBuilder builder(ind_mem_view);
  if (failed(setImmutableAddrAndIncrements(
          builder, comp, load_op, false, 0, 0, total_elements, immutable_addr,
          actual_immutable_addr, increment, unit)))
    return load_op->emitOpError("could not set immutable_addr and increment.");

  // Consumer for LoadAndExtractScalarOp is self.
  Value consumer;
  if (unit.getRegion().getArguments().empty())
    consumer = unit.getUnits()[0];
  else
    consumer = unit.getRegion().getArguments()[0];

  auto dbg_name_attr = getDbgNameAttr(load_op);
  auto element_width = access_details.getElementWidth();
  auto extract_op = sentient::LoadAndExtractScalarOp::create(
      builder, load_op->getLoc(), IndexType::get(load_op->getContext()),
      IndexType::get(load_op->getContext()), mutable_addr,
      actual_immutable_addr, increment, consumer, dbg_name_attr, total_elements,
      element_width);

  if (!extract_op)
    return ind_mem_view->emitError(
        "Unable to generate load_and_extract_scalar statement for the "
        "agen.vector_load operation");

  // Mark both the load_and_extract_scalar op and the indirect load op
  // associated to it with the extract_idx. This allows the indirect load op
  // to find the correct load_and_extract_scalar op to use when it is being
  // lowered. Increment the extract_idx for the next operation to use it.
  extract_op->setAttr("extract_idx", builder.getI8IntegerAttr(extract_idx));
  ind_load_op->setAttr("extract_idx", builder.getI8IntegerAttr(extract_idx));
  ++extract_idx;


}
// ---- 270/384  addStoreInputToDeleteList  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2987  (8L)
void e270_addStoreInputToDeleteList(
    Operation* input_op, SmallVectorImpl<Operation*>& to_be_deleted) {
  DT_CHECK(input_op);
  // Delete the receive or the constant_bitstream/shuffle operations.
  // The operation has already been verified by checkBasicConditions() to
  // ensure it fits one of the allowed patterns.
  to_be_deleted.push_back(input_op);
  if (auto shuffle_op = dyn_cast<vectorchain::ShuffleOp>(input_op))

}
// ---- 271/384  lowerCompositeMemoryInterleaveOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3775  (37L)
LogicalResult e271_lowerCompositeMemoryInterleaveOp(
    dataflow::ProgramUnitOp unit, Operation* op) {
  if (processInterleaveOp(unit, op).failed()) return LogicalResult::failure();

  auto interleave_op = cast_or_null<CompositeMemoryInterleaveOp>(op);
  DT_CHECK_MSG(interleave_op, "expecting a CompositeMemoryInterleaveOp");
  // Note: If granularity is not set, it is set to the maximum supported burst.
  //       Only L3 is supported at this time.
  SenSystemDef sysDef;
  auto granularity = sysDef.l3BurstSize;
  if (interleave_op->hasAttr("granularity"))
    granularity = interleave_op.getGranularity().value();

  Region& interleave_region = interleave_op.getRegion();
  SmallVector<Operation*> interleave_ops;
  OpBuilder builder(op);
  for (auto& region_op : interleave_region.getOps()) {
    if (isa<sentient::LoadAndSendOp, sentient::ReceiveAndStoreOp,
            sentient::LoadAndStoreOp>(region_op))
      interleave_ops.push_back(&region_op);
  }

  // Move the interleave_ops out of the region to prepare for interleaving.
  // If no ForOp is required, the operations will not be moved and they need
  // to be out of the region before the CompositeMemoryInterleaveOp is deleted
  // later.
  for (auto op_to_move : interleave_ops) op_to_move->moveBefore(op);

  SenComponents comp = dcc::getUnitType(op);
  if (dcc::burst_utils::processBurstSplitOrInterleave(
          builder, unit, comp, interleave_ops, granularity,
          interleave_loop_count_)
          .failed()) {
    op->emitOpError("interleaving burst failed");
    signalPassFailure();
  }


}
// ---- 272/384  insertInitializationStmt  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3834  (13L)
Value e272_insertInitializationStmt(
    SenComponents comp, Operation* loop_op, int imm_val,
    Value memory_view_start_addr) {
  OpBuilder builder(loop_op);
  auto const_op =
      mlir::arith::ConstantIndexOp::create(builder, loop_op->getLoc(), imm_val);

  if (is_any_of(comp, L3LU, L3SU)) return const_op.getResult();

  Value new_start_addr =
      cloneStartAddrOutsideLoop(loop_op, memory_view_start_addr);
  auto add_op = mlir::arith::AddIOp::create(
      builder, const_op->getLoc(), builder.getIndexType(), new_start_addr,

}
// ---- 273/384  lowerL0LXSyncOperationForAUnit  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:189  (180L)
LogicalResult e273_lowerL0LXSyncOperationForAUnit(
    mlir::Operation *op, OpBuilder builder,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops_corelet0,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops_corelet1,
    mlir::dataflow::GetUnitOp dst_unit) {
  auto send_op = llvm::dyn_cast<dataflow::SyncSendOp>(op);
  auto recv_op = llvm::dyn_cast<dataflow::SyncRecvOp>(op);
  auto implicit_sync_op =
      llvm::dyn_cast<dataflow::ImplicitSyncOnStreamingBufferOp>(op);
  DT_CHECK(send_op || recv_op || implicit_sync_op);
  SentientSyncMode sync_tag = SentientSyncMode::send;
  if (recv_op) sync_tag = SentientSyncMode::recv;

  auto dbg_name_attr = getDbgNameAttr(op);

  int implicit_sync_tile_size = -1;
  if (implicit_sync_op) {
    sync_tag = SentientSyncMode::sendrecv;
    auto parent_op = implicit_sync_op.getBufferSize().getDefiningOp();
    if (auto buffer_size_arith_def_op =
            llvm::dyn_cast<arith::ConstantIndexOp>(parent_op)) {
      implicit_sync_tile_size = buffer_size_arith_def_op.value();
    } else if (auto buffer_size_sen_def_op =
                   llvm::dyn_cast<sentient::ConstantOp>(parent_op)) {
      implicit_sync_tile_size = buffer_size_sen_def_op.getValue();
    } else {
      DT_ERROR("sync buffer size has to be a constant op");
    }
  }

  if (send_op && !send_op.getWaitImmediatelyForAsyncTransfers().has_value()) {
    op->emitError("Unknown async transfers modes for sentient");
    return LogicalResult::failure();
  }
  std::string dst_unit_name = dst_unit.getType().str();
  SenComponents dst_comp =
      EnumsConversion::stringToSenComponents.find(dst_unit_name)->second;
  if (dst_comp != SenComponents::L3LU && dst_comp != SenComponents::L3SU &&
      !dst_unit->hasAttr("corelet")) {
    dst_unit->emitError("Unknown corelet information for sentient");
    return LogicalResult::failure();
  }
  std::string src_unit_name;
  if (src_unit_ops_corelet0.size() > 0) {
    src_unit_name = getUnitNameFromAListOfGetUnitOp(src_unit_ops_corelet0);
  } else if (src_unit_ops_corelet1.size() > 0) {
    src_unit_name = getUnitNameFromAListOfGetUnitOp(src_unit_ops_corelet1);
  } else {
    op->emitError("List of src units cannot be empty");
    return LogicalResult::failure();
  }
  DT_CHECK(src_unit_ops_corelet0.size() == 0 ||
           src_unit_ops_corelet1.size() == 0 ||
           getUnitNameFromAListOfGetUnitOp(src_unit_ops_corelet0) ==
               getUnitNameFromAListOfGetUnitOp(src_unit_ops_corelet1));

  SenComponents src_comp =
      EnumsConversion::stringToSenComponents.find(src_unit_name)->second;
  auto gen_comp = EnumsConversion::senCompToGenericComp.at(src_comp);
  switch (gen_comp) {
    case SenComponents::L0LU:
    case SenComponents::L0SU:
      if (send_op && !send_op.getWaitImmediatelyForAsyncTransfers().value()) {
        op->emitError("Unsuported sync operation for L0");
        break;
      }
      if ((isSenComponentL0LU(src_comp) && isSenComponentL0SU(dst_comp)) ||
          (isSenComponentL0SU(src_comp) && isSenComponentL0LU(dst_comp))) {
        if (isSenComponentL0LU(dst_comp)) dst_unit_name = "l0lu";
        if (isSenComponentL0SU(dst_comp)) dst_unit_name = "l0su";
        if (sentient::SyncOp::create(
                builder, op->getLoc(), dbg_name_attr,
                SentientSyncModeAttr::get(builder.getContext(), sync_tag),
                ArrayAttr::get(
                    builder.getContext(),
                    {SentientLoadConsumerAttr::get(
                        op->getContext(),
                        symbolizeSentientLoadConsumer(dst_unit_name).value())}),
                builder.getBoolAttr(false),
                builder.getSI32IntegerAttr(implicit_sync_tile_size)))
          return LogicalResult::success();
        else
          op->emitError("Cannot create sync operation");
      }
      op->emitError("Unknown lowering of the L0 sync operation");
      break;
    case SenComponents::LXLU:
    case SenComponents::LXSU:
      DT_CHECK_MSG(implicit_sync_tile_size == -1,
                   "LX doesn't have implicit sync");
      if (dst_comp == SenComponents::L3LU || dst_comp == SenComponents::L3SU) {
        if (sentient::SyncOp::create(
                builder, op->getLoc(), dbg_name_attr,
                SentientSyncModeAttr::get(builder.getContext(), sync_tag),
                ArrayAttr::get(
                    builder.getContext(),
                    {SentientLoadConsumerAttr::get(
                        op->getContext(),
                        symbolizeSentientLoadConsumer(dst_unit_name).value())}),
                builder.getBoolAttr(false), builder.getSI32IntegerAttr(-1)))
          return LogicalResult::success();
        else
          op->emitError("Cannot create sync operation");
      }
      if (dst_comp == SenComponents::LXLU || dst_comp == SenComponents::LXSU) {
        if (send_op && !send_op.getWaitImmediatelyForAsyncTransfers().value()) {
          op->emitError("Unsuported sync operation for LX");
          break;
        }

        if (src_unit_ops_corelet0.size() == 0 ||
            src_unit_ops_corelet1.size() == 0) {
          if ((src_unit_ops_corelet0.size() > 0 &&
               dst_unit->getAttr("corelet") == builder.getI32IntegerAttr(1)) ||
              (src_unit_ops_corelet1.size() > 0 &&
               dst_unit->getAttr("corelet") == builder.getI32IntegerAttr(0))) {
            dst_unit_name += "N";
          }
          if (sentient::SyncOp::create(
                  builder, op->getLoc(), dbg_name_attr,
                  SentientSyncModeAttr::get(builder.getContext(), sync_tag),
                  ArrayAttr::get(
                      builder.getContext(),
                      {SentientLoadConsumerAttr::get(
                          op->getContext(),
                          symbolizeSentientLoadConsumer(dst_unit_name)
                              .value())}),
                  builder.getBoolAttr(false), builder.getSI32IntegerAttr(-1)))
            return LogicalResult::success();
          else
            op->emitError("Cannot create sync operation");
        } else {
          // create uniform regions
          std::vector<mlir::Value> src_unit_res;
          for (auto unit_op : src_unit_ops_corelet0)
            src_unit_res.push_back(unit_op.getResult(0));
          for (auto unit_op : src_unit_ops_corelet1)
            src_unit_res.push_back(unit_op.getResult(0));
          llvm::SmallVector<Attribute, 2> list_sizes = {
              builder.getI32IntegerAttr(src_unit_ops_corelet0.size()),
              builder.getI32IntegerAttr(src_unit_ops_corelet1.size())};
          OpBuilder builder_region0(op), builder_region1(op);
          std::tie(builder_region0, builder_region1) =
              createUniformRegionsWithTwoRegionsNoResult(
                  builder, op->getLoc(), src_unit_res, list_sizes);
          std::string dst_unit_name_for_src_corelet0 = dst_unit_name;
          std::string dst_unit_name_for_src_corelet1 = dst_unit_name;
          if (dst_unit->getAttr("corelet") == builder.getI32IntegerAttr(1))
            dst_unit_name_for_src_corelet0 += "N";
          else /*dst_unit corelet = 0*/
            dst_unit_name_for_src_corelet1 += "N";

          sentient::SyncOp::create(
              builder_region0, op->getLoc(), dbg_name_attr,
              SentientSyncModeAttr::get(builder_region0.getContext(), sync_tag),
              ArrayAttr::get(builder_region0.getContext(),
                             {SentientLoadConsumerAttr::get(
                                 builder_region0.getContext(),
                                 symbolizeSentientLoadConsumer(
                                     dst_unit_name_for_src_corelet0)
                                     .value())}),
              builder.getBoolAttr(false),
              builder_region0.getSI32IntegerAttr(-1));
          sentient::SyncOp::create(
              builder_region1, op->getLoc(), dbg_name_attr,
              SentientSyncModeAttr::get(builder_region1.getContext(), sync_tag),
              ArrayAttr::get(builder_region1.getContext(),
                             {SentientLoadConsumerAttr::get(
                                 builder_region1.getContext(),
                                 symbolizeSentientLoadConsumer(
                                     dst_unit_name_for_src_corelet1)
                                     .value())}),
              builder.getBoolAttr(false),
              builder_region1.getSI32IntegerAttr(-1));
          return LogicalResult::success();
        }
      }
      op->emitError("Unknown lowering of the LXLU/LXSU sync operation");
      break;
    default:

}}
// ---- 274/384  lowerL0LXSyncOperationForAGroupOfUnits  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:438  (222L)
DataflowToSentientLoweringPass::lowerL0LXSyncOperationForAGroupOfUnits(
    mlir::Operation *op, OpBuilder builder,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops_corelet0,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops_corelet1,
    dataflow::CreateGroupOp dst_unit_group) {
  auto send_op = llvm::dyn_cast<dataflow::SyncSendOp>(op);
  auto recv_op = llvm::dyn_cast<dataflow::SyncRecvOp>(op);
  auto implicit_sync_op =
      llvm::dyn_cast<dataflow::ImplicitSyncOnStreamingBufferOp>(op);
  DT_CHECK(send_op || recv_op || implicit_sync_op);
  SentientSyncMode sync_tag = SentientSyncMode::send;
  if (recv_op) sync_tag = SentientSyncMode::recv;

  int implicit_sync_tile_size = -1;
  if (implicit_sync_op) {
    sync_tag = SentientSyncMode::sendrecv;
    auto parent_op = implicit_sync_op.getBufferSize().getDefiningOp();
    if (auto buffer_size_arith_def_op =
            llvm::dyn_cast<arith::ConstantIndexOp>(parent_op)) {
      implicit_sync_tile_size = buffer_size_arith_def_op.value();
    } else if (auto buffer_size_sen_def_op =
                   llvm::dyn_cast<sentient::ConstantOp>(parent_op)) {
      implicit_sync_tile_size = buffer_size_sen_def_op.getValue();
    } else {
      DT_ERROR("sync buffer size has to be a constant op");
    }
  }

  auto dbg_name_attr = getDbgNameAttr(op);

  if (send_op && !send_op.getWaitImmediatelyForAsyncTransfers().has_value()) {
    op->emitError("Unknown async transfers modes for sentient");
    return LogicalResult::failure();
  }

  std::string src_unit_name;
  if (src_unit_ops_corelet0.size() > 0) {
    src_unit_name = getUnitNameFromAListOfGetUnitOp(src_unit_ops_corelet0);
  } else if (src_unit_ops_corelet1.size() > 0) {
    src_unit_name = getUnitNameFromAListOfGetUnitOp(src_unit_ops_corelet1);
  } else {
    op->emitError("List of src units cannot be empty");
    return LogicalResult::failure();
  }
  DT_CHECK(src_unit_ops_corelet0.size() == 0 ||
           src_unit_ops_corelet1.size() == 0 ||
           getUnitNameFromAListOfGetUnitOp(src_unit_ops_corelet0) ==
               getUnitNameFromAListOfGetUnitOp(src_unit_ops_corelet1));

  SenComponents src_comp =
      EnumsConversion::stringToSenComponents.find(src_unit_name)->second;
  auto gen_comp = EnumsConversion::senCompToGenericComp.at(src_comp);
  SmallVector<Attribute, 1> dst_unit_names;
  switch (gen_comp) {
    case SenComponents::L0LU:
    case SenComponents::L0SU: {
      if (send_op && !send_op.getWaitImmediatelyForAsyncTransfers().value()) {
        op->emitError("Unsuported sync operation for L0");
        break;
      }

      for (auto dst_unit_itr : dst_unit_group.getUnitIds()) {
        auto dst_unit =
            llvm::dyn_cast<dataflow::GetUnitOp>(dst_unit_itr.getDefiningOp());
        SenComponents dst_comp = EnumsConversion::stringToSenComponents
                                     .find(dst_unit.getType().str())
                                     ->second;
        if (dst_comp != SenComponents::L3LU &&
            dst_comp != SenComponents::L3SU && !dst_unit->hasAttr("corelet")) {
          dst_unit->emitError("Unknown corelet information for sentient");
          break;
        }
        if ((isSenComponentL0LU(src_comp) && !isSenComponentL0SU(dst_comp)) ||
            (isSenComponentL0SU(src_comp) && !isSenComponentL0LU(dst_comp))) {
          op->emitError("Unknown lowering of the L0 sync operation");
          break;
        }
        std::string dst_unit_name;
        if (isSenComponentL0LU(dst_comp)) dst_unit_name = "l0lu";
        if (isSenComponentL0SU(dst_comp)) dst_unit_name = "l0su";
        auto dst_unit_attr = SentientLoadConsumerAttr::get(
            op->getContext(),
            symbolizeSentientLoadConsumer(dst_unit_name).value());
        if (std::find(dst_unit_names.begin(), dst_unit_names.end(),
                      dst_unit_attr) == dst_unit_names.end())
          dst_unit_names.push_back(dst_unit_attr);
      }
      if (sentient::SyncOp::create(
              builder, op->getLoc(), dbg_name_attr,
              SentientSyncModeAttr::get(builder.getContext(), sync_tag),
              ArrayAttr::get(op->getContext(), dst_unit_names),
              BoolAttr::get(op->getContext(), false),
              builder.getSI32IntegerAttr(implicit_sync_tile_size)))
        return LogicalResult::success();

      op->emitError("Cannot create sync operation");
      break;
    }

    case SenComponents::LXLU:
    case SenComponents::LXSU: {
      DT_CHECK_MSG(implicit_sync_tile_size == -1,
                   "LX doesn't have implicit sync");
      if (send_op && !send_op.getWaitImmediatelyForAsyncTransfers().value()) {
        op->emitError("Unsuported sync operation for LX");
        break;
      }
      SmallVector<Attribute, 1> dst_unit_names_for_src_corelet0,
          dst_unit_names_for_src_corelet1, dst_unit_names_for_src_l3;
      for (auto dst_unit_itr : dst_unit_group.getUnitIds()) {
        auto dst_unit =
            llvm::dyn_cast<dataflow::GetUnitOp>(dst_unit_itr.getDefiningOp());
        SenComponents dst_comp = EnumsConversion::stringToSenComponents
                                     .find(dst_unit.getType().str())
                                     ->second;
        std::string dst_unit_name = dst_unit.getType().str();
        if (is_any_of(dst_comp, SenComponents::LXLU, SenComponents::LXLU0,
                      SenComponents::LXLU1, SenComponents::LXSU,
                      SenComponents::LXSU0, SenComponents::LXSU1)) {
          if (!dst_unit->hasAttr("corelet")) {
            dst_unit->emitError("Unknown corelet information for sentient");
            break;
          }
          auto dst_corelet = dst_unit->getAttr("corelet");
          if (is_any_of(dst_comp, SenComponents::LXLU0, SenComponents::LXLU1,
                        SenComponents::LXSU0, SenComponents::LXSU1))
            dst_unit_name.pop_back();  // delete last char "0" or "1"

          std::string dst_unit_name_neighbour = dst_unit_name + "N";
          if (dst_unit->getAttr("corelet") == builder.getI32IntegerAttr(0)) {
            pushBackTheUnitToListIfDoesnotExist(dst_unit_name,
                                                dst_unit_names_for_src_corelet0,
                                                op->getContext());
            pushBackTheUnitToListIfDoesnotExist(dst_unit_name_neighbour,
                                                dst_unit_names_for_src_corelet1,
                                                op->getContext());
          }
          if (dst_unit->getAttr("corelet") == builder.getI32IntegerAttr(1)) {
            pushBackTheUnitToListIfDoesnotExist(dst_unit_name_neighbour,
                                                dst_unit_names_for_src_corelet0,
                                                op->getContext());
            pushBackTheUnitToListIfDoesnotExist(dst_unit_name,
                                                dst_unit_names_for_src_corelet1,
                                                op->getContext());
          }
        } else if (dst_comp == SenComponents::L3LU ||
                   dst_comp == SenComponents::L3SU) {
          pushBackTheUnitToListIfDoesnotExist(
              dst_unit_name, dst_unit_names_for_src_l3, op->getContext());
        } else {
          op->emitError("Unknown lowering of the LX sync operation");
          break;
        }
      }

      if (src_unit_ops_corelet0.size() > 0 &&
          src_unit_ops_corelet1.size() == 0) {
        for (auto unit_attr : dst_unit_names_for_src_corelet0) {
          dst_unit_names.push_back(unit_attr);
        }
        for (auto unit_attr : dst_unit_names_for_src_l3) {
          dst_unit_names.push_back(unit_attr);
        }
        if (sentient::SyncOp::create(
                builder, op->getLoc(), dbg_name_attr,
                SentientSyncModeAttr::get(builder.getContext(), sync_tag),
                ArrayAttr::get(op->getContext(), dst_unit_names),
                BoolAttr::get(op->getContext(), false),
                builder.getSI32IntegerAttr(-1)))
          return LogicalResult::success();
      } else if (src_unit_ops_corelet1.size() > 0 &&
                 src_unit_ops_corelet0.size() == 0) {
        for (auto unit_attr : dst_unit_names_for_src_corelet1) {
          dst_unit_names.push_back(unit_attr);
        }
        for (auto unit_attr : dst_unit_names_for_src_l3) {
          dst_unit_names.push_back(unit_attr);
        }
        if (sentient::SyncOp::create(
                builder, op->getLoc(), dbg_name_attr,
                SentientSyncModeAttr::get(builder.getContext(), sync_tag),
                ArrayAttr::get(op->getContext(), dst_unit_names),
                BoolAttr::get(op->getContext(), false),
                builder.getSI32IntegerAttr(-1)))
          return LogicalResult::success();
      } else {
        std::vector<mlir::Value> src_unit_res;
        for (auto unit_op : src_unit_ops_corelet0)
          src_unit_res.push_back(unit_op.getResult(0));
        for (auto unit_op : src_unit_ops_corelet1)
          src_unit_res.push_back(unit_op.getResult(0));
        llvm::SmallVector<Attribute, 2> list_sizes = {
            builder.getI32IntegerAttr(src_unit_ops_corelet0.size()),
            builder.getI32IntegerAttr(src_unit_ops_corelet1.size())};
        OpBuilder builder_region0(op), builder_region1(op);
        std::tie(builder_region0, builder_region1) =
            createUniformRegionsWithTwoRegionsNoResult(
                builder, op->getLoc(), src_unit_res, list_sizes);
        sentient::SyncOp::create(
            builder_region0, op->getLoc(), dbg_name_attr,
            SentientSyncModeAttr::get(builder_region0.getContext(), sync_tag),
            ArrayAttr::get(op->getContext(), dst_unit_names_for_src_corelet0),
            BoolAttr::get(op->getContext(), false),
            builder_region0.getSI32IntegerAttr(-1));
        sentient::SyncOp::create(
            builder_region1, op->getLoc(), dbg_name_attr,
            SentientSyncModeAttr::get(builder_region1.getContext(), sync_tag),
            ArrayAttr::get(op->getContext(), dst_unit_names_for_src_corelet0),
            BoolAttr::get(op->getContext(), false),
            builder_region1.getSI32IntegerAttr(-1));
        sentient::SyncOp::create(
            builder, op->getLoc(), dbg_name_attr,
            SentientSyncModeAttr::get(builder.getContext(), sync_tag),
            ArrayAttr::get(op->getContext(), dst_unit_names_for_src_l3),
            BoolAttr::get(op->getContext(), false),
            builder.getSI32IntegerAttr(-1));
        return LogicalResult::success();
      }
      op->emitError("Cannot create sync operation with src LX");
      break;
    }
    default:

}}
// ---- 275/384  LowerSymbolQueryMap  —  dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:40  (68L)
void e275_LowerSymbolQueryMap(
    SenComponents comp, symbol::SymbolQueryMapOp query_map,
    std::vector<Operation*>& to_be_deleted_list) {
  auto immutable_mapping =
      query_map.getMap().getDefiningOp<symbol::SymbolImmutableMappingOp>();
  DT_CHECK(immutable_mapping);
  to_be_deleted_list.push_back(query_map);
  to_be_deleted_list.push_back(immutable_mapping);

  // Special case: Query map has only one key, value pair.
  // Then simply replace the query_map by the value.
  if (immutable_mapping.getKeys().size() == 1) {
    query_map.replaceAllUsesWith(immutable_mapping.getValues().back());
    return;
  }

  // Determine the number of uses of query_map other than in loop bounds.
  int num_non_loop_bound_uses = 0;
  bool found_loop_bound_use = false;

  // Store a list of (owner operation, operand number) for each use.
  // This is needed as we will replace the uses one-by-one.
  llvm::SmallVector<std::pair<Operation*, unsigned>>
      query_map_use_ownerop_and_operandnum;
  for (auto& use : query_map.getResult().getUses()) {
    Operation* owner_op = use.getOwner();
    query_map_use_ownerop_and_operandnum.push_back(
        std::make_pair(owner_op, use.getOperandNumber()));
    if (auto sentient_for = llvm::dyn_cast<sentient::ForOp>(owner_op)) {
      if (sentient_for.getBound().getDefiningOp() == query_map) {
        found_loop_bound_use = true;
        continue;
      }
    }
    ++num_non_loop_bound_uses;
  }

  // Create a conditional from the mapping. Its number of results is determined
  // as follows:
  // - 1 (JCR) result for loop bound uses (if any)
  // - 1 (LRF) result for non-loop-bound-uses (if any) for LX or below
  // - num_non_loop_bound_uses results for L3
  int num_results =
      (found_loop_bound_use ? 1 : 0) +
      ((num_non_loop_bound_uses > 0)
           ? (is_any_of(comp, L3LU, L3SU) ? num_non_loop_bound_uses : 1)
           : 0);
  sentient::IfOp if_op = createIfOpFromMapping(query_map, num_results);

  // Replace uses in loop bounds with the first IfOp result.
  // Replace other uses with different IfOp results if L3 or with the second
  // IfOp result if LX or below. Note: getUses() returns the list of uses in
  // reverse order so we process it backwards.
  int if_op_result_num_to_use_for_replacement = (found_loop_bound_use ? 1 : 0);
  for (auto it = query_map_use_ownerop_and_operandnum.rbegin();
       it != query_map_use_ownerop_and_operandnum.rend(); ++it) {
    Operation* owner_op = std::get<0>(*it);
    unsigned operand_num = std::get<1>(*it);
    if (auto sentient_for = llvm::dyn_cast<sentient::ForOp>(owner_op)) {
      if (sentient_for.getBound().getDefiningOp() == query_map) {
        sentient_for.setOperand(operand_num, if_op.getResults()[0]);
        continue;
      }
    }

    owner_op->setOperand(
        operand_num,
        if_op.getResults()[if_op_result_num_to_use_for_replacement]);

}}
// ---- 276/384  setReuseInformation  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/OperandReuse.cpp:17  (44L)
std::optional<bool> e276_setReuseInformation(
    Operation *user,
    llvm::SmallVectorImpl<std::optional<VectorOperand>> &operands) {
  // TODO: Check if there are no other operations (send, receive, store,
  // load) between the from and to operands.

  for (int i = 0; i < operands.size(); i++) {
    if (!operands[i].has_value()) return std::nullopt;
    auto &operand_i = operands[i].value();
    if (operand_i.type_ != Constant) {
      if (!this->insertIfNotExists(operand_i.op_)) {
        operand_i.setValue("latch");
      } else if (this->getAbsorbtionFlag(operand_i.op_).value()) {
        operand_i.setValue("latch");
      } else if (VectorOperand::sameBlock(user, operands[i]).failed()) {
        operand_i.setValue("latch");
      } else {
        for (int j = 0; j < i; j++) {
          if (!operands[j].has_value()) return std::nullopt;
          auto &operand_j = operands[j].value();
          if (operand_j.getName() == operand_i.getName() &&
              operand_i.type_ != LRF) {
            if (dominance_info_.dominates(operand_j.op_, operand_i.op_)) {
              operand_j.setValue("latch");
            } else if (dominance_info_.dominates(operand_i.op_,
                                                 operand_j.op_)) {
              operand_i.setValue("latch");
            } else {
              user->emitError(
                  "Impossible to determine dominance among operands");
              return std::nullopt;
            }
          }
        }
      }
    }
  }

  for (auto &operand : operands) {
    auto &from = operand.value();
    if (from.type_ != Constant && from.getName() != "latch") {
      this->setReuseFlag(from.op_);
    }
  }

}
// ---- 277/384  getGCVTorFCVTTypeFromIndicesAndCastInputs  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:188  (108L)
std::string e277_getGCVTorFCVTTypeFromIndicesAndCastInputs(
    std::vector<int> indices, int repetition,
    llvm::SmallVector<vectorchain::CastOp> sources,
    const dcc::DccExtContext& dcc_ext_ctx) {
  struct gcvt_fcvt_type {
    const std::string name_;
    const int num_inputs_;
    std::vector<int> vec_;
    int repetition_ = 8;
    int size_ = 0;
    std::string src_input_type_str_, src_output_type_str_;
    gcvt_fcvt_type(std::string name, int num_inputs, int repetition,
                   std::vector<int> vec, std::string src_input_type_str,
                   std::string src_output_type_str)
        : name_(name),
          num_inputs_(num_inputs),
          vec_(vec),
          src_input_type_str_(src_input_type_str),
          src_output_type_str_(src_output_type_str),
          repetition_(repetition),
          size_(vec.size()) {};
  };

  std::vector<gcvt_fcvt_type> gcvt_fcvt_insts = {
      {"gcvt_imm0",
       2,
       8,
       {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
       "fp16",
       "f8E5M2"},
      {"gcvt_imm1", 1, 8, {0, 1, 2, 3, 4, 5, 6, 7}, "f8E5M2", "fp16"},
      {"gcvt_imm2", 1, 8, {8, 9, 10, 11, 12, 13, 14, 15}, "f8E5M2", "fp16"},
      {"gcvt_imm4",
       2,
       8,
       {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
       "fp16",
       "f8E4M3FN"},
      {"gcvt_imm5", 1, 8, {0, 1, 2, 3, 4, 5, 6, 7}, "f8E4M3FN", "fp16"},
      {"gcvt_imm6", 1, 8, {8, 9, 10, 11, 12, 13, 14, 15}, "f8E4M3FN", "fp16"},
      {"gcvt_imm8", 1, 8, {0, 1, 2, 3, 4, 5, 6, 7}, "i4", "i16"},
      {"gcvt_imm16",
       1,
       4,
       {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
       "fp16",
       "bf16"},
      {"gcvt_imm17",
       1,
       4,
       {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15},
       "bf16",
       "fp16"},
      {"gcvt_imm24",
       2,
       8,
       {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15},
       "fp16",
       "f8E5M2"},
      {"gcvt_imm28",
       2,
       8,
       {0, 8, 1, 9, 2, 10, 3, 11, 4, 12, 5, 13, 6, 14, 7, 15},
       "fp16",
       "f8E4M3FN"},
      {"fcvt_imm0", 1, 8, {0, 1, 2, 3}, "fp16", "fp32"},
      {"fcvt_imm1", 1, 8, {4, 5, 6, 7}, "fp16", "fp32"},
      {"fcvt_imm2", 2, 8, {0, 1, 2, 3, 4, 5, 6, 7}, "fp32", "fp16"},
      {"fcvt_imm3",
       2,
       8,
       {0, 1, 2, 3, 4, 5, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1},
       "fp32",
       "f8E5M2"},
      /* TODO: Add FCVT mode 4. Need a way to differentiate it from mode 2.
      {"fcvt_imm4", 2, 8, {0, 1, 2, 3, 4, 5, 6, 7}, "fp32", "fp16"},
      */
      {"fcvt_imm5", 1, 8, {0, 1, 2, 3}, "bf16", "fp32"},
      {"fcvt_imm6", 1, 8, {4, 5, 6, 7}, "bf16", "fp32"},
      {"fcvt_imm7", 2, 8, {0, 1, 2, 3, 4, 5, 6, 7}, "fp32", "bf16"}};

  for (auto instr : gcvt_fcvt_insts) {
    bool found = true;
    if (indices.size() != instr.size_) continue;
    if (repetition != instr.repetition_) continue;
    if (instr.num_inputs_ != sources.size()) continue;

    for (int i = 0; i < indices.size() && found; i++) {
      if (indices[i] != instr.vec_[i]) found = false;
    }

    for (auto& src : sources) {
      OpBuilder builder(src);

      auto input_elem_type =
          mlir::dataflow::utils::getElementType(src.getInput().getType());
      std::string input_elem_type_str =
          convertTypeToString(builder, input_elem_type);
      auto output_elem_type =
          mlir::dataflow::utils::getElementType(src.getResult().getType());
      std::string output_elem_type_str =
          convertTypeToString(builder, output_elem_type);

      if (input_elem_type_str != instr.src_input_type_str_ ||
          output_elem_type_str != instr.src_output_type_str_) {
        found = false;
      }
    }

}}
// ---- 278/384  getOperandFromShuffleOp  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:311  (36L)
std::optional<VectorOperand> e278_getOperandFromShuffleOp(
    const dcc::DccExtContext &dcc_ext_ctx, vectorchain::ShuffleOp &op,
    const SenComponents comp) {
  DT_CHECK(isa<vectorchain::ShuffleOp>(op));
  if (vectorchain::utils::isShuffleNFWDVersion0(op)) {
    auto *parent = op.getOperand(0).getDefiningOp();
    auto operand = getOperand(dcc_ext_ctx, parent, comp, true);
    operand.value().type_ = NFWD;
    operand.value().setValue("nfwd0");
    return operand;
    //    return VectorOperand(NFWD, "nfwd0", op.getOperation());
  } else if (vectorchain::utils::isShuffleNFWDVersion2(op)) {
    auto *parent = op.getOperand(0).getDefiningOp();
    auto operand = getOperand(dcc_ext_ctx, parent, comp, true);
    operand.value().type_ = NFWD;
    operand.value().setValue("nfwd2");
    return operand;
    //    return VectorOperand(NFWD, "nfwd2", op.getOperation());
  } else if (vectorchain::utils::isCustomVectorTrivialShuffle(op)) {
    auto *parent = op.getOperand(0).getDefiningOp();
    auto const_bit_op = dyn_cast<vectorchain::ConstantBitstreamOp>(parent);
    DT_CHECK(const_bit_op);
    auto operand = getOperandFromConstantBitstreamOp(
        const_bit_op, /*is_constant_splatted_vector*/ true);
    operand.value().type_ = ConstantBitstream;
    return operand;
  }

  if (op.getOperation()->user_begin() != op.getOperation()->user_end()) {
    auto use = *op.getOperation()->user_begin();
    if (auto send_op = llvm::dyn_cast<dataflow::SendOp>(use)) {
      return getOperandFromSendOp(dcc_ext_ctx, send_op, comp);
    } else if (auto store_op = llvm::dyn_cast<agen::VectorStoreOp>(use)) {
      return getOperandFromLoadOrStoreOp(store_op.getOperation());
    }
    return std::nullopt;

}}
// ---- 279/384  createSplatOperation  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/Splat.cpp:70  (110L)
LogicalResult createSplatOperation(Operation *op, VectorOperand &from,
                                   VectorOperand &to,
                                   mlir::ConversionPatternRewriter &rewriter,
                                   SenComponents comp,
                                   const SenSystemDef &sys_def) {
  // default mask value
  auto mask_const_op = sentient::ConstantOp::create(rewriter, op->getLoc(),
                                                    rewriter.getIndexType(), 0);
  auto op_dbg_name = dataflow::getDbgNameAttr(op);
  if (auto const_op = llvm::dyn_cast<mlir::arith::ConstantOp>(from.op_)) {
    if (auto splat_attr = mlir::cast<SplatElementsAttr>(const_op.getValue())) {
      auto splat_value = splat_attr.getSplatValue<Attribute>();
      if (mlir::isa<IntegerAttr>(splat_value)) {
        int val = mlir::cast<IntegerAttr>(splat_value).getInt();
        auto sentient_const_op = sentient::ConstantOp::create(
            rewriter, op->getLoc(), rewriter.getI64Type(), val);
        auto dest_storage = sentient::LogicalPortOp::create(
            rewriter, op->getLoc(), rewriter.getIndexType(),
            sentient::SentientComputePortAttr::get(
                rewriter.getContext(),
                symbolizeSentientComputePort(to.getName()).value()));
        auto splat_op = sentient::SplatOp::create(
            rewriter, op->getLoc(), sentient_const_op.getResult(),
            dest_storage.getResult(), mask_const_op.getResult(), op_dbg_name,
            SentientSplatPad::none,
            symbolizeSentientPrecision(getInputPrecisionFromOperand(from))
                .value());
      } else {
        op->emitError("Unable to create the splat operation");
        return failure();
      }
    }
  } else if (auto shuffle_op = llvm::dyn_cast<vectorchain::ShuffleOp>(op)) {
    mlir::Value result;
    llvm::SmallVector<int, 8> vals;
    auto mask_operand = shuffle_op.getMask();
    if (mask_operand) {
      std::optional<Value> mask_val =
          comp != PT ? vectorchain::getMaskValueForNonPT(
                           op, mask_operand.getDefiningOp(), sys_def, rewriter)
                     : std::nullopt;
      if (mask_val.has_value())
        mask_const_op = mask_val.value().getDefiningOp<sentient::ConstantOp>();
    }

    if (auto const_bit_op =
            llvm::dyn_cast<vectorchain::ConstantBitstreamOp>(from.op_)) {
      result = createSentientConstants(rewriter, const_bit_op, shuffle_op);
    } else if (auto query_op =
                   llvm::dyn_cast<mlir::uniform::QueryMapOp>(from.op_)) {
      llvm::SmallVector<mlir::Value> new_map_keys, new_map_values;
      auto map_op = llvm::dyn_cast<mlir::uniform::DefImmutableMappingOp>(
          query_op.getMap().getDefiningOp());
      for (auto pair : llvm::zip(map_op.getKeys(), map_op.getValues())) {
        auto val = std::get<1>(pair);
        if (auto const_bit_op_from_val =
                llvm::dyn_cast<vectorchain::ConstantBitstreamOp>(
                    val.getDefiningOp())) {
          auto tmp_result = createSentientConstants(
              rewriter, const_bit_op_from_val, shuffle_op);
          new_map_values.emplace_back(tmp_result);
          new_map_keys.emplace_back(std::get<0>(pair));
        }
      }

      // create a map and query
      auto new_map = uniform::DefImmutableMappingOp::create(
          rewriter, rewriter.getUnknownLoc(), rewriter.getIndexType(),
          new_map_keys, new_map_values);

      // create a query op
      auto new_query_op = uniform::QueryMapOp::create(
          rewriter, new_map.getLoc(), new_map_values.front().getType(),
          new_map.getResult(), query_op.getKey());
      result = new_query_op.getResult();
    } else if (isa<dataflow::ReceiveOp, agen::VectorLoadOp>(from.op_)) {
      auto src_storage = sentient::LogicalPortOp::create(
          rewriter, op->getLoc(), rewriter.getIndexType(),
          sentient::SentientComputePortAttr::get(
              rewriter.getContext(),
              symbolizeSentientComputePort(from.getName()).value()));
      result = src_storage.getResult();
    }
    auto dest_storage = sentient::LogicalPortOp::create(
        rewriter, op->getLoc(), rewriter.getIndexType(),
        sentient::SentientComputePortAttr::get(
            rewriter.getContext(),
            symbolizeSentientComputePort(to.getName()).value()));

    sentient::SplatOp splat_op;
    if (vectorchain::utils::isFirstElemSplat(shuffle_op)) {
      // For SPLAT instruction with no padding
      splat_op = sentient::SplatOp::create(
          rewriter, op->getLoc(), result, dest_storage.getResult(),
          mask_const_op.getResult(), op_dbg_name, SentientSplatPad::none,
          symbolizeSentientPrecision(getInputPrecisionFromOperand(from))
              .value());
    } else if (vectorchain::utils::isPadLeftFor8FirstElemSplat(shuffle_op)) {
      // For SPLAT instruction with left padding
      splat_op = sentient::SplatOp::create(
          rewriter, op->getLoc(), result, dest_storage.getResult(),
          mask_const_op.getResult(), op_dbg_name, SentientSplatPad::left,
          symbolizeSentientPrecision(getInputPrecisionFromOperand(from))
              .value());
    } else {
      // For Register Init
      splat_op = sentient::SplatOp::create(
          rewriter, op->getLoc(), result, dest_storage.getResult(),
          mask_const_op.getResult(), op_dbg_name, SentientSplatPad::none,
          symbolizeSentientPrecision(getInputPrecisionFromOperand(from))

}}}
// ---- 280/384  cleanup  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1056  (7L)
    OpTy>::cleanup(Operation *op, OpInfo &op_info,
                   mlir::ConversionPatternRewriter &rewriter) {
  // Keep track of operations deleted (through rewriter) to avoid
  // double-deletions.
  std::vector<mlir::Operation *> erased_list;
  VectorOperand::eraseOperands(op_info.to_operands_, rewriter, erased_list);
  VectorOperand::eraseOp(op, rewriter, erased_list);

}
// ---- 281/384  OperationTreeBase  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/Analysis/LoopMaskTree.hpp:105  (2L)
  LoopMaskTree(dataflow::ProgramUnitOp &unit) : OperationTreeBase() {
    computeLoops(unit);

}
// ---- 282/384  updateLoopMaskTreeForConstantMask  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:20  (4L)
void e282_updateLoopMaskTreeForConstantMask(
    LoopMaskTree *pt_masking_tree, Operation *op, sentient::MacOp *mac_op,
    int mask_val) {
  auto parent_loop = op->getParentOfType<sentient::ForOp>();

}
// ---- 283/384  updateLoopMaskTreeForDynamicMask  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringPTMasks.cpp:28  (9L)
void e283_updateLoopMaskTreeForDynamicMask(
    LoopMaskTree *pt_masking_tree, Value val, sentient::MacOp *mac_op,
    int start_val, int increment) {
  DT_CHECK_MSG(isa<BlockArgument>(val), "expecting a block argument");
  auto parent_op = val.getParentRegion()->getParentOp();
  DT_CHECK(!parent_op ||
           isa<sentient::ForOp>(parent_op) &&
               "expecting either no parent op or a sentient::ForOp");
  DT_CHECK_MSG(cast<sentient::ForOp>(parent_op).getInductionVar() == val,

}
// ---- 284/384  hoistCommonConditionals  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:94  (96L)
Operation *CFGSDataflowConditionalTree::hoistCommonConditionals(
    Operation *if_op_where_last_hoist_occurred) {
  SmallVector<Operation *, 4> ops_to_delete;
  // Search the Conditional Tree for this pattern:
  //        if-node n
  //      /          |
  //    then        else
  //   |  |          |   |
  //  ...child_a...  ... child_b ...
  // where child_a and child_b correspond to equivalent, side-effect-free
  // if-ops.
  //
  // One hoist could create an opportunity for a new hoist out of the parent's
  // parent so hoistCommon will be executed in a kReverseBFS walk to facilitate
  // this. However, this means any one hoist will clobber the tree above it so
  // the tree will need to be recomputed. Thus, once hoisting has occurred, we
  // finish analyzing the current node then avoid hoisting for other nodes in
  // this iteration of the analysis.
  // If one hoist occurs, we will return the current node n's if-op so we may
  // resume hoisting from there in the next call to hoistCommonConditionals. If
  // no hoist occurs, we will return nullptr to indicate there are no more hoist
  // opportunities.
  Operation *cur_parent_if_op_of_hoist = nullptr;

  // We start analysis once if_op_where_last_hoist_occurred is reached in the
  // traversal. If this is nullptr, start immediately.
  bool start_analysis = (if_op_where_last_hoist_occurred == nullptr);
  auto hoistCommon = [&](CondNode *n) -> CondNode * {
    if (n == getRoot() || n->isLeaf() || n->isThenNode() || n->isElseNode() ||
        cur_parent_if_op_of_hoist)
      return nullptr;
    CondNode *then_node = n->getThenNode();
    CondNode *else_node = n->getElseNode();
    if (then_node->isLeaf() || else_node->isLeaf()) return nullptr;
    Operation *n_if_op = n->getOperation();
    DT_CHECK_MSG(n_if_op, "Expect valid if-op.");
    if (n_if_op == if_op_where_last_hoist_occurred) {
      start_analysis = true;
    }
    if (!start_analysis) return nullptr;

    // Search all pairs of children of (then_node, else_node) for if-ops which
    // are equivalent.
    for (CondNode *child_a = then_node->getFirstChild(); child_a;
         child_a = child_a->getNextSibling()) {
      // Skip nodes in the else branch corresponding to conditionals which have
      // been hoisted.
      SmallPtrSet<CondNode *, 8> nodes_to_skip;
      for (CondNode *child_b = else_node->getFirstChild(); child_b;
           child_b = child_b->getNextSibling()) {
        if (nodes_to_skip.contains(child_b)) continue;
        Operation *op_a = child_a->getOperation();
        Operation *op_b = child_b->getOperation();

        // Legality checks:
        // - op_a and op_b must be equivalent and have no side effects
        // - The block of op_a and op_b must be n's if-op's then/else block
        // (as opposed to a for-loop's block).
        // - The operations preceding op_a/op_b in their block must not
        // have side effects.
        // NOTE: It is likely the case that op_a/op_b depend on SSA values
        // preceding them in their block. We will hoist these along with the
        // common op; this hoist is valid because we will have verified that the
        // ops preceding op_a/op_b have no side effects.
        if (op_a && op_b && getOE().operationsAreEquivalent(*op_a, *op_b) &&
            isHoistable(op_a, &n->getThenRegion().front()) &&
            isHoistable(op_b, &n->getElseRegion().front())) {
          // Move op_a and its ancestors which do not dominate n's if-op
          // to right before n's if-op.
          moveAncestorsToMaintainDominance(op_a, n_if_op);

          // Replace uses of op_b by uses of op_a
          for (unsigned i = 0; i < op_b->getNumResults(); ++i)
            op_b->getResult(i).replaceAllUsesWith(op_a->getResult(i));

          // op_b will be deleted at the end.
          ops_to_delete.push_back(op_b);
          // Continue looking for common conditionals between the Then/Else
          // branches but skip these child_b. child_a will never be considered
          // again since child_a traverses the children of the then-node in
          // order.
          nodes_to_skip.insert(child_b);
          cur_parent_if_op_of_hoist = n_if_op;
          break;
        }
      }
    }
    return nullptr;
  };
  CondNode::walk<OperationNode::WalkOrder::kReverseBFS>(getRoot(), hoistCommon);
  // Delete all ops in ops_to_delete and their (non-constant) ancestors.
  for (auto *op : ops_to_delete) deleteAncestorsIfPossible(op);
  if (cur_parent_if_op_of_hoist) {
    recompute();
    getOE().clearCache();
  }

}
// ---- 285/384  replaceIfOpByIterArg  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:619  (56L)
void e285_replaceIfOpByIterArg() {
  auto for_op = std::get<0>(for_op_tuple_);
  int64_t num_iterations = std::get<4>(for_op_tuple_);
  if (num_iterations == 1) {
    // If the loop has one iteration only, then no need for a new iteration
    // argument. Simply replace if_op_ by seq_lb_.
    OpBuilder builder(if_op_);
    auto replacement = mlir::arith::ConstantIndexOp::create(
        builder, if_op_->getLoc(), seq_lb_);
    if_op_->getResult(0).replaceAllUsesWith(replacement);
    if_op_->erase();
  } else {
    // Need to mark if_op_ as we will be cloning its parent loop.
    OpBuilder builder(for_op);
    if_op_->setAttr(IF_OP_TO_BE_REPLACED_BY_ITER_ARG,
                    builder.getI32IntegerAttr(1));
    auto start_val = mlir::arith::ConstantIndexOp::create(
        builder, for_op->getLoc(), seq_lb_);
    auto step_val = mlir::arith::ConstantIndexOp::create(
        builder, for_op->getLoc(), seq_step_);
    // Clone the ForOp, adding 1 extra iterator argument.
    IRMapping ir_map;
    auto new_for_op =
        utils::createForOpWithAdditionalReturnValue(for_op, 1, ir_map);
    Value replacement;
    Operation *yield_op;
    unsigned num_iter_args;
    if (auto new_scf_for = llvm::dyn_cast<scf::ForOp>(new_for_op)) {
      num_iter_args = new_scf_for.getNumRegionIterArgs();
      replacement = new_scf_for.getRegionIterArgs()[num_iter_args - 1];
      yield_op = new_scf_for.getBody()->getTerminator();
    } else if (auto new_affine_for =
                   llvm::dyn_cast<affine::AffineForOp>(new_for_op)) {
      num_iter_args = new_affine_for.getNumRegionIterArgs();
      replacement = new_affine_for.getRegionIterArgs()[num_iter_args - 1];
      yield_op = new_affine_for.getBody()->getTerminator();
    } else {
      DT_ERROR("no matching for operation");
    }
    new_for_op->setOperand(num_iter_args - 1, start_val);
    builder.setInsertionPoint(yield_op);
    // Update the iteration argument.
    auto updated_iter_arg = mlir::arith::AddIOp::create(
        builder, yield_op->getLoc(), replacement, step_val);
    yield_op->setOperand(num_iter_args - 1, updated_iter_arg);

    // Replace the old IfOp with the new one.
    new_for_op->walk([&](scf::IfOp scf_if) {
      if (scf_if->hasAttr(IF_OP_TO_BE_REPLACED_BY_ITER_ARG)) {
        scf_if->getResult(0).replaceAllUsesWith(replacement);
        scf_if->erase();
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
  }

}
// ---- 286/384  runOnOperation  —  dcc/src/Transform/Dataflow/EnumerateCollectionUnit.cpp:94  (32L)
void e286_runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // Collect the total units operations
  SmallVector<dataflow::GetTotalUnitsInCollectionOp, 16> workList1;
  moduleOp.walk([&](dataflow::GetTotalUnitsInCollectionOp op) {
    workList1.push_back(op);
  });

  for (auto op : workList1) {
    auto type = mlir::dyn_cast<VectorType>(op.unit().getType());
    if (!type) return;

    int nUnits = type.getShape()[0];
    auto intType = IntegerType::get(op.getContext(), 32);
    OpBuilder builder(op);
    auto constTotalUnitsOp =
        ConstantIntOp::create(builder, op.getLoc(), nUnits, intType);
    op.replaceAllUsesWith(constTotalUnitsOp.getResult());
    op->erase();
  }

  // Collect the collection unit operations
  SmallVector<dataflow::ProgramCollectionOp, 16> workList2;
  moduleOp.walk([&](dataflow::ProgramCollectionOp collectionDefinitionOp) {
    workList2.push_back(collectionDefinitionOp);
  });

  for (auto collectionDefinitionOp : workList2) {
    enumerateCollectionUnit(moduleOp, collectionDefinitionOp);
    collectionDefinitionOp->erase();
  }

}
// ---- 287/384  flatten  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:384  (70L)
bool e287_flatten(Operation &op_) {
  clear();
  auto uniform_op = llvm::dyn_cast<uniform::UniformizeRegionsOp>(op_);
  if (!uniform_op) return false;
  std::vector<std::vector<mlir::Value>> units_of_old_local_regions;
  dcc::uniform::utils::getUnitsPerRegionsAsVectorOfVector(
      uniform_op, units_of_old_local_regions);

  auto new_node = new LocalOpNode(&op_);
  root_ = new_node;
  compute(new_node);

  llvm::MapVector<mlir::Value, std::vector<mlir::Value>> equivalence_classes;
  partitionUnits(equivalence_classes);

  // llvm::outs() << "--- Tree ---\n";
  // auto node = getRoot();
  // printTree(node, 2);
  // llvm::outs() << "------------\n";
  // printUnitToOpsMap(unit_to_ops);
  // printEquivalenceClasses(equivalence_classes);

  OpBuilder builder(&op_);

  std::vector<mlir::Value> new_unit_list;
  std::vector<Attribute> new_list_sizes;
  std::vector<mlir::Value> unit_rep_order;
  for (auto &eq_units : equivalence_classes) {
    unit_rep_order.push_back(eq_units.first);
    for (auto &u : eq_units.second) new_unit_list.push_back(u);
    new_list_sizes.push_back(builder.getI32IntegerAttr(eq_units.second.size()));
  }

  int num_of_regions = equivalence_classes.size();
  if (op_.getNumRegions() == num_of_regions) return false;

  auto new_uniform_op = mlir::uniform::UniformizeRegionsOp::create(
      builder, op_.getLoc(), mlir::TypeRange(), new_unit_list,
      ArrayAttr::get(builder.getContext(), new_list_sizes),
      /*regIndices*/ nullptr, /*regLocales*/ nullptr, num_of_regions);

  std::vector<std::vector<mlir::Value>> units_of_new_local_regions;
  dcc::uniform::utils::getUnitsPerRegionsAsVectorOfVector(
      new_uniform_op, units_of_new_local_regions);

  DT_CHECK(unit_rep_order.size() == num_of_regions);
  for (int i = 0; i < num_of_regions; i++) {
    auto &block = new_uniform_op.getRegion(i).emplaceBlock();
    auto block_arg =
        block.addArgument(builder.getIndexType(), new_uniform_op->getLoc());

    int old_region_idx = 0;
    for (; old_region_idx < units_of_old_local_regions.size();
         old_region_idx++) {
      if (std::find(units_of_old_local_regions.at(old_region_idx).begin(),
                    units_of_old_local_regions.at(old_region_idx).end(),
                    units_of_new_local_regions.at(i).at(0)) !=
          units_of_old_local_regions.at(old_region_idx).end())
        break;
    }
    IRMapping arg_map;
    arg_map.map(op_.getRegion(old_region_idx).getArgument(0), block_arg);
    OpBuilder builder_region(new_uniform_op.getRegion(i));
    cloneOpsForRegions(getRoot()->getFirstChild(),
                       unit_to_ops[unit_rep_order.at(i)], builder_region, 0,
                       arg_map, block_arg);
    auto yield_op = mlir::uniform::YieldOp::create(builder_region,
                                                   new_uniform_op->getLoc());
  }
  return true;

}
// ---- 288/384  runOnOperation  —  dcc/src/Transform/Dataflow/LoopUnrollingForPTLRFRegs.cpp:131  (22L)
void e288_runOnOperation() {
  ModuleOp module_op = getOperation();
  module_op.walk([&](dataflow::ProgramUnitOp unit) {
    DT_CHECK(unit.getUnits().size() >= 1);
    auto get_unit_op = unit.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>();
    auto record = EnumsConversion::stringToSenComponents.find(
        get_unit_op.getType().str());
    DT_CHECK_MSG(record != EnumsConversion::stringToSenComponents.end(),
                 "unexpected unit");
    SenComponents comp =
        EnumsConversion::senCompToGenericComp.at(record->second);

    for (const auto u : unit.getUnits()) {
      DT_CHECK(record ==
               EnumsConversion::senCompToGenericComp.find(
                   u.getDefiningOp<dataflow::GetUnitOp>().getType().str()));
    }

    if (is_any_of(comp, PT, PE, SFP)) {
      processComputeUnit(unit);
    }
  });

}
// ---- 289/384  initialize  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:694  (8L)
void e289_initialize(
    ExpressionEvaluator &evaluator, SmallVectorImpl<MASData> &mas_data,
    dataflow::GetLogicalMemoryViewOp mem_view_op, agen::AccessDetailsAffine &ad,
    int64_t &max_mutable) {
  int64_t num_elems_in_stick =
      dcc_ext_ctx_.getBytesPerStick() * 8 / ad.getElementWidth();
  DT_CHECK(dcc::agen::utils::hasValidL3ImmutableAddr(evaluator, mem_view_op,
                                                     num_elems_in_stick));

}
// ---- 290/384  createPartitions  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:983  (10L)
void e290_createPartitions(
    const SmallVectorImpl<MASData> &mas_data,
    const SmallVectorImpl<int64_t> &partition_sizes, Operation *op,
    const AffineMap &subscripts_map,
    const std::function<void(OpBuilder &, AffineMap &, int64_t start_addr_mod)>
        &op_creator) {
  DT_CHECK(!partition_sizes.empty());
  auto root_op = constructConditionals(mas_data, partition_sizes, op);
  DT_CHECK_MSG(root_op, "Root op of conditional tree could not be determined");


}
// ---- 291/384  calculatePartialShift  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:490  (66L)
int64_t e291_calculatePartialShift(
    SmallVectorImpl<int64_t> &shifts, agen::AccessDetailsAffine &ad,
    const int64_t immutable_space) {
  DT_CHECK(shifts.empty());

  // If all the mutable start address cannot be shifted, shift as much of it out
  // as possible.
  //
  // 1. Collect the constants for each dim along with their layout coefficient.
  //    Create a weight from them, order them with relation to that weight.
  //    The weight would be <layout coefficient> / <constant>.
  // 2. Starting with the highest weighted constant, calculate how much of it
  //    can be shifted out. This step considers EBR space.
  // 3. Repeat with all weights.
  auto layout_coeffs = ad.getLayoutCoeffs();
  auto subscripts_map = ad.getSubscriptsMap();

  // Turn the transfer order from a vector of indices into the subscripts
  // results. The subscript results need to be ordered to calculate the
  // coefficients and the weights but the new subscripts map that will be
  // created is created in the original order.
  SmallVector<int, 8> dim_order;
  auto transfer_order = ad.getTransferOrder();
  for (auto &expr : transfer_order.getResults()) {
    auto dim_expr = dyn_cast<AffineDimExpr>(expr);
    DT_CHECK(dim_expr);
    dim_order.push_back(dim_expr.getPosition());
  }

  SmallVector<DimWeight, 8> dim_weights;
  calculateDimWeights(dim_weights, ad);

  // Initialize all shifts to zero as the shifts may not be analyzed in order.
  for (int i = 0, e = subscripts_map.getNumResults(); i < e; ++i)
    shifts.push_back(0);

  int64_t curr_immutable_space = immutable_space;
  for (auto &dim_weight : dim_weights) {
    // Calculate how much of the constant offset can be shifted out. This is
    // equivalent to how many times the layout coefficient for that dim fits in
    // the remaining immutable space. If there isn't enough space to fit even
    // 1*<layout coeff>, move the the next dim. Even if there is not enough
    // space for the current dim, there may be space for another smaller dim.
    DT_CHECK(layout_coeffs.size() > dim_weight.dim_);
    int64_t constant_shift =
        curr_immutable_space / layout_coeffs[dim_weight.dim_];
    if (constant_shift <= 0) continue;

    // If the dim will fit into the remaining space in any form, update the
    // expression, calculate how much the immutable changes, and update the
    // remaining immutable space.
    if (constant_shift > dim_weight.offset_)
      constant_shift = dim_weight.offset_;

    curr_immutable_space -= constant_shift * layout_coeffs[dim_weight.dim_];
    DT_CHECK(shifts.size() > dim_weight.dim_);
    shifts[dim_weight.dim_] = constant_shift;
  }
  // If the amount of elements shifted to the immutable is not in terms of
  // sticks, need to shift a bit back to the mutable start address.
  int64_t total_shift = immutable_space - curr_immutable_space;
  DT_CHECK(total_shift >= 0);
  int64_t num_elems_in_stick =
      dcc_ext_ctx_.getBytesPerStick() * 8 / ad.getElementWidth();
  int64_t remainder = total_shift % num_elems_in_stick;
  total_shift -= remainder;

}
// ---- 292/384  transformSCFLoopWithNonConstantUpperBound  —  dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:169  (94L)
    transformSCFLoopWithNonConstantUpperBound(scf::ForOp scf_for) {
  auto lb_const =
      scf_for.getLowerBound().getDefiningOp<arith::ConstantIndexOp>();
  auto ub_const =
      scf_for.getUpperBound().getDefiningOp<arith::ConstantIndexOp>();
  auto step_const = scf_for.getStep().getDefiningOp<arith::ConstantIndexOp>();

  // we currently support lb being 0, step being 1, non-constant ub.
  if (lb_const.value() == 0 && step_const.value() == 1 && !ub_const) {
    // upper bound needs to be coming from select operation.
    auto cmpi_op = scf_for.getUpperBound().getDefiningOp<arith::SelectOp>();
    if (!cmpi_op) {
      return LogicalResult::failure();
    }

    // Both operands of cmpi has to be constant values.
    if (isa<arith::ConstantIndexOp>(cmpi_op.getTrueValue().getDefiningOp()) &&
        isa<arith::ConstantIndexOp>(cmpi_op.getFalseValue().getDefiningOp())) {
      OpBuilder builder(cmpi_op);

      // Construct if-operation with number of results being same of scf_for op
      auto if_op = scf::IfOp::create(builder, cmpi_op.getLoc(),
                                     scf_for->getResultTypes(),
                                     cmpi_op.getCondition(), true);
      if (auto dbg_name_attr = getDbgNameAttr(cmpi_op))
        setDbgNameAttr(if_op, dbg_name_attr);

      // replace uses of scf_for results with if_op results
      scf_for->replaceAllUsesWith(if_op.getResults());

      // work on then branch
      //      BlockAndValueMapping then_map;
      //      then_map.map(scf_for.upperBound(), cmpi_op.true_value());
      //      if_op.getThenBodyBuilder().clone(*scf_for, then_map);
      affine::AffineForOp then_forop;
      int then_ub = cmpi_op.getTrueValue()
                        .getDefiningOp<arith::ConstantIndexOp>()
                        .value();
      if (failed(transformSCFToAffineLoop(if_op.getThenBodyBuilder(), scf_for,
                                          lb_const.value(), then_ub,
                                          step_const.value(), then_forop))) {
        if_op->emitError("Unable to transform SCF loop into Affine loop");
        return LogicalResult::failure();
      }

      // Create the scf::YieldOp for the then region with the affine loop
      // results (only if the affine loop has results).
      if (then_forop.getNumResults() > 0) {
        OpBuilder then_builder = if_op.getThenBodyBuilder();
        then_builder.create<scf::YieldOp>(if_op.getLoc(),
                                          then_forop.getResults());
      }

      // work on else branch
      affine::AffineForOp else_forop;
      int else_ub = cmpi_op.getFalseValue()
                        .getDefiningOp<arith::ConstantIndexOp>()
                        .value();
      if (failed(transformSCFToAffineLoop(if_op.getElseBodyBuilder(), scf_for,
                                          lb_const.value(), else_ub,
                                          step_const.value(), else_forop))) {
        if_op->emitError("Unable to transform SCF loop into Affine loop");
        return LogicalResult::failure();
      }

      // Create the scf::YieldOp for the else region with the affine loop
      // results (only if the affine loop has results).
      if (else_forop.getNumResults() > 0) {
        OpBuilder else_builder = if_op.getElseBodyBuilder();
        else_builder.create<scf::YieldOp>(if_op.getLoc(),
                                          else_forop.getResults());
      }

      // analyze and process the loop in then branch
      auto then_loop = llvm::dyn_cast<mlir::LoopLikeOpInterface>(
          if_op.getThenRegion().front().front());
      analyzeAndTransform(then_loop);

      // analyze and process the loop in then branch
      auto else_loop = llvm::dyn_cast<mlir::LoopLikeOpInterface>(
          if_op.getElseRegion().front().front());
      analyzeAndTransform(else_loop);

      // Erase for-op and cmpi_op.
      scf_for->erase();
      if (cmpi_op->use_empty()) {
        cmpi_op.erase();
      }

      return LogicalResult::success();
    }
  }

  return LogicalResult::failure();

}
// ---- 293/384  runOn  —  dcc/src/Transform/Dataflow/TransformLoopToLegalizeForSentientLowering.cpp:447  (5L)
void e293_runOn(
    dataflow::ProgramUnitOp unit) {
  curr_unit_ =
      dcc::getUnitType(unit.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
  unit.walk<WalkOrder::PostOrder>(

}
// ---- 294/384  getPageValidity  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:114  (33L)
FlatLinearValueConstraints e294_getPageValidity(
    TPMVInfo &info, AffineMap &subscripts_map_sym, IntegerSet page_set) {
  FlatLinearValueConstraints page_set_constraints(page_set);
  DT_CHECK_MSG(page_set_constraints.isHyperRectangular(
                   0, page_set_constraints.getNumCols() - 1),
               "idx_set should be hyper rectangular");

  // Add the subscripts constraints to the page idx_set constraints.
  IntegerSet subscripts_in_page = page_set;
  subscripts_in_page = subscripts_in_page.replaceDimsAndSymbols(
      subscripts_map_sym.getResults(), {}, 0,
      subscripts_map_sym.getNumSymbols());
  FlatLinearValueConstraints page_sel_constraints(subscripts_in_page);

  if (page_sel_constraints.isEmpty()) {
    LLVM_DEBUG(llvm::dbgs() << "[TransformPagedMemViewPass] Invalid page - "
                               "subscripts and page do not intersect\n";
               page_sel_constraints.print(llvm::dbgs()); llvm::dbgs() << "\n");
    return page_sel_constraints;
  }

  addConstraintsForIVRanges(page_sel_constraints, info.indices_ranges_);

  if (page_sel_constraints.isEmpty()) {
    LLVM_DEBUG(llvm::dbgs()
                   << "[TransformPagedMemViewPass] Invalid page - IV "
                      "subscripts have bounds that do not intersect the page\n";
               page_sel_constraints.print(llvm::dbgs()); llvm::dbgs() << "\n");
    return page_sel_constraints;
  }

  LLVM_DEBUG(llvm::dbgs() << "[TransformPagedMemViewPass] Valid page\n";
             page_sel_constraints.print(llvm::dbgs()); llvm::dbgs() << "\n");

}
// ---- 295/384  createIterArgsForConditionals  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:401  (121L)
void e295_createIterArgsForConditionals(TPMVInfo &info,
                                             AffineMap &subscripts_map,
                                             SmallVectorImpl<Value> &indices) {
  DT_CHECK(info.conditional_iter_args_.empty());

  // Subscripts need to be converted into iter_args for use in a conditional
  // tree:
  //   - Each map result using a loop iterator will need an iter_arg.
  //   - Each loop contributing will yield an add of that arg + coeff.
  //   - The iter_arg will be initialized to the constant part of the result.
  //
  // As iter_args are created, store the innermost iter_args in order. Constant
  // expression results will be represented by nullptr. This list will be used
  // to form the conditionals.
  //
  // TODO: See if we want to do all the calculations twice to limit the number
  // of iter_args created. If we know which results contribute to page
  // selection, we can only create iter_args for those results instead of all
  // results involving loop iterators. But this would require full analysis
  // of all pages first.
  SmallVector<int> ordered_indices_idxs = setLoopIteratorOrder(indices);
  SmallVector<SmallVector<int64_t>> ordered_coeffs;
  for (auto &res : subscripts_map.getResults()) {
    if (isa<AffineConstantExpr>(res)) continue;

    // Collect the coefficients for this result.
    SmallVector<int64_t> coeffs;
    affine::FlatAffineValueConstraints constraints;
    auto flat_res = getFlattenedAffineExpr(res, subscripts_map.getNumDims(),
                                           subscripts_map.getNumSymbols(),
                                           &coeffs, &constraints);

    // Order the coefficients from outermost loop to innermost loop.
    SmallVector<int64_t> ordered_res_coeffs(coeffs.size());
    for (int i = 0, e = coeffs.size() - 1; i < e; ++i)
      ordered_res_coeffs[ordered_indices_idxs[i]] = coeffs[i];
    ordered_res_coeffs.push_back(coeffs.back());  // constant coeff
    ordered_coeffs.push_back(ordered_res_coeffs);
  }

  // Each loop will get one iter_arg per map result involving loop iterators.
  for (int i = 0, num_indices = indices.size(); i < num_indices; ++i) {
    // Loops are updated from outermost to innermost.
    int indices_idx = ordered_indices_idxs[i];

    // Create the new loop with an iter_arg representing each loop iterator
    // involved subscript.
    IRMapping ir_map;
    auto curr_loop =
        cast<BlockArgument>(indices[indices_idx]).getOwner()->getParentOp();
    auto for_op = cast<affine::AffineForOp>(
        dcc::utils::createForOpWithAdditionalReturnValue(
            curr_loop, ordered_coeffs.size(), ir_map, /* delete_op */ false));
    // Update the loop args and yields.
    auto region_iter_args = for_op.getRegionIterArgs();
    for (int c = 0, num_args = ordered_coeffs.size(); c < num_args; ++c) {
      // Update iter_arg initializations.
      OpBuilder builder(for_op);
      if (info.conditional_iter_args_.size() < num_args) {
        // If this is the outermost loop, initialize iter_args with the constant
        // coefficients.
        auto coeff_const = mlir::arith::ConstantIndexOp::create(
            builder, for_op.getLoc(), ordered_coeffs[c].back());
        for_op.setOperand(for_op.getNumOperands() - num_args + c,
                          coeff_const.getResult());
        info.conditional_iter_args_.push_back(
            region_iter_args[region_iter_args.size() - num_args + c]);
      } else {
        // If this is an inner loop, initialize to the previous iter_args
        // created.
        for_op.setOperand(for_op.getNumOperands() - num_args + c,
                          info.conditional_iter_args_[c]);
        info.conditional_iter_args_[c] =
            region_iter_args[region_iter_args.size() - num_args + c];
      }

      // For each subscript involving the iterator of the current loop, create
      // an addOp of the new loop iterator and its coefficient.
      auto terminator = for_op.getBody()->getTerminator();
      builder.setInsertionPoint(terminator);
      auto coeff_val = ordered_coeffs[c][i];
      if (coeff_val != 0) {
        auto coeff_const = mlir::arith::ConstantIndexOp::create(
            builder, for_op.getLoc(), ordered_coeffs[c][i]);
        auto add_op = mlir::arith::AddIOp::create(
            builder, terminator->getLoc(), info.conditional_iter_args_[c],
            coeff_const.getResult());
        terminator->setOperand(terminator->getNumOperands() - num_args + c,
                               add_op);
      }
    }

    // Update the indices to keep them in sync as we clone the loops.
    indices[indices_idx] = for_op.getInductionVar();
    for (int j = i + 1; j < num_indices; ++j) {
      auto new_index = ir_map.lookupOrNull(indices[ordered_indices_idxs[j]]);
      DT_CHECK(new_index);
      indices[ordered_indices_idxs[j]] =
          cast<affine::AffineForOp>(
              cast<BlockArgument>(new_index).getOwner()->getParentOp())
              .getInductionVar();
    }

    // Update mem_ops_ since they may have changed.
    for (int m = 0, num_ops = mem_ops_.size(); m < num_ops; ++m) {
      auto new_mem_op = ir_map.lookupOrNull(mem_ops_[m]);
      DT_CHECK(new_mem_op);
      mem_ops_[m] = new_mem_op;
    }
    // Update the paged_mem_view. The mem_view may not be in the innermost loop.
    auto new_mem_view = ir_map.lookupOrDefault(info.paged_mem_view_);
    info.paged_mem_view_ = cast<dataflow::GetPagedLogicalMemoryViewOp>(
        new_mem_view.getDefiningOp());

    // The other TPMVInfo objects need to be synced as the values they use for
    // paged_mem_view or indices may have changed as a result of the clones.
    MemoryOperandIndex mem_index = info.mem_index_;
    for (auto &ti : tpmv_info_) {
      if (ti.mem_index_ != mem_index) updateTPMVInfo(ti, ir_map);
    }


}}
// ---- 296/384  runOnOperation  —  dcc/src/Transform/Dataflow/UnitFiltering.cpp:362  (127L)
void e296_runOnOperation() {
  if (DisableThisPass) return;

  if (opts_.isIncludeList()) {
    DT_CHECK_MSG(FilterComponentsExcept.empty(),
                 "Should not use both filter-components command line option "
                 "and include list");
    filter_components_except_ = opts_.getInclExclList();
  } else {
    for (std::string comp : FilterComponentsExcept)
      filter_components_except_.insert(
          EnumsConversion::stringToSenComponents.find(comp)->second);
  }
  if (FilterFoldsExcept.empty()) {
    filter_folds_except_ = opts_.getFilterFoldsExcept();
  } else {
    DT_CHECK_MSG(opts_.getFilterFoldsExcept().empty(),
                 "Should not use both filter-folds command line options "
                 "(pass-specific and pipeline ones)");
    for (int fold_id : FilterFoldsExcept) filter_folds_except_.insert(fold_id);
  }
  DT_CHECK_MSG(
      filter_folds_except_.empty() || filter_folds_except_.count(0) > 0,
      "Should always keep fold #0 if filtering folds.");

  if (FilterCoresExcept.empty()) {
    filter_cores_except_ = opts_.getFilterCoresExcept();
  } else {
    DT_CHECK_MSG(opts_.getFilterCoresExcept().empty(),
                 "Should not use both filter-cores command line options "
                 "(pass-specific and pipeline ones)");
    for (int core_id : FilterCoresExcept) filter_cores_except_.insert(core_id);
  }

  if (FilterCoreletsExcept.empty()) {
    filter_corelets_except_ = opts_.getFilterCoreletsExcept();
  } else {
    DT_CHECK_MSG(opts_.getFilterCoreletsExcept().empty(),
                 "Should not use both filter-corelets command line options "
                 "(pass-specific and pipeline ones)");
    for (int corelet_id : FilterCoreletsExcept)
      filter_corelets_except_.insert(corelet_id);
  }

  if (FilterTransfersExcept.empty()) {
    filter_transfers_except_ = opts_.getFilterTransfersExcept();
  } else {
    DT_CHECK_MSG(opts_.getFilterTransfersExcept().empty(),
                 "Should not use both filter-transfers command line options "
                 "(pass-specific and pipeline ones)");
    for (std::string dbg_name : FilterTransfersExcept)
      filter_transfers_except_.insert(dbg_name);
  }

  bool filter_transfers = !filter_transfers_except_.empty();
  bool filter_folds = !filter_folds_except_.empty();
  bool filter_cores = !filter_cores_except_.empty();
  bool filter_corelets = !filter_corelets_except_.empty();

  // 1. Filter out program units, remove folds from program unit arguments
  ModuleOp module_op = getOperation();
  module_op.walk([&](dataflow::ProgramUnitOp unit_op) {
    bool remove_unit = !filter_components_except_.empty();
    // Filter out by component
    auto unit_type_str = unit_op.getUnits()
                             .front()
                             .getDefiningOp<dataflow::GetUnitOp>()
                             .getType()
                             .str();
    for (auto comp : filter_components_except_) {
      std::string comp_string = EnumsConversion::senComponentsToString.at(comp);
      if (comp_string == unit_type_str) {
        remove_unit = false;
        break;
      }
    }
    if (remove_unit) {
      to_delete_.push_back(unit_op);
    } else {
      // Filter out coreIDs/coreletIDs/foldIDs from list of units
      if (filter_cores || filter_corelets || filter_folds)
        removeCoresCoreletsFoldsFromProgramUnit(unit_op);
    }
    return WalkResult::skip();
  });

  for (Operation *op : to_delete_) op->erase();
  to_delete_.clear();
  if (!filter_folds && !filter_cores && !filter_corelets && !filter_transfers)
    return;

  // 2. Filter out folds/cores/corelets in DefImmutableMaps. Collect all data
  // transfers to be filtered out (if any).
  llvm::SmallVector<Operation *> transfers_to_filter_out;
  module_op.walk([&](Operation *op) {
    if (filter_cores || filter_corelets || filter_folds) {
      if (auto def_immutable_map = dyn_cast<uniform::DefImmutableMappingOp>(op))
        removeCoresCoreletsFoldsFromDefImmutMap(def_immutable_map);
    }
    if (filter_transfers && isDataTransfer(op) && !isDataTransferToKeep(op)) {
      transfers_to_filter_out.push_back(op);
    }
  });

  for (Operation *op : to_delete_) op->erase();
  to_delete_.clear();

  for (Operation *op : transfers_to_filter_out) {
    removeAncestors(op);
  }

  if (!filter_folds && !filter_cores && !filter_corelets) return;

  // 3. Filter out folds/cores/corelets in UniformizeRegions
  // NOTE: Must be done in a separate walk from 2 as we are cloning
  // UniformizeRegionsOps and do not want stale immutable ops
  // to persist in the cloned regions.
  module_op.walk([&](uniform::UniformizeRegionsOp uniformize_op) {
    removeCoresCoreletsFoldsFromUniformizeRegion(uniformize_op);
  });

  for (Operation *op : to_delete_) op->erase();
  to_delete_.clear();

  // 4. Remove dead GetUnitOps, CreateGroupOps, and CreateMulticastGroupOps.
  // Remove dead values from dataflow.get_unit's and update the fold number.
  cleanup(module_op);


}
// ==================================================================================================
// LEVEL 4
// ==================================================================================================

// ---- 297/384  constructTimeStepsInfo  —  dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:626  (42L)
LogicalResult e297_constructTimeStepsInfo(
    AccessContainer<AccessDetailsAffineComposite>& access_details,
    bool do_coalesce, bool do_burst_il_group_calc) {
  DT_CHECK(access_details.size() > 0 && access_details.size() <= 4);
  Operation* op = access_details.getFirst().getOp();

  DT_CHECK(
      (bool)(isa<CompositeLoadOp, CompositeIndirectLoadOp, CompositeStoreOp,
                 CompositeIndirectStoreOp, CompositeLoadAndStoreOp,
                 CompositeIndirectLoadAndStoreOp>(op)));

  for (auto& access : access_details) {
    // Construct time bounds
    SmallVector<int64_t> time_bounds;
    auto time_symbols = access.getTimeSymbols();
    auto time_set = access.getTimeSet();
    auto time_order = access.getTimeOrder();
    if (agen::utils::calculateTimeBounds(time_bounds, time_set, time_symbols,
                                         time_order)
            .failed())
      return failure();
    access.setTimeBounds(time_bounds);

    // construct time offsets
    // get address offsets for time dimensions
    auto time_offsets = access.getTimeOffsets();
    auto mem_view_layout_map = access.getMemViewLayoutMap();
    auto time_addr_map = access.getTimeAddrMap();
    if (agen::utils::calculateTimeOffsets(time_offsets, mem_view_layout_map,
                                          time_addr_map, time_order)
            .failed())
      return failure();
    access.setTimeOffsets(time_offsets);
  }

  // Coalesce time dimensions.
  // Note this step needs to be done considering all access_details for a given
  // transfer and it must happen before burst and il group calculation.
  if (do_coalesce) coalesceTimeDimensions(access_details);

  if (do_burst_il_group_calc)
    for (auto& access : access_details) access.computeBurstAndGroup();

}
// ---- 298/384  constructAffineDetailsAndAddrs  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2787  (16L)
LogicalResult e298_constructAffineDetailsAndAddrs(
    Operation* src_op, Operation* dst_op, ProgramUnitOp& unit,
    SenComponents comp, AccessContainer<AccessDetailsAffine>& access_details,
    AccessContainer<Value>& mutable_addrs,
    AccessContainer<Value>& immutable_addrs) {
  DT_CHECK(src_op);
  AccessDetailsAffine& src_ad =
      access_details.emplace_insert(MemoryOperandIndex::kDirSrc, src_op, comp);
  if (src_ad.constructDetails(MemoryOperandIndex::kDirSrc).failed())
    return src_op->emitError("unable to construct details for src");

  if (dst_op) {
    AccessDetailsAffine& dst_ad = access_details.emplace_insert(
        MemoryOperandIndex::kDirDst, dst_op, comp);
    if (dst_ad.constructDetails(MemoryOperandIndex::kDirDst).failed())
      return dst_op->emitError("unable to construct details for dst");

}}
// ---- 299/384  lowerAffineCompositeHelper  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2953  (15L)
LogicalResult e299_lowerAffineCompositeHelper(
    OpTy op, ProgramUnitOp& unit,
    AccessContainer<AccessDetailsAffineComposite>& access_details,
    AccessContainer<Value>& mutable_addrs,
    AccessContainer<Value>& immutable_addrs,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  auto result = constructTimeLoopsAndVectorOperations(
      unit, op, mutable_addrs, immutable_addrs, access_details);

  // The op may have changed due to loop cloning.
  auto candidate_op = findCandidateForLowering<OpTy>(unit);

  if (result.failed())
    return candidate_op->emitError(
        "Unable to generate loops and sentient statements for the composite "

}
// ---- 300/384  lowerSyncForAUnit  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:733  (7L)
LogicalResult e300_lowerSyncForAUnit(
    std::string src_unit_name, mlir::Operation *op,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops_corelet0,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops_corelet1,
    dataflow::GetUnitOp dst_unit) {
  OpBuilder builder(op);

}
// ---- 301/384  lowerSyncForAGroup  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:746  (8L)
LogicalResult e301_lowerSyncForAGroup(
    std::string src_unit_name, mlir::Operation *op,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops_corelet0,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops_corelet1,
    dataflow::CreateGroupOp dst_unit_group) {
  OpBuilder builder(op);
  if (src_unit_name.substr(0, 2) != "l3")

}
// ---- 302/384  lowerSyncLXL3ToLXL3  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:787  (928L)
LogicalResult e302_lowerSyncLXL3ToLXL3(
    mlir::Operation *op, mlir::OpBuilder builder,
    std::vector<mlir::Value> src_vs, std::vector<mlir::Value> dst_vs,
    int corelet_id, bool is_src_l3) {
  std::vector<std::pair<mlir::Value, mlir::Value>> src_dst_lx_corelet0,
      src_dst_lx_corelet1, src_dst_l3, src_dst_group;
  separateBasedOnDestinationUnits(builder, src_vs, dst_vs, src_dst_lx_corelet0,
                                  src_dst_lx_corelet1, src_dst_l3,
                                  src_dst_group);
  DT_CHECK(src_dst_lx_corelet0.size() != 0 || src_dst_lx_corelet1.size() != 0 ||
           src_dst_l3.size() != 0 || src_dst_group.size() != 0);
  if ((src_dst_lx_corelet1.size() == 0 && src_dst_l3.size() == 0 &&
       src_dst_group.size() == 0) ||
      (src_dst_lx_corelet0.size() == 0 && src_dst_l3.size() == 0 &&
       src_dst_group.size() == 0) ||
      (src_dst_lx_corelet0.size() == 0 && src_dst_lx_corelet1.size() == 0 &&
       src_dst_group.size() == 0)) {
    // only src_dst_lx_corelet0 is none empty or
    // only src_dst_lx_corelet1 is none empty or
    // only src_dst_l3 is none empty
    auto src_op =
        llvm::dyn_cast<dataflow::GetUnitOp>(src_vs[0].getDefiningOp());
    auto dst_op =
        llvm::dyn_cast<dataflow::GetUnitOp>(dst_vs[0].getDefiningOp());
    DT_CHECK(src_op && dst_op);
    if (is_src_l3) {
      if (lowerL3SyncOperationForAUnit(op, builder, {src_op}, dst_op).failed())
        return LogicalResult::failure();
    } else if (corelet_id == 0) {
      if (lowerL0LXSyncOperationForAUnit(op, builder, {src_op}, {}, dst_op)
              .failed())
        return LogicalResult::failure();
    } else {
      if (lowerL0LXSyncOperationForAUnit(op, builder, {}, {src_op}, dst_op)
              .failed())
        return LogicalResult::failure();
    }
    return LogicalResult::success();
  } else if (src_dst_lx_corelet0.size() == 0 &&
             src_dst_lx_corelet1.size() == 0 && src_dst_l3.size() == 0) {
    // only src_dst_group is none empty
    bool sameGroup = true;
    for (auto u : dst_vs) {
      if (u.getDefiningOp() == dst_vs[0].getDefiningOp()) sameGroup = false;
    }

    // Folding:
    // In case of folds, each value in the map refer to a create_group or
    // get_unit operation corresponding to a fold instance. It is not necessary
    // to create a uniformize_regions op to distinguish between them.
    // we could simply use only one instance of the fold.
    // Furthermore, it could introduce nested uniform regions which would
    // require flattening or simplify uniform regions pass to clean it up and
    // that could be compilation burden.
    bool all_keys_folds = true;
    for (auto u : src_vs) {
      if (u.getDefiningOp() != src_vs[0].getDefiningOp())
        all_keys_folds = false;
    }

    if (all_keys_folds) {
      sameGroup = true;
    }

    if (sameGroup) {
      auto src_op =
          llvm::dyn_cast<dataflow::GetUnitOp>(src_vs[0].getDefiningOp());
      auto group_op =
          llvm::dyn_cast<dataflow::CreateGroupOp>(dst_vs[0].getDefiningOp());
      if (is_src_l3) {
        if (lowerL3SyncOperationForAGroupOfUnits(op, builder, {src_op},
                                                 group_op)
                .failed())
          return LogicalResult::failure();
      } else if (corelet_id == 0) {
        if (lowerL0LXSyncOperationForAGroupOfUnits(op, builder, {src_op}, {},
                                                   group_op)
                .failed())
          return LogicalResult::failure();
      } else {
        if (lowerL0LXSyncOperationForAGroupOfUnits(op, builder, {}, {src_op},
                                                   group_op)
                .failed())
          return LogicalResult::failure();
      }
      return LogicalResult::success();
    } else {
      llvm::SmallVector<Attribute, 2> list_sizes;
      for (int i = 0; i < dst_vs.size(); i++) {
        list_sizes.push_back(builder.getI32IntegerAttr(1));
      }
      int num_of_regions = dst_vs.size();
      auto uniform_region = mlir::uniform::UniformizeRegionsOp::create(
          builder, op->getLoc(), mlir::TypeRange(), src_vs,
          ArrayAttr::get(builder.getContext(), list_sizes),
          /*regIndices*/ nullptr, /*regLocales*/ nullptr, num_of_regions);
      for (int i = 0; i < num_of_regions; i++) {
        auto &block = uniform_region.getRegion(i).emplaceBlock();
        block.addArgument(builder.getIndexType(), op->getLoc());
        OpBuilder builder_region(uniform_region.getRegion(i));
        auto src_op =
            llvm::dyn_cast<dataflow::GetUnitOp>(src_vs[i].getDefiningOp());
        auto group_op =
            llvm::dyn_cast<dataflow::CreateGroupOp>(dst_vs[i].getDefiningOp());
        if (is_src_l3) {
          if (lowerL3SyncOperationForAGroupOfUnits(op, builder_region, {src_op},
                                                   group_op)
                  .failed())
            return LogicalResult::failure();
        } else if (corelet_id == 0) {
          if (lowerL0LXSyncOperationForAGroupOfUnits(op, builder_region,
                                                     {src_op}, {}, group_op)
                  .failed())
            return LogicalResult::failure();
        } else {
          if (lowerL0LXSyncOperationForAGroupOfUnits(op, builder_region, {},
                                                     {src_op}, group_op)
                  .failed())
            return LogicalResult::failure();
        }
        uniform::YieldOp::create(builder_region, op->getLoc());
      }
    }
    return LogicalResult::success();
  } else if (src_dst_l3.size() == 0 && src_dst_group.size() == 0) {
    // only src_dst_lx_corelet0 and src_dst_lx_corelet1 are none empty
    std::vector<mlir::Value> sorted_units;
    for (auto src_dst : src_dst_lx_corelet0)
      sorted_units.push_back(src_dst.first);
    for (auto src_dst : src_dst_lx_corelet1)
      sorted_units.push_back(src_dst.first);
    llvm::SmallVector<Attribute, 2> list_sizes = {
        builder.getI32IntegerAttr(src_dst_lx_corelet0.size()),
        builder.getI32IntegerAttr(src_dst_lx_corelet1.size())};
    int num_of_regions = 2;
    auto uniform_region = mlir::uniform::UniformizeRegionsOp::create(
        builder, op->getLoc(), mlir::TypeRange(), sorted_units,
        ArrayAttr::get(builder.getContext(), list_sizes),
        /*regIndices*/ nullptr, /*regLocales*/ nullptr, num_of_regions);
    auto src0_corelet0_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_lx_corelet0[0].first.getDefiningOp());
    auto dst0_corelet0_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_lx_corelet0[0].second.getDefiningOp());
    auto src0_corelet1_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_lx_corelet1[0].first.getDefiningOp());
    auto dst0_corelet1_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_lx_corelet1[0].second.getDefiningOp());
    DT_CHECK(src0_corelet0_op && dst0_corelet0_op && src0_corelet1_op &&
             dst0_corelet1_op);
    auto &block0 = uniform_region.getRegion(0).emplaceBlock();
    block0.addArgument(builder.getIndexType(), op->getLoc());
    OpBuilder builder_region0(uniform_region.getRegion(0));
    if (is_src_l3) {
      if (lowerL3SyncOperationForAUnit(op, builder_region0, {src0_corelet0_op},
                                       dst0_corelet0_op)
              .failed())
        return LogicalResult::failure();
    } else if (corelet_id == 0) {
      if (lowerL0LXSyncOperationForAUnit(
              op, builder_region0, {src0_corelet0_op}, {}, dst0_corelet0_op)
              .failed())
        return LogicalResult::failure();
    } else {
      if (lowerL0LXSyncOperationForAUnit(op, builder_region0, {},
                                         {src0_corelet0_op}, dst0_corelet0_op)
              .failed())
        return LogicalResult::failure();
    }
    uniform::YieldOp::create(builder_region0, op->getLoc());
    auto &block1 = uniform_region.getRegion(1).emplaceBlock();
    block1.addArgument(builder.getIndexType(), op->getLoc());
    OpBuilder builder_region1(uniform_region.getRegion(1));
    if (is_src_l3) {
      if (lowerL3SyncOperationForAUnit(op, builder_region1, {src0_corelet1_op},
                                       dst0_corelet1_op)
              .failed())
        return LogicalResult::failure();
    } else if (corelet_id == 0) {
      if (lowerL0LXSyncOperationForAUnit(
              op, builder_region1, {src0_corelet1_op}, {}, dst0_corelet1_op)
              .failed())
        return LogicalResult::failure();
    } else {
      if (lowerL0LXSyncOperationForAUnit(op, builder_region1, {},
                                         {src0_corelet1_op}, dst0_corelet1_op)
              .failed())
        return LogicalResult::failure();
    }
    uniform::YieldOp::create(builder_region1, op->getLoc());
    return LogicalResult::success();
  } else if ((src_dst_lx_corelet0.size() == 0 &&
              src_dst_lx_corelet1.size() == 0) ||
             (src_dst_l3.size() == 0 && src_dst_lx_corelet0.size() == 0) ||
             (src_dst_l3.size() == 0 && src_dst_lx_corelet1.size() == 0)) {
    // only src_dst_l3 and src_dst_group are none empty or
    // only src_dst_lx_corelet0 and src_dst_group are none empty or
    // only src_dst_lx_corelet1 and src_dst_group are none empty
    bool only_l3_and_groups = false, only_lx_corelet0_and_groups = false,
         only_lx_corelet1_and_groups = false;
    if (src_dst_lx_corelet0.size() == 0 && src_dst_lx_corelet1.size() == 0) {
      only_l3_and_groups = true;
    } else if (src_dst_l3.size() == 0 && src_dst_lx_corelet0.size() == 0) {
      only_lx_corelet1_and_groups = true;
    } else if (src_dst_l3.size() == 0 && src_dst_lx_corelet1.size() == 0) {
      only_lx_corelet0_and_groups = true;
    }
    std::vector<mlir::Value> src_vs_from_l3_pair, src_vs_from_lx_corelet0_pair,
        src_vs_from_lx_corelet1_pair, src_vs_from_pair, dst_vs_from_pair;
    if (only_l3_and_groups) {
      for (auto u : src_dst_l3) src_vs_from_l3_pair.push_back(u.first);
    } else if (only_lx_corelet0_and_groups) {
      for (auto u : src_dst_lx_corelet0)
        src_vs_from_lx_corelet0_pair.push_back(u.first);
    } else if (only_lx_corelet1_and_groups) {
      for (auto u : src_dst_lx_corelet1)
        src_vs_from_lx_corelet1_pair.push_back(u.first);
    }
    for (auto u : src_dst_group) {
      src_vs_from_pair.push_back(u.first);
      dst_vs_from_pair.push_back(u.second);
    }

    bool sameGroup = true;
    for (auto u : dst_vs_from_pair) {
      if (u.getDefiningOp() == dst_vs_from_pair[0].getDefiningOp())
        sameGroup = false;
    }

    if (sameGroup) {
      std::vector<mlir::Value> sorted_units;
      llvm::SmallVector<Attribute, 2> list_sizes;
      if (only_l3_and_groups) {
        for (auto u : src_vs_from_l3_pair) sorted_units.push_back(u);
      } else if (only_lx_corelet0_and_groups) {
        for (auto u : src_vs_from_lx_corelet0_pair) sorted_units.push_back(u);
      } else if (only_lx_corelet1_and_groups) {
        for (auto u : src_vs_from_lx_corelet1_pair) sorted_units.push_back(u);
      }
      for (auto u : src_vs_from_pair) sorted_units.push_back(u);
      if (only_l3_and_groups) {
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_l3_pair.size()));
      } else if (only_lx_corelet0_and_groups) {
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_lx_corelet0_pair.size()));
      } else if (only_lx_corelet1_and_groups) {
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_lx_corelet1_pair.size()));
      }
      list_sizes.push_back(builder.getI32IntegerAttr(src_vs_from_pair.size()));
      int num_of_regions = 2;
      auto uniform_region = mlir::uniform::UniformizeRegionsOp::create(
          builder, op->getLoc(), mlir::TypeRange(), sorted_units,
          ArrayAttr::get(builder.getContext(), list_sizes),
          /*regIndices*/ nullptr, /*regLocales*/ nullptr, num_of_regions);
      auto &block0 = uniform_region.getRegion(0).emplaceBlock();
      block0.addArgument(builder.getIndexType(), op->getLoc());
      OpBuilder builder_region0(uniform_region.getRegion(0));
      // L3/LX
      mlir::dataflow::GetUnitOp src_op, dst_op;
      if (only_l3_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_l3_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_l3[0].second.getDefiningOp());
      } else if (only_lx_corelet0_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_lx_corelet0_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_lx_corelet0[0].second.getDefiningOp());
      } else if (only_lx_corelet1_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_lx_corelet1_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_lx_corelet1[0].second.getDefiningOp());
      }
      DT_CHECK(src_op && dst_op);
      if (is_src_l3) {
        if (lowerL3SyncOperationForAUnit(op, builder_region0, {src_op}, dst_op)
                .failed())
          return LogicalResult::failure();
      } else if (corelet_id == 0) {
        if (lowerL0LXSyncOperationForAUnit(op, builder_region0, {src_op}, {},
                                           dst_op)
                .failed())
          return LogicalResult::failure();
      } else {
        if (lowerL0LXSyncOperationForAUnit(op, builder_region0, {}, {src_op},
                                           dst_op)
                .failed())
          return LogicalResult::failure();
      }
      uniform::YieldOp::create(builder_region0, op->getLoc());
      auto &block1 = uniform_region.getRegion(1).emplaceBlock();
      block1.addArgument(builder.getIndexType(), op->getLoc());
      OpBuilder builder_region1(uniform_region.getRegion(1));
      // group
      src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
          src_vs_from_pair[0].getDefiningOp());
      auto group_op = llvm::dyn_cast<dataflow::CreateGroupOp>(
          dst_vs_from_pair[0].getDefiningOp());
      if (is_src_l3) {
        if (lowerL3SyncOperationForAGroupOfUnits(op, builder_region1, {src_op},
                                                 group_op)
                .failed())
          return LogicalResult::failure();
      } else if (corelet_id == 0) {
        if (lowerL0LXSyncOperationForAGroupOfUnits(op, builder_region1,
                                                   {src_op}, {}, group_op)
                .failed())
          return LogicalResult::failure();
      } else {
        if (lowerL0LXSyncOperationForAGroupOfUnits(op, builder_region1, {},
                                                   {src_op}, group_op)
                .failed())
          return LogicalResult::failure();
      }
      uniform::YieldOp::create(builder_region1, op->getLoc());
    } else {
      std::vector<mlir::Value> sorted_units;
      llvm::SmallVector<Attribute, 2> list_sizes;
      if (only_l3_and_groups) {
        for (auto u : src_vs_from_l3_pair) sorted_units.push_back(u);
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_l3_pair.size()));
      } else if (only_lx_corelet0_and_groups) {
        for (auto u : src_vs_from_lx_corelet0_pair) sorted_units.push_back(u);
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_lx_corelet0_pair.size()));
      } else if (only_lx_corelet1_and_groups) {
        for (auto u : src_vs_from_lx_corelet1_pair) sorted_units.push_back(u);
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_lx_corelet1_pair.size()));
      }
      for (auto src : src_vs_from_pair) sorted_units.push_back(src);
      for (int i = 0; i < dst_vs_from_pair.size(); i++) {
        list_sizes.push_back(builder.getI32IntegerAttr(1));
      }

      int num_of_regions = dst_vs_from_pair.size() + 1;
      auto uniform_region = mlir::uniform::UniformizeRegionsOp::create(
          builder, op->getLoc(), mlir::TypeRange(), sorted_units,
          ArrayAttr::get(builder.getContext(), list_sizes),
          /*regIndices*/ nullptr, /*regLocales*/ nullptr, num_of_regions);
      auto &block0 = uniform_region.getRegion(0).emplaceBlock();
      block0.addArgument(builder.getIndexType(), op->getLoc());
      OpBuilder builder_region0(uniform_region.getRegion(0));
      // L3/LX
      mlir::dataflow::GetUnitOp src_op, dst_op;
      if (only_l3_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_l3_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_l3[0].second.getDefiningOp());
      } else if (only_lx_corelet0_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_lx_corelet0_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_lx_corelet0[0].second.getDefiningOp());
      } else if (only_lx_corelet1_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_lx_corelet1_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_lx_corelet1[0].second.getDefiningOp());
      }
      DT_CHECK(src_op && dst_op);
      if (is_src_l3) {
        if (lowerL3SyncOperationForAUnit(op, builder_region0, {src_op}, dst_op)
                .failed())
          return LogicalResult::failure();
      } else if (corelet_id == 0) {
        if (lowerL0LXSyncOperationForAUnit(op, builder_region0, {src_op}, {},
                                           dst_op)
                .failed())
          return LogicalResult::failure();
      } else {
        if (lowerL0LXSyncOperationForAUnit(op, builder_region0, {}, {src_op},
                                           dst_op)
                .failed())
          return LogicalResult::failure();
      }
      uniform::YieldOp::create(builder_region0, op->getLoc());
      for (int i = 1; i < num_of_regions; i++) {
        auto &block1 = uniform_region.getRegion(i).emplaceBlock();
        block1.addArgument(builder.getIndexType(), op->getLoc());
        OpBuilder builder_region1(uniform_region.getRegion(i));
        // groups
        auto src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_pair[i - 1].getDefiningOp());
        auto group_op = llvm::dyn_cast<dataflow::CreateGroupOp>(
            dst_vs_from_pair[i - 1].getDefiningOp());
        if (is_src_l3) {
          if (lowerL3SyncOperationForAGroupOfUnits(op, builder_region1,
                                                   {src_op}, group_op)
                  .failed())
            return LogicalResult::failure();
        } else if (corelet_id == 0) {
          if (lowerL0LXSyncOperationForAGroupOfUnits(op, builder_region1,
                                                     {src_op}, {}, group_op)
                  .failed())
            return LogicalResult::failure();
        } else {
          if (lowerL0LXSyncOperationForAGroupOfUnits(op, builder_region1, {},
                                                     {src_op}, group_op)
                  .failed())
            return LogicalResult::failure();
        }
        uniform::YieldOp::create(builder_region1, op->getLoc());
      }
    }
    return LogicalResult::success();
  } else if (src_dst_group.size() == 0 && src_dst_lx_corelet0.size() == 0) {
    // only src_dst_lx_corelet1 and src_dst_l3 are none empty
    std::vector<mlir::Value> sorted_units;
    for (auto src_dst : src_dst_lx_corelet1)
      sorted_units.push_back(src_dst.first);
    for (auto src_dst : src_dst_l3) sorted_units.push_back(src_dst.first);
    llvm::SmallVector<Attribute, 2> list_sizes = {
        builder.getI32IntegerAttr(src_dst_lx_corelet1.size()),
        builder.getI32IntegerAttr(src_dst_l3.size())};
    int num_of_regions = 2;
    auto uniform_region = mlir::uniform::UniformizeRegionsOp::create(
        builder, op->getLoc(), mlir::TypeRange(), sorted_units,
        ArrayAttr::get(builder.getContext(), list_sizes),
        /*regIndices*/ nullptr, /*regLocales*/ nullptr, num_of_regions);
    auto src0_corelet1_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_lx_corelet1[0].first.getDefiningOp());
    auto dst0_corelet1_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_lx_corelet1[0].second.getDefiningOp());
    auto src0_l3_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_l3[0].first.getDefiningOp());
    auto dst0_l3_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_l3[0].second.getDefiningOp());
    DT_CHECK(src0_corelet1_op && dst0_corelet1_op && src0_l3_op && dst0_l3_op);

    auto &block0 = uniform_region.getRegion(0).emplaceBlock();
    block0.addArgument(builder.getIndexType(), op->getLoc());
    OpBuilder builder_region0(uniform_region.getRegion(0));
    if (is_src_l3) {
      if (lowerL3SyncOperationForAUnit(op, builder_region0, {src0_corelet1_op},
                                       dst0_corelet1_op)
              .failed())
        return LogicalResult::failure();
    } else if (corelet_id == 0) {
      if (lowerL0LXSyncOperationForAUnit(
              op, builder_region0, {src0_corelet1_op}, {}, dst0_corelet1_op)
              .failed())
        return LogicalResult::failure();
    } else {
      if (lowerL0LXSyncOperationForAUnit(op, builder_region0, {},
                                         {src0_corelet1_op}, dst0_corelet1_op)
              .failed())
        return LogicalResult::failure();
    }
    uniform::YieldOp::create(builder_region0, op->getLoc());

    auto &block1 = uniform_region.getRegion(1).emplaceBlock();
    block1.addArgument(builder.getIndexType(), op->getLoc());
    OpBuilder builder_region1(uniform_region.getRegion(1));
    if (is_src_l3) {
      if (lowerL3SyncOperationForAUnit(op, builder_region1, {src0_l3_op},
                                       dst0_l3_op)
              .failed())
        return LogicalResult::failure();
    } else if (corelet_id == 0) {
      if (lowerL0LXSyncOperationForAUnit(op, builder_region1, {src0_l3_op}, {},
                                         dst0_l3_op)
              .failed())
        return LogicalResult::failure();
    } else {
      if (lowerL0LXSyncOperationForAUnit(op, builder_region1, {}, {src0_l3_op},
                                         dst0_l3_op)
              .failed())
        return LogicalResult::failure();
    }
    uniform::YieldOp::create(builder_region1, op->getLoc());
    return LogicalResult::success();
  } else if (src_dst_group.size() == 0 && src_dst_lx_corelet1.size() == 0) {
    // only src_dst_lx_corelet0 and src_dst_l3 are none empty
    std::vector<mlir::Value> sorted_units;
    for (auto src_dst : src_dst_lx_corelet0)
      sorted_units.push_back(src_dst.first);
    for (auto src_dst : src_dst_l3) sorted_units.push_back(src_dst.first);
    llvm::SmallVector<Attribute, 2> list_sizes = {
        builder.getI32IntegerAttr(src_dst_lx_corelet0.size()),
        builder.getI32IntegerAttr(src_dst_l3.size())};
    int num_of_regions = 2;
    auto uniform_region = mlir::uniform::UniformizeRegionsOp::create(
        builder, op->getLoc(), mlir::TypeRange(), sorted_units,
        ArrayAttr::get(builder.getContext(), list_sizes),
        /*regIndices*/ nullptr, /*regLocales*/ nullptr, num_of_regions);
    auto src0_corelet0_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_lx_corelet0[0].first.getDefiningOp());
    auto dst0_corelet0_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_lx_corelet0[0].second.getDefiningOp());
    auto src0_l3_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_l3[0].first.getDefiningOp());
    auto dst0_l3_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_l3[0].second.getDefiningOp());
    DT_CHECK(src0_corelet0_op && dst0_corelet0_op && src0_l3_op && dst0_l3_op);
    auto &block0 = uniform_region.getRegion(0).emplaceBlock();
    block0.addArgument(builder.getIndexType(), op->getLoc());
    OpBuilder builder_region0(uniform_region.getRegion(0));
    if (is_src_l3) {
      if (lowerL3SyncOperationForAUnit(op, builder_region0, {src0_corelet0_op},
                                       dst0_corelet0_op)
              .failed())
        return LogicalResult::failure();
    } else if (corelet_id == 0) {
      if (lowerL0LXSyncOperationForAUnit(
              op, builder_region0, {src0_corelet0_op}, {}, dst0_corelet0_op)
              .failed())
        return LogicalResult::failure();
    } else {
      if (lowerL0LXSyncOperationForAUnit(op, builder_region0, {},
                                         {src0_corelet0_op}, dst0_corelet0_op)
              .failed())
        return LogicalResult::failure();
    }
    uniform::YieldOp::create(builder_region0, op->getLoc());
    auto &block1 = uniform_region.getRegion(1).emplaceBlock();
    block1.addArgument(builder.getIndexType(), op->getLoc());
    OpBuilder builder_region1(uniform_region.getRegion(1));
    if (is_src_l3) {
      if (lowerL3SyncOperationForAUnit(op, builder_region1, {src0_l3_op},
                                       dst0_l3_op)
              .failed())
        return LogicalResult::failure();
    } else if (corelet_id == 0) {
      if (lowerL0LXSyncOperationForAUnit(op, builder_region1, {src0_l3_op}, {},
                                         dst0_l3_op)
              .failed())
        return LogicalResult::failure();
    } else {
      if (lowerL0LXSyncOperationForAUnit(op, builder_region1, {}, {src0_l3_op},
                                         dst0_l3_op)
              .failed())
        return LogicalResult::failure();
    }
    uniform::YieldOp::create(builder_region1, op->getLoc());
    return LogicalResult::success();
  } else if (src_dst_group.size() == 0) {
    // only src_dst_lx_corelet0, src_dst_lx_corelet1 and src_dst_l3 are none
    // empty
    std::vector<mlir::Value> sorted_units;
    for (auto src_dst : src_dst_lx_corelet0)
      sorted_units.push_back(src_dst.first);
    for (auto src_dst : src_dst_lx_corelet1)
      sorted_units.push_back(src_dst.first);
    for (auto src_dst : src_dst_l3) sorted_units.push_back(src_dst.first);
    llvm::SmallVector<Attribute, 2> list_sizes = {
        builder.getI32IntegerAttr(src_dst_lx_corelet0.size()),
        builder.getI32IntegerAttr(src_dst_lx_corelet1.size()),
        builder.getI32IntegerAttr(src_dst_l3.size())};
    int num_of_regions = 3;
    auto uniform_region = mlir::uniform::UniformizeRegionsOp::create(
        builder, op->getLoc(), mlir::TypeRange(), sorted_units,
        ArrayAttr::get(builder.getContext(), list_sizes),
        /*regIndices*/ nullptr, /*regLocales*/ nullptr, num_of_regions);
    auto src0_corelet0_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_lx_corelet0[0].first.getDefiningOp());
    auto dst0_corelet0_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_lx_corelet0[0].second.getDefiningOp());
    auto src0_corelet1_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_lx_corelet1[0].first.getDefiningOp());
    auto dst0_corelet1_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_lx_corelet1[0].second.getDefiningOp());
    auto src0_l3_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_l3[0].first.getDefiningOp());
    auto dst0_l3_op = llvm::dyn_cast<dataflow::GetUnitOp>(
        src_dst_l3[0].second.getDefiningOp());
    DT_CHECK(src0_corelet0_op && dst0_corelet0_op && src0_corelet1_op &&
             dst0_corelet1_op && src0_l3_op && dst0_l3_op);

    auto &block0 = uniform_region.getRegion(0).emplaceBlock();
    block0.addArgument(builder.getIndexType(), op->getLoc());
    OpBuilder builder_region0(uniform_region.getRegion(0));
    if (is_src_l3) {
      if (lowerL3SyncOperationForAUnit(op, builder_region0, {src0_corelet0_op},
                                       dst0_corelet0_op)
              .failed())
        return LogicalResult::failure();
    } else if (corelet_id == 0) {
      if (lowerL0LXSyncOperationForAUnit(
              op, builder_region0, {src0_corelet0_op}, {}, dst0_corelet0_op)
              .failed())
        return LogicalResult::failure();
    } else {
      if (lowerL0LXSyncOperationForAUnit(op, builder_region0, {},
                                         {src0_corelet0_op}, dst0_corelet0_op)
              .failed())
        return LogicalResult::failure();
    }
    uniform::YieldOp::create(builder_region0, op->getLoc());
    auto &block1 = uniform_region.getRegion(1).emplaceBlock();
    block1.addArgument(builder.getIndexType(), op->getLoc());
    OpBuilder builder_region1(uniform_region.getRegion(1));
    if (is_src_l3) {
      if (lowerL3SyncOperationForAUnit(op, builder_region1, {src0_corelet1_op},
                                       dst0_corelet1_op)
              .failed())
        return LogicalResult::failure();
    } else if (corelet_id == 0) {
      if (lowerL0LXSyncOperationForAUnit(
              op, builder_region1, {src0_corelet1_op}, {}, dst0_corelet1_op)
              .failed())
        return LogicalResult::failure();
    } else {
      if (lowerL0LXSyncOperationForAUnit(op, builder_region1, {},
                                         {src0_corelet1_op}, dst0_corelet1_op)
              .failed())
        return LogicalResult::failure();
    }
    uniform::YieldOp::create(builder_region1, op->getLoc());
    auto &block2 = uniform_region.getRegion(2).emplaceBlock();
    block2.addArgument(builder.getIndexType(), op->getLoc());
    OpBuilder builder_region2(uniform_region.getRegion(2));
    if (is_src_l3) {
      if (lowerL3SyncOperationForAUnit(op, builder_region2, {src0_l3_op},
                                       dst0_l3_op)
              .failed())
        return LogicalResult::failure();
    } else if (corelet_id == 0) {
      if (lowerL0LXSyncOperationForAUnit(op, builder_region2, {src0_l3_op}, {},
                                         dst0_l3_op)
              .failed())
        return LogicalResult::failure();
    } else {
      if (lowerL0LXSyncOperationForAUnit(op, builder_region2, {}, {src0_l3_op},
                                         dst0_l3_op)
              .failed())
        return LogicalResult::failure();
    }
    uniform::YieldOp::create(builder_region2, op->getLoc());
    return LogicalResult::success();
  } else if (src_dst_l3.size() == 0 || src_dst_lx_corelet1.size() == 0 ||
             src_dst_lx_corelet0.size() == 0) {
    // only src_dst_lx_corelet0, src_dst_lx_corelet1 and src_dst_group are none
    // empty
    // only src_dst_lx_corelet0, src_dst_l3 and src_dst_group are none empty
    // only src_dst_lx_corelet1, src_dst_l3 and src_dst_group are none empty
    bool only_lx0_lx1_and_groups = false, only_lx0_l3_and_groups = false,
         only_lx1_l3_and_groups = false;
    if (src_dst_l3.size() == 0)
      only_lx0_lx1_and_groups = true;
    else if (src_dst_lx_corelet1.size() == 0)
      only_lx0_l3_and_groups = true;
    else if (src_dst_lx_corelet0.size() == 0)
      only_lx1_l3_and_groups = true;

    std::vector<mlir::Value> src_vs_from_l3_pair, src_vs_from_lx_corelet0_pair,
        src_vs_from_lx_corelet1_pair, src_vs_from_pair, dst_vs_from_pair;
    if (only_lx0_lx1_and_groups) {
      for (auto u : src_dst_lx_corelet0)
        src_vs_from_lx_corelet0_pair.push_back(u.first);
      for (auto u : src_dst_lx_corelet1)
        src_vs_from_lx_corelet1_pair.push_back(u.first);
    } else if (only_lx0_l3_and_groups) {
      for (auto u : src_dst_lx_corelet0)
        src_vs_from_lx_corelet0_pair.push_back(u.first);
      for (auto u : src_dst_l3) src_vs_from_l3_pair.push_back(u.first);
    } else if (only_lx1_l3_and_groups) {
      for (auto u : src_dst_lx_corelet1)
        src_vs_from_lx_corelet1_pair.push_back(u.first);
      for (auto u : src_dst_l3) src_vs_from_l3_pair.push_back(u.first);
    }
    for (auto u : src_dst_group) {
      src_vs_from_pair.push_back(u.first);
      dst_vs_from_pair.push_back(u.second);
    }

    bool sameGroup = true;
    for (auto u : dst_vs_from_pair) {
      if (u.getDefiningOp() == dst_vs_from_pair[0].getDefiningOp())
        sameGroup = false;
    }

    if (sameGroup) {
      std::vector<mlir::Value> sorted_units;
      llvm::SmallVector<Attribute, 2> list_sizes;
      if (only_lx0_lx1_and_groups) {
        for (auto u : src_vs_from_lx_corelet0_pair) sorted_units.push_back(u);
        for (auto u : src_vs_from_lx_corelet1_pair) sorted_units.push_back(u);
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_lx_corelet0_pair.size()));
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_lx_corelet1_pair.size()));
      } else if (only_lx0_l3_and_groups) {
        for (auto u : src_vs_from_lx_corelet0_pair) sorted_units.push_back(u);
        for (auto u : src_vs_from_l3_pair) sorted_units.push_back(u);
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_lx_corelet0_pair.size()));
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_l3_pair.size()));
      } else if (only_lx1_l3_and_groups) {
        for (auto u : src_vs_from_lx_corelet1_pair) sorted_units.push_back(u);
        for (auto u : src_vs_from_l3_pair) sorted_units.push_back(u);
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_lx_corelet1_pair.size()));
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_l3_pair.size()));
      }
      for (auto u : src_vs_from_pair) sorted_units.push_back(u);
      list_sizes.push_back(builder.getI32IntegerAttr(src_vs_from_pair.size()));
      int num_of_regions = 3;
      auto uniform_region = mlir::uniform::UniformizeRegionsOp::create(
          builder, op->getLoc(), mlir::TypeRange(), sorted_units,
          ArrayAttr::get(builder.getContext(), list_sizes),
          /*regIndices*/ nullptr, /*regLocales*/ nullptr, num_of_regions);
      auto &block0 = uniform_region.getRegion(0).emplaceBlock();
      block0.addArgument(builder.getIndexType(), op->getLoc());
      OpBuilder builder_region0(uniform_region.getRegion(0));
      // L3/LX
      mlir::dataflow::GetUnitOp src_op, dst_op;
      if (only_lx0_lx1_and_groups || only_lx0_l3_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_lx_corelet0_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_lx_corelet0[0].second.getDefiningOp());
      } else if (only_lx1_l3_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_lx_corelet1_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_lx_corelet1[0].second.getDefiningOp());
      }
      DT_CHECK(src_op && dst_op);
      if (is_src_l3) {
        if (lowerL3SyncOperationForAUnit(op, builder_region0, {src_op}, dst_op)
                .failed())
          return LogicalResult::failure();
      } else if (corelet_id == 0) {
        if (lowerL0LXSyncOperationForAUnit(op, builder_region0, {src_op}, {},
                                           dst_op)
                .failed())
          return LogicalResult::failure();
      } else {
        if (lowerL0LXSyncOperationForAUnit(op, builder_region0, {}, {src_op},
                                           dst_op)
                .failed())
          return LogicalResult::failure();
      }
      uniform::YieldOp::create(builder_region0, op->getLoc());
      auto &block1 = uniform_region.getRegion(1).emplaceBlock();
      block1.addArgument(builder.getIndexType(), op->getLoc());
      OpBuilder builder_region1(uniform_region.getRegion(1));
      // L3/LX
      if (only_lx0_lx1_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_lx_corelet1_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_lx_corelet1[0].second.getDefiningOp());
      } else if (only_lx0_l3_and_groups || only_lx1_l3_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_l3_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_l3[0].second.getDefiningOp());
      }
      DT_CHECK(src_op && dst_op);
      if (is_src_l3) {
        if (lowerL3SyncOperationForAUnit(op, builder_region1, {src_op}, dst_op)
                .failed())
          return LogicalResult::failure();
      } else if (corelet_id == 0) {
        if (lowerL0LXSyncOperationForAUnit(op, builder_region1, {src_op}, {},
                                           dst_op)
                .failed())
          return LogicalResult::failure();
      } else {
        if (lowerL0LXSyncOperationForAUnit(op, builder_region1, {}, {src_op},
                                           dst_op)
                .failed())
          return LogicalResult::failure();
      }
      uniform::YieldOp::create(builder_region1, op->getLoc());
      auto &block2 = uniform_region.getRegion(2).emplaceBlock();
      block2.addArgument(builder.getIndexType(), op->getLoc());
      OpBuilder builder_region2(uniform_region.getRegion(2));
      // group
      src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
          src_vs_from_pair[0].getDefiningOp());
      auto group_op = llvm::dyn_cast<dataflow::CreateGroupOp>(
          dst_vs_from_pair[0].getDefiningOp());
      if (is_src_l3) {
        if (lowerL3SyncOperationForAGroupOfUnits(op, builder_region2, {src_op},
                                                 group_op)
                .failed())
          return LogicalResult::failure();
      } else if (corelet_id == 0) {
        if (lowerL0LXSyncOperationForAGroupOfUnits(op, builder_region2,
                                                   {src_op}, {}, group_op)
                .failed())
          return LogicalResult::failure();
      } else {
        if (lowerL0LXSyncOperationForAGroupOfUnits(op, builder_region2, {},
                                                   {src_op}, group_op)
                .failed())
          return LogicalResult::failure();
      }
      uniform::YieldOp::create(builder_region2, op->getLoc());
    } else {
      std::vector<mlir::Value> sorted_units;
      llvm::SmallVector<Attribute, 2> list_sizes;
      if (only_lx0_lx1_and_groups) {
        for (auto u : src_vs_from_lx_corelet0_pair) sorted_units.push_back(u);
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_lx_corelet0_pair.size()));
        for (auto u : src_vs_from_lx_corelet1_pair) sorted_units.push_back(u);
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_lx_corelet1_pair.size()));
      } else if (only_lx0_l3_and_groups) {
        for (auto u : src_vs_from_lx_corelet0_pair) sorted_units.push_back(u);
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_lx_corelet0_pair.size()));
        for (auto u : src_vs_from_l3_pair) sorted_units.push_back(u);
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_l3_pair.size()));
      } else if (only_lx1_l3_and_groups) {
        for (auto u : src_vs_from_lx_corelet1_pair) sorted_units.push_back(u);
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_lx_corelet1_pair.size()));
        for (auto u : src_vs_from_l3_pair) sorted_units.push_back(u);
        list_sizes.push_back(
            builder.getI32IntegerAttr(src_vs_from_l3_pair.size()));
      }
      for (auto src : src_vs_from_pair) sorted_units.push_back(src);
      for (int i = 0; i < dst_vs_from_pair.size(); i++) {
        list_sizes.push_back(builder.getI32IntegerAttr(1));
      }

      int num_of_regions = dst_vs_from_pair.size() + 2;
      auto uniform_region = mlir::uniform::UniformizeRegionsOp::create(
          builder, op->getLoc(), mlir::TypeRange(), sorted_units,
          ArrayAttr::get(builder.getContext(), list_sizes),
          /*regIndices*/ nullptr, /*regLocales*/ nullptr, num_of_regions);
      auto &block0 = uniform_region.getRegion(0).emplaceBlock();
      block0.addArgument(builder.getIndexType(), op->getLoc());
      OpBuilder builder_region0(uniform_region.getRegion(0));
      // L3/LX
      mlir::dataflow::GetUnitOp src_op, dst_op;
      if (only_lx0_lx1_and_groups || only_lx0_l3_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_lx_corelet0_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_lx_corelet0[0].second.getDefiningOp());
      } else if (only_lx1_l3_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_lx_corelet1_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_lx_corelet1[0].second.getDefiningOp());
      }
      DT_CHECK(src_op && dst_op);
      if (is_src_l3) {
        if (lowerL3SyncOperationForAUnit(op, builder_region0, {src_op}, dst_op)
                .failed())
          return LogicalResult::failure();
      } else if (corelet_id == 0) {
        if (lowerL0LXSyncOperationForAUnit(op, builder_region0, {src_op}, {},
                                           dst_op)
                .failed())
          return LogicalResult::failure();
      } else {
        if (lowerL0LXSyncOperationForAUnit(op, builder_region0, {}, {src_op},
                                           dst_op)
                .failed())
          return LogicalResult::failure();
      }
      uniform::YieldOp::create(builder_region0, op->getLoc());

      auto &block1 = uniform_region.getRegion(1).emplaceBlock();
      block1.addArgument(builder.getIndexType(), op->getLoc());
      OpBuilder builder_region1(uniform_region.getRegion(1));
      // L3/LX
      if (only_lx0_lx1_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_lx_corelet1_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_lx_corelet1[0].second.getDefiningOp());
      } else if (only_lx0_l3_and_groups || only_lx1_l3_and_groups) {
        src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_l3_pair[0].getDefiningOp());
        dst_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_dst_l3[0].second.getDefiningOp());
      }
      DT_CHECK(src_op && dst_op);
      if (is_src_l3) {
        if (lowerL3SyncOperationForAUnit(op, builder_region1, {src_op}, dst_op)
                .failed())
          return LogicalResult::failure();
      } else if (corelet_id == 0) {
        if (lowerL0LXSyncOperationForAUnit(op, builder_region1, {src_op}, {},
                                           dst_op)
                .failed())
          return LogicalResult::failure();
      } else {
        if (lowerL0LXSyncOperationForAUnit(op, builder_region1, {}, {src_op},
                                           dst_op)
                .failed())
          return LogicalResult::failure();
      }
      uniform::YieldOp::create(builder_region1, op->getLoc());
      for (int i = 2; i < num_of_regions; i++) {
        auto &blocki = uniform_region.getRegion(i).emplaceBlock();
        blocki.addArgument(builder.getIndexType(), op->getLoc());
        OpBuilder builder_regioni(uniform_region.getRegion(i));
        // groups
        auto src_op = llvm::dyn_cast<dataflow::GetUnitOp>(
            src_vs_from_pair[i - 2].getDefiningOp());
        auto group_op = llvm::dyn_cast<dataflow::CreateGroupOp>(
            dst_vs_from_pair[i - 2].getDefiningOp());
        if (is_src_l3) {
          if (lowerL3SyncOperationForAGroupOfUnits(op, builder_regioni,
                                                   {src_op}, group_op)
                  .failed())
            return LogicalResult::failure();
        } else if (corelet_id == 0) {
          if (lowerL0LXSyncOperationForAGroupOfUnits(op, builder_regioni,
                                                     {src_op}, {}, group_op)
                  .failed())
            return LogicalResult::failure();
        } else {
          if (lowerL0LXSyncOperationForAGroupOfUnits(op, builder_regioni, {},
                                                     {src_op}, group_op)
                  .failed())
            return LogicalResult::failure();
        }
        uniform::YieldOp::create(builder_regioni, op->getLoc());
      }
    }

}}
// ---- 303/384  runOnOperation  —  dcc/src/Conversion/SymbolToSentient/SymbolToSentient.cpp:23  (15L)
void e303_runOnOperation() {
  ModuleOp module_op = getOperation();
  module_op.walk<WalkOrder::PreOrder>([&](dataflow::ProgramUnitOp unit) {
    auto comp = dcc::getUnitType(
        unit.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
    std::vector<Operation*> to_be_deleted_list;
    unit.walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (auto query_map = llvm::dyn_cast<symbol::SymbolQueryMapOp>(op))
        LowerSymbolQueryMap(comp, query_map, to_be_deleted_list);
    });
    for (Operation* op : to_be_deleted_list) {
      DT_CHECK_MSG(op->use_empty(), "uses of op still exist");
      op->erase();
    }
  });

}
// ---- 304/384  getOperandWithPrecision  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:389  (254L)
std::optional<VectorOperand> e304_getOperandWithPrecision(
    const dcc::DccExtContext &dcc_ext_ctx, Operation *op,
    const SenComponents &comp, bool &is_precision_converted,
    bool traverse_upwards) {
  is_precision_converted = false;
  if (isa<vector::LoadOp>(op)) {
    auto load_op = dyn_cast<vector::LoadOp>(op);
    auto operand = getOperandFromLoadOrStoreOp(load_op);
    operand.value().orig_precision_ = dataflow::utils::getPrecisionInString(
        mlir::dyn_cast<VectorType>(load_op.getType()).getElementType());
    operand.value().on_the_fly_conv_precision_ =
        operand.value().orig_precision_;
    return operand;
  } else if (isa<vector::StoreOp>(op)) {
    auto store_op = dyn_cast<vector::StoreOp>(op);
    auto operand = getOperandFromLoadOrStoreOp(store_op);
    operand.value().orig_precision_ = dataflow::utils::getPrecisionInString(
        mlir::dyn_cast<VectorType>(store_op.getValueToStore().getType())
            .getElementType());
    operand.value().on_the_fly_conv_precision_ =
        operand.value().orig_precision_;
    return operand;
  } else if (isa<agen::VectorStoreOp>(op)) {
    auto store_op = dyn_cast<agen::VectorStoreOp>(op);
    auto operand = getOperandFromLoadOrStoreOp(store_op);
    auto elem_type =
        dataflow::utils::getElementType(store_op.getValueToStore().getType());
    operand.value().orig_precision_ =
        dataflow::utils::getPrecisionInString(elem_type);
    operand.value().on_the_fly_conv_precision_ =
        operand.value().orig_precision_;
    return operand;
  } else if (isa<agen::VectorLoadOp>(op)) {
    auto load_op = dyn_cast<agen::VectorLoadOp>(op);
    auto operand = getOperandFromLoadOrStoreOp(load_op);
    auto elem_type =
        dataflow::utils::getElementType(load_op.getResult().getType());
    operand.value().orig_precision_ =
        dataflow::utils::getPrecisionInString(elem_type);
    operand.value().on_the_fly_conv_precision_ =
        operand.value().orig_precision_;
    return operand;
  } else if (isa<dataflow::SendOp>(op)) {
    auto send_op = dyn_cast<dataflow::SendOp>(op);
    auto operand = getOperandFromSendOp(dcc_ext_ctx, send_op, comp);
    auto elem_type =
        dataflow::utils::getElementType(send_op.getSendData().getType());
    operand.value().orig_precision_ =
        dataflow::utils::getPrecisionInString(elem_type);
    operand.value().on_the_fly_conv_precision_ =
        operand.value().orig_precision_;
    return operand;
  } else if (isa<dataflow::ReceiveOp>(op)) {
    auto recv_op = dyn_cast<dataflow::ReceiveOp>(op);
    auto operand = getOperandFromReceiveOp(dcc_ext_ctx, recv_op, comp);
    auto elem_type =
        dataflow::utils::getElementType(recv_op.getData().getType());
    if (operand.value().getFirstValue() == "pt") {
      // Input from PT is int16/fp16 in DD2 and int24/fp24 in Sen1p5
      if (elem_type.isIntOrIndex())
        operand.value().orig_precision_ =
            (dcc_ext_ctx.getArch() >= SEN1P5_ISA) ? "int24" : "int16";
      else
        operand.value().orig_precision_ =
            (dcc_ext_ctx.getArch() >= SEN1P5_ISA) ? "fp24" : "fp16";
    } else {
      operand.value().orig_precision_ =
          dataflow::utils::getPrecisionInString(elem_type);
    }
    operand.value().on_the_fly_conv_precision_ =
        operand.value().orig_precision_;
    return operand;
  } else if (isa<mlir::arith::FPToSIOp>(op)) {
    is_precision_converted = true;
    auto cast_op = dyn_cast<mlir::arith::FPToSIOp>(op);
    if (cast_op->hasOneUse()) {
      if (traverse_upwards) {
        if (cast_op.getIn().getDefiningOp()->getBlock() == op->getBlock()) {
          auto operand =
              getOperand(dcc_ext_ctx, cast_op.getIn().getDefiningOp(), comp);
          if (operand.has_value()) {
            auto elem_type = dataflow::utils::getElementType(cast_op.getType());
            operand.value().on_the_fly_conv_precision_ =
                dataflow::utils::getPrecisionInString(elem_type);
            return operand;
          }
        }
      } else {
        Operation *user = (*cast_op->getUses().begin()).getOwner();
        auto operand = getOperand(dcc_ext_ctx, user, comp);
        if (operand.has_value()) {
          auto elem_type = dataflow::utils::getElementType(cast_op.getType());
          operand.value().on_the_fly_conv_precision_ =
              dataflow::utils::getPrecisionInString(elem_type);
          return operand;
        }
      }
    }
  } else if (isa<mlir::arith::SIToFPOp>(op)) {
    is_precision_converted = true;
    auto cast_op = dyn_cast<mlir::arith::SIToFPOp>(op);
    if (cast_op->hasOneUse()) {
      if (traverse_upwards) {
        if (cast_op.getIn().getDefiningOp()->getBlock() == op->getBlock()) {
          auto operand =
              getOperand(dcc_ext_ctx, cast_op.getIn().getDefiningOp(), comp);
          if (operand.has_value()) {
            auto elem_type = dataflow::utils::getElementType(cast_op.getType());
            operand.value().on_the_fly_conv_precision_ =
                dataflow::utils::getPrecisionInString(elem_type);
            return operand;
          }
        }
      } else {
        Operation *user = (*cast_op->getUses().begin()).getOwner();
        auto operand = getOperand(dcc_ext_ctx, user, comp);
        if (operand.has_value()) {
          auto elem_type = dataflow::utils::getElementType(cast_op.getType());
          operand.value().on_the_fly_conv_precision_ =
              dataflow::utils::getPrecisionInString(elem_type);
          return operand;
        }
      }
    }
  } else if (isa<mlir::arith::ConstantOp>(op)) {
    auto const_op = dyn_cast<mlir::arith::ConstantOp>(op);
    auto operand = getOperandFromConstantOp(const_op);
    auto elem_type = dataflow::utils::getElementType(const_op.getType());
    operand.value().orig_precision_ =
        dataflow::utils::getPrecisionInString(elem_type);
    operand.value().on_the_fly_conv_precision_ =
        operand.value().orig_precision_;
    return operand;
  } else if (auto const_bit_op =
                 dyn_cast<vectorchain::ConstantBitstreamOp>(op)) {
    auto operand = getOperandFromConstantBitstreamOp(const_bit_op);
    auto elem_type = dataflow::utils::getElementType(const_bit_op.getType());
    operand.value().orig_precision_ =
        dataflow::utils::getPrecisionInString(elem_type);
    operand.value().on_the_fly_conv_precision_ =
        operand.value().orig_precision_;
    return operand;
  } else if (auto shuffle_op = dyn_cast<vectorchain::ShuffleOp>(op)) {
    auto operand = getOperandFromShuffleOp(dcc_ext_ctx, shuffle_op, comp);
    if (operand.has_value()) {
      auto elem_type = dataflow::utils::getElementType(shuffle_op.getType());
      operand.value().orig_precision_ =
          dataflow::utils::getPrecisionInString(elem_type);
      operand.value().on_the_fly_conv_precision_ =
          operand.value().orig_precision_;
    }
    return operand;
  } else if (auto cast_op = dyn_cast<vectorchain::CastOp>(op)) {
    if (cast_op->hasOneUse()) {
      is_precision_converted = true;
      if (traverse_upwards) {
        if (cast_op.getInput().getDefiningOp()->getBlock() == op->getBlock()) {
          auto operand =
              getOperand(dcc_ext_ctx, cast_op.getInput().getDefiningOp(), comp);
          if (operand.has_value()) {
            auto elem_type = dataflow::utils::getElementType(cast_op.getType());
            operand.value().on_the_fly_conv_precision_ =
                dataflow::utils::getPrecisionInString(elem_type);
            return operand;
          }
        }
      } else {
        Operation *user = (*cast_op->getUses().begin()).getOwner();
        auto operand = getOperand(dcc_ext_ctx, user, comp);
        if (operand.has_value()) {
          auto elem_type = dataflow::utils::getElementType(cast_op.getType());
          operand.value().on_the_fly_conv_precision_ =
              dataflow::utils::getPrecisionInString(elem_type);
          return operand;
        }
      }
    }
  } else if (auto neg_op = dyn_cast<vectorchain::NegOp>(op)) {
    auto operand = getOperandFromNegOp(dcc_ext_ctx, neg_op, comp);
    // The NegOp doesn't change the original precision
    return operand;
  } else if (isa<vectorchain::SelectOp>(op)) {
    auto select_op = dyn_cast<vectorchain::SelectOp>(op);
    if (select_op->hasOneUse()) {
      if (traverse_upwards) {
        auto operand =
            getOperand(dcc_ext_ctx, select_op.getData().getDefiningOp(), comp);
        auto elem_type =
            dataflow::utils::getElementType(select_op.getData().getType());
        operand.value().orig_precision_ =
            dataflow::utils::getPrecisionInString(elem_type);
        operand.value().on_the_fly_conv_precision_ =
            operand.value().orig_precision_;
        operand.value().splat_ = "east";
        return operand;
      } else {
        Operation *user = (select_op->getUses().begin())->getOwner();
        auto operand = getOperand(dcc_ext_ctx, user, comp);
        if (operand.has_value()) {
          auto elem_type = dataflow::utils::getElementType(select_op.getType());
          operand.value().orig_precision_ =
              dataflow::utils::getPrecisionInString(elem_type);
          operand.value().on_the_fly_conv_precision_ =
              operand.value().orig_precision_;
          return operand;
        }
      }
    }
  } else if (auto ew_compare_op =
                 llvm::dyn_cast<vectorchain::ElementWiseCompareOp>(op)) {
    if (traverse_upwards) {
      for (auto user : ew_compare_op->getUsers()) {
        if (isa<agen::VectorStoreOp>(user)) {
          auto operand =
              getOperandWithPrecision(dcc_ext_ctx, user, comp,
                                      is_precision_converted, traverse_upwards);
          auto elem_type =
              dataflow::utils::getElementType(ew_compare_op.getType());
          operand.value().orig_precision_ =
              dataflow::utils::getPrecisionInString(elem_type);
          operand.value().on_the_fly_conv_precision_ =
              operand.value().orig_precision_;
          return operand;
        }
      }

      // TODO: set appropriate istate number once translator changes are
      // implemented.
      VectorOperand operand =
          VectorOperand(VectorOperandType::ISTATE, "0", ew_compare_op);
      auto elem_type = dataflow::utils::getElementType(ew_compare_op.getType());
      operand.orig_precision_ =
          dataflow::utils::getPrecisionInString(elem_type);
      operand.on_the_fly_conv_precision_ = operand.orig_precision_;
      return operand;
    }
  } else if (auto query_op = llvm::dyn_cast<mlir::uniform::QueryMapOp>(op)) {
    std::optional<VectorOperand> folded_operand;
    auto map_op = llvm::dyn_cast<mlir::uniform::DefImmutableMappingOp>(
        query_op.getMap().getDefiningOp());
    for (auto val : map_op.getValues()) {
      auto operand = getOperand(dcc_ext_ctx, val.getDefiningOp(), comp);
      if (!folded_operand.has_value()) {
        folded_operand = operand;
      } else {
        folded_operand.value().values_.push_back(
            operand.value().values_.front());
      }
    }

    folded_operand.value().op_ = query_op;
    return folded_operand;
  } else if (isa<vectorchain::MultiplyOp, vectorchain::MultiplyAndAccumulateOp,
                 vectorchain::BinaryOp>(op)) {

}}
// ---- 305/384  runOnOperation  —  dcc/src/Transform/Dataflow/FlatteningLocalRegions.cpp:459  (17L)
void e305_runOnOperation() {
  if (DisableThisPass) return;

  ModuleOp module_op = getOperation();
  std::vector<mlir::Operation *> to_be_deleted;
  module_op.walk([&](mlir::Operation *op) {
    if (auto uniform_op = llvm::dyn_cast<uniform::UniformizeRegionsOp>(op)) {
      dcc::uniform::utils::flattenUniformRegion(uniform_op);
    }
  });
  module_op.walk([&](mlir::Operation *op) {
    if (auto uniform_op = llvm::dyn_cast<uniform::UniformizeRegionsOp>(op)) {
      FlatteningLocalRegionsTree tree;
      if (tree.flatten(*op)) to_be_deleted.push_back(op);
    }
  });
  for (auto op : to_be_deleted) op->erase();

}
// ---- 306/384  transformVectorLoad  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:298  (75L)
void e306_transformVectorLoad(MASCandidate &candidate) {
  auto op = dyn_cast<agen::VectorLoadOp>(candidate.op_);
  DT_CHECK(op);

  // Collect the relevant access details.
  agen::AccessContainer<agen::AccessDetailsAffine> access_details;
  agen::AccessDetailsAffine &ad =
      access_details.emplace_insert(candidate.mem_index_, op, candidate.comp_);
  auto result = ad.constructDetails(candidate.mem_index_);
  DT_CHECK(succeeded(result));

  ExpressionEvaluator evaluator;
  SmallVector<MASData> mas_data;
  auto mem_view_op =
      cast<dataflow::GetLogicalMemoryViewOp>(op.getMemRef().getDefiningOp());
  int64_t max_mutable = 0;
  initialize(evaluator, mas_data, mem_view_op, ad, max_mutable);
  if (!hasMutableAddrOverflow(mas_data, candidate.comp_, max_mutable,
                              ad.getElementWidth()))
    return;

  // Vector load and store patterns, despite being two separate operations,
  // will be later merged into one. Collect mem views from load and store
  // patterns, if any.
  SmallVector<Value> all_mem_views;
  Value store_mem_view = nullptr;
  all_mem_views.push_back(mem_view_op.getResult());
  for (auto use : op.getUseChain()) {
    if (auto store_op = dyn_cast<agen::VectorStoreOp>(use)) {
      store_mem_view = store_op.getMemRef();
      all_mem_views.push_back(store_mem_view);
    }
  }

  SmallVector<int64_t> partition_sizes;
  setupForPartitioning(evaluator, mas_data, partition_sizes, all_mem_views,
                       candidate.comp_, mem_view_op, max_mutable,
                       ad.getElementWidth());

  auto createOps = [&](OpBuilder &partition_builder,
                       AffineMap &new_subscripts_map, int64_t start_addr_mod) {
    adjustForEvenImmutableAddr(evaluator, mem_view_op.getStartAddress(),
                               new_subscripts_map, ad, start_addr_mod,
                               ad.getElementWidth());
    auto new_mem_view_op =
        createNewMemViewWithMod(evaluator, partition_builder, candidate.unit_,
                                op, mem_view_op, start_addr_mod);
    partition_builder.setInsertionPointAfter(new_mem_view_op);
    auto new_mem_op =
        op.cloneWithNewAccessInfo(partition_builder, new_mem_view_op,
                                  new_subscripts_map, ad.getIndices());
    partition_builder.setInsertionPointAfter(new_mem_op);
    op.cloneUseChainToNewOp(partition_builder, new_mem_op);
    // Every memory operand should have it's own unique mem view.
    // Clone the mem views for the other memory operands into the partition.
    partition_builder.setInsertionPoint(new_mem_op);
    for (auto use : new_mem_op.getUseChain()) {
      if (auto store_op = dyn_cast<agen::VectorStoreOp>(use)) {
        auto new_store_mem_view =
            partition_builder.clone(*store_mem_view.getDefiningOp());
        store_op.getMemRefMutable().assign(new_store_mem_view->getResult(0));
      }
    }
  };
  createPartitions(mas_data, partition_sizes, op, ad.getSubscriptsMap(),
                   createOps);

  op.eraseOpAndUseChain();

  // TODO: Connect these information messages to DT_DEEPRT_VERBOSE.
  LLVM_DEBUG(llvm::dbgs()
             << "[INFO][DCC][MutableAddrSplitting] Mutable address split in "
             << dcc_ext_ctx_.prog_name_ << " in "
             << EnumsConversion::senComponentsToString.at(candidate.comp_)
             << "\n");

}
// ---- 307/384  transformVectorStore  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:375  (74L)
void e307_transformVectorStore(MASCandidate &candidate) {
  auto op = dyn_cast<agen::VectorStoreOp>(candidate.op_);
  DT_CHECK(op);

  // Collect the relevant access details.
  agen::AccessContainer<agen::AccessDetailsAffine> access_details;
  agen::AccessDetailsAffine &ad =
      access_details.emplace_insert(candidate.mem_index_, op, candidate.comp_);
  auto result = ad.constructDetails(candidate.mem_index_);
  DT_CHECK(succeeded(result));

  ExpressionEvaluator evaluator;
  SmallVector<MASData> mas_data;
  auto mem_view_op =
      cast<dataflow::GetLogicalMemoryViewOp>(op.getMemRef().getDefiningOp());
  int64_t max_mutable = 0;
  initialize(evaluator, mas_data, mem_view_op, ad, max_mutable);
  if (!hasMutableAddrOverflow(mas_data, candidate.comp_, max_mutable,
                              ad.getElementWidth()))
    return;

  // Vector load and store patterns, despite being two separate operations,
  // will be later merged into one. Collect mem views from load and store
  // patterns, if any.
  SmallVector<Value> all_mem_views;
  Value load_mem_view = nullptr;
  all_mem_views.push_back(mem_view_op.getResult());
  for (auto use : op.getUseChain()) {
    if (auto load_op = dyn_cast<agen::VectorLoadOp>(use)) {
      load_mem_view = load_op.getMemRef();
      all_mem_views.push_back(load_mem_view);
    }
  }

  SmallVector<int64_t> partition_sizes;
  setupForPartitioning(evaluator, mas_data, partition_sizes, all_mem_views,
                       candidate.comp_, mem_view_op, max_mutable,
                       ad.getElementWidth());

  auto createOps = [&](OpBuilder &partition_builder,
                       AffineMap &new_subscripts_map, int64_t start_addr_mod) {
    adjustForEvenImmutableAddr(evaluator, mem_view_op.getStartAddress(),
                               new_subscripts_map, ad, start_addr_mod,
                               ad.getElementWidth());
    auto new_mem_view_op =
        createNewMemViewWithMod(evaluator, partition_builder, candidate.unit_,
                                op, mem_view_op, start_addr_mod);
    partition_builder.setInsertionPointAfter(new_mem_view_op);
    auto new_mem_op =
        op.cloneWithNewAccessInfo(partition_builder, new_mem_view_op,
                                  new_subscripts_map, ad.getIndices());
    // Every memory operand should have it's own unique mem view.
    // Clone the mem views for the other memory operands into the partition.
    partition_builder.setInsertionPoint(new_mem_op);
    for (auto use : new_mem_op.getUseChain()) {
      if (auto load_op = dyn_cast<agen::VectorLoadOp>(use)) {
        auto new_load_mem_view =
            partition_builder.clone(*load_mem_view.getDefiningOp());
        load_op.getMemRefMutable().assign(new_load_mem_view->getResult(0));
      }
    }
    op.cloneUseChainToNewOp(partition_builder, new_mem_op);
  };
  createPartitions(mas_data, partition_sizes, op, ad.getSubscriptsMap(),
                   createOps);

  op.eraseOpAndUseChain();

  // TODO: Connect these information messages to DT_DEEPRT_VERBOSE.
  LLVM_DEBUG(llvm::dbgs()
             << "[INFO][DCC][MutableAddrSplitting] Mutable address split in "
             << dcc_ext_ctx_.prog_name_ << " in "
             << EnumsConversion::senComponentsToString.at(candidate.comp_)
             << "\n");

}
// ---- 308/384  calculateShifts  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:396  (61L)
void e308_calculateShifts(
    ExpressionEvaluator &evaluator, SmallVectorImpl<int64_t> &shifts,
    dataflow::GetLogicalMemoryViewOp mem_view_op,
    agen::AccessDetailsAffine &ad) {
  DT_CHECK(shifts.empty());
  auto iter_coeff_dict = ad.getIndicesCoeffDict();
  auto const_offset = iter_coeff_dict[nullptr];

  int64_t num_elems_in_stick =
      dcc_ext_ctx_.getBytesPerStick() * 8 / ad.getElementWidth();

  // If the constant offset is 0, there is no mutable address to shift.
  auto subscripts_map = ad.getSubscriptsMap();
  auto transfer_order = ad.getTransferOrder();
  if (const_offset == 0) {
    for (int i = 0, e = subscripts_map.getNumResults(); i < e; ++i)
      shifts.push_back(0);

    // If arch is sen1p5 up, immutable addresses need to contain even values
    // only.
    if (dcc_ext_ctx_.getArch() < IsaCoreGen::SEN1P5_ISA) return;

    if (dcc::agen::utils::isL3ImmutableAddrEven(
            evaluator, mem_view_op.getStartAddress(), num_elems_in_stick))
      return;

    // If the immutable address is odd, it needs to be shifted by one stick.
    DT_CHECK_MSG(
        dcc::agen::utils::isL3ImmutableAddrAllOdd(
            evaluator, mem_view_op.getStartAddress(), num_elems_in_stick),
        "All immutable addrs must be odd to execute even shift.");
    offsetShifts(shifts, ad, -num_elems_in_stick);
    return;
  }

  // If the constant offset is not 0, how much mutable address we can shift
  // needs to be calculated based on immutable address space.
  auto max_immutable =
      dcc::agen::utils::getMaxImmutableAddress(evaluator, mem_view_op);
  auto max_immutable_range = getMaxImmutableRange(ad.getComp());
  int64_t immutable_space =
      (max_immutable_range / ad.getElementWidth()) - max_immutable;
  int64_t total_shift = 0;
  if (immutable_space >= const_offset)
    total_shift = calculateFullShift(shifts, ad);
  else
    total_shift = calculatePartialShift(shifts, ad, immutable_space);

  if (dcc_ext_ctx_.getArch() >= IsaCoreGen::SEN1P5_ISA) {
    // The immutable addr and the shift amount must either both be even or both
    // be odd to keep the immutable addr an even number of sticks.
    bool is_even = dcc::agen::utils::isL3ImmutableAddrEven(
        evaluator, mem_view_op.getStartAddress(), num_elems_in_stick);
    bool is_even_shift = (total_shift / num_elems_in_stick) % 2 == 0;
    if (is_even != is_even_shift) {
      DT_CHECK_MSG(is_even ? true
                           : dcc::agen::utils::isL3ImmutableAddrAllOdd(
                                 evaluator, mem_view_op.getStartAddress(),
                                 num_elems_in_stick),
                   "All immutable addrs must be all even or all odd to execute "
                   "even shift.");

}}}
// ---- 309/384  constructValidPage  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:190  (58L)
LogicalResult e309_constructValidPage(
    SmallVectorImpl<Operation *> &new_mem_ops, TPMVInfo &info,
    AffineMap &subscripts_map_sym, IntegerSet &page_set,
    FlatLinearValueConstraints &page_sel_constraints, int page_idx) {
  // If conditionals are created, every page may require different
  // subscripts/indices.
  AffineMap subscripts_map = info.subscripts_map_;
  SmallVector<Value> indices = info.indices_;

  FlatLinearValueConstraints page_set_constraints(page_set);

  SmallVector<Operation *, 16> insert_refs = mem_ops_;
  if (!indices.empty()) {
    if (page_sel_constraints.isHyperRectangular(
            0, page_sel_constraints.getNumCols() - 1)) {
      LLVM_DEBUG(llvm::dbgs() << "[TransformPagedMemViewPass] Hyperrectangular "
                                 "subscripts - creating conditionals\n");
      createConditionsForHyperRectSubscripts(insert_refs, page_sel_constraints,
                                             subscripts_map, indices,
                                             info.indices_ranges_);
    } else {
      if (info.conditional_iter_args_.empty()) {
        LLVM_DEBUG(llvm::dbgs()
                   << "[TransformPagedMemViewPass] Non-hyperrectangular "
                      "subscripts - creating iter_args\n");
        createIterArgsForConditionals(info, subscripts_map, indices);
        info.indices_ = indices;
        insert_refs = mem_ops_;  // Loop bodies may have been cloned.
      }
      LLVM_DEBUG(llvm::dbgs()
                 << "[TransformPagedMemViewPass] Non-hyperrectangular "
                    "subscripts - creating conditionals\n");
      createConditionsForNonHyperRectSubscripts(insert_refs, subscripts_map,
                                                info.conditional_iter_args_,
                                                page_set_constraints);
    }
  }

  SmallVector<int64_t, 16> start_elements =
      calculateStartElementsForPage(page_set_constraints);

  AffineMap new_subscripts_map =
      createNewSubscriptsFromStartElements(subscripts_map, start_elements);

  // Create the new load/store operations and update uses. If multiple memory
  // operands were paged, there may be multiple copies of the memory op (tracked
  // by mem_ops_) and all of them need to be
  // Note: Creates a new operation instead of cloning as there may be less
  // operands if indices were changed (for example, if ConstantOps were
  // removed from the indices).
  OpBuilder builder(mem_ops_[0]);
  for (auto [insert_ref, mem_op] : zip(insert_refs, mem_ops_)) {
    setBuilderToInsertRef(builder, insert_ref);
    auto mem_view =
        createNonPagedMemView(builder, info.paged_mem_view_, page_idx);
    auto mem_view_val = mem_view.getResult();
    new_mem_ops.emplace_back(createNewMemOp(builder, mem_op, info, mem_view_val,
                                            new_subscripts_map, indices));

}}
// ---- 310/384  analyzeValidPages  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:888  (33L)
LogicalResult e310_analyzeValidPages(TPMVInfo &info,
                                               AffineMap &subscripts_map_sym) {
  FlatLinearValueConstraints compare_constraints;
  bool do_compare = false;
  auto idx_sets = info.paged_mem_view_.getIdxSets();
  int valid_pages = 0;
  for (int page_idx = 0; page_idx < idx_sets.size(); ++page_idx) {
    auto idx_set = idx_sets[page_idx];
    DT_CHECK(isa<IntegerSetAttr>(idx_set));
    auto page_set = cast<IntegerSetAttr>(idx_set).getValue();
    DT_CHECK(page_set.getNumDims() == info.subscripts_map_.getNumResults());
    FlatLinearValueConstraints page_sel_constraints =
        getPageValidity(info, subscripts_map_sym, page_set);

    if (page_sel_constraints.isEmpty()) continue;

    ++valid_pages;

    if (do_compare) {
      gatherPageDependentDimsForPage(page_sel_constraints, compare_constraints);
    } else {
      compare_constraints = page_sel_constraints;
      do_compare = true;
    }
  }

  if (subscripts_map_sym.getNumSymbols() == 0 && valid_pages > 1)
    return info.paged_mem_view_->emitOpError(
        "expecting only one active page when subscripts are constant");

  if (valid_pages == 0)
    return info.paged_mem_view_->emitOpError("no valid pages found");



}
// ==================================================================================================
// LEVEL 5
// ==================================================================================================

// ---- 311/384  constructAffineCompDetailsAndAddrs  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2809  (33L)
LogicalResult e311_constructAffineCompDetailsAndAddrs(
    Operation* src_op, Operation* dst_op, ProgramUnitOp& unit,
    SenComponents comp,
    AccessContainer<AccessDetailsAffineComposite>& access_details,
    bool has_ind_src, bool has_ind_dst, AccessContainer<Value>& mutable_addrs,
    AccessContainer<Value>& immutable_addrs) {
  DT_CHECK(src_op);
  AccessDetailsAffineComposite& src_ad =
      access_details.emplace_insert(MemoryOperandIndex::kDirSrc, src_op, comp);
  if (src_ad.constructDetails(MemoryOperandIndex::kDirSrc).failed())
    return src_op->emitError("unable to construct details for src");

  if (dst_op) {
    AccessDetailsAffineComposite& dst_ad = access_details.emplace_insert(
        MemoryOperandIndex::kDirDst, dst_op, comp);
    if (dst_ad.constructDetails(MemoryOperandIndex::kDirDst).failed())
      return dst_op->emitError("unable to construct details for dst");
  }

  if (has_ind_src) {
    AccessDetailsAffineComposite& ind_src_ad = access_details.emplace_insert(
        MemoryOperandIndex::kIndSrc, src_op, comp);
    if (ind_src_ad.constructDetails(MemoryOperandIndex::kIndSrc).failed())
      return src_op->emitError("unable to construct details for indirect src");
  }
  if (has_ind_dst) {
    AccessDetailsAffineComposite& ind_dst_ad = access_details.emplace_insert(
        MemoryOperandIndex::kIndDst, dst_op, comp);
    if (ind_dst_ad.constructDetails(MemoryOperandIndex::kIndDst).failed())
      return dst_op->emitError("unable to construct details for indirect src");
  }

  if (AccessDetailsAffineComposite::constructTimeStepsInfo(access_details)

}
// ---- 312/384  lowerExtractVectorLoadOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3001  (21L)
LogicalResult e312_lowerExtractVectorLoadOp(
    VectorLoadOp& op, ProgramUnitOp& unit, SenComponents comp,
    unsigned& extract_idx, SmallVectorImpl<Operation*>& to_be_deleted) {
  AccessContainer<AccessDetailsAffine> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructAffineDetailsAndAddrs(op, nullptr, unit, comp, access_details,
                                     mutable_addrs, immutable_addrs)
          .failed())
    return failure();
  DT_CHECK(access_details.size() == 1 && mutable_addrs.size() == 1 &&
           immutable_addrs.size() == 1);

  auto candidate_op = findCandidateForLowering<VectorLoadOp>(unit);

  if (constructLoadAndExtractScalarOp(
          candidate_op, unit, comp, access_details[0], mutable_addrs[0],
          immutable_addrs[0], extract_idx, to_be_deleted)
          .failed())
    return candidate_op->emitError(
        "Unable to generate load_and_extract_scalar operation for the "
        "agen.vector_load op");

}
// ---- 313/384  lowerExtractVectorStoreOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3026  (20L)
LogicalResult e313_lowerExtractVectorStoreOp(
    VectorStoreOp& op, ProgramUnitOp& unit, SenComponents comp,
    unsigned& extract_idx, SmallVectorImpl<Operation*>& to_be_deleted) {
  AccessContainer<AccessDetailsAffine> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructAffineDetailsAndAddrs(op, nullptr, unit, comp, access_details,
                                     mutable_addrs, immutable_addrs)
          .failed())
    return failure();
  DT_CHECK(access_details.size() == 1 && mutable_addrs.size() == 1 &&
           immutable_addrs.size() == 1);

  auto candidate_op = findCandidateForLowering<VectorStoreOp>(unit);

  if (constructReceiveAndExtractScalarOp(candidate_op, extract_idx,
                                         to_be_deleted)
          .failed())
    return candidate_op->emitError(
        "Unable to generate load_and_extract_scalar operation for the "
        "agen.vector_load op");

}
// ---- 314/384  lowerVectorLoadOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3050  (20L)
LogicalResult e314_lowerVectorLoadOp(
    VectorLoadOp& op, ProgramUnitOp& unit, SenComponents comp,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  auto store_op = getStoreOpFromLoadStorePattern<VectorStoreOp>(op);
  bool is_load_store = store_op != nullptr;

  AccessContainer<AccessDetailsAffine> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructAffineDetailsAndAddrs(op, store_op, unit, comp, access_details,
                                     mutable_addrs, immutable_addrs)
          .failed())
    return failure();

  auto candidate_op = findCandidateForLowering<VectorLoadOp>(unit);

  // Operations may have been deleted/cloned so store op needs to be
  // recollected.
  store_op = getStoreOpFromLoadStorePattern<VectorStoreOp>(candidate_op);

  return lowerVectorLoadHelper<AccessDetailsAffine, VectorLoadOp>(

}
// ---- 315/384  lowerVectorStoreOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3074  (28L)
LogicalResult e315_lowerVectorStoreOp(
    VectorStoreOp& op, ProgramUnitOp& unit, SenComponents comp,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  AccessContainer<AccessDetailsAffine> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructAffineDetailsAndAddrs(op, nullptr, unit, comp, access_details,
                                     mutable_addrs, immutable_addrs)
          .failed())
    return failure();
  DT_CHECK(access_details.size() == 1 && mutable_addrs.size() == 1 &&
           immutable_addrs.size() == 1);

  auto candidate_op = findCandidateForLowering<VectorStoreOp>(unit);

  OpBuilder builder(candidate_op);
  Type element_type = candidate_op.getMemRef().getType().getElementType();
  if (constructReceiveAndStoreStmt<AccessDetailsAffine>(
          &builder, unit, candidate_op, element_type, access_details[0],
          mutable_addrs[0], immutable_addrs[0])
          .failed())
    return candidate_op->emitError(
        "Unable to generate receive_and_store statement for the "
        "agen.vector_store operation");

  to_be_deleted.push_back(candidate_op);

  Operation* input_op = candidate_op.getValueToStore().getDefiningOp();
  addStoreInputToDeleteList(input_op, to_be_deleted);

}
// ---- 316/384  lowerIndirectVectorLoadOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3169  (44L)
LogicalResult e316_lowerIndirectVectorLoadOp(
    IndirectVectorLoadOp& op, ProgramUnitOp& unit, SenComponents comp,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  if (comp != SenComponents::LXLU)
    return op->emitError("IndirectVectorLoadOp only supported in LXLU");

  AccessContainer<AccessDetailsAffine> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructAffineDetailsAndAddrs(op, nullptr, unit, comp, access_details,
                                     mutable_addrs, immutable_addrs)
          .failed())
    return failure();
  DT_CHECK(access_details.size() == 1 && mutable_addrs.size() == 1 &&
           immutable_addrs.size() == 1);

  auto candidate_op = findCandidateForLowering<IndirectVectorLoadOp>(unit);

  // When load_and_extract_scalar operations are lowered, they assign
  // indices to the indirect load they feed.
  if (!candidate_op->hasAttr("extract_idx"))
    return candidate_op->emitError(
        "indirect_vector_load operations should have an extract_idx "
        "attribute");

  // Locate the extract op that matches the extract_idx.
  auto extract_op =
      findExtractScalarOp<LoadAndExtractScalarOp>(candidate_op, unit);
  if (!extract_op)
    return candidate_op->emitError(
        "could not locate a load_and_extract_scalar operation matching "
        "the extract_idx used by op");

  OpBuilder builder(candidate_op);
  if (constructLoadAndSendStmt<AccessDetailsAffine>(
          &builder, unit, candidate_op, access_details[0], mutable_addrs[0],
          immutable_addrs[0], extract_op)
          .failed())
    return candidate_op->emitError(
        "Unable to generate load_and_send '"
        "statement for the agen.indirect_vector_load operation");

  addLoadChainToDeleteList(candidate_op, to_be_deleted);

  to_be_deleted.push_back(candidate_op);

}
// ---- 317/384  lowerIndirectVectorStoreOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3217  (46L)
LogicalResult e317_lowerIndirectVectorStoreOp(
    IndirectVectorStoreOp& op, ProgramUnitOp& unit, SenComponents comp,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  if (comp != SenComponents::LXSU)
    return op->emitError("IndirectVectorStoreOp only supported in LXSU");

  AccessContainer<AccessDetailsAffine> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructAffineDetailsAndAddrs(op, nullptr, unit, comp, access_details,
                                     mutable_addrs, immutable_addrs)
          .failed())
    return failure();
  DT_CHECK(access_details.size() == 1 && mutable_addrs.size() == 1 &&
           immutable_addrs.size() == 1);

  auto candidate_op = findCandidateForLowering<IndirectVectorStoreOp>(unit);

  // When load_and_extract_scalar operations are lowered, they assign
  // indices to the indirect load they feed.
  if (!candidate_op->hasAttr("extract_idx"))
    return candidate_op->emitError(
        "indirect_vector_store operations should have an extract_idx "
        "attribute");

  // Locate the extract op that matches the extract_idx.
  auto extract_op =
      findExtractScalarOp<ReceiveAndExtractScalarOp>(candidate_op, unit);
  if (!extract_op)
    return candidate_op->emitError(
        "could not locate a receive_and_extract_scalar operation matching "
        "the extract_idx used by op");

  OpBuilder builder(candidate_op);
  Type element_type = candidate_op.getDirectMemrefType().getElementType();
  if (constructReceiveAndStoreStmt<AccessDetailsAffine>(
          &builder, unit, candidate_op, element_type, access_details[0],
          mutable_addrs[0], immutable_addrs[0], extract_op)
          .failed())
    return candidate_op->emitError(
        "Unable to generate receive_and_store '"
        "statement for the agen.indirect_vector_store operation");

  to_be_deleted.push_back(candidate_op);

  Operation* input_op = candidate_op.getValue().getDefiningOp();
  addStoreInputToDeleteList(input_op, to_be_deleted);

}
// ---- 318/384  lowerLDCVTIPattern  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3444  (326L)
LogicalResult lowerLDCVTIPattern(AgenToSentientLoweringPass& pass,
                                 vectorchain::BinaryOp& bin_op,
                                 ProgramUnitOp& unit_op, SenComponents comp,
                                 SmallVectorImpl<Operation*>& to_be_deleted) {
  if (!bin_op->hasOneUse())
    return bin_op->emitError(
        "[lowerLDCVTIPattern] MultiplyOp expected to have one use");
  auto send_op = dyn_cast<dataflow::SendOp>(*bin_op->getUsers().begin());

  // Mark the send op so we can find it after loops are reconstructed later.
  {
    OpBuilder builder(send_op);
    send_op->setAttr("ldcvti_send_op", builder.getI8IntegerAttr(1));
  }

  // Binary op should have two vectorchain::ShuffleOp operands, either no mask
  // operand or an empty set, and an identity op_specific_map (reduction map).
  auto operand_1 =
      dyn_cast_or_null<vectorchain::ShuffleOp>(bin_op.getOp1().getDefiningOp());
  auto operand_2 =
      dyn_cast_or_null<vectorchain::ShuffleOp>(bin_op.getOp2().getDefiningOp());
  if (!operand_1 || !operand_2)
    return bin_op->emitError(
        "[lowerLDCVTIPattern] BinaryOp expected to have two ShuffleOp "
        "operands");
  if (!bin_op.getOpSpecificMap().isIdentity())
    return bin_op->emitError(
        "[lowerLDCVTIPattern] BinaryOp expected to have identity reduction "
        "map");
  // The mask operand should indicate no masking.
  auto mask_op = dyn_cast_or_null<vectorchain::CreateAffineMaskOp>(
      bin_op.getMask().getDefiningOp());
  if (!mask_op)
    return bin_op->emitError(
        "[lowerLDCVTIPattern] BinaryOp expected to have a mask operand");
  DT_CHECK(mask_op->hasAttr("mask_set"));
  auto mask_set = cast<IntegerSetAttr>(mask_op->getAttr("mask_set")).getValue();
  // Use FlatLinearValueConstraints to check if the constraints are
  // contradictory
  FlatLinearValueConstraints constraints(mask_set);
  if (!constraints.isEmpty())
    return bin_op->emitError(
        "[lowerLDCVTIPattern] BinaryOp mask should not mask anything");

  // Mark the binary op so we can find it after loops are reconstructed later.
  {
    OpBuilder builder(bin_op);
    bin_op->setAttr("ldcvti_bin_op", builder.getI8IntegerAttr(1));
  }

  // Both shuffles should have:
  //   - one constant variable operand
  //   - no pad or mask operands
  //   - an indices vector containing only a splat of that constant variable
  auto checkShuffle = [&](vectorchain::ShuffleOp shuffle) -> bool {
    if (!shuffle.getPad().empty() || shuffle.getMask()) return false;
    if (shuffle.getVariable().size() != 1) return false;
    auto var_op = isa_and_nonnull<arith::ConstantOp>(
        shuffle.getVariable()[0].getDefiningOp());
    if (!var_op) return false;
    if (!vectorchain::utils::isSplatOfFirstVar(shuffle)) return false;
    return true;
  };
  if (!checkShuffle(operand_1) || !checkShuffle(operand_2))
    return bin_op->emitError(
        "[lowerLDCVTIPattern] Inputs to BinaryOp do not match LDCVTI pattern");

  // One of the shuffles should have:
  //    - a vector_load input that loads from scale register, with all 0
  //      subscripts, and a start address of 0
  // The other shuffle should have either:
  //   - a vector_load input that loads from LX
  //   - a vectorchain::ShuffleOp input that has a vector_load input that loads
  //     from LX (this indicates non-default ldtype)
  //     - future plan
  vectorchain::ShuffleOp ldtype_shuffle = nullptr;
  auto identifyLoads = [&](vectorchain::ShuffleOp shuffle,
                           agen::VectorLoadOp& load_op) -> bool {
    auto input_op = shuffle.getInput().getDefiningOp();
    if (!input_op) return false;
    load_op = dyn_cast<agen::VectorLoadOp>(input_op);
    if (!load_op) {
      DT_CHECK_MSG(!ldtype_shuffle,
                   "[lowerLDCVTIPattern] ldtype already set! Only element load "
                   "can have ldtype.");
      ldtype_shuffle = dyn_cast<vectorchain::ShuffleOp>(input_op);
      if (!ldtype_shuffle) return false;
      load_op = dyn_cast_or_null<agen::VectorLoadOp>(
          ldtype_shuffle.getInput().getDefiningOp());
      if (!load_op) return false;
    }
    return true;
  };
  DT_CHECK_MSG(
      !ldtype_shuffle,
      "[lowerLDCVTIPattern] ldtype for LDCVTI not currently supported");

  agen::VectorLoadOp load_op_1 = nullptr, load_op_2 = nullptr;
  if (!identifyLoads(operand_1, load_op_1) ||
      !identifyLoads(operand_2, load_op_2))
    return bin_op.emitOpError(
        "[lowerLDCVTIPattern] Unable to identify element and scale load "
        "operations");
  DT_CHECK(load_op_1 && load_op_2);
  ExpressionEvaluator evaluator;
  agen::VectorLoadOp scale_load_op = nullptr;
  auto setScaleLoadIfFound = [&](agen::VectorLoadOp load_op) {
    if (scale_load_op) return;
    // Memview should be for LX scale register
    auto mem_view = dyn_cast<dataflow::GetLogicalMemoryViewOp>(
        load_op.getMemRef().getDefiningOp());
    DT_CHECK(mem_view);
    auto mem_view_comp =
        dcc::getUnitType(mem_view.getFromUnit().getDefiningOp());
    if (mem_view_comp != LXLUSCALEREG) return;

    // Memview start address should be 0
    const EvaluatedValue& start_addr_ev =
        evaluator.evaluateValue(mem_view.getStartAddress());
    EvaluatedValue::ScalarValue const_val;
    if (!start_addr_ev.getUniqueConstant(const_val) || const_val != 0) return;

    SmallVector<Value> indices;
    for (auto index : load_op.getMapOperands()) indices.push_back(index);
    auto coeffs = mlir::agen::utils::constructIteratorCoeffDict(
        load_op.getAffineMap(), load_op.getLoadOrder(), mem_view.getLayoutMap(),
        indices);
    for (auto& [val, coeff] : coeffs) {
      if (coeff != 0) return;
    }

    scale_load_op = load_op;
  };

  agen::VectorLoadOp element_load_op = nullptr;
  auto setElementLoadIfFound = [&](agen::VectorLoadOp load_op) {
    if (element_load_op) return;
    // Memview should be for LX
    auto mem_view = dyn_cast<dataflow::GetLogicalMemoryViewOp>(
        load_op.getMemRef().getDefiningOp());
    DT_CHECK(mem_view);
    auto mem_view_comp =
        dcc::getUnitType(mem_view.getFromUnit().getDefiningOp());
    if (mem_view_comp != LX) return;

    element_load_op = load_op;
  };

  // Set element and scale loads.
  setElementLoadIfFound(load_op_1);
  setElementLoadIfFound(load_op_2);
  setScaleLoadIfFound(load_op_1);
  setScaleLoadIfFound(load_op_2);
  DT_CHECK(element_load_op && scale_load_op);

  // Mark the load ops to find them later after loops are potentially
  // reconstructed.
  {
    OpBuilder builder(scale_load_op);
    scale_load_op->setAttr("ldcvti_scale_load_op", builder.getI8IntegerAttr(1));
    element_load_op->setAttr("ldcvti_element_load_op",
                             builder.getI8IntegerAttr(1));
  }

  // Determine which shuffle op corresponds to element and scale.
  vectorchain::ShuffleOp scale_shuffle_op = nullptr,
                         element_shuffle_op = nullptr;
  if (scale_load_op == load_op_1) {
    scale_shuffle_op = operand_1;
    element_shuffle_op = operand_2;
  } else {
    scale_shuffle_op = operand_2;
    element_shuffle_op = operand_1;
  }
  DT_CHECK(scale_shuffle_op && element_shuffle_op);

  // Mark the shuffle ops so we can find them after loops are reconstructed
  // later.
  {
    OpBuilder builder(scale_shuffle_op);
    scale_shuffle_op->setAttr("ldcvti_scale_shuffle_op",
                              builder.getI8IntegerAttr(1));
    element_shuffle_op->setAttr("ldcvti_element_shuffle_op",
                                builder.getI8IntegerAttr(1));
  }

  // TODO: Check that load_set/order match expected shapes.

  // Construct the access details including the mutable and immutable addresses
  // for the load_compute_and_send. Note: This will potentially reconstruct
  // loops in the IR. Expect previous references to be invalidated at this
  // point.
  AccessContainer<AccessDetailsAffine> element_ad;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (pass.constructAffineDetailsAndAddrs(element_load_op, nullptr, unit_op,
                                          comp, element_ad, mutable_addrs,
                                          immutable_addrs)
          .failed())
    return element_load_op.emitOpError(
        "[lowerLDCVTIPattern] Cannot set access details for element load op");
  DT_CHECK(element_ad.size() == 1 && mutable_addrs.size() == 1 &&
           immutable_addrs.size() == 1);

  // Find the ops forming the pattern since the original operations may have
  // been deleted.
  auto findOpWithAttribute = [&](StringRef attr_name) -> Operation* {
    Operation* found_op = nullptr;
    unit_op.walk([&](Operation* op) {
      if (op->hasAttr(attr_name)) {
        found_op = op;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    return found_op;
  };

  send_op = cast<dataflow::SendOp>(findOpWithAttribute("ldcvti_send_op"));
  bin_op = cast<vectorchain::BinaryOp>(findOpWithAttribute("ldcvti_bin_op"));
  scale_shuffle_op = cast<vectorchain::ShuffleOp>(
      findOpWithAttribute("ldcvti_scale_shuffle_op"));
  scale_load_op =
      cast<agen::VectorLoadOp>(findOpWithAttribute("ldcvti_scale_load_op"));
  element_shuffle_op = cast<vectorchain::ShuffleOp>(
      findOpWithAttribute("ldcvti_element_shuffle_op"));
  element_load_op =
      cast<agen::VectorLoadOp>(findOpWithAttribute("ldcvti_element_load_op"));
  DT_CHECK(send_op && bin_op && scale_shuffle_op && scale_load_op &&
           element_shuffle_op && element_load_op);

  // Create the sentient::ConstantOps for the scale and element indices to use
  // for the new operation.
  OpBuilder builder(unit_op);
  auto scale_const_op = dyn_cast<arith::ConstantOp>(
      scale_shuffle_op.getVariable()[0].getDefiningOp());
  DT_CHECK(scale_const_op);
  auto scale_index_attr = dyn_cast<IntegerAttr>(scale_const_op.getValue());
  DT_CHECK(scale_index_attr);
  int64_t scale_index_value = scale_index_attr.getInt();
  auto scale_index_op = sentient::ConstantOp::create(
      builder, bin_op->getLoc(), IndexType::get(bin_op->getContext()),
      scale_index_value);

  auto element_const_op = dyn_cast<arith::ConstantOp>(
      element_shuffle_op.getVariable()[0].getDefiningOp());
  DT_CHECK(element_const_op);
  auto element_index_attr = dyn_cast<IntegerAttr>(element_const_op.getValue());
  DT_CHECK(element_index_attr);
  int64_t element_index_value = element_index_attr.getInt();
  auto element_index_op = sentient::ConstantOp::create(
      builder, bin_op->getLoc(), IndexType::get(bin_op->getContext()),
      element_index_value);

  // The consumer comes from the send op.
  auto consumer = send_op.getToUnit().getDefiningOp();
  DT_CHECK(consumer);

  // src_total_elements and src_element_width comes from the element load op.
  auto src_total_elements = element_load_op.getVectorType().getNumElements();
  auto src_element_width =
      element_load_op.getVectorType().getElementTypeBitWidth();

  // dst_total_elements and dst_element_width comes from the binary op.
  auto bin_op_res_type = dyn_cast<VectorType>(bin_op.getResult().getType());
  DT_CHECK(bin_op_res_type);
  auto dst_total_elements = bin_op_res_type.getNumElements();
  auto dst_element_width = bin_op_res_type.getElementTypeBitWidth();

  auto shuffle_mode =
      symbolizeSentientShuffleMode(element_ad[0].getShuffleMode()).value();
  DT_CHECK(shuffle_mode == SentientShuffleMode::noshuffle);

  // Update immutable and increment
  Value actual_immutable_addr = nullptr, increment = nullptr;
  if (failed(pass.setImmutableAddrAndIncrements(
          builder, comp, element_load_op, false, 0, 0, src_total_elements,
          immutable_addrs[0], actual_immutable_addr, increment, unit_op)))
    return element_load_op->emitOpError(
        "[lowerLDCVTIPattern] Could not set immutable_addr and increment");

  const EvaluatedValue& increment_ev = evaluator.evaluateValue(increment);
  EvaluatedValue::ScalarValue increment_val;
  if (!increment_ev.getUniqueConstant(increment_val) || increment_val != 0)
    return element_load_op.emitError(
        "[lowerLDCVTIPattern] Expected increment to be 0");

  // The resulting load_compute_and_send operation will use scale shuffle to
  // determine the scale index, element shuffle to determine the element index,
  // and the element load to determine the address to load the stick to operate
  // on. If ldtype_shuffle existed, it is also used to determine the ldtype for
  // the element load (future work). Create the sentient.load_compute_and_send
  // operation.
  builder.setInsertionPointAfter(send_op);

  // Emit SetSendDestinationOp so that the downstream SentientToProgIR pass
  // can generate LX_SETDSTMASK with the correct mode (e.g. mode=pt).
  // The conventional load path does this via generateSetSendDestinationStmts;
  // we must do the same here because lowerLDCVTIPattern bypasses that path.
  if (failed(pass.generateSetSendDestinationStmts(builder, unit_op,
                                                  element_load_op, consumer)))
    return element_load_op->emitOpError(
        "[lowerLDCVTIPattern] Failed to generate set_send_destination op");

  auto load_compute_and_send_op = sentient::LoadComputeAndSendOp::create(
      builder, bin_op->getLoc(), IndexType::get(bin_op->getContext()),
      mutable_addrs[0], actual_immutable_addr, increment,
      element_index_op.getOut(), scale_index_op.getOut(),
      consumer->getResult(0),
      element_load_op.getDbgNameAttr(),  // dbgName
      src_total_elements,                // src_total_elements
      dst_total_elements,                // dst_total_elements
      src_element_width,                 // src_element_size
      dst_element_width,                 // dst_element_size
      nullptr,                           // dir
      shuffle_mode,                      // shuffle_mode
      SentientRegType::unknown,          // regLocale
      -1);                               // regIndex

  // Mark operations for deletion. Do this here to avoid memory movement during
  // above data collection for the operation.
  // TODO: When ldtype is supported, ensure the ldtype related shuffles are
  // removed.
  to_be_deleted.push_back(send_op);
  to_be_deleted.push_back(bin_op);
  to_be_deleted.push_back(scale_shuffle_op);
  to_be_deleted.push_back(scale_load_op);

}
// ---- 319/384  lowerSyncForAQueryMap  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1728  (166L)
LogicalResult e319_lowerSyncForAQueryMap(
    std::string src_unit_name, mlir::Operation *op,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops_corelet0,
    std::vector<mlir::dataflow::GetUnitOp> src_unit_ops_corelet1,
    uniform::QueryMapOp query_map) {
  OpBuilder builder(op);
  auto send_op = llvm::dyn_cast<dataflow::SyncSendOp>(op);
  auto recv_op = llvm::dyn_cast<dataflow::SyncRecvOp>(op);
  auto implicit_sync_op =
      llvm::dyn_cast<dataflow::ImplicitSyncOnStreamingBufferOp>(op);
  DT_CHECK(send_op || recv_op || implicit_sync_op);
  SentientSyncMode sync_tag = SentientSyncMode::send;
  if (recv_op) sync_tag = SentientSyncMode::recv;

  int implicit_sync_tile_size = -1;
  if (implicit_sync_op) {
    sync_tag = SentientSyncMode::sendrecv;
    auto parent_op = implicit_sync_op.getBufferSize().getDefiningOp();
    if (auto buffer_size_arith_def_op =
            llvm::dyn_cast<arith::ConstantIndexOp>(parent_op)) {
      implicit_sync_tile_size = buffer_size_arith_def_op.value();
    } else if (auto buffer_size_sen_def_op =
                   llvm::dyn_cast<sentient::ConstantOp>(parent_op)) {
      implicit_sync_tile_size = buffer_size_sen_def_op.getValue();
    } else {
      DT_ERROR("sync buffer size has to be a constant op");
    }
  }

  auto dbg_name_attr = getDbgNameAttr(op);

  if (send_op && !send_op.getWaitImmediatelyForAsyncTransfers().has_value()) {
    op->emitError("Unknown async transfers modes for sentient");
    return LogicalResult::failure();
  }
  std::vector<mlir::Value> src_vs =
      dcc::uniform::utils::getListOfKeyOpsFromUniformMapping(query_map);
  std::vector<mlir::Value> dst_vs =
      dcc::uniform::utils::getListOfValueOpsFromUniformMapping(query_map);

  if (src_unit_name.substr(0, 2) != "l3") {
    // LX/L0
    std::vector<mlir::Value> src_corelet0_vs, dst_with_src_corelet0_vs,
        src_corelet1_vs, dst_with_src_corelet1_vs;
    dcc::uniform::utils::separateKeyValuesBasedOnKeysCorelet(
        query_map, src_corelet0_vs, dst_with_src_corelet0_vs, src_corelet1_vs,
        dst_with_src_corelet1_vs);
    SenComponents src_comp =
        EnumsConversion::stringToSenComponents.find(src_unit_name)->second;
    auto gen_comp = EnumsConversion::senCompToGenericComp.at(src_comp);
    DT_CHECK(src_corelet0_vs.size() + src_corelet1_vs.size() != 0);

    switch (gen_comp) {
      case SenComponents::L0LU:
      case SenComponents::L0SU: {
        std::string dst_unit_name;
        for (int i = 0; i < dst_vs.size(); i++) {
          auto target_unit =
              llvm::dyn_cast<dataflow::GetUnitOp>(dst_vs[i].getDefiningOp());
          if (!target_unit) {
            auto target_group = llvm::dyn_cast<dataflow::CreateGroupOp>(
                dst_vs[i].getDefiningOp());
            DT_CHECK_MSG(
                target_group.getUnitIds().size() == 1,
                "For L0 units the target cannot be a group of multiple "
                "units");
            target_unit = llvm::dyn_cast<dataflow::GetUnitOp>(
                target_group.getUnitIds()[0].getDefiningOp());
            DT_CHECK_MSG(target_unit,
                         "For L0 units the target has to be a single unit.");
          }
          SenComponents dst_comp = EnumsConversion::stringToSenComponents
                                       .find(target_unit.getType().str())
                                       ->second;
          if (!((isSenComponentL0LU(src_comp) &&
                 isSenComponentL0SU(dst_comp)) ||
                (isSenComponentL0SU(src_comp) &&
                 isSenComponentL0LU(dst_comp)))) {
            op->emitError("Unsupported dst unit for L0 unit.");
            return LogicalResult::failure();
          }
          if (isSenComponentL0LU(dst_comp)) dst_unit_name = "l0lu";
          if (isSenComponentL0SU(dst_comp)) dst_unit_name = "l0su";
        }
        if (sentient::SyncOp::create(
                builder, op->getLoc(), dbg_name_attr,
                SentientSyncModeAttr::get(builder.getContext(), sync_tag),
                ArrayAttr::get(
                    builder.getContext(),
                    {SentientLoadConsumerAttr::get(
                        op->getContext(),
                        symbolizeSentientLoadConsumer(dst_unit_name).value())}),
                builder.getBoolAttr(false),
                builder.getSI32IntegerAttr(implicit_sync_tile_size))) {
          return LogicalResult::success();
        }
      }
      case SenComponents::LXLU:
      case SenComponents::LXSU: {
        DT_CHECK_MSG(implicit_sync_tile_size == -1,
                     "LX doesn't have implicit sync");
        if (src_corelet1_vs.size() == 0) {
          return lowerSyncLXL3ToLXL3(op, builder, src_corelet0_vs,
                                     dst_with_src_corelet0_vs, 0,
                                     /*is_src_l3 =*/false);
        } else if (src_corelet0_vs.size() == 0) {
          return lowerSyncLXL3ToLXL3(op, builder, src_corelet1_vs,
                                     dst_with_src_corelet1_vs, 1,
                                     /*is_src_l3 =*/false);
        } else if (isTargetL3(query_map)) {
          std::vector<mlir::Value> src_unit_res, dst_unit_res;
          for (auto u : src_corelet0_vs) src_unit_res.push_back(u);
          for (auto u : dst_with_src_corelet0_vs) dst_unit_res.push_back(u);
          for (auto u : src_corelet1_vs) src_unit_res.push_back(u);
          for (auto u : dst_with_src_corelet1_vs) dst_unit_res.push_back(u);
          return lowerSyncLXL3ToLXL3(op, builder, src_unit_res, dst_unit_res,
                                     -1,
                                     /*is_src_l3 =*/false);
        } else {
          std::vector<mlir::Value> src_unit_res;
          for (auto u : src_corelet0_vs) src_unit_res.push_back(u);
          for (auto u : src_corelet1_vs) src_unit_res.push_back(u);
          llvm::SmallVector<mlir::Attribute, 2> list_sizes;
          list_sizes.push_back(
              builder.getI32IntegerAttr(src_corelet0_vs.size()));
          list_sizes.push_back(
              builder.getI32IntegerAttr(src_corelet1_vs.size()));
          auto uniform_region = mlir::uniform::UniformizeRegionsOp::create(
              builder, op->getLoc(), mlir::TypeRange(), src_unit_res,
              ArrayAttr::get(builder.getContext(), list_sizes),
              /*regIndices*/ nullptr, /*regLocales*/ nullptr,
              /*num_of_regions = */ 2);
          auto &block0 = uniform_region.getRegion(0).emplaceBlock();
          block0.addArgument(builder.getIndexType(), op->getLoc());
          // block.getArgument()
          OpBuilder builder_region0(uniform_region.getRegion(0));
          if (lowerSyncLXL3ToLXL3(op, builder_region0, src_corelet0_vs,
                                  dst_with_src_corelet0_vs,
                                  /*corelet_id =*/0,
                                  /*is_src_l3 =*/false)
                  .failed())
            return LogicalResult::failure();
          uniform::YieldOp::create(builder_region0, op->getLoc());
          auto &block1 = uniform_region.getRegion(1).emplaceBlock();
          block1.addArgument(builder.getIndexType(), op->getLoc());
          OpBuilder builder_region1(uniform_region.getRegion(1));
          if (lowerSyncLXL3ToLXL3(op, builder_region1, src_corelet1_vs,
                                  dst_with_src_corelet1_vs,
                                  /*corelet_id =*/1,
                                  /*is_src_l3 =*/false)
                  .failed())
            return LogicalResult::failure();
          uniform::YieldOp::create(builder_region1, op->getLoc());
          // auto flattened_uniform_region =
          //     dcc::uniform::utils::flattenUniformRegion(uniform_region);
          // dcc::uniform::utils::simplifyUniformRegions(flattened_uniform_region);
          return LogicalResult::success();
        }
      }
      default:
        op->emitError("Unknown lowering of the sync operation");
        return LogicalResult::failure();
    }
  } else {
    // L3

}}
// ---- 320/384  getOperand  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:378  (4L)
std::optional<VectorOperand> e320_getOperand(
    const dcc::DccExtContext &dcc_ext_ctx, Operation *op,
    const SenComponents comp, bool traverse_upwards) {
  bool is_precision_converted;

}
// ---- 321/384  transformCompLoadAndStore  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:451  (92L)
void e321_transformCompLoadAndStore(
    MASCandidate &candidate) {
  auto op = dyn_cast<agen::CompositeLoadAndStoreOp>(candidate.op_);
  DT_CHECK(op);

  agen::AccessContainer<agen::AccessDetailsAffineComposite> access_details;
  agen::AccessDetailsAffineComposite &ad =
      access_details.emplace_insert(candidate.mem_index_, op, candidate.comp_);
  bool result = ad.constructDetails(candidate.mem_index_).succeeded();

  // Construct the time steps info without coalescing and without burst/IL
  // calcs.
  result &= agen::AccessDetailsAffineComposite::constructTimeStepsInfo(
                access_details, false, false)
                .succeeded();
  DT_CHECK(result);

  auto mem_view_op = dyn_cast_or_null<dataflow::GetLogicalMemoryViewOp>(
      ad.getMemRef().getDefiningOp());
  DT_CHECK(mem_view_op);

  ExpressionEvaluator evaluator;
  SmallVector<MASData> mas_data;
  int64_t max_mutable = 0;
  initialize(evaluator, mas_data, mem_view_op, ad, max_mutable);
  synthesizeTimeInfo(mas_data, ad, max_mutable);
  if (!hasMutableAddrOverflow(mas_data, candidate.comp_, max_mutable,
                              ad.getElementWidth()))
    return;

  SmallVector<Value> all_mem_views = {op.getSrcMemRef(), op.getDstMemRef()};

  SmallVector<int64_t> partition_sizes;
  setupForPartitioning(evaluator, mas_data, partition_sizes, all_mem_views,
                       candidate.comp_, mem_view_op, max_mutable,
                       ad.getElementWidth());

  auto subscripts_map = ad.getSubscriptsMap();
  auto indices = ad.getIndices();
  auto time_set = ad.getTimeSet();
  createExplicitTimeLoops(mas_data, partition_sizes, op, ad, subscripts_map,
                          indices, time_set);

  auto createOps = [&](OpBuilder &partition_builder,
                       AffineMap &new_subscripts_map, int64_t start_addr_mod) {
    adjustForEvenImmutableAddr(evaluator, mem_view_op.getStartAddress(),
                               new_subscripts_map, ad, start_addr_mod,
                               ad.getElementWidth());
    auto new_mem_view_op =
        createNewMemViewWithMod(evaluator, partition_builder, candidate.unit_,
                                op, mem_view_op, start_addr_mod);
    partition_builder.setInsertionPointAfter(new_mem_view_op);
    if (candidate.mem_index_ == agen::MemoryOperandIndex::kDirSrc) {
      // Every memory operand should have it's own unique mem view.
      // Clone the mem views for the other memory operands into the partition.
      auto new_dst_mem_view =
          partition_builder.clone(*op.getDstMemRef().getDefiningOp());
      op.getDstMemRefMutable().assign(new_dst_mem_view->getResult(0));

      SmallVector<Value, 16> dst_indices;
      for (auto index : op.getDstMapIndices()) dst_indices.push_back(index);
      auto new_mem_op = op.cloneWithNewAccessInfo(
          partition_builder, new_mem_view_op, op.getDstMemRef(),
          new_subscripts_map, op.getDstAffineMapAttr().getValue(), indices,
          dst_indices, time_set);
      partition_builder.setInsertionPoint(new_mem_op);
    } else {
      // Every memory operand should have it's own unique mem view.
      // Clone the mem views for the other memory operands into the partition.
      auto new_src_mem_view =
          partition_builder.clone(*op.getSrcMemRef().getDefiningOp());
      op.getSrcMemRefMutable().assign(new_src_mem_view->getResult(0));

      SmallVector<Value, 16> src_indices;
      for (auto index : op.getSrcMapIndices()) src_indices.push_back(index);
      auto new_mem_op = op.cloneWithNewAccessInfo(
          partition_builder, op.getSrcMemRef(), new_mem_view_op,
          op.getSrcAffineMapAttr().getValue(), new_subscripts_map, src_indices,
          indices, time_set);
      partition_builder.setInsertionPoint(new_mem_op);
    }
  };
  createPartitions(mas_data, partition_sizes, op, ad.getSubscriptsMap(),
                   createOps);

  op->erase();

  // TODO: Connect these information messages to DT_DEEPRT_VERBOSE.
  LLVM_DEBUG(llvm::dbgs()
             << "[INFO][DCC][MutableAddrSplitting] Mutable address split in "
             << dcc_ext_ctx_.prog_name_ << " in "
             << EnumsConversion::senComponentsToString.at(candidate.comp_)

}
// ---- 322/384  transformCompIndLoadAndStore  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:546  (124L)
void e322_transformCompIndLoadAndStore(
    MASCandidate &candidate) {
  auto op = dyn_cast<agen::CompositeIndirectLoadAndStoreOp>(candidate.op_);
  DT_CHECK(op);

  agen::AccessContainer<agen::AccessDetailsAffineComposite> access_details;
  agen::AccessDetailsAffineComposite &ad =
      access_details.emplace_insert(candidate.mem_index_, op, candidate.comp_);
  bool result = ad.constructDetails(candidate.mem_index_).succeeded();
  // Construct the time steps info without coalescing and without burst/IL
  // calcs.
  result &= agen::AccessDetailsAffineComposite::constructTimeStepsInfo(
                access_details, false, false)
                .succeeded();
  DT_CHECK(result);

  auto mem_view_op = dyn_cast_or_null<dataflow::GetLogicalMemoryViewOp>(
      ad.getMemRef().getDefiningOp());
  DT_CHECK(mem_view_op);

  ExpressionEvaluator evaluator;
  SmallVector<MASData> mas_data;
  int64_t max_mutable = 0;
  initialize(evaluator, mas_data, mem_view_op, ad, max_mutable);
  synthesizeTimeInfo(mas_data, ad, max_mutable);
  if (!hasMutableAddrOverflow(mas_data, candidate.comp_, max_mutable,
                              ad.getElementWidth()))
    return;

  // If the indirect part of the memory operation is for the same type of
  // operand as the memory operand triggering the transformation, the
  // operation is ineligible for transformation. The immutable address must be
  // zero for the side of the transfer involving the indirect.
  DT_CHECK(candidate.mem_index_ == agen::MemoryOperandIndex::kDirSrc
               ? !op.hasIndirectSrc()
               : !op.hasIndirectDst());

  SmallVector<Value> all_mem_views = {op.getDirectSrcMemref(),
                                      op.getDirectDstMemref()};
  if (op.hasIndirectSrc()) all_mem_views.push_back(op.getIndirectSrcMemref());
  if (op.hasIndirectDst()) all_mem_views.push_back(op.getIndirectDstMemref());

  SmallVector<int64_t> partition_sizes;
  setupForPartitioning(evaluator, mas_data, partition_sizes, all_mem_views,
                       candidate.comp_, mem_view_op, max_mutable,
                       ad.getElementWidth());

  auto subscripts_map = ad.getSubscriptsMap();
  auto indices = ad.getIndices();
  auto time_set = ad.getTimeSet();
  createExplicitTimeLoops(mas_data, partition_sizes, op, ad, subscripts_map,
                          indices, time_set);

  auto createOps = [&](OpBuilder &partition_builder,
                       AffineMap &new_subscripts_map, int64_t start_addr_mod) {
    adjustForEvenImmutableAddr(evaluator, mem_view_op.getStartAddress(),
                               new_subscripts_map, ad, start_addr_mod,
                               ad.getElementWidth());
    auto new_mem_view_op =
        createNewMemViewWithMod(evaluator, partition_builder, candidate.unit_,
                                op, mem_view_op, start_addr_mod);
    partition_builder.setInsertionPointAfter(new_mem_view_op);
    if (candidate.mem_index_ == agen::MemoryOperandIndex::kDirSrc) {
      // Every memory operand should have it's own unique mem view.
      // Clone the mem views for the other memory operands into the partition.
      auto new_dst_mem_view =
          partition_builder.clone(*op.getDirectDstMemref().getDefiningOp());
      op.getDirectDstMemrefMutable().assign(new_dst_mem_view->getResult(0));
      if (op.hasIndirectDst()) {
        auto new_ind_dst_mem_view =
            partition_builder.clone(*op.getIndirectDstMemref().getDefiningOp());
        op.getIndirectDstMemrefMutable().assign(
            new_ind_dst_mem_view->getResult(0));
      }

      SmallVector<Value, 16> dst_indices;
      for (auto index : op.getDirectDstMapIndices())
        dst_indices.push_back(index);
      auto new_mem_op = op.cloneWithNewAccessInfo(
          partition_builder, op.getIndirectSrcMemref(), new_mem_view_op,
          op.getIndirectDstMemref(), op.getDirectDstMemref(),
          partition_builder.getEmptyAffineMap(), new_subscripts_map,
          op.hasIndirectDst() ? op.getIndirectDstAffineMapAttr().getValue()
                              : partition_builder.getEmptyAffineMap(),
          op.getDirectDstAffineMapAttr().getValue(), indices, dst_indices,
          time_set);
      partition_builder.setInsertionPoint(new_mem_op);
    } else {
      // Every memory operand should have it's own unique mem view.
      // Clone the mem views for the other memory operands into the partition.
      auto new_src_mem_view =
          partition_builder.clone(*op.getDirectSrcMemref().getDefiningOp());
      op.getDirectSrcMemrefMutable().assign(new_src_mem_view->getResult(0));
      if (op.hasIndirectSrc()) {
        auto new_ind_src_mem_view =
            partition_builder.clone(*op.getIndirectSrcMemref().getDefiningOp());
        op.getIndirectSrcMemrefMutable().assign(
            new_ind_src_mem_view->getResult(0));
      }

      SmallVector<Value, 16> src_indices;
      for (auto index : op.getDirectSrcMapIndices())
        src_indices.push_back(index);
      auto new_mem_op = op.cloneWithNewAccessInfo(
          partition_builder, op.getIndirectSrcMemref(), op.getDirectSrcMemref(),
          op.getIndirectSrcMemref(), new_mem_view_op,
          op.hasIndirectSrc() ? op.getIndirectSrcAffineMapAttr().getValue()
                              : partition_builder.getEmptyAffineMap(),
          op.getDirectSrcAffineMapAttr().getValue(),
          partition_builder.getEmptyAffineMap(), new_subscripts_map,
          src_indices, indices, time_set);
      partition_builder.setInsertionPoint(new_mem_op);
    }
  };
  createPartitions(mas_data, partition_sizes, op, ad.getSubscriptsMap(),
                   createOps);

  op->erase();

  // TODO: Connect these information messages to DT_DEEPRT_VERBOSE.
  LLVM_DEBUG(llvm::dbgs()
             << "[INFO][DCC][MutableAddrSplitting] Mutable address split in "
             << dcc_ext_ctx_.prog_name_ << " in "
             << EnumsConversion::senComponentsToString.at(candidate.comp_)

}
// ---- 323/384  shiftMutableAddr  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:366  (19L)
AffineMap e323_shiftMutableAddr(
    dataflow::ProgramUnitOp unit, Operation *op,
    dataflow::GetLogicalMemoryViewOp mem_view_op,
    agen::AccessDetailsAffine &ad) {
  ExpressionEvaluator evaluator;
  DT_CHECK(dcc::agen::utils::hasValidL3ImmutableAddr(
      evaluator, mem_view_op,
      dcc_ext_ctx_.getBytesPerStick() * 8 / ad.getElementWidth(),
      /* req_even_toggle */ dcc_ext_ctx_.getArch() >= IsaCoreGen::SEN1P5_ISA
          ? true
          : false));

  SmallVector<int64_t> shifts;
  calculateShifts(evaluator, shifts, mem_view_op, ad);

  // If all the shifts are 0, there is nothing to shift. Return and report no
  // shift.
  if (std::all_of(shifts.begin(), shifts.end(),
                  [](int64_t s) { return s == 0; }))

}
// ---- 324/384  analyzeAndConstructValidPages  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:152  (32L)
LogicalResult e324_analyzeAndConstructValidPages(
    SmallVectorImpl<Operation *> &new_mem_ops, TPMVInfo &info,
    AffineMap &subscripts_map_sym) {
  DT_CHECK(new_mem_ops.empty());

  FlatLinearValueConstraints compare_constraints;
  auto idx_sets = info.paged_mem_view_.getIdxSets();
  int valid_pages = 0;
  for (int page_idx = 0; page_idx < idx_sets.size(); ++page_idx) {
    auto idx_set = idx_sets[page_idx];
    DT_CHECK(isa<IntegerSetAttr>(idx_set));
    auto page_set = cast<IntegerSetAttr>(idx_set).getValue();
    DT_CHECK(page_set.getNumDims() == info.subscripts_map_.getNumResults());
    FlatLinearValueConstraints page_sel_constraints =
        getPageValidity(info, subscripts_map_sym, page_set);

    if (page_sel_constraints.isEmpty()) continue;

    ++valid_pages;

    if (constructValidPage(new_mem_ops, info, subscripts_map_sym, page_set,
                           page_sel_constraints, page_idx)
            .failed())
      return LogicalResult::failure();
  }

  if (subscripts_map_sym.getNumSymbols() == 0 && valid_pages > 1)
    return info.paged_mem_view_->emitOpError(
        "expecting only one active page when subscripts are constant");

  if (valid_pages == 0)
    return info.paged_mem_view_->emitOpError("no valid pages found");

}
// ---- 325/384  transform_time  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:977  (98L)
LogicalResult e325_transform_time() {
  LLVM_DEBUG(llvm::dbgs()
             << "[TransformPagedMemViewPass] Transforming for time loops\n");
  DT_CHECK(mem_ops_.size() == 1);
  DT_CHECK(tpmv_info_.size() == tpmv_comp_info_.size());

  int explicit_loop_dim = -1;
  SmallVector<AffineMap> subscripts_map_time;
  for (int i = 0, e = tpmv_info_.size(); i < e; ++i) {
    LLVM_DEBUG(llvm::dbgs()
               << "[TransformPagedMemViewPass] Working on a memory operand\n");
    dcc::agen::utils::replaceConstOpsInSubscriptsMap(
        tpmv_info_[i].subscripts_map_, tpmv_info_[i].indices_);

    // Create a new set of subscripts considering the time steps.
    auto time_map = dcc::agen::utils::orderMap(
        tpmv_comp_info_[i].time_addr_map_, time_order_);
    subscripts_map_time.emplace_back(
        agen::utils::concatenateMaps(tpmv_info_[i].subscripts_map_, time_map));

    // If the TPMVInfo isn't for a paged mem view, skip analysis - we just need
    // the subscripts_map_time to update the subscripts of the non-paged TPMV
    // later.
    if (!tpmv_info_[i].paged_mem_view_) continue;

    // Create a copy of the subscripts_map that represents loop iterators as
    // symbols. This is used to form the constraints to determine which pages
    // are valid for mem_op_.
    AffineMap subscripts_map_sym =
        replaceDimsInMapWithSyms(subscripts_map_time[i]);

    // Ranges of the loop iterators are used to only choose pages within the
    // loop iteration space.
    calculateIndicesRanges(tpmv_info_[i].indices_,
                           tpmv_info_[i].indices_ranges_);

    // Add the ranges of the time dims.
    // Since all memory accesses for a memory op use the same time bounds,
    // just use the first access_details_ to grab time bounds.
    addTimeDimIndicesRanges(access_details_[0].getTimeBounds(),
                            tpmv_info_[i].indices_ranges_);
    DT_CHECK(subscripts_map_time[i].getNumDims() ==
             tpmv_info_[i].indices_ranges_.size());

    if (analyzeValidPages(tpmv_info_[i], subscripts_map_sym).failed())
      return LogicalResult::failure();

    explicit_loop_dim = identifyTimeDimForExplicitLoops(
        tpmv_info_[i].subscripts_map_.getNumDims());
  }

  // A explicit_loop_dim of -1 indicates all time dims can be preserved.
  // All other values indicate the innermost time dim to be explicitly created.
  if (explicit_loop_dim > -1) {
    // Ensure nothing done in the above loop added any mem ops.
    DT_CHECK(mem_ops_.size() == 1);
    SmallVector<Operation *, 16> for_ops =
        dcc::agen::utils::constructExplicitTimeLoops(
            mem_ops_[0], access_details_[0].getTimeBounds(), explicit_loop_dim);
    LLVM_DEBUG(llvm::dbgs() << "[TransformPagedMemViewPass] Created the "
                               "following time loops:\n";
               for_ops.front()->print(llvm::dbgs()); llvm::dbgs() << "\n\n");

    dcc::agen::utils::updateTimeSetForExplicitDims(explicit_loop_dim,
                                                   time_order_, time_set_);

    for (int i = 0, e = tpmv_info_.size(); i < e; ++i)
      dcc::agen::utils::updateSubscriptsAndIndicesForExplicitTimeLoops(
          for_ops, explicit_loop_dim, tpmv_info_[i].subscripts_map_,
          subscripts_map_time[i], tpmv_info_[i].indices_);

    // Replace mem_ops_ with an updated one that reflects the time dims being
    // removed.
    OpBuilder builder(context_);
    builder.setInsertionPointToStart(
        cast<affine::AffineForOp>(for_ops.back()).getBody());
    int paged_idx = -1;
    for (int i = 0, e = tpmv_info_.size(); i < e; ++i) {
      if (tpmv_info_[i].paged_mem_view_) {
        paged_idx = i;
        break;
      }
    }
    DT_CHECK(paged_idx >= 0);

    auto mem_view = tpmv_info_[paged_idx].paged_mem_view_.getResult();
    auto new_mem_op = createNewMemOp(
        builder, mem_ops_[0], tpmv_info_[paged_idx], mem_view,
        tpmv_info_[paged_idx].subscripts_map_, tpmv_info_[paged_idx].indices_);
    mem_ops_[0]->erase();
    mem_ops_[0] = new_mem_op;
  } else {
    LLVM_DEBUG(llvm::dbgs()
               << "[TransformPagedMemViewPass] Explicit loops did not need "
                  "to be created - pages do not cross page boundaries\n");
  }

  return LogicalResult::success();

}
// ---- 326/384  initialize_time  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:1095  (23L)
LogicalResult e326_initialize_time() {
  DT_CHECK(mem_ops_.size() == 1);
  auto op = dyn_cast<agen::CompositeLoadOp>(mem_ops_[0]);
  DT_CHECK(op);

  agen::AccessDetailsAffineComposite &ad =
      access_details_.emplace_insert(MemoryOperandIndex::kDirSrc, op, comp_);
  if (ad.constructDetails(MemoryOperandIndex::kDirSrc).failed())
    return op->emitError("Unable to construct details");

  // Construct the time steps info without coalescing and without burst/IL
  // calcs.
  if (AccessDetailsAffineComposite::constructTimeStepsInfo(access_details_,
                                                           false, false)
          .failed())
    return op->emitError("Unable to construct time step info");

  time_order_ = op.getTimeOrder();
  time_set_ = op.getTimeSet().getValue();

  tpmv_comp_info_.emplace_back(op.getTimeAddrMap(), access_details_[0]);

  return LogicalResult::success();


}
// ==================================================================================================
// LEVEL 6
// ==================================================================================================

// ---- 327/384  gatherSymbolicLoadStoreDetails  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:1051  (153L)
LogicalResult e327_gatherSymbolicLoadStoreDetails(
    SenComponents comp, AccessContainer<AccessDetailsSymbolic>& access_details,
    AccessContainer<Value>& mutable_addrs,
    AccessContainer<Value>& immutable_addrs) {
  for (auto& ad : access_details) {
    // For every loop involved in the indices, we need to add an iter_arg for
    // each memory operand. While doing this, we need to also keep the members
    // of the AccessDetails object up to date as cloning the loops may
    // invalidate Values or Operations.
    for (int indices_idx = 0, e = ad.getIndices().size(); indices_idx < e;
         ++indices_idx) {
      // Add an iter_arg to the loop.
      IRMapping ir_map;
      auto block_arg = dyn_cast<BlockArgument>(ad.getIndices()[indices_idx]);
      DT_CHECK(block_arg);
      auto* curr_loop = block_arg.getOwner()->getParentOp();
      auto new_loop = dcc::utils::createForOpWithAdditionalReturnValue(
          curr_loop, 1, ir_map, false);

      updateSymbolicAccessDetails(access_details, ir_map);

      // Update the yield of the op.
      // - If the index is a loop IV, apply the stride to the new iter_arg.
      // - If the index is an iter_arg, the stride has been handled upstream.
      //   Add the value feeding the yield to the new iter_arg.
      auto indices = ad.getIndices();
      int indices_iter_arg_idx = -1;
      Operation* terminator = nullptr;
      BlockArgument iter_arg = nullptr;
      if (auto affine_for = dyn_cast<affine::AffineForOp>(new_loop)) {
        indices_iter_arg_idx = getIndexOfIterArg<affine::AffineForOp>(
            indices[indices_idx], new_loop);
        if (indices_iter_arg_idx < 0)
          DT_CHECK(indices[indices_idx] == affine_for.getInductionVar());
        terminator = affine_for.getBody()->getTerminator();
        iter_arg = affine_for.getRegionIterArgs().back();
      } else if (auto scf_for = dyn_cast<scf::ForOp>(new_loop)) {
        indices_iter_arg_idx =
            getIndexOfIterArg<scf::ForOp>(indices[indices_idx], new_loop);
        if (indices_iter_arg_idx < 0)
          DT_CHECK(indices[indices_idx] == scf_for.getInductionVar());
        terminator = scf_for.getBody()->getTerminator();
        iter_arg = scf_for.getRegionIterArgs().back();
      } else {
        llvm_unreachable("unsupported loop operation");
      }
      DT_CHECK(terminator && iter_arg);
      OpBuilder builder(terminator);
      if (indices_iter_arg_idx < 0) {
        // Loop IV
        auto strides = ad.getStrides();
        auto stride_add = arith::AddIOp::create(builder, terminator->getLoc(),
                                                iter_arg, strides[indices_idx]);
        terminator->setOperand(terminator->getNumOperands() - 1,
                               stride_add.getResult());
      } else {
        // Loop-carried iter_arg
        auto feeding_val = terminator->getOperand(indices_iter_arg_idx);
        auto stride_add = arith::AddIOp::create(builder, terminator->getLoc(),
                                                iter_arg, feeding_val);
        terminator->setOperand(terminator->getNumOperands() - 1,
                               stride_add.getResult());
      }

      // Delete the original loop now that everything has been updated and set.
      curr_loop->erase();
    }
  }
  // Loops have been fully cloned where necessary and are stable for now.

  auto collectMutableIterArg =
      [&](Operation* loop_op,
          int mutable_idx) -> std::pair<BlockArgument, int> {
    BlockArgument iter_arg = nullptr;
    int iter_arg_operand_idx = -1, iter_arg_idx = -1, num_iter_args = -1;
    if (auto affine_for = dyn_cast<affine::AffineForOp>(loop_op)) {
      num_iter_args = affine_for.getNumRegionIterArgs();
      DT_CHECK(num_iter_args > 0);
      iter_arg_idx = num_iter_args - access_details.size() + mutable_idx;
      iter_arg = affine_for.getRegionIterArgs()[iter_arg_idx];
    } else if (auto scf_for = dyn_cast<scf::ForOp>(loop_op)) {
      num_iter_args = scf_for.getNumRegionIterArgs();
      DT_CHECK(num_iter_args > 0);
      iter_arg_idx = num_iter_args - access_details.size() + mutable_idx;
      iter_arg = scf_for.getRegionIterArgs()[iter_arg_idx];
    } else {
      llvm_unreachable("unsupport loop op");
    }
    DT_CHECK(iter_arg && iter_arg_idx >= 0);

    int iter_arg_start_idx = loop_op->getNumOperands() - num_iter_args;
    iter_arg_operand_idx = iter_arg_start_idx + iter_arg_idx;

    return std::make_pair(iter_arg, iter_arg_operand_idx);
  };

  AccessContainer<Value> mem_view_start_addrs;
  for (int ad_idx = 0, num_access_details = access_details.size();
       ad_idx < num_access_details; ++ad_idx) {
    // Order the indices according to loop order, outermost to
    // innermost.
    SmallVector<Value> sorted_indices = access_details[ad_idx].getIndices();
    llvm::sort(sorted_indices, [](Value& a, Value& b) -> bool {
      auto* a_op = cast<BlockArgument>(a).getOwner()->getParentOp();
      auto* b_op = cast<BlockArgument>(b).getOwner()->getParentOp();
      return a_op->isProperAncestor(b_op);
    });

    Value prev_arg = nullptr;
    for (auto& index_val : sorted_indices) {
      // Chain the new_iter_args together and properly initialize them.
      //
      // For L3, the new iter_arg chain should be initialized to 0. Symbolic
      // memory operations have a 1D identity layout map and the operation has
      // no affine map subscripts so there is no constant part of the indices
      // coeffs.
      //
      // For LX below, the new iter_arg chain should be initialized to the
      // memory view start address.
      auto loop = cast<BlockArgument>(index_val).getOwner()->getParentOp();
      auto mutable_iter_arg = collectMutableIterArg(loop, ad_idx);

      if (prev_arg) {
        loop->setOperand(mutable_iter_arg.second, prev_arg);
      } else {
        // First iter_arg in the chain. Initialize the iter_arg to the
        // appropriate value and set prev_arg for next iteration.
        // Note: If the op is in L3 the iter_arg is already initialized to 0
        //       when we added the new iter_arg so no change needed.
        if (!is_any_of(comp, L3LU, L3SU))
          loop->setOperand(mutable_iter_arg.second,
                           access_details[ad_idx].getMemViewStartAddr());
        prev_arg = mutable_iter_arg.first;
      }
    }
    // Collect the mem_view_start_addrs, and mutable_addrs. The mutable_addr
    // will be the last new iter_arg in the chain.
    mem_view_start_addrs.insert(access_details[ad_idx].getMemoryIndex(),
                                access_details[ad_idx].getMemViewStartAddr());
    auto inner_loop =
        cast<BlockArgument>(sorted_indices.back()).getOwner()->getParentOp();
    auto mutable_iter_arg = collectMutableIterArg(inner_loop, ad_idx);

    mutable_addrs.insert(access_details[ad_idx].getMemoryIndex(),
                         mutable_iter_arg.first);
  }

  // For L3, the immutable address should be set to the memory view start
  // address. It is expected these types of offsets will be handled in an
  // LBR/EBR, not LAR/EAR.
  if (failed(constructImmutableAddress(comp, access_details,
                                       mem_view_start_addrs, immutable_addrs)))
    return access_details[0].getOp()->emitError(

}
// ---- 328/384  adjustMutableAddrInitForIndirect  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:1447  (127L)
LogicalResult e328_adjustMutableAddrInitForIndirect(
    Operation* mem_op, Operation* extract_op) {
  DT_CHECK_MSG(mem_op && extract_op, "expecting valid operations");
  DT_CHECK(
      (isa<LoadAndSendOp>(mem_op) && isa<LoadAndExtractScalarOp>(extract_op)) ||
      (isa<ReceiveAndStoreOp>(mem_op) &&
       isa<ReceiveAndExtractScalarOp>(extract_op)) &&
          "invalid operation combination");

  unsigned mutable_addr_idx;
  if (auto load_op = dyn_cast<sentient::LoadAndSendOp>(mem_op)) {
    mutable_addr_idx = load_op.getMutableAddrMutable().getOperandNumber();
  } else if (auto store_op = dyn_cast<sentient::ReceiveAndStoreOp>(mem_op)) {
    mutable_addr_idx = store_op.getMutableAddrMutable().getOperandNumber();
  } else {
    llvm_unreachable("unsupported operation");
  }

  Value mutable_addr_replacement;
  if (auto load_and_extract_op = dyn_cast<LoadAndExtractScalarOp>(extract_op))
    mutable_addr_replacement = load_and_extract_op.getDataResult();
  else
    mutable_addr_replacement = extract_op->getResult(0);

  Value mutable_addr = mem_op->getOperand(mutable_addr_idx);
  if (auto block_arg = dyn_cast<BlockArgument>(mutable_addr)) {
    DT_CHECK((isa<affine::AffineForOp>(block_arg.getOwner()->getParentOp()) ||
              isa<scf::ForOp>(block_arg.getOwner()->getParentOp())) &&
             "mutable_addr parent op should be an expected loop type");
    // Note: The extract_op dominates the indirect op.
    // If the mutable_addr is an iter_arg from a loop:
    // 1. Determine the loop the extract op belongs to, if it belongs to one.
    // 2a. If the extract op belongs to a loop, traverse the iter_arg chains
    //    until a non-iter_arg is hit or the parent of the current loop is the
    //    loop containing the extract op.
    // 2b. If the extract op does not belong to a loop, traverse the iter_arg
    //     chains until a non-iter_arg is hit.
    // 3. Right before the loop the traversal stopped at, insert an add op
    //    adding the extract op result with the initializer of the iter_arg.
    //    Replace the initializer with the new add.
    //    Note: This is safe because every mutable has its own iter_arg.

    // If the extract_op is in the loop the block_arg is in, the mutable_addr
    // update will be inserted after the extract_op.
    auto extract_parent = extract_op->getParentOp();
    auto block_arg_parent = block_arg.getOwner()->getParentOp();
    OpBuilder builder(extract_op);
    if (block_arg_parent == extract_parent) {
      builder.setInsertionPointAfter(extract_op);
      auto new_mutable = sentient::AddOp::create(
          builder, extract_op->getLoc(), builder.getIndexType(), mutable_addr,
          mutable_addr_replacement);
      mem_op->setOperand(mutable_addr_idx, new_mutable);
      return LogicalResult::success();
    }

    // Traverse the iter_arg chain until a non-iter_arg is found or the parent
    // of the current loop is the loop containing the extract_op.
    Operation* curr_loop = block_arg_parent;
    BlockArgument curr_block_arg = block_arg;
    while (curr_loop) {
      int arg_idx = curr_block_arg.getArgNumber();
      Value init_val;
      if (auto affine_loop = dyn_cast<affine::AffineForOp>(curr_loop)) {
        init_val = affine_loop.getInits()[arg_idx - 1];
      } else {
        auto scf_loop = cast<scf::ForOp>(curr_loop);
        init_val = scf_loop.getInits()[arg_idx - 1];
      }

      if (auto init_arg = dyn_cast<BlockArgument>(init_val)) {
        auto init_val_parent = init_arg.getOwner()->getParentOp();
        DT_CHECK(isa<affine::AffineForOp>(init_val_parent));
        if (init_val_parent == extract_parent) {
          builder.setInsertionPoint(curr_loop);
          auto new_init = sentient::AddOp::create(
              builder, extract_op->getLoc(), builder.getIndexType(), init_val,
              mutable_addr_replacement);
          curr_loop->setOperand(arg_idx, new_init);
          return LogicalResult::success();
        } else {
          curr_loop = init_val_parent;
          curr_block_arg = init_arg;
        }
      } else {
        builder.setInsertionPoint(curr_loop);
        auto new_init = sentient::AddOp::create(
            builder, extract_op->getLoc(), builder.getIndexType(), init_val,
            mutable_addr_replacement);
        curr_loop->setOperand(arg_idx, new_init);
        return LogicalResult::success();
      }
    }
  } else {
    // If the mutable_addr is not an iter_arg:
    // 1. Determine if the extract op precedes the mutable_addr or not.
    // 2a. If the mutable_addr precedes the extract op, insert an add op after
    //     the extract op adding the mutable_addr and extract op result.
    // 2b. If the extract op precedes the mutable_addr, insert an add op after
    //     the mutable_addr adding the mutable_addr and extract op result.
    // 3. Replace the mutable_addr with this new add.
    OpBuilder builder(extract_op);
    Operation* mutable_addr_op = mutable_addr.getDefiningOp();
    Operation* curr_op = mem_op;
    while (curr_op) {
      if (curr_op == mutable_addr_op || curr_op == extract_op) {
        builder.setInsertionPointAfter(curr_op);
        break;
      }
      Operation* prev_op = curr_op->getPrevNode();
      if (prev_op)
        curr_op = prev_op;
      else {
        Operation* parent_block_op = curr_op->getBlock()->getParentOp();
        if (parent_block_op)
          curr_op = parent_block_op;
        else
          llvm_unreachable(
              "expecting mutable_addr_op and extract_op to dominate mem_op");
      }
    }

    auto new_mutable = sentient::AddOp::create(
        builder, extract_op->getLoc(), builder.getIndexType(), mutable_addr,
        mutable_addr_replacement);
    mem_op->setOperand(mutable_addr_idx, new_mutable->getResult(0));
  }

}
// ---- 329/384  lowerCompositeLoadOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3106  (17L)
LogicalResult e329_lowerCompositeLoadOp(
    CompositeLoadOp& op, ProgramUnitOp& unit, SenComponents comp,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  AccessContainer<AccessDetailsAffineComposite> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructAffineCompDetailsAndAddrs(
          op, nullptr, unit, comp, access_details,
          /* has_ind_src */ false,
          /* has_ind_dst */ false, mutable_addrs, immutable_addrs)
          .failed())
    return failure();
  DT_CHECK(access_details.size() == 1 && mutable_addrs.size() == 1 &&
           immutable_addrs.size() == 1);

  auto candidate_op = findCandidateForLowering<CompositeLoadOp>(unit);

  return lowerAffineCompositeHelper<CompositeLoadOp>(

}
// ---- 330/384  lowerCompositeStoreOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3127  (17L)
LogicalResult e330_lowerCompositeStoreOp(
    CompositeStoreOp& op, ProgramUnitOp& unit, SenComponents comp,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  AccessContainer<AccessDetailsAffineComposite> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructAffineCompDetailsAndAddrs(
          op, nullptr, unit, comp, access_details,
          /* has_ind_src */ false,
          /* has_ind_dst */ false, mutable_addrs, immutable_addrs)
          .failed())
    return failure();
  DT_CHECK(access_details.size() == 1 && mutable_addrs.size() == 1 &&
           immutable_addrs.size() == 1);

  auto candidate_op = findCandidateForLowering<CompositeStoreOp>(unit);

  return lowerAffineCompositeHelper<CompositeStoreOp>(

}
// ---- 331/384  lowerCompositeLoadAndStoreOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3148  (17L)
LogicalResult e331_lowerCompositeLoadAndStoreOp(
    CompositeLoadAndStoreOp& op, ProgramUnitOp& unit, SenComponents comp,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  AccessContainer<AccessDetailsAffineComposite> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructAffineCompDetailsAndAddrs(op, op, unit, comp, access_details,
                                         /* has_ind_src */ false,
                                         /* has_ind_dst */ false, mutable_addrs,
                                         immutable_addrs)
          .failed())
    return failure();
  DT_CHECK(access_details.size() == 2 && mutable_addrs.size() == 2 &&
           immutable_addrs.size() == 2);

  auto candidate_op = findCandidateForLowering<CompositeLoadAndStoreOp>(unit);

  return lowerAffineCompositeHelper<CompositeLoadAndStoreOp>(

}
// ---- 332/384  lowerCompositeIndirectLoadOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3267  (42L)
LogicalResult e332_lowerCompositeIndirectLoadOp(
    CompositeIndirectLoadOp& op, ProgramUnitOp& unit, SenComponents comp,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  if (comp != SenComponents::LXLU)
    return op->emitError("CompositeIndirectLoadOp only supported in LXLU");

  AccessContainer<AccessDetailsAffineComposite> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructAffineCompDetailsAndAddrs(
          op, nullptr, unit, comp, access_details,
          /* has_ind_src */ false, /* has_ind_dst */ false, mutable_addrs,
          immutable_addrs)
          .failed())
    return failure();

  auto candidate_op = findCandidateForLowering<CompositeIndirectLoadOp>(unit);

  // When load_and_extract_scalar operations are lowered, they assign
  // indices to the indirect load they feed.
  if (!candidate_op->hasAttr("extract_idx"))
    return candidate_op->emitError(
        "composite_indirect_load operations should have an extract_idx "
        "attribute");

  // Locate the extract op that matches the extract_idx.
  auto extract_op =
      findExtractScalarOp<LoadAndExtractScalarOp>(candidate_op, unit);
  if (!extract_op)
    return candidate_op->emitError(
        "could not locate a load_and_extract_scalar operation matching "
        "the extract_idx used by op");

  if (constructTimeLoopsAndVectorOperations(unit, candidate_op, mutable_addrs,
                                            immutable_addrs, access_details,
                                            extract_op)
          .failed())
    return candidate_op->emitError(
        "Unable to generate loops and "
        "sentient statements for the agen.composite_indirect_load "
        "operation");

  to_be_deleted.push_back(candidate_op);

}
// ---- 333/384  lowerCompositeIndirectStoreOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3313  (42L)
LogicalResult e333_lowerCompositeIndirectStoreOp(
    CompositeIndirectStoreOp& op, ProgramUnitOp& unit, SenComponents comp,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  if (comp != SenComponents::LXSU)
    return op->emitError("CompositeIndirectStoreOp only supported in LXSU");

  AccessContainer<AccessDetailsAffineComposite> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructAffineCompDetailsAndAddrs(
          op, nullptr, unit, comp, access_details,
          /* has_ind_src */ false, /* has_ind_dst */ false, mutable_addrs,
          immutable_addrs)
          .failed())
    return failure();

  auto candidate_op = findCandidateForLowering<CompositeIndirectStoreOp>(unit);

  // When receive_and_extract_scalar operations are lowered, they assign
  // indices to the indirect load they feed.
  if (!candidate_op->hasAttr("extract_idx"))
    return candidate_op->emitError(
        "composite_indirect_load operations should have an extract_idx "
        "attribute");

  // Locate the extract op that matches the extract_idx.
  auto extract_op =
      findExtractScalarOp<ReceiveAndExtractScalarOp>(candidate_op, unit);
  if (!extract_op)
    return candidate_op->emitError(
        "could not locate a receive_and_extract_scalar operation matching "
        "the extract_idx used by op");

  if (constructTimeLoopsAndVectorOperations(unit, candidate_op, mutable_addrs,
                                            immutable_addrs, access_details,
                                            extract_op)
          .failed())
    return candidate_op->emitError(
        "Unable to generate loops and "
        "sentient statements for the agen.composite_indirect_load "
        "operation");

  to_be_deleted.push_back(candidate_op);

}
// ---- 334/384  lowerCompositeIndirectLoadAndStoreOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3359  (17L)
LogicalResult e334_lowerCompositeIndirectLoadAndStoreOp(
    CompositeIndirectLoadAndStoreOp& op, ProgramUnitOp& unit,
    SenComponents comp, SmallVectorImpl<Operation*>& to_be_deleted) {
  AccessContainer<AccessDetailsAffineComposite> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructAffineCompDetailsAndAddrs(
          op, op, unit, comp, access_details, op.hasIndirectSrc(),
          op.hasIndirectDst(), mutable_addrs, immutable_addrs)
          .failed())
    return failure();
  DT_CHECK(access_details.size() >= 2 && mutable_addrs.size() >= 2 &&
           immutable_addrs.size() >= 2);

  auto candidate_op =
      findCandidateForLowering<CompositeIndirectLoadAndStoreOp>(unit);

  return lowerAffineCompositeHelper<CompositeIndirectLoadAndStoreOp>(

}
// ---- 335/384  insertCopyAndAddStmts  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3861  (13L)
Value e335_insertCopyAndAddStmts(Operation* loop_op,
                                                        int index,
                                                        Value copy_value,
                                                        int imm_val) {
  // update the loop iterator arguments
  auto stale_val = loop_op->getOperand(loop_op->getNumOperands() - index - 1);
  loop_op->setOperand(loop_op->getNumOperands() - index - 1, copy_value);
  stale_val.getDefiningOp()->erase();

  if (auto affine_for = dyn_cast<affine::AffineForOp>(loop_op)) {
    return insertCopyAndAddStmtsHelper<affine::AffineForOp>(affine_for, index,
                                                            imm_val);
  } else if (auto scf_for = dyn_cast<scf::ForOp>(loop_op)) {

}}
// ---- 336/384  adjustMutableAddrInitForStride  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:4034  (47L)
LogicalResult e336_adjustMutableAddrInitForStride(
    Operation* mem_op, int stride_step, Operation* outermost_comp_loop) {
  DT_CHECK((isa<sentient::LoadAndSendOp>(mem_op) ||
            isa<sentient::ReceiveAndStoreOp>(mem_op)) &&
           "only load_and_send and receive_and_store operations supported for "
           "mutable_addr stride adjustments");
  // If there were loops created as a part of lowering the memory operation,
  // the input to that loop tree needs to be adjusted for the stride.
  if (outermost_comp_loop) {
    auto for_op = dyn_cast<affine::AffineForOp>(outermost_comp_loop);
    DT_CHECK_MSG(for_op, "outermost composite loop should be an AffineForOp");
    auto iter_args = for_op.getRegionIterArgs();
    DT_CHECK_MSG(iter_args.size() == 1,
                 "expecting one iter_arg for outermost composite loop");
    int iter_arg_idx = iter_args[0].getArgNumber();
    Value init_val = for_op.getInits()[iter_arg_idx - 1];

    OpBuilder builder(for_op);
    auto const_op = mlir::arith::ConstantIndexOp::create(
        builder, for_op->getLoc(), -stride_step);
    // builder.setInsertionPointAfter(const_op);
    auto init_adjustment = mlir::arith::AddIOp::create(
        builder, for_op->getLoc(), builder.getIndexType(), init_val,
        const_op.getResult());

    for_op->replaceUsesOfWith(init_val, init_adjustment.getResult());
    return success();
  }

  // If there were no loops created as a part of lowering the memory operation,
  // the mutable_addr of the memory operation needs to be adjusted for the
  // stride.
  unsigned mutable_addr_idx;
  if (auto las = dyn_cast<sentient::LoadAndSendOp>(mem_op))
    mutable_addr_idx = las.getMutableAddrMutable().getOperandNumber();
  else {
    auto ras = cast<sentient::ReceiveAndStoreOp>(mem_op);
    mutable_addr_idx = ras.getMutableAddrMutable().getOperandNumber();
  }

  OpBuilder builder(mem_op);
  auto const_op = mlir::arith::ConstantIndexOp::create(
      builder, mem_op->getLoc(), -stride_step);
  auto init_adjustment = mlir::arith::AddIOp::create(
      builder, mem_op->getLoc(), builder.getIndexType(),
      mem_op->getOperand(mutable_addr_idx), const_op.getResult());
  mem_op->setOperand(mutable_addr_idx, init_adjustment.getResult());

}
// ---- 337/384  lowerSyncOperation  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:1901  (80L)
LogicalResult e337_lowerSyncOperation(
    mlir::ValueRange src_units, mlir::Operation *op) {
  DT_CHECK(src_units.size() > 0);
  std::string src_unit_name =
      dcc::getDirectUnitOpOrgetFirstIndirectUnitOpViaQueryMap(src_units[0])
          .getType()
          .str();
  std::vector<dataflow::GetUnitOp> src_unit_ops, src_unit_ops_corelet0,
      src_unit_ops_corelet1;
  std::vector<mlir::Value> src_unit_vals;
  for (auto src_unit : src_units) src_unit_vals.push_back(src_unit);
  // If op is in a uniform region, limit source units to the units in region
  auto region_index = dcc::uniform::utils::getRegionOpAndIndex(op);
  if (auto uniform_op =
          llvm::dyn_cast<uniform::UniformizeRegionsOp>(region_index.first)) {
    std::vector<std::vector<mlir::Value>> units_of_regions;
    dcc::uniform::utils::getUnitsPerRegionsAsVectorOfVector(uniform_op,
                                                            units_of_regions);
    DT_CHECK(region_index.second < units_of_regions.size());
    src_unit_vals.clear();
    for (auto src_unit : units_of_regions.at(region_index.second))
      src_unit_vals.push_back(src_unit);
  }
  for (auto src_unit : src_unit_vals) {
    auto src_unit_op =
        llvm::dyn_cast<dataflow::GetUnitOp>(src_unit.getDefiningOp());
    DT_CHECK(src_unit_op);
    src_unit_ops.push_back(src_unit_op);
    if (src_unit_op.getType().str() != src_unit_name) {
      op->emitError("Src unit types has to be the same.");
      return LogicalResult::failure();
    }
  }
  OpBuilder builder(op);
  if (src_unit_name.substr(0, 2) != "l3") {
    for (auto src_unit_op : src_unit_ops) {
      if (src_unit_op->hasAttr("corelet")) {
        if (src_unit_op->getAttr("corelet") == builder.getI32IntegerAttr(0))
          src_unit_ops_corelet0.push_back(src_unit_op);
        else /*corelet=1*/
          src_unit_ops_corelet1.push_back(src_unit_op);
      } else {
        op->emitError("Unknown corelet information for sentient");
        return LogicalResult::failure();
      }
    }
  }

  mlir::Value dst_units;
  if (auto send_op = llvm::dyn_cast<dataflow::SyncSendOp>(op)) {
    dst_units = send_op.getToUnit();
  } else if (auto recv_op = llvm::dyn_cast<dataflow::SyncRecvOp>(op)) {
    dst_units = recv_op.getFromUnit();
  } else if (auto implicit_sync_op =
                 llvm::dyn_cast<dataflow::ImplicitSyncOnStreamingBufferOp>(
                     op)) {
    dst_units = implicit_sync_op.getDstUnit();
  } else {
    op->emitError("The operation should be sync send or recv");
    return LogicalResult::failure();
  }

  if (auto dst_unit =
          llvm::dyn_cast<dataflow::GetUnitOp>(dst_units.getDefiningOp())) {
    return lowerSyncForAUnit(src_unit_name, op, src_unit_ops,
                             src_unit_ops_corelet0, src_unit_ops_corelet1,
                             dst_unit);
  }
  if (auto dst_unit_group =
          llvm::dyn_cast<dataflow::CreateGroupOp>(dst_units.getDefiningOp())) {
    return lowerSyncForAGroup(src_unit_name, op, src_unit_ops,
                              src_unit_ops_corelet0, src_unit_ops_corelet1,
                              dst_unit_group);
  }
  if (auto query_map =
          llvm::dyn_cast<uniform::QueryMapOp>(dst_units.getDefiningOp())) {
    return lowerSyncForAQueryMap(src_unit_name, op, src_unit_ops,
                                 src_unit_ops_corelet0, src_unit_ops_corelet1,
                                 query_map);
  }

}
// ---- 338/384  ConstructIFRecursively  —  dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:113  (125L)
Operation *StandardToSentientLoweringPass::ConstructIFRecursively(
    Operation *original_op, Operation *current_op, Value true_value,
    Value false_value, Block::iterator insertion_point,
    SmallVector<Operation *> &ops_to_be_erased) {
  // Base case: Operand of cmpIOp is non-cmpIOps, non-and, non-or
  if (auto cmpi_op = llvm::dyn_cast<mlir::arith::CmpIOp>(current_op)) {
    // Create sentient.if object corresponding to cmpIOp
    // return If(cmpPredicate, op.lhs, op.rhs) true_value, false_value
    OpBuilder builder(original_op);

    builder.setInsertionPoint(builder.getBlock(), insertion_point);
    ArrayAttr locale_attr = builder.getArrayAttr({SentientRegTypeAttr::get(
        builder.getContext(), SentientRegType::unknown)});
    auto if_op = sentient::IfOp::create(
        builder, original_op->getLoc(), TypeRange{true_value.getType()},
        CmpIPredicateAttr::get(
            builder.getContext(),
            getSentientCmpIPredicate(cmpi_op.getPredicate())),
        cmpi_op.getLhs(), cmpi_op.getRhs(), locale_attr, ArrayRef<int32_t>(),
        true);
    if (auto orig_dbg_name = getDbgNameAttr(cmpi_op))
      setDbgNameAttr(if_op, orig_dbg_name);
    builder.createBlock(&if_op.getThenRegion());
    builder.createBlock(&if_op.getElseRegion());

    builder.setInsertionPointToStart(&if_op.getThenRegion().front());
    sentient::YieldOp::create(builder, if_op.getLoc(), true_value);

    builder.setInsertionPointToStart(&if_op.getElseRegion().front());
    sentient::YieldOp::create(builder, if_op.getLoc(), false_value);

    ops_to_be_erased.push_back(cmpi_op);
    return if_op;
  } else if (auto and_op = llvm::dyn_cast<mlir::arith::AndIOp>(current_op)) {
    // Create sentient.if object corresponding to and
    auto lhs_if_op = ConstructIFRecursively(
        original_op, and_op.getLhs().getDefiningOp(), true_value, false_value,
        insertion_point, ops_to_be_erased);
    auto rhs_if_op = ConstructIFRecursively(
        original_op, and_op.getRhs().getDefiningOp(), true_value, false_value,
        insertion_point, ops_to_be_erased);
    if (auto orig_dbg_name = getDbgNameAttr(and_op)) {
      setDbgNameAttr(lhs_if_op, orig_dbg_name);
      setDbgNameAttr(rhs_if_op, orig_dbg_name);
    }

    // return If(lhs) {If(rhs) true_val; else false_val} else false_val;
    // Walk through lhs and replace block having true_val with rhs
    rhs_if_op->walk([&](mlir::Operation *op) {
      if (isa<sentient::YieldOp>(op)) {
        auto yield_op = llvm::dyn_cast<sentient::YieldOp>(op);
        if (yield_op.getOperand(0) == true_value) {
          OpBuilder builder(rhs_if_op);
          builder.setInsertionPoint(op);
          auto cloned_op = builder.clone(*lhs_if_op);
          yield_op.setOperand(0, cloned_op->getResult(0));
        }
      }
    });

    ops_to_be_erased.push_back(and_op);
    lhs_if_op->erase();
    return rhs_if_op;
  } else if (auto or_op = llvm::dyn_cast<mlir::arith::OrIOp>(current_op)) {
    // Create sentient.if object corresponding to and
    auto lhs_if = ConstructIFRecursively(
        original_op, or_op.getLhs().getDefiningOp(), true_value, false_value,
        insertion_point, ops_to_be_erased);
    auto rhs_if = ConstructIFRecursively(
        original_op, or_op.getRhs().getDefiningOp(), true_value, false_value,
        insertion_point, ops_to_be_erased);
    if (auto orig_dbg_name = getDbgNameAttr(or_op)) {
      setDbgNameAttr(lhs_if, orig_dbg_name);
      setDbgNameAttr(rhs_if, orig_dbg_name);
    }

    // return If(lhs) {true_val} else { if(rhs) {true_val} else {false_val}};
    rhs_if->walk([&](mlir::Operation *op) {
      if (isa<sentient::YieldOp>(op)) {
        auto yield_op = llvm::dyn_cast<sentient::YieldOp>(op);
        if (yield_op.getOperand(0) == false_value) {
          OpBuilder builder(rhs_if);
          builder.setInsertionPoint(op);
          auto cloned_op = builder.clone(*lhs_if);
          yield_op.setOperand(0, cloned_op->getResult(0));
        }
      }
    });

    ops_to_be_erased.push_back(or_op);
    lhs_if->erase();
    return rhs_if;
  } else if (current_op->getNumResults() == 1) {
    // Create sentient.if object otherwise
    // return If(cmpPredicate::eq, op.result, true) true_value, false_value
    OpBuilder builder(original_op);

    builder.setInsertionPoint(builder.getBlock(), insertion_point);
    ArrayAttr locale_attr = builder.getArrayAttr({SentientRegTypeAttr::get(
        builder.getContext(), SentientRegType::unknown)});

    auto val_true = sentient::ConstantOp::create(
        builder, builder.getUnknownLoc(), builder.getI1Type(), 1);
    auto if_op = sentient::IfOp::create(
        builder, original_op->getLoc(), TypeRange{true_value.getType()},
        CmpIPredicateAttr::get(builder.getContext(), CmpIPredicate::eq),
        current_op->getOpResults()[0], val_true, locale_attr,
        ArrayRef<int32_t>(), true);
    if (auto orig_dbg_name = getDbgNameAttr(original_op))
      setDbgNameAttr(if_op, orig_dbg_name);

    builder.createBlock(&if_op.getThenRegion());
    builder.createBlock(&if_op.getElseRegion());

    builder.setInsertionPointToStart(&if_op.getThenRegion().front());
    sentient::YieldOp::create(builder, if_op.getLoc(), true_value);

    builder.setInsertionPointToStart(&if_op.getElseRegion().front());
    sentient::YieldOp::create(builder, if_op.getLoc(), false_value);

    return if_op;
  } else {
    current_op->emitError(
        "The input condition to std.select should come from"
        "CMPI/AND/OR operations or an op with one return value");

}}
// ---- 339/384  SimplifyOrIOp  —  dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:389  (43L)
void e339_SimplifyOrIOp(mlir::Operation *op) {
  auto or_op = llvm::dyn_cast<mlir::arith::OrIOp>(op);
  OpBuilder builder(or_op);

  int andi_operand_num = 0;
  int cmpi_operand_num = 1;
  if (isa<mlir::arith::CmpIOp>(or_op.getOperand(0).getDefiningOp()) &&
      isa<mlir::arith::AndIOp>(or_op.getOperand(1).getDefiningOp())) {
    cmpi_operand_num = 0;
    andi_operand_num = 1;
  }
  auto cmpi_op = llvm::dyn_cast<mlir::arith::CmpIOp>(
      or_op.getOperand(cmpi_operand_num).getDefiningOp());
  auto andi_op = llvm::dyn_cast<mlir::arith::AndIOp>(
      or_op.getOperand(andi_operand_num).getDefiningOp());

  if (!cmpi_op || !andi_op) return;
  auto cmpi_op_neg = llvm::dyn_cast<mlir::arith::CmpIOp>(
      andi_op.getOperand(0).getDefiningOp());
  int other_operand_num = 1;
  if (!cmpi_op_neg) {
    cmpi_op_neg = llvm::dyn_cast<mlir::arith::CmpIOp>(
        andi_op.getOperand(1).getDefiningOp());
    other_operand_num = 0;
  }
  if (!((cmpi_op.getPredicate() == arith::CmpIPredicate::eq &&
         cmpi_op_neg.getPredicate() == arith::CmpIPredicate::ne) ||
        (cmpi_op.getPredicate() == arith::CmpIPredicate::ne &&
         cmpi_op_neg.getPredicate() == arith::CmpIPredicate::eq)))
    return;

  if (!((cmpi_op.getLhs() == cmpi_op_neg.getLhs() &&
         cmpi_op.getRhs() == cmpi_op_neg.getRhs()) ||
        (cmpi_op.getLhs() == cmpi_op_neg.getRhs() &&
         cmpi_op.getRhs() == cmpi_op_neg.getLhs())))
    return;

  auto new_or_op = mlir::arith::OrIOp::create(
      builder, op->getLoc(), cmpi_op, andi_op.getOperand(other_operand_num));
  op->replaceAllUsesWith(new_or_op);
  or_op.erase();
  if (andi_op.use_empty()) andi_op.erase();
  if (cmpi_op_neg.use_empty()) cmpi_op_neg.erase();

}
// ---- 340/384  analyzeAndFillOperandForwarding  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:535  (25L)
void e340_analyzeAndFillOperandForwarding(
    MLIRContext* context, const dcc::DccExtContext& dcc_ext_ctx,
    SenComponents comp, std::optional<VectorOperand>& from_operand,
    SmallVector<Attribute, 1>& op_forwarding) {
  if (from_operand.has_value()) {
    if (is_any_of(from_operand.value().type_, VectorOperandType::NFWD,
                  VectorOperandType::ConstantBitstream))
      return;

    if (!from_operand.value().splat_.empty()) {
      op_forwarding.push_back(SentientComputePortAttr::get(
          context,
          symbolizeSentientComputePort(from_operand.value().splat_).value()));
    }

    for (auto& use : from_operand.value().op_->getUses()) {
      Operation* user = use.getOwner();
      if (!isa<vectorchain::NegOp, vectorchain::CastOp, vectorchain::SelectOp,
               arith::SIToFPOp, arith::FPToSIOp>(user)) {
        auto to = VectorOperand::getOperand(dcc_ext_ctx, user, comp, false);
        if (to.has_value()) {
          op_forwarding.push_back(SentientComputePortAttr::get(
              context,
              symbolizeSentientComputePort(to.value().getName()).value()));
        }

}}}}
// ---- 341/384  analyzeNonComputeOpsForFusion  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.cpp:610  (86L)
void e341_analyzeNonComputeOpsForFusion(
    dataflow::ProgramUnitOp& unit, const dcc::DccExtContext& dcc_ext_ctx,
    OperandReuse& reuse_info, Operation* op, std::optional<VectorOperand>& from,
    const SenComponents comp, std::map<Operation*, bool>& is_visited,
    bool& is_fusion_respected,
    llvm::SmallVectorImpl<std::optional<VectorOperand>>& to_operands,
    bool& is_dangling_ops_present_after_fusion) {
  //  Add the current op to to_operands.
  auto to_operand = VectorOperand::getOperand(dcc_ext_ctx, op, comp, false);
  if (to_operand.has_value() &&
      isa<dataflow::SendOp, vector::StoreOp, agen::VectorStoreOp>(
          to_operand.value().op_)) {
    to_operands.push_back(to_operand);
  }

  // Trying to avoid fusing/merging multiple store operations that are
  // using the data from the from_op.
  bool is_store_op_encountered =
      isa<agen::VectorStoreOp, vector::StoreOp>(to_operand.value().op_);

  // Get the users
  SmallVector<Operation*> users;
  if (!isa<mlir::arith::ConstantOp>(from.value().op_)) {
    for (auto user : from.value().op_->getUsers()) {
      // Sometimes, getUsers link may not be updated!
      if (user) {
        users.push_back(user);
      }
    }
  } else {
    // Reuse info was constructed for a unit instead of the entire dataflow
    // graph. Some constant ops (from op) are moved to the outside the
    // unit definition causing issues since reuse info is meant per unit.
    for (auto user : from.value().op_->getUsers()) {
      // Sometimes, getUsers link may not be updated!
      auto is_ancestor = unit->isAncestor(user);
      if (user && is_ancestor) {
        users.push_back(user);
      } else if (user && !is_ancestor) {
        is_dangling_ops_present_after_fusion = true;
      }
    }
  }

  // Sort the users based on the dominance info
  llvm::sort(users, [&](Operation* left, Operation* right) -> bool {
    auto val = reuse_info.dominates(left, right);
    return val;
  });

  // TODO: what happens when you have two sends from a from-op.

  // Flag to determine if there is an op that wasn't fused.
  // In such cases, we shouldn't remove the from_op.
  for (const auto& user : users) {
    // Check if the user is send or store op -- so that we are working
    // only from (receive, load) --> (send, store).
    auto to = VectorOperand::getOperand(dcc_ext_ctx, user, comp, false);

    // Avoid processing current op, because we already know its a user
    // and we have added them!
    if (to.has_value() && to.value().op_ != op && !is_visited[to.value().op_]) {
      bool allow_merging =
          isa<dataflow::SendOp, vector::StoreOp, agen::VectorStoreOp>(
              to.value().op_);

      if (allow_merging) {
        allow_merging =
            !(isa<vector::StoreOp, agen::VectorStoreOp>(to.value().op_) &&
              is_store_op_encountered);

        if (isa<vector::StoreOp, agen::VectorStoreOp>(to.value().op_)) {
          is_store_op_encountered = true;
        }

        // Do one more check for store_op merging
        // If this fails, don't set for fusion failure.
        if (allow_merging) {
          to_operands.push_back(to);
          if (to.has_value()) {
            is_visited[to.value().op_] = true;
          }
        } else {
          is_dangling_ops_present_after_fusion = true;
        }
      } else {

}}}}
// ---- 342/384  analyzeAndFillResultForwarding  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorChainHelper.hpp:162  (27L)
void analyzeAndFillResultForwarding(
    Operation* op, SenComponents comp, MLIRContext* context,
    const dcc::DccExtContext& dcc_ext_ctx, BuilderType& builder,
    llvm::SmallVectorImpl<std::optional<VectorOperand>>& to_operands,
    llvm::SmallVectorImpl<Attribute>& result_forwarding,
    sentient::SentientComputePortAttr& logical_result_forwarding) {
  for (auto user : op->getUsers()) {
    auto to = VectorOperand::getOperand(dcc_ext_ctx, user, comp, false);
    to_operands.push_back(to);

    std::string dest = to.value().getName();
    if (dest == "sfpring") {
      if (isa<mlir::arith::FPToSIOp, mlir::arith::SIToFPOp,
              vectorchain::CastOp>(user)) {
        DT_CHECK_MSG(!user->getUsers().empty(),
                     "Cast operation expected to have one use");
        user = *user->getUsers().begin();
      }
      auto send_op = llvm::dyn_cast<dataflow::SendOp>(user);
      builder.setInsertionPointAfter(send_op);
      (void)sentient::SetSendDestinationOp::create(builder, op->getLoc(),
                                                   send_op.getToUnit());
    }

    // Store istate output operand separately in logical_result_forwarding
    if (dest.find("istate") != dest.npos)
      logical_result_forwarding = sentient::SentientComputePortAttr::get(

}}
// ---- 343/384  getOperandFromCastOp  —  dcc/src/Conversion/VectorChainLowering/CommonHelpers/VectorOperands.cpp:354  (5L)
std::optional<VectorOperand> e343_getOperandFromCastOp(
    const dcc::DccExtContext &dcc_ext_ctx, vectorchain::CastOp &op,
    const SenComponents comp) {
  DT_CHECK(isa<vectorchain::CastOp>(op));
  auto *parent = op.getOperand(0).getDefiningOp();

}
// ---- 344/384  lowerDanglingNonComputeOpsPESFP  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1274  (93L)
VectorChainToSentientPESFPLoweringPass::lowerDanglingNonComputeOpsPESFP(
    dataflow::ProgramUnitOp &unit, const SenComponents comp,
    const dcc::DccExtContext &dcc_ext_ctx, OperandReuse &reuse_info) {
  llvm::SmallVector<mlir::Operation *, 4> tobe_deleted;
  mlir::LogicalResult result = success();
  unit.walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<dataflow::ReceiveOp, vector::LoadOp, agen::VectorLoadOp>(op)) {
      if (!op->use_empty()) {
        op->emitError(
            "There is still a receive or load with uses, "
            "that shouldn't happen");
        result = failure();
        return WalkResult::interrupt();
      }
      auto from = VectorOperand::getOperand(dcc_ext_ctx, op, comp).value();
      auto absorption_flag = reuse_info.getAbsorbtionFlag(from.op_);
      if (absorption_flag.has_value() && !absorption_flag.value()) {
        // it has been used but not absorbed by some other op that is already
        // lowered
        auto *context = op->getContext();
        ArrayRef<Value> pointers = {};

        OpBuilder builder(op);
        sentient::ConstantOp mask_const_op = sentient::ConstantOp::create(
            builder, op->getLoc(), builder.getIndexType(), 0);
        std::string input_precision = getInputPrecisionFromOperand(from);
        std::string compute_precision = from.on_the_fly_conv_precision_;
        if (compute_precision != "fp32") {
          compute_precision = "fp16";
        }
        SentientFoldModeAttr fold_mode_attr =
            dcc::sentient::utils::getSentientFoldModeAttrForOperation(op, comp);
        // Set result precision to "none" to indicate dummy MAC
        std::string result_precision = "none";

        auto op_dbg_name = dataflow::getDbgNameAttr(op);
        if (from.getName() == "west" && isa<dataflow::ReceiveOp>(op)) {
          auto mac_op = sentient::MacOp::create(
              builder, op->getLoc(), TypeRange(), mask_const_op.getResult(),
              ValueRange(pointers), op_dbg_name, SentientComputePort::one,
              SentientComputePort::zero,
              symbolizeSentientComputePort(from.getName()).value(),
              ArrayAttr::get(context, {}), ArrayAttr::get(context, {}),
              ArrayAttr::get(context, {}), ArrayAttr::get(context, {}),
              fold_mode_attr,
              symbolizeSentientPrecision(input_precision).value(),
              symbolizeSentientPrecision(input_precision).value(),
              symbolizeSentientPrecision(input_precision).value(),
              symbolizeSentientPrecision(result_precision).value(),
              symbolizeSentientPrecision(compute_precision).value(), -1, -1,
              reuse_info.getId(from.op_).value());
        } else {
          auto mac_op = sentient::MacOp::create(
              builder, op->getLoc(), TypeRange(), mask_const_op,
              ValueRange(pointers), op_dbg_name,
              symbolizeSentientComputePort(from.getName()).value(),
              symbolizeSentientComputePort(from.getName()).value(),
              symbolizeSentientComputePort(from.getName()).value(),
              ArrayAttr::get(context, {}), ArrayAttr::get(context, {}),
              ArrayAttr::get(context, {}), ArrayAttr::get(context, {}),
              fold_mode_attr,
              symbolizeSentientPrecision(input_precision).value(),
              symbolizeSentientPrecision(input_precision).value(),
              symbolizeSentientPrecision(input_precision).value(),
              symbolizeSentientPrecision(result_precision).value(),
              symbolizeSentientPrecision(compute_precision).value(),
              reuse_info.getId(from.op_).value(), -1, -1);
          mac_op->setAttr("DataTransferOnly", builder.getBoolAttr(true));
        }
      } else if (!absorption_flag.has_value()) {
        from.op_->emitError("Dangling non-compute op has no use\n");
        result = failure();
        return WalkResult::interrupt();
      }

      tobe_deleted.push_back(op);
    } else if (isa<vectorchain::CreateAffineMaskOp>(op)) {
      // All CreateAffineMaskOps should be connected to other operations that
      // were already lowered.
      if (!op->use_empty()) {
        op->emitError(
            "There is still a create_affine_mask with uses - that shouldn't "
            "happen");
        result = failure();
        return WalkResult::interrupt();
      }
      tobe_deleted.push_back(op);
    }
    return WalkResult::advance();
  });

  for (auto op : tobe_deleted) {
    VectorOperand::eraseOp(op);

}}
// ---- 345/384  processXrfPtrPerUnit  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:337  (190L)
XrfPtrMap e345_processXrfPtrPerUnit(dataflow::ProgramUnitOp unit,
                                            LayoutExprMap expr_maps[]) {
  DT_CHECK_MSG(
      is_any_of(dccExtContext().getArch(), RCUDD1A_ISA, MPW4_ISA, SEN1P5_ISA),
      "XRF lowering is only enabled for SENARCH being "
      "one among the RCUDD1A_ISA, MPW4_ISA, SEN1P5_ISA");

  // a map of vector_load&vector_store and their corresponding xrf_ptr values
  // (both dummy mac's operand and result). The dummy mac ops are to be replaced
  // in following fuseNonComputeOps() and fuseComputeOps()
  XrfPtrMap vector_op_to_xrfptr_map;
  auto precision = unit.getPrecision().value().str();

  // work on write_ptr and read_ptr sequentially
  for (int i = 0; i < 2; i++) {
    auto expr_map = expr_maps[i];

    // write ptr: i=0; read ptr: i=1
    unsigned xrf_incr_after_prev_mac = 1;
    if (i == 1) {  // read pointer
      xrf_incr_after_prev_mac =
          dccExtContext().getXrfRdPtrIncrValAfterMAC(precision);
    }

    // a stack of forOp
    std::stack<Operation *> forop_stack;
    std::stack<Value> if_then_region_init_ptr_stack;
    // a stack of constantExpr vector for xrf accesses within each forOp region.
    llvm::SmallVector<llvm::SmallVector<int64_t, 4>, 4> xrf_access_offsets;
    // a stack of loop coefficents for each forOp
    std::stack<int64_t> loop_stride_step_stack;
    // keep track of the latest xrf ptr right before ifOp. this is used to help
    // lowering ifOp's else-region

    Type xrf_reg_type = IndexType::get(unit.getContext());
    // std::string xrf_ptr_name = (i == 0) ? "xrfwrptr" : "xrfrdptr";
    std::string xrf_ptr_name =
        "imm";  // regTypeAssignmentPass will assign reg type

    // initialize xrf reg to zero and update xrf_access_offsets
    OpBuilder init_builder(unit);
    init_builder.setInsertionPointToStart(&unit.getRegion().front());
    auto xrf_ptr = sentient::ConstantOp::create(init_builder, unit.getLoc(),
                                                xrf_reg_type, 0);
    Value xrf_ptr_val = xrf_ptr->getResult(0);
    xrf_access_offsets.push_back({0});

    // utility function to get the xrf access offset from outer forOp level
    auto getOffsetFromOuterForOp = [&]() {
      int forop_level = xrf_access_offsets.size() - 2;
      while (xrf_access_offsets[forop_level].size() == 0) {
        forop_level--;
      }
      return xrf_access_offsets[forop_level].back();
    };

    // walk throught all ops to insert xrf offset ops according to the 3
    // scenarios mentioned above
    SmallVector<Operation *, 16> ops_to_delete;
    unit.walk<WalkOrder::PreOrder>([&](Operation *op) {
      Location loc = op->getLoc();
      if (isa<sentient::ForOp, sentient::IfOp>(op)) {
        if (region_ops_with_xrf_access.count(op)) {
          // push forOp to stack
          forop_stack.push(op);
          xrf_access_offsets.push_back({});
          loop_stride_step_stack.push(INT64_MIN);

          // update loop coeff if it exists in expr_map
          auto it = expr_map.begin();
          while (it != expr_map.end()) {
            if (it->second.layout_map.count(op)) {
              loop_stride_step_stack.top() = it->second.layout_map.at(op);
              break;
            }
            it++;
          }

          if (isa<sentient::ForOp>(op)) {
            // reset forOp's iter_args and update current xrf_ptr_val
            auto old_ptr = op->getOperand(1 + i).getDefiningOp();
            op->setOperand(1 + i, xrf_ptr_val);
            xrf_ptr_val = getXrfValue(op, i);
            if (old_ptr->use_empty()) ops_to_delete.push_back(old_ptr);
          } else {
            // stack the init xrf ptr for else-region
            if_then_region_init_ptr_stack.push(xrf_ptr_val);
          }
        }
      } else if ((((1 - i) && isa<agen::VectorStoreOp, vector::StoreOp>(op)) ||
                  (i && isa<agen::VectorLoadOp, vector::LoadOp>(op))) &&
                 isXrfRelated(op)) {
        // type 1 scenario

        int64_t curr_const = expr_map[op].constant_val;
        // add constant bias of this xrf access to the stack
        // the default xrf ptr increment value of sentient.mac is included in
        // the total xrf offset. assume unroll==1 at vectorChain stage
        xrf_access_offsets.back().push_back(curr_const +
                                            xrf_incr_after_prev_mac);

        // insert xrf offset ops before agen:vector_store
        int64_t prev_const;
        if (xrf_access_offsets.back().size() > 1) {
          prev_const =
              xrf_access_offsets.back()[xrf_access_offsets.back().size() - 2];
        } else {
          prev_const = getOffsetFromOuterForOp();
        }

        auto const_offset_val = curr_const - prev_const;
        OpBuilder builder(op);
        xrf_ptr_val =
            insertConstAndAddOps(&builder, loc, xrf_reg_type, xrf_ptr_val,
                                 const_offset_val, xrf_ptr_name);
        vector_op_to_xrfptr_map[op].at(0).at(i) = xrf_ptr_val;  // bookkeeping
        // insert dummy mac_op as placeholder for next xrf operation which will
        // be replaced later
        xrf_ptr_val = insertDummyMacOp(&builder, op->getContext(), loc);
        vector_op_to_xrfptr_map[op].at(1).at(i) = xrf_ptr_val;  // bookkeeping

      } else if ((((1 - i) && isa<agen::VectorLoadOp, vector::LoadOp>(op)) ||
                  (i && isa<agen::VectorStoreOp, vector::StoreOp>(op))) &&
                 isXrfRelated(op)) {
        // type 1 scenario for inactive xrf ptr.
        vector_op_to_xrfptr_map[op].at(0).at(i) = xrf_ptr_val;  // bookkeeping
        OpBuilder builder(op);
        xrf_ptr_val = insertDummyMacOp(&builder, op->getContext(), loc);
        vector_op_to_xrfptr_map[op].at(1).at(i) = xrf_ptr_val;  // bookkeeping
      } else if (auto yield_op = dyn_cast<sentient::YieldOp>(op)) {
        auto parent_op = op->getParentRegion()->getParentOp();
        if (region_ops_with_xrf_access.count(parent_op)) {
          // type 2 scenario (applied to forOp and ifOp):
          // insert addOp to offset xrf index before next iteration and set
          // return value of the current forOp at the end of forop region.
          OpBuilder forop_builder(forop_stack.top());
          forop_builder.setInsertionPoint(op);
          int64_t prev_const = getOffsetFromOuterForOp();
          // if there is no xrf access in current loop, no address offset is
          // needed.
          int64_t curr_const = xrf_access_offsets.back().size() > 0
                                   ? xrf_access_offsets.back().back()
                                   : prev_const;
          auto const_offset_val = prev_const - curr_const +
                                  (loop_stride_step_stack.top() != INT64_MIN
                                       ? loop_stride_step_stack.top()
                                       : 0);
          xrf_ptr_val =
              insertConstAndAddOps(&forop_builder, loc, xrf_reg_type,
                                   xrf_ptr_val, const_offset_val, xrf_ptr_name);
          // update forOp's yield arguments
          xrf_ptr_val = updateYieldArgs(yield_op, xrf_ptr_val, i);

          if (isa<sentient::ForOp>(parent_op)) {
            // type 3 scenario for forOp only:
            // insert addOp to offset the travel distance made in the forOp
            // right after the current forop
            forop_builder.setInsertionPointAfter(forop_stack.top());
            const_offset_val = -(loop_stride_step_stack.top() != INT64_MIN
                                     ? loop_stride_step_stack.top()
                                     : 0) *
                               getForOpBound(forop_stack.top());
            xrf_ptr_val = insertConstAndAddOps(&forop_builder, loc,
                                               xrf_reg_type, xrf_ptr_val,
                                               const_offset_val, xrf_ptr_name);

            // pop 3 stacks
            forop_stack.pop();
            loop_stride_step_stack.pop();
            xrf_access_offsets.pop_back();
          } else {
            // for IfOp
            auto if_op = cast<sentient::IfOp>(parent_op);
            if (if_op.getThenRegion().isAncestor(yield_op->getParentRegion())) {
              xrf_access_offsets.back().clear();
              xrf_ptr_val = if_then_region_init_ptr_stack.top();
              if_then_region_init_ptr_stack.pop();
            } else {
              // pop 3 stacks
              forop_stack.pop();
              loop_stride_step_stack.pop();
              xrf_access_offsets.pop_back();
            }
          }
        }
      }
    });
    for (Operation *op : ops_to_delete) op->erase();
  }


}
// ---- 346/384  lowerDanglingNonComputeOps  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:882  (88L)
void e346_lowerDanglingNonComputeOps(
    dataflow::ProgramUnitOp &unit, const SenComponents comp,
    XrfPtrMap vector_op_to_xrfptr_map, OperandReuse &reuse_info,
    LoopMaskTree *pt_masking_tree) {
  llvm::SmallVector<mlir::Operation *, 4> tobe_deleted;
  unit.walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<dataflow::ReceiveOp, vector::LoadOp, agen::VectorLoadOp>(op)) {
      if (!op->use_empty()) {
        op->emitError(
            "There is still a receive or load with uses, "
            "that shouldn't happen");
        signalPassFailure();
        return;
      }

      auto from = VectorOperand::getOperand(dccExtContext(), op, comp).value();
      auto absorbtion_flag = reuse_info.getAbsorbtionFlag(from.op_);
      if (absorbtion_flag.has_value() && !absorbtion_flag.value()) {
        // it has been used but not absorbed by some other op that is already
        // lowered
        auto *context = op->getContext();
        ArrayRef<Value> pointers = {};
        if (isa<agen::VectorLoadOp>(op)) {
          pointers = vector_op_to_xrfptr_map.at(op).at(0);
        }

        OpBuilder builder(op);
        sentient::ConstantOp mask_const_op = sentient::ConstantOp::create(
            builder, op->getLoc(), builder.getIndexType(), 0);
        DT_CHECK_MSG(from.orig_precision_ == from.on_the_fly_conv_precision_,
                     "Expecting no on the fly conversions in PT");
        std::string precision = getInputPrecisionFromOperand(from);
        std::string compute_precision = computeUnitPrecision(unit, comp);
        SentientFoldModeAttr fold_mode_attr =
            dcc::sentient::utils::getSentientFoldModeAttrForOperation(op, comp);

        auto op_dbg_name = dataflow::getDbgNameAttr(op);
        if (from.getName() == "west" && isa<dataflow::ReceiveOp>(op)) {
          auto mac_op = sentient::MacOp::create(
              builder, op->getLoc(), TypeRange(), nullptr, ValueRange(pointers),
              op_dbg_name, SentientComputePort::one, SentientComputePort::zero,
              symbolizeSentientComputePort(from.getName()).value(),
              ArrayAttr::get(context, {}), ArrayAttr::get(context, {}),
              ArrayAttr::get(context, {}), ArrayAttr::get(context, {}),
              fold_mode_attr, symbolizeSentientPrecision(precision).value(),
              symbolizeSentientPrecision(precision).value(),
              symbolizeSentientPrecision(precision).value(),
              symbolizeSentientPrecision(precision).value(),
              symbolizeSentientPrecision(compute_precision).value(), -1, -1,
              reuse_info.getId(from.op_).value());
          updateLoopMaskTreeForConstantMask(pt_masking_tree, op, &mac_op, 0);
        } else {
          // Note: PT masking is not attached to operations.
          auto mac_op = sentient::MacOp::create(
              builder, op->getLoc(), TypeRange(), nullptr, ValueRange(pointers),
              op_dbg_name, symbolizeSentientComputePort(from.getName()).value(),
              SentientComputePort::one, SentientComputePort::zero,
              ArrayAttr::get(context, {}), ArrayAttr::get(context, {}),
              ArrayAttr::get(context, {}), ArrayAttr::get(context, {}),
              fold_mode_attr, symbolizeSentientPrecision(precision).value(),
              symbolizeSentientPrecision(precision).value(),
              symbolizeSentientPrecision(precision).value(),
              symbolizeSentientPrecision(precision).value(),
              symbolizeSentientPrecision(compute_precision).value(),
              reuse_info.getId(from.op_).value(), -1, -1);
          updateLoopMaskTreeForConstantMask(pt_masking_tree, op, &mac_op, 0);
        }
      } else if (!absorbtion_flag.has_value()) {
        from.op_->emitError("Dangling non-compute op has no use\n");
        signalPassFailure();
        return;
      }

      tobe_deleted.push_back(op);
    } else if (isa<vectorchain::CreateAffineMaskOp>(op)) {
      // All CreateAffineMaskOps should be connected to other operations that
      // were already lowered.
      if (!op->use_empty()) {
        op->emitError(
            "There is still a create_affine_mask with uses - that shouldn't "
            "happen");
        signalPassFailure();
        return;
      }
      tobe_deleted.push_back(op);
    }
  });


}
// ---- 347/384  topLevelConditionsMatch  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:279  (31L)
static bool topLevelConditionsMatch(dcc::OperationEquivalence &oe,
                                    Operation *if_op0, Operation *if_op1) {
  if (isa<affine::AffineIfOp>(if_op0) && isa<affine::AffineIfOp>(if_op1)) {
    auto affine_if_op0 = llvm::dyn_cast<affine::AffineIfOp>(if_op0);
    auto affine_if_op1 = llvm::dyn_cast<affine::AffineIfOp>(if_op1);
    // Check whether the constraint sets match and the operands are equivalent.
    if (affine_if_op0.getIntegerSet() != affine_if_op1.getIntegerSet())
      return false;
    for (auto pair : llvm::zip(if_op0->getOperands(), if_op1->getOperands())) {
      auto if0_operand = std::get<0>(pair);
      auto if1_operand = std::get<1>(pair);
      bool is_if0_operand_block_arg = mlir::isa<BlockArgument>(if0_operand);
      bool is_if1_operand_block_arg = mlir::isa<BlockArgument>(if1_operand);
      if (is_if0_operand_block_arg && is_if1_operand_block_arg) {
        auto *owner0 = mlir::cast<BlockArgument>(if0_operand).getOwner();
        auto *owner1 = mlir::cast<BlockArgument>(if1_operand).getOwner();
        auto idx0 = mlir::cast<BlockArgument>(if0_operand).getArgNumber();
        auto idx1 = mlir::cast<BlockArgument>(if1_operand).getArgNumber();
        if (owner0 != owner1 || idx0 != idx1) return false;
      } else if (is_if0_operand_block_arg || is_if1_operand_block_arg ||
                 !oe.operationsAreEquivalent(*if0_operand.getDefiningOp(),
                                             *if1_operand.getDefiningOp()))
        return false;
    }
  } else if (isa<scf::IfOp>(if_op0) && isa<scf::IfOp>(if_op1)) {
    // Check whether the conditions are equivalent.
    if (!oe.operationsAreEquivalent(*if_op0->getOperand(0).getDefiningOp(),
                                    *if_op1->getOperand(0).getDefiningOp()))
      return false;
  } else
    return false;

}
// ---- 348/384  singleOpBranchToYieldVal  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:498  (18L)
bool e348_singleOpBranchToYieldVal(
    Operation *op, int64_t &yielded_val, bool then_branch) {
  if (!op) return false;
  auto if_op = llvm::dyn_cast<mlir::scf::IfOp>(op);
  if (!if_op) return false;
  Block &bb = then_branch ? if_op.getThenRegion().front()
                          : if_op.getElseRegion().front();
  if (bb.getOperations().size() != 1) return false;
  auto *terminator = bb.getTerminator();
  Operation *yielded_val_op = terminator->getOperand(0).getDefiningOp();

  if (!yielded_val_op)
    return false;
  else if (auto const_idx_op =
               llvm::dyn_cast<mlir::arith::ConstantIndexOp>(yielded_val_op))
    yielded_val = const_idx_op.value();
  else
    return false;

}
// ---- 349/384  isLoopInvariant  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:681  (66L)
bool e349_isLoopInvariant(CondNode *n,
                                                  Operation *parent_for_op) {
  DT_CHECK(!n->isLeaf());
  Operation *if_op = n->getOperation();
  auto scf_if = llvm::dyn_cast<scf::IfOp>(if_op);
  DT_CHECK_MSG(scf_if, "Expect valid scf::IfOp");

  // Worklist of values needed to be checked for dependencies on loop iterator
  // arguments.
  SmallVector<Value> worklist;

  if (n->getThenNode()->isLeaf()) {
    Block &bb = scf_if.getThenRegion().front();
    if (bb.getOperations().size() != 1) return false;
    // Check that the yielded value does not depend on the loop iterator
    // arguments
    worklist.push_back(bb.getTerminator()->getOperand(0));
  } else {
    // Check loop invariance recursively inside the then-branch.
    auto then_node = n->getThenNode();
    if (then_node->getNumberOfChildren() != 1 ||
        !isLoopInvariant(then_node->getFirstChild(), parent_for_op))
      return false;
    // The then-branch body must be simple (only contain certain op types).
    for (Operation &op : n->getThenRegion().front().getOperations())
      if (!isa<mlir::arith::CmpIOp, scf::IfOp, mlir::arith::ConstantIndexOp,
               scf::YieldOp>(op))
        return false;
  }

  if (n->getElseNode()->isLeaf()) {
    Block &bb = scf_if.getElseRegion().front();
    if (bb.getOperations().size() != 1) return false;
    // Check that the yielded value does not depend on the loop iterator
    // arguments.
    worklist.push_back(bb.getTerminator()->getOperand(0));
  } else {
    // Check loop invariance recursively inside the else-branch.
    auto else_node = n->getElseNode();
    if (else_node->getNumberOfChildren() != 1 ||
        !isLoopInvariant(else_node->getFirstChild(), parent_for_op))
      return false;
    // The else-branch body must be simple (only contain certain op types).
    for (Operation &op : n->getElseRegion().front().getOperations())
      if (!isa<mlir::arith::CmpIOp, scf::IfOp, mlir::arith::ConstantIndexOp,
               scf::YieldOp>(op))
        return false;
  }

  // Check the condition for dependencies.
  worklist.push_back(scf_if.getCondition());

  while (!worklist.empty()) {
    Value v = worklist.back();
    worklist.pop_back();
    if (isa<BlockArgument>(v)) {
      Operation *for_op = cast<BlockArgument>(v).getOwner()->getParentOp();
      if (for_op == parent_for_op) return false;
    } else {
      // Check dependencies in each of the operands
      Operation *defining_op = v.getDefiningOp();
      for (auto opnd : v.getDefiningOp()->getOperands()) {
        worklist.push_back(opnd);
      }
    }
  }

}
// ---- 350/384  matchAndRewrite  —  dcc/src/Transform/Dataflow/DuplicateReusedToggle.cpp:33  (170L)
LogicalResult e350_matchAndRewrite(
    mlir::dataflow::GetLogicalMemoryViewOp mem_view_op,
    PatternRewriter &rewriter) {
  auto toggle_op = dyn_cast_or_null<arith::SubIOp>(
      mem_view_op.getStartAddress().getDefiningOp());
  if (!toggle_op) return failure();

  auto iter_arg = dyn_cast<BlockArgument>(toggle_op->getOperand(1));
  DT_CHECK_MSG(iter_arg,
               "expecting second operand of toggle op to be an iter_arg");

  // Any use of the toggle op that isn't either a yield operation or the memory
  // view needs to be duplicated.
  SmallVector<Operation *> candidates;
  auto toggle_parent = iter_arg.getOwner()->getParentOp();
  DT_CHECK(toggle_parent);
  for (auto user : toggle_op->getUsers()) {
    if (isa<affine::AffineYieldOp, scf::YieldOp>(user))
      DT_CHECK_MSG(user->getParentOp() == toggle_parent,
                   "[DuplicateReusedToggle] Toggle user is a yield for an "
                   "operation other than owning loop");
    else if (user != mem_view_op)
      candidates.push_back(user);
  }
  if (candidates.empty()) return failure();
  LLVM_DEBUG(llvm::dbgs() << "[DuplicateReusedToggle] Duplicating toggle "
                          << candidates.size() << " times:\n"
                          << *toggle_op << "\n");

  // Collect the iter_arg chain from the iter_arg used in the toggle to its
  // initializer. The args will be in innermost to outermost order.
  Value init_val = nullptr;
  auto iter_arg_chain =
      mlir::dataflow::utils::getIterArgChain(iter_arg, init_val);
  DT_CHECK(!iter_arg_chain.empty());
  DT_CHECK(dcc::utils::isConstant<arith::ConstantOp>(init_val));
  LLVM_DEBUG(llvm::dbgs() << "[DuplicateReusedToggle] Adding iter_args to "
                          << iter_arg_chain.size() << " loops\n");

  // For each loop involved in the iter_arg chain, add new iter_args for each
  // candidate. Iterate through the args in reverse to execute from outermost
  // to innermost loop.
  IRMapping ir_map;
  Block::BlockArgListType prev_upd_iter_args;
  Operation *prev_yield_op = nullptr;
  Block::BlockArgListType upd_iter_args;
  int num_orig_args = -1;
  for (int b = iter_arg_chain.size() - 1, i = b; i >= 0; --i) {
    // Clone the loop, adding an iter_arg for each candidate.
    auto loop = iter_arg_chain[i].getOwner()->getParentOp();
    auto new_loop = dcc::utils::createForOpWithAdditionalReturnValue(
        loop, candidates.size(), ir_map, /* delete_op */ false);

    // Collect the new iter_args added. These need to be properly initialized.
    Operation *yield_op = nullptr;
    if (auto affine_for = dyn_cast<affine::AffineForOp>(new_loop)) {
      upd_iter_args = affine_for.getRegionIterArgs();
      yield_op = affine_for.getBody()->getTerminator();
      DT_CHECK(isa<affine::AffineYieldOp>(yield_op));
    } else {
      auto scf_for = dyn_cast<scf::ForOp>(new_loop);
      DT_CHECK(scf_for);
      upd_iter_args = scf_for.getRegionIterArgs();
      yield_op = scf_for.getBody()->getTerminator();
      DT_CHECK(isa<scf::YieldOp>(yield_op));
    }

    num_orig_args = upd_iter_args.size() - candidates.size();
    DT_CHECK(num_orig_args > 0);

    // The outer loop has a slightly different transformation path than the
    // inner loops as it initializes the toggle pattern. The expected pattern of
    // the loop nest is as follows:
    //   - The outer most loop results should not have any uses. Otherwise, we
    //     wouldn't know what to do with the new results we will introduce as a
    //     part of the toggle duplication.
    //   - Loops will yield the results from the inner loops to feed the toggle
    //     chain. This propagates the toggle calculation through the loop nest.
    if (i != b) {
      // INNER LOOPS
      DT_CHECK(!prev_upd_iter_args.empty() && prev_yield_op);

      // Replace uses of original loop's results with the new ones. These should
      // only be yield operations.
      for (unsigned j = 0, e = loop->getNumResults(); j < e; ++j) {
        auto res = loop->getResult(j);
        DT_CHECK(res.getUsers().empty() ||
                 bool(isa<affine::AffineYieldOp, scf::ForOp>(
                     *(res.getUsers().begin()))));
        rewriter.replaceAllUsesWith(res, new_loop->getResult(j));
      }

      // The new iter_args for this loop should be initialized to the new
      // iter_args added to the previous created loop to continue the chain.
      for (unsigned j = num_orig_args,
                    k = prev_upd_iter_args.size() - candidates.size(),
                    e = upd_iter_args.size();
           j < e; ++j, ++k)
        mlir::dataflow::utils::setIterArgInit(upd_iter_args[j],
                                              prev_upd_iter_args[k]);

      // The yield op of the previous created parent loop needs to yield the
      // results associated to the new iter_args of the current loop.
      int operand_idx = prev_yield_op->getNumOperands() - candidates.size();
      DT_CHECK(operand_idx > 0);
      for (unsigned j = operand_idx, k = num_orig_args,
                    e = prev_yield_op->getNumOperands();
           j < e; ++j, ++k)
        prev_yield_op->setOperand(j, new_loop->getResult(k));
    } else {
      // OUTERMOST LOOPS
      DT_CHECK(prev_upd_iter_args.empty() && !prev_yield_op);
      // The outermost loop is expected to have no uses of the results. If there
      // were uses, we would have to be able to determine how they were used to
      // figure out how to use the new additional results properly.
      for (unsigned j = 0, e = loop->getNumResults(); j < e; ++j)
        DT_CHECK(loop->getResult(j).getUsers().empty());

      // The new iter_args in the new outer loop should be initialized to the
      // same value as the original toggle.
      for (unsigned j = num_orig_args, e = upd_iter_args.size(); j < e; ++j)
        mlir::dataflow::utils::setIterArgInit(upd_iter_args[j], init_val);
    }

    // Set prev_new_iter_args and prev_yield_op so the next loop in the nest can
    // appropriately initialize and yield.
    prev_upd_iter_args = upd_iter_args;
    prev_yield_op = yield_op;

    // Update toggle_op, iter_arg_chain, and candidates from the map.
    auto mapped_toggle_op = ir_map.lookupOrNull(toggle_op);
    DT_CHECK(mapped_toggle_op);
    toggle_op = cast<arith::SubIOp>(mapped_toggle_op.getDefiningOp());

    for (int j = i; j >= 0; --j) {
      auto new_arg = ir_map.lookupOrNull(iter_arg_chain[j]);
      DT_CHECK(new_arg);
      iter_arg_chain[j] = cast<BlockArgument>(new_arg);
    }

    for (int j = 0; j < candidates.size(); ++j) {
      candidates[j] = ir_map.lookupOrNull(candidates[j]);
      DT_CHECK(candidates[j]);
    }

    // Everything has been updated, delete this loop.
    rewriter.eraseOp(loop);
  }

  // Now that the new loop structures are in place, duplicate the toggle for
  // each candidate.
  OpBuilder builder(toggle_op);
  builder.setInsertionPointAfter(toggle_op);
  for (int i = 0, e = candidates.size(); i < e; ++i) {
    // Duplicate the toggle op.
    auto new_toggle_op = builder.clone(*toggle_op);
    builder.setInsertionPointAfter(new_toggle_op);

    // Update the iter_arg used in the new toggle to one of the new ones.
    auto new_arg = upd_iter_args[num_orig_args + i];
    new_toggle_op->setOperand(1, new_arg);

    // Update the candidate to use the new toggle operation.
    candidates[i]->replaceUsesOfWith(toggle_op.getResult(),
                                     new_toggle_op->getResult(0));

    // Update the yield operation to yield the new toggle instead of the
    // iter_arg used by the toggle.
    prev_yield_op->setOperand(num_orig_args + i, new_toggle_op->getResult(0));
  }

}
// ---- 351/384  runOnOperation  —  dcc/src/Transform/Dataflow/MutableAddrSplitting.cpp:226  (70L)
void e351_runOnOperation() {
  if (DisableThisPass) return;

  auto isCandidateMemView =
      [](mlir::dataflow::GetLogicalMemoryViewOp mem_view_op) -> bool {
    auto mem_view_unit =
        dcc::getUnitType(mem_view_op.getFromUnit().getDefiningOp());
    if (mem_view_unit != SenComponents::HBM) return false;

    return !isa_and_nonnull<mlir::symbol::CreateSymbolOp,
                            mlir::symbol::SymbolQueryMapOp>(
        mem_view_op.getStartAddress().getDefiningOp());
  };

  ModuleOp module_op = getOperation();
  std::vector<MASCandidate> candidates;
  std::unordered_set<Operation *> analyzed_candidates;
  module_op.walk<WalkOrder::PreOrder>([&](dataflow::ProgramUnitOp unit) {
    // Reset num_conditionals_ so every unit is not exceeding the max
    // conditionals.
    num_conditionals_ = 0;
    auto comp = dcc::getUnitType(
        unit.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
    if (!is_any_of(comp, SenComponents::L3LU, SenComponents::L3SU))
      return WalkResult::advance();
    unit.walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto mem_view = dyn_cast<dataflow::GetLogicalMemoryViewOp>(op);
      if (!mem_view || !isCandidateMemView(mem_view))
        return WalkResult::advance();
      DT_CHECK(mem_view->hasOneUse());

      agen::MemoryOperandIndex mem_index = agen::MemoryOperandIndex::kMax;
      auto mem_op = *mem_view->getUsers().begin();
      if (auto comp_las = dyn_cast<agen::CompositeLoadAndStoreOp>(mem_op)) {
        mem_index = comp_las.getSrcMemRef() == mem_view.getResult()
                        ? agen::MemoryOperandIndex::kDirSrc
                        : agen::MemoryOperandIndex::kDirDst;
      } else if (auto comp_ind_las =
                     dyn_cast<agen::CompositeIndirectLoadAndStoreOp>(mem_op)) {
        if (comp_ind_las.getDirectSrcMemref() == mem_view.getResult())
          mem_index = agen::MemoryOperandIndex::kDirSrc;
        else if (comp_ind_las.getDirectDstMemref() == mem_view.getResult())
          mem_index = agen::MemoryOperandIndex::kDirDst;
      } else {
        mem_index = agen::MemoryOperandIndex::kDirSrc;
      }
      DT_CHECK_MSG(mem_index != agen::MemoryOperandIndex::kMax,
                   "Invalid HBM memory operand.");

      auto res = analyzed_candidates.insert(mem_op);
      DT_CHECK_MSG(res.second,
                   "Data transfers should only have one HBM memory operand.");
      candidates.emplace_back(mem_op, unit, comp, mem_index);
      return WalkResult::advance();
    });
    return WalkResult::advance();
  });

  for (auto &candidate : candidates) {
    if (isa<agen::VectorLoadOp>(candidate.op_))
      transformVectorLoad(candidate);
    else if (isa<agen::VectorStoreOp>(candidate.op_))
      transformVectorStore(candidate);
    else if (isa<agen::CompositeLoadAndStoreOp>(candidate.op_))
      transformCompLoadAndStore(candidate);
    else if (isa<agen::CompositeIndirectLoadAndStoreOp>(candidate.op_))
      transformCompIndLoadAndStore(candidate);
    else
      llvm_unreachable("Unexpected candidate operation.");
  }

}
// ---- 352/384  transformVectorLoad  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:201  (25L)
void e352_transformVectorLoad(
    MSASCandidate &candidate) {
  auto op = dyn_cast<agen::VectorLoadOp>(candidate.op_);
  DT_CHECK(op);

  // Collect the relevant access details.
  agen::AccessContainer<agen::AccessDetailsAffine> access_details;
  agen::AccessDetailsAffine &ad =
      access_details.emplace_insert(candidate.mem_index_, op, candidate.comp_);
  auto result = ad.constructDetails(candidate.mem_index_);
  DT_CHECK(succeeded(result));

  auto mem_view_op =
      cast<dataflow::GetLogicalMemoryViewOp>(op.getMemRef().getDefiningOp());
  auto new_subscripts_map =
      shiftMutableAddr(candidate.unit_, op, mem_view_op, ad);
  // If the new subscripts map is empty that means there was no shifting
  // required.
  if (new_subscripts_map.isEmpty()) return;

  OpBuilder builder(op);
  auto new_mem_op = op.cloneWithNewAccessInfo(
      builder, mem_view_op, new_subscripts_map, ad.getIndices());

  if (!op->use_empty()) op->replaceAllUsesWith(new_mem_op);

}
// ---- 353/384  transformVectorStore  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:229  (24L)
void e353_transformVectorStore(
    MSASCandidate &candidate) {
  auto op = dyn_cast<agen::VectorStoreOp>(candidate.op_);
  DT_CHECK(op);

  // Collect the relevant access details.
  agen::AccessContainer<agen::AccessDetailsAffine> access_details;
  agen::AccessDetailsAffine &ad =
      access_details.emplace_insert(candidate.mem_index_, op, candidate.comp_);
  auto result = ad.constructDetails(candidate.mem_index_);
  DT_CHECK(succeeded(result));

  auto mem_view_op =
      cast<dataflow::GetLogicalMemoryViewOp>(op.getMemRef().getDefiningOp());
  auto new_subscripts_map =
      shiftMutableAddr(candidate.unit_, op, mem_view_op, ad);
  // If the new subscripts map is empty that means there was no shifting
  // required.
  if (new_subscripts_map.isEmpty()) return;

  OpBuilder builder(op);
  (void)op.cloneWithNewAccessInfo(builder, mem_view_op, new_subscripts_map,
                                  ad.getIndices());


}
// ---- 354/384  transformCompLoadAndStore  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:256  (39L)
void e354_transformCompLoadAndStore(
    MSASCandidate &candidate) {
  auto op = dyn_cast<agen::CompositeLoadAndStoreOp>(candidate.op_);
  DT_CHECK(op);

  // Collect the relevant access details.
  agen::AccessContainer<agen::AccessDetailsAffineComposite> access_details;
  agen::AccessDetailsAffineComposite &ad =
      access_details.emplace_insert(candidate.mem_index_, op, candidate.comp_);
  auto result = ad.constructDetails(candidate.mem_index_);
  DT_CHECK(succeeded(result));

  auto mem_view_op = dyn_cast_or_null<dataflow::GetLogicalMemoryViewOp>(
      ad.getMemRef().getDefiningOp());
  DT_CHECK(mem_view_op);

  auto new_subscripts_map =
      shiftMutableAddr(candidate.unit_, op, mem_view_op, ad);
  // If the new subscripts map is empty that means there was no shifting
  // required.
  if (new_subscripts_map.isEmpty()) return;

  OpBuilder builder(op);
  if (candidate.mem_index_ == agen::MemoryOperandIndex::kDirSrc) {
    SmallVector<Value, 16> dst_indices;
    for (auto index : op.getDstMapIndices()) dst_indices.push_back(index);
    (void)op.cloneWithNewAccessInfo(
        builder, op.getSrcMemRef(), op.getDstMemRef(), new_subscripts_map,
        op.getDstAffineMapAttr().getValue(), ad.getIndices(), dst_indices,
        op.getTimeSet().getValue());
  } else {
    SmallVector<Value, 16> src_indices;
    for (auto index : op.getSrcMapIndices()) src_indices.push_back(index);
    (void)op.cloneWithNewAccessInfo(
        builder, op.getSrcMemRef(), op.getDstMemRef(),
        op.getSrcAffineMapAttr().getValue(), new_subscripts_map, src_indices,
        ad.getIndices(), op.getTimeSet().getValue());
  }


}
// ---- 355/384  transformCompIndLoadAndStore  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:298  (54L)
void e355_transformCompIndLoadAndStore(
    MSASCandidate &candidate) {
  auto op = dyn_cast<agen::CompositeIndirectLoadAndStoreOp>(candidate.op_);
  DT_CHECK(op);

  // The memory operands on the side of the indirect must have a 0 immutable
  // address so no shifting can be done.
  if (candidate.mem_index_ == agen::MemoryOperandIndex::kDirSrc &&
      op.hasIndirectSrc())
    return;
  if (candidate.mem_index_ == agen::MemoryOperandIndex::kDirDst &&
      op.hasIndirectDst())
    return;

  // Collect the relevant access details.
  agen::AccessContainer<agen::AccessDetailsAffineComposite> access_details;
  agen::AccessDetailsAffineComposite &ad =
      access_details.emplace_insert(candidate.mem_index_, op, candidate.comp_);
  auto result = ad.constructDetails(candidate.mem_index_);
  DT_CHECK(succeeded(result));

  auto mem_view_op = dyn_cast_or_null<dataflow::GetLogicalMemoryViewOp>(
      ad.getMemRef().getDefiningOp());
  DT_CHECK(mem_view_op);

  auto new_subscripts_map =
      shiftMutableAddr(candidate.unit_, op, mem_view_op, ad);
  // If the new subscripts map is empty that means there was no shifting
  // required.
  if (new_subscripts_map.isEmpty()) return;

  OpBuilder builder(op);
  if (candidate.mem_index_ == agen::MemoryOperandIndex::kDirSrc) {
    SmallVector<Value, 16> dst_indices;
    for (auto index : op.getDirectDstMapIndices()) dst_indices.push_back(index);
    (void)op.cloneWithNewAccessInfo(
        builder, op.getIndirectSrcMemref(), op.getDirectSrcMemref(),
        op.getIndirectDstMemref(), op.getDirectDstMemref(),
        builder.getEmptyAffineMap(), new_subscripts_map,
        op.getIndirectDstAffineMapAttr().getValue(),
        op.getDirectDstAffineMapAttr().getValue(), ad.getIndices(), dst_indices,
        op.getTimeSet().getValue());
  } else {
    SmallVector<Value, 16> src_indices;
    for (auto index : op.getDirectSrcMapIndices()) src_indices.push_back(index);
    (void)op.cloneWithNewAccessInfo(
        builder, op.getIndirectSrcMemref(), op.getDirectSrcMemref(),
        op.getIndirectDstMemref(), op.getDirectDstMemref(),
        op.getIndirectSrcAffineMapAttr().getValue(),
        op.getDirectSrcAffineMapAttr().getValue(), builder.getEmptyAffineMap(),
        new_subscripts_map, src_indices, ad.getIndices(),
        op.getTimeSet().getValue());
  }


}
// ---- 356/384  transform  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:592  (39L)
LogicalResult e356_transform() {
  LLVM_DEBUG(
      llvm::dbgs()
      << "[TransformPagedMemViewPass] Transforming for paged mem views\n");
  for (auto &info : tpmv_info_) {
    LLVM_DEBUG(llvm::dbgs()
               << "[TransformPagedMemViewPass] Working on a memory operand\n");
    if (!info.paged_mem_view_) continue;
    dcc::agen::utils::replaceConstOpsInSubscriptsMap(info.subscripts_map_,
                                                     info.indices_);

    // Create a copy of the subscripts_map that represents loop iterators as
    // symbols. This is used to form the constraints to determine which pages
    // are valid for mem_ops_.
    AffineMap subscripts_map_sym =
        replaceDimsInMapWithSyms(info.subscripts_map_);

    // Ranges of the loop iterators are used to only choose pages within the
    // loop iteration space.
    calculateIndicesRanges(info.indices_, info.indices_ranges_);

    SmallVector<Operation *, 16> new_mem_ops;
    if (analyzeAndConstructValidPages(new_mem_ops, info, subscripts_map_sym)
            .failed())
      return LogicalResult::failure();

    for (auto &mem_op : mem_ops_) eraseMemOpAndUseChain(mem_op);
    // There will only be one paged_mem_view per TPMVInfo object at any given
    // time because we don't clone paged_mem_views during the lowering process.
    info.paged_mem_view_->erase();
    info.paged_mem_view_ = nullptr;

    // Re-populate mem_ops_ with the new memory ops we created. Because we could
    // be introducing conditionals for multiple memory operands, the mem_op may
    // get cloned into multiple conditional branches.
    mem_ops_ = new_mem_ops;
  }

  return LogicalResult::success();


}
// ==================================================================================================
// LEVEL 7
// ==================================================================================================

// ---- 357/384  generateAffineAddressManipulationStmts  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:625  (381L)
AgenToSentientLoweringPass::generateAffineAddressManipulationStmts(
    OpBuilder& builder, SenComponents comp,
    AccessContainer<AccessDetailsTy>& access_details,
    AccessContainer<Value>& mutable_addrs_base,
    AccessContainer<Value>& mutable_addrs,
    const DenseMap<Value, std::vector<int64_t>>& indices_coeff_dict) {
  DT_CHECK(mutable_addrs_base.size() == mutable_addrs.size() &&
           mutable_addrs.size() == access_details.size());
  LLVM_DEBUG({
    llvm::dbgs() << "\n--------------\nindices_coeff_dict:\n";
    for (auto& entry : indices_coeff_dict) {
      if (entry.first && isa<BlockArgument>(entry.first))
        llvm::dbgs()
            << "entry.first:" << entry.first << " from loop level:"
            << getLoopNestLevel(
                   cast<BlockArgument>(entry.first).getOwner()->getParentOp())
            << "\n";
      else
        llvm::dbgs() << "entry.first:" << entry.first << "\n";
      llvm::dbgs() << "entry.second:[";
      for (int i = 0; i < entry.second.size(); i++)
        llvm::dbgs() << entry.second[i]
                     << ((i + 1) == entry.second.size() ? "]" : ", ");
      llvm::dbgs() << "\n";
    }
    llvm::dbgs() << "\n--------------\n";
  });

  // Algorithm to insert the address register pointers.
  // 1.Sort the indices based on the loop nest
  // 2. Pick the outermost and insert an assignment before the outermost
  //  with the constant value
  // 3. Traverse from outermost used loop iterator till the last one before
  // 3.1 for them, insert assignment statements
  // 3.2 also insert add statements at the loop body ending
  // 4. for the last one, insert add statements after the load operation

  // Sort the indices from outermost to innnermost
  SmallVector<std::pair<Value, std::vector<int64_t>>> sorted_indices_coeff_pair;
  for (const auto& record : indices_coeff_dict) {
    if (record.first != nullptr) {
      sorted_indices_coeff_pair.push_back({record.first, record.second});
    }
  }

  // sort indices+coeff pairs so the outermost forOp is at position 0.
  llvm::sort(
      sorted_indices_coeff_pair,
      [&](std::pair<Value, std::vector<int64_t>> left,
          std::pair<Value, std::vector<int64_t>> right) -> bool {
        auto* left_op =
            mlir::cast<BlockArgument>(left.first).getOwner()->getParentOp();
        auto* right_op =
            mlir::cast<BlockArgument>(right.first).getOwner()->getParentOp();
        auto val = left_op->isProperAncestor(right_op);
        return val;
      });

  if (!sorted_indices_coeff_pair.empty()) {
    auto last_loop =
        mlir::cast<BlockArgument>(sorted_indices_coeff_pair.front().first)
            .getOwner()
            ->getParentOp();

    DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs()
                                      << "last_loop level:"
                                      << getLoopNestLevel(last_loop) << "\n");

    // Check if mutable base address value are defined within the outer for-loop
    // If the start address of a logical memory view are inside a loop, and if
    // this loop gets "cloned" to add a loop iterator argument in later steps,
    // then the reference to the start address becomes stale. Hence, we
    // need to retrieve its equivalent one after cloning.
    for (int i = 0; i < mutable_addrs_base.size(); i++) {
      Operation* def_op;
      const auto& base = mutable_addrs_base[i];
      if (!mlir::isa<BlockArgument>(base)) {
        def_op = base.getDefiningOp();
      } else {
        def_op = mlir::cast<BlockArgument>(base).getOwner()->getParentOp();
        DT_CHECK((bool)(isa<affine::AffineForOp, scf::ForOp>(def_op)));
      }

      if (!last_loop->isAncestor(def_op)) continue;

      // Add mutable addr index into operation so that we can index into
      // mutable_addr_base when retrieving.
      def_op->setAttr("index", builder.getI32IntegerAttr(i));

      DEBUG_WITH_TYPE(VerboseDebug, {
        if (!isa<affine::AffineForOp>(def_op))
          llvm::dbgs() << "marked op with 'index' attr:" << i
                       << "op:" << *def_op << "\n";
      });

      // if the def_op is a for-loop, we additionally have iter-index to
      // indicate which iterator argument will be used later.
      if (auto for_op = llvm::dyn_cast<affine::AffineForOp>(def_op)) {
        DEBUG_WITH_TYPE(
            VerboseDebug,
            llvm::dbgs() << "marked loop at level "
                         << dcc::utils::getLoopNestLevel<affine::AffineForOp>(
                                for_op)
                         << " with 'index' attr:" << i << ".\n");

        for (int j = 0; j < for_op.getNumRegionIterArgs(); j++) {
          auto& arg = for_op.getRegionIterArgs()[j];
          if (arg == base) {
            DEBUG_WITH_TYPE(VerboseDebug,
                            llvm::dbgs()
                                << "marked above loop: " << arg
                                << " with 'iter-index' attr: " << j << "\n");
            def_op->setAttr("iter-index", builder.getI32IntegerAttr(j));
            break;
          }
        }
      } else if (auto for_op = llvm::dyn_cast<scf::ForOp>(def_op)) {
        DEBUG_WITH_TYPE(VerboseDebug,
                        llvm::dbgs()
                            << "marked loop at level "
                            << dcc::utils::getLoopNestLevel<scf::ForOp>(for_op)
                            << " with 'index' attr:" << i << ".\n");

        for (int j = 0; j < for_op.getNumRegionIterArgs(); j++) {
          auto& arg = for_op.getRegionIterArgs()[j];
          if (arg == base) {
            DEBUG_WITH_TYPE(VerboseDebug,
                            llvm::dbgs()
                                << "marked above loop: " << arg
                                << " with 'iter-index' attr: " << j << "\n");
            def_op->setAttr("iter-index", builder.getI32IntegerAttr(j));
            break;
          }
        }
      }
    }
  } else {
    // Memory ops with address offsets that consist purely of a single constant
    // value simply need to adjust the mutable/immutable addr operands
    // appropriately. This function will modify the mutable address operand.
    // Other functions will modify the immutable address operand if required.
    // How the mutable/immutable address is modified depends on the unit.
    // L3: The mutable address holds the memory op offset and the immutable
    //     address holds the memory view start address.
    // LX below: The mutable address holds the combined value of the memory op
    //           offset and the immutable address should start at 0.
    for (int i = 0; i < mutable_addrs_base.size(); ++i) {
      DT_CHECK(indices_coeff_dict.find(nullptr) != indices_coeff_dict.end());

      auto insert_pt = builder.saveInsertionPoint();

      // Set the insertion point for the OpBuilder as close to the defining
      // operation as possible.
      Operation* def_op = mutable_addrs[i].getDefiningOp();
      if (auto block_arg = dyn_cast<BlockArgument>(mutable_addrs[i])) {
        def_op = block_arg.getOwner()->getParentOp();
        builder.setInsertionPointToStart(&def_op->getRegion(0).front());
      } else {
        DT_CHECK_MSG(def_op, "could not set def_op");
        builder.setInsertionPointAfter(def_op);
      }

      int init_value = indices_coeff_dict.find(nullptr)->getSecond()[i];
      if (is_any_of(comp, L3LU, L3SU)) {
        LLVM_DEBUG(llvm::dbgs()
                   << "overwriting mutable_addrs_base and mutable_addrs [" << i
                   << "] with:" << init_value << "\n");
        mutable_addrs[i] = mutable_addrs_base[i] =
            mlir::arith::ConstantIndexOp::create(
                builder, builder.getInsertionPoint()->getLoc(), init_value);
      } else if (init_value != 0) {
        // If the mutable is a constant, the constant can just be directly
        // updated. Otherwise, an addOp is required to properly adjust the
        // address. If the init_value is 0, no update needed.
        if (auto const_op =
                dyn_cast_or_null<mlir::arith::ConstantIndexOp>(def_op)) {
          auto offset_value = const_op.value();
          int64_t new_init_value = init_value + offset_value;
          LLVM_DEBUG(llvm::dbgs()
                     << "overwriting mutable_addrs_base and mutable_addrs ["
                     << i << "] with: " << "(" << init_value << " + "
                     << offset_value << ")" << new_init_value << "\n");
          builder.setInsertionPointAfter(def_op);
          mutable_addrs[i] = mutable_addrs_base[i] =
              mlir::arith::ConstantIndexOp::create(
                  builder, builder.getInsertionPoint()->getLoc(),
                  new_init_value);
        } else {
          LLVM_DEBUG(llvm::dbgs()
                     << "overwriting mutable_addrs_base and mutable_addrs ["
                     << i << "] with: " << mutable_addrs[i] << " + "
                     << init_value << "\n");
          auto init_const = mlir::arith::ConstantIndexOp::create(
              builder, builder.getInsertionPoint()->getLoc(), init_value);
          mutable_addrs[i] = mutable_addrs_base[i] =
              mlir::arith::AddIOp::create(
                  builder, builder.getInsertionPoint()->getLoc(),
                  mutable_addrs[i], init_const.getResult());
        }
      }
      builder.restoreInsertionPoint(insert_pt);
    }
    return LogicalResult::success();
  }

  // Create new loops with additional return values.
  // the forOp creation is in PostOrder (innermost first) to keep indices
  // tracking valid.
  Operation* outer_most_loop = nullptr;
  int n_outer_loop_extra_vars = 0;
  for (int i = sorted_indices_coeff_pair.size() - 1; i >= 0; --i) {
    auto loop_op = mlir::cast<BlockArgument>(sorted_indices_coeff_pair[i].first)
                       .getOwner()
                       ->getParentOp();
    n_outer_loop_extra_vars = sorted_indices_coeff_pair[i].second.size();
    DEBUG_WITH_TYPE(VerboseDebug,
                    llvm::dbgs()
                        << "creating a new loop based on loop at level "
                        << getLoopNestLevel(loop_op) << "\n");
    IRMapping ir_map;
    outer_most_loop = dcc::utils::createForOpWithAdditionalReturnValue(
        loop_op, n_outer_loop_extra_vars, ir_map);
    // Mark the new loop so that we can process it later.
    outer_most_loop->setAttr("marked", builder.getI32IntegerAttr(1));
    DEBUG_WITH_TYPE(VerboseDebug,
                    llvm::dbgs() << "new loop is at level "
                                 << getLoopNestLevel(outer_most_loop) << "\n");
  }

  // Add initialization statement
  if (outer_most_loop) {
    SmallVector<Operation*, 16> for_ops;
    outer_most_loop->walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (isa<affine::AffineForOp, scf::ForOp>(op)) {
        if (op->hasAttr("marked")) {
          for_ops.push_back(op);
        }

        op->removeAttr("marked");
      }
      if (op->hasAttr("index")) {
        // Get the information to index into mutable_addr
        auto val = mlir::cast<IntegerAttr>(op->getAttr("index")).getInt();

        // If it's a loop, update the mutable addr base with the iterator
        // argument indexed by "iter-index".
        if (auto loop = llvm::dyn_cast<affine::AffineForOp>(op)) {
          DEBUG_WITH_TYPE(
              VerboseDebug,
              llvm::dbgs() << "found loop at level "
                           << dcc::utils::getLoopNestLevel<affine::AffineForOp>(
                                  loop)
                           << " with 'index' attr.\n");
          DT_CHECK(
              op->hasAttr("iter-index") &&
              "perhaps main IV is used as a subscript instead of an iter-arg");
          auto iter_idx =
              mlir::cast<IntegerAttr>(op->getAttr("iter-index")).getInt();
          DEBUG_WITH_TYPE(VerboseDebug,
                          llvm::dbgs() << "iter-index = " << iter_idx << "\n");

          mutable_addrs_base[val] = loop.getRegionIterArgs()[iter_idx];
          op->removeAttr("iter-index");
        } else if (auto scf_loop = llvm::dyn_cast<scf::ForOp>(op)) {
          DT_CHECK(
              op->hasAttr("iter-index") &&
              "perhaps main IV is used as a subscript instead of an iter-arg");
          auto iter_idx =
              mlir::cast<IntegerAttr>(op->getAttr("iter-index")).getInt();
          mutable_addrs_base[val] = scf_loop.getRegionIterArgs()[iter_idx];
          op->removeAttr("iter-index");
        } else {
          // update with the result of operation.
          mutable_addrs_base[val] = op->getResult(0);
        }

        access_details[val].setMemViewStartAddr(mutable_addrs_base[val]);
        op->removeAttr("index");
      }
    });

#if defined(TOGGLE_INDIRECT_IMPL1)
    // The mem_view_start addr of an IBR access may be non-constant (eg. in case
    // of toggling JCR). As such we need to add it to the initial value of the
    // iter-arg corresponding to the kIndSrc/Dst immutable_addr. However the
    // operation calculating this start addr may be invariant at a certain level
    // within the outermost loop, so the initialization must happen at the
    // outermost *invariant* level.
    //
    // The following is a list of pairs where the first entry of the pair is the
    // index into the access details and the second is the original value of the
    // mem_view_start_addr. This list is used in a post-processing step to patch
    // up the initialization of corresponding iter-arg at the right loop level.
    SmallVector<std::pair<int, Value>> post_process_list;
#endif
    // Create initialization statement for each variable added to the outerloop.
    for (int i = 0; i < n_outer_loop_extra_vars; i++) {
      DT_CHECK(indices_coeff_dict.find(nullptr) != indices_coeff_dict.end());
      int init_value = indices_coeff_dict.find(nullptr)->getSecond()[i];
#if defined(TOGGLE_INDIRECT_IMPL1)
      if (access_details[i].getMemoryIndex() == MemoryOperandIndex::kIndSrc ||
          access_details[i].getMemoryIndex() == MemoryOperandIndex::kIndDst) {
        auto cv = llvm::dyn_cast_or_null<arith::ConstantOp>(
            mutable_addrs_base[i].getDefiningOp());
        if (cv) {
          if (mlir::dyn_cast<IntegerAttr>(cv.getValue()).getInt() != 0)
            post_process_list.push_back(
                std::make_pair(i, mutable_addrs_base[i]));
        } else if (isa<arith::SubIOp, symbol::CreateSymbolOp>(
                       mutable_addrs_base[i].getDefiningOp()))  // toggle
          post_process_list.push_back(std::make_pair(i, mutable_addrs_base[i]));
        else
          llvm_unreachable(
              "unexpected mem_view_start addr index for kIndSrc/Dst");
      }
#endif
      mutable_addrs_base[i] = this->insertInitializationStmt(
          comp, outer_most_loop, init_value, mutable_addrs_base[i]);
      mutable_addrs[i] = mutable_addrs_base[i];
    }

    // add copy and add statements at the end of each loop.
    for (unsigned dim_index = 0; dim_index < for_ops.size(); dim_index++) {
      for (int i = 0; i < n_outer_loop_extra_vars; i++) {
        int add_value = sorted_indices_coeff_pair[dim_index].second[i];
        mutable_addrs[i] = insertCopyAndAddStmts(for_ops[dim_index], i,
                                                 mutable_addrs[i], add_value);
      }
    }
#if defined(TOGGLE_INDIRECT_IMPL1)
    auto findInvariantLoop = [&](Operation* init) -> Operation* {
      if (isa<arith::ConstantOp, symbol::CreateSymbolOp>(init)) {
        if (outer_most_loop->isAncestor(init))
          init->moveBefore(outer_most_loop);
        return outer_most_loop;
      } else {
        // Going from inner to outermost loop, find the closest outer loop
        // containing the load/store operation such that init is invariant to
        // it. We look for an ancestor loop that is common between load/store
        // and init keeping track of the last parent loop of the load/store.
        // Once a common ancestor is found, the loop we are looking for is the
        // last parent loop of load/store.
        Operation* loop_containing_init = nullptr;
        Operation* parent = init;
        while (parent) {
          if (isa<affine::AffineForOp, scf::ForOp>(parent)) {
            loop_containing_init = parent;
            break;
          }
          parent = parent->getParentOp();
        }
        if (!loop_containing_init) return outer_most_loop;

        parent = for_ops.back();
        Operation* prev_loop = nullptr;
        while (parent) {
          if (parent == loop_containing_init) return prev_loop;
          if (isa<affine::AffineForOp, scf::ForOp>(parent)) prev_loop = parent;
          parent = parent->getParentOp();
        }
      }
      return nullptr;
    };
    for (auto v : post_process_list) {
      Operation* init = v.second.getDefiningOp();
      DT_CHECK_MSG(init, "expected operation for mem_view_start addr");
      Operation* loop = findInvariantLoop(init);
      if (loop) {
        DT_CHECK(outer_most_loop->isAncestor(loop));
        OpBuilder builder(loop);
        auto old_init = loop->getOperand(loop->getNumOperands() - v.first - 1);
        auto new_init = mlir::arith::AddIOp::create(builder, loop->getLoc(),
                                                    builder.getIndexType(),
                                                    old_init, v.second);
        loop->setOperand(loop->getNumOperands() - v.first - 1, new_init);
        mutable_addrs_base[v.first] = new_init;
      } else {
        init->emitWarning(
            "unable to find the right loop to add mem_view_start addr of "
            "indirect src/dst");
        return LogicalResult::failure();

#endif
}}}}
// ---- 358/384  constructLoadAndSendStmt  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:1910  (104L)
LogicalResult e358_constructLoadAndSendStmt(
    OpBuilder* builder, dataflow::ProgramUnitOp unit_op, Operation* op,
    AccessDetailsTy& access_details, Value& mutable_addr, Value& immutable_addr,
    unsigned burst_size, unsigned group_size, int stride_step,
    Operation* outermost_comp_loop, Operation* extract_op) {
  // It is guaranteed from the upstream passes that all units of a unit
  // operation will have same types.
  auto comp = dcc::getUnitType(
      unit_op.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());

  auto& mem_view_start_addr = immutable_addr;
  auto total_elements = access_details.getTotalElements();
  auto element_width = access_details.getElementWidth();
  auto chunk_size = access_details.getChunkSize();
  auto chunk_stride = access_details.getChunkStride();
  auto shuffle_mode = access_details.getShuffleMode();
  auto rotation_position = access_details.getRotationPosition();

  // Set Immutable address and increments
  Value actual_immutable_addr, increment;
  bool perform_burst_or_group = burst_size > 0 || group_size > 0;
  if (failed(setImmutableAddrAndIncrements(
          *builder, comp, op, perform_burst_or_group, stride_step, burst_size,
          total_elements, mem_view_start_addr, actual_immutable_addr, increment,
          unit_op))) {
    return op->emitOpError("could not set offset and increment.");
  }

  auto consumer_info = getLoadConsumer(op);
  if (!consumer_info.first || !consumer_info.second) {
    return op->emitError("can not extract the loadOp's consumer!");
  }

  auto send_op = dyn_cast<dataflow::SendOp>(consumer_info.first);
  DT_CHECK(send_op && "expected send as the consumer of load");

  // To avoid dominance issues in case the consumer of a vector_load is fed
  // with query maps or similar, insert the set send destination statements and
  // load_and_send op in front of the send operation.
  auto insert_pt = builder->saveInsertionPoint();
  if (isa<agen::VectorLoadOp>(op)) builder->setInsertionPoint(send_op);

  if (failed(generateSetSendDestinationStmts(*builder, unit_op, op,
                                             consumer_info.second))) {
    return op->emitError(
        "problem in generating set_send_destination operation");
  }

  dataflow::DataflowRoutingDirectionAttr routing_dir0(
      send_op.getDir() ? send_op.getDirAttr() : nullptr);
  SentientRoutingDirection srd;
  sentient::SentientRoutingDirectionAttr routing_dir = nullptr;
  if (send_op.getDir().has_value()) {
    auto ddir = send_op.getDir().value();
    if (ddir == DataflowRoutingDirection::BothWays)
      srd = SentientRoutingDirection::BothWays;
    else if (ddir == DataflowRoutingDirection::Clockwise)
      srd = SentientRoutingDirection::Clockwise;
    else if (ddir == DataflowRoutingDirection::CounterClockwise)
      srd = SentientRoutingDirection::CounterClockwise;
    else
      srd = SentientRoutingDirection::PseudoRandom;
    routing_dir = SentientRoutingDirectionAttr::get(builder->getContext(), srd);
  }

  if (failed(setldtype(comp, consumer_info.first, total_elements, element_width,
                       shuffle_mode))) {
    return op->emitError("could not set ldtype");
  }

  DT_CHECK_MSG(consumer_info.second->getNumResults() == 1,
               "expected a single ssa value");

  auto dbg_name_attr = getDbgNameAttr(op);

  auto load_and_send_op = sentient::LoadAndSendOp::create(
      *builder, op->getLoc(), IndexType::get(op->getContext()), mutable_addr,
      actual_immutable_addr, increment, consumer_info.second->getResult(0),
      dbg_name_attr, total_elements, element_width,
      rotation_position != 0 ? builder->getI32IntegerAttr(rotation_position)
                             : nullptr,
      routing_dir, chunk_size, chunk_stride, burst_size, group_size,
      symbolizeSentientShuffleMode(shuffle_mode).value());

  // Adjust the mutable address for stride.
  // This adjustment is only required for L0/LX units because stride_step
  // is part of mutable address and forms the effective address before
  // performing the memory operation. As a result, the initial value of it needs
  // to be reset.
  if (stride_step > 0 && is_any_of(comp, L0LU, L0SU, LXLU, LXSU)) {
    if (failed(adjustMutableAddrInitForStride(load_and_send_op, stride_step,
                                              outermost_comp_loop))) {
      return op->emitOpError("cannot update mutable_addr init for stride.");
    };
  }

  if (extract_op) {
    if (adjustMutableAddrInitForIndirect(load_and_send_op, extract_op)
            .failed()) {
      return op->emitOpError("cannot adjust mutable_addr for indirect load op");
    }
  }

  // Since the builder comes from outside this function, restore the insertion

}
// ---- 359/384  constructReceiveAndStoreStmt  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2025  (131L)
LogicalResult e359_constructReceiveAndStoreStmt(
    OpBuilder* builder, dataflow::ProgramUnitOp unit_op, Operation* op,
    Type data_elem_type, AccessDetailsTy& access_details, Value& mutable_addr,
    Value& immutable_addr, unsigned burst_size, unsigned group_size,
    int stride_step, Operation* outermost_comp_loop, Operation* extract_op) {
  auto& mem_view_start_addr = immutable_addr;
  auto total_elements = access_details.getTotalElements();
  auto element_width = access_details.getElementWidth();

  // TODO support coalesce fields

  // It is guaranteed from the upstream passes that all units of a unit
  // operation will have same types.
  auto comp = dcc::getUnitType(
      unit_op.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());

  // Set Immutable address and increments
  Value actual_immutable_addr, increment;
  bool perform_burst_or_group = burst_size > 0 || group_size > 0;
  if (failed(setImmutableAddrAndIncrements(
          *builder, comp, op, perform_burst_or_group, stride_step, burst_size,
          total_elements, mem_view_start_addr, actual_immutable_addr, increment,
          unit_op))) {
    return op->emitOpError("could not set offset and increment.");
  }

  // analyse producer
  auto producer_info = getStoreProducer(op);
  if (!producer_info.first || !producer_info.second) {
    return op->emitError("can not extract the StoreOp's producer!");
  }

  // There could be multiple results for a get unit op (because of folding), and
  // one of those results acting as a producer. This happens when producer's
  // fold size is not one and consumer fold size is one. Even though there exist
  // multiple results, the information needed to construct ReceiveAndStore
  // doesn't change. Hence, this assertion is not needed for such scenarios.
  if (!isa<dataflow::GetUnitOp>(producer_info.second)) {
    DT_CHECK_MSG(producer_info.second->getNumResults() == 1,
                 "expected a single result for non-get unit type producers");
  }

  OpResult producer = nullptr;
  if (auto bs_producer =
          dyn_cast<vectorchain::ConstantBitstreamOp>(producer_info.second)) {
    if (comp != L3LU) {
      return op->emitError(
          "ConstantBitstreamOp producers are only supported in L3");
    }
    // The producer for the receive_and_store will be a sentient.scalar_constant
    auto bs_value = bs_producer.getValue().getValue();
    int64_t imm_val = mlir::cast<IntegerAttr>(bs_value[0]).getInt();
    auto result_type = bs_producer.getResult().getType();
    mlir::Type elements_type;
    if (auto const_vtype = mlir::dyn_cast<VectorType>(result_type)) {
      elements_type = const_vtype.getElementType();
    } else if (auto const_custom_vtype =
                   mlir::dyn_cast<dataflow::CustomVectorType>(result_type)) {
      elements_type = const_custom_vtype.getElementType();
    }
    auto new_producer_op = sentient::ConstantOp::create(*builder, op->getLoc(),
                                                        elements_type, imm_val);
    producer = new_producer_op->getResult(0);
  }
  if (!producer) producer = producer_info.second->getResult(0);

  SentientShuffleModeAttr shuffle_mode = nullptr;
  if (failed(setsttype(builder, comp, producer_info.first, total_elements,
                       element_width, shuffle_mode))) {
    return op->emitError("could not set sttype");
  }

  // analyse coalesce store
  Value drop_first, multicast_info;
  unsigned subword_len = 1, stride = 1;
  bool is_coalesce = false;
  AffineMapAttr drop_first_map;
#ifdef COMMENT_OUT_COALESCE_STORE
#else
  if (auto comp_store_op = dyn_cast<CompositeStoreOp>(op)) {
    getCoalesceInfo(&comp_store_op, builder, is_coalesce, drop_first,
                    subword_len, stride, drop_first_map);
  }
  if (is_coalesce && is_src1_reg) {
    return op->emitError(
        "is_coalesce and is_src1_reg can not be true together.");
  }
#endif

  auto dbg_name_attr = getDbgNameAttr(op);

  sentient::ReceiveAndStoreOp receive_and_store_op;

  if (comp == L0SU && dccExtContext().getArch() >= SEN1P5_ISA) {
    // Only set the destination unit_op for the ReceiveAndStore in sen1p5 L0SU.
    receive_and_store_op = sentient::ReceiveAndStoreOp::create(
        *builder, op->getLoc(), IndexType::get(op->getContext()), mutable_addr,
        actual_immutable_addr, increment, producer, dbg_name_attr,
        access_details.getMemory(), drop_first, multicast_info, total_elements,
        element_width, burst_size, group_size, is_coalesce, subword_len,
        stride);
  } else {
    receive_and_store_op = sentient::ReceiveAndStoreOp::create(
        *builder, op->getLoc(), IndexType::get(op->getContext()), mutable_addr,
        actual_immutable_addr, increment, producer, dbg_name_attr, nullptr,
        drop_first, multicast_info, total_elements, element_width, burst_size,
        group_size, is_coalesce, subword_len, stride);
  }
  if (is_coalesce) {
    receive_and_store_op->setAttr("marked", builder->getI8IntegerAttr(1));
    receive_and_store_op->setAttr("drop_first_map", drop_first_map);
  }
  if (shuffle_mode) receive_and_store_op->setAttr("shuffle_mode", shuffle_mode);

  // Adjust the mutable address for stride.
  // This adjustment is only required for L0/LX units because stride_step
  // is part of mutable address and forms the effective address before
  // performing the memory operation. As a result, the initial value of it needs
  // to be reset.
  if (stride_step > 0 && is_any_of(comp, L0LU, L0SU, LXLU, LXSU)) {
    if (failed(adjustMutableAddrInitForStride(receive_and_store_op, stride_step,
                                              outermost_comp_loop))) {
      return op->emitOpError("cannot update mutable_addr init for stride.");
    };
  }

  if (extract_op) {
    if (adjustMutableAddrInitForIndirect(receive_and_store_op, extract_op)
            .failed()) {
      return op->emitOpError(
          "cannot adjust mutable_addr for indirect store op");

}}}
// ---- 360/384  constructSymbolicDetailsAndAddrs  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:2849  (16L)
LogicalResult e360_constructSymbolicDetailsAndAddrs(
    Operation* src_op, Operation* dst_op, ProgramUnitOp& unit,
    SenComponents comp, AccessContainer<AccessDetailsSymbolic>& access_details,
    AccessContainer<Value>& mutable_addrs,
    AccessContainer<Value>& immutable_addrs) {
  DT_CHECK(src_op);
  AccessDetailsSymbolic& src_ad =
      access_details.emplace_insert(MemoryOperandIndex::kDirSrc, src_op, comp);
  if (src_ad.constructDetails(MemoryOperandIndex::kDirSrc).failed())
    return src_op->emitError("unable to construct details for src");

  if (dst_op) {
    AccessDetailsSymbolic& dst_ad = access_details.emplace_insert(
        MemoryOperandIndex::kDirDst, dst_op, comp);
    if (dst_ad.constructDetails(MemoryOperandIndex::kDirDst).failed())
      return dst_op->emitError("unable to construct details for dst");

}}
// ---- 361/384  runOnOperation  —  dcc/src/Conversion/DataflowToSentient/DataflowToSentient.cpp:2014  (33L)
void e361_runOnOperation() {
  ModuleOp module_op = getOperation();
  std::vector<mlir::Operation *> to_be_deleted;
  module_op.walk([&](dataflow::ProgramUnitOp unit_op) {
    auto unit = unit_op.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>();
    unit_op.walk([&](mlir::Operation *op) {
      if (isa<dataflow::SyncSendOp, dataflow::SyncRecvOp,
              dataflow::ImplicitSyncOnStreamingBufferOp>(op)) {
        auto units = unit_op.getUnits();
        if (lowerSyncOperation(units, op).succeeded())
          to_be_deleted.push_back(op);
        else
          signalPassFailure();
      } else if (auto opaque_op = llvm::dyn_cast<dataflow::OpaqueOp>(op)) {
        OpBuilder builder(unit_op);
        builder.setInsertionPointToStart(&unit_op.getRegion().front());
        if (lowerOpaqueOperation(opaque_op).succeeded())
          to_be_deleted.push_back(op);
        else
          signalPassFailure();
      }
    });
    unit_op.walk([&](mlir::Operation *op) {
      if (isa<dataflow::CreateGroupOp>(op) && op->use_empty())
        to_be_deleted.push_back(op);
    });
  });
  for (auto op : to_be_deleted) op->erase();

  if (dtGetEnv<std::string>("CODEGEN_DUMP_IRS").has_value())
    dataflow::utils::dumpModule(
        module_op, "codegen_dumps/" + dccExtContext().prog_name_,
        (llvm::Twine("first_sentient_ir") + ".mlir").str());

}
// ---- 362/384  LowerSelectOpToSentient  —  dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:243  (19L)
void e362_LowerSelectOpToSentient(Operation *op) {
  SmallVector<Operation *> ops_to_be_erased;
  auto select_op = llvm::dyn_cast<mlir::arith::SelectOp>(op);
  OpBuilder builder(select_op);
  auto *cond_def_op = select_op.getCondition().getDefiningOp();
  auto insertion_point = builder.getInsertionPoint();
  auto transformed_op = ConstructIFRecursively(
      select_op, cond_def_op, select_op.getTrueValue(),
      select_op.getFalseValue(), insertion_point, ops_to_be_erased);
  if (transformed_op != op) {
    select_op->replaceAllUsesWith(transformed_op);
    select_op->erase();
    for (auto eraseOp = ops_to_be_erased.rbegin();
         eraseOp != ops_to_be_erased.rend(); ++eraseOp) {
      if ((*eraseOp)->use_empty()) {
        (*eraseOp)->erase();
      }
    }
  }

}
// ---- 363/384  LowerLogicalOpToSentient  —  dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:264  (19L)
void e363_LowerLogicalOpToSentient(
    mlir::arith::AndIOp op) {
  SmallVector<Operation *> ops_to_be_erased;
  OpBuilder builder(op);
  auto insertion_point = builder.getInsertionPoint();
  auto val_true = sentient::ConstantOp::create(builder, builder.getUnknownLoc(),
                                               builder.getI1Type(), 1);
  auto val_false = sentient::ConstantOp::create(
      builder, builder.getUnknownLoc(), builder.getI1Type(), 0);
  auto transformed_op = ConstructIFRecursively(
      op, op, val_true, val_false, insertion_point, ops_to_be_erased);
  if (transformed_op != op) {
    op->replaceAllUsesWith(transformed_op);
    for (auto eraseOp = ops_to_be_erased.rbegin();
         eraseOp != ops_to_be_erased.rend(); ++eraseOp) {
      if ((*eraseOp)->use_empty()) {
        (*eraseOp)->erase();
      }
    }

}}
// ---- 364/384  patternAgnosticFuseNonComputeOpsHelper  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:96  (222L)
    NonComputeOpPatternBase<OpTy>::patternAgnosticFuseNonComputeOpsHelper(
        Operation *op, std::optional<VectorOperand> &from,
        mlir::ConversionPatternRewriter &rewriter,
        bool is_precision_converted_global) {
  // If the send/store has a corresponding to receive/load.
  if (from.has_value() &&
      isa<dataflow::ReceiveOp, vector::LoadOp, agen::VectorLoadOp,
          mlir::arith::ConstantOp>(from.value().op_)) {
    auto *context = op->getContext();
    bool is_dangling_ops_present_after_fusion = false;
    llvm::SmallVector<std::optional<VectorOperand>, 3> to_operands;
    bool is_fusion_respected = true;
    analyzeNonComputeOpsForFusion(this->unit_, this->dcc_ext_ctx_,
                                  this->reuse_info_, op, from, this->comp_,
                                  is_visited_, is_fusion_respected, to_operands,
                                  is_dangling_ops_present_after_fusion);

    // Continue with other ops.
    if (!is_fusion_respected) return failure();

    SmallVector<std::optional<VectorOperand>, 1> from_operands;
    from_operands.push_back(from);
    this->reuse_info_.setReuseInformation(op, from_operands);

    // To operands are from the same block.
    if (VectorOperand::sameBlock(op, to_operands).succeeded()) {
      std::string op1 = from.value().getName(), op2 = "one", op3 = "zero";
      int from_ID[3] = {this->reuse_info_.getId(from.value().op_).value(), -1,
                        -1};

      /*
       * PE/SFP
       * Receive/Constant/Load + Store/Send
       * Receive + Store: --> LOGICAL (also supports latch)
       *  --> But no precision conversion (e.g., FMA fpuop)
       *  tgtrf = src0 --> not allowed!
       *  we need OR with zero.
       * Receive + Send:  --> LOGICAL (also supports latch)
       *  --> But no precision conversion (e.g., FMA fpuop)
       *  Use opA forwarding (result also.. but we need OR with zero).
       * Load + Store: --> LOGICAL (also supports latch)
       *  --> But no precision conversion (e.g., FMA fpuop)
       *  tgtrf = src0 --> not allowed!
       *  we need OR with zero.
       * Load + Send:  --> LOGICAL (also supports latch)
       *  --> But no precision conversion (e.g., FMA fpuop)
       *  Constant + Send:  --> FMA (also supports latch)
       *  Constant + Store:  --> IMMCOPY (also supports latch)
       *
       */

      // Use FMA output precision conversion in PE/SFP required, or
      // constant + send in PE/SFP, or receive from pt + send in PE (sen1p5)
      bool use_fma = false;
      bool sen1p5_receive_from_pt =
          isa<mlir::dataflow::ReceiveOp>(from.value().op_) &&
          from.value().getFirstValue() == "pt" &&
          this->dcc_ext_ctx_.getArch() >= IsaCoreGen::SEN1P5_ISA;
      if (is_any_of(this->comp_, PE, SFP) && !to_operands.empty()) {
        if (is_precision_converted_global) use_fma = true;

        if (isa<mlir::arith::ConstantOp>(from.value().op_)) {
          // constant+send
          for (auto &to : to_operands) {
            if (to.value().type_ == Link) {
              use_fma = true;
              op1 = "lrf14";  // doesn't influence the result
            }
          }
        } else if (sen1p5_receive_from_pt && this->comp_ == PE) {
          // PT FIFO not valid for LOGICAL in sen1p5 so need to realize as FMA.
          use_fma = true;
        }
      }

      // Use IMMCOPY if its constant storing into a register.
      bool use_immcopy =
          (isa<mlir::arith::ConstantOp>(from.value().op_) &&
           is_any_of(this->comp_, PE, SFP) && to_operands.size() == 1 &&
           to_operands[0].value().type_ == LRF);

      // Use logical by default if fma or immcopy doesn't work out.
      bool use_logical = !(use_fma || use_immcopy);

      rewriter.setInsertionPoint(op);
      SmallVector<Attribute, 1> opA_forwarding;
      SmallVector<Attribute, 1> opC_forwarding;
      SmallVector<Attribute, 1> result_forwarding;
      std::string opA_precision =
          getInputPrecisionFromOperand(from_operands[0]);
      auto compute_precision = getComputePrecisionOfOp(op);
      if (sen1p5_receive_from_pt && this->comp_ == PE) {
        // fp16 coming from PT is really fp24, which shouldn't be downcast to
        // fp16.
        if (compute_precision == "fp16") compute_precision = "fp32";
      }
      SentientFoldModeAttr fold_mode_attr;
      if (opA_precision == "fp32" && compute_precision == "fp16") {
        // Use the operation corresponding to the higher of opA/compute
        // precisions to determine foldMode.
        fold_mode_attr =
            dcc::sentient::utils::getSentientFoldModeAttrForOperation(
                from.value().op_, this->comp_, sen1p5_receive_from_pt);
        // Cast op should be interpreted as output on the fly conversion, so
        // opA_precision should be fp16 and compute&result precisions should be
        // fp32.
        std::swap(opA_precision, compute_precision);
      } else {
        fold_mode_attr =
            dcc::sentient::utils::getSentientFoldModeAttrForOperation(
                op, this->comp_, sen1p5_receive_from_pt);
      }
      std::string opB_precision = compute_precision;
      std::string opC_precision = compute_precision;
      for (auto &to : to_operands) {
        auto destination_type = to.value().type_;
        std::string dest = to.value().getName();
        if (use_fma) {
          // Use the accumulation part if plan to use FMA
          // Solves INT24 challenges in IMA8
          // Allows constants 0, 1, 2, 3
          std::swap(op1, op3);
          std::swap(opA_precision, opC_precision);
          std::swap(from_ID[0], from_ID[2]);
          result_forwarding.push_back(SentientComputePortAttr::get(
              context, symbolizeSentientComputePort(dest).value()));

          // A special case in DD1a where sfp unit forwarding one of
          // its input operands is forwarded to another sfp via MAC.
          if (dest == "sfpring") {
            // In case of SFPRing, only FMA result can be sent to data fifo.
            rewriter.setInsertionPointAfter(to.value().op_);
            auto send_op = llvm::dyn_cast<dataflow::SendOp>(to.value().op_);
            (void)sentient::SetSendDestinationOp::create(rewriter, op->getLoc(),
                                                         send_op.getToUnit());
          }
        } else if (use_immcopy) {
          DT_CHECK(destination_type == LRF);
        } else if (use_logical) {
          // Don't swap the operands.
          if (destination_type == LRF) {
            // LOGICAL doesn't allow src0/src2 port forwarding in tgtrf
            result_forwarding.push_back(SentientComputePortAttr::get(
                context, symbolizeSentientComputePort(dest).value()));
          } else {
            opA_forwarding.push_back(SentientComputePortAttr::get(
                context, symbolizeSentientComputePort(dest).value()));
          }
          // Precision of opC (0.0) should match opA's precision in LOGICAL.
          opC_precision = opA_precision;
        } else {
          llvm::errs() << "Unknown option to lower to sentient\n";
          return failure();
        }
      }

      sentient::ConstantOp mask_const_op = sentient::ConstantOp::create(
          rewriter, op->getLoc(), rewriter.getIndexType(), 0);

      auto op_dbg_name = dataflow::getDbgNameAttr(op);

      auto result_precision = getResultPrecisionFromOperands(to_operands);
      DT_CHECK_MSG(from_operands.size() == 1,
                   "Non compute ops expected to have one input operand");

      // If no result forwarding, set result_precision to default precision.
      if (result_forwarding.empty() && result_precision == "")
        result_precision = compute_precision;

      if (use_fma) {
        // PE/SFP associates mask directly to operations. To associate a
        // non-default mask, a compute operation would be used that
        // specifies a mask. Therefore the mask should be set to 0
        // (default).
        ArrayRef<Value> pointers = {};
        auto mac_op = sentient::MacOp::create(
            rewriter, op->getLoc(), TypeRange(), mask_const_op.getResult(),
            ValueRange(pointers), op_dbg_name,
            symbolizeSentientComputePort(op1).value(),
            symbolizeSentientComputePort(op2).value(),
            symbolizeSentientComputePort(op3).value(),
            ArrayAttr::get(context, opA_forwarding),
            ArrayAttr::get(context, {}),
            ArrayAttr::get(context, opC_forwarding),
            ArrayAttr::get(context, result_forwarding), fold_mode_attr,
            symbolizeSentientPrecision(opA_precision).value(),
            symbolizeSentientPrecision(opB_precision).value(),
            symbolizeSentientPrecision(opC_precision).value(),
            symbolizeSentientPrecision(result_precision).value(),
            symbolizeSentientPrecision(compute_precision).value(), from_ID[0],
            from_ID[1], from_ID[2]);
      } else if (use_immcopy) {
        if (failed(vectorchain::createSplatOperation(
                op, from.value(), to_operands[0].value(), rewriter, this->comp_,
                this->dcc_ext_ctx_.dsc_global_->sysDef)))
          return failure();
      } else if (use_logical) {
        // Bitwise operation so set compute precision to none.
        sentient::BinaryOp::create(
            rewriter, op->getLoc(), mask_const_op, op_dbg_name,
            symbolizeSentientComputePort(op1).value(),
            symbolizeSentientComputePort(op3).value(),
            SentientBinaryOperator::or0,
            ArrayAttr::get(context, opA_forwarding),
            ArrayAttr::get(context, {}),
            ArrayAttr::get(context, result_forwarding), nullptr, fold_mode_attr,
            symbolizeSentientPrecision(opA_precision).value(),
            symbolizeSentientPrecision(opC_precision).value(),
            symbolizeSentientPrecision(result_precision).value(),
            symbolizeSentientPrecision("none").value(), from_ID[0], from_ID[2]);
      }

      std::vector<mlir::Operation *> erased_list;
      VectorOperand::eraseOperands(to_operands, rewriter, erased_list);
      if (!is_dangling_ops_present_after_fusion) {
        VectorOperand::eraseOperands(from_operands, rewriter, erased_list);
      }
    } else {
      op->emitError(
          "All to operands should be in the same block "
          "in order to be fused.");
      return failure();

}}}
// ---- 365/384  fillOpInfo  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1070  (83L)
mlir::LogicalResult e365_fillOpInfo(
    Operation *op, int num_operands, const dcc::DccExtContext &dcc_ext_ctx,
    dataflow::ProgramUnitOp unit, SenComponents comp,
    ConversionPatternRewriter &rewriter,
    VectorChainToSentientPESFPLoweringPass::OpInfo &op_info) {
  auto *context = op->getContext();
  for (int i = 0; i < num_operands; i++) {
    auto operand_op = op->getOperand(i).getDefiningOp();
    op_info.from_operands_.push_back(
        VectorOperand::getOperand(dcc_ext_ctx, operand_op, comp));
  }

  if (auto shuffle_op = dyn_cast<vectorchain::ShuffleOp>(op)) {
    // Shuffle operations contain two variadic operands before the mask operand.
    // The variadic operands are to indicate what value substitutions or padding
    // values to use for the operation if any. Specifically set the mask for
    // shuffle operations.
    if (shuffle_op.getMask())
      op_info.mask_operand_ = shuffle_op.getMask().getDefiningOp();
    else
      op_info.mask_operand_ = std::nullopt;
  } else if (op->getNumOperands() == num_operands + 1) {
    op_info.mask_operand_ = op->getOperand(num_operands).getDefiningOp();
  } else {
    op_info.mask_operand_ = std::nullopt;
  }

  // analyze and fill operand forwarding
  analyzeAndFillOperandForwarding(context, dcc_ext_ctx, comp,
                                  op_info.from_operands_[0],
                                  op_info.opA_forwarding_);

  if (op_info.from_operands_.size() > 1) {
    analyzeAndFillOperandForwarding(context, dcc_ext_ctx, comp,
                                    op_info.from_operands_[1],
                                    op_info.opB_forwarding_);
  }

  if (op_info.from_operands_.size() > 2) {
    analyzeAndFillOperandForwarding(context, dcc_ext_ctx, comp,
                                    op_info.from_operands_[2],
                                    op_info.opC_forwarding_);
  }

  // PE/SFP associates mask directly to operations. Calculate the mask to
  // insert into the sentient.vector_mac. If no mask info is attached to the
  // op, use the default mask, 0.
  op_info.mask_val_ =
      op_info.mask_operand_.has_value()
          ? vectorchain::getMaskValueForNonPT(op, op_info.mask_operand_.value(),
                                              dcc_ext_ctx.dsc_global_->sysDef,
                                              rewriter)
          : sentient::ConstantOp::create(rewriter, op->getLoc(),
                                         rewriter.getIndexType(), 0);
  if (!op_info.mask_val_.has_value()) return failure();

  analyzeAndFillResultForwarding(
      op, comp, context, dcc_ext_ctx, rewriter, op_info.to_operands_,
      op_info.result_forwarding_, op_info.logical_result_forwarding_);

  op_info.result_precision_ =
      getResultPrecisionFromOperands(op_info.to_operands_);
  op_info.compute_precision_ = getComputePrecisionOfOp(op);

  // If no result forwarding, set result_precision to default precision.
  if (op_info.result_forwarding_.empty() && op_info.result_precision_ == "")
    op_info.result_precision_ = op_info.compute_precision_;

  op_info.fold_mode_attr_ =
      dcc::sentient::utils::getSentientFoldModeAttrForOperation(op, comp);

  int num_from_operands = op_info.from_operands_.size();
  op_info.opA_precision_ =
      getInputPrecisionFromOperand(op_info.from_operands_[0]);
  if (num_from_operands > 1)
    op_info.opB_precision_ =
        getInputPrecisionFromOperand(op_info.from_operands_[1]);
  else
    op_info.opB_precision_ = op_info.compute_precision_;
  if (num_from_operands > 2)
    op_info.opC_precision_ =
        getInputPrecisionFromOperand(op_info.from_operands_[2]);
  else

}
// ---- 366/384  fuseNonComputeOps  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1159  (80L)
void e366_fuseNonComputeOps(
    MLIRContext *context, dataflow::ProgramUnitOp unit_op,
    OperandReuse &reuse_info) {
  std::map<Operation *, bool> is_visited;
  unit_op.walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<dataflow::SendOp, vector::StoreOp, agen::VectorStoreOp>(op))
      is_visited[op] = false;
  });

  RewritePatternSet non_compute_ops_patterns(context);
  non_compute_ops_patterns
      .insert<SendOpLowering, StoreOpLowering, VectorStoreOpLowering>(
          context, dccExtContext(), unit_op, reuse_info, is_visited);

  dataflow::GetUnitOp get_unit_op = dyn_cast_or_null<dataflow::GetUnitOp>(
      unit_op.getUnits()[0].getDefiningOp());
  DT_CHECK_MSG(get_unit_op, "Cannot determine GetUnitOp!");
  auto comp = dcc::getUnitType(get_unit_op);

  ConversionTarget target(*context);
  target.addLegalDialect<
      arith::ArithDialect, mlir::vectorchain::VectorChainDialect,
      mlir::sentient::SentientDialect, mlir::memref::MemRefDialect,
      mlir::uniform::UniformDialect, mlir::symbol::SymbolDialect>();
  target.addLegalOp<
      agen::YieldOp, agen::VectorLoadOp, agen::CompositeLoadOp,
      agen::CompositeStoreOp, agen::CompositeLoadAndStoreOp,
      agen::CompositeIndirectLoadAndStoreOp, agen::CompositeIndirectLoadOp,
      agen::CompositeIndirectStoreOp, agen::IndirectVectorLoadOp,
      agen::IndirectVectorStoreOp, agen::CompositeMemoryInterleaveOp,
      agen::SetTransferMaskStateOp>();
  target.addLegalOp<trace::ProgramUnitTraceOp, trace::ReturnOp>();
  target.addLegalOp<
      dataflow::GetUnitOp, dataflow::GetLocalUnitOp, dataflow::ProgramUnitOp,
      dataflow::ReturnOp, dataflow::CreateGroupOp,
      dataflow::CreateMulticastGroupOp, dataflow::SyncSendOp,
      dataflow::SyncRecvOp, dataflow::ImplicitSyncOnStreamingBufferOp,
      dataflow::GetLogicalMemoryViewOp, dataflow::GetPagedLogicalMemoryViewOp,
      dataflow::ReceiveOp, dataflow::OpaqueOp>();

  // This lambda returns 'true' if the op *is illegal* (i.e., should be
  // converted)
  auto getDynamicLoweringLegality = [&](Operation *op) -> std::optional<bool> {
    // If the send/store has a corresponding to receive/load.
    bool is_precision_converted = false;
    Operation *data;
    if (auto send_op = llvm::dyn_cast<dataflow::SendOp>(op)) {
      data = send_op.getSendData().getDefiningOp();
    } else if (auto store_op = llvm::dyn_cast<vector::StoreOp>(op)) {
      data = store_op.getValueToStore().getDefiningOp();
    } else if (auto store_op = llvm::dyn_cast<agen::VectorStoreOp>(op)) {
      data = store_op.getValueToStore().getDefiningOp();
    }

    std::optional<VectorOperand> from = VectorOperand::getOperandWithPrecision(
        dccExtContext(), data, comp, is_precision_converted);
    if (from.has_value() &&
        isa<dataflow::ReceiveOp, vector::LoadOp, agen::VectorLoadOp,
            mlir::arith::ConstantOp>(from.value().op_)) {
      auto *context = op->getContext();
      bool is_dangling_ops_present_after_fusion = false;
      llvm::SmallVector<std::optional<VectorOperand>, 3> to_operands;
      bool is_fusion_respected = true;
      analyzeNonComputeOpsForFusion(unit_op, dccExtContext(), reuse_info, op,
                                    from, comp, is_visited, is_fusion_respected,
                                    to_operands,
                                    is_dangling_ops_present_after_fusion);
      // Continue with other ops.
      return !is_fusion_respected;
    }
    return true;
  };

  target.addDynamicallyLegalOp<dataflow::SendOp>(getDynamicLoweringLegality);
  target.addDynamicallyLegalOp<vector::StoreOp>(getDynamicLoweringLegality);
  target.addDynamicallyLegalOp<agen::VectorStoreOp>(getDynamicLoweringLegality);

  if (failed(applyPartialConversion(unit_op, target,
                                    std::move(non_compute_ops_patterns)))) {
    signalPassFailure();

}}
// ---- 367/384  createXrfIndexModifOps  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/LoweringXRF.cpp:564  (97L)
XrfPtrMap e367_createXrfIndexModifOps(dataflow::ProgramUnitOp unit,
                                              const SenComponents comp) {
  DT_CHECK(comp == PT);
  SmallVector<Operation *, 16> ops_list;
  LayoutExprMap xrf_layout_expr_maps[2] = {
      LayoutExprMap(),   // xrf write pointer
      LayoutExprMap()};  // xrf read pointer

  // if there is no xrf memory view, skip this pass
  bool use_xrf = false;
  unit.walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (isa<dataflow::GetLogicalMemoryViewOp>(op)) {
      if (isXrfRelated(op)) use_xrf = true;
    }
  });

  XrfPtrMap vector_op_to_xrfptr_map;
  if (!use_xrf) return vector_op_to_xrfptr_map;

  auto checkAndAddToRegionOpMap = [&](Operation *op) {
    if (auto for_op = dyn_cast<sentient::ForOp>(op)) {
      for (auto &each_op : for_op.getBody()->getOperations()) {
        if ((isa<agen::VectorLoadOp, agen::VectorStoreOp, vector::LoadOp,
                 vector::StoreOp>(each_op) &&
             isXrfRelated(&each_op)) ||
            region_ops_with_xrf_access.count(&each_op)) {
          region_ops_with_xrf_access.insert(op);
          return;
        }
      }
    } else if (auto if_op = dyn_cast<sentient::IfOp>(op)) {
      for (auto &each_op : if_op.getBody(0)->getOperations()) {
        if ((isa<agen::VectorLoadOp, agen::VectorStoreOp, vector::LoadOp,
                 vector::StoreOp>(each_op) &&
             isXrfRelated(&each_op)) ||
            region_ops_with_xrf_access.count(&each_op)) {
          region_ops_with_xrf_access.insert(op);
          return;
        }
      }
    }
  };

  // find out which forOp needs to be recreated to add xrf iter_args
  unit.walk<WalkOrder::PostOrder>(
      [&](Operation *op) { checkAndAddToRegionOpMap(op); });

  // add iter_args and return values for forop loops
  unit.walk<WalkOrder::PostOrder>([&](Operation *op) {
    if (dyn_cast<sentient::ForOp>(op) && region_ops_with_xrf_access.count(op)) {
      auto for_op = cast<sentient::ForOp>(op);
      createForOpWithReturnValue(for_op);
      region_ops_with_xrf_access.erase(op);
    } else if (dyn_cast<sentient::IfOp>(op) &&
               region_ops_with_xrf_access.count(op)) {
      auto if_op = cast<sentient::IfOp>(op);
      createIfOpWithReturnValue(if_op);
      region_ops_with_xrf_access.erase(op);
    }
  });

  // reflll after forOp clone
  unit.walk<WalkOrder::PostOrder>(
      [&](Operation *op) { checkAndAddToRegionOpMap(op); });

  // collect agen::vector_store, agen::vector_load and vector::load/store for
  // xrf accesses
  unit.walk([&](mlir::Operation *op) {
    if (isa<agen::VectorStoreOp, agen::VectorLoadOp, vector::StoreOp,
            vector::LoadOp>(op) &&
        isXrfRelated(op)) {
      ops_list.push_back(op);

      int stick_elem_num = dcc::utils::getStickElemNumForOp(op);
      DT_CHECK_MSG(stick_elem_num != -1,
                   "Could not compute stick element number");

      // assume that agen::vector_store is only used to store data to XRF in PT
      if (isa<agen::VectorStoreOp, vector::StoreOp>(op)) {
        xrf_layout_expr_maps[0][op] = getLayoutExpr(op, stick_elem_num);
      } else if (isa<agen::VectorLoadOp, vector::LoadOp>(op)) {
        xrf_layout_expr_maps[1][op] = getLayoutExpr(op, stick_elem_num);
      }
    }
  });

  // insert xrf ptr manipulation operations
  if (xrf_layout_expr_maps[0].size() > 0 ||
      xrf_layout_expr_maps[1].size() > 0) {
    if (areXrfAccessesLegal(xrf_layout_expr_maps)) {
      vector_op_to_xrfptr_map =
          processXrfPtrPerUnit(unit, xrf_layout_expr_maps);
    } else {
      unit->emitError("XRF accesses are illegal");
    }
  }


}
// ---- 368/384  fuseNonComputeOps  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:46  (191L)
void e368_fuseNonComputeOps(
    dataflow::ProgramUnitOp &unit, const SenComponents comp,
    XrfPtrMap vector_op_to_xrfptr_map, OperandReuse &reuse_info,
    LoopMaskTree *pt_masking_tree) {
  std::map<Operation *, bool> is_visited;
  SmallVector<Operation *, 16> ops_list;
  XrfPtrMap mac_op_to_xrfptr_map;
  DT_CHECK(comp == PT);

  unit.walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    if (isa<dataflow::SendOp, vector::StoreOp, agen::VectorStoreOp>(op)) {
      ops_list.push_back(op);
      is_visited[op] = false;
    }
  });

  for (auto &op : ops_list) {
    if (!is_visited[op] &&
        (isa<dataflow::SendOp, vector::StoreOp, agen::VectorStoreOp>(op))) {
      is_visited[op] = true;
      bool is_fusion_respected = true;
      auto *context = op->getContext();
      ArrayRef<Value> pointers = {};
      std::vector<Type> result_types;
      Operation *vector_op = op;
      if (auto send_op = dyn_cast<dataflow::SendOp>(op)) {
        vector_op = send_op.getSendData().getDefiningOp();
        for (auto use_ : send_op.getSendData().getDefiningOp()->getUsers()) {
          if (vector_op_to_xrfptr_map.count(use_) > 0) {
            pointers = vector_op_to_xrfptr_map.at(use_).at(0);
            result_types = {IndexType::get(context), IndexType::get(context)};
            vector_op = use_;
            break;
          }
        }
      }

      if (vector_op_to_xrfptr_map.count(vector_op) > 0) {
        pointers = vector_op_to_xrfptr_map.at(vector_op).at(0);
        result_types = {IndexType::get(context), IndexType::get(context)};
      }

      bool is_precision_converted_global = false;
      std::optional<VectorOperand> from;
      if (auto send_op = llvm::dyn_cast<dataflow::SendOp>(op)) {
        bool is_precision_converted = false;
        from = VectorOperand::getOperandWithPrecision(
            dccExtContext(), send_op.getSendData().getDefiningOp(), comp,
            is_precision_converted);
        is_precision_converted_global |= is_precision_converted;
      } else if (auto store_op = llvm::dyn_cast<vector::StoreOp>(op)) {
        bool is_precision_converted = false;
        from = VectorOperand::getOperandWithPrecision(
            dccExtContext(), store_op.getValueToStore().getDefiningOp(), comp,
            is_precision_converted);
        is_precision_converted_global |= is_precision_converted;
      } else if (auto store_op = llvm::dyn_cast<agen::VectorStoreOp>(op)) {
        bool is_precision_converted = false;
        from = VectorOperand::getOperandWithPrecision(
            dccExtContext(), store_op.getValueToStore().getDefiningOp(), comp,
            is_precision_converted);
        is_precision_converted_global |= is_precision_converted;
      } else {
        is_fusion_respected = false;
      }

      // If the send/store has a corresponding to receive/load.
      if (from.has_value() &&
          isa<dataflow::ReceiveOp, vector::LoadOp, agen::VectorLoadOp,
              mlir::arith::ConstantOp>(from.value().op_)) {
        bool is_dangling_ops_present_after_fusion = false;
        llvm::SmallVector<std::optional<VectorOperand>, 3> to_operands;
        analyzeNonComputeOpsForFusion(unit, dccExtContext(), reuse_info, op,
                                      from, comp, is_visited,
                                      is_fusion_respected, to_operands,
                                      is_dangling_ops_present_after_fusion);

        // Continue with other ops.
        if (!is_fusion_respected) continue;

        SmallVector<std::optional<VectorOperand>, 1> from_operands;
        from_operands.push_back(from);
        reuse_info.setReuseInformation(op, from_operands);

        std::string result_precision =
            getResultPrecisionFromOperands(to_operands);
        std::string compute_precision = computeUnitPrecision(unit, comp);

        DT_CHECK_MSG(from_operands.size() == 1,
                     "Non compute ops expected to have one input operand");
        std::string opA_precision =
            getInputPrecisionFromOperand(from_operands[0]);
        std::string opB_precision = compute_precision;
        std::string opC_precision = compute_precision;

        // To operands are from the same block.
        if (VectorOperand::sameBlock(op, to_operands).succeeded()) {
          std::string op1 = from.value().getName(), op2 = "one", op3 = "zero";
          int from_ID[3] = {reuse_info.getId(from.value().op_).value(), -1, -1};

          // Use FMA since unit is PT
          OpBuilder builder(op);
          SmallVector<Attribute, 1> opA_forwarding;
          SmallVector<Attribute, 1> opC_forwarding;
          SmallVector<Attribute, 1> result_forwarding;
          for (auto &to : to_operands) {
            std::string dest = to.value().getName();
            // Use the accumulation part if plan to use FMA
            // Solves INT24 challenges in IMA8
            // Allows constants 0, 1, 2, 3
            std::swap(op1, op3);
            std::swap(opA_precision, opC_precision);
            std::swap(from_ID[0], from_ID[2]);
            result_forwarding.push_back(SentientComputePortAttr::get(
                context, symbolizeSentientComputePort(dest).value()));

            // In architectures DD2 and smaller than that, the hardware
            // uses the src0 port value to transform resultant FP8 data
            // into FP9 and store into XRF for computations. Hence, we
            // need to set the operand with value zero as n-link. This
            // situation arises in case of block loading into xrf and
            // also forwarding data to the next PT row.
            if (dccExtContext().getArch() <= RCUDD1A_ISA) {
              if (compute_precision == "fp8" && op1 == "zero" && op2 == "one" &&
                  op3 == "north" &&
                  is_any_of(to.value().type_, VectorOperandType::LRF,
                            VectorOperandType::XRF, VectorOperandType::Link)) {
                op2 = "north";
              }
            }

            // A special case in DD1a where sfp unit forwarding one of
            // its input operands is forwarded to another sfp via MAC.
            if (dest == "sfpring") {
              // In case of SFPRing, only FMA result can be sent to data fifo.
              builder.setInsertionPointAfter(to.value().op_);
              auto send_op = llvm::dyn_cast<dataflow::SendOp>(to.value().op_);
              (void)sentient::SetSendDestinationOp::create(
                  builder, op->getLoc(), send_op.getToUnit());
            }
          }

          sentient::ConstantOp mask_const_op = sentient::ConstantOp::create(
              builder, op->getLoc(), builder.getIndexType(), 0);
          SentientFoldModeAttr fold_mode_attr =
              dcc::sentient::utils::getSentientFoldModeAttrForOperation(op,
                                                                        comp);

          auto op_dbg_name = dataflow::getDbgNameAttr(op);
          // PT does not associate mask directly to operations. Therefore, no
          // mask is set for PT MACs.
          auto mac_op = sentient::MacOp::create(
              builder, op->getLoc(), TypeRange(result_types), nullptr,
              ValueRange(pointers), op_dbg_name,
              symbolizeSentientComputePort(op1).value(),
              symbolizeSentientComputePort(op2).value(),
              symbolizeSentientComputePort(op3).value(),
              ArrayAttr::get(context, opA_forwarding),
              ArrayAttr::get(context, {}),
              ArrayAttr::get(context, opC_forwarding),
              ArrayAttr::get(context, result_forwarding), fold_mode_attr,
              symbolizeSentientPrecision(opA_precision).value(),
              symbolizeSentientPrecision(opB_precision).value(),
              symbolizeSentientPrecision(opC_precision).value(),
              symbolizeSentientPrecision(result_precision).value(),
              symbolizeSentientPrecision(compute_precision).value(), from_ID[0],
              from_ID[1], from_ID[2]);

          updateLoopMaskTreeForConstantMask(pt_masking_tree, op, &mac_op, 0);

          if (vector_op_to_xrfptr_map.count(vector_op) > 0) {
            LoweringXRF::setSentientMacXrfRegIncrAttr(
                &mac_op, &builder, to_operands[0].value().orig_precision_,
                dccExtContext());
            mac_op_to_xrfptr_map[mac_op] =
                vector_op_to_xrfptr_map.at(vector_op);
          }

          VectorOperand::eraseOperands(to_operands);
          if (!is_dangling_ops_present_after_fusion) {
            VectorOperand::eraseOperands(from_operands);
          }
        } else {
          op->emitError(
              "All to operands should be in the same block "
              "in order to be fused.");
          signalPassFailure();
          return;
        }
      }
    }

}}
// ---- 369/384  fuseComputeOps  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:245  (628L)
void e369_fuseComputeOps(
    dataflow::ProgramUnitOp &unit, const SenComponents comp,
    XrfPtrMap vector_op_to_xrfptr_map, OperandReuse &reuse_info,
    LoopMaskTree *pt_masking_tree) {
  SmallVector<Operation *, 16> ops_list;
  XrfPtrMap mac_op_to_xrfptr_map;
  // Collect all the compute operations
  unit.walk<WalkOrder::PreOrder>([&](mlir::Operation *op) {
    // clang-format off
    if (isa<vectorchain::MultiplyAndAccumulateOp>(op) ||
        isa<vectorchain::MultiplyOp>(op) ||
        isa<vectorchain::BinaryOp>(op)) {
      // clang-format on
      ops_list.push_back(op);
    }
  });

  // Iterate through each operation
  for (auto &op : ops_list) {
    if (auto shuffle_op = llvm::dyn_cast<vectorchain::ShuffleOp>(op)) {
      if (vectorchain::utils::isShuffleNFWDVersion0(shuffle_op) ||
          vectorchain::utils::isShuffleNFWDVersion2(shuffle_op) ||
          vectorchain::utils::isCustomVectorTrivialShuffle(shuffle_op)) {
        continue;
      }
    }
    OpBuilder builder(op);
    auto *context = op->getContext();
    ArrayRef<Value> pointers = {};
    std::vector<Type> result_types;
    Operation *vector_op = nullptr;

    // collect from_operands
    llvm::SmallVector<std::optional<VectorOperand>, 3> from_operands;

    int numOperands = 2;
    if (isa<vectorchain::MultiplyAndAccumulateOp>(op)) numOperands = 3;

    for (int i = 0; i < numOperands; i++) {
      auto operand_op = op->getOperand(i).getDefiningOp();
      if (vector_op_to_xrfptr_map.count(operand_op) > 0 &&
          isa<agen::VectorLoadOp, vector::LoadOp>(operand_op)) {
        vector_op = operand_op;
        pointers = vector_op_to_xrfptr_map.at(vector_op).at(0);
        result_types = {IndexType::get(context), IndexType::get(context)};
      }

      from_operands.push_back(
          VectorOperand::getOperand(dccExtContext(), operand_op, comp));
    }

    if (!vector_op) {
      for (auto user : op->getUsers()) {
        // There could be an intermediate operations between the operand and the
        // store op (e.g. NegOp, cast, SelectOp).
        while (user &&
               isa<arith::SIToFPOp, arith::FPToSIOp, vectorchain::CastOp,
                   vectorchain::NegOp, vectorchain::SelectOp>(user) &&
               user->hasOneUse()) {
          user = *user->getUsers().begin();
        }
        if (vector_op_to_xrfptr_map.count(user) > 0 &&
            isa<vector::StoreOp, agen::VectorStoreOp>(user)) {
          vector_op = user;
          pointers = vector_op_to_xrfptr_map.at(vector_op).at(0);
          result_types = {IndexType::get(context), IndexType::get(context)};
          break;
        }
      }
    }

    std::optional<mlir::Operation *> mask_operand;
    if (op->getNumOperands() == numOperands + 1) {
      mask_operand = op->getOperand(numOperands).getDefiningOp();
    } else {
      mask_operand = std::nullopt;
    }
    // set reuse information for finding latch possibilities
    reuse_info.setReuseInformation(op, from_operands);

    // analyze and fill operand forwarding
    SmallVector<Attribute, 1> opA_forwarding;
    analyzeAndFillOperandForwarding(context, dccExtContext(), comp,
                                    from_operands[0], opA_forwarding);

    SmallVector<Attribute, 1> opB_forwarding;
    if (from_operands.size() > 1) {
      analyzeAndFillOperandForwarding(context, dccExtContext(), comp,
                                      from_operands[1], opB_forwarding);
    }

    SmallVector<Attribute, 1> opC_forwarding;
    if (from_operands.size() > 2) {
      analyzeAndFillOperandForwarding(context, dccExtContext(), comp,
                                      from_operands[2], opC_forwarding);
    }

    std::optional<mlir::Value> mask_val;
    // PT does not associate mask directly to operations. Calculate the mask
    // to insert to the program. For constant masks, a constant is returned.
    // For dynamic masks, the loop iterator the mask operates on is returned.
    if (mask_operand.has_value()) {
      mask_val = getMaskValueForPT(op, mask_operand.value(), comp,
                                   dccExtContext().dsc_global_->sysDef);
      if (!mask_val.has_value()) {
        signalPassFailure();
        return;
      }
    }

    llvm::SmallVector<std::optional<VectorOperand>, 3> to_operands;
    SmallVector<Attribute, 1> result_forwarding;
    // Dummy attribute, should not appear in PT units.
    SentientComputePortAttr logical_result_forwarding = nullptr;
    analyzeAndFillResultForwarding(op, comp, context, dccExtContext(), builder,
                                   to_operands, result_forwarding,
                                   logical_result_forwarding);
    DT_CHECK_MSG(!logical_result_forwarding,
                 "LogicalResultForwarding should not appear in PT units");

    std::string result_precision = getResultPrecisionFromOperands(to_operands);
    std::string compute_precision = computeUnitPrecision(unit, comp);

    int num_from_operands = from_operands.size();
    std::string opA_precision = getInputPrecisionFromOperand(from_operands[0]);
    std::string opB_precision = compute_precision;
    if (num_from_operands >= 2)
      opB_precision = getInputPrecisionFromOperand(from_operands[1]);
    std::string opC_precision = compute_precision;
    if (num_from_operands >= 3)
      opC_precision = getInputPrecisionFromOperand(from_operands[2]);

    SentientFoldModeAttr fold_mode_attr =
        dcc::sentient::utils::getSentientFoldModeAttrForOperation(op, comp);
    auto op_dbg_name = dataflow::getDbgNameAttr(op);

    if (vector_op_to_xrfptr_map.count(vector_op) > 0) {
      // XRF-related scalar adds were already created, right before the
      // transfer's SendOp/StoreOp. Update the builder location to also be
      // before this SendOp/StoreOp in order to avoid use-before-def issues.
      builder.setInsertionPoint(to_operands[0].value().op_);
    }

    // create the required operations in the sentient.
    if (isa<vectorchain::MultiplyOp>(op)) {
      // MACOps in PT units do not have/use mask operand. Masking is controlled
      // by separate operations. For the purposes of calculating mask operation
      // placement later, this MACOp is considered to have a default mask.
      DT_CHECK_MSG(comp == PT && !mask_val.has_value(),
                   "expecting PT with no mask op");
      auto mac_op = sentient::MacOp::create(
          builder, op->getLoc(), TypeRange(result_types), nullptr,
          ValueRange(pointers), op_dbg_name,
          symbolizeSentientComputePort(from_operands[0].value().getName())
              .value(),
          symbolizeSentientComputePort(from_operands[1].value().getName())
              .value(),
          SentientComputePort::zero, ArrayAttr::get(context, opA_forwarding),
          ArrayAttr::get(context, opB_forwarding),
          ArrayAttr::get(context, opC_forwarding),
          ArrayAttr::get(context, result_forwarding), fold_mode_attr,
          symbolizeSentientPrecision(opA_precision).value(),
          symbolizeSentientPrecision(opB_precision).value(),
          symbolizeSentientPrecision(opC_precision).value(),
          symbolizeSentientPrecision(result_precision).value(),
          symbolizeSentientPrecision(compute_precision).value(),
          reuse_info.getId(from_operands[0].value().op_).value(),
          reuse_info.getId(from_operands[1].value().op_).value());
      updateLoopMaskTreeForConstantMask(pt_masking_tree, op, &mac_op, 0);

      // If the mac operation involves XRF, create XRF related operation
      // to manage the implicit addresses.
      if (vector_op_to_xrfptr_map.count(vector_op) > 0) {
        LoweringXRF::setSentientMacXrfRegIncrAttr(
            &mac_op, &builder, to_operands[0].value().orig_precision_,
            dccExtContext());
        mac_op_to_xrfptr_map[mac_op] = vector_op_to_xrfptr_map.at(vector_op);
      }
    } else if (isa<vectorchain::MultiplyAndAccumulateOp>(op)) {
      auto mode = SentientFMAmode::fused_mul_add;
      auto neg_input0 =
          llvm::dyn_cast<vectorchain::NegOp>(op->getOperand(0).getDefiningOp());
      auto neg_input1 =
          llvm::dyn_cast<vectorchain::NegOp>(op->getOperand(1).getDefiningOp());
      if ((neg_input0 || neg_input1) && !(neg_input0 && neg_input1))
        mode = SentientFMAmode::fused_neg_mul_sub;

      auto mac_op = sentient::MacOp::create(
          builder, op->getLoc(), TypeRange(result_types), nullptr,
          ValueRange(pointers), op_dbg_name,
          symbolizeSentientComputePort(from_operands[0].value().getName())
              .value(),
          symbolizeSentientComputePort(from_operands[1].value().getName())
              .value(),
          symbolizeSentientComputePort(from_operands[2].value().getName())
              .value(),
          ArrayAttr::get(context, opA_forwarding),
          ArrayAttr::get(context, opB_forwarding),
          ArrayAttr::get(context, opC_forwarding),
          ArrayAttr::get(context, result_forwarding), fold_mode_attr,
          symbolizeSentientPrecision(opA_precision).value(),
          symbolizeSentientPrecision(opB_precision).value(),
          symbolizeSentientPrecision(opC_precision).value(),
          symbolizeSentientPrecision(result_precision).value(),
          symbolizeSentientPrecision(compute_precision).value(),
          reuse_info.getId(from_operands[0].value().op_).value(),
          reuse_info.getId(from_operands[1].value().op_).value(),
          reuse_info.getId(from_operands[2].value().op_).value(), mode);

      // MACs in the PT unit do not have mask data attached to the operation.
      // Instead, separate operations control masking. Only
      // multiply_and_accumulate ops control this masking. If a mask operand was
      // found, previous calls to getMaskValueForPT() confirmed the mask
      // conforms to a supported format. Supported formats are:
      //   - constant mask values
      //   - subi <loop bound>, <loop iterator>
      // TODO: When more masking options are supported, this will need to
      //       be updated.
      if (mask_val.has_value()) {
        auto mask = mask_val.value();
        if (auto const_mask =
                dyn_cast_or_null<sentient::ConstantOp>(mask.getDefiningOp())) {
          // Mask will be set to the value described by the constant and will
          // not increment.
          updateLoopMaskTreeForConstantMask(pt_masking_tree, op, &mac_op,
                                            const_mask.getValue());
        } else {
          DT_CHECK_MSG(isa<BlockArgument>(mask), "expecting a block arg mask");
          //  Mask will start at 0, increment by 1 each loop iteration.
          updateLoopMaskTreeForDynamicMask(pt_masking_tree, mask, &mac_op, 0,
                                           1);
        }
      } else
        updateLoopMaskTreeForConstantMask(pt_masking_tree, op, &mac_op, 0);

      // If the mac operation involves XRF, create XRF related operation
      // to manage the implicit addresses.
      if (vector_op_to_xrfptr_map.count(vector_op) > 0) {
        LoweringXRF::setSentientMacXrfRegIncrAttr(
            &mac_op, &builder, to_operands[0].value().orig_precision_,
            dccExtContext());
        mac_op_to_xrfptr_map[mac_op] = vector_op_to_xrfptr_map.at(vector_op);
      }
    } else if (auto binary_op = llvm::dyn_cast<vectorchain::BinaryOp>(op)) {
      // create a binary operation
      DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");

      auto inp1 = from_operands[0].value();
      auto inp2 = from_operands[1].value();

      auto sentient_binary =
          getVectorBinaryToSentientBinary(binary_op.getBinaryOp());
      if (isSentientBinaryLogicalOp(sentient_binary)) {
        // Bitwise logical operation should have compute operand precision set
        // to none.
        compute_precision = "none";
      }

      sentient::BinaryOp::create(
          builder, op->getLoc(), mask_val.value(), op_dbg_name,
          symbolizeSentientComputePort(inp1.getName()).value(),
          symbolizeSentientComputePort(inp2.getName()).value(),
          getVectorBinaryToSentientBinary(binary_op.getBinaryOp()),
          ArrayAttr::get(context, {}), ArrayAttr::get(context, {}),
          ArrayAttr::get(context, result_forwarding), nullptr, fold_mode_attr,
          symbolizeSentientPrecision(opA_precision).value(),
          symbolizeSentientPrecision(opB_precision).value(),
          symbolizeSentientPrecision(result_precision).value(),
          symbolizeSentientPrecision(compute_precision).value(),
          reuse_info.getId(inp1.op_).value(),
          reuse_info.getId(inp2.op_).value());
    } else if (auto compare_op =
                   llvm::dyn_cast<vectorchain::ElementWiseCompareOp>(op)) {
      // create a binary operation
      DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");

      auto inp1 = from_operands[0].value();
      auto inp2 = from_operands[1].value();

      // Since Sentient ISA & Dialect doesn't support element-wise gt, ge, we
      // reorder the input operands before lowering to Sentient IR.
      if (compare_op.getCompareOp() ==
          VectorChainElementWiseCompareOperator::compare_gt) {
        sentient::BinaryOp::create(
            builder, op->getLoc(), mask_val.value(), op_dbg_name,
            symbolizeSentientComputePort(inp2.getName()).value(),
            symbolizeSentientComputePort(inp1.getName()).value(),
            SentientBinaryOperator::fcmp_lt, ArrayAttr::get(context, {}),
            ArrayAttr::get(context, {}),
            ArrayAttr::get(context, result_forwarding), nullptr, fold_mode_attr,
            symbolizeSentientPrecision(opA_precision).value(),
            symbolizeSentientPrecision(opB_precision).value(),
            symbolizeSentientPrecision(result_precision).value(),
            symbolizeSentientPrecision(compute_precision).value(),
            reuse_info.getId(inp2.op_).value(),
            reuse_info.getId(inp1.op_).value());
      } else if (compare_op.getCompareOp() ==
                 VectorChainElementWiseCompareOperator::compare_ge) {
        sentient::BinaryOp::create(
            builder, op->getLoc(), mask_val.value(), op_dbg_name,
            symbolizeSentientComputePort(inp2.getName()).value(),
            symbolizeSentientComputePort(inp1.getName()).value(),
            SentientBinaryOperator::fcmp_le, ArrayAttr::get(context, {}),
            ArrayAttr::get(context, {}),
            ArrayAttr::get(context, result_forwarding), nullptr, fold_mode_attr,
            symbolizeSentientPrecision(opA_precision).value(),
            symbolizeSentientPrecision(opB_precision).value(),
            symbolizeSentientPrecision(result_precision).value(),
            symbolizeSentientPrecision(compute_precision).value(),
            reuse_info.getId(inp2.op_).value(),
            reuse_info.getId(inp1.op_).value());
      } else {
        sentient::BinaryOp::create(
            builder, op->getLoc(), mask_val.value(), op_dbg_name,
            symbolizeSentientComputePort(inp1.getName()).value(),
            symbolizeSentientComputePort(inp2.getName()).value(),
            getVectorElementWiseCompareOperatorToSentientBinaryOperator(
                compare_op.getCompareOp()),
            ArrayAttr::get(context, {}), ArrayAttr::get(context, {}),
            ArrayAttr::get(context, result_forwarding), nullptr, fold_mode_attr,
            symbolizeSentientPrecision(opA_precision).value(),
            symbolizeSentientPrecision(opB_precision).value(),
            symbolizeSentientPrecision(result_precision).value(),
            symbolizeSentientPrecision(compute_precision).value(),
            reuse_info.getId(inp1.op_).value(),
            reuse_info.getId(inp2.op_).value());
      }
    } else if (auto ew_selection_op =
                   llvm::dyn_cast<vectorchain::ElementWiseSelectionOp>(op)) {
      // create a binary operation
      DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");

      // Check about the users of compare operation to see if it can be
      // fused.. compare + select turning into min/max.
      bool fusion_to_min = false;
      bool fusion_to_max = false;
      fuseCompareAndSelectIntoMinOrMax(ew_selection_op, fusion_to_min,
                                       fusion_to_max);

      auto state = from_operands[0].value();
      auto lhs = from_operands[1].value();
      auto rhs = from_operands[2].value();

      if (fusion_to_min || fusion_to_max) {
        sentient::BinaryOp::create(
            builder, op->getLoc(), mask_val.value(), op_dbg_name,
            symbolizeSentientComputePort(lhs.getName()).value(),
            symbolizeSentientComputePort(rhs.getName()).value(),
            fusion_to_min ? SentientBinaryOperator::min
                          : SentientBinaryOperator::max,
            ArrayAttr::get(context, {}), ArrayAttr::get(context, {}),
            ArrayAttr::get(context, result_forwarding), nullptr, fold_mode_attr,
            symbolizeSentientPrecision(opA_precision).value(),
            symbolizeSentientPrecision(opB_precision).value(),
            symbolizeSentientPrecision(result_precision).value(),
            symbolizeSentientPrecision(compute_precision).value(),
            reuse_info.getId(lhs.op_).value(),
            reuse_info.getId(rhs.op_).value());
      } else {
        sentient::TernaryOp::create(
            builder, op->getLoc(), mask_val.value(), op_dbg_name,
            symbolizeSentientComputePort(state.getName()).value(),
            symbolizeSentientComputePort(lhs.getName()).value(),
            symbolizeSentientComputePort(rhs.getName()).value(),
            getVectorTernaryToSentientTernary(ew_selection_op),
            ArrayAttr::get(context, opA_forwarding),
            ArrayAttr::get(context, opB_forwarding),
            ArrayAttr::get(context, opC_forwarding),
            ArrayAttr::get(context, result_forwarding), fold_mode_attr,
            symbolizeSentientPrecision(opA_precision).value(),
            symbolizeSentientPrecision(opB_precision).value(),
            symbolizeSentientPrecision(opC_precision).value(),
            symbolizeSentientPrecision(result_precision).value(),
            symbolizeSentientPrecision(compute_precision).value(),
            reuse_info.getId(state.op_).value(),
            reuse_info.getId(lhs.op_).value(),
            reuse_info.getId(rhs.op_).value());
      }
    } else if (auto shuffle_op = llvm::dyn_cast<vectorchain::ShuffleOp>(op)) {
      Operation *src = op->getOperand(0).getDefiningOp();
      if (auto src_cast = llvm::dyn_cast<vectorchain::CastOp>(src)) {
        std::vector<int> indices;
        int repetition = shuffle_op.getRepetition();

        if (failed(checkValidityOfPackAndShuffleLowering(op, indices,
                                                         repetition))) {
          signalPassFailure();
          return;
        }

        std::string name_str = getGCVTorFCVTTypeFromIndicesAndCastInputs(
            indices, repetition, {src_cast}, dcc_ext_ctx_);
        if (name_str.empty()) {
          op->emitError(
              "There is no FCVT/GCVT instruction corresponding to the "
              "following operation");
          op->dump();
          signalPassFailure();
          return;
        }
        DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");
        sentient::UnaryOp::create(
            builder, op->getLoc(), mask_val.value(), op_dbg_name,
            symbolizeSentientComputePort(from_operands[0].value().getName())
                .value(),
            symbolizeSentientUnary(name_str).value(),
            ArrayAttr::get(context, {}),
            ArrayAttr::get(context, result_forwarding), fold_mode_attr,
            symbolizeSentientPrecision(opA_precision).value(),
            symbolizeSentientPrecision(result_precision).value(),
            symbolizeSentientPrecision(compute_precision).value(),
            reuse_info.getId(src_cast.getInput().getDefiningOp()).value());
      } else {
        op->emitError("SPLAT instruction not available on PT units.");
        signalPassFailure();
        return;
      }
    } else if (auto scan_with_gap =
                   llvm::dyn_cast<vectorchain::ScanWithGapOp>(op)) {
      if (!is_any_of(result_precision, "fp16", "fp32")) {
        op->emitError("Reduction is supported only in FP16/FP32.");
        signalPassFailure();
        return;
      }
      if (scan_with_gap.getEvalOrder() !=
          mlir::vectorchain::VectorChainScanOpEvalOrders::left_to_right) {
        op->emitError("Only reduction order=left_to_right is supported.");
        signalPassFailure();
        return;
      }
      if (scan_with_gap.getGap().getSExtValue() != 8) {
        op->emitError("Only reduction gap=8 is supported.");
        signalPassFailure();
        return;
      }
      // clang-format off
      if (!is_any_of<VectorChainBinaryOperator>(scan_with_gap.getReductionOp(),
              mlir::vectorchain::VectorChainBinaryOperator::add,
              mlir::vectorchain::VectorChainBinaryOperator::min,
              mlir::vectorchain::VectorChainBinaryOperator::max,
              mlir::vectorchain::VectorChainBinaryOperator::abs_min,
              mlir::vectorchain::VectorChainBinaryOperator::abs_max)) {
        // clang-format on
        op->emitError("Unsuported operation for reduction.");
        signalPassFailure();
        return;
      }
      std::string reduction_func =
          "reduction_" +
          stringifyVectorChainBinaryOperator(scan_with_gap.getReductionOp())
              .str();
      // create a unary operation
      DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");
      sentient::UnaryOp::create(
          builder, op->getLoc(), mask_val.value(), op_dbg_name,
          symbolizeSentientComputePort(from_operands[0].value().getName())
              .value(),
          symbolizeSentientUnary(StringRef(reduction_func)).value(),
          ArrayAttr::get(context, {}),
          ArrayAttr::get(context, result_forwarding), fold_mode_attr,
          symbolizeSentientPrecision(opA_precision).value(),
          symbolizeSentientPrecision(result_precision).value(),
          symbolizeSentientPrecision(compute_precision).value(),
          reuse_info.getId(from_operands[0].value().op_).value());
    } else if (auto fast_exp_op = llvm::dyn_cast<vectorchain::FastExpOp>(op)) {
      // create a unary operation
      DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");
      sentient::UnaryOp::create(
          builder, op->getLoc(), mask_val.value(), op_dbg_name,
          symbolizeSentientComputePort(from_operands[0].value().getName())
              .value(),
          symbolizeSentientUnary(StringRef("fast_exp")).value(),
          ArrayAttr::get(context, {}),
          ArrayAttr::get(context, result_forwarding), fold_mode_attr,
          symbolizeSentientPrecision(opA_precision).value(),
          symbolizeSentientPrecision(result_precision).value(),
          symbolizeSentientPrecision(compute_precision).value(),
          reuse_info.getId(from_operands[0].value().op_).value());
    } else if (auto exp_est_op =
                   llvm::dyn_cast<vectorchain::ExpEstimateOp>(op)) {
      // create a unary operation
      std::string func_name =
          "exp_" +
          stringifyVectorChainExpEstimate(exp_est_op.getVersion()).str();
      DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");
      sentient::UnaryOp::create(
          builder, op->getLoc(), mask_val.value(), op_dbg_name,
          symbolizeSentientComputePort(from_operands[0].value().getName())
              .value(),
          symbolizeSentientUnary(StringRef(func_name)).value(),
          ArrayAttr::get(context, {}),
          ArrayAttr::get(context, result_forwarding), fold_mode_attr,
          symbolizeSentientPrecision(opA_precision).value(),
          symbolizeSentientPrecision(result_precision).value(),
          symbolizeSentientPrecision(compute_precision).value(),
          reuse_info.getId(from_operands[0].value().op_).value());
    } else if (isa<vectorchain::RecEstimateOp>(op)) {
      // create a unary operation
      DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");
      sentient::UnaryOp::create(
          builder, op->getLoc(), mask_val.value(), op_dbg_name,
          symbolizeSentientComputePort(from_operands[0].value().getName())
              .value(),
          SentientUnary::rec, ArrayAttr::get(context, {}),
          ArrayAttr::get(context, result_forwarding), fold_mode_attr,
          symbolizeSentientPrecision(opA_precision).value(),
          symbolizeSentientPrecision(result_precision).value(),
          symbolizeSentientPrecision(compute_precision).value(),
          reuse_info.getId(from_operands[0].value().op_).value());
    } else if (isa<vectorchain::LnEstimateOp>(op)) {
      // create a unary operation
      DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");
      sentient::UnaryOp::create(
          builder, op->getLoc(), mask_val.value(), op_dbg_name,
          symbolizeSentientComputePort(from_operands[0].value().getName())
              .value(),
          SentientUnary::ln, ArrayAttr::get(context, {}),
          ArrayAttr::get(context, result_forwarding), fold_mode_attr,
          symbolizeSentientPrecision(opA_precision).value(),
          symbolizeSentientPrecision(result_precision).value(),
          symbolizeSentientPrecision(compute_precision).value(),
          reuse_info.getId(from_operands[0].value().op_).value());
    } else if (isa<vectorchain::RsqrtEstimateOp>(op)) {
      // create a unary operation
      DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");
      sentient::UnaryOp::create(
          builder, op->getLoc(), mask_val.value(), op_dbg_name,
          symbolizeSentientComputePort(from_operands[0].value().getName())
              .value(),
          SentientUnary::rsqrt, ArrayAttr::get(context, {}),
          ArrayAttr::get(context, result_forwarding), fold_mode_attr,
          symbolizeSentientPrecision(opA_precision).value(),
          symbolizeSentientPrecision(result_precision).value(),
          symbolizeSentientPrecision(compute_precision).value(),
          reuse_info.getId(from_operands[0].value().op_).value());
    } else if (auto sigmoid_est_op =
                   llvm::dyn_cast<vectorchain::SigmoidEstimateOp>(op)) {
      // create a unary operation
      std::string func_name = "sigm_" + stringifyVectorChainEstimateVersions(
                                            sigmoid_est_op.getVersion())
                                            .str();
      DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");
      sentient::UnaryOp::create(
          builder, op->getLoc(), mask_val.value(), op_dbg_name,
          symbolizeSentientComputePort(from_operands[0].value().getName())
              .value(),
          symbolizeSentientUnary(func_name).value(),
          ArrayAttr::get(context, {}),
          ArrayAttr::get(context, result_forwarding), fold_mode_attr,
          symbolizeSentientPrecision(opA_precision).value(),
          symbolizeSentientPrecision(result_precision).value(),
          symbolizeSentientPrecision(compute_precision).value(),
          reuse_info.getId(from_operands[0].value().op_).value());
    } else if (auto tanh_est_op =
                   llvm::dyn_cast<vectorchain::TanhEstimateOp>(op)) {
      // create a unary operation
      std::string func_name =
          "tanh_" +
          stringifyVectorChainEstimateVersions(tanh_est_op.getVersion()).str();
      DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");
      sentient::UnaryOp::create(
          builder, op->getLoc(), mask_val.value(), op_dbg_name,
          symbolizeSentientComputePort(from_operands[0].value().getName())
              .value(),
          symbolizeSentientUnary(func_name).value(),
          ArrayAttr::get(context, {}),
          ArrayAttr::get(context, result_forwarding), fold_mode_attr,
          symbolizeSentientPrecision(opA_precision).value(),
          symbolizeSentientPrecision(result_precision).value(),
          symbolizeSentientPrecision(compute_precision).value(),
          reuse_info.getId(from_operands[0].value().op_).value());
    } else if (isa<vectorchain::PackOp>(op)) {
      std::vector<int> indices;
      auto pack_op = llvm::dyn_cast<vectorchain::PackOp>(op);
      int repetition = pack_op.getRepetition().getSExtValue();

      if (failed(
              checkValidityOfPackAndShuffleLowering(op, indices, repetition))) {
        signalPassFailure();
        return;
      }
      auto src0_cast = llvm::dyn_cast<vectorchain::CastOp>(
          op->getOperand(0).getDefiningOp());
      auto src1_cast = llvm::dyn_cast<vectorchain::CastOp>(
          op->getOperand(1).getDefiningOp());
      std::string name_str = "";
      if (src0_cast && src1_cast) {
        name_str = getGCVTorFCVTTypeFromIndicesAndCastInputs(
            indices, repetition, {src0_cast, src1_cast}, dcc_ext_ctx_);
      } else if (!src0_cast && !src1_cast) {
        // MERGE/PACK instructions are bitwise operations so compute precision
        // should be set to "none".
        compute_precision = "none";
        name_str = getMergeTypeFromIndices(
            indices, repetition, pack_op.getSignExtend(),
            dataflow::utils::getElementTypeBitWidth(
                pack_op.getResult().getType()));
      }

      if (name_str.empty()) {
        op->emitError(
            "There is no FCVT/GCVT, merge or pack instruction corresponding to "
            "the following operation");
        op->dump();
        signalPassFailure();
        return;
      }

      DT_CHECK_MSG(mask_val.has_value(), "expecting valid mask value");
      sentient::BinaryOp::create(
          builder, op->getLoc(), mask_val.value(), op_dbg_name,
          symbolizeSentientComputePort(from_operands[0].value().getName())
              .value(),
          symbolizeSentientComputePort(from_operands[1].value().getName())
              .value(),
          symbolizeSentientBinaryOperator(name_str).value(),
          ArrayAttr::get(context, opA_forwarding),
          ArrayAttr::get(context, opB_forwarding),
          ArrayAttr::get(context, result_forwarding), nullptr, fold_mode_attr,
          symbolizeSentientPrecision(opA_precision).value(),
          symbolizeSentientPrecision(opB_precision).value(),
          symbolizeSentientPrecision(result_precision).value(),
          symbolizeSentientPrecision(compute_precision).value(),
          reuse_info.getId(from_operands[0].value().op_).value(),
          reuse_info.getId(from_operands[1].value().op_).value());
    }
    VectorOperand::eraseOperands(to_operands);
    VectorOperand::eraseOp(op);

}}
// ---- 370/384  areShallowlyMergeable  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:343  (34L)
bool e370_areShallowlyMergeable(
    dcc::OperationEquivalence &oe, Operation *if_op0, Operation *if_op1) {
  DT_CHECK_MSG(if_op0 != if_op1, "Cannot merge a conditional with itself.");
  Block *bb = if_op0->getBlock();
  if (bb != if_op1->getBlock() ||
      (if_op0->getNumResults() > 0 && if_op1->getNumResults() > 0))
    return false;

  if (!topLevelConditionsMatch(oe, if_op0, if_op1)) return false;

  // Check that the ops strictly between if_op0, if_op1 have no side-effects
  // and no uses inside of if_op1.
  bool is_between_if_ops = false;
  for (auto &op : bb->getOperations()) {
    if (&op == if_op1) break;
    if (is_between_if_ops) {
      if (CFGSDataflowConditionalTree::opHasSideEffect(op)) return false;
      Operation *parent_op = op.getParentOp();
      // Traverse the parent ops of each usage of op, checking if
      // if_op1 appears, in which case it is not safe to merge (as the blocks
      // inside if_op1 would have to be merged into if_op0).
      // We allow for the CmpIOp condition of if_op1 (if it is an scf.if) to be
      // among these ops (since if_op0 has an equivalent condition preceding
      // it).
      if (&op != if_op1->getOperand(0).getDefiningOp() ||
          !isa<scf::IfOp>(if_op1))
        for (auto &use : op.getUses()) {
          for (Operation *cur_op = use.getOwner(); cur_op != parent_op;
               cur_op = cur_op->getParentOp())
            if (cur_op == if_op1) return false;
        }
    }
    if (&op == if_op0) is_between_if_ops = true;
  }

}
// ---- 371/384  hoistLoopInvariantConditionals  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:750  (42L)
void e371_hoistLoopInvariantConditionals() {
  // Search the Conditional Tree for this pattern:
  // loop {
  //   ...
  //   node n
  // }
  // where n is a simple, result-yielding, loop-invariant if-op which feeds into
  // a get_logical_mem_view or dataflow.send/receive.
  // If candidate is found, hoist it out of the parent loop,
  // along with any ancestors it is dependent on.
  auto hoistLIConditionals = [&](CondNode *n) -> CondNode * {
    if (n == getRoot() || n->isLeaf() || n->isThenNode() || n->isElseNode())
      return nullptr;

    Operation *n_if_op = n->getOperation();
    DT_CHECK_MSG(n_if_op, "Expect valid if-op.");
    if (n_if_op->getNumResults() != 1) return nullptr;

    // Restrict hoisting to IfOps only used in get_logical_mem_view or
    // dataflow.send/receive.
    for (auto user : n_if_op->getResult(0).getUsers())
      if (!isa<dataflow::GetLogicalMemoryViewOp, dataflow::ReceiveOp,
               dataflow::SendOp>(user))
        return nullptr;

    // Keep hoisting out of the parent loop until the parent op is not a loop or
    // the conditional is no longer loop invariant with respect to the parent
    // loop.
    while (1) {
      Operation *parent_for_op = n_if_op->getParentOp();
      if (!parent_for_op ||
          !isa<mlir::scf::ForOp, mlir::affine::AffineForOp>(parent_for_op))
        return nullptr;
      if (!isLoopInvariant(n, parent_for_op)) return nullptr;

      // Move n_if_op and its ancestors which do not dominate parent_for_op
      // to right before parent_for_op.
      moveAncestorsToMaintainDominance(n_if_op, parent_for_op);
    }
  };
  CondNode::walk<OperationNode::WalkOrder::kReverseBFS>(getRoot(),
                                                        hoistLIConditionals);

}
// ---- 372/384  runOnOperation  —  dcc/src/Transform/Dataflow/MutableStartAddrShifting.cpp:130  (69L)
void e372_runOnOperation() {
  if (DisableThisPass) return;

  auto isCandidateMemView =
      [](mlir::dataflow::GetLogicalMemoryViewOp mem_view_op) -> bool {
    auto mem_view_unit =
        dcc::getUnitType(mem_view_op.getFromUnit().getDefiningOp());
    if (mem_view_unit != SenComponents::HBM) return false;

    return !isa_and_nonnull<mlir::symbol::CreateSymbolOp,
                            mlir::symbol::SymbolQueryMapOp>(
        mem_view_op.getStartAddress().getDefiningOp());
  };

  ModuleOp module_op = getOperation();
  std::vector<MSASCandidate> candidates;
  std::unordered_set<Operation *> analyzed_candidates;
  module_op.walk<WalkOrder::PreOrder>([&](dataflow::ProgramUnitOp unit) {
    auto comp = dcc::getUnitType(
        unit.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
    if (!is_any_of(comp, SenComponents::L3LU, SenComponents::L3SU))
      return WalkResult::advance();
    unit.walk<WalkOrder::PreOrder>([&](Operation *op) {
      auto mem_view = dyn_cast<dataflow::GetLogicalMemoryViewOp>(op);
      if (!mem_view || !isCandidateMemView(mem_view))
        return WalkResult::advance();
      DT_CHECK_MSG(
          mem_view->hasOneUse(),
          "Expecting L3 memory views to be used in one memory operand.");

      agen::MemoryOperandIndex mem_index = agen::MemoryOperandIndex::kMax;
      auto mem_op = *mem_view->getUsers().begin();
      if (auto comp_las = dyn_cast<agen::CompositeLoadAndStoreOp>(mem_op)) {
        mem_index = comp_las.getSrcMemRef() == mem_view.getResult()
                        ? agen::MemoryOperandIndex::kDirSrc
                        : agen::MemoryOperandIndex::kDirDst;
      } else if (auto comp_ind_las =
                     dyn_cast<agen::CompositeIndirectLoadAndStoreOp>(mem_op)) {
        if (comp_ind_las.getDirectSrcMemref() == mem_view.getResult())
          mem_index = agen::MemoryOperandIndex::kDirSrc;
        else if (comp_ind_las.getDirectDstMemref() == mem_view.getResult())
          mem_index = agen::MemoryOperandIndex::kDirDst;
      } else {
        mem_index = agen::MemoryOperandIndex::kDirSrc;
      }
      DT_CHECK_MSG(mem_index != agen::MemoryOperandIndex::kMax,
                   "Invalid HBM memory operand.");

      auto res = analyzed_candidates.insert(mem_op);
      DT_CHECK_MSG(res.second,
                   "Data transfers should only have one HBM memory operand.");
      candidates.emplace_back(mem_op, unit, comp, mem_index);
      return WalkResult::advance();
    });
    return WalkResult::advance();
  });

  for (auto &candidate : candidates) {
    if (isa<agen::VectorLoadOp>(candidate.op_))
      transformVectorLoad(candidate);
    else if (isa<agen::VectorStoreOp>(candidate.op_))
      transformVectorStore(candidate);
    else if (isa<agen::CompositeLoadAndStoreOp>(candidate.op_))
      transformCompLoadAndStore(candidate);
    else if (isa<agen::CompositeIndirectLoadAndStoreOp>(candidate.op_))
      transformCompIndLoadAndStore(candidate);
    else
      llvm_unreachable("Unexpected candidate operation.");
  }

}
// ---- 373/384  run  —  dcc/src/Transform/Dataflow/TransformPagedMemView/TransformPagedMemViewImpl.cpp:647  (6L)
LogicalResult e373_run() {
  if (initialize().failed()) return LogicalResult::failure();

  if (transform().failed()) return LogicalResult::failure();

  return LogicalResult::success();


}
// ==================================================================================================
// LEVEL 8
// ==================================================================================================

// ---- 374/384  lowerSymbolicVectorLoadOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3380  (26L)
LogicalResult e374_lowerSymbolicVectorLoadOp(
    SymbolicVectorLoadOp& op, ProgramUnitOp& unit, SenComponents comp,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  Operation* store_op =
      getStoreOpFromLoadStorePattern<SymbolicVectorStoreOp>(op);

  AccessContainer<AccessDetailsSymbolic> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructSymbolicDetailsAndAddrs(op, store_op, unit, comp, access_details,
                                       mutable_addrs, immutable_addrs)
          .failed())
    return failure();

  auto candidate_op = dyn_cast_or_null<SymbolicVectorLoadOp>(
      access_details.get(MemoryOperandIndex::kDirSrc).getOp());
  DT_CHECK(candidate_op);

  // Operations may have been deleted/cloned so store op needs to be
  // recollected.
  if (access_details.has(MemoryOperandIndex::kDirDst)) {
    store_op = dyn_cast_or_null<SymbolicVectorStoreOp>(
        access_details.get(MemoryOperandIndex::kDirDst).getOp());
    DT_CHECK(store_op);
  }

  return lowerVectorLoadHelper<AccessDetailsSymbolic, SymbolicVectorLoadOp>(

}
// ---- 375/384  lowerSymbolicVectorStoreOp  —  dcc/src/Conversion/AgenToSentient/Helper.cpp:3410  (30L)
LogicalResult e375_lowerSymbolicVectorStoreOp(
    SymbolicVectorStoreOp& op, ProgramUnitOp& unit, SenComponents comp,
    SmallVectorImpl<Operation*>& to_be_deleted) {
  AccessContainer<AccessDetailsSymbolic> access_details;
  AccessContainer<Value> mutable_addrs, immutable_addrs;
  if (constructSymbolicDetailsAndAddrs(op, nullptr, unit, comp, access_details,
                                       mutable_addrs, immutable_addrs)
          .failed())
    return failure();
  DT_CHECK(access_details.size() == 1 && mutable_addrs.size() == 1 &&
           immutable_addrs.size() == 1);

  auto candidate_op = dyn_cast_or_null<SymbolicVectorStoreOp>(
      access_details.get(MemoryOperandIndex::kDirSrc).getOp());
  DT_CHECK(candidate_op);

  OpBuilder builder(candidate_op);
  Type element_type = candidate_op.getMemref().getType().getElementType();
  if (constructReceiveAndStoreStmt<AccessDetailsSymbolic>(
          &builder, unit, candidate_op, element_type, access_details[0],
          mutable_addrs[0], immutable_addrs[0])
          .failed())
    return candidate_op->emitError(
        "Unable to generate receive_and_store statement for the "
        "agen.vector_store operation");

  to_be_deleted.push_back(candidate_op);

  Operation* input_op = candidate_op.getValueToStore().getDefiningOp();
  addStoreInputToDeleteList(input_op, to_be_deleted);

}
// ---- 376/384  runOnOperation  —  dcc/src/Conversion/StandardToSentient/StandardToSentient.cpp:439  (34L)
void e376_runOnOperation() {
  ModuleOp module_op = getOperation();
  module_op.walk([&](mlir::Operation *op) {
    if (isa<mlir::arith::OrIOp>(op)) {
      SimplifyOrIOp(op);
    }
  });
  module_op.walk([&](mlir::Operation *op) {
    if (isa<mlir::arith::AddIOp>(op)) {
      LowerAddIOpToSentient(op);
    } else if (isa<mlir::arith::SubIOp>(op)) {
      LowerSubIOpToSentient(op);
    } else if (isa<mlir::arith::MulIOp>(op)) {
      op->emitError("Cannot lower mul expressions to sentient");
      signalPassFailure();
      // LowerMulIOpToSentient(op);
    } else if (isa<mlir::arith::SelectOp>(op)) {
      LowerSelectOpToSentient(op);
    } else if (isa<mlir::arith::ConstantIndexOp>(op)) {
      LowerConstantIndexToSentient(op);
    } else if (auto andop = dyn_cast<mlir::arith::AndIOp>(op)) {
      LowerLogicalOpToSentient(andop);
    } else if (auto orop = dyn_cast<mlir::arith::OrIOp>(op)) {
      LowerLogicalOpToSentient(orop);
    } else if (auto cmpop = dyn_cast<mlir::arith::CmpIOp>(op)) {
      LowerLogicalOpToSentient(cmpop);
    } else if (isa<mlir::arith::ConstantIntOp>(op)) {
      LowerConstantIntToSentient(op);
    } else if (isa<mlir::arith::ConstantOp>(op)) {
      op->dump();
      signalPassFailure();
      return;
    }
  });

}
// ---- 377/384  matchAndRewrite  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:44  (12L)
VectorChainToSentientPESFPLoweringPass::SendOpLowering::matchAndRewrite(
    mlir::dataflow::SendOp send_op, OpAdaptor adaptor,
    mlir::ConversionPatternRewriter &rewriter) {
  bool is_precision_converted = false;
  std::optional<VectorOperand> from = VectorOperand::getOperandWithPrecision(
      dcc_ext_ctx_, send_op.getSendData().getDefiningOp(), comp_,
      is_precision_converted);
  bool is_precision_converted_global = is_precision_converted;

  if (failed(patternAgnosticFuseNonComputeOpsHelper(
          send_op, from, rewriter, is_precision_converted_global)))
    return failure();

}
// ---- 378/384  runOnOperation  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPESFP/VectorChainToSentientPESFP.cpp:1374  (30L)
void e378_runOnOperation() {
  ModuleOp module_op = getOperation();
  MLIRContext *context = &getContext();

  module_op.walk([&](dataflow::ProgramUnitOp unit_op) {
    dataflow::GetUnitOp get_unit_op = dyn_cast_or_null<dataflow::GetUnitOp>(
        unit_op.getUnits()[0].getDefiningOp());
    DT_CHECK_MSG(get_unit_op, "Cannot determine GetUnitOp!");
    auto unit_comp = dcc::getUnitType(get_unit_op);
    if (is_any_of(unit_comp, PE, SFP)) {
      // redefine constant vectors
      redefineConstantVectors(module_op);

      // Order of these operations is important!
      OperandReuse reuse_info(unit_op);
      vectorchain::resetSentientFMAsIfExists(unit_op);

      fuseNonComputeOps(context, unit_op, reuse_info);

      fuseComputeOps(context, unit_op, reuse_info);

      if (failed(lowerDanglingNonComputeOpsPESFP(
              unit_op, unit_comp, dccExtContext(), reuse_info)) ||
          failed(vectorchain::validateLoweringAndSetMissingParameters(
              unit_op, reuse_info))) {
        signalPassFailure();
        return;
      }
    }
  });

}
// ---- 379/384  runOnOperation  —  dcc/src/Conversion/VectorChainLowering/VectorChainToSentientPT/VectorChainToSentientPT.cpp:975  (54L)
void e379_runOnOperation() {
  // Note: we couldn't use pattern-based rewriting system because we need to
  // process the different patterns in an order and the rewriting system may
  // not guarantee that order.

  ModuleOp module_op = getOperation();

  // redefine constant vectors
  redefineConstantVectors(module_op);

  module_op.walk([&](dataflow::ProgramUnitOp unit) {
    dataflow::GetUnitOp get_unit_op = dyn_cast_or_null<dataflow::GetUnitOp>(
        unit.getUnits()[0].getDefiningOp());
    DT_CHECK_MSG(get_unit_op, "Cannot determine GetUnitOp!");
    auto unit_comp = dcc::getUnitType(get_unit_op);
    if (unit_comp != PT) return WalkResult::skip();

    auto *context = module_op->getContext();
    DT_CHECK(unit.getUnits().size() >= 1);
    // preprocess xrf pointers before store/load lowering
    LoweringXRF xrf_processor(dcc_ext_ctx_);
    auto vector_op_to_xrfptr_map =
        xrf_processor.createXrfIndexModifOps(unit, unit_comp);

    // Populate the PT masking tree now that loops should be stable.
    // IMPORTANT: PT masking relies on loops being stable. If that changes,
    // the LoopMaskTree needs to be updated.
    LoopMaskTree *pt_masking_tree = nullptr;
    pt_masking_tree = new LoopMaskTree(unit);

    // Order of these operations is important!
    OperandReuse reuse_info(unit);
    vectorchain::resetSentientFMAsIfExists(unit);

    this->fuseNonComputeOps(unit, unit_comp, vector_op_to_xrfptr_map,
                            reuse_info, pt_masking_tree);
    this->fuseComputeOps(unit, unit_comp, vector_op_to_xrfptr_map, reuse_info,
                         pt_masking_tree);
    this->lowerDanglingNonComputeOps(unit, unit_comp, vector_op_to_xrfptr_map,
                                     reuse_info, pt_masking_tree);

    if (failed(vectorchain::validateLoweringAndSetMissingParameters(
            unit, reuse_info))) {
      signalPassFailure();
      WalkResult::interrupt();
    }

    LLVM_DEBUG(llvm::dbgs() << "[vc2sen]: PT Loop Mask Tree: \n";);
    LLVM_DEBUG(pt_masking_tree->print(llvm::outs()););
    insertPTMaskOps(pt_masking_tree);
    delete pt_masking_tree;

    return WalkResult::skip();
  });

}
// ---- 380/384  shallowlyMergeConditionals  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:198  (76L)
Operation *CFGSDataflowConditionalTree::shallowlyMergeConditionals(
    Operation *prev_merged_if_op) {
  SmallVector<Operation *, 4> ops_to_delete;
  // Search the Conditional Tree for conditionals which may be merged and merge
  // them. Each merge may create merge opportunities for its children, so this
  // findCandidatesAndMerge must be done in a PreOrder traversal. However,
  // once a merge has occurred the tree below is clobbered so we avoid further
  // merges (until the tree is recomputed).

  // If one merge occurs, we will return the merged_if_op so we may resume
  // merging from there in the next call to shallowlyMergeConditionals. If no
  // merge occurs, we will return nullptr to indicate there are no more merge
  // opportunities.
  Operation *cur_merged_if_op = nullptr;

  // We start analysis once prev_merged_if_op is reached in the
  // traversal. If this is nullptr, start immediately.
  bool start_analysis = (prev_merged_if_op == nullptr);
  auto findCandidatesAndMerge = [&](CondNode *n) -> CondNode * {
    if (n == getRoot() || n->isLeaf() || n->isThenNode() || n->isElseNode() ||
        cur_merged_if_op)
      return nullptr;
    Operation *n_if_op = n->getOperation();
    DT_CHECK_MSG(n_if_op, "Expect valid if-op.");
    if (n_if_op == prev_merged_if_op) {
      start_analysis = true;
    }
    if (!start_analysis) return nullptr;

    // Merge n with same-block siblings with the same condition
    // whenever either n or its sibling yields no result.
    for (CondNode *sibling = n->getNextSibling(); sibling;
         sibling = sibling->getNextSibling()) {
      Operation *sibling_if_op = sibling->getOperation();
      DT_CHECK_MSG(sibling_if_op, "Expect valid sibling if-op.");
      if (areShallowlyMergeable(getOE(), n_if_op, sibling_if_op)) {
        // The resulting conditional after the merge
        // will either be n_if_op or will be sibling_if_op moved right after
        // n_if_op.
        if (n_if_op->getNumResults() > 0) {
          // Merge the sibling if-op into n's. This is because
          // we need to use n_if_op's result types as it's yielding results.
          mergeShallow(/*src*/ sibling_if_op, /*dst*/ n_if_op,
                       /*dst_before_src*/ true);
          ops_to_delete.push_back(sibling_if_op);
          cur_merged_if_op = n_if_op;
        } else {
          // By default merge n's if-op into the sibling if-op.
          mergeShallow(/*src*/ n_if_op, /*dst*/ sibling_if_op,
                       /*dst_before_src*/ false);
          // Then move the sibling to right after n_if_op.
          sibling_if_op->moveAfter(n_if_op);
          // If the sibling is an scf.if, need to update its condition
          // to use the condition of n's if-op, and remove the unused condition.
          if (isa<scf::IfOp>(sibling_if_op)) {
            auto sibling_cond_op = sibling_if_op->getOperand(0).getDefiningOp();
            sibling_if_op->setOperand(0, n_if_op->getOperand(0));
            if (sibling_cond_op->use_empty()) sibling_cond_op->erase();
          }
          ops_to_delete.push_back(n_if_op);
          cur_merged_if_op = sibling_if_op;
        }
        break;
      }
    }
    return nullptr;
  };
  CondNode::walk<OperationNode::WalkOrder::kPreOrder>(getRoot(),
                                                      findCandidatesAndMerge);
  // Delete all ops in ops_to_delete and their (non-constant) ancestors in the
  // same block.
  for (auto *op : ops_to_delete) deleteAncestorsIfPossible(op);
  if (cur_merged_if_op) {
    recompute();
    getOE().clearCache();
  }

}
// ---- 381/384  simplifyValueBasedConditionals  —  dcc/src/Transform/Dataflow/Analysis/CFGSDataflowConditionalTree.cpp:466  (30L)
void e381_simplifyValueBasedConditionals() {
  bool tree_updated = false;
  // Every time we perform a simplification we need to recompute the
  // tree because we have cloned a loop so the IfOp pointers
  // are clobbered.
  do {
    tree_updated = false;
    for (auto child = getRoot()->getFirstChild(); child;
         child = child->getNextSibling()) {
      Operation *if_op = child->getOperation();
      if (if_op->hasAttr(PROCESSED_SIMPLIFICATIONS)) continue;
      ConditionalSimplificationManager instance(child);
      if (instance.parseConditional()) {
        instance.replaceIfOpByIterArg();
        tree_updated = true;
        break;
      } else {
        // If not a candidate for simplification, mark the IfOp so we do not
        // process it again.
        OpBuilder builder(if_op);
        if_op->setAttr(PROCESSED_SIMPLIFICATIONS, builder.getI32IntegerAttr(1));
      }
    }
    recompute();
  } while (tree_updated);
  for (auto child = getRoot()->getFirstChild(); child;
       child = child->getNextSibling()) {
    Operation *if_op = child->getOperation();
    if_op->removeAttr(PROCESSED_SIMPLIFICATIONS);
  }


}
// ==================================================================================================
// LEVEL 9
// ==================================================================================================

// ---- 382/384  fuseLoadOrStoreChainOps  —  dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:22  (144L)
LogicalResult e382_fuseLoadOrStoreChainOps(
    dataflow::ProgramUnitOp unit, SenComponents comp) {
  // extract_idx will increment every time a load_and_extract operation is
  // created. This index is used to connect load_and_extract operations to
  // the indirect loads that will use them.
  unsigned extract_idx = 0;

  while (true) {
    // Look for a candidate operation for lowering.
    Operation* vector_op = nullptr;
    unit.walk<WalkOrder::PreOrder>([&](Operation* op) {
      if (isa<VectorLoadOp, VectorStoreOp, CompositeLoadOp, CompositeStoreOp,
              CompositeLoadAndStoreOp, IndirectVectorLoadOp,
              IndirectVectorStoreOp, CompositeIndirectLoadOp,
              CompositeIndirectStoreOp, CompositeIndirectLoadAndStoreOp,
              SymbolicVectorLoadOp, SymbolicVectorStoreOp>(op)) {
        if (checkBasicConditions(op).failed())
          signalPassFailure();
        else
          vector_op = op;

        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    if (!vector_op) break;

    LLVM_DEBUG(llvm::dbgs() << "[Agen2Sen] Lowering: " << *vector_op << "\n");

    // Lower according to operation pattern.
    SmallVector<Operation*> to_be_deleted_list;
    auto loc = vector_op->getLoc();
    if (auto load_op = dyn_cast<VectorLoadOp>(vector_op)) {
      if (isLoadAndExtractScalarPattern(load_op)) {
        if (lowerExtractVectorLoadOp(load_op, unit, comp, extract_idx,
                                     to_be_deleted_list)
                .failed()) {
          emitError(
              loc,
              "cannot lower vector load with sentient.load_and_extract_scalar "
              "pattern");
          return failure();
        }
      } else {
        if (lowerVectorLoadOp(load_op, unit, comp, to_be_deleted_list)
                .failed()) {
          emitError(loc, "cannot lower agen.vector_load");
          return failure();
        }
      }
    } else if (auto store_op = dyn_cast<VectorStoreOp>(vector_op)) {
      if (isReceiveAndExtractScalarPattern(store_op)) {
        if (lowerExtractVectorStoreOp(store_op, unit, comp, extract_idx,
                                      to_be_deleted_list)
                .failed()) {
          emitError(loc,
                    "cannot lower vector store with "
                    "sentient.receive_and_extract_scalar pattern");
          return failure();
        }
      } else {
        if (lowerVectorStoreOp(store_op, unit, comp, to_be_deleted_list)
                .failed()) {
          emitError(loc, "cannot lower agen.vector_store");
          return failure();
        }
      }
    } else if (auto comp_load_op = dyn_cast<CompositeLoadOp>(vector_op)) {
      if (lowerCompositeLoadOp(comp_load_op, unit, comp, to_be_deleted_list)
              .failed()) {
        emitError(loc, "cannot lower agen.composite_load");
        return failure();
      }
    } else if (auto comp_store_op = dyn_cast<CompositeStoreOp>(vector_op)) {
      if (lowerCompositeStoreOp(comp_store_op, unit, comp, to_be_deleted_list)
              .failed()) {
        emitError(loc, "cannot lower agen.composite_store");
        return failure();
      }
    } else if (auto comp_load_store_op =
                   dyn_cast<CompositeLoadAndStoreOp>(vector_op)) {
      if (lowerCompositeLoadAndStoreOp(comp_load_store_op, unit, comp,
                                       to_be_deleted_list)
              .failed()) {
        emitError(loc, "cannot lower agen.composite_load_and_store");
        return failure();
      }
    } else if (auto ind_load_op = dyn_cast<IndirectVectorLoadOp>(vector_op)) {
      if (lowerIndirectVectorLoadOp(ind_load_op, unit, comp, to_be_deleted_list)
              .failed()) {
        emitError(loc, "cannot lower agen.indirect_vector_load");
        return failure();
      }
    } else if (auto ind_store_op = dyn_cast<IndirectVectorStoreOp>(vector_op)) {
      if (lowerIndirectVectorStoreOp(ind_store_op, unit, comp,
                                     to_be_deleted_list)
              .failed()) {
        emitError(loc, "cannot lower agen.indirect_vector_store");
        return failure();
      }
    } else if (auto comp_ind_load_op =
                   dyn_cast<CompositeIndirectLoadOp>(vector_op)) {
      if (lowerCompositeIndirectLoadOp(comp_ind_load_op, unit, comp,
                                       to_be_deleted_list)
              .failed()) {
        emitError(loc, "cannot lower agen.composite_indirect_load");
        return failure();
      }
    } else if (auto comp_ind_store_op =
                   dyn_cast<CompositeIndirectStoreOp>(vector_op)) {
      if (lowerCompositeIndirectStoreOp(comp_ind_store_op, unit, comp,
                                        to_be_deleted_list)
              .failed()) {
        emitError(loc, "cannot lower agen.composite_indirect_store");
        return failure();
      }
    } else if (auto comp_ind_load_store_op =
                   dyn_cast<CompositeIndirectLoadAndStoreOp>(vector_op)) {
      if (lowerCompositeIndirectLoadAndStoreOp(comp_ind_load_store_op, unit,
                                               comp, to_be_deleted_list)
              .failed()) {
        emitError(loc, "cannot lower agen.composite_indirect_load_and_store");
        return failure();
      }
    } else if (auto sym_load_op = dyn_cast<SymbolicVectorLoadOp>(vector_op)) {
      if (lowerSymbolicVectorLoadOp(sym_load_op, unit, comp, to_be_deleted_list)
              .failed()) {
        emitError(loc, "cannot lower agen.symbolic_vector_load");
        return failure();
      }
    } else if (auto sym_store_op = dyn_cast<SymbolicVectorStoreOp>(vector_op)) {
      if (lowerSymbolicVectorStoreOp(sym_store_op, unit, comp,
                                     to_be_deleted_list)
              .failed()) {
        emitError(loc, "cannot lower agen.symbolic_vector_store");
        return failure();
      }
    } else {
      llvm_unreachable("unsupported operation");
    }
    for (auto& op : to_be_deleted_list) op->erase();
  }


}
// ---- 383/384  runOnOperation  —  dcc/src/Transform/Dataflow/CFGSimplificationDataflowLevel.cpp:77  (80L)
void e383_runOnOperation() {
  if (DisableThisPass) return;
  LLVM_DEBUG(llvm::dbgs() << "CFG Simplification Dataflow Level Pass\n");
  ModuleOp module_op = getOperation();
  module_op.walk<WalkOrder::PreOrder>([&](dataflow::ProgramUnitOp unit_op) {
    Operation *unit = unit_op.getOperation();
    dcc::CFGSDataflowConditionalTree tree(*unit);
    tree.compute();
    if (tree.empty()) return;
    LLVM_DEBUG(llvm::dbgs() << "Initial tree:\n");
    LLVM_DEBUG(tree.print(llvm::dbgs()));

    // After every simplification, recompute the tree and redo the analysis,
    // repeating until no further hoisting is possible. Processed nodes are
    // marked to avoid re-analyzing them.

    // Hoist conditionals common to both branches of a parent conditional.
    Operation *cur_if_op_analyzed = nullptr;
    int count = 0;
    do {
      cur_if_op_analyzed = tree.hoistCommonConditionals(cur_if_op_analyzed);
      ++count;
      DEBUG_WITH_TYPE(
          VerboseDebug,
          llvm::dbgs()
              << "Tree after hoisting one set of common conditionals:\n");
      DEBUG_WITH_TYPE(VerboseDebug, tree.print(llvm::dbgs()));
    } while (cur_if_op_analyzed && count < MaxNumOfHoists);

    // Hoist loop invariant conditionals feeding into a get_memory_view.
    tree.hoistLoopInvariantConditionals();
    DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs()
                                      << "Tree after hoisting loop invariant "
                                         "conditionals out of loops:\n");
    DEBUG_WITH_TYPE(VerboseDebug, tree.print(llvm::dbgs()));

    // Hoisting may lead to duplicate, side-effect-free siblings. Remove these.
    tree.removeDuplicateConditionals();
    DEBUG_WITH_TYPE(
        VerboseDebug,
        llvm::dbgs()
            << "Tree after removing duplicate sibling conditionals:\n");
    DEBUG_WITH_TYPE(VerboseDebug, tree.print(llvm::dbgs()));

    // Whenver the then and else branches of a conditional are equivalent,
    // move the code out of the conditional and remove the conditional.
    tree.removeConditionWhenThenElseBranchesMatch();
    DEBUG_WITH_TYPE(VerboseDebug, llvm::dbgs()
                                      << "Tree after removing conditions when "
                                         "then/else branches match:\n");
    DEBUG_WITH_TYPE(VerboseDebug, tree.print(llvm::dbgs()));

    // Shallowly merge sibling conditionals when mergeable and one
    // or both do not yield values.
    // NOTE: We must leave the merging of sibling conditionals both yielding
    // values to a later Sentient-level pass as the Dataflow-level calls to the
    // Canonicalizer would partially undo those merges.
    cur_if_op_analyzed = nullptr;
    count = 0;
    do {
      cur_if_op_analyzed = tree.shallowlyMergeConditionals(cur_if_op_analyzed);
      count++;
      DEBUG_WITH_TYPE(VerboseDebug,
                      llvm::dbgs()
                          << "Tree after shallowly merging conditionals:\n");
      DEBUG_WITH_TYPE(VerboseDebug, tree.print(llvm::dbgs()));
    } while (cur_if_op_analyzed && count < MaxNumOfMerges);

    // Merging may lead to duplicate, side-effect-free siblings. Remove these.
    tree.removeDuplicateConditionals();

    // Replace top-level, 1-dimensional conditionals yielding a fixed-stride
    // sequence of Index values by a new iteration argument mimicking that
    // sequence. Skip conditionals feeding the dynamic bound of a loop.
    tree.simplifyValueBasedConditionals();

    LLVM_DEBUG(llvm::dbgs()
               << "Tree after simplifying value-based conditionals:\n");
    LLVM_DEBUG(tree.print(llvm::dbgs()));
  });


}
// ==================================================================================================
// LEVEL 10
// ==================================================================================================

// ---- 384/384  runOnOperation  —  dcc/src/Conversion/AgenToSentient/AgenToSentient.cpp:169  (81L)
void e384_runOnOperation() {
  ModuleOp module_op = getOperation();
  module_op.walk<WalkOrder::PreOrder>([&](dataflow::ProgramUnitOp unit) {
    auto comp = dcc::getUnitType(
        unit.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>());
    if (!is_any_of(comp, L0LU, L0SU, LXLU, LXSU, L3SU, L3LU)) {
      return;
    }

    // Lower LDCVTI patterns to sentient.load_compute_and_send operations first.
    // These patterns involve many operations including multiple
    // agen.vector_load operations. Lowering these patterns before more
    // conventional load/store patterns prevents handling this complexity
    // with all the other loads/stores. These patterns can be easily
    // identified by analyzing their anchor operation, a vectorchain.multiply.
    // LDCVTI patterns are currently the only patterns involving
    // vectorchain.multiply.
    SmallVector<vectorchain::BinaryOp> ldcvti_candidates;
    if (comp == LXLU && dcc_ext_ctx_.getArch() >= SEN1P5_ISA) {
      bool has_converged = false;
      while (!has_converged) {
        SmallVector<Operation*, 16> to_be_deleted;
        vectorchain::BinaryOp candidate = nullptr;
        unit.walk<WalkOrder::PreOrder>([&](vectorchain::BinaryOp op) {
          if (op.getBinaryOp() == vectorchain::VectorChainBinaryOperator::mul) {
            candidate = op;
            return WalkResult::interrupt();
          }
          return WalkResult::advance();
        });
        if (!candidate) {
          has_converged = true;
          break;
        }
        if (lowerLDCVTIPattern(*this, candidate, unit, comp, to_be_deleted)
                .failed()) {
          signalPassFailure();
          return;
        }
        for (auto& obsolete_op : to_be_deleted) obsolete_op->erase();
      }
    }

    // Lower the remaining load/store operation patterns.
    if (fuseLoadOrStoreChainOps(unit, comp).failed()) {
      signalPassFailure();
      return;
    }

    // Lower CompositeMemoryInterleaveOps. Requires the memory operations to be
    // lowered to SentientIR. Lowering involves moving and inserting operations
    // but not deleting them so a walk to collect them all then transform one by
    // one after should be safe.
    SmallVector<Operation*> candidates;
    unit.walk<WalkOrder::PreOrder>(
        [&](CompositeMemoryInterleaveOp op) { candidates.push_back(op); });
    for (auto& candidate : candidates) {
      if (lowerCompositeMemoryInterleaveOp(unit, candidate).failed()) {
        signalPassFailure();
        return;
      }
      candidate->erase();
    }

    // Lower SetTransferMaskStateOps. Involves inserting operations only so a
    // walk to collect them all then transform one by one after should be safe.
    candidates.clear();
    unit.walk<WalkOrder::PreOrder>(
        [&](SetTransferMaskStateOp op) { candidates.push_back(op); });
    for (auto& candidate : candidates) {
      if (lowerSetTransferMaskStateOp(unit, candidate).failed()) {
        signalPassFailure();
        return;
      }
      DT_CHECK_MSG(candidate->use_empty(), "uses of candidate still exist");
      candidate->erase();
    }

    // Remove set_send_dst operations that are trivially redundant.
    cleanupTriviallyRedundantSetSendDestination(unit);
  });

}
