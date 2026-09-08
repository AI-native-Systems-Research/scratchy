// bridge3.cpp -- IBM Spyre deeptools bridge 3: SentientIR -> ProgIR (dcc pass D76).
//
// Every body below is VERBATIM from the authority tree
//   /Users/nickm/git/deeptools-src   (repo_info.txt: deeptools|master|a0d29abbed...)
// and each banner gives that body's ORIGINAL <file>:<line>. PORT FROM THE AUTHORITY FILE AT THE
// CITED LINE; this unit tells you WHICH function and in WHAT ORDER.
//
// 130 units, 6879 declaration lines, from dcc/src/Conversion/SentientToProgIR/.
// Member definitions are rewritten as free functions (Class::foo -> eNNN_foo) so the unit needs no
// class declarations; the BODIES are untouched.
//
// Levels are the longest path over intra-span call edges: level 0 calls nothing else here, level N
// only levels below it.
#include "prelude.inc"


// ================================================================================================
// LEVEL 0
// ================================================================================================

// ---- 1/130  ConstructNOPInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:329  (10L)
UniformInstrInfo e001_ConstructNOPInstr(
    const SenComponents& comp, std::string comment_str) {
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::NOP);
  if (comment_str != "")
    super_instr.setCommonComment(comment_str);
  else
    super_instr.setCommonComment("NOP");
  return super_instr;
}

// ---- 2/130  ConstructReturnInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:340  (7L)
UniformInstrInfo e002_ConstructReturnInstr(
    const SenComponents& comp) {
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::RETURN);
  super_instr.setCommonComment("end of the program");
  return super_instr;
}

// ---- 3/130  normalizeBurstSize  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2562  (13L)
int e003_normalizeBurstSize(int burst_size,
                                                     SenComponents comp) {
  int max_threshold = dccExtContext().getMaxBurstSize(comp);
  DT_CHECK_MSG((burst_size >= 0 || burst_size <= max_threshold),
               "burst size is out of range");
  if (burst_size == max_threshold) {
    return 0;
  } else if (burst_size == 0) {
    return 1;
  } else {
    return burst_size;
  }
}

// ---- 4/130  rtrim  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3859  (6L)
static inline void e004_rtrim(std::string& s) {
  s.erase(std::find_if(s.rbegin(), s.rend(),
                       [](unsigned char ch) { return !std::isspace(ch); })
              .base(),
          s.end());
}

// ---- 5/130  ConstructIncrMaskInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4118  (11L)
UniformInstrInfo e005_ConstructIncrMaskInstr(
    const SenComponents& comp, sentient::IncrMaskOp op) {
  DT_CHECK_MSG(comp == PT,
               "INCRMASK instruction is currently only available in PT units");

  UniformInstrInfo super_instr;
  if (op.getDbgName().has_value())
    super_instr.setCommonComment(op.getDbgName().value().str());
  super_instr.setInstn(OpCodeT::INCRMASK);
  return super_instr;
}

// ---- 6/130  addToRegsToInit  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4157  (61L)
void e006_addToRegsToInit(
    Operation* op, std::vector<llvm::StringRef> locales,
    std::vector<int> reg_nums) {
  // From the list of GetUnitOps for the current ProgramUnitOp, collect the
  // units in {coreId, component} format where component is the SenComponents
  // representing the GetUnitOp including the corelet info.
  // The regstate_ member in psinfo is accessed by core and organized by
  // component including the corelet info.
  auto program_unit = op->getParentOfType<dataflow::ProgramUnitOp>();
  DT_CHECK_MSG(program_unit, "expecting ProgramUnitOp parent");
  std::vector<CoreAndComponent> unit_list;
  for (auto unit : program_unit.getUnits()) {
    auto get_unit = cast<dataflow::GetUnitOp>(unit.getDefiningOp());
    auto corelet = dcc::getCoreletId(get_unit);
    SenComponents comp =
        dcc::getSenComponentForProgramStateInfo(get_unit, corelet);
    auto core = dcc::getCoreId(get_unit);
    unit_list.push_back(std::make_pair(core, comp));
  }

  // For each pair of locale and register number provided, update the
  // regs_to_init_ map accordingly.
  for (auto [locale, reg_num] : llvm::zip(locales, reg_nums)) {
    // stringToRegType is uppercase.
    auto locale_upper = locale.upper();
    if (is_any_of(locale_upper, "IMM", "JCR")) continue;
    // Some reg_nums passed in (like those coming from results) may not actually
    // use a register (reg_num would be -1 to reflect this). In this case there
    // is nothing to track.
    if (reg_num < 0) continue;
    auto reg_type = ProgramAndStateInfo::stringToRegType.at(locale_upper);

    // Add the register information to the appropriate map for each unit.
    // Each unit contained in the current ProgramUnitOp will initialize the
    // register.
    // Note: This basic implementation does not differentiate regs only used
    //       in uniform regions. If a register is used in a uniform region,
    //       it will still be initialized in all the units contained by the
    //       ProgramUnitOp. This approach may potentially lead to IMMCOPYs
    //       getting added where they could be avoided. This is a temporary
    //       solution until the progtailor can be improved to no longer
    //       require DCC to do this work. If needed, we can extend this to
    //       only add the reginits to the appropriate units given uniform
    //       information.
    for (auto& unit : unit_list) {
      auto reg_map = regs_to_init_.find(unit);
      if (reg_map != regs_to_init_.end()) {
        if (auto reg_map_entry = regs_to_init_[unit].find(reg_type);
            reg_map_entry != regs_to_init_[unit].end()) {
          reg_map_entry->second.insert((unsigned)reg_num);
        } else {
          std::set<unsigned> regs({(unsigned)reg_num});
          reg_map->second[reg_type] = regs;
        }
      } else {
        UtilizedRegisters used_regs = {{reg_type, {(unsigned)reg_num}}};
        regs_to_init_[unit] = used_regs;
      }
    }
  }
}

// ---- 7/130  addPESFPLRFImmcopyToRegInit  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4226  (228L)
void e007_addPESFPLRFImmcopyToRegInit(
    mlir::Operation* const_input_op, DataFormats op_precision, int reg_val,
    sentient::SplatOp op, ProgIrGraphMap& reg_graph) {
  auto mask_constant =
      dyn_cast<sentient::ConstantOp>(op.getMask().getDefiningOp());
  DT_CHECK(mask_constant);
  DT_CHECK_MSG(mask_constant.getValue() == 0,
               "No masking is supported in register initialization");
  DT_CHECK_MSG(!op.getUnrollIncrResult(),
               "should not unroll target for a splat corresponding to register "
               "initialization");

  auto scalar_const_input =
      llvm::dyn_cast<sentient::ConstantOp>(const_input_op);
  auto vector_const_input =
      llvm::dyn_cast<sentient::VectorConstantOp>(const_input_op);
  auto query_op = llvm::dyn_cast<mlir::uniform::QueryMapOp>(const_input_op);

  auto scalar_const_val = [](sentient::ConstantOp scalar_const_input,
                             DataFormats op_precision) {
    uint32_t input = scalar_const_input.getValue();
    uint32_t val_32bits = 0u;
    if (op_precision == DataFormats::SEN169_FP16) {
      val_32bits = input | input << 16;
    } else if (op_precision == DataFormats::SENINT8 ||
               op_precision == DataFormats::SEN143_FP8) {
      val_32bits = input | input << 8 | input << 16 | input << 24;
    } else if (op_precision == DataFormats::SENINT4) {
      val_32bits = input | input << 4 | input << 8 | input << 12 |
                   (input | input << 4 | input << 8 | input << 12) << 16;
    } else if (op_precision == DataFormats::SENINT2) {
      val_32bits = input | input << 2 | input << 4 | input << 6 |
                   (input | input << 2 | input << 4 | input << 6) << 8 |
                   (input | input << 2 | input << 4 | input << 6 |
                    (input | input << 2 | input << 4 | input << 6) << 8)
                       << 16;
    } else if (op_precision == DataFormats::IEEE_FP32 ||
               op_precision == DataFormats::IEEE_INT32) {
      val_32bits = input;
    } else {
      DT_ERROR("unsupported constant type.");
    }
    std::array<uint32_t, 4> val_128bits = {val_32bits, val_32bits, val_32bits,
                                           val_32bits};
    return val_128bits;
  };

  auto vector_const_val = [](sentient::VectorConstantOp vector_const_input,
                             DataFormats op_precision) {
    std::vector<uint64_t> values;
    for (auto val_attr : vector_const_input.getValue()) {
      values.push_back((uint64_t)mlir::cast<IntegerAttr>(val_attr).getInt());
    }
    uint32_t val_32bits0 = 0u, val_32bits1 = 0u, val_32bits2 = 0u,
             val_32bits3 = 0u;
    if (op_precision == DataFormats::IEEE_FP32) {
      DT_CHECK_MSG(values.size() == 4,
                   "Expect four 32-bit values to construct 128-bit constant "
                   "LRF values per slice in PE/SFP");
      auto fillInt32 = [](uint64_t v0) { return v0; };
      val_32bits0 = fillInt32(values[0]);
      val_32bits1 = fillInt32(values[1]);
      val_32bits2 = fillInt32(values[2]);
      val_32bits3 = fillInt32(values[3]);
    } else if (op_precision == DataFormats::SEN169_FP16) {
      DT_CHECK_MSG(values.size() == 8,
                   "Expect eight 16-bit values to construct 128-bit constant "
                   "LRF values per slice in PE/SFP");
      auto fillInt32 = [](uint64_t v0, uint64_t v1) { return v0 << 16 | v1; };
      val_32bits0 = fillInt32(values[0], values[1]);
      val_32bits1 = fillInt32(values[2], values[3]);
      val_32bits2 = fillInt32(values[4], values[5]);
      val_32bits3 = fillInt32(values[6], values[7]);
    } else if (op_precision == DataFormats::SENINT8 ||
               op_precision == DataFormats::SEN143_FP8) {
      DT_CHECK_MSG(values.size() == 16,
                   "Expect sixteen 8-bit values to construct 128-bit constant "
                   "LRF values per slice in PE/SFP");
      auto fillInt32 = [](uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3) {
        return v0 << 24 | v1 << 16 | v2 << 8 | v3;
      };
      val_32bits0 = fillInt32(values[0], values[1], values[2], values[3]);
      val_32bits1 = fillInt32(values[4], values[5], values[6], values[7]);
      val_32bits2 = fillInt32(values[8], values[9], values[10], values[11]);
      val_32bits3 = fillInt32(values[12], values[13], values[14], values[15]);
    } else if (op_precision == DataFormats::SENINT4) {
      DT_CHECK_MSG(values.size() == 32,
                   "Expect thirty two 4-bit values to construct 128-bit "
                   "constant LRF values per slice in PE/SFP");
      auto fillInt32 = [](uint64_t v0, uint64_t v1, uint64_t v2, uint64_t v3,
                          uint64_t v4, uint64_t v5, uint64_t v6, uint64_t v7) {
        return v0 << 28 | v1 << 24 | v2 << 20 | v3 << 16 | v4 << 12 | v5 << 8 |
               v6 << 4 | v7;
      };
      val_32bits0 = fillInt32(values[0], values[1], values[2], values[3],
                              values[4], values[5], values[6], values[7]);
      val_32bits1 = fillInt32(values[8], values[9], values[10], values[11],
                              values[12], values[13], values[14], values[15]);
      val_32bits2 = fillInt32(values[16], values[17], values[18], values[19],
                              values[20], values[21], values[22], values[23]);
      val_32bits3 = fillInt32(values[24], values[25], values[26], values[27],
                              values[28], values[29], values[30], values[31]);
    } else {
      DT_ERROR("unsupported constant type.");
    }
    std::array<uint32_t, 4> val_128bits = {val_32bits0, val_32bits1,
                                           val_32bits2, val_32bits3};
    return val_128bits;
  };

  // find unit list from copyOp's parentOp
  SmallVector<Value> units;
  Value key;
  if (getQueryKeyAndUnitsFromParentRegion(op, key, units).failed()) {
    op->emitOpError("can't find parentOp");
    signalPassFailure();
    return;
  }
  if (key) {
    // assume the op is a global var.
    auto parent_op =
        mlir::cast<BlockArgument>(key).getParentRegion()->getParentOp();
    DT_CHECK(dyn_cast<dataflow::ProgramUnitOp>(parent_op) ||
             isa<mlir::uniform::UniformizeRegionsOp>(parent_op));
  }

  if (scalar_const_input) {
    OperandAttr val_attr;
    if (scalar_const_input->hasAttr("is_symbol")) {
      val_attr.setOperand(scalar_const_input.getValue(),
                          OperandAttr::Type::VARIABLE_SYMBOL);
    } else {
      std::array<uint32_t, 4> val_128bits =
          scalar_const_val(scalar_const_input, op_precision);
      val_attr.setOperand(val_128bits);
    }
    val_attr.setSenDataType(op_precision);
    for (auto unit : units) {
      reg_graph[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())]
          .addRegInit(reg_val, val_attr, RegType::LRF);
    }
  } else if (vector_const_input) {
    OperandAttr val_attr;
    std::array<uint32_t, 4> val_128bits =
        vector_const_val(vector_const_input, op_precision);
    val_attr.setOperand(val_128bits);
    val_attr.setSenDataType(op_precision);
    for (auto unit : units) {
      reg_graph[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())]
          .addRegInit(reg_val, val_attr, RegType::LRF);
    }
  } else if (query_op) {  // query/map operations
    FoldedVectorImmediateMap imm128bit_vals;
    std::vector<Value> empty_units;
    auto immutable_map =
        query_op.getMap().getDefiningOp<uniform::DefImmutableMappingOp>();
    auto values = immutable_map.getValuesFromKeys(units);
    for (int i = 0; i < units.size(); i++) {
      auto& unit = units[i];
      SdscFoldIdInput id = unit_foldid_map_.at(unit);
      auto& val = values[i];
      if (!val.has_value()) {
        empty_units.push_back(unit);
        continue;
      }

      if (auto val_scalar_const_op =
              llvm::dyn_cast<sentient::ConstantOp>(val->getDefiningOp())) {
        DT_CHECK_MSG(!val_scalar_const_op->hasAttr("is_symbol"),
                     "Symbolic sentient::ConstantOps not supported in uniform "
                     "operations");
        std::array<uint32_t, 4> val_128bits =
            scalar_const_val(val_scalar_const_op, op_precision);
        imm128bit_vals[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())]
                      [id] = val_128bits;
      } else if (auto val_vec_const_op =
                     llvm::dyn_cast<sentient::VectorConstantOp>(
                         val->getDefiningOp())) {
        std::array<uint32_t, 4> val_128bits =
            vector_const_val(val_vec_const_op, op_precision);
        imm128bit_vals[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())]
                      [id] = val_128bits;
      } else {
        DT_ERROR("unsupported constant type.");
      }
    }

    // if immutableMap doesn't contain an unit, use a random value for this
    // unit.
    if (!empty_units.empty()) {
      for (auto unit : empty_units) {
        DT_CHECK(!imm128bit_vals.empty());
        imm128bit_vals[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())] =
            imm128bit_vals.begin()->second;
      }
    }

    // Check for similarity and add to reg_graph.
    for (auto unit_imm_val_pair : imm128bit_vals) {
      OperandAttr imm_operand;
      std::vector<std::array<uint32_t, 4>> reg_values;
      for (auto fold_imm_val_pair : unit_imm_val_pair.second) {
        reg_values.push_back(fold_imm_val_pair.second);
      }

      auto all_same =
          std::all_of(reg_values.begin(), reg_values.end(),
                      [&reg_values](const std::array<uint32_t, 4>& element) {
                        return element == reg_values.front();
                      });
      if (!all_same) {
        for (auto fold_imm_val_pair : unit_imm_val_pair.second) {
          imm_operand.setOperand(fold_imm_val_pair.second,
                                 fold_imm_val_pair.first);
          imm_operand.setSenDataType(op_precision);
        }
      } else {
        imm_operand.setOperand(reg_values.front());
        imm_operand.setSenDataType(op_precision);
      }

      reg_graph[unit_imm_val_pair.first].addRegInit(reg_val, imm_operand,
                                                    RegType::LRF);
    }
  } else {
    DT_ERROR("Unsupported constant type for program header.");
  }
}

// ---- 8/130  addToRegInit  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:96  (41L)
void e008_addToRegInit(const mlir::Value& ssa_val,
                                                FoldedImmediateMap imm_vals,
                                                ProgIrGraphMap& reg_graph,
                                                bool is_symbolic) {
  auto reg_index = sentient::getValueRegIndex(ssa_val);
  auto locale = sentient::getValueRegLocaleAsString(ssa_val);
  auto reg_type = ProgramAndStateInfo::stringToRegType.at(locale.upper());
  for (auto unit_imm_val_pair : imm_vals) {
    OperandAttr imm_operand;

    std::vector<int64_t> reg_values;
    for (auto fold_imm_val_pair : unit_imm_val_pair.second) {
      reg_values.push_back(fold_imm_val_pair.second);
    }

    auto all_same = std::all_of(reg_values.begin(), reg_values.end(),
                                [&reg_values](const int64_t& element) {
                                  return element == reg_values.front();
                                });
    if (!all_same) {
      for (auto fold_imm_val_pair : unit_imm_val_pair.second) {
        if (is_symbolic)
          imm_operand.setOperand(fold_imm_val_pair.second,
                                 OperandAttr::Type::VARIABLE_SYMBOL,
                                 fold_imm_val_pair.first);
        else
          imm_operand.setOperand(fold_imm_val_pair.second,
                                 fold_imm_val_pair.first);
      }
    } else {
      if (is_symbolic)
        imm_operand.setOperand(reg_values.front(),
                               OperandAttr::Type::VARIABLE_SYMBOL);
      else
        imm_operand.setOperand(reg_values.front());
    }

    reg_graph[unit_imm_val_pair.first].addRegInit(reg_index, imm_operand,
                                                  reg_type);
  }
}

// ---- 9/130  GetAddressScale  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:502  (16L)
int e009_GetAddressScale(const SenComponents& comp,
                                                  StringRef locale) const {
  if (is_any_of(comp, L0SU, L0LU))
    return dccExtContext().getAddressGranularityScale({comp, L0});
  else if (is_any_of(comp, LXSU, LXLU))
    return dccExtContext().getAddressGranularityScale({comp, LX});
  else if (is_any_of(comp, L3LU, L3SU)) {
    if (locale == "ear" || locale == "ebr" || locale == "jcr")
      return dccExtContext().getAddressGranularityScale({comp, HBM});
    else if (locale == "lar" || locale == "lbr")
      return dccExtContext().getAddressGranularityScale({comp, LX});
    else
      llvm_unreachable("expected locale info to calculate scale in L3");
  }
  return 1;
}

// ---- 10/130  GetOpCodePrefix  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:519  (18L)
void e010_GetOpCodePrefix(
    const SenComponents& generic_unit, std::string& op_code) const {
  if (generic_unit == PT) {
    op_code = "PTOP";
  } else if (generic_unit == SFP) {
    op_code = "SFP";
  } else if (generic_unit == PE) {
    op_code = "PE";
  } else if (is_any_of(generic_unit, L0LU, L0SU)) {
    op_code = "L0";
  } else if (is_any_of(generic_unit, LXLU, LXSU)) {
    op_code = "LX";
  } else if (is_any_of(generic_unit, L3LU, L3SU)) {
    op_code = "L3";
  } else {
    llvm::errs() << "Unrecognized unit name in lowering to progir\n";
  }
}

// ---- 11/130  fillUnitToIdMap  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:967  (17L)
template <typename OpTy>
void e011_fillUnitToIdMap(OpTy op, UniformInstrBlock& block) {
  for (int idx = 0; idx < op.getNumRegions(); idx++) {
    auto units = op.getRegionUnitList(idx);
    for (auto unit : units) {
      if (auto unit_op = unit.template getDefiningOp<dataflow::GetUnitOp>()) {
        block.setUnitRegionIndex(getUnitName(unit_op), idx);
      } else if (auto group_op =
                     unit.template getDefiningOp<dataflow::CreateGroupOp>()) {
        for (auto subunit : group_op.getUnitIds()) {
          auto unit_op = subunit.template getDefiningOp<dataflow::GetUnitOp>();
          block.setUnitRegionIndex(getUnitName(unit_op), idx);
        }
      }
    }
  }
}

// ---- 12/130  setRegDef  --  dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:53  (7L)
void e012_setRegDef(SentientRegType regTy, int regNum) {
  if (regNum < 0 || is_any_of(regTy, SentientRegType::imm, SentientRegType::jcr,
                              SentientRegType::lccr, SentientRegType::xrfrdptr,
                              SentientRegType::xrfwrptr))
    return;
  reg_def_ctx_->regs()[dcc::getRegType(regTy)].set(regNum);
}

// ---- 13/130  addRegDefsForUnit  --  dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:61  (15L)
void e013_addRegDefsForUnit(
    mlir::dataflow::GetUnitOp get_unit) {
  DT_CHECK_MSG(enable_ctx_, "Unexpected call, reg def tracking is not enabled");
  auto &psi = tracker_.progStateInfo()[dcc::getCoreId(get_unit)];
  auto comp = dcc::getSenComponentForProgramStateInfo(
      get_unit, dcc::getCoreletId(get_unit));
  if (!psi.regDefs_.has_value()) {
    // RegDef data is initialized empty to indicate it is set.
    psi.regDefs_.emplace(ProgramAndStateInfo::RegDefs{});
  }
  for (int regTy = 0; regTy < regs_.size(); ++regTy) {
    auto key = std::make_pair(comp, static_cast<RegType>(regTy));
    psi.regDefs_.value()[key] |= regs_[regTy];
  }
}

// ---- 14/130  checkRegDefs  --  dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:106  (63L)
void e014_checkRegDefs(
    const std::map<int, ProgramAndStateInfo> &progstateinfo,
    const std::string &prog_name, int core, SenComponents comp) {
  if (!regDefChecking()) return;
  const ProgramAndStateInfo &psi = progstateinfo.at(core);
  // Compare the reg defs emitted by dcc in the psinfo with the reg refs
  // evaluated by visiting the instructions.  Ensure the instruction walk
  // didn't miss any references.
  DT_CHECK_MSG(psi.regDefs_,
               "Reg defs cannot be checked because data is not set");
  if (!is_any_of(EnumsConversion::senCompToGenericComp.at(comp),
                 SenComponents::L3, SenComponents::L3LU, SenComponents::L3SU,
                 SenComponents::LX, SenComponents::LXLU, SenComponents::LXSU,
                 SenComponents::SFP, SenComponents::PE))
    return;
  // Eval the set of referenced regs by walking inits/instructions.
  std::array<std::bitset<ProgramAndStateInfo::kMaxCompRegs>,
             RegType::MAX_VALUE + 1>
      regRefsEval;
  auto rsIt = psi.regState_.find(comp);
  if (rsIt != psi.regState_.end())
    for (auto &[regTy, regNums] : rsIt->second.getSimpleRegInit())
      for (auto &[regNum, regInit] : regNums) regRefsEval[regTy].set(regNum);
  DT_CHECK(psi.senCompProgram_.find(comp) != psi.senCompProgram_.end());
  for (auto &instr : psi.senCompProgram_.at(comp).getSimpleInstrVect()) {
    RegVisitor(comp).visitInstrRegRefs(
        instr, [&regRefsEval](RegType regTy, OperandAttr &init) {
          DT_CHECK(!init.isFolded());
          regRefsEval[regTy].set(init.asInt());
        });
  }
  // Compare the psinfo regDefs_ to the evaluated references.
  bool error = false;
  for (auto &[key, regnums] : psi.regDefs_.value()) {
    if (key.first != comp) continue;
    auto regTy = key.second;
    for (int regNum = 0; regNum < regnums.size(); ++regNum) {
      if (regnums[regNum] && !regRefsEval[regTy][regNum]) {
        error = true;
        auto compStr = EnumsConversion::senComponentsToString.at(comp);
        auto regTyStr = ProgramAndStateInfo::regTypeToString.at(
            static_cast<RegType>(regTy));
        std::cerr << "Reg ref discrepancy on " << prog_name << " core " << core
                  << " component " << compStr << std::endl
                  << "  " << regTyStr << " " << regNum << std::endl;
      }
    }
  }
  if (error) {
    auto compStr = EnumsConversion::senComponentsToString.at(comp);
    std::cerr << "Reg Inits and Program " << prog_name << " core " << core
              << " component " << compStr << std::endl;
    if (rsIt != psi.regState_.end()) {
      for (auto &[regTy, regNums] : rsIt->second.getSimpleRegInit()) {
        auto regTyStr = ProgramAndStateInfo::regTypeToString.at(regTy);
        for (auto &[regNum, regInit] : regNums)
          std::cerr << " " << regTyStr << regNum;
        std::cerr << std::endl;
      }
    }
    psi.senCompProgram_.at(comp).print(std::cerr);
  }
}

// ---- 15/130  initializeUtilizedRegisters  --  dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:429  (39L)
void e015_initializeUtilizedRegisters() {
  for (auto& unit_map : regs_to_init_) {
    auto unit_info = unit_map.first;
    auto core = unit_info.first;
    auto comp = unit_info.second;
    auto& reg_map = unit_map.second;

    // Collect the regState_ for the core at the appropriate component
    // Note: DIP requires the graph to be simple so this is replicated here
    auto reg_info = progstateinfo_[core].regState_.find(comp);
    if (reg_info == progstateinfo_[core].regState_.end()) {
      // No reg init for current component, init everything
      for (auto& reg_entry : reg_map) {
        auto reg_type = reg_entry.first;
        auto reg_nums = reg_entry.second;
        for (auto& reg_num : reg_nums)
          progstateinfo_[core].regState_[comp].addRegInit((unsigned)reg_num, 0,
                                                          reg_type);
      }
    } else {
      // Some reg init for current component, only init what isn't already in
      // the header
      auto reg_init =
          progstateinfo_[core].regState_.at(comp).getSimpleRegInit();
      for (auto& reg_entry : reg_map) {
        auto reg_type = reg_entry.first;
        auto reg_nums = reg_entry.second;
        auto reg_init_for_type = reg_init.find(reg_type);
        for (auto& reg_num : reg_nums) {
          if ((reg_init_for_type == reg_init.end()) ||
              (reg_init.at(reg_type).find((unsigned)reg_num) ==
               reg_init.at(reg_type).end()))
            progstateinfo_[core].regState_[comp].addRegInit((unsigned)reg_num,
                                                            0, reg_type);
        }
      }
    }
  }
}

// ---- 16/130  replaceProgramBodyWithSmcOp  --  dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:560  (49L)
void e016_replaceProgramBodyWithSmcOp(
    ModuleOp module_op) {
  // The lowered IR is redundant once the prog. IR exists, but the function
  // holding it stays. It is what says which program the module stands for: its
  // name and attributes (the grid a schedule was split for, say) survive, and
  // the module keeps the shape its declaration describes -- a function, not a
  // bare packet. The call in the declaration module resolves against the
  // declaration there, so this is about keeping the program's own signature
  // rather than about keeping any call valid.
  //
  // One program per module means one function, which is what every pathway
  // feeding this pass produces. More than one would make "the program's
  // function" ambiguous -- and DIP accepts only one init.smc per module -- so
  // say so rather than picking one.
  auto funcs = llvm::to_vector(module_op.getOps<mlir::func::FuncOp>());
  if (funcs.size() != 1) {
    module_op.emitError() << PASS_NAME
                          << " expects exactly one function in the module to "
                             "hold the init.smc reference, got "
                          << funcs.size();
    signalPassFailure();
    return;
  }

  mlir::func::FuncOp func = funcs.front();
  if (!func.getFunctionType().getResults().empty()) {
    func.emitError() << PASS_NAME
                     << " cannot discard the body of a function that returns "
                        "values: there would be nothing left to return";
    signalPassFailure();
    return;
  }

  // Block::clear() drops all references held by the contained operations
  // before erasing them in reverse order, so operations with uses (e.g. the
  // sentient.scalar_constant feeding a program unit) are safe to erase here.
  // Clearing every block first leaves no terminator to reference the blocks
  // that then go, keeping the entry block and the arguments it carries.
  Region& body = func.getFunctionBody();
  for (Block& block : body) block.clear();
  while (!body.hasOneBlock()) body.back().erase();

  OpBuilder builder(module_op.getContext());
  builder.setInsertionPointToEnd(&body.front());
  mlir::init::SmcOp::create(builder, func.getLoc(), dccExtContext().prog_name_);
  // func.func has no NoTerminator trait, unlike the module that used to hold
  // this op, so the emptied body needs one.
  mlir::func::ReturnOp::create(builder, func.getLoc());
}

// ---- 17/130  createNOPInstr  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:43  (8L)
UniformInstrInfo e017_createNOPInstr() {
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::NOP);
  auto random = rand();
  std::string target = "uniform_tgt_" + std::to_string(random);
  super_instr.setCommonComment(target);
  return super_instr;
}

// ---- 18/130  getUniformizedInstr  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:218  (21L)
InstrInfo e018_getUniformizedInstr(std::string unit_name) {
  InstrInfo ret;
  ret.instn_ = instn_;
  ret.instFields_ = instFields_;
  ret.setTag(tag_);
  ret.deadCode_ = deadCode_;
  std::string comment = (unit_to_comment_.count(unit_name) > 0)
                            ? unit_to_comment_[unit_name]
                            : unit_common_comment_;
  ret.setComment(comment);

  for (auto& field_operands_pair : operand_map_) {
    auto& unit_operand_map_ = field_operands_pair.second;
    // if no entry for a specific unit, use a random entry for uniform purpose
    ret.instFields_[Isa::to_instoperand(field_operands_pair.first)] =
        (unit_operand_map_.find(unit_name) != unit_operand_map_.end()
             ? unit_operand_map_.at(unit_name)
             : unit_operand_map_.begin()->second);
  }
  return ret;
}

// ---- 19/130  getRegularInstr  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:240  (11L)
InstrInfo e019_getRegularInstr() {
  InstrInfo ret;
  ret.instn_ = instn_;
  ret.instFields_ = instFields_;
  ret.setTag(tag_);
  ret.deadCode_ = deadCode_;
  DT_CHECK(unit_to_comment_.empty());
  ret.setComment(unit_common_comment_);
  DT_CHECK(operand_map_.empty());
  return ret;
}

// ---- 20/130  getCommonField  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:252  (5L)
const OperandAttr& e020_getCommonField(
    std::string field_name) const {
  DT_CHECK(operand_map_.find(field_name) == operand_map_.end());
  return instFields_.at(Isa::to_instoperand(field_name));
}

// ---- 21/130  setCommonField  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:257  (5L)
void e021_setCommonField(std::string field_name,
                                      OperandAttr field) {
  DT_CHECK(operand_map_.find(field_name) == operand_map_.end());
  instFields_[Isa::to_instoperand(field_name)] = field;
}

// ---- 22/130  hasCommonField  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:262  (3L)
bool e022_hasCommonField(std::string field_name) const {
  return instFields_.count(Isa::to_instoperand(field_name)) != 0;
}

// ---- 23/130  getUnitInstrList  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:266  (12L)
std::vector<UniformInstrInfo> e023_getUnitInstrList(
    std::string unit_name) {
  if (block_type_ == Type::REGULAR) {
    DT_CHECK(instr_lists_.size() == 1);
    return instr_lists_.front();
  }
  if (unit_to_region_idx_map_.find(unit_name) ==
      unit_to_region_idx_map_.end()) {
    return {};
  }
  return instr_lists_.at(unit_to_region_idx_map_.at(unit_name));
}

// ---- 24/130  insertInstruction  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:322  (5L)
void e024_insertInstruction(UniformInstrInfo instr) {
  DT_CHECK((int(current_region_) - int(instr_lists_.size())) <= 0);
  if (current_region_ == instr_lists_.size()) instr_lists_.emplace_back();
  instr_lists_.at(current_region_).push_back(instr);
}

// ---- 25/130  setCurrentRegion  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:327  (4L)
void e025_setCurrentRegion(unsigned region_id) {
  current_region_ = region_id;
  if (current_region_ > 0) DT_CHECK(block_type_ == Type::UNIFORM);
}

// ---- 26/130  getMaxInstrSize  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:331  (8L)
size_t e026_getMaxInstrSize() {
  size_t size = 0;
  if (instr_lists_.size() == 0) return size;
  for (int i = 0; i < instr_lists_.size(); i++) {
    size = std::max(size, instr_lists_[i].size());
  }
  return size;
}

// ---- 27/130  getCurrentRegionInstrSize  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:360  (5L)
size_t e027_getCurrentRegionInstrSize() {
  return (current_region_ >= instr_lists_.size())
             ? 0  // empty list
             : instr_lists_.at(current_region_).size();
}

// ---- 28/130  getInstr  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:365  (8L)
UniformInstrInfo& e028_getInstr(int idx, std::string unit_name) {
  if (unit_to_region_idx_map_.find(unit_name) ==
      unit_to_region_idx_map_.end()) {
    // TODO need to change to use MAX region
    return instr_lists_.at(0).at(idx);
  }
  return instr_lists_.at(unit_to_region_idx_map_.at(unit_name)).at(idx);
}

// ---- 29/130  getLastInstr  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:374  (3L)
UniformInstrInfo& e029_getLastInstr() {
  return instr_lists_.at(current_region_).back();
}

// ---- 30/130  appendEmptyUniformRegion  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:378  (4L)
void e030_appendEmptyUniformRegion() {
  DT_CHECK(block_type_ == Type::UNIFORM);
  instr_lists_.emplace_back();
}

// ---- 31/130  getMaxInstrSize  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:426  (7L)
size_t e031_getMaxInstrSize() {
  size_t size = 0;
  for (auto block : blocks_) {
    size += block.getMaxInstrSize();
  }
  return size;
}

// ---- 32/130  appendRegularBlock  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:442  (3L)
UniformInstrBlock& e032_appendRegularBlock() {
  return blocks_.emplace_back(UniformInstrBlock::Type::REGULAR);
}

// ---- 33/130  appendUniformBlock  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:445  (3L)
UniformInstrBlock& e033_appendUniformBlock() {
  return blocks_.emplace_back(UniformInstrBlock::Type::UNIFORM);
}

// ---- 34/130  getUniformInstr  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:449  (6L)
UniformInstrInfo& e034_getUniformInstr(InstrIndex index) {
  auto block_idx = std::get<0>(index);
  auto region_idx = std::get<1>(index);
  auto instr_idx = std::get<2>(index);
  return blocks_.at(block_idx).getInstrLists().at(region_idx).at(instr_idx);
}

// ---- 35/130  doesInstrWithThisIndexExist  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:456  (7L)
bool e035_doesInstrWithThisIndexExist(InstrIndex index) {
  auto block_idx = std::get<0>(index);
  auto region_idx = std::get<1>(index);
  auto instr_idx = std::get<2>(index);
  return (instr_idx <
          blocks_.at(block_idx).getInstrLists().at(region_idx).size());
}

// ---- 36/130  appendEmptyUniformRegion  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:497  (4L)
void e036_appendEmptyUniformRegion() {
  DT_CHECK(blocks_.back().getType() == UniformInstrBlock::Type::UNIFORM);
  blocks_.back().appendEmptyUniformRegion();
}

// ---- 37/130  getUnitRegionIndex  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.hpp:208  (6L)
size_t e037_getUnitRegionIndex(std::string unit) {
    if (getType() == UniformInstrBlock::Type::REGULAR) return 0;
    if (unit_to_region_idx_map_.find(unit) == unit_to_region_idx_map_.end())
      return -1;
    return unit_to_region_idx_map_.at(unit);
  }

// ---- 38/130  verifyOnTheFlyConversions  --  dcc/src/Conversion/SentientToProgIR/Utils.cpp:27  (56L)
void e038_verifyOnTheFlyConversions(const dcc::DccExtContext& dcc_ext_ctx,
                               SenComponents comp, std::string opA_precision,
                               std::string opB_precision,
                               std::string opC_precision, int src0_operand_idx,
                               std::string compute_precision,
                               std::string result_precision) {
  if (dcc_ext_ctx.getArch() <= RCUDD1A_ISA) {
    DT_CHECK_MSG(result_precision == compute_precision ||
                     is_any_of(result_precision, "int4", "int8"),
                 "Unsupported result precision conversion in DD2");
  } else {
    // Sentient 1.5
    if (opA_precision != compute_precision) {
      if (is_any_of(opA_precision, "fp24", "int24"))
        DT_CHECK_MSG(
            src0_operand_idx == 0 && comp == PE,
            "Unsupported input on the fly conversion for FMA/FMUL/FNMS");
      else
        DT_CHECK_MSG(
            opA_precision == "fp16" && compute_precision == "fp32",
            "Unsupported input on the fly conversion for FMA/FMUL/FNMS");
    }
    if (opB_precision != compute_precision) {
      if (is_any_of(opB_precision, "fp24", "int24"))
        DT_CHECK_MSG(
            src0_operand_idx == 1 && comp == PE,
            "Unsupported input on the fly conversion for FMA/FMUL/FNMS");
      else
        DT_CHECK_MSG(
            opB_precision == "fp16" && compute_precision == "fp32",
            "Unsupported input on the fly conversion for FMA/FMUL/FNMS");
    }
    if (opC_precision != compute_precision) {
      if (is_any_of(opC_precision, "fp24", "int24"))
        DT_CHECK_MSG(
            src0_operand_idx == 2 && comp == PE,
            "Unsupported input on the fly conversion for FMA/FMUL/FNMS");
      else
        DT_CHECK_MSG(
            opC_precision == "fp16" && compute_precision == "fp32",
            "Unsupported input on the fly conversion for FMA/FMUL/FNMS");
    }
    if (compute_precision != result_precision) {
      DT_CHECK_MSG(is_any_of(compute_precision, "fp16", "fp32"),
                   "Unsupported compute precision for FMA/FMUL/FNMS");
      if (is_any_of(result_precision, "fp16", "bf16"))
        DT_CHECK_MSG(
            compute_precision == "fp32",
            "Unsupported output on the fly conversion for FMA/FMUL/FNMS");
      else
        DT_CHECK_MSG(
            is_any_of(result_precision, "int4", "int8"),
            "Unsupported output on the fly conversion for FMA/FMUL/FNMS");
    }
  }
}

// ---- 39/130  getProperConsumer  --  dcc/src/Conversion/SentientToProgIR/Utils.cpp:86  (14L)
OperandAttr e039_getProperConsumer(const dcc::DccExtContext& dcc_ext_ctx,
                              SenComponents consumer, std::string name) {
  if (dcc_ext_ctx.getArch() >= RCUDD1A_ISA) {
    if (is_any_of(consumer, PT, L0SU) ||
        (dcc_ext_ctx.getArch() == SEN1P5_ISA && consumer == CROSSPTNLINK)) {
      return OperandAttr("sfp", OperandAttr::Type::DESCRIPTIVE);
    } else {
      DT_CHECK(is_any_of(consumer, PE, SFP));
      return OperandAttr(name, OperandAttr::Type::DESCRIPTIVE);
    }
  } else {
    return OperandAttr(name, OperandAttr::Type::DESCRIPTIVE);
  }
}

// ---- 40/130  updateProperConsumer  --  dcc/src/Conversion/SentientToProgIR/Utils.cpp:102  (14L)
void e040_updateProperConsumer(const dcc::DccExtContext& dcc_ext_ctx,
                          SenComponents consumer, std::string name,
                          OperandAttr& value, SdscFoldIdInput& id) {
  if (dcc_ext_ctx.getArch() >= RCUDD1A_ISA) {
    DT_CHECK(is_any_of(consumer, PE, PT, SFP, L0SU));
    if (is_any_of(consumer, PT, L0SU)) {
      value.setOperand("sfp", OperandAttr::Type::DESCRIPTIVE, id);
    } else {
      value.setOperand(name, OperandAttr::Type::DESCRIPTIVE, id);
    }
  } else {
    value.setOperand(name, OperandAttr::Type::DESCRIPTIVE, id);
  }
}

// ---- 41/130  getAddrWraparounded  --  dcc/src/Conversion/SentientToProgIR/Utils.cpp:117  (10L)
int64_t e041_getAddrWraparounded(int64_t val, const SenComponents& comp,
                            const DccExtContext& dcc_ext_cxt) {
  if (is_any_of(comp, L0LUROW0, L0LU, L0SU)) {
    return val % (int64_t)dcc_ext_cxt.getL0CapacityPerSlice();
  } else if (is_any_of(comp, SenComponents::PTXRF)) {
    auto num_xrf_per_ptrow = dcc_ext_cxt.getNumXRFPerPTRow();
    return (val + num_xrf_per_ptrow) % num_xrf_per_ptrow;
  }
  return val;
}


// ================================================================================================
// LEVEL 1
// ================================================================================================

// ---- 42/130  ConstructMVLoopInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:48  (63L)
UniformInstrInfo e042_ConstructMVLoopInstr(
    sentient::ForOp for_op, const SenComponents& comp,
    std::map<Operation*, std::string>& labels, int& labels_counter) {
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::MVLOOPCNT);
  auto bound_definition = for_op.getBound().getDefiningOp();
  auto indices = mlir::cast<ArrayAttr>(for_op.getRegIndicesAttr());
  std::string lccr =
      std::to_string(mlir::cast<IntegerAttr>(indices[0]).getInt());
  auto loop_dbg_name = for_op.getDbgName();
  if (isa<sentient::ConstantOp>(bound_definition)) {
    int value =
        llvm::dyn_cast<sentient::ConstantOp>(bound_definition).getValue();
    super_instr.setCommonField("imm", value);
    if (loop_dbg_name.has_value())
      super_instr.setCommonComment(loop_dbg_name.value().str());
    else
      super_instr.setCommonComment("for-loop-imm-lccr-" + lccr);
  } else if (auto query_op = dyn_cast<uniform::QueryMapOp>(bound_definition)) {
    if (loop_dbg_name.has_value())
      super_instr.setCommonComment(loop_dbg_name.value().str());
    else
      super_instr.setCommonComment("for-loop-imm-lccr-" + lccr);
    super_instr.setOperandMap(
        OperandMap(unit_foldid_map_, dccExtContext(), "imm", query_op));
  } else if (auto symbol_op =
                 dyn_cast<symbol::CreateSymbolOp>(bound_definition)) {
    if (loop_dbg_name.has_value())
      super_instr.setCommonComment(loop_dbg_name.value().str());
    else
      super_instr.setCommonComment("for-loop-imm-lccr-" + lccr);

    if (!symbol_op->hasAttr("SymbolId")) {
      symbol_op->emitError("No symbol id present in the loop bound");
      signalPassFailure();
    }

    auto symbol_idx = symbol_op.getSymbolID();
    super_instr.setCommonField(
        "imm", OperandAttr(symbol_idx, OperandAttr::Type::VARIABLE_SYMBOL));
  } else {
    std::string src0 =
        "jcr" + std::to_string(getValueRegIndex(for_op.getBound()));
    super_instr.setCommonField("dyn_loop", 1);
    super_instr.setCommonField(
        "src0", OperandAttr(src0, OperandAttr::Type::DESCRIPTIVE));

    std::string label_str = "for-loop-jcr-" + std::to_string(labels_counter++);
    if (loop_dbg_name.has_value())
      label_str = label_str + "(" + loop_dbg_name.value().str() + ")";
    super_instr.setCommonComment(label_str + "-begin");
    std::string end_label_str = label_str + "-end";
    auto next_op = for_op->getNextNode();
    if (labels.count(next_op) == 0)
      labels[next_op] = end_label_str;
    else
      end_label_str = labels[next_op];
    super_instr.setCommonField(
        "pc_target", OperandAttr(end_label_str, OperandAttr::Type::INSTR_TAG));
  }

  return super_instr;
}

// ---- 43/130  ConstructAssignInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:113  (204L)
std::optional<UniformInstrInfo>
e043_ConstructAssignInstr(
    const SenComponents& comp, mlir::Value src, mlir::Value tgt,
    int element_size, std::optional<CodeQualityStats>& cq_stats,
    bool program_header) {
  UniformInstrInfo super_instr;
  int region_idx = findUserUniformRegion(tgt);
  auto src_locale = sentient::getValueRegLocaleAsString(src);
  auto tgt_locale = sentient::getValueRegLocaleAsString(tgt);
  auto src_index = sentient::getValueRegIndex(src);
  auto tgt_index = sentient::getValueRegIndex(tgt);
  std::string imm_field_name = is_any_of(comp, LXLU, LXSU) ? "lrfimm" : "imm";
  if (src_index == tgt_index) return std::nullopt;

  DT_CHECK(
      !(src_locale == "imm" && (tgt_locale == "lbr" || tgt_locale == "ebr")) &&
      "LBR/EBR <- IMM not supported");
  if (src_locale == "imm" && tgt_locale == "jcr") {
    super_instr.setInstn(OpCodeT::JIMMCOPY);
    super_instr.setCommonField("jcr_target", tgt_index);
    super_instr.setCommonComment("JCR <- IMM");
    if (auto const_op =
            llvm::dyn_cast<sentient::ConstantOp>(src.getDefiningOp())) {
      super_instr.setCommonField("imm", (int)const_op.getValue());
    } else if (auto query_op =
                   dyn_cast<uniform::QueryMapOp>(src.getDefiningOp())) {
      OperandMap operand_map(unit_foldid_map_, dccExtContext(), "imm", query_op,
                             OperandMap::Mode::none);
      super_instr.setOperandMap(operand_map);
    } else if (auto symbol_op =
                   dyn_cast<symbol::CreateSymbolOp>(src.getDefiningOp())) {
      super_instr.setCommonField(
          "imm", OperandAttr(symbol_op.getSymbolID(),
                             OperandAttr::Type::VARIABLE_SYMBOL));
    } else {
      DT_ERROR(
          "unable to generate instruction JIMMCOPY with provided details.");
      signalPassFailure();
    }
  } else if (src_locale == "jcr" && tgt_locale == "jcr") {
    super_instr.setInstn(OpCodeT::JADD);
    super_instr.setCommonField("jcr_select", 1);
    super_instr.setCommonField("jcr_target", tgt_index);
    super_instr.setCommonField("src0", src_index);
    super_instr.setCommonField("imm", 0);
    super_instr.setCommonComment("JCR <- JCR");
  } else if (src_locale == "imm" && tgt_locale == "lrf") {
    if (program_header) {
      // don't generate the program here. Move to Program header
      return std::nullopt;
    } else {
      super_instr.setInstn(OpCodeT::IMMCOPY);
      super_instr.setCommonField("src0", tgt_index);
      super_instr.setCommonComment("LRF <- IMM");
      if (auto query_op = src.getDefiningOp<uniform::QueryMapOp>()) {
        float scale = static_cast<float>(element_size) / 8.0 /
                      (float)GetAddressScale(comp, tgt);
        super_instr.setOperandMap(OperandMap(unit_foldid_map_, dccExtContext(),
                                             imm_field_name, query_op,
                                             OperandMap::Mode::none, scale));
      } else if (auto uniform_op =
                     src.getDefiningOp<uniform::UniformizeRegionsOp>()) {
        auto yield_op = dyn_cast<uniform::YieldOp>(
            uniform_op.getRegion(region_idx).front().getTerminator());
        int result_index = getResultNum(uniform_op, src);
        return ConstructAssignInstr(comp, yield_op.getOperand(result_index),
                                    tgt, element_size, cq_stats,
                                    program_header);
      } else if (auto const_op = llvm::dyn_cast<sentient::ConstantOp>(
                     src.getDefiningOp())) {
        float scale = static_cast<float>(element_size) / 8.0 /
                      (float)GetAddressScale(comp, tgt);
        super_instr.setCommonField(
            imm_field_name,
            (int)(static_cast<int>(const_op.getValue()) * scale));
      } else if (auto symbol_op =
                     dyn_cast<symbol::CreateSymbolOp>(src.getDefiningOp())) {
        super_instr.setCommonField(
            imm_field_name, OperandAttr(symbol_op.getSymbolID(),
                                        OperandAttr::Type::VARIABLE_SYMBOL));
      } else {
        DT_ERROR(
            "unable to generate instruction JIMMCOPY with provided details.");
        signalPassFailure();
      }
    }
  } else if (src_locale == "lrf" && tgt_locale == "lrf") {
    super_instr.setInstn(OpCodeT::LRFREGCOPY);
    super_instr.setCommonField("src0", tgt_index);
    super_instr.setCommonField("src1", src_index);
    super_instr.setCommonComment("LRF <- LRF");
  } else if (src_locale == "imm" && tgt_locale == "xrfrdptr") {
    super_instr.setInstn(OpCodeT::XRFACCESS);
    super_instr.setCommonComment("set xrf rd index");
    super_instr.setCommonField(
        "rdptr_upd", OperandAttr("set", OperandAttr::Type::DESCRIPTIVE));
    super_instr.setCommonField(
        "wrptr_upd", OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
    int val = (int)llvm::dyn_cast<sentient::ConstantOp>(src.getDefiningOp())
                  .getValue();
    super_instr.setCommonField(
        "rdptr_imm", getAddrWraparounded(val, PTXRF, dccExtContext()));
  } else if (src_locale == "imm" && tgt_locale == "xrfwrptr") {
    super_instr.setInstn(OpCodeT::XRFACCESS);
    super_instr.setCommonComment("set xrf wr index");
    super_instr.setCommonField(
        "wrptr_upd", OperandAttr("set", OperandAttr::Type::DESCRIPTIVE));
    super_instr.setCommonField(
        "rdptr_upd", OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
    int val = (int)llvm::dyn_cast<sentient::ConstantOp>(src.getDefiningOp())
                  .getValue();
    super_instr.setCommonField(
        "wrptr_imm", getAddrWraparounded(val, PTXRF, dccExtContext()));
  } else if (src_locale == tgt_locale && tgt_index == src_index) {
    // doesn't require instruction.
  } else if (src_locale == "imm" &&
             (tgt_locale == "lar" || tgt_locale == "ear")) {
    if (program_header) {
      // don't generate the program here. Move to Program header
      return std::nullopt;
    } else {
      super_instr.setInstn(tgt_locale == "lar" ? OpCodeT::LARIMM
                                               : OpCodeT::EARIMM);
      super_instr.setCommonComment(tgt_locale == "lar" ? "LAR <- IMM"
                                                       : "EAR <- IMM");
      super_instr.setCommonField("src0", tgt_index);
      if (auto const_op = dyn_cast<sentient::ConstantOp>(src.getDefiningOp())) {
        float scale = static_cast<float>(element_size) / 8.0 /
                      (float)GetAddressScale(comp, tgt);
        super_instr.setCommonField("imm", (int)(const_op.getValue() * scale));
      } else if (auto get_unit_op =
                     dyn_cast<dataflow::GetUnitOp>(src.getDefiningOp())) {
        super_instr.setCommonField("imm", (int)(dcc::getCoreId(get_unit_op)));
      } else if (auto query_op = src.getDefiningOp<uniform::QueryMapOp>()) {
        auto value_ops = getListOfValueOpsFromUniformMapping(query_op);
        if (isa<sentient::ConstantOp>(value_ops.front().getDefiningOp())) {
          float scale = static_cast<float>(element_size) / 8.0 /
                        (float)GetAddressScale(comp, tgt);
          super_instr.setOperandMap(
              OperandMap(unit_foldid_map_, dccExtContext(), imm_field_name,
                         query_op, OperandMap::Mode::none, scale));
        } else if (isa<dataflow::GetUnitOp>(
                       value_ops.front().getDefiningOp())) {
          super_instr.setOperandMap(
              OperandMap(unit_foldid_map_, dccExtContext(), imm_field_name,
                         query_op, OperandMap::Mode::unit_id));
        } else {
          DT_ERROR(
              "unable to generate instruction LAR/EARIMMCOPY with provided "
              "details.");
          signalPassFailure();
        }
      } else if (auto symbol_op =
                     dyn_cast<symbol::CreateSymbolOp>(src.getDefiningOp())) {
        super_instr.setCommonField(
            "imm", OperandAttr(symbol_op.getSymbolID(),
                               OperandAttr::Type::VARIABLE_SYMBOL));
      } else {
        DT_ERROR(
            "unable to generate instruction LAR/EARIMMCOPY with provided "
            "details.");
        signalPassFailure();
      }
    }
  } else if ((src_locale == "lar" && tgt_locale == "lar") ||
             (src_locale == "ear" && tgt_locale == "ear")) {
    bool is_ear = src_locale == "ear";
    super_instr.setInstn(is_ear ? OpCodeT::EARREGCOPY : OpCodeT::LARREGCOPY);
    super_instr.setCommonField("src0", tgt_index);
    super_instr.setCommonField("src1", src_index);
    super_instr.setCommonComment(is_ear ? "EAR <- EAR" : "LAR <- LAR");
  } else if (src_locale == "imm" && tgt_locale == "gtr") {
    super_instr.setInstn(OpCodeT::GTRIMM);
    super_instr.setCommonField("src0", tgt_index);
    super_instr.setCommonComment("GTR <- IMM ");
    if (auto query_op = src.getDefiningOp<uniform::QueryMapOp>()) {
      super_instr.setOperandMap(OperandMap(unit_foldid_map_, dccExtContext(),
                                           imm_field_name, query_op));
    } else if (auto multicast_op =
                   src.getDefiningOp<dataflow::CreateMulticastGroupOp>()) {
      int gtr = (int)dcc::dataflow::utils::encodeMulticastGroupInfo(
          multicast_op, dcc_ext_ctx_);
      super_instr.setCommonField("imm", gtr);
      super_instr.setCommonComment(
          "GTR <- IMM: " +
          dcc::dataflow::utils::decodeMulticastGroupInfo(gtr, dcc_ext_ctx_));
    }
  } else {
    if (mlir::isa<BlockArgument>(tgt)) {
      mlir::cast<BlockArgument>(tgt).getOwner()->getParentOp()->emitError(
          "Unable to create assign instruction");
    } else {
      tgt.getDefiningOp()->emitError("Unable to create assign instruction");
    }
    signalPassFailure();
  }
  if (cq_stats.has_value() &&
      is_any_of(super_instr.getInstn(), OpCodeT::EARREGCOPY,
                OpCodeT::LARREGCOPY, OpCodeT::IMMCOPY, OpCodeT::JIMMCOPY,
                OpCodeT::LRFCOPY, OpCodeT::LRFREGCOPY))
    ++cq_stats.value().num_copy_ops_;

  return super_instr;
}

// ---- 44/130  ConstructBranchExitInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:319  (9L)
UniformInstrInfo e044_ConstructBranchExitInstr(
    sentient::YieldOp yieldOp, const SenComponents& comp) {
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::NOP);
  super_instr.setCommonField("be",
                             OperandAttr("be", OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonComment("Branch End");
  return super_instr;
}

// ---- 45/130  ConstructSyncInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:348  (96L)
UniformInstrInfo e045_ConstructSyncInstr(
    const SenComponents& comp, bool soft, int implicit_sync_boundary_tile_size,
    std::string mode, const ArrayAttr units,
    std::optional<StringRef> dbg_name) {
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::SYNC);
  std::map<std::string, int> mode_map{
      {"send", 1}, {"recv", 2}, {"sendrecv", 3}};
  std::map<std::string, int> lx_consumer_map{{"l3lu", 4},   {"l3su", 8},
                                             {"lxlu", 16},  {"lxsu", 32},
                                             {"lxluN", 64}, {"lxsuN", 128}};
  std::map<std::string, int> l3_consumer_map{{"l3lu", 4},   {"l3su", 8},
                                             {"lxlu0", 16}, {"lxsu0", 32},
                                             {"lxlu1", 64}, {"lxsu1", 128}};

  auto isL0Tethered = [&]() -> bool {
    bool l0_tethered = false;
    if (dcc_ext_ctx_.artifacts_) {
      l0_tethered = dcc_ext_ctx_.artifacts_->l0TetheredMode_;
    }
    return l0_tethered;
  };

  int synctag = mode_map[mode];
  std::string sync_comment = mode;
  if (is_any_of(comp, LXLU, LXSU)) {
    for (auto unit : units) {
      std::string consumer =
          stringifySentientLoadConsumer(
              mlir::cast<SentientLoadConsumerAttr>(unit).getValue())
              .str();
      sync_comment += "-" + consumer;
      synctag += lx_consumer_map[consumer];
    }
  } else if (is_any_of(comp, L3LU, L3SU)) {
    for (auto unit : units) {
      std::string consumer =
          stringifySentientLoadConsumer(
              mlir::cast<SentientLoadConsumerAttr>(unit).getValue())
              .str();
      sync_comment += "-" + consumer;
      synctag += l3_consumer_map[consumer];
    }
    if (soft && dccExtContext().getArch() >= RCUDD1A_ISA) {
      DT_CHECK_MSG(comp == L3LU, "soft sync only supported in L3LU");
      sync_comment += "-soft";
      super_instr.setCommonField(
          "soft", OperandAttr("yes", OperandAttr::Type::DESCRIPTIVE));
    }
  } else if (is_any_of(comp, L0LU, L0SU)) {
    // Handling of implicit sync feature of L0 units in RCUDD1a.
    if (implicit_sync_boundary_tile_size > 0) {
      DT_CHECK_MSG(dccExtContext().getArch() >= RCUDD1A_ISA,
                   "implicit sync is supported only on RCUDD1a");
      DT_CHECK_MSG(implicit_sync_boundary_tile_size > 0,
                   "implicit sync requires positive tile size");
      DT_CHECK_MSG(mode == "sendrecv",
                   "implicit sync requires mode to be sendrecv");
      sync_comment += "-implicit";
      super_instr.setCommonField(
          "implicit", OperandAttr("yes", OperandAttr::Type::DESCRIPTIVE));
      super_instr.setCommonField("tilesize", implicit_sync_boundary_tile_size);
      if (dcc_ext_ctx_.getArch() >= IsaCoreGen::SEN1P5_ISA) {
        // for implicit sync in sen1.5, sync tag is required.
        // TEMP: hardcode it to both for l0TetheredMode = true.
        // TODO: need to rely on sync node in scheduleTree to fill this field.
        super_instr.setCommonField("syncdest",
                                   OperandAttr(isL0Tethered() ? "both" : "self",
                                               OperandAttr::Type::DESCRIPTIVE));
      }
      if (dbg_name.has_value())
        super_instr.setCommonComment(dbg_name.value().str());
      else
        super_instr.setCommonComment("sync " + sync_comment);

      // we shouldn't set synctag attribute for implicit sync; hence returning
      return super_instr;
    } else {
      if (dcc_ext_ctx_.getArch() >= IsaCoreGen::SEN1P5_ISA) {
        // for explicit sync in sen1.5, sync tag is required.
        // TEMP: hardcode it to both for l0TetheredMode = true.
        // TODO: need to rely on sync node in scheduleTree to fill this field.
        super_instr.setCommonField("syncdest",
                                   OperandAttr(isL0Tethered() ? "both" : "self",
                                               OperandAttr::Type::DESCRIPTIVE));
      }
    }
  }

  super_instr.setCommonField("synctag", synctag);
  if (dbg_name.has_value())
    super_instr.setCommonComment(dbg_name.value().str());
  else
    super_instr.setCommonComment("sync " + sync_comment);
  return super_instr;
}

// ---- 46/130  ConstructJMPInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:446  (11L)
UniformInstrInfo e046_ConstructJMPInstr(
    const SenComponents& comp, std::string target) {
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::JCMP);
  super_instr.setCommonComment("jump");
  super_instr.setCommonField(
      "mode", OperandAttr("always", OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField("pc_target",
                             OperandAttr(target, OperandAttr::Type::INSTR_TAG));
  return super_instr;
}

// ---- 47/130  ConstructJCMPInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:458  (227L)
UniformInstrInfo e047_ConstructJCMPInstr(
    sentient::IfOp if_op, const SenComponents& comp,
    std::map<Operation*, std::string>& labels, int& labels_counter) {
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::JCMP);
  auto mode = if_op.getPredicate();

  auto is_symbol = [&](mlir::Value val, bool& flag) {
    if (isa<BlockArgument>(val)) return false;
    if (auto symbol_op =
            llvm::dyn_cast<symbol::CreateSymbolOp>(val.getDefiningOp())) {
      flag = true;
      return true;
    }

    return false;
  };

  bool is_imm_symbol = false;
  auto lhs_locale =
      is_symbol(if_op.getLhs(), is_imm_symbol)
          ? "imm"
          : sentient::getValueRegLocaleAsString(if_op.getLhs()).str();
  auto rhs_locale =
      is_symbol(if_op.getRhs(), is_imm_symbol)
          ? "imm"
          : sentient::getValueRegLocaleAsString(if_op.getRhs()).str();

  // invert condition checking because we jump when true
  if (mode == CmpIPredicate::eq) {
    super_instr.setCommonField(
        "mode", OperandAttr("ne", OperandAttr::Type::DESCRIPTIVE));
  } else if (mode == CmpIPredicate::ne) {
    super_instr.setCommonField(
        "mode", OperandAttr("eq", OperandAttr::Type::DESCRIPTIVE));
  } else if (mode == CmpIPredicate::slt) {
    if (lhs_locale == "imm") {
      super_instr.setCommonField(
          "mode", OperandAttr("ge", OperandAttr::Type::DESCRIPTIVE));
    } else if (rhs_locale == "imm") {
      super_instr.setCommonField(
          "mode", OperandAttr("le", OperandAttr::Type::DESCRIPTIVE));
    } else {
      super_instr.setCommonField(
          "mode", OperandAttr("ge", OperandAttr::Type::DESCRIPTIVE));
    }
  } else if (mode == CmpIPredicate::sle) {
    if (lhs_locale == "imm") {
      super_instr.setCommonField(
          "mode", OperandAttr("gt", OperandAttr::Type::DESCRIPTIVE));
    } else if (rhs_locale == "imm") {
      super_instr.setCommonField(
          "mode", OperandAttr("lt", OperandAttr::Type::DESCRIPTIVE));
    } else {
      super_instr.setCommonField(
          "mode", OperandAttr("gt", OperandAttr::Type::DESCRIPTIVE));
    }
  } else if (mode == CmpIPredicate::sgt) {
    if (lhs_locale == "imm") {
      super_instr.setCommonField(
          "mode", OperandAttr("le", OperandAttr::Type::DESCRIPTIVE));
    } else if (rhs_locale == "imm") {
      super_instr.setCommonField(
          "mode", OperandAttr("ge", OperandAttr::Type::DESCRIPTIVE));
    } else {
      super_instr.setCommonField(
          "mode", OperandAttr("le", OperandAttr::Type::DESCRIPTIVE));
    }
  } else if (mode == CmpIPredicate::sge) {
    if (lhs_locale == "imm") {
      // imm >= LCCR/JCR {S} --> sentient IR
      // imm < LCCR/JCR {S} --> prog IR (negated condition)
      // imm < LCCR/JCR {S} --> prog IR (src1 comp src0) & imm have to be src1
      super_instr.setCommonField(
          "mode", OperandAttr("lt", OperandAttr::Type::DESCRIPTIVE));
    } else if (rhs_locale == "imm") {
      // LCCR/JCR >= imm {S} --> sentient IR
      // LCCR/JCR < imm {S} --> prog IR (negated condition)
      // imm > LCCR/JCR {S} --> prog IR (src1 comp src0) & imm have to be src1
      super_instr.setCommonField(
          "mode", OperandAttr("gt", OperandAttr::Type::DESCRIPTIVE));
    } else {
      // LCCR/JCR1 >= LCCR/JCR2 {S} --> sentient IR
      // LCCR/JCR1 < LCCR/JCR2 {S} --> prog IR (negated condition)
      // src0 = LCCR/JCR2 and src1 = LCCR/JCR1
      super_instr.setCommonField(
          "mode", OperandAttr("lt", OperandAttr::Type::DESCRIPTIVE));
    }
  } else {
    if_op.emitError("Wrong mode");
    signalPassFailure();
  }

  int imm_value = -1;
  bool is_imm = false, use_jcr = false;

  std::string label = "if-label-" + std::to_string(labels_counter++);
  std::string end_label_str = label + "-end";
  auto next_node = if_op->getNextNode();

  if (labels.find(next_node) == labels.end()) {
    labels[next_node] = end_label_str;
  }

  if (!if_op.getElseRegion().empty()) {
    bool set_else_tag = true;
    std::string else_label_str = label + "-else";
    auto else_body_first_op = &if_op.getElseRegion().front().front();
    if (auto yield_op = llvm::dyn_cast<sentient::YieldOp>(else_body_first_op)) {
      if (yield_op->getNumOperands() == 0) {
        set_else_tag = false;
      } else {
        // This is to avoid comparing with all operands.
        // We break the loop after encountering first mismatch.
        set_else_tag = false;
        // Check if we don't generate instructions for operands going forward
        for (int i = 0; i < yield_op->getNumOperands(); i++) {
          auto yield_operand_locale =
              getValueRegLocaleAsString(yield_op.getOperand(i));
          auto yield_operand_index = getValueRegIndex(yield_op.getOperand(i));
          auto if_op_result_locale =
              getValueRegLocaleAsString(if_op.getResult(i));
          auto if_op_result_index = getValueRegIndex(if_op.getResult(i));
          if (yield_operand_locale != if_op_result_locale ||
              yield_operand_index != if_op_result_index) {
            set_else_tag = true;
            break;
          }
        }
      }
    }

    if (set_else_tag) {
      super_instr.setCommonField(
          "pc_target",
          OperandAttr(else_label_str, OperandAttr::Type::INSTR_TAG));
    } else {
      super_instr.setCommonField(
          "pc_target",
          OperandAttr(end_label_str, OperandAttr::Type::INSTR_TAG));
    }

    if (labels.count(else_body_first_op) == 0 && set_else_tag) {
      labels[else_body_first_op] = else_label_str;
    }
  } else {
    super_instr.setCommonField(
        "pc_target", OperandAttr(end_label_str, OperandAttr::Type::INSTR_TAG));
  }

  auto get_imm_value_or_symbol_id = [&](mlir::Value val) {
    DT_CHECK_MSG(!isa<BlockArgument>(val),
                 "Immediate value cannot be a block argument");
    if (auto const_op =
            llvm::dyn_cast<sentient::ConstantOp>(val.getDefiningOp())) {
      return const_op.getValue();
    } else if (auto symbol_op = llvm::dyn_cast<symbol::CreateSymbolOp>(
                   val.getDefiningOp())) {
      if (!symbol_op->hasAttr("SymbolId")) {
        symbol_op->emitError("No symbol id present in the loop bound");
        signalPassFailure();
      }
      return symbol_op.getSymbolID();
    } else {
      DT_ERROR("Unknown definition of immediate value");
    }
  };

  std::string src0, src1;
  if (lhs_locale == "imm" && rhs_locale == "jcr") {
    is_imm = use_jcr = true;
    src0 = "JCR" + std::to_string(sentient::getValueRegIndex(if_op.getRhs()));
    imm_value = get_imm_value_or_symbol_id(if_op.getLhs());
  } else if (lhs_locale == "jcr" && rhs_locale == "imm") {
    is_imm = use_jcr = true;
    src0 = "JCR" + std::to_string(sentient::getValueRegIndex(if_op.getLhs()));
    imm_value = get_imm_value_or_symbol_id(if_op.getRhs());
  } else if (lhs_locale == "imm" && rhs_locale == "lccr") {
    is_imm = true;
    use_jcr = false;
    src0 = "LCCR" + std::to_string(sentient::getValueRegIndex(if_op.getRhs()));
    imm_value = get_imm_value_or_symbol_id(if_op.getLhs());
  } else if (lhs_locale == "lccr" && rhs_locale == "imm") {
    is_imm = true;
    use_jcr = false;
    src0 = "LCCR" + std::to_string(sentient::getValueRegIndex(if_op.getLhs()));
    imm_value = get_imm_value_or_symbol_id(if_op.getRhs());
  } else if (lhs_locale == "lccr" && rhs_locale == "jcr") {
    is_imm = use_jcr = false;
    src0 = "LCCR" + std::to_string(sentient::getValueRegIndex(if_op.getLhs()));
    src1 = "JCR" + std::to_string(sentient::getValueRegIndex(if_op.getRhs()));
  } else if (lhs_locale == "jcr" && rhs_locale == "lccr") {
    is_imm = use_jcr = false;
    src0 = "LCCR" + std::to_string(sentient::getValueRegIndex(if_op.getRhs()));
    src1 = "JCR" + std::to_string(sentient::getValueRegIndex(if_op.getLhs()));
  } else if (lhs_locale == "jcr" && rhs_locale == "jcr") {
    is_imm = false;
    use_jcr = true;
    src0 = "JCR" + std::to_string(sentient::getValueRegIndex(if_op.getRhs()));
    src1 = "JCR" + std::to_string(sentient::getValueRegIndex(if_op.getLhs()));
  } else {
    if_op.emitError("Unknown Sentient IF operand for ProgIR Lowering");
    signalPassFailure();
    return super_instr;
  }

  super_instr.setCommonField("usejcr", OperandAttr(use_jcr));
  super_instr.setCommonField("src0",
                             OperandAttr(src0, OperandAttr::Type::DESCRIPTIVE));
  if (is_imm) {
    super_instr.setCommonField("isimm", OperandAttr(true));
    //    super_instr.setCommonField("cmp_imm", imm_value);
    if (!is_imm_symbol) {
      super_instr.setCommonField("src1", imm_value);
    } else {
      super_instr.setCommonField(
          "src1", OperandAttr(imm_value, OperandAttr::Type::VARIABLE_SYMBOL));
    }
  } else {
    super_instr.setCommonField("isimm", OperandAttr(false));
    super_instr.setCommonField(
        "src1", OperandAttr(src1, OperandAttr::Type::DESCRIPTIVE));
  }

  super_instr.setCommonComment(label + "-begin");
  return super_instr;
}

// ---- 48/130  ConstructJADDInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:687  (33L)
UniformInstrInfo e048_ConstructJADDInstr(
    const SenComponents& comp, sentient::AddOp add_op) {
  int src0, imm;
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::JADD);
  super_instr.setCommonComment("jcr add");
  auto lhs_locale = sentient::getValueRegLocaleAsString(add_op.getInp1());
  auto rhs_locale = sentient::getValueRegLocaleAsString(add_op.getInp2());
  if (lhs_locale == "lccr" && rhs_locale == "imm") {
    src0 = getValueRegIndex(add_op.getInp1());
    imm = add_op.getInp2().getDefiningOp<sentient::ConstantOp>().getValue();
  } else if (lhs_locale == "imm" && rhs_locale == "lccr") {
    src0 = getValueRegIndex(add_op.getInp2());
    imm = add_op.getInp1().getDefiningOp<sentient::ConstantOp>().getValue();
  } else if (lhs_locale == "jcr" && rhs_locale == "imm") {
    src0 = getValueRegIndex(add_op.getInp1());
    imm = add_op.getInp2().getDefiningOp<sentient::ConstantOp>().getValue();
    super_instr.setCommonField("jcr_select", 1);
  } else if (lhs_locale == "imm" && rhs_locale == "jcr") {
    src0 = getValueRegIndex(add_op.getInp2());
    imm = add_op.getInp1().getDefiningOp<sentient::ConstantOp>().getValue();
    super_instr.setCommonField("jcr_select", 1);
  } else {
    add_op->emitError("Unable to find Prog.IR for JADD\n");
    signalPassFailure();
  }

  int reg_index = add_op.getRegIndex();
  super_instr.setCommonField("jcr_target", reg_index);
  super_instr.setCommonField("src0", src0);
  super_instr.setCommonField("imm", imm);
  return super_instr;
}

// ---- 49/130  ConstructXRFADDInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:892  (47L)
UniformInstrInfo e049_ConstructXRFADDInstr(
    const SenComponents& comp, sentient::AddOp add_op) {
  UniformInstrInfo super_instr;
  super_instr.setCommonComment("xrf add");
  super_instr.setInstn(OpCodeT::XRFACCESS);
  auto lhs_locale = sentient::getValueRegLocaleAsString(add_op.getInp1());
  auto rhs_locale = sentient::getValueRegLocaleAsString(add_op.getInp2());

  if (lhs_locale == "xrfrdptr" && rhs_locale == "imm") {
    super_instr.setCommonField(
        "rdptr_upd", OperandAttr("incr", OperandAttr::Type::DESCRIPTIVE));
    super_instr.setCommonField(
        "wrptr_upd", OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
    int imm = add_op.getInp2().getDefiningOp<sentient::ConstantOp>().getValue();
    super_instr.setCommonField(
        "rdptr_imm", getAddrWraparounded(imm, PTXRF, dccExtContext()));
  } else if (lhs_locale == "imm" && rhs_locale == "xrfrdptr") {
    super_instr.setCommonField(
        "rdptr_upd", OperandAttr("incr", OperandAttr::Type::DESCRIPTIVE));
    super_instr.setCommonField(
        "wrptr_upd", OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
    int imm = add_op.getInp1().getDefiningOp<sentient::ConstantOp>().getValue();
    super_instr.setCommonField(
        "rdptr_imm", getAddrWraparounded(imm, PTXRF, dccExtContext()));
  } else if (lhs_locale == "xrfwrptr" && rhs_locale == "imm") {
    super_instr.setCommonField(
        "wrptr_upd", OperandAttr("incr", OperandAttr::Type::DESCRIPTIVE));
    super_instr.setCommonField(
        "rdptr_upd", OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
    int imm = add_op.getInp2().getDefiningOp<sentient::ConstantOp>().getValue();
    super_instr.setCommonField(
        "wrptr_imm", getAddrWraparounded(imm, PTXRF, dccExtContext()));
  } else if (lhs_locale == "imm" && rhs_locale == "xrfwrptr") {
    super_instr.setCommonField(
        "wrptr_upd", OperandAttr("incr", OperandAttr::Type::DESCRIPTIVE));
    super_instr.setCommonField(
        "rdptr_upd", OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
    int imm = add_op.getInp1().getDefiningOp<sentient::ConstantOp>().getValue();
    super_instr.setCommonField(
        "wrptr_imm", getAddrWraparounded(imm, PTXRF, dccExtContext()));
  } else {
    add_op->emitError("Unable to find Prog.IR for XRF ADD\n");
    signalPassFailure();
  }

  return super_instr;
}

// ---- 50/130  ConstructXRFADDInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:940  (22L)
UniformInstrInfo e050_ConstructXRFADDInstr(
    const SenComponents& comp, bool is_read, int val) {
  UniformInstrInfo super_instr;
  super_instr.setCommonComment("xrf add");
  super_instr.setInstn(OpCodeT::XRFACCESS);
  if (is_read) {
    super_instr.setCommonField(
        "rdptr_upd", OperandAttr("incr", OperandAttr::Type::DESCRIPTIVE));
    super_instr.setCommonField(
        "wrptr_upd", OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
    super_instr.setCommonField(
        "rdptr_imm", getAddrWraparounded(val, PTXRF, dccExtContext()));
  } else {
    super_instr.setCommonField(
        "rdptr_upd", OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
    super_instr.setCommonField(
        "wrptr_upd", OperandAttr("incr", OperandAttr::Type::DESCRIPTIVE));
    super_instr.setCommonField(
        "wrptr_imm", getAddrWraparounded(val, PTXRF, dccExtContext()));
  }
  return super_instr;
}

// ---- 51/130  ConstructJSUBInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:964  (34L)
UniformInstrInfo e051_ConstructJSUBInstr(
    const SenComponents& comp, sentient::SubOp sub_op) {
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::JSUB);
  super_instr.setCommonComment("jcr sub");
  auto lhs_locale = sentient::getValueRegLocaleAsString(sub_op.getInp1());
  auto rhs_locale = sentient::getValueRegLocaleAsString(sub_op.getInp2());

  int src0, imm;
  if (lhs_locale == "lccr" && rhs_locale == "imm") {
    src0 = getValueRegIndex(sub_op.getInp1());
    imm = sub_op.getInp2().getDefiningOp<sentient::ConstantOp>().getValue();
  } else if (lhs_locale == "imm" && rhs_locale == "lccr") {
    src0 = getValueRegIndex(sub_op.getInp2());
    imm = sub_op.getInp1().getDefiningOp<sentient::ConstantOp>().getValue();
  } else if (lhs_locale == "jcr" && rhs_locale == "imm") {
    src0 = getValueRegIndex(sub_op.getInp1());
    imm = sub_op.getInp2().getDefiningOp<sentient::ConstantOp>().getValue();
    super_instr.setCommonField("jcr_select", 1);
  } else if (lhs_locale == "imm" && rhs_locale == "jcr") {
    src0 = getValueRegIndex(sub_op.getInp2());
    imm = sub_op.getInp1().getDefiningOp<sentient::ConstantOp>().getValue();
    super_instr.setCommonField("jcr_select", 1);
  } else {
    sub_op->emitError("Unable to find Prog.IR for JCR SUB\n");
    signalPassFailure();
  }

  int reg_index = sub_op.getRegIndex();
  super_instr.setCommonField("jcr_target", reg_index);
  super_instr.setCommonField("src0", src0);
  super_instr.setCommonField("imm", imm);
  return super_instr;
}

// ---- 52/130  setSentientComputeInputProgIROperands  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1184  (125L)
void e052_setSentientComputeInputProgIROperands(
    Operation* op, const SenComponents& comp, UniformInstrInfo& super_instr,
    const StringRef& operand_value, std::string& port_name,
    std::string op_precision, std::string compute_precision,
    std::string result_precision, bool is_operand_forwarded,
    bool is_gcvt_fcvt) {
  std::string prog_ir_operand_value;
  // common for PT/PE/SFP
  if (operand_value.contains("lrf")) {
    if (comp == PT &&
        (compute_precision == "int4" || compute_precision == "int8" ||
         compute_precision == "int2")) {
      int reg_val = std::stoi(operand_value.substr(3).str());
      if (reg_val < 4) {
        prog_ir_operand_value = "R" + std::to_string(reg_val / 2);
      } else {
        prog_ir_operand_value = "R" + operand_value.substr(3).str();
      }
    } else {
      prog_ir_operand_value = "R" + operand_value.substr(3).str();
    }
  } else if (operand_value.contains("irf")) {
    prog_ir_operand_value = "irf" + operand_value.substr(3).str();
  } else if (operand_value.contains("latch")) {
    prog_ir_operand_value = "reuse";
  } else if (operand_value.contains("zero")) {
    prog_ir_operand_value = "0.0";
  } else if (operand_value.contains("one")) {
    prog_ir_operand_value = "1.0";
  } else if (comp == PT) {
    // PT only options
    if (operand_value == "xrf") {
      prog_ir_operand_value = operand_value.str();
    } else if (operand_value == "north" || operand_value == "west") {
      prog_ir_operand_value = operand_value.substr(0, 1).str() + "-link";
    } else if (operand_value == "crossptnlink") {
      prog_ir_operand_value = "n-link";
    } else {
      op->emitError("Unsupported operand for PT FMA");
      signalPassFailure();
    }
    // PE/SFP only options
  } else if (operand_value == "lx") {
    prog_ir_operand_value = "lxlu";
  } else if (operand_value == "nbrslice" || operand_value == "nfwd") {
    prog_ir_operand_value = operand_value.str();
  } else if (operand_value == "two") {
    prog_ir_operand_value = "2.0";
  } else if (operand_value == "three") {
    prog_ir_operand_value = "3.0";
  } else if (comp == SFP) {
    // SFP only options
    if (operand_value == "pe") {
      prog_ir_operand_value = operand_value.str();
    } else if (operand_value == "sfpring") {
      prog_ir_operand_value = "datafifo";
    } else if (operand_value == "nfwd0") {
      prog_ir_operand_value = "nfwd";
    } else if (operand_value == "nfwd2") {
      prog_ir_operand_value = "nfwd";
    } else {
      op->emitError("Unsupported operand for SFP FMA");
      signalPassFailure();
    }
  } else if (comp == PE) {
    // PE only options
    if (operand_value == "sfp") {
      prog_ir_operand_value = operand_value.str();
    } else if (operand_value == "pt") {
      // pt with op_precision int16 corresponds to "ptint16" in DD2 ISA
      if (dcc_ext_ctx_.getArch() <= IsaCoreGen::RCUDD1A_ISA &&
          op_precision == "int16")
        prog_ir_operand_value = "ptint16";
      else
        prog_ir_operand_value = operand_value.str();
    } else if (operand_value == "nfwd0") {
      prog_ir_operand_value = "nfwd";
    } else if (operand_value == "nfwd2") {
      prog_ir_operand_value = "nfwd";
    } else {
      op->emitError("Unsupported operand for PE FMA");
      signalPassFailure();
    }
  } else {
    op->emitError("Unrecognized operand for FMA");
    signalPassFailure();
  }

  // On the fly input conversions. Skip for dangling MAC op (i.e. when
  // result_precision is "none"), LOGICAL/MERGE/PACK (i.e. when
  // TODO : revisit whether we should set on the fly conversion for dangling
  // MacOp.
  if (dcc_ext_ctx_.getArch() >= IsaCoreGen::SEN1P5_ISA &&
      op_precision != compute_precision && compute_precision != "none" &&
      op_precision != "int1" && result_precision != "none" &&
      is_any_of(comp, PE, SFP)) {
    if (is_any_of(operand_value, "sfp", "pe", "lx", "l0", "pt")) {
      if (is_any_of(op_precision, "fp24", "int24")) {
        DT_CHECK_MSG(comp == PE,
                     "Only expect data received from PT in PE units");
        // TODO: change field name ptfp24tofp32 to ptfp24
        compute_precision = "fp32";
      } else if (!is_gcvt_fcvt) {
        DT_CHECK_MSG(op_precision == "fp16" && compute_precision == "fp32",
                     "Unsupported input on the fly conversion");
      }
      // e.g. fp16tofp32
      prog_ir_operand_value += op_precision + "to" + compute_precision;
      std::optional<SentientFoldMode> fold_mode =
          dcc::utils::getFoldModeAttributeIfExists(op);
      if (fold_mode.has_value() &&
          fold_mode.value() == SentientFoldMode::fold_AB_Both)
        prog_ir_operand_value += "fold";
      else
        DT_ERROR("l or h selection in on-the-fly conversion not supported");
    }
  }

  // Only set the field if it hasn't been set already
  if (!super_instr.hasCommonField(port_name)) {
    super_instr.setCommonField(
        port_name,
        OperandAttr(prog_ir_operand_value, OperandAttr::Type::DESCRIPTIVE));
  }
}

// ---- 53/130  setSentientComputeOutputProgIROperands  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1310  (112L)
void e053_setSentientComputeOutputProgIROperands(
    Operation* op, const SenComponents& comp, UniformInstrInfo& super_instr,
    const ArrayAttr& output_values, std::string port_name,
    std::string compute_precision, std::string result_precision) {
  for (auto output_val : output_values) {
    std::string field_name;
    auto output_port =
        mlir::cast<SentientComputePortAttr>(output_val).getValue();
    if (output_port == SentientComputePort::none) continue;

    // skip filling tgt-e for 1p5 since its either always
    // west data or stale data.
    if (dccExtContext().getArch() >= SEN1P5_ISA && comp == PT &&
        output_port == SentientComputePort::east) {
      continue;
    }

    StringRef output = stringifySentientComputePort(output_port);
    auto operand_name = port_name;
    if (comp == PT) {
      if (output == "east") {
        field_name = "tgte";
      } else if (output == "south") {
        field_name = "tgts";
      } else if (output.contains("lrf") || output.contains("irf") ||
                 output == "xrf") {
        field_name = "tgtrf";
      } else {
        op->emitError("Unsupported destination for PT FMA");
        signalPassFailure();
      }
    } else {  // PE/SFP
      if (dccExtContext().getArch() <= RCUDD1A_ISA) {
        if (output == "lx") {
          field_name = "tgtlx";
        } else if (output.contains("lrf")) {
          field_name = "tgtrf";
        } else if (comp == PE) {
          if (output == "sfp") {
            field_name = "tgtsfp";
          } else {
            op->emitError("Unsupported destination for PE FMA");
            signalPassFailure();
          }
        } else if (comp == SFP) {
          if (output == "pe") {
            field_name = "tgtpe";
          } else if (output == "pt") {
            field_name = "tgtpt";
          } else if (output == "l0") {
            field_name = "tgtl0";
          } else if (output == "sfpring") {
            if (operand_name != "result") {
              op->emitError("Only FMA result can be forwarded to the SFP ring");
              signalPassFailure();
            }
            field_name = "tgtdatafifo";
          } else {
            op->emitError("Unsupported destination for SFP FMA");
            signalPassFailure();
          }
        }
      } else {
        if (is_any_of(output, "pe", "pt", "l0", "sfp", "lx")) {
          if ((comp == SFP && output == "sfp") &&
              (comp == PE && output == "pe")) {
            op->emitError("tgtencoding can't be itself for SFP/PE.");
            signalPassFailure();
          }
          if (comp == PE && is_any_of(output, "l0", "pt")) {
            op->emitError("unsupported tgtencoding for PE.");
            signalPassFailure();
          }
          field_name = "tgtencoding";
          super_instr.setCommonField(
              field_name,
              OperandAttr(output.str(), OperandAttr::Type::DESCRIPTIVE));
          field_name = "fwdencoding";
        } else if (output.contains("lrf")) {
          field_name = "tgtrf";
        } else if (output.contains("sfpring")) {
          field_name = "tgtdatafifo";
        } else {
          op->emitError("unknown field: " + output);
          signalPassFailure();
        }
      }
    }

    if (super_instr.hasCommonField(field_name)) {
      op->emitError("Multiple operands forwarded in the same direction");
      signalPassFailure();
    }

    if (field_name == "tgtrf") {
      if (operand_name != "result") {
        op->emitError("Only FMA result can be written to register");
        signalPassFailure();
      }
      if (output == "xrf") {
        operand_name = "xrf";
      } else if (output.contains("irf")) {
        operand_name = "irf" + output.substr(3).str();
      } else {  // LRF
        operand_name = "R" + output.substr(3).str();
      }
    }

    super_instr.setCommonField(
        field_name, OperandAttr(operand_name, OperandAttr::Type::DESCRIPTIVE));
  }
}

// ---- 54/130  ConstructL3LoadInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2576  (95L)
UniformInstrInfo e054_ConstructL3LoadInstr(
    const SenComponents& comp, sentient::LoadAndSendOp load_op,
    std::optional<CodeQualityStats>& cq_stats) {
  DT_CHECK_MSG(comp == L3SU, "unexpected unit");
  Operation* consumer = load_op.getConsumer().getDefiningOp();
  DT_CHECK_MSG(consumer, "expected valid consumer");
  DT_CHECK((isa<dataflow::GetUnitOp, sentient::CopyOp, sentient::IfOp,
                uniform::QueryMapOp>(consumer)) &&
           "unexpected consumer");
  auto locale = sentient::getValueRegLocaleAsString(load_op.getConsumer());
  bool is_sent1p5 = (dccExtContext().getArch() >= IsaCoreGen::SEN1P5_ISA);
  bool is_multicast = locale == "gtr";
  bool is_dynamic = locale == "ear";

  int mutable_reg_index = getValueRegIndex(load_op.getMutableAddr());
  int immutable_reg_index = getValueRegIndex(load_op.getImmutableAddr());

  if (dccExtContext().getProgStitch() || forceFullRegInit.getValue() ||
      cq_stats.has_value()) {
    if (is_multicast)
      addToRegsToInit(load_op, {"gtr"},
                      {getValueRegIndex(load_op.getConsumer())});
    if (is_sent1p5)
      addToRegsToInit(load_op,
                      {getValueRegLocaleAsString(load_op.getMutableAddr())},
                      {getValueRegIndex(load_op.getMutableAddr())});
    else
      addToRegsToInit(load_op,
                      {getValueRegLocaleAsString(load_op.getMutableAddr()),
                       getValueRegLocaleAsString(load_op.getImmutableAddr()),
                       getValueRegLocaleAsString(load_op.getResult())},
                      {mutable_reg_index, immutable_reg_index,
                       getValueRegIndex(load_op.getResult())});
  }

  std::string instr_name = "ST";  // horrible instruction name
  if (is_multicast) instr_name += "G";
  if (dcc::utils::isUpdateMode(load_op)) instr_name += "U";

  UniformInstrInfo super_instr;
  if (load_op.getDbgName().has_value())
    super_instr.setCommonComment(load_op.getDbgName().value().str());
  super_instr.setInstn(Isa::to_instopcode(instr_name));
  super_instr.setCommonField("src0", mutable_reg_index);  // LAR
  if (is_sent1p5) {
    // lbr is removed in sent1.5
    auto const_op = llvm::dyn_cast_or_null<sentient::ConstantOp>(
        load_op.getImmutableAddr().getDefiningOp());
    DT_CHECK_MSG(const_op && const_op.getValue() == 0,
                 "Expect immutable address 0 (no LBRs).");
  } else {
    super_instr.setCommonField("src1", immutable_reg_index);  // LBR
  }
  if (is_sent1p5) {
    // for sent1.5, routing direction is specified by DRM
    if (load_op.getDir().has_value())
      super_instr.setCommonField(
          "drm", static_cast<int>(load_op.getDirAttr().getValue()));
  }
  super_instr.setCommonField("burst",
                             normalizeBurstSize(load_op.getBurstSize(), comp));
  super_instr.setCommonField(
      "group", is_multicast
                   ? (int)sentient::getValueRegIndex(load_op.getConsumer())
                   : 0);

  if (is_multicast && !is_dynamic) {
    DT_CHECK(load_op.getDir() && "expected routing direction");
    int node = static_cast<int>(load_op.getDirAttr().getValue());
    super_instr.setCommonField("node", node);
  } else if (is_dynamic) {
    // This path should only be taken when consumer/producer dynamically
    // changes.
    // Set the lower bits to the register index and the 8th bit to true.
    const int DYNAMIC_NODE_FLAG = 1 << 7;
    int node = mlir::sentient::getValueRegIndex(load_op.getConsumer());
    node |= DYNAMIC_NODE_FLAG;
    super_instr.setCommonField("node", node);
  } else if (!is_multicast && !is_dynamic) {
    auto consumer_def = load_op.getConsumer().getDefiningOp();
    if (auto get_unit_op = dyn_cast<dataflow::GetUnitOp>(consumer_def)) {
      super_instr.setCommonField("node", (int)(dcc::getCoreId(get_unit_op)));
    } else if (auto query_op = dyn_cast<uniform::QueryMapOp>(consumer_def)) {
      super_instr.setOperandMap(OperandMap(unit_foldid_map_, dccExtContext(),
                                           "node", query_op,
                                           OperandMap::Mode::unit_id));
    }
  } else {
    llvm_unreachable(
        "unsupported operation indicating the consumer/producer of data "
        "transfer");
  }

  return super_instr;
}

// ---- 55/130  ConstructL3StoreInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2672  (150L)
UniformInstrInfo e055_ConstructL3StoreInstr(
    const SenComponents& comp, sentient::ReceiveAndStoreOp store_op,
    std::optional<CodeQualityStats>& cq_stats) {
  DT_CHECK_MSG(comp == L3LU, "unexpected unit");
  Operation* producer = store_op.getProducer().getDefiningOp();
  DT_CHECK_MSG(producer, "expected valid producer");
  DT_CHECK((isa<dataflow::GetUnitOp, sentient::CopyOp, sentient::ConstantOp,
                sentient::IfOp, uniform::QueryMapOp>(producer)) &&
           "unexpected producer");
  bool is_sent1p5 = (dccExtContext().getArch() >= IsaCoreGen::SEN1P5_ISA);
  Value multicast = store_op.getMulticastInfo();
  DT_CHECK(
      (!multicast || sentient::getValueRegLocaleAsString(multicast) == "gtr") &&
      "expected GTR locale for multicast_info");
  DT_CHECK_MSG(!store_op.getDst(),
               "L0Scale destination not expected in L3 units");
  auto ldz_producer = dyn_cast<sentient::ConstantOp>(producer);
  DT_CHECK_MSG(!(ldz_producer && multicast),
               "multicast not supported for constant producers");

  int mutable_reg_index = getValueRegIndex(store_op.getMutableAddr());
  int immutable_reg_index = getValueRegIndex(store_op.getImmutableAddr());

  if (dccExtContext().getProgStitch() || forceFullRegInit.getValue() ||
      cq_stats.has_value()) {
    if (multicast)
      addToRegsToInit(store_op, {"gtr"}, {getValueRegIndex(multicast)});
    if (is_sent1p5)
      // lbr is removed in sent1.5
      addToRegsToInit(store_op,
                      {getValueRegLocaleAsString(store_op.getMutableAddr())},
                      {getValueRegIndex(store_op.getMutableAddr())});
    else
      addToRegsToInit(store_op,
                      {getValueRegLocaleAsString(store_op.getMutableAddr()),
                       getValueRegLocaleAsString(store_op.getImmutableAddr()),
                       getValueRegLocaleAsString(store_op.getResult())},
                      {mutable_reg_index, immutable_reg_index,
                       getValueRegIndex(store_op.getResult())});
  }

  bool is_dynamic =
      sentient::getValueRegLocaleAsString(store_op.getProducer()) == "ear";

  std::string instr_name = "LD";  // horrible instruction name
  if (multicast) instr_name += "G";
  if (ldz_producer) instr_name += "Z";
  if (dcc::utils::isUpdateMode(store_op)) instr_name += "U";

  UniformInstrInfo super_instr;
  if (store_op.getDbgName().has_value())
    super_instr.setCommonComment(store_op.getDbgName().value().str());
  super_instr.setInstn(Isa::to_instopcode(instr_name));
  super_instr.setCommonField("src0", mutable_reg_index);  // LAR
  if (is_sent1p5) {
    auto const_op = llvm::dyn_cast_or_null<sentient::ConstantOp>(
        store_op.getImmutableAddr().getDefiningOp());
    DT_CHECK_MSG(const_op && const_op.getValue() == 0,
                 "Expect immutable address 0 (no LBRs).");
  } else {
    super_instr.setCommonField("src1", immutable_reg_index);  // LBR
  }
  super_instr.setCommonField("burst",
                             normalizeBurstSize(store_op.getBurstSize(), comp));

  if (ldz_producer) {
    auto ldz_producer_result_type = ldz_producer->getResult(0).getType();
    DT_CHECK_MSG(ldz_producer_result_type.isIntOrFloat(),
                 "expecting int or float type for LDZ producer");
    unsigned bitwidth = ldz_producer_result_type.getIntOrFloatBitWidth();
    DT_CHECK(bitwidth == 4 || bitwidth == 8 || bitwidth == 16);
    int interpreted_imm;
    if (bitwidth == 4) {
      interpreted_imm = deeptools::Int64ToInt4BinToInt(ldz_producer.getValue());
    } else if (bitwidth == 8) {
      interpreted_imm = deeptools::Int64ToInt8BinToInt(ldz_producer.getValue());
    } else {  // bitwidth == 16
      interpreted_imm =
          deeptools::Int64ToInt16BinToInt(ldz_producer.getValue());
    }

    if (interpreted_imm == 0) {
      super_instr.setCommonField(
          "mode", OperandAttr("zero2lx", OperandAttr::Type::DESCRIPTIVE));
      super_instr.setCommonField("imm", interpreted_imm);
    } else if (bitwidth == 8) {
      super_instr.setCommonField(
          "mode", OperandAttr("imm2lx", OperandAttr::Type::DESCRIPTIVE));
      super_instr.setCommonField("imm", interpreted_imm);
    } else if (bitwidth == 16) {
      // non-zero imm16 LDZ do not need to set imm field because the value
      // is coming from the ZR, not the imm field.
      super_instr.setCommonField(
          "mode", OperandAttr("zr2lx", OperandAttr::Type::DESCRIPTIVE));
    } else if (min_ldz_imm_field_bits % bitwidth == 0) {
      // Bitwidth of producer is less than 8 bits. LDZ only supports 8 or 16 bit
      // imm fields. Splat the relevant bits of the interpreted value to form an
      // 8 bit entity for an imm2lx mode LDZ.
      DT_CHECK_MSG(ldz_producer.getValue() > 0, "LDZ imms are unsigned");
      int splat = 0, mask = interpreted_imm;
      for (int i = 0, e = min_ldz_imm_field_bits / bitwidth; i < e; ++i) {
        splat = splat | mask;
        mask = mask << bitwidth;
      }
      super_instr.setCommonField(
          "mode", OperandAttr("imm2lx", OperandAttr::Type::DESCRIPTIVE));
      super_instr.setCommonField("imm", splat);
    } else {
      llvm_unreachable("Unexpected LDZ mode!");
    }

    // LDZ must also include the datatype virtual field to tell senulator how to
    // interpret the data during simulation.
    std::string precision =
        ldz_producer_result_type.isInteger(bitwidth) ? "int" : "fp";
    precision += std::to_string(bitwidth);
    // Currently, only one datatype is supported but in the future, multiple
    // data types may be supported (for reinterpretation).
    int64_t data_type_one_hot = 1;
    super_instr.setCommonField(
        "datatype_virtual",
        data_type_one_hot << int(
            dcc::sentient::utils::getDataFormatFromSentientPrecisionAttr(
                precision)));
  } else {
    if (is_dynamic) {
      // This path should only be taken when consumer/producer dynamically
      // changes.
      // Set the lower bits to the register index and the 8th bit to true.
      const int DYNAMIC_NODE_FLAG = 1 << 7;
      int node = mlir::sentient::getValueRegIndex(store_op.getProducer());
      node |= DYNAMIC_NODE_FLAG;
      super_instr.setCommonField("node", node);
    } else {
      auto producer_def = store_op.getProducer().getDefiningOp();
      if (auto get_unit_op = dyn_cast<dataflow::GetUnitOp>(producer_def)) {
        super_instr.setCommonField("node", (int)(dcc::getCoreId(get_unit_op)));
      } else if (auto query_op = dyn_cast<uniform::QueryMapOp>(producer_def)) {
        super_instr.setOperandMap(OperandMap(unit_foldid_map_, dccExtContext(),
                                             "node", query_op,
                                             OperandMap::Mode::unit_id));
      }
    }

    super_instr.setCommonField(
        "group", multicast ? (int)sentient::getValueRegIndex(multicast) : 0);
  }

  return super_instr;
}

// ---- 56/130  ConstructZRAssignInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2828  (14L)
UniformInstrInfo e056_ConstructZRAssignInstr(
    const SenComponents& comp, int64_t imm) {
  DT_CHECK_MSG(comp == L3LU, "unexpected unit");

  UniformInstrInfo super_instr;

  super_instr.setInstn(OpCodeT::LDZimm16);
  int interpreted_imm = deeptools::Int64ToInt16BinToInt(imm);
  super_instr.setCommonField("imm", interpreted_imm);
  super_instr.setCommonField(
      "mode", OperandAttr("imm2zr", OperandAttr::Type::DESCRIPTIVE));

  return super_instr;
}

// ---- 57/130  ConstructL3LoadAndStoreInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2843  (179L)
UniformInstrInfo e057_ConstructL3LoadAndStoreInstr(
    const SenComponents& comp, sentient::LoadAndStoreOp ls_op,
    std::optional<CodeQualityStats>& cq_stats) {
  DT_CHECK_MSG(is_any_of(comp, L3SU, L3LU), "unexpected unit");
  auto multicast_copy_op = dyn_cast_or_null<sentient::CopyOp>(
      ls_op.getMulticastInfo() ? ls_op.getMulticastInfo().getDefiningOp()
                               : nullptr);
  DT_CHECK(!ls_op.getMulticastInfo() ||
           (multicast_copy_op &&
            multicast_copy_op.getRegLocale() == SentientRegType::gtr));
  dcc::utils::L3GatherScatterChecker gather_scatter(ls_op);

  StringRef src_mutable_locale =
      sentient::getValueRegLocaleAsString(ls_op.getSrcMutableAddr());
  StringRef src_immutable_locale =
      sentient::getValueRegLocaleAsString(ls_op.getSrcImmutableAddr());
  StringRef dst_mutable_locale =
      sentient::getValueRegLocaleAsString(ls_op.getDstMutableAddr());
  StringRef dst_immutable_locale =
      sentient::getValueRegLocaleAsString(ls_op.getDstImmutableAddr());
  int src_mutable_reg_index = getValueRegIndex(ls_op.getSrcMutableAddr());
  int src_immutable_reg_index = getValueRegIndex(ls_op.getSrcImmutableAddr());
  int dst_mutable_reg_index = getValueRegIndex(ls_op.getDstMutableAddr());
  int dst_immutable_reg_index = getValueRegIndex(ls_op.getDstImmutableAddr());

  bool is_sen1p5 = dccExtContext().getArch() >= IsaCoreGen::SEN1P5_ISA;

  bool skip_src_lbr = is_sen1p5 && dcc::utils::isLXUnit(ls_op.getSrc());
  bool skip_dst_lbr = is_sen1p5 && dcc::utils::isLXUnit(ls_op.getDst());
  if (skip_src_lbr) {
    auto const_op = llvm::dyn_cast_or_null<sentient::ConstantOp>(
        ls_op.getSrcImmutableAddr().getDefiningOp());
    DT_CHECK_MSG(const_op && const_op.getValue() == 0,
                 "Expect src immutable address 0 (no LBRs).");
  }
  if (skip_dst_lbr) {
    auto const_op = llvm::dyn_cast_or_null<sentient::ConstantOp>(
        ls_op.getDstImmutableAddr().getDefiningOp());
    DT_CHECK_MSG(const_op && const_op.getValue() == 0,
                 "Expect dst immutable address 0 (no LBRs).");
  }
  if (dccExtContext().getProgStitch() || forceFullRegInit.getValue() ||
      cq_stats.has_value()) {
    if (multicast_copy_op)
      addToRegsToInit(ls_op, {"gtr"}, {getValueRegIndex(multicast_copy_op)});
    if (skip_src_lbr)
      addToRegsToInit(ls_op,
                      {getValueRegLocaleAsString(ls_op.getSrcMutableAddr()),
                       getValueRegLocaleAsString(ls_op.getDstMutableAddr()),
                       getValueRegLocaleAsString(ls_op.getDstImmutableAddr())},
                      {getValueRegIndex(ls_op.getSrcMutableAddr()),
                       getValueRegIndex(ls_op.getDstMutableAddr()),
                       getValueRegIndex(ls_op.getDstImmutableAddr())});
    else if (skip_dst_lbr)
      addToRegsToInit(ls_op,
                      {getValueRegLocaleAsString(ls_op.getSrcMutableAddr()),
                       getValueRegLocaleAsString(ls_op.getSrcImmutableAddr()),
                       getValueRegLocaleAsString(ls_op.getDstMutableAddr())},
                      {getValueRegIndex(ls_op.getSrcMutableAddr()),
                       getValueRegIndex(ls_op.getSrcImmutableAddr()),
                       getValueRegIndex(ls_op.getDstMutableAddr())});
    else
      addToRegsToInit(
          ls_op,
          {src_mutable_locale, src_immutable_locale, dst_mutable_locale,
           dst_immutable_locale, getValueRegLocaleAsString(ls_op.getResult(0)),
           getValueRegLocaleAsString(ls_op.getResult(1))},
          {src_mutable_reg_index, src_immutable_reg_index,
           dst_mutable_reg_index, dst_immutable_reg_index,
           getValueRegIndex(ls_op.getResult(0)),
           getValueRegIndex(ls_op.getResult(1))});
  }

  // STZ instruction is used for an IBR write in L3SU (ibr=1), or for an
  // LX->QGI transfer in L3SU (ibr=0: data pushed to QGI).
  SenComponents dst_unit_type =
      dcc::getUnitType(ls_op.getDst().getDefiningOp());
  bool is_stz =
      (gather_scatter.isIBRWrite() || dst_unit_type == QGI) && comp == L3SU;
  std::string instr_name = comp == L3LU ? "LD" : "ST";
  if (is_stz)
    instr_name += "Z";
  else {
    if (gather_scatter.isIBRRead() || gather_scatter.isIBRWrite())
      instr_name += "I";
    if (multicast_copy_op) instr_name += "G";
    instr_name += "M";
    DT_CHECK(isa<sentient::ConstantOp>(ls_op.getSrcInc().getDefiningOp()) &&
             isa<sentient::ConstantOp>(ls_op.getDstInc().getDefiningOp()) &&
             "expected inc operands to be constant");
    auto src_inc =
        cast<sentient::ConstantOp>(ls_op.getSrcInc().getDefiningOp());
    auto dst_inc =
        cast<sentient::ConstantOp>(ls_op.getDstInc().getDefiningOp());
    // ISA only allows incrementing both lar and ear by the same amount (or
    // zero).
    DT_CHECK_MSG(src_inc.getValue() == dst_inc.getValue(),
                 "both src and dst increment should be equal");
    if (src_inc.getValue() != 0) instr_name += "U";
  }

  UniformInstrInfo super_instr;
  if (ls_op.getDbgName().has_value())
    super_instr.setCommonComment(ls_op.getDbgName().value().str());
  super_instr.setInstn(Isa::to_instopcode(instr_name));

  if (gather_scatter.isIBRWrite() && comp == L3LU) {
    super_instr.setCommonField("readibr", 0);
    // lar and lbr are unused, but we still map them, because senulator treats
    // lar as updated in update-mode.
    super_instr.setCommonField("src0", dst_mutable_reg_index);
    super_instr.setCommonField(
        "src2", getValueRegIndex(ls_op.getSrcMutableAddr()));  // ear
    DT_CHECK_MSG(!skip_src_lbr, "Do not expect src to be LX address");
    super_instr.setCommonField("src3", src_immutable_reg_index);  // ebr
  } else if (is_stz) {
    // ibr=1 for IBR-write (scatter); ibr=0 for LX->QGI
    super_instr.setCommonField("ibr", dst_unit_type == QGI ? 0 : 1);
    super_instr.setCommonField("src0", src_mutable_reg_index);  // lar
    if (!skip_src_lbr)
      super_instr.setCommonField("src1", src_immutable_reg_index);  // lbr
  } else if (src_mutable_locale == "lar") {
    DT_CHECK_MSG(dst_mutable_locale == "ear", "wrong locale for dst operand");
    DT_CHECK((skip_src_lbr || src_immutable_locale == "lbr") &&
             (dst_immutable_locale == "ebr" || dst_immutable_locale == "jcr") &&
             "wrong locale for immutable operands");
    DT_CHECK_MSG(
        (dst_immutable_locale != "jcr" || gather_scatter.isScatter()),
        "only scatter would read from lx and store into ear + ibr[jcr]");
    if (gather_scatter.isIBRRead() && comp == L3LU)
      super_instr.setCommonField("readibr", 1);
    super_instr.setCommonField("src0", src_mutable_reg_index);  // lar
    if (!skip_src_lbr) {
      super_instr.setCommonField("src1", src_immutable_reg_index);  // lbr
    } else {
      // for sent1.5, routing direction is specified by DRM
      if (ls_op.getDir().has_value())
        super_instr.setCommonField(
            "drm", static_cast<int>(ls_op.getDirAttr().getValue()));
    }
    super_instr.setCommonField("src2", dst_mutable_reg_index);  // ear
    DT_CHECK_MSG(!skip_dst_lbr, "Do not expect dst to be LX address");
    super_instr.setCommonField("src3", dst_immutable_reg_index);  // ebr/jcr
  } else {
    DT_CHECK_MSG(src_mutable_locale == "ear",
                 "wrong locale for the src operand");
    DT_CHECK_MSG(dst_mutable_locale == "lar",
                 "wrong locale for the dst operand");
    DT_CHECK((src_immutable_locale == "ebr" || src_immutable_locale == "jcr") &&
             (dst_immutable_locale == "lbr" || skip_dst_lbr) &&
             "wrong locale for immutable operands");
    DT_CHECK_MSG(
        (src_immutable_locale != "jcr" || gather_scatter.isGather()),
        "only gather would read from ear + ibr[jcr] and store into lx");
    if (gather_scatter.isIBRRead()) super_instr.setCommonField("readibr", 1);
    super_instr.setCommonField("src0", dst_mutable_reg_index);  // lar
    if (!skip_dst_lbr) {
      super_instr.setCommonField("src1", dst_immutable_reg_index);  // lbr
    } else {
      // for sent1.5, routing direction is specified by DRM
      if (ls_op.getDir().has_value())
        super_instr.setCommonField(
            "drm", static_cast<int>(ls_op.getDirAttr().getValue()));
    }
    super_instr.setCommonField("src2", src_mutable_reg_index);  // ear
    DT_CHECK_MSG(!skip_src_lbr, "Do not expect src to be LX address");
    super_instr.setCommonField("src3", src_immutable_reg_index);  // ebr/jcr
  }

  if (!is_stz)
    super_instr.setCommonField("burst",
                               normalizeBurstSize(ls_op.getBurstSize(), comp));
  if (!is_stz && (!gather_scatter.isValid() || comp == L3LU))
    super_instr.setCommonField(
        "group", multicast_copy_op
                     ? getValueRegIndex(multicast_copy_op.getResult())
                     : 0);
  return super_instr;
}

// ---- 58/130  ConstructLoadComputeInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3287  (99L)
UniformInstrInfo e058_ConstructLoadComputeInstr(
    const SenComponents& comp, sentient::LoadComputeAndSendOp load_compute_op,
    std::optional<CodeQualityStats>& cq_stats) {
  DT_CHECK(dccExtContext().getArch() >= IsaCoreGen::SEN1P5_ISA && comp == LXLU);

  // Collect reg info.
  int mutable_reg_index = getValueRegIndex(load_compute_op.getMutableAddr());
  int immutable_reg_index =
      getValueRegIndex(load_compute_op.getImmutableAddr());
  int result_reg_index = getValueRegIndex(load_compute_op.getResult());

  DT_CHECK(
      load_compute_op.getRegLocale() == SentientRegType::lrf &&
      getValueRegLocaleAsString(load_compute_op.getMutableAddr()) == "lrf" &&
      getValueRegLocaleAsString(load_compute_op.getImmutableAddr()) == "imm");

  if (dccExtContext().getProgStitch() || forceFullRegInit.getValue() ||
      cq_stats.has_value())
    addToRegsToInit(
        load_compute_op,
        {getValueRegLocaleAsString(load_compute_op.getMutableAddr()),
         getValueRegLocaleAsString(load_compute_op.getImmutableAddr()),
         getValueRegLocaleAsString(load_compute_op.getResult())},
        {mutable_reg_index, immutable_reg_index, result_reg_index});

  // Set instruction and debug names.
  UniformInstrInfo super_instr;
  if (load_compute_op.getDbgName().has_value())
    super_instr.setCommonComment(load_compute_op.getDbgName().value().str());
  std::string instr_name = "LX_LDCVTI";
  if (dcc::utils::isUpdateMode(load_compute_op)) instr_name += "U";
  super_instr.setInstn(Isa::to_instopcode(instr_name));

  // Set the imm field.
  auto immut_addr = dyn_cast_or_null<sentient::ConstantOp>(
      load_compute_op.getImmutableAddr().getDefiningOp());
  DT_CHECK(immut_addr);
  // The address is calculated relative to the dst_element_size.
  int imm_val = immut_addr.getValue() *
                (float)load_compute_op.getSrcElementSize() / 8.0 /
                (float)GetAddressScale(comp, load_compute_op);
  super_instr.setCommonField("imm", imm_val);
  super_instr.setCommonField("src0", result_reg_index);

  auto element_idx_op = dyn_cast_or_null<sentient::ConstantOp>(
      load_compute_op.getElementIndex().getDefiningOp());
  DT_CHECK(element_idx_op);
  int element_idx = element_idx_op.getValue();
  // Current support is only for 4 bit/256 elems to 16 bit/64 elems.
  DT_CHECK(load_compute_op.getSrcElementSize() == 4 &&
           load_compute_op.getDstElementSize() == 16 && element_idx >= 0 &&
           element_idx < 4);
  super_instr.setCommonField("elemidx", element_idx);

  auto scale_idx_op = dyn_cast_or_null<sentient::ConstantOp>(
      load_compute_op.getScaleIndex().getDefiningOp());
  DT_CHECK(scale_idx_op);
  int scale_idx = scale_idx_op.getValue();
  DT_CHECK(scale_idx >= 0 && scale_idx < 2);
  super_instr.setCommonField("scaleidx", scale_idx);

  // Set ldtype.
  std::string ldtype;
  switch (load_compute_op.getShuffleMode()) {
    case SentientShuffleMode::noshuffle:
      ldtype = "128b";
      break;
    case SentientShuffleMode::splat2b:
      ldtype = "2bsplat";
      break;
    case SentientShuffleMode::splat4b:
      ldtype = "4bsplat";
      break;
    case SentientShuffleMode::splat16b:
      ldtype = "16bsplat";
      break;
    default:
      llvm_unreachable("Unexpected shuffle_mode.");
      break;
  }
  super_instr.setCommonField(
      "ldtype", OperandAttr(ldtype, OperandAttr::Type::DESCRIPTIVE));

  if (auto consumer_unit = dyn_cast_or_null<dataflow::GetUnitOp>(
          load_compute_op.getConsumer().getDefiningOp())) {
    auto gen_unit = dcc::getUnitType(consumer_unit);
    auto consumer_name = EnumsConversion::senComponentsToString.at(gen_unit);
    super_instr.setCommonField(
        "consumertag",
        getProperConsumer(dccExtContext(), gen_unit, consumer_name));
  } else if (auto query_op = dyn_cast_or_null<uniform::QueryMapOp>(
                 load_compute_op.getConsumer().getDefiningOp())) {
    OperandMap operand_map(unit_foldid_map_, dccExtContext(), "consumertag",
                           query_op, OperandMap::Mode::unit_name);
    super_instr.setOperandMap(operand_map);
  }

  return super_instr;
}

// ---- 59/130  ConstructLRFCopyInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3509  (49L)
UniformInstrInfo e059_ConstructLRFCopyInstr(
    const SenComponents& comp, ReceiveAndExtractScalarOp receive_and_extract_op,
    std::optional<CodeQualityStats>& cq_stats) {
  DT_CHECK_MSG(comp == LXSU, "LRFCOPY only supported in LXSU");

  if (dccExtContext().getProgStitch() || forceFullRegInit.getValue() ||
      cq_stats.has_value())
    addToRegsToInit(
        receive_and_extract_op,
        {getValueRegLocaleAsString(receive_and_extract_op.getResult())},
        {getValueRegIndex(receive_and_extract_op.getResult())});

  auto unit_op = receive_and_extract_op.getUnit().getDefiningOp();
  DT_CHECK_MSG(unit_op && isa<dataflow::GetUnitOp>(unit_op),
               "expecting unit of ReceiveAndExtractScalarOp to be a GetUnitOp");
  auto get_unit_op = cast<dataflow::GetUnitOp>(unit_op);
  SenComponents unit_comp =
      EnumsConversion::stringToSenComponents.find(get_unit_op.getType().str())
          ->second;
  auto gen_comp = EnumsConversion::senCompToGenericComp.at(unit_comp);
  if (!is_any_of(gen_comp, SenComponents::SFP, SenComponents::PE,
                 SenComponents::LXLU)) {
    receive_and_extract_op->emitOpError(
        "unsupported unit in ReceiveAndExtractScalarOp");
    signalPassFailure();
  }

  auto position_op = receive_and_extract_op.getPosition().getDefiningOp();
  DT_CHECK_MSG(position_op && isa<sentient::ConstantOp>(position_op),
               "expecting position of ReceiveAndExtractScalarOp to be a "
               "sentient::ConstantOp");
  auto position = cast<sentient::ConstantOp>(position_op).getValue();
  if (position != 0) {
    receive_and_extract_op->emitOpError(
        "unsupported position in ReceiveAndExtractScalarOp");
    signalPassFailure();
  }
  UniformInstrInfo super_instr;
  if (receive_and_extract_op.getDbgName().has_value())
    super_instr.setCommonComment(
        receive_and_extract_op.getDbgName().value().str());
  super_instr.setInstn(OpCodeT::LRFCOPY);
  super_instr.setCommonField(
      "src0", getValueRegIndex(receive_and_extract_op.getResult()));
  super_instr.setCommonField(
      "producertag", EnumsConversion::senComponentsToString.at(gen_comp));

  return super_instr;
}

// ---- 60/130  ConstructSetDstMaskInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3559  (41L)
UniformInstrInfo e060_ConstructSetDstMaskInstr(
    const SenComponents& comp, const Operation* op) {
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::SETDSTMASK);
  DT_CHECK_MSG(
      is_any_of(comp, LXLU, LXSU),
      "setdstmask instruction is currently only available in the LX unit");
  std::string from_comp = EnumsConversion::senComponentsToString.at(comp);
  if (auto get_unit = dyn_cast<dataflow::GetUnitOp>(op)) {
    std::string consumer_str = get_unit.getType().lower();
    if (consumer_str.find("pt") != consumer_str.npos) consumer_str = "pt";
    DT_CHECK((consumer_str == "sfp" || consumer_str == "l0su" ||
              consumer_str == "pt") &&
             "unexpected destination unit");
    super_instr.setCommonField("mode", consumer_str);
    super_instr.setCommonComment("set dest for transfer from " + from_comp +
                                 " to " + consumer_str + ", via sfp");
    return super_instr;
  }
  auto query_op = dyn_cast<uniform::QueryMapOp>(op);
  auto key_ops = getListOfKeyOpsFromUniformMapping(query_op);
  auto fold_id = unit_foldid_map_.at(key_ops.front());

  OperandMap operand_map(unit_foldid_map_, dccExtContext(), "mode", query_op);
  for (auto& unit_operand_pair : operand_map["mode"]) {
    auto consumer_str = unit_operand_pair.second.asString(fold_id);
    if (consumer_str.find("pt") != consumer_str.npos)
      unit_operand_pair.second =
          OperandAttr("pt", OperandAttr::Type::DESCRIPTIVE);
    consumer_str = unit_operand_pair.second.asString(fold_id);
    DT_CHECK((consumer_str == "sfp" || consumer_str == "l0su" ||
              consumer_str == "pt") &&
             "unexpected destination unit");
    super_instr.setUniformizedComment("set dest for transfer from " +
                                          from_comp + " to " + consumer_str +
                                          ", via sfp",
                                      unit_operand_pair.first);
  }
  super_instr.setOperandMap(operand_map);
  return super_instr;
}

// ---- 61/130  ConstructSetDestInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3601  (54L)
UniformInstrInfo e061_ConstructSetDestInstr(
    const SenComponents& comp, const sentient::SetSendDestinationOp op) {
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::SETDEST);
  DT_CHECK_MSG(
      comp == SFP,
      "setdest instruction is currently only available in the SFP unit");

  if (auto get_unit = const_cast<sentient::SetSendDestinationOp*>(&op)
                          ->getUnits()
                          .getDefiningOp<dataflow::GetUnitOp>()) {
    DT_CHECK_MSG(get_unit, "expected a get_unit operation");
    std::string consumer_str =
        const_cast<dataflow::GetUnitOp*>(&get_unit)->getType().lower();
    int core_id = dcc::getCoreId(get_unit);
    int corelet_id = dcc::getCoreletId(get_unit);
    DT_CHECK_MSG(core_id >= 0 && corelet_id >= 0 && core_id < 32,
                 "invalid core/corelet id");
    auto prog_unit = op->getParentOfType<dataflow::ProgramUnitOp>();
    DT_CHECK_MSG(prog_unit,
                 "expected set_send_dst operation to be inside a program unit");
    auto curr_get_unit =
        (*prog_unit.getUnits().begin()).getDefiningOp<dataflow::GetUnitOp>();
    DT_CHECK_MSG(curr_get_unit, "unable to get the unit information");
    DT_CHECK(
        curr_get_unit.getType().lower() == consumer_str &&
        consumer_str == "sfp" &&
        "SETDEST can only be used to send data between SFP units of different "
        "cores");
    DT_CHECK(
        corelet_id == dcc::getCoreletId(curr_get_unit) &&
        "Currently SFP can only send data to the same corelet of another core");

    int64_t mask = (int64_t)1 << core_id;
    super_instr.setCommonField("imm", mask);
    super_instr.setCommonComment(
        llvm::Twine("set dest for transfer from SFP of core " +
                    llvm::Twine(dcc::getCoreId(curr_get_unit)) +
                    " to SFP of core " + llvm::Twine(core_id))
            .str());

    return super_instr;
  } else if (auto query_op = const_cast<sentient::SetSendDestinationOp*>(&op)
                                 ->getUnits()
                                 .getDefiningOp<uniform::QueryMapOp>()) {
    OperandMap operand_map(unit_foldid_map_, dccExtContext(), "imm", query_op,
                           OperandMap::Mode::set_dest_tgt, 1);
    super_instr.setCommonComment("set dest for transfer through sfpring");
    super_instr.setOperandMap(operand_map);
    return super_instr;
  }

  llvm_unreachable("unexpected operand to set_send_dst");
}

// ---- 62/130  ConstructImmCopyInstrFromSplatOp  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3657  (60L)
UniformInstrInfo e062_ConstructImmCopyInstrFromSplatOp(
    UniformInstrInfo super_instr, mlir::Operation* input,
    const SenComponents& comp, DataFormats op_precision, int reg_val,
    sentient::SplatOp op, std::optional<CodeQualityStats>& cq_stats) {
  auto const_input = llvm::dyn_cast<sentient::ConstantOp>(input);
  auto query_op = llvm::dyn_cast<mlir::uniform::QueryMapOp>(input);

  DT_CHECK_MSG(const_input || query_op,
               "Input to construction of Splat operation should be either "
               "constant input or query map with constant inputs");

  if (const_input) {
    OperandAttr val_attr(const_input.getValue());  // store encoded val as int
    val_attr.setSenDataType(op_precision);
    super_instr.setCommonField("imm", val_attr);
  } else {
    auto constant_kv_pair = getConstantTargetKeyValues(query_op);
    auto operand_map = OperandMap(unit_foldid_map_, "imm", query_op,
                                  constant_kv_pair, 1, op_precision);
    super_instr.setOperandMap(operand_map);
  }

  int replica = op.getPad() == SentientSplatPad::none ? 1 : 0;
  DT_CHECK_MSG(is_any_of(op.getPrecision(), SentientPrecision::fp16,
                         SentientPrecision::fp32),
               "expected fp16 or fp32 precision as per the ISA");
  std::string precision = stringifySentientPrecision(op.getPrecision()).str();
  if (dccExtContext().getArch() <= RCUDD1A_ISA && comp == PE) {
    DT_CHECK_MSG(precision != "fp32",
                 "fp32 precision only valid in SFP in DD2");
  } else {
    // Set mode bit
    super_instr.setCommonField(
        "mode", OperandAttr(precision, OperandAttr::Type::DESCRIPTIVE));
  }

  DT_CHECK_MSG(
      !op.getUnrollIncrResult(),
      "should not unroll target for a splat corresponding to an IMMCOPY");
  super_instr.setInstn(OpCodeT::IMMCOPY);
  if (cq_stats.has_value()) {
    ++cq_stats.value().num_copy_ops_;
  }
  super_instr.setCommonField("tgtrf", "R" + std::to_string(reg_val));
  DT_CHECK((op.getPad() == SentientSplatPad::none ^
            op.getPad() == SentientSplatPad::left) &&
           "unexpected pad string");
  auto mask_constant =
      dyn_cast<sentient::ConstantOp>(op.getMask().getDefiningOp());
  DT_CHECK(mask_constant);
  super_instr.setCommonField("mask",
                             OperandAttr(255 - mask_constant.getValue()));
  super_instr.setCommonField("replica", replica);
  auto splat_dbg_name = op.getDbgName();
  std::string desc = "splat/pad to create vector";
  if (splat_dbg_name.has_value())
    desc = splat_dbg_name.value().str() + " " + desc;
  super_instr.setCommonComment(desc);
  return super_instr;
}

// ---- 63/130  ConstructOpaqueInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3866  (167L)
std::vector<UniformInstrInfo>
e063_ConstructOpaqueInstr(const SenComponents& comp,
                                                   sentient::OpaqueOp op,
                                                   ProgIrGraphMap& reg_graph) {
  DT_CHECK_MSG(
      is_any_of(comp, PT, PE, SFP),
      "OPAQUE instruction is currently only available in PT, PE and SFP "
      "units");
  auto& unit_isa = dccExtContext().isa_per_unit_->at(comp);
  auto opName = op.getFuncName();
  mlir::DictionaryAttr param = op.getParameterDictionary();
  mlir::DictionaryAttr rwReg = op.getReadWriteRegisterDictionary();
  mlir::DictionaryAttr rReg = op.getReadOnlyRegisterDictionary();

  //  manually set the instructions
  auto DtDir = dtGetEnv<std::string>("DEEPTOOLS_PATH");
  if (!DtDir.has_value()) {
    DT_ERROR("Please specify DEEPTOOLS_PATH (i.e. ~/WORK/deeptools/)");
  }
  std::string DtDirStr = DtDir.value();
  std::string opaqueFileName;
  bool remove_redundant_setmask = false;
  if (opName.str() == "reciprocal") {
    auto unroll = param.getNamed(llvm::StringRef("unroll"));
    DT_CHECK(unroll.has_value());
    auto unrollAttr = llvm::dyn_cast<StringAttr>(unroll.value().getValue());
    opaqueFileName =
        DtDirStr +
        "/dcc/src/Conversion/SentientToProgIR/opaqueTemplates/reciprocalx" +
        unrollAttr.str() + ".smc";
  } else if (opName.str() == "layernormscale") {
    auto unroll = param.getNamed(llvm::StringRef("unroll"));
    DT_CHECK(unroll.has_value());
    auto unrollAttr = llvm::dyn_cast<StringAttr>(unroll.value().getValue());
    opaqueFileName = DtDirStr +
                     "/dcc/src/Conversion/SentientToProgIR/opaqueTemplates/"
                     "layernormscalex" +
                     unrollAttr.str() + ".smc";
  } else if (opName.str() == "rsqrt") {
    auto unroll = param.getNamed(llvm::StringRef("unroll"));
    DT_CHECK(unroll.has_value());
    auto unrollAttr = llvm::dyn_cast<StringAttr>(unroll.value().getValue());
    opaqueFileName = DtDirStr +
                     "/dcc/src/Conversion/SentientToProgIR/opaqueTemplates/"
                     "rsqrtx" +
                     unrollAttr.str() + ".smc";
  } else if (opName.str() == "sqrt") {
    auto unroll = param.getNamed(llvm::StringRef("unroll"));
    DT_CHECK(unroll.has_value());
    auto unrollAttr = llvm::dyn_cast<StringAttr>(unroll.value().getValue());
    opaqueFileName = DtDirStr +
                     "/dcc/src/Conversion/SentientToProgIR/opaqueTemplates/"
                     "sqrtx" +
                     unrollAttr.str() + ".smc";
  } else if (opName.str() == "realdiv") {
    auto unroll = param.getNamed(llvm::StringRef("unroll"));
    DT_CHECK(unroll.has_value());
    auto unrollAttr = llvm::dyn_cast<StringAttr>(unroll.value().getValue());
    opaqueFileName =
        DtDirStr +
        "/dcc/src/Conversion/SentientToProgIR/opaqueTemplates/realdivx" +
        unrollAttr.str() + ".smc";
  } else if (opName.str() == "softplus_p1") {
    auto unroll = param.getNamed(llvm::StringRef("unroll"));
    DT_CHECK(unroll.has_value());
    auto unrollAttr = llvm::dyn_cast<StringAttr>(unroll.value().getValue());
    opaqueFileName =
        DtDirStr +
        "/dcc/src/Conversion/SentientToProgIR/opaqueTemplates/softplus_p1x" +
        unrollAttr.str() + ".smc";
  } else if (opName.str() == "softplus_p2") {
    auto unroll = param.getNamed(llvm::StringRef("unroll"));
    DT_CHECK(unroll.has_value());
    auto unrollAttr = llvm::dyn_cast<StringAttr>(unroll.value().getValue());
    opaqueFileName =
        DtDirStr +
        "/dcc/src/Conversion/SentientToProgIR/opaqueTemplates/softplus_p2x" +
        unrollAttr.str() + ".smc";
  } else {
    if (opName.str() == "pt_slice_mask_arf_write" ||
        opName.str() == "pt_slice_mask_xrf_write") {
      auto loop_count = param.getNamed(llvm::StringRef("l0"));
      DT_CHECK(loop_count.has_value());
      auto loop_count_attr =
          llvm::dyn_cast<StringAttr>(loop_count.value().getValue());
      // SETMASK wraps around after every increment of 8.
      // So if the loop count is a multiple of 8, we can ignore the
      // state-resetting setmask at the end.
      // TODO: Remove this optimization by avoiding the use of templates.
      if (std::stoi(loop_count_attr.str()) % 8 == 0)
        remove_redundant_setmask = true;
    }
    opaqueFileName = DtDirStr +
                     "/dcc/src/Conversion/SentientToProgIR/opaqueTemplates/" +
                     opName.str() + ".smc";
  }
  std::ifstream senProgInStream(opaqueFileName);
  if (!senProgInStream.is_open()) {
    op.emitError(opaqueFileName + " does not exist");
    signalPassFailure();
  }

#ifdef DT_OBFUSCATE_LIBRARIES
  std::stringstream senProgStringStream;
  Obfuscator obfuscator;
  obfuscator.deobfuscate(senProgInStream, senProgStringStream);
  senProgInStream.close();
#else
  auto& senProgStringStream = senProgInStream;
#endif

  std::string line;
  std::vector<UniformInstrInfo> opaque_insts;
  std::string op_code;
  GetOpCodePrefix(comp, op_code);
  while (getline(senProgStringStream, line)) {
    if (line.length() == 0) continue;         // skip empty lines
    if (line.substr(0, 2) == "//") continue;  // skip comments
    // skip the redundant SETMASK
    if (remove_redundant_setmask && line.substr(0, 7) == "SETMASK" &&
        line.find("reset mask") != std::string::npos)
      continue;
    line = op_code + "_" + line;
    // search for substring
    auto pos = line.find("//");
    line = line.substr(0, pos - 1);
    rtrim(line);
    InstrInfo instr = Dpc::SMCLineToInstrInfo(&unit_isa, line);
    Dpc::convertArchDependentFields(instr, unit_isa, dcc_ext_ctx_.getArch());
    // now replace the src locations
    for (auto& field : instr.instFields_) {
      if (!field.second.isVariable()) continue;
      auto operand = field.second.asString();
      auto subsOp = rReg.getNamed(llvm::StringRef(operand));
      if (!subsOp.has_value()) {
        subsOp = rwReg.getNamed(llvm::StringRef(operand));
      }
      if (!subsOp.has_value()) {
        subsOp = param.getNamed(llvm::StringRef(operand));
      }
      if (!subsOp.has_value()) {
        op.emitError("OPAQUE was not provided with value for variable " +
                     operand);
        signalPassFailure();
      }

      // perform substitution
      auto subsOpAttr = llvm::dyn_cast<StringAttr>(subsOp.value().getValue());
      if (is_any_of(field.first, OperandT::imm)) {
        field.second.setOperand(stoi(subsOpAttr.str()));
      } else if (is_any_of(field.first, OperandT::unroll)) {
        auto unroll_fac = subsOpAttr.str();
        if (unroll_fac.at(0) != 'x') unroll_fac = 'x' + unroll_fac;
        field.second.setOperand(unroll_fac);
      } else {
        field.second.setOperand(subsOpAttr.str());
      }
    }
    UniformInstrInfo super_instr(instr);
    auto opaque_op_dbg_name = op.getDbgName();
    if (opaque_op_dbg_name.has_value())
      super_instr.setCommonComment(opaque_op_dbg_name.value().str());
    opaque_insts.push_back(super_instr);
  }

  return opaque_insts;
}

// ---- 64/130  ConstructSAMVInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4034  (33L)
UniformInstrInfo e064_ConstructSAMVInstr(
    const SenComponents& comp, sentient::SetActiveMaskValueOp op) {
  DT_CHECK_MSG(comp == LXLU,
               "SAMV instruction is currently only available in LXLU units");

  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::SAMV);
  auto samv_dbg_name = op.getDbgName();
  if (samv_dbg_name.has_value())
    super_instr.setCommonComment(samv_dbg_name.value().str());
  super_instr.setCommonField(
      "maskall", op.getMaskall()
                     ? OperandAttr("yes", OperandAttr::Type::DESCRIPTIVE)
                     : OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField("sliceidxsl", (int)op.getSliceidXsl());
  super_instr.setCommonField("numvalidentry", (int)op.getNumvalidentry());
  super_instr.setCommonField(
      "mvridx",
      OperandAttr("MVR" + std::to_string(getValueRegIndex(op.getMaskValue())),
                  OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField(
      "precision", OperandAttr(std::to_string(op.getPrecision()) + "b",
                               OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField(
      "xslinner", op.getXslinner()
                      ? OperandAttr("yes", OperandAttr::Type::DESCRIPTIVE)
                      : OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField("wsllen",
                             OperandAttr(std::to_string(op.getWsllen()) + "b",
                                         OperandAttr::Type::DESCRIPTIVE));

  return super_instr;
}

// ---- 65/130  ConstructSAMVResetInstruction  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4068  (26L)
UniformInstrInfo e065_ConstructSAMVResetInstruction(
    const SenComponents& comp) {
  DT_CHECK_MSG(has_samv_, "No SAMV was detected in the program");
  DT_CHECK_MSG(comp == LXLU,
               "SAMV instruction is currently only available in LXLU units");
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::SAMV);
  super_instr.setCommonField("maskall",
                             OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField("sliceidxsl", 7);
  super_instr.setCommonField("numvalidentry", 0);
  super_instr.setCommonField(
      "mvridx",
      OperandAttr(
          "MVR" + std::to_string(getValueRegIndex(has_samv_.getMaskValue())),
          OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField(
      "precision", OperandAttr(std::to_string(has_samv_.getPrecision()) + "b",
                               OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField("xslinner",
                             OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField("wsllen",
                             OperandAttr("1b", OperandAttr::Type::DESCRIPTIVE));

  return super_instr;
}

// ---- 66/130  ConstructSetMaskInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4095  (22L)
UniformInstrInfo e066_ConstructSetMaskInstr(
    const SenComponents& comp, sentient::SetMaskOp op) {
  UniformInstrInfo super_instr;
  DT_CHECK_MSG(comp == PT,
               "SETMASK instruction is currently only available in PT units");
  if (op.getDbgName().has_value())
    super_instr.setCommonComment(op.getDbgName().value().str());
  super_instr.setInstn(OpCodeT::SETMASK);

  if (auto const_op = dyn_cast_or_null<sentient::ConstantOp>(
          op.getMaskValue().getDefiningOp())) {
    int64_t imm = const_op.getValue();
    super_instr.setCommonField("imm", imm);
  } else
    llvm_unreachable("expecting ConstantOp from mask_value operand");

  // TODO: Once optimizations are put in place to eliminate unnecessary set_mask
  // ops, we may need to insert a set_mask reset operation at the end of the
  // program when progstitch is enabled.

  return super_instr;
}

// ---- 67/130  fillImmField  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:4130  (26L)
void e067_fillImmField(UniformInstrInfo& super_instr,
                                                Operation* op,
                                                const SenComponents& comp,
                                                std::string imm_field_name,
                                                int element_size, int scale) {
  if (auto const_op = dyn_cast<sentient::ConstantOp>(op)) {
    int64_t imm =
        const_op.getValue() * (float)element_size / 8.0 / (float)scale;
    // L0 MODLRF imm has to be within 1024 because imm has only 10bits.
    // and L0 addressing is cyclic. Hence, we can get remainder of the imm
    // with 1024 and use as immediate value.
    imm = getAddrWraparounded(imm, comp, dccExtContext());
    super_instr.setCommonField(imm_field_name, imm);
  } else if (auto query_op = dyn_cast<uniform::QueryMapOp>(op)) {
    bool l0_wrap_around = false;
    if (is_any_of(comp, L0LUROW0, L0LU, L0SU)) {
      l0_wrap_around = true;
    }
    OperandMap operand_map(unit_foldid_map_, dccExtContext(), imm_field_name,
                           query_op,
                           l0_wrap_around ? OperandMap::Mode::l0_wrap_around
                                          : OperandMap::Mode::none,
                           (float)element_size / 8.0 / (float)scale);
    super_instr.setOperandMap(operand_map);
  }
}

// ---- 68/130  getRegImmVals  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1140  (102L)
FoldedImmediateMap e068_getRegImmVals(
    Value val, SmallVector<Value>& units, const SenComponents& comp,
    int element_size, int scale, bool isMVRReg) {
  FoldedImmediateMap imm_vals;
  if (auto const_op = dyn_cast<sentient::ConstantOp>(val.getDefiningOp())) {
    auto imm_val = const_op.getValue();
    if (!isMVRReg) imm_val = imm_val * element_size / 8 / scale;
    // for l0 units, address register is 10bits
    imm_val = getAddrWraparounded(imm_val, comp, dccExtContext());
    for (auto unit : units) {
      SdscFoldIdInput id = unit_foldid_map_.at(unit);
      imm_vals[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())][id] =
          imm_val;
    }
  } else if (auto symbol_op =
                 dyn_cast<symbol::CreateSymbolOp>(val.getDefiningOp())) {
    auto imm_val = symbol_op.getSymbolID();
    for (auto unit : units) {
      SdscFoldIdInput id = unit_foldid_map_.at(unit);
      imm_vals[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())][id] =
          imm_val;
    }
  } else if (auto multicast_op = dyn_cast<dataflow::CreateMulticastGroupOp>(
                 val.getDefiningOp())) {
    auto imm_val = dcc::dataflow::utils::encodeMulticastGroupInfo(multicast_op,
                                                                  dcc_ext_ctx_);
    for (auto unit : units) {
      SdscFoldIdInput id = unit_foldid_map_.at(unit);
      imm_vals[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())][id] =
          imm_val;
    }
  } else if (auto get_unit_op =
                 dyn_cast<dataflow::GetUnitOp>(val.getDefiningOp())) {
    auto imm_val = dcc::getCoreId(get_unit_op);
    for (auto unit : units) {
      SdscFoldIdInput id = unit_foldid_map_.at(unit);
      imm_vals[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())][id] =
          imm_val;
    }
  } else if (auto query_op =
                 dyn_cast<uniform::QueryMapOp>(val.getDefiningOp())) {
    auto is_constant = isConstant<sentient::ConstantOp>(query_op);
    auto is_symbol = isSymbol(query_op);
    auto is_multicast = isMulticast(query_op);
    auto is_get_unit = isGetUnit(query_op);
    DT_CHECK_MSG(is_constant || is_symbol || is_multicast || is_get_unit,
                 "Unsupported operation used as a mapped value.");
    auto immutable_map =
        query_op.getMap().getDefiningOp<uniform::DefImmutableMappingOp>();
    std::vector<Value> empty_units;
    auto ret_values = immutable_map.getValuesFromKeys(units);
    for (int i = 0; i < units.size(); i++) {
      auto& unit = units[i];
      SdscFoldIdInput id = unit_foldid_map_.at(unit);
      auto& ret_val = ret_values[i];
      // scalar_copy may be defined outside uniformizeRegionsOp and only
      // applicable to subset of units.
      if (!ret_val.has_value()) {
        empty_units.push_back(unit);
        continue;
      }

      if (is_constant) {
        auto const_op = ret_val.value().getDefiningOp<sentient::ConstantOp>();
        auto imm_val = const_op.getValue();
        imm_val = imm_val * element_size / 8 / scale;
        // for l0 units, address register is 10bits
        imm_val = getAddrWraparounded(imm_val, comp, dccExtContext());
        imm_vals[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())][id] =
            imm_val;
      } else if (is_symbol) {
        auto symbol_op =
            ret_val.value().getDefiningOp<symbol::CreateSymbolOp>();
        auto imm_val = symbol_op.getSymbolID();
        imm_vals[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())][id] =
            imm_val;
      } else if (is_multicast) {
        auto multicast_op =
            ret_val.value().getDefiningOp<dataflow::CreateMulticastGroupOp>();
        auto imm_val = dcc::dataflow::utils::encodeMulticastGroupInfo(
            multicast_op, dcc_ext_ctx_);
        imm_vals[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())][id] =
            imm_val;
      } else {
        auto get_unit_op = ret_val.value().getDefiningOp<dataflow::GetUnitOp>();
        auto imm_val = dcc::getCoreId(get_unit_op);
        imm_vals[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())][id] =
            imm_val;
      }
    }
    // if immutableMap doesn't contain an unit, use a random value for this
    // unit.
    if (!empty_units.empty()) {
      for (auto unit : empty_units) {
        DT_CHECK(!imm_vals.empty());
        imm_vals[getUnitName(unit.getDefiningOp<dataflow::GetUnitOp>())] =
            imm_vals.begin()->second;
      }
    }
  }
  return imm_vals;
}

// ---- 69/130  recordOpRegDefs  --  dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:22  (30L)
void e069_recordOpRegDefs(Operation *op) {
  DT_CHECK_MSG(enabled(), "Unexpected call, reg def tracking is not enabled");
  DT_CHECK_MSG(reg_def_ctx_, "Expected active RegDefContext");
  // All operations that write to registers are expected to record register
  // information in the regLocale/regLocales and regIndex/regIndices attributes.
  // Operations that have a regLocale but not regIndex can be skipped.
  auto regLocaleAttrName = sentient::getRegLocaleAttrStrName();
  auto regIndexAttrName = sentient::getRegIndexAttrStrName();
  if (op->hasAttr(regIndexAttrName)) {
    DT_CHECK_MSG(op->hasAttr(regLocaleAttrName),
                 "op expected to have regLocale attr");
    setRegDef(mlir::cast<sentient::SentientRegTypeAttr>(
                  op->getAttr(regLocaleAttrName))
                  .getValue(),
              mlir::cast<IntegerAttr>(op->getAttr(regIndexAttrName)).getInt());
    return;
  }
  auto regIndicesAttrName = sentient::getRegIndicesAttrStrName();
  auto regLocalesAttrStrName = sentient::getRegLocalesAttrStrName();
  if (op->hasAttr(regIndicesAttrName)) {
    DT_CHECK_MSG(op->hasAttr(regLocalesAttrStrName),
                 "op expected to have reg_locales attr");
    for (auto [locale, regNum] :
         llvm::zip(mlir::cast<ArrayAttr>(op->getAttr(regLocalesAttrStrName)),
                   mlir::cast<ArrayAttr>(op->getAttr(regIndicesAttrName))))
      setRegDef(mlir::cast<sentient::SentientRegTypeAttr>(locale).getValue(),
                mlir::cast<IntegerAttr>(regNum).getInt());
    return;
  }
}

// ---- 70/130  ~UniformRegionContext  --  dcc/src/Conversion/SentientToProgIR/RegDefTracker.cpp:77  (28L)
e070_dtor_UniformRegionContext() noexcept(false) {
  if (!enable_ctx_) return;
  if (auto ur_op = dyn_cast<uniform::UniformizeRegionsOp>(uniform_op_)) {
    for (auto unit : ur_op.getRegionUnitList(region_idx_)) {
      if (auto get_unit = unit.getDefiningOp<mlir::dataflow::GetUnitOp>())
        addRegDefsForUnit(get_unit);
      else if (auto group_op =
                   unit.getDefiningOp<mlir::dataflow::CreateGroupOp>()) {
        for (auto subunit : group_op.getUnitIds())
          addRegDefsForUnit(subunit.getDefiningOp<mlir::dataflow::GetUnitOp>());
      }
    }
    return;
  }
  if (auto ep_op = dyn_cast<mlir::uniform::EqualizePatternOp>(uniform_op_)) {
    for (auto unit : ep_op.getRegionUnitList(region_idx_)) {
      if (auto get_unit = unit.getDefiningOp<mlir::dataflow::GetUnitOp>())
        addRegDefsForUnit(get_unit);
      else if (auto group_op =
                   unit.getDefiningOp<mlir::dataflow::CreateGroupOp>()) {
        for (auto subunit : group_op.getUnitIds())
          addRegDefsForUnit(subunit.getDefiningOp<mlir::dataflow::GetUnitOp>());
      }
    }
    return;
  }
  DT_ERROR("Unexpected operation for uniform region");
}

// ---- 71/130  createJmpInstr  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:30  (12L)
UniformInstrInfo e071_createJmpInstr() {
  UniformInstrInfo super_instr;
  super_instr.setInstn(OpCodeT::JCMP);
  super_instr.setCommonComment("jump");
  super_instr.setCommonField(
      "mode", OperandAttr("always", OperandAttr::Type::DESCRIPTIVE));
  auto random = rand();
  std::string target = "uniform_tgt_" + std::to_string(random);
  super_instr.setCommonField("pc_target",
                             OperandAttr(target, OperandAttr::Type::INSTR_TAG));
  return super_instr;
}

// ---- 72/130  addEntryToOperandMap  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:52  (121L)
void e072_addEntryToOperandMap(
    llvm::DenseMap<mlir::Value, SdscFoldIdInput>& unit_foldid_map,
    const dcc::DccExtContext& dcc_ext_ctx, std::string operand_name,
    mlir::uniform::QueryMapOp& query_op, Mode mode, double scale) {
  auto key_ops = getListOfKeyOpsFromUniformMapping(query_op);
  auto value_ops = getListOfValueOpsFromUniformMapping(query_op);

  // logic to check if a unit for all folds have same value. If so, then
  // don't entry fold id value into OperandAttr. This is to temporarily
  // help progir-opt in not handling fold id in all its checks.
  std::unordered_map<std::string, std::pair<Value, bool>>
      values_same_map_per_unit_name;
  for (int idx = 0; idx < key_ops.size(); idx++) {
    auto key_unit = key_ops.at(idx);
    auto unit_name =
        getUnitName(dyn_cast<dataflow::GetUnitOp>(key_unit.getDefiningOp()));
    auto value = value_ops.at(idx);

    auto entry = values_same_map_per_unit_name.find(unit_name);
    if (entry != values_same_map_per_unit_name.end()) {
      auto base_val = entry->second.first;

      // if the values are different, then perform semantic check
      // this is useful in case of get_unit operations.
      if (base_val != value) {
        auto base_op = base_val.getDefiningOp();
        auto val_op = value.getDefiningOp();
        auto base_mcg =
            llvm::dyn_cast<dataflow::CreateMulticastGroupOp>(base_op);
        auto val_mcg = llvm::dyn_cast<dataflow::CreateMulticastGroupOp>(val_op);

        if (base_op && val_op) {
          bool allow_diff_operands = false;
          if (base_mcg && val_mcg) {
            // Create multi-cast group ops across fold instances may be
            // different, but their values are the same.
            allow_diff_operands = base_mcg.getProducer().getDefiningOp() ==
                                  val_mcg.getProducer().getDefiningOp();
          }

          dcc::OperationEquivalence oe;
          if (!oe.operationsAreEquivalent(*base_val.getDefiningOp(),
                                          *value.getDefiningOp(), nullptr,
                                          allow_diff_operands)) {
            entry->second.second = false;
          }
        } else {
          entry->second.second = false;
        }
      }
    } else {
      values_same_map_per_unit_name[unit_name] = std::make_pair(value, true);
    }
  }

  for (int idx = 0; idx < key_ops.size(); idx++) {
    auto key_unit = key_ops.at(idx);
    auto unit_name =
        getUnitName(dyn_cast<dataflow::GetUnitOp>(key_unit.getDefiningOp()));
    SdscFoldIdInput id = unit_foldid_map.at(key_unit);
    bool folding_needed =
        !values_same_map_per_unit_name.find(unit_name)->second.second;
    DT_CHECK_MSG(!folding_needed,
                 "Folding in instruction fields are not supported");
    OperandAttr& value = operand_map_[operand_name][unit_name];
    if (auto constant_op =
            dyn_cast<sentient::ConstantOp>(value_ops.at(idx).getDefiningOp())) {
      auto imm = int64_t(constant_op.getValue() * scale);
      if ((mode == Mode::l0_wrap_around) && imm > 1024) imm %= 1024;
      folding_needed ? value.setOperand(imm, id) : value.setOperand(imm);
    } else if (auto get_unit_op = dyn_cast<dataflow::GetUnitOp>(
                   value_ops.at(idx).getDefiningOp())) {
      if (mode == Mode::unit_name) {
        auto consumer_unit_generic = EnumsConversion::senCompToGenericComp.at(
            EnumsConversion::stringToSenComponents.at(
                get_unit_op.getType().str()));
        const auto& consumer_unit_name =
            EnumsConversion::senComponentsToString.at(consumer_unit_generic);
        auto fold_id = folding_needed ? id : std::nullopt;
        updateProperConsumer(dcc_ext_ctx, consumer_unit_generic,
                             consumer_unit_name, value, fold_id);
      } else if (mode == Mode::set_dest_tgt) {
        std::string consumer_str = get_unit_op.getType().lower();
        int core_id = dcc::getCoreId(get_unit_op);
        int corelet_id = dcc::getCoreletId(get_unit_op);
        DT_CHECK_MSG(core_id >= 0 && corelet_id >= 0 && core_id < 32,
                     "invalid core/corelet id");
        auto curr_get_unit =
            dyn_cast<dataflow::GetUnitOp>(key_ops.at(idx).getDefiningOp());
        DT_CHECK_MSG(curr_get_unit,
                     "unable to get the current unit information");
        DT_CHECK(curr_get_unit.getType().lower() == consumer_str &&
                 consumer_str == "sfp" &&
                 "SETDEST can only be used to send data between SFP units of "
                 "different cores");
        DT_CHECK_MSG(corelet_id == dcc::getCoreletId(curr_get_unit),
                     "Currently SFP can only send data to the same corelet of "
                     "another core");
        int64_t mask = (int64_t)1 << core_id;
        folding_needed ? value.setOperand(mask, id) : value.setOperand(mask);
      } else if (mode == Mode::unit_id) {
        auto curr_get_unit =
            dyn_cast<dataflow::GetUnitOp>(value_ops.at(idx).getDefiningOp());
        int core_id = dcc::getCoreId(curr_get_unit);
        folding_needed ? value.setOperand(core_id, id)
                       : value.setOperand(core_id);
      } else {
        auto val = get_unit_op.getType().lower();
        folding_needed
            ? value.setOperand(val, OperandAttr::Type::DESCRIPTIVE, id)
            : value.setOperand(val, OperandAttr::Type::DESCRIPTIVE);
      }
    } else if (auto multicast_op =
                   llvm::dyn_cast<dataflow::CreateMulticastGroupOp>(
                       value_ops.at(idx).getDefiningOp())) {
      auto val = (int)dcc::dataflow::utils::encodeMulticastGroupInfo(
          multicast_op, dcc_ext_ctx);
      folding_needed ? value.setOperand(val, id) : value.setOperand(val);
    }
  }
}

// ---- 73/130  addEntryToOperandMap  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:174  (43L)
void e073_addEntryToOperandMap(
    llvm::DenseMap<mlir::Value, SdscFoldIdInput>& unit_foldid_map,
    std::string operand_name, uniform::QueryMapOp& query_op,
    llvm::DenseMap<mlir::Value, int64_t>& const_key_vals, double scale,
    DataFormats format) {
  // logic to check if a unit for all folds have same value. If so, then
  // don't entry fold id value into OperandAttr. This is to temporarily
  // help progir-opt in not handling fold id in all its checks.
  std::unordered_map<std::string, std::pair<int64_t, bool>>
      values_same_map_per_unit_name;
  for (auto& pair : const_key_vals) {
    auto key_unit = pair.first;
    auto unit_name =
        getUnitName(dyn_cast<dataflow::GetUnitOp>(key_unit.getDefiningOp()));
    auto value = pair.second;

    auto entry = values_same_map_per_unit_name.find(unit_name);
    if (entry != values_same_map_per_unit_name.end()) {
      if (entry->second.first != value) {
        entry->second.second = true;
      }
    } else {
      values_same_map_per_unit_name[unit_name] = std::make_pair(value, true);
    }
  }

  for (auto& pair : const_key_vals) {
    auto key_unit = pair.first;
    auto unit_name =
        getUnitName(dyn_cast<dataflow::GetUnitOp>(key_unit.getDefiningOp()));
    SdscFoldIdInput id = unit_foldid_map.at(key_unit);
    OperandAttr& value = operand_map_[operand_name][unit_name];
    bool folding_needed =
        !values_same_map_per_unit_name.find(unit_name)->second.second;
    DT_CHECK_MSG(!folding_needed,
                 "Folding in instruction fields are not supported");
    value.setOperand(int64_t(pair.second * scale),
                     folding_needed ? id : std::nullopt);
    if (format != DataFormats::INVALID) {
      value.setSenDataType(format);
    }
  }
}

// ---- 74/130  getMaxInstrRegionIndex  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:340  (12L)
size_t e074_getMaxInstrRegionIndex() {
  size_t index = 0;
  size_t max_size = getMaxInstrSize();
  if (instr_lists_.size() <= 1) return index;
  for (int i = 0; i < instr_lists_.size(); i++) {
    if (instr_lists_[i].size() == max_size) {
      index = i;
      break;
    }
  }
  return index;
}

// ---- 75/130  getRegionInstrSize  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:353  (7L)
size_t e075_getRegionInstrSize(std::string unit_name) {
  if (unit_to_region_idx_map_.find(unit_name) !=
      unit_to_region_idx_map_.end()) {
    return instr_lists_[unit_to_region_idx_map_.at(unit_name)].size();
  }
  return getMaxInstrSize();
}

// ---- 76/130  getUnitUniformInstrList  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:383  (9L)
std::vector<UniformInstrInfo> e076_getUnitUniformInstrList(
    std::string unit_name) {
  std::vector<UniformInstrInfo> ret;
  for (auto block : blocks_) {
    auto instr_list = block.getUnitInstrList(unit_name);
    ret.insert(ret.end(), instr_list.begin(), instr_list.end());
  }
  return ret;
}

// ---- 77/130  getLastBlockCurrentInstrSize  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:433  (3L)
size_t e077_getLastBlockCurrentInstrSize() {
  return blocks_.back().getCurrentRegionInstrSize();
}

// ---- 78/130  addInstructionToLastBlock  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:436  (6L)
void e078_addInstructionToLastBlock(UniformInstrInfo instr) {
  if (blocks_.size() == 0) {
    blocks_.emplace_back(UniformInstrBlock::Type::REGULAR);
  }
  blocks_.back().insertInstruction(instr);
}

// ---- 79/130  getNextInstrIndex  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:464  (19L)
InstrIndex e079_getNextInstrIndex() {
  // the returned index assumes next instruction would be in the current block
  // and region. for ifOp, the exact index of next instruction depends on the
  // type of next operation.
  // 1. ifOp {}; otherOp;
  //        -> next_instr_idx=blocks_.back().getCurrentRegionInstrSize()
  // 2. ifOp {}; uniformRegionOp;
  //        -> block_idx=blocks_.size()
  // 3. uniformRegionOp{{ifOp;yield}};
  //        -> block_idx=blocks_.size()
  // 4. uniformRegionOp{{ifOp;otherOp;yield}};
  //        -> next_instr_idx=blocks_.back().getCurrentRegionInstrSize()
  // the retuned index needs to be properly wrapped around based on specific
  // scenario.
  int block_idx = blocks_.size() - 1;
  int region_idx = blocks_.back().getCurrentRegion();
  int next_instr_idx = blocks_.back().getCurrentRegionInstrSize();
  return {block_idx, region_idx, next_instr_idx};
}

// ---- 80/130  setFCValueFromFoldMode  --  dcc/src/Conversion/SentientToProgIR/Utils.cpp:128  (22L)
void e080_setFCValueFromFoldMode(UniformInstrInfo& super_instr, mlir::Operation* op,
                            const SenComponents comp) {
  if (!is_any_of(comp, PE, SFP)) return;

  std::optional<SentientFoldMode> fold_mode =
      dcc::utils::getFoldModeAttributeIfExists(op);
  if (!fold_mode.has_value() || fold_mode.value() == SentientFoldMode::none ||
      fold_mode.value() == SentientFoldMode::fold_A) {
    super_instr.setCommonField(
        "foldctrl", OperandAttr("folda", OperandAttr::Type::DESCRIPTIVE));
  } else if (fold_mode.value() == SentientFoldMode::fold_B) {
    super_instr.setCommonField(
        "foldctrl", OperandAttr("foldb", OperandAttr::Type::DESCRIPTIVE));
  } else if (fold_mode.value() == SentientFoldMode::fold_AB_A ||
             fold_mode.value() == SentientFoldMode::fold_AB_B) {
    super_instr.setCommonField(
        "foldctrl", OperandAttr("2fold2instr", OperandAttr::Type::DESCRIPTIVE));
  } else if (fold_mode.value() == SentientFoldMode::fold_AB_Both) {
    super_instr.setCommonField(
        "foldctrl", OperandAttr("2fold1instr", OperandAttr::Type::DESCRIPTIVE));
  }
}


// ================================================================================================
// LEVEL 2
// ================================================================================================

// ---- 81/130  ConstructLRFADDInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:721  (88L)
std::vector<UniformInstrInfo>
e081_ConstructLRFADDInstr(
    const SenComponents& comp, sentient::AddOp add_op, int element_size,
    std::optional<CodeQualityStats>& cq_stats) {
  UniformInstrInfo super_instr;
  int scale = GetAddressScale(comp, add_op);
  super_instr.setCommonComment("lrf add");
  auto lhs_locale = sentient::getValueRegLocaleAsString(add_op.getInp1());
  auto rhs_locale = sentient::getValueRegLocaleAsString(add_op.getInp2());
  int src0_reg_index = sentient::getValueRegIndex(add_op.getInp1());
  int src1_reg_index = sentient::getValueRegIndex(add_op.getInp2());
  int tgt_reg_index = sentient::getValueRegIndex(add_op.getResult());

  if (dccExtContext().getProgStitch() || forceFullRegInit.getValue() ||
      cq_stats.has_value())
    addToRegsToInit(add_op,
                    {lhs_locale, rhs_locale,
                     sentient::getValueRegLocaleAsString(add_op.getResult())},
                    {src0_reg_index, src1_reg_index, tgt_reg_index});

  auto imm_field_name = is_any_of(comp, LXLU, LXSU) ? "lrfimm" : "imm";
  super_instr.setCommonField("src0", tgt_reg_index);
  if (lhs_locale == "lrf" && rhs_locale == "imm") {
    super_instr.setInstn(OpCodeT::MODLRFIMM);
    fillImmField(super_instr, add_op.getInp2().getDefiningOp(), comp,
                 imm_field_name, element_size, scale);

    // Create a copy operation and then do Add operation
    // This is because LRFMOD target requires to be one of its operand registers
    if (src0_reg_index != tgt_reg_index) {
      if (cq_stats.has_value()) ++cq_stats.value().num_copy_ops_;
      UniformInstrInfo copy_super_instr;
      copy_super_instr.setInstn(OpCodeT::LRFREGCOPY);
      copy_super_instr.setCommonField("src0", tgt_reg_index);
      copy_super_instr.setCommonField("src1", src0_reg_index);
      copy_super_instr.setCommonComment("LRF <- LRF");
      return {copy_super_instr, super_instr};
    } else {
      return {super_instr};
    }
  } else if (lhs_locale == "imm" && rhs_locale == "lrf") {
    super_instr.setInstn(OpCodeT::MODLRFIMM);
    super_instr.setCommonField("src0", src1_reg_index);
    fillImmField(super_instr, add_op.getInp1().getDefiningOp(), comp,
                 imm_field_name, element_size, scale);

    // Create a copy operation and then do Add operation
    // This is because LRFMOD target requires to be one of its operand registers
    if (src1_reg_index != tgt_reg_index) {
      if (cq_stats.has_value()) ++cq_stats.value().num_copy_ops_;
      UniformInstrInfo copy_super_instr;
      copy_super_instr.setInstn(OpCodeT::LRFREGCOPY);
      copy_super_instr.setCommonField("src0", tgt_reg_index);
      copy_super_instr.setCommonField("src1", src1_reg_index);
      copy_super_instr.setCommonComment("LRF <- LRF");
      return {copy_super_instr, super_instr};
    } else {
      return {super_instr};
    }
  } else if (lhs_locale == "lrf" && rhs_locale == "lrf") {
    super_instr.setInstn(OpCodeT::MODLRFREG);

    // Create a copy operation and then do Add operation
    // This is because LRFMOD target requires to be one of its operand registers
    if (src0_reg_index == tgt_reg_index) {
      super_instr.setCommonField("src1", src1_reg_index);
    } else if (src0_reg_index != tgt_reg_index &&
               src1_reg_index == tgt_reg_index) {
      super_instr.setCommonField("src1", src0_reg_index);
    } else {
      if (cq_stats.has_value()) ++cq_stats.value().num_copy_ops_;
      UniformInstrInfo copy_super_instr;
      copy_super_instr.setInstn(OpCodeT::LRFREGCOPY);
      copy_super_instr.setCommonField("src0", tgt_reg_index);
      copy_super_instr.setCommonField("src1", src0_reg_index);
      copy_super_instr.setCommonComment("LRF <- LRF");

      super_instr.setCommonField("src1", src1_reg_index);
      return {copy_super_instr, super_instr};
    }
  } else {
    add_op->emitError("Unable to find Prog.IR for LRF ADD\n");
    signalPassFailure();
    return {super_instr};
  }

  return {super_instr};
}

// ---- 82/130  ConstructLARorEARADDInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:810  (81L)
std::vector<UniformInstrInfo>
e082_ConstructLARorEARADDInstr(
    const SenComponents& comp, sentient::AddOp add_op, int element_size,
    std::optional<CodeQualityStats>& cq_stats) {
  UniformInstrInfo super_instr;
  super_instr.setCommonComment(
      llvm::Twine(stringifySentientRegType(add_op.getRegLocale()) + " add")
          .str());
  bool is_ear = stringifySentientRegType(add_op.getRegLocale()) == "ear";
  auto lhs_locale = sentient::getValueRegLocaleAsString(add_op.getInp1());
  auto rhs_locale = sentient::getValueRegLocaleAsString(add_op.getInp2());
  DT_CHECK((lhs_locale == stringifySentientRegType(add_op.getRegLocale()) ||
            rhs_locale == stringifySentientRegType(add_op.getRegLocale())) &&
           "one of the operands has incorrect locale");
  int src0_reg_index = sentient::getValueRegIndex(add_op.getInp1());
  int src1_reg_index = sentient::getValueRegIndex(add_op.getInp2());
  int tgt_reg_index = sentient::getValueRegIndex(add_op.getResult());

  if (dccExtContext().getProgStitch() || forceFullRegInit.getValue() ||
      cq_stats.has_value())
    addToRegsToInit(add_op,
                    {lhs_locale, rhs_locale,
                     sentient::getValueRegLocaleAsString(add_op.getResult())},
                    {src0_reg_index, src1_reg_index, tgt_reg_index});

  super_instr.setCommonField("src0", tgt_reg_index);
  if (((lhs_locale == "lar" || lhs_locale == "ear") && rhs_locale == "imm") ||
      ((rhs_locale == "lar" || rhs_locale == "ear") && lhs_locale == "imm")) {
    mlir::Value imm_operand =
        rhs_locale == "imm" ? add_op.getInp2() : add_op.getInp1();
    mlir::Value reg_operand =
        rhs_locale == "imm" ? add_op.getInp1() : add_op.getInp2();
    super_instr.setInstn(is_ear ? OpCodeT::ADDEARIMM : OpCodeT::ADDLARIMM);
    fillImmField(
        super_instr, imm_operand.getDefiningOp(), comp, "imm", element_size,
        GetAddressScale(comp, (rhs_locale == "imm" ? lhs_locale : rhs_locale)));
    int src_reg_index = sentient::getValueRegIndex(reg_operand);
    if (src_reg_index != tgt_reg_index) {
      // Create a copy operation and then do Add operation.
      // This is because ADDLARIMM/ADDEARIMM target requires to be one of its
      // operand registers.
      if (cq_stats.has_value()) ++cq_stats.value().num_copy_ops_;
      UniformInstrInfo copy_super_instr;
      copy_super_instr.setInstn(is_ear ? OpCodeT::EARREGCOPY
                                       : OpCodeT::LARREGCOPY);
      copy_super_instr.setCommonField("src0", tgt_reg_index);
      copy_super_instr.setCommonField("src1", src_reg_index);
      copy_super_instr.setCommonComment(is_ear ? "EAR <- EAR" : "LAR <- LAR");
      return {copy_super_instr, super_instr};
    } else {
      return {super_instr};
    }
  } else if ((lhs_locale == "lar" && rhs_locale == "lar") ||
             (lhs_locale == "ear" && rhs_locale == "ear")) {
    super_instr.setInstn(is_ear ? OpCodeT::MODEARREG : OpCodeT::MODLARREG);
    if (src0_reg_index == tgt_reg_index) {
      super_instr.setCommonField("src1", src1_reg_index);
    } else if (src1_reg_index == tgt_reg_index) {
      super_instr.setCommonField("src1", src0_reg_index);
    } else {
      // Create a copy operation and then do Add operation.
      // This is because MOD[EAR|LAR]REG target requires to be one of its
      // operand registers.
      if (cq_stats.has_value()) ++cq_stats.value().num_copy_ops_;
      UniformInstrInfo copy_super_instr;
      copy_super_instr.setInstn(is_ear ? OpCodeT::EARREGCOPY
                                       : OpCodeT::LARREGCOPY);
      copy_super_instr.setCommonField("src0", tgt_reg_index);
      copy_super_instr.setCommonField("src1", src0_reg_index);
      copy_super_instr.setCommonComment(is_ear ? "EAR <- EAR" : "LAR <- LAR");

      super_instr.setCommonField("src1", src1_reg_index);
      return {copy_super_instr, super_instr};
    }
  } else {
    add_op->emitError("Unable to find Prog.IR for LAR/EAR ADD\n");
    signalPassFailure();
  }

  return {super_instr};
}

// ---- 83/130  ConstructLRFSUBInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:999  (136L)
std::vector<UniformInstrInfo>
e083_ConstructLRFSUBInstr(
    const SenComponents& comp, sentient::SubOp sub_op, int element_size,
    std::optional<CodeQualityStats>& cq_stats) {
  UniformInstrInfo super_instr;
  super_instr.setCommonComment("lrf sub");
  auto lhs_locale = sentient::getValueRegLocaleAsString(sub_op.getInp1());
  auto rhs_locale = sentient::getValueRegLocaleAsString(sub_op.getInp2());
  int src0_reg_index = sentient::getValueRegIndex(sub_op.getInp1());
  int src1_reg_index = sentient::getValueRegIndex(sub_op.getInp2());
  int tgt_reg_index = sentient::getValueRegIndex(sub_op.getResult());

  if (dccExtContext().getProgStitch() || forceFullRegInit.getValue() ||
      cq_stats.has_value())
    addToRegsToInit(sub_op,
                    {lhs_locale, rhs_locale,
                     sentient::getValueRegLocaleAsString(sub_op.getResult())},
                    {src0_reg_index, src1_reg_index, tgt_reg_index});

  if (lhs_locale == "lrf" && rhs_locale == "imm") {
    super_instr.setInstn(OpCodeT::MODLRFIMM);
    auto imm_field_name = is_any_of(comp, LXLU, LXSU) ? "lrfimm" : "imm";
    fillImmField(super_instr, sub_op.getInp2().getDefiningOp(), comp,
                 imm_field_name, element_size,
                 GetAddressScale(comp, lhs_locale));

    // Create a copy operation and then do Add operation
    // This is because LRFMOD target requires to be one of its operand registers
    super_instr.setCommonField("src0", tgt_reg_index);
    if (src0_reg_index != tgt_reg_index) {
      if (cq_stats.has_value()) ++cq_stats.value().num_copy_ops_;
      UniformInstrInfo copy_super_instr;
      copy_super_instr.setInstn(OpCodeT::LRFREGCOPY);
      copy_super_instr.setCommonField("src0", tgt_reg_index);
      copy_super_instr.setCommonField("src1", src0_reg_index);
      copy_super_instr.setCommonComment("LRF <- LRF");
      return {copy_super_instr, super_instr};
    } else {
      return {super_instr};
    }
  } else if (lhs_locale == "imm" && rhs_locale == "lrf") {
    DT_CHECK_MSG(is_any_of(comp, LXLU, LXSU),
                 "LRF SUB with imm being first operand only supported in LXLU");
    super_instr.setCommonField("src0", tgt_reg_index);
    super_instr.setInstn(OpCodeT::SUBLRFIMM);
    std::vector<UniformInstrInfo> instr_vec;
    auto imm_field_name = is_any_of(comp, LXLU, LXSU) ? "lrfimm" : "imm";
    bool is_imm_out_of_range = false;
    if (auto const_op =
            sub_op.getInp1().getDefiningOp<sentient::ConstantOp>()) {
      int imm =
          sub_op.getInp1().getDefiningOp<sentient::ConstantOp>().getValue() *
          (float)element_size / 8.0 / (float)GetAddressScale(comp, rhs_locale);
      if (imm > 0x1FFFFF) {
        // IMM value can be larger than 2MB due to liveRangeReduction or Const
        // propagation. need to break IMM to 2 instructions. Note, IMM can never
        // be larger than 4MB.
        super_instr.setCommonField(imm_field_name, 0x1FFFFF);
        int imm_remain = imm - 0x1FFFFF;
        UniformInstrInfo add_super_instr;
        add_super_instr.setInstn(OpCodeT::MODLRFIMM);
        add_super_instr.setCommonField(imm_field_name, imm_remain);
        add_super_instr.setCommonField("src0", tgt_reg_index);
        instr_vec = {super_instr, add_super_instr};
      } else {
        super_instr.setCommonField(imm_field_name, imm);
        instr_vec = {super_instr};
      }
    } else if (auto query_op =
                   sub_op.getInp1().getDefiningOp<uniform::QueryMapOp>()) {
      // IMM value can be larger than 2MB due to liveRangeReduction or Const
      // propagation. need to break IMM to 2 instructions. Note, IMM can never
      // be larger than 4MB.
      // check if any const val is out of range
      auto constant_kv_pair = getConstantTargetKeyValues(query_op);
      float scale =
          (float)element_size / 8.0 / (float)GetAddressScale(comp, rhs_locale);
      for (auto& pair : constant_kv_pair) {
        pair.second *= scale;
        if (pair.second > 0x1FFFFF) {
          is_imm_out_of_range = true;
        }
      }
      if (is_imm_out_of_range) {
        // if out of range, reset const_vals and remaining vals for add_op
        llvm::DenseMap<mlir::Value, int64_t> remain_const_vals;
        for (auto& pair : constant_kv_pair) {
          if (pair.second > 0x1FFFFF) {
            remain_const_vals[pair.first] = pair.second - 0x1FFFFF;
            pair.second = 0x1FFFFF;
          } else {
            remain_const_vals[pair.first] = 0;
          }
        }

        OperandMap sub_operand_map(unit_foldid_map_, imm_field_name, query_op,
                                   constant_kv_pair);
        super_instr.setOperandMap(sub_operand_map);

        UniformInstrInfo add_super_instr;
        add_super_instr.setInstn(OpCodeT::MODLRFIMM);
        add_super_instr.setCommonField("src0", tgt_reg_index);
        OperandMap add_operand_map(unit_foldid_map_, imm_field_name, query_op,
                                   remain_const_vals);
        add_super_instr.setOperandMap(add_operand_map);
        instr_vec = {super_instr, add_super_instr};
      } else {
        OperandMap sub_operand_map(unit_foldid_map_, imm_field_name, query_op,
                                   constant_kv_pair);
        super_instr.setOperandMap(sub_operand_map);
        instr_vec = {super_instr};
      }
    }

    // Create a copy operation and then do Add operation
    // This is because LRFMOD target requires to be one of its operand registers
    if (src1_reg_index != tgt_reg_index) {
      if (cq_stats.has_value()) ++cq_stats.value().num_copy_ops_;
      UniformInstrInfo copy_super_instr;
      copy_super_instr.setInstn(OpCodeT::LRFREGCOPY);
      copy_super_instr.setCommonField("src0", tgt_reg_index);
      copy_super_instr.setCommonField("src1", src1_reg_index);
      copy_super_instr.setCommonComment("LRF <- LRF");
      std::vector<UniformInstrInfo> ret{copy_super_instr};
      ret.insert(ret.end(), instr_vec.begin(), instr_vec.end());
      return ret;
    } else {
      return instr_vec;
    }
  } else {
    sub_op->emitError("Unable to find Prog.IR for LRF SUB\n");
    signalPassFailure();
  }

  return {super_instr};
}

// ---- 84/130  ConstructLARorEARSUBInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1136  (46L)
std::vector<UniformInstrInfo>
e084_ConstructLARorEARSUBInstr(
    const SenComponents& comp, sentient::SubOp sub_op, int element_size,
    std::optional<CodeQualityStats>& cq_stats) {
  UniformInstrInfo super_instr;
  super_instr.setCommonComment(
      llvm::Twine(stringifySentientRegType(sub_op.getRegLocale()) + " sub")
          .str());
  auto lhs_locale = sentient::getValueRegLocaleAsString(sub_op.getInp1());
  auto rhs_locale = sentient::getValueRegLocaleAsString(sub_op.getInp2());
  DT_CHECK_MSG(rhs_locale == stringifySentientRegType(sub_op.getRegLocale()),
               "operand has incorrect locale");
  int src0_reg_index = sentient::getValueRegIndex(sub_op.getInp1());
  int src1_reg_index = sentient::getValueRegIndex(sub_op.getInp2());
  int tgt_reg_index = sentient::getValueRegIndex(sub_op.getResult());

  if (dccExtContext().getProgStitch() || forceFullRegInit.getValue() ||
      cq_stats.has_value())
    addToRegsToInit(sub_op,
                    {lhs_locale, rhs_locale,
                     sentient::getValueRegLocaleAsString(sub_op.getResult())},
                    {src0_reg_index, src1_reg_index, tgt_reg_index});

  if (rhs_locale == "lar" && lhs_locale == "imm") {
    super_instr.setInstn(OpCodeT::SUBLARIMM);
    super_instr.setCommonField("src0", tgt_reg_index);
    fillImmField(super_instr, sub_op.getInp1().getDefiningOp(), comp, "imm",
                 element_size, GetAddressScale(comp, rhs_locale));
    if (src1_reg_index != tgt_reg_index) {
      // Create a copy operation
      if (cq_stats.has_value()) ++cq_stats.value().num_copy_ops_;
      UniformInstrInfo copy_super_instr;
      copy_super_instr.setInstn(OpCodeT::LARREGCOPY);
      copy_super_instr.setCommonField("src0", tgt_reg_index);
      copy_super_instr.setCommonField("src1", src1_reg_index);
      copy_super_instr.setCommonComment("LAR <- LAR");
      return {copy_super_instr, super_instr};
    } else {
      return {super_instr};
    }
  } else {
    sub_op->emitError("Unable to find Prog.IR for LAR/EAR SUB\n");
    signalPassFailure();
    return {super_instr};
  }
}

// ---- 85/130  ConstructFMAInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1423  (291L)
UniformInstrInfo e085_ConstructFMAInstr(
    sentient::MacOp fma_op, const SenComponents& comp,
    XRFRegisterAnalyzer* xrf_reg_analyzer) {
  UniformInstrInfo super_instr;
  std::string opA_precision =
      stringifySentientPrecision(fma_op.getOpAPrecision()).str();
  std::string opB_precision =
      stringifySentientPrecision(fma_op.getOpBPrecision()).str();
  std::string opC_precision =
      stringifySentientPrecision(fma_op.getOpCPrecision()).str();
  std::string compute_precision =
      stringifySentientPrecision(fma_op.getComputePrecision()).str();
  std::string result_precision =
      stringifySentientPrecision(fma_op.getResultPrecision()).str();
  if (fma_op.getDbgName().has_value())
    super_instr.setCommonComment(fma_op.getDbgName().value().str());

  // Need to know which operand is src0 as src0 allowed precision has
  // different constraints than src1/src2.
  int src0_operand_idx = -1;
  if (fma_op.getOpAPortID() == 0)
    src0_operand_idx = 0;
  else if (fma_op.getOpBPortID() == 0)
    src0_operand_idx = 1;
  else if (fma_op.getOpCPortID() == 0)
    src0_operand_idx = 2;

  if (is_any_of(comp, PE, SFP) &&
      (result_precision != "none" || !fma_op.getResultForwarding()))
    verifyOnTheFlyConversions(dccExtContext(), comp, opA_precision,
                              opB_precision, opC_precision, src0_operand_idx,
                              compute_precision, result_precision);

  if (is_any_of(comp, PE, SFP) && result_precision != compute_precision &&
      result_precision != "none") {
    DT_CHECK_MSG(
        result_precision != "fp32" || dccExtContext().getArch() >= SEN1P5_ISA,
        "FPUOP is not well defined for FMA32 in RCUDD1A");
    // fp16 really means dlfp16
    std::string fpuop =
        (result_precision == "fp16") ? "dlfp16" : result_precision;
    super_instr.setCommonField(
        "fpuop", OperandAttr(fpuop, OperandAttr::Type::DESCRIPTIVE));
  }

  if (fma_op.getMode() == SentientFMAmode::fused_mul_add) {
    if (is_any_of(compute_precision, "int4", "mxint4")) {
      super_instr.setInstn(OpCodeT::IMA4);
    } else if (compute_precision == "int8") {
      super_instr.setInstn(OpCodeT::IMA8);
    } else if (is_any_of(compute_precision, "fp8", "fp80", "mxfp8")) {
      super_instr.setInstn(OpCodeT::FMA8);
    } else if (compute_precision == "mxfp4") {
      super_instr.setInstn(OpCodeT::FMA4);
    } else if (is_any_of(compute_precision, "fp16", "fp32", "bf16")) {
      super_instr.setInstn(OpCodeT::FMA);
      if (comp == SFP ||
          (dccExtContext().getArch() >= SEN1P5_ISA && comp == PE)) {
        super_instr.setCommonField(
            "mode",
            OperandAttr(compute_precision, OperandAttr::Type::DESCRIPTIVE));
      }
    } else {
      fma_op->emitError(
          "FMA instructions are currently "
          "supported only for int4, int8, mxint4, mxfp4, mxfp8, fp8, bf16, "
          "fp16 and "
          "fp32 types");
      signalPassFailure();
    }
  } else if (fma_op.getMode() == SentientFMAmode::fused_neg_mul_sub) {
    DT_CHECK_MSG(is_any_of(comp, PE, SFP),
                 "FNMS is supported only in PE and SFP");
    super_instr.setInstn(OpCodeT::FNMS);
    if (comp == SFP ||
        (dccExtContext().getArch() >= SEN1P5_ISA && comp == PE)) {
      super_instr.setCommonField(
          "mode",
          OperandAttr(compute_precision, OperandAttr::Type::DESCRIPTIVE));
    }
  }

  if (is_any_of(comp, SFP, PE)) {
    if (auto mask_constant =
            dyn_cast<sentient::ConstantOp>(fma_op.getMask().getDefiningOp())) {
      super_instr.setCommonField("mask",
                                 OperandAttr(255 - mask_constant.getValue()));
    } else {
      fma_op->emitError("Mask constant value has to exist.");
      signalPassFailure();
    }
  }

  std::string opA_hw, opB_hw, opC_hw;

  opA_hw = "src" + std::to_string(fma_op.getOpAPortID());
  opB_hw = "src" + std::to_string(fma_op.getOpBPortID());
  opC_hw = "src" + std::to_string(fma_op.getOpCPortID());

  /*if (comp == PT) {
    if (fma_op.getOpA().contains("west") || fma_op.getOpA().contains("one") ||
        fma_op.getOpB().contains("rf") || fma_op.getOpB().contains("zero")) {
      opA_hw = "src2";
      opB_hw = "src0";
    } else {
      opA_hw = "src0";
      opB_hw = "src2";
    }
    opC_hw = "src1";
  } else {  // PE/SFP
    if (fma_op.getOpA() == "zero" || fma_op.getOpA() == "one") {
      if (fma_op.getOpA() == "zero" && fma_op.getOpB() == "one") {
        // simple data forwarding case
        opA_hw = "src2";
        opB_hw = "src1";
        opC_hw = "src0";
      } else if (fma_op.getOpA() == "one" && fma_op.getOpB() == "zero") {
        // simple data forwarding case
        opA_hw = "src1";
        opB_hw = "src2";
        opC_hw = "src0";
      } else {
        opA_hw = "src1";
        opB_hw = "src0";
        opC_hw = "src2";
      }
    } else if (fma_op.getOpB() == "ptint16") {
      opA_hw = "src1";
      opB_hw = "src0";
      opC_hw = "src2";
    } else if (fma_op.getOpC() == "ptint16" &&
               (fma_op.getOpA() == "one" || fma_op.getOpB() == "one")) {
      // simple addition
      if (fma_op.getOpA() == "one") {
        opA_hw = "src1";
        opB_hw = "src2";
        opC_hw = "src0";
      } else {
        opA_hw = "src2";
        opB_hw = "src1";
        opC_hw = "src0";
      }
    } else {
      opA_hw = "src0";
      opB_hw = "src1";
      opC_hw = "src2";
    }
}*/

  setSentientComputeInputProgIROperands(
      fma_op, comp, super_instr, stringifySentientComputePort(fma_op.getOpA()),
      opA_hw, opA_precision, compute_precision, result_precision,
      !fma_op.getOpAForwarding().empty());
  setSentientComputeInputProgIROperands(
      fma_op, comp, super_instr, stringifySentientComputePort(fma_op.getOpB()),
      opB_hw, opB_precision, compute_precision, result_precision,
      !fma_op.getOpBForwarding().empty());
  setSentientComputeInputProgIROperands(
      fma_op, comp, super_instr, stringifySentientComputePort(fma_op.getOpC()),
      opC_hw, opC_precision, compute_precision, result_precision,
      !fma_op.getOpCForwarding().empty());

  setSentientComputeOutputProgIROperands(fma_op, comp, super_instr,
                                         fma_op.getOpAForwarding(), opA_hw,
                                         compute_precision, result_precision);
  setSentientComputeOutputProgIROperands(fma_op, comp, super_instr,
                                         fma_op.getOpBForwarding(), opB_hw,
                                         compute_precision, result_precision);
  setSentientComputeOutputProgIROperands(fma_op, comp, super_instr,
                                         fma_op.getOpCForwarding(), opC_hw,
                                         compute_precision, result_precision);
  setSentientComputeOutputProgIROperands(fma_op, comp, super_instr,
                                         fma_op.getResultForwarding(), "result",
                                         compute_precision, result_precision);

  if (!super_instr.hasCommonField("tgtrf")) {
    super_instr.setCommonField(
        "tgtrf", OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
  }

  super_instr.setCommonField(
      "unroll",
      OperandAttr(stringifySentientUnrollFactor(fma_op.getUnrollFactor()).str(),
                  OperandAttr::Type::DESCRIPTIVE));

  bool unroll_illegal = false;
  if (comp == PT && opA_hw == "src2") {
    unroll_illegal = fma_op.getUnrollIncrOpA();
  } else {
    super_instr.setCommonField("unrlfld" + opA_hw,
                               OperandAttr(fma_op.getUnrollIncrOpA()));
  }

  if (comp == PT && opB_hw == "src2") {
    unroll_illegal = fma_op.getUnrollIncrOpB();
  } else {
    super_instr.setCommonField("unrlfld" + opB_hw,
                               OperandAttr(fma_op.getUnrollIncrOpB()));
  }

  if (comp == PT && opC_hw == "src2") {
    unroll_illegal = fma_op.getUnrollIncrOpC();
  } else {
    super_instr.setCommonField("unrlfld" + opC_hw,
                               OperandAttr(fma_op.getUnrollIncrOpC()));
  }

  super_instr.setCommonField("unrlfldtgt",
                             OperandAttr(fma_op.getUnrollIncrResult()));
  if (unroll_illegal) {
    fma_op->emitError(
        "PT FMA/IMA does not support setting unrlfldsrc2, but it was "
        "requested");
    signalPassFailure();
  }

  if (comp != PT) {
    if (auto mask_constant =
            dyn_cast<sentient::ConstantOp>(fma_op.getMask().getDefiningOp())) {
      super_instr.setCommonField("mask",
                                 OperandAttr(255 - mask_constant.getValue()));
    } else {
      fma_op->emitError("Mask constant value has to exist.");
      signalPassFailure();
    }
  }

  if (dccExtContext().getArch() >= SEN1P5_ISA) {
    dcc::sentient::utils::setFCValueFromFoldMode(super_instr, fma_op, comp);

    if (dtGetEnv<bool>("SET_IFIFO_CONVERT").value_or(true)) {
      // In the Sentient IR MAC operation, MAC = opA x opB + opC
      // In sentient 1p5 PT fp8 mode, it is required that the input fp8 data
      // from n-link needs to be transformed into fp9 and store into XRF or
      // forward to south via an explicit bit, unlike DD2 which is automatically
      // handled in the hardware.
      if (comp == PT &&
          is_any_of(fma_op.getComputePrecision(), SentientPrecision::fp8,
                    SentientPrecision::mxfp8, SentientPrecision::mxfp4,
                    SentientPrecision::mxint4) &&
          is_any_of(SentientComputePort::zero, fma_op.getOpA(),
                    fma_op.getOpB()) &&
          fma_op.getOpC() == SentientComputePort::north) {
        DT_CHECK_MSG(is_any_of(fma_op.getOpCPrecision(), SentientPrecision::fp4,
                               SentientPrecision::fp8),
                     "N-link has to be in fp4/fp8 precision");
        if (is_any_of(fma_op.getComputePrecision(), SentientPrecision::mxfp8,
                      SentientPrecision::mxfp4, SentientPrecision::mxint4)) {
          // Check for the isDataWeight attribute set by AnnotateMacXRFWtRange
          // pass
          bool is_data_weight = false;
          if (auto is_data_weight_attr =
                  fma_op->getAttrOfType<BoolAttr>("isDataWeight")) {
            // Attribute is present, use it
            is_data_weight = is_data_weight_attr.getValue();
          } else {
            // Attribute not present (MAC with no XRF pointers)
            // Find closest previous FMA with XRF ptrs and use its annotation
            sentient::MacOp prev_fma_with_ptrs = nullptr;
            for (Operation* prev_op = fma_op->getPrevNode(); prev_op;
                 prev_op = prev_op->getPrevNode()) {
              if (auto prev_fma = dyn_cast<sentient::MacOp>(prev_op)) {
                if (prev_fma.getPointers().size() > 0) {
                  prev_fma_with_ptrs = prev_fma;
                  break;
                }
              }
            }
            DT_CHECK_MSG(prev_fma_with_ptrs,
                         "No previous MacOp with XRF ptrs found");

            // Get the annotation from the previous MAC with pointers
            auto prev_is_data_weight_attr =
                prev_fma_with_ptrs->getAttrOfType<BoolAttr>("isDataWeight");
            assert(prev_is_data_weight_attr);
            is_data_weight = prev_is_data_weight_attr.getValue();
          }

          if ((is_data_weight &&
               fma_op.getComputePrecision() == SentientPrecision::mxfp8) ||
              (!is_data_weight &&
               fma_op.getComputePrecision() == SentientPrecision::mxfp4))
            super_instr.setCommonField("ififo_conv", "yes");
        } else {
          super_instr.setCommonField("ififo_conv", "yes");
        }
      }
    }
  }
  return super_instr;
}

// ---- 86/130  ConstructBinaryInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:1715  (485L)
UniformInstrInfo e086_ConstructBinaryInstr(
    sentient::BinaryOp binary_op, const SenComponents& comp) {
  UniformInstrInfo super_instr;
  std::string opA_precision =
      stringifySentientPrecision(binary_op.getOpAPrecision()).str();
  std::string opB_precision =
      stringifySentientPrecision(binary_op.getOpBPrecision()).str();
  std::string compute_precision =
      stringifySentientPrecision(binary_op.getComputePrecision()).str();
  std::string result_precision =
      stringifySentientPrecision(binary_op.getResultPrecision()).str();
  if (binary_op.getDbgName().has_value())
    super_instr.setCommonComment(binary_op.getDbgName().value().str());

  // Need to know which operand is src0 as src0 allowed precision has
  // different constraints than src1/src2.
  int src0_operand_idx = -1;
  if (binary_op.getOpAPortID() == 0)
    src0_operand_idx = 0;
  else if (binary_op.getOpBPortID() == 0)
    src0_operand_idx = 1;

  bool instr_has_fpuop_field = false;
  bool is_fcmp_or_fminmax = false;
  bool is_gcvt_or_fcvt = false;
  if (binary_op.getBinaryOp() == SentientBinaryOperator::add ||
      binary_op.getBinaryOp() == SentientBinaryOperator::mul ||
      binary_op.getBinaryOp() == SentientBinaryOperator::mul_div2) {
    if (comp == PT) {
      if (binary_op.getBinaryOp() == SentientBinaryOperator::mul_div2) {
        binary_op.emitOpError(
            "Fused multiply and divide by 2 is not allowed in Pt units.");
        signalPassFailure();
        return super_instr;
      }
      if (is_any_of(compute_precision, "int4", "mxint4")) {
        super_instr.setInstn(OpCodeT::IMA4);
      } else if (compute_precision == "int8") {
        super_instr.setInstn(OpCodeT::IMA8);
      } else if (is_any_of(compute_precision, "fp8", "fp80", "mxfp8")) {
        super_instr.setInstn(OpCodeT::FMA8);
      } else if (compute_precision == "mxfp4") {
        super_instr.setInstn(OpCodeT::FMA4);
      } else if (compute_precision == "fp16") {
        super_instr.setInstn(OpCodeT::FMA);
      } else {
        binary_op->emitError(
            "FMA instructions in PT are currently "
            "supported only for int4, int8, mxint4, mxfp4, mxfp8, fp8 and fp16 "
            "types");
        signalPassFailure();
      }
      if (binary_op.getBinaryOp() == SentientBinaryOperator::mul) {
        super_instr.setCommonField(
            "src1", OperandAttr("0.0", OperandAttr::Type::DESCRIPTIVE));  // 0.0
      } else {
        super_instr.setCommonField(
            "src2", OperandAttr("1.0", OperandAttr::Type::DESCRIPTIVE));  // 1.0
      }
    } else {  // PE/SFP
      instr_has_fpuop_field = true;
      if (binary_op.getBinaryOp() == SentientBinaryOperator::mul) {
        super_instr.setInstn(OpCodeT::FMUL);
        super_instr.setCommonField(
            "src2",
            OperandAttr("0.0", OperandAttr::Type::DESCRIPTIVE));  // regular mul
      } else if (binary_op.getBinaryOp() == SentientBinaryOperator::add) {
        super_instr.setInstn(OpCodeT::FMA);
        super_instr.setCommonField(
            "src1", OperandAttr("1.0", OperandAttr::Type::DESCRIPTIVE));  // 1.0
      } else if (binary_op.getBinaryOp() == SentientBinaryOperator::mul_div2) {
        super_instr.setInstn(OpCodeT::FMUL);
        super_instr.setCommonField(
            "src2",
            OperandAttr("muldiv2",
                        OperandAttr::Type::DESCRIPTIVE));  // mul and div by 2
      } else {
        binary_op.emitOpError("Unknown binary operation.");
        signalPassFailure();
        return super_instr;
      }
      verifyOnTheFlyConversions(
          dccExtContext(), comp, opA_precision, opB_precision,
          /*opC_precision*/ compute_precision, src0_operand_idx,
          compute_precision, result_precision);
    }
  } else if (binary_op.getBinaryOp() == SentientBinaryOperator::sub &&
             is_any_of(comp, PE, SFP)) {
    instr_has_fpuop_field = true;
    super_instr.setInstn(OpCodeT::FNMS);
    verifyOnTheFlyConversions(
        dccExtContext(), comp, opA_precision, opB_precision,
        /*opC_precision*/ compute_precision, src0_operand_idx,
        compute_precision, result_precision);
    super_instr.setCommonField(
        "src1", OperandAttr("1.0", OperandAttr::Type::DESCRIPTIVE));  // 1.0
  } else {
    int second_opcode;
    if (comp == PT) {
      binary_op->emitError("Unsupported operation in PT");
      signalPassFailure();
    } else if (binary_op.getBinaryOp() == SentientBinaryOperator::min ||
               binary_op.getBinaryOp() == SentientBinaryOperator::max ||
               binary_op.getBinaryOp() == SentientBinaryOperator::abs_max ||
               binary_op.getBinaryOp() == SentientBinaryOperator::abs_min) {
      super_instr.setInstn(OpCodeT::FMINMAX);
      is_fcmp_or_fminmax = true;
      DT_CHECK_MSG(
          (opA_precision == compute_precision || opA_precision == "int1") &&
              opB_precision == compute_precision,
          "Input on the fly conversion not supported by  FMINMAX");
      DT_CHECK_MSG(compute_precision == result_precision,
                   "Output on the fly conversion not supported by FMINMAX");
      if (binary_op.getBinaryOp() == SentientBinaryOperator::max) {
        second_opcode = 0;
      } else if (binary_op.getBinaryOp() == SentientBinaryOperator::abs_max) {
        second_opcode = 2;  // ABS(X)  > ABS(Z) ? ABS(X) : ABS(Z)
      } else if (binary_op.getBinaryOp() == SentientBinaryOperator::min) {
        second_opcode = 4;
      } else if (binary_op.getBinaryOp() == SentientBinaryOperator::abs_min) {
        second_opcode = 6;  //  ABS(X) <= ABS(Z) ? ABS(X) : ABS(Z)
      } else {
        binary_op->emitError("Unsupported MINMAX variant");
        signalPassFailure();
      }
    } else if (binary_op.getBinaryOp() == SentientBinaryOperator::and0 ||
               binary_op.getBinaryOp() == SentientBinaryOperator::or0 ||
               binary_op.getBinaryOp() == SentientBinaryOperator::xnor ||
               binary_op.getBinaryOp() == SentientBinaryOperator::and_not) {
      DT_CHECK_MSG(compute_precision == "none",
                   "LOGICAL instruction precision expected to be none");
      DT_CHECK_MSG(
          opA_precision == result_precision &&
              opB_precision == result_precision,
          "Input precisions expected to match result precision for LOGICAL");
      super_instr.setInstn(OpCodeT::LOGICAL);
      if (binary_op.getBinaryOp() == SentientBinaryOperator::and0) {
        second_opcode = 0;
      } else if (binary_op.getBinaryOp() == SentientBinaryOperator::or0) {
        second_opcode = 1;
      } else if (binary_op.getBinaryOp() == SentientBinaryOperator::xnor) {
        second_opcode = 2;
      } else if (binary_op.getBinaryOp() == SentientBinaryOperator::and_not) {
        second_opcode = 3;
      } else {
        binary_op->emitError("Unsupported LOGICAL variant");
        signalPassFailure();
      }
    } else if (stringifySentientBinaryOperator(binary_op.getBinaryOp())
                   .contains("merge")) {
      DT_CHECK_MSG(compute_precision == "none",
                   "MERGE instruction precision expected to be none");
      DT_CHECK_MSG(
          opA_precision == result_precision &&
              opB_precision == result_precision,
          "Input precisions expected to match result precision for MERGE");

      // MERGE mode is always fp16
      super_instr.setInstn(OpCodeT::MERGE);
      auto bop = stringifySentientBinaryOperator(binary_op.getBinaryOp());
      if (bop.contains("16h")) {
        second_opcode = 0;
      } else if (bop.contains("16l")) {
        second_opcode = 1;
      } else if (bop.contains("32h")) {
        second_opcode = 2;
      } else if (bop.contains("32l")) {
        second_opcode = 3;
      } else if (bop.contains("64h")) {
        second_opcode = 4;
      } else if (bop.contains("64l")) {
        second_opcode = 5;
      } else if (bop.contains("8h")) {
        second_opcode = 6;
      } else if (bop.contains("8l")) {
        second_opcode = 7;
      } else {
        binary_op->emitError("Unsupported MERGE instruction");
        signalPassFailure();
      }
    } else if (stringifySentientBinaryOperator(binary_op.getBinaryOp())
                   .contains("pack")) {
      DT_CHECK_MSG(compute_precision == "none",
                   "PACK instruction precision expected to be none");
      DT_CHECK_MSG(
          opA_precision == result_precision &&
              opB_precision == result_precision,
          "Input precisions expected to match result precision for PACK");
      super_instr.setInstn(OpCodeT::PACK);
      auto bop_str =
          stringifySentientBinaryOperator(binary_op.getBinaryOp()).str();
      std::string pack_no = bop_str.substr(4, bop_str.size() - 4);
      pack_no.erase(std::remove_if(pack_no.begin(), pack_no.end(), ::isspace),
                    pack_no.end());
      second_opcode = stoi(pack_no);
      if (second_opcode < 0 || second_opcode > 27) {
        binary_op->emitError("Unsupported PACK instruction");
        signalPassFailure();
      }
    } else if (stringifySentientBinaryOperator(binary_op.getBinaryOp())
                   .contains("gcvt")) {
      is_gcvt_or_fcvt = true;
      DT_CHECK_MSG(compute_precision == "fp16",
                   "GCVT instruction precision expected to be fp16");
      super_instr.setInstn(OpCodeT::GCVT);
      auto bop_str =
          stringifySentientBinaryOperator(binary_op.getBinaryOp()).str();
      std::string gcvt_no = bop_str.substr(8, bop_str.size() - 8);
      second_opcode = stoi(gcvt_no);
      switch (second_opcode) {
        case 0:
        case 4:
        case 24:
        case 28:
          DT_CHECK_MSG(
              opA_precision == "fp16" && opB_precision == "fp16",
              "Unexpected input/result precisions for GCVT mode 0/4/24/28");
          break;
        default:
          binary_op->emitError("Unsupported GCVT instruction");
          signalPassFailure();
      }
    } else if (stringifySentientBinaryOperator(binary_op.getBinaryOp())
                   .contains("fcvt")) {
      is_gcvt_or_fcvt = true;
      DT_CHECK_MSG(compute_precision == "fp32",
                   "FCVT instruction precision expected to be fp32");
      super_instr.setInstn(OpCodeT::FCVT);
      auto bop_str =
          stringifySentientBinaryOperator(binary_op.getBinaryOp()).str();
      std::string fcvt_no = bop_str.substr(8, bop_str.size() - 8);
      second_opcode = stoi(fcvt_no);
      switch (second_opcode) {
        case 2:
          DT_CHECK_MSG(opA_precision == "fp32" && opB_precision == "fp32" &&
                           result_precision == "fp16",
                       "Unexpected input/result precisions for FCVT mode 2");
          break;
        case 3:
          DT_CHECK_MSG(opA_precision == "fp32" && opB_precision == "fp32" &&
                           result_precision == "fp8",
                       "Unexpected input/result precisions for FCVT mode 3");
          break;
        case 7:
          DT_CHECK_MSG(dccExtContext().getArch() >= SEN1P5_ISA,
                       "FCVT mode 7 only supported in Sentient1p5");
          DT_CHECK_MSG(opA_precision == "fp32" && opB_precision == "fp32" &&
                           result_precision == "bf16",
                       "Unexpected input/result precisions for FCVT mode 7");
          break;
        default:
          binary_op->emitError("Unsupported FCVT instruction");
          signalPassFailure();
      }
    } else if (is_any_of(binary_op.getBinaryOp(),
                         SentientBinaryOperator::fcmp_eq,
                         SentientBinaryOperator::fcmp_neq,
                         SentientBinaryOperator::fcmp_le,
                         SentientBinaryOperator::fcmp_lt)) {
      DT_CHECK_MSG(opA_precision == compute_precision &&
                       opB_precision == compute_precision,
                   "Input on the fly conversion not supported by FCMP");
      DT_CHECK_MSG(compute_precision == result_precision,
                   "Output on the fly conversion not supported by FCMP");
      super_instr.setInstn(OpCodeT::FCMP);
      is_fcmp_or_fminmax = true;
      if (binary_op.getBinaryOp() == SentientBinaryOperator::fcmp_neq) {
        second_opcode = 4;
      } else if (binary_op.getBinaryOp() == SentientBinaryOperator::fcmp_eq) {
        second_opcode = 5;
      } else if (binary_op.getBinaryOp() == SentientBinaryOperator::fcmp_lt) {
        second_opcode = 6;
      } else if (binary_op.getBinaryOp() == SentientBinaryOperator::fcmp_le) {
        second_opcode = 7;
      } else {
        binary_op->emitError("Unsupported FCMP variant");
        signalPassFailure();
      }
    } else {
      binary_op->emitError("Unsupported operation in PE/SFP");
      signalPassFailure();
    }

    if (auto logical_result_forwarding =
            binary_op.getLogicalResultForwarding()) {
      if (logical_result_forwarding.has_value()) {
        DT_CHECK_MSG(is_fcmp_or_fminmax,
                     "LogicalResultForwarding valid only for FCMP/FMINMAX");
        auto output_port = logical_result_forwarding.value();
        if (output_port == SentientComputePort::istate0) {
          second_opcode += (0 << 3);
        } else if (output_port == SentientComputePort::istate1) {
          second_opcode += (1 << 3);
        } else if (output_port == SentientComputePort::istate2) {
          second_opcode += (2 << 3);
        } else if (output_port == SentientComputePort::istate3) {
          second_opcode += (3 << 3);
        } else {
          second_opcode += (7 << 3);
        }
      }
    }

    super_instr.setCommonField("imm", OperandAttr(second_opcode));
  }

  // Set FPUOP if appropriate
  if (compute_precision != "none") {
    if (instr_has_fpuop_field) {
      if (compute_precision != result_precision) {
        DT_CHECK_MSG(is_any_of(comp, PE, SFP),
                     "On the fly conversions only allowed in PE/SFP units");
        DT_CHECK(result_precision != "none");
        // fp16 really means dlfp16 in Sen1p5
        std::string fpuop = (result_precision == "fp16" &&
                             dccExtContext().getArch() >= SEN1P5_ISA)
                                ? "dlfp16"
                                : result_precision;
        super_instr.setCommonField(
            "fpuop", OperandAttr(fpuop, OperandAttr::Type::DESCRIPTIVE));
      }
    } else if (!is_any_of(super_instr.getInstn(), OpCodeT::GCVT,
                          OpCodeT::FCVT)) {
      DT_CHECK_MSG(compute_precision == result_precision,
                   "Fpuop result conversion unsupported");
    }
  }

  if (dccExtContext().getArch() <= RCUDD1A_ISA && comp == PE) {
    DT_CHECK_MSG(compute_precision != "fp32",
                 "fp32 precision only valid in SFP in DD2");
  } else {
    // Set mode bit
    // MERGE/PACK/LOGICAL mode is always FP16
    std::string mode =
        (compute_precision == "none") ? "fp16" : compute_precision;
    DT_CHECK_MSG((mode == "fp16" || mode == "fp32"),
                 "Expecting fp16 or fp32 precision in SFP or Sen1p5 PE");
    if (super_instr.getInstn() != OpCodeT::LOGICAL) {  // LOGICAL has no mode
      super_instr.setCommonField(
          "mode", OperandAttr(mode, OperandAttr::Type::DESCRIPTIVE));
    }
  }

  std::string opA_hw = "src0", opB_hw;

  opA_hw = "src" + std::to_string(binary_op.getOpAPortID());
  opB_hw = "src" + std::to_string(binary_op.getOpBPortID());

  /*if (super_instr.getInstn().find("FMA") != std::string::npos ||
      super_instr.getInstn().find("IMA") != std::string::npos) {
    std::string opOther_hw;
    if (binary_op.getBinaryOp() == "mul") {
      if (comp == PT) {
        opB_hw = "src2";
        opOther_hw = "src1";
      } else {
        opB_hw = "src1";
        opOther_hw = "src2";
      }
      super_instr.setCommonField(opOther_hw,
          OperandAttr("0.0", OperandAttr::Type::DESCRIPTIVE));
    } else {  // add
      if (comp == PT) {
        opB_hw = "src1";
        opOther_hw = "src2";
      } else {
        opB_hw = "src2";
        opOther_hw = "src1";
      }
      super_instr.setCommonField(opOther_hw,
          OperandAttr("1.0", OperandAttr::Type::DESCRIPTIVE));
    }
    // try to correct assignment in case of restrictions on what a src supports
    if (comp == PT) {
      if (binary_op.getOpA().contains("west") ||
binary_op.getOpA().contains("one") || binary_op.getOpB().contains("rf") ||
binary_op.getOpB().contains("zero")) { std::swap(opA_hw, opB_hw);
      }
    } else {  // PE/SFP
      if (binary_op.getOpA() == "zero" || binary_op.getOpA() == "one" ||
          binary_op.getOpB() == "ptint16") {
        std::swap(opA_hw, opB_hw);
      }
    }
  } else {  // actual binary ops
    if (binary_op.getOpA() == "zero") {
      opA_hw = "src2";
      opB_hw = "src0";
    } else {
      opA_hw = "src0";
      opB_hw = "src2";
    }
}*/

  auto opA_port = stringifySentientComputePort(binary_op.getOpA());
  auto opB_port = stringifySentientComputePort(binary_op.getOpB());
  setSentientComputeInputProgIROperands(
      binary_op, comp, super_instr, opA_port, opA_hw, opA_precision,
      compute_precision, result_precision, binary_op.getOpAForwarding().empty(),
      is_gcvt_or_fcvt);
  setSentientComputeInputProgIROperands(
      binary_op, comp, super_instr, opB_port, opB_hw, opB_precision,
      compute_precision, result_precision, binary_op.getOpBForwarding().empty(),
      is_gcvt_or_fcvt);

  setSentientComputeOutputProgIROperands(binary_op, comp, super_instr,
                                         binary_op.getOpAForwarding(), opA_hw,
                                         compute_precision, result_precision);
  setSentientComputeOutputProgIROperands(binary_op, comp, super_instr,
                                         binary_op.getOpBForwarding(), opB_hw,
                                         compute_precision, result_precision);
  setSentientComputeOutputProgIROperands(
      binary_op, comp, super_instr, binary_op.getResultForwarding(), "result",
      compute_precision, result_precision);

  if (!super_instr.hasCommonField("tgtrf")) {
    super_instr.setCommonField(
        "tgtrf", OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
  }

  super_instr.setCommonField(
      "unroll",
      OperandAttr(
          stringifySentientUnrollFactor(binary_op.getUnrollFactor()).str(),
          OperandAttr::Type::DESCRIPTIVE));

  bool unroll_illegal = false;
  if (comp == PT && opA_hw == "src2") {
    unroll_illegal = binary_op.getUnrollIncrOpA();
  } else {
    super_instr.setCommonField("unrlfld" + opA_hw,
                               OperandAttr(binary_op.getUnrollIncrOpA()));
  }

  if (comp == PT && opB_hw == "src2") {
    unroll_illegal = binary_op.getUnrollIncrOpB();
  } else {
    super_instr.setCommonField("unrlfld" + opB_hw,
                               OperandAttr(binary_op.getUnrollIncrOpB()));
  }

  super_instr.setCommonField("unrlfldtgt",
                             OperandAttr(binary_op.getUnrollIncrResult()));

  if (dccExtContext().getArch() >= SEN1P5_ISA && comp == PE &&
      super_instr.getInstn() == OpCodeT::LOGICAL)
    // Reason: In Sen1p5, data received from PT is int24/fp24. But LOGICAL ops
    // have no on the fly conversions so cannot upcast to compute precision
    // fp32.
    DT_CHECK_MSG(!opA_port.contains("pt") && !opB_port.contains("pt"),
                 "Cannot have LOGICAL on pt input operands in Sen1p5");

  if (is_any_of(comp, PE, SFP) && dccExtContext().getArch() >= SEN1P5_ISA) {
    // In Sentient 1.5, unrlfldsrc1 controls the movement of the internal state
    // register.
    auto unroll_incr_logical_res = binary_op.getUnrollIncrLogicalResult();
    if (unroll_incr_logical_res)
      super_instr.setCommonField("unrlfldsrc1",
                                 OperandAttr(unroll_incr_logical_res));
  }

  if (unroll_illegal) {
    binary_op->emitError(
        "PT FMA/IMA does not support setting unrlfldsrc2, but it was "
        "requested");
    signalPassFailure();
  }

  if (comp != PT) {
    if (auto mask_constant = dyn_cast<sentient::ConstantOp>(
            binary_op.getMask().getDefiningOp())) {
      super_instr.setCommonField("mask",
                                 OperandAttr(255 - mask_constant.getValue()));
    } else {
      binary_op->emitError("Mask constant value has to exist.");
      signalPassFailure();
    }
  }

  if (dccExtContext().getArch() >= SEN1P5_ISA)
    dcc::sentient::utils::setFCValueFromFoldMode(super_instr, binary_op, comp);

  return super_instr;
}

// ---- 87/130  ConstructUnaryInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2201  (237L)
UniformInstrInfo e087_ConstructUnaryInstr(
    sentient::UnaryOp unary_op, const SenComponents& comp) {
  UniformInstrInfo super_instr;
  std::string opA_precision =
      stringifySentientPrecision(unary_op.getOpAPrecision()).str();
  std::string compute_precision =
      stringifySentientPrecision(unary_op.getComputePrecision()).str();
  std::string result_precision =
      stringifySentientPrecision(unary_op.getResultPrecision()).str();

  if (unary_op.getDbgName().has_value())
    super_instr.setCommonComment(unary_op.getDbgName().value().str());

  std::map<std::string, int> func_est_map{
      {"exp_a", 0},       {"exp_b", 1},      {"rec", 2},
      {"ln", 3},          {"rsqrt", 5},      {"sigm_slope", 6},
      {"sigm_offset", 7}, {"tanh_slope", 8}, {"tanh_offset", 9}};

  std::map<std::string, int> reduction_map{{"reduction_add", 1},
                                           {"reduction_max", 8},
                                           {"reduction_abs_max", 10},
                                           {"reduction_min", 12},
                                           {"reduction_abs_min", 14}};

  std::map<std::string, int> icvt_map{{"fast_exp", 7}};

  auto func = stringifySentientUnary(unary_op.getUnaryOp());
  std::string func_str = func.str();
  bool isEstimate = func_est_map.find(func_str) != func_est_map.end();
  bool isICVT = icvt_map.find(func_str) != icvt_map.end();
  bool isFloor = (func_str == "floor");
  bool isGCVT = func.contains("gcvt");
  bool isFCVT = func.contains("fcvt");
  bool isReduction = reduction_map.find(func_str) != reduction_map.end();

  if (isEstimate || isICVT || isFloor || isGCVT || isFCVT) {
    if (dccExtContext().getArch() <= RCUDD1A_ISA && comp == PE) {
      DT_CHECK_MSG(compute_precision != "fp32",
                   "fp32 precision only valid in SFP in DD2");
    } else {
      // Set mode bit
      DT_CHECK_MSG((compute_precision == "fp16" || compute_precision == "fp32"),
                   "Expecting fp16 or fp32 precision in SFP or Sen1p5 PE");
      super_instr.setCommonField(
          "mode",
          OperandAttr(compute_precision, OperandAttr::Type::DESCRIPTIVE));
    }

    if (isEstimate) {
      DT_CHECK_MSG(opA_precision == compute_precision,
                   "FEST does not support input on the fly conversions");
      DT_CHECK_MSG(result_precision == compute_precision,
                   "FEST does not support output on the fly conversions");
      super_instr.setInstn(OpCodeT::FEST);
      super_instr.setCommonField(
          "imm",
          OperandAttr(func_est_map[stringifySentientUnary(unary_op.getUnaryOp())
                                       .str()]));
      super_instr.setCommonField(
          "src2", OperandAttr("0.0", OperandAttr::Type::DESCRIPTIVE));
    } else if (isGCVT) {
      DT_CHECK_MSG(is_any_of(compute_precision, "fp16", "int16"),
                   "GCVT instruction precision expected to be fp16/int16");
      super_instr.setInstn(OpCodeT::GCVT);
      std::string gcvt_no = func_str.substr(8, func_str.size() - 8);
      int gcvt_mode = stoi(gcvt_no);
      switch (gcvt_mode) {
        case 1:
        case 2:
        case 5:
        case 6:
          DT_CHECK_MSG(
              opA_precision == "fp8" && result_precision == "fp16",
              "Unexpected input/result precisions for GCVT mode 1/2/5/6");
          break;
        case 8:
          DT_CHECK_MSG(opA_precision == "int4" && result_precision == "int16",
                       "Unexpected input/result precisions for GCVT mode 8");
          break;
        case 16:
          DT_CHECK_MSG(opA_precision == "fp16" && result_precision == "bf16",
                       "Unexpected input/result precisions for GCVT mode 16");
          break;
        case 17:
          DT_CHECK_MSG(opA_precision == "bf16" && result_precision == "fp16",
                       "Unexpected input/result precisions for GCVT mode 17");
          break;
        default:
          unary_op->emitError("Unsupported GCVT instruction");
          signalPassFailure();
      }
      super_instr.setCommonField("imm", gcvt_mode);
      super_instr.setCommonField(
          "src2", OperandAttr("0.0", OperandAttr::Type::DESCRIPTIVE));
    } else if (isFCVT) {
      DT_CHECK_MSG(comp == SFP ||
                       (comp == PE && dccExtContext().getArch() >= SEN1P5_ISA),
                   "FCVT instructions only supported in SFP or in Sen1p5 PE");
      super_instr.setInstn(OpCodeT::FCVT);
      std::string fcvt_no = func_str.substr(8, func_str.size() - 8);
      int fcvt_mode = stoi(fcvt_no);
      switch (fcvt_mode) {
        case 0:
        case 1:
          DT_CHECK_MSG(compute_precision == "fp32",
                       "FCVT imm0/1 compute precision expected to be fp32");
          DT_CHECK_MSG(result_precision == "fp32",
                       "FCVT imm0/1 result precision expected to be fp32");
          DT_CHECK_MSG((opA_precision == "fp16" &&
                        dccExtContext().getArch() <= RCUDD1A_ISA) ||
                           opA_precision == "fp16",
                       "Unexpected input precision for FCVT imm0/1");
          break;
        case 5:
        case 6:
          DT_CHECK_MSG(dccExtContext().getArch() >= SEN1P5_ISA,
                       "FCVT mode 5/6 only supported in Sentient1p5");
          DT_CHECK_MSG(compute_precision == "fp32",
                       "FCVT imm5/6 compute precision expected to be fp32");
          DT_CHECK_MSG(result_precision == "fp32",
                       "FCVT imm5/6 result precision expected to be fp32");
          DT_CHECK_MSG(opA_precision == "bf16",
                       "Unexpected input precision for FCVT imm5/6");
          break;
        default:
          unary_op->emitError("Unsupported FCVT instruction");
          signalPassFailure();
      }

      super_instr.setCommonField("imm", fcvt_mode);
    } else if (isFloor) {
      // Floor operation
      DT_CHECK_MSG(opA_precision == compute_precision,
                   "ICVT Floor does not support input on the fly conversions");
      DT_CHECK_MSG(result_precision == compute_precision,
                   "ICVT Floor does not support output on the fly conversions");
      super_instr.setInstn(OpCodeT::ICVT);
      int64_t floor_imm = (opA_precision == "fp16") ? 1 : 9;
      super_instr.setCommonField("imm", OperandAttr(floor_imm));
      super_instr.setCommonField(
          "src0",
          OperandAttr("icvtconst", OperandAttr::Type::DESCRIPTIVE));  // 0x0f
    } else {
      // isICVT == true (fast_exp)
      DT_CHECK_MSG(
          opA_precision == compute_precision,
          "ICVT Fast Exp does not support input on the fly conversions");
      DT_CHECK_MSG(
          result_precision == compute_precision,
          "ICVT Fast Exp does not support output on the fly conversions");
      super_instr.setInstn(OpCodeT::ICVT);
      super_instr.setCommonField(
          "imm",
          OperandAttr(
              icvt_map[stringifySentientUnary(unary_op.getUnaryOp()).str()]));
      super_instr.setCommonField(
          "src0",
          OperandAttr("icvtconst", OperandAttr::Type::DESCRIPTIVE));  // 0x0f
    }
    std::string opA_hw = "src" + std::to_string(unary_op.getOpAPortID());
    setSentientComputeInputProgIROperands(
        unary_op, comp, super_instr,
        stringifySentientComputePort(unary_op.getOpA()), opA_hw, opA_precision,
        compute_precision, result_precision,
        !unary_op.getOpAForwarding().empty(), (isGCVT || isFCVT));
    setSentientComputeOutputProgIROperands(unary_op, comp, super_instr,
                                           unary_op.getOpAForwarding(), opA_hw,
                                           compute_precision, result_precision);
    setSentientComputeOutputProgIROperands(
        unary_op, comp, super_instr, unary_op.getResultForwarding(), "result",
        compute_precision, result_precision);

    if (!super_instr.hasCommonField("tgtrf")) {
      super_instr.setCommonField(
          "tgtrf", OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
    }
    super_instr.setCommonField(
        "unroll",
        OperandAttr(
            stringifySentientUnrollFactor(unary_op.getUnrollFactor()).str(),
            OperandAttr::Type::DESCRIPTIVE));

    super_instr.setCommonField("unrlfld" + opA_hw,
                               OperandAttr(unary_op.getUnrollIncrOpA()));
    super_instr.setCommonField("unrlfldtgt",
                               OperandAttr(unary_op.getUnrollIncrResult()));

    if (auto mask_constant = dyn_cast<sentient::ConstantOp>(
            unary_op.getMask().getDefiningOp())) {
      super_instr.setCommonField("mask",
                                 OperandAttr(255 - mask_constant.getValue()));
    } else {
      unary_op->emitError("Mask constant value has to exist.");
      signalPassFailure();
    }
  } else if (isReduction) {
    DT_CHECK_MSG(opA_precision == compute_precision,
                 "REDUCE does not support input on the fly conversions");
    DT_CHECK_MSG(result_precision == compute_precision,
                 "REDUCE does not support output on the fly conversions");
    if (dccExtContext().getArch() <= RCUDD1A_ISA && comp == PE) {
      DT_CHECK_MSG(compute_precision != "fp32",
                   "fp32 precision only valid in SFP in DD2");
    } else {
      // Set mode bit
      DT_CHECK_MSG((compute_precision == "fp16" || compute_precision == "fp32"),
                   "Expecting fp16 or fp32 precision in SFP or Sen1p5 PE");
      super_instr.setCommonField(
          "mode",
          OperandAttr(compute_precision, OperandAttr::Type::DESCRIPTIVE));
    }
    super_instr.setInstn(OpCodeT::REDUCE);
    std::string reg_value =
        "R" + stringifySentientComputePort(unary_op.getOpA()).substr(3).str();
    super_instr.setCommonField(
        "rsm_src0", OperandAttr(reg_value, OperandAttr::Type::DESCRIPTIVE));
    super_instr.setCommonField(
        "rsm_src1",
        OperandAttr(reduction_map[stringifySentientUnary(unary_op.getUnaryOp())
                                      .str()]));
    super_instr.setCommonField(
        "rsm_tgtrf", OperandAttr(reg_value, OperandAttr::Type::DESCRIPTIVE));
    super_instr.setCommonField(
        "rsm_unroll",
        OperandAttr(
            stringifySentientUnrollFactor(unary_op.getUnrollFactor()).str(),
            OperandAttr::Type::DESCRIPTIVE));
  } else {
    unary_op->emitError("Unknown unary function.");
    signalPassFailure();
  }

  if (dccExtContext().getArch() >= SEN1P5_ISA)
    dcc::sentient::utils::setFCValueFromFoldMode(super_instr, unary_op, comp);

  return super_instr;
}

// ---- 88/130  ConstructTernaryInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:2439  (122L)
UniformInstrInfo e088_ConstructTernaryInstr(
    sentient::TernaryOp ternary_op, const SenComponents& comp) {
  UniformInstrInfo super_instr;
  std::string opA_precision =
      stringifySentientPrecision(ternary_op.getOpAPrecision()).str();
  std::string opB_precision =
      stringifySentientPrecision(ternary_op.getOpBPrecision()).str();
  std::string opC_precision =
      stringifySentientPrecision(ternary_op.getOpCPrecision()).str();
  std::string compute_precision =
      stringifySentientPrecision(ternary_op.getComputePrecision()).str();
  std::string result_precision =
      stringifySentientPrecision(ternary_op.getResultPrecision()).str();
  DT_CHECK_MSG(
      opA_precision == "int1" && opB_precision == compute_precision &&
          opC_precision == compute_precision,
      "Ternary instructions do not support input on the fly conversions");
  DT_CHECK_MSG(
      compute_precision == result_precision,
      "Ternary instructions do not support output on the fly conversions");
  if (ternary_op.getDbgName().has_value())
    super_instr.setCommonComment(ternary_op.getDbgName().value().str());

  if (dccExtContext().getArch() <= RCUDD1A_ISA && comp == PE) {
    DT_CHECK_MSG(compute_precision != "fp32",
                 "fp32 precision only valid in SFP in DD2");
  } else {
    // Set mode bit
    DT_CHECK_MSG((compute_precision == "fp16" || compute_precision == "fp32"),
                 "Expecting fp16 or fp32 precision in SFP or Sen1p5 PE");
    super_instr.setCommonField(
        "mode", OperandAttr(compute_precision, OperandAttr::Type::DESCRIPTIVE));
  }

  DT_CHECK_MSG(is_any_of(comp, PE, SFP),
               "Ternary operation such as SELECT is available only in PE/SFP");
  DT_CHECK_MSG(
      ternary_op.getTernaryOp() == SentientTernaryOperator::select,
      "Current, we support ternary operations for only select operations");

  super_instr.setInstn(OpCodeT::SELECT);

  int opcode = 0;
  if (dccExtContext().getArch() >= IsaCoreGen::SEN1P5_ISA) {
    auto state_reg_port = ternary_op.getOpA();
    if (state_reg_port == SentientComputePort::istate0) {
      opcode += (0 << 3);
    } else if (state_reg_port == SentientComputePort::istate1) {
      opcode += (1 << 3);
    } else if (state_reg_port == SentientComputePort::istate2) {
      opcode += (2 << 3);
    } else if (state_reg_port == SentientComputePort::istate3) {
      opcode += (3 << 3);
    }
  }

  super_instr.setCommonField("imm", OperandAttr(opcode));

  std::string opB_hw = "src" + std::to_string(ternary_op.getOpBPortID());
  std::string opC_hw = "src" + std::to_string(ternary_op.getOpCPortID());

  setSentientComputeInputProgIROperands(
      ternary_op, comp, super_instr,
      stringifySentientComputePort(ternary_op.getOpB()), opB_hw, opB_precision,
      compute_precision, result_precision,
      !ternary_op.getOpBForwarding().empty());
  setSentientComputeInputProgIROperands(
      ternary_op, comp, super_instr,
      stringifySentientComputePort(ternary_op.getOpC()), opC_hw, opC_precision,
      compute_precision, result_precision,
      !ternary_op.getOpCForwarding().empty());

  setSentientComputeOutputProgIROperands(ternary_op, comp, super_instr,
                                         ternary_op.getOpBForwarding(), opB_hw,
                                         compute_precision, result_precision);
  setSentientComputeOutputProgIROperands(ternary_op, comp, super_instr,
                                         ternary_op.getOpCForwarding(), opC_hw,
                                         compute_precision, result_precision);
  setSentientComputeOutputProgIROperands(
      ternary_op, comp, super_instr, ternary_op.getResultForwarding(), "result",
      compute_precision, result_precision);

  if (!super_instr.hasCommonField("tgtrf")) {
    super_instr.setCommonField(
        "tgtrf", OperandAttr("no", OperandAttr::Type::DESCRIPTIVE));
  }

  super_instr.setCommonField(
      "unroll",
      OperandAttr(
          stringifySentientUnrollFactor(ternary_op.getUnrollFactor()).str(),
          OperandAttr::Type::DESCRIPTIVE));

  if (is_any_of(comp, PE, SFP) && dccExtContext().getArch() >= SEN1P5_ISA) {
    // In Sentient 1.5, unrlfldsrc1 controls the movement of the internal state
    // register.
    auto unroll_incr_logical_res = ternary_op.getUnrollIncrLogicalResult();
    if (unroll_incr_logical_res)
      super_instr.setCommonField("unrlfldsrc1",
                                 OperandAttr(unroll_incr_logical_res));
  }

  super_instr.setCommonField("unrlfld" + opB_hw,
                             OperandAttr(ternary_op.getUnrollIncrOpA()));

  super_instr.setCommonField("unrlfld" + opC_hw,
                             OperandAttr(ternary_op.getUnrollIncrOpB()));

  super_instr.setCommonField("unrlfldtgt",
                             OperandAttr(ternary_op.getUnrollIncrResult()));

  if (auto mask_constant = dyn_cast<sentient::ConstantOp>(
          ternary_op.getMask().getDefiningOp())) {
    super_instr.setCommonField("mask",
                               OperandAttr(255 - mask_constant.getValue()));
  } else {
    ternary_op->emitError("Mask constant value has to exist.");
    signalPassFailure();
  }

  return super_instr;
}

// ---- 89/130  ConstructLoadInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3024  (201L)
UniformInstrInfo e089_ConstructLoadInstr(
    dataflow::ProgramUnitOp unit_op, const SenComponents& comp,
    sentient::LoadAndSendOp load_op,
    std::optional<CodeQualityStats>& cq_stats) {
  if (is_any_of(comp, L3LU, L3SU))
    return ConstructL3LoadInstr(comp, load_op, cq_stats);

  int mutable_reg_index = getValueRegIndex(load_op.getMutableAddr());
  int immutable_reg_index = getValueRegIndex(load_op.getImmutableAddr());
  int result_reg_index = getValueRegIndex(load_op.getResult());

  if (dccExtContext().getProgStitch() || forceFullRegInit.getValue() ||
      cq_stats.has_value())
    addToRegsToInit(load_op,
                    {getValueRegLocaleAsString(load_op.getMutableAddr()),
                     getValueRegLocaleAsString(load_op.getImmutableAddr()),
                     getValueRegLocaleAsString(load_op.getResult())},
                    {mutable_reg_index, immutable_reg_index, result_reg_index});

  // ldtype setting is dependent on unit and shuffle_mode. If shuffle mode
  // doesn't match a specific ldtype pattern, calculate ldtype.
  //

  // load type Checks for L0LU/LXLU
  int load_type = load_op.getElementSize() * load_op.getTotalElements() / 8;
  std::string ldtype = std::to_string(load_type) + "b";
  if (comp == L0LU) {
    switch (dccExtContext().getArch()) {
      case IsaCoreGen::SEN1P5_ISA:
        DT_CHECK(is_any_of(load_type, 8, 16, 32, 64));
        break;
      default:  // DD1 and DD2
        DT_CHECK(is_any_of(load_type, 2, 4, 8, 16));
        break;
    }
  } else if (comp == LXLU) {
    auto shuffle_mode = load_op.getShuffleMode();
    switch (shuffle_mode) {
      case SentientShuffleMode::splat2b:
        ldtype = "2bsplat";
        load_type = 2;
        break;
      case SentientShuffleMode::zpad16b:
        ldtype = "16bzpad";
        load_type = 16;
        break;
      case SentientShuffleMode::splat16b:
        ldtype = "16bsplat";
        load_type = 16;
        break;
      default:
        DT_CHECK_MSG(is_any_of(load_type, 2, 16, 128),
                     "Invalid ldtype for LX unit!");
        break;
    }
  }

  UniformInstrInfo super_instr;
  if (load_op.getDbgName().has_value())
    super_instr.setCommonComment(load_op.getDbgName().value().str());
  std::string instr_name;
  bool is_immutable_const =
      !mlir::isa<BlockArgument>(load_op.getImmutableAddr()) &&
      isa<sentient::ConstantOp>(load_op.getImmutableAddr().getDefiningOp());

  // If the program unit unit/unit iterator is used as the consumer, this is a
  // self load.
  auto consumer = load_op.getConsumer();
  DT_CHECK(consumer.getDefiningOp());
  auto consumer_unit = dcc::getUnitType(consumer.getDefiningOp());
  Value program_unit;
  if (unit_op.getRegion().getArguments().empty())
    program_unit = unit_op.getUnits()[0];
  else
    program_unit = unit_op.getRegion().getArguments()[0];
  bool is_lxlu_self_load = comp == LXLU && consumer_unit == LXLUSCALEREG;

  DT_CHECK(load_op.getChunkSize() * load_op.getElementSize() / 8 <= load_type);
  if (load_op.getBurstSize() > 1 || load_op.getInterleavedGroup() > 0 ||
      (is_any_of(comp, L0LU, L0SU) &&
       load_op.getChunkSize() != load_op.getTotalElements()) ||
      !is_immutable_const) {
    DT_CHECK(!is_lxlu_self_load);
    instr_name += "LDST";
    super_instr.setCommonField("src1", immutable_reg_index);
  } else {
    if (!is_lxlu_self_load) {
      instr_name += "LDSTI";
    } else {
      DT_CHECK(dccExtContext().getArch() >= IsaCoreGen::SEN1P5_ISA);
      DT_CHECK_MSG(load_type == 128,
                   "ldtype mode 00 only supported in LDCVTI self load");
      instr_name += "LDCVTI";
      // element_index and scale_index aren't used so set to 0.
      super_instr.setCommonField("elemidx", 0);
      super_instr.setCommonField("scaleidx", 0);
    }

    int imm_val =
        cast<sentient::ConstantOp>(load_op.getImmutableAddr().getDefiningOp())
            .getValue() *
        (float)load_op.getElementSize() / 8.0 /
        (float)GetAddressScale(comp, load_op);
    super_instr.setCommonField("imm", imm_val);
  }

  if (dcc::utils::isUpdateMode(load_op)) instr_name += "U";

  super_instr.setInstn(Isa::to_instopcode(instr_name));
  super_instr.setCommonField("src0", result_reg_index);
  super_instr.setCommonField(
      "ldtype", OperandAttr(ldtype, OperandAttr::Type::DESCRIPTIVE));
  if (load_op.getBurstSize() > 1) {
    super_instr.setCommonField("burst", 1);
    super_instr.setCommonField(
        "burstsize", normalizeBurstSize(load_op.getBurstSize(), comp));
  }
  if (int(load_op.getInterleavedGroup()) > 0) {
    int il_groups = int(load_op.getInterleavedGroup());
    // TODO: consider unrolling of interleaved groups in the earlier passes
    //  if it doesn't fit into ISA constraints similar to burst sizes.
    DT_CHECK_MSG(
        !(il_groups == 3 || il_groups > 4),
        "Sentient ISA doesn't support this interleaving groups factor");
    if (il_groups == 4)
      il_groups = 3;  // ISA encoding of 4 groups is 3 (bits:11)
    super_instr.setCommonField("group", int(il_groups));
  }
  if (is_any_of(comp, L0LU, L0SU)) {
    if (load_op.getShuffleMode() == SentientShuffleMode::splat) {
      super_instr.setCommonField(
          "splat", OperandAttr("splat", OperandAttr::Type::DESCRIPTIVE));
    }

    // TODO need to create a new function for LDSTU
    if (load_op.getChunkSize() != load_op.getTotalElements()) {
      int chunk_size_type =
          load_op.getElementSize() * load_op.getChunkSize() / 8;
      int chunk_stride_type =
          load_op.getElementSize() * load_op.getChunkStride() / 8;
      switch (dccExtContext().getArch()) {
        case IsaCoreGen::SEN1P5_ISA:
          DT_CHECK_MSG(is_any_of(chunk_size_type, 8, 16, 32, 64),
                       "Invalid chunk size for L0LU");
          DT_CHECK_MSG(
              is_any_of(chunk_stride_type, 8, 16, 32, 64, 96, 128, 192, 256),
              "Invalid chunk stride for L0LU");
          break;
        default:  // DD1 and DD2
          DT_CHECK_MSG(is_any_of(chunk_size_type, 2, 4),
                       "Invalid chunk size for L0LU");
          DT_CHECK_MSG(is_any_of(chunk_stride_type, 4, 8, 16, 32, 64),
                       "Invalid chunk stride for L0LU");
          break;
      }

      super_instr.setCommonField(
          "chunksize", OperandAttr(std::to_string(chunk_size_type) + "b",
                                   OperandAttr::Type::DESCRIPTIVE));
      super_instr.setCommonField(
          "chunkstride", OperandAttr(std::to_string(chunk_stride_type) + "b",
                                     OperandAttr::Type::DESCRIPTIVE));
    }
  } else if (is_any_of(comp, LXLU, LXSU)) {
    if (load_op.getRotateVal().has_value()) {
      DT_CHECK(!is_lxlu_self_load);
      // rotation value is in terms of number of elements.
      //  we need to convert to bytes.
      int rot_val = (load_op.getRotateVal().value() * load_op.getElementSize());
      DT_CHECK_MSG(rot_val % 128 == 0,
                   "rotation size should be a multiple of 16 Bytes");
      super_instr.setCommonField("rottype", rot_val / 128);
    }

    if (is_lxlu_self_load) {
      super_instr.setCommonField(
          "consumertag", OperandAttr("self", OperandAttr::Type::DESCRIPTIVE));
    } else if (auto load_op_get_unit = dyn_cast_or_null<dataflow::GetUnitOp>(
                   load_op.getConsumer().getDefiningOp())) {
      auto consumer_unit_generic = EnumsConversion::senCompToGenericComp.at(
          EnumsConversion::stringToSenComponents.at(
              load_op_get_unit.getType().str()));
      auto consumer_unit_name =
          EnumsConversion::senComponentsToString.at(consumer_unit_generic);
      super_instr.setCommonField(
          "consumertag",
          getProperConsumer(dccExtContext(), consumer_unit_generic,
                            consumer_unit_name));
    } else if (auto query_op = dyn_cast_or_null<uniform::QueryMapOp>(
                   load_op.getConsumer().getDefiningOp())) {
      OperandMap operand_map(unit_foldid_map_, dccExtContext(), "consumertag",
                             query_op, OperandMap::Mode::unit_name);
      super_instr.setOperandMap(operand_map);
    }
  } else {
    load_op->emitError("Unable to lower the load_op");
    signalPassFailure();
  }

  return super_instr;
}

// ---- 90/130  ConstructLoadInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3226  (60L)
UniformInstrInfo e090_ConstructLoadInstr(
    const SenComponents& comp, sentient::LoadAndExtractScalarOp extract_op,
    std::optional<CodeQualityStats>& cq_stats) {
  DT_CHECK_MSG(
      comp == LXLU,
      "load instructions constructed from load_and_extract_scalar ops are "
      "only supported in LXLU");

  if (dccExtContext().getProgStitch() || forceFullRegInit.getValue() ||
      cq_stats.has_value())
    addToRegsToInit(extract_op,
                    {getValueRegLocaleAsString(extract_op.getMutableAddr()),
                     getValueRegLocaleAsString(extract_op.getImmutableAddr())},
                    {getValueRegIndex(extract_op.getMutableAddr()),
                     getValueRegIndex(extract_op.getImmutableAddr())});

  int load_type =
      extract_op.getElementSize() * extract_op.getTotalElements() / 8;
  DT_CHECK_MSG(load_type == 128,
               "Invalid ldtype from load_and_extract_scalar op");
  std::string ldtype = std::to_string(load_type) + "b";

  UniformInstrInfo super_instr;
  if (extract_op.getDbgName().has_value())
    super_instr.setCommonComment(extract_op.getDbgName().value().str());
  std::string instr_name;
  if (dcc::utils::isConstant<sentient::ConstantOp>(
          extract_op.getImmutableAddr())) {
    instr_name += "LDSTI";
    auto immutable_op = extract_op.getImmutableAddr().getDefiningOp();
    if (auto constant_op = dyn_cast<sentient::ConstantOp>(immutable_op)) {
      int imm_val = constant_op.getValue() *
                    (float)extract_op.getElementSize() / 8.0 /
                    (float)GetAddressScale(comp, extract_op.getAddrResult());
      super_instr.setCommonField("imm", imm_val);
    } else if (auto query_op = dyn_cast<uniform::QueryMapOp>(immutable_op)) {
      float scale = (float)extract_op.getElementSize() / 8.0 /
                    (float)GetAddressScale(comp, extract_op.getAddrResult());
      super_instr.setOperandMap(OperandMap(unit_foldid_map_, dccExtContext(),
                                           "imm", query_op,
                                           OperandMap::Mode::none, scale));
    }
  } else {
    instr_name += "LDST";
    super_instr.setCommonField("src1",
                               getValueRegIndex(extract_op.getImmutableAddr()));
  }

  if (dcc::utils::isUpdateMode(extract_op)) instr_name += "U";

  super_instr.setInstn(Isa::to_instopcode(instr_name));
  super_instr.setCommonField("src0",
                             getValueRegIndex(extract_op.getAddrResult()));
  super_instr.setCommonField(
      "ldtype", OperandAttr(ldtype, OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField(
      "consumertag", OperandAttr("self", OperandAttr::Type::DESCRIPTIVE));

  return super_instr;
}

// ---- 91/130  ConstructStoreInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3387  (121L)
UniformInstrInfo e091_ConstructStoreInstr(
    const SenComponents& comp, sentient::ReceiveAndStoreOp store_op,
    std::optional<CodeQualityStats>& cq_stats) {
  if (is_any_of(comp, L3LU, L3SU))
    return ConstructL3StoreInstr(comp, store_op, cq_stats);

  int mutable_reg_index = getValueRegIndex(store_op.getMutableAddr());
  int immutable_reg_index = getValueRegIndex(store_op.getImmutableAddr());
  int result_reg_index = getValueRegIndex(store_op.getResult());

  if (dccExtContext().getProgStitch() || forceFullRegInit.getValue() ||
      cq_stats.has_value())
    addToRegsToInit(store_op,
                    {getValueRegLocaleAsString(store_op.getMutableAddr()),
                     getValueRegLocaleAsString(store_op.getImmutableAddr()),
                     getValueRegLocaleAsString(store_op.getResult())},
                    {mutable_reg_index, immutable_reg_index, result_reg_index});

  UniformInstrInfo super_instr;
  if (store_op.getDbgName().has_value())
    super_instr.setCommonComment(store_op.getDbgName().value().str());
  std::string instr_name;
  bool is_immutable_const =
      !mlir::isa<BlockArgument>(store_op.getImmutableAddr()) &&
      isa<sentient::ConstantOp>(store_op.getImmutableAddr().getDefiningOp());
  if (store_op.getBurstSize() > 1 || store_op.getInterleavedGroup() > 0 ||
      !is_immutable_const) {
    instr_name += "LDST";
    super_instr.setCommonField("src1", immutable_reg_index);
  } else {
    instr_name += "LDSTI";
    int imm_val =
        cast<sentient::ConstantOp>(store_op.getImmutableAddr().getDefiningOp())
            .getValue() *
        (float)store_op.getElementSize() / 8.0 /
        (float)GetAddressScale(comp, store_op);
    super_instr.setCommonField("imm", imm_val);
  }

  if (dcc::utils::isUpdateMode(store_op)) instr_name += "U";

  super_instr.setInstn(Isa::to_instopcode(instr_name));
  super_instr.setCommonField("src0", result_reg_index);
  if (store_op.getBurstSize() > 1) {
    super_instr.setCommonField("burst", 1);
    super_instr.setCommonField(
        "burstsize", normalizeBurstSize(store_op.getBurstSize(), comp));
  }
  if (int(store_op.getInterleavedGroup()) > 0) {
    int il_groups = int(store_op.getInterleavedGroup());
    // TODO: consider unrolling of interleaved groups in the earlier passes
    //  if it doesn't fit into ISA constraints similar to burst sizes.
    DT_CHECK_MSG(
        !(il_groups == 3 || il_groups > 4),
        "Sentient ISA doesn't support this interleaving groups factor");
    if (il_groups == 4)
      il_groups = 3;  // ISA encoding of 4 groups is 3 (bits:11)
    super_instr.setCommonField("group", int(il_groups));
  }
  if (is_any_of(comp, L0LU, L0SU)) {
    if (store_op.getCoalesce()) {
      super_instr.setCommonField("coalesce", 1);
      super_instr.setCommonField("stride", int(store_op.getStride()));
      super_instr.setCommonField("subwordlen",
                                 int(store_op.getSubwordLength()));
      if (store_op.getDropFirst()) {
        super_instr.setCommonField("src2",
                                   getValueRegIndex(store_op.getDropFirst()));
      }
      if (store_op.getPermute()) {
        super_instr.setCommonField("permute", 1);
      }
    }
    bool dst_is_l0_scale = false;
    if (store_op.getDst()) {
      std::optional<std::string> unit_str_optional =
          dcc::uniform::utils::findUnitType(store_op.getDst());
      if (unit_str_optional.has_value() &&
          unit_str_optional.value() == "l0scale")
        dst_is_l0_scale = true;
    }
    if (dccExtContext().getArch() >= SEN1P5_ISA) {
      DT_CHECK_MSG(comp == L0SU, "L0Scale implicit in L0LU");
      std::string mx_type = dst_is_l0_scale ? "scale" : "data";
      super_instr.setCommonField(
          "scale_array", OperandAttr(mx_type, OperandAttr::Type::DESCRIPTIVE));
    }
  } else if (is_any_of(comp, LXLU, LXSU)) {
    int store_size =
        store_op.getTotalElements() * store_op.getElementSize() / 8;
    auto shuffle_mode = store_op.getShuffleMode();
    if (shuffle_mode == SentientShuffleMode::masked2b)
      store_size = 2;
    else if (shuffle_mode == SentientShuffleMode::masked16b)
      store_size = 16;
    DT_CHECK(store_size == 2 || store_size == 16 ||
             store_size == 128 && "Invalid sttype for LX unit!");

    super_instr.setCommonField("sttype",
                               OperandAttr(std::to_string(store_size) + "b",
                                           OperandAttr::Type::DESCRIPTIVE));
    if (auto store_op_get_unit = dyn_cast_or_null<dataflow::GetUnitOp>(
            store_op.getProducer().getDefiningOp())) {
      DT_CHECK_MSG(store_op_get_unit,
                   "Expecting GetUnitOp for receive_and_store producer!");
      super_instr.setCommonField(
          "producertag", OperandAttr(store_op_get_unit.getType().lower(),
                                     OperandAttr::Type::DESCRIPTIVE));
    } else if (auto query_op = dyn_cast_or_null<uniform::QueryMapOp>(
                   store_op.getProducer().getDefiningOp())) {
      OperandMap operand_map(unit_foldid_map_, dccExtContext(), "producertag",
                             query_op, OperandMap::Mode::unit_name);
      super_instr.setOperandMap(operand_map);
    }
  } else {
    store_op->emitError("Unable to lower the store_op");
    signalPassFailure();
  }

  return super_instr;
}

// ---- 92/130  ConstructSplatInstrFromSplatOp  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3719  (65L)
UniformInstrInfo e092_ConstructSplatInstrFromSplatOp(
    UniformInstrInfo super_instr, sentient::LogicalPortOp input_port,
    sentient::LogicalPortOp output_port, sentient::SplatOp op,
    const SenComponents& comp, std::string precision) {
  std::string field_name_out, operand_name_out;
  StringRef output = stringifySentientComputePort(output_port.getPortName());
  StringRef input = stringifySentientComputePort(input_port.getPortName());
  int output_reg_val, input_reg_val;
  if (output.contains("lrf")) {
    field_name_out = "tgtrf";
    output_reg_val =
        std::stoi(stringifySentientComputePort(output_port.getPortName())
                      .str()
                      .substr(3));
    int max_reg = dccExtContext()
                      .dsc_global_->sysDef.regInfoPerUnit.at(comp)
                      .at(LRF)
                      .maxNum;
    DT_CHECK_MSG(output_reg_val >= 0 && output_reg_val < max_reg,
                 "invalid register number");
    operand_name_out = "R" + std::to_string(output_reg_val);
  } else {
    field_name_out = "tgt" + output.str();
    operand_name_out = "result";
  }

  DT_CHECK((op.getPad() == SentientSplatPad::none ^
            op.getPad() == SentientSplatPad::left) &&
           "unexpected pad string");
  int padding = op.getPad() == SentientSplatPad::none ? 1 : 0;

  super_instr.setInstn(OpCodeT::SPLAT);

  if (comp == SFP || (comp == PE && dccExtContext().getArch() >= SEN1P5_ISA)) {
    DT_CHECK_MSG((precision == "fp16" || precision == "fp32"),
                 "Expecting fp16 or fp32 precision for splat");
    super_instr.setCommonField(
        "mode", OperandAttr(precision, OperandAttr::Type::DESCRIPTIVE));
  }

  auto mask_constant =
      dyn_cast<sentient::ConstantOp>(op.getMask().getDefiningOp());
  DT_CHECK(mask_constant);
  super_instr.setCommonField("mask",
                             OperandAttr(255 - mask_constant.getValue()));
  std::string opA_hw = "src0";
  setSentientComputeInputProgIROperands(
      op, comp, super_instr,
      stringifySentientComputePort(input_port.getPortName()), opA_hw, precision,
      precision, precision,
      /* is_operand_forwarded*/ false);
  super_instr.setCommonField("imm", padding);
  super_instr.setCommonField(
      "src2", OperandAttr("0.0", OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField(
      field_name_out,
      OperandAttr(operand_name_out, OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField(
      "unroll",
      OperandAttr(stringifySentientUnrollFactor(op.getUnrollFactor()).str(),
                  OperandAttr::Type::DESCRIPTIVE));
  super_instr.setCommonField("unrlfldtgt",
                             OperandAttr(op.getUnrollIncrResult()));
  return super_instr;
}

// ---- 93/130  AddToLabelsMap  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:32  (8L)
void e093_AddToLabelsMap(
    UniformInstrBlocks& uniform_instr_region,
    std::map<std::string, std::set<InstrIndex>>& label_to_jumps,
    UniformInstrInfo& super_instr) {
  std::string label_str = super_instr.getCommonField("pc_target").asString();
  auto index = uniform_instr_region.getNextInstrIndex();
  label_to_jumps[label_str].insert(index);
}

// ---- 94/130  updateLabelAndAddToCodeGraph  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:47  (40L)
void e094_updateLabelAndAddToCodeGraph(
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region, Operation* op,
    UniformInstrInfo& super_instr, bool force_not_add_label) {
  std::string tag;
  if (!force_not_add_label) {
    if (labels.count(op) == 1) tag = labels[op];
  }

  // Check if the current instr is NOP
  bool is_instr_nop = super_instr.getInstn() == OpCodeT::NOP;

  // Check if the current operation is yield.
  is_instr_nop = is_instr_nop && isa<sentient::YieldOp>(op);

  // If both conditions are valid, then it is coming from MVLOOP
  // Checked with Exposed pipeline pass that we are not counting the cycles
  //  for yield in the exposed pipeline.

  // If a tag is present, don't merge with the last instruction!
  if (tag.empty() && is_instr_nop) {
    // check if the last block is empty.
    if (uniform_instr_region.getBlocks().back().getInstrLists().size() > 0) {
      auto& last_instr =
          uniform_instr_region.getBlocks().back().getInstrLists().back().back();

      // Check if there exists a be for the last instr,
      // If present, we shouldn't merge the instruction!
      if (!last_instr.hasCommonField("be") &&
          last_instr.getInstn() != OpCodeT::MVLOOPCNT) {
        last_instr.setCommonField(
            "be", OperandAttr("be", OperandAttr::Type::DESCRIPTIVE));
        return;
      }
    }
  }

  super_instr.setTag(tag);
  uniform_instr_region.addInstructionToLastBlock(super_instr);
}

// ---- 95/130  equalizeProgramLength  --  dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:469  (90L)
void e095_equalizeProgramLength(
    std::unordered_map<SenComponents, std::pair<std::string, int>>&
        max_length_unit_core_map) {
  // iterate through all unit types
  for (auto unit_core_pair : max_length_unit_core_map) {
    auto sen_comp = unit_core_pair.first;
    std::string unit_name = unit_core_pair.second.first;
    auto max_core_id = unit_core_pair.second.second;
    auto max_instr_list =
        static_cast<const ProgIrCodeBlock*>(progstateinfo_.at(max_core_id)
                                                .senCompProgram_.at(sen_comp)
                                                .blocks.front())
            ->instrVector;
    auto max_prog_length = max_instr_list.size();
    // iterate through psInfo for all cores
    for (auto& psinfo_pair : progstateinfo_) {
      auto& curr_sen_comp_program =
          progstateinfo_.at(psinfo_pair.first).senCompProgram_;
      // skip padding if current core doesn't has this unit type
      if (curr_sen_comp_program.find(sen_comp) == curr_sen_comp_program.end())
        continue;
      auto& curr_instr_vector =
          static_cast<ProgIrCodeBlock*>(
              curr_sen_comp_program.at(sen_comp).blocks.front())
              ->instrVector;
      auto curr_prog_length = curr_instr_vector.size();
      DT_CHECK_MSG(curr_prog_length > 0,
                   "The program contains no instruction.");

      // only pad instructions when the length of current program is shorter.
      if (max_prog_length - curr_prog_length == 1) {
        // if diff is one, insert NOP before RETURN
        UniformInstrInfo super_nop_instr = ConstructNOPInstr(
            EnumsConversion::senCompToGenericComp.at(sen_comp),
            "NOP for uniformization");
        curr_instr_vector.insert(curr_instr_vector.end() - 1,
                                 super_nop_instr.getRegularInstr());
      } else if (max_prog_length - curr_prog_length > 1) {
        auto padding_counter = uniformization_padding_counter_++;
        // use RETURN's tag if existing.
        std::string jcmp_tag =
            !curr_instr_vector.back().hasTag()
                ? ("uniformization_padding_" + std::to_string(padding_counter))
                : curr_instr_vector.back().getTagStr();
        curr_instr_vector.back().setTag(jcmp_tag);
        UniformInstrInfo super_jcmp_instr = ConstructJMPInstr(
            EnumsConversion::senCompToGenericComp.at(sen_comp), jcmp_tag);
        super_jcmp_instr.setCommonComment("jump for uniformization");
        curr_instr_vector.insert(curr_instr_vector.end() - 1,
                                 super_jcmp_instr.getRegularInstr());
        std::vector<std::string> pc_targets_in_dead_code;
        for (auto i = curr_prog_length; i < max_prog_length - 1; i++) {
          auto instr = max_instr_list.at(i);
          instr.setComment("padding for uniformization");
          instr.deadCode_ = true;
          // sanitize pc_targets
          if (instr.instFields_.find(OperandT::pc_target) !=
              instr.instFields_.end()) {
            auto target = instr.instFields_.at(OperandT::pc_target).asString();
            pc_targets_in_dead_code.push_back(target);
            // modify pc_target to avoid collision with the valid code
            instr.instFields_.at(OperandT::pc_target) = OperandAttr(
                target + "_uniformization", OperandAttr::Type::INSTR_TAG);
          }
          // sanitize tags
          if (instr.hasTag()) {
            if (std::find(pc_targets_in_dead_code.begin(),
                          pc_targets_in_dead_code.end(),
                          instr.getTagStr()) == pc_targets_in_dead_code.end()) {
              // remove tag is JCMP is missing.
              instr.clearTag();
            } else {
              // otherwise, modify it
              instr.setTag(instr.getTagStr() + "_uniformization");
            }
          }
          curr_instr_vector.insert(curr_instr_vector.end() - 1, instr);
        }
      } else if (max_prog_length - curr_prog_length == 0) {
        continue;
      } else {
        llvm::errs()
            << "curr_prog_length can't be larger than max_prog_length.";
        signalPassFailure();
        return;
      }
      DT_CHECK(curr_instr_vector.size() == max_prog_length);
    }
  }
}

// ---- 96/130  getUniformizedUnitInstrList  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:279  (42L)
std::pair<std::vector<UniformInstrInfo>, int>
e096_getUniformizedUnitInstrList(std::string unit_name) {
  if (block_type_ == Type::REGULAR) {
    DT_CHECK(instr_lists_.size() == 1);
    return {instr_lists_.front(), -1};
  }
  size_t max_region_idx = getMaxInstrRegionIndex();
  if (unit_to_region_idx_map_.find(unit_name) ==
          unit_to_region_idx_map_.end() ||
      (unit_to_region_idx_map_.at(unit_name) != max_region_idx &&
       instr_lists_.at(unit_to_region_idx_map_.at(unit_name)).size() <
           instr_lists_.at(max_region_idx).size())) {
    std::vector<UniformInstrInfo> ret_list;
    auto& max_region_list = instr_lists_.at(max_region_idx);
    if (unit_to_region_idx_map_.find(unit_name) !=
            unit_to_region_idx_map_.end() &&
        unit_to_region_idx_map_.at(unit_name) != max_region_idx) {
      ret_list = instr_lists_.at(unit_to_region_idx_map_.at(unit_name));
    }

    int jmp_pos = ret_list.size();
    int num_padded_instrs = max_region_list.size() - jmp_pos - 1;

    // An optimization to use NOP instruction instead of JCMP instruction
    // to save the cycles.
    if (num_padded_instrs == 0) {
      UniformInstrInfo nop_instr = createNOPInstr();
      ret_list.push_back(nop_instr);
    } else {
      // insert jmp instr to skip padding instructions
      UniformInstrInfo jmp_instr = createJmpInstr();
      ret_list.push_back(jmp_instr);
      // copy padding instructions
      for (int i = jmp_pos + 1; i < max_region_list.size(); i++) {
        ret_list.emplace_back(max_region_list.at(i)).setDeadCode();
      }
    }

    return {ret_list, jmp_pos};
  }
  return {instr_lists_.at(unit_to_region_idx_map_.at(unit_name)), -1};
}

// ---- 97/130  flattenIndex  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:484  (12L)
size_t e097_flattenIndex(InstrIndex index,
                                        std::string unit_name) {
  size_t ret = 0;
  auto block_idx = std::get<0>(index);
  auto region_idx = std::get<1>(index);
  auto instr_idx = std::get<2>(index);
  for (int current_block = 0; current_block < block_idx; current_block++) {
    ret += blocks_.at(current_block).getRegionInstrSize(unit_name);
  }
  ret += instr_idx;
  return ret;
}


// ================================================================================================
// LEVEL 3
// ================================================================================================

// ---- 98/130  ConstructSplatPadInstr  --  dcc/src/Conversion/SentientToProgIR/ConstructProgIRHelper.cpp:3792  (65L)
std::optional<UniformInstrInfo>
e098_ConstructSplatPadInstr(
    const SenComponents& comp, sentient::SplatOp op, ProgIrGraphMap& reg_graph,
    std::optional<CodeQualityStats>& cq_stats) {
  auto output_port = op.getOutput().getDefiningOp<sentient::LogicalPortOp>();
  DT_CHECK_MSG(output_port, "unknown output port");
  auto precision_stringref = stringifySentientPrecision(op.getPrecision());
  auto op_precision =
      dcc::sentient::utils::getDataFormatFromSentientPrecisionAttr(
          precision_stringref);

  auto const_input = op.getInput().getDefiningOp<sentient::ConstantOp>();
  auto vec_const_input =
      op.getInput().getDefiningOp<sentient::VectorConstantOp>();
  auto input_port = op.getInput().getDefiningOp<sentient::LogicalPortOp>();
  auto query_map = op.getInput().getDefiningOp<mlir::uniform::QueryMapOp>();

  DT_CHECK_MSG(
      (is_any_of(comp, PE, SFP)),
      "Vector register initialization and IMMCOPY and SPLAT instruction is "
      "currently only available in the PE and SFP "
      "units");

  UniformInstrInfo super_instr;
  if (op.getDbgName().has_value())
    super_instr.setCommonComment(op.getDbgName().value().str());
  if (const_input || vec_const_input || query_map) {
    DT_CHECK(stringifySentientComputePort(output_port.getPortName())
                 .contains("lrf") &&
             "only LRF is currently supported");
    int reg_val =
        std::stoi(stringifySentientComputePort(output_port.getPortName())
                      .str()
                      .substr(3));
    int max_reg = dccExtContext()
                      .dsc_global_->sysDef.regInfoPerUnit.at(comp)
                      .at(LRF)
                      .maxNum;
    DT_CHECK_MSG(reg_val >= 0 && reg_val < max_reg, "invalid register number");
    // TODO: If its program header and not a replica, we have to construct
    // an encoding of the value which is not required right now.
    // Hence, we revert to PE/SFP_IMMCOPY.
    if (op.getProgramHeader() && op.getPad() == SentientSplatPad::none) {
      addPESFPLRFImmcopyToRegInit(op.getInput().getDefiningOp(), op_precision,
                                  reg_val, op, reg_graph);
      return std::nullopt;
    } else if (const_input || query_map) {
      // Construct IMMCOPY
      super_instr = ConstructImmCopyInstrFromSplatOp(
          super_instr, op.getInput().getDefiningOp(), comp, op_precision,
          reg_val, op, cq_stats);
    } else {
      DT_ERROR("PE/SFP IMMCOPY doesn't support vector of constants.");
    }
  } else if (input_port) {
    // Construct SPLAT
    super_instr =
        ConstructSplatInstrFromSplatOp(super_instr, input_port, output_port, op,
                                       comp, precision_stringref.str());
  } else {
    llvm_unreachable(
        "Input to Splat Op is either a Constant or a LogicalPort Op.");
  }
  return super_instr;
}

// ---- 99/130  LowerLoadAndSendOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:144  (29L)
void e099_LowerLoadAndSendOperation(
    dataflow::ProgramUnitOp unit_op, const SenComponents& comp,
    LoadAndSendOp& load_op, std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region,
    std::optional<CodeQualityStats>& cq_stats) {
  mlir::Value src_val = load_op.getMutableAddr();
  auto src_index = sentient::getValueRegIndex(src_val);
  auto dst_index = sentient::getValueRegIndex(load_op.getResult());
  if (src_index != dst_index) {
    auto assign_instr = ConstructAssignInstr(
        comp, src_val, load_op.getResult(), load_op.getElementSize(), cq_stats);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, load_op,
                                 assign_instr.value());
    // We have copied the source register into the target register, so use
    // the target register in place of the original source register.
    setValueRegIndex(src_val, dst_index);
    auto load_instr = ConstructLoadInstr(unit_op, comp, load_op, cq_stats);
    // Since the copy is only materialized in progIR, restore the original
    // register value in sentient IR after progIR is generated, to preserve its
    // original state.
    setValueRegIndex(src_val, src_index);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, load_op,
                                 load_instr, true);
  } else {
    auto load_instr = ConstructLoadInstr(unit_op, comp, load_op, cq_stats);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, load_op,
                                 load_instr);
  }
}

// ---- 100/130  LowerReceiveAndStoreOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:174  (49L)
void e100_LowerReceiveAndStoreOperation(
    const SenComponents& comp, ReceiveAndStoreOp& store_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region,
    std::optional<CodeQualityStats>& cq_stats) {
  // If the comp unit is L3LU and the producer is a ConstantOp, the resulting
  // operation will be a form of LDZ. If the ConstantOp contains a 16-bit
  // non-zero value, an additional L3_LDZimm16 instruction is required to load
  // the imm16 value into ZR.
  bool is_ldz = false;
  if (comp == L3LU) {
    Operation* producer = store_op.getProducer().getDefiningOp();
    DT_CHECK_MSG(producer, "expected valid producer");
    auto ldz_producer = dyn_cast<sentient::ConstantOp>(producer);
    if (ldz_producer && ldz_producer.getValue() != 0 &&
        ldz_producer->getResultTypes()[0].getIntOrFloatBitWidth() == 16) {
      UniformInstrInfo zr_assign_instr =
          ConstructZRAssignInstr(comp, ldz_producer.getValue());
      updateLabelAndAddToCodeGraph(labels, uniform_instr_region, store_op,
                                   zr_assign_instr);
      is_ldz = true;
    }
  }

  mlir::Value src_val = store_op.getMutableAddr();
  auto src_index = sentient::getValueRegIndex(src_val);
  auto dst_index = sentient::getValueRegIndex(store_op.getResult());
  if (src_index != dst_index) {
    auto assign_instr =
        ConstructAssignInstr(comp, src_val, store_op.getResult(),
                             store_op.getElementSize(), cq_stats);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, store_op,
                                 assign_instr.value(), is_ldz);
    // We have copied the source register into the target register, so use
    // the target register in place of the original source register.
    setValueRegIndex(src_val, dst_index);
    auto store_instr = ConstructStoreInstr(comp, store_op, cq_stats);
    // Since the copy is only materialized in progIR, restore the original
    // register value in sentient IR after progIR is generated, to preserve its
    // original state.
    setValueRegIndex(src_val, src_index);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, store_op,
                                 store_instr, true);
  } else {
    auto store_instr = ConstructStoreInstr(comp, store_op, cq_stats);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, store_op,
                                 store_instr, is_ldz);
  }
}

// ---- 101/130  LowerLoadAndStoreOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:224  (55L)
void e101_LowerLoadAndStoreOperation(
    const SenComponents& comp, LoadAndStoreOp& ls_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region,
    std::optional<CodeQualityStats>& cq_stats) {
  mlir::Value src_val_0 = ls_op.getSrcMutableAddr();
  mlir::Value src_val_1 = ls_op.getDstMutableAddr();
  auto src_index_0 = sentient::getValueRegIndex(src_val_0);
  auto src_index_1 = sentient::getValueRegIndex(src_val_1);
  auto dst_index_0 = sentient::getValueRegIndex(ls_op.getResult(0));
  auto dst_index_1 = sentient::getValueRegIndex(ls_op.getResult(1));
  bool label_added_already = false;
  dcc::utils::L3GatherScatterChecker gather_scatter(ls_op);
  if (src_index_0 != dst_index_0) {
    auto assign_instr = ConstructAssignInstr(
        comp, src_val_0, ls_op.getResult(0), ls_op.getElementSize(), cq_stats);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, ls_op,
                                 assign_instr.value());
    label_added_already |= !uniform_instr_region.getBlocks()
                                .back()
                                .getLastInstr()
                                .getTag()
                                .empty();
    // We have copied the source register into the target register, so use
    // the target register in place of the original source register.
    setValueRegIndex(src_val_0, dst_index_0);
  }
  if (!gather_scatter.isIBRWrite() && src_index_1 != dst_index_1) {
    auto assign_instr = ConstructAssignInstr(
        comp, src_val_1, ls_op.getResult(1), ls_op.getElementSize(), cq_stats);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, ls_op,
                                 assign_instr.value(), label_added_already);
    label_added_already |= !uniform_instr_region.getBlocks()
                                .back()
                                .getLastInstr()
                                .getTag()
                                .empty();
    // We have copied the source register into the target register, so use
    // the target register in place of the original source register.
    setValueRegIndex(src_val_1, dst_index_1);
  }

  UniformInstrInfo ls_instr =
      ConstructL3LoadAndStoreInstr(comp, ls_op, cq_stats);

  // Since the copy is only materialized in progIR, restore the original
  // register value in sentient IR after progIR is generated, to preserve its
  // original state.
  if (src_index_0 != dst_index_0) setValueRegIndex(src_val_0, src_index_0);
  if (!gather_scatter.isIBRWrite() && src_index_1 != dst_index_1)
    setValueRegIndex(src_val_1, src_index_1);

  updateLabelAndAddToCodeGraph(labels, uniform_instr_region, ls_op, ls_instr,
                               label_added_already);
}

// ---- 102/130  LowerLoadAndExtractScalarOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:280  (24L)
void e102_LowerLoadAndExtractScalarOperation(
    dataflow::ProgramUnitOp unit_op, const SenComponents& comp,
    LoadAndExtractScalarOp load_and_extract_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region,
    std::optional<CodeQualityStats>& cq_stats) {
  DT_CHECK_MSG(comp == LXLU,
               "load_and_extract_scalar_op only supported in LXLU");
  auto mutable_addr = load_and_extract_op.getMutableAddr();
  auto reg_idx = sentient::getValueRegIndex(mutable_addr);

  auto consumer = load_and_extract_op.getConsumer();
  Value consumer_unit;
  if (unit_op.getRegion().getArguments().empty())
    consumer_unit = unit_op.getUnits()[0];
  else
    consumer_unit = unit_op.getRegion().getArguments()[0];
  DT_CHECK_MSG(consumer == consumer_unit,
               "expecting load_and_extract_scalar op consumer to be self");

  auto load_instr = ConstructLoadInstr(comp, load_and_extract_op, cq_stats);
  updateLabelAndAddToCodeGraph(labels, uniform_instr_region,
                               load_and_extract_op, load_instr);
}

// ---- 103/130  LowerReceiveAndExtractScalarOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:305  (13L)
void e103_LowerReceiveAndExtractScalarOperation(
    const SenComponents& comp,
    ReceiveAndExtractScalarOp& receive_and_extract_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region,
    std::optional<CodeQualityStats>& cq_stats) {
  DT_CHECK_MSG(comp == LXSU,
               "receive_and_extract_scalar_op only supported in LXSU");
  auto lrfcopy_instr =
      ConstructLRFCopyInstr(comp, receive_and_extract_op, cq_stats);
  updateLabelAndAddToCodeGraph(labels, uniform_instr_region,
                               receive_and_extract_op, lrfcopy_instr);
}

// ---- 104/130  LowerLoadComputeAndSendOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:319  (33L)
void e104_LowerLoadComputeAndSendOperation(
    const SenComponents& comp, LoadComputeAndSendOp& load_compute_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region,
    std::optional<CodeQualityStats>& cq_stats) {
  mlir::Value src_val = load_compute_op.getMutableAddr();
  auto src_index = sentient::getValueRegIndex(src_val);
  auto dst_index = sentient::getValueRegIndex(load_compute_op.getResult());
  if (src_index != dst_index) {
    // The address calculation is done with respect to the dst_element_size.
    auto assign_instr =
        ConstructAssignInstr(comp, src_val, load_compute_op.getResult(),
                             load_compute_op.getSrcElementSize(), cq_stats);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, load_compute_op,
                                 assign_instr.value());
    // We have copied the source register into the target register, so use
    // the target register in place of the original source register.
    setValueRegIndex(src_val, dst_index);
    auto load_compute_instr =
        ConstructLoadComputeInstr(comp, load_compute_op, cq_stats);
    // Since the copy is only materialized in progIR, restore the original
    // register value in sentient IR after progIR is generated, to preserve its
    // original state.
    setValueRegIndex(src_val, src_index);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, load_compute_op,
                                 load_compute_instr, true);
  } else {
    auto load_compute_instr =
        ConstructLoadComputeInstr(comp, load_compute_op, cq_stats);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, load_compute_op,
                                 load_compute_instr);
  }
}

// ---- 105/130  LowerSyncOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:353  (37L)
void e105_LowerSyncOperation(
    const SenComponents& comp, SyncOp& sync_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region,
    std::optional<CodeQualityStats>& cq_stats) {
  if (is_any_of(comp, L0LU, L0SU, LXLU, LXSU, L3LU, L3SU)) {
    bool soft = false;
    if (sync_op.getSoft().has_value() && sync_op.getSoft().value()) soft = true;

    // Implicit sync information
    int implicit_sync_boundary_tile_size = -1;
    if (sync_op.getImplicitSyncMemoryBoundary().has_value())
      implicit_sync_boundary_tile_size =
          sync_op.getImplicitSyncMemoryBoundary().value();

    UniformInstrInfo sync_op_instr =
        ConstructSyncInstr(comp, soft, implicit_sync_boundary_tile_size,
                           stringifySentientSyncMode(sync_op.getMode()).str(),
                           sync_op.getUnits(), sync_op.getDbgName());
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, sync_op,
                                 sync_op_instr);

    // DD2 hardware requirement: There should be an instruction gap b/w implicit
    // tile size settign instruction and loads/stores.
    if (dcc_ext_ctx_.getArch() == RCUDD1A_ISA &&
        implicit_sync_boundary_tile_size > 0 &&
        isa<LoadAndSendOp, ReceiveAndStoreOp>(sync_op->getNextNode())) {
      if (cq_stats.has_value()) ++cq_stats.value().num_nops_;
      UniformInstrInfo nop_instr =
          ConstructNOPInstr(comp, "dummy_nop_after_implicit_sync_set");
      uniform_instr_region.addInstructionToLastBlock(nop_instr);
    }
  } else {
    sync_op->emitError("Unknown unit for sync operation\n");
    signalPassFailure();
  }
}

// ---- 106/130  LowerNOPOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:391  (14L)
void e106_LowerNOPOperation(
    const SenComponents& comp, Operation* op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region,
    std::optional<CodeQualityStats>& cq_stats) {
  auto nop_op = llvm::dyn_cast<NOPOp>(op);
  DT_CHECK_MSG(nop_op, "Expect nop operation");
  std::string comment_str = "";
  if (nop_op.getDbgName().has_value())
    comment_str = nop_op.getDbgName().value().str();
  if (cq_stats.has_value()) ++cq_stats.value().num_nops_;
  UniformInstrInfo no_op_instr = ConstructNOPInstr(comp, comment_str);
  updateLabelAndAddToCodeGraph(labels, uniform_instr_region, op, no_op_instr);
}

// ---- 107/130  LowerCommonOperations  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:538  (26L)
void e107_LowerCommonOperations(
    const SenComponents& comp, std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region, Operation* op,
    std::vector<InstrIndex>& nop_for_labels,
    std::optional<CodeQualityStats>& cq_stats) {
  // If there is a label associated with an op that isn't lowered, attach label
  // to next, if possible, otherwise insert nop
  if (labels.count(op) >= 1) {
    auto next_op = op->getNextNode();
    if (isa<uniform::YieldOp>(op)) {
      // uniform subregions are all parallel so the next_op should be from
      // yieldOp's parentOp
      next_op = op->getParentOp()->getNextNode();
    }
    if (next_op != nullptr && !isa<uniform::UniformizeRegionsOp>(next_op) &&
        labels.count(next_op) == 0) {
      labels[next_op] = labels.at(op);
      labels.erase(op);
    } else {
      if (cq_stats.has_value()) ++cq_stats.value().num_nops_;
      nop_for_labels.push_back(uniform_instr_region.getNextInstrIndex());
      UniformInstrInfo nop_instr = ConstructNOPInstr(comp, "NOP for label");
      updateLabelAndAddToCodeGraph(labels, uniform_instr_region, op, nop_instr);
    }
  }
}

// ---- 108/130  LowerBinaryOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:565  (8L)
void e108_LowerBinaryOperation(
    dataflow::ProgramUnitOp& unit_op, const SenComponents& comp,
    BinaryOp& binary_op, std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region) {
  UniformInstrInfo super_instr = ConstructBinaryInstr(binary_op, comp);
  updateLabelAndAddToCodeGraph(labels, uniform_instr_region, binary_op,
                               super_instr);
}

// ---- 109/130  LowerUnaryOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:574  (7L)
void e109_LowerUnaryOperation(
    dataflow::ProgramUnitOp& unit_op, const SenComponents& comp,
    UnaryOp& unary_op, std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region) {
  UniformInstrInfo instr = ConstructUnaryInstr(unary_op, comp);
  updateLabelAndAddToCodeGraph(labels, uniform_instr_region, unary_op, instr);
}

// ---- 110/130  LowerTernaryOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:582  (8L)
void e110_LowerTernaryOperation(
    dataflow::ProgramUnitOp& unit_op, const SenComponents& comp,
    TernaryOp& ternary_op, std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region) {
  UniformInstrInfo super_instr = ConstructTernaryInstr(ternary_op, comp);
  updateLabelAndAddToCodeGraph(labels, uniform_instr_region, ternary_op,
                               super_instr);
}

// ---- 111/130  LowerForOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:591  (66L)
void e111_LowerForOperation(
    const SenComponents& comp, ForOp& for_op, int& labels_ctr,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region, ProgIrGraphMap& reg_graph,
    std::map<std::string, std::set<InstrIndex>>& label_to_jumps,
    std::optional<CodeQualityStats>& cq_stats) {
  bool label_attached = false;
  auto program_headers = for_op.getProgramHeaderAttr();
  for (int i = 0; i < for_op.getNumRegionIterArgs(); i++) {
    int scale = GetAddressScale(comp, for_op.getRegionIterArgs()[i]);
    // Get element_size if present as part of the attributes.
    int element_size =
        for_op->hasAttr("element_sizes")
            ? mlir::cast<IntegerAttr>(mlir::cast<ArrayAttr>(for_op->getAttr(
                                          "element_sizes"))[i + 1])
                  .getInt()
            : 8 * scale;
    auto reglocale =
        mlir::cast<SentientRegTypeAttr>(for_op.getRegLocalesAttr()[i + 1])
            .getValue();
    bool is_jcr_or_lccr =
        is_any_of(reglocale, SentientRegType::jcr, SentientRegType::lccr);
    if (!is_jcr_or_lccr)
      DT_CHECK_MSG(element_size != -1, "Invalid element_size for ForOp");

    bool ph = !program_headers.empty() &&
              mlir::cast<BoolAttr>(program_headers[i]).getValue();
    auto instr_assign = ConstructAssignInstr(comp, for_op.getIterOperands()[i],
                                             for_op.getRegionIterArgs()[i],
                                             element_size, cq_stats, ph);
    if (ph) {
      // find unit list from copyOp's parentOp
      SmallVector<Value> units;
      Value key;
      if (getQueryKeyAndUnitsFromParentRegion(for_op, key, units).failed()) {
        for_op->emitOpError("can't find parentOp");
        signalPassFailure();
        return;
      }
      if (key) {
        // assume the op is a global var.
        auto parent_op =
            mlir::cast<BlockArgument>(key).getParentRegion()->getParentOp();
        DT_CHECK(dyn_cast<dataflow::ProgramUnitOp>(parent_op) ||
                 isa<mlir::uniform::UniformizeRegionsOp>(parent_op));
      }
      auto imm_vals = getRegImmVals(for_op.getIterOperands()[i], units, comp,
                                    element_size, scale);
      addToRegInit(for_op.getRegionIterArgs()[i], imm_vals, reg_graph);
    }

    if (instr_assign.has_value()) {
      updateLabelAndAddToCodeGraph(labels, uniform_instr_region, for_op,
                                   instr_assign.value(), label_attached);
      label_attached = true;
    }
  }

  UniformInstrInfo super_instr =
      ConstructMVLoopInstr(for_op, comp, labels, labels_ctr);
  // for dynamic loop, add it to label_to_jumps for future NOP optimization
  if (super_instr.hasCommonField("pc_target"))
    AddToLabelsMap(uniform_instr_region, label_to_jumps, super_instr);
  updateLabelAndAddToCodeGraph(labels, uniform_instr_region, for_op,
                               super_instr, label_attached);
}

// ---- 112/130  LowerMACOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:658  (31L)
void e112_LowerMACOperation(
    dataflow::ProgramUnitOp& unit_op, const SenComponents& comp, MacOp& mac_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region,
    XRFRegisterAnalyzer* xrf_reg_analyzer) {
  UniformInstrInfo super_instr;
  auto precision =
      stringifySentientPrecision(mac_op.getComputePrecision()).str();
  super_instr = ConstructFMAInstr(mac_op, comp, xrf_reg_analyzer);
  updateLabelAndAddToCodeGraph(labels, uniform_instr_region, mac_op,
                               super_instr);
  if (mac_op.isXrfRdRelated()) {
    int required_xrf_read_incr_val =
        dccExtContext().getXrfRdPtrIncrValAfterMAC(precision);
    if (mac_op.getXrfReadIncrSigned() != required_xrf_read_incr_val) {
      super_instr = ConstructXRFADDInstr(
          comp, true,
          mac_op.getXrfReadIncrSigned() - required_xrf_read_incr_val);
      updateLabelAndAddToCodeGraph(labels, uniform_instr_region, mac_op,
                                   super_instr, true);
    }
  } else if (mac_op.isXrfWtRelated()) {
    int unroll_factor = mac_op.getUnrollFactorVal();
    if (unroll_factor != mac_op.getXrfWriteIncrSigned()) {
      super_instr = ConstructXRFADDInstr(
          comp, false, mac_op.getXrfWriteIncrSigned() - unroll_factor);
      updateLabelAndAddToCodeGraph(labels, uniform_instr_region, mac_op,
                                   super_instr, true);
    }
  }
}

// ---- 113/130  LowerSubOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:690  (40L)
void e113_LowerSubOperation(
    const SenComponents& comp, SubOp& sub_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region,
    std::optional<CodeQualityStats>& cq_stats) {
  if (sub_op.getRegLocale() == SentientRegType::jcr) {
    UniformInstrInfo instr = ConstructJSUBInstr(comp, sub_op);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, sub_op, instr);
  } else if (sub_op.getRegLocale() == SentientRegType::lrf ||
             sub_op.getRegLocale() == SentientRegType::lar ||
             sub_op.getRegLocale() == SentientRegType::ear) {
    int scale = GetAddressScale(comp, sub_op);
    // Get element_size if present as part of the attributes.
    int element_size =
        sub_op->hasAttr("element_size")
            ? mlir::cast<::mlir::IntegerAttr>(sub_op->getAttr("element_size"))
                  .getInt()
            : 8 * scale;
    DT_CHECK_MSG(element_size != -1, "Invalid element_size for SubOp");

    std::vector<UniformInstrInfo> instr;
    if (sub_op.getRegLocale() == SentientRegType::lrf)
      instr = ConstructLRFSUBInstr(comp, sub_op, element_size, cq_stats);
    else if (sub_op.getRegLocale() == SentientRegType::lar ||
             sub_op.getRegLocale() == SentientRegType::ear)
      instr = ConstructLARorEARSUBInstr(comp, sub_op, element_size, cq_stats);
    else
      llvm_unreachable("incorrect path taken for this locale");
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, sub_op,
                                 instr[0]);
    for (int i = 1; i < instr.size(); i++) {
      updateLabelAndAddToCodeGraph(labels, uniform_instr_region, sub_op,
                                   instr[i], true);
    }
  } else {
    sub_op->emitError("Unable to find Prog.IR for Sentient SUB\n");
    signalPassFailure();
    return;
  }
}

// ---- 114/130  LowerAddOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:731  (50L)
void e114_LowerAddOperation(
    const SenComponents& comp, AddOp& add_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region,
    std::optional<CodeQualityStats>& cq_stats) {
  if (add_op.getRegLocale() == SentientRegType::jcr) {
    UniformInstrInfo add_instr = ConstructJADDInstr(comp, add_op);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, add_op,
                                 add_instr);
  } else if (add_op.getRegLocale() == SentientRegType::lrf ||
             add_op.getRegLocale() == SentientRegType::lar ||
             add_op.getRegLocale() == SentientRegType::ear) {
    int scale = GetAddressScale(comp, add_op);
    // Get element_size if present as part of the attributes.
    int element_size =
        add_op->hasAttr("element_size")
            ? mlir::cast<::mlir::IntegerAttr>(add_op->getAttr("element_size"))
                  .getInt()
            : 8 * scale;
    DT_CHECK_MSG(element_size != -1, "Invalid element_size for AddOp");

    std::vector<UniformInstrInfo> add_instr;
    if (add_op.getRegLocale() == SentientRegType::lrf)
      add_instr = ConstructLRFADDInstr(comp, add_op, element_size, cq_stats);
    else if (add_op.getRegLocale() == SentientRegType::lar ||
             add_op.getRegLocale() == SentientRegType::ear)
      add_instr =
          ConstructLARorEARADDInstr(comp, add_op, element_size, cq_stats);
    else
      llvm_unreachable("incorrect path taken for this locale");
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, add_op,
                                 add_instr[0]);
    if (add_instr.size() > 1) {
      updateLabelAndAddToCodeGraph(labels, uniform_instr_region, add_op,
                                   add_instr[1], true);
    }
  } else if (add_op.getRegLocale() == SentientRegType::xrfrdptr) {
    UniformInstrInfo add_instr = ConstructXRFADDInstr(comp, add_op);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, add_op,
                                 add_instr);
  } else if (add_op.getRegLocale() == SentientRegType::xrfwrptr) {
    UniformInstrInfo add_instr = ConstructXRFADDInstr(comp, add_op);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, add_op,
                                 add_instr);
  } else {
    add_op->emitError("Unable to find Prog.IR for Sentient ADD\n");
    signalPassFailure();
    return;
  }
}

// ---- 115/130  LowerReturnOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:900  (7L)
void e115_LowerReturnOperation(
    const SenComponents& comp, Operation* op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region) {
  UniformInstrInfo return_instr = ConstructReturnInstr(comp);
  updateLabelAndAddToCodeGraph(labels, uniform_instr_region, op, return_instr);
}

// ---- 116/130  LowerSetSendDestinationOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:908  (18L)
void e116_LowerSetSendDestinationOperation(
    const SenComponents& comp, SetSendDestinationOp op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region) {
  if (is_any_of(comp, LXLU, LXSU)) {
    auto setdstmask_instr =
        ConstructSetDstMaskInstr(comp, op.getUnits().getDefiningOp());
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, op,
                                 setdstmask_instr);
  } else if (is_any_of(comp, SFP)) {
    auto setdest_instr = ConstructSetDestInstr(comp, op);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, op,
                                 setdest_instr);
  } else {
    op->emitError("did not expect set_send_dst in this unit\n");
    signalPassFailure();
  }
}

// ---- 117/130  LowerOpaqueOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:939  (12L)
void e117_LowerOpaqueOperation(
    const SenComponents& comp, sentient::OpaqueOp op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region, ProgIrGraphMap& reg_graph) {
  std::vector<UniformInstrInfo> opaque_expand =
      ConstructOpaqueInstr(comp, op, reg_graph);
  bool no_label = true;
  for (auto& inst : opaque_expand) {
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, op, inst, true);
    no_label = false;
  }
}

// ---- 118/130  LowerSAMVOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:952  (14L)
void e118_LowerSAMVOperation(
    const SenComponents& comp, SetActiveMaskValueOp& samv_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region) {
  DT_CHECK_MSG(comp == LXLU,
               "SAMV instruction is currently only available in LXLU units");
  auto samv_instr = ConstructSAMVInstr(comp, samv_op);
  updateLabelAndAddToCodeGraph(labels, uniform_instr_region, samv_op,
                               samv_instr);

  // TODO: Remove when progtailor can support the reset.
  if (dccExtContext().getProgStitch() || forceSAMVReset.getValue())
    has_samv_ = samv_op;
}

// ---- 119/130  LowerSetMaskOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1118  (10L)
void e119_LowerSetMaskOperation(
    const SenComponents& comp, SetMaskOp& set_mask_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region) {
  DT_CHECK_MSG(comp == PT,
               "SETMASK instruction is currently only available in PT units");
  auto set_mask_instr = ConstructSetMaskInstr(comp, set_mask_op);
  updateLabelAndAddToCodeGraph(labels, uniform_instr_region, set_mask_op,
                               set_mask_instr);
}

// ---- 120/130  LowerIncrMaskOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1129  (10L)
void e120_LowerIncrMaskOperation(
    const SenComponents& comp, IncrMaskOp& incrmask_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region) {
  DT_CHECK_MSG(comp == PT,
               "INCRMASK instruction is currently only available in PT units");
  UniformInstrInfo incrmask_instr = ConstructIncrMaskInstr(comp, incrmask_op);
  updateLabelAndAddToCodeGraph(labels, uniform_instr_region, incrmask_op,
                               incrmask_instr);
}

// ---- 121/130  getUniformizedUnitUniformInstrList  --  dcc/src/Conversion/SentientToProgIR/UniformInstrAndBlock.cpp:393  (32L)
std::vector<UniformInstrInfo>
e121_getUniformizedUnitUniformInstrList(std::string unit_name) {
  std::vector<UniformInstrInfo> ret;
  int prev_jmp_pos = -1;
  for (auto block : blocks_) {
    std::vector<UniformInstrInfo> instr_list;
    int new_jmp_pos;
    std::tie(instr_list, new_jmp_pos) =
        block.getUniformizedUnitInstrList(unit_name);
    if (prev_jmp_pos >= 0) {
      // if previous block has a jump op inserted, put tag to the first instr of
      // current block
      auto& jmp_instr = ret.at(prev_jmp_pos);
      if (jmp_instr.getInstn() == OpCodeT::JCMP) {
        if (!instr_list.front().getTag().empty()) {
          // if the first instr already has a tag, use this tag to replace
          // pc_target in prev jmp op.
          jmp_instr.setCommonField("pc_target",
                                   OperandAttr(instr_list.front().getTag(),
                                               OperandAttr::Type::INSTR_TAG));
        } else {
          instr_list.front().setTag(
              jmp_instr.getCommonField("pc_target").asString());
        }
      }
    }

    prev_jmp_pos = new_jmp_pos == -1 ? -1 : (new_jmp_pos + ret.size());
    ret.insert(ret.end(), instr_list.begin(), instr_list.end());
  }
  return ret;
}


// ================================================================================================
// LEVEL 4
// ================================================================================================

// ---- 122/130  LowerCopyOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:406  (95L)
void e122_LowerCopyOperation(
    dataflow::ProgramUnitOp unit_op, const SenComponents& comp, CopyOp& copy_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region, ProgIrGraphMap& reg_graph,
    std::vector<InstrIndex>& nop_for_labels,
    std::optional<CodeQualityStats>& cq_stats) {
  auto is_multicast = isMulticast(copy_op.getInp());
  auto is_constant = isConstant<sentient::ConstantOp>(copy_op.getInp());
  auto is_symbol = isSymbol(copy_op.getInp());
  auto is_get_unit = isGetUnit(copy_op.getInp());
  int scale = is_multicast ? 1 : GetAddressScale(comp, copy_op);
  // Get element_size if present as part of the attributes.
  int element_size =
      copy_op->hasAttr("element_size")
          ? mlir::cast<::mlir::IntegerAttr>(copy_op->getAttr("element_size"))
                .getInt()
          : 8 * scale;

  // In sen1p5 EBRs need to be shifted by 1.
  // For sen1p5 the EBR needs to be shifted right once so the value fits in
  // to the number of bits available.
  if (dccExtContext().getArch() >= IsaCoreGen::SEN1P5_ISA &&
      copy_op.getRegLocale() == SentientRegType::ebr) {
    ExpressionEvaluator evaluator;
    auto inp = copy_op.getInp();
    DT_CHECK(inp.getDefiningOp() &&
             (isa<sentient::ConstantOp>(inp.getDefiningOp()) ||
              isa<uniform::QueryMapOp>(inp.getDefiningOp())));
    const EvaluatedValue& ev = evaluator.evaluateValue(inp);

    DT_CHECK_MSG(copy_op->hasAttr("element_size") && element_size > 0,
                 "sen1p5 EBR requires element size to initialize reg");
    int64_t num_elems_in_stick =
        dccExtContext().getBytesPerStick() * 8 / element_size;
    DT_CHECK_MSG(ev.isDivisibleBy(2 * num_elems_in_stick),
                 "EBR is not divisible by an even number of sticks");

    const EvaluatedValue& new_ev =
        evaluator.evaluateShift(ev, 1, /* shift_right */ true);
    OpBuilder const_builder(unit_op);
    OpBuilder query_map_builder(
        dcc::uniform::utils::getLocalOrGlobalRegion(copy_op));
    auto new_const = new_ev.buildOffsetValue(const_builder, query_map_builder,
                                             copy_op->getLoc(),
                                             const_builder.getIndexType());
    copy_op.getInpMutable().assign(new_const);
  }

  // CopyOps with certain reg locales are allowed to have uninitialized
  // element_size.
  DT_CHECK((is_any_of(copy_op.getRegLocale(), SentientRegType::jcr,
                      SentientRegType::gtr, SentientRegType::ear,
                      SentientRegType::mvr) ||
            element_size != -1) &&
           "Invalid element_size for CopyOp");

  auto program_header = copy_op.getProgramHeader();
  if (program_header && !mlir::isa<BlockArgument>(copy_op.getInp())) {
    auto get_unit_op =
        llvm::dyn_cast<dataflow::GetUnitOp>(copy_op.getInp().getDefiningOp());
    DT_CHECK(is_constant || is_symbol || is_multicast || is_get_unit);
    // find unit list from copyOp's parentOp
    SmallVector<Value> units;
    Value key;
    if (getQueryKeyAndUnitsFromParentRegion(copy_op, key, units).failed()) {
      copy_op->emitOpError("can't find parentOp");
      signalPassFailure();
      return;
    }
    if (key) {
      // if copy_op is inside uniformizeRegionOp, region num has to be one.
      // if uniformizeRegionOp has region num > 1, copy_op has to be hoisted to
      // global region which is executed by regInit pass.
      auto parent_op =
          mlir::cast<BlockArgument>(key).getParentRegion()->getParentOp();
      if (isa<uniform::UniformizeRegionsOp, uniform::EqualizePatternOp>(
              parent_op)) {
        DT_CHECK(parent_op->getNumRegions() == 1);
      }
    }
    bool isMVRReg = (copy_op.getRegLocale() == SentientRegType::mvr);
    auto imm_vals = getRegImmVals(copy_op.getInp(), units, comp, element_size,
                                  scale, isMVRReg);
    addToRegInit(copy_op.getOut(), imm_vals, reg_graph,
                 /*is_symbolic*/ isSymbol(copy_op.getInp()));
    LowerCommonOperations(comp, labels, uniform_instr_region, copy_op,
                          nop_for_labels, cq_stats);
  } else {
    auto copy_op_instr =
        ConstructAssignInstr(comp, copy_op.getInp(), copy_op.getOut(),
                             element_size, cq_stats, program_header);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, copy_op,
                                 copy_op_instr.value());
  }
}

// ---- 123/130  LowerYieldOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:782  (117L)
void e123_LowerYieldOperation(
    const SenComponents& comp, YieldOp& yield_op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region,
    std::map<std::string, std::set<InstrIndex>>& label_to_jumps,
    std::vector<InstrIndex>& nop_for_labels,
    std::optional<CodeQualityStats>& cq_stats) {
  bool added_label = false;
  auto parent_op = yield_op->getParentOp();
  bool is_xrf = false;
  // lower yield_op's operands to assignment instructions if needed.
  // otherwise, insert NOP
  if (yield_op->getNumOperands() > 0) {
    for (int i = 0; i < yield_op->getNumOperands(); i++) {
      auto reg_locale = symbolizeSentientRegType(
                            getValueRegLocaleAsString(yield_op.getOperand(i)))
                            .value();
      // xrf has only 1 address register, no need to insert assignement.
      if (reg_locale == SentientRegType::xrfrdptr ||
          reg_locale == SentientRegType::xrfwrptr) {
        is_xrf = true;
        continue;
      }
      int element_size =
          parent_op->hasAttr("element_sizes")
              ? mlir::cast<IntegerAttr>(
                    mlir::cast<ArrayAttr>(parent_op->getAttr("element_sizes"))
                        [isa<sentient::ForOp>(parent_op) ? i + 1 : i])
                    .getInt()
              : -1;
      // if reg indices or types of src and dst are different, insert assignment
      // op.
      std::optional<UniformInstrInfo> instr;
      if (auto for_op = llvm::dyn_cast<sentient::ForOp>(parent_op)) {
        instr = ConstructAssignInstr(comp, yield_op.getOperand(i),
                                     for_op.getRegionIterArgs()[i],
                                     element_size, cq_stats);
      } else {
        // if-op
        instr = ConstructAssignInstr(comp, yield_op.getOperand(i),
                                     parent_op->getResult(i), element_size,
                                     cq_stats);
      }
      if (instr.has_value()) {
        // if yieldOp is lowered to an assignment instruction, update label and
        // add it to graph
        updateLabelAndAddToCodeGraph(labels, uniform_instr_region, yield_op,
                                     instr.value(), added_label);

        if (!uniform_instr_region.getBlocks()
                 .back()
                 .getLastInstr()
                 .getTag()
                 .empty())
          added_label = true;
      } else {
        // if yieldOp isn't lowered to an assignment instruction, insert NOP
        LowerCommonOperations(comp, labels, uniform_instr_region, yield_op,
                              nop_for_labels, cq_stats);
      }
    }
  } else {
    LowerCommonOperations(comp, labels, uniform_instr_region, yield_op,
                          nop_for_labels, cq_stats);
  }

  if (auto for_op = llvm::dyn_cast<sentient::ForOp>(parent_op)) {
    // construct BE instruction
    // insert reg assignement instructions if the reg types or indices of
    // for_op's results and iterators are different.
    UniformInstrInfo super_instr = ConstructBranchExitInstr(yield_op, comp);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, yield_op,
                                 super_instr, added_label);
    for (int i = 0; i < for_op.getNumResults(); i++) {
      if (!for_op.getResult(i).use_empty()) {
        int element_size = parent_op->hasAttr("element_sizes")
                               ? mlir::cast<IntegerAttr>(
                                     mlir::cast<ArrayAttr>(parent_op->getAttr(
                                         "element_sizes"))[i + 1])
                                     .getInt()
                               : -1;
        auto instr_assign =
            ConstructAssignInstr(comp, for_op.getRegionIterArgs()[i],
                                 for_op.getResult(i), element_size, cq_stats);
        if (instr_assign.has_value()) {
          updateLabelAndAddToCodeGraph(labels, uniform_instr_region, yield_op,
                                       instr_assign.value(), added_label);
        }
      }
    }
  } else if (auto if_op = llvm::dyn_cast<IfOp>(parent_op)) {
    // else region handling.
    if (!if_op.getElseRegion().empty()) {
      // append JCMP(mode=always) to the end of then region
      if (yield_op == if_op.getThenRegion().front().getTerminator()) {
        auto else_body_first_op = &if_op.getElseRegion().front().front();
        if (labels.count(else_body_first_op) != 0) {
          auto instr = ConstructJMPInstr(comp, labels[if_op->getNextNode()]);
          AddToLabelsMap(uniform_instr_region, label_to_jumps, instr);
          updateLabelAndAddToCodeGraph(labels, uniform_instr_region, yield_op,
                                       instr, added_label);
        } else {
          LowerCommonOperations(comp, labels, uniform_instr_region, yield_op,
                                nop_for_labels, cq_stats);
        }
      } else if (yield_op == if_op.getElseRegion().front().getTerminator()) {
        // only need to insert NOP if reglocale of yieldOp's operands is xrf.
        // for other reg types, assignment instruction or NOP has been inserted
        // in the code above.
        if (is_xrf) {
          LowerCommonOperations(comp, labels, uniform_instr_region, yield_op,
                                nop_for_labels, cq_stats);
        }
      }
    }
  }
}

// ---- 124/130  LowerSplatOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:927  (11L)
void e124_LowerSplatOperation(
    const SenComponents& comp, sentient::SplatOp op,
    std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region, ProgIrGraphMap& reg_graph,
    std::optional<CodeQualityStats>& cq_stats) {
  std::optional<UniformInstrInfo> splat_or_immcopy_instr =
      ConstructSplatPadInstr(comp, op, reg_graph, cq_stats);
  if (splat_or_immcopy_instr.has_value())
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, op,
                                 splat_or_immcopy_instr.value());
}

// ---- 125/130  LowerUniformYieldOperation  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:1056  (61L)
void e125_LowerUniformYieldOperation(
    const SenComponents& comp, std::map<Operation*, std::string>& labels,
    UniformInstrBlocks& uniform_instr_region, mlir::uniform::YieldOp& yield_op,
    std::vector<InstrIndex>& nop_for_labels,
    std::optional<CodeQualityStats>& cq_stats) {
  bool added_label = false;
  auto parent_op = yield_op->getParentOp();
  bool is_xrf = false;
  if (yield_op->getNumOperands() > 0) {
    for (int i = 0; i < yield_op->getNumOperands(); i++) {
      auto reg_locale = symbolizeSentientRegType(
                            getValueRegLocaleAsString(yield_op.getOperand(i)))
                            .value();
      // xrf has only 1 address register, no need to insert assignement.
      if (reg_locale == SentientRegType::xrfrdptr ||
          reg_locale == SentientRegType::xrfwrptr) {
        is_xrf = true;
        continue;
      }
      int element_size = parent_op->hasAttr("element_sizes")
                             ? mlir::cast<IntegerAttr>(
                                   mlir::cast<ArrayAttr>(
                                       parent_op->getAttr("element_sizes"))[i])
                                   .getInt()
                             : -1;
      // if reg indices or types of src and dst are different, insert assignment
      // op.
      std::optional<UniformInstrInfo> instr =
          ConstructAssignInstr(comp, yield_op.getOperand(i),
                               parent_op->getResult(i), element_size, cq_stats);
      if (instr.has_value()) {
        // if yieldOp is lowered to an assignment instruction, update label and
        // add it to graph
        updateLabelAndAddToCodeGraph(labels, uniform_instr_region, yield_op,
                                     instr.value(), added_label);

        if (!uniform_instr_region.getBlocks()
                 .back()
                 .getLastInstr()
                 .getTag()
                 .empty())
          added_label = true;
      }
    }
  } else {
    LowerCommonOperations(comp, labels, uniform_instr_region, yield_op,
                          nop_for_labels, cq_stats);
  }
  auto region_idx = yield_op->getParentRegion()->getRegionNumber();
  auto region_num = yield_op->getParentOp()->getNumRegions();
  // if uniform::yieldOp is the only op in the current region, append an empty
  // vector in the current block.
  if (isa<uniform::YieldOp>(yield_op->getBlock()->front())) {
    uniform_instr_region.appendEmptyUniformRegion();
  }
  // once region ops terminates, append a regular block. If nextNode is also
  // region op, the empty regular block will be erased.
  if (region_idx == region_num - 1) {
    (void)uniform_instr_region.appendRegularBlock();
  }
}


// ================================================================================================
// LEVEL 5
// ================================================================================================

// ---- 126/130  LowerUniformOperations  --  dcc/src/Conversion/SentientToProgIR/LowerSentientHelper.cpp:985  (70L)
void e126_LowerUniformOperations(
    Operation* uniform_op, const SenComponents& comp,
    UniformInstrBlocks& uniform_instr_region, ProgIrGraphMap& reg_graph,
    const std::map<SenComponents, Isa>& isaPerUnit,
    InstructionEstimatorImpl& instruction_estimator,
    std::map<std::string, std::set<InstrIndex>>& label_to_jumps,
    std::vector<InstrIndex>& nop_for_labels, int& labels_ctr,
    std::map<Operation*, std::string>& labels, dataflow::ProgramUnitOp unit_op,
    XRFRegisterAnalyzer* xrf_reg_analyzer,
    std::optional<CodeQualityStats>& cq_stats) {
  // skip the body, if region ops contain no operations to be lowered.
  if (instruction_estimator.getEstimatedInstructionCount(dccExtContext(),
                                                         uniform_op) == 0) {
    LowerCommonOperations(comp, labels, uniform_instr_region, uniform_op,
                          nop_for_labels, cq_stats);
    return;
  }
  // remove empty regular block for back-to-back uniform blocks and add uniform
  // block
  if (uniform_instr_region.getBlocks().size() > 0 &&
      uniform_instr_region.getBlocks().back().empty())
    uniform_instr_region.getBlocks().pop_back();
  auto& block = uniform_instr_region.appendUniformBlock();
  // initialize unit_to_region_idx_map
  if (auto ur_op = dyn_cast<uniform::UniformizeRegionsOp>(uniform_op)) {
    fillUnitToIdMap<uniform::UniformizeRegionsOp>(ur_op, block);
  } else if (auto ep_op = dyn_cast<uniform::EqualizePatternOp>(uniform_op)) {
    fillUnitToIdMap<uniform::EqualizePatternOp>(ep_op, block);
  }
  size_t region_idx = 0;
  // Determine the maximum stats across the different uniformize regions.
  std::optional<CodeQualityStats> max_cq_stats_across_regions;
  bool collecting_cq_stats = cq_stats.has_value();
  if (collecting_cq_stats)
    max_cq_stats_across_regions = CodeQualityStats{0, 0, 0};
  for (auto& region : uniform_op->getRegions()) {
    RegDefTracker::UniformRegionContext ctx(regDefTracker_, uniform_op,
                                            region_idx);
    block.setCurrentRegion(region_idx++);
    // Collect the code quality statistics for the region.
    std::optional<CodeQualityStats> cq_stats_region;
    if (collecting_cq_stats) cq_stats_region = CodeQualityStats{0, 0, 0};
    region.walk<WalkOrder::PreOrder>([&](mlir::Operation* op) {
      return GenerateProgIR(op, comp, uniform_instr_region, reg_graph,
                            isaPerUnit, instruction_estimator, label_to_jumps,
                            nop_for_labels, labels_ctr, labels, unit_op,
                            xrf_reg_analyzer, cq_stats_region);
    });
    if (collecting_cq_stats) {
      max_cq_stats_across_regions.value().num_conditional_jcmps_ =
          std::max(max_cq_stats_across_regions.value().num_conditional_jcmps_,
                   cq_stats_region.value().num_conditional_jcmps_);
      max_cq_stats_across_regions.value().ibuff_usage_percent_ =
          std::max(max_cq_stats_across_regions.value().ibuff_usage_percent_,
                   cq_stats_region.value().ibuff_usage_percent_);
      max_cq_stats_across_regions.value().num_copy_ops_ =
          std::max(max_cq_stats_across_regions.value().num_copy_ops_,
                   cq_stats_region.value().num_copy_ops_);
    }
  }
  // Update global cq_stats
  if (collecting_cq_stats) {
    cq_stats.value().num_conditional_jcmps_ +=
        max_cq_stats_across_regions.value().num_conditional_jcmps_;
    cq_stats.value().ibuff_usage_percent_ +=
        max_cq_stats_across_regions.value().ibuff_usage_percent_;
    cq_stats.value().num_copy_ops_ +=
        max_cq_stats_across_regions.value().num_copy_ops_;
  }
}

// ---- 127/130  GenerateProgIR  --  dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:53  (65L)
void e127_GenerateProgIR(
    dataflow::ProgramUnitOp unit_op, const SenComponents& comp,
    UniformInstrBlocks& uniform_instr_region, ProgIrGraphMap& reg_graph,
    const std::map<SenComponents, Isa>& isaPerUnit,
    InstructionEstimatorImpl& instruction_estimator,
    std::map<std::string, std::set<InstrIndex>>& label_to_jumps,
    std::vector<InstrIndex>& nop_for_labels,
    std::optional<CodeQualityStats>& cq_stats) {
  int labels_ctr = 0;
  std::map<Operation*, std::string> labels;

  // TODO: This code estimation is temporary to test the recently implemented
  //  IBUFF size implementation in Sentient Analyses of DCC.
  int instr_count = instruction_estimator.getEstimatedInstructionCount(
      dccExtContext(), unit_op);
  LLVM_DEBUG({ llvm::dbgs() << "Estimated IBUFF:" << instr_count << "\n"; });
  int max_instr_count = dccExtContext().getMaxIBuffEntries(unit_op);
  if (CollectCodeQualityStats) {
    cq_stats.value().ibuff_usage_percent_ = instr_count * 100 / max_instr_count;
  }
  if (checkProgIR.getValue() && instr_count > max_instr_count) {
    unit_op.dump();
    DT_CHECK(!unit_op.getUnits().empty());
    auto get_unit_op = unit_op.getUnits().front();
    llvm::errs() << "Require larger IBUFF\n";
    llvm::errs() << "Max IBUFF(" << max_instr_count << ") Current IBUFF("
                 << instr_count << ") for unit:\n"
                 << get_unit_op << "\n";
    signalPassFailure();
    return;
  }

  // Reset the has_samv_ member. If a SAMV is encountered in the program,
  // it will set this member to the SAMV operation to allow resetting the
  // mask at the end of the program in the case progstitching is enabled.
  has_samv_ = nullptr;

  // Create XRF Register Analyzer for PT units with mx precision
  XRFRegisterAnalyzer* xrf_reg_analyzer = nullptr;
  if (comp == PT && unit_op.getPrecision().has_value()) {
    std::string unit_precision = unit_op.getPrecision().value().str();
    if (unit_precision.substr(0, 2) == "mx") {
      xrf_reg_analyzer = new XRFRegisterAnalyzer(unit_op);
    }
  }

  // Walk the operation and lower into Prog. IR
  unit_op.walk<WalkOrder::PreOrder>([&](mlir::Operation* op) {
    // If SAMV operations were encountered, reset the mask at the end of the
    // program.
    if (has_samv_ && isa<dataflow::ReturnOp>(op)) {
      UniformInstrInfo samv_reset_instr = ConstructSAMVResetInstruction(comp);
      uniform_instr_region.addInstructionToLastBlock(samv_reset_instr);
      // In case there is more than one return, reset has_samv_.
      has_samv_ = nullptr;
    }
    return GenerateProgIR(op, comp, uniform_instr_region, reg_graph, isaPerUnit,
                          instruction_estimator, label_to_jumps, nop_for_labels,
                          labels_ctr, labels, unit_op, xrf_reg_analyzer,
                          cq_stats);
  });

  // Clean up XRF Register Analyzer
  delete xrf_reg_analyzer;
}

// ---- 128/130  GenerateProgIR  --  dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:119  (111L)
WalkResult e128_GenerateProgIR(
    Operation* op, const SenComponents& comp,
    UniformInstrBlocks& uniform_instr_region, ProgIrGraphMap& reg_graph,
    const std::map<SenComponents, Isa>& isaPerUnit,
    InstructionEstimatorImpl& instruction_estimator,
    std::map<std::string, std::set<InstrIndex>>& label_to_jumps,
    std::vector<InstrIndex>& nop_for_labels, int& labels_ctr,
    std::map<Operation*, std::string>& labels, dataflow::ProgramUnitOp unit_op,
    XRFRegisterAnalyzer* xrf_reg_analyzer,
    std::optional<CodeQualityStats>& cq_stats) {
  if (regDefTracker_.enabled()) regDefTracker_.recordOpRegDefs(op);
  if (auto for_op = dyn_cast<sentient::ForOp>(op)) {
    LowerForOperation(comp, for_op, labels_ctr, labels, uniform_instr_region,
                      reg_graph, label_to_jumps, cq_stats);
  } else if (auto mac_op = dyn_cast<sentient::MacOp>(op)) {
    LowerMACOperation(unit_op, comp, mac_op, labels, uniform_instr_region,
                      xrf_reg_analyzer);
  } else if (auto binary_op = dyn_cast<sentient::BinaryOp>(op)) {
    LowerBinaryOperation(unit_op, comp, binary_op, labels,
                         uniform_instr_region);
  } else if (auto unary_op = dyn_cast<sentient::UnaryOp>(op)) {
    LowerUnaryOperation(unit_op, comp, unary_op, labels, uniform_instr_region);
  } else if (auto ternary_op = dyn_cast<sentient::TernaryOp>(op)) {
    LowerTernaryOperation(unit_op, comp, ternary_op, labels,
                          uniform_instr_region);
  } else if (auto if_op = dyn_cast<sentient::IfOp>(op)) {
    if (CollectCodeQualityStats) ++cq_stats.value().num_conditional_jcmps_;
    UniformInstrInfo super_instr =
        ConstructJCMPInstr(if_op, comp, labels, labels_ctr);
    AddToLabelsMap(uniform_instr_region, label_to_jumps, super_instr);
    updateLabelAndAddToCodeGraph(labels, uniform_instr_region, op, super_instr);
  } else if (auto yield_op = dyn_cast<sentient::YieldOp>(op)) {
    LowerYieldOperation(comp, yield_op, labels, uniform_instr_region,
                        label_to_jumps, nop_for_labels, cq_stats);
  } else if (auto add_op = dyn_cast<sentient::AddOp>(op)) {
    LowerAddOperation(comp, add_op, labels, uniform_instr_region, cq_stats);
  } else if (auto sub_op = dyn_cast<sentient::SubOp>(op)) {
    LowerSubOperation(comp, sub_op, labels, uniform_instr_region, cq_stats);
  } else if (auto load_op = dyn_cast<sentient::LoadAndSendOp>(op)) {
    LowerLoadAndSendOperation(unit_op, comp, load_op, labels,
                              uniform_instr_region, cq_stats);
  } else if (auto store_op = dyn_cast<sentient::ReceiveAndStoreOp>(op)) {
    LowerReceiveAndStoreOperation(comp, store_op, labels, uniform_instr_region,
                                  cq_stats);
  } else if (auto ls_op = dyn_cast<sentient::LoadAndStoreOp>(op)) {
    LowerLoadAndStoreOperation(comp, ls_op, labels, uniform_instr_region,
                               cq_stats);
  } else if (auto load_and_extract_op =
                 dyn_cast<sentient::LoadAndExtractScalarOp>(op)) {
    LowerLoadAndExtractScalarOperation(unit_op, comp, load_and_extract_op,
                                       labels, uniform_instr_region, cq_stats);
  } else if (auto receive_and_extract_op =
                 dyn_cast<sentient::ReceiveAndExtractScalarOp>(op)) {
    LowerReceiveAndExtractScalarOperation(comp, receive_and_extract_op, labels,
                                          uniform_instr_region, cq_stats);
  } else if (auto load_compute_op =
                 dyn_cast<sentient::LoadComputeAndSendOp>(op)) {
    LowerLoadComputeAndSendOperation(comp, load_compute_op, labels,
                                     uniform_instr_region, cq_stats);
  } else if (auto sync_op = dyn_cast<sentient::SyncOp>(op)) {
    LowerSyncOperation(comp, sync_op, labels, uniform_instr_region, cq_stats);
  } else if (isa<dataflow::ReturnOp>(op)) {
    LowerReturnOperation(comp, op, labels, uniform_instr_region);
  } else if (isa<sentient::NOPOp>(op)) {
    LowerNOPOperation(comp, op, labels, uniform_instr_region, cq_stats);
  } else if (auto copy_op = dyn_cast<sentient::CopyOp>(op)) {
    LowerCopyOperation(unit_op, comp, copy_op, labels, uniform_instr_region,
                       reg_graph, nop_for_labels, cq_stats);
  } else if (isa<sentient::ConstantOp, sentient::LogicalPortOp,
                 dataflow::ProgramUnitOp, dataflow::GetUnitOp,
                 dataflow::GetLocalUnitOp, dataflow::CreateMulticastGroupOp,
                 dataflow::CreateGroupOp, mlir::uniform::DefImmutableMappingOp,
                 mlir::uniform::QueryMapOp, mlir::symbol::CreateSymbolOp,
                 mlir::symbol::SymbolImmutableMappingOp,
                 mlir::symbol::SymbolQueryMapOp>(op)) {
    LowerCommonOperations(comp, labels, uniform_instr_region, op,
                          nop_for_labels, cq_stats);
  } else if (auto set_send_dst_op =
                 dyn_cast<sentient::SetSendDestinationOp>(op)) {
    LowerSetSendDestinationOperation(comp, set_send_dst_op, labels,
                                     uniform_instr_region);
  } else if (auto splat_op = dyn_cast<sentient::SplatOp>(op)) {
    LowerSplatOperation(comp, splat_op, labels, uniform_instr_region, reg_graph,
                        cq_stats);
  } else if (auto opaque_op = dyn_cast<sentient::OpaqueOp>(op)) {
    LowerOpaqueOperation(comp, opaque_op, labels, uniform_instr_region,
                         reg_graph);
  } else if (auto samv_op = dyn_cast<sentient::SetActiveMaskValueOp>(op)) {
    LowerSAMVOperation(comp, samv_op, labels, uniform_instr_region);
  } else if (isa<mlir::uniform::UniformizeRegionsOp,
                 mlir::uniform::EqualizePatternOp>(op)) {
    LowerUniformOperations(op, comp, uniform_instr_region, reg_graph,
                           isaPerUnit, instruction_estimator, label_to_jumps,
                           nop_for_labels, labels_ctr, labels, unit_op,
                           xrf_reg_analyzer, cq_stats);
    return WalkResult::skip();
  } else if (auto yield_op = dyn_cast<mlir::uniform::YieldOp>(op)) {
    LowerUniformYieldOperation(comp, labels, uniform_instr_region, yield_op,
                               nop_for_labels, cq_stats);
  } else if (auto set_mask_op = dyn_cast<sentient::SetMaskOp>(op)) {
    LowerSetMaskOperation(comp, set_mask_op, labels, uniform_instr_region);
  } else if (auto incrmask_op = dyn_cast<sentient::IncrMaskOp>(op)) {
    LowerIncrMaskOperation(comp, incrmask_op, labels, uniform_instr_region);
  } else {
    op->dump();
    op->emitError("unable to lower the op into prog. IR");
    signalPassFailure();
    return WalkResult::interrupt();
  }
  return WalkResult::advance();
}


// ================================================================================================
// LEVEL 6
// ================================================================================================

// ---- 129/130  GenerateProgIRForProgramUnit  --  dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:234  (186L)
void e129_GenerateProgIRForProgramUnit(
    dataflow::ProgramUnitOp unit_op,
    std::map<int, ProgramAndStateInfo>& psinfo_map,
    const std::map<SenComponents, Isa>& isa_per_unit,
    InstructionEstimatorImpl& instruction_estimator,
    std::unordered_map<SenComponents, std::pair<std::string, int>>&
        max_length_unit_core_map) {
  DT_CHECK(unit_op.getUnits().size() >= 1);

  UniformInstrBlocks uniform_instr_region;
  ProgIrGraphMap reg_graph;
  auto get_unit_op = unit_op.getUnits()[0].getDefiningOp<dataflow::GetUnitOp>();
  auto record =
      EnumsConversion::stringToSenComponents.find(get_unit_op.getType().str());
  DT_CHECK_MSG(record != EnumsConversion::stringToSenComponents.end(),
               "unexpected unit");

  // Construct and cache fold ids for all units of the program unit op
  // If there is an SDSC, grab fold data from it.
  // Otherwise assume 1 fold.
  if (dcc_ext_ctx_.artifacts_) {
    const auto& fold_ids = dcc_ext_ctx_.artifacts_->foldIds_;
    for (auto const unit : unit_op.getUnits())
      unit_foldid_map_[unit] =
          fold_ids.at(getResultNum(unit.getDefiningOp(), unit));
  } else {
    for (auto const unit : unit_op.getUnits()) unit_foldid_map_[unit] = 0;
  }

  std::map<std::string, std::set<InstrIndex>> label_to_jumps;
  std::vector<InstrIndex> nop_for_labels;
  std::optional<CodeQualityStats> cq_stats;
  if (CollectCodeQualityStats) cq_stats = CodeQualityStats{0, 0, 0};
  SenComponents comp = EnumsConversion::senCompToGenericComp.at(record->second);

  GenerateProgIR(unit_op, comp, uniform_instr_region, reg_graph, isa_per_unit,
                 instruction_estimator, label_to_jumps, nop_for_labels,
                 cq_stats);

  if (CollectCodeQualityStats) {
    comp_to_code_quality_stats_[comp] = cq_stats.value();
    llvm::dbgs() << "*************************\nCODE QUALITY STATISTICS FOR : "
                 << EnumsConversion::senComponentsToString.at(comp)
                 << "\nNumber of conditional JCMPs: "
                 << cq_stats.value().num_conditional_jcmps_
                 << "\nNumber of Copy operations: "
                 << cq_stats.value().num_copy_ops_
                 << "\nNumber of NOPs (excluding those inserted for program "
                    "length equalization): "
                 << cq_stats.value().num_nops_
                 << "\nPercentage of available IBUFF used: "
                 << cq_stats.value().ibuff_usage_percent_ << " %\n";

    std::map<RegType, unsigned> reg_num_per_type;
    for (auto& unit_map : regs_to_init_) {
      auto unit_info = unit_map.first;
      auto unit_comp = unit_info.second;
      auto gen_comp = EnumsConversion::senCompToGenericComp.at(unit_comp);
      if (gen_comp != comp) continue;
      auto& reg_map = unit_map.second;
      for (auto& reg_entry : reg_map) {
        auto reg_type = reg_entry.first;
        reg_num_per_type[reg_type] = std::max(
            reg_num_per_type[reg_type], (unsigned)reg_entry.second.size());
      }
    }
    for (auto& record : reg_num_per_type) {
      llvm::dbgs() << "Number of "
                   << ProgramAndStateInfo::regTypeToString.at(record.first)
                   << " registers: " << record.second << "\n";
    }
    cq_stats.value().comp_to_num_regs_[comp] = reg_num_per_type;
    llvm::dbgs() << "*************************\n\n";
  }

  for (auto const unit : unit_op.getUnits()) {
    get_unit_op = unit.getDefiningOp<dataflow::GetUnitOp>();
    int fold_idx = getResultNum(get_unit_op, unit);

    // skip doing modifications to fold_idx > 0 because those variations
    // are captured in OperandAttr.
    if (fold_idx > 0) continue;

    int core = getCoreId(get_unit_op);
    int corelet = dcc::getCoreletId(get_unit_op);
    std::string unit_name = getUnitName(get_unit_op);
    SenComponents my_comp =
        getSenComponentForProgramStateInfo(get_unit_op, corelet);
    auto& psinfo = psinfo_map[core];

    /*
    Going through the list of extra NOP instructions which is added only
    because of labels, to remove the use of extra label and combine it with
    the next instruction.
    */
    std::vector<UniformInstrInfo> NOP_to_be_deleted;
    if (!uniform_instr_region.empty()) {
      DEBUG_WITH_TYPE(
          "dump-raw-progir", int ind = 0;
          std::cerr << "PROGIR before NOP optimization:\n";
          for (auto e : uniform_instr_region.getUniformizedUnitUniformInstrList(
                   unit_name)) {
            std::cout << "index- " << ind++ << ": ";
            e.getRegularInstr().print(std::cerr);
          });

      for (auto index : nop_for_labels) {
        int nop_index = std::get<2>(index);
        auto curr_label = uniform_instr_region.getUniformInstr(index).getTag();
        InstrIndex next_index = index;
        std::get<2>(next_index)++;
        if (!uniform_instr_region.doesInstrWithThisIndexExist(next_index))
          continue;
        auto next_label =
            uniform_instr_region.getUniformInstr(next_index).getTag();
        if (curr_label == "be" || next_label == "be") continue;
        // If the label field of the next instruction is empty, use the label of
        // the current instruction. None-empty next_label is needed when
        // updating pc_target.
        if (next_label.empty()) {
          next_label = curr_label;
          uniform_instr_region.getUniformInstr(next_index).setTag(next_label);
        }
        for (auto jump_index : label_to_jumps[curr_label]) {
          auto& super_instr = uniform_instr_region.getUniformInstr(jump_index);
          super_instr.setCommonField(
              "pc_target",
              OperandAttr(next_label, OperandAttr::Type::INSTR_TAG));
          label_to_jumps[next_label].insert(jump_index);
        }
        // Add the NOP instr to the list of instr to be skipped in final progir
        NOP_to_be_deleted.push_back(
            uniform_instr_region.getUniformInstr(index));
      }
    }

    // Adding instructions into code graph
    psinfo.senCompProgram_[my_comp].destroyGraph();
    if (!uniform_instr_region.empty()) {
      std::vector<UniformInstrInfo> instrs =
          uniform_instr_region.getUniformizedUnitUniformInstrList(unit_name);
      for (int i = 0; i < instrs.size(); i++) {
        // skip extra NOPs added for labels
        auto& super_instr = instrs.at(i);
        if (std::find(NOP_to_be_deleted.begin(), NOP_to_be_deleted.end(),
                      super_instr) == NOP_to_be_deleted.end())
          psinfo.senCompProgram_[my_comp].addInstruction(
              super_instr.getUniformizedInstr(unit_name));
      }
      // record core id of the max program length for each type
      DT_CHECK(psinfo.senCompProgram_[my_comp].blocks.size() == 1);
      auto instr_size = static_cast<const ProgIrCodeBlock*>(
                            psinfo.senCompProgram_[my_comp].blocks.front())
                            ->instrVector.size();
      auto max_length_unit_core_map_it = max_length_unit_core_map.find(my_comp);
      auto max_instr_size =
          max_length_unit_core_map_it == max_length_unit_core_map.end()
              ? 0
              : static_cast<const ProgIrCodeBlock*>(
                    psinfo_map[max_length_unit_core_map_it->second.second]
                        .senCompProgram_[my_comp]
                        .blocks.front())
                    ->instrVector.size();
      if (max_length_unit_core_map_it == max_length_unit_core_map.end() ||
          max_instr_size < instr_size ||
          (max_instr_size > 0 && max_instr_size == instr_size &&
           max_length_unit_core_map.at(my_comp).second > core)) {
        max_length_unit_core_map[my_comp] = std::make_pair(unit_name, core);
      }
    }

    psinfo.regState_[my_comp].destroyGraph();
    if (reg_graph[unit_name].head) {
      auto reg_info =
          static_cast<ProgIrRegBlock*>(reg_graph[unit_name].head)->regInfo;
      for (auto& reg_type_pair : reg_info) {
        for (auto& reg_idx_pair : reg_type_pair.second) {
          psinfo.regState_[my_comp].addRegInit(
              reg_idx_pair.first, reg_idx_pair.second, reg_type_pair.first);
        }
      }
    }

    if (reg_graph[unit_name].blocks.empty()) psinfo.regState_.erase(my_comp);
  }
}


// ================================================================================================
// LEVEL 7
// ================================================================================================

// ---- 130/130  runOnOperation  --  dcc/src/Conversion/SentientToProgIR/SentientToProgIR.cpp:610  (135L)
void e130_runOnOperation() {
  std::unordered_map<SenComponents, std::pair<std::string, int>>
      max_length_unit_core_map;
  ModuleOp module_op = getOperation();

  if (dtGetEnv<std::string>("CODEGEN_DUMP_IRS").has_value())
    dataflow::utils::dumpModule(
        module_op, "codegen_dumps/" + dccExtContext().prog_name_,
        (llvm::Twine("last_sentient_ir") + ".mlir").str());

  module_op.walk([&](dataflow::ProgramUnitOp program_unit) {
    RegDefTracker::ProgramUnitContext program_unit_ctx(regDefTracker_,
                                                       program_unit);
    InstructionEstimatorImpl& instruction_estimator =
        getChildAnalysis<AccurateInstructionEstimator>(program_unit);
    GenerateProgIRForProgramUnit(
        program_unit, progstateinfo_, *dccExtContext().isa_per_unit_,
        instruction_estimator, max_length_unit_core_map);
    return WalkResult::skip();  // No nested program units.
  });

  // equalize program length for all cores
  if (dccExtContext().getProgPatch()) {
    equalizeProgramLength(max_length_unit_core_map);
  }

  std::ofstream out;
  LLVM_DEBUG({ initializeOutputStream(out, std::cout); });
  bool validity_failed = false;
  const std::vector<SdscFoldId> fold_ids =
      dcc_ext_ctx_.artifacts_ ? dcc_ext_ctx_.artifacts_->foldIds_
                              : std::vector<SdscFoldId>{0};
  if (checkProgIR.getValue()) {
    for (const auto& [core_id, psinfo] : progstateinfo_) {
      LLVM_DEBUG({
        out << "Program for coreID : " << core_id << " foldID: " << 0 << "\n";
        psinfo.print(out, 0);
      });

      // Check validity
      const auto result =
          psinfo.checkProgramValidity(*dccExtContext().isa_per_unit_, fold_ids);
      if (result.first != ProgramAndStateInfo::ErrorType::NONE) {
        llvm::dbgs() << "Program verification failed for core " << core_id
                     << " node " << dccExtContext().prog_name_
                     << "\nError message: " << result.second << "\n";
        validity_failed = true;
      }
    }
  }

  // Initialize all used regs.
  if (forceFullRegInit.getValue()) initializeUtilizedRegisters();

  if (regDefTracker_.enabled()) {
    // Mark initialized registers as referenced.
    module_op.walk([&](dataflow::ProgramUnitOp program_unit) {
      for (auto const unit : program_unit.getUnits()) {
        auto getUnitOp = unit.getDefiningOp<dataflow::GetUnitOp>();
        int core = getCoreId(getUnitOp);
        int corelet = dcc::getCoreletId(getUnitOp);
        SenComponents comp =
            getSenComponentForProgramStateInfo(getUnitOp, corelet);
        auto& psinfo = progstateinfo_[core];
        auto rsIt = psinfo.regState_.find(comp);
        if (rsIt != psinfo.regState_.end()) {
          for (auto& [regTy, regInits] : rsIt->second.getSimpleRegInit()) {
            if (regTy == RegType::SPR) continue;
            for (auto& [regNum, regInit] : regInits) {
              // Reg defs will have been initialized during codegen.
              psinfo.regDefs_.value()[std::make_pair(comp, regTy)].set(regNum);
            }
          }
        }
        RegDefTracker::checkRegDefs(progstateinfo_, dccExtContext().prog_name_,
                                    core, comp);
      }
      return WalkResult::skip();  // No nested program units.
    });
  }

  // The prog. IR is fully created at this point; the lowered IR is no longer
  // needed and is collapsed into a reference to the SMC artifact.
  if (ReplaceModuleWithSmc) {
    replaceProgramBodyWithSmcOp(module_op);

    if (dtGetEnv<std::string>("CODEGEN_DUMP_IRS").has_value())
      dataflow::utils::dumpModule(module_op,
                                  "codegen_dumps/" + dccExtContext().prog_name_,
                                  (llvm::Twine("init_ir") + ".mlir").str());
  }

  // If dumping flag is enabled
  if (dumpProgIR.getValue()) {
    for (const auto& id : fold_ids) {
      // If there is non-empty out prog file mentioned, dump it into it
      // otherwise, dump to stdout.
      if (!progIRFileName.empty()) {
        out.open(progIRFileName + (id > 0 ? "_fold" + std::to_string(id) : ""));
      } else {
        // If there is no file name, dump to stdout.
        initializeOutputStream(out, std::cout);
        if (fold_ids.size() > 1) out << "\n\nFold:" << id << "\n\n";
      }

      // Translate into senprog and dump
      if (progIRFormat == DCC::kSenProg) {
        dpc_.convertIr2Senprog(progstateinfo_, *dccExtContext().isa_per_unit_,
                               out, id);
        if (!progIRFileName.empty()) {
          out.close();
        }
      } else if (progIRFormat == DCC::kSmc) {
        // Translate into smc and dump
        dpc_.convertIr2SMC(progstateinfo_, *dccExtContext().isa_per_unit_, out,
                           id);
        if (!progIRFileName.empty()) {
          out.close();
        }
      } else {
        // Just dump regular prog-ir
        for (auto& record : progstateinfo_) {
          out << "Program for coreID : " << record.first << "\n";
          auto& psinfo = record.second;
          psinfo.print(out, id);
        }
      }
    }
  }

  if (validity_failed) {
    signalPassFailure();
    return;
  }
}

