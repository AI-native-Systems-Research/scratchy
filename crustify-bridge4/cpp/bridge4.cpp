#include "prelude.inc"

/* ==========================================================================
 * ENTRY 001  isDescriptive
 * AUTHORITY: sys-arch-spec/progir/progir.h:103-103   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  bool isDescriptive() const { return type_ == Type::DESCRIPTIVE; }

/* ==========================================================================
 * ENTRY 002  isTag
 * AUTHORITY: sys-arch-spec/progir/progir.h:110-110   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  bool isTag() const { return type_ == Type::INSTR_TAG; }

/* ==========================================================================
 * ENTRY 003  isVariable
 * AUTHORITY: sys-arch-spec/progir/progir.h:104-104   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  bool isVariable() const { return type_ == Type::VARIABLE; }

/* ==========================================================================
 * ENTRY 004  asString
 * AUTHORITY: sys-arch-spec/progir/progir.h:62-66   (5 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  const std::string &asString(SdscFoldIdInput id = std::nullopt) const {
    if (!isVariable() && !isDescriptive() && !isTag())
      DT_ERROR("OperandAttr: attribute not string");
    return getValueImpl<std::string>(id);
  }

/* ==========================================================================
 * ENTRY 005  Isa::getFieldPosForName
 * AUTHORITY: sys-arch-spec/isa/isa.cpp:68-79   (12 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
int Isa::getFieldPosForName(int type, OperandT field) const {
  auto it = typeToFieldName.find(type);
  if (it != typeToFieldName.end()) {
    auto& fieldNames = it->second;
    for (unsigned i = 0, e = fieldNames.size(); i < e; i++) {
      if (fieldNames.at(i).first == field || fieldNames.at(i).second == field) {
        return i;
      }
    }
  }
  return -1;
}

/* ==========================================================================
 * ENTRY 006  Isa::getOpcodeType
 * AUTHORITY: sys-arch-spec/isa/isa.cpp:115-115   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
int Isa::getOpcodeType(OpCodeT opcode) const { return opcodeToType.at(opcode); }

/* ==========================================================================
 * ENTRY 007  Isa::isValidOpcode
 * AUTHORITY: sys-arch-spec/isa/isa.cpp:102-113   (12 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
bool Isa::isValidOpcode(OpCodeT opc, bool silent /*= false*/) const {
  if (opcodeToType.count(opc) != 0) {
    return true;
  }
  if (unsupportedOpcodes.count(opc) != 0 && !silent) {
    std::cerr << "\033[1;31mopcode " << opc << " is supported in "
              << IsaCoreGenName[unsupportedOpcodes.at(opc)]
              << " or newer (current ISA: " << IsaCoreGenName[sysDef.coreArch]
              << ")\033[0m" << std::endl;
  }
  return false;
}

/* ==========================================================================
 * ENTRY 008  Isa::checkIfFieldExists
 * AUTHORITY: sys-arch-spec/isa/isa.cpp:93-100   (8 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
int Isa::checkIfFieldExists(OpCodeT opcode, OperandT field) const {
  int fieldPos = -1;
  if (isValidOpcode(opcode)) {
    fieldPos = getFieldPosForName(getOpcodeType(opcode), field);
    if (fieldPos >= 0) return fieldPos;
  }
  return -1;
}

/* ==========================================================================
 * ENTRY 009  ProgramAndStateInfo::hasVariablesInInstr
 * AUTHORITY: sys-arch-spec/progir/progir.cpp:677-694   (18 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
bool ProgramAndStateInfo::hasVariablesInInstr() const {
  // check if any operand is a variable (string) anywhere in all codes
  for (auto const &unitProgPair : senCompProgram_) {
    for (auto const block : unitProgPair.second.blocks) {
      if (block->type == ProgIrBlock::Type::CONDITION)
        return true;
      else if (block->type == ProgIrBlock::Type::CODE) {
        for (auto const &instr :
             static_cast<const ProgIrCodeBlock *>(block)->instrVector) {
          for (auto const &operandPair : instr.instFields_) {
            if (operandPair.second.isVariable()) return true;
          }
        }
      }
    }
  }
  return false;
}

/* ==========================================================================
 * ENTRY 010  Dpc::checkProgFormatCompatibility
 * AUTHORITY: sys-arch-spec/dpc/dpc.cpp:87-96   (10 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
bool Dpc::checkProgFormatCompatibility(
    const std::map<int, ProgramAndStateInfo>& progstateinfo_,
    const ProgFormatNames targetFormat) {
  for (auto const& prog : progstateinfo_) {
    if (progFormatFeaturesMap.at(targetFormat).supportVariables == false &&
        prog.second.hasVariablesInInstr() == true)
      return false;
  }
  return true;
}

/* ==========================================================================
 * ENTRY 011  getCommentStr
 * AUTHORITY: sys-arch-spec/progir/progir.h:347-349   (3 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  const std::string &getCommentStr() const {
    return comment_ ? *comment_ : empty;
  }

/* ==========================================================================
 * ENTRY 012  Isa::getOpCodePrefix
 * AUTHORITY: sys-arch-spec/isa/isa.cpp:1480-1501   (22 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
std::string Isa::getOpCodePrefix(const SenComponents& unit) {
  auto generic_unit = EnumsConversion::senCompToGenericComp.at(unit);
  std::string op_code;
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
    DT_ERROR_FMT(
        "Invalid unit string: %s",
        EnumsConversion::senComponentsToString.at(generic_unit).c_str());
  }
  return op_code;
}

/* ==========================================================================
 * ENTRY 013  Isa::getOpCodeWithPrefix
 * AUTHORITY: sys-arch-spec/isa/isa.cpp:1503-1506   (4 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
std::string Isa::getOpCodeWithPrefix(const SenComponents& unit,
                                     OpCodeT opcode) {
  return getOpCodePrefix(unit) + "_" + to_string(opcode);
}

/* ==========================================================================
 * ENTRY 014  ProgramAndStateInfo::getSenComponent
 * AUTHORITY: sys-arch-spec/progir/progir.cpp:612-618   (7 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
SenComponents ProgramAndStateInfo::getSenComponent(std::string unit,
                                                   const int corelet) {
  if (senCompMap.find(std::make_pair(unit, corelet)) == senCompMap.end()) {
    DT_ERROR("Undefined type of unit: " + unit);
  }
  return senCompMap.at(std::make_pair(unit, corelet));
}

/* ==========================================================================
 * ENTRY 015  hasSenDataType
 * AUTHORITY: sys-arch-spec/progir/progir.h:119-119   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  bool hasSenDataType() const { return senDataType_ != DataFormats::INVALID; }

/* ==========================================================================
 * ENTRY 016  getSenDataType
 * AUTHORITY: sys-arch-spec/progir/progir.h:95-99   (5 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  DataFormats getSenDataType() const {
    if (!hasSenDataType())
      DT_ERROR("OperandAttr: attribute does not have SendDataType");
    return senDataType_;
  }

/* ==========================================================================
 * ENTRY 017  ProgIrGraph::isGraphSimple
 * AUTHORITY: sys-arch-spec/progir/progir.cpp:532-545   (14 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
bool ProgIrGraph::isGraphSimple(bool blocking) const {
  if (blocks.empty()) {
    if (blocking)
      DT_ERROR("Graph is empty, cannot proceed");
    else
      return false;
  } else if (blocks.size() > 1 || head->type != graphType) {
    if (blocking)
      DT_ERROR("Graph has multiple blocks, it must be simplified to proceed");
    else
      return false;
  }
  return true;
}

/* ==========================================================================
 * ENTRY 018  getSimpleInstrVect
 * AUTHORITY: sys-arch-spec/progir/progir.h:464-467   (4 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  std::vector<InstrInfo> &getSimpleInstrVect() const {
    isGraphSimple(true);
    return static_cast<ProgIrCodeBlock *>(head)->instrVector;
  }

/* ==========================================================================
 * ENTRY 019  getSimpleRegInit
 * AUTHORITY: sys-arch-spec/progir/progir.h:497-500   (4 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  UnitRegState &getSimpleRegInit() const {
    isGraphSimple(true);
    return static_cast<ProgIrRegBlock *>(head)->regInfo;
  }

/* ==========================================================================
 * ENTRY 020  getTagStr
 * AUTHORITY: sys-arch-spec/progir/progir.h:345-345   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  const std::string &getTagStr() const { return tag_ ? *tag_ : empty; }

/* ==========================================================================
 * ENTRY 021  hasComment
 * AUTHORITY: sys-arch-spec/progir/progir.h:335-335   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  bool hasComment() const { return (bool)comment_ && !getCommentStr().empty(); }

/* ==========================================================================
 * ENTRY 022  hasTag
 * AUTHORITY: sys-arch-spec/progir/progir.h:333-333   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  bool hasTag() const { return (bool)tag_ && !getTagStr().empty(); }

/* ==========================================================================
 * ENTRY 023  isBool
 * AUTHORITY: sys-arch-spec/progir/progir.h:108-108   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  bool isBool() const { return type_ == Type::BOOLEAN; }

/* ==========================================================================
 * ENTRY 024  isFloat
 * AUTHORITY: sys-arch-spec/progir/progir.h:107-107   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  bool isFloat() const { return type_ == Type::FLOAT; }

/* ==========================================================================
 * ENTRY 025  isInt
 * AUTHORITY: sys-arch-spec/progir/progir.h:106-106   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  bool isInt() const { return type_ == Type::INT; }

/* ==========================================================================
 * ENTRY 026  isInt128
 * AUTHORITY: sys-arch-spec/progir/progir.h:109-109   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  bool isInt128() const { return type_ == Type::INT128; }

/* ==========================================================================
 * ENTRY 027  isVariableSymbol
 * AUTHORITY: sys-arch-spec/progir/progir.h:105-105   (1 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  bool isVariableSymbol() const { return type_ == Type::VARIABLE_SYMBOL; }

/* ==========================================================================
 * ENTRY 028  strToupper
 * AUTHORITY: sys-arch-spec/dpc/dpc.cpp:50-53   (4 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
inline std::string strToupper(std::string inputStr) {
  std::transform(inputStr.begin(), inputStr.end(), inputStr.begin(), ::toupper);
  return inputStr;
}

/* ==========================================================================
 * ENTRY 029  asInt
 * AUTHORITY: sys-arch-spec/progir/progir.h:68-72   (5 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
  int64_t asInt(SdscFoldIdInput id = std::nullopt) const {
    if (!isInt() && !isBool() && !isVariableSymbol())
      DT_ERROR("OperandAttr: attribute not int");
    return getValueImpl<int64_t>(id);
  }

/* ==========================================================================
 * ENTRY 030  ProgramAndStateInfo::tagToLCCR
 * AUTHORITY: sys-arch-spec/progir/progir.cpp:720-807   (88 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
int ProgramAndStateInfo::tagToLCCR(const ProgIrCodeGraph &progGraph,
                                   const std::string &tag, bool allowNotFound) {
  int lccrIdx = -1, loopNesting = -1, shadowLccrIdx = -1;
  std::set<std::string> tags;
  std::map<std::string, int> pending_jumps;
  bool disableLoopTag = false, tagFoundOnDeadCode = false;
  for (const auto &instr : progGraph.getSimpleInstrVect()) {
    if (instr.hasTag()) {
      tags.insert(instr.getTagStr());
      if (pending_jumps.count(instr.getTagStr()) > 0) {
        if (pending_jumps.at(instr.getTagStr()) != 0) disableLoopTag = true;
        pending_jumps.erase(instr.getTagStr());
      }
    }
    if (instr.instn_ == OpCodeT::MVLOOPCNT) {
      if (instr.deadCode_) {  // dead code, can be disregarded
        ++shadowLccrIdx;
      } else {  // keep track of loops
        shadowLccrIdx = ++loopNesting;
        for (auto &jump : pending_jumps) jump.second++;
      }
      if (instr.getTagStr() == tag) {
        if (lccrIdx != -1 && !tagFoundOnDeadCode) {
          if (!instr.deadCode_)
            DT_ERROR_FMT("Tag: %s found in multiple lines", tag.c_str());
        } else if (disableLoopTag) {
          if (allowNotFound)
            return 0;
          else
            DT_ERROR_FMT(
                "Tag: %s found after JCMP that could alter "
                "LCCR index",
                tag.c_str());
        } else {
          lccrIdx = shadowLccrIdx;
          tagFoundOnDeadCode = instr.deadCode_;
        }
      }
    } else if (instr.getTagStr() == tag) {
      // most likely a loop turned JCMP because of progpatch
      if (allowNotFound)
        return shadowLccrIdx + 1;
      else
        DT_ERROR_FMT("Tag: %s found on op different from MVLOOPCNT",
                     tag.c_str());
    } else if (instr.instn_ == OpCodeT::RETURN) {
      if (!instr.deadCode_ && instr.instFields_.count(OperandT::subroutine)) {
        auto &subrField = instr.instFields_.at(OperandT::subroutine);
        if (((subrField.isBool() || subrField.isInt()) &&
             subrField.asInt() == 1) ||
            (subrField.isDescriptive() && subrField.asString() == "yes")) {
          disableLoopTag = true;
        }
      }
    }
    if (instr.instFields_.count(OperandT::pc_target) ||
        (instr.instn_ == OpCodeT::MVLOOPCNT &&
         instr.instFields_.count(OperandT::imm) &&
         instr.instFields_.at(OperandT::imm).isTag())) {
      // jumps can allow for unbalanced MVLOOPCNT/BE
      const auto &target =
          instr.instFields_.count(OperandT::pc_target)
              ? instr.instFields_.at(OperandT::pc_target).asString()
              : instr.instFields_.at(OperandT::imm).asString();
      if (tags.count(target)) disableLoopTag = true;
      pending_jumps[target] = 0;
    }
    if (instr.instFields_.count(OperandT::be)) {
      auto &be = instr.instFields_.at(OperandT::be);
      if (((be.isInt() || be.isBool()) && be.asInt() == 1) ||
          (be.isDescriptive() && be.asString() == "be")) {
        if (instr.deadCode_) {  // dead code, can be disregarded
          --shadowLccrIdx;
        } else {  // keep track of loops
          shadowLccrIdx = --loopNesting;
          for (auto &jump : pending_jumps) jump.second--;
        }
      }
    }
  }
  if (lccrIdx < 0) {
    if (allowNotFound)
      lccrIdx = 0;
    else
      DT_ERROR("No loop found with tag: " + tag);
  }
  return lccrIdx;
}

/* ==========================================================================
 * ENTRY 031  ProgramAndStateInfo::tagToPC
 * AUTHORITY: sys-arch-spec/progir/progir.cpp:696-718   (23 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
int ProgramAndStateInfo::tagToPC(const ProgIrCodeGraph &progGraph,
                                 const std::string &tag, bool allowNotFound) {
  const std::vector<InstrInfo> &prog = progGraph.getSimpleInstrVect();
  bool tagFoundOnDeadCode = false;
  int pc = -1;
  for (int i = 0; i < prog.size(); i++) {
    if (prog[i].getTagStr() == tag) {
      if (pc == -1 || tagFoundOnDeadCode) {
        pc = i;
        tagFoundOnDeadCode = prog[i].deadCode_;
      } else {
        DT_ERROR_FMT("Tag: %s found in lines %d and %d", tag.c_str(), pc, i);
      }
    }
  }
  if (pc == -1) {
    if (allowNotFound)
      pc = 0;
    else
      DT_ERROR("No instruction found with tag: " + tag);
  }
  return pc;
}

/* ==========================================================================
 * ENTRY 032  Dpc::convertIr2Senprog
 * AUTHORITY: sys-arch-spec/dpc/dpc.cpp:615-777   (163 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
void Dpc::convertIr2Senprog(
    const std::map<int, ProgramAndStateInfo>& progstateinfo_,
    const std::map<SenComponents, Isa>& isaPerUnit, std::ostream& outputStream,
    SdscFoldIdInput id, std::set<int> targetCores) {
  if (!checkProgFormatCompatibility(progstateinfo_, SENPROG))
    DT_ERROR(
        "Target format does not support all the required features for"
        " this program");

  if (targetCores.empty()) {  // convert all cores
    for (const auto& prog : progstateinfo_) targetCores.insert(prog.first);
  }
  for (int coreId : targetCores) {
    if (progstateinfo_.count(coreId) == 0)
      DT_ERROR_FMT("Core ID specified not found in progIR: %d", coreId);
    const ProgramAndStateInfo& progInfo = progstateinfo_.at(coreId);
    for (auto const& unitProgPair : progInfo.senCompProgram_) {
      SenComponents unit = unitProgPair.first;
      std::string unitName = EnumsConversion::senComponentsToString.at(unit);
      std::string corelet;
      if (unit == L3LU || unit == L3SU)
        corelet = "0";
      else {
        corelet = unitName.back();
        unitName.pop_back();
        if (unitName.find("row") != std::string::npos) {
          unitName.insert(2, 1, '_');  // add _ aftrer "pt"
          unitName.pop_back();         // get rid of the underscore at the end
        }
      }
      const Isa& isa =
          isaPerUnit.at(ProgramAndStateInfo::getSenComponent(unitName));

      // generate IBuff instructions
      const bool useOldFormat = false;
      outputStream << "===== START file: prog.txt =====\n"
                   << "========== Core: " << coreId << " Corelet: " << corelet
                   << " Unit: " << unitName << " Program START ============\n";
      std::string instrStart, shiftStr;
      if (unit == PE0 || unit == PE1 || unit == SFP0 || unit == SFP1) {
        instrStart = "instr64 = ( ";
        shiftStr = "UL << ";
      } else {
        instrStart = "instr_temp.push_back( ";
        shiftStr = " << ";
      }

      for (auto const& instr : unitProgPair.second.getSimpleInstrVect()) {
        int instrType = isa.getOpcodeType(instr.instn_);
        if (instr.hasTag()) outputStream << "///" << instr.getTagStr() << "\n";
        if (useOldFormat) outputStream << instrStart;
        if (instr.instn_ == OpCodeT::JCMPI)
          // L3_JCMP with isimm=1 is called L3_JCMPI because field names
          // depend on isimm bit, but it's not a real separate instruction
          outputStream << Isa::getOpCodeWithPrefix(unitProgPair.first,
                                                   OpCodeT::JCMP);
        else
          outputStream << Isa::getOpCodeWithPrefix(unitProgPair.first,
                                                   instr.instn_);

        for (auto const& operandPair : instr.instFields_) {
          int fieldPos =
              isa.checkIfFieldExists(instr.instn_, operandPair.first);
          if (fieldPos < 0) {
            std::string operandName = Isa::to_string(operandPair.first);
            DT_ERROR_FMT(
                "Illegal instruction/operand combination for this "
                "architecture.\n"
                "Instruction: %s\nOperand: %s\n",
                Isa::to_string(instr.instn_).c_str(), operandName.c_str());
          }
          outputStream << " | (";
          if (operandPair.second.isVariable())
            outputStream << "$" << operandPair.second.print(id) << "$";
          else if (operandPair.second.isVariableSymbol())
            outputStream << operandPair.second.print(id, true);
          else if (operandPair.second.isDescriptive())
            outputStream
                << isa.typeToFieldEncoding.at(instrType).at(fieldPos).at(
                       operandPair.second.asString(id));
          else if (operandPair.second.isTag()) {
            if (operandPair.first == OperandT::src0 &&
                (instr.instn_ == OpCodeT::JCMP ||
                 instr.instn_ == OpCodeT::JCMPI))
              outputStream << progInfo.tagToLCCR(
                  unitProgPair.second, operandPair.second.asString(id),
                  instr.deadCode_);
            else
              outputStream << progInfo.tagToPC(unitProgPair.second,
                                               operandPair.second.asString(id),
                                               instr.deadCode_);
          } else if (operandPair.second.isInt() || operandPair.second.isBool())
            outputStream << operandPair.second.print(id);
          else
            DT_ERROR("Unexpected operand type, conversion not supported");

          outputStream << shiftStr
                       << isa.typeToFieldBitShift.at(instrType).at(fieldPos)
                       << ")";
        }
        if (useOldFormat) outputStream << " );";
        if (instr.hasComment() || instr.deadCode_)
          outputStream << "  // " << instr.getCommentStr();
        if (instr.deadCode_) outputStream << " ==Dead code==";
        outputStream << "\n";
      }
      outputStream << "========== Core: " << coreId << " Corelet: " << corelet
                   << " Unit: " << unitName << " Program END ============\n"
                   << "===== END file: prog.txt =====\n";

      // generate register initialization, if necessary

      if (progInfo.regState_.count(unit) > 0) {
        outputStream << "===== START file: reg_initial.txt =====\n";
        for (auto const& regVecPair :
             progInfo.regState_.at(unit).getSimpleRegInit()) {
          RegType regType = regVecPair.first;
          auto unitRegMap = regVecPair.second;
          outputStream << "#\n";
          // build line with unit:core:corelet
          outputStream << strToupper(unitName);
          if (regType != RegType::LRF) {
            outputStream << '-'
                         << ProgramAndStateInfo::regTypeToString.at(regType);
          }
          outputStream << ':' << coreId;
          if (unit != L3LU && unit != L3SU) {
            outputStream << ':' << corelet;
          }
          outputStream << "\n";
          for (auto const& regPair : unitRegMap) {
            char buff[20];
            snprintf(buff, sizeof(buff), "%010x ", regPair.first);
            outputStream << buff;
            if (regPair.second.hasSenDataType()) {
              outputStream << "datatype:" << regPair.second.getSenDataType()
                           << ' ';
            }
            if (regPair.second.isVariable()) {
              if (unit == PE0 || unit == PE1 || unit == SFP0 || unit == SFP1) {
                for (int i = 0; i < 8; i++)
                  outputStream << '@' << regPair.second.asString(id) << '@';
              } else {  // L3, LX, L0
                outputStream << '$' << regPair.second.asString(id)
                             << "$.000000";
              }
            } else if (regPair.second.isVariableSymbol()) {
              outputStream << regPair.second.print(id, true);
            } else if (regPair.second.isInt()) {
              outputStream << regPair.second.print(id) << ".000000";
            } else if (regPair.second.isFloat() || regPair.second.isInt128()) {
              outputStream << regPair.second.print(id);
            } else {
              DT_ERROR("Unsupported reg init data type for senprog generation");
            }
            outputStream << "\n";
          }
        }
        outputStream << "===== END file: reg_initial.txt =====\n";
      }
    }
  }
}

/* ==========================================================================
 * ENTRY 033  OperandAttr::print
 * AUTHORITY: sys-arch-spec/progir/progir.cpp:25-62   (38 lines)
 * revision a0d29abbed  --  BODY VERBATIM, NOTHING APPENDED
 * ========================================================================== */
std::string OperandAttr::print(const SdscFoldIdInput id /*= std::nullopt*/,
                               const bool prettyPrint /*= false*/) const {
  char buff[40];
  std::string out;
  switch (type_) {
    case Type::VARIABLE:
      out = asString(id);
      if (prettyPrint) out = '"' + out + '"';
      return out;
    case Type::INSTR_TAG:
      out = asString(id);
      if (prettyPrint) out = '(' + out + ')';
      return out;
    case Type::DESCRIPTIVE:
      return asString(id);
    case Type::INT:
      return std::to_string(asInt(id));
    case Type::VARIABLE_SYMBOL:
      return "%" + std::to_string(asInt(id)) + "%";
    case Type::FLOAT:
      snprintf(buff, sizeof(buff), "%.6f", asFloat(id));
      return buff;
    case Type::BOOLEAN:
      if (prettyPrint)
        return (asBool(id) ? "true" : "false");
      else
        return std::to_string(asBool(id));
    case Type::INT128:
      std::array<uint32_t, 4> int32vect;
      int32vect = asInt128(id);
      snprintf(buff, sizeof(buff), "%08x%08x%08x%08x", int32vect.at(0),
               int32vect.at(1), int32vect.at(2), int32vect.at(3));
      return (prettyPrint ? "0x" : "") + std::string(buff);
    case Type::UNKNOWN:
    default:
      DT_ERROR("Uexpected OperandAttr type to print");
  }
}

