// SPDX-License-Identifier: Apache-2.0
// PLAIN-C++ STAND-INS FOR BRIDGE 2. ⛔ NO MLIR, NO LLVM, NO dcc HEADERS.
// Every name is declared the way the 384 bodies USE it: a template if applied to type arguments, a
// namespace if reached through ::, a member of Any if reached through . or ->, a free function if
// called. Shapes only — the behaviour being ported is in the bodies.
#pragma once
#include <cstdint>
#include <string>
#include <vector>
#include <map>
#include <set>
#include <utility>
#include <optional>
#include <algorithm>
#include <memory>
#include <functional>

#define DT_CHECK(...) ((void)0)
#define DT_CHECK_MSG(...) ((void)0)
#define DT_ERROR(...) ((void)0)

namespace stub {
struct Any {
  Any() = default;
  Any(const Any&) = default;
  Any& operator=(const Any&) { return *this; }
  template <class T> Any(T&&) {}
  template <class T> operator T() const { return T{}; }
  template <class T> operator T*() const { return nullptr; }
  explicit operator bool() const { return true; }
  template <class... A> Any operator()(A&&...) const { return {}; }
  Any operator[](Any) const { return {}; }
  Any operator*() const { return {}; }
  const Any* operator->() const { return this; }
  Any* operator->() { return this; }
  Any& operator++() { return *this; }  Any operator++(int) { return {}; }
  Any& operator--() { return *this; }  Any operator--(int) { return {}; }
  Any& operator+=(Any) { return *this; }  Any& operator-=(Any) { return *this; }
  Any& operator*=(Any) { return *this; }  Any& operator|=(Any) { return *this; }
  template <class... T, class... A> Any addArgument(A&&...) const { return {}; }
  template <class... T, class... A> Any addDynamicallyLegalOp(A&&...) const { return {}; }
  template <class... T, class... A> Any addIllegalDialect(A&&...) const { return {}; }
  template <class... T, class... A> Any addIllegalOp(A&&...) const { return {}; }
  template <class... T, class... A> Any addLegalDialect(A&&...) const { return {}; }
  template <class... T, class... A> Any addLegalOp(A&&...) const { return {}; }
  template <class... T, class... A> Any assign(A&&...) const { return {}; }
  template <class... T, class... A> Any at(A&&...) const { return {}; }
  template <class... T, class... A> Any back(A&&...) const { return {}; }
  template <class... T, class... A> Any clear(A&&...) const { return {}; }
  template <class... T, class... A> Any clearCache(A&&...) const { return {}; }
  template <class... T, class... A> Any clone(A&&...) const { return {}; }
  template <class... T, class... A> Any cloneUseChainToNewOp(A&&...) const { return {}; }
  template <class... T, class... A> Any cloneWithNewAccessInfo(A&&...) const { return {}; }
  template <class... T, class... A> Any cloneWithoutRegions(A&&...) const { return {}; }
  template <class... T, class... A> Any compose(A&&...) const { return {}; }
  template <class... T, class... A> Any composeMatchingMap(A&&...) const { return {}; }
  template <class... T, class... A> Any compute(A&&...) const { return {}; }
  template <class... T, class... A> Any computeBurstAndGroup(A&&...) const { return {}; }
  template <class... T, class... A> Any constructAffineDetailsAndAddrs(A&&...) const { return {}; }
  template <class... T, class... A> Any constructDetails(A&&...) const { return {}; }
  template <class... T, class... A> Any contains(A&&...) const { return {}; }
  template <class... T, class... A> Any create(A&&...) const { return {}; }
  template <class... T, class... A> Any createBlock(A&&...) const { return {}; }
  template <class... T, class... A> Any createXrfIndexModifOps(A&&...) const { return {}; }
  template <class... T, class... A> Any data(A&&...) const { return {}; }
  template <class... T, class... A> Any dominates(A&&...) const { return {}; }
  template <class... T, class... A> Any dropAllUses(A&&...) const { return {}; }
  template <class... T, class... A> Any dump(A&&...) const { return {}; }
  template <class... T, class... A> Any dyn_cast(A&&...) const { return {}; }
  template <class... T, class... A> Any emitError(A&&...) const { return {}; }
  template <class... T, class... A> Any emitOpError(A&&...) const { return {}; }
  template <class... T, class... A> Any emitWarning(A&&...) const { return {}; }
  template <class... T, class... A> Any emplace(A&&...) const { return {}; }
  template <class... T, class... A> Any emplaceBlock(A&&...) const { return {}; }
  template <class... T, class... A> Any emplace_insert(A&&...) const { return {}; }
  template <class... T, class... A> Any empty(A&&...) const { return {}; }
  template <class... T, class... A> Any erase(A&&...) const { return {}; }
  template <class... T, class... A> Any eraseOp(A&&...) const { return {}; }
  template <class... T, class... A> Any eraseOpAndUseChain(A&&...) const { return {}; }
  template <class... T, class... A> Any eraseOperand(A&&...) const { return {}; }
  template <class... T, class... A> Any evaluateValue(A&&...) const { return {}; }
  template <class... T, class... A> Any failed(A&&...) const { return {}; }
  template <class... T, class... A> Any flatten(A&&...) const { return {}; }
  template <class... T, class... A> Any fold(A&&...) const { return {}; }
  template <class... T, class... A> Any front(A&&...) const { return {}; }
  template <class... T, class... A> Any fuseComputeOps(A&&...) const { return {}; }
  template <class... T, class... A> Any fuseNonComputeOps(A&&...) const { return {}; }
  template <class... T, class... A> Any generateSetSendDestinationStmts(A&&...) const { return {}; }
  template <class... T, class... A> Any get(A&&...) const { return {}; }
  template <class... T, class... A> Any getAbsorbtionFlag(A&&...) const { return {}; }
  template <class... T, class... A> Any getAffineMap(A&&...) const { return {}; }
  template <class... T, class... A> Any getAffineMapAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getAllQueriedValues(A&&...) const { return {}; }
  template <class... T, class... A> Any getArch(A&&...) const { return {}; }
  template <class... T, class... A> Any getArgNumber(A&&...) const { return {}; }
  template <class... T, class... A> Any getArgument(A&&...) const { return {}; }
  template <class... T, class... A> Any getArguments(A&&...) const { return {}; }
  template <class... T, class... A> Any getArrayAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getAttrDictionary(A&&...) const { return {}; }
  template <class... T, class... A> Any getAttrs(A&&...) const { return {}; }
  template <class... T, class... A> Any getBF16Type(A&&...) const { return {}; }
  template <class... T, class... A> Any getBase(A&&...) const { return {}; }
  template <class... T, class... A> Any getBinaryOp(A&&...) const { return {}; }
  template <class... T, class... A> Any getBlock(A&&...) const { return {}; }
  template <class... T, class... A> Any getBlocks(A&&...) const { return {}; }
  template <class... T, class... A> Any getBody(A&&...) const { return {}; }
  template <class... T, class... A> Any getBoolAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getBound(A&&...) const { return {}; }
  template <class... T, class... A> Any getBufferSize(A&&...) const { return {}; }
  template <class... T, class... A> Any getBurstIndex(A&&...) const { return {}; }
  template <class... T, class... A> Any getBytesPerStick(A&&...) const { return {}; }
  template <class... T, class... A> Any getChunkSize(A&&...) const { return {}; }
  template <class... T, class... A> Any getChunkStride(A&&...) const { return {}; }
  template <class... T, class... A> Any getComp(A&&...) const { return {}; }
  template <class... T, class... A> Any getCompareOp(A&&...) const { return {}; }
  template <class... T, class... A> Any getCond(A&&...) const { return {}; }
  template <class... T, class... A> Any getCondition(A&&...) const { return {}; }
  template <class... T, class... A> Any getConstantBound(A&&...) const { return {}; }
  template <class... T, class... A> Any getConstantLowerBound(A&&...) const { return {}; }
  template <class... T, class... A> Any getConstantStep(A&&...) const { return {}; }
  template <class... T, class... A> Any getConstantUpperBound(A&&...) const { return {}; }
  template <class... T, class... A> Any getConstraint(A&&...) const { return {}; }
  template <class... T, class... A> Any getContext(A&&...) const { return {}; }
  template <class... T, class... A> Any getData(A&&...) const { return {}; }
  template <class... T, class... A> Any getDataResult(A&&...) const { return {}; }
  template <class... T, class... A> Any getDbgName(A&&...) const { return {}; }
  template <class... T, class... A> Any getDbgNameAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getDefiningOp(A&&...) const { return {}; }
  template <class... T, class... A> Any getDimSize(A&&...) const { return {}; }
  template <class... T, class... A> Any getDir(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirectAffineMapAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirectDstAffineMapAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirectDstMapIndices(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirectDstMemref(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirectDstMemrefMutable(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirectMapIndices(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirectMemref(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirectMemrefType(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirectSrcAffineMapAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirectSrcMapIndices(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirectSrcMemref(A&&...) const { return {}; }
  template <class... T, class... A> Any getDirectSrcMemrefMutable(A&&...) const { return {}; }
  template <class... T, class... A> Any getDstAffineMapAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getDstMapIndices(A&&...) const { return {}; }
  template <class... T, class... A> Any getDstMemRef(A&&...) const { return {}; }
  template <class... T, class... A> Any getDstMemRefMutable(A&&...) const { return {}; }
  template <class... T, class... A> Any getDstUnit(A&&...) const { return {}; }
  template <class... T, class... A> Any getElementType(A&&...) const { return {}; }
  template <class... T, class... A> Any getElementTypeBitWidth(A&&...) const { return {}; }
  template <class... T, class... A> Any getElementWidth(A&&...) const { return {}; }
  template <class... T, class... A> Any getElseBodyBuilder(A&&...) const { return {}; }
  template <class... T, class... A> Any getElseNode(A&&...) const { return {}; }
  template <class... T, class... A> Any getElseRegion(A&&...) const { return {}; }
  template <class... T, class... A> Any getEmptyAffineMap(A&&...) const { return {}; }
  template <class... T, class... A> Any getEquality(A&&...) const { return {}; }
  template <class... T, class... A> Any getEvalOrder(A&&...) const { return {}; }
  template <class... T, class... A> Any getExpectedTotalElements(A&&...) const { return {}; }
  template <class... T, class... A> Any getExtents(A&&...) const { return {}; }
  template <class... T, class... A> Any getF16Type(A&&...) const { return {}; }
  template <class... T, class... A> Any getF32Type(A&&...) const { return {}; }
  template <class... T, class... A> Any getFalseValue(A&&...) const { return {}; }
  template <class... T, class... A> Any getFilterCoreletsExcept(A&&...) const { return {}; }
  template <class... T, class... A> Any getFilterCoresExcept(A&&...) const { return {}; }
  template <class... T, class... A> Any getFilterFoldsExcept(A&&...) const { return {}; }
  template <class... T, class... A> Any getFilterTransfersExcept(A&&...) const { return {}; }
  template <class... T, class... A> Any getFirst(A&&...) const { return {}; }
  template <class... T, class... A> Any getFirstChild(A&&...) const { return {}; }
  template <class... T, class... A> Any getFirstValue(A&&...) const { return {}; }
  template <class... T, class... A> Any getFolding(A&&...) const { return {}; }
  template <class... T, class... A> Any getFromUnit(A&&...) const { return {}; }
  template <class... T, class... A> Any getFuncName(A&&...) const { return {}; }
  template <class... T, class... A> Any getGap(A&&...) const { return {}; }
  template <class... T, class... A> Any getGranularity(A&&...) const { return {}; }
  template <class... T, class... A> Any getI1Type(A&&...) const { return {}; }
  template <class... T, class... A> Any getI32IntegerAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getI64Type(A&&...) const { return {}; }
  template <class... T, class... A> Any getI8IntegerAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getId(A&&...) const { return {}; }
  template <class... T, class... A> Any getIdxSets(A&&...) const { return {}; }
  template <class... T, class... A> Any getIn(A&&...) const { return {}; }
  template <class... T, class... A> Any getInclExclList(A&&...) const { return {}; }
  template <class... T, class... A> Any getIncrement(A&&...) const { return {}; }
  template <class... T, class... A> Any getIndexType(A&&...) const { return {}; }
  template <class... T, class... A> Any getIndices(A&&...) const { return {}; }
  template <class... T, class... A> Any getIndicesCoeffDict(A&&...) const { return {}; }
  template <class... T, class... A> Any getIndirectDstAffineMapAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getIndirectDstMemref(A&&...) const { return {}; }
  template <class... T, class... A> Any getIndirectDstMemrefMutable(A&&...) const { return {}; }
  template <class... T, class... A> Any getIndirectSrcAffineMapAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getIndirectSrcMemref(A&&...) const { return {}; }
  template <class... T, class... A> Any getIndirectSrcMemrefMutable(A&&...) const { return {}; }
  template <class... T, class... A> Any getInductionVar(A&&...) const { return {}; }
  template <class... T, class... A> Any getInequality(A&&...) const { return {}; }
  template <class... T, class... A> Any getInitArgs(A&&...) const { return {}; }
  template <class... T, class... A> Any getInits(A&&...) const { return {}; }
  template <class... T, class... A> Any getInput(A&&...) const { return {}; }
  template <class... T, class... A> Any getInputVector(A&&...) const { return {}; }
  template <class... T, class... A> Any getInsertionPoint(A&&...) const { return {}; }
  template <class... T, class... A> Any getInt(A&&...) const { return {}; }
  template <class... T, class... A> Any getIntOrFloatBitWidth(A&&...) const { return {}; }
  template <class... T, class... A> Any getIntegerSet(A&&...) const { return {}; }
  template <class... T, class... A> Any getIntegerType(A&&...) const { return {}; }
  template <class... T, class... A> Any getInterleaveGroupIndex(A&&...) const { return {}; }
  template <class... T, class... A> Any getIterOperands(A&&...) const { return {}; }
  template <class... T, class... A> Any getKey(A&&...) const { return {}; }
  template <class... T, class... A> Any getKeys(A&&...) const { return {}; }
  template <class... T, class... A> Any getKind(A&&...) const { return {}; }
  template <class... T, class... A> Any getLayoutCoeffs(A&&...) const { return {}; }
  template <class... T, class... A> Any getLayoutMap(A&&...) const { return {}; }
  template <class... T, class... A> Any getLhs(A&&...) const { return {}; }
  template <class... T, class... A> Any getListSizes(A&&...) const { return {}; }
  template <class... T, class... A> Any getLoadInductionVar(A&&...) const { return {}; }
  template <class... T, class... A> Any getLoadOrder(A&&...) const { return {}; }
  template <class... T, class... A> Any getLoadSet(A&&...) const { return {}; }
  template <class... T, class... A> Any getLoc(A&&...) const { return {}; }
  template <class... T, class... A> Any getLowerBound(A&&...) const { return {}; }
  template <class... T, class... A> Any getLowerBoundMap(A&&...) const { return {}; }
  template <class... T, class... A> Any getLowerBoundOperands(A&&...) const { return {}; }
  template <class... T, class... A> Any getMap(A&&...) const { return {}; }
  template <class... T, class... A> Any getMapIndices(A&&...) const { return {}; }
  template <class... T, class... A> Any getMapOperands(A&&...) const { return {}; }
  template <class... T, class... A> Any getMask(A&&...) const { return {}; }
  template <class... T, class... A> Any getMaskParameter(A&&...) const { return {}; }
  template <class... T, class... A> Any getMaskValue(A&&...) const { return {}; }
  template <class... T, class... A> Any getMemRef(A&&...) const { return {}; }
  template <class... T, class... A> Any getMemRefMutable(A&&...) const { return {}; }
  template <class... T, class... A> Any getMemViewLayoutMap(A&&...) const { return {}; }
  template <class... T, class... A> Any getMemViewStartAddr(A&&...) const { return {}; }
  template <class... T, class... A> Any getMemory(A&&...) const { return {}; }
  template <class... T, class... A> Any getMemoryIndex(A&&...) const { return {}; }
  template <class... T, class... A> Any getMemref(A&&...) const { return {}; }
  template <class... T, class... A> Any getMulticastInfo(A&&...) const { return {}; }
  template <class... T, class... A> Any getMutableAddrMutable(A&&...) const { return {}; }
  template <class... T, class... A> Any getName(A&&...) const { return {}; }
  template <class... T, class... A> Any getNameAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getNextNode(A&&...) const { return {}; }
  template <class... T, class... A> Any getNextSibling(A&&...) const { return {}; }
  template <class... T, class... A> Any getNodePtr(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumArguments(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumCols(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumConstraints(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumDimVars(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumDims(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumElements(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumEqualities(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumInequalities(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumInputs(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumMaskedElements(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumOperands(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumRegionIterArgs(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumRegions(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumResults(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumSlices(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumSymbolVars(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumSymbols(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumUnmaskedElements(A&&...) const { return {}; }
  template <class... T, class... A> Any getNumberOfChildren(A&&...) const { return {}; }
  template <class... T, class... A> Any getOp(A&&...) const { return {}; }
  template <class... T, class... A> Any getOp1(A&&...) const { return {}; }
  template <class... T, class... A> Any getOp2(A&&...) const { return {}; }
  template <class... T, class... A> Any getOpADataID(A&&...) const { return {}; }
  template <class... T, class... A> Any getOpBDataID(A&&...) const { return {}; }
  template <class... T, class... A> Any getOpCDataID(A&&...) const { return {}; }
  template <class... T, class... A> Any getOpResult(A&&...) const { return {}; }
  template <class... T, class... A> Any getOpResults(A&&...) const { return {}; }
  template <class... T, class... A> Any getOpSpecificMap(A&&...) const { return {}; }
  template <class... T, class... A> Any getOperand(A&&...) const { return {}; }
  template <class... T, class... A> Any getOperandNumber(A&&...) const { return {}; }
  template <class... T, class... A> Any getOperands(A&&...) const { return {}; }
  template <class... T, class... A> Any getOperation(A&&...) const { return {}; }
  template <class... T, class... A> Any getOperations(A&&...) const { return {}; }
  template <class... T, class... A> Any getOps(A&&...) const { return {}; }
  template <class... T, class... A> Any getOut(A&&...) const { return {}; }
  template <class... T, class... A> Any getOwner(A&&...) const { return {}; }
  template <class... T, class... A> Any getPad(A&&...) const { return {}; }
  template <class... T, class... A> Any getPageStartAddrs(A&&...) const { return {}; }
  template <class... T, class... A> Any getParameterDictionary(A&&...) const { return {}; }
  template <class... T, class... A> Any getParentNode(A&&...) const { return {}; }
  template <class... T, class... A> Any getParentOfType(A&&...) const { return {}; }
  template <class... T, class... A> Any getParentOp(A&&...) const { return {}; }
  template <class... T, class... A> Any getParentRegion(A&&...) const { return {}; }
  template <class... T, class... A> Any getPosition(A&&...) const { return {}; }
  template <class... T, class... A> Any getPrecision(A&&...) const { return {}; }
  template <class... T, class... A> Any getPredicate(A&&...) const { return {}; }
  template <class... T, class... A> Any getPredicateAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getPrevNode(A&&...) const { return {}; }
  template <class... T, class... A> Any getReadOnlyRegisterDictionary(A&&...) const { return {}; }
  template <class... T, class... A> Any getReadWriteRegisterDictionary(A&&...) const { return {}; }
  template <class... T, class... A> Any getReductionOp(A&&...) const { return {}; }
  template <class... T, class... A> Any getRegIndicesIfExist(A&&...) const { return {}; }
  template <class... T, class... A> Any getRegLocales(A&&...) const { return {}; }
  template <class... T, class... A> Any getRegLocalesIfExist(A&&...) const { return {}; }
  template <class... T, class... A> Any getRegion(A&&...) const { return {}; }
  template <class... T, class... A> Any getRegionArg(A&&...) const { return {}; }
  template <class... T, class... A> Any getRegionIterArgs(A&&...) const { return {}; }
  template <class... T, class... A> Any getRegionNumber(A&&...) const { return {}; }
  template <class... T, class... A> Any getRegionUnitList(A&&...) const { return {}; }
  template <class... T, class... A> Any getRegions(A&&...) const { return {}; }
  template <class... T, class... A> Any getRepetition(A&&...) const { return {}; }
  template <class... T, class... A> Any getResult(A&&...) const { return {}; }
  template <class... T, class... A> Any getResultNumber(A&&...) const { return {}; }
  template <class... T, class... A> Any getResultTypes(A&&...) const { return {}; }
  template <class... T, class... A> Any getResults(A&&...) const { return {}; }
  template <class... T, class... A> Any getRhs(A&&...) const { return {}; }
  template <class... T, class... A> Any getRightShift(A&&...) const { return {}; }
  template <class... T, class... A> Any getRoot(A&&...) const { return {}; }
  template <class... T, class... A> Any getRotationPosition(A&&...) const { return {}; }
  template <class... T, class... A> Any getSExtValue(A&&...) const { return {}; }
  template <class... T, class... A> Any getSI32IntegerAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getSecond(A&&...) const { return {}; }
  template <class... T, class... A> Any getSelectionMap(A&&...) const { return {}; }
  template <class... T, class... A> Any getSendData(A&&...) const { return {}; }
  template <class... T, class... A> Any getShape(A&&...) const { return {}; }
  template <class... T, class... A> Any getShuffleMode(A&&...) const { return {}; }
  template <class... T, class... A> Any getSignExtend(A&&...) const { return {}; }
  template <class... T, class... A> Any getSingleConstantResult(A&&...) const { return {}; }
  template <class... T, class... A> Any getSliceMaskMap(A&&...) const { return {}; }
  template <class... T, class... A> Any getSplatValue(A&&...) const { return {}; }
  template <class... T, class... A> Any getSrcAffineMapAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getSrcMapIndices(A&&...) const { return {}; }
  template <class... T, class... A> Any getSrcMemRef(A&&...) const { return {}; }
  template <class... T, class... A> Any getSrcMemRefMutable(A&&...) const { return {}; }
  template <class... T, class... A> Any getStartAddr(A&&...) const { return {}; }
  template <class... T, class... A> Any getStartAddress(A&&...) const { return {}; }
  template <class... T, class... A> Any getStartVal(A&&...) const { return {}; }
  template <class... T, class... A> Any getStep(A&&...) const { return {}; }
  template <class... T, class... A> Any getStepAsInt(A&&...) const { return {}; }
  template <class... T, class... A> Any getStoreOrder(A&&...) const { return {}; }
  template <class... T, class... A> Any getStoreSet(A&&...) const { return {}; }
  template <class... T, class... A> Any getStrides(A&&...) const { return {}; }
  template <class... T, class... A> Any getStringAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getStringRef(A&&...) const { return {}; }
  template <class... T, class... A> Any getSubscriptsMap(A&&...) const { return {}; }
  template <class... T, class... A> Any getTerminator(A&&...) const { return {}; }
  template <class... T, class... A> Any getThenBodyBuilder(A&&...) const { return {}; }
  template <class... T, class... A> Any getThenNode(A&&...) const { return {}; }
  template <class... T, class... A> Any getThenRegion(A&&...) const { return {}; }
  template <class... T, class... A> Any getTimeAddrMap(A&&...) const { return {}; }
  template <class... T, class... A> Any getTimeBounds(A&&...) const { return {}; }
  template <class... T, class... A> Any getTimeOffsets(A&&...) const { return {}; }
  template <class... T, class... A> Any getTimeOrder(A&&...) const { return {}; }
  template <class... T, class... A> Any getTimeSet(A&&...) const { return {}; }
  template <class... T, class... A> Any getTimeSymbols(A&&...) const { return {}; }
  template <class... T, class... A> Any getToUnit(A&&...) const { return {}; }
  template <class... T, class... A> Any getTotalDataOriginsCount(A&&...) const { return {}; }
  template <class... T, class... A> Any getTotalElements(A&&...) const { return {}; }
  template <class... T, class... A> Any getTransferOrder(A&&...) const { return {}; }
  template <class... T, class... A> Any getTrueValue(A&&...) const { return {}; }
  template <class... T, class... A> Any getType(A&&...) const { return {}; }
  template <class... T, class... A> Any getTypeAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any getUniqueConstant(A&&...) const { return {}; }
  template <class... T, class... A> Any getUnit(A&&...) const { return {}; }
  template <class... T, class... A> Any getUnitIds(A&&...) const { return {}; }
  template <class... T, class... A> Any getUnits(A&&...) const { return {}; }
  template <class... T, class... A> Any getUnknownLoc(A&&...) const { return {}; }
  template <class... T, class... A> Any getUpperBound(A&&...) const { return {}; }
  template <class... T, class... A> Any getUpperBoundMap(A&&...) const { return {}; }
  template <class... T, class... A> Any getUpperBoundOperands(A&&...) const { return {}; }
  template <class... T, class... A> Any getUseChain(A&&...) const { return {}; }
  template <class... T, class... A> Any getUsers(A&&...) const { return {}; }
  template <class... T, class... A> Any getUses(A&&...) const { return {}; }
  template <class... T, class... A> Any getValue(A&&...) const { return {}; }
  template <class... T, class... A> Any getValueAsDouble(A&&...) const { return {}; }
  template <class... T, class... A> Any getValueToStore(A&&...) const { return {}; }
  template <class... T, class... A> Any getValues(A&&...) const { return {}; }
  template <class... T, class... A> Any getVariable(A&&...) const { return {}; }
  template <class... T, class... A> Any getVectorType(A&&...) const { return {}; }
  template <class... T, class... A> Any getVersion(A&&...) const { return {}; }
  template <class... T, class... A> Any getWaitImmediatelyForAsyncTransfers(A&&...) const { return {}; }
  template <class... T, class... A> Any getXrfRdPtrIncrValAfterMAC(A&&...) const { return {}; }
  template <class... T, class... A> Any has(A&&...) const { return {}; }
  template <class... T, class... A> Any hasAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any hasConstantBounds(A&&...) const { return {}; }
  template <class... T, class... A> Any hasConstantLowerBound(A&&...) const { return {}; }
  template <class... T, class... A> Any hasConstantUpperBound(A&&...) const { return {}; }
  template <class... T, class... A> Any hasIndirectDst(A&&...) const { return {}; }
  template <class... T, class... A> Any hasIndirectSrc(A&&...) const { return {}; }
  template <class... T, class... A> Any hasOneUse(A&&...) const { return {}; }
  template <class... T, class... A> Any has_value(A&&...) const { return {}; }
  template <class... T, class... A> Any hoistCommonConditionals(A&&...) const { return {}; }
  template <class... T, class... A> Any hoistLoopInvariantConditionals(A&&...) const { return {}; }
  template <class... T, class... A> Any inlineRegionBefore(A&&...) const { return {}; }
  template <class... T, class... A> Any insertChildNode(A&&...) const { return {}; }
  template <class... T, class... A> Any insertIfNotExists(A&&...) const { return {}; }
  template <class... T, class... A> Any insertInitializationStmt(A&&...) const { return {}; }
  template <class... T, class... A> Any isAncestor(A&&...) const { return {}; }
  template <class... T, class... A> Any isElseNode(A&&...) const { return {}; }
  template <class... T, class... A> Any isEmpty(A&&...) const { return {}; }
  template <class... T, class... A> Any isFullMask(A&&...) const { return {}; }
  template <class... T, class... A> Any isFunctionOfDim(A&&...) const { return {}; }
  template <class... T, class... A> Any isHyperRectangular(A&&...) const { return {}; }
  template <class... T, class... A> Any isIdentity(A&&...) const { return {}; }
  template <class... T, class... A> Any isIncludeList(A&&...) const { return {}; }
  template <class... T, class... A> Any isIntOrIndex(A&&...) const { return {}; }
  template <class... T, class... A> Any isLeaf(A&&...) const { return {}; }
  template <class... T, class... A> Any isLeafNode(A&&...) const { return {}; }
  template <class... T, class... A> Any isLoopNode(A&&...) const { return {}; }
  template <class... T, class... A> Any isMaskEquivalentToNode(A&&...) const { return {}; }
  template <class... T, class... A> Any isMaskNode(A&&...) const { return {}; }
  template <class... T, class... A> Any isProperAncestor(A&&...) const { return {}; }
  template <class... T, class... A> Any isSingleConstant(A&&...) const { return {}; }
  template <class... T, class... A> Any isThenNode(A&&...) const { return {}; }
  template <class... T, class... A> Any isUnmask(A&&...) const { return {}; }
  template <class... T, class... A> Any isXrfRdRelated(A&&...) const { return {}; }
  template <class... T, class... A> Any isXrfWtRelated(A&&...) const { return {}; }
  template <class... T, class... A> Any lookup(A&&...) const { return {}; }
  template <class... T, class... A> Any lookupOrDefault(A&&...) const { return {}; }
  template <class... T, class... A> Any lookupOrNull(A&&...) const { return {}; }
  template <class... T, class... A> Any lowerDanglingNonComputeOps(A&&...) const { return {}; }
  template <class... T, class... A> Any matchAndRewrite(A&&...) const { return {}; }
  template <class... T, class... A> Any modifyOpInPlace(A&&...) const { return {}; }
  template <class... T, class... A> Any moveAfter(A&&...) const { return {}; }
  template <class... T, class... A> Any moveBefore(A&&...) const { return {}; }
  template <class... T, class... A> Any nameAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any operationsAreEquivalent(A&&...) const { return {}; }
  template <class... T, class... A> Any parseConditional(A&&...) const { return {}; }
  template <class... T, class... A> Any pop(A&&...) const { return {}; }
  template <class... T, class... A> Any pop_back(A&&...) const { return {}; }
  template <class... T, class... A> Any print(A&&...) const { return {}; }
  template <class... T, class... A> Any projectOut(A&&...) const { return {}; }
  template <class... T, class... A> Any push(A&&...) const { return {}; }
  template <class... T, class... A> Any rbegin(A&&...) const { return {}; }
  template <class... T, class... A> Any regionsAreEquivalent(A&&...) const { return {}; }
  template <class... T, class... A> Any removeAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any removeConditionWhenThenElseBranchesMatch(A&&...) const { return {}; }
  template <class... T, class... A> Any removeDuplicateConditionals(A&&...) const { return {}; }
  template <class... T, class... A> Any removeRedundantConstraints(A&&...) const { return {}; }
  template <class... T, class... A> Any rend(A&&...) const { return {}; }
  template <class... T, class... A> Any replaceAllUsesExcept(A&&...) const { return {}; }
  template <class... T, class... A> Any replaceAllUsesWith(A&&...) const { return {}; }
  template <class... T, class... A> Any replaceDimsAndSymbols(A&&...) const { return {}; }
  template <class... T, class... A> Any replaceIfOpByIterArg(A&&...) const { return {}; }
  template <class... T, class... A> Any replaceOpWithNewOp(A&&...) const { return {}; }
  template <class... T, class... A> Any replaceUsesOfWith(A&&...) const { return {}; }
  template <class... T, class... A> Any replaceUsesWithIf(A&&...) const { return {}; }
  template <class... T, class... A> Any resize(A&&...) const { return {}; }
  template <class... T, class... A> Any restoreInsertionPoint(A&&...) const { return {}; }
  template <class... T, class... A> Any run(A&&...) const { return {}; }
  template <class... T, class... A> Any saveInsertionPoint(A&&...) const { return {}; }
  template <class... T, class... A> Any setAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any setAttrs(A&&...) const { return {}; }
  template <class... T, class... A> Any setImmutableAddrAndIncrements(A&&...) const { return {}; }
  template <class... T, class... A> Any setIndices(A&&...) const { return {}; }
  template <class... T, class... A> Any setInsertionPoint(A&&...) const { return {}; }
  template <class... T, class... A> Any setInsertionPointAfter(A&&...) const { return {}; }
  template <class... T, class... A> Any setInsertionPointToEnd(A&&...) const { return {}; }
  template <class... T, class... A> Any setInsertionPointToStart(A&&...) const { return {}; }
  template <class... T, class... A> Any setMemRef(A&&...) const { return {}; }
  template <class... T, class... A> Any setMemViewStartAddr(A&&...) const { return {}; }
  template <class... T, class... A> Any setMemory(A&&...) const { return {}; }
  template <class... T, class... A> Any setOp(A&&...) const { return {}; }
  template <class... T, class... A> Any setOpADataIDAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any setOpBDataIDAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any setOpCDataIDAttr(A&&...) const { return {}; }
  template <class... T, class... A> Any setOperand(A&&...) const { return {}; }
  template <class... T, class... A> Any setOperands(A&&...) const { return {}; }
  template <class... T, class... A> Any setOperation(A&&...) const { return {}; }
  template <class... T, class... A> Any setReuseFlag(A&&...) const { return {}; }
  template <class... T, class... A> Any setReuseInformation(A&&...) const { return {}; }
  template <class... T, class... A> Any setStrides(A&&...) const { return {}; }
  template <class... T, class... A> Any setTimeBounds(A&&...) const { return {}; }
  template <class... T, class... A> Any setTimeOffsets(A&&...) const { return {}; }
  template <class... T, class... A> Any setValue(A&&...) const { return {}; }
  template <class... T, class... A> Any shallowlyMergeConditionals(A&&...) const { return {}; }
  template <class... T, class... A> Any simplifyValueBasedConditionals(A&&...) const { return {}; }
  template <class... T, class... A> Any str(A&&...) const { return {}; }
  template <class... T, class... A> Any substr(A&&...) const { return {}; }
  template <class... T, class... A> Any succeeded(A&&...) const { return {}; }
  template <class... T, class... A> Any top(A&&...) const { return {}; }
  template <class... T, class... A> Any unit(A&&...) const { return {}; }
  template <class... T, class... A> Any use_empty(A&&...) const { return {}; }
  template <class... T, class... A> Any user_begin(A&&...) const { return {}; }
  template <class... T, class... A> Any user_end(A&&...) const { return {}; }
  template <class... T, class... A> Any value(A&&...) const { return {}; }
  template <class... T, class... A> Any walk(A&&...) const { return {}; }
  template <class... T, class... A> Any walkExprs(A&&...) const { return {}; }
  template <class... T, class... A> Any without_terminator(A&&...) const { return {}; }
  static Any absorbed_;
  static Any apply;
  static Any base_unit_core_id_;
  static Any base_unit_corelet_id_;
  static Any base_unit_program_;
  static Any bitSize;
  static Any bytesPerStick;
  static Any coalesce;
  static Any comp_;
  static Any composed_coeff_;
  static Any composite_indirect_load;
  static Any composite_indirect_load_and_store;
  static Any composite_indirect_store;
  static Any composite_load;
  static Any composite_load_and_store;
  static Any composite_store;
  static Any compute_precision_;
  static Any conditional_iter_args_;
  static Any constant_bitstream;
  static Any constant_val;
  static Any curr_unit_core_id_;
  static Any curr_unit_corelet_id_;
  static Any dcc_correct_ear_overflow;
  static Any dcc_ext_ctx_;
  static Any dim_;
  static Any dsc_global_;
  static Any element_bit_width;
  static Any first;
  static Any fold_mode_attr_;
  static Any forop;
  static Any from_operands_;
  static Any g;
  static Any id_;
  static Any indices_;
  static Any indices_ranges_;
  static Any indirect_vector_load;
  static Any indirect_vector_store;
  static Any is_in_region_num;
  static Any iter_arg_;
  static Any l3BurstSize;
  static Any layout_map;
  static Any load_and_extract_scalar;
  static Any location_;
  static Any logical_result_forwarding_;
  static Any mask_operand_;
  static Any mask_val_;
  static Any mem_index_;
  static Any mlir;
  static Any npos;
  static Any numSlicesPerStick;
  static Any num_inputs_;
  static Any num_iters_;
  static Any offset_;
  static Any on_the_fly_conv_precision_;
  static Any opA_forwarding_;
  static Any opA_precision_;
  static Any opB_forwarding_;
  static Any opB_precision_;
  static Any opC_forwarding_;
  static Any opC_precision_;
  static Any op_;
  static Any op_to_update_;
  static Any orig_precision_;
  static Any paged_mem_view_;
  static Any prog_name_;
  static Any receive;
  static Any receiveOp;
  static Any receive_and_extract_scalar;
  static Any regInfoPerUnit;
  static Any repetition;
  static Any repetition_;
  static Any result_forwarding_;
  static Any result_precision_;
  static Any reuse_info_;
  static Any second;
  static Any select;
  static Any send;
  static Any shuffle;
  static Any sign_extend;
  static Any size_;
  static Any splat_;
  static Any src_input_type_str_;
  static Any src_output_type_str_;
  static Any subscripts_map_;
  static Any sum;
  static Any symbolic_vector_load;
  static Any symbolic_vector_store;
  static Any sysDef;
  static Any time_addr_map_;
  static Any to_operands_;
  static Any type_;
  static Any unit_;
  static Any units;
  static Any units_list_;
  static Any update_with_;
  static Any values_;
  static Any vec;
  static Any vec_;
  static Any vector_load;
  static Any vector_store;
  static Any weight_;
};
inline Any Any::absorbed_;
inline Any Any::apply;
inline Any Any::base_unit_core_id_;
inline Any Any::base_unit_corelet_id_;
inline Any Any::base_unit_program_;
inline Any Any::bitSize;
inline Any Any::bytesPerStick;
inline Any Any::coalesce;
inline Any Any::comp_;
inline Any Any::composed_coeff_;
inline Any Any::composite_indirect_load;
inline Any Any::composite_indirect_load_and_store;
inline Any Any::composite_indirect_store;
inline Any Any::composite_load;
inline Any Any::composite_load_and_store;
inline Any Any::composite_store;
inline Any Any::compute_precision_;
inline Any Any::conditional_iter_args_;
inline Any Any::constant_bitstream;
inline Any Any::constant_val;
inline Any Any::curr_unit_core_id_;
inline Any Any::curr_unit_corelet_id_;
inline Any Any::dcc_correct_ear_overflow;
inline Any Any::dcc_ext_ctx_;
inline Any Any::dim_;
inline Any Any::dsc_global_;
inline Any Any::element_bit_width;
inline Any Any::first;
inline Any Any::fold_mode_attr_;
inline Any Any::forop;
inline Any Any::from_operands_;
inline Any Any::g;
inline Any Any::id_;
inline Any Any::indices_;
inline Any Any::indices_ranges_;
inline Any Any::indirect_vector_load;
inline Any Any::indirect_vector_store;
inline Any Any::is_in_region_num;
inline Any Any::iter_arg_;
inline Any Any::l3BurstSize;
inline Any Any::layout_map;
inline Any Any::load_and_extract_scalar;
inline Any Any::location_;
inline Any Any::logical_result_forwarding_;
inline Any Any::mask_operand_;
inline Any Any::mask_val_;
inline Any Any::mem_index_;
inline Any Any::mlir;
inline Any Any::npos;
inline Any Any::numSlicesPerStick;
inline Any Any::num_inputs_;
inline Any Any::num_iters_;
inline Any Any::offset_;
inline Any Any::on_the_fly_conv_precision_;
inline Any Any::opA_forwarding_;
inline Any Any::opA_precision_;
inline Any Any::opB_forwarding_;
inline Any Any::opB_precision_;
inline Any Any::opC_forwarding_;
inline Any Any::opC_precision_;
inline Any Any::op_;
inline Any Any::op_to_update_;
inline Any Any::orig_precision_;
inline Any Any::paged_mem_view_;
inline Any Any::prog_name_;
inline Any Any::receive;
inline Any Any::receiveOp;
inline Any Any::receive_and_extract_scalar;
inline Any Any::regInfoPerUnit;
inline Any Any::repetition;
inline Any Any::repetition_;
inline Any Any::result_forwarding_;
inline Any Any::result_precision_;
inline Any Any::reuse_info_;
inline Any Any::second;
inline Any Any::select;
inline Any Any::send;
inline Any Any::shuffle;
inline Any Any::sign_extend;
inline Any Any::size_;
inline Any Any::splat_;
inline Any Any::src_input_type_str_;
inline Any Any::src_output_type_str_;
inline Any Any::subscripts_map_;
inline Any Any::sum;
inline Any Any::symbolic_vector_load;
inline Any Any::symbolic_vector_store;
inline Any Any::sysDef;
inline Any Any::time_addr_map_;
inline Any Any::to_operands_;
inline Any Any::type_;
inline Any Any::unit_;
inline Any Any::units;
inline Any Any::units_list_;
inline Any Any::update_with_;
inline Any Any::values_;
inline Any Any::vec;
inline Any Any::vec_;
inline Any Any::vector_load;
inline Any Any::vector_store;
inline Any Any::weight_;
template <class L, class R> inline bool operator==(const L&, const R&) { return true; }
template <class L, class R> inline bool operator!=(const L&, const R&) { return true; }
template <class L, class R> inline bool operator<(const L&, const R&) { return true; }
template <class L, class R> inline bool operator>(const L&, const R&) { return true; }
template <class L, class R> inline bool operator<=(const L&, const R&) { return true; }
template <class L, class R> inline bool operator>=(const L&, const R&) { return true; }
inline Any operator+(const Any&, const Any&) { return {}; }
inline Any operator-(const Any&, const Any&) { return {}; }
inline Any operator*(const Any&, const Any&) { return {}; }
inline Any operator/(const Any&, const Any&) { return {}; }
inline Any operator%(const Any&, const Any&) { return {}; }
inline Any operator<<(const Any&, const Any&) { return {}; }
inline Any operator|(const Any&, const Any&) { return {}; }
inline Any operator&(const Any&, const Any&) { return {}; }
inline Any operator^(const Any&, const Any&) { return {}; }
}  // namespace stub
using stub::Any;

template <class... T> struct AccessContainer : Any { template <class... A> AccessContainer(A&&...) {} };
template <class... T> struct ArrayRef : Any { template <class... A> ArrayRef(A&&...) {} };
template <class... T> struct DenseMap : Any { template <class... A> DenseMap(A&&...) {} };
template <class... T> struct SmallPtrSet : Any { template <class... A> SmallPtrSet(A&&...) {} };
template <class... T> struct SmallVector : Any { template <class... A> SmallVector(A&&...) {} };
template <class... T> struct SmallVectorImpl : Any { template <class... A> SmallVectorImpl(A&&...) {} };
template <class... T> struct cast : Any { template <class... A> cast(A&&...) {} };
template <class... T> struct cast_or_null : Any { template <class... A> cast_or_null(A&&...) {} };
template <class... T> struct constructImmutableAddress : Any { template <class... A> constructImmutableAddress(A&&...) {} };
template <class... T> struct constructLoadAndSendStmt : Any { template <class... A> constructLoadAndSendStmt(A&&...) {} };
template <class... T> struct constructLoadAndStoreStmt : Any { template <class... A> constructLoadAndStoreStmt(A&&...) {} };
template <class... T> struct constructReceiveAndStoreStmt : Any { template <class... A> constructReceiveAndStoreStmt(A&&...) {} };
template <class... T> struct dtGetEnv : Any { template <class... A> dtGetEnv(A&&...) {} };
template <class... T> struct dyn_cast : Any { template <class... A> dyn_cast(A&&...) {} };
template <class... T> struct dyn_cast_or_null : Any { template <class... A> dyn_cast_or_null(A&&...) {} };
template <class... T> struct findCandidateForLowering : Any { template <class... A> findCandidateForLowering(A&&...) {} };
template <class... T> struct findExtractScalarOp : Any { template <class... A> findExtractScalarOp(A&&...) {} };
template <class... T> struct generateAffineAddressManipulationStmts : Any { template <class... A> generateAffineAddressManipulationStmts(A&&...) {} };
template <class... T> struct getIndexOfIterArg : Any { template <class... A> getIndexOfIterArg(A&&...) {} };
template <class... T> struct getParentOfType : Any { template <class... A> getParentOfType(A&&...) {} };
template <class... T> struct getStoreOpFromLoadStorePattern : Any { template <class... A> getStoreOpFromLoadStorePattern(A&&...) {} };
template <class... T> struct i : Any { template <class... A> i(A&&...) {} };
template <class... T> struct insertCopyAndAddStmtsHelper : Any { template <class... A> insertCopyAndAddStmtsHelper(A&&...) {} };
template <class... T> struct is_any_of : Any { template <class... A> is_any_of(A&&...) {} };
template <class... T> struct isa : Any { template <class... A> isa(A&&...) {} };
template <class... T> struct isa_and_nonnull : Any { template <class... A> isa_and_nonnull(A&&...) {} };
template <class... T> struct lowerAffineCompositeHelper : Any { template <class... A> lowerAffineCompositeHelper(A&&...) {} };
template <class... T> struct lowerVectorLoadHelper : Any { template <class... A> lowerVectorLoadHelper(A&&...) {} };
template <class... T> struct rn : Any { template <class... A> rn(A&&...) {} };
template <class... T> struct walk : Any { template <class... A> walk(A&&...) {} };

// ⭐ the qualified chains, nested exactly as the bodies write them.
namespace AccessDetailsAffineComposite {
  inline Any constructTimeStepsInfo;
}
namespace AffineExprKind {
  using Constant = Any;
  using DimId = Any;
  using Mod = Any;
}
namespace AffineMap {
  inline Any get;
}
namespace AgenRoutingDirection {
  using BothWays = Any;
  using Clockwise = Any;
  using CounterClockwise = Any;
}
namespace AgenToSentientLoweringPass {
  inline Any constructSetActiveMaskValueOp;
  inline Any createUniformizeRegionsOp;
  template <class... T> struct generateAffineAddressManipulationStmts : Any { template <class... A> generateAffineAddressManipulationStmts(A&&...) {} };
}
namespace ArrayAttr {
  inline Any get;
}
namespace Block {
  using BlockArgListType = Any;
  inline Any iterator;
}
namespace BoolAttr {
  inline Any get;
}
namespace CFGSDataflowConditionalTree {
  inline Any hoistCommonConditionals;
  inline Any opHasSideEffect;
  inline Any shallowlyMergeConditionals;
}
namespace CmpIPredicate {
  inline Any eq;
  inline Any ne;
  inline Any sge;
  inline Any sgt;
  inline Any sle;
  inline Any slt;
}
namespace CmpIPredicateAttr {
  inline Any get;
}
namespace CondNode {
  template <class... T> struct walk : Any { template <class... A> walk(A&&...) {} };
}
namespace ConstantIntOp {
  inline Any create;
}
namespace DataflowRoutingDirection {
  using BothWays = Any;
  using Clockwise = Any;
  using CounterClockwise = Any;
}
namespace DataflowToSentientLoweringPass {
  inline Any lowerL0LXSyncOperationForAGroupOfUnits;
  inline Any lowerL3SyncOperationForAGroupOfUnits;
}
namespace EnumsConversion {
  inline Any senCompToGenericComp;
  inline Any senComponentsToString;
  inline Any stringToSenComponents;
}
namespace EvaluatedValue {
  using ScalarValue = Any;
}
namespace IndexType {
  inline Any get;
}
namespace IntegerSet {
  inline Any get;
}
namespace IntegerType {
  inline Any get;
}
namespace IsaCoreGen {
  using SEN1P5_ISA = Any;
}
namespace LocalOpNode {
  template <class... T> struct walk : Any { template <class... A> walk(A&&...) {} };
}
namespace LogicalResult {
  inline Any failure;
  inline Any success;
}
namespace LoopMaskNode {
  using ActionFuncTy = Any;
  template <class... T> struct walk : Any { template <class... A> walk(A&&...) {} };
}
namespace LoopMaskTree {
  inline Any findNodeFromOp;
}
namespace LoweringXRF {
  inline Any setSentientMacXrfRegIncrAttr;
}
namespace MemoryOperandIndex {
  inline Any kDirDst;
  inline Any kDirSrc;
  inline Any kIndDst;
  inline Any kIndSrc;
  inline Any kMax;
}
namespace MutableAddrSplittingPass {
  inline Any calculateSubscriptsCoefficients;
  inline Any createNewMemViewWithMod;
}
namespace OpBuilder {
  using InsertionGuard = Any;
}
namespace Operation {
  inline Any operand_range;
}
namespace OperationNode {
  namespace WalkOrder {
    inline Any kBFS;
    inline Any kPostOrder;
    inline Any kPreOrder;
    inline Any kReverseBFS;
  }
  inline Any getFirstChild;
  inline Any getLastChild;
  inline Any getNextSibling;
  inline Any getParentNode;
  inline Any getPrevSibling;
}
namespace OperationTreeBase {
  inline Any getRoot;
}
namespace ProgramUnitOp {
  inline Any create;
}
namespace RegType {
  using EAR = Any;
  using EBR = Any;
}
namespace SenComponents {
  using CROSSPTNLINK = Any;
  using HBM = Any;
  using L0LU = Any;
  using L0SU = Any;
  using L3LU = Any;
  using L3SU = Any;
  using LX = Any;
  using LXLU = Any;
  using LXLU0 = Any;
  using LXLU1 = Any;
  using LXSU = Any;
  using LXSU0 = Any;
  using LXSU1 = Any;
  using LXVIRTUALIBR = Any;
  using PE = Any;
  using PT = Any;
  using SFP = Any;
}
namespace SentientBinaryOperator {
  inline Any abs_max;
  inline Any abs_min;
  inline Any add;
  inline Any and0;
  inline Any and_not;
  inline Any fcmp_eq;
  inline Any fcmp_le;
  inline Any fcmp_lt;
  inline Any fcmp_neq;
  inline Any max;
  inline Any min;
  inline Any mul;
  inline Any mul_div2;
  inline Any or0;
  inline Any sub;
  inline Any xnor;
}
namespace SentientComputePort {
  inline Any one;
  inline Any zero;
}
namespace SentientComputePortAttr {
  inline Any get;
}
namespace SentientFMAmode {
  inline Any fused_mul_add;
  inline Any fused_neg_mul_sub;
}
namespace SentientLoadConsumerAttr {
  inline Any get;
}
namespace SentientRegType {
  inline Any unknown;
  inline Any xrfrdptr;
  inline Any xrfwrptr;
}
namespace SentientRegTypeAttr {
  inline Any get;
}
namespace SentientRoutingDirection {
  using BothWays = Any;
  using Clockwise = Any;
  using CounterClockwise = Any;
  using PseudoRandom = Any;
}
namespace SentientRoutingDirectionAttr {
  inline Any get;
}
namespace SentientShuffleMode {
  inline Any masked16b;
  inline Any masked2b;
  inline Any noshuffle;
}
namespace SentientShuffleModeAttr {
  inline Any get;
}
namespace SentientSplatPad {
  inline Any left;
  inline Any none;
}
namespace SentientSyncMode {
  inline Any recv;
  inline Any send;
  inline Any sendrecv;
}
namespace SentientSyncModeAttr {
  inline Any get;
}
namespace SentientTernaryOperator {
  inline Any select;
}
namespace SentientUnary {
  inline Any ln;
  inline Any rec;
  inline Any rsqrt;
}
namespace SpecialTimeBoundValues {
  inline Any kCoalesced;
}
namespace StandardToSentientLoweringPass {
  using ConstructIFRecursively = Any;
}
namespace StringAttr {
  inline Any get;
}
namespace TPMVBase {
  inline Any createEqualityCondition;
  inline Any createInequalityCondition;
}
namespace TPMVVectorLoad {
  inline Any createNewMemOp;
}
namespace TransformLoopToLegalizeForSentientLowering {
  using KSplitParent = Any;
  using KUnroll = Any;
  inline Any analyzeLoop;
  inline Any kNone;
  inline Any transformSCFToAffineLoop;
}
namespace TransformType {
  using KUnroll = Any;
  inline Any kNone;
}
namespace VectorChainElementWiseCompareOperator {
  inline Any compare_eq;
  inline Any compare_ge;
  inline Any compare_gt;
  inline Any compare_le;
  inline Any compare_lt;
  inline Any compare_neq;
}
namespace VectorChainToSentientPESFPLoweringPass {
  namespace SendOpLowering {
    inline Any matchAndRewrite;
  }
  using OpInfo = Any;
  inline Any lowerDanglingNonComputeOpsPESFP;
}
namespace VectorOperand {
  inline Any eraseOp;
  inline Any eraseOperands;
  inline Any getOperand;
  inline Any getOperandWithPrecision;
  inline Any sameBlock;
}
namespace VectorOperandType {
  using ConstantBitstream = Any;
  using ISTATE = Any;
  using LRF = Any;
  using Link = Any;
  using NFWD = Any;
  using XRF = Any;
}
namespace VectorType {
  inline Any get;
}
namespace WalkOrder {
  using PostOrder = Any;
  using PreOrder = Any;
}
namespace WalkResult {
  inline Any advance;
  inline Any interrupt;
  inline Any skip;
}
namespace affine {
  namespace AffineForOp {
    inline Any create;
  }
  namespace AffineYieldOp {
    inline Any create;
  }
  using AffineApplyOp = Any;
  using AffineForOp = Any;
  using AffineForOps = Any;
  using AffineIfOp = Any;
  using AffineStoreOp = Any;
  using AffineVectorLoadOp = Any;
  using AffineYieldOp = Any;
  using FlatAffineValueConstraints = Any;
  inline Any canonicalizeMapAndOperands;
  inline Any fullyComposeAffineMapAndOperands;
}
namespace agen {
  namespace AccessDetailsAffineComposite {
    inline Any constructTimeStepsInfo;
  }
  namespace MemoryOperandIndex {
    inline Any kDirDst;
    inline Any kDirSrc;
    inline Any kMax;
  }
  namespace utils {
    inline Any calculateTimeBounds;
    inline Any calculateTimeOffsets;
    inline Any concatenateMaps;
    inline Any constructIteratorCoeffDict;
    inline Any getMapCoefficients;
    inline Any getSliceIDXsl;
    inline Any isDimAConstantRange;
    inline Any isDimValueZero;
    inline Any isGenericSAMV;
  }
  template <class... T> struct AccessContainer : Any { template <class... A> AccessContainer(A&&...) {} };
  using AccessDetailsAffine = Any;
  using AccessDetailsAffineComposite = Any;
  using CompositeIndirectLoadAndStoreOp = Any;
  using CompositeIndirectLoadOp = Any;
  using CompositeIndirectStoreOp = Any;
  using CompositeLoadAndStoreOp = Any;
  using CompositeLoadOp = Any;
  using CompositeMemoryInterleaveOp = Any;
  using CompositeStoreOp = Any;
  using IndirectVectorLoadOp = Any;
  using IndirectVectorStoreOp = Any;
  using MemoryOperandIndex = Any;
  using SetTransferMaskStateOp = Any;
  using SymbolicVectorLoadOp = Any;
  using SymbolicVectorStoreOp = Any;
  using VectorLoadOp = Any;
  using VectorStoreOp = Any;
  using YieldOp = Any;
}
namespace arith {
  namespace AddIOp {
    inline Any create;
  }
  namespace CmpIOp {
    inline Any create;
  }
  namespace CmpIPredicate {
    inline Any eq;
    inline Any ne;
    inline Any slt;
  }
  namespace ConstantIndexOp {
    inline Any create;
  }
  using ArithDialect = Any;
  using CmpIOp = Any;
  using ConstantIndexOp = Any;
  using ConstantIntOp = Any;
  using ConstantOp = Any;
  using FPToSIOp = Any;
  using SIToFPOp = Any;
  using SelectOp = Any;
  using SubIOp = Any;
}
namespace dataflow {
  namespace CustomMXFloatType {
    inline Any get;
  }
  namespace CustomMXIntType {
    inline Any get;
  }
  namespace GetUnitOp {
    inline Any create;
  }
  namespace utils {
    inline Any dumpModule;
    inline Any getElementType;
    inline Any getElementTypeBitWidth;
    inline Any getIntOrFloatBitWidth;
    inline Any getNewDbgNameFromList;
    inline Any getNumElements;
    inline Any getPrecisionInString;
    inline Any isIntOrFloatType;
  }
  using CreateGroupOp = Any;
  using CreateMulticastGroupOp = Any;
  using CustomVectorType = Any;
  using DataflowRoutingDirectionAttr = Any;
  using GetLocalUnitOp = Any;
  using GetLogicalMemoryViewOp = Any;
  using GetMyUnitInCollectionOp = Any;
  using GetPagedLogicalMemoryViewOp = Any;
  using GetTotalUnitsInCollectionOp = Any;
  using GetUnitOp = Any;
  using ImplicitSyncOnStreamingBufferOp = Any;
  using OpaqueOp = Any;
  using ProgramCollectionOp = Any;
  using ProgramUnitOp = Any;
  using ReceiveOp = Any;
  using ReturnOp = Any;
  using SendOp = Any;
  using SyncRecvOp = Any;
  using SyncSendOp = Any;
  inline Any getDbgNameAttr;
  inline Any setDbgNameAttr;
}
namespace dcc {
  namespace agen {
    namespace utils {
      inline Any constructExplicitTimeLoops;
      inline Any extractConstantOffsetsFromMapForDim;
      inline Any getMaxImmutableAddress;
      inline Any hasValidL3ImmutableAddr;
      inline Any isL3ImmutableAddrAllOdd;
      inline Any isL3ImmutableAddrEven;
      inline Any orderMap;
      inline Any replaceConstOpsInSubscriptsMap;
      inline Any updateMemViewStartAddress;
      inline Any updateSubscriptsAndIndicesForExplicitTimeLoops;
      inline Any updateTimeSetForExplicitDims;
    }
  }
  namespace burst_utils {
    inline Any processBurstSplitOrInterleave;
  }
  namespace dataflow {
    using DuplicateReusedTogglePattern = Any;
  }
  namespace sentient {
    namespace utils {
      inline Any getSentientFoldModeAttrForOperation;
    }
  }
  namespace uniform {
    namespace utils {
      inline Any findUnitType;
      inline Any flattenUniformRegion;
      inline Any getListOfKeyOpsFromUniformMapping;
      inline Any getListOfValueOpsFromUniformMapping;
      inline Any getLocalOrGlobalRegion;
      inline Any getRegionOpAndIndex;
      inline Any getUnitTypeFromUniformMappingAsString;
      inline Any getUnitsPerRegionsAsVectorOfVector;
      inline Any separateKeyValuesBasedOnKeysCorelet;
      inline Any simplifyQueryMapWithSameTarget;
    }
  }
  namespace utils {
    inline Any createForOpWithAdditionalReturnValue;
    inline Any findYieldsResolvingTo;
    inline Any getElementType;
    inline Any getLoopNestLevel;
    inline Any getNewDbgNameFromOp;
    inline Any getNewDbgNameFromOpOrCount;
    inline Any getStickElemNumForOp;
    inline Any getTotalVectorSize;
    inline Any isConstant;
    inline Any isLRFReg;
  }
  using CFGSDataflowConditionalTree = Any;
  using CondNode = Any;
  using ConditionalTree = Any;
  using DccExtContext = Any;
  using OperationEquivalence = Any;
  inline Any getCoreId;
  inline Any getCoreletId;
  inline Any getDirectUnitOpOrgetFirstIndirectUnitOpViaQueryMap;
  inline Any getUnitType;
}
namespace llvm {
  template <class... T> struct DenseMap : Any { template <class... A> DenseMap(A&&...) {} };
  using MapVector = Any;
  template <class... T> struct SmallPtrSet : Any { template <class... A> SmallPtrSet(A&&...) {} };
  template <class... T> struct SmallVector : Any { template <class... A> SmallVector(A&&...) {} };
  template <class... T> struct SmallVectorImpl : Any { template <class... A> SmallVectorImpl(A&&...) {} };
  using Twine = Any;
  inline Any all_of;
  template <class... T> struct cast : Any { template <class... A> cast(A&&...) {} };
  inline Any dbgs;
  template <class... T> struct dyn_cast : Any { template <class... A> dyn_cast(A&&...) {} };
  template <class... T> struct dyn_cast_or_null : Any { template <class... A> dyn_cast_or_null(A&&...) {} };
  inline Any errs;
  inline Any hasSingleElement;
  template <class... T> struct isa : Any { template <class... A> isa(A&&...) {} };
  inline Any outs;
  inline Any sort;
  inline Any to_vector;
  inline Any zip;
}
namespace mlir {
  namespace Float8E4M3FNType {
    inline Any get;
  }
  namespace Float8E5M2Type {
    inline Any get;
  }
  namespace affine {
    using AffineForOp = Any;
    using AffineIfOp = Any;
    inline Any expandAffineMap;
  }
  namespace agen {
    namespace utils {
      inline Any constructIteratorCoeffDict;
    }
  }
  namespace arith {
    namespace AddIOp {
      inline Any create;
    }
    namespace CmpIOp {
      inline Any create;
    }
    namespace CmpIPredicate {
      inline Any eq;
      inline Any ne;
      inline Any sge;
      inline Any sgt;
      inline Any sle;
      inline Any slt;
    }
    namespace ConstantIndexOp {
      inline Any create;
    }
    namespace DivSIOp {
      inline Any create;
    }
    namespace OrIOp {
      inline Any create;
    }
    namespace SubIOp {
      inline Any create;
    }
    using AddIOp = Any;
    using AndIOp = Any;
    using CmpIOp = Any;
    using CmpIPredicate = Any;
    using ConstantIndexOp = Any;
    using ConstantIntOp = Any;
    using ConstantOp = Any;
    using DivSIOp = Any;
    using FPToSIOp = Any;
    using MulIOp = Any;
    using OrIOp = Any;
    using SIToFPOp = Any;
    using SelectOp = Any;
    using SubIOp = Any;
  }
  namespace dataflow {
    namespace GetLogicalMemoryViewOp {
      inline Any create;
    }
    namespace utils {
      inline Any getElementType;
      inline Any getIterArgChain;
      inline Any setIterArgInit;
    }
    using DataflowDialect = Any;
    using GetLogicalMemoryViewOp = Any;
    using GetUnitOp = Any;
    using ReceiveOp = Any;
    using SendOp = Any;
  }
  namespace func {
    using FuncDialect = Any;
  }
  namespace memref {
    using MemRefDialect = Any;
  }
  namespace presburger {
    namespace BoundType {
      using LB = Any;
      using UB = Any;
    }
  }
  namespace scf {
    namespace ForOp {
      inline Any getOperationName;
    }
    namespace IfOp {
      inline Any create;
    }
    using ForOp = Any;
    using IfOp = Any;
  }
  namespace sentient {
    namespace CmpIPredicateAttr {
      inline Any get;
    }
    namespace SentientRegTypeAttr {
      inline Any get;
    }
    using SentientDialect = Any;
  }
  namespace symbol {
    using CreateSymbolOp = Any;
    using SymbolDialect = Any;
    using SymbolQueryMapOp = Any;
  }
  namespace uniform {
    namespace DefImmutableMappingOp {
      inline Any create;
    }
    namespace UniformizeRegionsOp {
      inline Any create;
    }
    namespace YieldOp {
      inline Any create;
    }
    using DefImmutableMappingOp = Any;
    using QueryMapOp = Any;
    using UniformDialect = Any;
    using YieldOp = Any;
  }
  namespace vectorchain {
    namespace VectorChainBinaryOperator {
      inline Any abs_max;
      inline Any abs_min;
      inline Any add;
      inline Any and0;
      inline Any and_not;
      inline Any max;
      inline Any min;
      inline Any mul;
      inline Any mul_div2;
      inline Any or0;
      inline Any sub;
      inline Any xnor;
    }
    namespace VectorChainScanOpEvalOrders {
      inline Any left_to_right;
    }
    using VectorChainDialect = Any;
  }
  using AffineMap = Any;
  using Attribute = Any;
  using BlockArgument = Any;
  using BoolAttr = Any;
  using ConversionPattern = Any;
  using ConversionPatternRewriter = Any;
  using Float8E4M3FNType = Any;
  using Float8E5M2Type = Any;
  using IntegerAttr = Any;
  using LogicalResult = Any;
  using LoopLikeOpInterface = Any;
  using MLIRContext = Any;
  using OpBuilder = Any;
  using OpOperand = Any;
  using Operation = Any;
  using Region = Any;
  using RewritePatternSet = Any;
  using Type = Any;
  using TypeRange = Any;
  using Value = Any;
  using ValueRange = Any;
  template <class... T> struct cast : Any { template <class... A> cast(A&&...) {} };
  template <class... T> struct dyn_cast : Any { template <class... A> dyn_cast(A&&...) {} };
  template <class... T> struct isa : Any { template <class... A> isa(A&&...) {} };
}
namespace scf {
  namespace ForOp {
    inline Any create;
  }
  namespace IfOp {
    inline Any create;
  }
  namespace YieldOp {
    inline Any create;
  }
  using ForOp = Any;
  using IfOp = Any;
  using ParallelOp = Any;
  using SCFDialect = Any;
  using YieldOp = Any;
}
namespace sentient {
  namespace AddOp {
    inline Any create;
  }
  namespace BinaryOp {
    inline Any create;
  }
  namespace ConstantOp {
    inline Any create;
  }
  namespace ForOp {
    inline Any create;
  }
  namespace IfOp {
    inline Any create;
  }
  namespace IncrMaskOp {
    inline Any create;
  }
  namespace LoadAndExtractScalarOp {
    inline Any create;
  }
  namespace LoadAndSendOp {
    inline Any create;
  }
  namespace LoadAndStoreOp {
    inline Any create;
    inline Any getIsIBRWriteAttrStrName;
  }
  namespace LoadComputeAndSendOp {
    inline Any create;
  }
  namespace LogicalPortOp {
    inline Any create;
  }
  namespace MacOp {
    inline Any create;
  }
  namespace MulOp {
    inline Any create;
  }
  namespace OpaqueOp {
    inline Any create;
  }
  namespace ReceiveAndExtractScalarOp {
    inline Any create;
  }
  namespace ReceiveAndStoreOp {
    inline Any create;
  }
  namespace SentientComputePortAttr {
    inline Any get;
  }
  namespace SetActiveMaskValueOp {
    inline Any create;
  }
  namespace SetMaskOp {
    inline Any create;
  }
  namespace SetSendDestinationOp {
    inline Any create;
  }
  namespace SplatOp {
    inline Any create;
  }
  namespace SubOp {
    inline Any create;
  }
  namespace SyncOp {
    inline Any create;
  }
  namespace TernaryOp {
    inline Any create;
  }
  namespace UnaryOp {
    inline Any create;
  }
  namespace VectorConstantOp {
    inline Any create;
  }
  namespace YieldOp {
    inline Any create;
  }
  using BinaryOp = Any;
  using ConstantOp = Any;
  using ForOp = Any;
  using IfOp = Any;
  using LoadAndSendOp = Any;
  using LoadAndStoreOp = Any;
  using MacOp = Any;
  using ReceiveAndStoreOp = Any;
  using SentientComputePortAttr = Any;
  using SentientRoutingDirectionAttr = Any;
  using SetSendDestinationOp = Any;
  using SplatOp = Any;
  using YieldOp = Any;
  inline Any getCoreOrCoreletID;
}
namespace std {
  namespace string {
    inline Any npos;
  }
  inline Any all_of;
  inline Any cout;
  inline Any endl;
  inline Any find;
  inline Any forward;
  inline Any function;
  inline Any get;
  inline Any log2;
  inline Any make_pair;
  inline Any make_tuple;
  inline Any map;
  inline Any move;
  inline Any nullopt;
  inline Any optional;
  inline Any pair;
  inline Any prev;
  inline Any reverse_copy;
  inline Any set;
  inline Any stack;
  inline Any string;
  inline Any swap;
  inline Any tie;
  inline Any to_string;
  inline Any tuple;
  inline Any unordered_set;
  inline Any vector;
}
namespace symbol {
  using CreateSymbolOp = Any;
  using SymbolImmutableMappingOp = Any;
  using SymbolQueryMapOp = Any;
}
namespace trace {
  using ProgramUnitTraceOp = Any;
  using ReturnOp = Any;
}
namespace uniform {
  namespace DefImmutableMappingOp {
    inline Any create;
  }
  namespace QueryMapOp {
    inline Any create;
  }
  namespace UniformizeRegionsOp {
    inline Any create;
  }
  namespace YieldOp {
    inline Any create;
  }
  using DefImmutableMappingOp = Any;
  using EqualizePatternOp = Any;
  using QueryMapOp = Any;
  using UniformizeRegionsOp = Any;
}
namespace utils {
  inline Any createForOpWithAdditionalReturnValue;
}
namespace vector {
  using LoadOp = Any;
  using StoreOp = Any;
}
namespace vectorchain {
  namespace VectorChainBinaryOperator {
    inline Any mul;
  }
  namespace utils {
    inline Any isCustomVectorTrivialShuffle;
    inline Any isFirstElemSplat;
    inline Any isPadLeftFor8FirstElemSplat;
    inline Any isShuffleNFWDVersion0;
    inline Any isShuffleNFWDVersion2;
    inline Any isSplatOfFirstVar;
  }
  using BinaryOp = Any;
  using CastOp = Any;
  using CoalesceOp = Any;
  using ConstantBitstreamOp = Any;
  using CreateAffineMaskOp = Any;
  using ElementWiseCompareOp = Any;
  using ElementWiseSelectionOp = Any;
  using ExpEstimateOp = Any;
  using FastExpOp = Any;
  using LnEstimateOp = Any;
  using MergeOp = Any;
  using MultiplyAndAccumulateOp = Any;
  using MultiplyOp = Any;
  using NegOp = Any;
  using PackOp = Any;
  using RecEstimateOp = Any;
  using RotateOp = Any;
  using RsqrtEstimateOp = Any;
  using ScanWithGapOp = Any;
  using SelectOp = Any;
  using ShuffleOp = Any;
  using SigmoidEstimateOp = Any;
  using TanhEstimateOp = Any;
  inline Any createSplatOperation;
  inline Any getMaskValueForNonPT;
  inline Any resetSentientFMAsIfExists;
  inline Any validateLoweringAndSetMissingParameters;
}

template <class... T, class... A> Any isa(A&&...) { return {}; }
template <class... T, class... A> Any dyn_cast(A&&...) { return {}; }
template <class... T, class... A> Any dyn_cast_or_null(A&&...) { return {}; }
template <class... T, class... A> Any cast(A&&...) { return {}; }
template <class... T, class... A> Any cast_or_null(A&&...) { return {}; }
template <class... T, class... A> Any isa_and_nonnull(A&&...) { return {}; }
template <class... T, class... A> Any AccessDetailsAffine(A&&...) { return {}; }
template <class... T, class... A> Any AccessDetailsBase(A&&...) { return {}; }
template <class... T, class... A> Any CFGSM(A&&...) { return {}; }
template <class... T, class... A> Any ConditionalSimplificationManager(A&&...) { return {}; }
template <class... T, class... A> Any ConstructIFRecursively(A&&...) { return {}; }
template <class... T, class... A> Any DEBUG_WITH_TYPE(A&&...) { return {}; }
template <class... T, class... A> Any DT_CHECK(A&&...) { return {}; }
template <class... T, class... A> Any DT_CHECK_MSG(A&&...) { return {}; }
template <class... T, class... A> Any DT_ERROR(A&&...) { return {}; }
template <class... T, class... A> Any ExtendUnitNameToCorelet(A&&...) { return {}; }
template <class... T, class... A> Any FMA(A&&...) { return {}; }
template <class... T, class... A> Any FlatteningLocalRegionsTree(A&&...) { return {}; }
template <class... T, class... A> Any IMMCOPY(A&&...) { return {}; }
template <class... T, class... A> Any LLVM_DEBUG(A&&...) { return {}; }
template <class... T, class... A> Any LMTLoopNode(A&&...) { return {}; }
template <class... T, class... A> Any LOGICAL(A&&...) { return {}; }
template <class... T, class... A> Any LayoutExprMap(A&&...) { return {}; }
template <class... T, class... A> Any Loop(A&&...) { return {}; }
template <class... T, class... A> Any LowerAddIOpToSentient(A&&...) { return {}; }
template <class... T, class... A> Any LowerConstantIndexToSentient(A&&...) { return {}; }
template <class... T, class... A> Any LowerConstantIntToSentient(A&&...) { return {}; }
template <class... T, class... A> Any LowerLogicalOpToSentient(A&&...) { return {}; }
template <class... T, class... A> Any LowerSelectOpToSentient(A&&...) { return {}; }
template <class... T, class... A> Any LowerSubIOpToSentient(A&&...) { return {}; }
template <class... T, class... A> Any LowerSymbolQueryMap(A&&...) { return {}; }
template <class... T, class... A> Any MaskNode(A&&...) { return {}; }
template <class... T, class... A> Any OperandReuse(A&&...) { return {}; }
template <class... T, class... A> Any SimplifyOrIOp(A&&...) { return {}; }
template <class... T, class... A> Any StringRef(A&&...) { return {}; }
template <class... T, class... A> Any TPMVComposite(A&&...) { return {}; }
template <class... T, class... A> Any TPMVVector(A&&...) { return {}; }
template <class... T, class... A> Any TransformationConditionalTree(A&&...) { return {}; }
template <class... T, class... A> Any TypeRange(A&&...) { return {}; }
template <class... T, class... A> Any ValueRange(A&&...) { return {}; }
template <class... T, class... A> Any addConstraintsForIVRanges(A&&...) { return {}; }
template <class... T, class... A> Any addLoadChainToDeleteList(A&&...) { return {}; }
template <class... T, class... A> Any addStoreInputToDeleteList(A&&...) { return {}; }
template <class... T, class... A> Any addTimeDimIndicesRanges(A&&...) { return {}; }
template <class... T, class... A> Any adjustForEvenImmutableAddr(A&&...) { return {}; }
template <class... T, class... A> Any adjustMutableAddrInitForIndirect(A&&...) { return {}; }
template <class... T, class... A> Any adjustMutableAddrInitForStride(A&&...) { return {}; }
template <class... T, class... A> Any analyzeAndConstructValidPages(A&&...) { return {}; }
template <class... T, class... A> Any analyzeAndFillOperandForwarding(A&&...) { return {}; }
template <class... T, class... A> Any analyzeAndFillResultForwarding(A&&...) { return {}; }
template <class... T, class... A> Any analyzeAndTransform(A&&...) { return {}; }
template <class... T, class... A> Any analyzeLoop(A&&...) { return {}; }
template <class... T, class... A> Any analyzeNonComputeOpsForFusion(A&&...) { return {}; }
template <class... T, class... A> Any analyzeValidPages(A&&...) { return {}; }
template <class... T, class... A> Any applyPartialConversion(A&&...) { return {}; }
template <class... T, class... A> Any areAnyOf(A&&...) { return {}; }
template <class... T, class... A> Any areCoreletsDifferent(A&&...) { return {}; }
template <class... T, class... A> Any areShallowlyMergeable(A&&...) { return {}; }
template <class... T, class... A> Any areXrfAccessesLegal(A&&...) { return {}; }
template <class... T, class... A> Any base_unit_core_id_(A&&...) { return {}; }
template <class... T, class... A> Any base_unit_corelet_id_(A&&...) { return {}; }
template <class... T, class... A> Any bindDims(A&&...) { return {}; }
template <class... T, class... A> Any bindSymbols(A&&...) { return {}; }
template <class... T, class... A> Any builder(A&&...) { return {}; }
template <class... T, class... A> Any builder_inner_region(A&&...) { return {}; }
template <class... T, class... A> Any builder_region(A&&...) { return {}; }
template <class... T, class... A> Any builder_region0(A&&...) { return {}; }
template <class... T, class... A> Any builder_region1(A&&...) { return {}; }
template <class... T, class... A> Any builder_region2(A&&...) { return {}; }
template <class... T, class... A> Any builder_regioni(A&&...) { return {}; }
template <class... T, class... A> Any calculateDimWeights(A&&...) { return {}; }
template <class... T, class... A> Any calculateFullShift(A&&...) { return {}; }
template <class... T, class... A> Any calculateIndicesRanges(A&&...) { return {}; }
template <class... T, class... A> Any calculatePartialShift(A&&...) { return {}; }
template <class... T, class... A> Any calculateShifts(A&&...) { return {}; }
template <class... T, class... A> Any calculateStartElementsForPage(A&&...) { return {}; }
template <class... T, class... A> Any calculateSubscriptsCoefficients(A&&...) { return {}; }
template <class... T, class... A> Any checkAffineSetConstraints(A&&...) { return {}; }
template <class... T, class... A> Any checkAndAddToRegionOpMap(A&&...) { return {}; }
template <class... T, class... A> Any checkBasicConditions(A&&...) { return {}; }
template <class... T, class... A> Any checkCompositeRegion(A&&...) { return {}; }
template <class... T, class... A> Any checkIndirectMemViewForExtractOp(A&&...) { return {}; }
template <class... T, class... A> Any checkShuffle(A&&...) { return {}; }
template <class... T, class... A> Any checkStoreOpFromExtractPattern(A&&...) { return {}; }
template <class... T, class... A> Any checkValidityOfPackAndShuffleLowering(A&&...) { return {}; }
template <class... T, class... A> Any cleanup(A&&...) { return {}; }
template <class... T, class... A> Any cleanupTriviallyRedundantSetSendDestination(A&&...) { return {}; }
template <class... T, class... A> Any cloneOpsForRegions(A&&...) { return {}; }
template <class... T, class... A> Any cloneStartAddrOutsideLoop(A&&...) { return {}; }
template <class... T, class... A> Any cloneUseChain(A&&...) { return {}; }
template <class... T, class... A> Any coalesceTimeDimensions(A&&...) { return {}; }
template <class... T, class... A> Any collectMutableIterArg(A&&...) { return {}; }
template <class... T, class... A> Any compressUnusedSymbols(A&&...) { return {}; }
template <class... T, class... A> Any computeLoops(A&&...) { return {}; }
template <class... T, class... A> Any computeUnitPrecision(A&&...) { return {}; }
template <class... T, class... A> Any compute_ops_patterns(A&&...) { return {}; }
template <class... T, class... A> Any cond_builder(A&&...) { return {}; }
template <class... T, class... A> Any constValToField(A&&...) { return {}; }
template <class... T, class... A> Any const_builder(A&&...) { return {}; }
template <class... T, class... A> Any constraints(A&&...) { return {}; }
template <class... T, class... A> Any constructAffineCompDetailsAndAddrs(A&&...) { return {}; }
template <class... T, class... A> Any constructChunkAndShuffleInfo(A&&...) { return {}; }
template <class... T, class... A> Any constructConditionals(A&&...) { return {}; }
template <class... T, class... A> Any constructExtentAndTotalElements(A&&...) { return {}; }
template <class... T, class... A> Any constructIndices(A&&...) { return {}; }
template <class... T, class... A> Any constructIteratorCoefficients(A&&...) { return {}; }
template <class... T, class... A> Any constructLdOrStType(A&&...) { return {}; }
template <class... T, class... A> Any constructLoadAndExtractScalarOp(A&&...) { return {}; }
template <class... T, class... A> Any constructReceiveAndExtractScalarOp(A&&...) { return {}; }
template <class... T, class... A> Any constructSetActiveMaskValueOp(A&&...) { return {}; }
template <class... T, class... A> Any constructSymbolicDetailsAndAddrs(A&&...) { return {}; }
template <class... T, class... A> Any constructTimeLoopsAndVectorOperations(A&&...) { return {}; }
template <class... T, class... A> Any constructValidPage(A&&...) { return {}; }
template <class... T, class... A> Any conversion(A&&...) { return {}; }
template <class... T, class... A> Any convertStringToType(A&&...) { return {}; }
template <class... T, class... A> Any convertTypeToString(A&&...) { return {}; }
template <class... T, class... A> Any copyLoopBody(A&&...) { return {}; }
template <class... T, class... A> Any createConditionalsForPartition(A&&...) { return {}; }
template <class... T, class... A> Any createConditionsForHyperRectSubscripts(A&&...) { return {}; }
template <class... T, class... A> Any createConditionsForNonHyperRectSubscripts(A&&...) { return {}; }
template <class... T, class... A> Any createDummyYieldInElseReg(A&&...) { return {}; }
template <class... T, class... A> Any createEqualityCondition(A&&...) { return {}; }
template <class... T, class... A> Any createExplicitTimeLoops(A&&...) { return {}; }
template <class... T, class... A> Any createForOpWithAdditionalReturnValue(A&&...) { return {}; }
template <class... T, class... A> Any createForOpWithReturnValue(A&&...) { return {}; }
template <class... T, class... A> Any createIfOpFromMapping(A&&...) { return {}; }
template <class... T, class... A> Any createIfOpWithReturnValue(A&&...) { return {}; }
template <class... T, class... A> Any createInequalityCondition(A&&...) { return {}; }
template <class... T, class... A> Any createIterArgsForConditionals(A&&...) { return {}; }
template <class... T, class... A> Any createNewMemOp(A&&...) { return {}; }
template <class... T, class... A> Any createNewMemViewWithMod(A&&...) { return {}; }
template <class... T, class... A> Any createNewSubscriptsFromStartElements(A&&...) { return {}; }
template <class... T, class... A> Any createNonPagedMemView(A&&...) { return {}; }
template <class... T, class... A> Any createPartitions(A&&...) { return {}; }
template <class... T, class... A> Any createSentientConstants(A&&...) { return {}; }
template <class... T, class... A> Any createSplatOperation(A&&...) { return {}; }
template <class... T, class... A> Any createUniformRegionsWithTwoRegionsNoResult(A&&...) { return {}; }
template <class... T, class... A> Any createUniformizeRegionsOp(A&&...) { return {}; }
template <class... T, class... A> Any curr_unit_core_id_(A&&...) { return {}; }
template <class... T, class... A> Any curr_unit_corelet_id_(A&&...) { return {}; }
template <class... T, class... A> Any dccExtContext(A&&...) { return {}; }
template <class... T, class... A> Any defined(A&&...) { return {}; }
template <class... T, class... A> Any deleteAncestorsIfPossible(A&&...) { return {}; }
template <class... T, class... A> Any e002_setCoalescedBoundValues(A&&...) { return {}; }
template <class... T, class... A> Any e031_checkIndirectMemViewForExtractOp(A&&...) { return {}; }
template <class... T, class... A> Any e032_findExtractScalarOp(A&&...) { return {}; }
template <class... T, class... A> Any e033_getLoadConsumer(A&&...) { return {}; }
template <class... T, class... A> Any e034_setldtype(A&&...) { return {}; }
template <class... T, class... A> Any e035_generateSetSendDestinationStmts(A&&...) { return {}; }
template <class... T, class... A> Any e036_getStoreOpFromLoadStorePattern(A&&...) { return {}; }
template <class... T, class... A> Any e037_findCandidateForLowering(A&&...) { return {}; }
template <class... T, class... A> Any e038_addLoadChainToDeleteList(A&&...) { return {}; }
template <class... T, class... A> Any e044_lowerOpaqueOperation(A&&...) { return {}; }
template <class... T, class... A> Any e047_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e049_LowerAddIOpToSentient(A&&...) { return {}; }
template <class... T, class... A> Any e050_LowerSubIOpToSentient(A&&...) { return {}; }
template <class... T, class... A> Any e051_LowerMulIOpToSentient(A&&...) { return {}; }
template <class... T, class... A> Any e053_LowerConstantIndexToSentient(A&&...) { return {}; }
template <class... T, class... A> Any e054_LowerConstantIntToSentient(A&&...) { return {}; }
template <class... T, class... A> Any e055_getId(A&&...) { return {}; }
template <class... T, class... A> Any e056_getAbsorbtionFlag(A&&...) { return {}; }
template <class... T, class... A> Any e057_setReuseFlag(A&&...) { return {}; }
template <class... T, class... A> Any e059_isSentientBinaryLogicalOp(A&&...) { return {}; }
template <class... T, class... A> Any e060_getInputPrecisionFromOperand(A&&...) { return {}; }
template <class... T, class... A> Any e061_getResultPrecisionFromOperands(A&&...) { return {}; }
template <class... T, class... A> Any e062_getComputePrecisionOfOp(A&&...) { return {}; }
template <class... T, class... A> Any e063_hasConstantBounds(A&&...) { return {}; }
template <class... T, class... A> Any e065_fuseCompareAndSelectIntoMinOrMax(A&&...) { return {}; }
template <class... T, class... A> Any e066_resetSentientFMAsIfExists(A&&...) { return {}; }
template <class... T, class... A> Any e067_redefineConstantVectors(A&&...) { return {}; }
template <class... T, class... A> Any e071_getOperandFromReceiveOp(A&&...) { return {}; }
template <class... T, class... A> Any e072_getOperandFromSendOp(A&&...) { return {}; }
template <class... T, class... A> Any e074_sameBlock(A&&...) { return {}; }
template <class... T, class... A> Any e075_eraseOp(A&&...) { return {}; }
template <class... T, class... A> Any e076_fuseComputeOps(A&&...) { return {}; }
template <class... T, class... A> Any e077_walk(A&&...) { return {}; }
template <class... T, class... A> Any e089_getMaskValueForPT(A&&...) { return {}; }
template <class... T, class... A> Any e090_getXrfValue(A&&...) { return {}; }
template <class... T, class... A> Any e091_getForOpBound(A&&...) { return {}; }
template <class... T, class... A> Any e092_setSentientMacXrfRegIncrAttr(A&&...) { return {}; }
template <class... T, class... A> Any e093_replaceAndEraseDummyMacOps(A&&...) { return {}; }
template <class... T, class... A> Any e094_computeUnitPrecision(A&&...) { return {}; }
template <class... T, class... A> Any e095_isOperationSelected(A&&...) { return {}; }
template <class... T, class... A> Any e098_getLhsRhsOfEQPredicate(A&&...) { return {}; }
template <class... T, class... A> Any e108_partitionUnits(A&&...) { return {}; }
template <class... T, class... A> Any e109_performFullUnroll(A&&...) { return {}; }
template <class... T, class... A> Any e110_getConstantTripCount(A&&...) { return {}; }
template <class... T, class... A> Any e111_getMaxMutableRange(A&&...) { return {}; }
template <class... T, class... A> Any e112_getMaxImmutableRange(A&&...) { return {}; }
template <class... T, class... A> Any e113_isEligibleForSplitting(A&&...) { return {}; }
template <class... T, class... A> Any e114_sortDataBasedOnWeight(A&&...) { return {}; }
template <class... T, class... A> Any e116_getMaxImmutableRange(A&&...) { return {}; }
template <class... T, class... A> Any e118_removeValuesFromIndices(A&&...) { return {}; }
template <class... T, class... A> Any e119_replaceDimsInMapWithSyms(A&&...) { return {}; }
template <class... T, class... A> Any e122_setBuilderToInsertRef(A&&...) { return {}; }
template <class... T, class... A> Any e123_calculateStartElementsForPage(A&&...) { return {}; }
template <class... T, class... A> Any e124_createNonPagedMemView(A&&...) { return {}; }
template <class... T, class... A> Any e125_cloneMemViewIfNonPaged(A&&...) { return {}; }
template <class... T, class... A> Any e126_getUseChain(A&&...) { return {}; }
template <class... T, class... A> Any e127_cloneUseChain(A&&...) { return {}; }
template <class... T, class... A> Any e129_eraseMemOpAndUseChain(A&&...) { return {}; }
template <class... T, class... A> Any e130_getStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any e131_addTimeDimIndicesRanges(A&&...) { return {}; }
template <class... T, class... A> Any e132_identifyTimeDimForExplicitLoops(A&&...) { return {}; }
template <class... T, class... A> Any e139_run(A&&...) { return {}; }
template <class... T, class... A> Any e140_removeCoresCoreletsFoldsFromProgramUnit(A&&...) { return {}; }
template <class... T, class... A> Any e143_constructExtentAndTotalElements(A&&...) { return {}; }
template <class... T, class... A> Any e144_constructLdOrStType(A&&...) { return {}; }
template <class... T, class... A> Any e145_initializeMemViewInfo(A&&...) { return {}; }
template <class... T, class... A> Any e146_constructIndices(A&&...) { return {}; }
template <class... T, class... A> Any e147_constructIteratorCoefficients(A&&...) { return {}; }
template <class... T, class... A> Any e148_computeBurstAndGroup(A&&...) { return {}; }
template <class... T, class... A> Any e151_checkCompositeRegion(A&&...) { return {}; }
template <class... T, class... A> Any e152_checkStoreOpFromExtractPattern(A&&...) { return {}; }
template <class... T, class... A> Any e153_isLoadAndExtractScalarPattern(A&&...) { return {}; }
template <class... T, class... A> Any e154_isReceiveAndExtractScalarPattern(A&&...) { return {}; }
template <class... T, class... A> Any e155_updateSymbolicAccessDetails(A&&...) { return {}; }
template <class... T, class... A> Any e156_getStoreProducer(A&&...) { return {}; }
template <class... T, class... A> Any e162_insertIfNotExists(A&&...) { return {}; }
template <class... T, class... A> Any e163_getMaskValueConstantForNonPT(A&&...) { return {}; }
template <class... T, class... A> Any e164_checkValidityOfPackAndShuffleLowering(A&&...) { return {}; }
template <class... T, class... A> Any e165_getMergeTypeFromIndices(A&&...) { return {}; }
template <class... T, class... A> Any e166_getOperandFromConstantOp(A&&...) { return {}; }
template <class... T, class... A> Any e167_getOperandFromConstantBitstreamOp(A&&...) { return {}; }
template <class... T, class... A> Any e168_getOperandFromNegOp(A&&...) { return {}; }
template <class... T, class... A> Any e169_getName(A&&...) { return {}; }
template <class... T, class... A> Any e170_getLayoutMapAndIndices(A&&...) { return {}; }
template <class... T, class... A> Any e171_isMaskEquivalentToNode(A&&...) { return {}; }
template <class... T, class... A> Any e172_areXrfAccessesLegal(A&&...) { return {}; }
template <class... T, class... A> Any e173_insertConstAndAddOps(A&&...) { return {}; }
template <class... T, class... A> Any e174_isXrfRelated(A&&...) { return {}; }
template <class... T, class... A> Any e175_updateYieldArgs(A&&...) { return {}; }
template <class... T, class... A> Any e176_opHasSideEffect(A&&...) { return {}; }
template <class... T, class... A> Any e177_mergeShallow(A&&...) { return {}; }
template <class... T, class... A> Any e179_clear(A&&...) { return {}; }
template <class... T, class... A> Any e180_traverseRegion(A&&...) { return {}; }
template <class... T, class... A> Any e181_inRegionEmpty(A&&...) { return {}; }
template <class... T, class... A> Any e182_cloneOpsForRegions(A&&...) { return {}; }
template <class... T, class... A> Any e183_expandAffineApplyOps(A&&...) { return {}; }
template <class... T, class... A> Any e184_getLoopTripCount(A&&...) { return {}; }
template <class... T, class... A> Any e185_hasMutableAddrOverflow(A&&...) { return {}; }
template <class... T, class... A> Any e186_calculatePartitionSizes(A&&...) { return {}; }
template <class... T, class... A> Any e187_constructConditionals(A&&...) { return {}; }
template <class... T, class... A> Any e189_synthesizeTimeInfo(A&&...) { return {}; }
template <class... T, class... A> Any e190_createExplicitTimeLoops(A&&...) { return {}; }
template <class... T, class... A> Any e191_calculateFullShift(A&&...) { return {}; }
template <class... T, class... A> Any e192_calculateDimWeights(A&&...) { return {}; }
template <class... T, class... A> Any e193_matchUnits(A&&...) { return {}; }
template <class... T, class... A> Any e195_transformLoop(A&&...) { return {}; }
template <class... T, class... A> Any e196_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e197_calculateIndicesRanges(A&&...) { return {}; }
template <class... T, class... A> Any e198_createConditionsForHyperRectSubscripts(A&&...) { return {}; }
template <class... T, class... A> Any e199_createConditionsForNonHyperRectSubscripts(A&&...) { return {}; }
template <class... T, class... A> Any e200_updateTPMVInfo(A&&...) { return {}; }
template <class... T, class... A> Any e201_setLoopIteratorOrder(A&&...) { return {}; }
template <class... T, class... A> Any e202_initialize(A&&...) { return {}; }
template <class... T, class... A> Any e203_removeCoresCoreletsFoldsFromDefImmutMap(A&&...) { return {}; }
template <class... T, class... A> Any e204_cleanup(A&&...) { return {}; }
template <class... T, class... A> Any e205_isDataTransferToKeep(A&&...) { return {}; }
template <class... T, class... A> Any e206_constructChunkAndShuffleInfo(A&&...) { return {}; }
template <class... T, class... A> Any e207_initialize(A&&...) { return {}; }
template <class... T, class... A> Any e210_checkBasicConditions(A&&...) { return {}; }
template <class... T, class... A> Any e211_processInterleaveOp(A&&...) { return {}; }
template <class... T, class... A> Any e212_gatherAffineLoadStoreDetails(A&&...) { return {}; }
template <class... T, class... A> Any e213_constructImmutableAddress(A&&...) { return {}; }
template <class... T, class... A> Any e214_setImmutableAddrAndIncrements(A&&...) { return {}; }
template <class... T, class... A> Any e215_setsttype(A&&...) { return {}; }
template <class... T, class... A> Any e216_constructReceiveAndExtractScalarOp(A&&...) { return {}; }
template <class... T, class... A> Any e217_lowerVectorLoadHelper(A&&...) { return {}; }
template <class... T, class... A> Any e218_lowerSetTransferMaskStateOp(A&&...) { return {}; }
template <class... T, class... A> Any e219_cloneStartAddrOutsideLoop(A&&...) { return {}; }
template <class... T, class... A> Any e220_cleanupTriviallyRedundantSetSendDestination(A&&...) { return {}; }
template <class... T, class... A> Any e223_lowerL3SyncOperationForAUnit(A&&...) { return {}; }
template <class... T, class... A> Any e226_createIfOpFromMapping(A&&...) { return {}; }
template <class... T, class... A> Any e228_validateLoweringAndSetMissingParameters(A&&...) { return {}; }
template <class... T, class... A> Any e232_getOperandFromLoadOrStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any e233_eraseOperands(A&&...) { return {}; }
template <class... T, class... A> Any e236_addMaskNode(A&&...) { return {}; }
template <class... T, class... A> Any e237_updateNode(A&&...) { return {}; }
template <class... T, class... A> Any e238_computeLoops(A&&...) { return {}; }
template <class... T, class... A> Any e239_insertPTMaskOps(A&&...) { return {}; }
template <class... T, class... A> Any e240_getLayoutExpr(A&&...) { return {}; }
template <class... T, class... A> Any e241_createForOpWithReturnValue(A&&...) { return {}; }
template <class... T, class... A> Any e242_createIfOpWithReturnValue(A&&...) { return {}; }
template <class... T, class... A> Any e243_insertDummyMacOp(A&&...) { return {}; }
template <class... T, class... A> Any e244_isHoistable(A&&...) { return {}; }
template <class... T, class... A> Any e245_enumerateCollectionUnit(A&&...) { return {}; }
template <class... T, class... A> Any e247_compute(A&&...) { return {}; }
template <class... T, class... A> Any e248_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e249_processComputeUnit(A&&...) { return {}; }
template <class... T, class... A> Any e250_initMASData(A&&...) { return {}; }
template <class... T, class... A> Any e251_setupForPartitioning(A&&...) { return {}; }
template <class... T, class... A> Any e252_fillPartitions(A&&...) { return {}; }
template <class... T, class... A> Any e253_adjustForEvenImmutableAddr(A&&...) { return {}; }
template <class... T, class... A> Any e254_offsetShifts(A&&...) { return {}; }
template <class... T, class... A> Any e255_applyShifts(A&&...) { return {}; }
template <class... T, class... A> Any e256_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e257_analyzeAndTransform(A&&...) { return {}; }
template <class... T, class... A> Any e258_addConstraintsForIVRanges(A&&...) { return {}; }
template <class... T, class... A> Any e259_createNewSubscriptsFromStartElements(A&&...) { return {}; }
template <class... T, class... A> Any e260_gatherPageDependentDimsForPage(A&&...) { return {}; }
template <class... T, class... A> Any e261_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e262_removeCoresCoreletsFoldsFromUniformizeRegion(A&&...) { return {}; }
template <class... T, class... A> Any e263_removeAncestors(A&&...) { return {}; }
template <class... T, class... A> Any e265_constructDetails(A&&...) { return {}; }
template <class... T, class... A> Any e266_coalesceTimeDimensions(A&&...) { return {}; }
template <class... T, class... A> Any e267_constructTimeLoopsAndVectorOperations(A&&...) { return {}; }
template <class... T, class... A> Any e268_constructLoadAndStoreStmt(A&&...) { return {}; }
template <class... T, class... A> Any e269_constructLoadAndExtractScalarOp(A&&...) { return {}; }
template <class... T, class... A> Any e270_addStoreInputToDeleteList(A&&...) { return {}; }
template <class... T, class... A> Any e271_lowerCompositeMemoryInterleaveOp(A&&...) { return {}; }
template <class... T, class... A> Any e272_insertInitializationStmt(A&&...) { return {}; }
template <class... T, class... A> Any e273_lowerL0LXSyncOperationForAUnit(A&&...) { return {}; }
template <class... T, class... A> Any e275_LowerSymbolQueryMap(A&&...) { return {}; }
template <class... T, class... A> Any e276_setReuseInformation(A&&...) { return {}; }
template <class... T, class... A> Any e277_getGCVTorFCVTTypeFromIndicesAndCastInputs(A&&...) { return {}; }
template <class... T, class... A> Any e278_getOperandFromShuffleOp(A&&...) { return {}; }
template <class... T, class... A> Any e282_updateLoopMaskTreeForConstantMask(A&&...) { return {}; }
template <class... T, class... A> Any e283_updateLoopMaskTreeForDynamicMask(A&&...) { return {}; }
template <class... T, class... A> Any e285_replaceIfOpByIterArg(A&&...) { return {}; }
template <class... T, class... A> Any e286_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e287_flatten(A&&...) { return {}; }
template <class... T, class... A> Any e288_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e289_initialize(A&&...) { return {}; }
template <class... T, class... A> Any e290_createPartitions(A&&...) { return {}; }
template <class... T, class... A> Any e291_calculatePartialShift(A&&...) { return {}; }
template <class... T, class... A> Any e293_runOn(A&&...) { return {}; }
template <class... T, class... A> Any e294_getPageValidity(A&&...) { return {}; }
template <class... T, class... A> Any e295_createIterArgsForConditionals(A&&...) { return {}; }
template <class... T, class... A> Any e296_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e297_constructTimeStepsInfo(A&&...) { return {}; }
template <class... T, class... A> Any e298_constructAffineDetailsAndAddrs(A&&...) { return {}; }
template <class... T, class... A> Any e299_lowerAffineCompositeHelper(A&&...) { return {}; }
template <class... T, class... A> Any e300_lowerSyncForAUnit(A&&...) { return {}; }
template <class... T, class... A> Any e301_lowerSyncForAGroup(A&&...) { return {}; }
template <class... T, class... A> Any e302_lowerSyncLXL3ToLXL3(A&&...) { return {}; }
template <class... T, class... A> Any e303_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e304_getOperandWithPrecision(A&&...) { return {}; }
template <class... T, class... A> Any e305_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e306_transformVectorLoad(A&&...) { return {}; }
template <class... T, class... A> Any e307_transformVectorStore(A&&...) { return {}; }
template <class... T, class... A> Any e308_calculateShifts(A&&...) { return {}; }
template <class... T, class... A> Any e309_constructValidPage(A&&...) { return {}; }
template <class... T, class... A> Any e310_analyzeValidPages(A&&...) { return {}; }
template <class... T, class... A> Any e311_constructAffineCompDetailsAndAddrs(A&&...) { return {}; }
template <class... T, class... A> Any e312_lowerExtractVectorLoadOp(A&&...) { return {}; }
template <class... T, class... A> Any e313_lowerExtractVectorStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any e314_lowerVectorLoadOp(A&&...) { return {}; }
template <class... T, class... A> Any e315_lowerVectorStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any e316_lowerIndirectVectorLoadOp(A&&...) { return {}; }
template <class... T, class... A> Any e317_lowerIndirectVectorStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any e319_lowerSyncForAQueryMap(A&&...) { return {}; }
template <class... T, class... A> Any e320_getOperand(A&&...) { return {}; }
template <class... T, class... A> Any e321_transformCompLoadAndStore(A&&...) { return {}; }
template <class... T, class... A> Any e322_transformCompIndLoadAndStore(A&&...) { return {}; }
template <class... T, class... A> Any e323_shiftMutableAddr(A&&...) { return {}; }
template <class... T, class... A> Any e324_analyzeAndConstructValidPages(A&&...) { return {}; }
template <class... T, class... A> Any e325_transform_time(A&&...) { return {}; }
template <class... T, class... A> Any e326_initialize_time(A&&...) { return {}; }
template <class... T, class... A> Any e327_gatherSymbolicLoadStoreDetails(A&&...) { return {}; }
template <class... T, class... A> Any e328_adjustMutableAddrInitForIndirect(A&&...) { return {}; }
template <class... T, class... A> Any e329_lowerCompositeLoadOp(A&&...) { return {}; }
template <class... T, class... A> Any e330_lowerCompositeStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any e331_lowerCompositeLoadAndStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any e332_lowerCompositeIndirectLoadOp(A&&...) { return {}; }
template <class... T, class... A> Any e333_lowerCompositeIndirectStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any e334_lowerCompositeIndirectLoadAndStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any e335_insertCopyAndAddStmts(A&&...) { return {}; }
template <class... T, class... A> Any e336_adjustMutableAddrInitForStride(A&&...) { return {}; }
template <class... T, class... A> Any e337_lowerSyncOperation(A&&...) { return {}; }
template <class... T, class... A> Any e339_SimplifyOrIOp(A&&...) { return {}; }
template <class... T, class... A> Any e340_analyzeAndFillOperandForwarding(A&&...) { return {}; }
template <class... T, class... A> Any e341_analyzeNonComputeOpsForFusion(A&&...) { return {}; }
template <class... T, class... A> Any e343_getOperandFromCastOp(A&&...) { return {}; }
template <class... T, class... A> Any e345_processXrfPtrPerUnit(A&&...) { return {}; }
template <class... T, class... A> Any e346_lowerDanglingNonComputeOps(A&&...) { return {}; }
template <class... T, class... A> Any e348_singleOpBranchToYieldVal(A&&...) { return {}; }
template <class... T, class... A> Any e349_isLoopInvariant(A&&...) { return {}; }
template <class... T, class... A> Any e350_matchAndRewrite(A&&...) { return {}; }
template <class... T, class... A> Any e351_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e352_transformVectorLoad(A&&...) { return {}; }
template <class... T, class... A> Any e353_transformVectorStore(A&&...) { return {}; }
template <class... T, class... A> Any e354_transformCompLoadAndStore(A&&...) { return {}; }
template <class... T, class... A> Any e355_transformCompIndLoadAndStore(A&&...) { return {}; }
template <class... T, class... A> Any e356_transform(A&&...) { return {}; }
template <class... T, class... A> Any e358_constructLoadAndSendStmt(A&&...) { return {}; }
template <class... T, class... A> Any e359_constructReceiveAndStoreStmt(A&&...) { return {}; }
template <class... T, class... A> Any e360_constructSymbolicDetailsAndAddrs(A&&...) { return {}; }
template <class... T, class... A> Any e361_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e362_LowerSelectOpToSentient(A&&...) { return {}; }
template <class... T, class... A> Any e363_LowerLogicalOpToSentient(A&&...) { return {}; }
template <class... T, class... A> Any e365_fillOpInfo(A&&...) { return {}; }
template <class... T, class... A> Any e366_fuseNonComputeOps(A&&...) { return {}; }
template <class... T, class... A> Any e367_createXrfIndexModifOps(A&&...) { return {}; }
template <class... T, class... A> Any e368_fuseNonComputeOps(A&&...) { return {}; }
template <class... T, class... A> Any e369_fuseComputeOps(A&&...) { return {}; }
template <class... T, class... A> Any e370_areShallowlyMergeable(A&&...) { return {}; }
template <class... T, class... A> Any e371_hoistLoopInvariantConditionals(A&&...) { return {}; }
template <class... T, class... A> Any e372_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e373_run(A&&...) { return {}; }
template <class... T, class... A> Any e374_lowerSymbolicVectorLoadOp(A&&...) { return {}; }
template <class... T, class... A> Any e375_lowerSymbolicVectorStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any e376_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e378_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e379_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e381_simplifyValueBasedConditionals(A&&...) { return {}; }
template <class... T, class... A> Any e382_fuseLoadOrStoreChainOps(A&&...) { return {}; }
template <class... T, class... A> Any e383_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any e384_runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any element_bit_width(A&&...) { return {}; }
template <class... T, class... A> Any enumerateCollectionUnit(A&&...) { return {}; }
template <class... T, class... A> Any equivalence_analysis(A&&...) { return {}; }
template <class... T, class... A> Any eraseMemOpAndUseChain(A&&...) { return {}; }
template <class... T, class... A> Any expandAffineApplyOps(A&&...) { return {}; }
template <class... T, class... A> Any failure(A&&...) { return {}; }
template <class... T, class... A> Any findInvariantLoop(A&&...) { return {}; }
template <class... T, class... A> Any findNodeFromOp(A&&...) { return {}; }
template <class... T, class... A> Any findOpWithAttribute(A&&...) { return {}; }
template <class... T, class... A> Any forop_builder(A&&...) { return {}; }
template <class... T, class... A> Any forwarding(A&&...) { return {}; }
template <class... T, class... A> Any fuseCompareAndSelectIntoMinOrMax(A&&...) { return {}; }
template <class... T, class... A> Any fuseLoadOrStoreChainOps(A&&...) { return {}; }
template <class... T, class... A> Any gatherPageDependentDimsForPage(A&&...) { return {}; }
template <class... T, class... A> Any gcvt_fcvt_type(A&&...) { return {}; }
template <class... T, class... A> Any getAffineConstantExpr(A&&...) { return {}; }
template <class... T, class... A> Any getAffineDimExpr(A&&...) { return {}; }
template <class... T, class... A> Any getAffineSymbolExpr(A&&...) { return {}; }
template <class... T, class... A> Any getCoalesceInfo(A&&...) { return {}; }
template <class... T, class... A> Any getComputePrecisionOfOp(A&&...) { return {}; }
template <class... T, class... A> Any getDataflowForLoopInfoIfIV(A&&...) { return {}; }
template <class... T, class... A> Any getFlattenedAffineExpr(A&&...) { return {}; }
template <class... T, class... A> Any getForOpBound(A&&...) { return {}; }
template <class... T, class... A> Any getGCVTorFCVTTypeFromIndicesAndCastInputs(A&&...) { return {}; }
template <class... T, class... A> Any getInputPrecisionFromOperand(A&&...) { return {}; }
template <class... T, class... A> Any getLastChild(A&&...) { return {}; }
template <class... T, class... A> Any getLayoutExpr(A&&...) { return {}; }
template <class... T, class... A> Any getLayoutMapAndIndices(A&&...) { return {}; }
template <class... T, class... A> Any getLdOrStSize(A&&...) { return {}; }
template <class... T, class... A> Any getLoadConsumer(A&&...) { return {}; }
template <class... T, class... A> Any getLoopNestLevel(A&&...) { return {}; }
template <class... T, class... A> Any getLoopTripCount(A&&...) { return {}; }
template <class... T, class... A> Any getMaskValueConstantForNonPT(A&&...) { return {}; }
template <class... T, class... A> Any getMaskValueForNonPT(A&&...) { return {}; }
template <class... T, class... A> Any getMaskValueForPT(A&&...) { return {}; }
template <class... T, class... A> Any getMaxImmutableRange(A&&...) { return {}; }
template <class... T, class... A> Any getMaxMutableRange(A&&...) { return {}; }
template <class... T, class... A> Any getMergeTypeFromIndices(A&&...) { return {}; }
template <class... T, class... A> Any getOE(A&&...) { return {}; }
template <class... T, class... A> Any getOffsetFromOuterForOp(A&&...) { return {}; }
template <class... T, class... A> Any getOperandFromConstantBitstreamOp(A&&...) { return {}; }
template <class... T, class... A> Any getOperandFromConstantOp(A&&...) { return {}; }
template <class... T, class... A> Any getOperandFromLoadOrStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any getOperandFromNegOp(A&&...) { return {}; }
template <class... T, class... A> Any getOperandFromReceiveOp(A&&...) { return {}; }
template <class... T, class... A> Any getOperandFromSendOp(A&&...) { return {}; }
template <class... T, class... A> Any getOperandFromShuffleOp(A&&...) { return {}; }
template <class... T, class... A> Any getOperandWithPrecision(A&&...) { return {}; }
template <class... T, class... A> Any getPageValidity(A&&...) { return {}; }
template <class... T, class... A> Any getPrevSibling(A&&...) { return {}; }
template <class... T, class... A> Any getResultPrecisionFromOperands(A&&...) { return {}; }
template <class... T, class... A> Any getSentientCmpIPredicate(A&&...) { return {}; }
template <class... T, class... A> Any getStoreProducer(A&&...) { return {}; }
template <class... T, class... A> Any getTransferSet(A&&...) { return {}; }
template <class... T, class... A> Any getUnitNameFromAListOfGetUnitOp(A&&...) { return {}; }
template <class... T, class... A> Any getVectorBinaryToSentientBinary(A&&...) { return {}; }
template <class... T, class... A> Any getVectorElementWiseCompareOperatorToSentientBinaryOperator(A&&...) { return {}; }
template <class... T, class... A> Any getVectorTernaryToSentientTernary(A&&...) { return {}; }
template <class... T, class... A> Any getXrfValue(A&&...) { return {}; }
template <class... T, class... A> Any group(A&&...) { return {}; }
template <class... T, class... A> Any guard(A&&...) { return {}; }
template <class... T, class... A> Any hasMutableAddrOverflow(A&&...) { return {}; }
template <class... T, class... A> Any identifyLoads(A&&...) { return {}; }
template <class... T, class... A> Any identifyTimeDimForExplicitLoops(A&&...) { return {}; }
template <class... T, class... A> Any init_builder(A&&...) { return {}; }
template <class... T, class... A> Any initialize(A&&...) { return {}; }
template <class... T, class... A> Any initializeMemViewInfo(A&&...) { return {}; }
template <class... T, class... A> Any insertConstAndAddOps(A&&...) { return {}; }
template <class... T, class... A> Any insertCopyAndAddStmts(A&&...) { return {}; }
template <class... T, class... A> Any insertDummyMacOp(A&&...) { return {}; }
template <class... T, class... A> Any insertMaskOps(A&&...) { return {}; }
template <class... T, class... A> Any insertPTMaskOps(A&&...) { return {}; }
template <class... T, class... A> Any instance(A&&...) { return {}; }
template <class... T, class... A> Any inversePermutation(A&&...) { return {}; }
template <class... T, class... A> Any isCandidateMemView(A&&...) { return {}; }
template <class... T, class... A> Any isDataTransfer(A&&...) { return {}; }
template <class... T, class... A> Any isDataTransferToKeep(A&&...) { return {}; }
template <class... T, class... A> Any isFirstElemSplat(A&&...) { return {}; }
template <class... T, class... A> Any isHoistable(A&&...) { return {}; }
template <class... T, class... A> Any isLoadAndExtractScalarPattern(A&&...) { return {}; }
template <class... T, class... A> Any isLoopInvariant(A&&...) { return {}; }
template <class... T, class... A> Any isOperationSelected(A&&...) { return {}; }
template <class... T, class... A> Any isReceiveAndExtractScalarPattern(A&&...) { return {}; }
template <class... T, class... A> Any isRightZeroPadFromFirstElem(A&&...) { return {}; }
template <class... T, class... A> Any isSameListOfUnits(A&&...) { return {}; }
template <class... T, class... A> Any isSenComponentL0LU(A&&...) { return {}; }
template <class... T, class... A> Any isSenComponentL0SU(A&&...) { return {}; }
template <class... T, class... A> Any isSentientBinaryLogicalOp(A&&...) { return {}; }
template <class... T, class... A> Any isSplatFromFirstElem(A&&...) { return {}; }
template <class... T, class... A> Any isTargetL3(A&&...) { return {}; }
template <class... T, class... A> Any isXrfRelated(A&&...) { return {}; }
template <class... T, class... A> Any iterator(A&&...) { return {}; }
template <class... T, class... A> Any llvm_unreachable(A&&...) { return {}; }
template <class... T, class... A> Any locale_list(A&&...) { return {}; }
template <class... T, class... A> Any loopUnrollByFactor(A&&...) { return {}; }
template <class... T, class... A> Any loopUnrollFull(A&&...) { return {}; }
template <class... T, class... A> Any lowerCompositeIndirectLoadAndStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerCompositeIndirectLoadOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerCompositeIndirectStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerCompositeLoadAndStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerCompositeLoadOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerCompositeMemoryInterleaveOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerCompositeStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerDanglingNonComputeOpsPESFP(A&&...) { return {}; }
template <class... T, class... A> Any lowerExtractVectorLoadOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerExtractVectorStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerIndirectVectorLoadOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerIndirectVectorStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerL0LXSyncOperationForAGroupOfUnits(A&&...) { return {}; }
template <class... T, class... A> Any lowerL0LXSyncOperationForAUnit(A&&...) { return {}; }
template <class... T, class... A> Any lowerL3SyncOperationForAGroupOfUnits(A&&...) { return {}; }
template <class... T, class... A> Any lowerL3SyncOperationForAUnit(A&&...) { return {}; }
template <class... T, class... A> Any lowerLDCVTIPattern(A&&...) { return {}; }
template <class... T, class... A> Any lowerOpaqueOperation(A&&...) { return {}; }
template <class... T, class... A> Any lowerSetTransferMaskStateOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerSymbolicVectorLoadOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerSymbolicVectorStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerSyncForAGroup(A&&...) { return {}; }
template <class... T, class... A> Any lowerSyncForAQueryMap(A&&...) { return {}; }
template <class... T, class... A> Any lowerSyncForAUnit(A&&...) { return {}; }
template <class... T, class... A> Any lowerSyncLXL3ToLXL3(A&&...) { return {}; }
template <class... T, class... A> Any lowerSyncOperation(A&&...) { return {}; }
template <class... T, class... A> Any lowerVectorLoadOp(A&&...) { return {}; }
template <class... T, class... A> Any lowerVectorStoreOp(A&&...) { return {}; }
template <class... T, class... A> Any mask_set_flat(A&&...) { return {}; }
template <class... T, class... A> Any matchUnits(A&&...) { return {}; }
template <class... T, class... A> Any mergeShallow(A&&...) { return {}; }
template <class... T, class... A> Any merge_and_pack_type(A&&...) { return {}; }
template <class... T, class... A> Any moveAncestorsToMaintainDominance(A&&...) { return {}; }
template <class... T, class... A> Any name(A&&...) { return {}; }
template <class... T, class... A> Any name_(A&&...) { return {}; }
template <class... T, class... A> Any non_compute_ops_patterns(A&&...) { return {}; }
template <class... T, class... A> Any num_inputs_(A&&...) { return {}; }
template <class... T, class... A> Any offsetShifts(A&&...) { return {}; }
template <class... T, class... A> Any op_creator(A&&...) { return {}; }
template <class... T, class... A> Any ordered_res_coeffs(A&&...) { return {}; }
template <class... T, class... A> Any page_sel_constraints(A&&...) { return {}; }
template <class... T, class... A> Any page_set_constraints(A&&...) { return {}; }
template <class... T, class... A> Any partitionUnits(A&&...) { return {}; }
template <class... T, class... A> Any partition_builder(A&&...) { return {}; }
template <class... T, class... A> Any pattern(A&&...) { return {}; }
template <class... T, class... A> Any patternAgnosticFuseNonComputeOpsHelper(A&&...) { return {}; }
template <class... T, class... A> Any patterns(A&&...) { return {}; }
template <class... T, class... A> Any performFullUnroll(A&&...) { return {}; }
template <class... T, class... A> Any pow(A&&...) { return {}; }
template <class... T, class... A> Any processComputeUnit(A&&...) { return {}; }
template <class... T, class... A> Any processInterleaveOp(A&&...) { return {}; }
template <class... T, class... A> Any processXrfPtrPerUnit(A&&...) { return {}; }
template <class... T, class... A> Any pushBackTheUnitToListIfDoesnotExist(A&&...) { return {}; }
template <class... T, class... A> Any query_map_builder(A&&...) { return {}; }
template <class... T, class... A> Any range_eq_constraints(A&&...) { return {}; }
template <class... T, class... A> Any recompute(A&&...) { return {}; }
template <class... T, class... A> Any redefineConstantVectors(A&&...) { return {}; }
template <class... T, class... A> Any region(A&&...) { return {}; }
template <class... T, class... A> Any removeAncestors(A&&...) { return {}; }
template <class... T, class... A> Any removeCoresCoreletsFoldsFromDefImmutMap(A&&...) { return {}; }
template <class... T, class... A> Any removeCoresCoreletsFoldsFromProgramUnit(A&&...) { return {}; }
template <class... T, class... A> Any removeCoresCoreletsFoldsFromUniformizeRegion(A&&...) { return {}; }
template <class... T, class... A> Any repetition_(A&&...) { return {}; }
template <class... T, class... A> Any replaceDimsInMapWithSyms(A&&...) { return {}; }
template <class... T, class... A> Any reuse_info(A&&...) { return {}; }
template <class... T, class... A> Any rewriter(A&&...) { return {}; }
template <class... T, class... A> Any routing_dir0(A&&...) { return {}; }
template <class... T, class... A> Any runOnOperation(A&&...) { return {}; }
template <class... T, class... A> Any separateBasedOnDestinationUnits(A&&...) { return {}; }
template <class... T, class... A> Any setBuilderToInsertRef(A&&...) { return {}; }
template <class... T, class... A> Any setBurstIndex(A&&...) { return {}; }
template <class... T, class... A> Any setChunkSize(A&&...) { return {}; }
template <class... T, class... A> Any setChunkStride(A&&...) { return {}; }
template <class... T, class... A> Any setCoalescedBoundValues(A&&...) { return {}; }
template <class... T, class... A> Any setDbgNameAttr(A&&...) { return {}; }
template <class... T, class... A> Any setElementLoadIfFound(A&&...) { return {}; }
template <class... T, class... A> Any setElementWidth(A&&...) { return {}; }
template <class... T, class... A> Any setExpectedTotalElements(A&&...) { return {}; }
template <class... T, class... A> Any setExtents(A&&...) { return {}; }
template <class... T, class... A> Any setIndicesCoeffDict(A&&...) { return {}; }
template <class... T, class... A> Any setInterleaveGroupIndex(A&&...) { return {}; }
template <class... T, class... A> Any setLayoutCoeffs(A&&...) { return {}; }
template <class... T, class... A> Any setLdOrStSize(A&&...) { return {}; }
template <class... T, class... A> Any setLoopIteratorOrder(A&&...) { return {}; }
template <class... T, class... A> Any setMemViewLayoutMap(A&&...) { return {}; }
template <class... T, class... A> Any setMemoryIndex(A&&...) { return {}; }
template <class... T, class... A> Any setOE(A&&...) { return {}; }
template <class... T, class... A> Any setRotationPosition(A&&...) { return {}; }
template <class... T, class... A> Any setScaleLoadIfFound(A&&...) { return {}; }
template <class... T, class... A> Any setShuffleMode(A&&...) { return {}; }
template <class... T, class... A> Any setSubscriptsMap(A&&...) { return {}; }
template <class... T, class... A> Any setTimeAddrMap(A&&...) { return {}; }
template <class... T, class... A> Any setTimeSymbols(A&&...) { return {}; }
template <class... T, class... A> Any setTotalElements(A&&...) { return {}; }
template <class... T, class... A> Any setTransferOrder(A&&...) { return {}; }
template <class... T, class... A> Any setTransferSet(A&&...) { return {}; }
template <class... T, class... A> Any setldtype(A&&...) { return {}; }
template <class... T, class... A> Any setsttype(A&&...) { return {}; }
template <class... T, class... A> Any setupForPartitioning(A&&...) { return {}; }
template <class... T, class... A> Any shiftMutableAddr(A&&...) { return {}; }
template <class... T, class... A> Any sign_extend(A&&...) { return {}; }
template <class... T, class... A> Any signalPassFailure(A&&...) { return {}; }
template <class... T, class... A> Any simplifyAffineMap(A&&...) { return {}; }
template <class... T, class... A> Any size_(A&&...) { return {}; }
template <class... T, class... A> Any src_input_type_str_(A&&...) { return {}; }
template <class... T, class... A> Any src_output_type_str_(A&&...) { return {}; }
template <class... T, class... A> Any stringifyVectorChainBinaryOperator(A&&...) { return {}; }
template <class... T, class... A> Any stringifyVectorChainEstimateVersions(A&&...) { return {}; }
template <class... T, class... A> Any stringifyVectorChainExpEstimate(A&&...) { return {}; }
template <class... T, class... A> Any success(A&&...) { return {}; }
template <class... T, class... A> Any symbolizeSentientBinaryOperator(A&&...) { return {}; }
template <class... T, class... A> Any symbolizeSentientComputePort(A&&...) { return {}; }
template <class... T, class... A> Any symbolizeSentientLoadConsumer(A&&...) { return {}; }
template <class... T, class... A> Any symbolizeSentientPrecision(A&&...) { return {}; }
template <class... T, class... A> Any symbolizeSentientShuffleMode(A&&...) { return {}; }
template <class... T, class... A> Any symbolizeSentientUnary(A&&...) { return {}; }
template <class... T, class... A> Any synthesizeTimeInfo(A&&...) { return {}; }
template <class... T, class... A> Any target(A&&...) { return {}; }
template <class... T, class... A> Any topLevelConditionsMatch(A&&...) { return {}; }
template <class... T, class... A> Any tpmv(A&&...) { return {}; }
template <class... T, class... A> Any tpmv_manager(A&&...) { return {}; }
template <class... T, class... A> Any transfer_set_csts(A&&...) { return {}; }
template <class... T, class... A> Any transform(A&&...) { return {}; }
template <class... T, class... A> Any transformCompIndLoadAndStore(A&&...) { return {}; }
template <class... T, class... A> Any transformCompLoadAndStore(A&&...) { return {}; }
template <class... T, class... A> Any transformSCFLoopWithNonConstantUpperBound(A&&...) { return {}; }
template <class... T, class... A> Any transformSCFToAffineLoop(A&&...) { return {}; }
template <class... T, class... A> Any transformVectorLoad(A&&...) { return {}; }
template <class... T, class... A> Any transformVectorStore(A&&...) { return {}; }
template <class... T, class... A> Any traverseRegion(A&&...) { return {}; }
template <class... T, class... A> Any tree(A&&...) { return {}; }
template <class... T, class... A> Any types(A&&...) { return {}; }
template <class... T, class... A> Any unit_matching_info(A&&...) { return {}; }
template <class... T, class... A> Any updateLoopMaskTreeForConstantMask(A&&...) { return {}; }
template <class... T, class... A> Any updateLoopMaskTreeForDynamicMask(A&&...) { return {}; }
template <class... T, class... A> Any updateSymbolicAccessDetails(A&&...) { return {}; }
template <class... T, class... A> Any updateTPMVInfo(A&&...) { return {}; }
template <class... T, class... A> Any updateYieldArgs(A&&...) { return {}; }
template <class... T, class... A> Any vec(A&&...) { return {}; }
template <class... T, class... A> Any vec_(A&&...) { return {}; }
template <class... T, class... A> Any verifyLoopNest(A&&...) { return {}; }
template <class... T, class... A> Any xrf_processor(A&&...) { return {}; }
template <class... T, class... A> Any yielded_vals(A&&...) { return {}; }
template <class... T, class... A> Any zip(A&&...) { return {}; }

inline Any A;
inline Any AND;
inline Any CFG;
inline Any CMPI;
inline Any CODEGEN_DUMP_IRS;
inline Any COMMENT_OUT_COALESCE_STORE;
inline Any CROSS;
inline Any CROSSPTNLINK;
inline Any DCC;
inline Any DD1;
inline Any EAR;
inline Any FCVT;
inline Any FP16;
inline Any FP32;
inline Any GCVT;
inline Any HBM;
inline Any IBR;
inline Any IF_OP_TO_BE_REPLACED_BY_ITER_ARG;
inline Any INFO;
inline Any INT32_MAX;
inline Any INT64_MIN;
inline Any IRF;
inline Any ISTATE;
inline Any IV;
inline Any L0;
inline Any L0LU;
inline Any L0SU;
inline Any L3;
inline Any L3IBR;
inline Any L3LU;
inline Any L3SU;
inline Any LCCR;
inline Any LDCVTI;
inline Any LHS;
inline Any LINK;
inline Any LRF;
inline Any LRFREG;
inline Any LU;
inline Any LX;
inline Any LXLU;
inline Any LXLUSCALEREG;
inline Any LXSU;
inline Any MPW4_ISA;
inline Any N;
inline Any NFWD;
inline Any OR;
inline Any PE;
inline Any PESTATE;
inline Any PROCESSED_SIMPLIFICATIONS;
inline Any PT;
inline Any PTIRF;
inline Any PTXRF;
inline Any RCUDD1A_ISA;
inline Any RHS;
inline Any SAMV;
inline Any SCF;
inline Any SEN1P5_ISA;
inline Any SENARCH;
inline Any SETDSTMASK;
inline Any SFP;
inline Any SFPSTATE;
inline Any SPLAT;
inline Any T;
inline Any TOGGLE_INDIRECT_IMPL1;
inline Any XRF;
inline Any access_details_;
inline Any context_;
inline Any curr_unit_;
inline Any data_origins_;
inline Any dominance_info_;
inline Any element_width_;
inline Any exp_;
inline Any expected_total_elements_;
inline Any extents_;
inline Any filter_components_except_;
inline Any filter_corelets_except_;
inline Any filter_cores_except_;
inline Any filter_folds_except_;
inline Any filter_transfers_except_;
inline Any for_op_tuple_;
inline Any if_op_;
inline Any index_mapping_;
inline Any indices_coeff_dict_;
inline Any interleave_group_index_;
inline Any interleave_loop_count_;
inline Any is_visited_;
inline Any layout_coeffs_;
inline Any mem_ops_;
inline Any mem_view_layout_map_;
inline Any mem_view_start_addr_;
inline Any memory_index_;
inline Any new_op_;
inline Any num_conditionals_;
inline Any op_to_node_;
inline Any opr_;
inline Any opts_;
inline Any page_dependent_time_syms_;
inline Any program_units_;
inline Any reducible_groups_;
inline Any reduction_;
inline Any root_;
inline Any rotation_position_;
inline Any seq_lb_;
inline Any seq_step_;
inline Any set_incr_mask_count_;
inline Any shuffle_mode_;
inline Any sigm_;
inline Any strides_;
inline Any tanh_;
inline Any time_bounds_;
inline Any time_offsets_;
inline Any time_order_;
inline Any time_set_;
inline Any time_symbols_;
inline Any to_delete_;
inline Any total_elements_;
inline Any tpmv_comp_info_;
inline Any tpmv_info_;
inline Any transfer_order_;
inline Any transfer_set_;
inline Any use_;
inline Any val_array_;
