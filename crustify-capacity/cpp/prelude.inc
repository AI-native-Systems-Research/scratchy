// prelude.inc — the name catalogue for capacity.cpp's 19 bodies.
//
// ⛔ THIS IS NOT A PORTING INPUT. It carries NAMES ONLY. The behaviour being ported lives in
// the bodies, and the authority for every one of them is
//   /Users/nickm/git/deeptools-src/<file>:<line>
// as the banner above each body states.
//
// ⭐ UNLIKE bridge 1 AND bridge 2, THIS CAMPAIGN'S DECLARATIONS ARE ALL READABLE. Every type
// below is declared in a plain, header-only, MLIR-free part of the authority tree — open the
// real header rather than a stand-in:
//   /Users/nickm/git/deeptools-src/dsc/dims.h                DataStructDims, DimPaddingInfo,
//                                                            SymbolicDimInfo, PaddingFormType
//   /Users/nickm/git/deeptools-src/dsc/dsc2.h                ScheduleNode, AllocateNode, LoopNode,
//                                                            BlockNode, DataStage, CoordinateType,
//                                                            CoordinateFoldPosition,
//                                                            CoordinateCategory, FoldManager
//   /Users/nickm/git/deeptools-src/dsc/designSpaceConfig.h   DesignSpaceConfig, LabeledDsInfo,
//                                                            PrimaryDsInfo   (getLayoutDimSet's and
//                                                            getStickDims' inline int overloads are
//                                                            at :238 and :244)
//   /Users/nickm/git/deeptools-src/dsc/dscdefn.h             PadType, DsTypes, SenComponents,
//                                                            PrimaryDimTypes, MemOrgInfo
//
// The reference's own assertion macros: DT_CHECK / DT_CHECK_MSG abort the compiler, and DT_ERROR
// FAILS THE PASS — so in Rust a stop is faithful where one of those stands, and a fallback value
// is not.
#pragma once
#include <cmath>
#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

// ---- 16 qualified scopes the bodies reach
// scope: AllocateNode
// scope: CoordinateCategory
// scope: CoordinateFoldPosition
// scope: DataStructDims
// scope: DesignSpaceConfig
// scope: EnumsConversion
// scope: IndirectAllocType
// scope: LabeledDsInfo
// scope: LoopNode
// scope: PadType
// scope: PrimaryDimTypes
// scope: ScaledLdsCategory
// scope: ScheduleNode
// scope: SenComponents
// scope: dsc2
// scope: std
//
// ---- 46 trailing-underscore members the bodies read or write
// member: allocateCoordinates_
// member: allocateNode_
// member: backGapCore_
// member: component_
// member: coordinates_
// member: coreIdToWkSlice_
// member: coreletSplit_
// member: denId_
// member: dim_
// member: dims_
// member: dsType_
// member: el_
// member: gapStickSpread_
// member: granularity_
// member: ignoreSymbolicVolumeLimits_
// member: indirectAllocType_
// member: labeledDs_
// member: layoutDimOrder_
// member: ldsIdx_
// member: maxDimSizes_
// member: maxSize_
// member: maxSymbolicVolume_
// member: memOrg_
// member: mxInfo_
// member: name_
// member: nodeType_
// member: nonUnifiedAllocInHBM_
// member: numBuffers_
// member: padBack_
// member: padFront_
// member: paddingSizes_
// member: padding_
// member: peSfpSplit_
// member: prev_
// member: referenceLdsIdx_
// member: rowSplit_
// member: scale_
// member: scaledLdsCategory_
// member: sliceViewCoordinates_
// member: ss_
// member: stride_
// member: symbolicDimInfo_
// member: unneededPadBack_
// member: unneededPadFront_
// member: unneededPad_
// member: windowDim_
//
// ---- 44 member calls the bodies make
// call: any_of
// call: at
// call: begin
// call: calculate_padded
// call: ceil
// call: clear
// call: compound
// call: count
// call: emplace
// call: emplace_back
// call: empty
// call: end
// call: erase
// call: find
// call: foldConstructed
// call: getBufferCapacityForNode
// call: getBufferCapacityForNodePerDim
// call: getBufferCapacityForNodePerDimCustomLocation
// call: getCoordinateCategoryOfPos
// call: getCumulativeStickSizes
// call: getDimIndexInLayoutOrder
// call: getFoldDimSize
// call: getLayoutDimSet
// call: getLayoutDims
// call: getNumDims
// call: getOwnerLoop
// call: getPadding
// call: getPageSize
// call: getPrev
// call: getSizeDataStageForNode
// call: insert
// call: insert_or_assign
// call: isParametricLoop
// call: max
// call: min
// call: parametricStride
// call: primaryDimToValHandler_st
// call: primaryDimToVal_base_st
// call: primaryDimToVal_clView_st
// call: primaryDimToVal_st
// call: pruneMaxSymbolicVolumes
// call: scaleFromMaxToGranularity
// call: size
// call: try_emplace
