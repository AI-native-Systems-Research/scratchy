// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! AN ATTRIBUTE'S KEY, AS A TYPE.
//!
//! An operation's attributes were keyed by spelling, so a producer writing `"shape"` and a
//! consumer reading `"shapes"` agreed at run time by accident and disagreed the same way. The
//! key is a variant: the set of attributes an operation can carry is the set declared here.

/// Every attribute key this tree names.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub enum AttrKey {
    /// `_result_shape`
    ResultShape,
    /// `base_map`
    BaseMap,
    /// `bb0_names`
    Bb0Names,
    /// `consumer_tiles_per_group`
    ConsumerTilesPerGroup,
    /// `coordinate_order`
    CoordinateOrder,
    /// `coordinate_set`
    CoordinateSet,
    /// `dense_list`
    DenseList,
    /// `dim`
    Dim,
    /// `dim_data`
    DimData,
    /// `dim_kinds`
    DimKinds,
    /// `dim_map_0`
    DimMap0,
    /// `dimensions`
    Dimensions,
    /// `dtype`
    Dtype,
    /// `groups`
    Groups,
    /// `indexing_maps`
    IndexingMaps,
    /// `intermediate_vars`
    IntermediateVars,
    /// `is_tensor`
    IsTensor,
    /// `iter_args`
    IterArgs,
    /// `iter_var`
    IterVar,
    /// `lx_core_id`
    LxCoreId,
    /// `memory_space`
    MemorySpace,
    /// `n_ins`
    NIns,
    /// `names`
    Names,
    /// `num_results`
    NumResults,
    /// `outs_var`
    OutsVar,
    /// `permutation`
    Permutation,
    /// `predicate`
    Predicate,
    /// `producer_tiles_per_group`
    ProducerTilesPerGroup,
    /// `reduce_fn`
    ReduceFn,
    /// `result_names`
    ResultNames,
    /// `shape`
    Shape,
    /// `sizes_dyn`
    SizesDyn,
    /// `slice_offsets`
    SliceOffsets,
    /// `slice_sizes`
    SliceSizes,
    /// `slice_strides`
    SliceStrides,
    /// `strides`
    Strides,
    /// `target_shape`
    TargetShape,
    /// `value`
    Value,
    /// `variables_space_order`
    VariablesSpaceOrder,
    /// `variables_space_set`
    VariablesSpaceSet,
}
