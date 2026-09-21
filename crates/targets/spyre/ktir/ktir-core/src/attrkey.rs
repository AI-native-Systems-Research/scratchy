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

    // ⛔⛔⛔ A NEW KEY GOES HERE, AT THE END, AND NOT AT ITS ALPHABETICAL HOME. The list above is
    // alphabetical by SPELLING and this one is not, deliberately: this enum is fieldless, so
    // `#[derive(Hash)]` hashes the DISCRIMINANT, and `Operation`'s own derived `Hash` carries
    // `attributes: &[(AttrKey, Attr)]` straight into `bundle_fingerprint_grouped` — which is the
    // `superdsc-code/<fp>` directory name every baked bundle is stored and found under. Inserting a
    // variant in the middle therefore shifts every later discriminant and RENAMES EVERY BUNDLE IN
    // THE TREE, with no change to a single descriptor byte.
    //
    // MEASURED, granite-8b fp16 at `SCRATCHY_SUPERDSC_GROUP_SIZE=512`: `DimSubs` at its alphabetical
    // position (between `DimMap0` and `Dimensions`) moved the 134-bundle set from `ac71651adc80` to
    // `a8d5f0a7e848` with 0 of 134 fingerprints in common. Appended here, the set is main's.
    // `the_declaration_order_is_a_baked_fingerprint_input` below is the guard that says so.
    /// `dim_subs` — ONE `AffineMapList`, one map per OUTPUT DIMENSION of a
    /// `ktdp.construct_indirect_access_tile`, carrying that dim's SUBSCRIPT
    /// EXPRESSIONS.
    ///
    /// ⭐ WHY THIS KEY EXISTS. `parse_dim_subscripts` could build neither a
    /// `direct_sub` dim nor an `indirect` dim with an explicit index expression,
    /// because no key could carry the expression: it failed loudly on the first
    /// and silently addressed the index view by the bare enumeration point on
    /// the second. A gather whose index is an EXPRESSION — `table[ids[%off +
    /// %d0], %d1]`, which is what any grid of more than one work item over one
    /// index vector needs — was therefore inexpressible.
    ///
    /// ⛔ AND IT IS **ONE** KEY FOR EVERY DIM, NOT `dim_sub_0`, `dim_sub_1`, ...
    /// [`AttrKey::DimMap0`] is the per-dim spelling and it is a dead end: only
    /// `dim_map_0` is declared, so the SECOND `direct_expr` dim of any op is
    /// unrepresentable and the handler refuses it by arity. A list indexed by
    /// dimension has no such ceiling.
    ///
    /// # THE DOMAIN IS THE ENUMERATION POINT; CAPTURES ARE SYMBOLS
    ///
    /// Map `d` is evaluated at the intermediate-variable point, so `Dim(i)` is
    /// enumeration variable `i` — NOT an operand, and NOT the captures-first
    /// domain MLIR's own `per_dim_subscript_maps` uses. Outer SSA scalars enter
    /// as `Sym(j)`, naming `intermediate_vars[j]`, whose value is read from the
    /// value table when the op executes. That is exactly
    /// [`crate::memref::SubExpr`]'s `(expr, syms)` split, and it is what lets an
    /// `AffineExpr<'static>` — whose children are BORROWS and so cannot be built
    /// while a handler runs — be read straight off the program.
    ///
    /// Result COUNT per map is the subscript arity of that dim: the index view's
    /// RANK for an `indirect` dim (one expression per view axis, dotted with the
    /// view's strides), and 1 for a `direct_sub` dim.
    DimSubs,
}

/// ⛔⛔⛔ THE DECLARATION ORDER OF [`AttrKey`] IS AN INPUT TO EVERY BAKED BUNDLE'S NAME, and this is
/// the test that says so — the one that would have caught `DimSubs` at its alphabetical position.
///
/// The chain, every link of it derived rather than written: [`AttrKey`] is FIELDLESS, so
/// `#[derive(Hash)]` hashes its DISCRIMINANT; `Operation` derives `Hash` over
/// `attributes: &[(AttrKey, Attr)]`; `scratchy-target-spyre`'s `bundle_fingerprint_grouped` hashes
/// `k.func` for every emitted op; and that fingerprint IS the `$OUT_DIR/superdsc-code/<fp>` directory
/// the bake writes each bundle's device code into and the runtime finds it under. So a variant inserted
/// in the middle shifts every later discriminant and RENAMES EVERY BUNDLE — while not moving one byte
/// of one descriptor, which is why reading the diff cannot find it and only the set can.
///
/// The three pins below are the recorded positions. `Dimensions == 12` is the load-bearing one: it is
/// the first variant an alphabetical `dim_*` insertion displaces, and it was 12 when granite-8b fp16's
/// 134-bundle set read `a8d5f0a7e848` instead of `ac71651adc80`.
#[cfg(test)]
mod declaration_order {
    use super::AttrKey;

    #[test]
    fn the_declaration_order_is_a_baked_fingerprint_input() {
        assert_eq!(
            AttrKey::ResultShape as u32,
            0,
            "`_result_shape` sorts first and is the anchor of the recorded order"
        );
        assert_eq!(
            AttrKey::Dimensions as u32,
            11,
            "a NEW KEY WAS INSERTED ABOVE `Dimensions` INSTEAD OF AT THE END OF THE ENUM. That \
             renames every baked bundle (see this module's doc): `Dimensions` is hashed by \
             discriminant into every `Operation` that carries it, and every `superdsc-code/<fp>` \
             directory name follows. Move the new variant to the end."
        );
        assert_eq!(
            AttrKey::VariablesSpaceSet as u32,
            39,
            "`variables_space_set` is the last of the ALPHABETICAL block"
        );
        assert_eq!(
            AttrKey::DimSubs as u32,
            40,
            "`dim_subs` is appended AFTER that block, not at its alphabetical home"
        );
    }
}
