// SPDX-License-Identifier: Apache-2.0
//! ⭐ STAGE 1 OF THE TWO-STAGE MODEL BUILD: each selected canonical as a type.
//!
//! For every canonical this build compiles (the same features as scratchy-models, forwarded), a
//! `pub struct` implementing [`scratchy_ir::facts::ModelFacts`], and [`dispatch`], which runs a
//! [`scratchy_ir::facts::FactsVisitor`] at one of them. scratchy-models' build script takes this
//! crate as a build-dependency, so every lowering it runs is generic over a model's facts and
//! monomorphised for the selected canonicals only. See `build.rs`.

use scratchy_ir::facts::FactsVisitor;

/// One selected canonical in [`dispatch`]'s table: its key, and the visitor instantiated at its
/// type.
struct Entry<V: FactsVisitor> {
    arch: &'static str,
    canonical: &'static str,
    visit: fn(V) -> V::Out,
}

include!(concat!(env!("OUT_DIR"), "/facts.rs"));
