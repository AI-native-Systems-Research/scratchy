// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Input batch assembly moved down into `scratchy-serving-engine` (below
//! `compiler` and the target crates) so backend worker impls in `targets/*` can
//! use it without a dependency cycle.

pub use scratchy_serving_engine::input_batch::*;
