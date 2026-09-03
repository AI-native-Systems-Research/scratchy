// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! The `Worker` trait + `NoopWorker` moved down into `scratchy-serving-engine`
//! (below `compiler` and the target crates) so backend worker impls in
//! `targets/*` can implement the trait without a dependency cycle.

pub use scratchy_serving_engine::worker::*;
