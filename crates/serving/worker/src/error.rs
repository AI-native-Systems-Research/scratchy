// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Executor error types. The definitions moved down into
//! `scratchy-serving-engine` (below `compiler` and the target crates) so backend
//! worker impls in `targets/*` can name them without a dependency cycle.

pub use scratchy_serving_engine::error::{ExecutorError, ExecutorResult};
