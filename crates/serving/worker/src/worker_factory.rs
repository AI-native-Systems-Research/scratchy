// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Backend-neutral worker construction (the `WorkerFactory` trait + inventory
//! registry, `WorkerCreateConfig`, `resolve_model_path`) moved down into
//! `scratchy-serving-engine` so backend factories in `targets/*` can register
//! against the same registry without a dependency cycle.

pub use scratchy_serving_engine::worker_factory::*;
