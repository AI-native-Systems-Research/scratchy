// SPDX-License-Identifier: Apache-2.0
//! Integration-test re-export of the crate's MTL4 single-op dispatch
//! helper ([`scratchy_target_metal::mtl4_dispatch`]). Kept as `mod common`
//! so the converted parity tests reach it as `common::dispatch_threadgroups`,
//! `common::shared_slice`, `common::Buffer`, etc.
#![allow(unused_imports)]

pub use scratchy_target_metal::mtl4_dispatch::*;
