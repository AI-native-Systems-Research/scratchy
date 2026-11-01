// SPDX-License-Identifier: Apache-2.0
//! Tensor-parallel build scope. This crate carries no code: its `2`, `4` and `8` features select
//! which sharded worlds `#[forward]` bakes for spyre (`tp/<W>` from `scratchy-cli`), forwarded through
//! scratchy-models to the forward macro. See `Cargo.toml`.
