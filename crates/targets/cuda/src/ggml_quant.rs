// SPDX-License-Identifier: Apache-2.0
//! GGML quantized data types — pure data, no CUDA calls.
//!
//! `GgmlDType` / `GgmlStorage` live in the `scratchy-quantizations` crate (so
//! `LinearLayer::Ggml` can name `GgmlStorage` below the target crates without a
//! cycle). Re-exported here under the historic `crate::ggml_quant` path so
//! `GpuWeights`'s `HashMap<String, GgmlStorage>` field and every
//! `crate::ggml_quant::*` / `crate::ggml::GgmlStorage` call site keep resolving
//! against the same type.
pub use scratchy_quantizations::ggml_quant::{GgmlDType, GgmlStorage};
