// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! KTIR core — the dependency-free IR/parse/data layer of the KTIR CPU stack
//! (RFC 0682): IR types, the MLIR-text parser, affine expressions/maps/sets,
//! dtypes, the tile/memref value types, and the f16 codec. The execution layer
//! (`ktir-cpu`) and the optimizer (`ktir-optimizer`) build on these.
//!
//! Module names mirror the Python `ktir_cpu` package for diffability.

pub mod affine;
pub mod arena;
pub mod attrkey;
pub mod codec;
pub mod consttest;
pub mod dtypes;
pub mod fxhash;
pub mod ir;
pub mod irtype;
pub mod memref;
pub mod opkind;
pub mod tile;
