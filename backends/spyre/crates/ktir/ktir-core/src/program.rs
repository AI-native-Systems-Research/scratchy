// Copyright 2025 The Torch-Spyre Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License").
//
//! What a KTIR program computes, as its producer states it: the node kind the emulator's fused
//! runner groups programs by. The device lowering reads the program's own ops, never this tag.

/// Which pointwise function a program computes — see [`Program::Elementwise`].
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Elementwise {
    Silu,
    Gelu,
    Mul,
    Add,
    QuickGelu,
    GeluErf,
    Sub,
    Exp,
    Log,
    Neg,
    Rsqrt,
    Sqrt,
    Abs,
    Reciprocal,
    Sigmoid,
    Tanh,
    Mish,
    RealDiv,
    Maximum,
    Minimum,
    Equal,
}

impl Elementwise {
    /// Every variant.
    pub const ALL: &'static [Elementwise] = &[
        Elementwise::Silu,
        Elementwise::Gelu,
        Elementwise::Mul,
        Elementwise::Add,
        Elementwise::QuickGelu,
        Elementwise::GeluErf,
        Elementwise::Sub,
        Elementwise::Exp,
        Elementwise::Log,
        Elementwise::Neg,
        Elementwise::Rsqrt,
        Elementwise::Sqrt,
        Elementwise::Abs,
        Elementwise::Reciprocal,
        Elementwise::Sigmoid,
        Elementwise::Tanh,
        Elementwise::Mish,
        Elementwise::RealDiv,
        Elementwise::Maximum,
        Elementwise::Minimum,
        Elementwise::Equal,
    ];
}

/// The node kind a program computes.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Program {
    Elementwise(Elementwise),
    SiluMul,
    RmsNorm,
    /// The norm with no gain.
    RmsNormUnit,
    /// `cap · tanh(x / cap)`.
    TanhSoftCap,
    /// `x · w` for a loaded one-element `w`.
    ScalarWeightMul,
    /// A copy that re-views a whole buffer.
    Reshape,
    ScalarMul,
    Matmul,
    /// The last-row extraction of an lm-head tail.
    LmLast,
    /// A whole source copied into a column block of a wider destination.
    BlockCopy,
    Rope,
    /// Paged attention over the KV pools.
    PagedAttn,
    /// The paged KV write.
    PagedKvWrite,
    /// A 2-D transpose of a whole tensor.
    Transpose,
    /// The norm applied per head window.
    WindowedRmsNorm {
        unit: bool,
    },
    /// The token draw: one row of logits to the index it samples.
    Sample,
    /// The chained head's input rows: each request's token embedding read out of the tied
    /// weight at the token the previous step drew.
    EmbedRowGather,
    /// The chained head's stage: each request's drawn token decomposed into the next step's
    /// gather indices.
    ChainStage,
}
