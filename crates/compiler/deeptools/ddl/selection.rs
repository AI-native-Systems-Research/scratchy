// SPDX-License-Identifier: Apache-2.0
//! IBM'S TEMPLATE-SELECTION TABLE, PORTED TO RUST — which `.ddl` serves each op-func, per core generation.
//!
//! Ported from three places in the deeptools source:
//!
//! | fact | authority path |
//! |---|---|
//! | the (op-func → ordered candidates) table | `ddc/ddl/ddl_conversion.h:86-271` (`opFuncToDdlTemplate`) |
//! | each `OpFuncs` variant's DDL spelling | `sys-arch-spec/arch_enums.cpp:322-499` (`opFuncsToString`) |
//! | the core generations | `sys-arch-spec/isa/isa.hpp:25-32` (`enum IsaCoreGen`) |
//!
//! ⛔ REVIEWED: RE-CITED AGAINST THE AUTHORITY. The three ranges were first read off the pod (2026-08-13) as
//! `:86-267`, `:308-481` and `:25-32`; against the tree this campaign declares its authority only the last held.
//! `:267` stops three rows SHORT of the table's `};` at `:271` — and stopping there is exactly what dropped four of
//! its rows, see `OP_FUNC_TEMPLATES`.
//!
//! ⭐ THIS RESOLVES EVERY OP-FUNC, where the file stems alone resolve only the 73 that exactly one template
//! declares — and it corrects the guess those stems invite, because `_dd1` is MPW4 rather than RCUDD1A, so plain
//! `bmm.ddl` is the RCUDD1A one.
//!
//! ⭐ TWO SPELLING CONVENTIONS ARE REAL. Most op-funcs spell lowercase (`matmul`, `batchmatmulfp8mb`), while the four
//! restickify ones spell CamelCase (`ReStickifyOpLx`) — exactly as `opFuncsToString` writes them, because the `.ddl`
//! templates match against these strings.
//!
//! ⛔ `CSQ_INT4` IS DECLARED TWICE IN THE C++ AND THE FIRST IS COMMENTED OUT (`ddl_conversion.h:210-211`), so the
//! table holds 121 op-funcs across 122 declarations. Both spell the same candidate, so a port that missed the comment
//! would agree on behaviour and disagree on the count.
//!
//! This file lives in `ddl/`, outside `src/`, because it holds TEXT: `build.rs` includes it, so the template file
//! names cannot reach an island.

/// A CORE GENERATION — `IsaCoreGen` (`isa.hpp:25-32`).
///
/// ⛔ `MPW2`/`MPW3` ARE ABSENT ON PURPOSE. `opFuncToDdlTemplate` names only these three, and the crate builds for two
/// of them (`arch-rcudd1a`, `arch-sen1p5`); MPW4 appears in the table but has no cargo feature, so a candidate tagged
/// for it is never selected. Adding a generation here is a build error until every table arm names it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IsaGen {
    /// `MPW4_ISA` — in the table, but the crate has no feature for it.
    Mpw4,
    /// `RCUDD1A_ISA`, the crate's default.
    Rcudd1a,
    /// `SEN1P5_ISA`.
    Sen1p5,
}

/// WHICH GENERATIONS ONE CANDIDATE SERVES.
///
/// ⛔ AN ABSENT ISA TAG IS A FACT, NOT A MISSING VALUE, which is why this is an enum and not an `Option<IsaGen>`.
/// `opFuncToDdlTemplate` omits the tag for a candidate that serves every generation — 25 of the entries are written
/// that way — so reading absence as "no generation" would reject most of the table, and reading it as a default
/// generation would silently pick one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Serves {
    /// The candidate is written with no ISA tag: every generation.
    EveryArch,
    /// The candidate names exactly one generation.
    Only(IsaGen),
}

impl Serves {
    /// Whether this candidate may be selected when building for `arch`.
    pub const fn covers(self, arch: IsaGen) -> bool {
        match self {
            Self::EveryArch => true,
            Self::Only(only) => match (only, arch) {
                (IsaGen::Mpw4, IsaGen::Mpw4) => true,
                (IsaGen::Rcudd1a, IsaGen::Rcudd1a) => true,
                (IsaGen::Sen1p5, IsaGen::Sen1p5) => true,
                (IsaGen::Mpw4, IsaGen::Rcudd1a) => false,
                (IsaGen::Mpw4, IsaGen::Sen1p5) => false,
                (IsaGen::Rcudd1a, IsaGen::Mpw4) => false,
                (IsaGen::Rcudd1a, IsaGen::Sen1p5) => false,
                (IsaGen::Sen1p5, IsaGen::Mpw4) => false,
                (IsaGen::Sen1p5, IsaGen::Rcudd1a) => false,
            },
        }
    }
}

/// ONE CANDIDATE TEMPLATE for one op-func.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Candidate {
    /// The `.ddl` file name, as `opFuncToDdlTemplate` writes it.
    pub template: &'static str,
    pub serves: Serves,
}

/// EVERY OP-FUNC'S CANDIDATES, IN THE C++'S DECLARATION ORDER.
///
/// ⛔ THE ORDER INSIDE AN ENTRY IS LOAD-BEARING. `opFuncToDdlTemplate` is an ORDERED list and dxp takes the first
/// candidate whose ISA tag admits the target, so re-sorting an entry changes which template serves an op-func.
///
/// ⛔ REVIEWED: FOUR ROWS WERE MISSING AND THREE ARE NOW BACK — `batchmatmulmxfp4w` (`ddl_conversion.h:107`),
/// `addi32toi32` (`:169`) and `addi64toi64` (`:170`), each of which `ddl_templates` was answering
/// with the *"no DDL available for op"* `None` that belongs to an op-func the table does not name at all. All four
/// live past the `:267` the first read stopped at.
/// ⛔ THE FOURTH IS STILL MISSING AND IT IS BLOCKED, NOT DECIDED: `StzLatch` → `stz_latch.ddl` (`:270`). That
/// template is the one `ddc/ddl_templates/*.ddl` the crate has not vendored, so `Template` has no variant to name it
/// and the row cannot be written until it is. Until then `selectAndParseDdlTemplate` answers *"no DDL available for
/// op StzLatch"* where the reference would try `stz_latch.ddl`.
///
/// 120 of the authority's 121 op-funcs, naming 30 of its 31 distinct templates.
pub const OP_FUNC_TEMPLATES: &[(&str, &[Candidate])] = &[
    (
        "matmul",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "matmulfp8",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "matmulint8",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "matmulint4",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "batchmatmul",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "batchmatmulmxfp4w",
        &[Candidate {
            template: "bmm_sen1p5.ddl",
            serves: Serves::Only(IsaGen::Sen1p5),
        }],
    ),
    (
        "batchmatmulfp8",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "batchmatmulfp8mb",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "batchmatmulint8",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "batchmatmulint8mbkg3",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "batchmatmulint4",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "batchmatmulxrf",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "batchmatmulxrffp8",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "batchmatmulxrfint8",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "batchmatmulxrfint4",
        &[
            Candidate {
                template: "bmm.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "bmm_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "bmm_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "batchmatmulmxfp8",
        &[Candidate {
            template: "bmm_sen1p5.ddl",
            serves: Serves::Only(IsaGen::Sen1p5),
        }],
    ),
    (
        "scaledgroupmatmulfp4",
        &[Candidate {
            template: "bmm_sen1p5.ddl",
            serves: Serves::Only(IsaGen::Sen1p5),
        }],
    ),
    (
        "reciprocal",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "layernormscale",
        &[
            Candidate {
                template: "unary_parallel.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "layernormscale_32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "tanh",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "tanhbackward",
        &[Candidate {
            template: "gelu_bwd.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "leakyrelufwd",
        &[Candidate {
            template: "unary_pipeline.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "relufwd",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "relu6fwd",
        &[Candidate {
            template: "unary_pipeline.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "clip",
        &[
            Candidate {
                template: "unary_pipeline.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "unary_parallel.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "fastexp",
        &[
            Candidate {
                template: "unary_pipeline.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "unary_parallel.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "gelufwd",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "gelubackward",
        &[Candidate {
            template: "gelu_bwd.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "sigmoid",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "fastsigmoid",
        &[Candidate {
            template: "unary_pipeline.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "silu",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "exp",
        &[
            Candidate {
                template: "unary_parallel.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "unary_pipeline.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "log",
        &[Candidate {
            template: "unary_pipeline.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "layernormnorm",
        &[
            Candidate {
                template: "layernormnorm.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "layernormnorm_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "layernormbackwardnorm",
        &[Candidate {
            template: "layernormbackwardnorm.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "maxpoolfwd",
        &[Candidate {
            template: "pooling.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "avgpoolfwd",
        &[Candidate {
            template: "pooling.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "add",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "addi32toi32",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "addi64toi64",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "muli32toi32",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "sub",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "mul",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "revsub",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "realdiv",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "biasadd",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "stridedadd",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "batchnormfwd",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "fnms",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "where3",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "maximum",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "minimum",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "sinkcorrectionfactor",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "sum",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "sumnonstick",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "mean",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "meannonstick",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "max",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "maxnonstick",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "absmax",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "absmaxnonstick",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "min",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "minnonstick",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "prodnonstick",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "exx2",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "exx2_32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "exx2_zeromean",
        &[
            Candidate {
                template: "summeanmaxexx2.ddl",
                serves: Serves::EveryArch,
            },
            Candidate {
                template: "summeanmaxexx2_fp32.ddl",
                serves: Serves::EveryArch,
            },
        ],
    ),
    (
        "csqint8wt",
        &[Candidate {
            template: "quantization_no_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "qfp8",
        &[Candidate {
            template: "quantization_single_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "qfp8ch",
        &[Candidate {
            template: "quantization_double_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "qfp8wt",
        &[Candidate {
            template: "quantization_no_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "qfp8mb",
        &[Candidate {
            template: "quantization_single_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "csqint4",
        &[Candidate {
            template: "quantization_double_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "csqint4wt",
        &[Candidate {
            template: "quantization_no_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "csqint8",
        &[Candidate {
            template: "quantization_single_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "csqint8ch",
        &[Candidate {
            template: "quantization_double_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "csqint8mb",
        &[Candidate {
            template: "quantization_single_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "csqint8mbv2",
        &[Candidate {
            template: "quantization_single_pad_v2.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "csqint8v2",
        &[Candidate {
            template: "quantization_single_pad_v2.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "quantscalepertoken",
        &[Candidate {
            template: "quant_scale_per_token.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "quantscalepertokenfp8",
        &[Candidate {
            template: "quant_scale_per_token.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "conv2dint8",
        &[
            Candidate {
                template: "convolution2d.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "convolution2d_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
        ],
    ),
    (
        "conv2d",
        &[
            Candidate {
                template: "convolution2d.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "convolution2d_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
        ],
    ),
    (
        "conv2dint4",
        &[
            Candidate {
                template: "convolution2d.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "convolution2d_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
        ],
    ),
    (
        "conv2dfp8",
        &[
            Candidate {
                template: "convolution2d.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "convolution2d_dd1.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
        ],
    ),
    (
        "conv2dgenos1",
        &[Candidate {
            template: "convolution2d_os1.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "conv2dos1",
        &[Candidate {
            template: "convolution2d_os1.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "conv2dint8os1",
        &[Candidate {
            template: "convolution2d_os1.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "conv2dxrfint8os1",
        &[Candidate {
            template: "convolution2d_os1.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "depthwiseconv2dnative",
        &[Candidate {
            template: "depthwise_conv_fwd.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "rope64p1",
        &[Candidate {
            template: "rope.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "rope64p2",
        &[Candidate {
            template: "rope.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "sqrt",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "rsqrt",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "identity",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "shuffle",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "avgpoolnmapfwd",
        &[Candidate {
            template: "pooling.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "lstmactp2",
        &[Candidate {
            template: "lstmactp2.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "mish",
        &[Candidate {
            template: "unary_pipeline.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "abs",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "neg",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "greaterequal",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "lesserequal",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "greaterthan",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "lesserthan",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "equal",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "notequal",
        &[Candidate {
            template: "broadcast_ops.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "dl16tofp32",
        &[Candidate {
            template: "quantization_double_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "fp32todl16",
        &[Candidate {
            template: "quantization_double_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "interslicetranspose_fp16",
        &[Candidate {
            template: "inter_slice_transpose.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "interslicetranspose_fp8",
        &[Candidate {
            template: "inter_slice_transpose.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "fp8todl16",
        &[Candidate {
            template: "quantization_double_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "dl16tobf16",
        &[Candidate {
            template: "quantization_no_pad.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "topkvalue",
        &[Candidate {
            template: "topk.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "topkindex",
        &[Candidate {
            template: "topk.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "maskbyindex",
        &[Candidate {
            template: "topk.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "softplus",
        &[Candidate {
            template: "unary_pipeline.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "ReStickifyOpLx",
        &[
            Candidate {
                template: "restickify.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "restickify.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "restickify_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "ReStickifyOpHBM",
        &[
            Candidate {
                template: "restickify.ddl",
                serves: Serves::Only(IsaGen::Rcudd1a),
            },
            Candidate {
                template: "restickify.ddl",
                serves: Serves::Only(IsaGen::Mpw4),
            },
            Candidate {
                template: "restickify_sen1p5.ddl",
                serves: Serves::Only(IsaGen::Sen1p5),
            },
        ],
    ),
    (
        "floor",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
    (
        "int32idxtoaddr",
        &[Candidate {
            template: "unary_parallel.ddl",
            serves: Serves::EveryArch,
        }],
    ),
];
