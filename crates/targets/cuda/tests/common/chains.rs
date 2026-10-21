// SPDX-License-Identifier: Apache-2.0
//! GEMM chains: the canonicals the tape tests lower, and the `static` tapes written out by hand
//! from the issue's numbers, which `tape_layout` checks against what the compiler lowers.
//!
//! Every source is declared in the IR as `[K, N]`; the weight's bytes on the device are `[N, K]`.
//! Source 0 is the activation, rebound to the embedded hidden: colour 0, the forward's input.

use scratchy_subtile::fixtures::front_end_lowered;
use scratchy_subtile::handoff::{LoweredDecode, SourceBinding};
use scratchy_subtile::lower::{GemmWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{SourceShape, SubOp};
use scratchy_target_cuda::step::{CudaStep, GemmBf16};
use scratchy_target_cuda::tape::{ArenaLayout, ArenaOffset, ArenaSpan, CudaTape, WeightIx};
use scratchy_target_cuda_compiler::canonical::{CanonicalAt, lower_canonical};

/// Panic unless `tape` is, field by field, what CUDA's compiler lowers `l` to.
pub fn assert_tape_is_lowered(name: &str, tape: &CudaTape, l: &LoweredDecode) {
    let at = CanonicalAt {
        stem: name,
        m: u64::from(tape.bucket_m),
    };
    let c = lower_canonical(l, at).unwrap_or_else(|e| panic!("{name}: lower_canonical: {e}"));
    assert_eq!(tape.bucket_m, c.bucket_m, "{name}: bucket_m");
    assert_eq!(
        tape.arena.colour_bytes, c.colour_bytes,
        "{name}: colour bytes"
    );
    assert_eq!(
        tape.arena.colour_offsets, c.colour_offsets,
        "{name}: offsets"
    );
    assert_eq!(tape.arena.total_bytes, c.total_bytes, "{name}: total");
    assert_eq!(tape.input, c.input, "{name}: input");
    assert_eq!(tape.result, c.result, "{name}: result");
    assert_eq!(tape.weight_bytes, c.weight_bytes, "{name}: weight_bytes");
    assert_eq!(tape.steps, c.steps, "{name}: steps");
}

/// The chains' activation width: `x` is `[M, HIDDEN]`.
pub const HIDDEN: u32 = 512;

/// `x [m, k0]` through one dense GEMM per `ns` entry, each reading the one before; the last is
/// the result. Every source bound to a weight, as the shared front end binds them.
pub fn chain_input(m: u32, k0: u32, ns: &[u32]) -> LoweringInput {
    let mut sources = vec![SourceShape { rows: m, cols: k0 }];
    let mut ops = Vec::new();
    let mut k = k0;
    for (i, &n) in ns.iter().enumerate() {
        sources.push(SourceShape { rows: k, cols: n });
        let act = match i {
            0 => InputRef::Ext(0),
            i => InputRef::Op(i - 1),
        };
        ops.push(OpDesc {
            op: SubOp::MatmulTile {
                n,
                weight: GemmWeight::Dense,
            },
            m,
            inputs: vec![act, InputRef::Ext(i + 1)],
        });
        k = n;
    }
    LoweringInput {
        exported: None,
        result: ops.len() - 1,
        sources,
        ops,
    }
}

/// `input` as the shared front end binds it, with source 0 rebound to the embedded hidden.
pub fn with_hidden(input: LoweringInput) -> LoweredDecode {
    let mut l = front_end_lowered(input);
    l.bindings[0] = SourceBinding::EmbeddedHidden;
    l
}

/// `x [m, 512] → W1 (N = 1024) → y1 [m, 1024] → W2 (N = 512) → y2 [m, 512]`.
pub fn two_gemm(m: u32) -> LoweredDecode {
    with_hidden(chain_input(m, HIDDEN, &[1024, 512]))
}

/// [`two_gemm`] at `m = 256`, then `y2 → W3 (N = 1024) → y3 [256, 1024]`.
pub fn three_gemm() -> LoweredDecode {
    with_hidden(chain_input(256, HIDDEN, &[1024, 512, 1024]))
}

const fn gemm(a: u64, b: u32, c: u64, m: u32, n: u32, k: u32) -> CudaStep {
    CudaStep::Gemm(GemmBf16 {
        a: ArenaOffset(a),
        b: WeightIx(b),
        c: ArenaOffset(c),
        m,
        n,
        k,
    })
}

const fn span(offset: u64, bytes: u64) -> ArenaSpan {
    ArenaSpan {
        offset: ArenaOffset(offset),
        bytes,
    }
}

/// [`two_gemm`]`(128)`: colours x, y1, y2.
pub static TWO_GEMM_M128: CudaTape = CudaTape {
    bucket_m: 128,
    arena: ArenaLayout {
        colour_bytes: &[131072, 262144, 131072],
        colour_offsets: &[0, 131072, 393216],
        total_bytes: 524288,
    },
    input: span(0, 131072),
    result: span(393216, 131072),
    weight_bytes: &[0, 1048576, 1048576],
    steps: &[
        gemm(0, 1, 131072, 128, 1024, 512),
        gemm(131072, 2, 393216, 128, 512, 1024),
    ],
};

/// [`two_gemm`]`(256)`: colours x, y1, y2.
pub static TWO_GEMM_M256: CudaTape = CudaTape {
    bucket_m: 256,
    arena: ArenaLayout {
        colour_bytes: &[262144, 524288, 262144],
        colour_offsets: &[0, 262144, 786432],
        total_bytes: 1048576,
    },
    input: span(0, 262144),
    result: span(786432, 262144),
    weight_bytes: &[0, 1048576, 1048576],
    steps: &[
        gemm(0, 1, 262144, 256, 1024, 512),
        gemm(262144, 2, 786432, 256, 512, 1024),
    ],
};

/// [`three_gemm`]: `y1` is dead after step 1, so `y3` takes its colour, and the arena is the
/// two-GEMM one. `y3` is the result.
pub static THREE_GEMM_M256: CudaTape = CudaTape {
    bucket_m: 256,
    arena: ArenaLayout {
        colour_bytes: &[262144, 524288, 262144],
        colour_offsets: &[0, 262144, 786432],
        total_bytes: 1048576,
    },
    input: span(0, 262144),
    result: span(262144, 524288),
    weight_bytes: &[0, 1048576, 1048576, 1048576],
    steps: &[
        gemm(0, 1, 262144, 256, 1024, 512),
        gemm(262144, 2, 786432, 256, 512, 1024),
        gemm(786432, 3, 262144, 256, 1024, 512),
    ],
};
