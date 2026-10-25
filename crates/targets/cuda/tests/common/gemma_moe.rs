// SPDX-License-Identifier: Apache-2.0
//! Gemma 4 26B-A4B's sparse MoE block at a small geometry (#885), as the shared bridge's
//! `GemmaMoe` arm expands it under `DenseExperts::Lower`: the router's own pre-norm into the MoE
//! scratch, its 128-expert projection, the Gemma routing (top-k of the raw logits, the gathered
//! scores times `hidden^-0.5`, a softmax over the picks, the per-expert scale), the dense
//! SwitchGLU bank's GeGLU experts, and the combine.
//!
//! Geometry: hidden `H = 256`, 128 experts, top-8, expert width `I = 128` (each a multiple of 64
//! and not of 256 but `H`, as Gemma's 2816 and 704 are 64-column-tile widths).
//!
//! Sources, every weight one tensor of its bundle (`SourceBinding::BundleTensor`): 0 `x` (the
//! embedded hidden), 1 the router scale `[1, H]`, 2 the router projection `[H, E]`, 3 the
//! per-expert scale `[1, E]`, 4 the gate `[E·I, H]`, 5 the up `[E·I, H]`, 6 the down `[E·H, I]`.

use scratchy_subtile::handoff::{
    BundleTensor, Expansion, ExpansionId, LoweredDecode, SourceBinding, UnrollIndex,
};
use scratchy_subtile::lower::{ExpertWeight, InputRef, LoweringInput, OpDesc};
use scratchy_subtile::subtile_ir::{
    ExpertBundle, ExpertProj, GatedAct, NumExperts, RouterBundle, SharedExpertBound, SourceShape,
    SubOp, TopK,
};
use scratchy_target_cuda_compiler::canonical::CudaModelFacts;

use super::chains::{GEMMA4_FACTS, with_hidden};

pub const H: u32 = 256;
pub const EXPERTS: u32 = 128;
pub const TOP_K: u32 = 8;
pub const I: u32 = 128;
pub const EPS: f32 = 1e-6;
/// The router's and the experts' bundle ids, and the block's layer.
pub const ROUTER: u32 = 20;
pub const BANK: u32 = 21;
pub const LAYER: u64 = 0;

/// Gemma 4's facts for the block: its GEMMs also take the 64-column tile.
pub const GEMMA_MOE_FACTS: CudaModelFacts = GEMMA4_FACTS;

/// The block's ops at `m` rows, as the bridge emits them.
pub fn gemma_moe_input(m: u32) -> LoweringInput {
    let nz = |n| std::num::NonZeroU32::new(n).expect("non-zero");
    let (experts, k, bundle) = (
        NumExperts::new(nz(EXPERTS)),
        TopK::new(nz(TOP_K)),
        ExpertBundle::SwitchGlu,
    );
    let gemma = RouterBundle::Gemma;
    let op = |op, inputs| OpDesc { op, m, inputs };
    let expert = |proj, n, inputs| {
        op(
            SubOp::ExpertMatmul {
                proj,
                n,
                k,
                weight: ExpertWeight::Dense,
                bundle,
            },
            inputs,
        )
    };
    use InputRef::{Ext, Op};
    LoweringInput {
        exported: None,
        sources: vec![
            SourceShape { rows: m, cols: H },
            SourceShape { rows: 1, cols: H },
            SourceShape {
                rows: H,
                cols: EXPERTS,
            },
            SourceShape {
                rows: 1,
                cols: EXPERTS,
            },
            SourceShape {
                rows: EXPERTS * I,
                cols: H,
            },
            SourceShape {
                rows: EXPERTS * I,
                cols: H,
            },
            SourceShape {
                rows: EXPERTS * H,
                cols: I,
            },
        ],
        ops: vec![
            op(
                SubOp::RouterNorm {
                    eps: EPS,
                    router: gemma,
                },
                vec![Ext(0), Ext(1)],
            ),
            op(
                SubOp::RouterLogits {
                    experts,
                    router: gemma,
                },
                vec![Op(0), Ext(2)],
            ),
            op(SubOp::RouteArgsort, vec![Op(1)]),
            op(SubOp::RouteTopK { k }, vec![Op(2)]),
            op(SubOp::RouteGatherScores, vec![Op(1), Op(3)]),
            op(
                SubOp::RouteScale {
                    scale: (H as f32).powf(-0.5),
                },
                vec![Op(4)],
            ),
            op(SubOp::RouteSoftmax, vec![Op(5)]),
            op(
                SubOp::RouteExpertScale { router: gemma },
                vec![Op(6), Op(3), Ext(3)],
            ),
            op(
                SubOp::ExpertSort { experts, k, bundle },
                vec![Ext(0), Op(3)],
            ),
            expert(ExpertProj::Gate, I, vec![Op(8), Op(8), Ext(4)]),
            expert(ExpertProj::Up, I, vec![Op(8), Op(8), Ext(5)]),
            op(
                SubOp::ExpertGatedAct {
                    act: GatedAct::Gelu,
                },
                vec![Op(9), Op(10)],
            ),
            expert(ExpertProj::Down, H, vec![Op(11), Op(8), Ext(6)]),
            op(SubOp::ExpertUnsort, vec![Op(12), Op(8)]),
            op(
                SubOp::ExpertCombine {
                    hidden: H,
                    shared: SharedExpertBound(None),
                },
                vec![Op(13), Op(7)],
            ),
        ],
        result: 14,
    }
}

/// The bundle tensor each weight source is, by source index.
pub const TENSORS: [(usize, u32, BundleTensor); 6] = [
    (1, ROUTER, BundleTensor::RouterScale),
    (2, ROUTER, BundleTensor::RouterGate),
    (3, ROUTER, BundleTensor::RouterPerExpertScale),
    (4, BANK, BundleTensor::ExpertGateW),
    (5, BANK, BundleTensor::ExpertUpW),
    (6, BANK, BundleTensor::ExpertDownW),
];

/// The block at `m` rows as the front end hands it over: source 0 the embedded hidden, each
/// weight one bundle tensor of layer 0's router or bank, every op one MoE expansion.
pub fn gemma_moe(m: u32) -> LoweredDecode {
    let mut l = with_hidden(gemma_moe_input(m));
    for (ix, id, tensor) in TENSORS {
        l.bindings[ix] = SourceBinding::BundleTensor {
            id,
            index: Some(UnrollIndex(LAYER)),
            tensor,
        };
    }
    for e in l.op_expansion.iter_mut() {
        *e = Some(Expansion {
            id: ExpansionId(0),
            guard: None,
        });
    }
    l
}
