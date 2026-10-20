// SPDX-License-Identifier: Apache-2.0
//! ONE CANONICAL, LOWERED — CUDA's orchestration of the shared passes.
//!
//! In metal's order (`scratchy-target-metal-compiler`'s `canonical::lower_canonical`), under CUDA's
//! declared tables (`scratchy_target_cuda::op_abi`): the sampled rows, the graph and tape built
//! from whole ops, the fold pass, the colourer. Then CUDA's opcode lowering ([`steps_from_tape`]).
//!
//! No KV codec (CUDA is bf16 only) and no wave order (that pass exists because metal's barriers
//! drain everything in flight; CUDA runs one stream in order). No re-roll either: the unrolled
//! tape is what the colourer and the fold pass read, and nothing here lays steps out along a loop.
//!
//! [`steps_from_tape`]: crate::steps_from_tape

use scratchy_subtile::handoff::LoweredDecode;
use scratchy_subtile::sample_rows::{SampleRowsError, expand_sample_rows};
use scratchy_subtile::subtile_ir::{SubtileIR, ValidatedGraph, lower_region};
use scratchy_subtile::subtile_tape::{Instr, SubtileTape, lower_dag_to_tape};
use scratchy_subtile::tape_colouring::{ColourCount, ColourError, TapeColouring, colour_tape};
use scratchy_subtile::tape_folding::{FoldError, ModelFoldFacts, TapeFolds, fold_tape};
use scratchy_target_cuda::op_abi::{
    CUDA_COLOUR_FACTS, CUDA_COLUMN_BLOCK, CUDA_FUSIONS, CUDA_SAMPLE_ROWS, CudaFusion,
};
use scratchy_target_cuda::step::CudaStep;

use crate::steps_from_tape::lower_step;

/// Which canonical this is, for the refusals.
#[derive(Clone, Copy, Debug)]
pub struct CanonicalAt<'a> {
    pub stem: &'a str,
    /// The canonical's rows.
    pub m: u64,
}

/// The shared passes' result for one canonical, before opcode lowering.
pub struct CudaPlan {
    /// The canonical after CUDA's sampled rows were inserted: what the graph was built from.
    pub lowered: LoweredDecode,
    pub graph: SubtileIR,
    /// The unrolled tape of [`Self::graph`].
    pub tape: SubtileTape,
    pub folds: TapeFolds<CudaFusion>,
    pub colouring: TapeColouring,
}

/// One decode canonical, lowered.
pub struct CudaCanonical {
    pub colours: ColourCount,
    /// One step per `Compute` on the tape, in tape order.
    pub steps: Vec<CudaStep>,
}

/// Why a canonical could not be lowered.
#[derive(Debug)]
pub enum CanonicalRefusal {
    SampleRows(SampleRowsError),
    /// The graph built from the canonical is not a valid DAG.
    Graph(String),
    Fold(FoldError),
    Colour(ColourError),
    /// The `step`-th `Compute` on the tape is an op CUDA has no lowering for yet.
    NotLowered {
        step: usize,
        op: &'static str,
    },
}

impl std::fmt::Display for CanonicalRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SampleRows(e) => write!(f, "the shared sample-rows pass refused: {e}"),
            Self::Graph(e) => write!(f, "the graph is not a valid DAG: {e}"),
            Self::Fold(e) => write!(f, "the shared fold pass refused: {e}"),
            Self::Colour(e) => write!(f, "tape colorer refused: {e}"),
            Self::NotLowered { step, op } => write!(
                f,
                "step {step} computes {op}, which the cuda backend does not lower yet"
            ),
        }
    }
}

/// Run the shared passes on one decode canonical under CUDA's tables: sampled rows, graph and
/// tape, folds, colours.
pub fn plan_canonical(
    l: &LoweredDecode,
    at: CanonicalAt<'_>,
) -> Result<CudaPlan, CanonicalRefusal> {
    let CanonicalAt { stem, m } = at;
    // CUDA's lm_head slice: none is declared, so this inserts nothing — but it runs, so a row
    // added to the table takes effect without touching this function.
    let lowered = expand_sample_rows(l, &CUDA_SAMPLE_ROWS).map_err(CanonicalRefusal::SampleRows)?;
    let graph = lower_region(&lowered.input, CUDA_COLUMN_BLOCK);
    let tape = {
        let valid = ValidatedGraph::new(&graph)
            .map_err(|e| CanonicalRefusal::Graph(format!("{stem} m={m}: {e}")))?;
        lower_dag_to_tape(&valid)
    };
    // The facts only gate patterns, and CUDA's table declares none.
    let model = ModelFoldFacts {
        fold_projections: false,
        matvec_ends: false,
        matvec_norms: false,
        row_programs: false,
        hidden_out: false,
    };
    let folds =
        fold_tape(&graph, &tape, &lowered, &CUDA_FUSIONS, model).map_err(CanonicalRefusal::Fold)?;
    let colouring = colour_tape(
        &graph,
        &tape,
        &lowered.bindings,
        &folds.fold_facts(),
        &CUDA_COLOUR_FACTS,
    )
    .map_err(CanonicalRefusal::Colour)?;
    Ok(CudaPlan {
        lowered,
        graph,
        tape,
        folds,
        colouring,
    })
}

/// Lower one decode canonical: [`plan_canonical`], then every `Compute` on the tape through CUDA's
/// opcode lowering, in tape order.
pub fn lower_canonical(
    l: &LoweredDecode,
    at: CanonicalAt<'_>,
) -> Result<CudaCanonical, CanonicalRefusal> {
    let plan = plan_canonical(l, at)?;
    let steps = (plan.tape.instrs().iter())
        .filter_map(|i| match i {
            Instr::Compute { node, .. } => Some(&plan.graph.nodes[node.index()].op),
            _ => None,
        })
        .enumerate()
        .map(|(step, op)| lower_step(step, op))
        .collect::<Result<Vec<_>, _>>()?;
    Ok(CudaCanonical {
        colours: plan.colouring.count(),
        steps,
    })
}
