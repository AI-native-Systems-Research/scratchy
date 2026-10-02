// SPDX-License-Identifier: Apache-2.0
//! THE SAMPLED ROWS OF A RESULT MATMUL, INSERTED FROM A TARGET'S DECLARED FACTS.
//!
//! The sampler reads one row of the forward's result per sequence, so on a multi-row step the
//! result matmul — the vocabulary projection — need only compute those rows. A target that can
//! compute them apart declares for which weights ([`SampleRowsFacts`]); this pass wraps each such
//! result matmul of a multi-row canonical in one construct (one expansion, no guard):
//!
//! 1. [`SubOp::SampleRowsGather`] moves each sequence's sampled row to the front of the
//!    activation, in place, and the matmul reads it;
//! 2. the matmul;
//! 3. [`SubOp::SampleRowsScatter`] moves its front rows back to the sampled rows, in place;
//! 4. [`SubOp::AllRowsMatmul`], the matmul over every row taking over that output — what a step
//!    that reads every row runs instead of 1–3, and never on a step 1–3 run on.
//!
//! Whether a bake point slices at all, and on which steps each part runs, is the target's
//! lowering: the tape states the dataflow, and a part that does not run leaves the buffer it would
//! rewrite in place as it was. A canonical of one row per sequence is left as it is. A target that
//! samples no rows never calls this.

use std::cmp::Ordering;

use crate::handoff::{Expansion, ExpansionId, LoweredDecode};
use crate::lower::{GemmWeightKind, InputRef, OpDesc};
use crate::subtile_ir::SubOp;

/// A target's sampled rows, as data.
pub struct SampleRowsFacts {
    /// The result matmul weights whose rows the target samples.
    pub weights: &'static [GemmWeightKind],
}

/// Why a canonical's sampled rows could not be inserted, naming the op at fault.
#[derive(Debug)]
pub enum SampleRowsError {
    /// The result matmul has no activation operand.
    NoActivation { op: usize, name: &'static str },
    /// The result matmul is already part of another construct's expansion.
    AlreadyExpanded { op: usize, name: &'static str },
}

impl std::fmt::Display for SampleRowsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoActivation { op, name } => write!(f, "op {op} ({name}): no activation operand"),
            Self::AlreadyExpanded { op, name } => {
                write!(f, "op {op} ({name}): already part of another expansion")
            }
        }
    }
}

/// Wrap `l`'s result matmul in its sampled rows, when `facts` sample its weight and the canonical
/// has more than one row per sequence.
pub fn expand_sample_rows(
    l: &LoweredDecode,
    facts: &SampleRowsFacts,
) -> Result<LoweredDecode, SampleRowsError> {
    let r = l.input.result;
    let od = &l.input.ops[r];
    let SubOp::MatmulTile { weight, .. } = od.op else {
        return Ok(l.clone());
    };
    if od.m < 2 || !facts.weights.contains(&weight.kind()) {
        return Ok(l.clone());
    }
    let name = od.op.name();
    if l.op_expansion[r].is_some() {
        return Err(SampleRowsError::AlreadyExpanded { op: r, name });
    }
    let (rows, weights) =
        (od.inputs.split_first()).ok_or(SampleRowsError::NoActivation { op: r, name })?;
    // The construct takes op `r`'s place; every reader of the result reads its end.
    let (gather, matmul, scatter, all_rows) = (r, r + 1, r + 2, r + 3);
    let at = |j: usize| match j.cmp(&r) {
        Ordering::Less => j,
        Ordering::Equal => all_rows,
        Ordering::Greater => j + 3,
    };
    let input = |i: &InputRef| match *i {
        InputRef::Op(j) => InputRef::Op(at(j)),
        ext => ext,
    };
    let desc = |op, inputs| OpDesc {
        op,
        m: od.m,
        inputs,
    };
    let weights: Vec<InputRef> = weights.iter().map(input).collect();
    let chain = |first: Vec<InputRef>| first.into_iter().chain(weights.iter().copied()).collect();
    let construct = [
        desc(SubOp::SampleRowsGather, vec![input(rows)]),
        desc(od.op, chain(vec![InputRef::Op(gather)])),
        desc(SubOp::SampleRowsScatter, vec![InputRef::Op(matmul)]),
        desc(
            SubOp::AllRowsMatmul,
            chain(vec![InputRef::Op(gather), InputRef::Op(scatter)]),
        ),
    ];
    let mut x = l.clone();
    for od in &mut x.input.ops {
        od.inputs.iter_mut().for_each(|i| *i = input(i));
    }
    x.input.ops.splice(r..=r, construct);
    x.input.result = all_rows;
    // The gather is its activation's buffer; the scatter and the all-rows matmul the matmul's.
    let rows_tile = match *rows {
        InputRef::Op(j) => l.op_tiles[j],
        InputRef::Ext(_) => None,
    };
    let tile = l.op_tiles[r];
    x.op_tiles.splice(r..=r, [rows_tile, tile, tile, tile]);
    let id = l.op_expansion.iter().flatten().map(|e| e.id.0 + 1).max();
    let own = Some(Expansion {
        id: ExpansionId(id.unwrap_or(0)),
        guard: None,
    });
    x.op_expansion.splice(r..=r, [own; 4]);
    let op_of = |j: usize| if j == r { matmul } else { at(j) };
    x.norm_gain_add_tiles = (l.norm_gain_add_tiles.iter())
        .map(|(op, tile)| (op_of(*op), *tile))
        .collect();
    Ok(x)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::handoff::SourceBinding;
    use crate::lower::{AffineInt4, GemmWeight, LoweringInput};
    use crate::subtile_ir::{GainConvention, SourceShape, ValidatedGraph, lower_region};
    use crate::subtile_tape::lower_dag_to_tape;
    use crate::tape_colouring::{ColourFacts, ColourRule, FoldFacts, OutputAlias, colour_tape};
    use crate::tape_steps::OperandIx;
    use InputRef::{Ext, Op};

    const AFFINE: SampleRowsFacts = SampleRowsFacts {
        weights: &[GemmWeightKind::Affine],
    };

    fn affine(k: u32) -> GemmWeight {
        let affine = AffineInt4::mint(4, 64, k).expect("a 4-bit g64 weight");
        GemmWeight::Affine { affine }
    }

    /// A final norm and the result projection onto a 512-wide vocabulary, `m` rows each: op `i`
    /// realizes tile `i`.
    fn head(m: u32, weight: GemmWeight) -> LoweredDecode {
        let desc = |op, inputs| OpDesc { op, m, inputs };
        let norm = SubOp::RmsNorm {
            eps: 1e-5,
            gain: GainConvention::Scale,
        };
        let input = LoweringInput {
            sources: vec![
                SourceShape { rows: 1, cols: 256 },
                SourceShape { rows: 1, cols: 256 },
                SourceShape {
                    rows: 256,
                    cols: 512,
                },
            ],
            ops: vec![
                desc(norm, vec![Ext(0), Ext(1)]),
                desc(SubOp::MatmulTile { n: 512, weight }, vec![Op(0), Ext(2)]),
            ],
            result: 1,
        };
        LoweredDecode {
            input,
            bindings: vec![
                SourceBinding::EmbeddedHidden,
                SourceBinding::Weight { id: 0, index: None },
                SourceBinding::Weight { id: 1, index: None },
            ],
            op_tiles: vec![Some((0, 0)), Some((1, 0))],
            norm_gain_add_tiles: Default::default(),
            op_expansion: vec![None; 2],
        }
    }

    fn names(l: &LoweredDecode) -> Vec<&'static str> {
        l.input.ops.iter().map(|od| od.op.name()).collect()
    }

    /// A multi-row result matmul: the gather over its activation, which it reads; the scatter
    /// over its output; the all-rows matmul over the gathered rows and the scattered output, with
    /// the matmul's weight — the new result. One construct, unguarded; the gather is its
    /// activation's tile, the rest the matmul's.
    #[test]
    fn a_multi_row_result_matmul_is_sampled() {
        let x = expand_sample_rows(&head(2, affine(256)), &AFFINE).expect("expands");
        assert_eq!(
            names(&x),
            [
                "RmsNorm",
                "SampleRowsGather",
                "MatmulTile",
                "SampleRowsScatter",
                "AllRowsMatmul"
            ]
        );
        let ins = |i: usize| x.input.ops[i].inputs.clone();
        assert_eq!(ins(1), [Op(0)]);
        assert_eq!(ins(2), [Op(1), Ext(2)]);
        assert_eq!(ins(3), [Op(2)]);
        assert_eq!(ins(4), [Op(1), Op(3), Ext(2)]);
        assert_eq!(x.input.result, 4);
        let (t0, t1) = (Some((0, 0)), Some((1, 0)));
        assert_eq!(x.op_tiles, [t0, t0, t1, t1, t1]);
        let id = x.op_expansion[1].map(|e| e.id);
        assert!(id.is_some() && x.op_expansion[1..].iter().all(|e| e.map(|e| e.id) == id));
        assert!(
            x.op_expansion[1..]
                .iter()
                .all(|e| e.is_some_and(|e| e.guard.is_none()))
        );
        assert_eq!(x.op_expansion[0], None);
        assert!(x.input.ops.iter().all(|od| od.m == 2));
    }

    /// One row per sequence, a weight the target does not sample, or a result that is not a
    /// matmul: the op list comes back as it was.
    #[test]
    fn only_a_listed_multi_row_result_matmul_is_sampled() {
        let unchanged = |l: LoweredDecode| {
            let x = expand_sample_rows(&l, &AFFINE).expect("passes through");
            assert_eq!(names(&x), names(&l));
            assert_eq!(x.input.result, l.input.result);
        };
        unchanged(head(1, affine(256)));
        unchanged(head(2, GemmWeight::Dense));
        let mut capped = head(2, affine(256));
        let cap = OpDesc {
            op: SubOp::TanhSoftCap,
            m: 2,
            inputs: vec![Op(1)],
        };
        capped.input.ops.push(cap);
        capped.input.result = 2;
        capped.op_tiles.push(Some((2, 0)));
        capped.op_expansion.push(None);
        unchanged(capped);
    }

    /// A result matmul some construct already expanded to is refused, by name.
    #[test]
    fn an_expanded_result_matmul_is_refused() {
        let mut l = head(2, affine(256));
        l.op_expansion[1] = Some(Expansion {
            id: ExpansionId(0),
            guard: None,
        });
        match expand_sample_rows(&l, &AFFINE) {
            Err(SampleRowsError::AlreadyExpanded { op: 1, name }) => assert_eq!(name, "MatmulTile"),
            other => panic!("expected AlreadyExpanded at op 1, got {other:?}"),
        }
    }

    /// The construct takes the next expansion id after every existing one.
    #[test]
    fn the_construct_takes_a_fresh_expansion_id() {
        let mut l = head(2, affine(256));
        l.op_expansion[0] = Some(Expansion {
            id: ExpansionId(6),
            guard: None,
        });
        let x = expand_sample_rows(&l, &AFFINE).expect("expands");
        assert_eq!(x.op_expansion[2].map(|e| e.id), Some(ExpansionId(7)));
    }

    /// Under a target's in-place facts for the construct, the sampled canonical colours as the
    /// plain one does: every inserted step aliases a buffer the plain tape already holds, so the
    /// colour count and the result's colour are the matmul's.
    #[test]
    fn the_construct_colours_as_the_plain_matmul() {
        fn rule(op: &SubOp) -> ColourRule {
            let fresh = ColourRule::FRESH;
            let over = |k| fresh.output(OutputAlias::Operand(OperandIx(k)));
            match op {
                SubOp::SampleRowsGather | SubOp::SampleRowsScatter => over(0),
                SubOp::AllRowsMatmul => over(1),
                SubOp::MatmulTile { .. } => fresh.reads_across(),
                _ => fresh,
            }
        }
        let facts = ColourFacts {
            rule,
            colour_zero: SourceBinding::EmbeddedHidden,
        };
        let colours = |l: &LoweredDecode| {
            let graph = lower_region(&l.input, std::num::NonZeroU32::MAX);
            let tape = lower_dag_to_tape(&ValidatedGraph::new(&graph).expect("a valid graph"));
            let c = colour_tape(&graph, &tape, &l.bindings, &FoldFacts::default(), &facts)
                .expect("colours");
            (c.count(), c.result())
        };
        let plain = head(2, affine(256));
        let sampled = expand_sample_rows(&plain, &AFFINE).expect("expands");
        assert_eq!(colours(&sampled), colours(&plain));
    }
}
