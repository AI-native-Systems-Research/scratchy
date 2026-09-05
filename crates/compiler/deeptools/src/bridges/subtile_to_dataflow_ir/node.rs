//! ONE NODE OF THE TAPE, IN A VOCABULARY THIS CRATE CAN NAME.
//!
//! ⛔⛔ THIS IS NOT A SECOND `SubtileNode`, AND IT IS NOT A `Dsc`. It is the half of bridge 1 that
//! can live here at all: `deeptools` must never depend on scratchy, so the side that speaks
//! `SubOp` and `TensorRegion` sits in scratchy and fills these in. What crosses is an op-func, a
//! format, and extents — every one of them a token or a newtype, so the crossing itself is checked.
//!
//! ⛔ AND NOTHING IS RECOVERED HERE. Every field is STATED by the producer. The nuked attempt built
//! a lowering, threw the shape into string dimension names, and tried to read it back; the fields
//! below exist precisely so there is nothing left to recover.

use crate::arch::{Bytes, Elements};
use crate::generated::{DataType, OpFunc};

/// A ROW COUNT — how many tokens a tensor covers.
///
/// ⛔ A SEPARATE TYPE FROM [`Cols`] BECAUSE TRANSPOSING THEM IS THE DEFECT. `[rows, cols]` and
/// `[cols, rows]` are both plausible and only one addresses the tensor the tape wrote. An
/// `Op<Target, 8, 64, 100, 1, 0>` was written once and thrown away for exactly this: five bare
/// integers in a row, where a transposition compiles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Rows(pub u32);

/// A COLUMN COUNT — how wide a tensor is.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Cols(pub u32);

/// THE CONTRACTED EXTENT of a matmul — the K a dot product runs over.
///
/// ⛔ NOT A [`Cols`], THOUGH IT IS READ FROM ONE. K comes from the A-slice's column extent, and the
/// output's own `cols` is N. Two column counts in one op, and using the wrong one gives a matmul
/// that contracts over its output width.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Contraction(pub u32);

/// WHICH HBM SEGMENT a tensor lives in.
///
/// ⛔⛔ THE SEGMENT IS PART OF THE ADDRESS, NOT A DETAIL. The worker H2Ds one device region per
/// occupied segment, and an offset means nothing without the segment it is an offset INTO. An
/// address that dropped the segment is what makes a program read memory nothing filled.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct Segment(pub u32);

/// WHERE AN OPERAND ACTUALLY IS.
///
/// ⛔⛔ THIS ENUM IS THE WHOLE LESSON OF THE NUKED BRANCH. Every operand there was addressed into
/// the LX, including weights that only ever exist in HBM, so the emitted programs read an address
/// nothing ever wrote. A weight is [`Residence::Hbm`] and it reaches the LX only by a transfer the
/// program itself contains — see [`super::transfer`] and [`crate::islands::dataflow_ir::op::Op`]'s
/// `CompositeLoadAndStore`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Residence {
    /// In the device's global memory, at a byte offset within its segment.
    ///
    /// ⛔ THE OFFSET IS IN BYTES HERE AND IN ELEMENTS IN THE VIEW. scratchy's placements are byte
    /// offsets; DataflowIR addresses in elements (`Dataflow.td`, `get_logical_memory_view`). The
    /// conversion is [`Operand::start`], and it is a type change so that it cannot be applied
    /// twice or not at all.
    Hbm {
        /// Which segment.
        segment: Segment,
        /// The byte offset within that segment's region.
        offset: Bytes,
    },
    /// Staged in a core's scratchpad, at an element offset.
    ///
    /// ⭐ AN LX RESIDENCE IS EARNED, NOT ASSUMED. It means something already put the tensor there:
    /// a previous node's output, or a transfer this program emitted.
    Lx {
        /// The element offset within the LX.
        offset: Elements,
    },
}

/// ONE OPERAND: its shape, its format, and where it is.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Operand {
    /// How many rows.
    pub rows: Rows,
    /// How many columns.
    pub cols: Cols,
    /// The element format.
    pub format: DataType,
    /// Where it lives.
    pub at: Residence,
}

impl Operand {
    /// HOW MANY ELEMENTS THIS OPERAND COVERS.
    #[must_use]
    pub const fn elements(self) -> Elements {
        Elements(self.rows.0 as u64 * self.cols.0 as u64)
    }

    /// THE VIEW'S START ADDRESS, IN ELEMENTS.
    ///
    /// ⛔⛔ THE ONE PLACE BYTES BECOME ELEMENTS. A `Residence::Hbm` offset is in bytes because
    /// that is what a placement is; a view's `start_address` is in elements because that is what
    /// DataflowIR addresses in. Doing this conversion anywhere else means doing it twice
    /// somewhere.
    ///
    /// ⛔ AND THE SEGMENT IS NOT IN IT. This is the offset WITHIN a segment; which segment it is
    /// belongs to the unit the view is taken over, not to the start.
    /// # Panics
    ///
    /// If an HBM placement's byte offset is not a whole number of elements of this format, which
    /// would put the view's first element half-way through one.
    #[must_use]
    pub fn start(self) -> Elements {
        match self.at {
            Residence::Hbm { offset, .. } => {
                // ⛔ VIA BITS, BECAUSE int4 AND fp4 ARE HALF A BYTE. `offset_bytes / bytes_per_elem`
                // has no answer for them; `offset_bytes * 8 / bits` has the right one, and for a
                // whole-byte format it is the same division.
                let bits = u64::from(self.format.bits().0);
                let offset_bits = offset.0 * 8;
                assert!(
                    offset_bits % bits == 0,
                    "an HBM placement whose byte offset is not a whole number of elements: the \
                     view would start part-way through one"
                );
                Elements(offset_bits / bits)
            }
            Residence::Lx { offset } => offset,
        }
    }
}

/// ONE NODE OF THE FORWARD, READY TO LOWER.
///
/// ⭐ THE SCHEDULE AND THE SHAPE, SIDE BY SIDE. [`Node::op_func`] and [`Node::format`] choose the
/// template, which is the SCHEDULE and knows no extents; [`Node::inputs`] and [`Node::output`] are
/// the SHAPE. Neither can be derived from the other, which is why both are here.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Node {
    /// Which op-func — the sealed set scratchy can emit.
    pub op_func: OpFunc,
    /// The precision the template is selected at.
    ///
    /// ⛔ PART OF THE QUESTION, NOT DECORATION. A template serves an op-func AT A PRECISION, and
    /// resolving without it hands an fp16 op the fp32 kernel.
    pub format: DataType,
    /// The operands read, in the order the op-func's `operation_bind` declares them.
    pub inputs: Vec<Operand>,
    /// The operand written.
    pub output: Operand,
}

impl Node {
    /// THE CONTRACTED EXTENT, for the ops that have one.
    ///
    /// ⭐ READ FROM THE FIRST INPUT'S COLUMNS, which is the A-slice's `[m, k]`. The output's own
    /// `cols` is N, so taking K from there would contract over the output width.
    #[must_use]
    pub fn contraction(&self) -> Option<Contraction> {
        self.inputs.first().map(|a| Contraction(a.cols.0))
    }
}
