// SPDX-License-Identifier: Apache-2.0
//! ⭐ ONE LAYOUT PER MLX-AFFINE CODES TENSOR, DECIDED AT THE BAKE.
//!
//! Each MLX-affine codes tensor a model's tapes read is stored one way: [`decide`] picks it from
//! the tensor's own facts ([`CodesTensor`]) and the chip class the tape is baked for, once per
//! tensor, and the model's tapes for that class carry the result as one table
//! ([`AffineCodesTable`], [`LoweredMetalTape::codes`]). Every command reading a tensor's codes
//! takes its codes constants from that tensor's entry; the load stores every entry's tensor that
//! way once, before any command binds it. No reader decides, so no two readers can disagree.
//!
//! [`LoweredMetalTape::codes`]: crate::tape::lowered::LoweredMetalTape::codes

use crate::tape::ids::{AffineBits, AffineGroupSize, KDim, LayerId, SourceIx};
pub use crate::tape::kernel_constants::{AffineCodes, PlanesK, RowCodes};
use crate::tape::lowered::{CodesRole, GenClass, WeightTensor};

/// What decides how one codes tensor is stored: its role in the model, its code width, its rows'
/// input count and its group size.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CodesTensor {
    pub role: CodesRole,
    pub bits: AffineBits,
    pub k: KDim,
    pub group: AffineGroupSize,
}

/// ⭐ THE ONE DECISION: how the codes of `t` are stored on `class`, for a model that is a drafter
/// when `drafter` is `Some`.
///
/// Row-whole ([`RowCodes::of`]: offset-8 for 4-bit codes on a class storing them so) unless every
/// law below lets the tensor be regrouped into planes, which the wide class's W4A8 small-M GEMMs
/// read (Qwen3.8-27B's verify shapes, 16 rows, µs row-whole / planes: gate/up 149 / 139, down
/// 144 / 137, gdn qkv 88 / 80, lm_head 1356 / 1256; the one-row matvec within ±4%). Both class
/// choices are hand-tuned to those measurements, not derived:
///
/// - a projection only ([`CodesRole::Projection`]): every other role is read a row at a time;
/// - the wide class only ([`GenClass::M5Wide`]);
/// - never a drafter's: a DFlash2 drafter's projections stored as planes miscompute (Qwen3.8-27B,
///   replies 6-14 of 16 identical) — stated by hand, its cause not found;
/// - offset-8 codes only (the planes kernels read signed codes), in whole 64-input chunks
///   ([`PlanesK`]);
/// - a group the matrix-unit kernels take: a weight of group size 32 has no W4A8 GEMM, so its
///   small-M steps take the bf16 twin, which reads the codes row-major.
pub fn decide(
    class: GenClass,
    drafter: Option<scratchy_ir::model_facts::Drafter>,
    t: CodesTensor,
) -> AffineCodes {
    let rows = RowCodes::of(class, t.bits);
    let planes = match (t.role, rows) {
        (CodesRole::Projection, RowCodes::Offset8)
            if class == GenClass::M5Wide && drafter.is_none() && t.group.nax().is_some() =>
        {
            PlanesK::new(t.k.get())
        }
        _ => None,
    };
    match planes {
        Some(k) => AffineCodes::Planes { k },
        None => AffineCodes::Rows(rows),
    }
}

/// One codes tensor a tape reads: the family, tensor and layer a command binds it at, with the
/// facts [`decide`] takes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct CodesEntry {
    pub ix: SourceIx,
    pub which: WeightTensor,
    pub layer: LayerId,
    pub codes: AffineCodes,
}

impl CodesEntry {
    /// The order a table holds its entries in, which [`AffineCodesTable::of`] searches by.
    fn key(&self) -> (u32, WeightTensor, u32) {
        (self.ix.0, self.which, self.layer.0)
    }
}

/// How a model's tapes for one chip class store every MLX-affine codes tensor they read: one
/// entry per `(family, tensor, layer)`, ordered by it. Every bucket's tape for the class carries
/// the same table.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, serde::Serialize)]
pub struct AffineCodesTable {
    pub entries: &'static [CodesEntry],
}

impl AffineCodesTable {
    /// The table of a tape that reads no codes.
    pub const EMPTY: &'static AffineCodesTable = &AffineCodesTable { entries: &[] };

    /// How the `which` tensor of family `ix` at `layer` is stored; `None` for a tensor the table's
    /// tapes read no codes of.
    pub fn of(&self, ix: SourceIx, which: WeightTensor, layer: LayerId) -> Option<AffineCodes> {
        let key = (ix.0, which, layer.0);
        (self.entries.binary_search_by(|e| e.key().cmp(&key)))
            .ok()
            .map(|at| self.entries[at].codes)
    }
}

/// The DSL weight (`WeightId`) a codes tensor is: one tensor whichever families bind it — a
/// tied head is its table's ([`CodesRead::of`]).
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct CodesWeight(pub u32);

/// One read of a codes tensor by a step row: the family a command binds it through (`ix`), the
/// DSL weight it is (`of`), the tensor and every layer the row reads it at (each iteration of the
/// loops the row is rolled into), and the tensor's facts.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodesRead {
    pub ix: SourceIx,
    pub of: CodesWeight,
    pub which: WeightTensor,
    pub layers: Vec<LayerId>,
    pub tensor: CodesTensor,
}

impl CodesRole {
    /// The role of a tensor read in both roles: a projection's codes some reader reads row-whole
    /// (a tied head's, by its table's gather) are row-whole.
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (CodesRole::Projection, role) | (role, _) => role,
        }
    }
}

/// Why a model's codes table could not be baked: two step rows read one tensor at two widths,
/// input counts or groups.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CodesDisagree {
    pub of: CodesWeight,
    pub which: WeightTensor,
    pub layer: LayerId,
    pub first: CodesTensor,
    pub then: CodesTensor,
}

impl std::fmt::Display for CodesDisagree {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "codes tensor {:?} of DSL weight {} at layer {} is read as {:?} and as {:?}",
            self.which, self.of.0, self.layer.0, self.first, self.then
        )
    }
}

/// The codes table of a model's tapes for `class`: every tensor `reads` names, decided once
/// ([`decide`]) over every role it is read in, each family binding it naming that one decision,
/// ordered for [`AffineCodesTable::of`]. Its entries are the bake's for the life of the process:
/// the tapes lowered from it hold it by `&'static`.
pub fn bake_table(
    class: GenClass,
    drafter: Option<scratchy_ir::model_facts::Drafter>,
    reads: impl IntoIterator<Item = CodesRead>,
) -> Result<&'static AffineCodesTable, CodesDisagree> {
    type Key = (u32, WeightTensor, u32);
    // Each tensor, by the family whose tensor it is, with its facts; and each key a command binds
    // a tensor through, with the tensor's.
    let mut tensors: std::collections::BTreeMap<Key, CodesTensor> = Default::default();
    let mut bound: std::collections::BTreeMap<Key, Key> = Default::default();
    for read in reads {
        for layer in read.layers {
            let tensor = (read.of.0, read.which, layer.0);
            bound.insert((read.ix.0, read.which, layer.0), tensor);
            let Some(first) = tensors.get_mut(&tensor) else {
                tensors.insert(tensor, read.tensor);
                continue;
            };
            let facts = |t: &CodesTensor| (t.bits, t.k, t.group);
            if facts(first) != facts(&read.tensor) {
                return Err(CodesDisagree {
                    of: read.of,
                    which: read.which,
                    layer,
                    first: *first,
                    then: read.tensor,
                });
            }
            first.role = first.role.join(read.tensor.role);
        }
    }
    let decided: std::collections::BTreeMap<Key, AffineCodes> = (tensors.into_iter())
        .map(|(key, t)| (key, decide(class, drafter, t)))
        .collect();
    let entries: Vec<CodesEntry> = (bound.into_iter())
        .map(|((ix, which, layer), tensor)| CodesEntry {
            ix: SourceIx(ix),
            which,
            layer: LayerId(layer),
            codes: decided[&tensor],
        })
        .collect();
    Ok(Box::leak(Box::new(AffineCodesTable {
        entries: Box::leak(entries.into_boxed_slice()),
    })))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tensor(role: CodesRole, k: u32) -> CodesTensor {
        CodesTensor {
            role,
            bits: AffineBits::B4,
            k: KDim(k),
            group: AffineGroupSize::of(64),
        }
    }

    /// Only a projection's codes become planes on the wide class: a tied head's two names — the
    /// head family and the table family — are both row-whole, so the one tensor is stored one way
    /// whichever family a reader binds (gemma-4's tied lm_head on an M5 Max, #1365).
    #[test]
    fn only_a_projection_becomes_planes_and_a_tied_head_is_row_whole_by_both_names() {
        let wide = GenClass::M5Wide;
        let planes = AffineCodes::Planes {
            k: PlanesK::of(2816),
        };
        let offset8 = AffineCodes::Rows(RowCodes::Offset8);
        assert_eq!(
            decide(wide, None, tensor(CodesRole::Projection, 2816)),
            planes
        );
        for role in [CodesRole::Head, CodesRole::Table, CodesRole::ExpertBank] {
            assert_eq!(decide(wide, None, tensor(role, 2816)), offset8, "{role:?}");
        }
        assert_eq!(
            decide(GenClass::M5, None, tensor(CodesRole::Projection, 2816)),
            offset8
        );
        assert_eq!(
            decide(GenClass::Mid, None, tensor(CodesRole::Projection, 2816)),
            AffineCodes::Rows(RowCodes::AsWritten)
        );
    }

    /// Two rows reading one tensor at two input counts refuse the bake, naming it.
    #[test]
    fn one_tensor_read_two_ways_refuses_the_bake() {
        let read = |k| CodesRead {
            ix: SourceIx(3),
            of: CodesWeight(3),
            which: WeightTensor::Weight,
            layers: vec![LayerId(0), LayerId(1)],
            tensor: tensor(CodesRole::Projection, k),
        };
        let table = bake_table(GenClass::M5Wide, None, [read(2816), read(2816)]).expect("agree");
        assert_eq!(table.entries.len(), 2);
        assert!(
            table
                .of(SourceIx(3), WeightTensor::Weight, LayerId(1))
                .is_some()
        );
        assert!(
            table
                .of(SourceIx(3), WeightTensor::Weight, LayerId(2))
                .is_none()
        );
        let refused = bake_table(GenClass::M5Wide, None, [read(2816), read(4096)]);
        assert!(matches!(
            refused,
            Err(CodesDisagree {
                layer: LayerId(0),
                ..
            })
        ));
    }

    /// Two families binding one DSL weight — a tied head and its table — are one tensor: one
    /// decision over both roles, which both families' entries name. Row-whole on the wide class,
    /// where the head read as a projection alone would be planes.
    #[test]
    fn two_families_of_one_weight_are_one_decision() {
        let (head, table) = (SourceIx(1), SourceIx(2));
        let read = |ix, role| CodesRead {
            ix,
            of: CodesWeight(7),
            which: WeightTensor::Weight,
            layers: vec![LayerId(0)],
            tensor: tensor(role, 2816),
        };
        let reads = [
            read(head, CodesRole::Projection),
            read(table, CodesRole::Table),
        ];
        let t = bake_table(GenClass::M5Wide, None, reads).expect("one tensor");
        let rows = Some(AffineCodes::Rows(RowCodes::Offset8));
        assert_eq!(t.of(head, WeightTensor::Weight, LayerId(0)), rows);
        assert_eq!(t.of(table, WeightTensor::Weight, LayerId(0)), rows);
    }
}
