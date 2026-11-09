// SPDX-License-Identifier: Apache-2.0
//! ⭐ A BUNDLE'S BAKED FACTS, WALKED ONCE — every field of the [`BundleLayout`] and of each launch's
//! [`KvShifts`], in the order the artifact ships them, handed to a [`FactSink`].
//!
//! Two sinks read this walk: the macro's token emission (what ships in the binary) and the bundle's
//! content name (what the bake keys it by). Both read THE WALK, never the struct, so the name and the
//! shipped bytes cannot disagree about which facts exist or what they hold: one statement of the
//! bundle's content, rendered twice.
//!
//! ⛔ EXHAUSTIVE BY CONSTRUCTION. Every record is destructured with no `..`, so a field added to
//! [`BundleLayout`], [`KvShifts`] or anything they hold does not compile until it is walked, and a
//! field the walk names is a field both sinks see.
//!
//! The vocabulary's own values ([`PlaceId`], [`Fragment`], [`Rank`], [`WindowCut`]) are LEAVES, not
//! records: each has its one naming site in the crate that mints it (`PlaceId`'s `Display`, the
//! `const fn baked` re-mints), so a sink renders the value through that site rather than through a
//! second walk of its variants here.

use crate::collective::{CollectiveKind, Fragment, Rank, Tp, TpConsume, WindowCut};
use crate::{BundleLayout, FoldRows, KernelWeight, KvShifts, PlaceId, Placement, Walk, WeightTail};

/// A type's path below the bundle crate's root (`["collective", "TpConsume"]`).
pub type Path = &'static [&'static str];

/// How a list ships.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ListShape {
    /// A fixed-size array field: `[a, b, …]`.
    Array,
    /// A borrowed slice: `Cow::Borrowed(&[a, b, …])`.
    Slice,
    /// A placement table. Bundles of one ladder share one resident layout, so equal tables ship as
    /// one `static`, borrowed in place.
    PlaceTable,
}

/// What a walk renders each value as. The walk is a FOLD: every value is rendered from its parts'
/// renderings, so a sink holds no state of the walk and cannot see a half-built value.
pub trait FactSink {
    type Out;
    /// A struct value: its fields, named, in shipped order.
    fn record(&mut self, path: Path, fields: Vec<(&'static str, Self::Out)>) -> Self::Out;
    /// An enum value; a unit variant has no fields.
    fn variant(
        &mut self,
        path: Path,
        variant: &'static str,
        fields: Vec<(&'static str, Self::Out)>,
    ) -> Self::Out;
    fn option(&mut self, v: Option<Self::Out>) -> Self::Out;
    fn list(&mut self, shape: ListShape, vs: Vec<Self::Out>) -> Self::Out;
    fn pair(&mut self, a: Self::Out, b: Self::Out) -> Self::Out;
    fn u32(&mut self, v: u32) -> Self::Out;
    fn u64(&mut self, v: u64) -> Self::Out;
    fn bool(&mut self, v: bool) -> Self::Out;
    /// An `f32`, exactly: a rounded scale is a quietly wrong model, and the page walk's `-inf`
    /// seed has no literal.
    fn f32(&mut self, v: f32) -> Self::Out;
    /// A weight fold's scale, which ships as a suffixed literal (`0.0078125f32`): the shortest
    /// round-trip spelling, exact for every finite value, and a build error at expansion for any
    /// other.
    fn fold_scale(&mut self, v: f32) -> Self::Out;
    fn place(&mut self, id: PlaceId) -> Self::Out;
    fn fragment(&mut self, f: Fragment) -> Self::Out;
    fn rank<T: Tp>(&mut self, r: Rank<T>) -> Self::Out;
    fn cut(&mut self, c: WindowCut) -> Self::Out;
}

/// A value the walk can hand to a sink.
pub trait Facts {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out;
}

impl Facts for u32 {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        s.u32(*self)
    }
}

impl Facts for u64 {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        s.u64(*self)
    }
}

impl Facts for bool {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        s.bool(*self)
    }
}

impl Facts for f32 {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        s.f32(*self)
    }
}

impl Facts for PlaceId {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        s.place(*self)
    }
}

impl<X: Facts> Facts for Option<X> {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        let v = self.as_ref().map(|x| x.visit(s));
        s.option(v)
    }
}

/// One list, element by element.
fn list<S: FactSink, X: Facts>(s: &mut S, shape: ListShape, xs: &[X]) -> S::Out {
    let vs = xs.iter().map(|x| x.visit(s)).collect();
    s.list(shape, vs)
}

impl Facts for Placement {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        let Placement {
            id,
            segment,
            bank,
            offset,
            size,
            is_logits,
        } = self;
        let fields = vec![
            ("id", id.visit(s)),
            ("segment", segment.visit(s)),
            ("bank", bank.visit(s)),
            ("offset", offset.visit(s)),
            ("size", size.visit(s)),
            ("is_logits", is_logits.visit(s)),
        ];
        s.record(&["Placement"], fields)
    }
}

impl Facts for KernelWeight<'_> {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        let KernelWeight {
            id,
            device_size,
            stride_map,
            stick_size,
            word_length,
        } = self;
        let fields = vec![
            ("id", id.visit(s)),
            ("device_size", list(s, ListShape::Slice, device_size)),
            ("stride_map", list(s, ListShape::Slice, stride_map)),
            ("stick_size", stick_size.visit(s)),
            ("word_length", word_length.visit(s)),
        ];
        s.record(&["KernelWeight"], fields)
    }
}

impl Facts for WeightTail {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        match self {
            WeightTail::InPlace => s.variant(&["WeightTail"], "InPlace", Vec::new()),
            WeightTail::Spilled { segment, bytes } => {
                let fields = vec![("segment", segment.visit(s)), ("bytes", bytes.visit(s))];
                s.variant(&["WeightTail"], "Spilled", fields)
            }
        }
    }
}

impl Facts for BundleLayout<'_> {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        let BundleLayout {
            segment_bytes,
            weight_bank_bytes,
            weight_tail,
            places,
            kernel_weights,
            scalarmul_scales,
            weight_scale_folds,
            kv_request_stride_bytes,
            lx_op_local_bytes,
            lx_prob_pin_bytes,
            lx_pin_window_bytes,
            num_sources,
            resident_source_tids,
        } = self;
        let fields = vec![
            ("segment_bytes", list(s, ListShape::Array, segment_bytes)),
            ("weight_bank_bytes", list(s, ListShape::Slice, weight_bank_bytes)),
            ("weight_tail", weight_tail.visit(s)),
            ("places", list(s, ListShape::PlaceTable, places)),
            ("kernel_weights", list(s, ListShape::Slice, kernel_weights)),
            ("scalarmul_scales", list(s, ListShape::Slice, scalarmul_scales)),
            ("weight_scale_folds", {
                let vs = weight_scale_folds
                    .iter()
                    .map(|&(tid, scale)| {
                        let (tid, scale) = (s.u32(tid), s.fold_scale(scale));
                        s.pair(tid, scale)
                    })
                    .collect();
                s.list(ListShape::Slice, vs)
            }),
            ("kv_request_stride_bytes", kv_request_stride_bytes.visit(s)),
            ("lx_op_local_bytes", lx_op_local_bytes.visit(s)),
            ("lx_prob_pin_bytes", lx_prob_pin_bytes.visit(s)),
            ("lx_pin_window_bytes", lx_pin_window_bytes.visit(s)),
            ("num_sources", num_sources.visit(s)),
            ("resident_source_tids", list(s, ListShape::Slice, resident_source_tids)),
        ];
        s.record(&["BundleLayout"], fields)
    }
}

impl Facts for FoldRows {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        let variant = match self {
            FoldRows::WholeBatch => "WholeBatch",
            FoldRows::PerRequest => "PerRequest",
        };
        s.variant(&["FoldRows"], variant, Vec::new())
    }
}

impl Facts for Walk {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        let Walk { trips, step_bytes } = self;
        let fields = vec![("trips", trips.visit(s)), ("step_bytes", step_bytes.visit(s))];
        s.record(&["Walk"], fields)
    }
}

impl Facts for CollectiveKind {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        let variant = match self {
            CollectiveKind::AllReduce => "AllReduce",
            CollectiveKind::AllGather => "AllGather",
        };
        s.variant(&["collective", "CollectiveKind"], variant, Vec::new())
    }
}

impl<T: Tp> Facts for TpConsume<T> {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        let TpConsume {
            kind,
            operand,
            fragment,
            plays_at,
            cut,
        } = *self;
        let fields = vec![
            ("kind", kind.visit(s)),
            ("operand", operand.visit(s)),
            ("fragment", s.fragment(fragment)),
            ("plays_at", s.rank(plays_at)),
            ("cut", s.cut(cut)),
        ];
        s.record(&["collective", "TpConsume"], fields)
    }
}

impl<T: Tp> Facts for KvShifts<T> {
    fn visit<S: FactSink>(&self, s: &mut S) -> S::Out {
        let KvShifts {
            slot_stride_bytes,
            slab_stride_bytes,
            page_slots,
            request,
            slot_offset,
            page_fold,
            batched_requests,
            gathered,
            fold_rows,
            tp,
            walk,
        } = self;
        let fields = vec![
            ("slot_stride_bytes", slot_stride_bytes.visit(s)),
            ("slab_stride_bytes", slab_stride_bytes.visit(s)),
            ("page_slots", page_slots.visit(s)),
            ("request", request.visit(s)),
            ("slot_offset", slot_offset.visit(s)),
            ("page_fold", page_fold.visit(s)),
            ("batched_requests", batched_requests.visit(s)),
            ("gathered", gathered.visit(s)),
            ("fold_rows", fold_rows.visit(s)),
            ("walk", walk.visit(s)),
            ("tp", tp.visit(s)),
        ];
        s.record(&["KvShifts"], fields)
    }
}
