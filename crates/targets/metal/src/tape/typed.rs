// SPDX-License-Identifier: Apache-2.0
//! A BAKED TAPE'S WEIGHT READS, TYPED.
//!
//! Each model's bake emits one [`Source`] marker per weight family its tapes bind: the family's
//! [`SourceIx`] and the typed handle (`scratchy_tensors::typed::W2<K, N, _>`, …) its `Weights`
//! field holds, reached through [`Source::at`] — the resolver's one path to it. A dense GEMM's
//! record (`Gemm`: its dims and the weight it reads, which nothing else of the command names)
//! takes the weight's index and dims from its slot's one [`GemmSlot::WEIGHT`], read off the
//! handle; each binding of a typed family's tensor states its index as its slot's. So the loader
//! (the field's type), the tape (the GEMM's weight and dims) and the kernel (picked and baked from
//! those dims) read one type, and rustc checks they agree.
//!
//! What rustc refuses, each refusal one edit from the version it accepts. (ii) A GEMM's dims are
//! its slot's handle's — `lm_head`'s are `[2048, 128256]` — and a slot whose handle has no GEMM
//! dims (the norm's rank-1 gain) has none to give:
//!
//! ```
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{GemmSlot, Source};
//! # use scratchy_tensors::typed::{Dim, W1, W2};
//! # enum Hidden {}
//! # impl Dim for Hidden { const N: u32 = 2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { const N: u32 = 128256; }
//! # struct Layer;
//! # struct Weights { lm_head: W2<Hidden, Vocab, Layer>, norm: W1<Hidden, Layer> }
//! # enum Norm {}
//! # impl Source for Norm {
//! #     type Model = Weights;
//! #     type Handle = W1<Hidden, Layer>;
//! #     const IX: SourceIx = SourceIx(1);
//! #     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.norm }
//! # }
//! # enum LmHead {}
//! # impl Source for LmHead {
//! #     type Model = Weights;
//! #     type Handle = W2<Hidden, Vocab, Layer>;
//! #     const IX: SourceIx = SourceIx(0);
//! #     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.lm_head }
//! # }
//! let w = <LmHead as GemmSlot>::WEIGHT;
//! assert_eq!((w.ix, w.n, w.k), (SourceIx(0), 128256, 2048));
//! ```
//!
//! ```compile_fail,E0277
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{GemmSlot, Source};
//! # use scratchy_tensors::typed::{Dim, W1, W2};
//! # enum Hidden {}
//! # impl Dim for Hidden { const N: u32 = 2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { const N: u32 = 128256; }
//! # struct Layer;
//! # struct Weights { lm_head: W2<Hidden, Vocab, Layer>, norm: W1<Hidden, Layer> }
//! # enum Norm {}
//! # impl Source for Norm {
//! #     type Model = Weights;
//! #     type Handle = W1<Hidden, Layer>;
//! #     const IX: SourceIx = SourceIx(1);
//! #     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.norm }
//! # }
//! let w = <Norm as GemmSlot>::WEIGHT;
//! ```
//!
//! (iii) A slot whose handle is not the type of the field it reaches — `lm_head` declared
//! `[vocab, hidden]` over a `[hidden, vocab]` field:
//!
//! ```
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{GemmSlot, Source};
//! # use scratchy_tensors::typed::{Dim, W1, W2};
//! # enum Hidden {}
//! # impl Dim for Hidden { const N: u32 = 2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { const N: u32 = 128256; }
//! # struct Layer;
//! # struct Weights { lm_head: W2<Hidden, Vocab, Layer>, norm: W1<Hidden, Layer> }
//! # enum Norm {}
//! # impl Source for Norm {
//! #     type Model = Weights;
//! #     type Handle = W1<Hidden, Layer>;
//! #     const IX: SourceIx = SourceIx(1);
//! #     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.norm }
//! # }
//! # enum LmHead {}
//! impl Source for LmHead {
//!     type Model = Weights;
//!     type Handle = W2<Hidden, Vocab, Layer>;
//!     const IX: SourceIx = SourceIx(0);
//!     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.lm_head }
//! }
//! ```
//!
//! ```compile_fail,E0308
//! # use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
//! # use scratchy_target_metal::tape::typed::{GemmSlot, Source};
//! # use scratchy_tensors::typed::{Dim, W1, W2};
//! # enum Hidden {}
//! # impl Dim for Hidden { const N: u32 = 2048; }
//! # enum Vocab {}
//! # impl Dim for Vocab { const N: u32 = 128256; }
//! # struct Layer;
//! # struct Weights { lm_head: W2<Hidden, Vocab, Layer>, norm: W1<Hidden, Layer> }
//! # enum Norm {}
//! # impl Source for Norm {
//! #     type Model = Weights;
//! #     type Handle = W1<Hidden, Layer>;
//! #     const IX: SourceIx = SourceIx(1);
//! #     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.norm }
//! # }
//! # enum LmHead {}
//! impl Source for LmHead {
//!     type Model = Weights;
//!     type Handle = W2<Vocab, Hidden, Layer>;
//!     const IX: SourceIx = SourceIx(0);
//!     fn at(w: &Weights, _: LayerId) -> &Self::Handle { &w.lm_head }
//! }
//! ```

use scratchy_tensors::typed::{Dim, Shape2};

use super::ids::{LayerId, SourceIx};

/// A weight family of model `Model`: its index in the model's source manifest, and the typed
/// handle its field holds.
pub trait Source {
    type Model;
    type Handle;
    const IX: SourceIx;
    fn at(model: &Self::Model, layer: LayerId) -> &Self::Handle;
}

/// A GEMM's weight, as its slot states it: the slot's index and the handle's dims, `n` (output
/// columns) and `k` (the inner dim).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemmSlotWeight {
    pub ix: SourceIx,
    pub n: u32,
    pub k: u32,
}

/// A slot a GEMM reads: only a slot whose handle is a rank-2 weight has a [`GemmSlotWeight`].
pub trait GemmSlot {
    const WEIGHT: GemmSlotWeight;
}

impl<S: Source<Handle: Shape2>> GemmSlot for S {
    const WEIGHT: GemmSlotWeight = GemmSlotWeight {
        ix: S::IX,
        n: <<S::Handle as Shape2>::N as Dim>::N,
        k: <<S::Handle as Shape2>::K as Dim>::N,
    };
}
