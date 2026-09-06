//! EVERY PRODUCER-CONSUMER PAIRING, AS ONE VALUE WITH TWO ENDS.
//!
//! # 🛑 A PAIRING IS NOT TWO FACTS THAT AGREE
//!
//! ⛔⛔ THE EMITTER BUILT BOTH ENDS SEPARATELY AND TRUSTED THEM TO MATCH:
//!
//! ```text
//! let to   = computers.as_ref().map_or(lx, Units::first);   // the mover's destination
//! let from = loaders.as_ref().map_or(lx, Units::first);     // the compute's source
//! ```
//!
//! Two independent lookups. Nothing said the mover's `to` was the compute's unit, or that the
//! compute's `from` was the mover's — and if a schedule named neither, BOTH fall back to `lx` and
//! the program describes a wire from a memory to itself. Every type in this crate stayed silent on
//! that.
//!
//! ⭐⭐ SO THE PAIRING IS THE VALUE. A [`Link`] is minted once and yields its two ends ONCE, by
//! consuming itself. The producer spends the [`SendEnd`], the consumer spends the [`RecvEnd`], and
//! `Link<Lxlu, Sfp>` is a different type from `Link<Sfp, Lxlu>` — so the ends cannot be swapped and
//! a send cannot be paired with a receive from a different wire.
//!
//! # 🛑 AND THE FORM, THE UNIT AND THE WIRE ALL COME FROM THE RESIDENCIES
//!
//! ⛔⛔ `getDataTransferType(src_is_fifo, dst_is_fifo)` is TOTAL over three cases
//! (`DataTransferLowering.cpp:165-172`), and each case fixes the ops emitted, the unit kind that can
//! lower them, and whether a wire exists at all:
//!
//! | src | dst | ops | unit | wire |
//! |---|---|---|---|---|
//! | memref | memref | `agen.composite_load_and_store` (`:255, :311`) | an L3 half (`Helper.cpp:2177-2179`) | none |
//! | memref | FIFO | `agen.vector_load` + `dataflow.send` (`:274, :424`) | the LX load unit | one |
//! | FIFO | memref | `dataflow.receive` + `agen.vector_store` (`:293, :495`) | the LX store unit | one |
//! | FIFO | FIFO | *"FIFO to FIFO transfers are not allowed"* | — | — |
//!
//! ⛔ CHOOSING THEM SEPARATELY IS WHAT PUT A COMPOSITE TRANSFER ON AN `lxlu`, which
//! `Helper.cpp:2177-2179` refuses with a bare `LogicalResult::failure()` and no message at all — so
//! dbo-opt printed only the caller's wrapper, *"Unable to generate loops and sentient statements for
//! the composite vector operations"* (`:2965-2967`), and the emitter read that as being about the
//! transfer's shape.
//!
//! ⛔ AN EARLIER ATTEMPT AT THIS LOCKED THE LABEL AND NOT THE PAIRING. `Units::moves_memory()`
//! returned "is this kind an L3 half" and had NO CALLER; the emitter went on hand-placing the
//! transfer beside it. A bool that describes half of a pairing enforces nothing.

use crate::islands::dataflow_ir::dialects::Val;
use crate::units::DfirUnit;

/// A UNIT KIND, AS A TYPE.
///
/// ⛔ SEALED, AND ONE PER KIND THAT TERMINATES A WIRE. These exist so that the two ends of a
/// [`Link`] are distinguishable in the type system; a kind that never terminates a wire has no
/// marker here and cannot be named as an end.
pub trait UnitKind: sealed::Sealed {
    /// Which kind this is, for the `dataflow.get_unit` that binds it.
    const KIND: DfirUnit;
}

mod sealed {
    pub trait Sealed {}
}

/// The L3 load half — the only kind that may run a memory-to-memory transfer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L3lu;
/// The LX load unit — reads the scratchpad and drives a wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lxlu;
/// The SFP — drains a wire, computes, drives another.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Sfp;
/// The LX store unit — drains a wire and writes the scratchpad.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Lxsu;

impl sealed::Sealed for L3lu {}
impl sealed::Sealed for Lxlu {}
impl sealed::Sealed for Sfp {}
impl sealed::Sealed for Lxsu {}
impl UnitKind for L3lu {
    const KIND: DfirUnit = DfirUnit::L3lu;
}
impl UnitKind for Lxlu {
    const KIND: DfirUnit = DfirUnit::Lxlu;
}
impl UnitKind for Sfp {
    const KIND: DfirUnit = DfirUnit::Sfp;
}
impl UnitKind for Lxsu {
    const KIND: DfirUnit = DfirUnit::Lxsu;
}

/// THE PRODUCER'S END OF ONE WIRE — what `dataflow.send`'s `to` must be.
///
/// ⛔ THE `Val` IS PRIVATE AND ONLY [`Link::ends`] MINTS ONE. A bare unit handle can no longer be
/// written into a send.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SendEnd(Val);

impl SendEnd {
    /// The unit the data goes to — the FIRST HOP, which is what the op carries.
    #[must_use]
    pub const fn val(self) -> Val {
        self.0
    }
}

/// THE CONSUMER'S END OF ONE WIRE — what `dataflow.receive`'s `from` must be.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RecvEnd(Val);

impl RecvEnd {
    /// The unit the data comes from.
    #[must_use]
    pub const fn val(self) -> Val {
        self.0
    }
}

/// ONE WIRE BETWEEN TWO UNITS.
///
/// ⭐⭐ ITS TWO ENDS ARE HANDED OUT ONCE, BY CONSUMING IT. So one `Link` is one send and one
/// receive: a wire cannot be driven twice, and a send cannot be paired with a receive belonging to
/// a different wire.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Link<From: UnitKind, To: UnitKind> {
    from: Val,
    to: Val,
    ends: core::marker::PhantomData<(From, To)>,
}

impl<From: UnitKind, To: UnitKind> Link<From, To> {
    /// A wire between two BOUND units.
    ///
    /// ⛔ THE KINDS ARE THE TYPE PARAMETERS, so the caller states which end is which and
    /// `Link<Lxlu, Sfp>` cannot be passed where `Link<Sfp, Lxsu>` is wanted.
    #[must_use]
    pub fn between(from: Val, to: Val) -> Link<From, To> {
        Link {
            from,
            to,
            ends: core::marker::PhantomData,
        }
    }

    /// THE TWO ENDS, ONCE.
    ///
    /// ⛔ CONSUMES THE LINK. The producer's `to` and the consumer's `from` come out of this single
    /// call, so they are the same wire by construction rather than by two lookups agreeing.
    #[must_use]
    pub fn ends(self) -> (SendEnd, RecvEnd) {
        (SendEnd(self.to), RecvEnd(self.from))
    }

    /// Which kinds this wire runs between — for the `get_unit`s that must bind them.
    #[must_use]
    pub const fn kinds() -> (DfirUnit, DfirUnit) {
        (From::KIND, To::KIND)
    }
}
