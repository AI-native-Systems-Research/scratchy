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

/// ONE ROW OF THE PT — ⛔⛔ ADDED BECAUSE THE ISLAND COULD NOT STATE THE REFERENCE'S OWN PROGRAM.
///
/// `dcc/test/Conversion/AgenToSentient/lx-to-sfp-bypass-1.mlir:126-127` is
/// `%pt = dataflow.get_unit {..., type = "ptrow0"}` followed by `dataflow.send %pt, %9`, and the
/// expected SentientIR for it is `sentient.set_send_dst(%[[VAL_10]])` on that same value (`:47`).
/// With no marker for a PT row, a send to one was not constructible — so the bypass rule
/// `generateSetSendDestinationStmts` exists to implement had no input it could be tested on.
///
/// ⭐ THE ROW IS THE CONST PARAMETER, so `PtRowUnit<0>` and `PtRowUnit<1>` are different types and a
/// wire to row 0 cannot be spent as a wire to row 1.
///
/// ⛔ AND A ROW THIS ARCH DOES NOT HAVE IS A BUILD FAILURE, not a runtime one: [`UnitKind::KIND`] is
/// a const, so `PtRowUnit<9>::KIND` fails const evaluation on an 8-row arch at the point of use.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PtRowUnit<const ROW: u32>;
/// The L0 store unit — one of the four destinations that make a load bypass the SFP
/// (`Helper.cpp:2764-2765`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L0su;
/// The L0 load unit.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct L0lu;
/// The link out of this partition — the fourth bypass destination.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CrossPtnLink;

/// The PE — drains a wire, computes, drives another.
///
/// ⛔ ADDED FOR ENTRY 304'S FIXTURE: `mixed_precision.mlir:745` is a **PE** program unit receiving
/// from a `ptrow7`, and with no marker for the PE that wire was not constructible — the receive
/// whose precision the reference overrides had no input the test could state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Pe;

impl sealed::Sealed for L3lu {}
impl sealed::Sealed for Lxlu {}
impl sealed::Sealed for Sfp {}
impl sealed::Sealed for Lxsu {}
impl sealed::Sealed for Pe {}
impl UnitKind for Pe {
    const KIND: DfirUnit = DfirUnit::Pe;
}
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
impl<const ROW: u32> sealed::Sealed for PtRowUnit<ROW> {}
impl sealed::Sealed for L0su {}
impl sealed::Sealed for L0lu {}
impl sealed::Sealed for CrossPtnLink {}
impl<const ROW: u32> UnitKind for PtRowUnit<ROW> {
    // ⛔ THE `None` ARM IS A CONST PANIC, WHICH IS A COMPILE ERROR. `Row` carries this arch's row
    // count, so naming a row it does not have fails to build rather than falling back to a wrong
    // unit — the trap `PtRow::unit()` was deleted for.
    const KIND: DfirUnit = match crate::units::Row::checked(ROW) {
        Some(row) => DfirUnit::PtRow(row),
        None => panic!("this arch's PT has no such row"),
    };
}
impl UnitKind for L0su {
    const KIND: DfirUnit = DfirUnit::L0su;
}
impl UnitKind for L0lu {
    const KIND: DfirUnit = DfirUnit::L0lu;
}
impl UnitKind for CrossPtnLink {
    const KIND: DfirUnit = DfirUnit::CrossPtnLink;
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

    /// THE DEGENERATE WIRE A UNIT DRIVES TO ITSELF — *"Consumer for LoadAndExtractScalarOp is
    /// self"* (`Helper.cpp:2437-2442`).
    ///
    /// ⛔⛔ NOT EXPRESSIBLE AS A [`Link`], AND DELIBERATELY SO. A `Link<A, B>` joins two units and
    /// hands out one send and one receive; `sentient.load_and_extract_scalar` names the unit it is
    /// ITSELF running on as its `$consumer` and there is no receive anywhere to pair with. The
    /// vendor prints exactly that — `consumer(%arg0)` inside the program unit bound by `%arg0`
    /// (`dcc/test/Conversion/AgenToSentient/lx_indirect_loads_stores_composite.mlir:34`).
    ///
    /// ⛔ SO IT IS A NAMED CONSTRUCTOR AND NOT `SendEnd(val)`: the only two ways to obtain an end
    /// stay [`Link::ends`] and this, and reaching for this one says in the source that the wire is a
    /// self-loop rather than a pairing someone forgot to make.
    #[must_use]
    pub const fn to_self(unit: Val) -> SendEnd {
        SendEnd(unit)
    }

    /// THE END'S VALUE, MUTABLY — for RENUMBERING ONLY.
    ///
    /// ⛔ THIS DOES NOT REPAIR THE WIRE, AND IT IS NOT A PUBLIC SETTER. A clone rewrites every value
    /// of an op it copied (`dialects::parts_mut`), and a rewriter redirects a single use in place
    /// (`dialects::vals_mut`, `VectorChainHelper.cpp:600-602`); the send's destination is one of the
    /// values either one names. Both are SUBSTITUTION, NOT MINTING — the [`Link`] the end came from
    /// still says which two units it joins, because renaming a value does not move it, and
    /// [`Link::ends`] remains the only way to obtain an end in the first place. Anything that wants a
    /// DIFFERENT destination takes a different [`Link`].
    pub(crate) const fn val_mut(&mut self) -> &mut Val {
        &mut self.0
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

    /// THE END'S VALUE, MUTABLY — for RENUMBERING ONLY. See [`SendEnd::val_mut`].
    pub(crate) const fn val_mut(&mut self) -> &mut Val {
        &mut self.0
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

/// ONE SIDE OF A RENDEZVOUS — the peer this unit signals, then waits on.
///
/// ⛔ THE PAIR IS EMITTED TOGETHER. `dataflow.sync_recv` is BLOCKING — *"it does not return until
/// the matching signal has been received"* (`Dataflow.td:209-212`) — so a `sync_send` whose peer
/// never signals back is a unit that waits forever.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Half<Peer: UnitKind> {
    peer: Val,
    of: core::marker::PhantomData<Peer>,
}

impl<Peer: UnitKind> Half<Peer> {
    /// The peer to signal and then wait on.
    #[must_use]
    pub const fn peer(self) -> Val {
        self.peer
    }
}

/// A RENDEZVOUS BETWEEN TWO UNITS.
///
/// # 🛑 BOTH SIDES, OR NEITHER
///
/// ⭐⭐ IBM EMITS IT SYMMETRICALLY. Their `l3lu` unit does `sync_send %30`, `sync_send %32`,
/// `sync_recv %30`, `sync_recv %32` (`/tmp/ktir_ref/export/debug/dfir.mlir:95-98`) and the `lxlu`
/// unit it is synchronising with does the mirror, `sync_send %32` then `sync_recv %32`
/// (`:119-120`). The `lxsu` and `l3su` units do the same at `:173-174` and `:193-196`. Signal, then
/// wait — on BOTH sides.
///
/// ⛔ THE EMITTER WROTE ONE SIDE. The cache walk put `sync_send` and `sync_recv` in the COMPUTE's
/// body naming a mover it minted inline, and the mover's own body had no mirror at all. Two
/// independent statements again, and this time the missing one is a deadlock rather than a
/// diagnostic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Rendezvous<A: UnitKind, B: UnitKind> {
    a: Val,
    b: Val,
    between: core::marker::PhantomData<(A, B)>,
}

impl<A: UnitKind, B: UnitKind> Rendezvous<A, B> {
    /// A rendezvous between two BOUND units.
    #[must_use]
    pub fn between(a: Val, b: Val) -> Rendezvous<A, B> {
        Rendezvous {
            a,
            b,
            between: core::marker::PhantomData,
        }
    }

    /// THE TWO SIDES, ONCE.
    ///
    /// ⛔ CONSUMES THE RENDEZVOUS, so `A`'s side and `B`'s side come from one value. `A` is handed
    /// the peer it signals — which is `B` — and vice versa; getting that backwards is not
    /// expressible because the halves are typed by whose side they are.
    #[must_use]
    pub fn halves(self) -> (Half<B>, Half<A>) {
        (
            Half {
                peer: self.b,
                of: core::marker::PhantomData,
            },
            Half {
                peer: self.a,
                of: core::marker::PhantomData,
            },
        )
    }
}

/// WHERE ONE TENSOR SITS IN THE SCRATCHPAD — written by one unit, read by another.
///
/// # 🛑 NOTHING DOWNSTREAM CHECKS THIS ONE
///
/// ⛔⛔ dbo-opt COMPILES ONE PROGRAM UNIT AT A TIME AND HAS NO CROSS-UNIT ALIAS ANALYSIS. The L3
/// half writes an LX range and the LX loader views the same range; that they name the same bytes is
/// an obligation on US, not something a pass will refuse. So this pairing has no refusal to drive
/// it and would surface as wrong numerics on hardware.
///
/// ⛔ AND IT IS ALREADY WRONG. The staging loop wrote `dst_start = 0` for EVERY operand, so in
/// `g6_1_matmul` the activation view (`memref<1x2048xf16>` at element 0) and the weight view
/// (`memref<2048x2048xf16>` at element 0) name the same address — the weight lands on top of the
/// activation. That is the "invented addresses" failure this crate's own CLAUDE.md records.
///
/// ⭐ THE ADDRESS IS IN ELEMENTS, which is what `dataflow.get_logical_memory_view`'s `start`
/// carries (`Dataflow.td:250`) — not bytes, and not sticks.
///
/// ⛔ MINTED BY THE WRITER, SPENT BY EACH READER. A reader cannot build a view out of a loose
/// address and shape it computed for itself.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Placed {
    start: i64,
    rows: u64,
    cols: u64,
}

impl Placed {
    /// ⛔ THE ONLY CONSTRUCTOR, AND IT IS THE WRITER'S. Whoever puts the bytes there says where
    /// they went; every reader takes this value rather than recomputing one.
    #[must_use]
    pub const fn written_at(start: i64, rows: u64, cols: u64) -> Placed {
        Placed { start, rows, cols }
    }

    /// Its start, in ELEMENTS.
    #[must_use]
    pub const fn start(self) -> i64 {
        self.start
    }

    /// Its rows.
    #[must_use]
    pub const fn rows(self) -> u64 {
        self.rows
    }

    /// Its columns.
    #[must_use]
    pub const fn cols(self) -> u64 {
        self.cols
    }
}
