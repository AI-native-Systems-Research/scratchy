// SPDX-License-Identifier: Apache-2.0
//! WHAT A LOAD'S VALUE REACHES, AND WHICH UNIT CONSUMES IT.
//!
//! Ported from `AgenToSentientLoweringPass::getLoadConsumer`
//! (`dcc/src/Conversion/AgenToSentient/Helper.cpp:1242-1278`).
//!
//! # 🛑 MOST OF THAT FUNCTION IS DISCOVERY, AND WE DO NOT NEED IT
//!
//! ⛔⛔ THE C++ IS REVERSE-ENGINEERING A CHAIN IT DID NOT BUILD. It takes the load's result (or, for a
//! composite, its *load induction variable*), asserts the value `hasOneUse()`, walks
//! `getUsers().begin()`, and pattern-matches what it finds. Every line of that exists because the pass
//! receives an MLIR module from elsewhere and has to work out how the ops relate.
//!
//! ⭐ WE BUILD THE RELATION, SO WE READ IT. `dataflow::Op::Send` carries `to: SendEnd` — one end of a
//! [`crate::islands::dataflow_ir::link::Link`], minted once with its matching `RecvEnd` and unable to
//! be a unit handle a caller chose. So *"the consumer is the send's `to_unit`"*, which costs the C++ a
//! use-list walk and two `emitError` arms, is a field access here.
//!
//! ⛔ WHAT DOES **NOT** SURVIVE, and must not be re-derived: `hasOneUse`, `getUsers`, `getResult(0)`
//! versus `getLoadInductionVar()`, and the two `return std::make_pair(nullptr, nullptr)` arms. The last
//! two are refusals, and this bridge has none.
//!
//! # 🛑 WHAT DOES SURVIVE IS THE SHAPE CONSTRAINT, AND IT IS WORTH A TYPE
//!
//! ⭐⭐ THE REFERENCE STATES IT AS A COMMENT AND THEN ENFORCES IT WITH AN ERROR:
//!
//! ```text
//! // Assume loadOp is followed by only:
//! //   - a sendOp, or
//! //   - a selectOp/shuffleOp followed by a sendOp, or
//! //   - a storeOp
//! ```
//!
//! Three shapes, and anything else is `"unsupported loadOp consumer!"`. [`LoadChain`] is those three
//! and no fourth, so the arm that emits that error has no counterpart here — not because we skipped it,
//! but because its input is unconstructible.
//!
//! ⛔ AND "EXACTLY ONE CONSUMER" IS THE OTHER HALF OF THE RULE — the reference's
//! `if (!consumer_root.hasOneUse())` with *"loadOp should have one consumer"*. A `LoadChain` is
//! move-only and `#[must_use]`, following [`crate::islands::dataflow_ir::dialects::dataflow::Received`]:
//! a chain that is never spent is a load whose value nothing consumes, which is the
//! "Dangling non-compute op has no use" family of refusal, and a chain cannot be spent twice.

use crate::islands::dataflow_ir::dialects::Val;
use crate::islands::dataflow_ir::link::SendEnd;

/// A LANE REARRANGEMENT THAT MAY SIT BETWEEN A LOAD AND ITS SEND.
///
/// ⛔ THREE OPS, AND THE SET IS THE REFERENCE'S: `isa<vectorchain::SelectOp, vectorchain::ShuffleOp,
/// vectorchain::RotateOp>(user)` (`Helper.cpp:1268-1269`). A fourth vectorchain op between a load and a
/// send is not a shape the lowering handles, so it must not be nameable here.
///
/// ⛔ AND ONLY **ONE** MAY INTERVENE. The reference tests `user->hasOneUse()` and then looks for the
/// send directly — it does not recurse, so two chained rearrangements are not the middle case, they are
/// the unsupported one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rearrangement {
    /// `vectorchain.select`.
    Select,
    /// `vectorchain.shuffle`.
    Shuffle,
    /// `vectorchain.rotate`.
    Rotate,
}

/// WHERE ONE LOAD'S VALUE GOES — the three shapes and no fourth.
///
/// ⛔⛔ MOVE-ONLY AND `#[must_use]`, for the reason
/// [`crate::islands::dataflow_ir::dialects::dataflow::Received`] is: a load whose value nothing
/// consumes is what dbo-opt refuses as *"Dangling non-compute op has no use"*, and a value consumed
/// twice is a load the reference would reject for not having `hasOneUse()`. Spending the chain is what
/// makes both unrepresentable rather than checked.
#[must_use = "a load chain that is never spent is a load whose value nothing consumes — dbo-opt \
              refuses that as a dangling non-compute op"]
#[derive(Debug, PartialEq, Eq)]
pub enum LoadChain {
    /// `load -> send`. The consumer is the send's own end of the wire.
    ToSend {
        /// The loaded value the send carries.
        data: Val,
        /// Which unit receives it.
        to: SendEnd,
    },
    /// `load -> select | shuffle | rotate -> send`.
    ///
    /// ⛔ EXACTLY ONE REARRANGEMENT — see [`Rearrangement`].
    ThroughRearrangement {
        /// The loaded value entering the rearrangement.
        data: Val,
        /// Which rearrangement.
        via: Rearrangement,
        /// The value the rearrangement binds, which the send carries.
        rearranged: Val,
        /// Which unit receives it.
        to: SendEnd,
    },
    /// `load -> store`. ⭐ NO SEND AND THEREFORE NO CONSUMER UNIT: the value never leaves this unit,
    /// which is why [`LoadChain::consumer`] answers `None` for it rather than inventing an end.
    ToStore {
        /// The loaded value the store writes.
        data: Val,
    },
}

impl LoadChain {
    /// WHICH UNIT CONSUMES THE LOAD — the reference's second return value.
    ///
    /// ⭐ A FIELD READ, NOT A SEARCH. The C++ reaches this through `send_op.getToUnit().getDefiningOp()`
    /// after a use-list walk; the send already holds its [`SendEnd`] here.
    ///
    /// ⛔ `None` FOR A STORE, AND THAT IS AN ABSENCE RATHER THAN A FAILURE. A load feeding a store
    /// never puts its value on a wire, so there is no consuming unit to name — distinct from the
    /// reference, which returns the same `nullptr` pair for "it was a store" and for "unsupported",
    /// making the two indistinguishable at the call site.
    #[must_use]
    pub const fn consumer(&self) -> Option<SendEnd> {
        match self {
            LoadChain::ToSend { to, .. } | LoadChain::ThroughRearrangement { to, .. } => Some(*to),
            LoadChain::ToStore { .. } => None,
        }
    }

    /// THE VALUE THE LOAD BOUND — the root the reference derives per load kind.
    ///
    /// ⛔ THE C++ PICKS IT BY OP KIND: `getResult(0)` for a vector load, but
    /// `getLoadInductionVar()` for a `CompositeLoad` or `CompositeIndirectLoad`
    /// (`Helper.cpp:1245-1250`) — a composite's consumer hangs off its induction variable, not its
    /// result. That distinction is the caller's to make when it builds the chain, because only the
    /// caller knows which op it loaded with; carrying it here would be storing the answer to a
    /// question this type does not ask.
    #[must_use]
    pub const fn data(&self) -> Val {
        match self {
            LoadChain::ToSend { data, .. }
            | LoadChain::ThroughRearrangement { data, .. }
            | LoadChain::ToStore { data } => *data,
        }
    }

    /// SPEND THE CHAIN, yielding what the lowering needs to emit the transfer.
    ///
    /// ⛔ TAKES `self`. One load, one consumer, once — which is `hasOneUse()` as a type rather than a
    /// check with an error arm.
    #[must_use]
    pub fn spend(self) -> (Val, Option<SendEnd>) {
        let consumer = self.consumer();
        (self.data(), consumer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::islands::dataflow_ir::link::{Link, Lxlu, Sfp};

    /// A wire whose ends are minted together, as the island requires.
    fn wire() -> SendEnd {
        let (send, _recv) = Link::<Lxlu, Sfp>::between(Val(1), Val(2)).ends();
        send
    }

    /// 🎯 A SEND'S CONSUMER IS THE WIRE'S END, not a unit a caller picked.
    #[test]
    fn the_consumer_comes_from_the_wire() {
        let to = wire();
        let chain = LoadChain::ToSend { data: Val(7), to };
        assert_eq!(chain.consumer(), Some(to));
        assert_eq!(chain.data(), Val(7));
    }

    /// 🎯 A REARRANGEMENT DOES NOT CHANGE THE CONSUMER, only what reaches it.
    ///
    /// ⭐ `data` IS WHAT ENTERED and `rearranged` IS WHAT THE SEND CARRIES — two values, because the
    /// reference's middle case has an op between them binding a new one.
    #[test]
    fn a_rearrangement_keeps_the_consumer_and_changes_the_value() {
        let to = wire();
        let chain = LoadChain::ThroughRearrangement {
            data: Val(7),
            via: Rearrangement::Shuffle,
            rearranged: Val(8),
            to,
        };
        assert_eq!(chain.consumer(), Some(to));
        assert_eq!(chain.data(), Val(7), "the value that ENTERED the rearrangement");
    }

    /// 🎯 A STORE HAS NO CONSUMING UNIT, AND THAT IS NOT A FAILURE.
    ///
    /// ⛔ THE REFERENCE CANNOT SAY THIS. It returns `(nullptr, nullptr)` both for a store and for an
    /// unsupported consumer, so a caller cannot tell "no unit, by design" from "I could not work it
    /// out". Here the first is `None` and the second is unconstructible.
    #[test]
    fn a_store_has_no_consuming_unit() {
        let chain = LoadChain::ToStore { data: Val(7) };
        assert_eq!(chain.consumer(), None);
        assert_eq!(chain.data(), Val(7));
    }

    /// 🎯 SPENDING YIELDS BOTH HALVES AND CONSUMES THE CHAIN.
    ///
    /// ⭐ THAT THE CHAIN CANNOT BE SPENT TWICE IS THE COMPILER'S TEST, not this one's — `spend` takes
    /// `self`, so a second call does not build. This only pins what it yields.
    #[test]
    fn spending_yields_the_value_and_the_consumer() {
        let to = wire();
        assert_eq!(
            LoadChain::ToSend { data: Val(3), to }.spend(),
            (Val(3), Some(to))
        );
        assert_eq!(
            LoadChain::ToStore { data: Val(4) }.spend(),
            (Val(4), None)
        );
    }
}
