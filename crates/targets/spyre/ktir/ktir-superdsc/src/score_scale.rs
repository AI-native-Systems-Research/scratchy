// SPDX-License-Identifier: Apache-2.0
//! THE SCORE-SCALE LAW, AS TYPES — every way an attention's score multiplier can go missing,
//! mis-placed, or double-applied, each locked by a type instead of a comment.
//!
//! The score scale `s` must appear EXACTLY ONCE on each side of the score dot product —
//! `scores = (q·√s)·(k·√s)ᵀ = (q·kᵀ)·s` — and the two halves may live in different places: both in
//! the program (`qk · s`), both in the staged weights (√s folded into `W_q`/`W_k`, the node
//! rewritten to 1.0), or split (the door's `qs = Q·√s` with the pool storing `K·√s`). What must
//! never happen is a scale that lives NOWHERE — a program stating no multiplier over a pool of
//! raw K is `t = 1.0`, fluent garbage — or a scale applied TWICE — pre-scaled weights behind a
//! program that still states the multiplier.
//!
//! ⛔ THIS MODULE EXISTS BECAUSE EVERY ONE OF THOSE FACTS WAS A VALUE OR A COMMENT:
//!
//! | the wall (as it stood) | the type that closes it |
//! |---|---|
//! | `BundleAttnParams::scale_in_weights: bool` — a runtime flag standing in for a proof; a `bool` cannot say WHICH of the two legal skip reasons holds, so its consumer cannot be made exhaustive | [`ScoreScaleLaw`] — three arms, each minted only where its proof exists, consumed by exhaustive match |
//! | "The resident cache stores K ALREADY scaled" (`lower_ktir_to_superdsc.rs`, the door's √s split) — a comment invariant | [`NewK`] — the K a pool write carries, typed by its scale law; the write cannot accept a bare name |
//! | "PLAIN rotated K — the fp8 k_proj's scale fold is refused on this path, so the readers own the scale" (`ktir_superdsc_door.rs`, the rope append) — the OTHER law, same `&str` field | the same [`NewK`], one constructor per law, so the two paths must NAME which they write |
//! | "the fold rewrote the scale to 1.0 upstream" — a rewritten `1.0` is indistinguishable from a model's native `1.0` (`gemma-4` states it honestly; a half-applied fold states it while the weights carry nothing) | the law is minted from the fold's own receipt (membership in `attn_scale_weight_folds`, post-gates) and from the IR node's own payload — the IR is shared-immutable (`&SubtileIR`), every rewrite happens on a clone, so the value read at the mint is the arch's own statement and not a rewrite's residue |
//! | the walk lowering a multiplier-less attention program over a raw-K pool — silent `t = 1.0` | the `Program::Attn` door cross-checks the program's own statement against the bundle's law; a disagreement is a BUILD ERROR naming both, not an emission |
//!
//! The one fact that stays a value and not a type is the multiplier's MAGNITUDE: the program is
//! the single reading of it (`program_score_scale`), and the door's old caller-stated `f32` twin
//! was deleted precisely because a second copy of one config number can only ever agree. What
//! the types lock is WHERE the scale lives — the only question whose wrong answer is invisible
//! in every dump, because a missing √s multiplies by 1 and a double-applied one by `s`, and
//! neither is a shape a buffer can contradict.

/// ⭐⭐⭐ WHERE ONE BUNDLE'S ATTENTION SCORE SCALE LIVES — proof-carrying, minted only where the
/// proof exists, consumed by exhaustive match.
///
/// One law per bundle (the one-value-per-bundle refusal lives where it is minted,
/// `attn_bundle_params`): a graph whose attention nodes disagree about where their scales live
/// cannot lower one without mislowering the other.
///
/// The arms and their proofs:
///   * [`StatedByProgram`](Self::StatedByProgram) — the DEFAULT, and the live law for every model
///     whose attention multiplier is a real number below 1 (√s < 1 cannot land in the weights
///     through the f16 floor gate, so the fold is refused for all of them): the program states
///     the multiplier and the walk lowers it. Proof: the absence of every other proof.
///   * [`CommutedIntoWeights`](Self::CommutedIntoWeights) — the fold's own receipt: √s was
///     commuted into BOTH projection weights by `attn_scale_weight_folds`' gates (fp8 payload
///     refused, shared readers refused, f16 floor cleared pair-wise). Minted from membership in
///     that list and from nothing else — a refused fold mints nothing and the law stays
///     `StatedByProgram` with the multiplier on the tape, which is the safe direction.
///   * [`UnitByModel`](Self::UnitByModel) — the model's own multiplier IS 1.0 (`gemma-4`'s arch
///     defaults `attention_multiplier` to it): √1 = 1, so nothing to state, fold, or carry, and a
///     program that states no multiplier is telling the truth. Minted from the IR node's own
///     payload, which is the arch's statement — the IR is `&`-shared and every fold rewrite
///     happens on a clone, so a `1.0` read here is the model's, never a rewrite's.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ScoreScaleLaw {
    /// The program states the multiplier (`qk · s`); the walk lowers it. The pool's K is RAW and
    /// every reader owns the whole scale in its own program.
    StatedByProgram,
    /// √s was commuted into both projection weights at load — the fold's receipt. The node is
    /// rewritten to 1.0 where it is lowered, the program truthfully states no multiplier, and
    /// the pool's K is born already scaled.
    CommutedIntoWeights,
    /// The model's own multiplier is 1.0 — √1 = 1 leaves nothing for anyone to carry.
    UnitByModel,
}

impl ScoreScaleLaw {
    /// ⭐ THE POOL LAW THIS LAW'S WRITES CARRY — the pairing, stated ONCE and consumed by BOTH
    /// sides of the pool contract. The write mint (the rope append's [`NewK`]) and every read
    /// check (the door's `Program::Attn` cross-check, the splice's `SCALE` derivation) derive
    /// the pool's bytes through THIS method, so the two sides cannot disagree about what the
    /// pool carries: a reader that stated its obligation from one spelling of the pairing over
    /// a write minted from another would read a `SqrtScaled` pool as raw (scores missing √s,
    /// the ~10× dilution) or a `Raw` one as pre-scaled (scores carrying a stray s) — both
    /// invisible to every extent law, both now one derivation.
    pub const fn pool_k(self) -> PoolKLaw {
        match self {
            ScoreScaleLaw::CommutedIntoWeights => PoolKLaw::SqrtScaled,
            ScoreScaleLaw::StatedByProgram | ScoreScaleLaw::UnitByModel => PoolKLaw::Raw,
        }
    }
}

/// What a pool write's K BYTES carry — a byte fact about the write, stated by the constructor
/// that minted it. The reader's OBLIGATION is derived from the bundle's [`ScoreScaleLaw`] (the
/// proof), not from this: a `Raw` pool under `UnitByModel` needs no reader-side multiply just as
/// a `SqrtScaled` one needs none under `CommutedIntoWeights`.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum PoolKLaw {
    /// The K carries √s — written by an explicit multiply, or born from fold-carrying weights.
    SqrtScaled,
    /// The K carries no score factor. Readers under [`ScoreScaleLaw::StatedByProgram`] must
    /// state the whole scale in their own programs; readers under
    /// [`ScoreScaleLaw::UnitByModel`] need nothing (the whole scale is 1).
    Raw,
}

/// ⭐ THE K A KV-POOL WRITE CARRIES, TYPED BY ITS SCALE LAW — the replacement for the `&str`
/// field that let two paths with OPPOSITE scale laws hand the recipe the same bare name.
///
/// ⛔ THE WALL THIS CLOSES. The recipe ([`crate::emit::kv_cache_write`]) was given
/// `new_k: &str` and its own doc admitted the hole: "The recipe writes whatever it is given;
/// the scale law belongs to the caller." The two callers hold opposite laws — the attention
/// door multiplies both sides by √s and stores `K·√s` ("future steps' prefix scores need no
/// further K scaling"), while the rope append on the spliced path stores PLAIN rotated K
/// (the fp8 k_proj's fold is refused there, so the readers own the scale). Both correct in
/// isolation, both spelled identically at the signature, and nothing between them and a
/// cross-wiring but two comments. Now the write's law is the TYPE of its input, minted by one
/// of three constructors that each name the proof they demand:
///
///   * [`NewK::reader_stated`] — RAW K. The live spelling for the spliced path's append (and
///     for a unit-scale model, whose raw K IS its scaled K).
///   * [`NewK::born_scaled_by_fold`] — K from a GEMM whose weights the fold proved carry √s.
///     Demands the bundle's [`ScoreScaleLaw::CommutedIntoWeights`] receipt; the name is the
///     contract.
///   * [`NewK::sqrt_scaled_by_multiply`] — K that an EMITTED multiply just scaled (`qs`'s twin
///     on the K side). Door-private: only the attention door's multiply arm can honestly claim
///     it, because only that arm emits the multiply.
///
/// There is no `From<&str>` and no public field access to the law's setter: a caller cannot
/// spell a law it does not hold, and the recipe cannot write a K whose law it was not told.
pub struct NewK<'a> {
    law: PoolKLaw,
    name: &'a str,
}

impl<'a> NewK<'a> {
    /// RAW K — no scale factor in the bytes. The readers own the whole scale in their own
    /// programs (or the model's scale is 1 and there is nothing to own).
    pub fn reader_stated(name: &'a str) -> Self {
        NewK {
            law: PoolKLaw::Raw,
            name,
        }
    }

    /// K BORN SCALED from a projection whose weights the fold proved carry √s — the receipt is
    /// the bundle's [`ScoreScaleLaw::CommutedIntoWeights`], held by the caller that matched it
    /// out of its bundle params. The rope append's spelling for a folded model: the staged K
    /// leaves the k_proj GEMM already carrying √s, so the pool stores it and future prefix
    /// scores state no multiplier of their own.
    pub fn born_scaled_by_fold(name: &'a str) -> Self {
        NewK {
            law: PoolKLaw::SqrtScaled,
            name,
        }
    }

    /// K SCALED BY AN EMITTED MULTIPLY — `new_k_scaled`, the twin of the door's `qs`. Private to
    /// this crate because only the attention door's multiply arm emits that multiply; claiming
    /// this law without emitting the op would write a pool the score legs trust for a √s no
    /// bytes carry.
    pub(crate) fn sqrt_scaled_by_multiply(name: &'a str) -> Self {
        NewK {
            law: PoolKLaw::SqrtScaled,
            name,
        }
    }

    /// The law these bytes carry — for the read side, which must agree with what its program
    /// states (the `Program::Attn` door cross-check).
    pub fn law(&self) -> PoolKLaw {
        self.law
    }

    /// The tensor's name — the recipe's spelling, unchanged from the `&str` it replaced.
    pub fn name(&self) -> &'a str {
        self.name
    }
}
