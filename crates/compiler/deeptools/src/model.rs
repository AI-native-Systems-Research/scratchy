//! THE MODEL, AS CONSTANTS — the geometry the emitter is allowed to know, carried by a trait.
//!
//! ⭐ THE DOOR IS SCRATCHY'S AND IT ALREADY EXISTS. `subtile`'s `model_geometry` turns a config's
//! `(nqh, nkvh, head_dim)` into CONST GENERICS at the bake
//! (`with_config_attn_geometry` -> `OnAttnGeometry::on_geometry<NQH, NKVH, HD>`), with the GQA
//! divisibility proof already spent at `ModelAttnGeometry::mint`. This trait is what that door hands
//! to: scratchy implements it inside the arm, where the numbers are constants, and the emitter is
//! generic over it. The dependency stays one-way, `subtile -> deeptools`.
//!
//! ⛔ SO DO NOT RE-DERIVE A GEOMETRY HERE. `mint` is the ONE place the kv/query division is decided
//! (`model_geometry.rs:93-106`), and it returns `None` for a config whose kv-head count does not
//! divide its query-head count — a model whose attention would otherwise read another head's keys.
//! Anything in this file that recomputed that would be a second opinion about a settled fact.

/// ONE MODEL'S GEOMETRY, AS A TYPE.
///
/// ⛔ EVERY CONSTANT IS STATED, none defaulted — the same rule [`crate::arch::Arch`] follows, and for
/// the same reason: an associated const with a default is a value a new model can forget to set and
/// still compile. The one exception is [`Model::GQA`], which is DERIVED rather than stated, and is
/// therefore the one thing an impl must not be able to write.
pub trait Model {
    /// `num_attention_heads`.
    const QUERY_HEADS: u32;

    /// `num_key_value_heads`. Equal to [`Model::QUERY_HEADS`] only for a non-GQA model, which is
    /// exactly why it is a separate constant and not an assumption.
    const KV_HEADS: u32;

    /// `head_dim` — how wide ONE head is. Not a hidden size, not a stick width, not a lane count:
    /// the three quantities a head dim is most often confused with, all of which are also small
    /// powers of two.
    const HEAD_DIM: u32;

    /// `hidden_size`.
    const HIDDEN: u32;

    /// `num_hidden_layers`.
    const LAYERS: u32;

    /// `intermediate_size` — the FFN width.
    const FFN: u32;

    /// `vocab_size`.
    const VOCAB: u32;

    /// QUERY HEADS PER KV HEAD, divided out here and nowhere else.
    ///
    /// ⛔ NO `max(1)`, NO ROUNDING. The division is exact because [`Model::WELL_FORMED`] refuses a
    /// model where it is not, at compile time. The arithmetic that used to paper over it —
    /// `(nqh / nkvh.max(1)).max(1)` — yields a plausible group size whose attention reads another
    /// head's keys, which is fluent wrong output rather than a fault.
    const GQA: u32 = {
        let () = Self::WELL_FORMED;
        Self::QUERY_HEADS / Self::KV_HEADS
    };

    /// The `[rows, nqh*hd]` token-stream width — the query projection's and the attention output's
    /// column count.
    const Q_WIDTH: u32 = {
        let () = Self::WELL_FORMED;
        Self::QUERY_HEADS * Self::HEAD_DIM
    };

    /// The `[rows, nkvh*hd]` kv-stream width — one K or V projection's column count, narrower than
    /// [`Model::Q_WIDTH`] by exactly [`Model::GQA`].
    const KV_WIDTH: u32 = {
        let () = Self::WELL_FORMED;
        Self::KV_HEADS * Self::HEAD_DIM
    };

    /// ⭐⭐ THE MODEL'S OWN INVARIANTS, AS A CONST THE COMPILER EVALUATES.
    ///
    /// A trait's associated const is monomorphised per impl, so every assertion below is checked once
    /// per model at BUILD time. That is what makes this a lock rather than a comment — but only for a
    /// model something forces, which is what [`Model::check`] is for.
    const WELL_FORMED: () = {
        assert!(Self::QUERY_HEADS > 0, "a model with no query heads");
        assert!(Self::KV_HEADS > 0, "a model with no kv heads");
        assert!(Self::HEAD_DIM > 0, "a model with a zero-wide head");
        assert!(Self::HIDDEN > 0, "a model with no hidden size");
        assert!(Self::LAYERS > 0, "a model with no layers");
        assert!(Self::FFN > 0, "a model with no FFN width");
        assert!(Self::VOCAB > 0, "a model with an empty vocabulary");
        assert!(
            Self::QUERY_HEADS % Self::KV_HEADS == 0,
            "kv-head count does not divide query-head count: this model has no GQA grouping, and \
             any group size derived from it would address another head's keys"
        );
        // ⛔ AND NOT `GQA * KV_HEADS == QUERY_HEADS`, WHICH IS WHAT A CYCLE LOOKS LIKE. [`Model::GQA`]
        // reads this const before dividing, so an assertion here ABOUT `GQA` makes the two constants
        // depend on each other and rustc refuses the crate with E0391 rather than the model. The
        // divisibility above is the same fact stated where it does not close a loop: given it, the
        // division is exact, so there is nothing left for a second assertion to catch.
    };

    /// FORCE [`Model::WELL_FORMED`] where nothing else has.
    ///
    /// ⛔⛔ AN UNREFERENCED ASSOCIATED CONST IS NEVER EVALUATED, AND THAT WAS MEASURED, NOT ASSUMED.
    /// The first version of this file put the assertions in `WELL_FORMED` alone and offered this
    /// method to force them; a model with `nqh=32, nkvh=5` then COMPILED CLEAN, because nothing on a
    /// live path read the const. The fix is that [`Model::GQA`], [`Model::Q_WIDTH`] and
    /// [`Model::KV_WIDTH`] each read `WELL_FORMED` before computing, so the invariant fires on any
    /// use of a derived quantity rather than on a call someone remembered to write.
    ///
    /// ⭐ THIS REMAINS FOR THE EMITTER'S ENTRY POINT, which forces it once at the top so that a model
    /// whose derived constants happen not to be read on some path is still checked.
    fn check() {
        let () = Self::WELL_FORMED;
    }
}
