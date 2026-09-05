//! THE DEEPTOOLS IRs, AND THE BRIDGES BETWEEN THEM.
//!
//! ```text
//! SubtileIR tape ──► DataflowIR ──► SentientIR ──► ProgIR ──► SenProg ──► init_binary
//!    (scratchy)      island 1       island 2       island 3    (ported)
//!               ^bridge 1       ^bridge 2      ^bridge 3   ^bridge 4
//! ```
//!
//! # ⭐⭐ AN ISLAND IS AN IR AND NOTHING ELSE
//!
//! [`islands`]`::<ir>` holds one IR's types, its invariants and its printer. It knows nothing
//! about who produces it or who consumes it — that is what keeps it checkable against the
//! reference files on its own terms, rather than against whatever we happened to emit.
//!
//! # ⭐⭐ A BRIDGE IS WHERE TWO VOCABULARIES MEET
//!
//! [`bridges`]`::<from>_to_<to>` is exactly where a join gets fudged, so each gets its own
//! directory and its own tests rather than being spread through the island it feeds.
//!
//! # ⛔⛔ NO SCRATCHY DEPENDENCY, EVER
//!
//! Anything that speaks scratchy's `TensorRegion` or `SubtileNode` lives on the scratchy side of
//! bridge 1. The half of that bridge which speaks the tape belongs there; the half that speaks
//! DataflowIR belongs here. An edge back into scratchy is what lets a lowering be *recovered*
//! from its own output instead of read from its input.

pub mod bridges;
pub mod islands;
