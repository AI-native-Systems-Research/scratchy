//! The Python side: a bounded AST, the census gate that enforces its bounds, and the
//! owned parser that produces it.
//!
//! [`ast`] is this crate's OWN Python AST. It intentionally does not mirror CPython's
//! `ast` module wholesale -- it carries exactly the node kinds the measured census
//! reaches, so a construct outside scope has nowhere to go and must be refused by name.
//!
//! [`parser`] lexes and parses a strict Python subset into [`ast`] with no
//! dependencies, the way the torch carriers are parsed. It is the only parser; there
//! is no feature gate behind which a third-party one could return.

pub mod ast;
pub mod census;
pub mod parser;
