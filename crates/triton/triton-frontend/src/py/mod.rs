//! The Python side: a bounded AST, the census gate that enforces its bounds, and the
//! parser adapter.
//!
//! [`ast`] is this crate's OWN Python AST. It intentionally does not mirror CPython's
//! `ast` module wholesale -- it carries exactly the node kinds the measured census
//! reaches, so a construct outside scope has nowhere to go and must be refused by name.
//!
//! [`ruff_adapter`] (feature `ruff`) lowers `ruff_python_ast` into [`ast`], and is the
//! ONLY place in the crate that mentions a ruff type.

pub mod ast;
pub mod census;

#[cfg(feature = "ruff")]
pub mod ruff_adapter;
