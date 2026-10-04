//! # triton-frontend: Python kernel source -> TTIR, in Rust.
//!
//! This is BRIDGE ONE of `Triton -> KTIR -> SuperDSC`. It takes the text of a
//! `@triton.jit` kernel plus its signature and constexpr bindings, and produces an
//! in-memory [`ttir::Module`]. **No Python runs. Nothing is shelled out to.**
//!
//! ## NO TEXT BOUNDARIES IN THE PIPELINE.
//!
//! The product of this crate is a VALUE, not a file. Bridge two consumes
//! [`ttir::Module`] directly. There is no text form of a program at any point in the
//! compile path -- the same rule KTIR downstream states outright. Two text facilities
//! exist and both are TEST INSTRUMENTS:
//!
//!   * [`ttir::print`] renders a module as TTIR text, for the golden diff and for reading
//!     by eye when something is wrong;
//!   * [`ttir::parse`] reads TTIR text, for ONE purpose: loading the goldens the existing
//!     Python toolchain emits so [`diff`] can compare against them.
//!
//! Neither is reachable from the lowering path. Printing a module so another stage can
//! parse it back would be the bug this note exists to prevent.
//!
//! ## THE SWAPPABLE BOUNDARY.
//!
//! The shared vocabulary between the two bridges is confined to ONE module: [`ttir`]
//! (`src/ttir/mod.rs`). Whether the KTIR types downstream end up being scratchy's
//! vendored `ktir-core` or ours is an open decision, so replacing this vocabulary must be
//! a contained change. What depends on it:
//!
//!   * [`semantic`] and [`codegen`] construct it;
//!   * [`ttir::print`] / [`ttir::parse`] render and read it (tests only);
//!   * [`diff`] compares two of them;
//!   * bridge two consumes it.
//!
//! [`py`] (the Python AST) never mentions a TTIR type, and [`ttir`] never mentions a
//! Python one. The two meet only inside [`codegen`].
//!
//! ## THE PYTHON PARSER IS BEHIND A CARGO FEATURE.
//!
//! Our other crates are dependency-free so `cargo test --offline` works on the pod, which
//! has no crates.io access. `ruff_python_parser` pulls 84 packages, so it lives behind the
//! `ruff` feature and only [`py::ruff_adapter`] ever sees a ruff type. Everything else --
//! the TTIR value type, the printer, the golden reader, the semantic layer, the AST walk
//! and the diff -- builds and tests with no dependencies at all.
//!
//! ## SCOPE IS MEASURED, AND OUTSIDE IT IS A NAMED ERROR.
//!
//! `../triton-superdsc/tools/ast_census.py` walks the five fixtures' `@triton.jit`
//! functions and reports exactly 33 AST node types and 33 call targets. That census is the
//! scope boundary, enforced in [`py::census`]. A construct outside it is a compile error
//! that NAMES the construct and its source line -- never a silent partial lowering.
//!
//! ## ONE DELIBERATE DIVERGENCE FROM TRITON'S SEMANTICS.
//!
//! Triton promotes `f16 / f16 -> f32`. That promotion is WRONG ON THIS DEVICE and is
//! turned off by [`target::Target::div_promotes_narrow_floats`]. See [`target`] for the
//! measurement and the reasoning; it is documented there rather than here so a reader who
//! goes looking for the divergence finds it next to the switch.

pub mod diff;
pub mod mangle;
pub mod ttir;

pub mod py;
pub mod semantic;
pub mod target;
pub mod codegen;
pub mod opt;

pub use ttir::Module;

/// A compile error: a message, and where in the Python source it came from.
///
/// Every refusal in this crate carries a line and column, because the whole point of the
/// census gate is that an unsupported construct is named AT ITS SOURCE SITE rather than
/// producing a partial lowering that fails later somewhere worse.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Error {
    pub message: String,
    pub line: u32,
    pub col: u32,
}

impl Error {
    pub fn new(message: impl Into<String>, line: u32, col: u32) -> Error {
        Error {
            message: message.into(),
            line,
            col,
        }
    }
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "at {}:{}: {}", self.line, self.col, self.message)
    }
}

impl std::error::Error for Error {}

pub type Result<T> = std::result::Result<T, Error>;
