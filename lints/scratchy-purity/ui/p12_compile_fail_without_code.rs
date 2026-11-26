//! A crate whose type proofs are doctests.
//!
//! ```compile_fail
//! let x: u32 = "no";
//! ```

// Flagged, one per doc comment form.

/// A type proof that names no error code.
///
/// ```compile_fail
/// let x: u32 = "no";
/// ```
pub fn a() {}

/**
 * The same, in a block doc comment.
 *
 * ```compile_fail
 * let x: u32 = "no";
 * ```
 */
pub fn b() {}

// Clean: a fence that names its code, beside other attributes or not, and other fences.

/// ```compile_fail,E0277
/// let x: u32 = 1u64;
/// ```
pub fn c() {}

/// ```compile_fail,E0599,edition2021
/// 1u32.no_such_method();
/// ```
pub fn d() {}

/// ```rust
/// let x: u32 = 1;
/// ```
pub fn e() {}

// A fence in a plain comment is not a doctest.
// ```compile_fail
// let x: u32 = "no";
// ```
pub fn f() -> &'static str {
    "```compile_fail"
}

fn main() {
    a();
    b();
    c();
    d();
    e();
    let _ = f();
}
