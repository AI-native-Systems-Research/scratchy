//! Superseded by the typed table.

// Flagged, one per comment form.

// This helper moved from the old module.
fn a() {}

/// The table used to be a Vec.
fn b() {}

/* The cache was removed. */
fn c() {}

// See `/tmp/report.md` for the numbers.
fn d() {}

/// Copied from (~/tmp/src/lib.rs).
fn e() {}

// Clean: comment-like text inside strings, and tmp as a word.

fn f() -> (&'static str, &'static str) {
    ("// moved from x", r#"/tmp/a and used to be"#)
}

// The gate/up/tmp/out tensors and {out}_tmp are names, not paths.
fn g() {}

// It moves to the next row; at most 4 tokens, no longer.
fn h() {}

fn main() {
    a();
    b();
    c();
    d();
    e();
    let _ = f();
    g();
    h();
}
