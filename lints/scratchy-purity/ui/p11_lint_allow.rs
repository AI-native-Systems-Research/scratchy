// Flagged: every attribute that lowers a lint's level, for any lint.
#![allow(unused_variables)]
#![expect(unused_mut)]
#![allow(rustdoc::broken_intra_doc_links)]

#[allow(dead_code)]
fn a() {}

#[expect(dead_code)]
fn b() {}

#[allow(clippy::needless_return)]
pub fn i() -> u32 {
    return 1;
}

#[cfg_attr(all(), allow(dead_code))]
fn c() {}

#[cfg_attr(all(), cfg_attr(all(), expect(dead_code)))]
fn d() {}

// Flagged too: code no build compiles (a test, a disabled feature).
#[cfg(any())]
mod disabled {
    #[allow(dead_code)]
    pub fn e() {}
}

// A macro that stamps an allow is flagged at the macro.
macro_rules! stamp {
    ($name:ident) => {
        #[allow(dead_code)]
        fn $name() {}
    };
}
stamp!(f);

// An allow whose name a macro argument supplies is flagged where it expands.
macro_rules! level {
    ($level:ident, $name:ident) => {
        #[$level(dead_code)]
        fn $name() {}
    };
}
level!(allow, g);

// Clean: tightening levels, and the word as a name, in a string or in a comment.
#[deny(unused_must_use)]
#[warn(unused_imports)]
#[forbid(unsafe_code)]
pub struct Policy {
    pub allow: bool,
    pub allow_list: Vec<u8>,
}

/// Never write `#[allow(dead_code)]`; fix what the lint reports.
pub fn h(p: &Policy) -> (&'static str, bool) {
    (r#"#[allow(dead_code)]"#, p.allow)
}

fn main() {
    let mut x = 1;
    let y = 2;
    let _ = i();
    let _ = h(&Policy { allow: true, allow_list: vec![x] });
}
