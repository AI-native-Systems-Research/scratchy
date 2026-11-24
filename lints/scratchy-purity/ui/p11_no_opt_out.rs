// A site never opts out of a purity lint: an allow is itself an error.
#![allow(p11_tombstone_comment, p11_escape_hatch_constructor)]

// This helper moved from `old::module`.
pub struct Heads(u32);

impl Heads {
    pub fn from_untyped(n: u32) -> Self {
        Self(n)
    }
}

fn main() {
    let _ = Heads::from_untyped(1).0;
}
