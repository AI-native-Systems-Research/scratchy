// Flagged: the escape-hatch names, wherever a fn is defined.
pub struct Heads(u32);

impl Heads {
    pub fn of(n: u32) -> Option<Self> {
        (n > 0).then_some(Self(n))
    }
    pub fn from_untyped(n: u32) -> Self {
        Self(n)
    }
    pub fn conflated_with_kv(n: u32) -> Self {
        Self(n)
    }
}

pub fn conflated_with_heads(n: u32) -> Heads {
    Heads(n)
}

macro_rules! stamp {
    ($t:ident) => {
        impl $t {
            pub fn from_untyped(n: u32) -> Self {
                Self(n)
            }
        }
    };
}

pub struct Rows(u32);
stamp!(Rows);

// Clean: other names, and calls to a flagged fn (only its definition is flagged).
pub fn from_untypedness() {}
pub fn untyped() {}

fn main() {
    let _ = Heads::from_untyped(1);
    let _ = Heads::of(1).map(|h| h.0);
    let _ = Rows::from_untyped(2).0;
}
