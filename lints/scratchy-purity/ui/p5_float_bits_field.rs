// Flagged: a float stored as bits in a struct field, read back or written through a float.
pub struct Norm {
    pub eps_bits: u32,
    pub theta_bits: u64,
}

impl Norm {
    pub fn new(eps: f32) -> Self {
        Self { eps_bits: eps.to_bits(), theta_bits: 0 }
    }
    pub fn theta(&self) -> f64 {
        f64::from_bits(self.theta_bits)
    }
}

pub enum Rope {
    Scaled { factor_bits: u32 },
}

impl Rope {
    pub fn factor(&self) -> f32 {
        let Rope::Scaled { factor_bits } = self;
        f32::from_bits(*factor_bits)
    }
}

// Clean: other widths, a float, a raw word no float is read from, and bits that are not a field.
pub struct Counts {
    pub count_bits: u8,
    pub eps: f32,
    pub mask_bits: u32,
}

pub fn from_bits(x_bits: u32) -> f32 {
    let y_bits: u32 = x_bits;
    f32::from_bits(y_bits)
}

fn main() {
    let n = Norm::new(1e-6);
    let c = Counts { count_bits: 0, eps: 0.0, mask_bits: 0xFFFE };
    let _ = (n.eps_bits, n.theta(), c.count_bits, c.eps, c.mask_bits, Rope::Scaled { factor_bits: 0 }.factor(), from_bits(0));
}
