// SPDX-License-Identifier: Apache-2.0
//! The symbolic dimension: a config bound, a literal, or a product or quotient of those, and the one
//! canonical order products are written in. The vocabulary every shape in the compiler is spelled
//! in, shared by the classifier, the weights manifest, the shape solver and the typed-Rust emit pass.

/// A dimension expression.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum Dim {
    /// Concrete integer literal.
    Lit(u64),
    /// Named bound — resolves to a config.json integer per-model.
    Bound(String),
    /// Product of subterms. Canonically flattened: no nested Mul.
    Mul(Vec<Dim>),
    /// Quotient: `numerator / denominator`. Surfaced today only by
    /// DSL-authored reshape arithmetic (`num_tokens /
    /// vision_merge_factor`). The Solver treats it as concrete:
    /// `unify` checks structural equality, `walk` / `close_dim`
    /// recurse into both children. `eval_closed_dim` evaluates by
    /// integer division (caller's responsibility to ensure the
    /// dividend is exactly divisible at runtime — the merger
    /// reshape's `total_l / spatial_merge_size**2` is always
    /// exact).
    Div(Box<Dim>, Box<Dim>),
    /// Unresolved variable. Must be resolved before inference
    /// completes or shape inference reports an error.
    Var(DimVar),
}

/// Fresh-var id. Allocated by the [`Solver`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DimVar(pub(crate) u32);

/// A tensor shape — an ordered sequence of dims.
pub type Shape = Vec<Dim>;

/// Flatten and sort a Mul's children to a canonical form. Single-
/// element products collapse to their inner dim.
pub fn canonical_mul(mut children: Vec<Dim>) -> Dim {
    // Flatten nested Muls.
    let mut flat = Vec::with_capacity(children.len());
    for c in children.drain(..) {
        match c {
            Dim::Mul(inner) => flat.extend(inner),
            other => flat.push(other),
        }
    }
    // Drop Lit(1)s.
    flat.retain(|d| !matches!(d, Dim::Lit(1)));
    // Deterministic ordering for canonicalization.
    flat.sort_by(dim_cmp);
    match flat.len() {
        0 => Dim::Lit(1),
        1 => flat.pop().unwrap(),
        _ => Dim::Mul(flat),
    }
}

fn dim_cmp(a: &Dim, b: &Dim) -> std::cmp::Ordering {
    // Bound < Lit < Mul < Var ordering, breaking ties within each.
    use std::cmp::Ordering;
    fn rank(d: &Dim) -> u8 {
        match d {
            Dim::Bound(_) => 0,
            Dim::Lit(_) => 1,
            Dim::Mul(_) => 2,
            Dim::Div(_, _) => 3,
            Dim::Var(_) => 4,
        }
    }
    match rank(a).cmp(&rank(b)) {
        Ordering::Equal => match (a, b) {
            (Dim::Bound(x), Dim::Bound(y)) => x.cmp(y),
            (Dim::Lit(x), Dim::Lit(y)) => x.cmp(y),
            (Dim::Mul(x), Dim::Mul(y)) => x.len().cmp(&y.len()),
            // Div ordering is opaque (no internal order needed for
            // canonicalization; only one Div per merger reshape today).
            (Dim::Div(_, _), Dim::Div(_, _)) => Ordering::Equal,
            (Dim::Var(x), Dim::Var(y)) => x.0.cmp(&y.0),
            _ => unreachable!(),
        },
        other => other,
    }
}
