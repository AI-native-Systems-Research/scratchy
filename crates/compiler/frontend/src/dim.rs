// SPDX-License-Identifier: Apache-2.0
//! The symbolic dimension: a config bound, a literal, or a product or quotient of those, and the one
//! canonical order products are written in. The vocabulary every shape in the compiler is spelled
//! in, shared by the classifier, the weights manifest and the typed-Rust emit pass.

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
    /// vision_merge_factor`), as the typed `Div<A, B>`. `eval_closed_dim` evaluates by
    /// integer division (caller's responsibility to ensure the
    /// dividend is exactly divisible at runtime — the merger
    /// reshape's `total_l / spatial_merge_size**2` is always
    /// exact).
    Div(Box<Dim>, Box<Dim>),
    /// Sum of terms: a packed projection's width, its parts side by side (`gdn_conv_dim`, a
    /// Mamba-2 `in_proj`'s). Canonically flattened like `Mul`: no nested `Add`.
    Add(Vec<Dim>),
}

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

/// Flatten and sort a sum's terms to a canonical form, in [`canonical_mul`]'s order. Single-term
/// sums collapse to their inner dim; `Lit(0)` terms drop.
pub fn canonical_add(mut terms: Vec<Dim>) -> Dim {
    let mut flat = Vec::with_capacity(terms.len());
    for t in terms.drain(..) {
        match t {
            Dim::Add(inner) => flat.extend(inner),
            other => flat.push(other),
        }
    }
    flat.retain(|d| !matches!(d, Dim::Lit(0)));
    flat.sort_by(dim_cmp);
    match flat.len() {
        0 => Dim::Lit(0),
        1 => flat.pop().unwrap(),
        _ => Dim::Add(flat),
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
            Dim::Add(_) => 3,
            Dim::Div(_, _) => 4,
        }
    }
    match rank(a).cmp(&rank(b)) {
        Ordering::Equal => match (a, b) {
            (Dim::Bound(x), Dim::Bound(y)) => x.cmp(y),
            (Dim::Lit(x), Dim::Lit(y)) => x.cmp(y),
            (Dim::Mul(x), Dim::Mul(y)) | (Dim::Add(x), Dim::Add(y)) => x.len().cmp(&y.len()),
            // Div ordering is opaque (no internal order needed for
            // canonicalization; only one Div per merger reshape today).
            (Dim::Div(_, _), Dim::Div(_, _)) => Ordering::Equal,
            _ => unreachable!(),
        },
        other => other,
    }
}

/// A derived dim's rule over config bounds: the arithmetic a `weights.json` name like `kv_dim`
/// stands for. The one table of these, [`DERIVED_DIMS`], is what the config loader derives each
/// model's value from and what the typed-Rust emit pass spells as a type, so the two cannot drift.
#[derive(Clone, Copy, Debug)]
pub enum DerivedRule {
    /// A config bound.
    Bound(&'static str),
    /// A literal.
    Lit(u64),
    /// The product of the factors.
    Mul(&'static [DerivedRule]),
    /// The sum of the terms.
    Add(&'static [DerivedRule]),
    /// `on` where the config's `flag` is nonzero, else `off` (an absent flag is zero).
    Gated {
        flag: &'static str,
        on: &'static DerivedRule,
        off: &'static DerivedRule,
    },
}

impl DerivedRule {
    /// The config flag a gated rule chooses its arm by.
    pub fn gate_flag(&self) -> Option<&'static str> {
        match self {
            Self::Gated { flag, .. } => Some(flag),
            _ => None,
        }
    }

    /// The rule as the dim it derives, a gated rule's arm chosen by its flag in `bounds` (an absent
    /// flag is zero).
    pub fn dim(&self, bounds: &std::collections::BTreeMap<String, u64>) -> Dim {
        match self {
            Self::Bound(name) => Dim::Bound(name.to_string()),
            Self::Lit(n) => Dim::Lit(*n),
            Self::Mul(factors) => canonical_mul(factors.iter().map(|f| f.dim(bounds)).collect()),
            Self::Add(terms) => canonical_add(terms.iter().map(|t| t.dim(bounds)).collect()),
            Self::Gated { flag, on, off } => match bounds.get(*flag).copied().unwrap_or(0) {
                0 => off.dim(bounds),
                _ => on.dim(bounds),
            },
        }
    }

    /// The rule's value under `bounds`, `None` when a bound it reads is absent.
    pub fn eval(&self, bounds: &std::collections::BTreeMap<String, u64>) -> Option<u64> {
        match self {
            Self::Bound(name) => bounds.get(*name).copied(),
            Self::Lit(n) => Some(*n),
            Self::Mul(factors) => {
                (factors.iter()).try_fold(1, |acc, f| Some(acc * f.eval(bounds)?))
            }
            Self::Add(terms) => (terms.iter()).try_fold(0, |acc, t| Some(acc + t.eval(bounds)?)),
            Self::Gated { flag, on, off } => {
                if bounds.get(*flag).copied().unwrap_or(0) != 0 {
                    on.eval(bounds)
                } else {
                    off.eval(bounds)
                }
            }
        }
    }
}

/// When the config loader derives a [`DerivedDim`]: with the implicit bounds, or after the generic
/// defaults (the global attention class's widths read `global_head_dim` /
/// `num_global_key_value_heads`, which default then).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeriveStage {
    Implicit,
    AfterDefaults,
}

/// A config bound derived from others, where no config states it.
#[derive(Clone, Copy, Debug)]
pub struct DerivedDim {
    pub name: &'static str,
    pub stage: DeriveStage,
    pub rule: DerivedRule,
}

/// Every derived dim, in derivation order (a rule may read one derived above it).
pub const DERIVED_DIMS: [DerivedDim; 11] = [
    DerivedDim {
        name: "attn_q_dim",
        stage: DeriveStage::Implicit,
        rule: DerivedRule::Mul(&[
            DerivedRule::Bound("num_attention_heads"),
            DerivedRule::Bound("head_dim"),
        ]),
    },
    // `attn_output_gate` (the Qwen3.5 family) doubles q_proj's output: per head `[query | gate]`.
    DerivedDim {
        name: "q_gate_dim",
        stage: DeriveStage::Implicit,
        rule: DerivedRule::Gated {
            flag: "attn_output_gate",
            on: &DerivedRule::Mul(&[DerivedRule::Lit(2), DerivedRule::Bound("attn_q_dim")]),
            off: &DerivedRule::Bound("attn_q_dim"),
        },
    },
    DerivedDim {
        name: "kv_dim",
        stage: DeriveStage::Implicit,
        rule: DerivedRule::Mul(&[
            DerivedRule::Bound("num_key_value_heads"),
            DerivedRule::Bound("head_dim"),
        ]),
    },
    DerivedDim {
        name: "gdn_value_dim",
        stage: DeriveStage::Implicit,
        rule: DerivedRule::Mul(&[
            DerivedRule::Bound("linear_num_value_heads"),
            DerivedRule::Bound("linear_value_head_dim"),
        ]),
    },
    // Gated-DeltaNet conv channels: q and k at key width, then v at value width (`in_proj_qkv`'s
    // output, `conv1d`'s channels).
    DerivedDim {
        name: "gdn_conv_dim",
        stage: DeriveStage::Implicit,
        rule: DerivedRule::Add(&[
            DerivedRule::Mul(&[
                DerivedRule::Lit(2),
                DerivedRule::Bound("linear_num_key_heads"),
                DerivedRule::Bound("linear_key_head_dim"),
            ]),
            DerivedRule::Bound("gdn_value_dim"),
        ]),
    },
    // Multi-head latent attention (the DeepSeek family). q_proj's (or q_b_proj's, under Q-LoRA)
    // output: per head, the no-rope and rope parts of a query.
    DerivedDim {
        name: "q_proj_out",
        stage: DeriveStage::Implicit,
        rule: DerivedRule::Mul(&[
            DerivedRule::Bound("num_attention_heads"),
            DerivedRule::Add(&[
                DerivedRule::Bound("qk_nope_head_dim"),
                DerivedRule::Bound("qk_rope_head_dim"),
            ]),
        ]),
    },
    // kv_b_proj's decompressed output: per head, a key's no-rope part and a value.
    DerivedDim {
        name: "kv_lora_out",
        stage: DeriveStage::Implicit,
        rule: DerivedRule::Mul(&[
            DerivedRule::Bound("num_attention_heads"),
            DerivedRule::Add(&[
                DerivedRule::Bound("qk_nope_head_dim"),
                DerivedRule::Bound("v_head_dim"),
            ]),
        ]),
    },
    // o_proj's input: every head's value.
    DerivedDim {
        name: "attn_out",
        stage: DeriveStage::Implicit,
        rule: DerivedRule::Mul(&[
            DerivedRule::Bound("num_attention_heads"),
            DerivedRule::Bound("v_head_dim"),
        ]),
    },
    // kv_a_proj_with_mqa's output: the compressed KV, then the one shared rope key.
    DerivedDim {
        name: "kv_a_proj_out",
        stage: DeriveStage::Implicit,
        rule: DerivedRule::Add(&[
            DerivedRule::Bound("kv_lora_rank"),
            DerivedRule::Bound("qk_rope_head_dim"),
        ]),
    },
    // The global attention class's projection widths (Gemma 4's dual-class attention):
    // q_proj_global's output, and the one shared global K/V projection's.
    DerivedDim {
        name: "q_global_dim",
        stage: DeriveStage::AfterDefaults,
        rule: DerivedRule::Mul(&[
            DerivedRule::Bound("num_attention_heads"),
            DerivedRule::Bound("global_head_dim"),
        ]),
    },
    DerivedDim {
        name: "k_global_dim",
        stage: DeriveStage::AfterDefaults,
        rule: DerivedRule::Mul(&[
            DerivedRule::Bound("num_global_key_value_heads"),
            DerivedRule::Bound("global_head_dim"),
        ]),
    },
];

/// Derive into `bounds` every [`DERIVED_DIMS`] entry of `stage` that no config states and whose
/// inputs `bounds` has, each into `derived` as the dim its rule derives it from. Explicit values
/// always win.
pub fn derive_dims(
    bounds: &mut std::collections::BTreeMap<String, u64>,
    stage: DeriveStage,
    derived: &mut std::collections::BTreeMap<String, Dim>,
) {
    for d in DERIVED_DIMS.iter().filter(|d| d.stage == stage) {
        if !bounds.contains_key(d.name)
            && let Some(v) = d.rule.eval(bounds)
        {
            derived.insert(d.name.to_string(), d.rule.dim(bounds));
            bounds.insert(d.name.to_string(), v);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every bound a rule reads, nested rules included.
    fn reads(rule: &DerivedRule, out: &mut Vec<&'static str>) {
        match rule {
            DerivedRule::Bound(name) => out.push(name),
            DerivedRule::Lit(_) => {}
            DerivedRule::Mul(rules) | DerivedRule::Add(rules) => {
                rules.iter().for_each(|r| reads(r, out));
            }
            DerivedRule::Gated { flag, on, off } => {
                out.push(flag);
                reads(on, out);
                reads(off, out);
            }
        }
    }

    /// The table's order is load-bearing: a rule reading another derived dim finds it derived only
    /// if that one comes EARLIER, at the same or an earlier stage. So a reordering that would
    /// silently change a value fails here.
    #[test]
    fn a_derived_dim_reads_only_derived_dims_above_it() {
        let rank = |s: DeriveStage| match s {
            DeriveStage::Implicit => 0,
            DeriveStage::AfterDefaults => 1,
        };
        for (i, d) in DERIVED_DIMS.iter().enumerate() {
            let mut names = Vec::new();
            reads(&d.rule, &mut names);
            for name in names {
                if let Some(j) = DERIVED_DIMS.iter().position(|e| e.name == name) {
                    assert!(
                        j < i,
                        "`{}` reads `{name}`, which the table derives after it",
                        d.name
                    );
                    assert!(
                        rank(DERIVED_DIMS[j].stage) <= rank(d.stage),
                        "`{}` reads `{name}`, which is derived at a later stage",
                        d.name
                    );
                }
            }
        }
    }
}
