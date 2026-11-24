//! P5, newtypes for every quantity.

use clippy_utils::diagnostics::span_lint_and_help;
use rustc_ast::{Expr, ExprKind, FieldDef, ItemKind, TyKind, UnOp};
use rustc_lint::{EarlyContext, EarlyLintPass};
use rustc_session::{declare_lint, impl_lint_pass};
use rustc_span::{Span, Symbol};
use std::collections::HashSet;

declare_lint! {
    /// P5: no raw scalars. A float stored as bits in a `u32` or `u64` field is a raw scalar;
    /// a newtype that compares its bits inside carries the float and its `Eq`.
    pub P5_FLOAT_BITS_FIELD,
    Forbid,
    "a struct field that stores a float as bits in a u32 or u64"
}

pub static LINTS: &[&rustc_lint::Lint] = &[P5_FLOAT_BITS_FIELD];

/// The `*_bits` integer fields a crate defines, and the names it converts to or from a float.
/// A field is a float stored as bits when both hold: the name alone does not decide, since a
/// raw hardware word or a bit width is named `*_bits` too and never meets a float.
#[derive(Default)]
pub struct P5 {
    fields: Vec<(Span, Symbol)>,
    floats: HashSet<Symbol>,
}

impl_lint_pass!(P5 => [P5_FLOAT_BITS_FIELD]);

/// Whether a field named `name`, of a type spelled as the one-segment path `ty`, can hold a
/// float's bits.
pub fn bits_field(name: &str, ty: &str) -> bool {
    let stem = name.strip_suffix("_bits").unwrap_or_default();
    !stem.is_empty() && matches!(ty, "u32" | "u64")
}

/// Whether a path, by its segments, names a float's bit conversion: `f32::from_bits` and the
/// like, or `to_bits` called as a method or through its type.
pub fn float_conversion(segments: &[&str]) -> bool {
    match segments {
        [ty, f] => matches!(*ty, "f32" | "f64") && matches!(*f, "from_bits" | "to_bits"),
        _ => false,
    }
}

/// The field or binding name an expression reads, through derefs, references and parentheses.
fn read_name(expr: &Expr) -> Option<Symbol> {
    match &expr.kind {
        ExprKind::Field(_, ident) => Some(ident.name),
        ExprKind::Path(None, path) if path.segments.len() == 1 => Some(path.segments[0].ident.name),
        ExprKind::Unary(UnOp::Deref, inner)
        | ExprKind::AddrOf(_, _, inner)
        | ExprKind::Paren(inner) => read_name(inner),
        _ => None,
    }
}

/// Whether an expression yields a float's bits: a `to_bits()` call, as a method or by path.
fn to_bits(expr: &Expr) -> bool {
    match &expr.kind {
        ExprKind::MethodCall(call) => call.seg.ident.name.as_str() == "to_bits",
        ExprKind::Call(func, _) => conversion_path(func),
        ExprKind::Paren(inner) => to_bits(inner),
        _ => false,
    }
}

fn conversion_path(func: &Expr) -> bool {
    let ExprKind::Path(None, path) = &func.kind else {
        return false;
    };
    let segments: Vec<&str> = path
        .segments
        .iter()
        .map(|s| s.ident.name.as_str())
        .collect();
    float_conversion(&segments)
}

impl P5 {
    fn field(&mut self, field: &FieldDef) {
        let Some(name) = field.ident else {
            return;
        };
        let TyKind::Path(None, path) = &field.ty.kind else {
            return;
        };
        let [segment] = path.segments.as_slice() else {
            return;
        };
        if segment.args.is_none() && bits_field(name.as_str(), segment.ident.as_str()) {
            self.fields.push((field.span, name.name));
        }
    }
}

/// The fields of every struct, union and enum variant; never a local or a parameter.
impl EarlyLintPass for P5 {
    fn check_item(&mut self, _: &EarlyContext<'_>, item: &rustc_ast::Item) {
        if let ItemKind::Struct(_, _, data) | ItemKind::Union(_, _, data) = &item.kind {
            data.fields().iter().for_each(|field| self.field(field));
        }
    }

    fn check_variant(&mut self, _: &EarlyContext<'_>, variant: &rustc_ast::Variant) {
        variant
            .data
            .fields()
            .iter()
            .for_each(|field| self.field(field));
    }

    fn check_expr(&mut self, _: &EarlyContext<'_>, expr: &Expr) {
        match &expr.kind {
            // A float read out of a name: `f32::from_bits(self.x_bits)`.
            ExprKind::Call(func, args) if conversion_path(func) => {
                self.floats.extend(args.iter().filter_map(|a| read_name(a)));
            }
            // A float written into one: `Self { x_bits: x.to_bits() }`, `self.x_bits = x.to_bits()`.
            ExprKind::Struct(literal) => {
                let written = literal
                    .fields
                    .iter()
                    .filter(|f| to_bits(&f.expr))
                    .map(|f| f.ident.name);
                self.floats.extend(written);
            }
            ExprKind::Assign(lhs, rhs, _) if to_bits(rhs) => self.floats.extend(read_name(lhs)),
            _ => {}
        }
    }

    fn check_crate_post(&mut self, cx: &EarlyContext<'_>, _: &rustc_ast::Crate) {
        for (span, name) in self
            .fields
            .iter()
            .filter(|(_, name)| self.floats.contains(name))
        {
            span_lint_and_help(
                cx,
                P5_FLOAT_BITS_FIELD,
                *span,
                format!(
                    "P5: `{name}` stores a float as bits, a raw scalar; use a newtype that compares its bits inside"
                ),
                None,
                "wrap it in a newtype that implements Eq over its bits internally",
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_bits_field_of_u32_or_u64_can_hold_a_float() {
        assert!(bits_field("eps_bits", "u32"));
        assert!(bits_field("theta_bits", "u64"));
        for (name, ty) in [
            ("count_bits", "u8"),
            ("eps_bits", "f32"),
            ("eps", "u32"),
            ("bits", "u32"),
            ("_bits", "u32"),
            ("bitset", "u64"),
            ("eps_bits", "U32"),
        ] {
            assert!(!bits_field(name, ty), "{name}: {ty}");
        }
    }

    #[test]
    fn only_a_float_types_bit_conversions_count() {
        for path in [
            ["f32", "from_bits"],
            ["f64", "from_bits"],
            ["f32", "to_bits"],
            ["f64", "to_bits"],
        ] {
            assert!(float_conversion(&path), "{path:?}");
        }
        assert!(!float_conversion(&["u32", "from_bits"]));
        assert!(!float_conversion(&["from_bits"]));
        assert!(!float_conversion(&["half", "f16", "from_bits"]));
        assert!(!float_conversion(&["f32", "from_le_bytes"]));
    }
}
