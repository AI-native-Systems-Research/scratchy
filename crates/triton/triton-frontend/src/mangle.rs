//! A port of Triton's function-name mangling.
//!
//! When a `@triton.jit` function calls another, Triton does not inline it: it generates a
//! `tt.func private` whose NAME encodes the callee's fully-qualified Python name plus the
//! mangled type of every argument, then emits a `tt.call`. The mangled name is the cache
//! key, so a helper called at two different constexpr shapes becomes two functions.
//!
//! Measured in the goldens:
//!
//! ```text
//! tt.call @triton.language.standard.zeros__Tc64_c128T_cfp16() : () -> tensor<64x128xf16>
//! tt.call @triton.language.standard.zeros__Tc64_c64T_cfp16()  : () -> tensor<64x64xf16>
//! tt.call @triton.language.standard.sum__fp16S64_64S_c1_cFalse_cNone(%x)
//! tt.call @triton.language.standard.max__fp16S64_64S_c1_cFalse_cTrue_cFalse(%x)
//! tt.call @triton.language.standard._elementwise_max__fp32_fp32(%a, %b)
//! ```
//!
//! ## The rules, each from a definition
//!
//! `mangle_fn` (`compiler/code_generator.py:32`):
//!
//! ```text
//! mangled = '_'.join(ty.mangle() for ty in arg_tys)
//! mangled = mangled.replace("'", '_sq_').replace('[', '_').replace(']', '_')
//! return f'{name}__{mangled}'
//! ```
//!
//! where `name` is `get_full_name(fn)` = `f"{fn.__module__}.{fn.__qualname__}"`
//! (`runtime/jit.py:461`), and each type mangles as:
//!
//! | type | rule | source | example |
//! |---|---|---|---|
//! | integer `dtype` | `i`/`u` + bitwidth | `core.py:645` | `i32`, `u8` |
//! | float `dtype` | `str(self)` | `core.py:645` | `fp16`, `fp32` |
//! | void `dtype` | `V` | `core.py:645` | `V` |
//! | `pointer_type` | `P` + pointee | `core.py:703` | `Pfp16` |
//! | `block_type` | elem + `S` + dims joined `_` + `S` | `core.py:755` | `fp16S64_64S` |
//! | `constexpr_type` | `c` + (value's own mangle, else `repr(value)`) | `core.py:196` | `c64`, `cfp16`, `cFalse`, `cNone` |
//! | `tuple_type` | `T` + members joined `_` + `T` | `core.py:796` | `Tc64_c128T` |
//!
//! The `repr` case is why a Python bool mangles `cTrue`/`cFalse` and `None` mangles `cNone`
//! rather than anything C-like: it is literally `repr()`.

use crate::semantic::Val;
use crate::ttir::{FloatKind, Signedness, Type};

/// The mangled spelling of a dtype, as `dtype.mangle()` produces it.
pub fn mangle_type(t: &Type) -> String {
    match t {
        Type::Int(bits, sign) => {
            let prefix = if *sign == Signedness::Unsigned { 'u' } else { 'i' };
            format!("{prefix}{bits}")
        }
        // `str(dtype)` for a float is Triton's dtype NAME (`fp16`), not MLIR's (`f16`).
        Type::Float(k) => match k {
            FloatKind::F16 => "fp16".to_string(),
            FloatKind::BF16 => "bf16".to_string(),
            FloatKind::F32 => "fp32".to_string(),
            FloatKind::F64 => "fp64".to_string(),
            FloatKind::F8E4M3FN => "fp8e4nv".to_string(),
            FloatKind::F8E5M2 => "fp8e5".to_string(),
        },
        Type::Ptr(e, _) => format!("P{}", mangle_type(e)),
        Type::Tensor(shape, elem) => {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("{}S{}S", mangle_type(elem), dims.join("_"))
        }
        // `tensor_descriptor_base_type.mangle` (`core.py:1387`) is `TD` + the BLOCK type's
        // mangle. Measured in the attention golden:
        // `_attn_fwd_inner__..._TDfp16S64_128S_TDfp16S64_128S_TDfp16S64_64S_...`
        Type::TensorDesc(shape, elem) => {
            let dims: Vec<String> = shape.iter().map(|d| d.to_string()).collect();
            format!("TD{}S{}S", mangle_type(elem), dims.join("_"))
        }
        Type::Void => "V".to_string(),
    }
}

/// CPython's `repr(float)`, INCLUDING ITS EXPONENT-FORM SWITCH.
///
/// # Why this is not `format!("{f}")`.
///
/// Rust's `Display` for f64 never uses exponent notation: it prints `1e-05` as `0.00001`.
/// CPython's `float_repr` goes through `PyOS_double_to_string` in repr mode, which prints the
/// SHORTEST round-tripping digits and then chooses the form by where the decimal point falls:
/// fixed while `-4 < decpt <= 16`, exponential otherwise, with the exponent signed and padded
/// to at least two digits.
///
/// It matters because a float constexpr is mangled with `repr` INTO A SYMBOL NAME, so a
/// different spelling is a different function. Measured on `decoder_block.py`, whose
/// `rms_norm_eps` is `1e-05`:
///
/// ```text
/// golden: ..._c64_c64_c1e-05_c0.0078125_c0.011271055_c0.22
/// ours:   ..._c64_c64_c0.00001_c0.0078125_c0.011271055_c0.22   <- before this function
/// ```
///
/// and the `tt.call` that names it disagreed too, so the diff reported a missing function, an
/// extra function AND a callee mismatch from one formatting rule. `1.0`, `0.22` and
/// `0.0078125` all agreed, which is exactly why nothing caught it earlier: every float
/// constexpr in the tree until now was inside the fixed-notation window.
fn py_float_repr(f: f64) -> String {
    if f.is_nan() {
        return "nan".to_string();
    }
    if f.is_infinite() {
        return if f > 0.0 { "inf" } else { "-inf" }.to_string();
    }
    if f == 0.0 {
        // Python keeps the sign of negative zero: `repr(-0.0) == '-0.0'`.
        return if f.is_sign_negative() { "-0.0" } else { "0.0" }.to_string();
    }
    let (sign, a) = if f < 0.0 { ("-", -f) } else { ("", f) };
    // Rust's LowerExp gives the shortest round-tripping digits and the exponent, which is the
    // same pair CPython works from -- only the assembly differs.
    let e = format!("{a:e}");
    let (mant, exp) = match e.split_once('e') {
        Some(x) => x,
        None => return format!("{sign}{a}"),
    };
    let exp: i32 = match exp.parse() {
        Ok(x) => x,
        Err(_) => return format!("{sign}{a}"),
    };
    let digits: String = mant.chars().filter(|c| *c != '.').collect();
    // `decpt` is where the decimal point sits relative to the digit string, which is what
    // CPython's `format_float_short` branches on.
    let decpt = exp + 1;
    if decpt > -4 && decpt <= 16 {
        // Fixed notation. An integral value still shows `.0`, which is `Py_DTSF_ADD_DOT_0`.
        if a == a.trunc() {
            return format!("{sign}{a:.1}");
        }
        return format!("{sign}{a}");
    }
    let m = if digits.len() == 1 {
        digits
    } else {
        format!("{}.{}", &digits[..1], &digits[1..])
    };
    let esign = if exp < 0 { '-' } else { '+' };
    format!("{sign}{m}e{esign}{:02}", exp.abs())
}

/// Python's `repr` for the literal kinds a constexpr can hold.
///
/// `repr` is what `constexpr_type.mangle` falls back to, so the spelling has to match
/// Python's exactly -- `True`, not `true`; `None`, not `null`; and a float that is integral
/// still prints a trailing `.0`.
fn py_repr(v: &Val) -> Option<String> {
    Some(match v {
        Val::Bool(b) => if *b { "True" } else { "False" }.to_string(),
        Val::Int(i) => i.to_string(),
        Val::Float(f) => py_float_repr(*f),
        Val::None => "None".to_string(),
        Val::Str(s) => format!("'{s}'"),
        _ => return None,
    })
}

/// The mangled spelling of one call argument.
///
/// A [`Val::Ir`] argument is a runtime value and mangles as its TYPE; everything else is a
/// constexpr and mangles as `c<repr>` (or `T..T` for a sequence).
pub fn mangle_arg(v: &Val, ty_of: &dyn Fn(&Val) -> Option<Type>) -> Option<String> {
    match v {
        // A descriptor mangles ONCE, from its language type, even though it flattens to
        // `1 + 2 * rank` IR arguments -- `mangle_fn` runs over `arg.type`, not over the
        // flattened handles. Conflating the two would produce a symbol with fifteen entries
        // where the oracle has three.
        Val::Ir(_) | Val::Desc { .. } => ty_of(v).map(|t| mangle_type(&t)),
        Val::Dtype(t) => Some(format!("c{}", mangle_type(t))),
        Val::Seq(items) => {
            let mut parts = Vec::with_capacity(items.len());
            for it in items {
                parts.push(mangle_arg(it, ty_of)?);
            }
            Some(format!("T{}T", parts.join("_")))
        }
        other => py_repr(other).map(|r| format!("c{r}")),
    }
}

/// `mangle_fn`: the full symbol name for a generated callee.
pub fn mangle_fn(full_name: &str, arg_mangles: &[String]) -> String {
    let joined = arg_mangles.join("_");
    // `code_generator.py:35-37`: quotes and brackets are not legal in an LLVM identifier.
    let joined = joined.replace('\'', "_sq_").replace(['[', ']'], "_");
    format!("{full_name}__{joined}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::rc::Rc;

    fn no_types(_: &Val) -> Option<Type> {
        None
    }

    #[test]
    fn dtype_mangles_match_tritons_names() {
        assert_eq!(mangle_type(&Type::f16()), "fp16");
        assert_eq!(mangle_type(&Type::f32()), "fp32");
        assert_eq!(mangle_type(&Type::i32()), "i32");
        assert_eq!(
            mangle_type(&Type::Int(8, Signedness::Unsigned)),
            "u8"
        );
        assert_eq!(mangle_type(&Type::ptr(Type::f16())), "Pfp16");
        assert_eq!(
            mangle_type(&Type::Tensor(vec![64, 64], Rc::new(Type::f16()))),
            "fp16S64_64S"
        );
    }

    /// The four symbol names the goldens actually contain, reproduced exactly. These are the
    /// test that matters: they come from the oracle, not from reading the mangling code.
    #[test]
    fn the_golden_symbol_names_are_reproduced() {
        let zeros_128 = mangle_fn(
            "triton.language.standard.zeros",
            &[
                mangle_arg(
                    &Val::Seq(vec![Val::Int(64), Val::Int(128)]),
                    &no_types,
                )
                .unwrap(),
                mangle_arg(&Val::Dtype(Type::f16()), &no_types).unwrap(),
            ],
        );
        assert_eq!(zeros_128, "triton.language.standard.zeros__Tc64_c128T_cfp16");

        let zeros_64 = mangle_fn(
            "triton.language.standard.zeros",
            &[
                mangle_arg(&Val::Seq(vec![Val::Int(64), Val::Int(64)]), &no_types).unwrap(),
                mangle_arg(&Val::Dtype(Type::f16()), &no_types).unwrap(),
            ],
        );
        assert_eq!(zeros_64, "triton.language.standard.zeros__Tc64_c64T_cfp16");

        // sum(input: tensor<64x64xf16>, axis=1, keep_dims=False, dtype=None)
        let tensor = Type::Tensor(vec![64, 64], Rc::new(Type::f16()));
        let sum = mangle_fn(
            "triton.language.standard.sum",
            &[
                mangle_type(&tensor),
                "c1".to_string(),
                mangle_arg(&Val::Bool(false), &no_types).unwrap(),
                mangle_arg(&Val::None, &no_types).unwrap(),
            ],
        );
        assert_eq!(
            sum,
            "triton.language.standard.sum__fp16S64_64S_c1_cFalse_cNone"
        );

        let emax = mangle_fn(
            "triton.language.standard._elementwise_max",
            &[mangle_type(&Type::f32()), mangle_type(&Type::f32())],
        );
        assert_eq!(
            emax,
            "triton.language.standard._elementwise_max__fp32_fp32"
        );
    }

    #[test]
    fn repr_spellings_are_pythons() {
        assert_eq!(mangle_arg(&Val::Bool(true), &no_types).unwrap(), "cTrue");
        assert_eq!(mangle_arg(&Val::Bool(false), &no_types).unwrap(), "cFalse");
        assert_eq!(mangle_arg(&Val::None, &no_types).unwrap(), "cNone");
        assert_eq!(mangle_arg(&Val::Int(-3), &no_types).unwrap(), "c-3");
        // An integral float keeps Python's trailing `.0`.
        assert_eq!(mangle_arg(&Val::Float(1.0), &no_types).unwrap(), "c1.0");
        assert_eq!(mangle_arg(&Val::Float(0.5), &no_types).unwrap(), "c0.5");
    }
}
