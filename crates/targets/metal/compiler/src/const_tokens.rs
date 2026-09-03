//! Generic const-constructor emission: any `serde::Serialize` value →
//! the tokens of a `const`/`static` expression constructing it.
//!
//! This is the ONE serializer for the whole metal tape type tree. It
//! walks serde's data model (which carries struct/variant/field names),
//! so a new kernel or binding variant needs ZERO edits here — the
//! `#[derive(Serialize)]` on the tape types keeps emission in lockstep
//! with the definitions. The only per-type fact is which module a type
//! lives in ([`module_of`]), a const table.
//!
//! Representation notes:
//! - every sequence is emitted as `&[..]` — the tape wire types hold
//!   `&'static [T]` exclusively (no `Vec` survives into statics);
//! - integers keep their width (`3u32`, `0u8`) so struct fields
//!   type-check without inference context;
//! - floats are emitted as `from_bits(0x..)` so the baked value is
//!   BIT-EXACT (the tape's `ConstantValue` is already bit-packed, but
//!   this keeps any future float field honest too).

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use serde::ser::{self, Serialize};
use std::fmt;

/// ⭐ THE MODULE ALIASES THE EMITTED TAPE RESOLVES THROUGH, DEFINED ONCE.
///
/// The bake emits on the order of 1.5M struct literals per arch. Spelling
/// `::scratchy_target_metal::tape::lowered::` on every one of them cost 38 characters and ~8
/// tokens EACH — tens of megabytes of the generated file, and a name-resolution walk per
/// occurrence, all of it inside rustc's single-threaded front end. Through a 4-character alias
/// the same literal is 2 tokens.
///
/// Emitted at the top of the same block as the tape (see `metal_static_tape`), so the aliases
/// and the paths [`module_of`] hands out are defined in ONE place and cannot drift apart.
pub(crate) fn alias_preamble() -> TokenStream {
    quote! {
        use ::scratchy_target_metal::tape::lowered as __tl;
        use ::scratchy_target_metal::tape::constants as __tc;
        use ::scratchy_target_metal::tape::ids as __ti;
    }
}

/// The module each tape type lives in — the single per-type fact the
/// emitter needs. A type missing here is a macro-time error naming it.
///
/// Returns an ALIAS, not the absolute path — see [`alias_preamble`]. The alias is in scope only
/// inside the block that preamble opens, which is where every emitted value goes.
fn module_of(ty: &str) -> Result<TokenStream, Error> {
    let m = match ty {
        "KernelId" | "Binding" | "DispatchShape" | "MScaling" | "MScaleAxis" | "RuntimeGate"
        | "LoweredCommand" | "GatedCommand" | "GemmDims" | "WeightLocator" | "WeightTensor"
        | "WeightBundleKind" | "RuntimeBindingKind" | "ActivationWidth" | "LoweredMetalTape"
        | "ClassedTape" | "CapPatch" | "ScratchPatch" | "ScratchField" | "GenClass"
        | "PatchTarget" | "TapeLoop" => {
            quote!(__tl)
        }
        "ConstantValue" | "ConstantType" | "ConstSlot" => {
            quote!(__tc)
        }
        "BucketM" | "LayerId" | "ArenaSlotIdx" | "PhysicalBlockIdx" | "LogicalBlockIdx"
        | "SlotInBlock" | "SeqIdx" | "QTokenIdx" | "NumTokens" | "BindingIdx" | "HeadDim"
        | "NumQHeads" | "NumKvHeads" | "RotDim" | "RopePairOff" | "BlockSize"
        | "BlocksPerChunk" | "MaxBlocksPerSeq" | "QSize" | "IntermediateSize" | "HiddenSize"
        | "KDim" | "NDim" | "SplitK" | "AttnDebugMode" | "AttnWindow" | "KDimI32" | "NDimI32"
        | "MDimI32" | "KPartitionSizeI32" | "AttnScale" | "RmsNormEps" => {
            quote!(__ti)
        }
        other => return Err(Error(format!("const_tokens: unmapped type `{other}`"))),
    };
    Ok(m)
}

/// Emit the const-constructor tokens for `value`.
pub fn const_tokens<T: Serialize>(value: &T) -> Result<TokenStream, Error> {
    value.serialize(Emitter)
}

#[derive(Debug)]
pub struct Error(pub String);

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}
impl std::error::Error for Error {}
impl ser::Error for Error {
    fn custom<T: fmt::Display>(msg: T) -> Self {
        Error(msg.to_string())
    }
}

struct Emitter;

macro_rules! emit_int {
    ($fn:ident, $ty:ty, $suffix:literal) => {
        fn $fn(self, v: $ty) -> Result<TokenStream, Error> {
            let lit = proc_macro2::Literal::from_str(&format!(concat!("{}", $suffix), v))
                .expect("int literal");
            Ok(quote!(#lit))
        }
    };
}
use std::str::FromStr;

impl ser::Serializer for Emitter {
    type Ok = TokenStream;
    type Error = Error;
    type SerializeSeq = SeqEmit;
    type SerializeTuple = TupleEmit;
    type SerializeTupleStruct = TupleStructEmit;
    type SerializeTupleVariant = TupleVariantEmit;
    type SerializeMap = ser::Impossible<TokenStream, Error>;
    type SerializeStruct = StructEmit;
    type SerializeStructVariant = StructVariantEmit;

    fn serialize_bool(self, v: bool) -> Result<TokenStream, Error> {
        Ok(quote!(#v))
    }
    emit_int!(serialize_i8, i8, "i8");
    emit_int!(serialize_i16, i16, "i16");
    emit_int!(serialize_i32, i32, "i32");
    emit_int!(serialize_i64, i64, "i64");
    emit_int!(serialize_u8, u8, "u8");
    emit_int!(serialize_u16, u16, "u16");
    emit_int!(serialize_u32, u32, "u32");
    emit_int!(serialize_u64, u64, "u64");
    fn serialize_f32(self, v: f32) -> Result<TokenStream, Error> {
        let bits = v.to_bits();
        Ok(quote!(f32::from_bits(#bits)))
    }
    fn serialize_f64(self, v: f64) -> Result<TokenStream, Error> {
        let bits = v.to_bits();
        Ok(quote!(f64::from_bits(#bits)))
    }
    fn serialize_char(self, v: char) -> Result<TokenStream, Error> {
        Ok(quote!(#v))
    }
    fn serialize_str(self, v: &str) -> Result<TokenStream, Error> {
        Ok(quote!(#v))
    }
    fn serialize_bytes(self, _: &[u8]) -> Result<TokenStream, Error> {
        Err(Error("const_tokens: bytes unsupported".into()))
    }
    fn serialize_none(self) -> Result<TokenStream, Error> {
        Ok(quote!(None))
    }
    fn serialize_some<T: Serialize + ?Sized>(self, v: &T) -> Result<TokenStream, Error> {
        let inner = v.serialize(Emitter)?;
        Ok(quote!(Some(#inner)))
    }
    fn serialize_unit(self) -> Result<TokenStream, Error> {
        Ok(quote!(()))
    }
    fn serialize_unit_struct(self, name: &'static str) -> Result<TokenStream, Error> {
        let m = module_of(name)?;
        let id = format_ident!("{name}");
        Ok(quote!(#m::#id))
    }
    fn serialize_unit_variant(
        self,
        name: &'static str,
        _idx: u32,
        variant: &'static str,
    ) -> Result<TokenStream, Error> {
        let m = module_of(name)?;
        let ty = format_ident!("{name}");
        let var = format_ident!("{variant}");
        Ok(quote!(#m::#ty::#var))
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        v: &T,
    ) -> Result<TokenStream, Error> {
        let m = module_of(name)?;
        let ty = format_ident!("{name}");
        let inner = v.serialize(Emitter)?;
        Ok(quote!(#m::#ty(#inner)))
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        _idx: u32,
        variant: &'static str,
        v: &T,
    ) -> Result<TokenStream, Error> {
        let m = module_of(name)?;
        let ty = format_ident!("{name}");
        let var = format_ident!("{variant}");
        let inner = v.serialize(Emitter)?;
        Ok(quote!(#m::#ty::#var(#inner)))
    }
    fn serialize_seq(self, _len: Option<usize>) -> Result<SeqEmit, Error> {
        Ok(SeqEmit(Vec::new()))
    }
    fn serialize_tuple(self, _len: usize) -> Result<TupleEmit, Error> {
        Ok(TupleEmit(Vec::new()))
    }
    fn serialize_tuple_struct(
        self,
        name: &'static str,
        _len: usize,
    ) -> Result<TupleStructEmit, Error> {
        let m = module_of(name)?;
        let ty = format_ident!("{name}");
        Ok(TupleStructEmit(quote!(#m::#ty), Vec::new()))
    }
    fn serialize_tuple_variant(
        self,
        name: &'static str,
        _idx: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<TupleVariantEmit, Error> {
        let m = module_of(name)?;
        let ty = format_ident!("{name}");
        let var = format_ident!("{variant}");
        Ok(TupleVariantEmit(quote!(#m::#ty::#var), Vec::new()))
    }
    fn serialize_map(
        self,
        _len: Option<usize>,
    ) -> Result<ser::Impossible<TokenStream, Error>, Error> {
        Err(Error("const_tokens: maps unsupported".into()))
    }
    fn serialize_struct(self, name: &'static str, _len: usize) -> Result<StructEmit, Error> {
        let m = module_of(name)?;
        let ty = format_ident!("{name}");
        Ok(StructEmit(quote!(#m::#ty), Vec::new()))
    }
    fn serialize_struct_variant(
        self,
        name: &'static str,
        _idx: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<StructVariantEmit, Error> {
        let m = module_of(name)?;
        let ty = format_ident!("{name}");
        let var = format_ident!("{variant}");
        Ok(StructVariantEmit(quote!(#m::#ty::#var), Vec::new()))
    }
}

pub struct SeqEmit(Vec<TokenStream>);
impl ser::SerializeSeq for SeqEmit {
    type Ok = TokenStream;
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> {
        self.0.push(v.serialize(Emitter)?);
        Ok(())
    }
    fn end(self) -> Result<TokenStream, Error> {
        let items = self.0;
        // Every tape sequence is a `&'static [T]`.
        Ok(quote!(&[ #(#items),* ]))
    }
}

pub struct TupleEmit(Vec<TokenStream>);
impl ser::SerializeTuple for TupleEmit {
    type Ok = TokenStream;
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> {
        self.0.push(v.serialize(Emitter)?);
        Ok(())
    }
    fn end(self) -> Result<TokenStream, Error> {
        let items = self.0;
        Ok(quote!(( #(#items),* )))
    }
}

pub struct TupleStructEmit(TokenStream, Vec<TokenStream>);
impl ser::SerializeTupleStruct for TupleStructEmit {
    type Ok = TokenStream;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> {
        self.1.push(v.serialize(Emitter)?);
        Ok(())
    }
    fn end(self) -> Result<TokenStream, Error> {
        let (path, items) = (self.0, self.1);
        Ok(quote!(#path( #(#items),* )))
    }
}

pub struct TupleVariantEmit(TokenStream, Vec<TokenStream>);
impl ser::SerializeTupleVariant for TupleVariantEmit {
    type Ok = TokenStream;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> {
        self.1.push(v.serialize(Emitter)?);
        Ok(())
    }
    fn end(self) -> Result<TokenStream, Error> {
        let (path, items) = (self.0, self.1);
        Ok(quote!(#path( #(#items),* )))
    }
}

pub struct StructEmit(TokenStream, Vec<(proc_macro2::Ident, TokenStream)>);
impl ser::SerializeStruct for StructEmit {
    type Ok = TokenStream;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        v: &T,
    ) -> Result<(), Error> {
        self.1.push((format_ident!("{key}"), v.serialize(Emitter)?));
        Ok(())
    }
    fn end(self) -> Result<TokenStream, Error> {
        let (path, fields) = (self.0, self.1);
        let (names, vals): (Vec<_>, Vec<_>) = fields.into_iter().unzip();
        Ok(quote!(#path { #(#names: #vals),* }))
    }
}

pub struct StructVariantEmit(TokenStream, Vec<(proc_macro2::Ident, TokenStream)>);
impl ser::SerializeStructVariant for StructVariantEmit {
    type Ok = TokenStream;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        v: &T,
    ) -> Result<(), Error> {
        self.1.push((format_ident!("{key}"), v.serialize(Emitter)?));
        Ok(())
    }
    fn end(self) -> Result<TokenStream, Error> {
        let (path, fields) = (self.0, self.1);
        let (names, vals): (Vec<_>, Vec<_>) = fields.into_iter().unzip();
        Ok(quote!(#path { #(#names: #vals),* }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_target_metal::tape::constants::{ConstantType, ConstantValue};
    use scratchy_target_metal::tape::lowered::{
        Binding, DispatchShape, GatedCommand, KernelId, LoweredCommand, RuntimeGate,
    };

    /// A command exercising every emission form: unit variant, struct
    /// variant, newtype, tuple, slice, option, str, mixed int widths.
    fn sample() -> GatedCommand {
        GatedCommand {
            command: LoweredCommand {
                kernel: KernelId::Embed,
                library: "default_lib",
                function: "metal_embed_f16",
                constants: scratchy_target_metal::tape::lowered::baked(vec![ConstantValue {
                    index: 3u16,
                    bits: 1.5f32.to_bits(),
                    ty: ConstantType::Float,
                }]),
                dispatch: DispatchShape {
                    threadgroups: (8, 1, 1),
                    threads_per_threadgroup: (256, 1, 1),
                    m_scaling: None,
                },
                bindings: scratchy_target_metal::tape::lowered::baked(vec![
                    Binding::ArenaSlot {
                        slot: 2,
                        binding_index: 0,
                    },
                    Binding::Source {
                        ix: 7,
                        binding_index: 1,
                    },
                ]),
                gemm_dims: None,
            },
            gate: Some(RuntimeGate::OnlyIfTurboquant),
        }
    }

    #[test]
    fn emits_parseable_constructor_with_exact_values() {
        let toks = const_tokens(&sample()).expect("emit");
        // Must parse as a Rust expression…
        let expr: syn::Expr = syn::parse2(toks.clone()).expect("parse");
        let s = quote!(#expr).to_string();
        // …naming the runtime's own types at absolute paths, with widths
        // and bit-exact floats.
        for needle in [
            ":: scratchy_target_metal :: tape :: lowered :: KernelId :: Embed",
            "Binding :: ArenaSlot { slot : 2u32 , binding_index : 0u8 }",
            "Binding :: Source { ix : 7u32 , binding_index : 1u8 }",
            "index : 3u16",
            "RuntimeGate :: OnlyIfTurboquant",
            "\"metal_embed_f16\"",
        ] {
            assert!(s.contains(needle), "missing `{needle}` in:\n{s}");
        }
        assert!(
            s.contains(&format!("bits : {}u32", 1.5f32.to_bits())),
            "float bits not exact:\n{s}"
        );
    }
}
