// SPDX-License-Identifier: Apache-2.0
//! Generic const-constructor emission: any `serde::Serialize` value →
//! the tokens of a `const`/`static` expression constructing it.
//!
//! This is the ONE serializer for every target's static tapes. It walks
//! serde's data model (which carries struct/variant/field names), so a new
//! kernel or binding variant needs ZERO edits here — the
//! `#[derive(Serialize)]` on a target's tape types keeps emission in
//! lockstep with the definitions. The only per-type fact is which module a
//! type lives in, and that is the TARGET's: a const [`ModuleTable`] it
//! passes in, beside the values.
//!
//! Representation notes:
//! - every sequence is emitted as `&[..]` — the tape wire types hold
//!   `&'static [T]` exclusively (no `Vec` survives into statics);
//! - integers keep their width (`3u32`, `0u8`) so struct fields
//!   type-check without inference context;
//! - floats are emitted as `from_bits(0x..)` so the baked value is
//!   BIT-EXACT.

use proc_macro2::TokenStream;
use quote::{format_ident, quote};
use serde::ser::{self, Serialize};
use std::fmt;
use std::str::FromStr;

/// One module of a target's tape types: the alias the emitted values name it by, its absolute
/// path, and the types that live in it.
#[derive(Clone, Copy, Debug)]
pub struct ModuleRow {
    /// The short name every emitted value spells the module with (`__tl`).
    pub alias: &'static str,
    /// The module's absolute path, `::`-separated (`::scratchy_target_metal::tape::lowered`).
    pub path: &'static str,
    /// The type names (serde's struct and enum names) that live in it.
    pub types: &'static [&'static str],
}

/// ⭐ THE TARGET'S ONE FACT: which module each of its tape types lives in, and the aliases the
/// emitted tape resolves through.
///
/// A target bakes on the order of a million struct literals per arch. Spelling the absolute path
/// on every one costs tens of characters and several tokens EACH, all of it inside rustc's
/// single-threaded front end; through a short alias the same literal is two tokens. The aliases
/// ([`Self::preamble`]) and the paths [`const_tokens`] hands out come from the same rows, so they
/// cannot drift apart.
#[derive(Clone, Copy, Debug)]
pub struct ModuleTable {
    pub rows: &'static [ModuleRow],
}

impl ModuleTable {
    /// `use <path> as <alias>;` for every row, in order: emitted at the top of the block that
    /// holds the values, the only place the aliases are in scope.
    pub fn preamble(&self) -> TokenStream {
        let uses = self.rows.iter().map(|row| {
            let path = row
                .path
                .split("::")
                .filter(|seg| !seg.is_empty())
                .map(|seg| {
                    let seg = format_ident!("{seg}");
                    quote!(:: #seg)
                });
            let alias = format_ident!("{}", row.alias);
            quote! { use #(#path)* as #alias; }
        });
        quote! { #(#uses)* }
    }

    /// The alias of the module `ty` lives in. A type missing here is a macro-time error naming
    /// it.
    fn module_of(&self, ty: &str) -> Result<TokenStream, Error> {
        let row = (self.rows.iter())
            .find(|row| row.types.contains(&ty))
            .ok_or_else(|| Error(format!("const_tokens: unmapped type `{ty}`")))?;
        let alias = format_ident!("{}", row.alias);
        Ok(quote!(#alias))
    }
}

/// Emit the const-constructor tokens for `value`, naming each type through `table`'s aliases.
pub fn const_tokens<T: Serialize + ?Sized>(
    table: &ModuleTable,
    value: &T,
) -> Result<TokenStream, Error> {
    value.serialize(Emitter(table))
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

#[derive(Clone, Copy)]
struct Emitter<'t>(&'t ModuleTable);

macro_rules! emit_int {
    ($fn:ident, $ty:ty, $suffix:literal) => {
        fn $fn(self, v: $ty) -> Result<TokenStream, Error> {
            let lit = proc_macro2::Literal::from_str(&format!(concat!("{}", $suffix), v))
                .expect("int literal");
            Ok(quote!(#lit))
        }
    };
}

impl<'t> ser::Serializer for Emitter<'t> {
    type Ok = TokenStream;
    type Error = Error;
    type SerializeSeq = SeqEmit<'t>;
    type SerializeTuple = TupleEmit<'t>;
    type SerializeTupleStruct = TupleStructEmit<'t>;
    type SerializeTupleVariant = TupleVariantEmit<'t>;
    type SerializeMap = ser::Impossible<TokenStream, Error>;
    type SerializeStruct = StructEmit<'t>;
    type SerializeStructVariant = StructVariantEmit<'t>;

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
        let inner = v.serialize(self)?;
        Ok(quote!(Some(#inner)))
    }
    fn serialize_unit(self) -> Result<TokenStream, Error> {
        Ok(quote!(()))
    }
    fn serialize_unit_struct(self, name: &'static str) -> Result<TokenStream, Error> {
        let m = self.0.module_of(name)?;
        let id = format_ident!("{name}");
        Ok(quote!(#m::#id))
    }
    fn serialize_unit_variant(
        self,
        name: &'static str,
        _idx: u32,
        variant: &'static str,
    ) -> Result<TokenStream, Error> {
        let m = self.0.module_of(name)?;
        let ty = format_ident!("{name}");
        let var = format_ident!("{variant}");
        Ok(quote!(#m::#ty::#var))
    }
    fn serialize_newtype_struct<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        v: &T,
    ) -> Result<TokenStream, Error> {
        let m = self.0.module_of(name)?;
        let ty = format_ident!("{name}");
        let inner = v.serialize(self)?;
        Ok(quote!(#m::#ty(#inner)))
    }
    fn serialize_newtype_variant<T: Serialize + ?Sized>(
        self,
        name: &'static str,
        _idx: u32,
        variant: &'static str,
        v: &T,
    ) -> Result<TokenStream, Error> {
        let m = self.0.module_of(name)?;
        let ty = format_ident!("{name}");
        let var = format_ident!("{variant}");
        let inner = v.serialize(self)?;
        Ok(quote!(#m::#ty::#var(#inner)))
    }
    fn serialize_seq(self, _len: Option<usize>) -> Result<SeqEmit<'t>, Error> {
        Ok(SeqEmit(self, Vec::new()))
    }
    fn serialize_tuple(self, _len: usize) -> Result<TupleEmit<'t>, Error> {
        Ok(TupleEmit(self, Vec::new()))
    }
    fn serialize_tuple_struct(
        self,
        name: &'static str,
        _len: usize,
    ) -> Result<TupleStructEmit<'t>, Error> {
        let m = self.0.module_of(name)?;
        let ty = format_ident!("{name}");
        Ok(TupleStructEmit(self, quote!(#m::#ty), Vec::new()))
    }
    fn serialize_tuple_variant(
        self,
        name: &'static str,
        _idx: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<TupleVariantEmit<'t>, Error> {
        let m = self.0.module_of(name)?;
        let ty = format_ident!("{name}");
        let var = format_ident!("{variant}");
        Ok(TupleVariantEmit(self, quote!(#m::#ty::#var), Vec::new()))
    }
    fn serialize_map(
        self,
        _len: Option<usize>,
    ) -> Result<ser::Impossible<TokenStream, Error>, Error> {
        Err(Error("const_tokens: maps unsupported".into()))
    }
    fn serialize_struct(self, name: &'static str, _len: usize) -> Result<StructEmit<'t>, Error> {
        let m = self.0.module_of(name)?;
        let ty = format_ident!("{name}");
        Ok(StructEmit(self, quote!(#m::#ty), Vec::new()))
    }
    fn serialize_struct_variant(
        self,
        name: &'static str,
        _idx: u32,
        variant: &'static str,
        _len: usize,
    ) -> Result<StructVariantEmit<'t>, Error> {
        let m = self.0.module_of(name)?;
        let ty = format_ident!("{name}");
        let var = format_ident!("{variant}");
        Ok(StructVariantEmit(self, quote!(#m::#ty::#var), Vec::new()))
    }
}

pub struct SeqEmit<'t>(Emitter<'t>, Vec<TokenStream>);
impl ser::SerializeSeq for SeqEmit<'_> {
    type Ok = TokenStream;
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> {
        self.1.push(v.serialize(self.0)?);
        Ok(())
    }
    fn end(self) -> Result<TokenStream, Error> {
        let items = self.1;
        // Every tape sequence is a `&'static [T]`.
        Ok(quote!(&[ #(#items),* ]))
    }
}

pub struct TupleEmit<'t>(Emitter<'t>, Vec<TokenStream>);
impl ser::SerializeTuple for TupleEmit<'_> {
    type Ok = TokenStream;
    type Error = Error;
    fn serialize_element<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> {
        self.1.push(v.serialize(self.0)?);
        Ok(())
    }
    fn end(self) -> Result<TokenStream, Error> {
        let items = self.1;
        Ok(quote!(( #(#items),* )))
    }
}

pub struct TupleStructEmit<'t>(Emitter<'t>, TokenStream, Vec<TokenStream>);
impl ser::SerializeTupleStruct for TupleStructEmit<'_> {
    type Ok = TokenStream;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> {
        self.2.push(v.serialize(self.0)?);
        Ok(())
    }
    fn end(self) -> Result<TokenStream, Error> {
        let (path, items) = (self.1, self.2);
        Ok(quote!(#path( #(#items),* )))
    }
}

pub struct TupleVariantEmit<'t>(Emitter<'t>, TokenStream, Vec<TokenStream>);
impl ser::SerializeTupleVariant for TupleVariantEmit<'_> {
    type Ok = TokenStream;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(&mut self, v: &T) -> Result<(), Error> {
        self.2.push(v.serialize(self.0)?);
        Ok(())
    }
    fn end(self) -> Result<TokenStream, Error> {
        let (path, items) = (self.1, self.2);
        Ok(quote!(#path( #(#items),* )))
    }
}

pub struct StructEmit<'t>(
    Emitter<'t>,
    TokenStream,
    Vec<(proc_macro2::Ident, TokenStream)>,
);
impl ser::SerializeStruct for StructEmit<'_> {
    type Ok = TokenStream;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        v: &T,
    ) -> Result<(), Error> {
        self.2.push((format_ident!("{key}"), v.serialize(self.0)?));
        Ok(())
    }
    fn end(self) -> Result<TokenStream, Error> {
        let (path, fields) = (self.1, self.2);
        let (names, vals): (Vec<_>, Vec<_>) = fields.into_iter().unzip();
        Ok(quote!(#path { #(#names: #vals),* }))
    }
}

pub struct StructVariantEmit<'t>(
    Emitter<'t>,
    TokenStream,
    Vec<(proc_macro2::Ident, TokenStream)>,
);
impl ser::SerializeStructVariant for StructVariantEmit<'_> {
    type Ok = TokenStream;
    type Error = Error;
    fn serialize_field<T: Serialize + ?Sized>(
        &mut self,
        key: &'static str,
        v: &T,
    ) -> Result<(), Error> {
        self.2.push((format_ident!("{key}"), v.serialize(self.0)?));
        Ok(())
    }
    fn end(self) -> Result<TokenStream, Error> {
        let (path, fields) = (self.1, self.2);
        let (names, vals): (Vec<_>, Vec<_>) = fields.into_iter().unzip();
        Ok(quote!(#path { #(#names: #vals),* }))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde::Serialize;

    #[derive(Serialize)]
    struct Id(u32);

    #[derive(Serialize)]
    enum Kind {
        Unit,
        Wrapped(Id),
        Fields { at: u64, scale: f32 },
    }

    #[derive(Serialize)]
    struct Record {
        kind: Kind,
        ids: &'static [Id],
        pair: (u8, i16),
        name: &'static str,
        maybe: Option<Id>,
    }

    const TABLE: ModuleTable = ModuleTable {
        rows: &[
            ModuleRow {
                alias: "__a",
                path: "::some_crate::a",
                types: &["Record", "Kind"],
            },
            ModuleRow {
                alias: "__b",
                path: "::some_crate::b",
                types: &["Id"],
            },
        ],
    };

    #[test]
    fn emits_parseable_constructor_through_the_table() {
        let value = Record {
            kind: Kind::Fields { at: 7, scale: 1.5 },
            ids: &[Id(3), Id(4)],
            pair: (1, -2),
            name: "k",
            maybe: Some(Id(9)),
        };
        let toks = const_tokens(&TABLE, &value).expect("emit");
        let expr: syn::Expr = syn::parse2(toks).expect("parse");
        let s = quote!(#expr).to_string();
        for needle in [
            "__a :: Record {",
            "kind : __a :: Kind :: Fields { at : 7u64",
            "ids : & [__b :: Id (3u32) , __b :: Id (4u32)]",
            "pair : (1u8 , - 2i16)",
            "name : \"k\"",
            "maybe : Some (__b :: Id (9u32))",
        ] {
            assert!(s.contains(needle), "missing `{needle}` in:\n{s}");
        }
        assert!(
            s.contains(&format!("f32 :: from_bits ({}u32)", 1.5f32.to_bits())),
            "float bits not exact:\n{s}"
        );
        let unit = const_tokens(&TABLE, &Kind::Unit).expect("emit").to_string();
        assert_eq!(unit, "__a :: Kind :: Unit");
        let wrapped = const_tokens(&TABLE, &Kind::Wrapped(Id(1)))
            .expect("emit")
            .to_string();
        assert_eq!(wrapped, "__a :: Kind :: Wrapped (__b :: Id (1u32))");
    }

    #[test]
    fn an_unmapped_type_is_an_error_naming_it() {
        #[derive(Serialize)]
        struct Stray(u8);
        let err = const_tokens(&TABLE, &Stray(1)).expect_err("unmapped");
        assert_eq!(err.0, "const_tokens: unmapped type `Stray`");
    }

    #[test]
    fn the_preamble_aliases_every_row() {
        let s = TABLE.preamble().to_string();
        assert_eq!(
            s,
            "use :: some_crate :: a as __a ; use :: some_crate :: b as __b ;"
        );
    }
}
