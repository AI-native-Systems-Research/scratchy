//! Metal's const-constructor emission: the shared serializer
//! (`scratchy_const_tokens`) under metal's module table.
//!
//! The walk over serde's data model is shared by every target; the one fact
//! that is metal's is which module each metal tape type lives in, and the
//! aliases the emitted tape resolves through ([`METAL_MODULES`]), a const
//! table handed to the shared serializer. A new tape type needs one entry
//! there; a missing one is a macro-time error naming it.

use proc_macro2::TokenStream;
use scratchy_const_tokens::{ModuleRow, ModuleTable};
use serde::ser::Serialize;

pub use scratchy_const_tokens::Error;

/// ⭐ THE MODULE ALIASES THE EMITTED TAPE RESOLVES THROUGH, DEFINED ONCE.
///
/// The bake emits on the order of 1.5M struct literals per arch. Spelling
/// `::scratchy_target_metal::tape::lowered::` on every one of them cost 38 characters and ~8
/// tokens EACH — tens of megabytes of the generated file, and a name-resolution walk per
/// occurrence, all of it inside rustc's single-threaded front end. Through a 4-character alias
/// the same literal is 2 tokens.
///
/// The preamble ([`alias_preamble`]) is emitted at the top of the same block as the tape (see
/// `static_tape`), and it and the paths [`const_tokens`] hands out come from these rows, so they
/// cannot drift apart.
pub const METAL_MODULES: ModuleTable = ModuleTable {
    rows: &[
        ModuleRow {
            alias: "__tl",
            path: "::scratchy_target_metal::tape::lowered",
            types: &[
                "KernelId",
                "Binding",
                "DispatchShape",
                "MScaling",
                "MScaleAxis",
                "RuntimeGate",
                "LoweredCommand",
                "GatedCommand",
                "GemmDims",
                "WeightTensor",
                "RuntimeBindingKind",
                "ActivationWidth",
                "LoweredMetalTape",
                "ClassedTape",
                "GenClass",
                "KvAddressing",
                "TapeLoop",
                "TapeCommands",
                "Fence",
                "SetAxis",
                "LogitsRows",
                "RowTile",
                "DecodePartials",
                "SplitSteps",
            ],
        },
        ModuleRow {
            alias: "__tc",
            path: "::scratchy_target_metal::tape::constants",
            types: &["ConstantValue", "ConstantType", "ConstSlot"],
        },
        ModuleRow {
            alias: "__ti",
            path: "::scratchy_target_metal::tape::ids",
            types: &[
                "BucketM",
                "LayerId",
                "SourceIx",
                "ArenaSlotIdx",
                "PhysicalBlockIdx",
                "LogicalBlockIdx",
                "SlotInBlock",
                "SeqIdx",
                "QTokenIdx",
                "NumTokens",
                "BindingIdx",
                "HeadDim",
                "NumQHeads",
                "NumKvHeads",
                "RotDim",
                "RopePairOff",
                "BlockSize",
                "BlocksPerChunk",
                "MaxBlocksPerSeq",
                "TqDecodeHeads",
                "CommandIx",
                "QSize",
                "IntermediateSize",
                "HiddenSize",
                "KDim",
                "NDim",
                "SplitK",
                "AttnDebugMode",
                "AttnWindow",
                "KDimI32",
                "NDimI32",
                "MDimI32",
                "KPartitionSizeI32",
                "AttnScale",
                "RmsNormEps",
            ],
        },
    ],
};

/// The `use … as __tl;` lines the emitted metal tape names its types through.
pub(crate) fn alias_preamble() -> TokenStream {
    METAL_MODULES.preamble()
}

/// Emit the const-constructor tokens for a metal tape `value`.
pub fn const_tokens<T: Serialize>(value: &T) -> Result<TokenStream, Error> {
    scratchy_const_tokens::const_tokens(&METAL_MODULES, value)
}

#[cfg(test)]
mod tests {
    use super::*;
    use quote::quote;
    use scratchy_target_metal::tape::constants::{ConstantType, ConstantValue};
    use scratchy_target_metal::tape::ids::{LayerId, SourceIx};
    use scratchy_target_metal::tape::lowered::{
        Binding, DispatchShape, GatedCommand, KernelId, LoweredCommand, RuntimeGate, WeightTensor,
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
                        ix: SourceIx(7),
                        which: WeightTensor::AffineScales,
                        layer: LayerId(3),
                        binding_index: 1,
                    },
                ]),
                gemm_dims: None,
            },
            gate: Some(RuntimeGate::OnlyIfDecodeStep),
        }
    }

    #[test]
    fn emits_parseable_constructor_with_exact_values() {
        let toks = const_tokens(&sample()).expect("emit");
        // Must parse as a Rust expression…
        let expr: syn::Expr = syn::parse2(toks.clone()).expect("parse");
        let s = quote!(#expr).to_string();
        // …naming the runtime's own types through the tape's module aliases, with widths
        // and bit-exact floats.
        for needle in [
            "kernel : __tl :: KernelId :: Embed",
            "Binding :: ArenaSlot { slot : 2u32 , binding_index : 0u8 }",
            "Binding :: Source { ix : __ti :: SourceIx (7u32) , which : __tl :: WeightTensor :: \
             AffineScales , layer : __ti :: LayerId (3u32) , binding_index : 1u8 }",
            "index : 3u16",
            "RuntimeGate :: OnlyIfDecodeStep",
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
