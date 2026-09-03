// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Specialized pipeline cache for scratchy-target-metal Phase 5.B.
//!
//! Builds (and memoizes) `MTLComputePipelineState` objects keyed on the
//! `(kernel name, function-constant bag)` pair.

pub use crate::tape::constants::{ConstSlot, ConstantType, ConstantValue};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4Compiler, MTL4CompilerDescriptor, MTL4ComputePipelineDescriptor,
    MTL4LibraryFunctionDescriptor, MTL4SpecializedFunctionDescriptor, MTLComputePipelineState,
    MTLDataType, MTLDevice, MTLFunctionConstantValues, MTLLibrary,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::{Mutex, OnceLock};

use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;
pub type FunctionConstantValues = Retained<MTLFunctionConstantValues>;

type MetallibBytes = &'static [u8];

/// The objc mapping the neutral `ConstantType` deliberately does not carry.
pub(crate) fn metal_data_type(t: ConstantType) -> MTLDataType {
    match t {
        ConstantType::UInt => MTLDataType::UInt,
        ConstantType::Int => MTLDataType::Int,
        ConstantType::Float => MTLDataType::Float,
        ConstantType::Bool => MTLDataType::Bool,
    }
}

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct PipelineKey {
    pub kernel_name: &'static str,
    pub library_name: &'static str,
    pub constants: Vec<ConstantValue>,
}

impl PipelineKey {
    pub fn new(
        library_name: &'static str,
        kernel_name: &'static str,
        mut constants: Vec<ConstantValue>,
    ) -> Self {
        constants.sort_by_key(|c| c.index);
        Self {
            kernel_name,
            library_name,
            constants,
        }
    }
}

pub struct SpecializedPipelineCache {
    device: Device,
    libraries: HashMap<&'static str, Library>,
    pipelines: Mutex<HashMap<PipelineKey, ComputePipelineState>>,
    /// Lazy-built MTL4 compiler. Pipelines built through this compiler
    /// run correctly when dispatched (`dispatchThreadgroups`) on an
    /// `MTL4ComputeCommandEncoder`; the legacy MTL3
    /// `MTLDevice.newComputePipelineStateWithDescriptor` path produces
    /// pipelines that emit wrong kernel state on MTL4 encoders.
    compiler: OnceLock<Retained<ProtocolObject<dyn MTL4Compiler>>>,
}

impl SpecializedPipelineCache {
    pub fn new(
        device: Device,
        libraries_in: &[(&'static str, MetallibBytes)],
    ) -> Result<Self, MetalStreamError> {
        let mut libraries = HashMap::with_capacity(libraries_in.len());
        for (name, bytes) in libraries_in {
            let lib = load_library_from_bytes(&device, bytes).map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("load library `{name}`: {e}"))
            })?;
            libraries.insert(*name, lib);
        }
        Ok(Self {
            device,
            libraries,
            pipelines: Mutex::new(HashMap::new()),
            compiler: OnceLock::new(),
        })
    }

    /// Register a synthesized kernel library from precompiled metallib
    /// bytes. The macro AOT-compiles synthesized `.metal` sources at
    /// proc-macro expansion time (shells out to `xcrun metal -c` +
    /// `xcrun metallib`) and embeds the resulting bytes as
    /// `&'static [u8]`. Same `newLibraryWithData` path used by all
    /// hand-written shaders — NOT the runtime MSL→AIR compile path
    /// (`newLibraryWithSource`), which produces different binaries
    /// across Apple GPU generations and was the source of a real M1
    /// runtime failure.
    ///
    /// Used by the compiler-driven megakernel synthesis pass
    /// (`scratchy-forward-compiler-macro/src/fuse_pass.rs`).
    pub fn register_metallib_library(
        &mut self,
        name: &'static str,
        bytes: &'static [u8],
    ) -> Result<(), MetalStreamError> {
        let lib = load_library_from_bytes(&self.device, bytes).map_err(|e| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "load synthesized metallib `{name}`: {e}"
            ))
        })?;
        self.libraries.insert(name, lib);
        Ok(())
    }

    #[cfg(test)]
    #[allow(dead_code)] // test/diagnostic constructor; not on the production path
    pub(crate) fn from_sources(
        device: Device,
        sources: &[(&'static str, &str)],
    ) -> Result<Self, MetalStreamError> {
        let mut libraries = HashMap::with_capacity(sources.len());
        for (name, source) in sources {
            let opts = objc2_metal::MTLCompileOptions::new();
            let ns_source = NSString::from_str(source);
            let lib = device
                .newLibraryWithSource_options_error(&ns_source, Some(&opts))
                .map_err(|e| {
                    MetalStreamError::ShaderCompilationFailed(format!(
                        "compile library `{name}`: {e:?}"
                    ))
                })?;
            libraries.insert(*name, lib);
        }
        Ok(Self {
            device,
            libraries,
            pipelines: Mutex::new(HashMap::new()),
            compiler: OnceLock::new(),
        })
    }

    pub fn with_standard_shaders(device: Device) -> Result<Self, MetalStreamError> {
        let mut cache = Self::new(
            device,
            &[
                ("rmsnorm", crate::embedded_metallib!("rmsnorm")),
                (
                    "fused_add_rmsnorm",
                    crate::embedded_metallib!("fused_add_rmsnorm"),
                ),
                (
                    "fused_gate_up_silu_mul",
                    crate::embedded_metallib!("fused_gate_up_silu_mul"),
                ),
                (
                    "fused_qkv_rope_cache",
                    crate::embedded_metallib!("fused_qkv_rope_cache"),
                ),
                (
                    "fused_affine_qkv_rope_cache",
                    crate::embedded_metallib!("fused_affine_qkv_rope_cache"),
                ),
                ("attention", crate::embedded_metallib!("attention")),
                (
                    "attention_dense_gather",
                    crate::embedded_metallib!("attention_dense_gather"),
                ),
                (
                    "attention_causal_softmax",
                    crate::embedded_metallib!("attention_causal_softmax"),
                ),
                (
                    "attention_layout_convert",
                    crate::embedded_metallib!("attention_layout_convert"),
                ),
                (
                    "attention_steel",
                    crate::embedded_metallib!("attention_steel"),
                ),
                (
                    "attention_steel_paged",
                    crate::embedded_metallib!("attention_steel_paged"),
                ),
                (
                    "gather_last_token",
                    crate::embedded_metallib!("gather_last_token"),
                ),
                ("rope", crate::embedded_metallib!("rope")),
                ("turboquant", crate::embedded_metallib!("turboquant")),
                ("embed", crate::embedded_metallib!("embed")),
                ("activation", crate::embedded_metallib!("activation")),
                ("elementwise", crate::embedded_metallib!("elementwise")),
                (
                    "quantized_dequantize",
                    crate::embedded_metallib!("quantized_dequantize"),
                ),
                ("quantized_qmv", crate::embedded_metallib!("quantized_qmv")),
                ("quantized_qmm", crate::embedded_metallib!("quantized_qmm")),
                // `quantized_qmm_nax` is intentionally absent — runtime-
                // compiled below (offline metallib miscompiles MPP).
                ("quantized_qvm", crate::embedded_metallib!("quantized_qvm")),
                (
                    "quantized_splitk_reduce",
                    crate::embedded_metallib!("quantized_splitk_reduce"),
                ),
                ("silu_mul", crate::embedded_metallib!("silu_mul")),
                ("gemm", crate::embedded_metallib!("gemm")),
                // MoE-on-Metal: router decomposition kernels.
                // `lower_metal_moe` (Phase A) emits commands that
                // reference these libraries by name; without them
                // the per-(library, function) pipeline lookup in
                // `get_or_build` panics at first MoE forward.
                ("softmax", crate::embedded_metallib!("softmax")),
                ("argpartition", crate::embedded_metallib!("argpartition")),
                (
                    "take_along_axis",
                    crate::embedded_metallib!("take_along_axis"),
                ),
                (
                    "slice_trailing_cols",
                    crate::embedded_metallib!("slice_trailing_cols"),
                ),
                (
                    "moe_weighted_sum",
                    crate::embedded_metallib!("moe_weighted_sum"),
                ),
                // Gemma-4 `gemma_moe` op: per-expert score scale gather-mul.
                (
                    "moe_per_expert_scale",
                    crate::embedded_metallib!("moe_per_expert_scale"),
                ),
                // Gemma-4 grouped expert-GEMM prefill: counting sort
                // (offsets/init/scatter) + un-scatter.
                ("moe_group", crate::embedded_metallib!("moe_group")),
                // Qwen3.5 Gated-DeltaNet + attention-output-gate kernels.
                // `lower_one` (gate_apply/gate_split) and the GDN op lowering
                // reference these libraries by name; without registration the
                // per-(library, function) lookup in `get_or_build` panics at
                // first forward.
                ("gate_apply", crate::embedded_metallib!("gate_apply")),
                ("gate_split", crate::embedded_metallib!("gate_split")),
                // Qwen3.5-MoE shared-expert combine.
                ("gate_scale", crate::embedded_metallib!("gate_scale")),
                ("gdn_gating", crate::embedded_metallib!("gdn_gating")),
                (
                    "gdn_rms_norm_gated",
                    crate::embedded_metallib!("gdn_rms_norm_gated"),
                ),
                (
                    "gdn_conv1d_varlen",
                    crate::embedded_metallib!("gdn_conv1d_varlen"),
                ),
                (
                    "gdn_scan_varlen",
                    crate::embedded_metallib!("gdn_scan_varlen"),
                ),
                // Vision-tower (VL) kernels.
                (
                    "vision_rope_2d",
                    crate::embedded_metallib!("vision_rope_2d"),
                ),
                (
                    "vision_varlen_attn",
                    crate::embedded_metallib!("vision_varlen_attn"),
                ),
                (
                    "vision_layernorm",
                    crate::embedded_metallib!("vision_layernorm"),
                ),
                (
                    "embedding_gather",
                    crate::embedded_metallib!("embedding_gather"),
                ),
                ("avg_pool_2d", crate::embedded_metallib!("avg_pool_2d")),
            ],
        )?;
        // NAX qmm_t (`affine_qmm_t_nax_*`) MUST be compiled from source at
        // runtime via `newLibraryWithSource`: the offline `xcrun metal`
        // metallib toolchain miscompiles MetalPerformancePrimitives
        // `matmul2d` cooperative tensors (each MMA reduces only half its
        // K → ~95%-wrong qmm_t). The runtime compiler is correct; this is
        // also the path mlx uses. The cross-generation `newLibraryWithSource`
        // caveat noted on `register_metallib_library` does not apply here —
        // NAX only runs on M5+ (`is_nax_capable`), and there it's the only
        // correct option.
        let nax_lib = crate::shader_cache::compile_nax_library_from_source(&cache.device)
            .map_err(MetalStreamError::ShaderCompilationFailed)?;
        cache.libraries.insert("quantized_qmm_nax", nax_lib);
        // NAX paged attention (`attention_steel_nax_paged`) — same runtime-
        // compile requirement as the qmm (MPP miscompiled offline). Only
        // compiled where the runtime driver supports MPP; on non-NAX GPUs
        // the source still parses but the dispatcher never selects it.
        if let Ok(nax_attn_lib) =
            crate::shader_cache::compile_nax_attention_paged_library_from_source(&cache.device)
        {
            cache
                .libraries
                .insert("attention_steel_nax_paged", nax_attn_lib);
        }
        Ok(cache)
    }

    pub fn len(&self) -> usize {
        self.pipelines.lock().unwrap().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_or_build(
        &self,
        key: &PipelineKey,
    ) -> Result<ComputePipelineState, MetalStreamError> {
        {
            let map = self.pipelines.lock().unwrap();
            if let Some(p) = map.get(key) {
                return Ok(p.clone());
            }
        }

        let library = self.libraries.get(key.library_name).ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "no library `{}` in SpecializedPipelineCache (call `new` with this library)",
                key.library_name
            ))
        })?;

        let constants = MTLFunctionConstantValues::new();
        for c in &key.constants {
            unsafe {
                constants.setConstantValue_type_atIndex(
                    NonNull::new(&c.bits as *const u32 as *mut c_void).unwrap(),
                    metal_data_type(c.ty),
                    c.index as usize,
                );
            }
        }

        let ns_name = NSString::from_str(key.kernel_name);

        // MTL4 function-descriptor chain:
        //   MTL4LibraryFunctionDescriptor       (where to find the
        //                                         function: library +
        //                                         entry-point name)
        //   MTL4SpecializedFunctionDescriptor   (wraps it + bakes
        //                                         function-constant
        //                                         values for this
        //                                         specialization)
        //   MTL4ComputePipelineDescriptor       (the descriptor the
        //                                         compiler consumes;
        //                                         carries the MTL4
        //                                         dispatch-support flag)
        let lib_fn_desc = MTL4LibraryFunctionDescriptor::new();
        lib_fn_desc.setName(Some(&ns_name));
        lib_fn_desc.setLibrary(Some(library));

        let spec_fn_desc = MTL4SpecializedFunctionDescriptor::new();
        // Upcast: spec descriptor's `setFunctionDescriptor` accepts
        // any `MTL4FunctionDescriptor` subclass.
        let lib_fn_super: &::objc2_metal::MTL4FunctionDescriptor = &lib_fn_desc;
        spec_fn_desc.setFunctionDescriptor(Some(lib_fn_super));
        spec_fn_desc.setConstantValues(Some(&constants));

        let pipe_desc = MTL4ComputePipelineDescriptor::new();
        let spec_fn_super: &::objc2_metal::MTL4FunctionDescriptor = &spec_fn_desc;
        pipe_desc.setComputeFunctionDescriptor(Some(spec_fn_super));

        let compiler = self.compiler.get_or_init(|| {
            let cdesc = MTL4CompilerDescriptor::new();
            self.device
                .newCompilerWithDescriptor_error(&cdesc)
                .expect("newCompilerWithDescriptor")
        });
        let pipeline = compiler
            .newComputePipelineStateWithDescriptor_compilerTaskOptions_error(&pipe_desc, None)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "MTL4 build pipeline `{}` (lib `{}`, {} constants): {e:?}",
                    key.kernel_name,
                    key.library_name,
                    key.constants.len(),
                ))
            })?;

        // GUARD (threadgroup-overflow class): a kernel whose total threadgroup
        // memory exceeds the device limit faults at GPU exec with a cryptic
        // `MTLCommandBufferStatus(5)`. Catch it here at pipeline build with the
        // FUNCTION NAME instead. Runtime-compiled MPP/NAX kernels' threadgroup
        // size (incl. matmul2d internals) isn't knowable to Rust at compile
        // time, so this build-time assert is the earliest possible guard.

        let mut map = self.pipelines.lock().unwrap();
        Ok(map.entry(key.clone()).or_insert(pipeline).clone())
    }
}
