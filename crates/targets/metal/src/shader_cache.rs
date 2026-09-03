// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Shader compilation and caching infrastructure.

use dispatch2::DispatchData;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLCompileOptions, MTLComputePipelineDescriptor, MTLComputePipelineState, MTLDevice,
    MTLFunctionConstantValues, MTLLanguageVersion, MTLLibrary, MTLMathMode, MTLPipelineOption,
};
use std::collections::HashMap;
use std::ffi::c_void;
use std::ptr::NonNull;
use std::sync::Mutex;

use crate::specialized_pipeline_cache::ConstantValue;
use crate::stream::MetalStreamError;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

/// Cache for compiled Metal shaders.
///
/// Two pipeline pots: `pipelines` for kernels that don't take any
/// `[[function_constant(N)]]` (keyed on symbol name only), and
/// `specialized_pipelines` for the qmv / qmm_t family which bake
/// K / N / M (and friends) in as function constants — the same
/// trade `SpecializedPipelineCache` makes on the worker side.
/// Standalone test paths reach the specialized variant via
/// [`Self::get_pipeline_specialized`].
pub struct ShaderCache {
    device: Device,
    libraries: HashMap<String, Library>,
    pipelines: Mutex<HashMap<String, ComputePipelineState>>,
    specialized_pipelines: Mutex<HashMap<(String, Vec<ConstantValue>), ComputePipelineState>>,
}

impl ShaderCache {
    /// Create a new shader cache with all shader libraries.
    pub fn new(device: Device) -> Result<Self, MetalStreamError> {
        let mut libraries = HashMap::new();
        for (name, bytes) in [
            ("activation", &crate::embedded_metallib!("activation")[..]),
            ("rope", &crate::embedded_metallib!("rope")[..]),
            ("rmsnorm", &crate::embedded_metallib!("rmsnorm")[..]),
            (
                "fused_add_rmsnorm",
                &crate::embedded_metallib!("fused_add_rmsnorm")[..],
            ),
            (
                "fused_gate_up_silu_mul",
                &crate::embedded_metallib!("fused_gate_up_silu_mul")[..],
            ),
            (
                "fused_qkv_rope_cache",
                &crate::embedded_metallib!("fused_qkv_rope_cache")[..],
            ),
            (
                "fused_affine_qkv_rope_cache",
                &crate::embedded_metallib!("fused_affine_qkv_rope_cache")[..],
            ),
            (
                "quantized_dequantize",
                &crate::embedded_metallib!("quantized_dequantize")[..],
            ),
            (
                "quantized_qmv",
                &crate::embedded_metallib!("quantized_qmv")[..],
            ),
            (
                "quantized_qmm",
                &crate::embedded_metallib!("quantized_qmm")[..],
            ),
            // NOTE: `quantized_qmm_nax` is NOT loaded in this static
            // list — its library is selected below (AOT metallib by
            // default; runtime JIT only as an opt-in fallback).
            (
                "quantized_qvm",
                &crate::embedded_metallib!("quantized_qvm")[..],
            ),
            (
                "quantized_splitk_reduce",
                &crate::embedded_metallib!("quantized_splitk_reduce")[..],
            ),
            ("silu_mul", &crate::embedded_metallib!("silu_mul")[..]),
            ("softmax", &crate::embedded_metallib!("softmax")[..]),
            (
                "argpartition",
                &crate::embedded_metallib!("argpartition")[..],
            ),
            (
                "take_along_axis",
                &crate::embedded_metallib!("take_along_axis")[..],
            ),
            (
                "moe_weighted_sum",
                &crate::embedded_metallib!("moe_weighted_sum")[..],
            ),
            ("moe_group", &crate::embedded_metallib!("moe_group")[..]),
            (
                "slice_trailing_cols",
                &crate::embedded_metallib!("slice_trailing_cols")[..],
            ),
        ] {
            let lib = load_library_from_bytes(&device, bytes).map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!("load `{name}.metallib`: {e}"))
            })?;
            libraries.insert(name.to_string(), lib);
        }

        // NAX qmm_t: AOT-embedded metallib by DEFAULT.
        //
        // `quantized_qmm_nax.metal` pulls in MetalPerformancePrimitives
        // `matmul2d` cooperative tensors. On SDK 26.5 the offline
        // `xcrun metal` frontend used to miscompile those (each call
        // reduced only half its K → `affine_qmm_t_nax_*` came out ~95%
        // wrong, worst_abs ~10.45), which is why this library was JIT'd
        // from source at startup. That was MLX issue #3586 and the fix
        // (MLX PR #3622) is `-mmacosx-version-min=26.2`: build.rs now
        // compiles this stem with that flag (see `NAX_MPP_SHADER_STEMS`)
        // and the embedded metallib is bit-exact with the CPU reference
        // (verified by the `affine_qmm_t_nax_*` golden) — and matches
        // the mlx WHEEL's precompiled codegen at ~13 TFLOPS, faster than
        // the ~10 TFLOPS the runtime `newLibraryWithSource` compile got.
        //
        let nax_lib =
            load_library_from_bytes(&device, &crate::embedded_metallib!("quantized_qmm_nax")[..])
                .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "load embedded `quantized_qmm_nax.metallib`: {e}"
                ))
            })?;
        libraries.insert("quantized_qmm_nax".to_string(), nax_lib);

        Ok(Self {
            device,
            libraries,
            pipelines: Mutex::new(HashMap::new()),
            specialized_pipelines: Mutex::new(HashMap::new()),
        })
    }

    fn library_for(&self, name: &str) -> Result<&Library, MetalStreamError> {
        let lib = if name.starts_with("rope_") {
            self.libraries.get("rope")
        } else if name.starts_with("rmsnorm_") {
            self.libraries.get("rmsnorm")
        } else if name.starts_with("fused_add_rmsnorm_") {
            self.libraries.get("fused_add_rmsnorm")
        } else if name.starts_with("fused_gate_up_silu_mul_") {
            self.libraries.get("fused_gate_up_silu_mul")
        } else if name.starts_with("fused_affine_qkv_rope_cache_") {
            self.libraries.get("fused_affine_qkv_rope_cache")
        } else if name.starts_with("fused_qkv_rope_cache_") {
            self.libraries.get("fused_qkv_rope_cache")
        } else if name.starts_with("affine_dequantize_") || name.starts_with("affine_embed_") {
            // Both kernels live in shaders/quantized_dequantize.metal —
            // affine_embed reuses the dequant math under a gather
            // indirection (P6).
            self.libraries.get("quantized_dequantize")
        } else if name.starts_with("nvfp4_qmm_t_nax_") {
            // NVFP4 NAX prefill — same metallib as affine NAX (both in
            // quantized_qmm_nax.metal). Before the generic nvfp4_qmm_t_.
            self.libraries.get("quantized_qmm_nax")
        } else if name.starts_with("nvfp4_qmm_t_") {
            // NVFP4 standard prefill qmm_t — lives in quantized_qmm.metal.
            self.libraries.get("quantized_qmm")
        } else if name.starts_with("nvfp4_qmv_") {
            // NVFP4 decode matvec — shares quantized_qmv.metal with affine.
            self.libraries.get("quantized_qmv")
        } else if name.starts_with("affine_qmm_t_nax_") {
            // NAX (Apple9 / M4+) qmm_t — lives in its own metallib
            // since `quantized_qmm_nax.metal` pulls in the
            // MetalPerformancePrimitives headers. Routed before the
            // generic `affine_qmm_t_` prefix below since the prefixes
            // overlap.
            self.libraries.get("quantized_qmm_nax")
        } else if name.starts_with("affine_qmm_t_") || name.starts_with("affine_qmm_n_") {
            // Matches `affine_qmm_t_<dtype>_*`,
            // `affine_qmm_t_splitk_<dtype>_*`, and
            // `affine_qmm_n_<dtype>_*` by prefix — all live in
            // shaders/quantized_qmm.metal.
            self.libraries.get("quantized_qmm")
        } else if name.starts_with("affine_qmv_") || name.starts_with("affine_gather_qmv_") {
            // Also matches `affine_qmv_quad_*` and `affine_qmv_fast_*`
            // by prefix, and the MoE gather variants
            // `affine_gather_qmv_*` / `affine_gather_qmv_fast_*` that
            // live alongside the plain qmv kernels in
            // shaders/quantized_qmv.metal (added in the MoE-on-Metal
            // port).
            self.libraries.get("quantized_qmv")
        } else if name.starts_with("affine_qvm_") {
            // Matches `affine_qvm_<dtype>_*` and
            // `affine_qvm_split_k_<dtype>_*` by prefix. Distinct
            // from qmv (the letter order matters): qmv =
            // matvec-transpose=true (in quantized_qmv.metal); qvm
            // = vector × matrix transpose=false
            // (in quantized_qvm.metal).
            self.libraries.get("quantized_qvm")
        } else if name.starts_with("splitk_reduce_") {
            self.libraries.get("quantized_splitk_reduce")
        } else if name.starts_with("silu_mul") {
            self.libraries.get("silu_mul")
        } else if name.starts_with("block_softmax_")
            || name.starts_with("looped_softmax_")
            || name.starts_with("topk_renorm_")
        {
            self.libraries.get("softmax")
        } else if name.starts_with("c_arg_block_sort_") {
            self.libraries.get("argpartition")
        } else if name.starts_with("take_along_axis_") {
            self.libraries.get("take_along_axis")
        } else if name.starts_with("moe_weighted_sum_") {
            self.libraries.get("moe_weighted_sum")
        } else if name.starts_with("slice_trailing_cols_") {
            self.libraries.get("slice_trailing_cols")
        } else {
            self.libraries.get("activation")
        };
        lib.ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "No library found for kernel '{}'",
                name
            ))
        })
    }

    /// Get or compile a pipeline for a given kernel name (no function
    /// constants — for kernels whose symbol uniquely determines the
    /// pipeline).
    pub fn get_pipeline(&self, name: &str) -> Result<ComputePipelineState, MetalStreamError> {
        {
            let pipelines = self.pipelines.lock().unwrap();
            if let Some(pipeline) = pipelines.get(name) {
                return Ok(pipeline.clone());
            }
        }

        let library = self.library_for(name)?;

        let ns_name = NSString::from_str(name);
        let function = library.newFunctionWithName(&ns_name).ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed(format!("Failed to get function '{}'", name))
        })?;

        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "Failed to create pipeline for '{}': {:?}",
                    name, e
                ))
            })?;

        {
            let mut pipelines = self.pipelines.lock().unwrap();
            pipelines.insert(name.to_string(), pipeline.clone());
        }

        Ok(pipeline)
    }

    /// Get or compile a pipeline specialized on the provided function
    /// constants. The qmv / qmm_t family uses this — K / N / M ride
    /// as `[[function_constant(N)]]` in `quantized_qmv.metal` and
    /// `quantized_qmm.metal` so the same kernel symbol can serve
    /// every per-Linear shape with a per-shape pipeline.
    ///
    /// Cache key is `(symbol_name, sorted-by-index constants)`. The
    /// sort matches `SpecializedPipelineCache::PipelineKey::new` so
    /// callers can hand identical constant bags to either cache and
    /// get the same lookup behavior.
    pub fn get_pipeline_specialized(
        &self,
        name: &str,
        constants: &[ConstantValue],
    ) -> Result<ComputePipelineState, MetalStreamError> {
        let mut sorted_constants = constants.to_vec();
        sorted_constants.sort_by_key(|c| c.index);
        let cache_key = (name.to_string(), sorted_constants.clone());
        {
            let pipelines = self.specialized_pipelines.lock().unwrap();
            if let Some(pipeline) = pipelines.get(&cache_key) {
                return Ok(pipeline.clone());
            }
        }

        let library = self.library_for(name)?;

        let constant_values = MTLFunctionConstantValues::new();
        for c in &sorted_constants {
            unsafe {
                constant_values.setConstantValue_type_atIndex(
                    NonNull::new(&c.bits as *const u32 as *mut c_void).unwrap(),
                    crate::specialized_pipeline_cache::metal_data_type(c.ty),
                    c.index as usize,
                );
            }
        }

        let ns_name = NSString::from_str(name);
        let function = library
            .newFunctionWithName_constantValues_error(&ns_name, &constant_values)
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "newFunctionWithName('{}', {} constants): {e:?}",
                    name,
                    sorted_constants.len(),
                ))
            })?;

        let descriptor = MTLComputePipelineDescriptor::new();
        descriptor.setComputeFunction(Some(&function));

        let pipeline = self
            .device
            .newComputePipelineStateWithDescriptor_options_reflection_error(
                &descriptor,
                MTLPipelineOption::None,
                None,
            )
            .map_err(|e| {
                MetalStreamError::ShaderCompilationFailed(format!(
                    "build pipeline '{}' with {} constants: {e:?}",
                    name,
                    sorted_constants.len(),
                ))
            })?;

        {
            let mut pipelines = self.specialized_pipelines.lock().unwrap();
            pipelines.insert(cache_key, pipeline.clone());
        }

        Ok(pipeline)
    }
}

/// Wrap a static byte slice as a `dispatch_data_t` and load it as a Metal
/// library. The bytes typically come from `include_bytes!` so we keep the
/// destructor as no-op (default behavior of `DispatchData::from`'s
/// implementation copies into a managed buffer).
pub fn load_library_from_bytes(device: &Device, bytes: &'static [u8]) -> Result<Library, String> {
    let data = DispatchData::from_static_bytes(bytes);
    device
        .newLibraryWithData_error(&data)
        .map_err(|e| format!("newLibraryWithData failed: {:?}", e))
}

/// Compile the NAX qmm_t library (`quantized_qmm_nax`) from MSL **source
/// at runtime** via `newLibraryWithSource`. `ShaderCache::new` loads the
/// build.rs AOT metallib instead; this is used directly by
/// `specialized_pipeline_cache` and tests.
///
/// History: anything that includes `metal_nax.h` uses MetalPerformance-
/// Primitives `matmul2d` cooperative tensors. The offline `xcrun metal`
/// toolchain **used to miscompile** those on SDK 26.5 — each `matmul2d`
/// reduced only half its K, so `affine_qmm_t_nax_*` came out ~95% wrong
/// (verified on M5/applegpu_g17g). That was MLX issue #3586; MLX PR #3622
/// showed the fix is `-mmacosx-version-min=26.2`, which build.rs now
/// passes. The AOT metallib is therefore correct AND matches mlx's
/// precompiled codegen (~13 TFLOPS) — faster than this runtime compile
/// (~10 TFLOPS). Math mode Safe (fastMath off) and Metal 4.0 match mlx's
/// `MTLCompileOptions`.
///
/// The runtime compiler has no `-I` for our shader dir, so we inline the
/// `#include "metal_nax.h"` ourselves; the `<MetalPerformancePrimitives/…>`
/// framework include inside `metal_nax.h` is resolved by the runtime
/// compiler. Function constants (`QMM_K/N/M`) and `[[host_name]]`
/// instantiations resolve normally via `newFunctionWithName`.
pub fn compile_nax_library_from_source(device: &Device) -> Result<Library, String> {
    const NAX_HEADER: &str =
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/metal_nax.h"));
    const QMM_NAX_SRC: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/quantized_qmm_nax.metal"
    ));
    let body = QMM_NAX_SRC.replace("#include \"metal_nax.h\"", "");
    let source = format!("{NAX_HEADER}\n{body}");

    nax_library_from_msl(device, &source, "quantized_qmm_nax")
}

/// Compile the NAX **paged attention** library (`attention_steel_nax_paged`)
/// from MSL source at runtime via `newLibraryWithSource` — same reason as
/// the qmm: anything pulling in `metal_nax.h` / MPP `matmul2d` must use the
/// runtime driver compiler, not the offline `xcrun metal` toolchain (which
/// reduces only half the K and silently miscompiles MPP). The kernel's
/// `#include "metal_nax.h"` and the per-kernel header are inlined here since
/// the runtime compiler has no `-I` for our shader dir.
pub fn compile_nax_attention_paged_library_from_source(device: &Device) -> Result<Library, String> {
    const NAX_HEADER: &str =
        include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/shaders/metal_nax.h"));
    const KERNEL_HEADER: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/mlx_steel_attn/steel_attention_nax_paged_kernel.h"
    ));
    const NAX_PAGED_SRC: &str = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/shaders/attention_steel_nax_paged.metal"
    ));
    // Inline both includes (runtime compiler has no include path).
    let kernel_header = KERNEL_HEADER.replace("#include \"metal_nax.h\"", "");
    let body = NAX_PAGED_SRC
        .replace("#include \"metal_nax.h\"", "")
        .replace(
            "#include \"mlx_steel_attn/steel_attention_nax_paged_kernel.h\"",
            "",
        );
    // `<metal_stdlib>`/`<metal_simdgroup>` + `using namespace metal` BEFORE
    // metal_nax.h: NAXTile::row_reduce calls unqualified `simd_shuffle_xor`,
    // which two-phase lookup only resolves if `metal` is in scope at the
    // template's *definition* point (the qmm never instantiates row_reduce
    // so this was latent).
    let source = format!(
        "#include <metal_stdlib>\n#include <metal_simdgroup>\nusing namespace metal;\n{NAX_HEADER}\n{kernel_header}\n{body}"
    );

    nax_library_from_msl(device, &source, "attention_steel_nax_paged")
}

/// Shared runtime NAX compile: math mode Safe (fastMath off) + Metal 4.0,
/// matching mlx's `MTLCompileOptions` (the only correct config for MPP).
fn nax_library_from_msl(device: &Device, source: &str, label: &str) -> Result<Library, String> {
    let opts = MTLCompileOptions::new();
    opts.setMathMode(MTLMathMode::Safe); // == fastMath off; what mlx uses
    opts.setLanguageVersion(MTLLanguageVersion::Version4_0);
    device
        .newLibraryWithSource_options_error(&NSString::from_str(source), Some(&opts))
        .map_err(|e| format!("newLibraryWithSource({label}) failed: {e:?}"))
}
