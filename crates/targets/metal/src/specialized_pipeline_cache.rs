// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Specialized pipeline cache for scratchy-target-metal Phase 5.B.
//!
//! Builds (and memoizes) `MTLComputePipelineState` objects keyed on the
//! `(kernel name, baked constant set)` pair.

pub use crate::tape::constants::{ConstSlot, ConstantType, ConstantValue};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTL4Compiler, MTL4CompilerDescriptor, MTL4ComputePipelineDescriptor,
    MTL4LibraryFunctionDescriptor, MTLComputePipelineState, MTLDevice, MTLLibrary,
};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::sync::{Mutex, OnceLock};

use crate::shader_cache::load_library_from_bytes;
use crate::stream::MetalStreamError;
use crate::tape::lowered::BakedKernel;

pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;
pub type Library = Retained<ProtocolObject<dyn MTLLibrary>>;
pub type ComputePipelineState = Retained<ProtocolObject<dyn MTLComputePipelineState>>;

type MetallibBytes = &'static [u8];

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
    /// Kernels compiled at expansion with their constants ([`crate::aot::bake`]), by their key.
    baked: Mutex<HashMap<PipelineKey, Library>>,
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
            baked: Mutex::default(),
            pipelines: Mutex::new(HashMap::new()),
            compiler: OnceLock::new(),
        })
    }

    /// Load each of `kernels` not yet loaded: the libraries of [`crate::aot::baked_library`]
    /// kernels.
    pub fn register_baked<'k>(
        &self,
        kernels: impl IntoIterator<Item = &'k BakedKernel>,
    ) -> Result<(), MetalStreamError> {
        let mut baked = self.baked.lock().unwrap();
        for k in kernels {
            let key = PipelineKey::new(k.library, k.function, k.constants.to_vec());
            if let Entry::Vacant(slot) = baked.entry(key) {
                let lib = load_library_from_bytes(&self.device, k.metallib).map_err(|e| {
                    MetalStreamError::ShaderCompilationFailed(format!(
                        "load baked `{}`: {e}",
                        k.function
                    ))
                })?;
                slot.insert(lib);
            }
        }
        Ok(())
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

        // A baked kernel has its constants compiled in, under the full key; any other kernel
        // takes no constants.
        let library = if crate::aot::baked_library(key.library_name).is_some() {
            self.baked.lock().unwrap().get(key).cloned()
        } else if key.constants.is_empty() {
            self.libraries.get(key.library_name).cloned()
        } else {
            None
        };
        let library = library.ok_or_else(|| {
            MetalStreamError::ShaderCompilationFailed(format!(
                "`{}` of `{}` has no library compiled with constants {:?}",
                key.kernel_name, key.library_name, key.constants
            ))
        })?;

        // MTL4 function-descriptor chain: the library function (library + entry-point name) in
        // the descriptor the compiler consumes, which carries the MTL4 dispatch-support flag.
        let lib_fn_desc = MTL4LibraryFunctionDescriptor::new();
        lib_fn_desc.setName(Some(&NSString::from_str(key.kernel_name)));
        lib_fn_desc.setLibrary(Some(&library));
        let pipe_desc = MTL4ComputePipelineDescriptor::new();
        let lib_fn_super: &::objc2_metal::MTL4FunctionDescriptor = &lib_fn_desc;
        pipe_desc.setComputeFunctionDescriptor(Some(lib_fn_super));

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
        // FUNCTION NAME instead. MPP/NAX kernels' threadgroup size (incl.
        // matmul2d internals) isn't knowable to Rust at compile time, so this
        // pipeline-build assert is the earliest possible guard.

        let mut map = self.pipelines.lock().unwrap();
        Ok(map.entry(key.clone()).or_insert(pipeline).clone())
    }
}
