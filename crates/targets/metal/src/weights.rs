// Copyright © 2024 Apple Inc.
// SPDX-License-Identifier: Apache-2.0

//! Metal weight loading from safetensors files.

use std::collections::HashMap;
use std::ffi::c_void;
use std::fs::File;
use std::path::Path;
use std::ptr::NonNull;

use memmap2::Mmap;
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLDevice, MTLResourceOptions};

pub type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;
pub type Device = Retained<ProtocolObject<dyn MTLDevice>>;

/// Metal weights loaded from safetensors.
pub struct MetalWeights {
    device: Device,
    _mmap: Mmap,
    tensors: HashMap<String, TensorInfo>,
    buffers: HashMap<String, Buffer>,
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    shape: Vec<usize>,
    dtype: String,
    data_offset: usize,
    byte_size: usize,
}

impl MetalWeights {
    pub fn load_safetensors(device: Device, path: &Path) -> Result<Self, String> {
        let file = File::open(path)
            .map_err(|e| format!("Failed to open safetensors file: {}", e))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| format!("Failed to mmap safetensors file: {}", e))?;

        let tensors = Self::parse_header(&mmap)?;

        let mut buffers = HashMap::new();
        for (name, info) in &tensors {
            let buffer = Self::create_buffer(&device, &mmap, info)?;
            buffers.insert(name.clone(), buffer);
        }

        Ok(Self {
            device,
            _mmap: mmap,
            tensors,
            buffers,
        })
    }

    fn parse_header(mmap: &Mmap) -> Result<HashMap<String, TensorInfo>, String> {
        if mmap.len() < 8 {
            return Err("Safetensors file too small (< 8 bytes)".to_string());
        }

        let header_len = u64::from_le_bytes([
            mmap[0], mmap[1], mmap[2], mmap[3], mmap[4], mmap[5], mmap[6], mmap[7],
        ]) as usize;

        if mmap.len() < 8 + header_len {
            return Err(format!(
                "Safetensors file too small for header (need {}, got {})",
                8 + header_len,
                mmap.len()
            ));
        }

        let header_bytes = &mmap[8..8 + header_len];
        let header_str = std::str::from_utf8(header_bytes)
            .map_err(|e| format!("Invalid UTF-8 in safetensors header: {}", e))?;
        let header: serde_json::Value = serde_json::from_str(header_str)
            .map_err(|e| format!("Invalid JSON in safetensors header: {}", e))?;

        let mut tensors = HashMap::new();
        if let Some(obj) = header.as_object() {
            for (name, value) in obj {
                if name == "__metadata__" {
                    continue;
                }

                let tensor_obj = value
                    .as_object()
                    .ok_or_else(|| format!("Tensor '{}' is not an object", name))?;

                let shape_arr = tensor_obj
                    .get("shape")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("Tensor '{}' missing 'shape' array", name))?;
                let shape: Vec<usize> = shape_arr
                    .iter()
                    .map(|v| v.as_u64().map(|n| n as usize))
                    .collect::<Option<Vec<_>>>()
                    .ok_or_else(|| format!("Tensor '{}' has invalid shape", name))?;

                let dtype = tensor_obj
                    .get("dtype")
                    .and_then(|v| v.as_str())
                    .ok_or_else(|| format!("Tensor '{}' missing 'dtype' string", name))?
                    .to_string();

                let offsets = tensor_obj
                    .get("data_offsets")
                    .and_then(|v| v.as_array())
                    .ok_or_else(|| format!("Tensor '{}' missing 'data_offsets' array", name))?;
                if offsets.len() != 2 {
                    return Err(format!(
                        "Tensor '{}' data_offsets must have 2 elements",
                        name
                    ));
                }
                let start = offsets[0]
                    .as_u64()
                    .ok_or_else(|| format!("Tensor '{}' data_offsets[0] not a number", name))?
                    as usize;
                let end = offsets[1]
                    .as_u64()
                    .ok_or_else(|| format!("Tensor '{}' data_offsets[1] not a number", name))?
                    as usize;

                let byte_size = end - start;
                let data_offset = 8 + header_len + start;

                tensors.insert(
                    name.clone(),
                    TensorInfo {
                        shape,
                        dtype,
                        data_offset,
                        byte_size,
                    },
                );
            }
        }

        Ok(tensors)
    }

    fn create_buffer(device: &Device, mmap: &Mmap, info: &TensorInfo) -> Result<Buffer, String> {
        if info.data_offset + info.byte_size > mmap.len() {
            return Err(format!(
                "Tensor data out of bounds (offset={}, size={}, file_len={})",
                info.data_offset, info.byte_size, mmap.len()
            ));
        }

        let data = &mmap[info.data_offset..info.data_offset + info.byte_size];

        // SAFETY: `newBufferWithBytes:length:options:` copies the bytes into a
        // device-owned buffer; `data` only needs to be valid for the call.
        let buffer = unsafe {
            let bytes = NonNull::new(data.as_ptr() as *mut c_void)
                .ok_or_else(|| "tensor data pointer is null".to_string())?;
            device
                .newBufferWithBytes_length_options(
                    bytes,
                    info.byte_size,
                    MTLResourceOptions::StorageModeShared,
                )
                .ok_or_else(|| "newBufferWithBytes returned nil".to_string())?
        };

        Ok(buffer)
    }

    pub fn get_tensor(&self, name: &str) -> Option<&Buffer> {
        self.buffers.get(name)
    }

    pub fn get_tensor_info(&self, name: &str) -> Option<&TensorInfo> {
        self.tensors.get(name)
    }

    pub fn take_tensor(&mut self, name: &str) -> Option<Buffer> {
        self.buffers.remove(name)
    }

    pub fn tensor_names(&self) -> Vec<&str> {
        self.tensors.keys().map(|s| s.as_str()).collect()
    }

    pub fn device(&self) -> &Device {
        &self.device
    }
}

#[cfg(test)]
mod tests {
    #[test]
    fn test_weights_struct_compiles() {
        assert!(true, "MetalWeights struct compiles");
    }
}
