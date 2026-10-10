// SPDX-License-Identifier: Apache-2.0

//! `select_rows` — a speculative step's inputs that depend on how many drafts the step before it
//! kept, picked on the device (`shaders/select_rows.metal`): the host lays the step out while that
//! one still runs, so it writes every outcome ([`Selection`]) and the device copies the one that
//! happened, at the head of the step's command buffer. Baked for a model that drafts
//! ([`crate::off_tape`]).

use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer as _, MTLSize};

use crate::off_tape::{OffTapeKernels, OffTapePipeline};
use crate::shader_cache::Device;
use crate::stream::MetalStreamError;

pub struct SelectRowsKernel {
    pub pipeline: OffTapePipeline,
}

impl SelectRowsKernel {
    /// `None` for a model that does not draft.
    pub fn new(
        device: &Device,
        kernels: &OffTapeKernels,
    ) -> Result<Option<Self>, MetalStreamError> {
        (kernels.select_rows.as_ref())
            .map(|k| {
                Ok(Self {
                    pipeline: OffTapePipeline::new(device, k)?,
                })
            })
            .transpose()
    }
}

/// One copy the device picks: `len` words of `table` from word `*selector · stride` — variants
/// `stride` words apart, overlapping where `stride < len` — to `at` bytes into `to`.
pub struct Selection {
    pub table: Vec<u32>,
    pub stride: usize,
    pub len: usize,
    /// The variant's index: a `u32` `.1` bytes into `.0`, which an earlier command buffer wrote.
    pub selector: (crate::mtl4_dispatch::Buffer, usize),
    pub to: SelectInto,
    pub at: usize,
}

/// Where a [`Selection`] lands.
pub enum SelectInto {
    /// A runtime input of the forward.
    Input(crate::interpreter::metal::DeviceInputInto),
    /// A buffer outside the runtime inputs that the forward's command buffer reads.
    Buffer(crate::mtl4_dispatch::Buffer),
}

/// The kernel's op, as `select_rows.metal` lays it out: the table `src_at` bytes into the staged
/// region, the destination and selector by address.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct SelectOp {
    pub src_at: u32,
    pub stride: u32,
    pub len: u32,
    pub _pad: u32,
    pub dst: u64,
    pub sel: u64,
}

impl SelectOp {
    pub fn new(src_at: usize, s: &Selection, dst: u64) -> Self {
        let words = |n: usize| u32::try_from(n).expect("a selection fits u32 words");
        Self {
            src_at: words(src_at),
            stride: words(s.stride),
            len: words(s.len),
            _pad: 0,
            dst,
            sel: s.selector.0.gpuAddress() + s.selector.1 as u64,
        }
    }

    /// The op's bytes, for the staged region.
    pub fn bytes(&self) -> [u8; size_of::<Self>()] {
        // SAFETY: `SelectOp` is `repr(C)` plain data with no padding bytes left uninitialized.
        unsafe { std::mem::transmute(*self) }
    }
}

/// Encode the ops at `ops` (in the staged region at `region`) onto `encoder`, behind a barrier on
/// the input writes before them; the forward's dispatches wait on them with the barrier after.
pub fn encode_select_rows(
    kernel: &crate::shader_cache::ComputePipelineState,
    encoder: &ProtocolObject<dyn objc2_metal::MTL4ComputeCommandEncoder>,
    device: &Device,
    (region, ops): (u64, u64),
    count: usize,
) {
    use objc2_metal::{
        MTL4CommandEncoder as _, MTL4ComputeCommandEncoder as _, MTL4VisibilityOptions, MTLStages,
    };
    encoder.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
        MTLStages::Blit | MTLStages::Dispatch,
        MTLStages::Dispatch,
        MTL4VisibilityOptions::Device,
    );
    let table = crate::mtl4_dispatch::build_arg_table(device, &[(ops, 0), (region, 1)], region);
    encoder.setComputePipelineState(kernel);
    encoder.setArgumentTable(Some(&table));
    let size = |width| MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    encoder.dispatchThreadgroups_threadsPerThreadgroup(size(count), size(64));
}
