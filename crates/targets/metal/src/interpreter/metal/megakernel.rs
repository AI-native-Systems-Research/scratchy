// SPDX-License-Identifier: Apache-2.0
//! THE DECODE MEGAKERNEL, LAUNCHED: a worker's side of a baked [`MegakernelTape`].
//!
//! Everything was decided at expansion: the kernel's MSL — the whole decode forward, its work
//! split, its grid barriers. At load the worker compiles the tape's library once ([`MK_BODIES`]
//! and the generated source), specializes the kernel with the device's persistent threadgroups
//! `MK_P` and the load's scalars, and fills the address table — the same resolved bindings its
//! argument tables get — at the positions the bake fixed. A forward is then ONE launch.

use std::ops::Range;
use std::time::{Duration, Instant};

use ::objc2::rc::Retained;
use ::objc2::runtime::ProtocolObject;
use ::objc2_metal::{
    MTL4ArgumentTable, MTL4CommandEncoder as _, MTL4ComputeCommandEncoder, MTL4VisibilityOptions,
    MTLComputePipelineState as _, MTLStages,
};

use super::__re::{
    Buffer, ComputePipelineState, Device, MTL4ArgumentTableDescriptor, MTLBuffer, MTLDevice,
    MTLResourceOptions, MTLSize,
};
use super::pipelines::{PipelineLookupError, SpecializedPipelines};
use super::worker::WorkerError;
use crate::tape::constants::ConstantValue;
use crate::tape::lowered::{
    GatedCommand, MK_FC_P, MK_FC_SPIN_LIMIT, MK_THREADS, MegakernelError, MegakernelTape,
    MkLoadSource,
};

/// The adapters' bodies: `shaders/megakernel/megakernel.metal` with its local includes inlined
/// (build.rs). A tape's generated kernel completes it.
pub const MK_BODIES: &str = include_str!(concat!(env!("OUT_DIR"), "/mk_bodies.metal"));

/// Polls one grid-barrier wait may spend before it records a stall and gives up.
const MK_SPIN_LIMIT: u32 = 1 << 20;

/// The launch synchronization block: the stall word alone on its 128-byte line, then one 128-byte
/// line per persistent threadgroup (the grid barriers it has arrived at). Mirrors `MkSync` in
/// `shaders/megakernel/megakernel.metal`.
#[repr(C)]
#[derive(Clone, Copy)]
struct MkSyncBlock {
    stall: [u32; 4],
    line: [u32; 28],
}

/// Bytes of the synchronization block of a launch of `threadgroups`.
fn sync_bytes(threadgroups: usize) -> usize {
    size_of::<MkSyncBlock>() * (1 + threadgroups)
}

/// A bucket's megakernel, as one worker launches it.
pub struct MegakernelBaking {
    /// The expanded commands the launch replaces.
    commands: Range<usize>,
    pipeline: ComputePipelineState,
    table: Retained<ProtocolObject<dyn MTL4ArgumentTable>>,
    /// Persistent threadgroups per launch (`MK_P`).
    threadgroups: usize,
    /// The address table; the argument table points at it.
    addresses: Buffer,
    /// The launch's synchronization block (host-visible: its stall word is read after every
    /// forward).
    sync: Buffer,
}

// Metal handles: retain/release is thread-safe, and a worker is checked out by one forward at a
// time (the same grounds as `Mtl4Step`).
unsafe impl Send for MegakernelBaking {}
unsafe impl Sync for MegakernelBaking {}

/// What one load measured, for the trace.
pub struct MegakernelLoad {
    pub steps: usize,
    pub grid_barriers: u32,
    /// The generated library's compile, when this load compiled it (`None`: an earlier load of
    /// the pool did).
    pub compiled: Option<Duration>,
    pub source_bytes: usize,
    pub pipeline_built: Duration,
    pub max_threads: usize,
    pub threadgroups: usize,
}

/// A `(buffer, offset, binding index)` triple — what the argument tables are filled from.
pub type BoundBinding = (Buffer, u64, u64);

fn mk_error(e: MegakernelError) -> WorkerError {
    WorkerError::Megakernel(e)
}

impl MegakernelBaking {
    /// Compile `tape`'s library and kernel and fill its address table from `commands` (the
    /// bucket's materialized, expanded commands), `baked_of` (each one's baked position) and
    /// `bound` (each one's resolved bindings).
    pub fn bake(
        tape: &MegakernelTape,
        commands: &[GatedCommand],
        baked_of: &[usize],
        bound: &[Vec<BoundBinding>],
        pipelines: &SpecializedPipelines,
        device: &Device,
    ) -> Result<(Self, MegakernelLoad), WorkerError> {
        let lookup = |e: PipelineLookupError| match e {
            PipelineLookupError::Build(e) => mk_error(MegakernelError::Compile(e.to_string())),
            e => WorkerError::PipelineLookup(e),
        };
        let compiled = pipelines
            .megakernel_library(tape.library, || format!("{MK_BODIES}\n{}", tape.source))
            .map_err(lookup)?;
        let threadgroups = crate::device::gpu_cores(device).map_or(1, |c| c.get() as usize);
        // The load's scalars, read from the materialized commands.
        let mut constants = vec![
            ConstantValue::uint(MK_FC_P, threadgroups as u32),
            ConstantValue::uint(MK_FC_SPIN_LIMIT, MK_SPIN_LIMIT),
        ];
        for l in tape.load_constants {
            let at = baked_of.iter().position(|&b| b == l.baked as usize);
            let cmd = at.map(|i| &commands[i].command);
            let value = cmd.and_then(|c| match l.source {
                MkLoadSource::Constant(slot) => c
                    .constants
                    .iter()
                    .find(|v| v.index == slot.get())
                    .map(|v| ConstantValue {
                        index: l.index.get(),
                        ..*v
                    }),
                MkLoadSource::Threadgroups(ax) => {
                    let (x, y, z) = c.dispatch.threadgroups;
                    Some(ConstantValue::uint(l.index, [x, y, z][ax.index()]))
                }
            });
            let symbol = cmd.map_or("?", |c| c.function);
            constants.push(value.ok_or(mk_error(MegakernelError::LoadConstant { symbol }))?);
        }
        let started = Instant::now();
        let pipeline = pipelines
            .megakernel(tape.library, tape.kernel, constants)
            .map_err(lookup)?;
        let pipeline_built = started.elapsed();
        let max_threads = pipeline.maxTotalThreadsPerThreadgroup();
        if max_threads < MK_THREADS as usize {
            return Err(mk_error(MegakernelError::ThreadCap {
                needed: MK_THREADS,
                cap: max_threads as u32,
            }));
        }
        let needed = pipeline.staticThreadgroupMemoryLength();
        let budget = device.maxThreadgroupMemoryLength();
        if needed > budget {
            return Err(mk_error(MegakernelError::ThreadgroupMemory {
                needed: needed as u32,
                budget: budget as u32,
            }));
        }

        // The address table: instance `k` of a step at `table_at + k · row_len`.
        let mut table = vec![0u64; tape.table_len as usize];
        let mut instances = vec![0u32; tape.steps.len()];
        let span = tape.commands.start as usize..tape.commands.end as usize;
        for i in span.clone() {
            let Some(s) = tape
                .steps
                .iter()
                .position(|s| s.baked as usize == baked_of[i])
            else {
                continue;
            };
            let step = &tape.steps[s];
            let found = commands[i].command.function;
            if found != step.function {
                return Err(mk_error(MegakernelError::AdapterMismatch {
                    expected: step.function,
                    found,
                }));
            }
            let row = (step.table_at + instances[s] * step.row_len) as usize;
            instances[s] += 1;
            for (buf, off, idx) in &bound[i] {
                table[row + *idx as usize] = buf.gpuAddress() + off;
            }
        }
        let shared = MTLResourceOptions::StorageModeShared;
        let alloc = |len: usize| {
            device
                .newBufferWithLength_options(len.max(16), shared)
                .expect("newBufferWithLength_options returned nil (megakernel)")
        };
        let addresses = alloc(table.len() * 8);
        let sync = alloc(sync_bytes(threadgroups));
        // SAFETY: fresh shared buffers of at least these lengths, not yet visible to the GPU.
        unsafe {
            let dst = addresses.contents().as_ptr().cast::<u64>();
            std::ptr::copy_nonoverlapping(table.as_ptr(), dst, table.len());
            std::ptr::write_bytes(
                sync.contents().as_ptr().cast::<u8>(),
                0,
                sync_bytes(threadgroups),
            );
        }
        let desc = MTL4ArgumentTableDescriptor::new();
        desc.setMaxBufferBindCount(2);
        let arguments = device
            .newArgumentTableWithDescriptor_error(&desc)
            .expect("newArgumentTableWithDescriptor (megakernel)");
        // SAFETY: a table of 2 slots; both addresses are inside live, resident buffers.
        unsafe {
            arguments.setAddress_atIndex(addresses.gpuAddress(), 0);
            arguments.setAddress_atIndex(sync.gpuAddress(), 1);
        }
        let load = MegakernelLoad {
            steps: instances.iter().sum::<u32>() as usize,
            grid_barriers: tape.grid_barriers,
            compiled,
            source_bytes: MK_BODIES.len() + tape.source.len(),
            pipeline_built,
            max_threads,
            threadgroups,
        };
        let baking = Self {
            commands: span,
            pipeline,
            table: arguments,
            threadgroups,
            addresses,
            sync,
        };
        Ok((baking, load))
    }

    /// The buffers the GPU reads by address, for the worker's residency set.
    pub fn buffers(&self) -> [&Buffer; 2] {
        [&self.addresses, &self.sync]
    }

    /// The expanded commands the launch replaces.
    pub fn commands(&self) -> Range<usize> {
        self.commands.clone()
    }

    /// Encode the launch: a barrier, then the kernel over `MK_P` threadgroups. The caller fences
    /// the next dispatch.
    pub fn encode(&self, enc: &ProtocolObject<dyn MTL4ComputeCommandEncoder>) {
        enc.barrierAfterEncoderStages_beforeEncoderStages_visibilityOptions(
            MTLStages::Dispatch,
            MTLStages::Dispatch,
            MTL4VisibilityOptions::None,
        );
        enc.setArgumentTable(Some(&self.table));
        enc.setComputePipelineState(&self.pipeline);
        let one = |width| MTLSize {
            width,
            height: 1,
            depth: 1,
        };
        enc.dispatchThreadgroups_threadsPerThreadgroup(
            one(self.threadgroups),
            one(MK_THREADS as usize),
        );
    }

    /// After a forward's command buffer completed: a grid barrier that gave up, as a typed error
    /// (the synchronization block is reset for the next forward).
    pub fn take_stall(&self) -> Result<(), MegakernelError> {
        // SAFETY: a shared `MkSyncBlock` the GPU no longer touches (the forward completed).
        let block = unsafe {
            let p = self.sync.contents().as_ptr().cast::<MkSyncBlock>();
            let b = std::ptr::read_volatile(p);
            if b.stall[0] != 0 {
                std::ptr::write_bytes(p.cast::<u8>(), 0, sync_bytes(self.threadgroups));
            }
            b
        };
        match block.stall {
            [0, ..] => Ok(()),
            [site, arrived, 0, of] => Err(MegakernelError::CoResidency {
                site: site - 1,
                arrived,
                of,
            }),
            [site, arrived, ordinal, of] => Err(MegakernelError::Stall {
                site: site - 1,
                ordinal,
                arrived,
                of,
            }),
        }
    }
}
