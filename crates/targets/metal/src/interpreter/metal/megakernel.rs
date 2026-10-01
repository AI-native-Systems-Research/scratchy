// SPDX-License-Identifier: Apache-2.0
//! THE DECODE MEGAKERNEL, LOADED: a worker's side of a baked [`MegakernelTape`].
//!
//! Everything was decided at expansion: the segment kernels' MSL — each a run of the decode
//! forward in which no threadgroup waits on another, its work split in the GPU's cores `MK_P` —
//! compiled at build time into the tape's library ([`MK_BODIES`] and the generated source), as
//! every shader is. At load the worker loads that library once, specializes each segment kernel
//! with the device's cores `MK_P` and the load's scalars, and fills the address table — the same
//! resolved bindings its argument tables get — at the positions the bake fixed. Every launch is
//! then an ordinary dispatch step ([`BucketStep`]): `MK_P` threadgroups of its segment's kernel,
//! binding its instance's block of the table, a barrier before it — played by the worker's
//! dispatch loop like every other step.

use std::collections::HashMap;
use std::time::{Duration, Instant};

use ::objc2_metal::MTLComputePipelineState as _;

use super::__re::{Buffer, Device, MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize};
use super::mtl4::{Mtl4Step, bake_mtl4_steps};
use super::pipelines::{PipelineLookupError, SpecializedPipelines};
use super::worker::{BucketStep, WorkerError};
use crate::tape::constants::ConstantValue;
use crate::tape::lowered::{
    GatedCommand, MK_FC_P, MK_THREADS, MegakernelError, MegakernelTape, MkLoadSource,
};

/// The adapters' bodies: `shaders/megakernel/megakernel.metal` with its local includes inlined
/// (build.rs). A tape's generated kernels complete it; the bake compiles the two together.
pub const MK_BODIES: &str = include_str!(concat!(env!("OUT_DIR"), "/mk_bodies.metal"));

/// A bucket's segmented decode, as one worker plays it: every launch of a forward as a step of
/// the dispatch loop, and the address table they bind (pinned with the worker's buffers).
pub struct Segmented {
    pub steps: Vec<Mtl4Step>,
    pub addresses: Buffer,
}

/// What one load measured, for the trace.
pub struct SegmentedLoad {
    /// Commands the launches play, loops played out.
    pub commands: usize,
    pub kernels: usize,
    pub launches: usize,
    /// The generated library's load, when this load loaded it (`None`: an earlier load of the
    /// pool did).
    pub loaded: Option<Duration>,
    pub metallib_bytes: usize,
    pub pipelines_built: Duration,
    /// The kernels' `maxTotalThreadsPerThreadgroup`, least and most.
    pub max_threads: (usize, usize),
    pub threadgroups: usize,
}

/// A `(buffer, offset, binding index)` triple — what the argument tables are filled from.
pub type BoundBinding = (Buffer, u64, u64);

fn mk_error(e: MegakernelError) -> WorkerError {
    WorkerError::Megakernel(e)
}

/// `tape`'s launches as dispatch steps: its library loaded and each kernel specialized, its
/// address table filled from `commands` (the bucket's materialized, expanded commands),
/// `baked_of` (each one's baked position) and `bound` (each one's resolved bindings). The
/// launches follow the tape's own expansion: one per expanded instance of a segment's opening
/// command.
pub fn bake(
    tape: &MegakernelTape,
    commands: &[GatedCommand],
    baked_of: &[usize],
    bound: &[Vec<BoundBinding>],
    pipelines: &SpecializedPipelines,
    device: &Device,
) -> Result<(Segmented, SegmentedLoad), WorkerError> {
    let lookup = |e: PipelineLookupError| match e {
        PipelineLookupError::Build(e) => mk_error(MegakernelError::Compile(e.to_string())),
        e => WorkerError::PipelineLookup(e),
    };
    let loaded = pipelines
        .megakernel_library(tape.library, tape.metallib)
        .map_err(lookup)?;
    let cores = crate::device::gpu_cores(device).ok_or(mk_error(MegakernelError::NoGpuCores))?;
    let threadgroups = cores.get() as usize;
    // The load's scalars, read from the materialized commands.
    let mut constants = vec![ConstantValue::uint(MK_FC_P, cores.get())];
    for l in tape.load_constants {
        let at = baked_of.iter().position(|&b| b == l.baked as usize);
        let cmd = at.map(|i| &commands[i].command);
        let value = cmd.and_then(|c| match l.source {
            MkLoadSource::Constant(slot) => {
                c.constants
                    .iter()
                    .find(|v| v.index == slot.get())
                    .map(|v| ConstantValue {
                        index: l.index.get(),
                        ..*v
                    })
            }
            MkLoadSource::Threadgroups(ax) => {
                let (x, y, z) = c.dispatch.threadgroups;
                Some(ConstantValue::uint(l.index, [x, y, z][ax.index()]))
            }
        });
        let symbol = cmd.map_or("?", |c| c.function);
        constants.push(value.ok_or(mk_error(MegakernelError::LoadConstant { symbol }))?);
    }
    let budget = device.maxThreadgroupMemoryLength();
    let started = Instant::now();
    let mut max_threads = (usize::MAX, 0);
    let mut kernels = Vec::with_capacity(tape.segments.len());
    for segment in tape.segments {
        let pipeline = pipelines
            .megakernel(tape.library, segment.kernel, constants.clone())
            .map_err(lookup)?;
        let cap = pipeline.maxTotalThreadsPerThreadgroup();
        if cap < MK_THREADS as usize {
            return Err(mk_error(MegakernelError::ThreadCap {
                needed: MK_THREADS,
                cap: cap as u32,
            }));
        }
        let needed = pipeline.staticThreadgroupMemoryLength();
        if needed > budget {
            return Err(mk_error(MegakernelError::ThreadgroupMemory {
                needed: needed as u32,
                budget: budget as u32,
            }));
        }
        max_threads = (max_threads.0.min(cap), max_threads.1.max(cap));
        kernels.push(pipeline);
    }
    let pipelines_built = started.elapsed();

    // The address table: instance `k` of a step at its segment's block `k`, the step's row in it.
    let mut table = vec![0u64; tape.table_len as usize];
    let step_of: HashMap<usize, usize> = (tape.steps.iter().enumerate())
        .map(|(s, step)| (step.baked as usize, s))
        .collect();
    let mut instances = vec![0u32; tape.steps.len()];
    for (i, b) in baked_of.iter().enumerate() {
        let Some(&s) = step_of.get(b) else {
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
        let segment = &tape.segments[step.segment as usize];
        let row = segment.table_at + instances[s] * segment.block_len + step.row_at;
        instances[s] += 1;
        for (buf, off, idx) in &bound[i] {
            table[row as usize + *idx as usize] = buf.gpuAddress() + off;
        }
    }
    let addresses = device
        .newBufferWithLength_options(
            table.len().max(1) * 8,
            MTLResourceOptions::StorageModeShared,
        )
        .expect("newBufferWithLength_options returned nil (megakernel)");
    // SAFETY: a fresh shared buffer of at least this length, not yet visible to the GPU.
    unsafe {
        let dst = addresses.contents().as_ptr().cast::<u64>();
        std::ptr::copy_nonoverlapping(table.as_ptr(), dst, table.len());
    }

    // The launches, in the tape's expanded order: instance `k` of a segment binds block `k`.
    let segment_of: HashMap<usize, usize> = (tape.segments.iter().enumerate())
        .map(|(k, segment)| (segment.opens as usize, k))
        .collect();
    let one = |width| MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let mut launched = vec![0u32; tape.segments.len()];
    let mut steps: Vec<BucketStep> = Vec::new();
    for b in baked_of {
        let Some(&k) = segment_of.get(b) else {
            continue;
        };
        let segment = &tape.segments[k];
        let block = segment.table_at + launched[k] * segment.block_len;
        launched[k] += 1;
        steps.push(BucketStep::Dispatch {
            kernel: None,
            pipeline: kernels[k].clone(),
            direct_bindings: vec![vec![(addresses.clone(), u64::from(block) * 8, 0)]],
            direct_dispatch: vec![(one(threadgroups), one(MK_THREADS as usize))],
            direct_m_scaling: vec![None],
            barrier_before: vec![true],
            runtime_gate: vec![None],
        });
    }
    let load = SegmentedLoad {
        commands: instances.iter().sum::<u32>() as usize,
        kernels: kernels.len(),
        launches: steps.len(),
        loaded,
        metallib_bytes: tape.metallib.len(),
        pipelines_built,
        max_threads,
        threadgroups,
    };
    let steps = bake_mtl4_steps(&steps, device).ok_or(WorkerError::WeightLookupFailed {
        reason: "a segment launch's argument table could not be made",
    })?;
    Ok((Segmented { steps, addresses }, load))
}
