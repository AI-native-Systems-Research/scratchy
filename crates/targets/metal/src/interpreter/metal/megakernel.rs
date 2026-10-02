// SPDX-License-Identifier: Apache-2.0
//! THE DECODE MEGAKERNEL, LOADED: a worker's side of a baked [`MegakernelTape`].
//!
//! The bake compiled every launch a decode forward may play — each region's runs, in which no
//! threadgroup waits on another — at build time into the tape's library ([`MK_BODIES`] and the
//! generated source), as every shader is. At load the worker measures the device
//! ([`DeviceFacts`]), solves which launches play each region and how each run's work splits over
//! the cores ([`split::solve`]), loads the library once, specializes each chosen run's kernel
//! with the device's cores `MK_P`, the load's scalars and its split, and fills the address table
//! — the same resolved bindings its argument tables get — at the positions the bake fixed. Every
//! launch is then an ordinary dispatch step ([`BucketStep`]): `MK_P` threadgroups of a run's
//! kernel binding its region instance's block of the table, or a command's own dispatch kernel,
//! a barrier before each — played by the worker's dispatch loop like every other step.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::time::{Duration, Instant};

use ::objc2_metal::MTLComputePipelineState as _;

use super::__re::{
    Buffer, ComputePipelineState, Device, MTLBuffer, MTLDevice, MTLResourceOptions, MTLSize,
};
use super::mtl4::{Mtl4Step, bake_mtl4_steps};
use super::pipelines::{PipelineLookupError, SpecializedPipelines};
use super::split::{self, Costed, DeviceFacts, RegionPlan, Tile};
use super::worker::{BucketStep, WorkerError};
use crate::tape::constants::ConstantValue;
use crate::tape::lowered::{
    GateCtx, GatedCommand, KernelId, MK_FC_P, MK_THREADS, MScaling, MegakernelError,
    MegakernelTape, MkLoadSource, MkPlace,
};

/// The adapters' bodies: `shaders/megakernel/megakernel.metal` with its local includes inlined
/// (build.rs). A tape's generated kernels complete it; the bake compiles the two together.
pub const MK_BODIES: &str = include_str!(concat!(env!("OUT_DIR"), "/mk_bodies.metal"));

/// A bucket's segmented decode, as one worker plays it: every launch of a forward as a step of
/// the dispatch loop, and the address table the generated ones bind (pinned with the worker's
/// buffers).
pub struct Segmented {
    pub steps: Vec<Mtl4Step>,
    pub addresses: Buffer,
}

/// An expanded command as its own dispatch kernel plays it — what a native launch encodes.
pub struct Native {
    pub kernel: KernelId,
    pub pipeline: ComputePipelineState,
    pub dispatch: (MTLSize, MTLSize),
    pub m_scaling: Option<MScaling>,
}

/// What one load measured and solved, for the trace.
pub struct SegmentedLoad {
    /// Commands the launches play, loops played out.
    pub commands: usize,
    /// Generated kernels the plan plays, and launches per forward (of them, by a command's own
    /// kernel).
    pub kernels: usize,
    pub launches: usize,
    pub natives: usize,
    /// The generated library's load, when this load loaded it (`None`: an earlier load of the
    /// pool did).
    pub loaded: Option<Duration>,
    pub metallib_bytes: usize,
    pub pipelines_built: Duration,
    /// The kernels' `maxTotalThreadsPerThreadgroup`, least and most.
    pub max_threads: (usize, usize),
    pub threadgroups: usize,
    /// The device's facts (this load's measuring of them, `measured`: none when an earlier load
    /// measured everything), and each region's plan and instances per forward.
    pub facts: DeviceFacts,
    pub measured: Duration,
    pub plans: Vec<RegionPlan>,
    pub instances: Vec<u32>,
}

/// A `(buffer, offset, binding index)` triple — what the argument tables are filled from.
pub type BoundBinding = (Buffer, u64, u64);

fn mk_error(e: MegakernelError) -> WorkerError {
    WorkerError::Megakernel(e)
}

/// `tape`'s launches as dispatch steps, solved for `device`: its library loaded and each chosen
/// run's kernel specialized, its address table filled from `commands` (the bucket's materialized,
/// expanded commands), `baked_of` (each one's baked position), `bound` (each one's resolved
/// bindings) and `natives` (each one's own dispatch kernel). The launches follow the tape's own
/// expansion: each expanded instance of a region's opening command plays the region's plan.
pub fn bake(
    tape: &MegakernelTape,
    commands: &[GatedCommand],
    baked_of: &[usize],
    bound: &[Vec<BoundBinding>],
    natives: &[Native],
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

    // The region instances, in the tape's expanded order: each the admitted commands of its steps.
    let ctx = GateCtx::decode_one(false);
    let region_of: HashMap<usize, usize> = (tape.regions.iter().enumerate())
        .map(|(r, region)| (region.opens as usize, r))
        .collect();
    let steps_of: Vec<std::ops::Range<usize>> = (0..tape.regions.len())
        .map(|r| {
            let first = tape.steps.iter().position(|s| s.region as usize == r);
            let len = tape.steps.iter().filter(|s| s.region as usize == r).count();
            first.map_or(0..0, |f| f..f + len)
        })
        .collect();
    let mut instances: Vec<(usize, Vec<usize>)> = Vec::new();
    for i in (0..commands.len()).filter(|&i| commands[i].gate.is_none_or(|g| g.admits(ctx))) {
        if let Some(&r) = region_of.get(&baked_of[i]) {
            instances.push((r, Vec::new()));
        }
        let Some((r, held)) = instances.last_mut() else {
            return Err(mk_error(MegakernelError::AdapterMismatch {
                expected: tape.steps.first().map_or("?", |s| s.function),
                found: commands[i].command.function,
            }));
        };
        let step = tape.steps.get(steps_of[*r].start + held.len());
        let found = commands[i].command.function;
        match step {
            Some(s) if s.function == found && s.baked as usize == baked_of[i] => held.push(i),
            _ => {
                return Err(mk_error(MegakernelError::AdapterMismatch {
                    expected: step.map_or("?", |s| s.function),
                    found,
                }));
            }
        }
    }

    // Each step as the solver weighs it, its grid the first instance's.
    let mut costed: Vec<Option<Costed>> = vec![None; tape.steps.len()];
    for (r, held) in &instances {
        for (j, &i) in held.iter().enumerate() {
            let s = steps_of[*r].start + j;
            if costed[s].is_none() {
                let work = tape.steps[s].work;
                let (x, y, z) = commands[i].command.dispatch.threadgroups;
                let grid = if work.load { (x, y, z) } else { work.grid };
                let vtgs = split::vtgs(&work, grid, cores);
                costed[s] = Some(Costed { work, vtgs });
            }
        }
    }
    let costed: Vec<Costed> = costed
        .into_iter()
        .map(|c| c.expect("the bake made every region from an instance the forward plays"))
        .collect();

    // The device's facts — each calibration measured on synthetic weights — and each region's
    // plan.
    let measuring = Instant::now();
    let facts = DeviceFacts::of(device, pipelines, cores, tape.library, tape.calibrations)
        .map_err(mk_error)?;
    let measured = measuring.elapsed();
    let plans: Vec<RegionPlan> = (tape.regions.iter())
        .map(|region| split::solve(region, &costed, &facts))
        .collect();

    // The load's scalars, read from the materialized commands, and the device's heads.
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
    for s in tape.steps {
        if let Some(h) = s.work.heads {
            let v = ConstantValue::uint(h.slot, split::heads(&s.work, cores));
            if !constants.contains(&v) {
                constants.push(v);
            }
        }
    }

    // Each chosen run's kernel, specialized with its split.
    let budget = device.maxThreadgroupMemoryLength();
    let started = Instant::now();
    let mut max_threads = (usize::MAX, 0);
    let mut kernels: Vec<Vec<Option<ComputePipelineState>>> = Vec::with_capacity(plans.len());
    for (region, plan) in tape.regions.iter().zip(&plans) {
        let mut chosen = Vec::with_capacity(plan.tiles.len());
        for tile in &plan.tiles {
            let Tile::Run { run, split } = tile else {
                chosen.push(None);
                continue;
            };
            let run = &region.runs[*run];
            let mut c = constants.clone();
            c.extend(split.constants(run));
            let pipeline = pipelines
                .megakernel(tape.library, run.kernel, c)
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
            chosen.push(Some(pipeline));
        }
        kernels.push(chosen);
    }
    let pipelines_built = started.elapsed();

    // The address table: instance `k` of a region at its block `k`, each step's row in it.
    let mut table = vec![0u64; tape.table_len as usize];
    let mut count = vec![0u32; tape.regions.len()];
    for (r, held) in &instances {
        let region = &tape.regions[*r];
        let block = region.table_at + count[*r] * region.block_len;
        count[*r] += 1;
        for (j, &i) in held.iter().enumerate() {
            let row = block + tape.steps[steps_of[*r].start + j].row_at;
            for (buf, off, idx) in &bound[i] {
                table[row as usize + *idx as usize] = buf.gpuAddress() + off;
            }
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

    // The launches, in the tape's expanded order: each region instance its plan's tiles.
    let one = |width| MTLSize {
        width,
        height: 1,
        depth: 1,
    };
    let mut launched = vec![0u32; tape.regions.len()];
    let mut steps: Vec<BucketStep> = Vec::new();
    let mut native_launches = 0;
    for (r, held) in &instances {
        let region = &tape.regions[*r];
        let block = region.table_at + launched[*r] * region.block_len;
        launched[*r] += 1;
        for (tile, kernel) in plans[*r].tiles.iter().zip(&kernels[*r]) {
            match (tile, kernel) {
                (Tile::Run { .. }, Some(pipeline)) => steps.push(BucketStep::Dispatch {
                    kernel: None,
                    pipeline: pipeline.clone(),
                    direct_bindings: vec![vec![(addresses.clone(), u64::from(block) * 8, 0)]],
                    direct_dispatch: vec![(one(threadgroups), one(MK_THREADS as usize))],
                    direct_m_scaling: vec![None],
                    barrier_before: vec![true],
                    runtime_gate: vec![None],
                }),
                (Tile::Native { unit, .. }, _) => {
                    let u = region.units[*unit];
                    let start = steps_of[*r].start as u32;
                    for j in u.first - start..u.end - start {
                        let i = held[j as usize];
                        let n = &natives[i];
                        native_launches += 1;
                        steps.push(BucketStep::Dispatch {
                            kernel: Some(n.kernel),
                            pipeline: n.pipeline.clone(),
                            direct_bindings: vec![bound[i].clone()],
                            direct_dispatch: vec![n.dispatch],
                            direct_m_scaling: vec![n.m_scaling],
                            barrier_before: vec![true],
                            runtime_gate: vec![None],
                        });
                    }
                }
                (Tile::Run { .. }, None) => unreachable!("every run tile has its kernel"),
            }
        }
    }
    let load = SegmentedLoad {
        commands: instances.iter().map(|(_, h)| h.len()).sum(),
        kernels: kernels.iter().flatten().flatten().count(),
        launches: steps.len(),
        natives: native_launches,
        loaded,
        metallib_bytes: tape.metallib.len(),
        pipelines_built,
        max_threads,
        threadgroups,
        facts,
        measured,
        plans,
        instances: count,
    };
    let steps = bake_mtl4_steps(&steps, device).ok_or(WorkerError::WeightLookupFailed {
        reason: "a segment launch's argument table could not be made",
    })?;
    Ok((Segmented { steps, addresses }, load))
}

impl SegmentedLoad {
    /// What the load measured and chose, for the trace: the device's facts, then per region its
    /// launches — each a run's steps (where each runs, a spread step's virtual threadgroups per
    /// item) or a unit's own dispatch kernels — with what each costs, beside what the region's
    /// longest runs and every command's own kernel would.
    pub fn explain(&self, tape: &MegakernelTape) -> String {
        let f = &self.facts;
        let mut out = String::new();
        let _ = writeln!(
            out,
            "[split] device: {} cores, bare launch boundary {}, dependent step {} (this load \
             measured for {:?})",
            f.cores.get(),
            f.launch,
            f.step,
            self.measured,
        );
        for (c, s) in tape.calibrations.iter().zip(&f.streams) {
            let lane: Vec<String> = s.lane.iter().map(ToString::to_string).collect();
            let _ = writeln!(
                out,
                "[split]   {} (k {}, {} rows of {}-bit codes per virtual threadgroup): GB/s one \
                 core streams by virtual threadgroups per item 1..{}: {}; its own kernel, every \
                 core: {}; a launch boundary streaming it: {}",
                c.lane,
                c.k,
                c.rows,
                c.bits,
                c.widest,
                lane.join(" "),
                s.native,
                s.boundary,
            );
        }
        let forward = |cost: &dyn Fn(&RegionPlan) -> split::Seconds| {
            let each = self.plans.iter().zip(&self.instances);
            let total: f64 = each.map(|(p, &n)| cost(p).0 * f64::from(n)).sum();
            format!("{:.2} ms", total * 1e3)
        };
        let _ = writeln!(
            out,
            "[split] a decode forward, as solved: {} (its longest runs: {}; every command its own \
             kernel: {})",
            forward(&RegionPlan::seconds),
            forward(&|p| p.longest),
            forward(&|p| p.dispatch),
        );
        for (r, (region, plan)) in tape.regions.iter().zip(&self.plans).enumerate() {
            let _ = writeln!(
                out,
                "[split] region {r} (opens baked {}, x{} per forward): {} launches (the dataflow \
                 requires {}), {} (longest runs {}, every command its own kernel {})",
                region.opens,
                self.instances[r],
                plan.tiles.len(),
                region.required,
                plan.seconds(),
                plan.longest,
                plan.dispatch,
            );
            for tile in &plan.tiles {
                let units = tile.units(region);
                let mut parts = Vec::new();
                let mut ord = 0;
                for u in units.clone() {
                    let unit = region.units[u as usize];
                    let names: Vec<&str> = (unit.first..unit.end)
                        .map(|s| tape.steps[s as usize].function)
                        .collect();
                    let how = match tile {
                        Tile::Native { .. } => "own kernel".to_string(),
                        Tile::Run { run, split } => {
                            match region.runs[*run].places[(u - units.start) as usize] {
                                MkPlace::Spread => {
                                    ord += 1;
                                    format!("spread, {} vtgs/item", split.k[ord - 1])
                                }
                                MkPlace::Lane(g) => format!("lane {}", split.lanes[g as usize]),
                                MkPlace::Everywhere => "every core".to_string(),
                            }
                        }
                    };
                    parts.push(format!("{} ({how})", names.join(" > ")));
                }
                let _ = writeln!(out, "[split]   {}: {}", tile.seconds(), parts.join(" | "));
            }
        }
        out
    }
}
