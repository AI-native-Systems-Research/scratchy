// SPDX-License-Identifier: Apache-2.0
//! THE LAUNCH SPLIT, SOLVED FOR THE DEVICE AT LOAD.
//!
//! The bake leaves every launch a decode forward may play — each region's runs ([`MkRun`]),
//! beside each command's own dispatch kernel — and each step's work in the tape's own terms
//! ([`MkWork`]). What those cost differs from GPU to GPU, so it is measured HERE, on the GPU the
//! forward runs on, with synthetic work — no model, no weights ([`DeviceFacts`]) — and the plan is
//! solved from it: per region, the cheapest tiling of its units by runs and dispatch kernels
//! ([`cheapest_tiling`], exact), each run's work split over the cores chosen to finish its busiest
//! core first ([`split_run`]).
//!
//! # The costs
//!
//! A launch costs a dependent launch boundary — a bare launch's ([`DeviceFacts::launch`]), or
//! for a launch streaming weights its body's draining and filling ([`StreamFacts::boundary`]) —
//! plus its busiest core's work. A spread step's items go round-robin over the cores; an item of `K` virtual threadgroups
//! of a streaming step costs its weight bytes at the rate ONE core streams its body's calibration
//! ([`MkCalibration`]: the matvec body on synthetic weights at the step's row length) with items
//! of `K` while every core does ([`StreamFacts::lane`]); every other item, and every step of a
//! pinned or copied unit, costs at least a dependent step inside one threadgroup
//! ([`DeviceFacts::step`]). A step played by its own dispatch kernel costs a launch boundary plus
//! its weight bytes at the rate the whole GPU streams the calibration one virtual threadgroup per
//! threadgroup ([`StreamFacts::native`]), or a dependent step per wave of its threadgroups.

use std::collections::hash_map::Entry;
use std::collections::{BTreeSet, HashMap};
use std::fmt;
use std::ops::{Add, Range};
use std::sync::{Mutex, OnceLock};

use ::objc2_metal::MTLDevice as _;
use scratchy_subtile::megakernel_plan::cheapest_tiling;

use super::__re::{ComputePipelineState, Device, MTLBuffer as _, MTLResourceOptions, MTLSize};
use super::pipelines::{PipelineLookupError, SpecializedPipelines};
use crate::mtl4_dispatch::{Buffer, Mtl4DispatchBatch, shared_slice, shared_zeroed};
use crate::tape::constants::{ConstSlot, ConstantValue};
use crate::tape::ids::{GpuCores, HeadDim, NumKvHeads, NumQHeads, TqDecodeHeads};
use crate::tape::lowered::{
    MK_FC_CAL, MK_THREADS, MegakernelError, MkCalibration, MkPlace, MkRegion, MkRun, MkWork,
};

/// Time on the GPU.
#[derive(Clone, Copy, Default, PartialEq, PartialOrd, Debug)]
pub struct Seconds(pub f64);

impl Add for Seconds {
    type Output = Self;
    fn add(self, o: Self) -> Self {
        Self(self.0 + o.0)
    }
}

impl Seconds {
    fn max(self, o: Self) -> Self {
        if o > self { o } else { self }
    }
}

impl fmt::Display for Seconds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1} us", self.0 * 1e6)
    }
}

/// A streaming rate.
#[derive(Clone, Copy, PartialEq, PartialOrd, Debug)]
pub struct BytesPerSecond(pub f64);

impl BytesPerSecond {
    fn time(self, bytes: f64) -> Seconds {
        Seconds(bytes / self.0)
    }
}

impl fmt::Display for BytesPerSecond {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{:.1}", self.0 / 1e9)
    }
}

/// What the split's costs are, on one GPU — each measured with synthetic work.
#[derive(Clone, Debug)]
pub struct DeviceFacts {
    /// The threadgroups a generated launch runs: one per core.
    pub cores: GpuCores,
    /// A dependent launch boundary before a launch streaming nothing: a barrier, then a launch (a
    /// chain of launches of a generated launch's shape that each bump what the last wrote). A
    /// streaming launch's is its body's ([`StreamFacts::boundary`]).
    pub launch: Seconds,
    /// A dependent step inside one threadgroup, as small as a norm — the least any step costs.
    pub step: Seconds,
    /// Per calibration of the tape ([`crate::tape::lowered::MegakernelTape::calibrations`]),
    /// its streaming rates.
    pub streams: Vec<StreamFacts>,
}

/// How fast the device streams one calibration's matvec body.
#[derive(Clone, Debug)]
pub struct StreamFacts {
    /// Bytes per second ONE core streams with items of `K` virtual threadgroups (at `K − 1`), one
    /// threadgroup per core taking items round-robin as a run spreads a step.
    pub lane: Vec<BytesPerSecond>,
    /// Bytes per second the whole GPU streams one virtual threadgroup per threadgroup, as the
    /// step's own dispatch kernel.
    pub native: BytesPerSecond,
    /// What a launch boundary costs a launch streaming this body: its stream split into dependent
    /// launches of its own kernel, against one launch of it all, per boundary — the previous
    /// launch draining, this one filling.
    pub boundary: Seconds,
}

/// Weight bytes a streaming calibration reads per pass: far beyond any Apple GPU's last-level
/// cache, so every pass streams from DRAM.
const STREAM_BYTES: u64 = 128 << 20;
/// GPU time the first calibration streams untimed before any is measured; dependent steps run
/// untimed before the first launch and step are.
const WARM: Seconds = Seconds(0.05);
const WARM_STEPS: u32 = 25_000;
/// Passes per timed batch, and timed batches per measurement (the fastest counts).
const PASSES: usize = 2;
const TRIALS: usize = 2;
/// Dependent launches a calibration's stream is split into to time a launch boundary.
const SPLITS: u32 = 64;
/// Timed batches of each, the fastest counting.
const SPLIT_TRIALS: usize = 4;
/// Dependent launches, and dependent steps in one threadgroup, per timed chain.
const CHAIN_LAUNCHES: usize = 512;
const CHAIN_STEPS: u32 = 2048;

const LIBRARY: &str = "device_facts";

/// What this process measured on one device: its launch boundary and dependent step, and each
/// calibration's rates by `(library, kernel)` — each measured once, by the first load needing it.
struct Measured {
    launch: Seconds,
    step: Seconds,
    streams: HashMap<(&'static str, &'static str), StreamFacts>,
}

/// Every device's measurements, by the device's registry id.
fn measured() -> &'static Mutex<HashMap<u64, Measured>> {
    static FACTS: OnceLock<Mutex<HashMap<u64, Measured>>> = OnceLock::new();
    FACTS.get_or_init(Mutex::default)
}

fn facts_error(e: impl fmt::Display) -> MegakernelError {
    MegakernelError::DeviceFacts(e.to_string())
}

impl DeviceFacts {
    /// `device`'s facts for a tape whose calibrations, in `library`, are `calibrations` — each
    /// measured the first time a load needs it.
    pub fn of(
        device: &Device,
        pipelines: &SpecializedPipelines,
        cores: GpuCores,
        library: &'static str,
        calibrations: &[MkCalibration],
    ) -> Result<Self, MegakernelError> {
        let mut all = measured().lock().map_err(facts_error)?;
        let id = device.registryID();
        let mut probe: Option<Probe> = None;
        let known = match all.entry(id) {
            Entry::Occupied(e) => e.into_mut(),
            Entry::Vacant(e) => {
                let p = probe.insert(Probe::new(device, pipelines, cores)?);
                p.chain(WARM_STEPS)?;
                let (launch, step) = (p.launch()?, p.step()?);
                e.insert(Measured {
                    launch,
                    step,
                    streams: HashMap::new(),
                })
            }
        };
        let mut streams = Vec::with_capacity(calibrations.len());
        for c in calibrations {
            let facts = match known.streams.entry((library, c.lane)) {
                Entry::Occupied(e) => e.into_mut(),
                Entry::Vacant(e) => {
                    let p = match probe {
                        Some(ref p) => p,
                        None => probe.insert(Probe::new(device, pipelines, cores)?),
                    };
                    e.insert(p.stream(library, c)?)
                }
            };
            streams.push(facts.clone());
        }
        Ok(Self {
            cores,
            launch: known.launch,
            step: known.step,
            streams,
        })
    }
}

/// The synthetic kernels' measurements.
struct Probe<'a> {
    device: &'a Device,
    pipelines: &'a SpecializedPipelines,
    cores: GpuCores,
    sink: Buffer,
    /// Whether the GPU has streamed long enough to run as a decode keeps it (clocks up, the
    /// weights' pages mapped).
    warm: std::cell::Cell<bool>,
}

impl<'a> Probe<'a> {
    fn new(
        device: &'a Device,
        pipelines: &'a SpecializedPipelines,
        cores: GpuCores,
    ) -> Result<Self, MegakernelError> {
        Ok(Self {
            device,
            pipelines,
            cores,
            sink: shared_zeroed(device, 16),
            warm: std::cell::Cell::new(false),
        })
    }

    fn pipeline(
        &self,
        library: &'static str,
        function: &'static str,
        constants: Vec<ConstantValue>,
    ) -> Result<ComputePipelineState, MegakernelError> {
        self.pipelines
            .megakernel(library, function, constants)
            .map_err(|e: PipelineLookupError| facts_error(e))
    }

    /// The fastest of `trials` batches `encode` fills, each timed by the GPU (its start to its end).
    fn fastest(
        &self,
        trials: usize,
        mut encode: impl FnMut(&mut Mtl4DispatchBatch),
    ) -> Result<Seconds, MegakernelError> {
        let mut best = f64::INFINITY;
        for _ in 0..trials {
            let mut batch = Mtl4DispatchBatch::begin(self.device)
                .ok_or_else(|| facts_error("no Metal 4 command queue"))?;
            encode(&mut batch);
            let gpu = batch.try_commit().map_err(facts_error)?;
            best = best.min(gpu.as_secs_f64());
        }
        Ok(Seconds(best))
    }

    /// A dependent launch boundary: a chain of launches of a generated launch's shape.
    fn launch(&self) -> Result<Seconds, MegakernelError> {
        let pso = self.pipeline(LIBRARY, "df_launch", Vec::new())?;
        let chain = self.fastest(5, |batch| {
            for i in 0..CHAIN_LAUNCHES {
                if i > 0 {
                    batch.barrier();
                }
                batch.encode(
                    &pso,
                    &[(&self.sink, 0)],
                    &[],
                    &[],
                    &[],
                    size(self.cores.get()),
                    size(MK_THREADS),
                );
            }
        })?;
        Ok(Seconds(chain.0 / CHAIN_LAUNCHES as f64))
    }

    /// A dependent step inside one threadgroup.
    fn step(&self) -> Result<Seconds, MegakernelError> {
        let mut best = f64::INFINITY;
        for _ in 0..5 {
            best = best.min(self.chain(CHAIN_STEPS)?.0);
        }
        Ok(Seconds(best / f64::from(CHAIN_STEPS)))
    }

    /// `steps` dependent steps inside one threadgroup, in one launch: how long they ran.
    fn chain(&self, steps: u32) -> Result<Seconds, MegakernelError> {
        let pso = self.pipeline(LIBRARY, "df_chain", Vec::new())?;
        let words = shared_zeroed(self.device, MK_THREADS as usize * 4);
        self.fastest(1, |batch| {
            batch.encode(
                &pso,
                &[(&words, 0)],
                &[(steps, 1)],
                &[],
                &[],
                size(1),
                size(MK_THREADS),
            );
        })
    }

    /// A shared buffer of `len` bytes whose every page is written (so no read is served by a
    /// shared zero page).
    fn touched(&self, len: u64) -> Result<Buffer, MegakernelError> {
        let len = len.max(16) as usize;
        let buf = (self.device)
            .newBufferWithLength_options(len, MTLResourceOptions::StorageModeShared)
            .ok_or_else(|| facts_error("a synthetic weight buffer could not be allocated"))?;
        // SAFETY: a fresh shared buffer of `len` bytes, not yet visible to the GPU.
        unsafe {
            let bytes = buf.contents().as_ptr().cast::<u8>();
            for at in (0..len).step_by(4096) {
                *bytes.add(at) = (at / 4096) as u8 | 1;
            }
        }
        Ok(buf)
    }

    /// How fast the device streams calibration `c` (in `library`) over synthetic weights: per
    /// items of `K = 1..=widest` virtual threadgroups round-robin over one threadgroup per core
    /// (the busiest core's bytes over the pass), and as its own dispatch kernel (every byte over
    /// the pass).
    fn stream(
        &self,
        library: &'static str,
        c: &MkCalibration,
    ) -> Result<StreamFacts, MegakernelError> {
        let row = u64::from(c.k * c.bits / 8 + 2 * c.k.div_ceil(c.group) * c.scale_bytes);
        let vtg_bytes = u64::from(c.rows) * row;
        let slices = u64::from(SPLITS);
        let vtgs = (STREAM_BYTES / vtg_bytes / slices).max(1) * slices;
        let vtgs = u32::try_from(vtgs).map_err(facts_error)?;
        let n = vtgs * c.rows;
        let w_row = u64::from(c.k * c.bits / 8);
        let s_row = u64::from(c.k.div_ceil(c.group) * c.scale_bytes);
        let at = c.bindings;
        // Every buffer the body binds, at its binding, and how far a row moves it.
        let mut bound = vec![
            (at.weights, self.touched(u64::from(n) * w_row)?, w_row),
            (at.input, self.touched(u64::from(c.k * c.act_bytes))?, 0),
            (
                at.output,
                self.touched(u64::from(n * c.act_bytes))?,
                u64::from(c.act_bytes),
            ),
        ];
        for slot in [at.scales, at.biases].into_iter().flatten() {
            bound.push((slot, self.touched(u64::from(n) * s_row)?, s_row));
        }
        let addresses = |first: u64| {
            let mut a = vec![
                0u64;
                bound
                    .iter()
                    .map(|(i, ..)| usize::from(*i) + 1)
                    .max()
                    .unwrap_or(0)
            ];
            for (i, buf, row) in &bound {
                a[usize::from(*i)] = buf.gpuAddress() + first * row;
            }
            a
        };
        let table = shared_slice(self.device, &addresses(0));
        let resident: Vec<&Buffer> = bound.iter().map(|(_, b, _)| b).collect();
        let n_slot = MK_FC_CAL;
        let batch = |pso: &ComputePipelineState, k: u32, grid: u32, tpg: MTLSize, trials| {
            self.fastest(trials, |batch| {
                for p in 0..PASSES {
                    if p > 0 {
                        batch.barrier();
                    }
                    batch.encode(
                        pso,
                        &[(&table, 0)],
                        &[(k, 1)],
                        &[],
                        &resident,
                        size(grid),
                        tpg,
                    );
                }
            })
        };
        let pso = self.pipeline(library, c.native, vec![ConstantValue::uint(n_slot, n)])?;
        let (tx, ty, tz) = c.tpg;
        let tpg = MTLSize {
            width: tx as usize,
            height: ty as usize,
            depth: tz as usize,
        };
        // The GPU streaming first, as a decode keeps it, untimed: a batch — and before the
        // probe's first measurement, at least `WARM` of streaming.
        let mut warmed = batch(&pso, 1, vtgs, tpg, 1)?;
        if !self.warm.replace(true) {
            while warmed < WARM {
                warmed = warmed + batch(&pso, 1, vtgs, tpg, 1)?;
            }
        }
        let pass = |pso: &ComputePipelineState, k: u32, grid: u32, tpg: MTLSize| {
            let secs = batch(pso, k, grid, tpg, TRIALS)?;
            Ok::<f64, MegakernelError>(secs.0.max(f64::MIN_POSITIVE) / PASSES as f64)
        };
        let secs = pass(&pso, 1, vtgs, tpg)?;
        let native = BytesPerSecond(f64::from(vtgs) * vtg_bytes as f64 / secs);
        // The stream in one launch, and split into `SPLITS` dependent launches of a slice each.
        let slice = vtgs / SPLITS;
        let sliced = vec![ConstantValue::uint(n_slot, slice * c.rows)];
        let sliced = self.pipeline(library, c.native, sliced)?;
        let tables: Vec<Buffer> = (0..SPLITS)
            .map(|i| shared_slice(self.device, &addresses(u64::from(i * slice * c.rows))))
            .collect();
        let one = self.fastest(SPLIT_TRIALS, |batch| {
            batch.encode(
                &pso,
                &[(&table, 0)],
                &[(1, 1)],
                &[],
                &resident,
                size(vtgs),
                tpg,
            );
        })?;
        let split = self.fastest(SPLIT_TRIALS, |batch| {
            for (i, t) in tables.iter().enumerate() {
                if i > 0 {
                    batch.barrier();
                }
                batch.encode(
                    &sliced,
                    &[(t, 0)],
                    &[(1, 1)],
                    &[],
                    &resident,
                    size(slice),
                    tpg,
                );
            }
        })?;
        let boundary = Seconds(((split.0 - one.0) / f64::from(SPLITS - 1)).max(0.0));
        let cores = self.cores.get();
        let pso = self.pipeline(library, c.lane, vec![ConstantValue::uint(n_slot, n)])?;
        let mut lane = Vec::with_capacity(c.widest as usize);
        for k in 1..=c.widest {
            let secs = pass(&pso, k, cores, size(MK_THREADS))?;
            let busiest = vtgs.div_ceil(k).div_ceil(cores) * k;
            lane.push(BytesPerSecond(f64::from(busiest) * vtg_bytes as f64 / secs));
        }
        Ok(StreamFacts {
            lane,
            native,
            boundary,
        })
    }
}

fn size(width: u32) -> MTLSize {
    MTLSize {
        width: width as usize,
        height: 1,
        depth: 1,
    }
}

/// Every thread count a streaming step of `work` may play an item with.
pub fn lane_threads(work: &MkWork) -> impl Iterator<Item = u32> + '_ {
    (1..=work.widest).map(|k| k * work.vtg_threads)
}

/// A step's virtual threadgroups on this load: its grid (the materialized command's, where the
/// load sizes it), one per head group where a decode attention groups its heads.
pub fn vtgs(work: &MkWork, grid: (u32, u32, u32), cores: GpuCores) -> u32 {
    let h = heads(work, cores);
    grid.0 * (grid.1 / h) * grid.2
}

/// A decode attention's heads per virtual threadgroup on `cores`: the device's pick.
pub fn heads(work: &MkWork, cores: GpuCores) -> u32 {
    work.heads.map_or(1, |h| {
        TqDecodeHeads::for_group(HeadDim(h.head_dim), NumQHeads(h.q), NumKvHeads(h.kv), cores).get()
    })
}

/// A step as the solver weighs it: its work and its virtual threadgroups on this load.
#[derive(Clone, Copy, Debug)]
pub struct Costed {
    pub work: MkWork,
    pub vtgs: u32,
}

impl Costed {
    /// One item of `k` virtual threadgroups on one core.
    fn item(&self, k: u32, facts: &DeviceFacts) -> Seconds {
        match self.work.stream {
            None => facts.step,
            Some(st) => {
                let lane = &facts.streams[st.calibration as usize].lane;
                let rate = lane[(k.clamp(1, lane.len() as u32) - 1) as usize];
                facts
                    .step
                    .max(rate.time(f64::from(k) * f64::from(st.bytes)))
            }
        }
    }

    /// Every item, one threadgroup playing them all at its widest.
    fn whole(&self, facts: &DeviceFacts) -> Seconds {
        let k = self.vtgs.min(self.work.widest).max(1);
        let one = self.item(k, facts);
        Seconds(one.0 * f64::from(self.vtgs.div_ceil(k)))
    }

    /// The launch boundary before a launch that streams this step.
    fn boundary(&self, facts: &DeviceFacts) -> Seconds {
        match self.work.stream {
            None => facts.launch,
            Some(st) => facts
                .launch
                .max(facts.streams[st.calibration as usize].boundary),
        }
    }

    /// Its own dispatch kernel: a launch boundary, then its grid.
    fn native(&self, facts: &DeviceFacts) -> Seconds {
        let t = self.work.vtg_threads;
        let waves = self
            .vtgs
            .div_ceil(facts.cores.get() * (MK_THREADS / t).max(1));
        let floor = Seconds(facts.step.0 * f64::from(waves));
        let run = match self.work.stream {
            None => floor,
            Some(st) => {
                let rate = facts.streams[st.calibration as usize].native;
                floor.max(rate.time(f64::from(self.vtgs) * f64::from(st.bytes)))
            }
        };
        self.boundary(facts) + run
    }
}

/// A run's work split over the launch's threadgroups: per spread unit (in order) its items'
/// virtual threadgroups `k` and its first item's cursor `c`; per lane group its threadgroup; and
/// what the launch costs.
#[derive(Clone, Debug, PartialEq)]
pub struct RunSplit {
    pub k: Vec<u32>,
    pub c: Vec<u32>,
    pub lanes: Vec<u32>,
    pub seconds: Seconds,
}

impl RunSplit {
    /// The run's split as its kernel's function constants ([`MkRun::split_at`]).
    pub fn constants(&self, run: &MkRun) -> Vec<ConstantValue> {
        let at = run.split_at.get();
        let spread = self.k.iter().zip(&self.c).enumerate();
        let mut out: Vec<ConstantValue> = spread
            .flat_map(|(i, (&k, &c))| {
                let i = i as u16;
                [
                    ConstantValue::uint(ConstSlot(at + 2 * i), k),
                    ConstantValue::uint(ConstSlot(at + 2 * i + 1), c),
                ]
            })
            .collect();
        let groups = (2 * self.k.len()) as u16;
        out.extend(
            (self.lanes.iter().enumerate())
                .map(|(g, &l)| ConstantValue::uint(ConstSlot(at + groups + g as u16), l)),
        );
        out
    }
}

/// `run`'s work split over the device's cores, and its cost: every spread streaming step plays
/// items of the same threads per core `t` (as many virtual threadgroups as `t` holds, at least
/// one, at most its widest), every other spread step its widest; items go round-robin, the
/// cursor continuing from step to step; each lane group takes the least loaded threadgroup,
/// heaviest first; copied units load every threadgroup. The `t` whose busiest threadgroup
/// finishes first wins, the fewest threads on a tie.
pub fn split_run(
    region: &MkRegion,
    run: &MkRun,
    steps: &[Costed],
    facts: &DeviceFacts,
) -> RunSplit {
    let p = facts.cores.get() as usize;
    let units = &region.units[run.first as usize..run.end as usize];
    let unit_steps = |u: usize| &steps[units[u].first as usize..units[u].end as usize];
    // A pinned or copied unit's work: each step whole on its threadgroup.
    let held = |u: usize| {
        (unit_steps(u).iter())
            .map(|s| s.whole(facts))
            .fold(Seconds(0.0), Add::add)
    };
    let mut copies = Seconds(0.0);
    let mut groups = vec![
        Seconds(0.0);
        run.places
            .iter()
            .filter_map(lane_group)
            .max()
            .map_or(0, |g| g + 1) as usize
    ];
    let mut spread: Vec<Costed> = Vec::new();
    for (u, place) in run.places.iter().enumerate() {
        match *place {
            MkPlace::Spread => spread.push(unit_steps(u)[0]),
            MkPlace::Lane(g) => groups[g as usize] = groups[g as usize] + held(u),
            MkPlace::Everywhere => copies = copies + held(u),
        }
    }
    let mut widths: BTreeSet<u32> = (spread.iter())
        .filter(|s| s.work.stream.is_some())
        .flat_map(|s| lane_threads(&s.work))
        .collect();
    if widths.is_empty() {
        widths.insert(0);
    }
    // The launch's boundary: its streaming steps' (the dearest), or a bare launch's.
    let boundary = (spread.iter())
        .map(|s| s.boundary(facts))
        .fold(facts.launch, Seconds::max);
    let mut order: Vec<usize> = (0..groups.len()).collect();
    order.sort_by(|&a, &b| {
        groups[b]
            .partial_cmp(&groups[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let mut best: Option<RunSplit> = None;
    for t in widths {
        let k: Vec<u32> = (spread.iter())
            .map(|s| match s.work.stream {
                None => s.work.widest,
                Some(_) => (t / s.work.vtg_threads).clamp(1, s.work.widest),
            })
            .collect();
        let mut load = vec![copies; p];
        let mut cursor = 0u32;
        let mut c = Vec::with_capacity(spread.len());
        for (s, &k) in spread.iter().zip(&k) {
            let n = s.vtgs.div_ceil(k);
            let one = s.item(k, facts);
            let (base, extra) = (n / p as u32, n as usize % p);
            for (l, w) in load.iter_mut().enumerate() {
                let at = (l + p - cursor as usize % p) % p;
                let mine = base + u32::from(at < extra);
                *w = *w + Seconds(one.0 * f64::from(mine));
            }
            c.push(cursor);
            cursor += n;
        }
        let mut lanes = vec![0u32; groups.len()];
        for &g in &order {
            let (l, _) = (load.iter().enumerate())
                .min_by(|a, b| a.1.partial_cmp(b.1).unwrap_or(std::cmp::Ordering::Equal))
                .expect("a launch has a threadgroup");
            load[l] = load[l] + groups[g];
            lanes[g] = l as u32;
        }
        let busiest = load.iter().copied().fold(Seconds(0.0), Seconds::max);
        let split = RunSplit {
            k,
            c,
            lanes,
            seconds: boundary + busiest,
        };
        if best.as_ref().is_none_or(|b| split.seconds < b.seconds) {
            best = Some(split);
        }
    }
    best.expect("at least one width is weighed")
}

fn lane_group(p: &MkPlace) -> Option<u32> {
    match *p {
        MkPlace::Lane(g) => Some(g),
        _ => None,
    }
}

/// How a region's launches play it.
#[derive(Clone, Debug, PartialEq)]
pub enum Tile {
    /// Run `run` of the region, split as `split`.
    Run { run: usize, split: RunSplit },
    /// Unit `unit` of the region, each of its steps by its own dispatch kernel.
    Native { unit: usize, seconds: Seconds },
}

impl Tile {
    pub fn seconds(&self) -> Seconds {
        match self {
            Self::Run { split, .. } => split.seconds,
            Self::Native { seconds, .. } => *seconds,
        }
    }

    pub fn units(&self, region: &MkRegion) -> Range<u32> {
        match self {
            Self::Run { run, .. } => region.runs[*run].first..region.runs[*run].end,
            Self::Native { unit, .. } => *unit as u32..*unit as u32 + 1,
        }
    }
}

/// A region's plan: its tiles, and — for the trace — what its longest runs and every command's
/// own dispatch kernel would cost.
#[derive(Clone, Debug)]
pub struct RegionPlan {
    pub tiles: Vec<Tile>,
    pub longest: Seconds,
    pub dispatch: Seconds,
}

impl RegionPlan {
    pub fn seconds(&self) -> Seconds {
        self.tiles
            .iter()
            .map(Tile::seconds)
            .fold(Seconds(0.0), Add::add)
    }
}

/// The cheapest way to play `region` on this device: every run a candidate at its best split,
/// every unit a candidate by its own dispatch kernels; the tiling minimizing the total.
/// `steps` are the tape's steps ([`crate::tape::lowered::MegakernelTape::steps`]) as weighed on this load.
pub fn solve(region: &MkRegion, steps: &[Costed], facts: &DeviceFacts) -> RegionPlan {
    let mut tiles: Vec<Tile> = (0..region.runs.len())
        .map(|run| Tile::Run {
            run,
            split: split_run(region, &region.runs[run], steps, facts),
        })
        .collect();
    let native: Vec<Seconds> = (region.units.iter())
        .map(|u| {
            (steps[u.first as usize..u.end as usize].iter())
                .map(|s| s.native(facts))
                .fold(Seconds(0.0), Add::add)
        })
        .collect();
    tiles
        .extend((native.iter().enumerate()).map(|(unit, &seconds)| Tile::Native { unit, seconds }));
    let candidates: Vec<(Range<u32>, Seconds)> = tiles
        .iter()
        .map(|t| (t.units(region), t.seconds()))
        .collect();
    let n = region.units.len() as u32;
    let chosen = cheapest_tiling(n, &candidates).expect("every unit is a candidate of its own");
    // The longest runs, from the first unit on.
    let mut longest = Seconds(0.0);
    let mut at = 0;
    while at < n {
        let (k, _) = (candidates.iter().enumerate())
            .filter(|(k, (r, _))| r.start == at && *k < region.runs.len())
            .max_by_key(|(_, (r, _))| r.end)
            .expect("every unit opens a run");
        longest = longest + candidates[k].1;
        at = candidates[k].0.end;
    }
    RegionPlan {
        tiles: chosen.iter().map(|&k| tiles[k].clone()).collect(),
        longest,
        dispatch: native.iter().copied().fold(Seconds(0.0), Add::add),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tape::lowered::{MkStreamWork, MkUnit};

    fn facts(cores: u32) -> DeviceFacts {
        DeviceFacts {
            cores: GpuCores(cores),
            launch: Seconds(5e-6),
            step: Seconds(1.5e-6),
            streams: vec![StreamFacts {
                lane: (1..=16)
                    .map(|k| BytesPerSecond(10e9 + 1e8 * f64::from(k)))
                    .collect(),
                native: BytesPerSecond(100e9),
                boundary: Seconds(5e-6),
            }],
        }
    }

    fn qmv(vtgs: u32) -> Costed {
        Costed {
            work: MkWork {
                vtg_threads: 64,
                widest: 16,
                stream: Some(MkStreamWork {
                    bytes: 13824,
                    calibration: 0,
                }),
                grid: (1, vtgs, 1),
                load: false,
                heads: None,
            },
            vtgs,
        }
    }

    fn norm() -> Costed {
        Costed {
            work: MkWork {
                vtg_threads: 1024,
                widest: 1,
                stream: None,
                grid: (1, 1, 1),
                load: false,
                heads: None,
            },
            vtgs: 1,
        }
    }

    /// A copied norm, two spread matvecs and a pinned step: at every core count each spread
    /// step's items are one to its widest virtual threadgroups, the cursors follow the items, and
    /// every lane group is a threadgroup of the launch.
    #[test]
    fn a_run_splits_validly_at_every_core_count() {
        let steps = [norm(), qmv(384), qmv(17), norm()];
        let units: &[MkUnit] = &[
            MkUnit { first: 0, end: 1 },
            MkUnit { first: 1, end: 2 },
            MkUnit { first: 2, end: 3 },
            MkUnit { first: 3, end: 4 },
        ];
        let run = MkRun {
            kernel: "mk_r0",
            first: 0,
            end: 4,
            places: &[
                MkPlace::Everywhere,
                MkPlace::Spread,
                MkPlace::Spread,
                MkPlace::Lane(0),
            ],
            waits: &[],
            split_at: ConstSlot(16384),
        };
        let region = MkRegion {
            opens: 0,
            required: 1,
            table_at: 0,
            block_len: 0,
            units,
            runs: Box::leak(Box::new([run])),
        };
        for p in 1..=128 {
            let f = facts(p);
            let s = split_run(&region, &region.runs[0], &steps, &f);
            assert!(
                s.k.iter().all(|&k| (1..=16).contains(&k)),
                "{p} cores: {:?}",
                s.k
            );
            assert_eq!(s.c, vec![0, 384u32.div_ceil(s.k[0])], "{p} cores");
            assert!(s.lanes.iter().all(|&l| l < p), "{p} cores: {:?}", s.lanes);
            assert!(s.seconds > f.launch, "{p} cores");
        }
    }

    /// The tiling weighs a launch boundary against a copy and against a step's own kernel: with
    /// launches dear, the norm is copied into its reader's launch; with a dispatch kernel
    /// streaming far faster than a core's threads, the matvec plays alone by its own kernel.
    #[test]
    fn the_plan_weighs_launches_copies_and_dispatch_kernels() {
        let steps = [norm(), qmv(640)];
        let units: &[MkUnit] = &[MkUnit { first: 0, end: 1 }, MkUnit { first: 1, end: 2 }];
        let runs: &[MkRun] = &[
            MkRun {
                kernel: "mk_r0",
                first: 0,
                end: 1,
                places: &[MkPlace::Lane(0)],
                waits: &[],
                split_at: ConstSlot(16384),
            },
            MkRun {
                kernel: "mk_r1",
                first: 0,
                end: 2,
                places: &[MkPlace::Everywhere, MkPlace::Spread],
                waits: &[(1, 0)],
                split_at: ConstSlot(16385),
            },
            MkRun {
                kernel: "mk_r2",
                first: 1,
                end: 2,
                places: &[MkPlace::Spread],
                waits: &[],
                split_at: ConstSlot(16387),
            },
        ];
        let region = MkRegion {
            opens: 0,
            required: 1,
            table_at: 0,
            block_len: 0,
            units,
            runs,
        };
        let f = facts(10);
        let plan = solve(&region, &steps, &f);
        assert!(
            matches!(plan.tiles[..], [Tile::Run { run: 1, .. }]),
            "{:?}",
            plan.tiles
        );
        let mut fast = facts(10);
        fast.streams[0].native = BytesPerSecond(1e13);
        let plan = solve(&region, &steps, &fast);
        assert!(
            matches!(
                plan.tiles[..],
                [Tile::Run { run: 0, .. }, Tile::Native { unit: 1, .. }]
            ),
            "{:?}",
            plan.tiles
        );
    }
}
