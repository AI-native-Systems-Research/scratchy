// SPDX-License-Identifier: Apache-2.0
//! A metal tape, projected onto the shared megakernel dataflow pass.
//!
//! Each command becomes a [`StepFlow`]: the memory its bindings name, split into what the kernel
//! writes ([`kernel_writes`]) and what it only reads, and how many threadgroups it launches. The
//! analysis itself — waits, units, coherence — is [`scratchy_subtile::megakernel_plan`]'s.

use std::num::NonZeroU32;
use std::ops::Range;

use scratchy_subtile::megakernel_plan::{self, Items, MegakernelPlan, StepFlow};

use crate::tape::kernel_memory::kernel_writes;
use crate::tape::lowered::{Binding, GatedCommand, RuntimeBindingKind};

/// One of the worker's shared scratch buffers, taken WHOLE. The tape records a region's offset
/// but not its length, and one bucket may pack two layouts into the same buffer (GDN and MoE
/// share `moe_scratch`), so two offsets are not proof of two disjoint regions.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum ScratchBuffer {
    SplitK,
    RopedK,
    AttnUnfused,
    Moe,
}

/// Memory a command can touch, as the tape names it. Weights, sources and inline scalars are
/// absent: nothing on the tape writes them.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum MetalLoc {
    Arena(u32),
    Scratch(ScratchBuffer),
    Runtime(RuntimeBindingKind),
}

fn loc(b: &Binding) -> Option<(u8, MetalLoc)> {
    match *b {
        Binding::ArenaSlot { slot, binding_index } => Some((binding_index, MetalLoc::Arena(slot))),
        Binding::Scratch { binding_index } => Some((binding_index, MetalLoc::Scratch(ScratchBuffer::SplitK))),
        Binding::RopedKScratch { binding_index } => {
            Some((binding_index, MetalLoc::Scratch(ScratchBuffer::RopedK)))
        }
        Binding::AttnUnfusedScratch { binding_index, .. } => {
            Some((binding_index, MetalLoc::Scratch(ScratchBuffer::AttnUnfused)))
        }
        Binding::MoeScratch { binding_index, .. } => {
            Some((binding_index, MetalLoc::Scratch(ScratchBuffer::Moe)))
        }
        Binding::Runtime { kind, binding_index } => Some((binding_index, MetalLoc::Runtime(kind))),
        Binding::Source { .. } | Binding::Inline { .. } => None,
    }
}

/// A command's dataflow, or `None` when its kernel's writes are not declared.
pub fn step_flow(cmd: &GatedCommand) -> Option<StepFlow<MetalLoc>> {
    let c = &cmd.command;
    let writes = kernel_writes(c.kernel)?;
    let (x, y, z) = c.dispatch.threadgroups;
    let items = Items(NonZeroU32::new(x * y * z)?);
    let mut flow = StepFlow { reads: Vec::new(), writes: Vec::new(), items };
    for (index, l) in c.bindings.iter().filter_map(loc) {
        if writes.contains(&index) {
            flow.writes.push(l);
        } else {
            flow.reads.push(l);
        }
    }
    Some(flow)
}

/// Commands one megakernel dispatch plays, and their plan.
pub struct Run {
    pub commands: Range<usize>,
    pub plan: MegakernelPlan<MetalLoc>,
}

/// Split an expanded tape into maximal runs of commands whose dataflow is declared, and plan
/// each. A command outside every run is dispatched on its own.
pub fn plan_runs(commands: &[GatedCommand]) -> Vec<Run> {
    let flows: Vec<Option<StepFlow<MetalLoc>>> = commands.iter().map(step_flow).collect();
    let mut runs = Vec::new();
    let mut i = 0;
    while i < flows.len() {
        if flows[i].is_none() {
            i += 1;
            continue;
        }
        let start = i;
        while i < flows.len() && flows[i].is_some() {
            i += 1;
        }
        let steps: Vec<StepFlow<MetalLoc>> = flows[start..i].iter().flatten().cloned().collect();
        runs.push(Run { commands: start..i, plan: megakernel_plan::plan(&steps) });
    }
    runs
}
