// SPDX-License-Identifier: Apache-2.0
//! A rung's KTIR programs lowered to SuperDSC against the rung's placement.
//!
//! Two passes over one rung. [`place_rung`] plans every program once without addresses and places
//! what the plans need beside the graph's own buffers — each staged weight's device layout, the
//! constant sticks the descriptors read, and the scratch every program's intermediates are packed
//! into. [`lower_op`] then emits one program's descriptors against those placements.

use ktir_core::ddl_binds::Baked;
use scratchy_spyre_bundle::collective::Tp;
use scratchy_spyre_bundle::program::KtirNode;
use scratchy_spyre_bundle::reserved_tids::{SCRATCH_TID, stick_const_tid};
use scratchy_triton_splice::EmittedOp;
use spyre_superdsc::format::DataFormat;
use spyre_superdsc::graph::ConstStick;
use spyre_superdsc::opfunc::OpFunc;
use spyre_superdsc::wire::Sdsc;
use spyre_superdsc::{Held, Planned, Residency};

use crate::placement::{BundleLayout, RetileDescriptor, SegRole, TensorPlacement, align128};

/// One program's descriptors, in launch order.
pub type Descriptors = Vec<(OpFunc, Sdsc)>;

/// The tensor each of `k`'s parameters addresses.
fn tids(k: &KtirNode) -> impl Iterator<Item = u32> + '_ {
    k.bindings.iter().map(|b| b.get())
}

/// Whether `k`'s `i`-th parameter is a staged weight: written once from the host, so its stager
/// may lay it out however its readers need.
fn residency(layout: &BundleLayout, k: &KtirNode) -> Vec<Residency> {
    tids(k)
        .map(|t| match layout.placements.get(&t) {
            Some(p) if p.role == SegRole::Weight => Residency::Staged,
            _ => Residency::Device,
        })
        .collect()
}

/// Each parameter's held layout under the rung's agreed layouts.
fn param_layouts(layout: &BundleLayout, k: &KtirNode) -> Vec<Option<Held>> {
    tids(k).map(|t| layout.held.get(&t).cloned()).collect()
}

/// `k` planned under the rung's agreed layouts.
fn plan(layout: &BundleLayout, k: &KtirNode) -> Result<Planned, String> {
    spyre_superdsc::plan::<Baked>(&k.func, &param_layouts(layout, k))
        .map_err(|e| format!("{}: {e:?}", k.func.name))
}

/// Plan every program of a rung and place what the plans need: each staged weight's device layout
/// (one per weight, agreed by every program reading it), the constant sticks, and the scratch.
pub fn place_rung<'a, T: Tp + 'a>(
    ops: impl IntoIterator<Item = &'a EmittedOp<T>> + Clone,
    layout: &mut BundleLayout,
) -> Result<(), String> {
    // Each staged weight's layout, agreed by its readers.
    for e in ops.clone() {
        let k = &e.ktir;
        let wanted = spyre_superdsc::layouts::<Baked>(&k.func, &residency(layout, k))
            .map_err(|err| format!("{}: {err:?}", e.op_name))?;
        for (t, w) in tids(k).zip(wanted) {
            let Some(w) = w else { continue };
            match layout.held.get(&t) {
                Some(h) if *h != w => {
                    return Err(format!(
                        "{}: t{t} is read in two layouts ({h:?} and {w:?}) — one staged weight \
                         holds one",
                        e.op_name
                    ));
                }
                _ => {
                    layout.held.insert(t, w);
                }
            }
        }
    }
    // Every program planned: the constants it reads, the scratch it packs, and the room each of
    // its buffers must hold.
    let mut consts: std::collections::BTreeSet<ConstStick> = Default::default();
    let mut scratch = 0u64;
    for e in ops {
        let k = &e.ktir;
        let p = plan(layout, k).map_err(|err| format!("{}: {err}", e.op_name))?;
        consts.extend(p.graph.consts.iter().copied());
        scratch = scratch.max(p.footprint.scratch_bytes);
        for (i, t) in tids(k).enumerate() {
            if let (true, Some(view), Some(format)) = (
                layout.held.contains_key(&t),
                p.graph.param_views[i].first(),
                p.graph.params[i],
            ) {
                layout.held_views.insert(t, (view.shape(), format));
            }
        }
        for (t, room) in tids(k).zip(&p.footprint.room) {
            let size = layout.placements.get(&t).map_or(0, |p| p.size);
            if *room > size {
                return Err(format!(
                    "{}: t{t} is placed at {size} B but its descriptors reach {room} B",
                    e.op_name
                ));
            }
        }
    }
    place_consts(layout, &consts.into_iter().collect::<Vec<_>>());
    place_scratch(layout, scratch);
    retile_weights(layout);
    Ok(())
}

/// Slots in a rung's constant table. Each value's slot is keyed by the value itself, so a value has
/// one address in every rung of a model: a session stages its own bundle's constants and every
/// sibling's, and a sibling reads its own at the address the session wrote them.
pub const STICK_CONST_SLOTS: u64 = 1024;

/// The slot `c` hashes to: FNV-1a over its format and bits, a fixed function of the value.
fn const_slot(c: &ConstStick) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in [c.format as u8].into_iter().chain(c.bits.to_le_bytes()) {
        h ^= u64::from(b);
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h % STICK_CONST_SLOTS
}

/// The constant table at the end of the activation segment: each value one stick at its own
/// slot, two values of one rung on one slot stepping on to the next free one in value order.
fn place_consts(layout: &mut BundleLayout, consts: &[ConstStick]) {
    let seg = SegRole::Activation.segment();
    let base = align128(layout.segment_bytes[seg]);
    let mut taken: std::collections::BTreeSet<u64> = Default::default();
    for c in consts {
        let mut slot = const_slot(c);
        while !taken.insert(slot) {
            slot = (slot + 1) % STICK_CONST_SLOTS;
        }
        let tid = stick_const_tid(slot as usize);
        layout.placements.insert(
            tid,
            TensorPlacement {
                tid,
                role: SegRole::Activation,
                segment: seg,
                bank: 0,
                offset: base + slot * 128,
                size: 128,
            },
        );
        layout.stick_consts.push((tid, *c));
    }
    layout.segment_bytes[seg] = base + STICK_CONST_SLOTS * 128;
}

/// The rung's scratch, after everything else in the intermediate segment.
fn place_scratch(layout: &mut BundleLayout, bytes: u64) {
    if bytes == 0 {
        return;
    }
    let seg = SegRole::Intermediate.segment();
    let offset = align128(layout.segment_bytes[seg]);
    layout.placements.insert(
        SCRATCH_TID,
        TensorPlacement {
            tid: SCRATCH_TID,
            role: SegRole::Intermediate,
            segment: seg,
            bank: 0,
            offset,
            size: align128(bytes),
        },
    );
    layout.segment_bytes[seg] = offset + align128(bytes);
}

/// Each staged weight's walk from its host bytes into its held layout.
fn retile_weights(layout: &mut BundleLayout) {
    let walks: Vec<(u32, RetileDescriptor)> = layout
        .held
        .iter()
        .filter_map(|(&t, h)| {
            let (shape, format) = layout.held_views.get(&t)?;
            Some((t, retile(shape, *format, h)))
        })
        .collect();
    layout.kernel_weights.extend(walks);
}

/// The walk from a row-major host buffer of `shape` into `held`: per device axis, outermost first,
/// its extent and the host stride it advances.
pub fn retile(shape: &[u64], format: DataFormat, held: &Held) -> RetileDescriptor {
    let lanes = format.lanes::<Baked>();
    let host = |a: usize| shape[a + 1..].iter().product::<u64>();
    let (outer, last) = held.order.split_at(held.order.len() - 1);
    let inner = last[0];
    let mut axes: Vec<(u64, u64)> = Vec::with_capacity(held.order.len() + 1);
    if held.columns {
        axes.push((shape[inner].div_ceil(lanes), host(inner) * lanes));
        axes.extend(outer.iter().map(|&a| (shape[a], host(a))));
        axes.push((lanes, host(inner)));
    } else {
        axes.extend(outer.iter().map(|&a| (shape[a], host(a))));
        axes.push((shape[inner], host(inner)));
    }
    RetileDescriptor {
        axes,
        stick_size: lanes as u32,
        word_length: format.word_bytes() as u32,
    }
}

/// The bytes each buffer of `ir` must hold: every element any program's descriptors reach, a
/// division's padding included — a contraction pads its stick axis past the tensor's own extent to
/// divide across more cores, and the placement holds the padding. Every node is lowered as a card
/// bake lowers it, each staged weight held as its readers need.
pub fn rooms<T: Tp, F: scratchy_subtile::subtile_ir::RopeForm>(
    ir: &scratchy_subtile::subtile_ir::SubtileIR<F>,
    weight_ids: &std::collections::HashSet<u32>,
    rows_are_requests: bool,
    query_rows: scratchy_spyre_bundle::rung::QueryExtent,
) -> std::collections::BTreeMap<u32, u64> {
    use crate::lower_subtile_tape_to_ktir::{NodeLowering, lower_one_node};
    use rayon::prelude::*;
    let per_node: Vec<Vec<(u32, u64)>> = ir
        .nodes
        .par_iter()
        .map(|node| {
            let ops = match lower_one_node::<T, F>(
                node,
                ir,
                scratchy_spyre_bundle::program::ActiveCap::FULL,
                rows_are_requests,
                query_rows,
            ) {
                NodeLowering::Ops(v) => v,
                NodeLowering::HostRouted(_) | NodeLowering::Unhandled(_) => Vec::new(),
            };
            let mut out = Vec::new();
            for e in &ops {
                let k = &e.ktir;
                let residency: Vec<Residency> = tids(k)
                    .map(|t| {
                        if weight_ids.contains(&t) {
                            Residency::Staged
                        } else {
                            Residency::Device
                        }
                    })
                    .collect();
                // A program the lowering cannot plan reserves nothing here: the card's own
                // lowering of it ([`place_rung`]) refuses it by name, and no other target reads it.
                let Ok(layouts) = spyre_superdsc::layouts::<Baked>(&k.func, &residency) else {
                    continue;
                };
                let Ok(p) = spyre_superdsc::plan::<Baked>(&k.func, &layouts) else {
                    continue;
                };
                out.extend(tids(k).zip(p.footprint.room.iter().copied()));
            }
            out
        })
        .collect();
    let mut room = std::collections::BTreeMap::new();
    for (t, b) in per_node.into_iter().flatten() {
        let e = room.entry(t).or_insert(0u64);
        *e = (*e).max(b);
    }
    room
}

/// `e`'s descriptors against the rung's placements.
pub fn lower_op<T: Tp>(e: &EmittedOp<T>, layout: &BundleLayout) -> Result<Descriptors, String> {
    let k = &e.ktir;
    let planned = plan(layout, k)?;
    let params: Vec<u64> = tids(k)
        .map(|t| {
            layout
                .address(t)
                .ok_or_else(|| format!("{}: t{t} has no placement", e.op_name))
        })
        .collect::<Result<_, _>>()?;
    let consts: Vec<(ConstStick, u64)> = layout
        .stick_consts
        .iter()
        .filter_map(|(t, c)| Some((*c, layout.address(*t)?)))
        .collect();
    let scratch = layout.address(SCRATCH_TID).unwrap_or(0);
    let bindings = spyre_superdsc::Bindings {
        params: &params,
        consts: &consts,
        scratch,
    };
    planned
        .emit::<Baked>(&bindings)
        .map_err(|err| format!("{}: {err:?}", e.op_name))
}
