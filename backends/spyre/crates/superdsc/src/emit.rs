// SPDX-License-Identifier: Apache-2.0
//! [`Graph`] → SuperDSC: one descriptor per device op.
//!
//! Each op's space gets the DDL's dimension labels, a work division, and for each operand a
//! layout read straight off its view: the unit-stride axis is the stick, every other axis is
//! ordered by its stride, and each axis's extent in storage is the ratio of the next stride to its
//! own. Every address is concrete — the parameter's, the constant stick's or the intermediate's
//! byte address, plus the core's corner along each axis times that axis's stride.

use std::collections::BTreeMap;

use ktir_core::ddl_binds::DeviceGen;

use crate::dim::{Dim, MATMUL_BATCH, OUTER};
use crate::divide::{Cuttable, divide};
use crate::format::DataFormat;
use crate::graph::{ConstStick, Graph, Node, Operand, Store};
use crate::opfunc::{Family, OpFunc};
use crate::qty::{Bytes, Cores, Extent};
use crate::wire;

/// A byte address in device memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct DevAddr(u64);

impl DevAddr {
    pub const fn new(byte: u64) -> DevAddr {
        DevAddr(byte)
    }

    pub const fn get(self) -> u64 {
        self.0
    }

    /// The address `bytes` past this one.
    pub const fn add_bytes(self, bytes: u64) -> DevAddr {
        DevAddr(self.0 + bytes)
    }
}

/// Where the function's storage lives on the device.
#[derive(Clone, Debug)]
pub struct Bindings<'a> {
    /// Each parameter's buffer, by parameter.
    pub params: &'a [DevAddr],
    /// Each constant's stick, by constant.
    pub consts: &'a [(ConstStick, DevAddr)],
    /// The region the intermediates are packed into, if one is placed.
    pub scratch: Option<DevAddr>,
}

/// What a graph needs placed: the scratch its intermediates take, packed from the scratch base in
/// local order, and the bytes each parameter's buffer must hold — every element any descriptor
/// reaches, padding included.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Footprint {
    pub scratch_bytes: Bytes,
    pub room: Vec<Bytes>,
    /// Each intermediate's byte offset from the scratch base.
    locals: Vec<Bytes>,
}

/// Why an op has no descriptor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum EmitError {
    /// An operand whose stick axis (the op's, or its role's) is not the view's unit-stride axis.
    Stick { node: usize, operand: usize },
    /// An operand whose strides do not nest whole sticks inside each other.
    Nesting { node: usize, operand: usize },
    /// A parameter the bindings give no address.
    Unbound { param: usize },
    /// A constant the bindings give no stick.
    Unplaced { constant: usize },
    /// An intermediate with no scratch region placed to hold it.
    Unscratched { local: usize },
    /// An operand held in a format dxp binds the op over none of ([`OpFunc::binds`]).
    Format {
        node: usize,
        operand: usize,
        func: OpFunc,
        format: DataFormat,
    },
}

/// One space axis.
#[derive(Clone, Copy, Debug)]
struct SAxis {
    extent: u64,
    dim: Dim,
}

/// The op's work division: each axis's split, its per-core extent, and each core's slice.
struct Division<'a> {
    axes: &'a [SAxis],
    split: &'a [u64],
    per_core: &'a [u64],
    slices: &'a [Vec<u64>],
}

/// Where a space axis lands in an operand: one of its view's axes, or the lanes of the stick each
/// of its values sits on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum At {
    View(usize),
    Lanes,
    /// An axis the operand repeats along without a view axis of its own.
    Broadcast,
}

/// One operand projected onto the space: where each space axis it lists lands, and the space axis
/// it is sticked on.
struct Proj<'g> {
    operand: &'g Operand,
    at: Vec<Option<At>>,
    stick: usize,
    /// The space axis an index tensor walks, when this operand is the one it indexes.
    indexed: Option<usize>,
}

impl Proj<'_> {
    /// Element stride along space axis `i` (0 for an axis the operand does not advance along).
    fn stride(&self, i: usize) -> Option<u64> {
        self.at[i].map(|a| match a {
            At::View(j) => self.operand.view.axes[j].stride.get(),
            At::Lanes => 1,
            At::Broadcast => 0,
        })
    }

    /// Element distance of position `c` along space axis `i`.
    fn distance(&self, i: usize, c: u64) -> u64 {
        match self.at[i] {
            Some(At::View(j)) => self.operand.view.distance(j, Extent::new(c)).get(),
            Some(At::Lanes) => c,
            Some(At::Broadcast) | None => 0,
        }
    }

    /// Whether the operand holds one position along space axis `i` (of `extent` in the space): the
    /// space's own extent is one, or the operand's view axis is one position the stick's padding
    /// widened — every position of it reads that one element.
    fn one_position(&self, i: usize, extent: u64) -> bool {
        extent == 1
            || matches!(self.at[i], Some(At::View(j)) if self.operand.view.axes[j].extent == Extent::ONE)
    }

    /// Element distance between consecutive sticks along space axis `i`, when it is held in stick
    /// columns.
    fn columns(&self, i: usize) -> Option<u64> {
        match (self.at[i], self.operand.view.tile) {
            (Some(At::View(j)), Some(t)) if t.axis == j => Some(t.pitch.get()),
            _ => None,
        }
    }
}

/// One op's space, division and operand projections — everything but addresses.
struct Plan {
    axes: Vec<SAxis>,
    split: Vec<u64>,
    per_core: Vec<u64>,
    slices: Vec<Vec<u64>>,
    /// Per operand (inputs, then the result): its stride along each space axis it lists, and the
    /// space axis it is sticked on.
    projs: Vec<(Vec<Option<At>>, usize)>,
    /// The operand read or written through an index tensor, and the space axis the index walks.
    gathered: Option<(usize, usize)>,
}

impl Plan {
    /// One past the last element operand `oi` reaches over the padded space.
    fn end(&self, node: &Node, oi: usize) -> u64 {
        let p = self.proj(node, oi);
        p.operand.view.offset.get()
            + self
                .axes
                .iter()
                .enumerate()
                .map(|(i, a)| p.distance(i, a.extent.saturating_sub(1)))
                .sum::<u64>()
            + 1
    }

    fn proj<'g>(&self, node: &'g Node, oi: usize) -> Proj<'g> {
        let operand = if oi < node.inputs.len() {
            &node.inputs[oi]
        } else {
            &node.output
        };
        let (at, stick) = &self.projs[oi];
        Proj {
            operand,
            at: at.clone(),
            stick: *stick,
            // Every operand of an access through an index walks the indexed axis.
            indexed: self.gathered.map(|(_, a)| a),
        }
    }
}

/// The storage `g`'s descriptors reach on generation `G`.
pub fn footprint<G: DeviceGen>(g: &Graph) -> Footprint {
    let mut elements: Vec<u64> = g.locals.iter().map(|l| l.elements.get()).collect();
    let mut room: Vec<u64> = vec![0; g.params.len()];
    // Each intermediate's live range: from the first op that touches it to the last.
    let mut live: Vec<Option<(usize, usize)>> = vec![None; g.locals.len()];
    for (ni, node) in g.nodes.iter().enumerate() {
        let p = plan::<G>(node);
        for oi in 0..=node.inputs.len() {
            let o = if oi < node.inputs.len() {
                &node.inputs[oi]
            } else {
                &node.output
            };
            let end = p.end(node, oi);
            match o.store {
                Store::Local(l) => {
                    elements[l] = elements[l].max(end);
                    live[l] = Some(live[l].map_or((ni, ni), |(a, _)| (a, ni)));
                }
                Store::Param(q) => room[q] = room[q].max(end * o.format.word_bytes().get()),
                Store::Const(_) => {}
            }
        }
    }
    // Packed by live range: an intermediate takes the lowest bytes no intermediate live beside it
    // holds, so one dead before another is born shares its bytes.
    let bytes: Vec<u64> = g
        .locals
        .iter()
        .zip(&elements)
        .map(|(l, e)| (e * l.format.word_bytes().get()).next_multiple_of(G::FLIT_BYTES))
        .collect();
    let mut order: Vec<usize> = (0..g.locals.len()).collect();
    order.sort_by_key(|&l| (live[l].map_or(usize::MAX, |r| r.0), l));
    let mut locals = vec![0u64; g.locals.len()];
    let mut placed: Vec<(u64, u64, usize)> = Vec::new();
    let mut scratch_bytes = 0u64;
    for l in order {
        let Some((first, last)) = live[l] else {
            continue;
        };
        placed.retain(|&(_, _, end)| end >= first);
        let mut taken: Vec<(u64, u64)> = placed.iter().map(|&(a, b, _)| (a, b)).collect();
        taken.sort_unstable();
        let mut at = 0u64;
        for (a, b) in taken {
            if at + bytes[l] <= a {
                break;
            }
            at = at.max(b);
        }
        locals[l] = at;
        placed.push((at, at + bytes[l], last));
        scratch_bytes = scratch_bytes.max(at + bytes[l]);
    }
    Footprint {
        scratch_bytes: Bytes::new(scratch_bytes),
        room: room.into_iter().map(Bytes::new).collect(),
        locals: locals.into_iter().map(Bytes::new).collect(),
    }
}

/// `g`'s descriptors in launch order, against `bindings`, for generation `G`.
pub fn emit<G: DeviceGen>(
    g: &Graph,
    footprint: &Footprint,
    bindings: &Bindings<'_>,
) -> Result<Vec<(OpFunc, wire::Sdsc)>, EmitError> {
    let base = |s: Store| -> Result<DevAddr, EmitError> {
        match s {
            Store::Param(p) => bindings
                .params
                .get(p)
                .copied()
                .ok_or(EmitError::Unbound { param: p }),
            Store::Local(l) => bindings
                .scratch
                .map(|s| s.add_bytes(footprint.locals[l].get()))
                .ok_or(EmitError::Unscratched { local: l }),
            Store::Const(c) => g
                .consts
                .get(c)
                .and_then(|k| bindings.consts.iter().find(|(b, _)| b == k))
                .map(|(_, a)| *a)
                .ok_or(EmitError::Unplaced { constant: c }),
        }
    };
    g.nodes
        .iter()
        .enumerate()
        .map(|(ni, node)| Ok((node.func, op::<G>(ni, node, &plan::<G>(node), &base)?)))
        .collect()
}

/// For each parameter some op reads on a unit stride: the widest padded extent any op's division
/// reaches along that stick, counted from the start of the buffer's row.
pub fn staged_widths<G: DeviceGen>(g: &Graph) -> Vec<Option<Extent>> {
    let mut out = vec![None; g.params.len()];
    for node in &g.nodes {
        let p = plan::<G>(node);
        for oi in 0..=node.inputs.len() {
            let o = if oi < node.inputs.len() {
                &node.inputs[oi]
            } else {
                &node.output
            };
            let Store::Param(q) = o.store else { continue };
            let pr = p.proj(node, oi);
            if pr.stride(pr.stick) != Some(1) {
                continue;
            }
            // The row's start: the offset minus whatever the outer axes advance it.
            let pitch = (0..p.axes.len())
                .filter(|&i| i != pr.stick)
                .filter_map(|i| pr.stride(i))
                .filter(|&s| s > 1)
                .min();
            let offset = o.view.offset.get();
            let in_row = pitch.map_or(offset, |pt| offset % pt);
            let w = Extent::new(in_row + p.axes[pr.stick].extent);
            out[q] = Some(out[q].map_or(w, |x: Extent| x.max(w)));
        }
    }
    out
}

/// One operand's position along each space axis, and its stick axis.
type Projection = (Vec<Option<At>>, usize);

/// The space, labels, division and operand projections of `node`.
fn plan<G: DeviceGen>(node: &Node) -> Plan {
    let family = node.func.family();
    // A conversion computes on whole sticks of its wider side: a core's slab of the narrower one is
    // then whole sticks too.
    let lanes = node
        .inputs
        .iter()
        .filter(|_| node.func.converts())
        .map(|o| o.format.lanes::<G>().get())
        .fold(node.output.format.lanes::<G>().get(), u64::max);
    let mut space: Vec<u64> = node.space.iter().map(|e| e.get()).collect();
    let rank = space.len();
    let mut operands: Vec<&Operand> = node.inputs.iter().collect();
    operands.push(&node.output);
    // Each operand's strides along the space, and the space axis that is the op's stick.
    let (projs, stick_axis, labels): (Vec<Projection>, usize, Vec<Dim>) = match family {
        Family::Contraction => {
            let nb = rank - 3;
            let (m, n, k) = (nb, nb + 1, nb + 2);
            let role = |_: &Operand, listed: [usize; 2]| -> Vec<Option<At>> {
                let mut s = vec![None; rank];
                for (i, slot) in s.iter_mut().enumerate().take(nb) {
                    *slot = Some(At::View(i));
                }
                s[listed[0]] = Some(At::View(nb));
                s[listed[1]] = Some(At::View(nb + 1));
                s
            };
            let mut labels: Vec<Dim> = (0..nb).map(|b| MATMUL_BATCH[nb - 1 - b]).collect();
            labels.extend([Dim::Mb, Dim::Out, Dim::In]);
            (
                vec![
                    (role(operands[0], [m, k]), k),
                    (role(operands[1], [k, n]), n),
                    (role(operands[2], [m, n]), n),
                ],
                n,
                labels,
            )
        }
        Family::Pointwise | Family::StickReduce | Family::NonStickReduce => {
            let src = if family == Family::Pointwise {
                &node.output
            } else {
                &node.inputs[0]
            };
            let unit = src
                .view
                .axes
                .iter()
                .position(|a| a.is_unit())
                .or_else(|| src.view.axes.iter().position(|a| a.stride.get() == 1));
            let (stick, lanes_axis) = match unit {
                Some(s) => (s, false),
                // Every value sits on its own stick: the stick's lanes are an axis of their own.
                None => (rank, true),
            };
            if lanes_axis {
                space.push(lanes);
            }
            let mut labels = Vec::with_capacity(space.len());
            let mut outer = OUTER.iter();
            for i in 0..space.len() {
                labels.push(if i == stick {
                    Dim::Out
                } else {
                    *outer.next().unwrap_or(&Dim::J)
                });
            }
            let projs = operands
                .iter()
                .map(|o| {
                    let mut s: Vec<Option<At>> =
                        (0..o.view.axes.len()).map(|j| Some(At::View(j))).collect();
                    // A constant is one value: its lanes repeat like every other axis of it.
                    if lanes_axis {
                        s.push(Some(if matches!(o.store, Store::Const(_)) {
                            At::Broadcast
                        } else {
                            At::Lanes
                        }));
                    }
                    // A restick reads each operand on its own stick.
                    let own = o.view.axes.iter().position(|a| a.is_unit());
                    let stick = match (node.func, own) {
                        (OpFunc::Restick, Some(own)) => own,
                        _ => stick,
                    };
                    (s, stick)
                })
                .collect();
            (projs, stick, labels)
        }
    };
    // A partial stick computes on the whole stick: the stick axis rounds up to whole sticks, and
    // every operand must hold those lanes as padding of its own (checked per operand).
    space[stick_axis] = space[stick_axis].next_multiple_of(lanes);
    // ── the work division
    let reduced = match family {
        Family::Contraction => Some(rank - 1),
        Family::StickReduce | Family::NonStickReduce => node
            .output
            .view
            .axes
            .iter()
            .position(|a| a.stride.get() == 0 && a.spreads()),
        Family::Pointwise => None,
    };
    // An access through an index tensor: the indexed axis stays whole on every core.
    let gathered = operands
        .iter()
        .position(|o| o.gather.is_some())
        .and_then(|oi| operands[oi].gather.as_ref().map(|g| (oi, g.axis)));
    let reduced = reduced.or(gathered.map(|(_, a)| a));
    let bytes: u64 = operands
        .iter()
        .map(|o| o.view.numel().bytes(o.format.word_bytes()).get())
        .sum();
    // An access through an index tensor runs on one core: dxp converts its indices to addresses
    // on one (`GatherIndexConversion`'s `idx2addr`).
    let budget = match (family, gathered) {
        (_, Some(_)) => 1,
        (Family::Contraction, None) => u64::from(G::CORES),
        (_, None) => bytes
            .div_ceil(G::BYTES_PER_CORE)
            .clamp(1, u64::from(G::CORES)),
    };
    // Every axis some operand is sticked on cuts in whole sticks.
    let sticks: Vec<usize> = projs
        .iter()
        .zip(&operands)
        .filter(|((at, st), o)| {
            at.get(*st).copied().flatten().is_some_and(|a| match a {
                At::View(j) => o.view.axes[j].stride.get() == 1,
                At::Lanes => true,
                At::Broadcast => false,
            })
        })
        .map(|((_, st), _)| *st)
        .chain([stick_axis])
        .collect();
    // Every axis some operand is sticked on computes on whole sticks.
    for &a in &sticks {
        space[a] = space[a].next_multiple_of(lanes);
    }
    let cuttable = |space: &[u64]| -> Vec<Cuttable> {
        space
            .iter()
            .enumerate()
            .map(|(i, &e)| Cuttable {
                units: Extent::new(if Some(i) == reduced {
                    1
                } else if sticks.contains(&i) {
                    if e % lanes == 0 { e / lanes } else { 1 }
                } else {
                    e
                }),
            })
            .collect()
    };
    let split_of = |space: &[u64]| -> Vec<u64> {
        divide(&cuttable(space), Cores::new(budget))
            .into_iter()
            .map(Cores::get)
            .collect()
    };
    let mut split = split_of(&space);
    // A contraction whose stick count factors badly divides badly: a prime count of output sticks
    // splits across one core. Such an axis is padded by up to an eighth to the count that splits
    // across the most cores — judged on the stick axis alone, so a weight's padding is the same at
    // every row count that reads it. The padding is lanes every operand's buffer holds.
    if family == Family::Contraction {
        let units = space[stick_axis] / lanes;
        let splits = |u: u64| {
            (1..=budget.min(u))
                .rev()
                .find(|d| u.is_multiple_of(*d))
                .unwrap_or(1)
        };
        if splits(units) < budget.min(units) / 2 {
            let best = (units..=units + units / 8)
                .max_by_key(|&u| (splits(u), std::cmp::Reverse(u)))
                .unwrap_or(units);
            space[stick_axis] = best * lanes;
            split = split_of(&space);
        }
    }
    let axes: Vec<SAxis> = space
        .iter()
        .zip(&labels)
        .map(|(&extent, &dim)| SAxis { extent, dim })
        .collect();
    let cores: u64 = split.iter().product();
    let per_core: Vec<u64> = axes.iter().zip(&split).map(|(a, s)| a.extent / s).collect();
    // Core `c`'s slice along each axis: the first axis varies fastest.
    let slices: Vec<Vec<u64>> = (0..cores)
        .map(|c| {
            let mut r = c;
            split
                .iter()
                .map(|&s| {
                    let v = r % s;
                    r /= s;
                    v
                })
                .collect()
        })
        .collect();
    Plan {
        axes,
        split,
        per_core,
        slices,
        projs,
        gathered,
    }
}

fn op<G: DeviceGen>(
    ni: usize,
    node: &Node,
    plan: &Plan,
    base: &dyn Fn(Store) -> Result<DevAddr, EmitError>,
) -> Result<wire::Sdsc, EmitError> {
    let family = node.func.family();
    let mut operands: Vec<&Operand> = node.inputs.iter().collect();
    operands.push(&node.output);
    if let Some(o) = operands.iter().position(|o| !node.func.binds(o.format)) {
        return Err(EmitError::Format {
            node: ni,
            operand: o,
            func: node.func,
            format: operands[o].format,
        });
    }
    let Plan {
        axes,
        split,
        per_core,
        slices,
        projs,
        gathered,
    } = plan;
    let cores: u64 = split.iter().product();
    let space: Vec<u64> = axes.iter().map(|a| a.extent).collect();
    // ── the operands
    let n_in = node.inputs.len();
    let division = Division {
        axes,
        split,
        per_core,
        slices,
    };
    let mut tensors = Vec::with_capacity(projs.len() + 1);
    for oi in 0..projs.len() {
        let p = plan.proj(node, oi);
        tensors.push(tensor::<G>((ni, oi), &p, &division, base)?);
    }
    // ── an access through an index tensor: the index is one more tensor, over the indexed axis
    let index_at = match gathered {
        Some((vi, g)) => {
            let gather = operands[*vi].gather.as_ref().ok_or(EmitError::Stick {
                node: ni,
                operand: *vi,
            })?;
            let index = &gather.index;
            let lanes = index.format.lanes::<G>().get();
            let n = axes[*g].extent as i64;
            let affine = |alpha: i64| wire::Func::Affine { alpha, beta: 0 };
            let prop = |factor: u64, label| wire::Prop::new(factor, label);
            let coord = wire::Coord {
                spatial: 3,
                temporal: 0,
                elem_arr: 2,
                padding: "nopad",
                folds: wire::Folds {
                    dim_prop_func: vec![
                        affine(n),
                        affine(0),
                        affine(0),
                        affine(lanes as i64),
                        affine(1),
                    ],
                    dim_prop_attr: vec![
                        prop(1, "core_fold"),
                        prop(1, "corelet_fold"),
                        prop(1, "row_fold"),
                        // The sticks the `n` indices span — every stick dimension's own law.
                        prop((n as u64).div_ceil(lanes), "elem_arr_1"),
                        prop(lanes, "elem_arr_0"),
                    ],
                    data: None,
                },
            };
            let addr = base(index.store)?
                .add_bytes(index.view.offset.bytes(index.format.word_bytes()).get())
                .get();
            // The value's indexed axis holds one row per index. Rows one stick apart are the rows of
            // a stick array, the rest of which are a gap; a whole multi-stick row per index is the
            // value's outermost block, past which there is nothing.
            let one_stick =
                plan.proj(node, *vi).stride(*g) == Some(operands[*vi].format.lanes::<G>().get());
            let vt = &mut tensors[*vi];
            if !one_stick {
                // dxp sizes a whole row's own dims itself; it is told only that each index names
                // one row.
                vt.max_dims.iter_mut().for_each(|m| *m = -1);
            }
            if let Some(k) = vt.order.iter().position(|d| *d == axes[*g].dim) {
                vt.max_dims[k] = 1;
            }
            if one_stick {
                vt.back_gap = Some((axes[*g].dim, gather.rows.get().saturating_sub(1)));
            }
            tensors.push(Tensor {
                format: index.format,
                order: vec![axes[*g].dim],
                stick: axes[*g].dim,
                max_dims: vec![-1],
                scales: vec![1],
                coords: BTreeMap::from([(axes[*g].dim.wire(), coord)]),
                addrs: vec![addr; slices.len()],
                back_gap: None,
            });
            // The index tensor leads the operand list — dxp's identity binds its input and
            // output after it.
            if let Some(index) = tensors.pop() {
                tensors.insert(0, index);
            }
            Some((*vi + 1, 0))
        }
        None => None,
    };
    // Where the op's own inputs and result sit in the operand list.
    let first = usize::from(index_at.is_some());
    let out_idx = first + n_in;
    // ── the layout labels: one per distinct (order, stick), a contraction's and a conversion's by role
    let mut labels_of: Vec<&'static str> = Vec::with_capacity(tensors.len());
    let mut primary: BTreeMap<&'static str, wire::LayoutInfo> = BTreeMap::new();
    let mut seen: Vec<(Vec<Dim>, Dim, u64, &'static str)> = Vec::new();
    const NAMES: [&str; 4] = ["OUTPUT", "KERNEL", "INPUT", "KERNEL_IDX"];
    const ROLES: [&str; 3] = ["INPUT", "KERNEL", "OUTPUT"];
    for (ti, t) in tensors.iter().enumerate() {
        let label = if index_at.is_some_and(|(_, ii)| ii == ti) {
            "KERNEL_IDX"
        } else if family == Family::Contraction {
            ROLES[ti]
        } else if node.func.converts() {
            if ti < out_idx { "INPUT" } else { "OUTPUT" }
        } else if let Some((_, _, _, l)) = seen.iter().find(|(o, s, w, _)| {
            *o == t.order && *s == t.stick && *w == t.format.lanes::<G>().get()
        }) {
            l
        } else {
            let l = NAMES[seen.len().min(NAMES.len() - 1)];
            seen.push((t.order.clone(), t.stick, t.format.lanes::<G>().get(), l));
            l
        };
        labels_of.push(label);
        primary.entry(label).or_insert_with(|| wire::LayoutInfo {
            layout_dim_order: t.order.iter().map(|d| d.wire()).collect(),
            stick_dim_order: vec![t.stick.wire()],
            stick_size: vec![t.format.lanes::<G>().get()],
        });
    }
    let core_prop = |n: u64| wire::Prop::new(n, "core");
    let one = |label| wire::Prop::new(1, label);
    let schedule: Vec<wire::Allocate> = tensors
        .iter()
        .enumerate()
        .map(|(ti, t)| wire::Allocate {
            node_type: "allocate",
            name: format!("allocate-Tensor{ti}_hbm"),
            prev: "",
            lds_idx: ti,
            component: "hbm",
            layout_dim_order: t.order.iter().map(|d| d.wire()).collect(),
            max_dim_sizes: t.max_dims.clone(),
            indirect_alloc_type: index_at.map(|(vi, ii)| {
                if ti == vi {
                    "value_tensor"
                } else if ti == ii {
                    "index_tensor"
                } else {
                    "no_indirection"
                }
            }),
            related_indirect_access_alloc: index_at.and_then(|(vi, ii)| {
                if ti == vi {
                    Some(format!("allocate-Tensor{ii}_hbm"))
                } else if ti == ii {
                    Some(format!("allocate-Tensor{vi}_hbm"))
                } else {
                    None
                }
            }),
            index_tensor_type: index_at.and_then(|(_, ii)| (ti == ii).then_some("index")),
            start_address_core_corelet: wire::Folds {
                dim_prop_func: vec![
                    wire::Func::Map {},
                    wire::Func::Const {},
                    wire::Func::Const {},
                ],
                dim_prop_attr: vec![core_prop(cores), one("corelet"), one("time")],
                data: Some(
                    t.addrs
                        .iter()
                        .enumerate()
                        .map(|(c, a)| (format!("[{c}, 0, 0]"), a.to_string()))
                        .collect(),
                ),
            },
            back_gap_core: t.back_gap.map(|(d, gap)| {
                BTreeMap::from([(d.wire(), BTreeMap::from([("-1", gap.to_string())]))])
            }),
            coordinates: wire::Coordinates {
                coord_info: t.coords.clone(),
                core_id_to_wk_slice: BTreeMap::new(),
            },
        })
        .collect();
    let labeled: Vec<wire::LabeledDs> = tensors
        .iter()
        .enumerate()
        .map(|(ti, t)| wire::LabeledDs {
            lds_idx: ti,
            ds_name: format!("Tensor{ti}"),
            ds_type: labels_of[ti],
            scale: t.scales.clone(),
            word_length: t.format.word_bytes().get(),
            data_format: t.format.wire(),
            // An index tensor is read from HBM only: indices never stage through LX.
            mem_org: wire::MemOrg {
                hbm: Some(wire::MemPresence { is_present: 1 }),
                lx: (!index_at.is_some_and(|(_, ii)| ii == ti))
                    .then_some(wire::MemPresence { is_present: 1 }),
            },
        })
        .collect();
    let mut inputs: Vec<String> = (first..first + n_in)
        .map(|i| format!("Tensor{i}-idx{i}"))
        .collect();
    if node.func.accumulates() {
        inputs.push(format!("Tensor{out_idx}-idx{out_idx}"));
    }
    let compute_format = node
        .func
        .compute_format(node.inputs.first().map_or(node.output.format, |o| o.format));
    let space_of = |name, ext: &[u64]| wire::Space {
        name,
        dims: axes
            .iter()
            .zip(ext)
            .map(|(a, &e)| (a.dim.extent_key(), e))
            .collect(),
    };
    let dsc = wire::Dsc {
        num_cores_used: cores as u32,
        num_corelets_used: 1,
        core_ids_used: (0..cores as u32).collect(),
        n: space_of("n", &space),
        coordinate_masking: BTreeMap::new(),
        masking_const_id: -1,
        data_stage_param: BTreeMap::from([(
            "0",
            wire::Stage {
                ss: space_of("core", per_core),
                el: space_of("core", per_core),
            },
        )]),
        primary_ds_info: primary,
        schedule_tree: schedule,
        labeled_ds: labeled,
        constant_info: wire::Constants::Empty("{}"),
        compute_op: vec![wire::ComputeOp {
            ex_unit: node.func.unit().wire(),
            op_func_name: node.func.wire(),
            attributes: wire::ComputeAttrs {
                data_format: compute_format.wire(),
                fidelity: "regular",
            },
            location: "Inner",
            input_labeled_ds: inputs,
            output_labeled_ds: vec![format!("Tensor{out_idx}-idx{out_idx}")],
            indirect_access_index_labeled_ds: index_at
                .map(|(_, ii)| vec![format!("Tensor{ii}-idx{ii}")])
                .unwrap_or_default(),
        }],
    };
    Ok(wire::Sdsc {
        sdsc_fold_props: vec![one("time")],
        sdsc_folds: wire::Folds {
            dim_prop_func: vec![wire::Func::Affine { alpha: 1, beta: 0 }],
            dim_prop_attr: vec![one("time")],
            data: Some(BTreeMap::from([("[0]", "0")])),
        },
        core_fold_prop: core_prop(cores),
        corelet_fold_prop: one("corelet"),
        num_cores_used: cores as u32,
        core_id_to_dsc: (0..cores).map(|c| (c.to_string(), 0)).collect(),
        num_wk_slices_per_dim: axes
            .iter()
            .zip(split)
            .map(|(a, &s)| (a.dim.wire(), s))
            .collect(),
        core_id_to_wk_slice: slices
            .iter()
            .enumerate()
            .map(|(c, s)| {
                (
                    c.to_string(),
                    axes.iter()
                        .zip(s)
                        .map(|(a, &v)| (a.dim.wire(), v))
                        .collect(),
                )
            })
            .collect(),
        core_id_to_dsc_schedule: (0..cores)
            .map(|c| (c.to_string(), vec![[-1, 0, 0, 0]]))
            .collect(),
        dscs: vec![BTreeMap::from([(node.func.wire(), dsc)])],
    })
}

/// One operand's descriptor facts.
struct Tensor {
    format: DataFormat,
    order: Vec<Dim>,
    stick: Dim,
    max_dims: Vec<i64>,
    scales: Vec<i64>,
    coords: BTreeMap<&'static str, wire::Coord>,
    addrs: Vec<u64>,
    /// The gap after an indexed axis's one row per index: the buffer's other rows.
    back_gap: Option<(Dim, u64)>,
}

fn tensor<G: DeviceGen>(
    (ni, oi): (usize, usize),
    p: &Proj<'_>,
    d: &Division<'_>,
    base: &dyn Fn(Store) -> Result<DevAddr, EmitError>,
) -> Result<Tensor, EmitError> {
    let Division {
        axes,
        split,
        per_core,
        slices,
    } = *d;
    let o = p.operand;
    let lanes = o.format.lanes::<G>().get();
    #[derive(Clone, Copy)]
    struct Entry {
        axis: usize,
        scale: i64,
        key: u64,
    }
    let mut entries: Vec<Entry> = Vec::new();
    for (axis, ax) in axes.iter().enumerate() {
        let Some(stride) = p.stride(axis) else {
            continue;
        };
        let e = if axis == p.stick {
            match stride {
                // The stick's index advances a stick at a time — or a whole stick column.
                1 => Entry {
                    axis,
                    scale: 1,
                    key: p.columns(axis).unwrap_or(lanes),
                },
                0 => Entry {
                    axis,
                    scale: -2,
                    key: 0,
                },
                _ => {
                    return Err(EmitError::Stick {
                        node: ni,
                        operand: oi,
                    });
                }
            }
        } else if p.indexed != Some(axis) && (stride == 0 || p.one_position(axis, ax.extent)) {
            Entry {
                axis,
                scale: -1,
                key: u64::MAX,
            }
        } else if stride == 0 {
            // The indexed axis of an operand holding one row: outermost, one row.
            Entry {
                axis,
                scale: 1,
                key: u64::MAX - 1,
            }
        } else {
            Entry {
                axis,
                scale: 1,
                key: stride,
            }
        };
        entries.push(e);
    }
    // A contiguous stick sits inside every other axis of its key; a stick held in columns sits
    // outside every row its columns are pitched past.
    let columns = p.columns(p.stick).is_some();
    // An indexed axis one stick apart sits directly outside the lanes: dxp reads it as the rows.
    entries.sort_by_key(|e| {
        (
            e.key,
            p.indexed != Some(e.axis),
            (e.axis == p.stick) == columns,
        )
    });
    let real: Vec<Entry> = entries
        .iter()
        .copied()
        .filter(|e| e.key < u64::MAX - 1)
        .collect();
    // Each axis's extent in storage: the next axis's step over this one's — counted in elements
    // for the stick, which holds `lanes` per step.
    let mut max_dims = Vec::with_capacity(entries.len());
    for e in &entries {
        if e.key >= u64::MAX - 1 {
            max_dims.push(1);
            continue;
        }
        let pos = real.iter().position(|r| r.axis == e.axis).unwrap_or(0);
        let md = match (e.scale, real.get(pos + 1)) {
            (-2, Some(n)) => {
                if n.key % lanes != 0 {
                    return Err(EmitError::Nesting {
                        node: ni,
                        operand: oi,
                    });
                }
                lanes as i64
            }
            (-2, None) => lanes as i64,
            (_, Some(n)) => {
                if n.key % e.key != 0 {
                    return Err(EmitError::Nesting {
                        node: ni,
                        operand: oi,
                    });
                }
                let ratio = n.key / e.key;
                (if e.axis == p.stick {
                    ratio * lanes
                } else {
                    ratio
                }) as i64
            }
            (_, None) => -1,
        };
        max_dims.push(md);
    }
    let order: Vec<Dim> = entries.iter().map(|e| axes[e.axis].dim).collect();
    let scales: Vec<i64> = entries.iter().map(|e| e.scale).collect();
    let mut coords = BTreeMap::new();
    for e in &entries {
        let tiled = e.scale == 1;
        let ns = if tiled { split[e.axis] } else { 1 };
        let size = if tiled { per_core[e.axis] } else { 1 } as i64;
        let affine = |alpha: i64| wire::Func::Affine { alpha, beta: 0 };
        let prop = |factor: u64, label| wire::Prop::new(factor, label);
        let c = if e.axis != p.stick {
            wire::Coord {
                spatial: 3,
                temporal: 0,
                elem_arr: 1,
                padding: "nopad",
                folds: wire::Folds {
                    dim_prop_func: vec![affine(size), affine(0), affine(0), affine(1)],
                    dim_prop_attr: vec![
                        prop(ns, "core_fold"),
                        prop(1, "corelet_fold"),
                        prop(1, "row_fold"),
                        prop(size as u64, "elem_arr_0"),
                    ],
                    data: None,
                },
            }
        } else {
            let red = e.scale == -2;
            let l = lanes as i64;
            wire::Coord {
                spatial: 3,
                temporal: 0,
                elem_arr: 2,
                padding: "nopad",
                folds: wire::Folds {
                    dim_prop_func: vec![
                        affine(if red { l } else { size }),
                        affine(0),
                        affine(0),
                        affine(l),
                        affine(if red { 0 } else { 1 }),
                    ],
                    dim_prop_attr: vec![
                        prop(ns, "core_fold"),
                        prop(1, "corelet_fold"),
                        prop(1, "row_fold"),
                        prop(
                            if red {
                                1
                            } else {
                                (size as u64).div_ceil(lanes)
                            },
                            "elem_arr_1",
                        ),
                        prop(lanes, "elem_arr_0"),
                    ],
                    data: None,
                },
            }
        };
        coords.insert(axes[e.axis].dim.wire(), c);
    }
    // The window starts a stick, and a partial-stick window owns the rest of its last stick.
    if p.stride(p.stick) == Some(1) {
        let logical = match p.at[p.stick] {
            Some(At::View(j)) => o.view.axes[j].extent.get(),
            _ => axes[p.stick].extent,
        };
        let outer_aligned = (0..axes.len()).all(|i| {
            i == p.stick
                || p.one_position(i, axes[i].extent)
                || p.stride(i).is_none_or(|s| s.is_multiple_of(lanes))
        });
        if !o.view.offset.get().is_multiple_of(lanes) || !outer_aligned {
            return Err(EmitError::Nesting {
                node: ni,
                operand: oi,
            });
        }
        if logical % lanes != 0 && logical != axes[p.stick].extent {
            let pitch = max_dims
                .get(entries.iter().position(|e| e.axis == p.stick).unwrap_or(0))
                .copied()
                .unwrap_or(-1);
            if pitch != -1 && (pitch as u64) < axes[p.stick].extent {
                return Err(EmitError::Nesting {
                    node: ni,
                    operand: oi,
                });
            }
        }
    }
    let origin = base(o.store)?;
    let word = o.format.word_bytes().get();
    let addrs = slices
        .iter()
        .map(|s| {
            let elems: u64 = entries
                .iter()
                .filter(|e| e.scale == 1)
                .map(|e| p.distance(e.axis, s[e.axis] * per_core[e.axis]))
                .sum();
            origin.add_bytes((o.view.offset.get() + elems) * word).get()
        })
        .collect();
    Ok(Tensor {
        format: o.format,
        order,
        stick: axes[p.stick].dim,
        max_dims,
        scales,
        coords,
        addrs,
        back_gap: None,
    })
}

/// The files of one dxp group: `sdsc_<i>.json` per op in order, then `bundle.mlir`.
pub fn group_files(
    ops: &[(OpFunc, wire::Sdsc)],
) -> Result<Vec<(String, String)>, serde_json::Error> {
    let mut files = Vec::with_capacity(ops.len() + 1);
    let mut body = String::new();
    for (i, (func, sdsc)) in ops.iter().enumerate() {
        let key = format!("{i}_{}", func.wire());
        let json = serde_json::to_string(&BTreeMap::from([(key, sdsc)]))?;
        let name = format!("sdsc_{i}.json");
        body.push_str(&format!(
            "    sdscbundle.sdsc_execute () {{sdsc_filename=\"{name}\"}}\n"
        ));
        files.push((name, json));
    }
    files.push((
        "bundle.mlir".to_string(),
        format!("module {{\n  func.func @sdsc_bundle() {{\n{body}    return\n  }}\n}}\n"),
    ));
    Ok(files)
}
