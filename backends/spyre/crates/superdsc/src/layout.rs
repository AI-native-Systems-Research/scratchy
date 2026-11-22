// SPDX-License-Identifier: Apache-2.0
//! The stick each operand must be read on, and what to do when its view's unit-stride axis is
//! another one.
//!
//! An op's ABI fixes its operands' sticks ([`required`]): a pointwise op reads every operand on
//! its result's stick, a contraction reads `a` on `k` and `b` and the result on `n`. When an
//! operand's view puts its unit stride elsewhere, either its buffer is host-staged — and is then
//! held re-laid-out, so the view states the needed stick ([`staged_layouts`]) — or it is copied
//! into an intermediate that holds the needed stick ([`restick`]).

use crate::graph::{Graph, Local, Node, Operand, Store};
use crate::lift::{Held, ParamLayout};
use crate::opfunc::{Family, OpFunc};
use crate::view::View;

/// Who writes a parameter's buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Residency {
    /// The host stages it before any launch and no device op writes it (a weight, a table).
    Staged,
    /// A device op writes it (an activation, the KV pool).
    Device,
}

/// The operand axis operand `oi` of `node` must have as its unit stride, if the op fixes one.
pub fn required(node: &Node, oi: usize) -> Option<usize> {
    let operand = if oi < node.inputs.len() { &node.inputs[oi] } else { &node.output };
    let unit = |o: &Operand| o.view.axes.iter().position(|a| a.stride == 1 && a.extent > 1);
    match (node.func, node.func.family()) {
        (OpFunc::Restick, _) => None,
        (_, Family::Contraction) => Some(operand.view.axes.len() - 1),
        (_, Family::Pointwise) => unit(&node.output),
        (_, Family::StickReduce | Family::NonStickReduce) => {
            if oi < node.inputs.len() {
                None
            } else {
                unit(&node.inputs[0])
            }
        }
    }
}

/// Is `o` readable with `axis` as its stick?
fn on_stick(o: &Operand, axis: usize) -> bool {
    let a = o.view.axes[axis];
    a.stride == 1 || a.extent == 1 || (a.stride == 0 && o.gather.is_none())
}

/// For each staged parameter: held so every op reads it on the stick it needs — innermost the
/// axis a mismatched read needs, and in stick columns when a contraction reads it as its KERNEL.
/// The analysis runs on the function as lifted with no re-layout.
pub fn staged_layouts(g: &Graph, residency: &[Residency]) -> Vec<ParamLayout> {
    let mut out: Vec<ParamLayout> = vec![None; g.params.len()];
    for node in &g.nodes {
        for oi in 0..=node.inputs.len() {
            let o = if oi < node.inputs.len() { &node.inputs[oi] } else { &node.output };
            let Store::Param(p) = o.store else { continue };
            if residency.get(p) != Some(&Residency::Staged) {
                continue;
            }
            let Some(axis) = required(node, oi) else { continue };
            let kernel = node.func.family() == Family::Contraction && oi == 1;
            if on_stick(o, axis) && !kernel {
                continue;
            }
            // The memory-view axis the needed stick walks: the one whose stride it carries.
            let want = o.view.axes[axis].stride;
            let Some(mv) = g.param_views[p].first() else { continue };
            let Some(a) = mv.axes.iter().position(|x| x.stride == want) else { continue };
            let mut order: Vec<usize> = (0..mv.axes.len()).filter(|&x| x != a).collect();
            order.push(a);
            out[p] = Some(Held {
                order,
                inner: mv.axes[a].extent,
                columns: kernel,
            });
        }
    }
    out
}

/// Copy every operand some op reads off its stick into an intermediate that holds it there.
pub fn restick<G: ktir_core::ddl_binds::DeviceGen>(g: &mut Graph) {
    let mut i = 0;
    while i < g.nodes.len() {
        for oi in 0..g.nodes[i].inputs.len() {
            let Some(axis) = required(&g.nodes[i], oi) else { continue };
            let o = g.nodes[i].inputs[oi].clone();
            if on_stick(&o, axis) {
                continue;
            }
            // The copy: the same shape, `axis` innermost, the rest in the order they were held.
            let shape = o.view.shape();
            let mut order: Vec<usize> = (0..shape.len()).filter(|&a| a != axis).collect();
            order.sort_by_key(|&a| (std::cmp::Reverse(o.view.axes[a].stride), a));
            order.push(axis);
            let mut held: Vec<u64> = order.iter().map(|&a| shape[a]).collect();
            if let Some(l) = held.last_mut() {
                *l = l.next_multiple_of(o.format.lanes::<G>());
            }
            let physical = View::row_major(0, &held);
            let mut view = View::row_major(0, &shape);
            for (k, &a) in order.iter().enumerate() {
                view.axes[a].stride = physical.axes[k].stride;
            }
            g.locals.push(Local {
                elements: view.end(),
                format: o.format,
            });
            let copy = Operand {
                store: Store::Local(g.locals.len() - 1),
                view,
                format: o.format,
                gather: None,
            };
            g.nodes[i].inputs[oi] = copy.clone();
            g.nodes.insert(
                i,
                Node {
                    func: OpFunc::Restick,
                    space: shape,
                    inputs: vec![o],
                    output: copy,
                },
            );
            i += 1;
        }
        i += 1;
    }
}

/// The view an access through an index needs: the indexed axis one stick apart, directly outside
/// the stick's lanes, and the stick's own columns pitched past every row — dxp converts an index
/// to an address one stick per row.
fn indexable(v: &View, axis: usize, lanes: u64) -> bool {
    let stick = v.axes.iter().position(|a| a.stride == 1 && a.extent > 1);
    let one_column = stick.is_none_or(|s| v.axes[s].extent <= lanes || v.tile.is_some_and(|t| t.axis == s));
    v.axes[axis].stride == lanes && one_column
}

/// Is `v` read through an index along `axis` one whole row per index — the indexed axis outermost,
/// every other axis densely inside it, the unit-stride axis whole sticks? dxp addresses such a row
/// as one block per index, however many sticks it holds: a page of the KV pool gathers in place.
fn row_indexable(v: &View, axis: usize, lanes: u64) -> bool {
    if v.tile.is_some() {
        return false;
    }
    let mut rest: Vec<crate::view::Axis> = v
        .axes
        .iter()
        .enumerate()
        .filter(|&(i, a)| i != axis && a.extent > 1)
        .map(|(_, a)| *a)
        .collect();
    rest.sort_by_key(|a| a.stride);
    let mut unit = 1u64;
    for (k, a) in rest.iter().enumerate() {
        if a.stride != unit {
            return false;
        }
        unit *= if k == 0 { a.extent.next_multiple_of(lanes) } else { a.extent };
    }
    unit > lanes && v.axes[axis].stride == unit
}

/// `v` with its unit-stride axis `s` split into its sticks and their lanes — `[…, R·lanes, …]` read
/// as `[…, R, lanes, …]` — a row of several sticks as dxp's indirect access walks it.
fn split_sticks(v: &View, s: usize, lanes: u64) -> View {
    let mut out = v.clone();
    let a = out.axes[s];
    out.axes[s] = crate::view::Axis { extent: lanes, stride: a.stride };
    out.axes.insert(s, crate::view::Axis { extent: a.extent / lanes, stride: a.stride * lanes });
    out
}

/// `shape` row-major, its innermost axis whole sticks: the rows a row gather lands or reads.
fn row_view(shape: &[u64], lanes: u64) -> View {
    let mut held = shape.to_vec();
    if let Some(l) = held.last_mut() {
        *l = l.next_multiple_of(lanes);
    }
    let physical = View::row_major(0, &held);
    let mut v = View::row_major(0, shape);
    for (a, p) in v.axes.iter_mut().zip(&physical.axes) {
        a.stride = p.stride;
    }
    v
}

/// `shape` held with `axis` directly outside the stick's lanes and the stick axis `stick` in
/// columns pitched past all `rows` of it.
fn indexable_view(shape: &[u64], axis: usize, stick: usize, lanes: u64) -> View {
    let rows = shape[axis];
    let cols = shape[stick].next_multiple_of(lanes);
    let mut v = View::row_major(0, shape);
    v.axes[axis].stride = lanes;
    v.axes[stick].stride = 1;
    let mut unit = rows * cols;
    for a in (0..shape.len()).rev().filter(|&a| a != axis && a != stick) {
        v.axes[a].stride = unit;
        unit *= shape[a];
    }
    v.tile = Some(crate::view::Tile {
        axis: stick,
        lanes,
        pitch: rows * lanes,
    });
    v
}

/// Make every access through an index indexable: a read copies the indexed buffer into an
/// intermediate held that way first; a write lands in such an intermediate, between a copy of the
/// buffer in and a copy of it back out.
pub fn index_inner<G: ktir_core::ddl_binds::DeviceGen>(g: &mut Graph) {
    let mut i = 0;
    while i < g.nodes.len() {
        let node = &g.nodes[i];
        let (is_input, operand) = match (node.inputs.first(), &node.output) {
            (Some(o), _) if o.gather.is_some() => (true, o.clone()),
            (_, o) if o.gather.is_some() => (false, o.clone()),
            _ => {
                i += 1;
                continue;
            }
        };
        let Some(gather) = operand.gather.clone() else {
            i += 1;
            continue;
        };
        let lanes = operand.format.lanes::<G>();
        let axis = gather.axis;
        let Some(stick) = operand.view.axes.iter().position(|a| a.stride == 1 && a.extent > 1) else {
            i += 1;
            continue;
        };
        // A whole row per index reads (or lands) in place; the other side of the copy holds its
        // rows contiguous, one per index.
        let rows = row_indexable(&operand.view, axis, lanes);
        // The other side of the copy is held the same way, rows one per index.
        let other = if is_input { g.nodes[i].output.clone() } else { g.nodes[i].inputs[0].clone() };
        let other_shape = other.view.shape();
        let other_held = if rows {
            row_view(&other_shape, lanes)
        } else {
            indexable_view(&other_shape, axis, stick, lanes)
        };
        if other.view != other_held {
            g.locals.push(Local {
                elements: other_held.end(),
                format: other.format,
            });
            let staged = Operand {
                store: Store::Local(g.locals.len() - 1),
                view: other_held,
                format: other.format,
                gather: None,
            };
            let copy = Node {
                func: OpFunc::Identity,
                space: other_shape,
                inputs: vec![if is_input { staged.clone() } else { other.clone() }],
                output: if is_input { other } else { staged.clone() },
            };
            if is_input {
                g.nodes[i].output = staged;
                g.nodes.insert(i + 1, copy);
            } else {
                g.nodes[i].inputs[0] = staged;
                g.nodes.insert(i, copy);
                i += 1;
            }
        }
        if rows {
            // A row of several sticks: its sticks are one more axis of the copy, its lanes the
            // stick — on both sides, so the index walks whole rows of whole sticks.
            let s_extent = operand.view.axes[stick].extent;
            if s_extent > lanes && s_extent.is_multiple_of(lanes) {
                let node = &mut g.nodes[i];
                let split_axis = |o: &mut Operand| {
                    o.view = split_sticks(&o.view, stick, lanes);
                    if let Some(gt) = o.gather.as_mut()
                        && gt.axis > stick
                    {
                        gt.axis += 1;
                    }
                };
                split_axis(&mut node.inputs[0]);
                split_axis(&mut node.output);
                node.space[stick] = lanes;
                node.space.insert(stick, s_extent / lanes);
            }
            i += 1;
            continue;
        }
        if indexable(&operand.view, axis, lanes) {
            i += 1;
            continue;
        }
        // The whole buffer along the indexed axis, as the access's view states the rest of it.
        let mut whole = operand.view.clone();
        whole.axes[axis].extent = gather.rows;
        whole.tile = operand.view.tile;
        let shape = whole.shape();
        let held = indexable_view(&shape, axis, stick, lanes);
        g.locals.push(Local {
            elements: held.end(),
            format: operand.format,
        });
        let local = Store::Local(g.locals.len() - 1);
        let in_local = Operand {
            store: local,
            view: held.clone(),
            format: operand.format,
            gather: None,
        };
        let plain = Operand {
            gather: None,
            ..operand.clone()
        };
        let whole_buffer = Operand {
            view: whole,
            ..plain
        };
        let mut access = held.clone();
        access.axes[axis].extent = operand.view.axes[axis].extent;
        let rewritten = Operand {
            store: local,
            view: access,
            format: operand.format,
            gather: Some(gather),
        };
        let copy = |from: &Operand, to: &Operand| Node {
            func: OpFunc::Identity,
            space: shape.clone(),
            inputs: vec![from.clone()],
            output: to.clone(),
        };
        if is_input {
            g.nodes[i].inputs[0] = rewritten;
            g.nodes.insert(i, copy(&whole_buffer, &in_local));
            i += 2;
        } else {
            g.nodes[i].output = rewritten;
            g.nodes.insert(i + 1, copy(&in_local, &whole_buffer));
            g.nodes.insert(i, copy(&whole_buffer, &in_local));
            i += 3;
        }
    }
}

/// Does `o`'s window start inside a stick along its unit-stride axis? A descriptor addresses whole
/// sticks, so such a window is unreachable where it lies.
fn starts_inside_a_stick(o: &Operand, lanes: u64) -> Option<usize> {
    let s = o.view.axes.iter().position(|a| a.stride == 1 && a.extent > 1)?;
    let outer_aligned = o
        .view
        .axes
        .iter()
        .enumerate()
        .all(|(i, a)| i == s || a.extent == 1 || a.stride.is_multiple_of(lanes));
    (o.view.tile.is_none() && outer_aligned && !o.view.offset.is_multiple_of(lanes)).then_some(s)
}

/// The parameters some op reads or writes in a window that starts inside a stick: each is worked
/// on relocated — copied in sticked on another axis, and copied back whole when written.
pub fn misaligned<G: ktir_core::ddl_binds::DeviceGen>(g: &Graph) -> Vec<usize> {
    let mut moved: Vec<usize> = Vec::new();
    for node in &g.nodes {
        for o in node.inputs.iter().chain([&node.output]) {
            if let Store::Param(p) = o.store
                && starts_inside_a_stick(o, o.format.lanes::<G>()).is_some()
                && !moved.contains(&p)
            {
                moved.push(p);
            }
        }
    }
    moved
}
