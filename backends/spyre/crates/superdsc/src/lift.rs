// SPDX-License-Identifier: Apache-2.0
//! KTIR → [`Graph`]: run the function's dataflow symbolically.
//!
//! Every index is a compile-time constant (the grid, the loop bounds, the tile corners), so the
//! function is executed with indices as values and tensors as views: a load is a view of its
//! buffer, a shape op re-describes a view, and each tensor op becomes one device op writing a new
//! intermediate. Loops with constant trip counts unroll; each grid point runs the body once.

use std::collections::HashMap;

use ktir_core::attrkey::AttrKey;
use ktir_core::ddl_binds::DeviceGen;
use ktir_core::ir::{Attr, IRFunction, Operation, Ssa};
use ktir_core::opkind::OpKind;

use crate::format::DataFormat;
use crate::graph::{ConstStick, Gather, Graph, Local, Node, Operand, Store};
use crate::opfunc::{Axis as RAxis, Combine, Compare, OpFunc};
use crate::view::{Axis, View, ViewError};

/// Why a function did not lift.
#[derive(Clone, Debug, PartialEq)]
pub enum LiftError {
    /// An op this lowering has no device form for.
    Op(OpKind),
    /// An op whose operands or attributes are not the form its KTIR definition states.
    Malformed { op: OpKind, what: &'static str },
    /// An index the function computes from data rather than from constants.
    DynamicIndex { op: OpKind },
    /// A shape op the views cannot express.
    View { op: OpKind, error: ViewError },
    /// An SSA value read before any op defined it.
    Undefined(Ssa),
}

#[derive(Clone, Debug)]
enum Val {
    Int(i64),
    Float(f64),
    /// A parameter's address, `offset` elements in.
    Param {
        param: usize,
        offset: u64,
    },
    /// A memory view or an access tile over a store.
    Mem {
        store: Store,
        view: View,
        format: DataFormat,
    },
    /// An access tile read or written through an index tensor: `base` is the view with the indexed
    /// axis one row per index, `shape` the tile's own shape.
    Indirect {
        base: Operand,
        gather: Gather,
        shape: Vec<u64>,
    },
    /// An access tile at a row offset read from memory: its rows along `axis`, each one row
    /// read or written through the offset.
    Rows {
        axis: usize,
        rows: Vec<(Operand, Gather)>,
        shape: Vec<u64>,
    },
    /// An index the function reads from memory: `index[0] · scale + add`.
    Dyn {
        index: Operand,
        scale: u64,
        add: u64,
    },
    Tensor(Operand),
    Splat {
        value: f64,
        format: DataFormat,
        shape: Vec<u64>,
    },
    /// An uninitialized tensor: what a destination-passing op writes into.
    Empty,
    Tuple(Vec<Val>),
}

/// How a staged parameter's buffer holds its memory view: the view's axes, outermost first, in the
/// order the buffer stores them row-major, the innermost held `inner` elements wide (whole sticks,
/// at least the view's extent — the padding every op reading it divides over).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Held {
    pub order: Vec<usize>,
    pub inner: u64,
    /// The innermost axis held in stick columns (`[inner/lanes, rest…, lanes]`): the layout a
    /// contraction's KERNEL is read in, each core's slice of columns contiguous.
    pub columns: bool,
}

/// A parameter's [`Held`] layout, or `None` for a buffer held as its view states.
pub type ParamLayout = Option<Held>;

/// Lift `f` for generation `G`, each parameter's buffer held as `layouts` states.
pub fn lift<G: DeviceGen>(f: &IRFunction<'_>, layouts: &[ParamLayout]) -> Result<Graph, LiftError> {
    lift_relocating::<G>(f, layouts, &[])
}

/// [`lift`], with the parameters in `relocated` worked on relocated: copied at their first view
/// into an intermediate sticked on another axis, and copied back whole at the end when written.
pub fn lift_relocating<G: DeviceGen>(
    f: &IRFunction<'_>,
    layouts: &[ParamLayout],
    relocated: &[usize],
) -> Result<Graph, LiftError> {
    let mut cx = Cx::<G> {
        graph: Graph {
            params: vec![None; f.arguments.len()],
            param_views: vec![Vec::new(); f.arguments.len()],
            ..Graph::default()
        },
        env: HashMap::new(),
        consts: HashMap::new(),
        layouts,
        relocated,
        moved: HashMap::new(),
        _g: std::marker::PhantomData,
    };
    let (gx, gy, gz) = f.grid;
    for point in 0..gx.max(1) * gy.max(1) * gz.max(1) {
        cx.env.clear();
        for (i, (ssa, _)) in f.arguments.iter().enumerate() {
            cx.env.insert(
                *ssa,
                Val::Param {
                    param: i,
                    offset: 0,
                },
            );
        }
        let coord = [
            point % gx.max(1),
            (point / gx.max(1)) % gy.max(1),
            point / (gx.max(1) * gy.max(1)),
        ];
        cx.run(f.operations, &coord)?;
    }
    // Every relocated buffer the function wrote is copied back whole.
    let mut back: Vec<(usize, Store, View, View, DataFormat)> = cx
        .moved
        .iter()
        .map(|(&p, (local, lv, pv, fmt))| (p, *local, lv.clone(), pv.clone(), *fmt))
        .collect();
    back.sort_by_key(|b| b.0);
    for (p, local, lv, pv, format) in back {
        if cx
            .graph
            .nodes
            .iter()
            .filter(|n| n.output.store == local)
            .count()
            > 1
        {
            cx.graph.nodes.push(Node {
                func: OpFunc::Restick,
                space: pv.shape(),
                inputs: vec![Operand {
                    store: local,
                    view: lv,
                    format,
                    gather: None,
                }],
                output: Operand {
                    store: Store::Param(p),
                    view: pv,
                    format,
                    gather: None,
                },
            });
        }
    }
    forward_stores(&mut cx.graph);
    convert_through_index(&mut cx.graph);
    Ok(cx.graph)
}

/// `operands` re-stated over `space`, a space their own differs from only in axes of one position.
/// An operand read through an index keeps its space: its indexed axis is counted in it.
fn over_space(operands: &[Operand], space: &[u64]) -> Option<Vec<Operand>> {
    operands
        .iter()
        .map(|o| match &o.gather {
            Some(_) if o.view.axes.len() != space.len() => None,
            _ => Some(Operand {
                view: o.view.over_space(space)?,
                ..o.clone()
            }),
        })
        .collect()
}

/// A conversion reads its input through the index: rows gathered by a copy whose only reader is a
/// format conversion are converted straight out of the indexed buffer. dxp binds a copy for fp16
/// and fp32 words only, so an fp8 gather exists only as the conversion that widens it.
fn convert_through_index(g: &mut Graph) {
    let mut i = 0;
    while i < g.nodes.len() {
        let n = &g.nodes[i];
        let Store::Local(l) = n.output.store else {
            i += 1;
            continue;
        };
        if n.func != OpFunc::Identity || n.inputs.len() != 1 || n.inputs[0].gather.is_none() {
            i += 1;
            continue;
        }
        let touches = |m: &Node| {
            m.inputs.iter().any(|o| {
                o.store == Store::Local(l)
                    || o.gather
                        .as_ref()
                        .is_some_and(|gt| gt.index.store == Store::Local(l))
            })
        };
        let writers = g
            .nodes
            .iter()
            .filter(|m| m.output.store == Store::Local(l))
            .count();
        let readers: Vec<usize> = (0..g.nodes.len())
            .filter(|&j| touches(&g.nodes[j]))
            .collect();
        let fold = match readers[..] {
            [j] if writers == 1 && j > i => {
                let r = &g.nodes[j];
                (r.func.converts()
                    && r.inputs.len() == 1
                    && r.inputs[0].store == Store::Local(l)
                    && r.inputs[0].view.same_elements(&n.output.view)
                    && r.space == n.space)
                    .then_some(j)
            }
            _ => None,
        };
        let Some(j) = fold else {
            i += 1;
            continue;
        };
        let gathered = g.nodes[i].inputs[0].clone();
        g.nodes[j].inputs[0] = gathered;
        g.nodes.remove(i);
    }
}

struct Cx<'l, G> {
    graph: Graph,
    env: HashMap<Ssa, Val>,
    consts: HashMap<ConstStick, usize>,
    layouts: &'l [ParamLayout],
    relocated: &'l [usize],
    /// Each relocated parameter's copy: its store, its view, the parameter's view, its format.
    moved: HashMap<usize, (Store, View, View, DataFormat)>,
    _g: std::marker::PhantomData<G>,
}

fn int_list<'a>(op: &Operation<'a>, key: AttrKey) -> Option<&'a [i64]> {
    match op.attr(key) {
        Some(Attr::IntList(v)) => Some(v),
        _ => None,
    }
}

fn u64s(v: &[i64]) -> Vec<u64> {
    v.iter().map(|&x| x as u64).collect()
}

impl<G: DeviceGen> Cx<'_, G> {
    fn get(&self, s: Ssa) -> Result<Val, LiftError> {
        self.env.get(&s).cloned().ok_or(LiftError::Undefined(s))
    }

    fn bind(&mut self, op: &Operation<'_>, v: Val) {
        match (op.attr(AttrKey::ResultNames), v) {
            (Some(Attr::Ssas(names)), Val::Tuple(vs)) if names.len() > 1 => {
                for (n, v) in names.iter().zip(vs) {
                    self.env.insert(*n, v);
                }
            }
            (_, v) => {
                if let Some(r) = op.result {
                    self.env.insert(r, v);
                }
            }
        }
    }

    fn int(&self, op: &Operation<'_>, s: Ssa) -> Result<i64, LiftError> {
        match self.get(s)? {
            Val::Int(i) => Ok(i),
            _ => Err(LiftError::DynamicIndex { op: op.op_type }),
        }
    }

    fn format(op: &Operation<'_>) -> Result<DataFormat, LiftError> {
        match (
            op.attr(AttrKey::Dtype),
            op.result_type.and_then(|t| t.elem()),
        ) {
            (Some(Attr::Dtype(d)), _) => Ok(DataFormat::of(*d)),
            (_, Some(d)) => Ok(DataFormat::of(d)),
            _ => Err(LiftError::Malformed {
                op: op.op_type,
                what: "a tensor op states its element type",
            }),
        }
    }

    fn shape(op: &Operation<'_>) -> Result<Vec<u64>, LiftError> {
        int_list(op, AttrKey::Shape)
            .map(u64s)
            .ok_or(LiftError::Malformed {
                op: op.op_type,
                what: "a tensor op states its shape",
            })
    }

    /// An access tile's `coordinate_order` when it is not the identity: result axis `i` is
    /// memory axis `order[i]`.
    fn coordinate_order(op: &Operation<'_>) -> Result<Option<Vec<usize>>, LiftError> {
        let Some(Attr::AffineMap(m)) = op.attr(AttrKey::CoordinateOrder) else {
            return Ok(None);
        };
        let order = m
            .exprs
            .iter()
            .map(|e| match e {
                ktir_core::affine::AffineExpr::Dim(d) => Ok(*d),
                _ => Err(LiftError::Malformed {
                    op: op.op_type,
                    what: "a coordinate order permutes the tile's axes",
                }),
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((order.iter().enumerate().any(|(i, &d)| i != d)).then_some(order))
    }

    fn const_stick(&mut self, format: DataFormat, value: f64) -> usize {
        let key = ConstStick {
            format,
            bits: (value as f32).to_bits(),
        };
        let n = self.graph.consts.len();
        *self.consts.entry(key).or_insert_with(|| {
            self.graph.consts.push(key);
            n
        })
    }

    /// A value as an operand over `shape`.
    fn operand(
        &mut self,
        op: &Operation<'_>,
        v: &Val,
        shape: &[u64],
        format: DataFormat,
    ) -> Result<Operand, LiftError> {
        match v {
            Val::Tensor(o) => {
                let mut o = o.clone();
                if o.view.axes.len() == shape.len() {
                    o.view = o.view.broadcast_to(shape);
                }
                Ok(o)
            }
            Val::Splat { value, format, .. } => {
                let c = self.const_stick(*format, *value);
                Ok(Operand {
                    store: Store::Const(c),
                    view: View::of_memory(shape, &vec![0; shape.len()]),
                    format: *format,
                    gather: None,
                })
            }
            Val::Float(value) => {
                let c = self.const_stick(format, *value);
                Ok(Operand {
                    store: Store::Const(c),
                    view: View::of_memory(shape, &vec![0; shape.len()]),
                    format,
                    gather: None,
                })
            }
            Val::Int(value) => {
                let c = self.const_stick(format, *value as f64);
                Ok(Operand {
                    store: Store::Const(c),
                    view: View::of_memory(shape, &vec![0; shape.len()]),
                    format,
                    gather: None,
                })
            }
            _ => Err(LiftError::Malformed {
                op: op.op_type,
                what: "a tensor operand is a tensor or a scalar",
            }),
        }
    }

    /// A fresh intermediate for a result of `shape`. A result whose dense inputs all hold one
    /// value per stick (a reduction's result and what is computed from it) is held the same way.
    fn local(&mut self, shape: &[u64], format: DataFormat, per_stick: bool) -> Operand {
        self.local_like(shape, format, per_stick, None)
    }

    /// A fresh intermediate whose axes are stored in the order of `like`'s strides (outermost
    /// first), else row-major.
    fn local_like(
        &mut self,
        shape: &[u64],
        format: DataFormat,
        per_stick: bool,
        like: Option<&View>,
    ) -> Operand {
        let lanes = format.lanes::<G>();
        let numel: u64 = shape.iter().product();
        // The innermost axis is held in whole sticks: a partial stick pads to the stick.
        let padded = |v: View| -> View {
            let Some(inner) = v
                .axes
                .iter()
                .position(|a| a.stride == 1 && a.extent > 1)
                .or_else(|| v.axes.iter().position(|a| a.stride == 1))
            else {
                return v;
            };
            let pitch = v.axes[inner].extent.next_multiple_of(lanes);
            let mut out = v.clone();
            for (i, a) in v.axes.iter().enumerate() {
                if i != inner && a.stride != 0 {
                    out.axes[i].stride = a.stride / v.axes[inner].extent * pitch;
                }
            }
            out
        };
        let ordered = |s: &[u64]| -> View {
            padded(match like {
                Some(l) if l.axes.len() == s.len() && l.axes.iter().any(|a| a.stride == 1) => {
                    let mut order: Vec<usize> = (0..s.len()).collect();
                    // An axis `like` repeats along (stride 0) holds nothing of its own: outermost.
                    order.sort_by_key(|&a| {
                        (
                            l.axes[a].stride != 0,
                            std::cmp::Reverse(l.axes[a].stride),
                            a,
                        )
                    });
                    let held: Vec<u64> = order.iter().map(|&a| s[a]).collect();
                    let physical = View::row_major(0, &held);
                    let mut v = View::row_major(0, s);
                    for (k, &a) in order.iter().enumerate() {
                        v.axes[a].stride = physical.axes[k].stride;
                    }
                    v
                }
                _ => View::row_major(0, s),
            })
        };
        let (elements, view) = if per_stick {
            let mut v = View::row_major(0, shape);
            for a in &mut v.axes {
                a.stride *= lanes;
            }
            (numel * lanes, v)
        } else {
            let v = ordered(shape);
            (v.end(), v)
        };
        self.graph.locals.push(Local { elements, format });
        Operand {
            store: Store::Local(self.graph.locals.len() - 1),
            view,
            format,
            gather: None,
        }
    }

    /// Does `o` hold one value per stick (no axis is the lanes of a stick)?
    fn per_stick(o: &Operand) -> bool {
        matches!(o.store, Store::Local(_) | Store::Param(_))
            && o.view.axes.iter().all(|a| a.stride != 1 || a.extent == 1)
            && o.view.axes.iter().any(|a| a.stride != 0)
    }

    fn pointwise(&mut self, op: &Operation<'_>, func: OpFunc) -> Result<(), LiftError> {
        let shape = Self::shape(op)?;
        let format = Self::format(op)?;
        let in_format = match func {
            OpFunc::Cmp(_) | OpFunc::Select => format,
            _ => format,
        };
        let mut inputs = Vec::with_capacity(op.operands.len());
        for &s in op.operands {
            let v = self.get(s)?;
            inputs.push(self.operand(op, &v, &shape, in_format)?);
        }
        let dense: Vec<&Operand> = inputs
            .iter()
            .filter(|o| o.view.axes.iter().any(|a| a.stride != 0))
            .collect();
        // The result holds one value per stick when every input it reads does: a per-stick input
        // repeated along an outer axis reads the same stick at every position of it. Repeated
        // along the innermost axis it is a broadcast INTO a stick, which materializes as whole
        // sticks.
        let per_stick = !dense.is_empty()
            && dense.iter().all(|o| {
                Self::per_stick(o)
                    && o.view
                        .axes
                        .last()
                        .is_none_or(|a| a.stride != 0 || a.extent == 1)
            });
        let like = dense
            .iter()
            .find(|o| o.view.axes.iter().any(|a| a.stride == 1 && a.extent > 1))
            .or(dense.first())
            .map(|o| o.view.clone());
        let output = self.local_like(&shape, format, per_stick, like.as_ref());
        self.graph.nodes.push(Node {
            func,
            space: shape,
            inputs,
            output: output.clone(),
        });
        self.bind(op, Val::Tensor(output));
        Ok(())
    }

    fn reduce(&mut self, op: &Operation<'_>) -> Result<(), LiftError> {
        let combine = match op.attr(AttrKey::ReduceFn) {
            Some(Attr::Op(OpKind::ArithAddf)) => Combine::Sum,
            Some(Attr::Op(OpKind::ArithMaxnumf | OpKind::ArithMaximumf | OpKind::ArithMaxf)) => {
                Combine::Max
            }
            Some(Attr::Op(OpKind::ArithMinnumf | OpKind::ArithMinimumf | OpKind::ArithMinf)) => {
                Combine::Min
            }
            Some(Attr::Op(k)) => return Err(LiftError::Op(*k)),
            _ => {
                return Err(LiftError::Malformed {
                    op: op.op_type,
                    what: "linalg.reduce states its combiner",
                });
            }
        };
        let dims = int_list(op, AttrKey::Dimensions).ok_or(LiftError::Malformed {
            op: op.op_type,
            what: "linalg.reduce states its dimensions",
        })?;
        let format = Self::format(op)?;
        let [axis] = dims else {
            return Err(LiftError::Malformed {
                op: op.op_type,
                what: "linalg.reduce folds one dimension",
            });
        };
        let axis = *axis as usize;
        let input = match self.get(op.operands[0])? {
            Val::Tensor(o) => o,
            _ => {
                return Err(LiftError::Malformed {
                    op: op.op_type,
                    what: "linalg.reduce reads a tensor",
                });
            }
        };
        let space = input.view.shape();
        let stick = input.view.axes[axis].stride == 1;
        let out_shape: Vec<u64> = space
            .iter()
            .enumerate()
            .filter(|&(i, _)| i != axis)
            .map(|(_, &e)| e)
            .collect();
        // A stick reduction leaves one value per stick, and so does any reduction of an input
        // that already holds one.
        let result = self.local(&out_shape, format, stick || Self::per_stick(&input));
        let mut out = result.clone();
        out.view.axes.insert(
            axis,
            crate::view::Axis {
                extent: space[axis],
                stride: 0,
            },
        );
        let func = OpFunc::Reduce(combine, if stick { RAxis::Stick } else { RAxis::NonStick });
        self.graph.nodes.push(Node {
            func,
            space,
            inputs: vec![input],
            output: out,
        });
        // A non-identity init folds in after the reduction.
        let init = self.get(op.operands[1])?;
        let folded = match (&init, func.identity()) {
            (Val::Splat { value, .. }, Some(id)) if *value == id => None,
            _ => Some(init),
        };
        let value = match folded {
            None => result,
            Some(init) => {
                let rhs = self.operand(op, &init, &out_shape, format)?;
                let combine_op = match combine {
                    Combine::Sum => OpFunc::Add,
                    Combine::Max => OpFunc::Maximum,
                    Combine::Min => OpFunc::Minimum,
                };
                let per_stick = Self::per_stick(&result);
                let o = self.local(&out_shape, format, per_stick);
                self.graph.nodes.push(Node {
                    func: combine_op,
                    space: out_shape.clone(),
                    inputs: vec![result, rhs],
                    output: o.clone(),
                });
                o
            }
        };
        self.bind(op, Val::Tensor(value));
        Ok(())
    }

    fn matmul(&mut self, op: &Operation<'_>, batch: bool) -> Result<(), LiftError> {
        let malformed = |what| LiftError::Malformed {
            op: op.op_type,
            what,
        };
        let (Val::Tensor(a), Val::Tensor(b)) =
            (self.get(op.operands[0])?, self.get(op.operands[1])?)
        else {
            return Err(malformed("a contraction reads two tensors"));
        };
        let transpose_b = match op.attr(AttrKey::IndexingMaps) {
            Some(Attr::AffineMapList(maps)) if maps.len() == 3 => {
                match maps[1].result_dims().as_deref() {
                    Some([2, 1]) | Some([0, 3, 2]) => false,
                    Some([1, 2]) | Some([0, 2, 3]) => true,
                    _ => return Err(malformed("the B map is (k, n) or (n, k)")),
                }
            }
            None => false,
            Some(_) => return Err(malformed("indexing_maps lists A, B and C")),
        };
        let rank = a.view.axes.len();
        let b = if transpose_b {
            let mut perm: Vec<usize> = (0..rank).collect();
            perm.swap(rank - 2, rank - 1);
            let view = b.view.permute(&perm).map_err(|error| LiftError::View {
                op: op.op_type,
                error,
            })?;
            Operand { view, ..b }
        } else {
            b
        };
        let (m, k) = (a.view.axes[rank - 2].extent, a.view.axes[rank - 1].extent);
        let n = b.view.axes[rank - 1].extent;
        let batches: Vec<u64> = a.view.axes[..rank - 2].iter().map(|x| x.extent).collect();
        if batch != !batches.is_empty() {
            return Err(malformed("a batch matmul carries a batch axis"));
        }
        let format = Self::format(op)?;
        let mut out_shape = batches.clone();
        out_shape.extend([m, n]);
        let mut space = batches;
        space.extend([m, n, k]);
        let func = if b.format == DataFormat::Fp8 {
            OpFunc::MatmulFp8
        } else {
            OpFunc::Matmul
        };
        let product = self.local(&out_shape, format, false);
        self.graph.nodes.push(Node {
            func,
            space,
            inputs: vec![a, b],
            output: product.clone(),
        });
        // `outs` accumulates: a non-zero accumulator adds in after the contraction.
        let acc = self.get(op.operands[2])?;
        let value = match acc {
            Val::Splat { value: 0.0, .. } | Val::Empty => product,
            acc => {
                let rhs = self.operand(op, &acc, &out_shape, format)?;
                let o = self.local(&out_shape, format, false);
                self.graph.nodes.push(Node {
                    func: OpFunc::Add,
                    space: out_shape,
                    inputs: vec![product, rhs],
                    output: o.clone(),
                });
                o
            }
        };
        self.bind(op, Val::Tensor(value));
        Ok(())
    }

    fn view_op(
        &mut self,
        op: &Operation<'_>,
        f: impl Fn(&View) -> Result<View, ViewError>,
    ) -> Result<(), LiftError> {
        let v = match self.get(op.operands[0])? {
            Val::Tensor(o) => match f(&o.view) {
                Ok(view) => Val::Tensor(Operand { view, ..o }),
                // A value the shape op cannot re-describe in place is copied out contiguous first,
                // which every shape op can.
                Err(ViewError::NotContiguous { .. } | ViewError::Tiled) => {
                    let shape = o.view.shape();
                    let copy = self.local(&shape, o.format, false);
                    self.graph.nodes.push(Node {
                        func: OpFunc::Identity,
                        space: shape,
                        inputs: vec![o],
                        output: copy.clone(),
                    });
                    let view = f(&copy.view).map_err(|error| LiftError::View {
                        op: op.op_type,
                        error,
                    })?;
                    Val::Tensor(Operand { view, ..copy })
                }
                Err(error) => {
                    return Err(LiftError::View {
                        op: op.op_type,
                        error,
                    });
                }
            },
            Val::Splat { value, format, .. } => Val::Splat {
                value,
                format,
                shape: Self::shape(op)?,
            },
            _ => {
                return Err(LiftError::Malformed {
                    op: op.op_type,
                    what: "a shape op reads a tensor",
                });
            }
        };
        self.bind(op, v);
        Ok(())
    }

    /// An access tile whose corner along axis `a` is read from memory: `index · scale + add`. The
    /// axis splits into blocks of the tile's extent, and the index names one block — so the
    /// corner must be a whole number of blocks (`scale` is the tile's extent, `add` none).
    fn block_tile(
        &mut self,
        op: &Operation<'_>,
        (store, view, format): (Store, &View, DataFormat),
        (corner, shape): (&[u64], &[u64]),
        (a, index, scale, add): (usize, Operand, u64, u64),
    ) -> Result<Val, LiftError> {
        let e = shape[a];
        let axis = view.axes[a];
        if scale != e || add != 0 || axis.extent % e != 0 {
            return Err(LiftError::DynamicIndex { op: op.op_type });
        }
        let mut base = view
            .window(corner, &{
                let mut s = shape.to_vec();
                s[a] = axis.extent;
                s
            })
            .map_err(|error| LiftError::View {
                op: op.op_type,
                error,
            })?;
        // Axis `a` as (block, row in block): the index walks blocks.
        base.axes[a] = crate::view::Axis {
            extent: 1,
            stride: axis.stride * e,
        };
        base.axes.insert(
            a + 1,
            crate::view::Axis {
                extent: e,
                stride: axis.stride,
            },
        );
        if base.tile.is_some() {
            return Err(LiftError::View {
                op: op.op_type,
                error: ViewError::Tiled,
            });
        }
        Ok(Val::Indirect {
            base: Operand {
                store,
                view: base,
                format,
                gather: None,
            },
            gather: Gather {
                axis: a,
                index: Box::new(index),
                rows: axis.extent / e,
            },
            shape: shape.to_vec(),
        })
    }

    /// An access tile whose corner along axis `a` is a row read from memory, `index + add`: its
    /// `e` rows, row `k` the one-row block the index names of the axis from `add + k` on.
    fn row_tiles(
        &mut self,
        op: &Operation<'_>,
        (store, view, format): (Store, &View, DataFormat),
        (corner, shape): (&[u64], &[u64]),
        (a, index, add): (usize, Operand, u64),
    ) -> Result<Val, LiftError> {
        let mut one = shape.to_vec();
        one[a] = 1;
        let mut rows = Vec::with_capacity(shape[a] as usize);
        for k in 0..shape[a] {
            let mut at = vec![0u64; view.axes.len()];
            at[a] = add + k;
            let mut extents: Vec<u64> = view.axes.iter().map(|x| x.extent).collect();
            extents[a] = extents[a]
                .checked_sub(add + k)
                .filter(|&n| n > 0)
                .ok_or(LiftError::DynamicIndex { op: op.op_type })?;
            let from = view
                .window(&at, &extents)
                .map_err(|error| LiftError::View {
                    op: op.op_type,
                    error,
                })?;
            let Val::Indirect { base, gather, .. } = self.block_tile(
                op,
                (store, &from, format),
                (corner, &one),
                (a, index.clone(), 1, 0),
            )?
            else {
                return Err(LiftError::Malformed {
                    op: op.op_type,
                    what: "a row of a tile",
                });
            };
            rows.push((base, gather));
        }
        Ok(Val::Rows {
            axis: a,
            rows,
            shape: shape.to_vec(),
        })
    }

    /// `ktdp.construct_indirect_access_tile`: along each `indirect` dimension the row is the index
    /// view's entry at `offset + d`; along each `direct_sub` dimension the coordinate is
    /// `offset + d`.
    fn indirect_tile(&mut self, op: &Operation<'_>) -> Result<(), LiftError> {
        let malformed = |what| LiftError::Malformed {
            op: op.op_type,
            what,
        };
        let Val::Mem {
            store,
            view,
            format,
        } = self.get(op.operands[0])?
        else {
            return Err(malformed("an indirect tile indexes a memory view"));
        };
        let shape = Self::shape(op)?;
        let kinds = match op.attr(AttrKey::DimKinds) {
            Some(Attr::StrList(k)) => *k,
            _ => return Err(malformed("an indirect tile states its dimension kinds")),
        };
        let data = int_list(op, AttrKey::DimData)
            .ok_or(malformed("an indirect tile states its index views"))?;
        let offsets = match op.attr(AttrKey::IntermediateVars) {
            Some(Attr::Ssas(v)) => v.to_vec(),
            _ => return Err(malformed("an indirect tile states its offsets")),
        };
        let mut corner = vec![0u64; shape.len()];
        let mut indirect = None;
        for (d, kind) in kinds.iter().enumerate() {
            match *kind {
                "direct_sub" => corner[d] = self.int(op, offsets[d])? as u64,
                "indirect" if indirect.is_none() => {
                    let Val::Mem {
                        store: istore,
                        view: iview,
                        format: ifmt,
                    } = self.get(op.operands[1 + data[d] as usize])?
                    else {
                        return Err(malformed("an index view is a memory view"));
                    };
                    let n = shape[d];
                    // The index entries: a static window of the index view, or a block of it at an
                    // offset read from memory, gathered first.
                    let index = match self.get(offsets[d])? {
                        Val::Int(o) => Operand {
                            store: istore,
                            view: iview.window(&[o as u64], &[n]).map_err(|error| {
                                LiftError::View {
                                    op: op.op_type,
                                    error,
                                }
                            })?,
                            format: ifmt,
                            gather: None,
                        },
                        Val::Dyn { index, scale, add } => {
                            let Val::Indirect {
                                base,
                                gather,
                                shape: s,
                            } = self.block_tile(
                                op,
                                (istore, &iview, ifmt),
                                (&[0], &[n]),
                                (0, index, scale, add),
                            )?
                            else {
                                return Err(malformed("a block of an index view"));
                            };
                            let value = Operand {
                                gather: Some(gather),
                                ..base
                            };
                            let out = self.local(&value.view.shape(), ifmt, false);
                            self.graph.nodes.push(Node {
                                func: OpFunc::Identity,
                                space: value.view.shape(),
                                inputs: vec![value],
                                output: out.clone(),
                            });
                            let v = out.view.reshape(&s).map_err(|error| LiftError::View {
                                op: op.op_type,
                                error,
                            })?;
                            Operand { view: v, ..out }
                        }
                        _ => return Err(LiftError::DynamicIndex { op: op.op_type }),
                    };
                    self.note_param(istore, ifmt);
                    indirect = Some((d, index));
                }
                _ => return Err(malformed("an indirect tile has one indirect dimension")),
            }
        }
        let Some((d, index)) = indirect else {
            return Err(malformed("an indirect tile has an indirect dimension"));
        };
        let rows = view.axes[d].extent;
        let mut full = shape.clone();
        full[d] = rows;
        let mut base = view
            .window(&corner, &full)
            .map_err(|error| LiftError::View {
                op: op.op_type,
                error,
            })?;
        base.axes[d].extent = shape[d];
        self.bind(
            op,
            Val::Indirect {
                base: Operand {
                    store,
                    view: base,
                    format,
                    gather: None,
                },
                gather: Gather {
                    axis: d,
                    index: Box::new(index),
                    rows,
                },
                shape,
            },
        );
        Ok(())
    }

    /// The relocated copy of parameter `param` seen through `view`: made at its first view, sticked
    /// on the axis outermost in the view (padded to whole sticks), the rest in their own order.
    fn relocate(&mut self, param: usize, view: &View, format: DataFormat) -> (Store, View) {
        if let Some((store, lv, _, _)) = self.moved.get(&param) {
            return (*store, lv.clone());
        }
        let shape = view.shape();
        let lanes = format.lanes::<G>();
        let t = (0..shape.len())
            .filter(|&a| view.axes[a].stride != 1 && shape[a] > 1)
            .max_by_key(|&a| view.axes[a].stride)
            .unwrap_or(0);
        let mut order: Vec<usize> = (0..shape.len()).filter(|&a| a != t).collect();
        order.sort_by_key(|&a| (std::cmp::Reverse(view.axes[a].stride), a));
        order.push(t);
        let mut held: Vec<u64> = order.iter().map(|&a| shape[a]).collect();
        if let Some(l) = held.last_mut() {
            *l = l.next_multiple_of(lanes);
        }
        let physical = View::row_major(0, &held);
        let mut lv = View::row_major(0, &shape);
        for (k, &a) in order.iter().enumerate() {
            lv.axes[a].stride = physical.axes[k].stride;
        }
        self.graph.locals.push(Local {
            elements: lv.end(),
            format,
        });
        let store = Store::Local(self.graph.locals.len() - 1);
        self.graph.nodes.push(Node {
            func: OpFunc::Restick,
            space: shape,
            inputs: vec![Operand {
                store: Store::Param(param),
                view: view.clone(),
                format,
                gather: None,
            }],
            output: Operand {
                store,
                view: lv.clone(),
                format,
                gather: None,
            },
        });
        self.moved
            .insert(param, (store, lv.clone(), view.clone(), format));
        (store, lv)
    }

    fn store(&mut self, op: &Operation<'_>) -> Result<(), LiftError> {
        let value = self.get(op.operands[0])?;
        let (store, view, format) = match self.get(op.operands[1])? {
            Val::Mem {
                store,
                view,
                format,
            } => (store, view, format),
            Val::Indirect {
                base,
                gather,
                shape,
            } => {
                // The rows the index names, written: a scatter.
                let input = self.operand(op, &value, &shape, base.format)?;
                let rows = base.view.shape();
                let input = Operand {
                    view: input.view.reshape(&rows).map_err(|error| LiftError::View {
                        op: op.op_type,
                        error,
                    })?,
                    ..input
                };
                self.note_param(base.store, base.format);
                self.graph.nodes.push(Node {
                    func: OpFunc::Identity,
                    space: rows,
                    inputs: vec![input],
                    output: Operand {
                        gather: Some(gather),
                        ..base
                    },
                });
                return Ok(());
            }
            Val::Rows { axis, rows, shape } => {
                // Each row of the value, written to the row the offset names.
                let format = rows
                    .first()
                    .map(|(b, _)| b.format)
                    .ok_or(LiftError::Malformed {
                        op: op.op_type,
                        what: "a tile has rows",
                    })?;
                let input = self.operand(op, &value, &shape, format)?;
                for (k, (base, gather)) in rows.into_iter().enumerate() {
                    let space = base.view.shape();
                    let row =
                        Self::row_of(&input.view, axis, k as u64, &space).map_err(|error| {
                            LiftError::View {
                                op: op.op_type,
                                error,
                            }
                        })?;
                    self.note_param(base.store, base.format);
                    self.graph.nodes.push(Node {
                        func: OpFunc::Identity,
                        space,
                        inputs: vec![Operand {
                            view: row,
                            ..input.clone()
                        }],
                        output: Operand {
                            gather: Some(gather),
                            ..base
                        },
                    });
                }
                return Ok(());
            }
            _ => {
                return Err(LiftError::Malformed {
                    op: op.op_type,
                    what: "ktdp.store writes an access tile",
                });
            }
        };
        let shape = view.shape();
        let input = self.operand(op, &value, &shape, format)?;
        self.note_param(store, format);
        self.graph.nodes.push(Node {
            func: OpFunc::Identity,
            space: shape,
            inputs: vec![input],
            output: Operand {
                store,
                view,
                format,
                gather: None,
            },
        });
        Ok(())
    }

    /// Row `k` of `v` along `axis`, seen as `space`.
    fn row_of(v: &View, axis: usize, k: u64, space: &[u64]) -> Result<View, ViewError> {
        let mut at = vec![0u64; v.axes.len()];
        at[axis] = k;
        let mut one = v.shape();
        one[axis] = 1;
        v.window(&at, &one)?.reshape(space)
    }

    fn note_param(&mut self, store: Store, format: DataFormat) {
        if let Store::Param(p) = store {
            self.graph.params[p] = Some(format);
        }
    }

    fn run(
        &mut self,
        ops: &[Operation<'_>],
        tile: &[usize; 3],
    ) -> Result<Option<Vec<Val>>, LiftError> {
        for op in ops {
            if let Some(y) = self.step(op, tile)? {
                return Ok(Some(y));
            }
        }
        Ok(None)
    }

    fn step(
        &mut self,
        op: &Operation<'_>,
        tile: &[usize; 3],
    ) -> Result<Option<Vec<Val>>, LiftError> {
        use OpKind as K;
        let malformed = |what| LiftError::Malformed {
            op: op.op_type,
            what,
        };
        match op.op_type {
            K::ArithConstant => {
                let v = match op.attr(AttrKey::Value) {
                    Some(Attr::Int(i)) => Val::Int(*i),
                    Some(Attr::Float(f)) => Val::Float(*f),
                    Some(Attr::Bool(b)) => Val::Int(i64::from(*b)),
                    _ => return Err(malformed("arith.constant states its value")),
                };
                self.bind(op, v);
            }
            K::KtdpGetComputeTileId => {
                let n = match op.attr(AttrKey::ResultNames) {
                    Some(Attr::Ssas(names)) => names.len(),
                    _ => 1,
                };
                let v = if n > 1 {
                    Val::Tuple(
                        tile[..n.min(3)]
                            .iter()
                            .map(|&t| Val::Int(t as i64))
                            .collect(),
                    )
                } else {
                    Val::Int(tile[0] as i64)
                };
                self.bind(op, v);
            }
            K::ArithIndexCast
            | K::ArithIndexCastui
            | K::ArithExtsi
            | K::ArithExtui
            | K::ArithTrunci => {
                let v = self.get(op.operands[0])?;
                self.bind(op, v);
            }
            K::ArithAddi
            | K::ArithSubi
            | K::ArithMuli
            | K::ArithDivsi
            | K::ArithDivui
            | K::ArithRemsi
            | K::ArithRemui
            | K::ArithMinsi
            | K::ArithMaxsi
            | K::ArithFloordivsi
            | K::ArithCeildivsi => {
                let (a, b) = (self.get(op.operands[0])?, self.get(op.operands[1])?);
                let v = match (a, b) {
                    (Val::Int(x), Val::Int(y)) => Val::Int(match op.op_type {
                        K::ArithAddi => x + y,
                        K::ArithSubi => x - y,
                        K::ArithMuli => x * y,
                        K::ArithDivsi | K::ArithDivui => x / y,
                        K::ArithRemsi | K::ArithRemui => x % y,
                        K::ArithMinsi => x.min(y),
                        K::ArithMaxsi => x.max(y),
                        K::ArithFloordivsi => x.div_euclid(y),
                        _ => (x + y - 1).div_euclid(y),
                    }),
                    (Val::Param { param, offset }, Val::Int(y))
                    | (Val::Int(y), Val::Param { param, offset })
                        if op.op_type == K::ArithAddi =>
                    {
                        Val::Param {
                            param,
                            offset: offset + y as u64,
                        }
                    }
                    (Val::Dyn { index, scale, add }, Val::Int(y))
                    | (Val::Int(y), Val::Dyn { index, scale, add })
                        if y >= 0 =>
                    {
                        match op.op_type {
                            K::ArithAddi => Val::Dyn {
                                index,
                                scale,
                                add: add + y as u64,
                            },
                            K::ArithMuli => Val::Dyn {
                                index,
                                scale: scale * y as u64,
                                add: add * y as u64,
                            },
                            _ => return Err(LiftError::DynamicIndex { op: op.op_type }),
                        }
                    }
                    _ => return Err(LiftError::DynamicIndex { op: op.op_type }),
                };
                self.bind(op, v);
            }
            K::KtdpConstructMemoryView => {
                let shape = int_list(op, AttrKey::Shape)
                    .map(u64s)
                    .ok_or(malformed("a view states its shape"))?;
                // A view stating no strides is row-major over its shape.
                let strides = int_list(op, AttrKey::Strides).map(u64s).unwrap_or_else(|| {
                    View::row_major(0, &shape)
                        .axes
                        .iter()
                        .map(|a| a.stride)
                        .collect()
                });
                let format = Self::format(op)?;
                let (param, offset) = match self.get(op.operands[0])? {
                    Val::Param { param, offset } => (param, offset),
                    _ => return Err(malformed("a memory view's base is a parameter")),
                };
                let mut view = View::of_memory(&shape, &strides);
                if let Some(Some(Held {
                    order,
                    inner,
                    columns,
                })) = self.layouts.get(param)
                {
                    if view != View::row_major(0, &shape) {
                        return Err(malformed("a re-laid-out buffer's view is row-major"));
                    }
                    let lanes = format.lanes::<G>();
                    let (outer, last) = order.split_at(order.len() - 1);
                    let inner_axis = last[0];
                    if *columns {
                        // Stick columns: the outer axes row-major, one stick per position, and
                        // the innermost axis's sticks a whole column apart.
                        let held: Vec<u64> = outer.iter().map(|&a| shape[a]).collect();
                        let physical = View::row_major(0, &held);
                        for (k, &a) in outer.iter().enumerate() {
                            view.axes[a].stride = physical.axes[k].stride * lanes;
                        }
                        view.axes[inner_axis].stride = 1;
                        view.tile = Some(crate::view::Tile {
                            axis: inner_axis,
                            lanes,
                            pitch: held.iter().product::<u64>() * lanes,
                        });
                    } else {
                        // Row-major in `order`, the innermost axis `inner` wide.
                        let mut held: Vec<u64> = order.iter().map(|&a| shape[a]).collect();
                        if let Some(l) = held.last_mut() {
                            *l = (*inner).max(l.next_multiple_of(lanes));
                        }
                        let physical = View::row_major(0, &held);
                        for (k, &a) in order.iter().enumerate() {
                            view.axes[a].stride = physical.axes[k].stride;
                        }
                    }
                }
                view.offset = offset;
                self.graph.param_views[param].push(view.clone());
                let v = if self.relocated.contains(&param) {
                    let (store, view) = self.relocate(param, &view, format);
                    Val::Mem {
                        store,
                        view,
                        format,
                    }
                } else {
                    Val::Mem {
                        store: Store::Param(param),
                        view,
                        format,
                    }
                };
                self.bind(op, v);
            }
            K::KtdpConstructAccessTile => {
                let Val::Mem {
                    store,
                    view,
                    format,
                } = self.get(op.operands[0])?
                else {
                    return Err(malformed("an access tile windows a memory view"));
                };
                // A tile read in another order states its shape in that order: the window is the
                // memory-order block, read permuted.
                let order = Self::coordinate_order(op)?;
                let shape = {
                    let stated = Self::shape(op)?;
                    match &order {
                        Some(o) => {
                            let mut block = vec![0u64; stated.len()];
                            for (i, &a) in o.iter().enumerate() {
                                block[a] = stated[i];
                            }
                            block
                        }
                        None => stated,
                    }
                };
                let mut corner = Vec::with_capacity(op.operands.len() - 1);
                let mut dynamic = None;
                for (a, &s) in op.operands[1..].iter().enumerate() {
                    match self.get(s)? {
                        Val::Int(i) => corner.push(i as u64),
                        Val::Dyn { index, scale, add } if dynamic.is_none() => {
                            dynamic = Some((a, index, scale, add));
                            corner.push(0);
                        }
                        _ => return Err(LiftError::DynamicIndex { op: op.op_type }),
                    }
                }
                let v = match (dynamic, order) {
                    (None, order) => {
                        let window =
                            view.window(&corner, &shape)
                                .map_err(|error| LiftError::View {
                                    op: op.op_type,
                                    error,
                                })?;
                        let view = match order {
                            Some(o) => window.permute(&o).map_err(|error| LiftError::View {
                                op: op.op_type,
                                error,
                            })?,
                            None => window,
                        };
                        Val::Mem {
                            store,
                            view,
                            format,
                        }
                    }
                    (Some(_), Some(_)) => return Err(LiftError::DynamicIndex { op: op.op_type }),
                    (Some((a, index, 1, add)), None) if shape[a] > 1 => self.row_tiles(
                        op,
                        (store, &view, format),
                        (&corner, &shape),
                        (a, index, add),
                    )?,
                    (Some((a, index, scale, add)), None) => self.block_tile(
                        op,
                        (store, &view, format),
                        (&corner, &shape),
                        (a, index, scale, add),
                    )?,
                };
                self.bind(op, v);
            }
            K::KtdpConstructIndirectAccessTile => self.indirect_tile(op)?,
            K::TensorExtract => {
                // A scalar read out of an index tensor: the element, as a one-entry index.
                let Val::Tensor(o) = self.get(op.operands[0])? else {
                    return Err(malformed("tensor.extract reads a tensor"));
                };
                let mut at = Vec::with_capacity(op.operands.len() - 1);
                for &s in &op.operands[1..] {
                    at.push(self.int(op, s)? as u64);
                }
                let elem = o.view.shifted(&at);
                let index = Operand {
                    view: View {
                        offset: elem.offset,
                        axes: vec![crate::view::Axis {
                            extent: 1,
                            stride: 1,
                        }],
                        tile: None,
                    },
                    ..o
                };
                self.bind(
                    op,
                    Val::Dyn {
                        index,
                        scale: 1,
                        add: 0,
                    },
                );
            }
            K::KtdpLoad => {
                let v = match self.get(op.operands[0])? {
                    Val::Mem {
                        store,
                        view,
                        format,
                    } => {
                        self.note_param(store, format);
                        Val::Tensor(Operand {
                            store,
                            view,
                            format,
                            gather: None,
                        })
                    }
                    Val::Indirect {
                        base,
                        gather,
                        shape,
                    } => {
                        // The rows the index names, copied out.
                        let value = Operand {
                            gather: Some(gather),
                            ..base
                        };
                        let out = self.local(&value.view.shape(), value.format, false);
                        self.graph.nodes.push(Node {
                            func: OpFunc::Identity,
                            space: value.view.shape(),
                            inputs: vec![value],
                            output: out.clone(),
                        });
                        let view = out.view.reshape(&shape).map_err(|error| LiftError::View {
                            op: op.op_type,
                            error,
                        })?;
                        Val::Tensor(Operand { view, ..out })
                    }
                    Val::Rows { axis, rows, shape } => {
                        // Each row the offset names, copied into its row of one tile.
                        let format = rows
                            .first()
                            .map(|(b, _)| b.format)
                            .ok_or(malformed("a tile has rows"))?;
                        let out = self.local(&shape, format, false);
                        for (k, (base, gather)) in rows.into_iter().enumerate() {
                            let value = Operand {
                                gather: Some(gather),
                                ..base
                            };
                            let space = value.view.shape();
                            let row = Self::row_of(&out.view, axis, k as u64, &space).map_err(
                                |error| LiftError::View {
                                    op: op.op_type,
                                    error,
                                },
                            )?;
                            self.graph.nodes.push(Node {
                                func: OpFunc::Identity,
                                space,
                                inputs: vec![value],
                                output: Operand {
                                    view: row,
                                    ..out.clone()
                                },
                            });
                        }
                        Val::Tensor(out)
                    }
                    _ => return Err(malformed("ktdp.load reads an access tile")),
                };
                self.bind(op, v);
            }
            K::KtdpStore => self.store(op)?,
            K::TensorSplat => {
                let shape = Self::shape(op)?;
                let v = match self.get(op.operands[0])? {
                    Val::Float(value) => Val::Splat {
                        value,
                        format: Self::format(op)?,
                        shape,
                    },
                    Val::Int(i) => Val::Splat {
                        value: i as f64,
                        format: Self::format(op)?,
                        shape,
                    },
                    // A value read from memory repeats its one element over every axis.
                    Val::Tensor(o)
                    | Val::Dyn {
                        index: o,
                        scale: 1,
                        add: 0,
                    } if o.view.numel() == 1 => {
                        let axes = shape
                            .iter()
                            .map(|&extent| Axis { extent, stride: 0 })
                            .collect();
                        Val::Tensor(Operand {
                            view: View {
                                offset: o.view.offset,
                                axes,
                                tile: None,
                            },
                            ..o
                        })
                    }
                    _ => return Err(LiftError::DynamicIndex { op: op.op_type }),
                };
                self.bind(op, v);
            }
            K::TensorEmpty => {
                self.bind(op, Val::Empty);
            }
            K::LinalgFill => {
                let value = match self.get(op.operands[0])? {
                    Val::Float(f) => f,
                    Val::Int(i) => i as f64,
                    _ => return Err(LiftError::DynamicIndex { op: op.op_type }),
                };
                let v = Val::Splat {
                    value,
                    format: Self::format(op)?,
                    shape: Self::shape(op)?,
                };
                self.bind(op, v);
            }
            K::TensorExpandShape | K::TensorCollapseShape | K::TensorReshape => {
                let to = int_list(op, AttrKey::TargetShape)
                    .or_else(|| int_list(op, AttrKey::Shape))
                    .map(u64s)
                    .ok_or(malformed("a reshape states its result shape"))?;
                self.view_op(op, |v| v.reshape(&to))?;
            }
            K::LinalgBroadcast => {
                let shape = Self::shape(op)?;
                let added = int_list(op, AttrKey::Dimensions)
                    .ok_or(malformed("linalg.broadcast states its dimensions"))?
                    .iter()
                    .map(|&d| d as usize)
                    .collect::<Vec<_>>();
                self.view_op(op, |v| Ok(v.broadcast(&shape, &added)))?;
            }
            K::LinalgTranspose => {
                let perm = int_list(op, AttrKey::Permutation)
                    .ok_or(malformed("linalg.transpose states its permutation"))?
                    .iter()
                    .map(|&d| d as usize)
                    .collect::<Vec<_>>();
                self.view_op(op, |v| v.permute(&perm))?;
            }
            K::LinalgReduce => self.reduce(op)?,
            K::LinalgMatmul => self.matmul(op, false)?,
            K::LinalgBatchMatmul => self.matmul(op, true)?,
            K::ArithAddf => self.pointwise(op, OpFunc::Add)?,
            K::ArithSubf => self.pointwise(op, OpFunc::Sub)?,
            K::ArithMulf => self.pointwise(op, OpFunc::Mul)?,
            K::ArithDivf => self.pointwise(op, OpFunc::Div)?,
            K::ArithMaximumf | K::ArithMaxnumf | K::ArithMaxf => {
                self.pointwise(op, OpFunc::Maximum)?
            }
            K::ArithMinimumf | K::ArithMinnumf | K::ArithMinf => {
                self.pointwise(op, OpFunc::Minimum)?
            }
            K::ArithNegf => self.pointwise(op, OpFunc::Neg)?,
            K::ArithAbsf | K::MathAbsf => self.pointwise(op, OpFunc::Abs)?,
            K::MathExp => self.pointwise(op, OpFunc::Exp)?,
            K::MathLog => self.pointwise(op, OpFunc::Log)?,
            K::MathSqrt => self.pointwise(op, OpFunc::Sqrt)?,
            K::MathRsqrt => self.pointwise(op, OpFunc::Rsqrt)?,
            K::MathTanh => self.pointwise(op, OpFunc::Tanh)?,
            K::MathErf => self.pointwise(op, OpFunc::Erf)?,
            K::MathFloor => self.pointwise(op, OpFunc::Floor)?,
            K::ArithSelect => self.pointwise(op, OpFunc::Select)?,
            K::ArithCmpf => {
                let pred = match op.attr(AttrKey::Predicate) {
                    Some(Attr::Str(p)) => *p,
                    _ => return Err(malformed("arith.cmpf states its predicate")),
                };
                let c = match pred {
                    "ogt" | "ugt" => Compare::Gt,
                    "oge" | "uge" => Compare::Ge,
                    "olt" | "ult" => Compare::Lt,
                    "ole" | "ule" => Compare::Le,
                    "oeq" | "ueq" => Compare::Eq,
                    "one" | "une" => Compare::Ne,
                    _ => {
                        return Err(malformed(
                            "arith.cmpf's predicate is an ordered or unordered compare",
                        ));
                    }
                };
                self.pointwise(op, OpFunc::Cmp(c))?;
            }
            K::ArithExtf | K::ArithTruncf | K::ArithConvertf => {
                let to = Self::format(op)?;
                let v = self.get(op.operands[0])?;
                let from = match &v {
                    Val::Tensor(o) => Some(o.format),
                    Val::Splat { format, .. } => Some(*format),
                    _ => None,
                };
                match (from, v) {
                    (Some(f), v) if f == to => self.bind(op, v),
                    (_, Val::Splat { value, shape, .. }) => self.bind(
                        op,
                        Val::Splat {
                            value,
                            format: to,
                            shape,
                        },
                    ),
                    (Some(DataFormat::Fp16), _) if to == DataFormat::Fp32 => {
                        self.pointwise(op, OpFunc::Widen)?
                    }
                    (Some(DataFormat::Fp32), _) if to == DataFormat::Fp16 => {
                        self.pointwise(op, OpFunc::Narrow)?
                    }
                    (Some(DataFormat::Fp8), _) if to == DataFormat::Fp16 => {
                        self.pointwise(op, OpFunc::Fp8ToFp16)?
                    }
                    (Some(DataFormat::Fp16), _) if to == DataFormat::Fp8 => {
                        self.pointwise(op, OpFunc::Fp16ToFp8)?
                    }
                    _ => return Err(LiftError::Op(op.op_type)),
                }
            }
            K::ScfFor => {
                let lb = self.int(op, op.operands[0])?;
                let ub = self.int(op, op.operands[1])?;
                let step = self.int(op, op.operands[2])?.max(1);
                let iv = match op.attr(AttrKey::IterVar) {
                    Some(Attr::Ssas(v)) if !v.is_empty() => v[0],
                    _ => return Err(malformed("scf.for states its induction variable")),
                };
                let args: Vec<Ssa> = match op.attr(AttrKey::IterArgs) {
                    Some(Attr::Ssas(v)) => v.to_vec(),
                    _ => Vec::new(),
                };
                let mut carried: Vec<Val> = op.operands[3..]
                    .iter()
                    .map(|&s| self.get(s))
                    .collect::<Result<_, _>>()?;
                let body = op.regions.first().copied().unwrap_or(&[]);
                let mut i = lb;
                while i < ub {
                    self.env.insert(iv, Val::Int(i));
                    for (a, v) in args.iter().zip(&carried) {
                        self.env.insert(*a, v.clone());
                    }
                    if let Some(y) = self.run(body, tile)? {
                        carried = y;
                    }
                    i += step;
                }
                match carried.len() {
                    0 => {}
                    1 => {
                        let v = carried
                            .pop()
                            .ok_or(malformed("scf.for yields its carried value"))?;
                        self.bind(op, v);
                    }
                    _ => self.bind(op, Val::Tuple(carried)),
                }
            }
            K::ScfYield => {
                let vs = op
                    .operands
                    .iter()
                    .map(|&s| self.get(s))
                    .collect::<Result<Vec<_>, _>>()?;
                return Ok(Some(vs));
            }
            K::FuncReturn | K::RegionBb0Args => {}
            other => return Err(LiftError::Op(other)),
        }
        Ok(None)
    }
}

/// Store forwarding: an intermediate written by one op and only ever copied out whole into a
/// parameter buffer is written straight into that buffer.
fn forward_stores(g: &mut Graph) {
    let mut i = 0;
    while i < g.nodes.len() {
        let n = &g.nodes[i];
        let forwardable = n.func == OpFunc::Identity
            && matches!(n.output.store, Store::Param(_))
            && n.inputs.len() == 1
            && n.inputs[0].gather.is_none();
        if forwardable {
            let src = n.inputs[0].clone();
            if let Store::Local(l) = src.store {
                let writers: Vec<usize> = g
                    .nodes
                    .iter()
                    .enumerate()
                    .filter(|(_, m)| m.output.store == Store::Local(l))
                    .map(|(j, _)| j)
                    .collect();
                let readers = g
                    .nodes
                    .iter()
                    .filter(|m| {
                        m.inputs.iter().any(|o| {
                            o.store == Store::Local(l)
                                || o.gather
                                    .as_ref()
                                    .is_some_and(|gt| gt.index.store == Store::Local(l))
                        })
                    })
                    .count();
                if let [w] = writers[..]
                    && readers == 1
                    && w < i
                    && g.nodes[w].output.view.same_elements(&src.view)
                    && g.locals[l].elements == src.view.numel()
                    && g.nodes[w].output.format == g.nodes[i].output.format
                    && let Some(inputs) = over_space(&g.nodes[w].inputs, &g.nodes[i].space)
                {
                    let (space, out) = (g.nodes[i].space.clone(), g.nodes[i].output.clone());
                    let producer = &mut g.nodes[w];
                    producer.space = space;
                    producer.inputs = inputs;
                    producer.output = out;
                    g.nodes.remove(i);
                    continue;
                }
            }
        }
        i += 1;
    }
}
