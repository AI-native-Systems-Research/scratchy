// SPDX-License-Identifier: Apache-2.0
//! Lower a [`SubtileIR`] to **KTIR** (the `ktdp` MLIR dialect) as TEXT, for
//! the IBM Spyre path. Sibling of [`crate::lower_subtile_tape_to_tk_tape`].
//! Numeric validation runs out-of-band on the open-source `ktir-emulator`
//! emulator (an independent, foreign implementation of RFC 0682 — see
//! `examples/dump_ktir.rs`).
//!
//! ## Per-op kernels (matches Spyre's per-bundle execution)
//!
//! Each SubtileIR **node** becomes one `func.func`, with **all I/O through
//! HBM** — every input tensor and the output tensor is an HBM pointer arg.
//! A host/checker allocates one HBM buffer per tensor (sources + op-output
//! "intermediate" buffers) and runs the node funcs in topological order,
//! threading each output buffer into the next op's input. This is exactly
//! Spyre's host-orchestrated per-bundle model (NOT a megakernel): there are
//! no cross-op SSA intermediates, so the GEMM tile-loads its activation `A`
//! from HBM like any weight, and LX is bounded to one op's working set
//! (each func is its own region, freed on return).
//!
//! ## Status: SPIKE
//!
//! Single-core (`grid=[1,1]`), `m=1`. Arms: `Add`, `RmsNorm`, `MatmulTile`
//! (K-tiled to fit 2MB LX), `SiluMul`. Element offsets, f32 tiles.
//! `RopeRotate` / `AttnDecode` and SPMD 32-core distribution are follow-ons.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::fmt::Write as _;

use scratchy_subtile::subtile_ir::{
    EwKind, KvCacheProducer, RopeForm, SubOp, SubtileIR, SubtileNode, TensorId, TensorRegion,
    TensorShape,
};

/// Element type emitted for tile tensors. `f16` is the canonical KTIR tile
/// dtype: every reference fixture (`examples/triton-ktir/*.mlir`) is f16
/// end-to-end — activations, weight memrefs, and the matmul accumulator
/// (`iter_args(%acc) -> tensor<MxNxf16>`) alike — and the metal backend
/// lowers unannotated tiles to `half` (the NAX tensor engine consumes bf16
/// internally; the host/KTIR boundary is f16). `ktir-emulator` widens f16→f32 for
/// the actual arithmetic, so reductions/softmax still accumulate in f32 and
/// the functional agreement vs the f32 `eval_dag` oracle is ~1e-2 (not
/// bit-exact). Use f32 only for a bit-exact oracle diff while debugging.
const ELEM: &str = "f16";

/// One `func.func` arg — an HBM base pointer (`index`) + the metadata a
/// harness needs to bind a buffer to it by name.
#[derive(Clone, Debug)]
pub struct KtirArg {
    /// SSA arg name WITHOUT the leading `%` (the `ktir-emulator` kwarg name).
    pub name: String,
    /// The [`TensorId`] this pointer backs.
    pub tensor: u32,
    pub rows: u32,
    pub cols: u32,
    /// True iff this is the node's OUTPUT (where the func writes).
    pub is_output: bool,
}

/// A single-node KTIR kernel (one func wrapped in a module).
#[derive(Clone, Debug)]
pub struct KtirKernel {
    pub text: String,
    pub func_name: String,
    pub args: Vec<KtirArg>,
    /// `(mask_tensor_id, prefix_capacity)` iff this node emits the AttnDecode
    /// runtime length mask (a synthetic `[1, capacity]` HBM source).
    pub mask: Option<(u32, u32)>,
}

/// One node lowered to its OWN single-func KTIR module. ktir-emulator's regex
/// parser does not scope args correctly across multiple funcs in one module
/// (it returns the last func's signature for every func), so each node is a
/// standalone module the orchestrator loads + runs individually.
#[derive(Clone, Debug)]
pub struct NodeKtir {
    pub func_name: String,
    pub args: Vec<KtirArg>,
    pub module_text: String,
}

/// A whole SubtileIR lowered to per-node KTIR funcs in one module, plus the
/// orchestration manifest a host/checker needs to thread HBM buffers.
#[derive(Clone, Debug)]
pub struct GraphKtir {
    pub nodes: Vec<NodeKtir>,
    pub num_sources: u32,
    pub result_tensor: u32,
    /// `tensor_shapes[id] = (rows, cols)` for every tensor. When `attn_mask`
    /// is set, the LAST entry is the synthetic mask tensor `[1, capacity]`.
    pub tensor_shapes: Vec<(u32, u32)>,
    /// Tensor id of the shared AttnDecode runtime length-mask source, if the
    /// graph has a maskable decode. The host fills it `[1, capacity]` per
    /// forward step (0 on valid columns, -inf past `decode_position`); it is
    /// NOT one of `num_sources` and is written by no node.
    pub attn_mask: Option<u32>,
}

/// Per-func emit state. No `slot_ssa` / cross-op threading — each node func
/// is self-contained (inputs loaded from HBM, output stored to HBM).
struct KtirState<'g, F: RopeForm> {
    graph: &'g SubtileIR<F>,
    body: String,
    next_ssa: u32,
    /// `TensorId.0` → bare arg name (deduped, first-seen).
    arg_of_tensor: BTreeMap<usize, String>,
    /// Tensor ids in first-seen order (the func-arg order).
    arg_order: Vec<usize>,
    /// Set iff this node's `AttnDecode` emits the runtime length mask:
    /// `(mask_tensor_id, prefix_capacity)`. The mask tensor is a synthetic
    /// HBM source (id beyond `graph.tensors`) whose shape isn't in
    /// `graph.tensors`, so it's carried here for arg/shape resolution.
    mask: Option<(u32, u32)>,
    /// SPMD core grid `[GX, GY]` for this node's func (default `[1, 1]`). A node
    /// sets it >1 to split work across cores (GEMM: M rows; attention: heads) so
    /// the per-core LX working set stays small — the prefill-beyond-LX fix.
    grid: (u32, u32),
}

impl<'g, F: RopeForm> KtirState<'g, F> {
    fn new(graph: &'g SubtileIR<F>) -> Self {
        Self {
            graph,
            body: String::new(),
            next_ssa: 0,
            arg_of_tensor: BTreeMap::new(),
            arg_order: Vec::new(),
            mask: None,
            grid: (1, 1),
        }
    }

    fn fresh(&mut self, hint: &str) -> String {
        let n = self.next_ssa;
        self.next_ssa += 1;
        format!("%{hint}{n}")
    }

    /// Bare arg name for a tensor's HBM base pointer (deduped, first-seen).
    fn arg_for(&mut self, t: TensorId) -> String {
        if let Some(a) = self.arg_of_tensor.get(&t.index()) {
            return a.clone();
        }
        let a = format!("t{}_ptr", t.index());
        self.arg_of_tensor.insert(t.index(), a.clone());
        self.arg_order.push(t.index());
        a
    }

    /// `construct_memory_view` for the whole HBM tensor; returns the view SSA.
    /// Loop-invariant — emit once, before any `scf.for` that tiles into it.
    fn emit_view(&mut self, t: TensorId) -> String {
        let arg = self.arg_for(t);
        let shape = self.graph.shape(t);
        let (sr, sc) = (shape.rows, shape.cols);
        let view = self.fresh("view");
        writeln!(
            self.body,
            "    {view} = ktdp.construct_memory_view %{arg}, sizes: [{sr}, {sc}], strides: [{sc}, 1] {{"
        )
        .unwrap();
        writeln!(
            self.body,
            "      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + {} >= 0, d1 >= 0, -d1 + {} >= 0)>,",
            sr - 1,
            sc - 1
        )
        .unwrap();
        writeln!(
            self.body,
            "      memory_space = #ktdp.spyre_memory_space<HBM>"
        )
        .unwrap();
        writeln!(self.body, "    }} : memref<{sr}x{sc}x{ELEM}>").unwrap();
        view
    }

    /// `construct_access_tile` of shape `[tr, tc]` at base `[base_r, base_c]`
    /// into `view` (whose tensor has `shape`). Base indices are SSA strings
    /// (`"%c0"`, or a loop var like `"%k3"`), enabling tiled / offset access.
    fn emit_tile_access(
        &mut self,
        view: &str,
        shape: TensorShape,
        base_r: &str,
        base_c: &str,
        tr: u32,
        tc: u32,
    ) -> String {
        let (sr, sc) = (shape.rows, shape.cols);
        let acc = self.fresh("acc");
        writeln!(
            self.body,
            "    {acc} = ktdp.construct_access_tile {view}[{base_r}, {base_c}] {{"
        )
        .unwrap();
        writeln!(
            self.body,
            "      access_tile_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + {} >= 0, d1 >= 0, -d1 + {} >= 0)>,",
            tr - 1,
            tc - 1
        )
        .unwrap();
        writeln!(
            self.body,
            "      access_tile_order = affine_map<(d0, d1) -> (d0, d1)>"
        )
        .unwrap();
        writeln!(
            self.body,
            "    }} : memref<{sr}x{sc}x{ELEM}> -> !ktdp.access_tile<{tr}x{tc}xindex>"
        )
        .unwrap();
        acc
    }

    /// `ktdp.load` of a `[tr, tc]` access tile → `tensor<tr x tc x f32>` SSA.
    fn emit_load_tile(&mut self, acc: &str, tr: u32, tc: u32) -> String {
        let v = self.fresh("v");
        writeln!(
            self.body,
            "    {v} = ktdp.load {acc} : !ktdp.access_tile<{tr}x{tc}xindex> -> tensor<{tr}x{tc}x{ELEM}>"
        )
        .unwrap();
        v
    }

    /// Index SSA for a constant offset: `%c0` for 0 (the common case), else a
    /// fresh `arith.constant`.
    fn idx_const(&mut self, off: u32) -> String {
        if off == 0 {
            "%c0".to_string()
        } else {
            let f = self.fresh("off");
            writeln!(self.body, "    {f} = arith.constant {off} : index").unwrap();
            f
        }
    }

    /// Region load for add/rmsnorm/silumul. Honors the region offset — a fused
    /// operand can be a column/row slice of a larger tensor (e.g. the `up` half
    /// of a gate-up projection) — defaulting to the offset-0 whole-tensor path.
    fn emit_load(&mut self, tr: &TensorRegion) -> String {
        let shape = self.graph.shape(tr.tensor);
        let (r, c) = (tr.region.rows.len, tr.region.cols.len);
        let view = self.emit_view(tr.tensor);
        let rs = self.idx_const(tr.region.rows.start);
        let cs = self.idx_const(tr.region.cols.start);
        let acc = self.emit_tile_access(&view, shape, &rs, &cs, r, c);
        self.emit_load_tile(&acc, r, c)
    }

    /// Region store of `val` into `out`. Honors the region offset.
    fn emit_store(&mut self, val: &str, out: &TensorRegion) {
        let shape = self.graph.shape(out.tensor);
        let (r, c) = (out.region.rows.len, out.region.cols.len);
        let view = self.emit_view(out.tensor);
        let rs = self.idx_const(out.region.rows.start);
        let cs = self.idx_const(out.region.cols.start);
        let acc = self.emit_tile_access(&view, shape, &rs, &cs, r, c);
        writeln!(
            self.body,
            "    ktdp.store {val}, {acc} : tensor<{r}x{c}x{ELEM}>, !ktdp.access_tile<{r}x{c}xindex>"
        )
        .unwrap();
    }

    /// `construct_memory_view` with EXPLICIT 2D sizes/strides — reinterpret a
    /// buffer as a different shape (rope: view x`[1, heads*hd]` as `[heads, hd]`).
    fn emit_view_shaped(&mut self, t: TensorId, sr: u32, sc: u32) -> String {
        let arg = self.arg_for(t);
        let view = self.fresh("view");
        writeln!(
            self.body,
            "    {view} = ktdp.construct_memory_view %{arg}, sizes: [{sr}, {sc}], strides: [{sc}, 1] {{"
        )
        .unwrap();
        writeln!(
            self.body,
            "      coordinate_set = affine_set<(d0, d1) : (d0 >= 0, -d0 + {} >= 0, d1 >= 0, -d1 + {} >= 0)>,",
            sr - 1,
            sc - 1
        )
        .unwrap();
        writeln!(
            self.body,
            "      memory_space = #ktdp.spyre_memory_space<HBM>"
        )
        .unwrap();
        writeln!(self.body, "    }} : memref<{sr}x{sc}x{ELEM}>").unwrap();
        view
    }

    /// 1D load: view tensor `t` (`full_len` contiguous elements) as a 1D memref
    /// and load `prefix_len` elements starting at `off` → `tensor<prefix_len x
    /// f16>`. Used for rope cos/sin (the `half` of token-row ri's `[.., hd]`
    /// table at off=ri*hd) and rmsnorm gamma (off=0).
    fn emit_load_1d(&mut self, t: TensorId, full_len: u32, off: u32, prefix_len: u32) -> String {
        let arg = self.arg_for(t);
        let view = self.fresh("view");
        writeln!(
            self.body,
            "    {view} = ktdp.construct_memory_view %{arg}, sizes: [{full_len}], strides: [1] {{"
        )
        .unwrap();
        writeln!(
            self.body,
            "      coordinate_set = affine_set<(d0) : (d0 >= 0, -d0 + {} >= 0)>, memory_space = #ktdp.spyre_memory_space<HBM>",
            full_len - 1
        )
        .unwrap();
        writeln!(self.body, "    }} : memref<{full_len}x{ELEM}>").unwrap();
        let offc = self.idx_const(off);
        let acc = self.fresh("acc");
        writeln!(
            self.body,
            "    {acc} = ktdp.construct_access_tile {view}[{offc}] {{"
        )
        .unwrap();
        writeln!(
            self.body,
            "      access_tile_set = affine_set<(d0) : (d0 >= 0, -d0 + {} >= 0)>,",
            prefix_len - 1
        )
        .unwrap();
        writeln!(
            self.body,
            "      access_tile_order = affine_map<(d0) -> (d0)>"
        )
        .unwrap();
        writeln!(
            self.body,
            "    }} : memref<{full_len}x{ELEM}> -> !ktdp.access_tile<{prefix_len}xindex>"
        )
        .unwrap();
        let v = self.fresh("v");
        writeln!(
            self.body,
            "    {v} = ktdp.load {acc} : !ktdp.access_tile<{prefix_len}xindex> -> tensor<{prefix_len}x{ELEM}>"
        )
        .unwrap();
        v
    }

    /// Tiled store of `val` (shape `[tr, tc]`) into `view` at `[base_r, base_c]`.
    #[allow(clippy::too_many_arguments)]
    fn emit_store_tile(
        &mut self,
        view: &str,
        shape: TensorShape,
        base_r: &str,
        base_c: &str,
        tr: u32,
        tc: u32,
        val: &str,
    ) {
        let acc = self.emit_tile_access(view, shape, base_r, base_c, tr, tc);
        writeln!(
            self.body,
            "    ktdp.store {val}, {acc} : tensor<{tr}x{tc}x{ELEM}>, !ktdp.access_tile<{tr}x{tc}xindex>"
        )
        .unwrap();
    }

    /// NeoX rotary of `x_t[1, heads*hd]` with `cos_t`/`sin_t` first-half tables,
    /// writing the rotated result to `out_t`. Shared by RopeRotate and
    /// RopeAppend (whose host eval is rotation-only). m=1.
    /// RoPE over `rows` token positions (rows=1 decode, rows=m prefill). x is
    /// `[rows, heads*hd]`, viewed as `[rows*heads, hd]` so each (token,head) is a
    /// row. cos/sin are `[rows, hd]` (one position per token row); each is
    /// broadcast across heads to `[rows*heads, half]`. At rows=1 this is exactly
    /// the old single-token path (heads rows, one cos/sin), so decode is
    /// unchanged; at rows>1 every token row rotates by its OWN position's table.
    #[allow(clippy::too_many_arguments)]
    fn emit_rope(
        &mut self,
        x_t: TensorId,
        cos_t: TensorId,
        sin_t: TensorId,
        out_t: TensorId,
        cols: u32,
        hd: u32,
        rows: u32,
        // Row stride of the cos/sin table tensors. They are now pre-tiled to the
        // rope's full width (`heads * hd`) by the host, so position `ri`'s row
        // starts at `ri * tbl_cols` (NOT `ri * hd`); the first `half` of each row
        // holds the rotary values (every head's slice is identical).
        tbl_cols: u32,
    ) {
        let heads = cols / hd;
        let half = hd / 2;
        let mh = rows * heads;
        let mhshape = TensorShape { rows: mh, cols: hd };
        let half_c = self.fresh("half");
        writeln!(self.body, "    {half_c} = arith.constant {half} : index").unwrap();
        let x_view = self.emit_view_shaped(x_t, mh, hd);
        let out_view = self.emit_view_shaped(out_t, mh, hd);
        let ty = format!("tensor<{heads}x{half}x{ELEM}>");
        // Per token row: rotate its `heads` head-rows (rows [ri*heads ..
        // (ri+1)*heads] of the [mh, hd] view) by position ri's cos/sin (the
        // `half` at offset ri*hd of the [rows, hd] table), broadcast across
        // heads (dim 0). Per-row so each token uses its OWN position table — and
        // it avoids tensor.collapse_shape, which ktir_emulator does not rank-reduce.
        // rows=1 ⇒ one iteration = the gate-proven single-token path.
        for ri in 0..rows {
            let rbase = self.idx_const(ri * heads);
            let xf_acc = self.emit_tile_access(&x_view, mhshape, &rbase, "%c0", heads, half);
            let xf = self.emit_load_tile(&xf_acc, heads, half);
            let xs_acc = self.emit_tile_access(&x_view, mhshape, &rbase, &half_c, heads, half);
            let xs = self.emit_load_tile(&xs_acc, heads, half);
            let mk = |st: &mut Self, t: TensorId, tag: &str| -> String {
                let h1 = st.emit_load_1d(t, rows * tbl_cols, ri * tbl_cols, half);
                let bi = st.fresh("cbi");
                writeln!(
                    st.body,
                    "    {bi} = tensor.empty() : tensor<{heads}x{half}x{ELEM}>"
                )
                .unwrap();
                let b = st.fresh(tag);
                writeln!(st.body, "    {b} = linalg.broadcast ins({h1} : tensor<{half}x{ELEM}>) outs({bi} : tensor<{heads}x{half}x{ELEM}>) dimensions = [0]").unwrap();
                b
            };
            let cosb = mk(self, cos_t, "cosb");
            let sinb = mk(self, sin_t, "sinb");
            let a1 = self.fresh("a1");
            writeln!(self.body, "    {a1} = arith.mulf {xf}, {cosb} : {ty}").unwrap();
            let a2 = self.fresh("a2");
            writeln!(self.body, "    {a2} = arith.mulf {xs}, {sinb} : {ty}").unwrap();
            let of = self.fresh("of");
            writeln!(self.body, "    {of} = arith.subf {a1}, {a2} : {ty}").unwrap();
            let b1 = self.fresh("b1");
            writeln!(self.body, "    {b1} = arith.mulf {xf}, {sinb} : {ty}").unwrap();
            let b2 = self.fresh("b2");
            writeln!(self.body, "    {b2} = arith.mulf {xs}, {cosb} : {ty}").unwrap();
            let os = self.fresh("os");
            writeln!(self.body, "    {os} = arith.addf {b1}, {b2} : {ty}").unwrap();
            self.emit_store_tile(&out_view, mhshape, &rbase, "%c0", heads, half, &of);
            self.emit_store_tile(&out_view, mhshape, &rbase, &half_c, heads, half, &os);
        }
    }
}

/// Emit the func body for one node: load inputs from HBM, compute, store the
/// output to HBM.
fn lower_node<F: RopeForm>(st: &mut KtirState<F>, node: &SubtileNode<F>) {
    let op = node.op;
    let out = node.output;
    let inputs = node.inputs.clone();

    match op {
        SubOp::Elementwise(EwKind::Add) => {
            let a = st.emit_load(&inputs[0]);
            let b = st.emit_load(&inputs[1]);
            let (r, c) = (out.region.rows.len, out.region.cols.len);
            let d = st.fresh("add");
            writeln!(
                st.body,
                "    {d} = arith.addf {a}, {b} : tensor<{r}x{c}x{ELEM}>"
            )
            .unwrap();
            st.emit_store(&d, &out);
        }

        SubOp::ScalarMul { scale } => {
            // out[i,j] = x[i,j] * scale (compile-time const — Granite's activation
            // multipliers: embedding / residual / recip(logits_scaling)). PURE ELEM
            // (f16): a splat const + one `arith.mulf`, like the `Elementwise(Add/Mul)`
            // arms and the attention-scale constant. NOTE: only the power-of-two scales
            // (0.0078125=2^-7, 0.0625=2^-4) are exact in f16; 12.0 is exact (1.5·2^3), but
            // residual_multiplier 0.22 (=11/50) is NON-dyadic ⇒ exact in NO binary float
            // (IEEE-f16 → 0.2200927, SEN169 → 901/4096=0.219971; Kani-proven in the sdsc
            // crate's `granite2b_mup_scales_sen169_exactness`). On the SEN169 superdsc path
            // this per-layer residual scale is a proven ~1.3e-4 precision gap vs the fp32
            // golden. `{scale:?}` guarantees a decimal literal so the MLIR float parse never
            // sees a bare integer.
            let x = st.emit_load(&inputs[0]);
            let (r, c) = (out.region.rows.len, out.region.cols.len);
            let sc = st.fresh("smc");
            writeln!(st.body, "    {sc} = arith.constant {scale:?} : {ELEM}").unwrap();
            let scs = st.fresh("smcs");
            writeln!(
                st.body,
                "    {scs} = tensor.splat {sc} : tensor<{r}x{c}x{ELEM}>"
            )
            .unwrap();
            let d = st.fresh("sm");
            writeln!(
                st.body,
                "    {d} = arith.mulf {x}, {scs} : tensor<{r}x{c}x{ELEM}>"
            )
            .unwrap();
            st.emit_store(&d, &out);
        }

        SubOp::RmsNorm {
            eps,
            // ⛔ SCALE ONLY. This body multiplies by the STORED gain; the gemma-class
            // (1 + w) convention is a different kernel, and running this one over a
            // zero-centred gain scales every normalized activation by roughly nothing.
            // Binding the variant in the PATTERN (rather than checking inside) puts the
            // refusal in the enumerated not-lowered arm below, where every other op this
            // emitter declines already lives.
            gain: scratchy_subtile::subtile_ir::GainConvention::Scale,
        } => {
            // out[0,j] = x[0,j] * (1/sqrt(mean_j(x[0,:]^2) + eps)) * gamma[j].
            let x = st.emit_load(&inputs[0]);
            let (r, c) = (out.region.rows.len, out.region.cols.len);
            // Variance (x², its sum/mean, the rsqrt) is computed in f32, NOT the
            // tile dtype: a real residual stream grows past |x|≈256, where x²
            // overflows f16 (361k > 65504 → inf → mean=inf → scale=0 → the layer
            // dies). HBM/tiles stay f16 (the residual itself fits); only this
            // reduction widens. The final rescale narrows back (the scale ≈ 1/rms
            // is small — safe in f16). The synthetic-source gate never exercises
            // this (its values are tiny), so this is real-weights-only.
            let xf = st.fresh("xf");
            writeln!(
                st.body,
                "    {xf} = arith.extf {x} : tensor<{r}x{c}x{ELEM}> to tensor<{r}x{c}xf32>"
            )
            .unwrap();
            let x2 = st.fresh("x2");
            writeln!(
                st.body,
                "    {x2} = arith.mulf {xf}, {xf} : tensor<{r}x{c}xf32>"
            )
            .unwrap();
            let zero = st.fresh("zero");
            writeln!(st.body, "    {zero} = arith.constant 0.0 : f32").unwrap();
            let sinit = st.fresh("sinit");
            writeln!(
                st.body,
                "    {sinit} = tensor.splat {zero} : tensor<{r}xf32>"
            )
            .unwrap();
            let ssum = st.fresh("ssum");
            writeln!(st.body, "    {ssum} = linalg.reduce {{ arith.addf }}").unwrap();
            writeln!(st.body, "      ins({x2} : tensor<{r}x{c}xf32>)").unwrap();
            writeln!(st.body, "      outs({sinit} : tensor<{r}xf32>)").unwrap();
            writeln!(st.body, "      dimensions = [1]").unwrap();
            let dt = st.fresh("dt");
            writeln!(st.body, "    {dt} = arith.constant {c}.0 : f32").unwrap();
            let dts = st.fresh("dts");
            writeln!(st.body, "    {dts} = tensor.splat {dt} : tensor<{r}xf32>").unwrap();
            let mean = st.fresh("mean");
            writeln!(
                st.body,
                "    {mean} = arith.divf {ssum}, {dts} : tensor<{r}xf32>"
            )
            .unwrap();
            let epsc = st.fresh("epsc");
            writeln!(st.body, "    {epsc} = arith.constant {eps} : f32").unwrap();
            let epst = st.fresh("epst");
            writeln!(
                st.body,
                "    {epst} = tensor.splat {epsc} : tensor<{r}xf32>"
            )
            .unwrap();
            let meps = st.fresh("meps");
            writeln!(
                st.body,
                "    {meps} = arith.addf {mean}, {epst} : tensor<{r}xf32>"
            )
            .unwrap();
            let rms = st.fresh("rms");
            writeln!(st.body, "    {rms} = math.sqrt {meps} : tensor<{r}xf32>").unwrap();
            let one = st.fresh("one");
            writeln!(st.body, "    {one} = arith.constant 1.0 : f32").unwrap();
            let onet = st.fresh("onet");
            writeln!(st.body, "    {onet} = tensor.splat {one} : tensor<{r}xf32>").unwrap();
            let inv = st.fresh("inv");
            writeln!(
                st.body,
                "    {inv} = arith.divf {onet}, {rms} : tensor<{r}xf32>"
            )
            .unwrap();
            // Per-row scale: each of the r rows has its own inv_rms. Narrow the
            // [r] f32 inv to the tile dtype, then broadcast across the c columns
            // (dim 1). At r=1 this is one row ⇒ identical to the old scalar
            // splat (decode unchanged), but it now scales each prefill row by
            // its OWN rms instead of row 0's.
            let inv_e = st.fresh("invE");
            writeln!(
                st.body,
                "    {inv_e} = arith.truncf {inv} : tensor<{r}xf32> to tensor<{r}x{ELEM}>"
            )
            .unwrap();
            let invinit = st.fresh("invinit");
            writeln!(
                st.body,
                "    {invinit} = tensor.empty() : tensor<{r}x{c}x{ELEM}>"
            )
            .unwrap();
            let invb = st.fresh("invb");
            writeln!(
                st.body,
                "    {invb} = linalg.broadcast ins({inv_e} : tensor<{r}x{ELEM}>) outs({invinit} : tensor<{r}x{c}x{ELEM}>) dimensions = [1]"
            )
            .unwrap();
            let xs = st.fresh("xs");
            writeln!(
                st.body,
                "    {xs} = arith.mulf {x}, {invb} : tensor<{r}x{c}x{ELEM}>"
            )
            .unwrap();
            // gamma is one row [1,c]; load it rank-1 ([c]) and broadcast across
            // the r rows so each row is scaled by the same per-column weight.
            // (rank-1 load avoids tensor.collapse_shape, which ktir_emulator does not
            // reduce — it left [1,c] and the broadcast produced a [1,1,c].) At
            // r=1 the broadcast is a no-op copy (decode numerically unchanged).
            let gcol = st.emit_load_1d(inputs[1].tensor, c, 0, c);
            let ginit = st.fresh("ginit");
            writeln!(
                st.body,
                "    {ginit} = tensor.empty() : tensor<{r}x{c}x{ELEM}>"
            )
            .unwrap();
            let gb = st.fresh("gb");
            writeln!(
                st.body,
                "    {gb} = linalg.broadcast ins({gcol} : tensor<{c}x{ELEM}>) outs({ginit} : tensor<{r}x{c}x{ELEM}>) dimensions = [0]"
            )
            .unwrap();
            let yv = st.fresh("y");
            writeln!(
                st.body,
                "    {yv} = arith.mulf {xs}, {gb} : tensor<{r}x{c}x{ELEM}>"
            )
            .unwrap();
            st.emit_store(&yv, &out);
        }

        SubOp::SiluMul => {
            // out[j] = (gate / (1 + exp(-gate))) * up.
            let gate = st.emit_load(&inputs[0]);
            let up = st.emit_load(&inputs[1]);
            let (r, c) = (out.region.rows.len, out.region.cols.len);
            let neg = st.fresh("neg");
            writeln!(
                st.body,
                "    {neg} = arith.negf {gate} : tensor<{r}x{c}x{ELEM}>"
            )
            .unwrap();
            let e = st.fresh("e");
            writeln!(st.body, "    {e} = math.exp {neg} : tensor<{r}x{c}x{ELEM}>").unwrap();
            let one_s = st.fresh("one");
            writeln!(st.body, "    {one_s} = arith.constant 1.0 : {ELEM}").unwrap();
            let one = st.fresh("onet");
            writeln!(
                st.body,
                "    {one} = tensor.splat {one_s} : tensor<{r}x{c}x{ELEM}>"
            )
            .unwrap();
            let denom = st.fresh("denom");
            writeln!(
                st.body,
                "    {denom} = arith.addf {one}, {e} : tensor<{r}x{c}x{ELEM}>"
            )
            .unwrap();
            let silu = st.fresh("silu");
            writeln!(
                st.body,
                "    {silu} = arith.divf {gate}, {denom} : tensor<{r}x{c}x{ELEM}>"
            )
            .unwrap();
            let yv = st.fresh("y");
            writeln!(
                st.body,
                "    {yv} = arith.mulf {silu}, {up} : tensor<{r}x{c}x{ELEM}>"
            )
            .unwrap();
            st.emit_store(&yv, &out);
        }

        SubOp::MatmulTile { .. } => {
            // out[m,n] = A[m,k] @ W[k,n], W row-major [K,N] (matches eval_node).
            // K is tiled into [kb, bw] blocks (scf.for + iter_args accumulator)
            // that fit the 2 MB LX. N is ALSO tiled into `bw`-wide column blocks
            // when the whole-N K-loop's n-wide tiles (acc + partial + add-result
            // + W) wouldn't fit — i.e. a large-vocab lm_head. Each column block
            // K-loops independently and stores to its output column range. Both
            // A and W are HBM tensors.
            let a_reg = inputs[0];
            let w_reg = inputs[1];
            let m = out.region.rows.len;
            let n = out.region.cols.len;
            let kdim = a_reg.region.cols.len;
            let a_t = a_reg.tensor;
            let w_t = w_reg.tensor;
            let a_shape = st.graph.shape(a_t);
            let a_view = st.emit_view(a_t);
            // Weight binds VERBATIM as its on-disk [out, in] = [n, k] buffer; the
            // matmul reads it with a transpose-B `indexing_maps` (B's map ends in
            // the reduction dim d2, so the contraction reduces over k in place),
            // so no transpose and no strided gather — native transpose-B at NAX
            // speed. B's view is the natural [n, k] (strides [k, 1]).
            let w_nk = TensorShape {
                rows: n,
                cols: kdim,
            };
            let w_view = st.emit_view_shaped(w_t, n, kdim);
            let out_view = st.emit_view(out.tensor);
            let out_shape = st.graph.shape(out.tensor);
            // M (token rows) split ACROSS GRID CORES: for m>1 (prefill) the func
            // runs grid=[m,1] and each core computes ONE output row at
            // `pid = ktdp.get_compute_tile_id`, reading A row `pid` + the FULL
            // (shared) W, writing output row `pid`. Per-core the tiles are [1,·]
            // so the working set is the m=1 size and fits each core's own 2MB LX
            // — the prefill-beyond-LX fix; the shared-weight grid is also exactly
            // ktir-emulator's cross-core GEMM combine. m=1 keeps grid=[1,1], pid=0 ⇒
            // byte-identical to the validated decode.
            let row_idx = if m > 1 {
                let pid = st.fresh("pid");
                writeln!(st.body, "    {pid} = ktdp.get_compute_tile_id : index").unwrap();
                st.grid = (m, 1);
                pid
            } else {
                "%c0".to_string()
            };
            let nblk = pick_n_block(n, m);
            let mut n_off = 0u32;
            while n_off < n {
                let bw = nblk.min(n - n_off); // this column block's width
                let kb = pick_k_block(kdim, bw);
                assert_eq!(
                    kdim % kb,
                    0,
                    "KTIR spike: K ({kdim}) must be divisible by the K-block ({kb})"
                );
                let noff = if n_off == 0 {
                    "%c0".to_string()
                } else {
                    let c = st.fresh("noff");
                    writeln!(st.body, "    {c} = arith.constant {n_off} : index").unwrap();
                    c
                };
                let kc = st.fresh("K");
                writeln!(st.body, "    {kc} = arith.constant {kdim} : index").unwrap();
                let kbc = st.fresh("KB");
                writeln!(st.body, "    {kbc} = arith.constant {kb} : index").unwrap();
                let azero = st.fresh("azero");
                writeln!(
                    st.body,
                    "    {azero} = arith.constant dense<0.0> : tensor<1x{bw}x{ELEM}>"
                )
                .unwrap();
                let acc = st.fresh("mm");
                let kv = st.fresh("k");
                let accit = st.fresh("accit");
                writeln!(
                    st.body,
                    "    {acc} = scf.for {kv} = %c0 to {kc} step {kbc} iter_args({accit} = {azero}) -> (tensor<1x{bw}x{ELEM}>) {{"
                )
                .unwrap();
                let a_acc = st.emit_tile_access(&a_view, a_shape, &row_idx, &kv, 1, kb);
                let a_val = st.emit_load_tile(&a_acc, 1, kb);
                // B tile = contiguous row-block [n_off..+bw, kv..+kb] of [n, k].
                let w_acc = st.emit_tile_access(&w_view, w_nk, &noff, &kv, bw, kb);
                let w_val = st.emit_load_tile(&w_acc, bw, kb);
                let cinit = st.fresh("cinit");
                writeln!(
                    st.body,
                    "    {cinit} = arith.constant dense<0.0> : tensor<1x{bw}x{ELEM}>"
                )
                .unwrap();
                let part = st.fresh("part");
                writeln!(
                    st.body,
                    "    {part} = linalg.matmul indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d1, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>] ins({a_val}, {w_val} : tensor<1x{kb}x{ELEM}>, tensor<{bw}x{kb}x{ELEM}>) outs({cinit} : tensor<1x{bw}x{ELEM}>) -> tensor<1x{bw}x{ELEM}>"
                )
                .unwrap();
                let accnext = st.fresh("accnext");
                writeln!(
                    st.body,
                    "    {accnext} = arith.addf {accit}, {part} : tensor<1x{bw}x{ELEM}>"
                )
                .unwrap();
                writeln!(st.body, "    scf.yield {accnext} : tensor<1x{bw}x{ELEM}>").unwrap();
                writeln!(st.body, "    }}").unwrap();
                st.emit_store_tile(&out_view, out_shape, &row_idx, &noff, 1, bw, &acc);
                n_off += bw;
            }
        }

        SubOp::RopeRotate { head_dim, .. } => {
            let cols = inputs[0].region.cols.len;
            let tbl_cols = inputs[1].region.cols.len;
            st.emit_rope(
                inputs[0].tensor,
                inputs[1].tensor,
                inputs[2].tensor,
                out.tensor,
                cols,
                head_dim.get(),
                out.region.rows.len,
                tbl_cols,
            );
        }
        SubOp::RopeAppend { head_dim, .. } => {
            // Host eval (eval_node) is rotation-only: rotate K (input[0]) with
            // cos/sin. The V + paged KV-cache write are GPU/Spyre-side and flow
            // through graph edges, so for the emulator this == RopeRotate.
            let cols = inputs[0].region.cols.len;
            let tbl_cols = inputs[1].region.cols.len;
            st.emit_rope(
                inputs[0].tensor,
                inputs[1].tensor,
                inputs[2].tensor,
                out.tensor,
                cols,
                head_dim.get(),
                out.region.rows.len,
                tbl_cols,
            );
        }
        SubOp::AttnDecode {
            geom,
            scale,
            producer,
            ..
        } => {
            // Single-segment decode attention, m=1. inputs [Q, K, V].
            // Per q-head h (unrolled): scores = scale*(q_h · K_kvhᵀ) [1,S],
            // softmax over S, out_h = softmax · V_kvh [1,hd]. GQA: kv-head =
            // h/gqa (constant col offset). Multi-segment (prefix+new) concat
            // is a follow-on.
            assert!(
                inputs.len() >= 3 && inputs.len() % 2 == 1,
                "AttnDecode: Q + (K,V) segment pairs"
            );
            let nq = geom.nqh().get();
            let hd = geom.hd().get();
            let gqa = geom.gqa().get();
            let nseg = (inputs.len() - 1) / 2;
            // Runtime length mask: for the production decode shape (a prefix
            // cache written THIS forward by a RopeAppend, segment 0, + the new
            // token as a later segment) the prefix cache TENSOR spans the full
            // structural capacity, but only `decode_position` of its rows are
            // valid at run time. `lower_region` SLICES the host-eval prefix to
            // `valid_len - 1`, baking the length into the shape; here we keep
            // the FULL capacity and add a runtime additive mask instead — one
            // length-independent bundle (cuda's `right_fill` + `DecodePosition`
            // equivalent). The mask is a shared `[1, capacity]` HBM tile (0 on
            // valid columns, -inf past `decode_position`) the host fills per
            // forward step; masked columns drop out of the softmax. Only the
            // prefix segment is masked — the new-token segment is always valid.
            let mask_prefix =
                nseg >= 2 && matches!(producer, KvCacheProducer::SameForwardRopeAppend { .. });
            // mq query rows (m=1 decode, m>1 prefill). Q/out are [mq, nq*hd];
            // per (query row qi, head h) we load a [1,hd] tile at row qi, col
            // h*hd and reuse the validated m=1 per-row MLIR once per query row.
            // mq is an emit-time constant so this unrolls — ktir_emulator has no perf
            // constraint, and it stays inside the m=1 supported op subset.
            let mq = out.region.rows.len as usize;
            let qview_shape = TensorShape {
                rows: mq as u32,
                cols: nq * hd,
            };
            let q_view = st.emit_view_shaped(inputs[0].tensor, mq as u32, nq * hd);
            let out_view = st.emit_view_shaped(out.tensor, mq as u32, nq * hd);
            // Per segment: (k_t, v_t, kshape, vshape, row_start, seq_len, col_start).
            let mut segs: Vec<(TensorId, TensorId, TensorShape, TensorShape, u32, u32, u32)> =
                Vec::with_capacity(nseg);
            for i in 0..nseg {
                let kr = inputs[1 + 2 * i];
                let vr = inputs[2 + 2 * i];
                let kshape = st.graph.shape(kr.tensor);
                // The masked prefix segment reads its cache to FULL capacity
                // (rows 0..cap); the host mask bounds it to the valid prefix.
                let (row_start, seq_len) = if mask_prefix && i == 0 {
                    (0, kshape.rows)
                } else {
                    (kr.region.rows.start, kr.region.rows.len)
                };
                segs.push((
                    kr.tensor,
                    vr.tensor,
                    kshape,
                    st.graph.shape(vr.tensor),
                    row_start,
                    seq_len,
                    kr.region.cols.start,
                ));
            }
            // Shared mask tile `[1, capacity]`, loaded once: a synthetic HBM
            // source at id == graph.tensors.len() (beyond the IR tensors), the
            // same id for every AttnDecode (all layers share one prefix
            // capacity ⇒ one mask per forward step).
            let mask_tile = if mask_prefix {
                let cap = segs[0].5;
                let mask_id = st.graph.tensors.len() as u32;
                st.mask = Some((mask_id, cap));
                let mview = st.emit_view_shaped(TensorId::from_index(mask_id as usize), 1, cap);
                let macc = st.emit_tile_access(
                    &mview,
                    TensorShape { rows: 1, cols: cap },
                    "%c0",
                    "%c0",
                    1,
                    cap,
                );
                Some(st.emit_load_tile(&macc, 1, cap))
            } else {
                None
            };
            let mut kviews = Vec::with_capacity(nseg);
            let mut vviews = Vec::with_capacity(nseg);
            for seg in segs.iter().take(nseg) {
                kviews.push(st.emit_view(seg.0));
            }
            for seg in segs.iter().take(nseg) {
                vviews.push(st.emit_view(seg.1));
            }
            let scale_c = st.fresh("scale");
            writeln!(st.body, "    {scale_c} = arith.constant {scale} : {ELEM}").unwrap();
            let ninf = st.fresh("ninf");
            writeln!(st.body, "    {ninf} = arith.constant -1.0e38 : {ELEM}").unwrap();
            let zc = st.fresh("zc");
            writeln!(st.body, "    {zc} = arith.constant 0.0 : {ELEM}").unwrap();

            // Heads split ACROSS GRID CORES for prefill (mq>1): grid=[nq,1],
            // each core computes ONE head (all mq query rows), so the per-core
            // LX working set is one head's tiles — 1/nq of the m*nq the single-
            // core unroll kept live (which overflowed). The query-row (qi) loop
            // stays emit-time so the causal SLICE (slen=qi+1) still works (no
            // runtime mask). m=1 decode keeps grid=[1,1] + all heads on one core
            // ⇒ byte-identical.
            let head_pid = if mq > 1 {
                let p = st.fresh("hpid");
                writeln!(st.body, "    {p} = ktdp.get_compute_tile_id : index").unwrap();
                st.grid = (nq, 1);
                Some(p)
            } else {
                None
            };
            let hd_c = st.fresh("hdc");
            writeln!(st.body, "    {hd_c} = arith.constant {hd} : index").unwrap();
            let gqa_c = st.fresh("gqac");
            writeln!(st.body, "    {gqa_c} = arith.constant {gqa} : index").unwrap();
            for qi in 0..mq {
                let cqi = st.fresh("qrow");
                writeln!(st.body, "    {cqi} = arith.constant {qi} : index").unwrap();
                let nh = if head_pid.is_some() { 1 } else { nq };
                for h in 0..nh {
                    // Q/out column (ch) + kv-column base (kvcol) for this head:
                    // runtime (the core's head pid) for prefill, emit-time const for
                    // decode. kvcol is now an SSA (was an int) — folded with the
                    // segment col start via arith.addi below.
                    let (ch, kvcol) = if let Some(ref pid) = head_pid {
                        let ch = st.fresh("qcol");
                        writeln!(st.body, "    {ch} = arith.muli {pid}, {hd_c} : index").unwrap();
                        let kvh = st.fresh("kvh");
                        writeln!(st.body, "    {kvh} = arith.divui {pid}, {gqa_c} : index")
                            .unwrap();
                        let kvcol = st.fresh("kvcol");
                        writeln!(st.body, "    {kvcol} = arith.muli {kvh}, {hd_c} : index")
                            .unwrap();
                        (ch, kvcol)
                    } else {
                        let ch = st.fresh("qcol");
                        writeln!(st.body, "    {ch} = arith.constant {} : index", h * hd).unwrap();
                        let kvcol = st.fresh("kvcol");
                        writeln!(
                            st.body,
                            "    {kvcol} = arith.constant {} : index",
                            (h / gqa) * hd
                        )
                        .unwrap();
                        (ch, kvcol)
                    };
                    let qh_acc = st.emit_tile_access(&q_view, qview_shape, &cqi, &ch, 1, hd);
                    let qh = st.emit_load_tile(&qh_acc, 1, hd);

                    // Pass 1: per-seg scores [1,slen] + running global max scalar.
                    let mut scores: Vec<(String, u32)> = Vec::with_capacity(nseg);
                    let mut gmax: Option<String> = None;
                    for i in 0..nseg {
                        let (_kt, _vt, kshape, _vs, rstart, slen_full, cstart) = segs[i];
                        // Causal slicing: the new-token segment holds the mq query
                        // rows' own K/V; query row qi attends only rows 0..=qi (its
                        // own + earlier new tokens), so load qi+1 rows instead of all
                        // mq. Pure causality via tile slicing — no per-element mask
                        // const (ktir_emulator can't execute dense<[...]> lists; only
                        // splat). The prefix segment stays full (the runtime prefix
                        // mask bounds it). At mq=1 ⇒ 1 row ⇒ decode unchanged.
                        let is_new = !(mask_prefix && i == 0);
                        let slen = if is_new && slen_full == mq as u32 {
                            (qi as u32) + 1
                        } else {
                            slen_full
                        };
                        let crow = st.fresh("kr");
                        writeln!(st.body, "    {crow} = arith.constant {rstart} : index").unwrap();
                        let kcs = st.fresh("kcs");
                        writeln!(st.body, "    {kcs} = arith.constant {cstart} : index").unwrap();
                        let ccol = st.fresh("kc");
                        writeln!(st.body, "    {ccol} = arith.addi {kcs}, {kvcol} : index")
                            .unwrap();
                        let kacc = st.emit_tile_access(&kviews[i], kshape, &crow, &ccol, slen, hd);
                        let kk = st.emit_load_tile(&kacc, slen, hd);
                        let kti = st.fresh("kti");
                        writeln!(
                            st.body,
                            "    {kti} = tensor.empty() : tensor<{hd}x{slen}x{ELEM}>"
                        )
                        .unwrap();
                        let kt = st.fresh("kt");
                        writeln!(st.body, "    {kt} = linalg.transpose ins({kk} : tensor<{slen}x{hd}x{ELEM}>) outs({kti} : tensor<{hd}x{slen}x{ELEM}>) permutation = [1, 0]").unwrap();
                        let sci = st.fresh("sci");
                        writeln!(
                            st.body,
                            "    {sci} = arith.constant dense<0.0> : tensor<1x{slen}x{ELEM}>"
                        )
                        .unwrap();
                        let scr = st.fresh("scr");
                        writeln!(st.body, "    {scr} = linalg.matmul ins({qh}, {kt} : tensor<1x{hd}x{ELEM}>, tensor<{hd}x{slen}x{ELEM}>) outs({sci} : tensor<1x{slen}x{ELEM}>) -> tensor<1x{slen}x{ELEM}>").unwrap();
                        let sclt = st.fresh("sclt");
                        writeln!(
                            st.body,
                            "    {sclt} = tensor.splat {scale_c} : tensor<1x{slen}x{ELEM}>"
                        )
                        .unwrap();
                        let sc = st.fresh("sc");
                        writeln!(
                            st.body,
                            "    {sc} = arith.mulf {scr}, {sclt} : tensor<1x{slen}x{ELEM}>"
                        )
                        .unwrap();
                        // Add the runtime length mask to the prefix segment's
                        // scores: -inf on columns past `decode_position` ⇒ those
                        // cache positions vanish from the softmax (exp(-inf)=0).
                        let sc = if mask_prefix && i == 0 {
                            let m = mask_tile
                                .as_ref()
                                .expect("mask tile present when mask_prefix");
                            let scm = st.fresh("scm");
                            writeln!(
                                st.body,
                                "    {scm} = arith.addf {sc}, {m} : tensor<1x{slen}x{ELEM}>"
                            )
                            .unwrap();
                            scm
                        } else {
                            sc
                        };
                        let mi = st.fresh("mi");
                        writeln!(st.body, "    {mi} = tensor.splat {ninf} : tensor<1x{ELEM}>")
                            .unwrap();
                        let mx = st.fresh("mx");
                        writeln!(st.body, "    {mx} = linalg.reduce {{ arith.maximumf }}").unwrap();
                        writeln!(st.body, "      ins({sc} : tensor<1x{slen}x{ELEM}>)").unwrap();
                        writeln!(st.body, "      outs({mi} : tensor<1x{ELEM}>)").unwrap();
                        writeln!(st.body, "      dimensions = [1]").unwrap();
                        let mxs = st.fresh("mxs");
                        writeln!(
                            st.body,
                            "    {mxs} = tensor.extract {mx}[%c0] : tensor<1x{ELEM}>"
                        )
                        .unwrap();
                        scores.push((sc, slen));
                        gmax = Some(match gmax {
                            None => mxs,
                            Some(g) => {
                                let nm = st.fresh("gm");
                                writeln!(st.body, "    {nm} = arith.maximumf {g}, {mxs} : {ELEM}")
                                    .unwrap();
                                nm
                            }
                        });
                    }
                    let gmax = gmax.unwrap();

                    // Pass 2: exp(scores - gmax) + running global sum scalar.
                    let mut es: Vec<(String, u32)> = Vec::with_capacity(nseg);
                    let mut gsum: Option<String> = None;
                    for (sc, slen) in scores {
                        let gmb = st.fresh("gmb");
                        writeln!(
                            st.body,
                            "    {gmb} = tensor.splat {gmax} : tensor<1x{slen}x{ELEM}>"
                        )
                        .unwrap();
                        let sh = st.fresh("sh");
                        writeln!(
                            st.body,
                            "    {sh} = arith.subf {sc}, {gmb} : tensor<1x{slen}x{ELEM}>"
                        )
                        .unwrap();
                        let ex = st.fresh("ex");
                        writeln!(
                            st.body,
                            "    {ex} = math.exp {sh} : tensor<1x{slen}x{ELEM}>"
                        )
                        .unwrap();
                        let zit = st.fresh("zit");
                        writeln!(st.body, "    {zit} = tensor.splat {zc} : tensor<1x{ELEM}>")
                            .unwrap();
                        let su = st.fresh("su");
                        writeln!(st.body, "    {su} = linalg.reduce {{ arith.addf }}").unwrap();
                        writeln!(st.body, "      ins({ex} : tensor<1x{slen}x{ELEM}>)").unwrap();
                        writeln!(st.body, "      outs({zit} : tensor<1x{ELEM}>)").unwrap();
                        writeln!(st.body, "      dimensions = [1]").unwrap();
                        let sus = st.fresh("sus");
                        writeln!(
                            st.body,
                            "    {sus} = tensor.extract {su}[%c0] : tensor<1x{ELEM}>"
                        )
                        .unwrap();
                        es.push((ex, slen));
                        gsum = Some(match gsum {
                            None => sus,
                            Some(g) => {
                                let ng = st.fresh("gs");
                                writeln!(st.body, "    {ng} = arith.addf {g}, {sus} : {ELEM}")
                                    .unwrap();
                                ng
                            }
                        });
                    }
                    let gsum = gsum.unwrap();

                    // Pass 3: weighted V, accumulated across segments → out_h [1,hd].
                    let mut out_acc: Option<String> = None;
                    for (i, (ex, slen)) in es.into_iter().enumerate() {
                        let (_kt, _vt, _ks, vshape, rstart, _sl, cstart) = segs[i];
                        let gsb = st.fresh("gsb");
                        writeln!(
                            st.body,
                            "    {gsb} = tensor.splat {gsum} : tensor<1x{slen}x{ELEM}>"
                        )
                        .unwrap();
                        let w = st.fresh("w");
                        writeln!(
                            st.body,
                            "    {w} = arith.divf {ex}, {gsb} : tensor<1x{slen}x{ELEM}>"
                        )
                        .unwrap();
                        let vrw = st.fresh("vr");
                        writeln!(st.body, "    {vrw} = arith.constant {rstart} : index").unwrap();
                        let vcs = st.fresh("vcs");
                        writeln!(st.body, "    {vcs} = arith.constant {cstart} : index").unwrap();
                        let vcl = st.fresh("vc");
                        writeln!(st.body, "    {vcl} = arith.addi {vcs}, {kvcol} : index").unwrap();
                        let vacc = st.emit_tile_access(&vviews[i], vshape, &vrw, &vcl, slen, hd);
                        let vv = st.emit_load_tile(&vacc, slen, hd);
                        let oi = st.fresh("oi");
                        writeln!(
                            st.body,
                            "    {oi} = arith.constant dense<0.0> : tensor<1x{hd}x{ELEM}>"
                        )
                        .unwrap();
                        let ov = st.fresh("ov");
                        writeln!(st.body, "    {ov} = linalg.matmul ins({w}, {vv} : tensor<1x{slen}x{ELEM}>, tensor<{slen}x{hd}x{ELEM}>) outs({oi} : tensor<1x{hd}x{ELEM}>) -> tensor<1x{hd}x{ELEM}>").unwrap();
                        out_acc = Some(match out_acc {
                            None => ov,
                            Some(o) => {
                                let na = st.fresh("oa");
                                writeln!(
                                    st.body,
                                    "    {na} = arith.addf {o}, {ov} : tensor<1x{hd}x{ELEM}>"
                                )
                                .unwrap();
                                na
                            }
                        });
                    }
                    let oh = out_acc.unwrap();
                    st.emit_store_tile(&out_view, qview_shape, &cqi, &ch, 1, hd, &oh);
                }
            }
        }
        // ── SubOps this emitter does not lower ─────────────────────
        //
        // Enumerated, never a `_` catch-all: adding a
        // `SubOp` and forgetting the KTIR arm must be E0004 at
        // compile time, not a panic the first time a model happens to
        // emit it. The KTIR path is the decode/prefill emitter for
        // the arches spyre serves; these three are carried by the
        // SuperDSC emitter's own decomposition instead.
        SubOp::SumReduce | SubOp::RmsNormReduce { .. } | SubOp::RmsNormApply { .. } => {
            unimplemented!("KTIR: {op:?} is lowered by the SuperDSC emitter, not here")
        }
        // The rest of the arch vocabulary. The shared front end EXPRESSES every op now
        // — vision loads, MoE, gated-delta-net, the (1 + w) rmsnorm — so they reach
        // every target's emitter, and the emitter is where "do I have a kernel for
        // this?" is answered. Enumerated for the same reason as the arm above: a new
        // SubOp must be E0004 here, not a surprise at emission time.
        SubOp::RmsNorm { .. }
        // quick-gelu and exact-erf gelu are DISTINCT functions from `Gelu`, and `Sub`
        // takes a [m, 1] broadcast operand — none of the three is the nearest arm.
        | SubOp::Elementwise(EwKind::QuickGelu | EwKind::GeluErf | EwKind::Sub)
        | SubOp::TanhSoftCap
        | SubOp::RmsNormUnit { .. }
        | SubOp::ScalarWeightMul
        | SubOp::GateSplit { .. }
        | SubOp::GateApply
        | SubOp::GateScale
        | SubOp::LoadPixels { .. }
        | SubOp::LoadPosEmbeds { .. }
        | SubOp::EmbeddingGather { .. }
        | SubOp::VisionRope
        | SubOp::VarlenAttention { .. }
        | SubOp::EncoderAttn { .. }
        | SubOp::GatedDeltaNet
        | SubOp::GemmaMoe { .. }
        | SubOp::Moe { .. }
        | SubOp::Mean => {
            unimplemented!("KTIR: no kernel for {op:?}")
        }
        // Silu and Mul never reach a node on their own: the shared
        // `lower` fuses them into `SiluMul` before either backend
        // sees the tape: the fusion that matters to both happens
        // once, in the shared lower.
        SubOp::Elementwise(EwKind::Silu | EwKind::Mul) => {
            unimplemented!("KTIR: {op:?} should have been fused into SiluMul by `lower`")
        }
        // ⛔ A DISTINCT CASE FROM THE ABOVE, AND SAYING SO MATTERS. Silu/Mul
        // land there because `lower` was supposed to fuse them; gelu is fused
        // into nothing, so borrowing that message would send a reader looking
        // for a fusion bug that does not exist. The dxp DDL has a real `gelu`
        // primitive and the SuperDSC path emits it; this emulator emits `arith`
        // and has no gelu, so a gelu arch runs on the SuperDSC path only.
        // The emulator emits `arith` over flat buffers and has no stick
        // layout at all, so a reshape here is only meaningful once the
        // SuperDSC path defines the restickify it lowers to.
        SubOp::Reshape => {
            unimplemented!(
                "KTIR emulator has no reshape — the SuperDSC path must define the \
                 restickify first"
            )
        }
        SubOp::Elementwise(EwKind::Gelu) => {
            unimplemented!(
                "KTIR emulator has no gelu primitive (the SuperDSC path emits the DDL's \
                 OpFunc::Gelu) — a gelu arch must run under --features superdsc"
            )
        }
    }
}

/// Emit one node's `func.func` block (NOT module-wrapped) + its arg manifest +
/// the optional synthetic mask `(tensor_id, capacity)` this node introduced.
fn emit_node_func<F: RopeForm>(
    graph: &SubtileIR<F>,
    node: &SubtileNode<F>,
    fname: &str,
) -> (String, Vec<KtirArg>, Option<(u32, u32)>) {
    let mut st = KtirState::new(graph);
    lower_node(&mut st, node);
    let out_tensor = node.output.tensor.index();
    let mask = st.mask;
    let args: Vec<KtirArg> = st
        .arg_order
        .iter()
        .map(|&t| {
            // The synthetic mask tensor's shape is `[1, capacity]` and isn't in
            // `graph.tensors`; resolve it from `st.mask` instead of `shape()`.
            let (rows, cols) = match mask {
                Some((mid, cap)) if mid as usize == t => (1, cap),
                _ => {
                    let s = graph.shape(TensorId::from_index(t));
                    (s.rows, s.cols)
                }
            };
            KtirArg {
                name: st.arg_of_tensor[&t].clone(),
                tensor: t as u32,
                rows,
                cols,
                is_output: t == out_tensor,
            }
        })
        .collect();
    let arglist = args
        .iter()
        .map(|a| format!("%{}: index", a.name))
        .collect::<Vec<_>>()
        .join(", ");
    let (gx, gy) = st.grid;
    let func = format!(
        "  func.func @{fname}({arglist}) attributes {{grid = [{gx}, {gy}]}} {{\n    %c0 = arith.constant 0 : index\n{}    return\n  }}\n",
        st.body
    );
    (func, args, mask)
}

/// Lower a single node to a one-func KTIR module (single-op convenience).
pub fn lower_node_to_ktir<F: RopeForm>(
    graph: &SubtileIR<F>,
    node: &SubtileNode<F>,
    fname: &str,
) -> KtirKernel {
    let (func, args, mask) = emit_node_func(graph, node, fname);
    KtirKernel {
        text: format!("module {{\n{func}}}\n"),
        func_name: fname.to_string(),
        args,
        mask,
    }
}

/// Lower a whole SubtileIR to per-node KTIR funcs in one module + the
/// orchestration manifest. Node `i` → func `{base}_n{i}`.
pub fn lower_graph_to_ktir<F: RopeForm>(graph: &SubtileIR<F>, base: &str) -> GraphKtir {
    let mut nodes = Vec::with_capacity(graph.nodes.len());
    let mut mask: Option<(u32, u32)> = None;
    for (i, node) in graph.nodes.iter().enumerate() {
        let fname = format!("{base}_n{i}");
        let k = lower_node_to_ktir(graph, node, &fname);
        if let Some(m) = k.mask {
            debug_assert!(
                mask.is_none() || mask == Some(m),
                "AttnDecode mask must be uniform across layers (one shared prefix capacity)"
            );
            mask = Some(m);
        }
        nodes.push(NodeKtir {
            func_name: k.func_name,
            args: k.args,
            module_text: k.text,
        });
    }
    let mut tensor_shapes: Vec<(u32, u32)> =
        graph.tensors.iter().map(|t| (t.rows, t.cols)).collect();
    let attn_mask = mask.map(|(mid, cap)| {
        // The mask tensor sits at id == graph.tensors.len(); append its
        // `[1, capacity]` shape so host buffer allocation covers it.
        debug_assert_eq!(
            mid as usize,
            tensor_shapes.len(),
            "mask id must be the next free tensor id"
        );
        tensor_shapes.push((1, cap));
        mid
    });
    GraphKtir {
        nodes,
        num_sources: graph.num_sources,
        result_tensor: graph.result.index() as u32,
        tensor_shapes,
        attn_mask,
    }
}

/// Largest power-of-two K-block dividing `k` such that the tiles co-resident in
/// LX during the K-loop all fit under the 2 MB per-core scratchpad: the weight
/// tile `[kb, n]`, the accumulator `[m=1, n]`, and the partial product
/// `[m=1, n]` — i.e. `(kb + 2)` rows of width `n`. Sizing only the W tile (the
/// old behaviour) overflowed on a large vocab: at n=128256 the accumulator +
/// partial alone are ~1 MB, so kb must drop to 1.
fn pick_k_block(k: u32, n: u32) -> u32 {
    // Cap the W tile (`[kb, n]`) small. The K-loop also holds ~4 `[m, n]` tiles,
    // and across sequential N-blocks each block leaves its accumulator +
    // zero-init in scope — ktir-emulator reclaims LX only at scope exit, a residue of
    // ~2·N_total floats. A small W tile keeps W + that residue + the working
    // tiles under the 2 MB LX cap for vocab up to ~170K (the Llama family).
    const W_TILE_CAP_BYTES: u32 = 400_000;
    let max_kb = (W_TILE_CAP_BYTES / (n * 4)).max(1);
    let mut kb = 1;
    let mut cand = 2;
    while cand <= max_kb && k.is_multiple_of(cand) {
        kb = cand;
        cand *= 2;
    }
    kb
}

/// Output column-block width, bounded by BOTH the vocab `n` and the row count
/// `m`. ktir-emulator's full-M GEMM offload holds the `[m, bw]` accumulator (plus its
/// K-loop sibling tiles + the cross-block residue) resident in the 2 MB LX, so
/// the constraint is on `m · bw`, NOT `n` alone — a small-vocab lm_head at
/// prefill (e.g. n=49152, m=32 ⇒ `[32, 49152]` = 3 MB) overflows just as a
/// large-vocab one at decode does. Whole-N is emitted only when the `[m, n]`
/// working set fits (the validated m=1 budget was `n ≤ 90_000`); otherwise tile
/// into 16_384-wide blocks — the cap validated for Llama (`[m, 16384]` f16 = 1 MB
/// at m=32) — shrunk further if a `[m, 16384]` block alone would overflow LX.
fn pick_n_block(n: u32, m: u32) -> u32 {
    const WHOLE_N_FITS_M1: u32 = 90_000; // whole-N working-set budget at m=1
    const BLOCK_N: u32 = 16_384; // Llama-validated column-block width
    // f16 element budget for one block's resident `[m, bw]` tiles, with headroom
    // for the ~4 sibling tiles + the W tile: keep `m · bw ≤ 512 K` (= 1 MB f16).
    const BLOCK_MN_BUDGET: u32 = 512 * 1024;
    let m = m.max(1);
    if m.saturating_mul(n) <= WHOLE_N_FITS_M1 {
        n
    } else {
        BLOCK_N.min(BLOCK_MN_BUDGET / m).max(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_subtile::fixtures::add_only_input;
    use scratchy_subtile::subtile_ir::{ValidatedGraph, lower_region};
    use std::num::NonZeroU32;

    /// Cheap, no-external-dep smoke test: per-node lowering of the (1-node)
    /// add fixture is structurally well-formed. Numeric correctness is the
    /// out-of-band ktir-emulator gate (see examples/dump_ktir.rs).
    #[test]
    fn add_emits_wellformed_graph_ktir() {
        let nb = NonZeroU32::new(2048).unwrap();
        let g = lower_region(&add_only_input(), nb);
        ValidatedGraph::new(&g).expect("fixture validates");
        let gk = lower_graph_to_ktir(&g, "add");
        assert_eq!(gk.nodes.len(), 1, "add fixture is one node");
        let m = &gk.nodes[0].module_text;
        assert!(m.contains("func.func @add_n0"), "{m}");
        assert!(m.contains("ktdp.construct_memory_view"));
        assert!(m.contains("ktdp.load"));
        assert!(m.contains("arith.addf"));
        assert!(m.contains("ktdp.store"));
        // 2 source ptrs + 1 output ptr; exactly one marked output.
        let n0 = &gk.nodes[0];
        assert_eq!(n0.args.len(), 3, "args: {:?}", n0.args);
        assert_eq!(n0.args.iter().filter(|a| a.is_output).count(), 1);
    }
}
