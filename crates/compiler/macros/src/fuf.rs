// SPDX-License-Identifier: Apache-2.0
//! Phase 6: unroll the per-model CFG into a flat numeric tile graph
//! — the Fully Unrolled Forward (FUF).
//!
//! The FUF has no loops, no identifiers, no strings. Every tile is:
//!
//! ```text
//! FufNode {
//!     id: TileId,               // dense u32 index
//!     op: OpKind,               // small enum, one per DSL op
//!     inputs: Vec<FufInput>,    // per-arg edges
//!     outputs: Vec<Shape>,      // one shape per output slot
//! }
//! ```
//!
//! and `FufInput` is one of:
//!   - `Tile { id, slot }` — the `slot`-th output of another tile
//!     (slot 0 for single-output ops, 0/1/2 for rope_append);
//!   - `Weight { id, index }` — a reference to an interned weight,
//!     optionally indexed with a concrete unrolled integer;
//!   - `Extern { kind, index }` — a reference to a non-weight
//!     parameter (`input_ids`, `kv_cache`, …), optionally indexed.
//!
//! Nothing below here has any idea what a "transformer" is.

use std::collections::HashMap;

use crate::cfg::{BlockId, BoolPredResolved, Cfg, Instr, Terminator};
use crate::classified::{Expr, ExternKind, LocalId, OpKind, UnrollIndex, WeightId};
use crate::quantization::StorageFormat;
use crate::shape::{Inferred, Shape};

// ── Types ─────────────────────────────────────────────────────────

pub use scratchy_subtile::handoff::TileId;

/// An input edge for a tile.
#[derive(Clone, Debug)]
pub enum FufInput {
    /// Output `slot` of tile `id`.
    Tile { id: TileId, slot: u8 },
    /// Reference to a weight, optionally indexed by a concrete
    /// unrolled integer (the former loop variable).
    ///
    /// `storage` is the bits-on-disk format resolved at macro time
    /// from the model's `quantization_config` (see
    /// [`crate::quantization::storage_format_for_weight`] for the
    /// full resolver rules). `fuf::unroll` always emits `Dense`; the
    /// per-model [`Fuf::annotate_storage_formats`] pass overwrites
    /// it for quantized models before the solver runs.
    Weight {
        id: WeightId,
        index: Option<UnrollIndex>,
        storage: StorageFormat,
    },
    /// Reference to a non-weight extern, optionally indexed.
    Extern {
        kind: ExternKind,
        index: Option<UnrollIndex>,
    },
    /// A compile-time scalar constant. Produced by DSL operators
    /// like `w + 1.0` — the `1.0` rides the FUF as a `Scalar`
    /// input. Rank-0 for shape-inference purposes; Impls emit it
    /// as a literal in the generated kernel call.
    Scalar(f64),
}

/// A tile.
#[derive(Clone, Debug)]
pub struct FufNode {
    pub id: TileId,
    pub op: OpKind,
    pub inputs: Vec<FufInput>,
    /// One `Shape` per output. Length 1 for most ops; length 3 for
    /// `rope_append` (q, k, v).
    pub outputs: Vec<Shape>,
}

/// The FUF for one model specialization.
#[derive(Clone, Debug)]
pub struct Fuf {
    pub nodes: Vec<FufNode>,
}

impl Fuf {
    pub fn get(&self, id: TileId) -> &FufNode {
        &self.nodes[id.0 as usize]
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Populate the `storage` field on every `FufInput::Weight` by
    /// running [`crate::quantization::storage_format_for_weight`]
    /// over the model's config. `fuf::unroll` emits every Weight
    /// input as [`StorageFormat::Dense`]; this pass overwrites them
    /// for quantized models before the solver runs.
    ///
    /// Called once per model in the macro drive between unroll and
    /// solve. The resolver's rules (modules_to_not_convert,
    /// tie_word_embeddings, Gemm-only) are the single source of
    /// truth — codegen's `FieldLoad` planner consults the same
    /// function, so matcher gating stays in lockstep with loader
    /// emission.
    pub fn annotate_storage_formats(
        &mut self,
        program: &crate::classified::Program,
        model: &crate::config::ModelParams,
    ) {
        // Resolver reads immutable `self`; precompute per (WeightId,
        // index) pair, then write the results back. MLX-native mixed /
        // dynamic quant (OptiQ) gives different numbered layers different
        // bit-widths, so the same WeightId's storage varies with the
        // unrolled `index` — key the cache on `(WeightId, index)`.
        use std::collections::HashMap;
        let mut cache: HashMap<(WeightId, Option<UnrollIndex>), StorageFormat> = HashMap::new();
        for node in &self.nodes {
            for input in &node.inputs {
                if let FufInput::Weight { id, index, .. } = input {
                    cache.entry((*id, *index)).or_insert_with(|| {
                        crate::quantization::storage_format_for_weight(
                            program, self, *id, *index, model,
                        )
                    });
                }
            }
        }
        for node in &mut self.nodes {
            for input in &mut node.inputs {
                if let FufInput::Weight { id, index, storage } = input
                    && let Some(fmt) = cache.get(&(*id, *index))
                {
                    *storage = fmt.clone();
                }
            }
        }
    }
}

/// Error produced during unrolling. All shape-related issues have
/// already been caught in Phase 4; this error set is specific to
/// structural problems in the CFG.
#[derive(Debug)]
pub enum UnrollError {
    MissingLocal {
        id: LocalId,
    },
    /// Phase 5 should have rejected nested loops; we assert.
    UnsupportedCfgShape(String),
    /// A local has no shape in the Inferred map.
    MissingShape {
        id: LocalId,
    },
}

impl std::fmt::Display for UnrollError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingLocal { id } => {
                write!(f, "read of local {:?} with no prior definition", id.0)
            }
            Self::UnsupportedCfgShape(s) => write!(f, "unsupported CFG shape: {s}"),
            Self::MissingShape { id } => write!(f, "no inferred shape for local {:?}", id.0),
        }
    }
}

impl std::error::Error for UnrollError {}

// ── Unroller ──────────────────────────────────────────────────────

pub fn unroll(cfg: &Cfg, inferred: &Inferred) -> Result<Fuf, UnrollError> {
    let mut u = Unroller {
        cfg,
        inferred,
        nodes: Vec::new(),
        local_to_tile: HashMap::new(),
        loop_vars: HashMap::new(),
    };
    u.walk(cfg.entry)?;
    Ok(Fuf { nodes: u.nodes })
}

struct Unroller<'a> {
    cfg: &'a Cfg,
    inferred: &'a Inferred,
    nodes: Vec<FufNode>,
    /// The current `TileId` + slot for every local that's been
    /// assigned so far. Overwritten on each assignment (straight-
    /// line SSA within the unrolled stream).
    local_to_tile: HashMap<LocalId, (TileId, u8)>,
    /// Current concrete value of each for-loop induction variable.
    /// Populated at loop entry, popped at exit.
    loop_vars: HashMap<LocalId, u64>,
}

impl<'a> Unroller<'a> {
    fn walk(&mut self, start: BlockId) -> Result<(), UnrollError> {
        let mut cur = start;
        loop {
            let block = self.cfg.block(cur);
            for instr in &block.instrs {
                self.lower_instr(instr)?;
            }
            match &block.term {
                Terminator::Jump(next) => cur = *next,
                Terminator::JumpWithCarry { target, carry } => {
                    self.apply_carry(carry);
                    cur = *target;
                }
                Terminator::CondJump {
                    cond,
                    then_b,
                    else_b,
                } => {
                    cur = if self.eval_pred(cond)? {
                        *then_b
                    } else {
                        *else_b
                    };
                }
                Terminator::LoopHeader {
                    ivar,
                    start,
                    end,
                    body,
                    exit,
                    loop_carry,
                } => {
                    // Unroll: for each iteration value, inject the
                    // concrete ivar and walk the body. After each
                    // iteration's body, propagate loop-carried
                    // bindings so the next iteration reads the
                    // previous iteration's outputs.
                    for i in *start..*end {
                        self.loop_vars.insert(*ivar, i);
                        self.walk_body_once(*body, cur)?;
                        for (outer, inner) in loop_carry {
                            if let Some(&inner_binding) = self.local_to_tile.get(inner) {
                                self.local_to_tile.insert(*outer, inner_binding);
                            }
                        }
                    }
                    self.loop_vars.remove(ivar);
                    cur = *exit;
                }
                Terminator::Return => return Ok(()),
            }
        }
    }

    /// Walk the loop body once: process instrs, follow jumps,
    /// stop when we see a jump back to the header.
    fn walk_body_once(&mut self, entry: BlockId, header: BlockId) -> Result<(), UnrollError> {
        let mut cur = entry;
        loop {
            let block = self.cfg.block(cur);
            for instr in &block.instrs {
                self.lower_instr(instr)?;
            }
            match &block.term {
                Terminator::Jump(next) => {
                    if *next == header {
                        return Ok(());
                    }
                    cur = *next;
                }
                Terminator::JumpWithCarry { target, carry } => {
                    self.apply_carry(carry);
                    if *target == header {
                        return Ok(());
                    }
                    cur = *target;
                }
                Terminator::CondJump {
                    cond,
                    then_b,
                    else_b,
                } => {
                    cur = if self.eval_pred(cond)? {
                        *then_b
                    } else {
                        *else_b
                    };
                }
                Terminator::LoopHeader { .. } => {
                    return Err(UnrollError::UnsupportedCfgShape(
                        "nested loops not yet supported".into(),
                    ));
                }
                Terminator::Return => return Ok(()),
            }
        }
    }

    fn apply_carry(&mut self, carry: &[(LocalId, LocalId)]) {
        for (merge_id, source_id) in carry {
            if let Some(&binding) = self.local_to_tile.get(source_id) {
                self.local_to_tile.insert(*merge_id, binding);
            }
        }
    }

    fn eval_pred(&self, cond: &BoolPredResolved) -> Result<bool, UnrollError> {
        match cond {
            BoolPredResolved::Modulo {
                ivar,
                divisor,
                remainder,
            } => {
                let v = self.loop_var_value(*ivar)?;
                if *divisor == 0 {
                    return Err(UnrollError::UnsupportedCfgShape(
                        "`if ivar % 0 == ...` is undefined".into(),
                    ));
                }
                Ok(v % *divisor == *remainder)
            }
            BoolPredResolved::NotModulo {
                ivar,
                divisor,
                remainder,
            } => {
                let v = self.loop_var_value(*ivar)?;
                if *divisor == 0 {
                    return Err(UnrollError::UnsupportedCfgShape(
                        "`if ivar % 0 != ...` is undefined".into(),
                    ));
                }
                Ok(v % *divisor != *remainder)
            }
            BoolPredResolved::Less { ivar, bound } => {
                let v = self.loop_var_value(*ivar)?;
                Ok(v < *bound)
            }
            BoolPredResolved::In { ivar, members } => {
                let v = self.loop_var_value(*ivar)?;
                Ok(members.contains(&v))
            }
        }
    }

    fn lower_instr(&mut self, instr: &Instr) -> Result<(), UnrollError> {
        match instr {
            Instr::Assign { target, value } => {
                let shape = self
                    .inferred
                    .locals
                    .get(target)
                    .cloned()
                    .ok_or(UnrollError::MissingShape { id: *target })?;
                let (op, inputs) = match value {
                    Expr::Call { op, args } => {
                        let inputs = args
                            .iter()
                            .map(|a| self.resolve_arg(a).map(|(inp, _)| inp))
                            .collect::<Result<Vec<_>, _>>()?;
                        (*op, inputs)
                    }
                    // Top-level `Mul` — e.g. `x = tile * scalar` for
                    // Gemma's embed scale. Emits a `Mul` tile that
                    // the `ScalarMulImpl` (or a future tensor×tensor
                    // fusion) claims.
                    Expr::Mul { lhs, rhs } => {
                        let (l, _) = self.resolve_arg(lhs)?;
                        let (r, _) = self.resolve_arg(rhs)?;
                        (OpKind::Mul, vec![l, r])
                    }
                    other => {
                        return Err(UnrollError::UnsupportedCfgShape(format!(
                            "expected a Call or Mul on RHS, got {other:?}"
                        )));
                    }
                };
                let tile_id = self.push_tile(op, inputs, vec![shape]);
                self.local_to_tile.insert(*target, (tile_id, 0));
                Ok(())
            }
            Instr::AssignTuple { targets, value } => {
                let inputs = self.resolve_args_from_expr(value)?;
                let op = match value {
                    Expr::Call { op, .. } => *op,
                    other => {
                        return Err(UnrollError::UnsupportedCfgShape(format!(
                            "tuple-RHS must be a Call, got {other:?}"
                        )));
                    }
                };
                let outputs: Vec<Shape> = targets
                    .iter()
                    .map(|t| {
                        self.inferred
                            .locals
                            .get(t)
                            .cloned()
                            .ok_or(UnrollError::MissingShape { id: *t })
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                let tile_id = self.push_tile(op, inputs, outputs);
                for (slot, t) in targets.iter().enumerate() {
                    self.local_to_tile.insert(*t, (tile_id, slot as u8));
                }
                Ok(())
            }
        }
    }

    /// Extract the input edges from a Call's arg list. Nested calls
    /// (e.g. `silu(gemm(..))`) and `Mul` expressions are inlined as
    /// helper tiles (one per sub-op). The returned FufInputs point
    /// at those helper tiles.
    fn resolve_args_from_expr(&mut self, expr: &Expr) -> Result<Vec<FufInput>, UnrollError> {
        match expr {
            Expr::Call { args, .. } => args
                .iter()
                .map(|a| self.resolve_arg(a).map(|(inp, _shape)| inp))
                .collect::<Result<Vec<_>, _>>(),
            _ => unreachable!("RHS must be a Call; caller checks"),
        }
    }

    /// Turn a classified Expr that appears as an op arg into a
    /// `FufInput` plus the resolved shape of that input.
    ///
    /// The shape is load-bearing: nested calls push helper tiles
    /// whose output shape must be correct so downstream cost
    /// functions can evaluate them. A Mul tile whose operands came
    /// from nested Calls, for example, feeds a downstream GEMM —
    /// if the Mul's output shape is empty, the GEMM's cost returns
    /// None and the solver errors out.
    fn resolve_arg(&mut self, expr: &Expr) -> Result<(FufInput, Shape), UnrollError> {
        match expr {
            Expr::Local(id) => {
                let &(tile, slot) = self
                    .local_to_tile
                    .get(id)
                    .ok_or(UnrollError::MissingLocal { id: *id })?;
                let shape = self.nodes[tile.0 as usize]
                    .outputs
                    .get(slot as usize)
                    .cloned()
                    .unwrap_or_default();
                Ok((FufInput::Tile { id: tile, slot }, shape))
            }
            Expr::Extern { kind, index } => {
                let index = index
                    .map(|lid| self.loop_var_value(lid))
                    .transpose()?
                    .map(UnrollIndex);
                let shape = crate::shape::extern_shape(*kind);
                Ok((FufInput::Extern { kind: *kind, index }, shape))
            }
            Expr::Weight { id, index } => {
                let index = index
                    .map(|lid| self.loop_var_value(lid))
                    .transpose()?
                    .map(UnrollIndex);
                let shape = self.inferred.weights.get(id).cloned().unwrap_or_default();
                Ok((
                    FufInput::Weight {
                        id: *id,
                        index,
                        storage: StorageFormat::Dense,
                    },
                    shape,
                ))
            }
            Expr::Call { op, args } => {
                // Nested call — promote it to its own tile. Compute
                // its output shape via the op's signature so the
                // FUF carries real shapes for downstream cost
                // evaluation.
                let resolved: Vec<(FufInput, Shape)> = args
                    .iter()
                    .map(|a| self.resolve_arg(a))
                    .collect::<Result<_, _>>()?;
                let (inputs, input_shapes): (Vec<FufInput>, Vec<Shape>) =
                    resolved.into_iter().unzip();
                let mut throwaway = crate::shape::Solver::new();
                let sig = crate::shape::apply_signature_with_geometry(
                    &mut throwaway,
                    *op,
                    &input_shapes,
                    self.inferred.hybrid_attention_geometry,
                )
                .map_err(|e| {
                    UnrollError::UnsupportedCfgShape(format!("nested {} shape: {e}", op.as_str()))
                })?;
                let out_shape = sig.output.clone();
                let tile_id = self.push_tile(*op, inputs, vec![sig.output]);
                Ok((
                    FufInput::Tile {
                        id: tile_id,
                        slot: 0,
                    },
                    out_shape,
                ))
            }
            Expr::Mul { lhs, rhs } => {
                // Elementwise multiplication (DSL's `*`, e.g.
                // `gate * up`). Output shape equals either operand;
                // we take the left operand's shape since shape
                // inference has already unified the two.
                let (l, l_shape) = self.resolve_arg(lhs)?;
                let (r, _r_shape) = self.resolve_arg(rhs)?;
                let tile_id = self.push_tile(OpKind::Mul, vec![l, r], vec![l_shape.clone()]);
                Ok((
                    FufInput::Tile {
                        id: tile_id,
                        slot: 0,
                    },
                    l_shape,
                ))
            }
            Expr::ScalarLit(v) => {
                // Scalar literal — rides along as a tile input with
                // an empty shape (rank-0) so shape inference treats
                // it as a broadcastable scalar.
                Ok((FufInput::Scalar(*v), Shape::new()))
            }
            Expr::SqrtBound(_) => unreachable!(
                "Expr::SqrtBound should have been folded to Expr::ScalarLit by \
                 cfg.rs::fold_scalars before reaching the unroller"
            ),
            Expr::ConfigScalar { .. } => unreachable!(
                "Expr::ConfigScalar should have been folded to Expr::ScalarLit by \
                 cfg.rs::fold_scalars before reaching the unroller"
            ),
        }
    }

    fn loop_var_value(&self, id: LocalId) -> Result<u64, UnrollError> {
        self.loop_vars
            .get(&id)
            .copied()
            .ok_or(UnrollError::MissingLocal { id })
    }

    fn push_tile(&mut self, op: OpKind, inputs: Vec<FufInput>, outputs: Vec<Shape>) -> TileId {
        let id = TileId(self.nodes.len() as u32);
        self.nodes.push(FufNode {
            id,
            op,
            inputs,
            outputs,
        });
        id
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cfg::build_cfg;
    use crate::classify::classify;
    use crate::config::{self, ModelParams};
    use crate::parse_python::parse_body;
    use crate::shape::{Dim, infer};
    use std::path::PathBuf;

    fn classify_src(src: &str) -> crate::classified::Program {
        let ast = parse_body(src).expect("parse DSL");
        classify(&ast).expect("classify")
    }

    fn llama_3_2_1b_params() -> ModelParams {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/arch")
            .join("configs")
            .join("llama")
            .join("llama-3.2-1b.json");
        config::load_file(&path).expect("load config")
    }

    /// Glue helper: classify → infer shapes → build CFG → unroll.
    fn unroll_src(src: &str, params: &ModelParams) -> Fuf {
        let program = classify_src(src);
        let inferred = infer(
            &program,
            &crate::weights_manifest::WeightsManifest::llama_test_conventions(),
            &std::collections::BTreeMap::new(),
        )
        .expect("infer");
        let cfg = build_cfg(&program, params).expect("build cfg");
        unroll(&cfg, &inferred).expect("unroll")
    }

    #[test]
    fn straight_line_produces_one_tile_per_op() {
        let params = llama_3_2_1b_params();
        let fuf = unroll_src("hidden_states = embed(input_ids, embed_tokens);", &params);
        assert_eq!(fuf.nodes.len(), 1);
        assert_eq!(fuf.nodes[0].op, OpKind::Embed);
    }

    #[test]
    fn loop_body_unrolls_n_times() {
        let params = llama_3_2_1b_params(); // num_hidden_layers = 16
        let fuf = unroll_src(
            "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(num_hidden_layers):
                hidden_states = add(hidden_states, hidden_states)
            ",
            &params,
        );
        // 1 embed + 16 adds = 17 tiles.
        assert_eq!(fuf.nodes.len(), 17, "expected 1 embed + 16 unrolled adds");

        // Every add's inputs should reference the previous tile.
        for (i, node) in fuf.nodes.iter().enumerate().skip(1) {
            assert_eq!(node.op, OpKind::Add);
            match &node.inputs[0] {
                FufInput::Tile { id, .. } => {
                    assert_eq!(id.0, (i as u32) - 1, "add #{i} should read tile {}", i - 1);
                }
                other => panic!("expected tile input, got {other:?}"),
            }
        }
    }

    #[test]
    fn weight_index_resolves_to_concrete_integer() {
        let params = llama_3_2_1b_params();
        let fuf = unroll_src(
            "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(3):
                normed = rmsnorm(hidden_states, input_layernorm[layer])
                hidden_states = add(normed, hidden_states)
            ",
            &params,
        );
        // Inside the loop body, each rmsnorm reads input_layernorm
        // with the concrete loop index. Collect those indices.
        let mut rmsnorm_indices = Vec::new();
        for node in &fuf.nodes {
            if node.op == OpKind::RmsNorm {
                match &node.inputs[1] {
                    FufInput::Weight { index: Some(i), .. } => rmsnorm_indices.push(i.0),
                    other => panic!("expected indexed weight, got {other:?}"),
                }
            }
        }
        assert_eq!(rmsnorm_indices, vec![0, 1, 2], "unrolled indices");
    }

    #[test]
    fn fuf_contains_no_strings() {
        // Structural invariant: FufNode carries no String
        // anywhere. Check each input/output via pattern. Body is
        // a minimal attention flow (q_proj must be followed by
        // o_proj so the hidden_states = add(...) shape contract
        // holds — conventions now pin q_proj's output to heads-
        // layout, which isn't hidden_size directly).
        let params = llama_3_2_1b_params();
        let fuf = unroll_src(
            "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(2):
                normed = rmsnorm(hidden_states, input_layernorm[layer])
                q = gemm(normed, self_attn.q_proj[layer])
                oproj = gemm(q, self_attn.o_proj[layer])
                hidden_states = add(oproj, hidden_states)
            ",
            &params,
        );
        for node in &fuf.nodes {
            for input in &node.inputs {
                match input {
                    FufInput::Tile { .. }
                    | FufInput::Weight { .. }
                    | FufInput::Extern { .. }
                    | FufInput::Scalar(_) => {
                        // All numeric/enum. No String.
                    }
                }
            }
            for shape in &node.outputs {
                for dim in shape {
                    // Dim::Bound carries a String by design — that's
                    // the config.json name. Not a tile-level string,
                    // a shape-level one, and it resolves per-model
                    // at codegen time. Acceptable.
                    match dim {
                        Dim::Lit(_)
                        | Dim::Bound(_)
                        | Dim::Mul(_)
                        | Dim::Div(_, _)
                        | Dim::Var(_) => {}
                    }
                }
            }
        }
    }

    #[test]
    fn rope_append_tuple_maps_three_locals_to_same_tile() {
        let params = llama_3_2_1b_params();
        let fuf = unroll_src(
            "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(1):
                normed = rmsnorm(hidden_states, input_layernorm[layer])
                q = gemm(normed, self_attn.q_proj[layer])
                k = gemm(normed, self_attn.k_proj[layer])
                v = gemm(normed, self_attn.v_proj[layer])
                (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
                attn = attention(q, k, v, kv_cache[layer], block_table)
                oproj = gemm(attn, self_attn.o_proj[layer])
                hidden_states = add(oproj, hidden_states)
            ",
            &params,
        );

        // Find the rope_append tile and the attention tile.
        let rope_tile = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::RopeAppend)
            .expect("rope_append present");
        let attn_tile = fuf
            .nodes
            .iter()
            .find(|n| n.op == OpKind::Attention)
            .expect("attention present");

        // rope_append has 3 outputs (q, k, v).
        assert_eq!(rope_tile.outputs.len(), 3);

        // attention's first 3 inputs all reference rope_tile, but
        // with slots 0, 1, 2.
        let slots: Vec<u8> = attn_tile.inputs[..3]
            .iter()
            .map(|inp| match inp {
                FufInput::Tile { id, slot } => {
                    assert_eq!(*id, rope_tile.id, "attn read from rope tile");
                    *slot
                }
                other => panic!("expected tile input, got {other:?}"),
            })
            .collect();
        assert_eq!(slots, vec![0, 1, 2]);
    }

    #[test]
    fn swiglu_mul_emits_opkind_mul_not_add() {
        // The real Llama SwiGLU MLP uses `silu(gate) * up`. The
        // unroller emits the `*` as an OpKind::Mul tile. Regression
        // check for the pre-fix shortcut that emitted OpKind::Add
        // for Expr::Mul (silent wrong-answer at codegen time).
        let params = llama_3_2_1b_params();
        let fuf = unroll_src(
            "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(1):
                normed = rmsnorm(hidden_states, input_layernorm[layer])
                gate = silu(gemm(normed, mlp.gate_proj[layer]))
                up = gemm(normed, mlp.up_proj[layer])
                down = gemm(gate * up, mlp.down_proj[layer])
                hidden_states = add(down, hidden_states)
            ",
            &params,
        );
        let n_mul = fuf.nodes.iter().filter(|n| n.op == OpKind::Mul).count();
        let n_add = fuf.nodes.iter().filter(|n| n.op == OpKind::Add).count();
        assert_eq!(n_mul, 1, "expected exactly one Mul tile for `gate * up`");
        assert_eq!(n_add, 1, "expected exactly one Add tile (residual)");
    }

    #[test]
    fn if_alternates_attention_and_sliding_attention() {
        // Real-shape acid test: a body that branches on `layer % 2`
        // and picks `attention` for even layers, `sliding_attention`
        // for odd. With 4 concrete iterations we expect 2 of each,
        // in the order [Attention, Sliding, Attention, Sliding].
        // Every downstream Add reads the merge binding for `attn`
        // which must point at the taken arm's tile per iteration.
        let params = llama_3_2_1b_params();
        let fuf = unroll_src(
            "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(4):
                normed = rmsnorm(hidden_states, input_layernorm[layer])
                q = gemm(normed, self_attn.q_proj[layer])
                k = gemm(normed, self_attn.k_proj[layer])
                v = gemm(normed, self_attn.v_proj[layer])
                (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
                if layer % 2 == 0:
                    attn = attention(q, k, v, kv_cache[layer], block_table)
                else:
                    attn = sliding_attention(q, k, v, kv_cache[layer], block_table)
                oproj = gemm(attn, self_attn.o_proj[layer])
                hidden_states = add(oproj, hidden_states)
            ",
            &params,
        );

        // Exactly one attention-family tile per iteration.
        let attn_family: Vec<OpKind> = fuf
            .nodes
            .iter()
            .filter_map(|n| match n.op {
                OpKind::Attention | OpKind::SlidingAttention => Some(n.op),
                _ => None,
            })
            .collect();
        assert_eq!(attn_family.len(), 4, "one attn tile per iteration");
        assert_eq!(
            attn_family,
            vec![
                OpKind::Attention,
                OpKind::SlidingAttention,
                OpKind::Attention,
                OpKind::SlidingAttention,
            ],
            "alternating per layer index"
        );

        // Each iteration's `oproj = gemm(attn, ...)` must read from
        // the attention tile that *that iteration's arm* produced.
        let attn_tile_ids: Vec<TileId> = fuf
            .nodes
            .iter()
            .filter(|n| matches!(n.op, OpKind::Attention | OpKind::SlidingAttention))
            .map(|n| n.id)
            .collect();
        let oproj_attn_refs: Vec<TileId> = fuf
            .nodes
            .iter()
            .filter(|n| n.op == OpKind::Gemm && n.inputs.len() == 2)
            // `oproj` is the only gemm whose first input is an
            // attention-family tile; filter by that.
            .filter_map(|n| match &n.inputs[0] {
                FufInput::Tile { id, .. } if attn_tile_ids.contains(id) => Some(*id),
                _ => None,
            })
            .collect();
        assert_eq!(
            oproj_attn_refs, attn_tile_ids,
            "each oproj consumes the corresponding iteration's attn output"
        );
    }

    #[test]
    fn if_less_with_config_bound_resolves_per_iteration() {
        // Pattern used by DeepSeek-V3: "first N layers are dense,
        // rest use a different variant." Here we stand that in with
        // attention vs. sliding_attention gated by `layer < N`.
        let mut params = llama_3_2_1b_params();
        params.bounds.insert("num_dense_layers".to_string(), 2);
        let fuf = unroll_src(
            "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(5):
                normed = rmsnorm(hidden_states, input_layernorm[layer])
                q = gemm(normed, self_attn.q_proj[layer])
                k = gemm(normed, self_attn.k_proj[layer])
                v = gemm(normed, self_attn.v_proj[layer])
                (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
                if layer < num_dense_layers:
                    attn = attention(q, k, v, kv_cache[layer], block_table)
                else:
                    attn = sliding_attention(q, k, v, kv_cache[layer], block_table)
                oproj = gemm(attn, self_attn.o_proj[layer])
                hidden_states = add(oproj, hidden_states)
            ",
            &params,
        );
        let attn_family: Vec<OpKind> = fuf
            .nodes
            .iter()
            .filter_map(|n| match n.op {
                OpKind::Attention | OpKind::SlidingAttention => Some(n.op),
                _ => None,
            })
            .collect();
        assert_eq!(
            attn_family,
            vec![
                OpKind::Attention,
                OpKind::Attention,
                OpKind::SlidingAttention,
                OpKind::SlidingAttention,
                OpKind::SlidingAttention,
            ],
            "first 2 dense, remaining 3 sliding"
        );
    }

    #[test]
    fn if_in_literal_array_alternates_per_iteration() {
        // Mirrors Qwen2.5-VL's `fullatt_block_indexes = [7, 15, 23, 31]`
        // dispatch — at the unroll-time literal predicate level, only
        // layers in the set take the `then` arm. Stand it in here
        // with `[1, 3]` over an 8-iteration loop and check the
        // attention-family ordering matches the membership pattern.
        let params = llama_3_2_1b_params();
        let fuf = unroll_src(
            "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(8):
                normed = rmsnorm(hidden_states, input_layernorm[layer])
                q = gemm(normed, self_attn.q_proj[layer])
                k = gemm(normed, self_attn.k_proj[layer])
                v = gemm(normed, self_attn.v_proj[layer])
                (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
                if layer in [1, 3]:
                    attn = attention(q, k, v, kv_cache[layer], block_table)
                else:
                    attn = sliding_attention(q, k, v, kv_cache[layer], block_table)
                oproj = gemm(attn, self_attn.o_proj[layer])
                hidden_states = add(oproj, hidden_states)
            ",
            &params,
        );
        let attn_family: Vec<OpKind> = fuf
            .nodes
            .iter()
            .filter_map(|n| match n.op {
                OpKind::Attention | OpKind::SlidingAttention => Some(n.op),
                _ => None,
            })
            .collect();
        assert_eq!(
            attn_family,
            vec![
                OpKind::SlidingAttention, // layer 0
                OpKind::Attention,        // layer 1 ∈ {1, 3}
                OpKind::SlidingAttention, // layer 2
                OpKind::Attention,        // layer 3 ∈ {1, 3}
                OpKind::SlidingAttention, // layer 4
                OpKind::SlidingAttention, // layer 5
                OpKind::SlidingAttention, // layer 6
                OpKind::SlidingAttention, // layer 7
            ],
            "in-set picks `then` arm; outside picks `else`"
        );
    }

    #[test]
    fn full_llama_body_tile_count() {
        // Acid test: realistic Llama body at realistic N (16).
        let params = llama_3_2_1b_params();
        let fuf = unroll_src(
            "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(num_hidden_layers):
                normed = rmsnorm(hidden_states, input_layernorm[layer])
                q = gemm(normed, self_attn.q_proj[layer])
                k = gemm(normed, self_attn.k_proj[layer])
                v = gemm(normed, self_attn.v_proj[layer])
                (q, k, v) = rope_append(q, k, v, positions, rotary, kv_cache[layer])
                attn = attention(q, k, v, kv_cache[layer], block_table)
                oproj = gemm(attn, self_attn.o_proj[layer])
                hidden_states = add(oproj, hidden_states)

                normed2 = rmsnorm(hidden_states, post_attention_layernorm[layer])
                up = gemm(normed2, mlp.up_proj[layer])
                down = gemm(up, mlp.down_proj[layer])
                hidden_states = add(down, hidden_states)
            normed = rmsnorm(hidden_states, norm)
            logits = gemm(normed, lm_head)
            ",
            &params,
        );

        // Per-iteration body: rmsnorm, gemm_q, gemm_k, gemm_v,
        //   rope_append, attention, gemm_o, add (attn), rmsnorm2,
        //   gemm_up, gemm_down, add (mlp) = 12 tiles.
        // NL = 16 → 192 body tiles.
        // Pre: embed (1). Post: rmsnorm_norm + gemm_lm_head (2).
        // Total: 1 + 192 + 2 = 195.
        assert_eq!(
            fuf.nodes.len(),
            195,
            "expected 1 embed + 12 tiles × 16 iters + 2 post-loop = 195 tiles"
        );
    }
}
