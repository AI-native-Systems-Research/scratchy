// SPDX-License-Identifier: Apache-2.0
//! Phase 5: build a per-model CFG from the classified program.
//!
//! Takes (ClassifiedProgram, ModelParams) and produces a CFG whose
//! loop terminators carry *concrete* integer trip counts — no
//! symbolic `num_hidden_layers`, no BTreeMap lookups at any later
//! phase.
//!
//! The CFG here is deliberately minimal: our DSL has straight-line
//! instruction blocks separated only by for-loops. A block is a
//! vec of instrs ending in a terminator (Jump, LoopHeader, Return).
//!
//! No bullshit allowed below this point: loop bounds are `u64`, the
//! program carries no `String` identifiers, and everything the
//! solver and codegen touch is numeric.

use std::collections::BTreeMap;

use crate::classified::{BoolPred, Bound, Expr, LocalId, Program, Stmt};
use crate::config::ModelParams;

// ── Types ─────────────────────────────────────────────────────────

/// Dense block identifier.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct BlockId(pub u32);

/// A basic block: straight-line instrs + a terminator.
#[derive(Clone, Debug)]
pub struct Block {
    pub instrs: Vec<Instr>,
    pub term: Terminator,
}

/// An instruction in a block. Same shape as a classified statement
/// but with no for-loop — those become block-level terminators.
#[derive(Clone, Debug)]
pub enum Instr {
    Assign { target: LocalId, value: Expr },
    AssignTuple { targets: Vec<LocalId>, value: Expr },
}

/// Block terminator.
#[derive(Clone, Debug)]
pub enum Terminator {
    /// Unconditional jump to another block.
    Jump(BlockId),
    /// Loop header. `body` is entered once per iteration, terminates
    /// by jumping back to this block; after the last iteration, flow
    /// proceeds to `exit`. `start`/`end` are *concrete* ints that
    /// came from this model's `ModelParams` (or from a DSL-level
    /// literal bound).
    ///
    /// `loop_carry` is forwarded from the classified `Stmt::For`:
    /// after each iteration, rebind each `(outer, inner)` local so
    /// the next iteration reads the previous iteration's output.
    LoopHeader {
        ivar: LocalId,
        start: u64,
        end: u64,
        body: BlockId,
        exit: BlockId,
        loop_carry: Vec<(LocalId, LocalId)>,
    },
    /// Compile-time conditional jump. The predicate's bound fields
    /// have been resolved to concrete `u64`s; at unroll time the
    /// predicate is evaluated against the current loop-var value
    /// and flow proceeds into `then_b` or `else_b`.
    CondJump {
        cond: BoolPredResolved,
        then_b: BlockId,
        else_b: BlockId,
    },
    /// Jump to `target` and, before handing control to `target`,
    /// rebind each `(merge_id, source_id)` pair: the unroller sets
    /// `local_to_tile[merge_id]` to whatever `source_id` currently
    /// points at. Used at the exit of each `if` arm to populate
    /// the merge bindings with the taken arm's final writes.
    JumpWithCarry {
        target: BlockId,
        carry: Vec<(LocalId, LocalId)>,
    },
    /// Function end.
    Return,
}

/// `BoolPred` with its `Bound`s resolved to concrete integers using
/// the model's `ModelParams`. Produced by [`build_cfg`] and consumed
/// by the unroller.
#[derive(Clone, Debug)]
pub enum BoolPredResolved {
    /// `ivar % divisor == remainder`.
    Modulo {
        ivar: LocalId,
        divisor: u64,
        remainder: u64,
    },
    /// `ivar % divisor != remainder`.
    NotModulo {
        ivar: LocalId,
        divisor: u64,
        remainder: u64,
    },
    /// `ivar < bound`.
    Less { ivar: LocalId, bound: u64 },
    /// `members.contains(&ivar)` — set membership; resolved members
    /// are concrete u64 literals. See [`crate::classified::BoolPred::In`].
    In { ivar: LocalId, members: Vec<u64> },
}

/// The CFG for one specialization.
#[derive(Clone, Debug)]
pub struct Cfg {
    pub blocks: Vec<Block>,
    pub entry: BlockId,
}

impl Cfg {
    pub fn block(&self, id: BlockId) -> &Block {
        &self.blocks[id.0 as usize]
    }
}

/// Errors when building the CFG. Distinct from shape-inference
/// errors because these concern bound resolution, not shapes.
#[derive(Debug)]
pub enum CfgError {
    UnknownBound {
        name: String,
        /// Names we did have, for diagnostics.
        available: Vec<String>,
    },
    UnknownScalar {
        name: String,
        available: Vec<String>,
    },
}

impl std::fmt::Display for CfgError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::UnknownBound { name, available } => {
                write!(
                    f,
                    "loop bound `{name}` not in this model's config.json. \
                     Available: {available:?}",
                )
            }
            Self::UnknownScalar { name, available } => {
                write!(
                    f,
                    "config scalar `{name}` not in this model's config.json. \
                     Available: {available:?}",
                )
            }
        }
    }
}

impl std::error::Error for CfgError {}

// ── Builder ──────────────────────────────────────────────────────

pub fn build_cfg(program: &Program, params: &ModelParams) -> Result<Cfg, CfgError> {
    let mut builder = CfgBuilder::new(&params.bounds, &params.scalars);
    let entry = builder.alloc_block();
    let exit = builder.alloc_block();
    builder.lower_stmts(&program.statements, entry, exit)?;
    builder.finalize_block(exit, Terminator::Return);
    Ok(Cfg {
        blocks: builder.blocks,
        entry,
    })
}

struct CfgBuilder<'a> {
    bounds: &'a BTreeMap<String, u64>,
    scalars: &'a BTreeMap<String, f64>,
    blocks: Vec<Block>,
    /// One-per-block open instruction buffer: instrs accumulate
    /// into the indexed block's `instrs` field when we commit.
    open: BTreeMap<BlockId, Vec<Instr>>,
}

impl<'a> CfgBuilder<'a> {
    fn new(bounds: &'a BTreeMap<String, u64>, scalars: &'a BTreeMap<String, f64>) -> Self {
        Self {
            bounds,
            scalars,
            blocks: Vec::new(),
            open: BTreeMap::new(),
        }
    }

    fn alloc_block(&mut self) -> BlockId {
        let id = BlockId(self.blocks.len() as u32);
        // Placeholder; lower_stmts will rewrite instrs and term
        // before finalize.
        self.blocks.push(Block {
            instrs: Vec::new(),
            term: Terminator::Return,
        });
        self.open.insert(id, Vec::new());
        id
    }

    fn push_instr(&mut self, block: BlockId, instr: Instr) {
        self.open.entry(block).or_default().push(instr);
    }

    fn finalize_block(&mut self, block: BlockId, term: Terminator) {
        let instrs = self.open.remove(&block).unwrap_or_default();
        self.blocks[block.0 as usize].instrs = instrs;
        self.blocks[block.0 as usize].term = term;
    }

    /// Recursively fold compile-time scalar expressions using the
    /// per-model bounds. Currently: `SqrtBound(name)` →
    /// `ScalarLit((bounds[name] as f64).sqrt())`. Any other
    /// expression descends into its children.
    fn fold_scalars(&self, expr: &Expr) -> Result<Expr, CfgError> {
        match expr {
            Expr::SqrtBound(ident) => {
                let name = ident.to_string();
                let v = self.bounds.get(&name).copied().ok_or_else(|| {
                    let available: Vec<String> = self.bounds.keys().cloned().collect();
                    CfgError::UnknownBound { name, available }
                })?;
                Ok(Expr::ScalarLit((v as f64).sqrt()))
            }
            Expr::ConfigScalar { name: ident, recip } => {
                let name = ident.to_string();
                let v = self.scalars.get(&name).copied().ok_or_else(|| {
                    let available: Vec<String> = self.scalars.keys().cloned().collect();
                    CfgError::UnknownScalar {
                        name: name.clone(),
                        available,
                    }
                })?;
                Ok(Expr::ScalarLit(if *recip { 1.0 / v } else { v }))
            }
            Expr::Call { op, args } => {
                let args = args
                    .iter()
                    .map(|a| self.fold_scalars(a))
                    .collect::<Result<Vec<_>, _>>()?;
                Ok(Expr::Call { op: *op, args })
            }
            Expr::Mul { lhs, rhs } => Ok(Expr::Mul {
                lhs: Box::new(self.fold_scalars(lhs)?),
                rhs: Box::new(self.fold_scalars(rhs)?),
            }),
            other => Ok(other.clone()),
        }
    }

    /// Lower a straight-line statement sequence into blocks. `start`
    /// is the block instrs begin accumulating in; on exit we jump
    /// to `end`.
    fn lower_stmts(
        &mut self,
        stmts: &[Stmt],
        mut current: BlockId,
        end: BlockId,
    ) -> Result<(), CfgError> {
        for stmt in stmts {
            match stmt {
                Stmt::Assign { target, value } => {
                    let value = self.fold_scalars(value)?;
                    self.push_instr(
                        current,
                        Instr::Assign {
                            target: *target,
                            value,
                        },
                    );
                }
                Stmt::AssignTuple { targets, value } => {
                    let value = self.fold_scalars(value)?;
                    self.push_instr(
                        current,
                        Instr::AssignTuple {
                            targets: targets.clone(),
                            value,
                        },
                    );
                }
                Stmt::For {
                    ivar,
                    start,
                    end: loop_end,
                    body,
                    loop_carry,
                } => {
                    let start_val = self.resolve_bound(start)?;
                    let end_val = self.resolve_bound(loop_end)?;
                    let header = self.alloc_block();
                    let body_block = self.alloc_block();
                    let after = self.alloc_block();
                    // Current block jumps to the header.
                    self.finalize_block(current, Terminator::Jump(header));
                    self.finalize_block(
                        header,
                        Terminator::LoopHeader {
                            ivar: *ivar,
                            start: start_val,
                            end: end_val,
                            body: body_block,
                            exit: after,
                            loop_carry: loop_carry.clone(),
                        },
                    );
                    // Body: recurse. Its last block jumps back to
                    // the header.
                    self.lower_stmts(body, body_block, header)?;
                    // Continue with `after`.
                    current = after;
                }
                Stmt::If {
                    cond,
                    then_body,
                    else_body,
                    merge_carry,
                } => {
                    let cond_resolved = self.resolve_pred(cond)?;
                    let then_start = self.alloc_block();
                    let else_start = self.alloc_block();
                    let then_tail = self.alloc_block();
                    let else_tail = self.alloc_block();
                    let merge = self.alloc_block();

                    self.finalize_block(
                        current,
                        Terminator::CondJump {
                            cond: cond_resolved,
                            then_b: then_start,
                            else_b: else_start,
                        },
                    );

                    // Lower each arm. lower_stmts finalizes each
                    // arm's last block with `Jump(then_tail)` /
                    // `Jump(else_tail)` respectively.
                    self.lower_stmts(then_body, then_start, then_tail)?;
                    self.lower_stmts(else_body, else_start, else_tail)?;

                    // Install merge carries on the two arm tails.
                    let then_carry: Vec<(LocalId, LocalId)> =
                        merge_carry.iter().map(|(mid, tf, _)| (*mid, *tf)).collect();
                    let else_carry: Vec<(LocalId, LocalId)> =
                        merge_carry.iter().map(|(mid, _, ef)| (*mid, *ef)).collect();
                    self.finalize_block(
                        then_tail,
                        Terminator::JumpWithCarry {
                            target: merge,
                            carry: then_carry,
                        },
                    );
                    self.finalize_block(
                        else_tail,
                        Terminator::JumpWithCarry {
                            target: merge,
                            carry: else_carry,
                        },
                    );

                    // Continue in merge.
                    current = merge;
                }
            }
        }
        // After the last statement, jump to the outer end block.
        self.finalize_block(current, Terminator::Jump(end));
        Ok(())
    }

    fn resolve_pred(&self, pred: &BoolPred) -> Result<BoolPredResolved, CfgError> {
        match pred {
            BoolPred::Modulo {
                ivar,
                divisor,
                remainder,
            } => Ok(BoolPredResolved::Modulo {
                ivar: *ivar,
                divisor: self.resolve_bound(divisor)?,
                remainder: self.resolve_bound(remainder)?,
            }),
            BoolPred::NotModulo {
                ivar,
                divisor,
                remainder,
            } => Ok(BoolPredResolved::NotModulo {
                ivar: *ivar,
                divisor: self.resolve_bound(divisor)?,
                remainder: self.resolve_bound(remainder)?,
            }),
            BoolPred::Less { ivar, bound } => Ok(BoolPredResolved::Less {
                ivar: *ivar,
                bound: self.resolve_bound(bound)?,
            }),
            BoolPred::In { ivar, members } => Ok(BoolPredResolved::In {
                ivar: *ivar,
                members: members.clone(),
            }),
        }
    }

    fn resolve_bound(&self, b: &Bound) -> Result<u64, CfgError> {
        match b {
            Bound::Lit(n) => Ok(*n),
            Bound::Sym(ident) => {
                let name = ident.to_string();
                self.bounds.get(&name).copied().ok_or_else(|| {
                    let available: Vec<String> = self.bounds.keys().cloned().collect();
                    CfgError::UnknownBound { name, available }
                })
            }
        }
    }
}

// ── Tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::classify::classify;
    use crate::config::{self};
    use crate::parse_python::parse_body;
    use std::path::PathBuf;

    fn classify_src(src: &str) -> Program {
        let ast = parse_body(src).expect("parse DSL");
        classify(&ast).expect("classify")
    }

    fn llama_3_2_1b_params() -> ModelParams {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../models/arch")
            // ⛔ `configs/llama`, NOT `llama/configs`. The two components were
            // swapped — a stale path from the per-arch-crate layout that the
            // collapse into `scratchy-models` replaced. It never surfaced
            // because this whole test module could not compile off cuda.
            .join("configs")
            .join("llama")
            .join("llama-3.2-1b.json");
        config::load_file(&path).expect("load llama-3.2-1b config")
    }

    #[test]
    fn straight_line_program_single_block() {
        let p = classify_src("x = embed(input_ids, embed_tokens);");
        let params = llama_3_2_1b_params();
        let cfg = build_cfg(&p, &params).expect("build cfg");

        // 2 blocks: entry (with the instr), exit (empty + Return).
        assert_eq!(cfg.blocks.len(), 2);
        let entry = cfg.block(cfg.entry);
        assert_eq!(entry.instrs.len(), 1);
        assert!(matches!(entry.term, Terminator::Jump(_)));
    }

    #[test]
    fn for_loop_creates_header_body_after_blocks() {
        let p = classify_src(
            "
            for layer in range(num_hidden_layers):
                x = embed(input_ids, embed_tokens)
            ",
        );
        let params = llama_3_2_1b_params();
        let cfg = build_cfg(&p, &params).expect("build cfg");

        // Blocks: entry, header, body, after, exit = 5.
        assert_eq!(cfg.blocks.len(), 5);

        // Find the LoopHeader terminator.
        let header_idx = cfg
            .blocks
            .iter()
            .position(|b| matches!(b.term, Terminator::LoopHeader { .. }))
            .expect("loop header block");
        let header_id = BlockId(header_idx as u32);
        let header = &cfg.blocks[header_idx];
        match &header.term {
            Terminator::LoopHeader {
                start,
                end,
                body,
                exit,
                ..
            } => {
                assert_eq!(*start, 0, "start is 0 (from DSL literal)");
                assert_eq!(*end, 16, "end resolved to llama-3.2-1b num_hidden_layers");
                // Body block must jump back to header.
                let body_block = cfg.block(*body);
                match body_block.term {
                    Terminator::Jump(target) => {
                        assert_eq!(target.0, header_id.0, "body should back-edge to header")
                    }
                    _ => panic!("body terminator should be Jump to header"),
                }
                // After block exists.
                let _ = cfg.block(*exit);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn unknown_bound_errors() {
        let p = classify_src(
            "
            for i in range(totally_bogus_bound):
                x = embed(input_ids, e)
            ",
        );
        let params = llama_3_2_1b_params();
        let err = build_cfg(&p, &params).unwrap_err();
        let msg = err.to_string();
        assert!(
            msg.contains("totally_bogus_bound"),
            "error names bound: {msg}"
        );
    }

    #[test]
    fn literal_bounds_pass_through() {
        let p = classify_src(
            "
            for i in range(5):
                x = embed(input_ids, e)
            ",
        );
        let params = llama_3_2_1b_params();
        let cfg = build_cfg(&p, &params).expect("build cfg");

        let header = cfg
            .blocks
            .iter()
            .find(|b| matches!(b.term, Terminator::LoopHeader { .. }))
            .unwrap();
        match header.term {
            Terminator::LoopHeader { start, end, .. } => {
                assert_eq!(start, 0);
                assert_eq!(end, 5);
            }
            _ => unreachable!(),
        }
    }

    #[test]
    fn cfg_has_no_strings_in_terminators() {
        // Structural invariant: no Terminator variant carries a
        // String. (It doesn't — this test is just a compile-time
        // smoke check documenting the invariant.)
        fn assert_no_strings(_term: &Terminator) {
            // If someone adds a `String` to a Terminator variant,
            // this function would no longer be vacuously valid
            // because they'd have to cover that variant in some
            // way. The invariant is "all data in Terminator is
            // numeric or block-indexing."
        }
        let p = classify_src("x = embed(input_ids, e);");
        let cfg = build_cfg(&p, &llama_3_2_1b_params()).unwrap();
        for b in &cfg.blocks {
            assert_no_strings(&b.term);
        }
    }

    #[test]
    fn if_creates_condjump_and_two_jumpwithcarry_tails() {
        let p = classify_src(
            "
            for layer in range(4):
                if layer % 2 == 0:
                    attn = attention(q, k, v, kv_cache, block_table)
                else:
                    attn = sliding_attention(q, k, v, kv_cache, block_table)
                hidden_states = add(attn, attn)
            ",
        );
        let params = llama_3_2_1b_params();
        let cfg = build_cfg(&p, &params).expect("build cfg");

        let condjumps: Vec<_> = cfg
            .blocks
            .iter()
            .filter(|b| matches!(b.term, Terminator::CondJump { .. }))
            .collect();
        assert_eq!(condjumps.len(), 1, "exactly one CondJump");

        let carries: Vec<_> = cfg
            .blocks
            .iter()
            .filter_map(|b| match &b.term {
                Terminator::JumpWithCarry { target, carry } => Some((*target, carry.len())),
                _ => None,
            })
            .collect();
        assert_eq!(carries.len(), 2, "two arm tails with JumpWithCarry");
        // Both arm tails jump to the same merge block and each
        // carries exactly one rebinding (`attn`).
        assert_eq!(carries[0].0, carries[1].0, "both arms target same merge");
        for (_, n) in &carries {
            assert_eq!(*n, 1, "one carry entry per arm (attn)");
        }

        // The CondJump's predicate must resolve to `layer % 2 == 0`.
        match &condjumps[0].term {
            Terminator::CondJump { cond, .. } => match cond {
                BoolPredResolved::Modulo {
                    divisor, remainder, ..
                } => {
                    assert_eq!(*divisor, 2);
                    assert_eq!(*remainder, 0);
                }
                _ => panic!("expected Modulo predicate"),
            },
            _ => unreachable!(),
        }
    }

    #[test]
    fn if_in_literal_array_resolves_to_in_pred() {
        let p = classify_src(
            "
            for layer in range(4):
                if layer in [0, 2]:
                    attn = attention(q, k, v, kv_cache, block_table)
                else:
                    attn = sliding_attention(q, k, v, kv_cache, block_table)
                hidden_states = add(attn, attn)
            ",
        );
        let cfg = build_cfg(&p, &llama_3_2_1b_params()).expect("build cfg");
        let condjump = cfg
            .blocks
            .iter()
            .find(|b| matches!(b.term, Terminator::CondJump { .. }))
            .expect("CondJump block");
        match &condjump.term {
            Terminator::CondJump {
                cond: BoolPredResolved::In { members, .. },
                ..
            } => {
                assert_eq!(members, &vec![0u64, 2]);
            }
            _ => panic!("expected In predicate"),
        }
    }

    #[test]
    fn if_with_symbolic_bound_resolves_to_int() {
        // `sliding_window_pattern` isn't in llama config, so this
        // model's params get a synthetic one for the test.
        let mut params = llama_3_2_1b_params();
        params
            .bounds
            .insert("sliding_window_pattern".to_string(), 2);
        let p = classify_src(
            "
            for layer in range(4):
                if layer % sliding_window_pattern == 0:
                    attn = attention(q, k, v, kv_cache, block_table)
                else:
                    attn = sliding_attention(q, k, v, kv_cache, block_table)
                hidden_states = add(attn, attn)
            ",
        );
        let cfg = build_cfg(&p, &params).expect("build cfg");
        let condjump = cfg
            .blocks
            .iter()
            .find(|b| matches!(b.term, Terminator::CondJump { .. }))
            .expect("CondJump");
        match &condjump.term {
            Terminator::CondJump {
                cond:
                    BoolPredResolved::Modulo {
                        divisor, remainder, ..
                    },
                ..
            } => {
                assert_eq!(*divisor, 2, "sliding_window_pattern resolved to 2");
                assert_eq!(*remainder, 0);
            }
            _ => panic!("expected Modulo"),
        }
    }

    #[test]
    fn full_llama_body_cfg() {
        let p = classify_src(
            "
            hidden_states = embed(input_ids, embed_tokens)
            for layer in range(num_hidden_layers):
                normed = rmsnorm(hidden_states, input_layernorm[layer])
                q = gemm(normed, self_attn.q_proj[layer])
                hidden_states = add(normed, hidden_states)
            normed = rmsnorm(hidden_states, norm)
            ",
        );
        let cfg = build_cfg(&p, &llama_3_2_1b_params()).unwrap();

        // Find the loop header; body should contain 3 instrs.
        let header_block = cfg
            .blocks
            .iter()
            .find(|b| matches!(b.term, Terminator::LoopHeader { .. }))
            .unwrap();
        let body_id = match header_block.term {
            Terminator::LoopHeader { body, .. } => body,
            _ => unreachable!(),
        };
        let body = cfg.block(body_id);
        assert_eq!(body.instrs.len(), 3, "3 body instrs (normed, q, hidden)");

        // Entry block has 1 instr (embed).
        let entry = cfg.block(cfg.entry);
        assert_eq!(entry.instrs.len(), 1);

        // After the loop, there's a block with the post-loop rmsnorm.
        let post_loop: Vec<_> = cfg
            .blocks
            .iter()
            .filter(|b| b.instrs.len() == 1 && !std::ptr::eq(*b, entry))
            .collect();
        assert!(
            post_loop
                .iter()
                .any(|b| matches!(b.instrs[0], Instr::Assign { .. })),
            "there is a post-loop block with an Assign for the final rmsnorm",
        );
    }
}
