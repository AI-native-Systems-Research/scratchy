//! THE DATAFLOWIR OPS. One variant per operation the emitted programs contain, and no more.
//!
//! The set is the union of `Dataflow.td`'s own ops and the standard-dialect ops a real program uses,
//! read off IBM's `dcc/test/PT/xrfbmm_int8_fwd.mlir` — a complete int8 BMM in about sixty lines.
//!
//! ⛔ WHAT IS NOT HERE IS THE POINT. No register, no port, no result forwarding, no unroll factor,
//! no precision per operand. Those are `sentient.*`, and dcc's 76 passes derive them
//! (`dbo/docs/pass_pipeline.md` D1-D76). An op here that named a register would be one rung down the
//! ladder from where this crate stands.
//!
//! ⭐⭐ ONE MODULE PER DIALECT, AND THE DIALECT IS THE OUTER ARM. Which `.td` declares an op is a
//! fact about the op, so the type carries it rather than a prefix on a variant's name.
//! `agen.vector_load` and `affine.vector_load` are two operations of two dialects — see
//! [`agen::Op::VectorLoad`] for which of them the scheduler's own producer emits — and as sibling
//! variants of one flat enum nothing but the spelling said so.

pub mod affine;
pub mod agen;
pub mod arith;
pub mod dataflow;
pub mod scf;
pub mod symbol;
pub mod uniform;
pub mod vector;
pub mod vectorchain;

use crate::islands::dataflow_ir::Values;

/// AN SSA VALUE, minted by the builder and never spelled by hand.
///
/// ⛔ A NEWTYPE OVER THE NUMBER, so a value cannot be confused with an extent, an address or a loop
/// bound — all of which are also small integers in this IR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct Val(pub u32);

/// ONE INDEX OF A LOAD OR STORE: an induction variable, an applied map, or a literal.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Index {
    /// An SSA value — a loop's induction variable or an `affine.apply` result.
    Val(Val),
    /// A literal, as `%lrf_memory[4, 0]` writes it.
    Const(i64),
    /// ⭐⭐ A SUM OF STRIDED INDUCTION VARIABLES, WRITTEN INLINE — `%arg9 + %arg8 * 8`.
    ///
    /// ⛔⛔ INLINE, NOT AN `affine.apply`. One view dimension is walked by EVERY enclosing loop that
    /// strides its axis, so an index is a sum, not a single variable. IBM writes those sums straight
    /// into the index list — `agen.vector_store %48, %49[0, %arg9 + %arg8 * 8, 0]`
    /// (`dcc/test/PT/bf16-pt.mlir:161`) — and `bf16-pt.mlir` contains **zero** `affine.apply`.
    /// Emitting one per index instead made `dbo-opt` refuse outright: "'affine.apply' op Expanded
    /// affine.apply operation does not resolve to a constant".
    ///
    /// ⭐ A STRIDE OF ONE PRINTS BARE. `%arg9 * 1` is the same address written longer, and the
    /// vendored files never write it.
    ///
    /// ⭐⭐ AND A CONSTANT ADDEND, WHICH IS HOW A FAN-OUT'S SLICES DIFFER. One `ddl.data_transfer`
    /// to a row-expanding `unit="pt"` becomes one send PER ROW, and the eight PT rows are a systolic
    /// accumulation chain (`bmm.ddl:256-260`: row 0 seeds with `%zero_const`, rows 1-7 add
    /// `%pt_src02_north`) each MACing from its OWN XRF — so they need eight DIFFERENT slices, not
    /// one broadcast. The i-th slice sits `i * stride/count` along the axis the innermost enclosing
    /// loop strides.
    ///
    /// ⛔ INLINE, NOT AN `affine.apply` — same reason as the terms above.
    Strided(Vec<(Val, i64)>, i64),
}

/// ONE DATAFLOWIR OPERATION, under the dialect that declares it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Op {
    /// Upstream `arith` — the constants and the integer predicates.
    Arith(arith::Op),
    /// Upstream `scf` — the undecided branch.
    Scf(scf::Op),
    /// Upstream `affine` — the loop nest, the applied maps, and `dcc-opt`'s vector accesses.
    Affine(affine::Op),
    /// Upstream `vector` — the two plain accesses the vectorchain lowerings still read.
    Vector(vector::Op),
    /// `Dataflow.td` — units, views, transfers between units, and the opaque bodies.
    Dataflow(dataflow::Op),
    /// `Agen.td` — the address-generator's accesses and composite transfers.
    Agen(agen::Op),
    /// `VectorChain.td` — everything the PE and the SFP compute.
    VectorChain(vectorchain::Op),
    /// Upstream `symbol` — a scalar the schedule fixes later.
    Symbol(symbol::Op),
    /// `Uniform.td` — one program written once and mapped onto many units.
    Uniform(uniform::Op),
}

// ─────────────────────────────── THE USE LIST ────────────────────────────────

/// EVERY VALUE ONE OP **READS**, in the order the op names them.
///
/// # 🛑 THIS EXISTS SO THAT `hasOneUse` CAN BE ASKED HONESTLY
///
/// ⛔⛔ A SEARCH THAT ONLY INSPECTS THE OPS IT EXPECTS CANNOT COUNT USES. `getLoadConsumer`
/// (`Helper.cpp:1256`) refuses a load whose result has more than one use, and the whole point of that
/// refusal is the use it did not expect — a compute that also reads the loaded vector. A census
/// restricted to sends and rearrangements would report one use for a value that has three and the
/// refusal would never fire.
///
/// ⛔ SO IT IS TOTAL OVER THE ENUM, WITH NO WILDCARD ANYWHERE. A new op must state what it reads;
/// falling through to "nothing" would silently under-count.
///
/// ⛔ AND IT DOES NOT DESCEND INTO REGIONS. An op's operands are its own; a value read inside an
/// `affine.for` body is read by the op in that body, which is what [`uses`] walks.
#[must_use]
pub fn operands(op: &Op) -> Vec<Val> {
    let mut reads: Vec<Val> = Vec::new();
    match op {
        Op::Arith(op) => match op {
            // A literal reads nothing.
            arith::Op::Constant { .. }
            | arith::Op::ConstantInt { .. }
            | arith::Op::DenseConstant { .. } => {}
            // ⭐ THE ADDRESS ARITHMETIC READS TWO VALUES. `insertCopyAndAddStmtsHelper` closes a
            // carrying loop with `arith.addi %iter_arg, %c<coeff>` (`AgenToSentient.hpp:502-520`),
            // and BOTH the carried argument and the coefficient constant are uses of their values.
            arith::Op::AddI(bin)
            | arith::Op::SubI(bin)
            | arith::Op::MulI(bin)
            | arith::Op::DivSI(bin)
            | arith::Op::RemSI(bin) => {
                reads.extend([bin.lhs, bin.rhs]);
            }
            // ⭐ ALL THREE, AND THE CONDITION FIRST — `arith.select`'s operand order is
            // `$condition, $true_value, $false_value`.
            arith::Op::Select {
                condition,
                true_value,
                false_value,
                ..
            } => reads.extend([*condition, *true_value, *false_value]),
            // ⭐ THE PREDICATE IS NOT AN OPERAND — it is `arith.cmpi`'s first token, an
            // attribute. Both compared values are uses; which comparison it is, is not.
            arith::Op::Compare { lhs, rhs, .. } => reads.extend([*lhs, *rhs]),
            arith::Op::Logic { operands, .. } => reads.extend(operands.iter().copied()),
        },
        // ⭐ A SYMBOL READS NOTHING — its id is an attribute, not an operand (`Symbol.td:59`).
        Op::Symbol(symbol::Op::CreateSymbol { .. }) => {}
        Op::Uniform(op) => match op {
            // ⛔⛔ THE UNITS ARE ONE FLAT OPERAND RANGE IN REGION ORDER, and this is where the
            // reference's `$units` / `$list_sizes` pair is put back together.
            // `Variadic<AnyType>:$units` (`Uniform.td:86`) is sliced per region by the prefix sum of
            // `$list_sizes` (`Uniform.cpp:184-192`), so concatenating the per-region lists in region
            // order **is** that range — see [`uniform::LocalRegion`]. The REGION ARGUMENTS are not
            // here: they are block arguments (`Uniform.td:96`) and belong to [`block_args`].
            uniform::Op::UniformizeRegions { regions, .. } => {
                for region in regions {
                    reads.extend(region.units.iter().copied());
                }
            }
            // ⭐ A TERMINATOR READS WHAT IT YIELDS (`Uniform.td:71`).
            uniform::Op::Yield { operands } => reads.extend(operands.iter().copied()),
            // ⭐ KEYS AND VALUES BOTH, KEY FIRST — the two `Variadic` ranges in declaration order
            // (`Uniform.td:132-133`), *"paired positionally"* (`:124-125`).
            uniform::Op::DefImmutableMapping { pairs, .. } => {
                for (key, value) in pairs {
                    reads.extend([*key, *value]);
                }
            }
            uniform::Op::QueryMap { map, key, .. } => reads.extend([*map, *key]),
        },
        Op::Scf(op) => match op {
            // ⛔ THE INDUCTION VARIABLES ARE NOT OPERANDS. `scf.parallel`'s `ivs` are the region's
            // arguments — values it DEFINES — so counting them here would make every loop a user of
            // its own variable.
            scf::Op::Parallel { ivs: _, body: _ } => {}
            // ⭐ ALL THREE BOUNDS ARE OPERANDS — `scf.for`'s lower bound, upper bound and step are
            // SSA values it reads, which is exactly why the trip count has to be reconstructed from
            // their defining constants (`LoopUnrollForShuffleOp.cpp:168-170`). The `iv` is the
            // region's argument, not an operand, same as `scf.parallel`'s.
            //
            // ⭐⭐ AND IT IS THE REASON `TransformLoopToLegalizeForSentientLowering` EXISTS:
            // `scf.for %i = %c0 to %10 step %c1` reads `%10`, and when `%10` is an `arith.select` the
            // loop's trip count is not affine. A census that skipped them would report the select as
            // unused and let a rewrite erase the value the loop counts to.
            scf::Op::For {
                iv: _,
                lo,
                hi,
                step,
                carried,
                body: _,
                dbg_name: _,
            } => {
                reads.extend([*lo, *hi, *step]);
                reads.extend(carried.iter().map(|carried| carried.init));
            }
            scf::Op::If { cond, .. } => reads.push(*cond),
            scf::Op::Yield { operands } => reads.extend(operands.iter().copied()),
        },
        Op::Affine(op) => match op {
            // Same as `scf.parallel`: `iv` is the body's argument, not an operand. A dynamic bound
            // IS one.
            affine::Op::For {
                iv: _,
                lo,
                hi,
                carried,
                body: _,
                dbg_name: _,
            } => {
                for bound in [lo, hi] {
                    if let affine::Bound::Val(val) = bound {
                        reads.push(*val);
                    }
                }
                // ⭐ AN `iter_args` INITIALISER IS AN OPERAND OF THE LOOP, evaluated outside it —
                // so the address a carrying loop starts from is USED by the `affine.for` itself.
                // Its `arg` and its `result` are not: they are [`block_args`] and [`results`].
                reads.extend(carried.iter().map(|carried| carried.init));
            }
            affine::Op::Apply { args, .. } => reads.extend(args.iter().copied()),
            // ⭐ THE SET'S OPERANDS ARE USES — both lists. `affine.if #set0(%arg9)` reads `%arg9`
            // as surely as `scf.if %53` reads its predicate; the SET itself is an attribute.
            affine::Op::If {
                args, symbol_args, ..
            } => {
                reads.extend(args.iter().copied());
                reads.extend(symbol_args.iter().copied());
            }
            affine::Op::Yield { operands } => reads.extend(operands.iter().copied()),
            affine::Op::VectorLoad { view, indices, .. } => {
                reads.push(*view);
                index_operands(indices, &mut reads);
            }
            affine::Op::VectorStore {
                value,
                view,
                indices,
                ..
            } => {
                reads.extend([*value, *view]);
                index_operands(indices, &mut reads);
            }
        },
        Op::Dataflow(op) => match op {
            dataflow::Op::GetUnit { .. } | dataflow::Op::Opaque(_) => {}
            dataflow::Op::GetLocalUnit { of, .. } => reads.push(*of),
            // ⭐ EVERY MEMBER IS AN OPERAND — `Variadic<Index>:$unit_ids` (`Dataflow.td:152`).
            dataflow::Op::CreateGroup { unit_ids, .. } => reads.extend(unit_ids.iter().copied()),
            dataflow::Op::GetLogicalMemoryView { from, start, .. } => reads.extend([*from, *start]),
            // ⭐ EVERY PAGE'S START ADDRESS IS AN OPERAND — `Variadic<Index>:$page_start_addrs`
            // (`Dataflow.td:267-299`) — and the extents beside them are attributes, so they are not.
            dataflow::Op::GetPagedLogicalMemoryView(view) => {
                reads.extend([view.unit, view.start_addr]);
                reads.extend(view.pages.iter().map(|page| page.start_addr));
            }
            dataflow::Op::ProgramUnit { units, .. } => reads.extend(units.iter().copied()),
            // ⭐ THE SEND'S DESTINATION IS AN OPERAND, and it is the one `getLoadConsumer` follows
            // back to a `get_unit` (`Helper.cpp:1266`).
            dataflow::Op::Send { to, data, .. } => reads.extend([to.val(), *data]),
            dataflow::Op::Receive { from, .. } => reads.push(from.val()),
            dataflow::Op::SyncSend { to, .. } => reads.push(*to),
            dataflow::Op::SyncRecv { from, .. } => reads.push(*from),
            dataflow::Op::ImplicitSync {
                view, dst, size, ..
            } => reads.extend([*view, *dst, *size]),
        },
        Op::Agen(op) => match op {
            agen::Op::Yield => {}
            agen::Op::VectorLoad { view, indices, .. } => {
                reads.push(*view);
                index_operands(indices, &mut reads);
            }
            agen::Op::VectorStore {
                value,
                view,
                indices,
                ..
            } => {
                reads.extend([*value, *view]);
                index_operands(indices, &mut reads);
            }
            // ⛔ THE STRIDES AND THE MULTICAST HANDLE ARE OPERANDS TOO — `operands1` holds the
            // subscript, then the strides, then at most one multicast group (`Agen.td:1128-1130`),
            // and a use-chain walk that missed them would treat a symbolic access as reading
            // nothing but its view.
            agen::Op::SymbolicVectorLoad {
                view,
                indices,
                strides,
                multicast,
                ..
            } => {
                reads.push(*view);
                index_operands(indices, &mut reads);
                reads.extend(strides.iter().copied());
                reads.extend(*multicast);
            }
            agen::Op::SymbolicVectorStore {
                value,
                view,
                indices,
                strides,
                ..
            } => {
                reads.extend([*value, *view]);
                index_operands(indices, &mut reads);
                reads.extend(strides.iter().copied());
            }
            // ⛔ `load_iv` IS THE REGION'S ARGUMENT, not an operand — see [`block_args`].
            agen::Op::CompositeLoadAndStore(transfer) => {
                reads.push(transfer.src);
                index_operands(&transfer.src_indices, &mut reads);
                reads.push(transfer.dst);
                index_operands(&transfer.dst_indices, &mut reads);
            }
            // ⛔ THE INTERLEAVE HAS NO OPERANDS AT ALL — granularity is an attribute and the
            // transfers it splits are in its region (`Agen.td:1030-1031`).
            agen::Op::CompositeMemoryInterleave { .. } => {}
            // ⭐ ONE OPERAND, THE MASK VALUE. The slice map and the patterns are attributes.
            agen::Op::SetTransferMaskState { mask_value, .. } => reads.push(*mask_value),
        },
        // ⭐ `$base` AND `$indices`, WHICH IS ALL A PLAIN ACCESS HAS. Same operand list as the
        // `agen` pair above; what it lacks is the two attributes, not the operands.
        Op::Vector(op) => match op {
            vector::Op::Load { base, indices, .. } => {
                reads.push(*base);
                index_operands(indices, &mut reads);
            }
            vector::Op::Store {
                value,
                base,
                indices,
                ..
            } => {
                reads.extend([*value, *base]);
                index_operands(indices, &mut reads);
            }
        },
        Op::VectorChain(op) => match op {
            vectorchain::Op::ConstantBitstream { .. }
            | vectorchain::Op::CreateAffineMask { .. } => {}
            // ⛔ THE MASK PARAMETER IS AN OPERAND, AND ENTRY 089 IS THE READER THAT PROVES IT: it
            // reaches the parameter's DEFINING op (`arith.constant` or `arith.subi`) to decide
            // whether the mask is static (`Helper.cpp:64-79`, `:122-131`). A use-walk that missed it
            // would call the value dead.
            vectorchain::Op::CreateAffineMaskSet { mask_parameter, .. } => {
                reads.extend(mask_parameter.iter().copied());
            }
            vectorchain::Op::Estimate { input, .. }
            | vectorchain::Op::FastExp { input, .. }
            | vectorchain::Op::Floor { input, .. }
            | vectorchain::Op::ScanWithGap { input, .. }
            | vectorchain::Op::Select { input, .. }
            | vectorchain::Op::Shuffle { input, .. }
            | vectorchain::Op::Cast { input, .. } => reads.push(*input),
            vectorchain::Op::Rotate {
                input, position, ..
            } => reads.extend([*input, *position]),
            vectorchain::Op::Multiply { a, b, .. } => reads.extend([*a, *b]),
            vectorchain::Op::MultiplyAccumulate { a, b, acc, .. } => reads.extend([*a, *b, *acc]),
            vectorchain::Op::ElementWiseCompare { op1, op2, mask, .. } => {
                reads.extend([*op1, *op2]);
                reads.extend(mask.map(|m| m.val()));
            }
            vectorchain::Op::ElementWiseSelection {
                cond,
                lhs,
                rhs,
                mask,
                ..
            } => {
                reads.extend([cond.val(), *lhs, *rhs]);
                reads.extend(mask.map(|m| m.val()));
            }
            vectorchain::Op::Binary { op1, op2, mask, .. }
            | vectorchain::Op::Pack {
                op1, op2, mask, ..
            } => {
                reads.extend([*op1, *op2]);
                reads.extend(mask.map(|m| m.val()));
            }
            // ⭐ THE ONE UNARY OP WITH A MASK — see [`vectorchain::Op::Neg`].
            vectorchain::Op::Neg { input, mask, .. } => {
                reads.push(*input);
                reads.extend(mask.map(|m| m.val()));
            }
            // ⛔ NO MASK — a merge states its two sides as iteration spaces, and those are
            // attributes and not operands (`VectorChain.td:164-185`).
            vectorchain::Op::Merge { op1, op2, .. } => reads.extend([*op1, *op2]),
        },
    }
    reads
}

/// THE VALUES AN OP **DEFINES AS RESULTS** — what `getDefiningOp` answers this op for.
///
/// ⛔ RESULTS ONLY. A region argument is defined by no op at all, and MLIR's `getDefiningOp()`
/// returns null for one — a distinction `getLoadConsumer` depends on, since a send whose `to` is a
/// block argument yields the null second half of its pair (`Helper.cpp:1266`). Those are
/// [`block_args`].
#[must_use]
pub fn results(op: &Op) -> Vec<Val> {
    match op {
        Op::Arith(op) => match op {
            arith::Op::Constant { result, .. }
            | arith::Op::ConstantInt { result, .. }
            | arith::Op::Compare { result, .. }
            | arith::Op::Select { result, .. }
            | arith::Op::Logic { result, .. }
            | arith::Op::DenseConstant { result, .. } => vec![*result],
            arith::Op::AddI(bin)
            | arith::Op::SubI(bin)
            | arith::Op::MulI(bin)
            | arith::Op::DivSI(bin)
            | arith::Op::RemSI(bin) => vec![bin.result],
        },
        // ⛔ A `symbol.create_symbol` BINDS ITS EXTENT, and entry 091 reads it: the backward walk from
        // a lowered loop's bound ends at either an `arith.constant` or this op
        // (`LoweringXRF.cpp:278-288`), which it can only do if this walk answers for it.
        Op::Symbol(symbol::Op::CreateSymbol { result, .. }) => vec![*result],
        // ⭐ A LOAD BINDS ITS VECTOR AND A STORE BINDS NOTHING — `results = (outs
        // AnyVectorOfAnyRank:$result)` on `Vector_LoadOp`, and no `results` block at all on
        // `Vector_StoreOp`.
        Op::Vector(vector::Op::Load { result, .. }) => vec![*result],
        Op::Vector(vector::Op::Store { .. }) => Vec::new(),
        // ⛔ A `uniform.uniformize_regions` MAY BIND RESULTS, and 47 of the 352 under `dcc/test` do —
        // see [`uniform::Op::UniformizeRegions::results`]. A census that answered "none" would make
        // the op's own verifier, which ties the count to every region's terminator
        // (`Uniform.cpp:170-179`), a statement about nothing.
        Op::Uniform(uniform::Op::UniformizeRegions { results, .. }) => results.clone(),
        Op::Uniform(
            uniform::Op::DefImmutableMapping { result, .. } | uniform::Op::QueryMap { result, .. },
        ) => vec![*result],
        // ⭐ A TERMINATOR BINDS NOTHING; the values it carries out are its PARENT's results.
        Op::Uniform(uniform::Op::Yield { .. }) => Vec::new(),
        Op::Scf(op) => match op {
            // ⭐ A CARRYING `scf.for` BINDS RESULTS, one per `iter_args` entry — the reference's own
            // input to `TransformLoopToLegalizeForSentientLowering` is
            // `%11 = scf.for %arg4 = %c0 to %10 step %c1 iter_args(%arg5 = %arg3) -> (index)`
            // (`scf_loop_with_result.mlir:32`), and `transformSCFToAffineLoop` branches on
            // `scf_for.getNumResults() > 0` to decide whether to write a yield at all (`:129`). A
            // census that answered "none" would make that branch unreachable.
            scf::Op::For { carried, .. } => carried.iter().map(|carried| carried.result).collect(),
            // ⭐ AND AN `scf.if` BINDS WHAT ITS ARMS YIELD — `%13 = scf.if %12 -> (index)`
            // (`dcc/test/Transform/CFGSimplificationDataflowLevel/merging.mlir:129`). ⛔ THREE
            // DECISIONS OF THE SHALLOW MERGE READ THIS LIST'S LENGTH: which of two candidates becomes
            // the destination (`CFGSDataflowConditionalTree.cpp:237`), which region's terminator
            // survives the splice (`:420`, `:440`), and whether the pair is mergeable at all
            // (`:348`).
            scf::Op::If { results, .. } => results.clone(),
            // ⛔ `scf.parallel`'s results would be the values its reductions carry, and nothing this
            // crate emits reads one.
            scf::Op::Yield { .. } | scf::Op::Parallel { .. } => Vec::new(),
        },
        Op::Affine(op) => match op {
            // ⭐ A CARRYING LOOP DOES BIND RESULTS — one per `iter_args` entry, which is how the
            // address a nest computes leaves it (`AgenToSentient.hpp:502-520`). A plain counted
            // loop carries nothing and binds nothing.
            affine::Op::For { carried, .. } => {
                carried.iter().map(|carried| carried.result).collect()
            }
            affine::Op::Yield { .. } | affine::Op::VectorStore { .. } => Vec::new(),
            // ⭐ AND AN `affine.if` BINDS WHAT IT YIELDS — `%52 = affine.if #set0(%arg9) -> index`
            // (`dcc/test/PT/issue-236.mlir:59`), which the `arith.cmpi` on the next line reads.
            affine::Op::If { results, .. } => results.clone(),
            affine::Op::Apply { result, .. } | affine::Op::VectorLoad { result, .. } => {
                vec![*result]
            }
        },
        Op::Dataflow(op) => match op {
            dataflow::Op::GetUnit { result, .. }
            | dataflow::Op::GetLocalUnit { result, .. }
            | dataflow::Op::CreateGroup { result, .. }
            | dataflow::Op::GetLogicalMemoryView { result, .. }
            | dataflow::Op::Receive { result, .. } => vec![*result],
            dataflow::Op::GetPagedLogicalMemoryView(view) => vec![view.result],
            // ⛔ `dataflow.send` HAS NO RESULT (`Dataflow.td`), which is why `getLoadConsumer`
            // returns the send op itself rather than a value.
            dataflow::Op::ProgramUnit { .. }
            | dataflow::Op::Send { .. }
            | dataflow::Op::SyncSend { .. }
            | dataflow::Op::SyncRecv { .. }
            | dataflow::Op::ImplicitSync { .. }
            | dataflow::Op::Opaque(_) => Vec::new(),
        },
        Op::Agen(op) => match op {
            agen::Op::VectorLoad { result, .. }
            | agen::Op::SymbolicVectorLoad { result, .. }
            | agen::Op::SetTransferMaskState { result, .. } => {
                vec![*result]
            }
            agen::Op::VectorStore { .. }
            | agen::Op::SymbolicVectorStore { .. }
            | agen::Op::Yield
            | agen::Op::CompositeMemoryInterleave { .. }
            | agen::Op::CompositeLoadAndStore(_) => Vec::new(),
        },
        Op::VectorChain(op) => match op {
            vectorchain::Op::Estimate { result, .. }
            | vectorchain::Op::FastExp { result, .. }
            | vectorchain::Op::Floor { result, .. }
            | vectorchain::Op::Neg { result, .. }
            | vectorchain::Op::ScanWithGap { result, .. }
            | vectorchain::Op::Select { result, .. }
            | vectorchain::Op::Multiply { result, .. }
            | vectorchain::Op::MultiplyAccumulate { result, .. }
            | vectorchain::Op::ElementWiseCompare { result, .. }
            | vectorchain::Op::ElementWiseSelection { result, .. }
            | vectorchain::Op::Binary { result, .. }
            | vectorchain::Op::ConstantBitstream { result, .. }
            | vectorchain::Op::Shuffle { result, .. }
            | vectorchain::Op::Rotate { result, .. }
            | vectorchain::Op::Cast { result, .. }
            | vectorchain::Op::Pack { result, .. }
            | vectorchain::Op::Merge { result, .. }
            | vectorchain::Op::CreateAffineMask { result, .. }
            | vectorchain::Op::CreateAffineMaskSet { result, .. } => vec![*result],
        },
    }
}

/// EVERY VALUE ONE OP **READS**, AS A PLACE A REWRITE MAY WRITE — MLIR's `getOpOperands()`.
///
/// # ⛔⛔ THE MIRROR OF [`operands`], AND THE TWO MUST STAY IN STEP
///
/// It exists for [`replace_uses_of_with`], which is what cloning a use chain needs
/// (`Agen.cpp:143-145`): every op after the first reads the previous op's result, and the clone has
/// to read the previous CLONE's result instead. Total over the enum with no wildcard, for the reason
/// [`operands`] gives.
///
/// # ⛔⛔ TWO OPERANDS ARE DELIBERATELY ABSENT, AND NEITHER IS AN OVERSIGHT
///
/// * A **link end** — [`dataflow::Op::Send`]'s `to` and [`dataflow::Op::Receive`]'s `from`. A
///   [`crate::islands::dataflow_ir::link::Link`] hands its two ends out once, by consuming itself, so
///   one wire is one send and one receive; a re-pointed end would name a unit no receive is paired
///   with. A send's DATA is here, and the data is what a chain clone substitutes.
/// * A **mask or condition vector** — [`vectorchain::Predicate`], which carries the type the value
///   was DEFINED at and never recomputes it at the use. Writing the value without the type would
///   keep the old width on the new value.
///
/// So this is every operand a rewrite may re-point, and the two it may not are the two whose pairing
/// with something else would be broken by re-pointing them alone.
#[must_use]
pub fn operands_mut(op: &mut Op) -> Vec<&mut Val> {
    let mut places: Vec<&mut Val> = Vec::new();
    match op {
        Op::Arith(op) => match op {
            arith::Op::Constant { .. }
            | arith::Op::ConstantInt { .. }
            | arith::Op::DenseConstant { .. } => {}
            arith::Op::AddI(bin)
            | arith::Op::SubI(bin)
            | arith::Op::MulI(bin)
            | arith::Op::DivSI(bin)
            | arith::Op::RemSI(bin) => {
                places.extend([&mut bin.lhs, &mut bin.rhs]);
            }
            arith::Op::Compare { lhs, rhs, .. } => places.extend([lhs, rhs]),
            arith::Op::Select {
                condition,
                true_value,
                false_value,
                ..
            } => places.extend([condition, true_value, false_value]),
            arith::Op::Logic { operands, .. } => places.extend(operands.iter_mut()),
        },
        Op::Scf(op) => match op {
            scf::Op::Parallel { ivs: _, body: _ } => {}
            // ⭐ THE THREE BOUNDS ARE PLAIN VALUES, WHICH IS WHY THIS ARM IS SHORTER THAN
            // `affine.for`'s: an `scf.for` has no `Bound` to unwrap and no `iter_args` to re-point.
            scf::Op::For {
                iv: _,
                lo,
                hi,
                step,
                carried,
                body: _,
                dbg_name: _,
            } => {
                places.extend([lo, hi, step]);
                places.extend(carried.iter_mut().map(|carried| &mut carried.init));
            }
            scf::Op::If { cond, .. } => places.push(cond),
            scf::Op::Yield { operands } => places.extend(operands.iter_mut()),
        },
        Op::Affine(op) => match op {
            affine::Op::For {
                iv: _,
                lo,
                hi,
                carried,
                body: _,
                dbg_name: _,
            } => {
                for bound in [lo, hi] {
                    if let affine::Bound::Val(val) = bound {
                        places.push(val);
                    }
                }
                places.extend(carried.iter_mut().map(|carried| &mut carried.init));
            }
            affine::Op::Apply { args, .. } => places.extend(args.iter_mut()),
            affine::Op::Yield { operands } => places.extend(operands.iter_mut()),
            // ⭐ BOTH OPERAND LISTS, DIMS THEN SYMBOLS — the order `affine.if #set(%d)[%s]` writes
            // them in, and the order the set's two position spaces number them in.
            affine::Op::If {
                args, symbol_args, ..
            } => {
                places.extend(args.iter_mut());
                places.extend(symbol_args.iter_mut());
            }
            affine::Op::VectorLoad { view, indices, .. } => {
                places.push(view);
                index_operands_mut(indices, &mut places);
            }
            affine::Op::VectorStore {
                value,
                view,
                indices,
                ..
            } => {
                places.extend([value, view]);
                index_operands_mut(indices, &mut places);
            }
        },
        Op::Dataflow(op) => match op {
            dataflow::Op::GetUnit { .. } | dataflow::Op::Opaque { .. } => {}
            dataflow::Op::GetLocalUnit { of, .. } => places.push(of),
            dataflow::Op::CreateGroup { unit_ids, .. } => places.extend(unit_ids.iter_mut()),
            dataflow::Op::GetLogicalMemoryView { from, start, .. } => places.extend([from, start]),
            dataflow::Op::GetPagedLogicalMemoryView(view) => {
                places.extend([&mut view.unit, &mut view.start_addr]);
                places.extend(view.pages.iter_mut().map(|page| &mut page.start_addr));
            }
            dataflow::Op::ProgramUnit { units, .. } => places.extend(units.iter_mut()),
            // ⛔ `to` IS A LINK END — see the exclusions above. The DATA is the operand a rewrite
            // re-points, and it is the one a use-chain clone substitutes.
            dataflow::Op::Send { to: _, data, .. } => places.push(data),
            dataflow::Op::Receive { from: _, .. } => {}
            dataflow::Op::SyncSend { to, .. } => places.push(to),
            dataflow::Op::SyncRecv { from, .. } => places.push(from),
            dataflow::Op::ImplicitSync {
                view, dst, size, ..
            } => places.extend([view, dst, size]),
        },
        Op::Agen(op) => match op {
            agen::Op::Yield => {}
            agen::Op::VectorLoad { view, indices, .. } => {
                places.push(view);
                index_operands_mut(indices, &mut places);
            }
            agen::Op::VectorStore {
                value,
                view,
                indices,
                ..
            } => {
                places.extend([value, view]);
                index_operands_mut(indices, &mut places);
            }
            agen::Op::SymbolicVectorLoad {
                view,
                indices,
                strides,
                multicast,
                ..
            } => {
                places.push(view);
                index_operands_mut(indices, &mut places);
                places.extend(strides.iter_mut());
                places.extend(multicast.as_mut());
            }
            agen::Op::SymbolicVectorStore {
                value,
                view,
                indices,
                strides,
                ..
            } => {
                places.extend([value, view]);
                index_operands_mut(indices, &mut places);
                places.extend(strides.iter_mut());
            }
            agen::Op::CompositeLoadAndStore(transfer) => {
                places.push(&mut transfer.src);
                index_operands_mut(&mut transfer.src_indices, &mut places);
                places.push(&mut transfer.dst);
                index_operands_mut(&mut transfer.dst_indices, &mut places);
            }
            // ⛔ ARM FOR ARM WITH [`operands`]: the interleave names no value, the mask state names one.
            agen::Op::CompositeMemoryInterleave { .. } => {}
            agen::Op::SetTransferMaskState { mask_value, .. } => places.push(mask_value),
        },
        // ⭐ ARM FOR ARM WITH [`operands`] — see the note there.
        Op::Vector(op) => match op {
            vector::Op::Load { base, indices, .. } => {
                places.push(base);
                index_operands_mut(indices, &mut places);
            }
            vector::Op::Store {
                value,
                base,
                indices,
                ..
            } => {
                places.extend([value, base]);
                index_operands_mut(indices, &mut places);
            }
        },
        Op::VectorChain(op) => match op {
            vectorchain::Op::ConstantBitstream { .. }
            | vectorchain::Op::CreateAffineMask { .. } => {}
            // ⭐ THE MASK PARAMETER IS A PLAIN OPERAND, NOT A [`vectorchain::Predicate`]: it is an
            // `index` substituted for the set's `s0`, so re-pointing it leaves no stale width behind.
            vectorchain::Op::CreateAffineMaskSet { mask_parameter, .. } => {
                places.extend(mask_parameter.iter_mut());
            }
            vectorchain::Op::Estimate { input, .. }
            | vectorchain::Op::FastExp { input, .. }
            | vectorchain::Op::Floor { input, .. }
            // ⛔ ITS MASK IS ABSENT HERE FOR THE SAME REASON THE ELEMENTWISE FAMILY'S IS — see the
            // exclusions above.
            | vectorchain::Op::Neg { input, .. }
            | vectorchain::Op::ScanWithGap { input, .. }
            | vectorchain::Op::Select { input, .. }
            | vectorchain::Op::Shuffle { input, .. }
            | vectorchain::Op::Cast { input, .. } => places.push(input),
            vectorchain::Op::Rotate {
                input, position, ..
            } => places.extend([input, position]),
            vectorchain::Op::Multiply { a, b, .. } => places.extend([a, b]),
            vectorchain::Op::MultiplyAccumulate { a, b, acc, .. } => places.extend([a, b, acc]),
            // ⛔ THE MASK AND THE CONDITION VECTOR ARE ABSENT — see the exclusions above.
            vectorchain::Op::ElementWiseCompare { op1, op2, .. } => places.extend([op1, op2]),
            vectorchain::Op::ElementWiseSelection { lhs, rhs, .. } => places.extend([lhs, rhs]),
            vectorchain::Op::Binary { op1, op2, .. }
            | vectorchain::Op::Pack { op1, op2, .. }
            | vectorchain::Op::Merge { op1, op2, .. } => places.extend([op1, op2]),
        },
        // ⛔ A SYMBOL READS NOTHING. `symbol.create_symbol` names an extent that is not yet a value;
        // entry 091's backward walk stops AT it, never through it.
        Op::Symbol(symbol::Op::CreateSymbol { .. }) => {}
        // ⭐ ARM FOR ARM WITH [`operands`] — the units flattened in region order, and both halves of
        // every mapping pair.
        Op::Uniform(op) => match op {
            uniform::Op::UniformizeRegions { regions, .. } => {
                for region in regions {
                    places.extend(region.units.iter_mut());
                }
            }
            uniform::Op::Yield { operands } => places.extend(operands.iter_mut()),
            uniform::Op::DefImmutableMapping { pairs, .. } => {
                for (key, value) in pairs {
                    places.extend([key, value]);
                }
            }
            uniform::Op::QueryMap { map, key, .. } => places.extend([map, key]),
        },
    }
    places
}

/// THE VALUES AN OP **DEFINES AS RESULTS**, AS PLACES — the mirror of [`results`].
///
/// It exists for [`clone_with_fresh_results`]: a cloned op binds its own values, never the ones the
/// original bound.
#[must_use]
pub fn results_mut(op: &mut Op) -> Vec<&mut Val> {
    match op {
        Op::Arith(op) => match op {
            arith::Op::Constant { result, .. }
            | arith::Op::ConstantInt { result, .. }
            | arith::Op::Compare { result, .. }
            | arith::Op::Select { result, .. }
            | arith::Op::Logic { result, .. }
            | arith::Op::DenseConstant { result, .. } => vec![result],
            arith::Op::AddI(bin)
            | arith::Op::SubI(bin)
            | arith::Op::MulI(bin)
            | arith::Op::DivSI(bin)
            | arith::Op::RemSI(bin) => {
                vec![&mut bin.result]
            }
        },
        Op::Symbol(symbol::Op::CreateSymbol { result, .. }) => vec![result],
        Op::Vector(vector::Op::Load { result, .. }) => vec![result],
        Op::Vector(vector::Op::Store { .. }) => Vec::new(),
        Op::Uniform(uniform::Op::UniformizeRegions { results, .. }) => results.iter_mut().collect(),
        Op::Uniform(
            uniform::Op::DefImmutableMapping { result, .. } | uniform::Op::QueryMap { result, .. },
        ) => vec![result],
        Op::Uniform(uniform::Op::Yield { .. }) => Vec::new(),
        // ⛔ THIS WAS `Op::Scf(_) => Vec::new()`, AND THAT DISAGREED WITH [`results`]. The read side
        // has answered the carried results of an `scf.for` since the variant landed; the write side
        // did not, so a clone that reminted an op's results left a carrying loop binding the
        // ORIGINAL's names — which is exactly what [`crate::islands::dataflow_ir::Values::
        // clone_without_regions`] asks for. The two are now arm for arm.
        Op::Scf(op) => match op {
            scf::Op::For { carried, .. } => carried
                .iter_mut()
                .map(|carried| &mut carried.result)
                .collect(),
            // ⛔ AND THE `scf.if` LIST JOINED BOTH WHEN THE SHALLOW MERGE NEEDED TO READ ITS LENGTH.
            scf::Op::If { results, .. } => results.iter_mut().collect(),
            scf::Op::Yield { .. } | scf::Op::Parallel { .. } => Vec::new(),
        },
        Op::Affine(op) => match op {
            affine::Op::For { carried, .. } => carried
                .iter_mut()
                .map(|carried| &mut carried.result)
                .collect(),
            // ⛔ `affine.if` BINDS A RESULT LIST, AND ENTRY 096 TURNS ON IT BEING EMPTY: a
            // value-yielding conditional is not a candidate for a dummy `else`
            // (`CFGSDataflowConditionalTree.cpp:383-386`), so the list is a fact this walk must state.
            affine::Op::If { results, .. } => results.iter_mut().collect(),
            affine::Op::Yield { .. } | affine::Op::VectorStore { .. } => Vec::new(),
            affine::Op::Apply { result, .. } | affine::Op::VectorLoad { result, .. } => {
                vec![result]
            }
        },
        Op::Dataflow(op) => match op {
            dataflow::Op::GetUnit { result, .. }
            | dataflow::Op::GetLocalUnit { result, .. }
            | dataflow::Op::CreateGroup { result, .. }
            | dataflow::Op::GetLogicalMemoryView { result, .. }
            | dataflow::Op::Receive { result, .. } => vec![result],
            dataflow::Op::GetPagedLogicalMemoryView(view) => vec![&mut view.result],
            dataflow::Op::ProgramUnit { .. }
            | dataflow::Op::Send { .. }
            | dataflow::Op::SyncSend { .. }
            | dataflow::Op::SyncRecv { .. }
            | dataflow::Op::ImplicitSync { .. }
            | dataflow::Op::Opaque { .. } => Vec::new(),
        },
        Op::Agen(op) => match op {
            agen::Op::VectorLoad { result, .. }
            | agen::Op::SymbolicVectorLoad { result, .. }
            | agen::Op::SetTransferMaskState { result, .. } => {
                vec![result]
            }
            agen::Op::VectorStore { .. }
            | agen::Op::SymbolicVectorStore { .. }
            | agen::Op::Yield
            | agen::Op::CompositeMemoryInterleave { .. }
            | agen::Op::CompositeLoadAndStore(_) => Vec::new(),
        },
        Op::VectorChain(op) => match op {
            vectorchain::Op::CreateAffineMaskSet { result, .. } => vec![result],
            vectorchain::Op::Estimate { result, .. }
            | vectorchain::Op::FastExp { result, .. }
            | vectorchain::Op::Floor { result, .. }
            | vectorchain::Op::Neg { result, .. }
            | vectorchain::Op::ScanWithGap { result, .. }
            | vectorchain::Op::Select { result, .. }
            | vectorchain::Op::Multiply { result, .. }
            | vectorchain::Op::MultiplyAccumulate { result, .. }
            | vectorchain::Op::ElementWiseCompare { result, .. }
            | vectorchain::Op::ElementWiseSelection { result, .. }
            | vectorchain::Op::Binary { result, .. }
            | vectorchain::Op::ConstantBitstream { result, .. }
            | vectorchain::Op::Shuffle { result, .. }
            | vectorchain::Op::Rotate { result, .. }
            | vectorchain::Op::Cast { result, .. }
            | vectorchain::Op::Pack { result, .. }
            | vectorchain::Op::Merge { result, .. }
            | vectorchain::Op::CreateAffineMask { result, .. } => vec![result],
        },
    }
}

/// RE-POINT EVERY USE OF `from` AT `to` — `Operation::replaceUsesOfWith`.
///
/// ⭐ ONE ENTRY PER USE, so an op that reads a value twice has both re-pointed, exactly as MLIR walks
/// its own operand list. See [`operands_mut`] for the two operands it does not reach and why.
pub fn replace_uses_of_with(op: &mut Op, from: Val, to: Val) {
    for place in operands_mut(op) {
        if *place == from {
            *place = to;
        }
    }
}

/// A COPY OF ONE OP BINDING ITS OWN VALUES — `OpBuilder::clone`.
///
/// The operands are the original's; every result is freshly minted, because two ops binding one
/// value is not a diagnosis anyone enjoys making from MLIR's error (see
/// [`crate::islands::dataflow_ir::Values`]).
///
/// # ⛔⛔ IT DOES NOT REMAP A REGION, AND THE CALLERS HAVE NONE
///
/// MLIR's `clone` deep-copies an op's regions and gives the copy's blocks their own arguments. An op
/// whose region binds values would therefore need those reminted and every use inside the region
/// re-pointed, which is not written here. Both callers are region-free by construction: entry 125
/// clones a `dataflow.get_logical_memory_view`, and entry 127 clones a use chain, whose members are
/// single-result computes and a terminating send or store (`Agen.cpp:114-134`). [`regions`] is what
/// says which ops those are.
#[must_use]
pub fn clone_with_fresh_results(op: &Op, values: &mut Values) -> Op {
    let mut clone = op.clone();
    for result in results_mut(&mut clone) {
        *result = values.mint();
    }
    clone
}

/// THE VALUES AN OP'S REGIONS BIND — arguments, which no op defines.
#[must_use]
pub fn block_args(op: &Op) -> Vec<Val> {
    match op {
        // ⭐ THE REGION BINDS THE INDUCTION VARIABLE FIRST, THEN ONE ARGUMENT PER CARRIED VALUE —
        // the order `affine.for`'s body block declares them in.
        Op::Affine(affine::Op::For { iv, carried, .. }) => {
            let mut args = vec![*iv];
            args.extend(carried.iter().map(|carried| carried.arg));
            args
        }
        Op::Scf(scf::Op::Parallel { ivs, .. }) => ivs.clone(),
        // ⭐ SAME ORDER AS `affine.for`: the induction variable first, then one argument per carried
        // value. `transformSCFToAffineLoop` maps them positionally
        // (`TransformLoopToLegalizeForSentientLowering.cpp:115-119`), so the two orders must agree.
        Op::Scf(scf::Op::For { iv, carried, .. }) => {
            let mut args = vec![*iv];
            args.extend(carried.iter().map(|carried| carried.arg));
            args
        }
        // ⭐ THE COMPOSITE TRANSFER'S `load_iv` IS ITS REGION'S ARGUMENT — the loaded vector the
        // body reads. `getLoadInductionVar()` is what `getLoadConsumer` roots a composite load's
        // consumer chain at (`Helper.cpp:1250`).
        Op::Agen(agen::Op::CompositeLoadAndStore(transfer)) => vec![transfer.load_iv],
        // ⛔⛔ ONE ARGUMENT PER REGION, AND EACH REGION BINDS ITS OWN.
        // `getRegionArg(i) { return getRegion(i).getArgument(0); }` (`Uniform.td:96`) — so a
        // two-region op binds two values, and entry 182 maps the one belonging to the region it is
        // cloning out of (`FlatteningLocalRegions.cpp:246-256`). A census that missed them would let
        // a clone keep reading the ORIGINAL region's argument.
        Op::Uniform(uniform::Op::UniformizeRegions { regions, .. }) => {
            regions.iter().map(|region| region.arg).collect()
        }
        Op::Uniform(
            uniform::Op::Yield { .. }
            | uniform::Op::DefImmutableMapping { .. }
            | uniform::Op::QueryMap { .. },
        ) => Vec::new(),
        Op::Arith(_)
        | Op::Affine(_)
        | Op::Scf(_)
        | Op::Dataflow(_)
        | Op::Agen(_)
        | Op::Vector(_)
        | Op::VectorChain(_)
        | Op::Symbol(_) => Vec::new(),
    }
}

/// THE OPS AN OP'S REGIONS HOLD, in the order they are written.
#[must_use]
pub fn regions(op: &Op) -> Vec<&[Op]> {
    match op {
        Op::Affine(affine::Op::For { body, .. })
        | Op::Scf(scf::Op::Parallel { body, .. } | scf::Op::For { body, .. }) => {
            vec![body.as_slice()]
        }
        // ⭐ TWO REGIONS EACH, `then` FIRST — the order `getRegions()[0]` / `getRegions()[1]`
        // indexes them in, which is what `createDummyYieldInElseReg` relies on
        // (`CFGSDataflowConditionalTree.cpp:386-388`).
        Op::Scf(scf::Op::If {
            body, else_body, ..
        })
        | Op::Affine(affine::Op::If {
            body, else_body, ..
        }) => vec![body.as_slice(), else_body.as_slice()],
        Op::Dataflow(dataflow::Op::ProgramUnit { body, .. }) => vec![body.as_slice()],
        Op::Agen(agen::Op::CompositeLoadAndStore(transfer)) => vec![transfer.body.as_slice()],
        Op::Agen(agen::Op::CompositeMemoryInterleave { body, .. }) => vec![body.as_slice()],
        // ⛔⛔ AS MANY REGIONS AS IT HAS UNIT LISTS — `VariadicRegion<AnyRegion>:$regions`
        // (`Uniform.td:91`), one per [`uniform::LocalRegion`]. This count IS the pass's decision:
        // `flatten` declines when `op_.getNumRegions() == num_of_regions`
        // (`FlatteningLocalRegions.cpp:418`).
        Op::Uniform(uniform::Op::UniformizeRegions { regions, .. }) => {
            regions.iter().map(|region| region.body.as_slice()).collect()
        }
        Op::Uniform(
            uniform::Op::Yield { .. }
            | uniform::Op::DefImmutableMapping { .. }
            | uniform::Op::QueryMap { .. },
        ) => Vec::new(),
        Op::Arith(_)
        | Op::Affine(_)
        | Op::Scf(_)
        | Op::Dataflow(_)
        | Op::Agen(_)
        | Op::Vector(_)
        | Op::VectorChain(_)
        | Op::Symbol(_) => Vec::new(),
    }
}

/// THE OPS AN OP'S REGIONS HOLD, MUTABLY — [`regions`] arm for arm.
///
/// ⛔⛔ THE TWO MUST STAY IN STEP, WHICH IS WHY THEY SIT TOGETHER AND MATCH IN THE SAME ORDER. A
/// reader that descends with [`regions`] and a writer that descends with this one would disagree
/// about where an op is the moment one of them gained an arm the other lacks — and a *position* is
/// how `e075_eraseOp` names the op it removes.
///
/// ⭐ `&mut Vec<Op>` AND NOT `&mut [Op]`: the one caller removes elements, which is a `Vec`
/// operation. The read side hands out slices because nothing reading needs the length to change.
#[must_use]
pub fn regions_mut(op: &mut Op) -> Vec<&mut Vec<Op>> {
    match op {
        Op::Affine(affine::Op::For { body, .. })
        | Op::Scf(scf::Op::Parallel { body, .. } | scf::Op::For { body, .. }) => {
            vec![body]
        }
        // ⭐ TWO REGIONS EACH, `then` FIRST — arm for arm with [`regions`], which is the order entry
        // 096 indexes when it pushes a terminator into the second.
        Op::Scf(scf::Op::If {
            body, else_body, ..
        })
        | Op::Affine(affine::Op::If {
            body, else_body, ..
        }) => vec![body, else_body],
        Op::Dataflow(dataflow::Op::ProgramUnit { body, .. }) => vec![body],
        Op::Agen(agen::Op::CompositeLoadAndStore(transfer)) => vec![&mut transfer.body],
        Op::Agen(agen::Op::CompositeMemoryInterleave { body, .. }) => vec![body],
        // ⭐ ARM FOR ARM WITH [`regions`], which is what entry 182's per-region recursion indexes.
        Op::Uniform(uniform::Op::UniformizeRegions { regions, .. }) => {
            regions.iter_mut().map(|region| &mut region.body).collect()
        }
        Op::Uniform(
            uniform::Op::Yield { .. }
            | uniform::Op::DefImmutableMapping { .. }
            | uniform::Op::QueryMap { .. },
        ) => Vec::new(),
        Op::Arith(_)
        | Op::Affine(_)
        | Op::Scf(_)
        | Op::Dataflow(_)
        | Op::Agen(_)
        | Op::Vector(_)
        | Op::VectorChain(_)
        | Op::Symbol(_) => Vec::new(),
    }
}

/// THE OP'S `dbgName`, or `None` when it carries none — `dataflow::getDbgNameAttr`.
///
/// ```cpp
/// auto mlir::dataflow::getDbgNameAttr(Operation* op) -> StringAttr {
///   if (auto iface = dyn_cast<DebugNameOpInterface>(op); iface) {
///     return iface.getDbgNameAttr();
///   }
///   if (const auto result =
///           op->getAttrOfType<StringAttr>(DebugNameOpInterface::kDbgNameAttrName);
///       result) {
///     return result;
///   }
///   return nullptr;
/// }
/// ```
/// (`DataflowOpInterfaces.cpp:24-36`, with `kDbgNameAttrName = "dbgName"` at
/// `DataflowInterfaces.td:68`)
///
/// ⭐ THE TWO PATHS COLLAPSE TO ONE FIELD HERE. An op of IBM's own dialect implements the interface
/// and holds the name in a property; an upstream `scf.if` does not and holds it as a discardable
/// attribute — and the reference's getter erases that difference, which is why one `Option<String>`
/// field per op that can carry one is the whole of it.
///
/// ⛔ `None` ALSO MEANS "COULD NOT CARRY ONE", exactly as the reference's `nullptr` does: an op with
/// no such attribute and an op whose attribute is absent are the same answer.
#[must_use]
pub fn dbg_name(op: &Op) -> Option<&str> {
    match op {
        Op::Scf(scf::Op::For { dbg_name, .. } | scf::Op::If { dbg_name, .. })
        | Op::Affine(affine::Op::For { dbg_name, .. } | affine::Op::If { dbg_name, .. })
        | Op::Dataflow(dataflow::Op::Opaque(dataflow::Opaque { dbg_name, .. })) => {
            dbg_name.as_deref()
        }
        // ── the ops of this island that carry no name at all ─────────────────────────────────────
        Op::Scf(scf::Op::Yield { .. } | scf::Op::Parallel { .. })
        | Op::Affine(
            affine::Op::Apply { .. }
            | affine::Op::Yield { .. }
            | affine::Op::VectorLoad { .. }
            | affine::Op::VectorStore { .. },
        )
        | Op::Arith(_)
        | Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        // ⛔ AND `vector` HAS NO FIELD TO CARRY ONE. Upstream's `vector.load`/`vector.store` do print
        // an `attr-dict`, so MLIR would let a `dbgName` ride on either — but
        // [`super::vector::Op`] represents neither that dict nor any optional attribute, for the
        // reason recorded there, and none of the nine occurrences in the authority tree carries one.
        | Op::Vector(_)
        | Op::Symbol(_)
        // ⚠️ NO `uniform` OP CARRIES ONE ANYWHERE IN THE AUTHORITY'S FIXTURES. Its printer would
        // show it — `printOptionalAttrDict(op->getAttrs(), {"list_sizes", "newly_added"})`
        // (`Uniform.cpp:119-120`) hides only those two — and `dbgName` appears beside a
        // `uniform.` op in none of `dcc/test`'s 352 `uniformize_regions`. The names in this pass's
        // output belong to the `scf.if`s it clones (`flatten_local_region4.mlir:351`, `:357`).
        | Op::Uniform(_) => None,
    }
}

/// THE SLOT THE OP'S `dbgName` LIVES IN, or `None` for an op that cannot hold one —
/// `dataflow::setDbgNameAttr`.
///
/// ```cpp
/// void mlir::dataflow::setDbgNameAttr(Operation* op, StringAttr name) {
///   if (auto iface = dyn_cast<DebugNameOpInterface>(op)) {
///     iface.setDbgNameAttr(name);
///     return;
///   }
///   if (name != nullptr) {
///     op->setAttr(DebugNameOpInterface::kDbgNameAttrName, name);
///     return;
///   }
///   op->removeAttr(DebugNameOpInterface::kDbgNameAttrName);
/// }
/// ```
/// (`DataflowOpInterfaces.cpp:38-50`)
///
/// ⭐ A SLOT AND NOT A SETTER, because the reference's one function both writes and REMOVES: passing
/// `nullptr` erases the attribute. `*slot = None` is that erasure, and `*slot = Some(..)` the write.
///
/// ⛔ AND `None` FROM THIS FUNCTION IS A DIFFERENT ANSWER FROM [`dbg_name`]'s: it says the op has no
/// place to put a name, which for the reference is impossible (any `Operation*` takes a discardable
/// attribute). Nothing in bridge 2 names an op that this island does not give a field to — the only
/// caller is `mergeShallow`'s last statement, whose `dst` is one of the two conditionals.
#[must_use]
pub fn dbg_name_mut(op: &mut Op) -> Option<&mut Option<String>> {
    match op {
        Op::Scf(scf::Op::For { dbg_name, .. } | scf::Op::If { dbg_name, .. })
        | Op::Affine(affine::Op::For { dbg_name, .. } | affine::Op::If { dbg_name, .. })
        | Op::Dataflow(dataflow::Op::Opaque(dataflow::Opaque { dbg_name, .. })) => Some(dbg_name),
        Op::Scf(scf::Op::Yield { .. } | scf::Op::Parallel { .. })
        | Op::Affine(
            affine::Op::Apply { .. }
            | affine::Op::Yield { .. }
            | affine::Op::VectorLoad { .. }
            | affine::Op::VectorStore { .. },
        )
        | Op::Arith(_)
        | Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_)
        // ⛔ `vector` AGAIN, AND HERE THE `None` IS THE STRONGER CLAIM — see [`dbg_name`]: the island
        // gives these two ops no slot, so `mergeShallow` could not name one if it were handed one.
        // It is not: its `dst` is a conditional.
        | Op::Vector(_)
        | Op::Symbol(_)
        | Op::Uniform(_) => None,
    }
}

// ───────────────────────── THE SAME POSITIONS, MUTABLY ──────────────────────────

/// EVERY `Val` POSITION OF ONE OP, MUTABLY, GROUPED BY WHAT THE POSITION IS.
///
/// # ⭐⭐ THIS IS WHAT `IRMapping` + `OpBuilder::clone` NEEDS AND NOTHING ELSE
///
/// `transformSCFToAffineLoop` clones a loop body under a value mapping
/// (`TransformLoopToLegalizeForSentientLowering.cpp:120-127`), and a clone has to do three different
/// things to three different kinds of value: REWRITE what the op reads, MINT what it defines, and
/// MINT what its regions bind. A single `Vec<&mut Val>` could not tell them apart, so the groups are
/// the type.
///
/// ⛔ AUDIT IT AGAINST [`operands`], [`block_args`], [`results`] AND [`regions`] — arm for arm, in
/// the same order. It names exactly the positions those four read; a position this misses is a value
/// a clone leaves pointing into the ORIGINAL op, which is an SSA graph with two definitions of one
/// name and no diagnostic saying so.
pub struct OpPartsMut<'a> {
    /// What the op READS — see [`operands`].
    pub operands: Vec<&'a mut Val>,
    /// What the op's REGIONS BIND — see [`block_args`].
    pub block_args: Vec<&'a mut Val>,
    /// What the op DEFINES — see [`results`].
    pub results: Vec<&'a mut Val>,
    /// The op's REGIONS — see [`regions`].
    pub regions: Vec<&'a mut Vec<Op>>,
}

/// EVERY `Val` POSITION OF ONE OP, MUTABLY. See [`OpPartsMut`].
#[must_use]
pub fn parts_mut(op: &mut Op) -> OpPartsMut<'_> {
    let mut operands: Vec<&mut Val> = Vec::new();
    let mut block_args: Vec<&mut Val> = Vec::new();
    let mut results: Vec<&mut Val> = Vec::new();
    let mut regions: Vec<&mut Vec<Op>> = Vec::new();
    match op {
        Op::Arith(op) => match op {
            arith::Op::Constant { result, .. }
            | arith::Op::ConstantInt { result, .. }
            | arith::Op::DenseConstant { result, .. } => results.push(result),
            arith::Op::AddI(bin)
            | arith::Op::SubI(bin)
            | arith::Op::MulI(bin)
            | arith::Op::DivSI(bin)
            | arith::Op::RemSI(bin) => {
                operands.extend([&mut bin.lhs, &mut bin.rhs]);
                results.push(&mut bin.result);
            }
            arith::Op::Compare {
                result, lhs, rhs, ..
            } => {
                operands.extend([lhs, rhs]);
                results.push(result);
            }
            arith::Op::Select {
                result,
                condition,
                true_value,
                false_value,
                ..
            } => {
                operands.extend([condition, true_value, false_value]);
                results.push(result);
            }
            arith::Op::Logic {
                result,
                operands: reads,
                ..
            } => {
                operands.extend(reads.iter_mut());
                results.push(result);
            }
        },
        Op::Scf(op) => match op {
            scf::Op::Parallel { ivs, body } => {
                block_args.extend(ivs.iter_mut());
                regions.push(body);
            }
            scf::Op::For {
                iv,
                lo,
                hi,
                step,
                carried,
                body,
                // ⛔ NOT A VALUE. `dbgName` is a string attribute the pass COPIES verbatim
                // (`:110-111`), so a clone leaves it exactly as it found it.
                dbg_name: _,
            } => {
                operands.extend([lo, hi, step]);
                block_args.push(iv);
                for carried in carried.iter_mut() {
                    operands.push(&mut carried.init);
                    block_args.push(&mut carried.arg);
                    results.push(&mut carried.result);
                }
                regions.push(body);
            }
            scf::Op::If {
                cond,
                results: binds,
                body,
                else_body,
                // ⛔ NOT A VALUE, exactly as on the loop above.
                dbg_name: _,
            } => {
                operands.push(cond);
                results.extend(binds.iter_mut());
                regions.extend([body, else_body]);
            }
            scf::Op::Yield { operands: reads } => operands.extend(reads.iter_mut()),
        },
        Op::Affine(op) => match op {
            affine::Op::For {
                iv,
                lo,
                hi,
                carried,
                body,
                dbg_name: _,
            } => {
                for bound in [lo, hi] {
                    if let affine::Bound::Val(val) = bound {
                        operands.push(val);
                    }
                }
                block_args.push(iv);
                for carried in carried.iter_mut() {
                    operands.push(&mut carried.init);
                    block_args.push(&mut carried.arg);
                    results.push(&mut carried.result);
                }
                regions.push(body);
            }
            affine::Op::Apply { result, args, .. } => {
                operands.extend(args.iter_mut());
                results.push(result);
            }
            // ⭐ FOUR GROUPS IN ONE OP, arm for arm with [`operands_mut`], [`results_mut`] and
            // [`regions_mut`]: dims then symbols read, the result list bound, both regions held.
            affine::Op::If {
                set: _,
                args,
                symbol_args,
                results: binds,
                body,
                else_body,
                // ⛔ NOT A VALUE, as on every other op that carries one.
                dbg_name: _,
            } => {
                operands.extend(args.iter_mut());
                operands.extend(symbol_args.iter_mut());
                results.extend(binds.iter_mut());
                regions.extend([body, else_body]);
            }
            affine::Op::Yield { operands: reads } => operands.extend(reads.iter_mut()),
            affine::Op::VectorLoad {
                result,
                view,
                indices,
                ..
            } => {
                operands.push(view);
                index_operands_mut(indices, &mut operands);
                results.push(result);
            }
            affine::Op::VectorStore {
                value,
                view,
                indices,
                ..
            } => {
                operands.extend([value, view]);
                index_operands_mut(indices, &mut operands);
            }
        },
        Op::Dataflow(op) => match op {
            dataflow::Op::GetUnit { result, .. } => results.push(result),
            dataflow::Op::Opaque { .. } => {}
            dataflow::Op::GetLocalUnit { result, of, .. } => {
                operands.push(of);
                results.push(result);
            }
            dataflow::Op::CreateGroup { result, unit_ids } => {
                operands.extend(unit_ids.iter_mut());
                results.push(result);
            }
            dataflow::Op::GetLogicalMemoryView {
                result,
                from,
                start,
                ..
            } => {
                operands.extend([from, start]);
                results.push(result);
            }
            // ⭐ THE PAGE START ADDRESSES ARE OPERANDS, the extents beside them attributes — see
            // [`operands`].
            dataflow::Op::GetPagedLogicalMemoryView(view) => {
                operands.extend([&mut view.unit, &mut view.start_addr]);
                operands.extend(view.pages.iter_mut().map(|page| &mut page.start_addr));
                results.push(&mut view.result);
            }
            dataflow::Op::ProgramUnit { units, body, .. } => {
                operands.extend(units.iter_mut());
                regions.push(body);
            }
            dataflow::Op::Send { to, data, .. } => operands.extend([to.val_mut(), data]),
            dataflow::Op::Receive { result, from, .. } => {
                operands.push(from.val_mut());
                results.push(result);
            }
            dataflow::Op::SyncSend { to, .. } => operands.push(to),
            dataflow::Op::SyncRecv { from, .. } => operands.push(from),
            dataflow::Op::ImplicitSync {
                view, dst, size, ..
            } => operands.extend([view, dst, size]),
        },
        Op::Agen(op) => match op {
            agen::Op::Yield => {}
            agen::Op::VectorLoad {
                result,
                view,
                indices,
                ..
            } => {
                operands.push(view);
                index_operands_mut(indices, &mut operands);
                results.push(result);
            }
            agen::Op::VectorStore {
                value,
                view,
                indices,
                ..
            } => {
                operands.extend([value, view]);
                index_operands_mut(indices, &mut operands);
            }
            agen::Op::SymbolicVectorLoad {
                result,
                view,
                indices,
                strides,
                multicast,
                ..
            } => {
                operands.push(view);
                index_operands_mut(indices, &mut operands);
                operands.extend(strides.iter_mut());
                operands.extend(multicast.as_mut());
                results.push(result);
            }
            agen::Op::SymbolicVectorStore {
                value,
                view,
                indices,
                strides,
                ..
            } => {
                operands.extend([value, view]);
                index_operands_mut(indices, &mut operands);
                operands.extend(strides.iter_mut());
            }
            agen::Op::CompositeLoadAndStore(transfer) => {
                let transfer = transfer.as_mut();
                operands.push(&mut transfer.src);
                index_operands_mut(&mut transfer.src_indices, &mut operands);
                operands.push(&mut transfer.dst);
                index_operands_mut(&mut transfer.dst_indices, &mut operands);
                block_args.push(&mut transfer.load_iv);
                regions.push(&mut transfer.body);
            }
            // ⛔ A REGION AND NOTHING ELSE, and it binds NO block argument: the transfers inside carry
            // their own `load_iv` (`Agen.td:1031` — a bare `SizedRegion<1>`).
            agen::Op::CompositeMemoryInterleave { body, .. } => regions.push(body),
            agen::Op::SetTransferMaskState {
                result, mask_value, ..
            } => {
                operands.push(mask_value);
                results.push(result);
            }
        },
        // ⭐ NO REGION AND NO BLOCK ARGUMENT — arm for arm with [`operands_mut`] and [`results_mut`].
        Op::Vector(op) => match op {
            vector::Op::Load {
                result,
                base,
                indices,
                ..
            } => {
                operands.push(base);
                index_operands_mut(indices, &mut operands);
                results.push(result);
            }
            vector::Op::Store {
                value,
                base,
                indices,
                ..
            } => {
                operands.extend([value, base]);
                index_operands_mut(indices, &mut operands);
            }
        },
        Op::VectorChain(op) => match op {
            // ⭐ THE PARAMETER IS AN OPERAND AND THE MASK IS THE RESULT — see [`operands_mut`].
            vectorchain::Op::CreateAffineMaskSet {
                result,
                mask_parameter,
                ..
            } => {
                operands.extend(mask_parameter.iter_mut());
                results.push(result);
            }
            vectorchain::Op::ConstantBitstream { result, .. }
            | vectorchain::Op::CreateAffineMask { result, .. } => results.push(result),
            vectorchain::Op::Estimate { result, input, .. }
            | vectorchain::Op::FastExp { result, input, .. }
            | vectorchain::Op::Floor { result, input, .. }
            | vectorchain::Op::Neg { result, input, .. }
            | vectorchain::Op::ScanWithGap { result, input, .. }
            | vectorchain::Op::Select { result, input, .. }
            | vectorchain::Op::Shuffle { result, input, .. }
            | vectorchain::Op::Cast { result, input, .. } => {
                operands.push(input);
                results.push(result);
            }
            vectorchain::Op::Rotate {
                result,
                input,
                position,
                ..
            } => {
                operands.extend([input, position]);
                results.push(result);
            }
            vectorchain::Op::Multiply { result, a, b, .. } => {
                operands.extend([a, b]);
                results.push(result);
            }
            vectorchain::Op::MultiplyAccumulate {
                result, a, b, acc, ..
            } => {
                operands.extend([a, b, acc]);
                results.push(result);
            }
            vectorchain::Op::ElementWiseCompare {
                result,
                op1,
                op2,
                mask,
                ..
            } => {
                operands.extend([op1, op2]);
                if let Some(mask) = mask {
                    operands.push(mask.val_mut());
                }
                results.push(result);
            }
            vectorchain::Op::ElementWiseSelection {
                result,
                cond,
                lhs,
                rhs,
                mask,
                ..
            } => {
                operands.extend([cond.val_mut(), lhs, rhs]);
                if let Some(mask) = mask {
                    operands.push(mask.val_mut());
                }
                results.push(result);
            }
            vectorchain::Op::Binary {
                result,
                op1,
                op2,
                mask,
                ..
            }
            | vectorchain::Op::Pack {
                result,
                op1,
                op2,
                mask,
                ..
            } => {
                operands.extend([op1, op2]);
                if let Some(mask) = mask {
                    operands.push(mask.val_mut());
                }
                results.push(result);
            }
            vectorchain::Op::Merge {
                result, op1, op2, ..
            } => {
                operands.extend([op1, op2]);
                results.push(result);
            }
        },
        // ⛔ A SYMBOL READS NOTHING AND HOLDS NOTHING; it binds the extent entry 091's walk stops at.
        Op::Symbol(symbol::Op::CreateSymbol { result, .. }) => results.push(result),
        // ⛔⛔ FOUR GROUPS FOR ONE OP, AND ENTRY 182 USES THREE OF THEM. `cloneWithoutRegions`
        // (`FlatteningLocalRegions.cpp:258`) rewrites the OPERANDS through its mapping, takes fresh
        // RESULTS, and leaves the regions empty for the recursion to fill — so `units` must land in
        // `operands`, `arg` in `block_args`, and `body` in `regions`, or the clone reads the
        // original's values under a name the printer has already numbered.
        Op::Uniform(op) => match op {
            uniform::Op::UniformizeRegions {
                regions: local_regions,
                results: bound,
            } => {
                for region in local_regions {
                    operands.extend(region.units.iter_mut());
                    block_args.push(&mut region.arg);
                    regions.push(&mut region.body);
                }
                results.extend(bound.iter_mut());
            }
            uniform::Op::Yield { operands: yielded } => operands.extend(yielded.iter_mut()),
            uniform::Op::DefImmutableMapping { result, pairs } => {
                for (key, value) in pairs {
                    operands.extend([key, value]);
                }
                results.push(result);
            }
            uniform::Op::QueryMap { result, map, key } => {
                operands.extend([map, key]);
                results.push(result);
            }
        },
    }
    OpPartsMut {
        operands,
        block_args,
        results,
        regions,
    }
}


/// EVERY **USE** OF ONE VALUE IN `scope`, INNERMOST OPS INCLUDED — one entry per use.
///
/// ⛔⛔ ONE ENTRY PER USE, NOT PER USER, because that is what MLIR counts. `Value::hasOneUse()` is
/// false for a value one op reads twice, and `getUsers()` is a mapped range over the same use list —
/// so `uses(v, scope).len() == 1` is exactly `hasOneUse()` and `uses(v, scope).first()` is exactly
/// `*getUsers().begin()`.
///
/// ⛔ AND IT DESCENDS INTO REGIONS. A load's result is read by a send inside an `affine.for` body,
/// and a walk that stopped at the top level would call that value unused.
#[must_use]
pub fn uses(of: Val, scope: &[Op]) -> Vec<&Op> {
    let mut users: Vec<&Op> = Vec::new();
    for op in scope {
        for read in operands(op) {
            if read == of {
                users.push(op);
            }
        }
        for region in regions(op) {
            users.extend(uses(of, region));
        }
    }
    users
}

/// THE OP THAT DEFINES A VALUE AS A RESULT — `Value::getDefiningOp()`.
///
/// ⛔ `None` FOR A REGION ARGUMENT, which is what the reference gets as a null pointer. See
/// [`results`].
#[must_use]
pub fn defining_op(val: Val, scope: &[Op]) -> Option<&Op> {
    for op in scope {
        if results(op).contains(&val) {
            return Some(op);
        }
        for region in regions(op) {
            if let Some(found) = defining_op(val, region) {
                return Some(found);
            }
        }
    }
    None
}

/// THE OP WHOSE REGION BINDS A VALUE AS AN ARGUMENT — `BlockArgument::getOwner()->getParentOp()`.
///
/// ⭐ THE MIRROR OF [`defining_op`], AND EXACTLY ONE OF THE TWO ANSWERS. A value is either an op's
/// result or a region's argument, never both — so `region_owner(v, scope).is_some()` is
/// `isa<BlockArgument>(v)` and `defining_op(v, scope).is_some()` is `!isa<BlockArgument>(v)`.
///
/// ⛔ IT ANSWERS WITH THE **PARENT OP**, NOT THE BLOCK. MLIR's `getOwner()` is the block and callers
/// immediately ask it for `getParentOp()` — `cast<BlockArgument>(index).getOwner()->getParentOp()`
/// then `isa<affine::AffineForOp, scf::ForOp>(loop_op)`
/// (`dcc/src/Conversion/AgenToSentient/AccessDetails.cpp:360-362`). This island has no block between
/// an op and its region's arguments, so the op is the whole answer.
///
/// ⛔ AND IT DESCENDS INTO REGIONS, for [`uses`]' reason: the loop whose induction variable a
/// subscript names is nested inside the program unit, not beside it.
#[must_use]
pub fn region_owner(val: Val, scope: &[Op]) -> Option<&Op> {
    for op in scope {
        if block_args(op).contains(&val) {
            return Some(op);
        }
        for region in regions(op) {
            if let Some(found) = region_owner(val, region) {
                return Some(found);
            }
        }
    }
    None
}

/// THE SSA VALUES ONE INDEX LIST READS.
///
/// ⛔ A STRIDED SUM READS EVERY VARIABLE IN IT. `%arg9 + %arg8 * 8` is two uses, not one — see
/// [`Index::Strided`].
fn index_operands(indices: &[Index], into: &mut Vec<Val>) {
    for index in indices {
        match index {
            Index::Val(val) => into.push(*val),
            Index::Const(_) => {}
            Index::Strided(terms, _) => into.extend(terms.iter().map(|(val, _)| *val)),
        }
    }
}

/// THE SSA VALUES ONE INDEX LIST READS, AS PLACES — the mirror of [`index_operands`].
fn index_operands_mut<'o>(indices: &'o mut [Index], into: &mut Vec<&'o mut Val>) {
    for index in indices {
        match index {
            Index::Val(val) => into.push(val),
            Index::Const(_) => {}
            Index::Strided(terms, _) => into.extend(terms.iter_mut().map(|(val, _)| val)),
        }
    }
}

// ───────────────────────────── THE SAME WALK, IN PLACE ──────────────────────────────

/// WHICH ROLE A `Val` PLAYS IN THE OP THAT NAMES IT.
///
/// ⛔ THREE ROLES AND NO FOURTH, because the three walks above are exactly these three questions:
/// [`operands`] is what an op READS, [`results`] what it DEFINES, [`block_args`] what its regions
/// BIND. A mutable walk that did not carry the role would let a rewrite meant for an operand land on
/// a result and silently re-define a value.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Role {
    /// A value the op READS — see [`operands`].
    Operand,
    /// A value the op DEFINES as a result — see [`results`].
    Result,
    /// A value the op's regions BIND — see [`block_args`].
    BlockArg,
}

/// EVERY `Val` ONE OP NAMES, BY ROLE, ASSIGNABLE IN PLACE.
///
/// # 🛑 `setOperand` IS AN OPERATION THIS ISLAND DID NOT HAVE
///
/// ⛔⛔ `redefineConstantVectors` REDIRECTS **ONE** USE OF A VALUE AND LEAVES THE OTHERS ALONE —
/// `group.op_to_update_->setOperand(group.location_, group.update_with_)`
/// (`VectorChainHelper.cpp:600-602`). Nothing here could express that: [`operands`] hands out
/// COPIES. Rebuilding the op from a fresh literal instead means restating every other field of the
/// variant at the rewrite site, which is how the wrong field gets reset to a default.
///
/// ⭐⭐ ONE TOTAL MATCH FOR ALL THREE ROLES, AND [`operands_mut`] IS A FILTER OVER IT. Three separate
/// mutable walks could drift out of step with [`operands`], [`results`] and [`block_args`] one
/// variant at a time; one walk that says which role each slot is cannot. The order WITHIN a role is
/// theirs, and `unit_tests::the_mutable_walk_agrees_with_the_immutable_ones` freezes that.
///
/// ⛔ AND IT DOES NOT DESCEND INTO REGIONS, exactly like the three walks it mirrors. [`regions_mut`]
/// is how a caller goes down.
#[must_use]
pub fn vals_mut(op: &mut Op) -> Vec<(Role, &mut Val)> {
    let mut vals: Vec<(Role, &mut Val)> = Vec::new();
    match op {
        Op::Arith(op) => match op {
            arith::Op::Constant { result, .. }
            | arith::Op::ConstantInt { result, .. }
            | arith::Op::DenseConstant { result, .. } => vals.push((Role::Result, result)),
            arith::Op::AddI(bin)
            | arith::Op::SubI(bin)
            | arith::Op::MulI(bin)
            | arith::Op::DivSI(bin)
            | arith::Op::RemSI(bin) => {
                vals.push((Role::Operand, &mut bin.lhs));
                vals.push((Role::Operand, &mut bin.rhs));
                vals.push((Role::Result, &mut bin.result));
            }
            arith::Op::Compare {
                result, lhs, rhs, ..
            } => {
                vals.push((Role::Operand, lhs));
                vals.push((Role::Operand, rhs));
                vals.push((Role::Result, result));
            }
            arith::Op::Select {
                result,
                condition,
                true_value,
                false_value,
                ..
            } => {
                vals.push((Role::Operand, condition));
                vals.push((Role::Operand, true_value));
                vals.push((Role::Operand, false_value));
                vals.push((Role::Result, result));
            }
            arith::Op::Logic {
                result, operands, ..
            } => {
                vals.extend(operands.iter_mut().map(|val| (Role::Operand, val)));
                vals.push((Role::Result, result));
            }
        },
        Op::Scf(op) => match op {
            // ⛔ THE INDUCTION VARIABLES ARE THE REGION'S ARGUMENTS, not operands — see [`operands`].
            scf::Op::Parallel { ivs, body: _ } => {
                vals.extend(ivs.iter_mut().map(|iv| (Role::BlockArg, iv)));
            }
            // ⭐ THE IV IS THE ONE THING [`operands_mut`] HAS NO ROLE FOR — an `scf.for` binds it
            // as its body's argument, exactly as [`scf::Op::Parallel`] binds its `ivs`.
            scf::Op::For {
                iv,
                lo,
                hi,
                step,
                carried,
                body: _,
                dbg_name: _,
            } => {
                vals.extend([lo, hi, step].map(|val| (Role::Operand, val)));
                vals.push((Role::BlockArg, iv));
                for carried in carried.iter_mut() {
                    vals.push((Role::Operand, &mut carried.init));
                    vals.push((Role::BlockArg, &mut carried.arg));
                    vals.push((Role::Result, &mut carried.result));
                }
            }
            scf::Op::If { cond, results, .. } => {
                vals.push((Role::Operand, cond));
                vals.extend(results.iter_mut().map(|val| (Role::Result, val)));
            }
            scf::Op::Yield { operands } => {
                vals.extend(operands.iter_mut().map(|val| (Role::Operand, val)));
            }
        },
        Op::Affine(op) => match op {
            // ⭐ THE THREE ROLES INTERLEAVE IN ONE VARIANT, and this is the only op where they do:
            // each `iter_args` entry contributes an operand (its initialiser), a result and a region
            // argument. They are collected in ONE pass over `carried` and appended in role order, so
            // the sequence matches [`operands`] ++ [`results`] ++ [`block_args`] and not the field
            // order of [`affine::Carried`].
            affine::Op::For {
                iv,
                lo,
                hi,
                carried,
                body: _,
                dbg_name: _,
            } => {
                for bound in [lo, hi] {
                    if let affine::Bound::Val(val) = bound {
                        vals.push((Role::Operand, val));
                    }
                }
                let mut inits: Vec<(Role, &mut Val)> = Vec::new();
                let mut results: Vec<(Role, &mut Val)> = Vec::new();
                let mut args: Vec<(Role, &mut Val)> = Vec::new();
                for carried in carried.iter_mut() {
                    inits.push((Role::Operand, &mut carried.init));
                    results.push((Role::Result, &mut carried.result));
                    args.push((Role::BlockArg, &mut carried.arg));
                }
                vals.extend(inits);
                vals.extend(results);
                vals.push((Role::BlockArg, iv));
                vals.extend(args);
            }
            // ⭐ ROLE ORDER, [`operands`] ++ [`results`], AND NO BLOCK ARGUMENT: `affine.if`'s regions
            // take none — the set's dims are OPERANDS of the op, not arguments of its blocks.
            affine::Op::If {
                set: _,
                args,
                symbol_args,
                results,
                body: _,
                else_body: _,
                dbg_name: _,
            } => {
                vals.extend(args.iter_mut().map(|val| (Role::Operand, val)));
                vals.extend(symbol_args.iter_mut().map(|val| (Role::Operand, val)));
                vals.extend(results.iter_mut().map(|val| (Role::Result, val)));
            }
            affine::Op::Apply { result, args, .. } => {
                vals.extend(args.iter_mut().map(|val| (Role::Operand, val)));
                vals.push((Role::Result, result));
            }
            affine::Op::Yield { operands } => {
                vals.extend(operands.iter_mut().map(|val| (Role::Operand, val)));
            }
            affine::Op::VectorLoad {
                result,
                view,
                indices,
                ..
            } => {
                vals.push((Role::Operand, view));
                index_vals_mut(indices, &mut vals);
                vals.push((Role::Result, result));
            }
            affine::Op::VectorStore {
                value,
                view,
                indices,
                ..
            } => {
                vals.push((Role::Operand, value));
                vals.push((Role::Operand, view));
                index_vals_mut(indices, &mut vals);
            }
        },
        Op::Dataflow(op) => match op {
            dataflow::Op::GetUnit { result, .. } => vals.push((Role::Result, result)),
            dataflow::Op::Opaque { .. } => {}
            dataflow::Op::GetPagedLogicalMemoryView(view) => {
                vals.push((Role::Operand, &mut view.unit));
                vals.push((Role::Operand, &mut view.start_addr));
                vals.extend(
                    view.pages
                        .iter_mut()
                        .map(|page| (Role::Operand, &mut page.start_addr)),
                );
                vals.push((Role::Result, &mut view.result));
            }
            dataflow::Op::GetLocalUnit { result, of, .. } => {
                vals.push((Role::Operand, of));
                vals.push((Role::Result, result));
            }
            dataflow::Op::CreateGroup { result, unit_ids } => {
                vals.extend(unit_ids.iter_mut().map(|val| (Role::Operand, val)));
                vals.push((Role::Result, result));
            }
            dataflow::Op::GetLogicalMemoryView {
                result,
                from,
                start,
                ..
            } => {
                vals.push((Role::Operand, from));
                vals.push((Role::Operand, start));
                vals.push((Role::Result, result));
            }
            dataflow::Op::ProgramUnit { units, .. } => {
                vals.extend(units.iter_mut().map(|val| (Role::Operand, val)));
            }
            dataflow::Op::Send { to, data, .. } => {
                vals.push((Role::Operand, to.val_mut()));
                vals.push((Role::Operand, data));
            }
            dataflow::Op::Receive { result, from, .. } => {
                vals.push((Role::Operand, from.val_mut()));
                vals.push((Role::Result, result));
            }
            dataflow::Op::SyncSend { to, .. } => vals.push((Role::Operand, to)),
            dataflow::Op::SyncRecv { from, .. } => vals.push((Role::Operand, from)),
            dataflow::Op::ImplicitSync {
                view, dst, size, ..
            } => {
                vals.push((Role::Operand, view));
                vals.push((Role::Operand, dst));
                vals.push((Role::Operand, size));
            }
        },
        Op::Agen(op) => match op {
            agen::Op::Yield => {}
            agen::Op::VectorLoad {
                result,
                view,
                indices,
                ..
            } => {
                vals.push((Role::Operand, view));
                index_vals_mut(indices, &mut vals);
                vals.push((Role::Result, result));
            }
            agen::Op::VectorStore {
                value,
                view,
                indices,
                ..
            } => {
                vals.push((Role::Operand, value));
                vals.push((Role::Operand, view));
                index_vals_mut(indices, &mut vals);
            }
            agen::Op::SymbolicVectorLoad {
                result,
                view,
                indices,
                strides,
                multicast,
                ..
            } => {
                vals.push((Role::Operand, view));
                index_vals_mut(indices, &mut vals);
                vals.extend(strides.iter_mut().map(|val| (Role::Operand, val)));
                vals.extend(multicast.as_mut().map(|val| (Role::Operand, val)));
                vals.push((Role::Result, result));
            }
            agen::Op::SymbolicVectorStore {
                value,
                view,
                indices,
                strides,
                ..
            } => {
                vals.push((Role::Operand, value));
                vals.push((Role::Operand, view));
                index_vals_mut(indices, &mut vals);
                vals.extend(strides.iter_mut().map(|val| (Role::Operand, val)));
            }
            agen::Op::CompositeLoadAndStore(transfer) => {
                let transfer = transfer.as_mut();
                vals.push((Role::Operand, &mut transfer.src));
                index_vals_mut(&mut transfer.src_indices, &mut vals);
                vals.push((Role::Operand, &mut transfer.dst));
                index_vals_mut(&mut transfer.dst_indices, &mut vals);
                vals.push((Role::BlockArg, &mut transfer.load_iv));
            }
            // ⭐ ARM FOR ARM WITH [`parts_mut`], minus the region this one does not visit.
            agen::Op::CompositeMemoryInterleave { .. } => {}
            agen::Op::SetTransferMaskState {
                result, mask_value, ..
            } => {
                vals.push((Role::Operand, mask_value));
                vals.push((Role::Result, result));
            }
        },
        // ⭐ ARM FOR ARM WITH [`parts_mut`]; a plain access binds no block argument.
        Op::Vector(op) => match op {
            vector::Op::Load {
                result,
                base,
                indices,
                ..
            } => {
                vals.push((Role::Operand, base));
                index_vals_mut(indices, &mut vals);
                vals.push((Role::Result, result));
            }
            vector::Op::Store {
                value,
                base,
                indices,
                ..
            } => {
                vals.push((Role::Operand, value));
                vals.push((Role::Operand, base));
                index_vals_mut(indices, &mut vals);
            }
        },
        Op::VectorChain(op) => match op {
            vectorchain::Op::ConstantBitstream { result, .. }
            | vectorchain::Op::CreateAffineMask { result, .. } => {
                vals.push((Role::Result, result));
            }
            // ⭐ THE MASK PARAMETER IS AN OPERAND — see [`operands_mut`].
            vectorchain::Op::CreateAffineMaskSet {
                result,
                mask_parameter,
                ..
            } => {
                vals.extend(mask_parameter.iter_mut().map(|val| (Role::Operand, val)));
                vals.push((Role::Result, result));
            }
            vectorchain::Op::Estimate { result, input, .. }
            | vectorchain::Op::ScanWithGap { result, input, .. }
            | vectorchain::Op::Select { result, input, .. }
            | vectorchain::Op::Shuffle { result, input, .. }
            | vectorchain::Op::FastExp { result, input, .. }
            | vectorchain::Op::Floor { result, input, .. }
            | vectorchain::Op::Neg { result, input, .. }
            | vectorchain::Op::Cast { result, input, .. } => {
                vals.push((Role::Operand, input));
                vals.push((Role::Result, result));
            }
            vectorchain::Op::Rotate {
                result,
                input,
                position,
                ..
            } => {
                vals.push((Role::Operand, input));
                vals.push((Role::Operand, position));
                vals.push((Role::Result, result));
            }
            vectorchain::Op::Multiply { result, a, b, .. } => {
                vals.push((Role::Operand, a));
                vals.push((Role::Operand, b));
                vals.push((Role::Result, result));
            }
            vectorchain::Op::MultiplyAccumulate {
                result, a, b, acc, ..
            } => {
                vals.push((Role::Operand, a));
                vals.push((Role::Operand, b));
                vals.push((Role::Operand, acc));
                vals.push((Role::Result, result));
            }
            vectorchain::Op::ElementWiseCompare {
                result,
                op1,
                op2,
                mask,
                ..
            } => {
                vals.push((Role::Operand, op1));
                vals.push((Role::Operand, op2));
                if let Some(mask) = mask.as_mut() {
                    vals.push((Role::Operand, mask.val_mut()));
                }
                vals.push((Role::Result, result));
            }
            vectorchain::Op::ElementWiseSelection {
                result,
                cond,
                lhs,
                rhs,
                mask,
                ..
            } => {
                vals.push((Role::Operand, cond.val_mut()));
                vals.push((Role::Operand, lhs));
                vals.push((Role::Operand, rhs));
                if let Some(mask) = mask.as_mut() {
                    vals.push((Role::Operand, mask.val_mut()));
                }
                vals.push((Role::Result, result));
            }
            vectorchain::Op::Binary {
                result,
                op1,
                op2,
                mask,
                ..
            }
            | vectorchain::Op::Pack {
                result,
                op1,
                op2,
                mask,
                ..
            } => {
                vals.push((Role::Operand, op1));
                vals.push((Role::Operand, op2));
                if let Some(mask) = mask.as_mut() {
                    vals.push((Role::Operand, mask.val_mut()));
                }
                vals.push((Role::Result, result));
            }
            // ⛔ NO MASK — see [`operands`].
            vectorchain::Op::Merge {
                result, op1, op2, ..
            } => {
                vals.push((Role::Operand, op1));
                vals.push((Role::Operand, op2));
                vals.push((Role::Result, result));
            }
        },
        // ⛔ A SYMBOL BINDS ITS EXTENT AND READS NOTHING — see [`operands_mut`].
        Op::Symbol(symbol::Op::CreateSymbol { result, .. }) => {
            vals.push((Role::Result, result));
        }
        // ⭐ THE UNITS ARE READ, THE REGION ARGUMENT IS BOUND, AND THE RESULTS ARE DEFINED — the
        // three roles [`operands`], [`block_args`] and [`results`] give the same op, in that order.
        Op::Uniform(op) => match op {
            uniform::Op::UniformizeRegions { regions, results } => {
                for region in regions {
                    vals.extend(region.units.iter_mut().map(|unit| (Role::Operand, unit)));
                    vals.push((Role::BlockArg, &mut region.arg));
                }
                vals.extend(results.iter_mut().map(|result| (Role::Result, result)));
            }
            uniform::Op::Yield { operands } => {
                vals.extend(operands.iter_mut().map(|read| (Role::Operand, read)));
            }
            uniform::Op::DefImmutableMapping { result, pairs } => {
                for (key, value) in pairs {
                    vals.push((Role::Operand, key));
                    vals.push((Role::Operand, value));
                }
                vals.push((Role::Result, result));
            }
            uniform::Op::QueryMap { result, map, key } => {
                vals.push((Role::Operand, map));
                vals.push((Role::Operand, key));
                vals.push((Role::Result, result));
            }
        },
    }
    vals
}

/// THE SSA SLOTS ONE INDEX LIST READS — [`index_operands`], assignable.
fn index_vals_mut<'a>(indices: &'a mut [Index], into: &mut Vec<(Role, &'a mut Val)>) {
    for index in indices {
        match index {
            Index::Val(val) => into.push((Role::Operand, val)),
            Index::Const(_) => {}
            Index::Strided(terms, _) => {
                into.extend(terms.iter_mut().map(|(val, _)| (Role::Operand, val)));
            }
        }
    }
}

#[cfg(test)]
mod unit_tests {
    use super::{
        Index, Op, Role, Val, block_args, operands, operands_mut, regions, regions_mut, results,
        vals_mut,
    };
    use super::{affine, agen, arith, dataflow, scf, vectorchain};
    use crate::islands::dataflow_ir::link::{Link, Lxlu, Sfp};
    use crate::islands::dataflow_ir::ty::{
        AffineMap, ElemType, IntegerSet, MemRef, ScalarTy, Vector,
    };

    fn vector() -> Vector {
        Vector {
            len: 64,
            elem: ElemType::F16,
        }
    }

    fn memref() -> MemRef {
        MemRef {
            shape: vec![8, 64],
            elem: ElemType::F16,
        }
    }

    fn predicate(val: u32) -> vectorchain::Predicate {
        vectorchain::LaneMask::prefix_of(64, vector()).binds(Val(val))
    }

    /// ONE OP PER MECHANISM THE MUTABLE WALK HAS TO GET RIGHT.
    ///
    /// ⛔ THIS IS NOT A SAMPLE OF THE ENUM AND IT IS NOT TRYING TO BE. Coverage of the variants is
    /// the COMPILER'S job: [`vals_mut`] and [`regions_mut`] have no wildcard arm, so a new op cannot
    /// be added without stating its slots. What a test has to check instead is the part the compiler
    /// cannot — that the slots come out in the ORDER the three immutable walks name them — and the
    /// list below holds one op for each shape that could get that wrong: a sub-struct's fields, a
    /// `Vec` of operands, the one variant where all three roles interleave, an index list with a
    /// strided sum, the two ops whose operand is a private link end, the two private
    /// [`vectorchain::Predicate`]s of a masked selection, a boxed payload with a region argument, and
    /// an op with two regions.
    fn one_of_each_mechanism() -> Vec<Op> {
        vec![
            // A sub-struct's three fields, in one arm.
            Op::Arith(arith::Op::AddI(arith::IntBinary {
                result: Val(3),
                lhs: Val(1),
                rhs: Val(2),
                ty: ScalarTy::Index,
            })),
            // A `Vec` of operands ahead of the result.
            Op::Arith(arith::Op::Logic {
                result: Val(7),
                kind: arith::LogicKind::And,
                operands: vec![Val(4), Val(5), Val(6)],
            }),
            // ⭐ THE HARD ONE: a dynamic bound, and one carried value contributing an operand, a
            // result and a region argument.
            Op::Affine(affine::Op::For {
                iv: Val(10),
                lo: affine::Bound::Const(0),
                hi: affine::Bound::Val(Val(8)),
                carried: vec![affine::Carried {
                    init: Val(9),
                    arg: Val(11),
                    result: Val(12),
                }],
                body: Vec::new(),
                dbg_name: None,
            }),
            // An index list holding a strided sum and a literal.
            Op::Agen(agen::Op::VectorLoad {
                result: Val(20),
                view: Val(13),
                indices: vec![
                    Index::Const(0),
                    Index::Strided(vec![(Val(14), 1), (Val(15), 8)], 4),
                    Index::Val(Val(16)),
                ],
                view_ty: memref(),
                ty: vector(),
            }),
            // The private link ends.
            Op::Dataflow(dataflow::Op::Send {
                to: Link::<Lxlu, Sfp>::between(Val(30), Val(31)).ends().0,
                data: Val(32),
                ty: vector(),
            }),
            Op::Dataflow(dataflow::Op::Receive {
                result: Val(35),
                from: Link::<Lxlu, Sfp>::between(Val(33), Val(34)).ends().1,
                ty: vector(),
            }),
            // Two private predicates, one of them optional.
            Op::VectorChain(vectorchain::Op::ElementWiseSelection {
                result: Val(45),
                cond: predicate(40),
                lhs: Val(41),
                rhs: Val(42),
                mask: Some(predicate(43)),
                ty: vector(),
            }),
            // A boxed payload with two index lists and a region argument.
            Op::Agen(agen::Op::CompositeLoadAndStore(Box::new(
                agen::CompositeTransfer {
                    src: Val(50),
                    src_indices: vec![Index::Val(Val(51))],
                    src_ty: memref(),
                    dst: Val(52),
                    dst_indices: vec![Index::Val(Val(53))],
                    dst_ty: memref(),
                    load_iv: Val(54),
                    load_iv_ty: vector(),
                    load_set: IntegerSet::from_sizes(&[8]),
                    load_order: AffineMap::identity(1),
                    store_set: IntegerSet::from_sizes(&[8]),
                    store_order: AffineMap::identity(1),
                    time_set: IntegerSet::from_sizes(&[8]),
                    time_order: AffineMap::identity(1),
                    load_time_addr_map: AffineMap::identity(1),
                    store_time_addr_map: AffineMap::identity(1),
                    body: vec![Op::Agen(agen::Op::Yield)],
                },
            ))),
            // Two regions.
            Op::Scf(scf::Op::If {
                cond: Val(60),
                results: Vec::new(),
                body: vec![Op::Agen(agen::Op::Yield)],
                else_body: Vec::new(),
                dbg_name: None,
            }),
        ]
    }

    /// ⛔⛔ THE MUTABLE WALK IS ONLY USEFUL IF A POSITION MEANS THE SAME THING IN BOTH. `setOperand`
    /// is indexed by `use.getOperandNumber()` (`VectorChainHelper.cpp:600-602`), so slot `i` of
    /// [`operands_mut`] has to be value `i` of [`operands`] — and the same for the other two roles.
    #[test]
    fn the_mutable_walk_agrees_with_the_immutable_ones() {
        for op in one_of_each_mechanism() {
            let expected_operands = operands(&op);
            let expected_results = results(&op);
            let expected_args = block_args(&op);
            let expected_regions: Vec<usize> =
                regions(&op).iter().map(|region| region.len()).collect();

            let mut walked = op.clone();
            let by_role = |role: Role| -> Vec<Val> {
                let mut copy = op.clone();
                vals_mut(&mut copy)
                    .into_iter()
                    .filter(|(seen, _)| *seen == role)
                    .map(|(_, val)| *val)
                    .collect()
            };

            assert_eq!(by_role(Role::Operand), expected_operands, "{op:?}");
            assert_eq!(by_role(Role::Result), expected_results, "{op:?}");
            assert_eq!(by_role(Role::BlockArg), expected_args, "{op:?}");
            // ⛔ [`operands_mut`] IS A SUBSEQUENCE OF THE ROLE, NOT THE WHOLE OF IT. It withholds a
            // link end and a [`vectorchain::Predicate`] on purpose, for the reason its own doc gives;
            // what has to hold is that every place it DOES hand out appears in [`operands`] in the
            // same relative order, so neither walk shifts the other's slots.
            let narrow: Vec<Val> = operands_mut(&mut walked)
                .into_iter()
                .map(|val| *val)
                .collect();
            let mut wide = expected_operands.iter();
            for place in &narrow {
                assert!(wide.any(|seen| seen == place), "{place:?} of {op:?}");
            }
            assert_eq!(
                regions_mut(&mut walked)
                    .iter()
                    .map(|region| region.len())
                    .collect::<Vec<_>>(),
                expected_regions,
                "{op:?}"
            );
        }
    }

    /// ⛔ ASSIGNING ONE SLOT MOVES ONE VALUE. The whole reason `redefineConstantVectors` needs a
    /// positional rewrite is that a value read TWICE by one op gets two DIFFERENT clones, one per
    /// position (`VectorChainHelper.cpp:589-595`) — so a walk that rewrote every matching operand at
    /// once would be the wrong operation.
    #[test]
    fn assigning_one_operand_slot_leaves_the_others_alone() {
        let mut op = Op::Arith(arith::Op::Logic {
            result: Val(7),
            kind: arith::LogicKind::And,
            operands: vec![Val(4), Val(4), Val(6)],
        });

        *operands_mut(&mut op)[1] = Val(99);

        assert_eq!(operands(&op), vec![Val(4), Val(99), Val(6)]);
        assert_eq!(results(&op), vec![Val(7)]);
    }

    /// ⛔ A MASK'S TYPE SURVIVES THE SUBSTITUTION. [`vectorchain::Predicate`] exists so that the type
    /// travels with the value from its definition, and `val_mut` is deliberately unable to touch it.
    /// ⭐ ONLY [`vals_mut`] REACHES A CONDITION VECTOR AT ALL — [`operands_mut`] withholds it, so
    /// this substitution is one a positional rewrite can make and a use re-pointing cannot.
    #[test]
    fn substituting_a_mask_value_keeps_the_type_it_was_defined_at() {
        let mut op = Op::VectorChain(vectorchain::Op::ElementWiseSelection {
            result: Val(45),
            cond: predicate(40),
            lhs: Val(41),
            rhs: Val(42),
            mask: None,
            ty: vector(),
        });

        *vals_mut(&mut op)[0].1 = Val(70);

        let Op::VectorChain(vectorchain::Op::ElementWiseSelection { cond, .. }) = &op else {
            unreachable!("the op above is an element_wise_selection")
        };
        assert_eq!(cond.val(), Val(70));
        assert_eq!(cond.ty(), vector());
    }
}
