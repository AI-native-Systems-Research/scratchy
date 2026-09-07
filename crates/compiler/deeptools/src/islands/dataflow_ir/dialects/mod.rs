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
    /// `Dataflow.td` — units, views, transfers between units, and the opaque bodies.
    Dataflow(dataflow::Op),
    /// `Agen.td` — the address-generator's accesses and composite transfers.
    Agen(agen::Op),
    /// `VectorChain.td` — everything the PE and the SFP compute.
    VectorChain(vectorchain::Op),
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
            arith::Op::AddI(bin) | arith::Op::SubI(bin) | arith::Op::MulI(bin) => {
                reads.extend([bin.lhs, bin.rhs]);
            }
            // ⭐ THE PREDICATE IS NOT AN OPERAND — it is `arith.cmpi`'s first token, an
            // attribute. Both compared values are uses; which comparison it is, is not.
            arith::Op::Compare { lhs, rhs, .. } => reads.extend([*lhs, *rhs]),
            arith::Op::Logic { operands, .. } => reads.extend(operands.iter().copied()),
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
            // ⛔ `load_iv` IS THE REGION'S ARGUMENT, not an operand — see [`block_args`].
            agen::Op::CompositeLoadAndStore(transfer) => {
                reads.push(transfer.src);
                index_operands(&transfer.src_indices, &mut reads);
                reads.push(transfer.dst);
                index_operands(&transfer.dst_indices, &mut reads);
            }
        },
        Op::VectorChain(op) => match op {
            vectorchain::Op::ConstantBitstream { .. }
            | vectorchain::Op::CreateAffineMask { .. } => {}
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
            | arith::Op::Logic { result, .. }
            | arith::Op::DenseConstant { result, .. } => vec![*result],
            arith::Op::AddI(bin) | arith::Op::SubI(bin) | arith::Op::MulI(bin) => vec![bin.result],
        },
        Op::Scf(op) => match op {
            // ⭐ A CARRYING `scf.for` BINDS RESULTS, one per `iter_args` entry — the reference's own
            // input to `TransformLoopToLegalizeForSentientLowering` is
            // `%11 = scf.for %arg4 = %c0 to %10 step %c1 iter_args(%arg5 = %arg3) -> (index)`
            // (`scf_loop_with_result.mlir:32`), and `transformSCFToAffineLoop` branches on
            // `scf_for.getNumResults() > 0` to decide whether to write a yield at all (`:129`). A
            // census that answered "none" would make that branch unreachable.
            scf::Op::For { carried, .. } => carried.iter().map(|carried| carried.result).collect(),
            // ⛔ AND NEITHER OF THE OTHER TWO BINDS ONE IN THIS ISLAND. `scf.if`'s and
            // `scf.parallel`'s results would be the values their yields carry, and nothing this
            // crate emits reads one.
            scf::Op::If { .. } | scf::Op::Yield { .. } | scf::Op::Parallel { .. } => Vec::new(),
        },
        Op::Affine(op) => match op {
            // ⭐ A CARRYING LOOP DOES BIND RESULTS — one per `iter_args` entry, which is how the
            // address a nest computes leaves it (`AgenToSentient.hpp:502-520`). A plain counted
            // loop carries nothing and binds nothing.
            affine::Op::For { carried, .. } => {
                carried.iter().map(|carried| carried.result).collect()
            }
            affine::Op::Yield { .. } | affine::Op::VectorStore { .. } => Vec::new(),
            affine::Op::Apply { result, .. } | affine::Op::VectorLoad { result, .. } => {
                vec![*result]
            }
        },
        Op::Dataflow(op) => match op {
            dataflow::Op::GetUnit { result, .. }
            | dataflow::Op::GetLocalUnit { result, .. }
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
            agen::Op::VectorLoad { result, .. } => vec![*result],
            agen::Op::VectorStore { .. } | agen::Op::Yield | agen::Op::CompositeLoadAndStore(_) => {
                Vec::new()
            }
        },
        Op::VectorChain(op) => match op {
            vectorchain::Op::Estimate { result, .. }
            | vectorchain::Op::FastExp { result, .. }
            | vectorchain::Op::Floor { result, .. }
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
            | vectorchain::Op::CreateAffineMask { result, .. } => vec![*result],
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
            arith::Op::AddI(bin) | arith::Op::SubI(bin) | arith::Op::MulI(bin) => {
                places.extend([&mut bin.lhs, &mut bin.rhs]);
            }
            arith::Op::Compare { lhs, rhs, .. } => places.extend([lhs, rhs]),
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
            agen::Op::CompositeLoadAndStore(transfer) => {
                places.push(&mut transfer.src);
                index_operands_mut(&mut transfer.src_indices, &mut places);
                places.push(&mut transfer.dst);
                index_operands_mut(&mut transfer.dst_indices, &mut places);
            }
        },
        Op::VectorChain(op) => match op {
            vectorchain::Op::ConstantBitstream { .. }
            | vectorchain::Op::CreateAffineMask { .. } => {}
            vectorchain::Op::Estimate { input, .. }
            | vectorchain::Op::FastExp { input, .. }
            | vectorchain::Op::Floor { input, .. }
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
            | arith::Op::Logic { result, .. }
            | arith::Op::DenseConstant { result, .. } => vec![result],
            arith::Op::AddI(bin) | arith::Op::SubI(bin) | arith::Op::MulI(bin) => {
                vec![&mut bin.result]
            }
        },
        Op::Scf(_) => Vec::new(),
        Op::Affine(op) => match op {
            affine::Op::For { carried, .. } => carried
                .iter_mut()
                .map(|carried| &mut carried.result)
                .collect(),
            affine::Op::Yield { .. } | affine::Op::VectorStore { .. } => Vec::new(),
            affine::Op::Apply { result, .. } | affine::Op::VectorLoad { result, .. } => {
                vec![result]
            }
        },
        Op::Dataflow(op) => match op {
            dataflow::Op::GetUnit { result, .. }
            | dataflow::Op::GetLocalUnit { result, .. }
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
            agen::Op::VectorLoad { result, .. } => vec![result],
            agen::Op::VectorStore { .. } | agen::Op::Yield | agen::Op::CompositeLoadAndStore(_) => {
                Vec::new()
            }
        },
        Op::VectorChain(op) => match op {
            vectorchain::Op::Estimate { result, .. }
            | vectorchain::Op::FastExp { result, .. }
            | vectorchain::Op::Floor { result, .. }
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
        Op::Arith(_)
        | Op::Affine(_)
        | Op::Scf(_)
        | Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_) => Vec::new(),
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
        Op::Scf(scf::Op::If {
            body, else_body, ..
        }) => vec![body.as_slice(), else_body.as_slice()],
        Op::Dataflow(dataflow::Op::ProgramUnit { body, .. }) => vec![body.as_slice()],
        Op::Agen(agen::Op::CompositeLoadAndStore(transfer)) => vec![transfer.body.as_slice()],
        Op::Arith(_)
        | Op::Affine(_)
        | Op::Scf(_)
        | Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_) => Vec::new(),
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
        Op::Scf(scf::Op::If {
            body, else_body, ..
        }) => vec![body, else_body],
        Op::Dataflow(dataflow::Op::ProgramUnit { body, .. }) => vec![body],
        Op::Agen(agen::Op::CompositeLoadAndStore(transfer)) => vec![&mut transfer.body],
        Op::Arith(_)
        | Op::Affine(_)
        | Op::Scf(_)
        | Op::Dataflow(_)
        | Op::Agen(_)
        | Op::VectorChain(_) => Vec::new(),
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
            arith::Op::AddI(bin) | arith::Op::SubI(bin) | arith::Op::MulI(bin) => {
                operands.extend([&mut bin.lhs, &mut bin.rhs]);
                results.push(&mut bin.result);
            }
            arith::Op::Compare {
                result, lhs, rhs, ..
            } => {
                operands.extend([lhs, rhs]);
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
                body,
                else_body,
            } => {
                operands.push(cond);
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
            agen::Op::CompositeLoadAndStore(transfer) => {
                let transfer = transfer.as_mut();
                operands.push(&mut transfer.src);
                index_operands_mut(&mut transfer.src_indices, &mut operands);
                operands.push(&mut transfer.dst);
                index_operands_mut(&mut transfer.dst_indices, &mut operands);
                block_args.push(&mut transfer.load_iv);
                regions.push(&mut transfer.body);
            }
        },
        Op::VectorChain(op) => match op {
            vectorchain::Op::ConstantBitstream { result, .. }
            | vectorchain::Op::CreateAffineMask { result, .. } => results.push(result),
            vectorchain::Op::Estimate { result, input, .. }
            | vectorchain::Op::FastExp { result, input, .. }
            | vectorchain::Op::Floor { result, input, .. }
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
