// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! THE NINE GRANITE CONFIGURATIONS, **EXECUTED**, AND COMPARED AGAINST TORCH.
//!
//! ```text
//!   triton_frontend::codegen::compile        .py SOURCE  -> raw ttir            value
//!   triton_frontend::opt::make_ttir          inline/fold/cse/dce/hoist          value
//!   triton_ktir::from_ttir::convert          ttir        -> our KTIR            value
//!   triton_ktir::make_ktir                   the KTDP/KTDF passes               value
//!   triton_ktir::passes::to_ktir::run        KTDP        -> KTIR                value
//!   to_ktir_emit::lower                      ours        -> ktir_core::IRFunction
//!   ktir_emulator                            IRFunction  -> output tensors
//!   compare                                  vs test/numeric/<config>/ref_out.bin
//! ```
//!
//! No arrow above serialises, and NO STAGE IS HANDED A CHECKED-IN ttir GOLDEN. `emit_ktir` reads
//! one, so a fixture change or a pass change cannot appear in it; reading a golden dump while
//! chasing a matmul shape cost this tree an afternoon and four reverted fixes.
//!
//! # ⚠️ WHAT THE ORACLE IS, AND WHAT IT IS NOT
//!
//! The executor is scratchy's `ktir-emulator`, and what it rounds was read out of its code rather
//! than taken from its description:
//!
//! * **A CONTRACTION accumulates in f32 and rounds ONCE**, at the end -- `linalg.matmul` ->
//!   `blas::sgemm_rowmajor` with an `f32` accumulator (`src/blas.rs:42`), then a single
//!   `Tile::compute` (`src/dialects/linalg.rs:463`). Never f64, never per-K-step. An `outs` operand
//!   adds a second round (`linalg.rs:494`), so `tl.dot(a, b)` is one and `tl.dot(a, b, acc)` is two.
//! * ⚠️ **EVERY ELEMENTWISE STEP ROUNDS**, to the tile's own dtype: `Tile::compute` calls
//!   `round_to_dtype` on every result (`ktir-core:src/tile.rs:151`), and its own comment says an f16
//!   chain "rounds after each step the way NumPy does, rather than accumulating in f32 and rounding
//!   only at store". So "accumulates WIDE" is true of the K loop and WOULD BE WRONG read as "rounds
//!   only at the store" -- the derived envelopes are a COUNT OF f16 ROUNDS along each path.
//! * **Round-to-nearest-EVEN** (`ktir-core:src/codec.rs:67,74`), where the device is round-half-up.
//!   Hence `U_F16 = 2^-11`, half an ULP, against the pod's `u = 2^-10`.
//! * **`rsqrt` is `1.0 / x.sqrt()` in f32 and THERE IS NO SIGMOID AT ALL** -- `math.rs`'s
//!   registration table has no entry for one. So `eps_rsqrt` and `eps_sigmoid`, which DOMINATE the
//!   pod's device bounds, are error terms this oracle never commits.
//!
//! So it is a **CORRECTNESS** oracle, not a fidelity one: it proves the program computes the right
//! FUNCTION, and it does NOT prove the device's rounding matches.
//!
//! That is exactly the right instrument for the bug class this tree has. `dxp_standalone` compiles
//! and executes no arithmetic, and two decoders passed that gate for HOURS while contracting
//! `q @ k` where the program says `q @ k.T`. A wrong contraction, a dropped transpose or a reduction
//! on the wrong axis is a change of FUNCTION, and this sees all three.
//!
//! ⛔ THE COROLLARY, AND IT IS NOT A DETAIL: a tolerance for this executor must be RE-DERIVED, never
//! copied from `test/pod/*_eps_derivation.py`. MEASURED COST OF GETTING THAT WRONG -- `rmsnorm`'s
//! drop-the-max-energy-stick control has a worst ratio of **2.18** against the re-derived bound and
//! **0.28** against the device's `k+ = 0.0477`. The same control, under a copied bound, is VACUOUS.
//! See [`bounds`].
//!
//! # WHAT A TEST HERE OWES
//!
//! Three rules, each of which this tree has been burned by:
//!
//! 1. **THE TOLERANCE IS DERIVED BEFORE THE RUN**, from the arithmetic, with the algebra written
//!    down. A bound chosen after seeing the error is not a test. [`bounds`] holds the derivations.
//! 2. **EVERY COMPARISON HAS A CONTROL THAT FAILS.** A comparison that cannot fail proves nothing.
//!    [`mutants`] holds the algorithm mutants and the BRACKETING control -- perturb the
//!    largest-bound element by 1.5x its bound (must exceed) and by 0.5x (must stay within), which is
//!    what proves the threshold sits where the derivation puts it rather than merely somewhere.
//! 3. **REPORT `max|err|` AND THE BOUND**, never "passes". [`Comparison`] carries both.

pub mod bounds;
pub mod data;
pub mod mutants;

use ktir_core::arena::Arena;
use ktir_core::ir::{IRFunction, Ssa};
use ktir_emulator::interpreter::{Arg, Output};
use triton_frontend::codegen::KernelSpec;
use triton_frontend::target::Target;
use triton_frontend::{codegen, opt};
use triton_ktir_superdsc::cases;

/// What went wrong, and at which leg. Same shape as `triton_ktir_superdsc::Refusal`, and for the
/// same reason: a driver tabulating `REFUSED <stage> <message>` shows a stage-less failure as a
/// blank row, which is how a footprint mismatch went unread once already.
#[derive(Debug, Clone)]
pub struct Refusal {
    pub stage: &'static str,
    pub message: String,
}

impl std::fmt::Display for Refusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "REFUSED at {}: {}", self.stage, self.message)
    }
}

fn refuse(stage: &'static str, message: impl Into<String>) -> Refusal {
    Refusal { stage, message: message.into() }
}

pub type Result<T> = std::result::Result<T, Refusal>;

/// One configuration, lowered as far as the executor's input language.
pub struct Lowered {
    /// THEIR function type, arena-backed by [`Arena::global`] so it is `'static` -- which is the
    /// lifetime the emulator's entry points take. `Arena::global` exists for exactly this and is
    /// documented upstream as "THE ONE ARENA A BUILD HAS".
    pub func: IRFunction<'static>,
    /// The launch contract the configuration was compiled at, kept so a test can read the
    /// `tl.constexpr`s it derived its tolerance from rather than restating them.
    pub spec: KernelSpec,
    pub grid: Vec<i64>,
}

/// `.py` SOURCE to their `IRFunction`. Every leg is a value; nothing reads a golden.
pub fn lower(config: &str) -> Result<Lowered> {
    let (spec, grid, _program) = cases::case(config)
        .ok_or_else(|| refuse("case", format!("`{config}` is not one of: {}", cases::ALL.join(" "))))?;

    // LEG 1 -- Triton `.py` -> raw ttir -> the six `make_ttir` passes.
    let src = std::fs::read_to_string(&spec.file)
        .map_err(|e| refuse("read", format!("cannot read {}: {e}", spec.file)))?;
    let mut tt = codegen::compile(&src, &spec, Target::spyre())
        .map_err(|e| refuse("codegen", e.to_string()))?;
    opt::make_ttir(&mut tt).map_err(|e| refuse("make_ttir", e.to_string()))?;

    // LEG 2 -- ttir value -> our KTIR value. No text in between.
    let mut m = triton_ktir::from_ttir::convert(&tt)
        .map_err(|e| refuse("from_ttir", e.to_string()))?;
    triton_ktir::make_ktir(&mut m, &grid).map_err(|e| refuse("make_ktir", e.to_string()))?;
    triton_ktir::passes::to_ktir::run(&mut m, &grid)
        .map_err(|e| refuse("to_ktir", e.to_string()))?;
    refold_matmul_accumulators(&mut m);

    // LEG 3 -- our module -> THEIR `IRFunction`, in the process-wide arena so it is `'static`.
    let func = triton_ktir::passes::to_ktir_emit::lower(&m, Arena::global())
        .map_err(|e| refuse("to_ktir_emit", e.to_string()))?;
    Ok(Lowered { func, spec, grid })
}

/// Fold `decompose_matmul_accumulators`' rewrite BACK to the accumulate form before the emulator
/// sees it — the inverse of `to_ktir.rs`'s pass, scoped to THIS crate's lowering only.
///
/// ⛔ WHY THIS CRATE FOLDS INSTEAD OF THE PASS MOVING. The decompose is a DOOR artifact: the
/// SuperDSC whole-function door reads two inputs (`n_in = 2`), so it needs the accumulation
/// visible as an `arith.addf`. But the EMULATOR charges LX per SSA tile, and the decompose's
/// three-op shape (`splat` + `matmul` + `addf`) makes it charge the fresh `dot` result WHILE the
/// old accumulator is still live for the addf — MEASURED on granite tiled_k: trip 2's down
/// matmul charges `1597440 + 524288 > 2097152` (the `[64, 4096]` dot over a base of 1597440),
/// where the program's own statement (`tl.dot(a, b, acc)`, one tile, consumed on last use) fit
/// and was card-verified at `508e52e74` (2026-09-19, max|err| 2.148438e-2 under a derived
/// bound). The emulator models the PROGRAM's residency, not the door's spelling of it, so this
/// crate runs the pass (the door's requirement is not ours to remove from `to_ktir::run`) and
/// then folds the exact shape it minted.
///
/// The fold is SEMANTICS-PRESERVING and FAIL-CLOSED to that exact shape: a `linalg.matmul`
/// whose `outs` is the pass's own zero splat, followed by the pass's own `arith.addf(dot, acc)`
/// with `dot` dead after the addf, becomes `matmul ins(a, b) outs(acc)` — the form
/// `accumulate_outs` already executes and the consume-on-last-use model already charges as ONE
/// tile. Anything else (a nonzero splat, a live `dot`, an addf the pass did not mint) is left
/// untouched, so a program that states a genuine materialize-then-add still pays it.
fn refold_matmul_accumulators(m: &mut triton_ktir::ir::Module) {
    use triton_ktir::ir::OpKind;
    use triton_ktir::passes::dot_to_linalg::is_zero_const;
    use triton_ktir::passes::walk;
    use triton_ktir::Ssa as KSsa;

    // The zero `outs` of a decomposed matmul: either the pass's scalar zero constant or its
    // `tensor.splat` of one (the two spellings `decompose_matmul_accumulators` mints).
    let is_pass_zero = |m: &triton_ktir::ir::Module, v: KSsa| -> bool {
        if is_zero_const(m, v) {
            return true;
        }
        m.def_of(v).is_some_and(|d| {
            d.kind == OpKind::TensorSplat
                && d.operands.first().is_some_and(|&s| is_zero_const(m, s))
        })
    };

    for path in triton_ktir::passes::walk::paths(m).into_iter().rev() {
        // `walk::block_mut` on the MATMUL's path addresses the block holding all three ops (they
        // are spliced into one block by the decompose); index arithmetic is on that block.
        let Some(mm) = triton_ktir::passes::walk::at(m, &path).cloned() else { continue };
        if mm.kind != OpKind::LinalgMatmul || mm.operands.len() != 3 {
            continue;
        }
        let zero_t = mm.operands[2];
        if !is_pass_zero(m, zero_t) {
            continue;
        }
        let Some(dot) = mm.results.first().copied() else { continue };
        let block = match triton_ktir::passes::walk::block_mut(m, &path) {
            Some(blk) => blk,
            None => continue,
        };
        let idx = path.index();
        // The addf must be the matmul's own `arith.addf(dot, acc)` and `dot` must be dead after
        // it — the exact consumer shape the decompose minted. A `dot` read anywhere else means
        // this is a genuine materialize-then-add program, not the pass's artifact.
        let Some(addf) = block
            .iter()
            .skip(idx + 1)
            .find(|o| o.kind == OpKind::ArithAddf && o.operands.first() == Some(&dot))
            .cloned()
        else {
            continue;
        };
        let acc = addf.operands[1];
        let dot_live_after = block[idx + 1..]
            .iter()
            .any(|o| o != &addf && o.operands.contains(&dot));
        if dot_live_after {
            continue;
        }
        // Rewrite: the matmul accumulates into `acc` and takes the addf's result name; the addf
        // itself is dropped. The rewrite is POSITIONAL, not a range delete — the addf may be far
        // from the matmul (other ops can sit between them and stay), and the splat/constant that
        // fed `zero_t` is dropped only when the matmul was its LAST reader (it always is when the
        // decompose minted it, but the check keeps the fold safe on any other program).
        let sum = addf.results.first().copied();
        let Some(sum) = sum else { continue };
        let addf_idx = block
            .iter()
            .position(|o| o == &addf)
            .expect("the addf came from this block");
        // The matmul: outs := acc, result := the addf's name (the name every consumer reads).
        {
            let o = &mut block[idx];
            o.operands[2] = acc;
            o.results[0] = sum;
            if let Some(ty) = addf.result_types.first() {
                o.result_types[0] = ty.clone();
            }
        }
        // Drop the addf.
        block.remove(addf_idx);
        // Drop the splat/constant chain behind `zero_t` when the matmul was its last reader —
        // dead code once nothing references it (the emulator would charge the splat's tile).
        let mut dead: Vec<KSsa> = vec![zero_t];
        while let Some(&v) = dead.last() {
            let Some(def) = m.def_of(v) else { break };
            let still_read = walk::paths(m).into_iter().any(|p| {
                walk::at(m, &p).is_some_and(|o| o.operands.contains(&v))
            });
            if still_read {
                break;
            }
            if def.kind == OpKind::TensorSplat || def.kind == OpKind::ArithConstant {
                dead.push(def.operands[0]);
            } else {
                break;
            }
        }
        dead.pop(); // `dead` now holds only the provably-dead defs; the last push was not one
        // Remove the dead defs from whichever block holds them (they may be in this block).
        for v in dead {
            for p in walk::paths(m).into_iter().rev() {
                if walk::at(m, &p).is_some_and(|o| o.results.first() == Some(&v)) {
                    if let Some(blk) = walk::block_mut(m, &p) {
                        let i = p.index();
                        if i < blk.len() && blk[i].results.first() == Some(&v) {
                            blk.remove(i);
                        }
                    }
                    break;
                }
            }
        }
    }
}

/// One argument's binding: the fixture's own parameter NAME, and the bytes for it.
///
/// ⛔ THE BINDING IS POSITIONAL, AND THAT IS A CLAIM THAT GETS CHECKED. `to_ktir_emit::lower`
/// builds `IRFunction::arguments` from the body region's arguments in order, and those are the
/// `.py` kernel's POINTER parameters in declaration order (the `tl.constexpr`s never become
/// arguments -- they are resolved at codegen). So argument *i* of the `IRFunction` is pointer
/// parameter *i* of the kernel. [`execute`] REFUSES a count mismatch rather than binding a
/// prefix, because binding a prefix is a silently wrong answer: every buffer after the missing
/// one would be off by one and the run would still produce numbers.
pub struct Binding {
    /// The `.py` parameter name (`desc_x`, `desc_w`, `desc_o`, ...). Diagnostic and ordering only.
    pub name: String,
    /// Raw bytes already in the argument's own dtype -- f16 little-endian for a `*fp16`, i32 for a
    /// `*i32`. Handed to [`Arg::TensorBytes`], which copies them into HBM with NO conversion, so
    /// the executor sees the same bytes the pod's torch wrote.
    pub bytes: Vec<u8>,
    pub shape: Vec<usize>,
    pub dtype: ktir_core::dtypes::DType,
}

/// What came back: every tensor argument, read out of HBM after the run.
pub struct Run {
    /// Keyed by the `.py` parameter name, so a caller asks for `desc_o` and not for `%arg2`.
    pub outputs: std::collections::HashMap<String, Output>,
}

impl Run {
    /// One output tensor's values, widened to f64 for comparison. `Output::data` is already f32 --
    /// the executor's own dtype-agnostic view -- and widening f32 to f64 is exact, so this loses
    /// nothing and lets the bound algebra stay in f64 throughout.
    pub fn f64s(&self, arg: &str) -> Result<Vec<f64>> {
        let o = self.outputs.get(arg).ok_or_else(|| {
            let mut have: Vec<&str> = self.outputs.keys().map(|s| s.as_str()).collect();
            have.sort();
            refuse("readback", format!("no output for `{arg}`; the run returned {have:?}"))
        })?;
        Ok(o.data.iter().map(|&v| v as f64).collect())
    }
}

/// Execute a lowered configuration against `bindings`.
///
/// ⚖️ WHY `execute_function_with_latency` AND NOT `execute_function`. Upstream's
/// `interpreter::execute_function` is `pub(crate)`, and `execute_function_in` requires a caller to
/// build the memory hierarchy and marshal the arguments itself (`marshal_inputs` is private). The
/// latency-tracked twin is PUBLIC, takes the same `&[(Ssa, Arg)]`, and does the same marshal, run
/// and read-back -- it just also meters each op. We drop the report. This is a door choice, not an
/// arithmetic one: both call `comm_sched::execute_with_communication` on the same operations.
pub fn execute(l: &Lowered, bindings: &[Binding]) -> Result<Run> {
    let params: Vec<Ssa> = l.func.arguments.iter().map(|(s, _)| *s).collect();
    if params.len() != bindings.len() {
        return Err(refuse(
            "bind",
            format!(
                "the lowered function takes {} argument(s) and {} binding(s) were supplied \
                 ({:?}). REFUSING rather than binding a prefix: every buffer after a missing one \
                 would be off by one and the run would still produce numbers.",
                params.len(),
                bindings.len(),
                bindings.iter().map(|b| b.name.as_str()).collect::<Vec<_>>(),
            ),
        ));
    }

    let mut args: Vec<(Ssa, Arg)> = Vec::with_capacity(bindings.len());
    for (ssa, b) in params.iter().zip(bindings) {
        let want = b.shape.iter().product::<usize>() * b.dtype.bytes_per_elem();
        if b.bytes.len() != want {
            return Err(refuse(
                "bind",
                format!(
                    "`{}` is {:?} of {:?} = {want} bytes, but {} bytes were supplied",
                    b.name,
                    b.shape,
                    b.dtype,
                    b.bytes.len()
                ),
            ));
        }
        args.push((
            *ssa,
            Arg::TensorBytes { data: b.bytes.clone(), shape: b.shape.clone(), dtype: b.dtype },
        ));
    }

    let (raw, _latency) = ktir_emulator::interpreter::execute_function_with_latency(
        &l.func,
        &args,
        ktir_emulator::latency::HardwareConfig::default(),
    )
    .map_err(|e| refuse("execute", e))?;

    // Re-key the read-back by the `.py` parameter name. The executor keys by `Ssa`, which is an
    // opaque `u32`: a test reporting "no output for %arg2" sends its reader to the wrong file.
    let mut outputs = std::collections::HashMap::new();
    for (ssa, b) in params.iter().zip(bindings) {
        if let Some(o) = raw.get(ssa) {
            outputs.insert(b.name.clone(), o.clone());
        }
    }
    Ok(Run { outputs })
}

/// The verdict on one comparison, carrying BOTH numbers.
///
/// ⛔ "PASSES" IS NOT A RESULT. A report that says a config passed cannot be checked against a
/// later run, cannot show a bound that was loose, and cannot show an error that crept. So every
/// field a reader needs is here and [`Comparison::report`] prints all of them.
#[derive(Debug, Clone)]
pub struct Comparison {
    pub config: String,
    pub elements: usize,
    /// `max |got - reference|` over every element.
    pub max_abs_err: f64,
    /// ⛔⛔ THE BOUND AT THE ELEMENT WHERE `max_abs_err` OCCURRED -- **NOT** at the worst-ratio
    /// element, and not the largest bound anywhere.
    ///
    /// THIS FIELD EXISTED AND WAS WRONG, AND THE WRONGNESS WAS THE FLATTERING KIND. It used to be
    /// assigned inside the worst-RATIO branch, so `report()` printed `max|err|` from one element
    /// beside a bound from another. On `rmsnorm_granite` that read as
    /// `max|err| = 7.812500e-3   bound = 6.093244e-3` -- an apparent 1.28x VIOLATION sitting next to
    /// `exceeded 0/262144`, which is incoherent and would have been quoted as a violation or, worse,
    /// as evidence the bound was fine. The two numbers must come from the SAME element or they are
    /// not a comparison.
    pub bound_at_max_err: f64,
    pub reference_at_max_err: f64,
    pub got_at_max_err: f64,
    pub max_err_index: usize,
    /// The bound at the WORST-RATIO element, which is the one the verdict is actually about.
    pub bound_at_worst: f64,
    /// The largest `|got - reference| / bound` over every element. This, not `max_abs_err`, is what
    /// the verdict is: a sign-aware envelope's threshold varies per element.
    pub worst_ratio: f64,
    pub worst_index: usize,
    pub reference_at_worst: f64,
    pub got_at_worst: f64,
    /// How many elements exceeded their own bound.
    pub exceeded: usize,
}

impl Comparison {
    /// Compare `got` against `reference` under a per-element sign-aware envelope.
    pub fn new(config: &str, got: &[f64], reference: &[f64], env: &bounds::Envelope) -> Result<Self> {
        if got.len() != reference.len() {
            return Err(refuse(
                "compare",
                format!(
                    "the run produced {} element(s) and the reference has {}: a comparison over \
                     the shorter of two lengths is a comparison that cannot see a shape defect",
                    got.len(),
                    reference.len()
                ),
            ));
        }
        if reference.is_empty() {
            return Err(refuse("compare", "an empty reference: a comparison over nothing passes"));
        }
        let mut c = Comparison {
            config: config.to_string(),
            elements: reference.len(),
            max_abs_err: 0.0,
            bound_at_max_err: 0.0,
            reference_at_max_err: 0.0,
            got_at_max_err: 0.0,
            max_err_index: 0,
            bound_at_worst: 0.0,
            worst_ratio: 0.0,
            worst_index: 0,
            reference_at_worst: 0.0,
            got_at_worst: 0.0,
            exceeded: 0,
        };
        let mut max_abs = (0.0f64, 0usize);
        for (i, (&g, &r)) in got.iter().zip(reference).enumerate() {
            let err = (g - r).abs();
            let bound = env.abs_bound(r);
            if err > max_abs.0 {
                max_abs = (err, i);
                // Recorded HERE, in the max-|err| branch, so the pair `max_abs_err` /
                // `bound_at_max_err` is one element's two numbers.
                c.bound_at_max_err = bound;
                c.reference_at_max_err = r;
                c.got_at_max_err = g;
                c.max_err_index = i;
            }
            // A NaN on either side must not compare "within bound" by falling through a `>`.
            let ratio = if !err.is_finite() || !bound.is_finite() {
                f64::INFINITY
            } else if bound == 0.0 {
                if err == 0.0 { 0.0 } else { f64::INFINITY }
            } else {
                err / bound
            };
            if ratio > 1.0 {
                c.exceeded += 1;
            }
            if ratio > c.worst_ratio {
                c.worst_ratio = ratio;
                c.worst_index = i;
                c.bound_at_worst = bound;
                c.reference_at_worst = r;
                c.got_at_worst = g;
            }
        }
        c.max_abs_err = max_abs.0;
        // A run where every element matched EXACTLY has no worst-ratio element, so the report would
        // show element 0 as though it were the interesting one. Point it at the max-|err| element,
        // which in that case is also element 0's equal -- every error being zero.
        if c.worst_ratio == 0.0 {
            let i = max_abs.1;
            c.worst_index = i;
            c.reference_at_worst = reference[i];
            c.got_at_worst = got[i];
            c.bound_at_worst = env.abs_bound(reference[i]);
        }
        // ⛔ AND THE SAME HOLE ON THE max-|err| SIDE, WHICH `embedding_granite` WAS THE FIRST
        // CONFIGURATION TO FALL INTO. The three `*_at_max_err` fields are assigned inside
        // `if err > max_abs.0`, and for a run that matches BIT FOR BIT that branch never fires -- so
        // they keep their initialised zeros and `report()` prints
        // `max|err| = 0.000000e0 at [0] ... (ref +0.000000e0, got +0.000000e0)`. Which reads as "the
        // reference is zeros", i.e. as a vacuous comparison, when it is the opposite: an EXACT match
        // on 1,048,576 non-zero elements (`ref_out.bin[0]` is +16.421875). Same class as the bug this
        // struct's own doc records -- a number paired with the wrong element -- and the same fix:
        // state element `max_abs.1`'s actual numbers.
        if c.max_abs_err == 0.0 {
            let i = max_abs.1;
            c.max_err_index = i;
            c.reference_at_max_err = reference[i];
            c.got_at_max_err = got[i];
            c.bound_at_max_err = env.abs_bound(reference[i]);
        }
        Ok(c)
    }

    /// Within bound at EVERY element. Not "close enough on average" -- an averaged verdict is how a
    /// single wrong stick hides in 8192 lanes.
    pub fn within_bound(&self) -> bool {
        self.exceeded == 0 && self.worst_ratio <= 1.0
    }

    /// Both numbers, per rule 3 -- AND EACH PAIRED WITH THE ELEMENT IT CAME FROM.
    ///
    /// Two lines, because there are two different elements to report and putting their numbers on one
    /// line is what made an earlier version of this read as a violation it was not: the largest
    /// ABSOLUTE error and the largest error RELATIVE TO ITS OWN ALLOWANCE are generally different
    /// elements under a per-element envelope, and the VERDICT is the second one.
    pub fn report(&self) -> String {
        format!(
            "{:<26} n={:<8} max|err| = {:.6e} at [{}] vs its own bound {:.6e}              (ref {:+.6e}, got {:+.6e})\n{:<26} {:<10} VERDICT: worst ratio = {:.4} at [{}]              (err {:.6e} / bound {:.6e}, ref {:+.6e})   exceeded {}/{}",
            self.config,
            self.elements,
            self.max_abs_err,
            self.max_err_index,
            self.bound_at_max_err,
            self.reference_at_max_err,
            self.got_at_max_err,
            "",
            "",
            self.worst_ratio,
            self.worst_index,
            (self.got_at_worst - self.reference_at_worst).abs(),
            self.bound_at_worst,
            self.reference_at_worst,
            self.exceeded,
            self.elements,
        )
    }
}
