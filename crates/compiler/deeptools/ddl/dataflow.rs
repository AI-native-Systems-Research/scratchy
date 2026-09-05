//! Which statements an op-func's dataflow runs — the input to bridge 1's splice.
//!
//! A template file declares several `ddl.operation_bind`s (`summeanmaxexx2.ddl` binds `sum`, `sumnonstick`,
//! `mean`, `max`, …) and its `ddl.dataflow` block gates statements behind `ddl.if`. So "op1's dataflow" is a
//! walk of that block with the conditions evaluated against ONE op-bind.
//!
//! ⭐ THERE ARE THREE KINDS OF CONDITION AND ONLY ONE IS DECIDABLE HERE.
//!
//! | condition | example | verdict |
//! |---|---|---|
//! | the active op-bind, or an alias over it | `ddl.if (%mean_op)`, `ddl.if (%max_or_maxnonstick_op)` | `then` only |
//! | a DIFFERENT op-bind | `ddl.if (%prod_nonstick_op)` when compiling `mean` | `else` only |
//! | not an op-bind at all | `ddl.if (%psum_end)`, `ddl.if (%cond_not_first_loopreduce_dims)` | BOTH arms |
//!
//! The third kind is a LOOP POSITION, not an op-func property: a loop that really iterates executes the
//! first-iteration form once and the steady-state form afterwards, so both arms are statements this op-func
//! can run. Treating them as decided would drop half the loop body.
//!
//! Deliberately not a port of the old crate's `ddl_resolve.rs`: its dim binding, datastage exploration,
//! parametric trip counts and layout/alias tables were dxp RECOVERING what scratchy states.

use crate::ast::{AttrValue, Operand, Operation};

/// What a `ddl.if`'s condition resolves to once the active op-bind is known.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Verdict {
    /// Names the active op-bind: the `then` arm runs.
    Then,
    /// Names a DIFFERENT op-bind: only the `else` arm runs.
    Else,
    /// The op-bind facts do not settle it, so both arms are kept.
    Both,
    /// ⛔ A LOOP-POSITION PREDICATE — see [`Truth::PerLoopPosition`]. The arms are MUTUALLY EXCLUSIVE and
    /// which one runs depends on the loop's extent, which is not known until expansion.
    PerLoopPosition(CondTree),
}

/// WHICH TRIP A `ddl.condition` NAMES — `value_expr="first"` / `"last"`.
///
/// ⛔ THE VENDORED SET IS CLOSED: `eq`+`first` × 73, `eq`+`last` × 67, `ne`+`last` × 2 (a negated
/// `Last`). `processCondition` also admits integer value expressions (`ddl_conversion.cpp:260-276`); no
/// vendored template states one, so that spelling is a parse refusal rather than a variant nothing
/// constructs.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Place {
    /// `value_expr="first"` — `iv == lower bound` (`SNControlFlowLowering.cpp:105-108`).
    First,
    /// `value_expr="last"` — `iv == upper bound - 1` (`SNControlFlowLowering.cpp:100-104`).
    Last,
}

/// ⛔⛔⛔ THE PREDICATE'S OWN SHAPE, KEPT — `ddl.condition` leaves under their `_and`/`_or`/`_not`
/// connectives, with every op-bind leaf already resolved away.
///
/// The reference carries the same structure into the schedule: a live condition becomes
/// `LoopCondComposite`'s `twoLevelOrOfAnds_` with a `negated_` flag (`ddl_conversion.cpp:317-328`), and
/// `SNControlFlowLowering::constructConditionalOperation` rebuilds it as nested `scf.if`s over the loop
/// induction variables (`SNControlFlowLowering.cpp:66-200`).
///
/// 🛑 A SINGLE `Sense` FLAG SAT HERE ONCE, AND IT COULD NOT SAY WHICH LOOP. `bmm.ddl:281`'s
/// `%cond_first_blkaccum2loop` is an AND of two different loops' firsts; collapsed to one bit, the guard
/// then recorded the `ddl.if`'s own depth — a depth no loop registers — and expansion resolved every
/// live-loop branch as if the loop were dropped: only the `cond_first` matmul variant survived, one south
/// per dispatch, 64 against the PE's 32, and rows 5-7 of every core deadlocked.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CondTree {
    /// A trip of the loop this label names.
    Position {
        /// The `loop_label=` attribute, resolved to a depth at the `ddl.if` site.
        label: String,
        /// Which trip.
        place: Place,
    },
    /// `ddl.condition_and`.
    All(Vec<CondTree>),
    /// `ddl.condition_or`.
    Any(Vec<CondTree>),
    /// `ddl.condition_not`.
    Not(Box<CondTree>),
}

/// [`CondTree`] with each label resolved to the depth of the `ddl.loop` STATEMENT it names — the index
/// bridge 1's walk registers that loop's trips under.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ResolvedCond {
    /// A trip of the loop whose statement sits at this depth.
    Position {
        /// The named loop's own statement depth.
        loop_depth: u16,
        /// Which trip.
        place: Place,
    },
    /// `ddl.condition_and`.
    All(Vec<ResolvedCond>),
    /// `ddl.condition_or`.
    Any(Vec<ResolvedCond>),
    /// `ddl.condition_not`.
    Not(Box<ResolvedCond>),
}

impl CondTree {
    /// Resolve every label to the depth of the enclosing loop that carries it.
    ///
    /// ⛔ AN ENCLOSING LOOP, BY CONSTRUCTION. `processCondition` looks labels up in a global table
    /// (`ddl_conversion.cpp:296-317`), but a position predicate over a loop that does not enclose the
    /// branch has no induction variable to compare — `constructConditionalOperation` reads
    /// `getInductionVar()` of the named loop, which must be in scope. Every vendored guard names an
    /// enclosing loop; one that does not is refused here with both lists printed.
    fn resolve(&self, enclosing: &[(Option<&str>, u16)]) -> ResolvedCond {
        match self {
            Self::Position { label, place } => {
                let depth = enclosing
                    .iter()
                    .rev()
                    .find_map(|(carried, depth)| (*carried == Some(label.as_str())).then_some(*depth))
                    .unwrap_or_else(|| {
                        panic!(
                            "a loop-position predicate names {label:?}, which encloses no statement of \
                             this branch; the enclosing loops here are {enclosing:?}"
                        )
                    });
                ResolvedCond::Position {
                    loop_depth: depth,
                    place: *place,
                }
            }
            Self::All(list) => {
                ResolvedCond::All(list.iter().map(|cond| cond.resolve(enclosing)).collect())
            }
            Self::Any(list) => {
                ResolvedCond::Any(list.iter().map(|cond| cond.resolve(enclosing)).collect())
            }
            Self::Not(inner) => ResolvedCond::Not(Box::new(inner.resolve(enclosing))),
        }
    }
}

/// The SSA name an op binds, if it binds one.
fn result_of(op: &Operation) -> Option<&str> {
    op.results.first().map(String::as_str)
}

/// THE SAME, KEYED BY THE BIND'S OWN SSA NAME.
///
/// ⛔ AN OP-FUNC KEY CANNOT NAME A CHAINED ARM. `bmm.ddl:79-80` declares `%stradd_op` and `%stradd2_op` with ONE
/// `opFuncName` — `stridedadd` — and they read DIFFERENT operands (`%resadd` against `%resadd2`), so a lookup by
/// op-func answers with the first and silently gives the second stage the first's tensor.
pub fn bind_tensors_named<'a>(module: &'a [Operation], bound: &str) -> Option<BoundTensors<'a>> {
    bind_tensors_where(module, &|op| op.results.iter().any(|r| r == bound))
}

fn bind_tensors_where<'a>(
    module: &'a [Operation],
    matches_bind: &dyn Fn(&Operation) -> bool,
) -> Option<BoundTensors<'a>> {
    for op in module {
        if op.name != "ddl.operation_bind" {
            continue;
        }
        if !matches_bind(op) {
            continue;
        }
        // `([types], [inputs], [outputs], [internals])` — four bracketed lists, in that order.
        let lists: Vec<&'a [String]> = op
            .operands
            .iter()
            .filter_map(|operand| match operand {
                Operand::RefList(names) => Some(names.as_slice()),
                Operand::Ref(_) => None,
            })
            .collect();
        // ⭐ THE INTERNALS LIST IS OPTIONAL, AND THE CENSUS SAYS SO: 118 of the 222 vendored binds state four
        // lists and 104 state three. A template that allocates nothing of its own has no fourth list, which is an
        // absent LIST rather than an empty one — so three is a shape, not a defect.
        return Some(match lists.as_slice() {
            [types, inputs, outputs, internals] => BoundTensors {
                types,
                inputs,
                outputs,
                internals,
            },
            [types, inputs, outputs] => BoundTensors {
                types,
                inputs,
                outputs,
                internals: &[],
            },
            several => panic!(
                "a `ddl.operation_bind` binding {:?} states {} bracketed lists; the vendored shapes are three \
                 (types, inputs, outputs) and four (with internals) — `bmm.ddl:52` is the four-list form",
                op.results,
                several.len()
            ),
        });
    }
    None
}

/// The four lists one `ddl.operation_bind` names, in position order.
#[derive(Debug, Clone, Copy)]
pub struct BoundTensors<'a> {
    /// ⭐⭐ THE DATA FORMATS THIS BIND ADMITS — `[%type_fp16, %type_fp32]`, as `ddl.type` SSA names.
    ///
    /// ⛔⛔ AND IT IS WHAT DECIDES WHICH TEMPLATE SERVES AN OP-FUNC, not just documentation. Two
    /// templates bind `exp`: `unary_parallel.ddl:30` for `%type_fp32` and `unary_pipeline.ddl:19`
    /// for `%type_fp16`, each with its own precision-specific `.smc`. dxp picks between them by
    /// MATCHING this list against the op's own format (`ddl_conversion.cpp:2137-2144`, "Check
    /// supported data formats for op"); reading only the arch tag hands an fp16 `exp` the fp32
    /// kernel and its fp32 constants.
    pub types: &'a [String],
    /// The op's inputs, in the order island 1's `Compute::inputs` holds them.
    pub inputs: &'a [String],
    /// The op's outputs — every vendored bind states exactly one.
    pub outputs: &'a [String],
    /// Tensors that exist only INSIDE the template (`%ptsum_fp`, `%pesum`), so they are not the op's at all.
    pub internals: &'a [String],
}

/// The `opFuncName` each `ddl.operation_bind` declares, with the SSA name it binds.
pub fn op_binds(module: &[Operation]) -> Vec<(&str, &str)> {
    let mut out = Vec::new();
    for op in module {
        if op.name != "ddl.operation_bind" {
            continue;
        }
        let Some(bound) = result_of(op) else { continue };
        for (key, value) in &op.attrs {
            if key == "opFuncName"
                && let AttrValue::String(name) = value
            {
                out.push((bound, name.as_str()));
            }
        }
    }
    out
}

/// WHETHER A `ddl.if`'s CONDITION HOLDS FOR THIS ACTIVE SET — three-valued, because two of the three answers
/// are facts and the third is the honest absence of one.
///
/// ⛔⛔ `Unknown` IS NOT A DEFAULT, IT IS THE LOOP-POSITION CASE. `ddl.condition(%in){loop_label=…,
/// condition="eq", value_expr="last"}` (`bmm.ddl:305`) asks whether this is the last trip of a loop, which is a
/// RUN-TIME fact — bridge 1 emits both arms for it because the machine takes both over the loop's life. An
/// op-bind condition is a different thing entirely: it is decided when the DSC names its `computeOp_`s.
#[derive(Debug, Clone, PartialEq, Eq)]
enum Truth {
    True,
    False,
    /// The op-bind facts do not settle it — the name is not declared in this walk's reach.
    Unknown,
    /// ⛔⛔⛔ A LOOP-POSITION PREDICATE, WHICH IS A DIFFERENT FACT FROM "UNKNOWN" AND MUST NOT SHARE ITS
    /// VALUE.
    ///
    /// `ddl.condition(%in){loop_label="accum2bottom_loop", condition="eq", value_expr="first"}`
    /// (`bmm.ddl:239`) is decidable — just not HERE. `processCondition` resolves it to a compile-time bool
    /// when the loop's dim is DROPPED, and `FIRST` and `LAST` both take `condVal = 0` so an `EQ` makes BOTH
    /// true (`ddl_conversion.cpp:274-292`); otherwise it becomes a `LoopCond` carried into the schedule
    /// (`:317-319`) and the branch survives into codegen. Neither answer is available at parse time, because
    /// both need the dim's extent.
    ///
    /// 🛑 SHARING `Unknown`'s VALUE MEANT THE WALK EMITTED BOTH ARMS OF A MUTUALLY EXCLUSIVE BRANCH,
    /// ADJACENT AND IN TEXTUAL ORDER. `bmm.ddl:256-269` is
    /// `if(first) { 0 + store arf } else { if(last) { arf + south } else { arf + store arf } }` — exactly one
    /// runs per trip, and the PT program came out with all three back to back. The second then read the ARF
    /// the first had written, three instructions before the write-back delay would land it, and every matmul
    /// refused with an untyped accumulator.
    PerLoopPosition(CondTree),
}

impl Truth {
    fn not(self) -> Truth {
        match self {
            Truth::True => Truth::False,
            Truth::False => Truth::True,
            // 🔑 The negation of a loop-position predicate is still one — the tree keeps the negation as
            // structure, the same `negated_` the reference flips (`ddl_conversion.cpp:322-328`).
            Truth::PerLoopPosition(tree) => Truth::PerLoopPosition(CondTree::Not(Box::new(tree))),
            Truth::Unknown => Truth::Unknown,
        }
    }
}

/// EVALUATE ONE CONDITION NAME AGAINST THE ACTIVE BIND SET.
///
/// ⭐⭐ THE CONNECTIVES ARE THE WHOLE POINT, AND WITHOUT THEM A BARE MATMUL EMITS THE FUSED ONE'S STATEMENTS.
/// `bmm.ddl:307-309` writes
///
/// ```text
///   %is_any_aux_op = ddl.condition_or(%stradd_op, %stradd2_op, %bn_op, %bias_op, %relu_op)
///   %cond_aux_ops_iter = ddl.condition_and(%cond_last_inaccum, %is_any_aux_op)
///   %cond_psum_or_aux_ops_iter = ddl.condition_or(%psum_op, %cond_aux_ops_iter)
/// ```
///
/// and gates the epilogue section on the last of those. For a matmul with no fused arm every one of those binds
/// is INACTIVE, so `%is_any_aux_op` is false, so the `condition_and` is false whatever the loop position says,
/// so the whole section is `Else`. Reading only "is this name a bind or an alias of one" answered `Unknown` for
/// all three and emitted BOTH arms — an extra PE→SFP-LRF `data_transfer` on every matmul the port compiled.
///
/// ⛔ AND `condition_not` IS WHY A CONJUNCTION MUST BE ABLE TO COME OUT *TRUE* TOO, not only false:
/// `%not_psum = ddl.condition_not(%psum_op)` (`:310`) is TRUE for every op that is not a partial reduction, and
/// leaving it `Unknown` keeps an arm the op does not run.
fn truth_of(all: &[&Operation], active: &[&str], binds: &[&str], name: &str) -> Truth {
    truth_with_depth(all, active, binds, name, 0)
}

/// The recursion, bounded. The condition graph is a DAG in every vendored template, and a bound is what makes
/// that a checked property rather than an assumption — a cycle would otherwise be a build that never ends.
fn truth_with_depth<'a>(
    all: &[&'a Operation],
    active: &[&str],
    binds: &[&str],
    name: &str,
    depth: u32,
) -> Truth {
    const MAX_DEPTH: u32 = 64;
    if depth > MAX_DEPTH {
        panic!(
            "the condition graph is more than {MAX_DEPTH} deep at {name}; the vendored templates chain \
             `ddl.condition_or`/`_and`/`_not` a handful of levels, so this is a cycle rather than a deep nest"
        );
    }
    // ⭐ AN OP-BIND IS THE BASE CASE, AND ITS ANSWER IS EXACT. Active means this walk runs it; a bind the
    // template declares and this walk does not run is FALSE, not unknown.
    if active.contains(&name) {
        return Truth::True;
    }
    if binds.contains(&name) {
        return Truth::False;
    }
    let Some(op) = all.iter().find(|op| op.results.iter().any(|r| r == name)) else {
        return Truth::Unknown;
    };
    let operands = |op: &'a Operation| -> Vec<&'a str> {
        op.operands
            .iter()
            .flat_map(|operand| match operand {
                Operand::Ref(r) => vec![r.as_str()],
                Operand::RefList(list) => list.iter().map(String::as_str).collect(),
            })
            .collect()
    };
    match op.name.as_str() {
        "ddl.condition_or" => {
            let mut unknown = false;
            // 🔑 AN OR OVER LOOP-POSITION OPERANDS IS ITSELF PER-POSITION, unless something else already
            // makes it true — and EVERY position operand is kept, as a disjunct of the tree.
            let mut positions: Vec<CondTree> = Vec::new();
            for operand in operands(op) {
                match truth_with_depth(all, active, binds, operand, depth + 1) {
                    Truth::True => return Truth::True,
                    Truth::False => {}
                    Truth::Unknown => unknown = true,
                    Truth::PerLoopPosition(tree) => positions.push(tree),
                }
            }
            // 🛑 An UNKNOWN operand beside a loop-position one leaves the whole thing unknown: the
            // extent cannot settle a name this walk never found.
            match (unknown, positions.len()) {
                (true, _) => Truth::Unknown,
                (false, 0) => Truth::False,
                (false, 1) => Truth::PerLoopPosition(positions.remove(0)),
                (false, _) => Truth::PerLoopPosition(CondTree::Any(positions)),
            }
        }
        "ddl.condition_and" => {
            let mut unknown = false;
            // 🔑 An AND over loop-position operands is per-position too, unless something else already
            // makes it false — and EVERY position operand is kept, as a conjunct of the tree.
            let mut positions: Vec<CondTree> = Vec::new();
            for operand in operands(op) {
                match truth_with_depth(all, active, binds, operand, depth + 1) {
                    Truth::False => return Truth::False,
                    Truth::True => {}
                    Truth::Unknown => unknown = true,
                    Truth::PerLoopPosition(tree) => positions.push(tree),
                }
            }
            match (unknown, positions.len()) {
                (true, _) => Truth::Unknown,
                (false, 0) => Truth::True,
                (false, 1) => Truth::PerLoopPosition(positions.remove(0)),
                (false, _) => Truth::PerLoopPosition(CondTree::All(positions)),
            }
        }
        "ddl.condition_not" => match operands(op).as_slice() {
            [only] => truth_with_depth(all, active, binds, only, depth + 1).not(),
            several => panic!(
                "a `ddl.condition_not` binding {name} states {} operands; a negation has one",
                several.len()
            ),
        },
        // ⭐ A LOOP-POSITION PREDICATE IS THE ONE THING THAT IS GENUINELY NOT DECIDABLE HERE — and it is
        // its OWN answer, not an unknown. The leaf carries WHICH loop and WHICH trip; the closed vendored
        // set is `eq`+`first`/`last` and `ne`+`last` — see [`Place`].
        "ddl.condition" => {
            let attr = |key: &str| -> &str {
                op.attrs
                    .iter()
                    .find_map(|(name, value)| match (name == key, value) {
                        (true, AttrValue::String(text)) => Some(text.as_str()),
                        _ => None,
                    })
                    .unwrap_or_else(|| {
                        panic!("a `ddl.condition` binding {name} states no string {key}=")
                    })
            };
            let label = attr("loop_label").to_owned();
            let place = match (attr("condition"), attr("value_expr")) {
                ("eq", "first") => {
                    return Truth::PerLoopPosition(CondTree::Position {
                        label,
                        place: Place::First,
                    });
                }
                ("eq", "last") => Place::Last,
                ("ne", "last") => {
                    return Truth::PerLoopPosition(CondTree::Not(Box::new(CondTree::Position {
                        label,
                        place: Place::Last,
                    })));
                }
                (condition, value_expr) => panic!(
                    "a `ddl.condition` binding {name} states condition={condition:?} \
                     value_expr={value_expr:?}; the vendored set is eq+first, eq+last and ne+last"
                ),
            };
            Truth::PerLoopPosition(CondTree::Position { label, place })
        }
        // 🛑 Anything else binding a condition name is a statement this walk has no rule for, which is a
        // different thing again from either.
        _ => Truth::Unknown,
    }
}

/// WHICH SIDE OF A LOOP-POSITION BRANCH — the parse-time twin of `template::Arm`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Arm {
    /// The `then` body.
    Then,
    /// The `else` body.
    Else,
}

/// ONE LOOP-POSITION BRANCH a statement sits inside: the predicate tree, labels resolved to the depths
/// of the loops they name, and the arm.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArmOf {
    /// The predicate, each named loop resolved to its own statement depth.
    pub cond: ResolvedCond,
    /// Which side of it.
    pub arm: Arm,
}

/// ONE CONSTRUCT ENCLOSING A STATEMENT — a loop, or one arm of an undecided branch.
///
/// ⛔⛔ DEPTH AND GUARD SEPARATELY CANNOT SAY WHICH NESTS OUTSIDE WHICH, and the templates contain
/// both orders. `broadcast_ops.ddl:189-192` puts a `ddl.loop` INSIDE an `else` arm, while
/// `bmm.ddl:182-187` puts a `ddl.if` inside a loop. A statement carrying "depth 2, guard [g]" is in
/// both cases the same pair of numbers, and emitting loops-then-ifs turns `if { loop { S } }` into
/// `loop { if { S } }` — a loop that runs unconditionally where the template says it may not run at
/// all. So the ORDER is recorded, not the two counts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Enclosing {
    /// A `ddl.loop`, named by the index of its own statement in the walked tape.
    Loop {
        /// Where the loop's own statement sits in the output.
        stmt: usize,
    },
    /// One arm of a branch the walk could not decide.
    Arm(ArmOf),
}

/// ONE STATEMENT A WALK YIELDED — the operation, how many loops enclose it, and which loop-position
/// branches it sits inside.
///
/// ⭐ A STRUCT AND NOT A TUPLE. Three heterogeneous members threaded through six signatures is a shape
/// nothing checks: `(op, depth, guard)` and `(op, guard, depth)` differ only by which one the reader
/// happens to be looking at, and clippy calls the type "very complex" for the same reason.
#[derive(Debug, Clone)]
pub struct Walked<'a> {
    /// The statement itself.
    pub op: &'a Operation,
    /// How many `ddl.loop`s enclose it.
    pub depth: u16,
    /// EVERY CONSTRUCT ENCLOSING IT, OUTERMOST FIRST AND IN ORDER — loops and undecided arms
    /// interleaved as the template writes them. See [`Enclosing`].
    ///
    /// ⛔ A SEPARATE `guard: Vec<ArmOf>` SAT BESIDE THIS AND IS GONE. It listed the same arms with
    /// the loops between them removed, so it could not say whether a loop was inside an arm or the
    /// other way round — and 895 of the vendored nestings are the first order. `path` carries every
    /// arm `guard` did, in the position the template puts it, so keeping both was carrying one fact
    /// twice with only one of the copies complete.
    pub path: Vec<Enclosing>,
}

/// EVERY OPERATION IN EVERY REGION, in source order — what a scan for declarations has to look at.
///
/// ⛔ A DECLARATION IS NOT ONLY A TOP-LEVEL STATEMENT. A `ddl.condition_or` or a `ddl.operation_bind` reference may
/// sit inside a `ddl.loop` or a `ddl.if` arm, and a scan that stops at the top level treats it as absent — which
/// makes an op-bind condition undecidable and keeps both arms.
fn flattened(ops: &[Operation]) -> Vec<&Operation> {
    let mut out = Vec::new();
    fn descend<'a>(ops: &'a [Operation], out: &mut Vec<&'a Operation>) {
        for op in ops {
            out.push(op);
            descend(&op.body, out);
            descend(else_body(op), out);
        }
    }
    descend(ops, &mut out);
    out
}

/// Every statement one FUSION SET of op-binds needs, flattened in source order — DECLARATIONS AND DATAFLOW BOTH.
///
/// ⭐⭐ THE SET IS THE KEY, NOT ONE BIND. A bare matmul is the set `[%mm_fp8_op]`; a matmul with a fused
/// stridedadd epilogue is `[%mm_fp8_op, %stradd_op]`, and the two walks differ STRUCTURALLY rather than by a
/// union — `ddl.if (%stradd_op)` (`bmm.ddl:335`) takes its `then` arm in one and its (empty) `else` in the
/// other. Appending the arm's statements to the bare walk would put the epilogue's compute after the matmul's
/// stores instead of inside the accumulation loop, which is why this is one walk and not two.
///
/// ⭐ THE WHOLE MODULE, NOT JUST THE `ddl.dataflow` REGION, and that was a real hole. A template declares names
/// before its dataflow and the dataflow references them: `bmm.ddl:111` binds `%kertensor_lx_allocation` with
/// `ddl.get_external_data_transfer_allocation`, `ddl.dataflow` opens at `:122`, and `:177` uses that name as a
/// `ddl.unit`'s allocation. Walking only the dataflow gave bridge 2 a tape in which that operand was bound
/// nowhere, so it could not tell a register from a port. `bmm.ddl` declares 78 names that way.
///
/// ⛔ AND THE DECLARATIONS ARE NOT ALWAYS SHARED, which is why they go through the same verdict logic rather than
/// into a per-template table. `rope.ddl:33` wraps two `ddl.constraint`s in `ddl.if(%rope_p2_op)` — a declaration
/// section gated on an op-bind. One walk per op-bind is correct for that by construction; a per-template table
/// would have had to special-case it, or be quietly wrong.
///
/// Depth-first, `then` before `else`, descending every region — the order the unit executes them in, which is the
/// order bridge 1 splices them.
pub fn statements_for<'a>(
    module: &'a [Operation],
    dataflow: &'a [Operation],
    active_binds: &[&'a str],
) -> Vec<Walked<'a>> {
    // ⛔ THE WHOLE MODULE *AND* THE DATAFLOW, because a condition may be declared in either. `rope.ddl:33` wraps
    // constraints in `ddl.if(%rope_p2_op)` at declaration level, while `broadcast_ops.ddl:173` declares a
    // `ddl.condition_or` inside two `ddl.loop`s — so the graph a condition is resolved against is every operation
    // of every region.
    let mut all = flattened(module);
    all.extend(flattened(dataflow));
    // ⛔⛔ THE RAW BIND LIST, NOT AN ALIAS-CLOSED ONE, AND THAT DISTINCTION COST A SILENT WRONG ANSWER. An
    // earlier `op_bind_names` closed this set over `ddl.condition_or`, so `%is_any_aux_op` (`bmm.ddl:307`) was
    // itself treated as a bind — and [`truth_of`]'s base case then answered FALSE for it without ever evaluating
    // the disjunction, which dropped the `stridedadd` arm from the FUSED walk too. The two walks came out
    // byte-identical, which reads exactly like "fusion changes nothing" rather than like a bug.
    //
    // ⭐ THE CLOSURE IS NOT LOST, IT IS COMPUTED PROPERLY: `truth_of` descends the `_or`/`_and`/`_not` graph, so
    // an alias's value follows from its operands instead of from being mistaken for one of them.
    let binds: Vec<&str> = op_binds(module)
        .into_iter()
        .map(|(bound, _)| bound)
        .collect();
    let mut out = Vec::new();
    let scope = WalkScope {
        all: &all,
        active: active_binds,
        bind_names: &binds,
    };
    walk(module, &scope, 0, &[], &[], &[], &mut out);
    out
}

/// How many mutually exclusive branches this walk has FLATTENED — see [`Truth::PerLoopPosition`].
///
/// ⭐ A LEDGER, BECAUSE THE ARM IS A STAND-IN AND A STAND-IN WITH NO NUMBER ON IT IS INVISIBLE. The
/// resolution needs the loop's extent, which is not known until expansion; until it is carried there, this
/// says how much of the templates' branch structure is being discarded.
pub static FLATTENED_EXCLUSIVE_BRANCHES: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// How many have been flattened so far.
#[must_use]
pub fn flattened_exclusive_branches() -> usize {
    FLATTENED_EXCLUSIVE_BRANCHES.load(std::sync::atomic::Ordering::Relaxed)
}

/// 🛑 Of those, how many had BOTH arms populated — the ones that emit two statements where one runs.
pub static FLATTENED_WITH_BOTH_ARMS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// How many of the flattened branches carried an `else`.
#[must_use]
pub fn flattened_with_both_arms() -> usize {
    FLATTENED_WITH_BOTH_ARMS.load(std::sync::atomic::Ordering::Relaxed)
}

/// The walk's RECURSION-INVARIANT context — the same three slices at every level, bundled so the
/// recursive calls cannot transpose them (`active` and `bind_names` share a type, which is exactly
/// the swap a positional argument list cannot catch).
struct WalkScope<'a, 'b> {
    all: &'b [&'a Operation],
    active: &'b [&'b str],
    bind_names: &'b [&'b str],
}

fn walk<'a>(
    stmts: &'a [Operation],
    scope: &WalkScope<'a, '_>,
    depth: u16,
    // 🔑 The loops enclosing these statements, outermost first: each one's `label=` and the depth its
    // own statement sits at — what a loop-position predicate resolves its `loop_label` against.
    enclosing: &[(Option<&'a str>, u16)],
    // 🔑 The loop-position branches enclosing these statements, outermost first.
    guard: &[ArmOf],
    // 🔑 Every enclosing construct in ORDER — what says whether a loop is inside an arm or the
    // other way round.
    path: &[Enclosing],
    out: &mut Vec<Walked<'a>>,
) {
    for op in stmts {
        match op.name.as_str() {
            "ddl.if" => match verdict(op, scope.all, scope.active, scope.bind_names) {
                Verdict::Then => walk(&op.body, scope, depth, enclosing, guard, path, out),
                Verdict::Else => walk(else_body(op), scope, depth, enclosing, guard, path, out),
                Verdict::Both => {
                    walk(&op.body, scope, depth, enclosing, guard, path, out);
                    walk(else_body(op), scope, depth, enclosing, guard, path, out);
                }
                // ⛔⛔⛔ BOTH ARMS, AND THAT IS WRONG — STATED HERE RATHER THAN HIDDEN IN `Both`.
                //
                // These arms are MUTUALLY EXCLUSIVE: `processCondition` either resolves the predicate to a
                // compile-time bool, when the loop's dim is dropped and `FIRST`/`LAST` both become true
                // (`ddl_conversion.cpp:274-292`), or carries it into the schedule as a `LoopCond` (`:317-319`)
                // so the branch survives into codegen. It never emits both.
                //
                // ⛔ THE EXTENT IS WHAT DECIDES, AND IT IS NOT KNOWN AT PARSE TIME — it is what the tiler
                // computes per op. So the resolution belongs at EXPANSION and the question has to be carried
                // there, which is the port this arm is standing in for. Until it is, the arms are emitted in
                // textual order and a PT program can contain an ARF read three instructions after its write.
                Verdict::PerLoopPosition(tree) => {
                    FLATTENED_EXCLUSIVE_BRANCHES.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    // 🔑 An `if` with no `else` flattens HARMLESSLY: keeping the `then` arm on every trip is
                    // wrong about WHEN it runs, but it emits no instruction that contradicts another. A
                    // branch with BOTH arms populated emits two statements where one runs.
                    if !else_body(op).is_empty() {
                        FLATTENED_WITH_BOTH_ARMS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    }
                    // ⭐ BOTH ARMS ARE STILL WALKED, AND EACH NOW CARRIES WHICH ONE IT IS. Resolving the
                    // predicate needs the loops' EXTENTS, which the tiler computes per op — so the question
                    // travels to expansion instead of being answered wrongly here. See `template::Arm`.
                    let cond = tree.resolve(enclosing);
                    let mut inner = guard.to_vec();
                    inner.push(ArmOf {
                        cond: cond.clone(),
                        arm: Arm::Then,
                    });
                    let mut then_path = path.to_vec();
                    then_path.push(Enclosing::Arm(ArmOf {
                        cond: cond.clone(),
                        arm: Arm::Then,
                    }));
                    walk(&op.body, scope, depth, enclosing, &inner, &then_path, out);
                    let mut otherwise = guard.to_vec();
                    otherwise.push(ArmOf {
                        cond: cond.clone(),
                        arm: Arm::Else,
                    });
                    let mut else_path = path.to_vec();
                    else_path.push(Enclosing::Arm(ArmOf {
                        cond,
                        arm: Arm::Else,
                    }));
                    walk(
                        else_body(op),
                        scope,
                        depth,
                        enclosing,
                        &otherwise,
                        &else_path,
                        out,
                    );
                }
            },
            // ⭐ A LOOP IS EMITTED AND THEN DESCENDED INTO, which an earlier version of this did not do — it
            // walked the body and dropped the loop, so the NEST was invisible in the tape. That matters: a
            // transfer's chunk size is the extent of the `chunk_datastage` its enclosing loop pairs, and the
            // templates carry 450 `ddl.loop`s. The depth restores the tree from a flat list, pre-order.
            "ddl.loop" | "ddl.parametric_loop" => {
                out.push(Walked {
                    op,
                    depth,
                    path: path.to_vec(),
                });
                // The loop's body is enclosed BY this loop, named by where its own statement landed.
                let stmt = out.len() - 1;
                // 🔑 The loop's `label=`, if it states one — what a `ddl.condition`'s `loop_label` names.
                let label = op
                    .attrs
                    .iter()
                    .find_map(|(key, value)| match (key.as_str(), value) {
                        ("label", AttrValue::String(text)) => Some(text.as_str()),
                        _ => None,
                    });
                let mut nested = enclosing.to_vec();
                nested.push((label, depth));
                let mut deeper = path.to_vec();
                deeper.push(Enclosing::Loop { stmt });
                walk(&op.body, scope, depth + 1, &nested, guard, &deeper, out);
            }
            // Containers the walk passes through — they are not loops and do not nest anything.
            "ddl.dataflow" | "ddl.transformations" => {
                walk(&op.body, scope, depth, enclosing, guard, path, out)
            }
            // The op-binds are the KEY this walk is run per; emitting them would put the question in the answer.
            "ddl.operation_bind" => {}
            // Every other statement is one this op-func runs. Bridge 1 decides which KINDS carry
            // instructions; this walk only decides which ones APPLY.
            _ => out.push(Walked {
                op,
                depth,
                path: path.to_vec(),
            }),
        }
    }
}

/// A `ddl.if`'s `else` region, empty when it has none.
fn else_body(op: &Operation) -> &[Operation] {
    op.else_body.as_deref().unwrap_or(&[])
}

fn verdict(op: &Operation, all: &[&Operation], active: &[&str], bind_names: &[&str]) -> Verdict {
    let Some(Operand::Ref(cond)) = op.operands.first() else {
        // No condition operand: nothing to decide against, so the body is unconditional.
        return Verdict::Then;
    };
    match truth_of(all, active, bind_names, cond.as_str()) {
        Truth::True => Verdict::Then,
        Truth::False => Verdict::Else,
        Truth::Unknown => Verdict::Both,
        Truth::PerLoopPosition(sense) => Verdict::PerLoopPosition(sense),
    }
}
