//! ⭐⭐ EVERY CONSTANT, MEASURED BY COMPARING TWO EMISSIONS.
//!
//! ⛔⛔ EXPRESSING A CONSTANT ONLY PUTS A VALUE IN THE OUTPUT; EXPLOITING IT CHANGES WHAT THE OUTPUT
//! IS. Three versions of this crate carried constants that nothing branched on, and each looked
//! correct until someone asked what would differ if the value changed. So every test here lowers
//! the SAME tape twice, differing in exactly one constant, and asserts a STRUCTURAL difference —
//! an op that is present in one and absent in the other, not a bound that reads differently.
//!
//! ⛔ AND THE VALUES ARE CARRIED, NOT THE RATIOS. "the decode program is smaller" passes on an
//! emitter that dropped the wrong loop. Each assertion names the op it expects to vanish and counts
//! it.

use deeptools::arch::{Arch, Dd2};
use deeptools::bridges::subtile_to_dataflow_ir::node::{
    Cols, Node, Operand, Residence, Rows, Segment,
};
use deeptools::bridges::subtile_to_dataflow_ir::tape;
use deeptools::generated::{DataType, OpFunc};
use deeptools::islands::dataflow_ir::GroupId;
use deeptools::islands::dataflow_ir::dialects::{Op, affine, agen, dataflow, scf, vectorchain};
use deeptools::model::Model;
use deeptools::workload::Workload;

/// A model whose hidden size is a whole number of sticks — 2048 is 32 sticks of 64.
struct Aligned;
impl Model for Aligned {
    const QUERY_HEADS: u32 = 32;
    const KV_HEADS: u32 = 8;
    const HEAD_DIM: u32 = 64;
    const HIDDEN: u32 = 2048;
    const LAYERS: u32 = 40;
    const FFN: u32 = 8192;
    const VOCAB: u32 = 49152;
}

/// The same model with a RAGGED hidden size: 2050 is 32 sticks and two elements.
struct Ragged;
impl Model for Ragged {
    const QUERY_HEADS: u32 = 32;
    const KV_HEADS: u32 = 8;
    const HEAD_DIM: u32 = 64;
    const HIDDEN: u32 = 2050;
    const LAYERS: u32 = 40;
    const FFN: u32 = 8192;
    const VOCAB: u32 = 49152;
}

/// A decode rung: one row.
struct Decode;
impl Workload for Decode {
    const ROWS: u32 = 1;
    const ACTIVE_CAP: u32 = 64;
}

/// A prefill rung: 96 rows, the ceiling of `PREFILL_RUNGS`.
struct Prefill;
impl Workload for Prefill {
    const ROWS: u32 = 96;
    const ACTIVE_CAP: u32 = 64;
}

/// A decode rung whose cache span is one stick — nothing to walk.
struct NarrowCache;
impl Workload for NarrowCache {
    const ROWS: u32 = 1;
    const ACTIVE_CAP: u32 = 64;
}

/// A decode rung whose cache span is many sticks, and too large for the scratchpad.
struct WideCache;
impl Workload for WideCache {
    const ROWS: u32 = 1;
    const ACTIVE_CAP: u32 = 8192;
}

/// A rung that sweeps NO resident prefix — `ActiveCap::NONE` resolved to zero.
///
/// ⭐ A REAL BUNDLE, NOT A DEGENERATE ONE: every single-chunk prompt is one, because a chunk whose
/// `start == 0` has no resident prefix to attend. It is also the common TTFT case.
struct NoSweep;
impl Workload for NoSweep {
    const ROWS: u32 = 96;
    const ACTIVE_CAP: u32 = 0;
}

/// A two-node tape: one op reading a weight from the HBM, one reading an activation already staged.
fn tape_of(op_func: OpFunc) -> Vec<Node> {
    let hbm = Operand {
        rows: Rows(1),
        cols: Cols(64),
        format: DataType::Sen169Fp16,
        at: Residence::Hbm {
            segment: Segment(1),
            offset: deeptools::arch::Bytes(0),
        },
    };
    let staged = Operand {
        rows: Rows(1),
        cols: Cols(64),
        format: DataType::Sen169Fp16,
        at: Residence::Lx {
            offset: deeptools::arch::Elements(0),
        },
    };
    vec![
        Node {
            op_func,
            format: DataType::Sen169Fp16,
            inputs: vec![hbm, staged],
            output: staged,
        },
        Node {
            op_func,
            format: DataType::Sen169Fp16,
            inputs: vec![staged, staged],
            output: staged,
        },
    ]
}

/// Count the ops of one kind anywhere in a run, regions included.
fn count(run: &deeptools::islands::dataflow_ir::Run<Dd2>, want: fn(&Op) -> bool) -> usize {
    fn walk(ops: &[Op], want: fn(&Op) -> bool, n: &mut usize) {
        for op in ops {
            if want(op) {
                *n += 1;
            }
            match op {
                Op::Affine(affine::Op::For { body, .. })
                | Op::Dataflow(dataflow::Op::ProgramUnit { body, .. }) => walk(body, want, n),
                Op::Scf(scf::Op::If {
                    body, else_body, ..
                }) => {
                    walk(body, want, n);
                    walk(else_body, want, n);
                }
                Op::Agen(agen::Op::CompositeLoadAndStore(t)) => walk(&t.body, want, n),
                _ => {}
            }
        }
    }
    let mut n = 0;
    for program in &run.programs {
        // ⭐ THE PREAMBLE AND EVERY PROGRAM UNIT. A count that read only one of them would miss the
        // ops the other holds — and the split between them is exactly what these tests measure.
        walk(&program.preamble, want, &mut n);
        for unit in program.units.iter() {
            walk(&unit.body, want, &mut n);
        }
    }
    n
}

const IS_FOR: fn(&Op) -> bool = |op| matches!(op, Op::Affine(affine::Op::For { .. }));
const IS_MASK: fn(&Op) -> bool = |op| {
    matches!(
        op,
        Op::VectorChain(vectorchain::Op::CreateAffineMask { .. })
    )
};
const IS_SELECT: fn(&Op) -> bool = |op| {
    matches!(
        op,
        Op::VectorChain(vectorchain::Op::ElementWiseSelection { .. })
    )
};
const IS_TRANSFER: fn(&Op) -> bool =
    |op| matches!(op, Op::Agen(agen::Op::CompositeLoadAndStore(_)));
const IS_SYNC: fn(&Op) -> bool = |op| matches!(op, Op::Dataflow(dataflow::Op::SyncRecv { .. }));
const IS_SEND: fn(&Op) -> bool = |op| matches!(op, Op::Dataflow(dataflow::Op::Send { .. }));
const IS_RECEIVE: fn(&Op) -> bool = |op| matches!(op, Op::Dataflow(dataflow::Op::Receive { .. }));
const IS_LOAD: fn(&Op) -> bool = |op| matches!(op, Op::Agen(agen::Op::VectorLoad { .. }));
const IS_STORE: fn(&Op) -> bool = |op| matches!(op, Op::Agen(agen::Op::VectorStore { .. }));

/// ⭐⭐ EVERY OPERAND CROSSES THE WIRE, AND THE TWO ENDS AGREE ON HOW MANY.
///
/// ⛔⛔ THIS IS THE TEST THE TYPE CANNOT BE. `Received` stops the compute reading the MOVER's views,
/// which is what dbo-opt refused — but it is equally satisfied by a mover that sends into the void
/// and a compute that receives nothing and computes on an empty operand list. Only counting catches
/// that, and only counting AGAINST THE NODE'S OWN ARITY: "sends == receives" passes at zero.
///
/// ⛔ AND THE LOADS MOVED, THEY DID NOT MULTIPLY. One `agen.vector_load` per operand, in the mover;
/// a compute that still loaded would show more loads than sends.
#[test]
fn every_operand_reaches_the_compute_on_a_wire() {
    let tape = tape_of(OpFunc::Add);
    let run = tape::compile::<Dd2, Aligned, Decode>(&tape, GroupId(0)).expect("lowers");

    // Two nodes, two inputs each: four operands. And each node's RESULT crosses a second wire.
    let operands: usize = tape.iter().map(|node| node.inputs.len()).sum();
    assert_eq!(operands, 4, "the fixture tape has two nodes of two inputs");
    let results = tape.len();

    // ⭐⭐ TWO WIRES PER NODE, AND EVERY END IS SPENT. The loader sends one vector per operand to
    // the compute; the compute sends one result to the store. A `Link` hands out its two ends once,
    // so a send with no matching receive is not a program that can be built — but a program with
    // NO wires at all would still satisfy that, which is why the counts are against the node's own
    // arity and the tape's own length rather than against each other.
    assert_eq!(
        count(&run, IS_SEND),
        operands + results,
        "{operands} operand sends from the loader, plus {results} result sends from the compute"
    );
    assert_eq!(
        count(&run, IS_RECEIVE),
        operands + results,
        "and every one of them is taken off the wire at the other end"
    );

    // ⛔ THE LOADS MOVED, THEY DID NOT MULTIPLY. One `agen.vector_load` per operand, in the loader;
    // a compute that still loaded for itself would show more loads than operands.
    assert_eq!(
        count(&run, IS_LOAD),
        operands,
        "one load per operand, in the loader"
    );

    // ⛔ AND EVERY RESULT IS WRITTEN. A compute whose result is never stored is never lowered at
    // all — see `Computed`; the failure surfaced as "Dangling non-compute op has no use" naming a
    // RECEIVE, three passes away from the cause.
    assert_eq!(
        count(&run, IS_STORE),
        results,
        "one `agen.vector_store` per node, in the store unit"
    );
}

/// ⭐ EVERY NODE OF THE TAPE BECOMES A PROGRAM. The tape is the deliverable; a lowering that
/// emitted fewer programs than the tape has nodes has silently skipped work.
#[test]
fn every_node_of_the_tape_becomes_a_program() {
    let tape = tape_of(OpFunc::Add);
    let run = tape::compile::<Dd2, Aligned, Decode>(&tape, GroupId(0)).expect("the tape lowers");
    assert_eq!(
        run.programs.len(),
        tape.len(),
        "the tape has {} nodes; the run must have that many programs",
        tape.len()
    );
    // And each names the node it came from, so the run's order is the tape's order.
    for (at, program) in run.programs.iter().enumerate() {
        assert_eq!(
            program.name.index.0 as usize, at,
            "programs must be in tape order"
        );
    }
}

/// ⭐ IS_DECODE REMOVES THE ROW NEST — it does not set its bound to one.
#[test]
fn is_decode_removes_the_row_nest() {
    let tape = tape_of(OpFunc::Add);
    let decode = tape::compile::<Dd2, Aligned, Decode>(&tape, GroupId(0)).expect("lowers");
    let prefill = tape::compile::<Dd2, Aligned, Prefill>(&tape, GroupId(0)).expect("lowers");

    let at_decode = count(&decode, IS_FOR);
    let at_prefill = count(&prefill, IS_FOR);

    // ⛔ CARRY THE VALUE. Two nodes, one row loop each PER UNIT at prefill, none at decode.
    //
    // ⭐⭐ THE FACTOR IS THE UNIT COUNT, AND IT IS THREE. The loader, the compute and the store walk
    // ONE iteration space from three points along it — the loader reads and sends, the compute
    // receives and computes and sends on, the store receives and writes — so a node with a row loop
    // has one in each. A count of 4 here would mean a unit was missing its nest; a count of 2, that
    // two units were sharing one.
    assert_eq!(at_decode, 0, "a decode rung must emit NO row loop at all");
    assert_eq!(
        at_prefill, 6,
        "a prefill rung emits one row loop per node per unit: 2 nodes x 3 units"
    );
}

/// ⭐ STICK_ALIGNED REMOVES THE MASK AND EVERYTHING THAT CONSUMES IT.
#[test]
fn stick_alignment_removes_the_mask_and_its_consumer() {
    let tape = tape_of(OpFunc::Add);
    let aligned = tape::compile::<Dd2, Aligned, Decode>(&tape, GroupId(0)).expect("lowers");
    let ragged = tape::compile::<Dd2, Ragged, Decode>(&tape, GroupId(0)).expect("lowers");

    assert_eq!(
        count(&aligned, IS_MASK),
        0,
        "a hidden size of {} is {} whole sticks of {}: no lane needs predicating",
        Aligned::HIDDEN,
        Aligned::HIDDEN / Dd2::SLICES_PER_STICK,
        Dd2::SLICES_PER_STICK
    );
    assert_eq!(count(&aligned, IS_SELECT), 0, "and nothing consumes it");

    // 2050 = 32 sticks and 2 elements, so both the mask and its selection appear, once per node.
    assert_eq!(count(&ragged, IS_MASK), 2, "a ragged row needs a predicate");
    assert_eq!(
        count(&ragged, IS_SELECT),
        2,
        "and an element_wise_selection consuming it"
    );
}

/// ⭐ THE SK BUCKET CHANGES THE ATTENTION WALK — both whether it exists and what is inside it.
#[test]
fn the_sk_bucket_changes_the_cache_walk() {
    let tape = tape_of(OpFunc::Batchmatmul);
    let narrow = tape::compile::<Dd2, Aligned, NarrowCache>(&tape, GroupId(0)).expect("lowers");
    let wide = tape::compile::<Dd2, Aligned, WideCache>(&tape, GroupId(0)).expect("lowers");

    // ⛔ CARRY THE VALUE. active_cap=64 is exactly one stick of 64, so KV_VECTORS is 1 and there is
    // nothing to step over: no walk at all. active_cap=8192 is 128 sticks, so each node gets one —
    // in EACH of its two units, which walk the same space from opposite ends.
    assert_eq!(
        count(&narrow, IS_FOR),
        0,
        "a cache span of one vector has no walk to emit"
    );
    assert_eq!(
        count(&wide, IS_FOR),
        6,
        "a 128-vector span emits one walk per attention node per unit: 2 nodes x 3 units"
    );

    // And the wide rung's cache does not fit the scratchpad, so each step waits for its own slice.
    assert_eq!(
        count(&narrow, IS_SYNC),
        0,
        "a staged cache needs no per-step wait"
    );
    assert_eq!(
        count(&wide, IS_SYNC),
        6,
        "a streamed cache waits once per node's walk, in each of the three units that walk it"
    );
}

/// ⭐⭐ A RUNG THAT SWEEPS NOTHING EMITS NO WALK — not a walk of zero trips.
///
/// ⛔ THIS WAS A REAL BUG, FOUND BY THE ACCEPTANCE BUILD. `ActiveCap` is a sentinel type whose
/// `FULL` is 0 and `NONE` is `u32::MAX`, and the call site passed `.get()` — the sentinel — where a
/// resolved extent belongs, so the door saw `active_cap 0` and `active_cap 4294967295` and refused
/// every rung. Resolving it surfaced the second half: `NO_CACHE_WALK` was `KV_VECTORS == 1`, which
/// is FALSE at zero vectors, so a bundle sweeping no resident prefix would have emitted a loop that
/// runs never — the exact shape `audit_op_work` refuses elsewhere.
#[test]
fn a_rung_that_sweeps_nothing_emits_no_walk() {
    let tape = tape_of(OpFunc::Batchmatmul);
    let none = tape::compile::<Dd2, Aligned, NoSweep>(&tape, GroupId(0)).expect("lowers");

    // ⛔ CARRY THE VALUE. 96 rows is a prefill rung, so the ROW nest is present — one per node, in
    // each of its two units — and the cache walk is absent. Counting only "fewer loops" would pass
    // on losing the wrong one.
    assert_eq!(
        count(&none, IS_FOR),
        6,
        "the row nest survives at 96 rows in all three units; only the cache walk goes"
    );
    assert_eq!(
        count(&none, IS_SYNC),
        0,
        "and with no walk there is no per-step wait"
    );
}

/// ⭐ AND THE RESIDENCE DECIDES WHETHER A TRANSFER EXISTS AT ALL.
///
/// ⛔⛔ THIS IS THE ONE THAT WAS WRONG BEFORE. Every operand used to be addressed into the LX, so
/// no transfer was ever emitted and the programs read memory nothing filled. An HBM operand must
/// produce a `composite_load_and_store`; an operand already staged must not.
#[test]
fn an_hbm_operand_is_transferred_and_a_staged_one_is_not() {
    let tape = tape_of(OpFunc::Add);
    let run = tape::compile::<Dd2, Aligned, Decode>(&tape, GroupId(0)).expect("lowers");

    // Node 0 reads one HBM operand and one staged; node 1 reads two staged. One transfer in total.
    assert_eq!(
        count(&run, IS_TRANSFER),
        1,
        "exactly the HBM-resident operand crosses into the scratchpad"
    );

    // And the HBM is named as a global unit — no core, no corelet — in every program.
    let text = deeptools::islands::dataflow_ir::print::run(&run);
    assert!(
        text.contains(r#"dataflow.get_unit {name = "hbm", type = "hbm"}"#),
        "the HBM must be bound as a global unit"
    );
    assert!(
        !text.contains(r#"core = 0 : i32, name = "hbm""#),
        "a global unit carries neither core nor corelet"
    );
}
