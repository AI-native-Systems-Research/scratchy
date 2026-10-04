// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Licensed under the MIT terms in the crate root.

//! ⛔⛔ THE CONTROLS FOR THE ONE OP CLASS WHERE A WRONG ANSWER LOOKS PLAUSIBLE.
//!
//! `embedding_granite` is the only configuration whose kernel gathers, and a gather's failure modes
//! all produce WELL-FORMED OUTPUT: an off-by-one row, a token-major/head-major swap, a stale index,
//! a block offset applied to the wrong axis. Every one of them is the right shape, the right dtype
//! and the right distribution — the table's rows are independent `randn` draws, so the WRONG row is
//! statistically indistinguishable from the right one. Nothing but a per-element comparison against
//! the row the ids actually name can see any of it.
//!
//! The arithmetic, meanwhile, is ONE MULTIPLY (`embedding.py:117`). So the tolerance is not where
//! this configuration's risk lives and a tight bound is not what makes it verified — these two
//! controls are:
//!
//! 1. [`the_real_gathers_off_by_one_row_exceeds_the_embedding_bound`] — `mutants::embedding`'s
//!    authored mutant, wired to the CHECKED-IN ids and table rather than to a synthetic pair, and
//!    gated against the FIXTURE'S OWN `ref_out.bin`.
//! 2. [`an_absent_table_row_expands_to_zero_and_that_diverges`] — the property that makes storing
//!    the 384 MiB table as 2 MiB sound, MEASURED rather than asserted in a docstring.
//!
//! ⚖️ WHY A SEPARATE FILE FROM `executed.rs`. That file's one sweep is deliberately one table over
//! every configuration; these are two controls about ONE of them, each of which allocates the
//! expanded 384 MiB table. Keeping them here makes the sweep's report readable and lets a reader
//! run the gather's own controls alone.

use triton_numeric::{bounds, data, mutants, Comparison};

/// The configuration whose bytes both controls read. `embedding_granite_bm128` shares them
/// (`data_dir`), so the bytes are stated once and there is nothing to choose between.
const CONFIG: &str = "embedding_granite";

/// One row of the expanded f16 table, widened to f64.
///
/// Read out of the BINDING BYTES — the same buffer [`triton_numeric::execute`] copies into HBM —
/// through `ktir_core::codec::decode`, which is the decoder the executor's own read-back uses. So a
/// control cannot disagree with the run about what an f16 bit pattern means.
fn table_row(bytes: &[u8], row: usize, d_model: usize) -> Vec<f64> {
    let bpe = 2usize;
    let off = row * d_model * bpe;
    data::decode_f64(&bytes[off..off + d_model * bpe], d_model, ktir_core::dtypes::DType::F16)
        .expect("an f16 row decodes")
}

/// The fixture, its bindings, and the extents read from `meta.json` rather than restated.
fn load() -> (data::Fixture, Vec<triton_numeric::Binding>, usize, usize, usize, f64) {
    let f = data::Fixture::load(CONFIG).expect("the embedding data");
    let bindings = f.bindings().expect("the embedding bindings");
    let n_tok = f.int("N_TOK").expect("N_TOK") as usize;
    let v = f.int("V").expect("V") as usize;
    let d_model = f.int("D_MODEL").expect("D_MODEL") as usize;
    let emb_scale = f.float("EMB_SCALE").expect("EMB_SCALE");
    (f, bindings, n_tok, v, d_model, emb_scale)
}

fn binding<'a>(b: &'a [triton_numeric::Binding], name: &str) -> &'a triton_numeric::Binding {
    b.iter().find(|x| x.name == name).unwrap_or_else(|| panic!("no binding `{name}`"))
}

/// The token ids, as the kernel reads them: raw i32 little-endian.
fn ids_of(b: &[triton_numeric::Binding], n_tok: usize) -> Vec<i64> {
    let raw = &binding(b, "desc_ids").bytes;
    assert_eq!(raw.len(), n_tok * 4, "desc_ids is {} bytes for {n_tok} i32 ids", raw.len());
    raw.as_chunks::<4>().0.iter()
        .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64)
        .collect()
}

/// ⭐ THE OFF-BY-ONE ROW, ON THE REAL TABLE, AGAINST THE FIXTURE'S OWN ANSWER.
///
/// `mutants::embedding::off_by_one_row` existed and ran on a `[97, 128]` synthetic pair. That proves
/// the mutant is discriminating against the ENVELOPE; it proves nothing about the bytes this
/// configuration is verified on — a 4096-wide Granite row and a 128-wide toy row are different
/// stimuli, and the reference in the toy test is the mutant module's own `truth`, not the fixture's.
/// Here both sides come from the tree: the ids and the table are the checked-in bytes, and the thing
/// the mutant is compared against is `ref_out.bin`, computed by `torch.nn.functional.embedding` on
/// the pod.
///
/// # THE COMPACT TABLE, WHICH IS A REINDEXING AND NOT A DIFFERENT STIMULUS
///
/// `off_by_one_row` takes `table: &[f64]`, and the real table widened to f64 is 3.2 GB. It needs
/// exactly two rows per token — `ids[t]` and `ids[t] + 1` — so this builds a `[2 * N_TOK, D_MODEL]`
/// table holding real row `ids[t]` at compact row `2t` and real row `ids[t] + 1` at `2t + 1`, and
/// passes `ids[t] = 2t`. The mutant's `+1` skew then lands on real row `ids[t] + 1` BY CONSTRUCTION,
/// which is the mutant it is named for, on the real bytes, in 16 MB.
///
/// ⛔ AND THE CONSTRUCTION IS CHECKED BEFORE IT IS TRUSTED: the FAITHFUL gather over the compact
/// table has to reproduce `ref_out.bin` inside the derived envelope. If the reindexing were wrong,
/// that comparison fails first and this control never gets to report a margin it has not earned.
#[test]
fn the_real_gathers_off_by_one_row_exceeds_the_embedding_bound() {
    let (f, b, n_tok, v, d_model, emb_scale) = load();
    let env = bounds::embedding(emb_scale);
    let reference = f.reference().expect("ref_out.bin");
    let ids = ids_of(&b, n_tok);
    let table = &binding(&b, "desc_table").bytes;

    // Two rows per token: the one the ids name, and its successor.
    let mut compact = Vec::with_capacity(2 * n_tok * d_model);
    for &id in &ids {
        assert!(id >= 0 && (id as usize) < v, "id {id} is outside the table");
        compact.extend(table_row(table, id as usize, d_model));
        compact.extend(table_row(table, ((id + 1) as usize) % v, d_model));
    }
    let cids: Vec<i64> = (0..n_tok as i64).map(|t| 2 * t).collect();
    let cv = 2 * n_tok;

    // THE CONSTRUCTION'S OWN GATE. `truth` over the compact table IS the kernel's function, so it
    // must land inside the envelope against the fixture's reference before any margin below means
    // anything.
    let truth = mutants::embedding::truth(&cids, &compact, cv, d_model, emb_scale);
    let c = Comparison::new(CONFIG, &truth, &reference, &env).expect("a comparison");
    println!("  reindexed faithful gather vs ref_out.bin:\n  {}", c.report());
    assert!(
        c.within_bound(),
        "the compact reindexing does NOT reproduce the fixture's answer, so this control's mutant \
         margin would be measured against the wrong truth. {}",
        c.report()
    );

    let mutant = mutants::embedding::off_by_one_row(&cids, &compact, cv, d_model, emb_scale);
    let (dev, bound) = mutants::max_abs_deviation(&reference, &mutant, &env);
    let ratio = mutants::worst_ratio(&reference, &mutant, &env);
    let frac = mutants::changed_row_fraction(&reference, &mutant, d_model);
    println!(
        "  off_by_one_row on the REAL table: max|dev| = {dev:.6e}  bound = {bound:.6e}  \
         worst ratio = {ratio:.4e}  changed-row fraction = {frac:.4}"
    );
    assert!(
        mutants::exceeds(&reference, &mutant, &env),
        "⛔⛔ AN OFF-BY-ONE GATHERED ROW SITS INSIDE THE EMBEDDING ENVELOPE ON THE REAL TABLE. \
         max|dev| = {dev:.6e} against a bound of {bound:.6e} (worst ratio {ratio:.4e}). The gather \
         is the ONLY thing this kernel can get wrong — the arithmetic is one multiply — so if this \
         control cannot see a one-row skew then `embedding_granite`'s green result says nothing \
         about which rows were read. REPORT THIS rather than loosening anything."
    );
    // A mutant must change essentially every word of its thinnest row, not a handful — the same
    // predeclared floor `transposing_a_square_projection_exceeds_the_decoder_bound` uses.
    assert!(
        frac >= mutants::MUTANT_MIN_ROW_FRACTION,
        "the off-by-one gather changes only {frac:.4} of the words in its thinnest row; the \
         predeclared floor is {}",
        mutants::MUTANT_MIN_ROW_FRACTION
    );
}

/// ⭐⭐⭐ THE INDEX-STICK WRAP, ON THE REAL TABLE, AGAINST THE FIXTURE'S OWN ANSWER — the defect the
/// CARD produced, encoded as the gate that would catch its return.
///
/// A gather op's index reaches the L3LU IBR as ONE stick transfer — `SenUint32`, 32 entries, 128 B
/// (`CopyDims::ENTRIES_PER_OP`) — and each core indexes it at its own work-slice word, so an op whose
/// index is longer than one stick WRAPS. `embedding_granite`'s descriptor spans the whole node
/// (`node_rows` = the output view's 256 rows, the grid folded in), pages `mb` by ONE position, and so
/// declares 256 entries. MEASURED on the card, through the ladder this tree bakes:
///
/// ```text
/// rows[0:32]    within_2pct_strict = 1.000000   (131072 elements, max|err| 0.0625 = the f16 ulp)
/// rows[32:64]   within_2pct_strict = 0.006226
/// rows[0:256]   within_2pct_strict = 0.130597   ≈ 32/256, and 256/256 rows fit `entry = r mod 32`
/// ```
///
/// with `got_rms` 11.9603 against the reference's 11.9844 — the RIGHT distribution, because the
/// table's rows are independent draws. `dxp_standalone --bundle` exited 0 and the launch returned
/// rc=0 in 0.265 s. Nothing in the compile path or the runtime saw anything wrong.
///
/// ⭐ THE EMISSION SIDE NOW **CUTS** THE NODE into one op per index stick
/// (`assemble_pointwise_broadcast_gather`), each reading its own 32 entries and writing its own 32-row
/// window of the output, and all three embedding configurations are exact on card at
/// `within_2pct_strict` 1.000000 — with seven discriminating controls at 256 tokens, two of which (ids
/// rotated by one stick, ids reversed within each stick) exist only because the node is cut. The
/// one-stick ceiling is STILL enforced, on the op that declares the index
/// (`one_gathered_leg`), because that is what the wrap is a property of. THIS test is the other side:
/// it asks whether the NUMERIC gate can see the wrap at all, so that a cut which silently dropped a
/// leg's entry base — every leg reading leg 0's — cannot pass here either.
///
/// # THE COMPACT TABLE, which is the same reindexing the sibling control above uses
///
/// `index_stick_wrap` takes `table: &[f64]`, and the real table widened to f64 is 3.2 GB. It needs two
/// rows per token — `ids[t]` and `ids[t % 32]` — so this builds a `[2 · N_TOK, D_MODEL]` table holding
/// real row `ids[t]` at compact row `2t` and real row `ids[t % 32]` at `2t + 1`, and hands the mutant
/// `cids[t] = 2t` with a stick of... no: the wrap has to be expressed in the COMPACT index, and
/// `cids[t % 32]` is `2·(t % 32)`, which is a DIFFERENT compact row from `2t + 1`. So the mutant is
/// called on `cids` directly — compact row `2·(t % 32)` IS real row `ids[t % 32]` — and the odd rows
/// are not needed at all. Only `2t` is ever read.
///
/// ⛔ AND THE CONSTRUCTION IS GATED BEFORE IT IS TRUSTED, exactly as above: the FAITHFUL gather over
/// the compact table must reproduce `ref_out.bin` inside the derived envelope first. If the reindexing
/// were wrong, that comparison fails and this control never reports a margin it has not earned.
#[test]
fn the_real_gathers_index_stick_wrap_exceeds_the_embedding_bound() {
    let (f, b, n_tok, v, d_model, emb_scale) = load();
    let env = bounds::embedding(emb_scale);
    let reference = f.reference().expect("ref_out.bin");
    let ids = ids_of(&b, n_tok);
    let table = &binding(&b, "desc_table").bytes;
    // The index's own stick, as the emitter spells it — 32 `SenUint32` entries in one 128-byte stick.
    // Stated here rather than imported so this test does not depend on the emission crate; the number
    // is checked against the node it is about, which is what makes a bare 32 safe.
    const INDEX_STICK: usize = 32;
    assert!(
        n_tok > INDEX_STICK,
        "this configuration's {n_tok} entries already fit one {INDEX_STICK}-entry stick, so there is \
         no wrap to measure and this control would be vacuous"
    );

    // Two rows per token: the one the ids name, and the one a wrapped index would reach.
    let mut compact = Vec::with_capacity(2 * n_tok * d_model);
    for (t, &id) in ids.iter().enumerate() {
        assert!(id >= 0 && (id as usize) < v, "id {id} is outside the table");
        compact.extend(table_row(table, id as usize, d_model));
        let w = ids[t % INDEX_STICK];
        compact.extend(table_row(table, w as usize, d_model));
    }
    let cids: Vec<i64> = (0..n_tok as i64).map(|t| 2 * t).collect();
    let cv = 2 * n_tok;

    // THE CONSTRUCTION'S OWN GATE, before any margin below means anything.
    let truth = mutants::embedding::truth(&cids, &compact, cv, d_model, emb_scale);
    let c = Comparison::new(CONFIG, &truth, &reference, &env).expect("a comparison");
    println!("  reindexed faithful gather vs ref_out.bin:\n  {}", c.report());
    assert!(
        c.within_bound(),
        "the compact reindexing does NOT reproduce the fixture's answer, so this control's mutant \
         margin would be measured against the wrong truth. {}",
        c.report()
    );

    let mutant =
        mutants::embedding::index_stick_wrap(&cids, &compact, cv, d_model, emb_scale, INDEX_STICK);
    let (dev, bound) = mutants::max_abs_deviation(&reference, &mutant, &env);
    let ratio = mutants::worst_ratio(&reference, &mutant, &env);
    let frac = mutants::changed_row_fraction(&reference, &mutant, d_model);
    println!(
        "  index_stick_wrap({INDEX_STICK}) on the REAL table: max|dev| = {dev:.6e}  \
         bound = {bound:.6e}  worst ratio = {ratio:.4e}  changed-row fraction = {frac:.4}"
    );
    assert!(
        mutants::exceeds(&reference, &mutant, &env),
        "⛔⛔ A WRAPPED GATHER INDEX SITS INSIDE THE EMBEDDING ENVELOPE ON THE REAL TABLE. \
         max|dev| = {dev:.6e} against a bound of {bound:.6e} (worst ratio {ratio:.4e}). This is the \
         shape the CARD produced — the first {INDEX_STICK} rows exact and every later row repeating \
         them — so if this control cannot see it, neither the emission guard nor this suite is \
         standing between that descriptor and a green result. REPORT THIS rather than loosening \
         anything."
    );
    // ⛔ AND THE FIRST STICK MUST BE UNTOUCHED, which is what makes this the WRAP rather than a
    // generic scramble: rows 0..31 of the mutant and of the truth are the same bytes, and everything
    // the margin above measures comes from rows 32.. . Without this the test would pass for any
    // mutant that ruined the whole tensor, and the thing the card did is much narrower than that.
    let first = INDEX_STICK * d_model;
    assert_eq!(
        &truth[..first],
        &mutant[..first],
        "the wrap changed the FIRST index stick, which the card's own result shows it does not \
         (rows[0:32] came back exact) — so this mutant is not the defect it is named for"
    );
    assert!(
        truth[first..] != mutant[first..],
        "the wrap changed nothing past the first stick, so it is not a mutant at all"
    );
    // The same predeclared floor the sibling control uses: a mutant must change essentially every
    // word of its thinnest CHANGED row. Rows 0..31 are unchanged BY CONSTRUCTION (checked above), so
    // the floor is applied to the rows the wrap actually reaches.
    let frac_tail = mutants::changed_row_fraction(&reference[first..], &mutant[first..], d_model);
    assert!(
        frac_tail >= mutants::MUTANT_MIN_ROW_FRACTION,
        "past the first stick the wrap changes only {frac_tail:.4} of the words in its thinnest row; \
         the predeclared floor is {}",
        mutants::MUTANT_MIN_ROW_FRACTION
    );
}

/// ⭐ THE SPARSE TABLE'S SOUNDNESS PROPERTY, MEASURED.
///
/// `desc_table` is `[49159, 4096]` f16 — 384 MiB — of which the kernel gathers 256 rows, so the tree
/// stores 255 distinct rows in 2,090,012 B (`SPRSTBL1`). The whole argument that this cannot mask a
/// wrong gather is one sentence in `data::Encoding` and `gen_numeric_data.py`: AN ABSENT ROW EXPANDS
/// TO ZERO, so a kernel that reads a row the ids never named reads zeros, the product is zero where
/// the reference is not, and the comparison DIVERGES. False FAILURE possible; false PASS impossible.
///
/// ⛔ THAT WAS PROSE IN THREE FILES AND AN EXECUTED FACT IN NONE. It is the reason the 2 MiB file is
/// allowed to stand in for the 384 MiB one, so it is checked here, and checked with the control that
/// makes it non-vacuous: an expansion that produced zeros EVERYWHERE would satisfy "absent rows are
/// zero" and mask everything, so a PRESENT row is required to be non-zero in the same breath.
#[test]
fn an_absent_table_row_expands_to_zero_and_that_diverges() {
    let (f, b, n_tok, v, d_model, emb_scale) = load();
    let env = bounds::embedding(emb_scale);
    let reference = f.reference().expect("ref_out.bin");
    let ids = ids_of(&b, n_tok);
    let table = &binding(&b, "desc_table").bytes;
    assert_eq!(table.len(), v * d_model * 2, "the expanded table is not [V, D_MODEL] f16");

    // `write_sparse` stores exactly the DISTINCT ids, so any other row index is absent. Which rows
    // those are is not assumed from that rule: the expansion itself is read below.
    let named: std::collections::HashSet<i64> = ids.iter().copied().collect();
    let absent = (0..v as i64)
        .find(|r| !named.contains(r))
        .expect("256 ids cannot name all 49159 rows") as usize;
    let absent_row = table_row(table, absent, d_model);
    assert!(
        absent_row.iter().all(|&x| x == 0.0),
        "row {absent} is named by no id and did NOT expand to zero. Zero is the whole soundness \
         argument for storing this table sparsely: a row the kernel cannot have meant must read as \
         something it cannot have meant."
    );
    // THE DISCRIMINATING HALF. An all-zero expansion would pass the assertion above and hide every
    // gather defect there is.
    let present_row = table_row(table, ids[0] as usize, d_model);
    let present_energy = present_row.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    assert!(
        present_energy > 0.0,
        "the row id 0 names ({}) expanded to zeros too, so 'absent rows are zero' is vacuous here \
         and the expansion is empty",
        ids[0]
    );

    // RETARGET ONE ID AT THAT ABSENT ROW. Token 0 then reads zeros, so its whole output row becomes
    // zero and the deviation from the fixture's reference is that row's own energy.
    let mut mutated = reference.clone();
    for j in 0..d_model {
        mutated[j] = 0.0;
    }
    let (dev, bound) = mutants::max_abs_deviation(&reference, &mutated, &env);
    let ratio = mutants::worst_ratio(&reference, &mutated, &env);
    let range = reference.iter().fold(0.0f64, |m, &x| m.max(x.abs()));
    println!(
        "  id[0] retargeted from {} (present) to {absent} (absent -> zero): max|dev| = {dev:.6e}  \
         bound = {bound:.6e}  worst ratio = {ratio:.4e}   reference range +-{range:.6e}",
        ids[0]
    );
    assert!(
        mutants::exceeds(&reference, &mutated, &env),
        "⛔⛔ READING AN ABSENT (ZERO) ROW SITS INSIDE THE EMBEDDING ENVELOPE. max|dev| = {dev:.6e} \
         against {bound:.6e}. Then sparsifying the table CAN mask a wrong gather and the 2 MiB file \
         may not stand in for the 384 MiB one — regenerate `in_desc_table.sparse` densely."
    );
    // The energy the mutation removes IS the row's, so the two numbers must agree: a deviation
    // smaller than the row's own peak would mean the reference row and the table row disagree.
    let scaled = present_energy * emb_scale;
    assert!(
        (dev - scaled).abs() <= 1e-2 * scaled.max(1.0),
        "the deviation {dev:.6e} is not the retargeted row's own energy {scaled:.6e} \
         (|table[{}]|max x EMB_SCALE). The reference's row 0 and the table row id 0 names are then \
         not the same numbers, which is a data defect and not a gather one.",
        ids[0]
    );
}
