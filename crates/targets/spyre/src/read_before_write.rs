// SPDX-License-Identifier: Apache-2.0
//! ⭐⭐⭐⭐⭐ THE READ-BEFORE-WRITE WALK — no op of a forward may read a byte that
//! no earlier op of the same forward wrote and nothing staged.
//!
//! The defect class is the one the card keeps paying for: an emission that drops
//! or reorders a producer leaves a consumer reading a buffer whose bytes were
//! NEVER WRITTEN — not written with wrong values, never written at all — so the
//! read returns whatever the device held there, and the model is fluently wrong
//! from the first token with nothing downstream able to say why. The window
//! guards check an op against its OWN buffers; the LX direction guard checks a
//! pin has SOME producer; neither checks the thing that actually ran wrong on
//! the card — that the bytes a read touches were written by an op that ran
//! EARLIER IN THE SAME FORWARD.
//!
//! ⛔⛔⛔ THE WALK READS ONE THING: THE OPS' OWN BINDINGS. Every fact it judges
//! with — the buffer name, the direction, the materialized footprint and
//! touched blocks, and now the PLACEMENT FACTS (role, segment, bank, the
//! intra-segment byte base with the element-offset slice already folded in, the
//! LX verdict) — ARRIVED on the `arg_bindings`, stamped there by
//! `EmittedOp::new` from the door's (`resolve_seg_base`'s) one resolution at
//! emission time. The walk resolves NOTHING: it holds no layout, consults no
//! pin table, no placement row, no synth map, and multiplies no element offset
//! by no word. Records 53 and 54 were the same defect twice — the walk's own
//! private copy of the door's address resolution drifting from the door's at a
//! different spot each time (the word source, then the arm order) — and the
//! cure is structural: there is ONE resolution in the codebase, it travels on
//! the binding, and a consumer that wants it reads it there. The walk also
//! never opens `EmittedOp::ktir`, never consults a program's values, constants
//! or splats, and knows nothing about any kernel's source — writedeness is a
//! fact about descriptors in the bundle's op list and the bindings they carry,
//! and an implementation that derives it anywhere else is wrong here.
//!
//! ⛔ AND A DESCRIPTOR THE WALK CANNOT SEE IS A REFUSAL, NOT A PASS: an op whose
//! descriptor is present but whose `arg_bindings` is EMPTY names buffers the
//! walk cannot check, and silence is the hazard this walk exists to remove —
//! `EmittedOp::new` is the sole binding writer and refuses a facts/bindings
//! length mismatch, so an invisible operand can only mean a walk-model defect,
//! and it refuses naming the op. (`EmittedOp::bare` ops — no descriptor, no
//! bindings — are markers, not accesses, and pass.)
//!
//! ⛔ KEYED ON CARRIED FACTS AND BYTE INTERVALS ONLY. A buffer is a `(segment,
//! bank)` and a byte interval; a read is admitted iff the honest set covers its
//! role or the union of intervals earlier ops WROTE covers it. There is no
//! concept of any op family, carry, score or pool geometry in the code, the
//! types or the refusal text — the walk cannot grow an opinion about WHAT an
//! op computes, only about whether the bytes it reads were written first.
//!
//! ⭐ THE HONEST SET — written at session start, never refused, and NOTHING
//! ELSE (see [`session_written`]): the weights, the host-staged activations,
//! and the paged pool. Both pool limitations are stated where the set is
//! decided:
//!
//! 1. **An in-bounds pool read with no aimed write is not caught.** The pool is
//!    zero-filled when the session loads it (the load refuses on fill failure),
//!    so every pool byte reads a deterministic value before anything writes it.
//!    A read aimed at the wrong pool offset — a launch-time rebase the baked
//!    intervals cannot see, since the runtime adds to the baked addresses after
//!    the bake — is therefore admitted by this walk: it reads zeroed-or-written
//!    bytes, not never-written memory. Wrong-VALUE reads inside the pool are
//!    the launch plan's to catch, never this walk's.
//! 2. **Intra-forward pool write→read ordering is not checked.** The pool is
//!    the one segment whose writes are MEANT to outlive the forward that wrote
//!    them, so its reads are admitted by the honest set whether the writer ran
//!    in this forward, an earlier one, or never — the walk does not track pool
//!    writes at all, which keeps (1) and (2) one decision rather than two
//!    places to disagree.
//!
//! ⚠️ TWO STATED PREMISES, both carried on the bindings themselves:
//! * **LX bindings are out of domain.** A binding whose carried `lx` is set
//!   lives in the per-core scratchpad, not in a segment; its ordering is the LX
//!   slot alternation walk's own law. The walk skips those bindings by the
//!   carried flag, never by a pin lookup — the flag IS the door's pin lookup's
//!   answer, resolved once at emission.
//! * **The rolled body's per-layer shifts are an affine replay.** The walk
//!   checks each body instance at its baked (layer-0) offsets, in the true
//!   launch order the producer sorted; the executor's per-layer segment-base
//!   shifts replay the same static pattern at other addresses, and a threading
//!   defect that only exists at a shifted address is not this walk's to see.
//!   A single-class model (every layer one body) is modeled exactly; a
//!   class-split hand-off that reads bytes no instance ever wrote at the
//!   static offsets refuses — loudly, for a ruling — rather than passing.
//!
//! The walk changes no emission: it reads the ops the declare pass already
//! lowered (the descriptors it computed and the bindings they carry) and either
//! returns `Ok` or refuses the bake, naming the op, the buffer, and the exact
//! unwritten interval.

use ktir_superdsc::emit::EmittedOp;
use ktir_superdsc::placement::SegRole;
/// ⭐ THE HONEST SET — which roles the SESSION writes before any op runs.
///
/// A read of a buffer in one of these roles is never refused: the weights are
/// uploaded once at load, the activations are staged by the host every step,
/// and the pool is zero-filled at load (see the module doc for both pool
/// limitations — they are this function's decision, stated once, here).
/// ⛔ NOTHING ELSE. An `Intermediate` or `Logits` byte is only ever correct to
/// read if an earlier op of the same forward wrote it; growing this set is a
/// ruling, never a match arm.
fn session_written(role: SegRole) -> bool {
    matches!(role, SegRole::Weight | SegRole::Activation | SegRole::Kv)
}

/// The written-byte table: per `(segment, bank)`, the merged union of intervals
/// earlier ops wrote. Union coverage (a read admitted when the union of earlier
/// writes covers it, even piecewise — a strided write and a strided read of the
/// same blocks must match block for block, and no single interval would) is the
/// coverage model; STALE-but-written bytes are deliberately NOT judged here —
/// reading a dead tensor's leftovers is a wrong-value defect, and this walk
/// exists for the never-written class alone.
#[derive(Default)]
struct Written {
    spans: std::collections::BTreeMap<(usize, u32), Vec<(u64, u64)>>,
}

impl Written {
    /// Record one written interval, merging it with any span it touches so the
    /// table stays a sorted disjoint union.
    fn insert(&mut self, seg: usize, bank: u32, start: u64, end: u64) {
        let spans = self.spans.entry((seg, bank)).or_default();
        let (mut lo, mut hi) = (start, end);
        let mut out: Vec<(u64, u64)> = Vec::with_capacity(spans.len() + 1);
        for &(s, e) in spans.iter() {
            if e < lo || s > hi {
                out.push((s, e));
            } else {
                lo = lo.min(s);
                hi = hi.max(e);
            }
        }
        out.push((lo, hi));
        out.sort_unstable();
        *spans = out;
    }

    /// Does the union cover `[start, end)` wholly? Returns the first uncovered
    /// sub-interval when it does not — the refusal names the exact unwritten
    /// bytes, not just the buffer. The returned hole is CLAMPED to this span's
    /// own end: a later-written span may cover part of the tail, and naming
    /// bytes an op did write as unwritten would make the refusal text false.
    fn uncovered(&self, seg: usize, bank: u32, start: u64, end: u64) -> Option<(u64, u64)> {
        let mut cursor = start;
        for &(s, e) in self.spans.get(&(seg, bank)).into_iter().flatten() {
            if e <= cursor {
                continue;
            }
            if s > cursor {
                // A hole before this span — but only up to where this span
                // (or a later one) starts writing again.
                return Some((cursor, end.min(s)));
            }
            cursor = e;
            if cursor >= end {
                return None;
            }
        }
        Some((cursor, end))
    }
}

/// ⭐ THE WALK. `seq` is one forward's op-list instances in run order — the
/// prefix, each body instance as the producer's sorted launch sequence runs
/// it, the suffix — each with a label the refusal names. `Ok(())` or a
/// refusal naming the op, the buffer, and the unwritten interval. No layout:
/// every placement fact it needs arrived on the bindings.
pub fn walk_read_before_write(seq: &[(&str, &[EmittedOp])]) -> Result<(), String> {
    let mut written = Written::default();
    for (list_label, ops) in seq {
        for (oi, e) in ops.iter().enumerate() {
            let where_ = format!("{} op #{oi} '{}'", list_label, e.op_name);
            // ⛔ A DESCRIPTOR OP WITH NO BINDINGS IS A REFUSAL. `EmittedOp::new`
            // is the sole binding writer and stamps one binding per spec operand
            // (refusing a facts/bindings length mismatch), so a descriptor op
            // with an EMPTY binding list names buffers this walk cannot see —
            // a walk-model defect, never a pass. Bare ops (no descriptor) are
            // markers and carry no accesses.
            if e.op.is_some() && e.arg_bindings.is_empty() {
                return Err(format!(
                    "{where_} has a descriptor but no arg bindings — the walk cannot see \
                     what it names, and an invisible operand is a walk-model defect, not a \
                     pass. Every descriptor op's bindings are stamped at emission \
                     (EmittedOp::new), so this shape must refuse rather than silently \
                     check nothing"
                ));
            }
            // A time-tiled op's trips run in order; each trip's reads see the
            // writes of every earlier op AND every earlier trip. The per-trip
            // advance is the binding's own `affine_strides` "out" entry — the
            // same number `concrete_trips` adds onto the base address.
            let trips = e.time.max(1);
            for t in 0..trips {
                // Reads first, then writes: an op's own writes never cover its
                // reads — a read-modify-write of never-written bytes refuses.
                let mut reads: Vec<(&ktir_superdsc::emit::ArgBinding, u64, u64)> = Vec::new();
                let mut writes: Vec<(usize, u32, u64, u64)> = Vec::new();
                for (bi, b) in e.arg_bindings.iter().enumerate() {
                    // The carried LX verdict skips the binding — the door's own
                    // pin-lookup answer, resolved once at emission. Core-local
                    // bytes are the alternation walk's domain, never this one's.
                    if b.lx {
                        continue;
                    }
                    let shift = t as u64
                        * e.affine_strides
                            .get(bi)
                            .and_then(|m| m.get("out"))
                            .copied()
                            .unwrap_or(0)
                            .max(0) as u64;
                    // The carried byte base — the door's intra-segment base,
                    // the placement/synth offset WITH the element-offset slice
                    // already folded in. The walk adds only the per-trip shift;
                    // it re-derives nothing.
                    let base = b.base_bytes + shift;
                    for off in b.touched.offsets() {
                        let (s, en) = (base + off, base + off + b.touched.run_bytes);
                        if b.is_input {
                            reads.push((b, s, en));
                        } else {
                            writes.push((b.segment, b.bank, s, en));
                        }
                    }
                }
                for (b, s, en) in &reads {
                    if session_written(b.role) {
                        continue;
                    }
                    if let Some((gs, ge)) = written.uncovered(b.segment, b.bank, *s, *en) {
                        return Err(format!(
                            "{where_} reads '{}' (seg{} bank {}) at bytes [{s}, {en}), and \
                             [{gs}, {ge}) of that is written by NO earlier op of this forward — \
                             the buffer is neither a weight, a host-staged input, nor the paged \
                             pool, so nothing staged it either. The read returns whatever the \
                             device held there, not a value this program computed: fix the \
                             emission so an earlier op writes those bytes",
                            b.buffer, b.segment, b.bank,
                        ));
                    }
                }
                for (seg, bank, s, en) in writes {
                    written.insert(seg, bank, s, en);
                }
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ktir_superdsc::emit::{ArgBinding, TouchedBlocks};

    /// One binding with HAND-STATED carried facts — the door's answer as the
    /// emitter stamps it: the role, the `(segment, bank)` table key, and the
    /// intra-segment byte base (the placement/synth offset with the
    /// element-offset slice already folded in). The fixtures state the facts
    /// directly because the walk no longer resolves anything: what it reads is
    /// exactly these fields.
    #[allow(clippy::too_many_arguments)]
    fn binding(
        name: &str,
        is_input: bool,
        bytes: u64,
        role: SegRole,
        segment: usize,
        bank: u32,
        base_bytes: u64,
    ) -> ArgBinding {
        ArgBinding {
            buffer: name.to_string(),
            is_input,
            footprint_bytes: bytes,
            touched: TouchedBlocks {
                run_bytes: bytes,
                outer: Vec::new(),
            },
            offset_elems: 0,
            segment,
            bank,
            role,
            base_bytes,
            lx: false,
        }
    }

    /// The common fixture: an Intermediate in its own segment, bank 0, at
    /// `base_bytes`.
    fn inter(name: &str, is_input: bool, bytes: u64, base_bytes: u64) -> ArgBinding {
        binding(
            name,
            is_input,
            bytes,
            SegRole::Intermediate,
            SegRole::Intermediate.segment(),
            0,
            base_bytes,
        )
    }

    /// A strided binding over the same carried facts: `run_bytes` per block,
    /// `outer` `(count, stride_bytes)` pairs — the piecewise-coverage shape.
    fn inter_strided(
        name: &str,
        is_input: bool,
        run_bytes: u64,
        outer: Vec<(u64, u64)>,
        base_bytes: u64,
    ) -> ArgBinding {
        ArgBinding {
            buffer: name.to_string(),
            is_input,
            footprint_bytes: run_bytes,
            touched: TouchedBlocks { run_bytes, outer },
            offset_elems: 0,
            segment: SegRole::Intermediate.segment(),
            bank: 0,
            role: SegRole::Intermediate,
            base_bytes,
            lx: false,
        }
    }

    fn op(name: &str, bindings: Vec<ArgBinding>) -> EmittedOp {
        let mut e = EmittedOp::bare(name.to_string());
        e.arg_bindings = bindings;
        e
    }

    /// A REAL emission with its bindings stripped — the exact shape the
    /// rider refuses: a descriptor the door built (so `op` is `Some`) whose
    /// operand list the walk cannot see. Built by emitting a one-op matmul
    /// and clearing the bindings, because the refusal keys on the descriptor
    /// being present, which only a genuine emission provides.
    fn descriptor_op_with_no_bindings(name: &str) -> EmittedOp {
        let op = ktir_superdsc::ir::bridge::tiled_op_sdsc_op::matmul_opspec(
            64, 64, 64, 1, "tA", "tW", "tO",
        )
        .expect("a small matmul is a legal OpSpec");
        let folds = ktir_superdsc::superdsc_opspec::SdscFoldSet::new(op.iter.cores_used());
        let mut e = ktir_superdsc::emit::emit_sdsc_tiled(name, &op, &folds, &mut 0, None)
            .expect("the small matmul emits");
        assert!(
            e.op.is_some() && !e.arg_bindings.is_empty(),
            "the fixture premise: a fresh emission carries a descriptor AND its bindings"
        );
        e.arg_bindings.clear();
        e
    }

    #[test]
    fn a_chain_where_the_producer_precedes_the_consumer_passes() {
        let a = op("wr", vec![inter("t9", false, 256, 0)]);
        let b = op("rd", vec![inter("t9", true, 256, 0)]);
        assert!(walk_read_before_write(&[("list", &[a, b])]).is_ok());
    }

    #[test]
    fn a_read_before_its_write_names_the_op_and_the_interval() {
        let a = op("rd", vec![inter("t9", true, 256, 0)]);
        let b = op("wr", vec![inter("t9", false, 256, 0)]);
        let err = walk_read_before_write(&[("list", &[a, b])]).unwrap_err();
        assert!(err.contains("'rd'"), "{err}");
        assert!(err.contains("[0, 256)"), "{err}");
        assert!(err.contains("t9"), "{err}");
    }

    #[test]
    fn a_never_written_intermediate_read_refuses() {
        let a = op("rd", vec![inter("t9", true, 256, 0)]);
        assert!(walk_read_before_write(&[("list", &[a])]).is_err());
    }

    #[test]
    fn the_honest_set_is_never_refused() {
        let ops = [
            op(
                "r1",
                vec![binding("w", true, 256, SegRole::Weight, 1, 0, 0)],
            ),
            op(
                "r2",
                vec![binding("a", true, 256, SegRole::Activation, 3, 0, 0)],
            ),
            op("r3", vec![binding("k", true, 256, SegRole::Kv, 2, 0, 0)]),
        ];
        assert!(walk_read_before_write(&[("list", &ops)]).is_ok());
    }

    #[test]
    fn a_partially_covered_read_names_the_uncovered_tail() {
        let a = op("wr", vec![inter("t9", false, 128, 0)]);
        let b = op("rd", vec![inter("t9", true, 256, 0)]);
        let err = walk_read_before_write(&[("list", &[a, b])]).unwrap_err();
        assert!(err.contains("[128, 256)"), "{err}");
    }

    #[test]
    fn a_hole_clamps_to_where_the_next_write_resumes() {
        // Writes at [0, 64) and [192, 256), read over all 256: the hole is
        // [64, 192) EXACTLY. The refusal must not name [64, 256) — the tail
        // was written, and a refusal naming written bytes is a false one.
        // The two write bases are carried facts now — the door's BYTE bases,
        // stated by hand where the old fixture derived them from an element
        // offset: the recorded geometry's tail write sat at Fp16 element 96 =
        // byte 192 (the ×2 the walk no longer performs).
        let head = op("wrH", vec![inter("t9", false, 64, 0)]);
        let tail = op("wrT", vec![inter("t9", false, 64, 192)]);
        let rd = op("rd", vec![inter("t9", true, 256, 0)]);
        let err = walk_read_before_write(&[("list", &[head, tail, rd])]).unwrap_err();
        assert!(err.contains("[64, 192)"), "{err}");
        assert!(!err.contains("[64, 256)"), "{err}");
    }

    #[test]
    fn strided_writes_cover_strided_reads_piecewise() {
        let mk = |is_input: bool| inter_strided("t9", is_input, 64, vec![(2, 4096)], 0);
        let a = op("wr", vec![mk(false)]);
        let b = op("rd", vec![mk(true)]);
        assert!(walk_read_before_write(&[("list", &[a.clone(), b])]).is_ok());
        // A read one block wider than the write refuses on the extra block.
        let wider = inter_strided("t9", true, 128, vec![(2, 4096)], 0);
        let c = op("rd2", vec![wider]);
        let err = walk_read_before_write(&[("list", &[a, c])]).unwrap_err();
        assert!(err.contains("[64, 128)"), "{err}");
    }

    #[test]
    fn an_op_never_covers_its_own_read_with_its_own_write() {
        // Read-modify-write of never-written bytes: both bindings on one op.
        let a = op(
            "rmw",
            vec![inter("t9", true, 256, 0), inter("t9", false, 256, 0)],
        );
        assert!(walk_read_before_write(&[("list", std::slice::from_ref(&a))]).is_err());
        // ...but a genuine RMW over EARLIER-written bytes passes.
        let w = op("wr", vec![inter("t9", false, 256, 0)]);
        assert!(walk_read_before_write(&[("list", &[w, a])]).is_ok());
    }

    #[test]
    fn a_time_tiled_op_shifts_by_its_own_affine_stride() {
        let mut wr = op("wr", vec![inter("t9", false, 4096, 0)]);
        wr.time = 2;
        wr.affine_strides = vec![std::collections::BTreeMap::from([("out", 4096i64)])];
        // Trip t's read is covered by trip t's write of the earlier op.
        let mut rd = op("rd", vec![inter("t9", true, 4096, 0)]);
        rd.time = 2;
        rd.affine_strides = vec![std::collections::BTreeMap::from([("out", 4096i64)])];
        assert!(walk_read_before_write(&[("list", &[wr, rd.clone()])]).is_ok());
        // Strip the writer's stride and the second trip's read must refuse.
        let mut wr0 = op("wr", vec![inter("t9", false, 4096, 0)]);
        wr0.time = 2;
        wr0.affine_strides = vec![std::collections::BTreeMap::new()];
        let err = walk_read_before_write(&[("list", &[wr0, rd])]).unwrap_err();
        assert!(err.contains("[4096, 8192)"), "{err}");
    }

    #[test]
    fn slot_reuse_across_names_is_admitted_when_the_bytes_were_written() {
        // Two names over the SAME bytes of one segment (liveness coloring
        // reuses slots): the walk judges bytes, not names — a later name
        // reading an earlier name's write is ADMITTED, because the never-
        // written class is the only one judged here. Stale-value reads are a
        // different detector's defect, stated in the module doc. The shared
        // slot is stated as the shared carried facts: same segment, bank, base.
        let a = op("wrA", vec![inter("tA", false, 256, 0)]);
        let b = op("rdB", vec![inter("tB", true, 256, 0)]);
        assert!(walk_read_before_write(&[("list", &[a, b])]).is_ok());
    }

    #[test]
    fn the_honest_set_covers_a_spilled_tail_by_role() {
        // A weight moved to a spill segment keeps its ROLE (spill_weight_tail
        // moves only the placement), so the honest set admits it wherever it
        // sits — keyed on the carried role, never on the segment number.
        let a = op(
            "rd",
            vec![binding("t5", true, 256, SegRole::Weight, 6, 0, 0)],
        );
        assert!(walk_read_before_write(&[("list", &[a])]).is_ok());
    }

    #[test]
    fn an_op_with_a_descriptor_but_no_bindings_refuses() {
        // The rider: a descriptor op the walk cannot see is a walk-model
        // defect, never a pass. The refusal names the op.
        let a = descriptor_op_with_no_bindings("invisible");
        let err = walk_read_before_write(&[("list", &[a])]).unwrap_err();
        assert!(err.contains("'invisible'"), "{err}");
        assert!(err.contains("no arg bindings"), "{err}");
    }

    #[test]
    fn a_bare_marker_op_without_a_descriptor_passes() {
        // The ANTI-WEAKENING pin for the refusal above: it keys on the
        // DESCRIPTOR being present, not on the binding list being empty —
        // `EmittedOp::bare` markers carry no descriptor and no accesses, and
        // refusing those would make the refusal a blanket empty check that
        // says nothing about visibility.
        let a = op("marker", vec![]);
        assert!(walk_read_before_write(&[("list", &[a])]).is_ok());
    }

    #[test]
    fn an_lx_binding_is_skipped_not_judged() {
        // The carried LX verdict is the skip rule: a core-local binding's
        // bytes are the alternation walk's domain, and a read of never-written
        // HBM-interval bytes at the same base is still refused — the flag
        // skips exactly the LX binding and nothing beside it.
        let mut lx_rd = inter("t77", true, 256, 4096);
        lx_rd.lx = true;
        let hbm_rd = inter("t78", true, 256, 4096);
        let a = op("rdLX", vec![lx_rd]);
        assert!(walk_read_before_write(&[("list", &[a])]).is_ok());
        let b = op("rdHBM", vec![hbm_rd]);
        let err = walk_read_before_write(&[("list", &[b])]).unwrap_err();
        assert!(err.contains("t78"), "{err}");
    }
}
