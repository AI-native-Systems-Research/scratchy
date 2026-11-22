// SPDX-License-Identifier: Apache-2.0
//! THE FORWARD TAPE — one decode step, as steps rather than as a function body.
//!
//! `superdsc_forward_chunk` spells a forward as ~110 lines of interleaved
//! `acts.push(...)` / `run_step(...)` / `read_tensor(...)`. Every one of those
//! lines is a kernel run somewhere plus a transfer, which is what
//! [`scratchy_subtile::host_tape`] models. This module supplies the two halves
//! that are spyre's: WHICH kernels a forward runs, and WHAT each produces.
//!
//! The third half — the launcher — is the WORKER's, because it owns the session
//! that a launch and a readback go through. That split is the architecture: the
//! kernels and the tape are per-model facts, the launcher is per-target I/O.

use crate::bundle_code::PlaceId;
use scratchy_subtile::host_tape::{KernelId, Operand, Site, Step, TensorId, Transfer};

/// What a host kernel produced, in the format it is uploaded in.
///
/// ⛔ THE DTYPE IS THE KERNEL'S, NOT THE CALLER'S. A value tensor is built as f32 and narrowed by
/// the bind loop; an index table and the chain's lane words are not values, and each rides the
/// channel that carries it verbatim.
pub enum Staged {
    /// ⭐ `Cow`, BECAUSE ONE KERNEL PRODUCES NOTHING. Every other host kernel computes its rows
    /// per forward and genuinely owns them; `Const(i)` hands back a table the macro already put
    /// in the binary, and copying it to satisfy an owned type was `2·hd²` floats per token for
    /// the identity and rope-P kernels alone.
    F32(std::borrow::Cow<'static, [f32]>),
    /// ⭐ INT32 INDEX WORDS, LITTLE-ENDIAN — the paged step's page table, write rows and query rows
    /// (a decode batch's chunk pages and replicated rows), at the card's own layout.
    ///
    /// ⛔ NOT A NARROWABLE NUMBER. These are ADDRESSES-IN-WAITING: the walk gathers the pool rows a
    /// page id names and the KV write scatters to the rows its index names. f16 carries integers
    /// exactly only to 2048 and a pool's rows pass that within a few pages, so narrowing would not
    /// lose precision visibly — it would gather and scatter a DIFFERENT ROW'S KEYS and read as fluent
    /// text.
    I32(Vec<u8>),
}

/// A forward's ARGUMENTS — the data its kernels compute FROM.
///
/// ⛔ NOT WHAT THEY PRODUCE. The tokens, the positions and the request's KV blocks are a forward's
/// inputs; the embedding rows, the rotary tables and the paged step's tables are what the kernels
/// COMPUTE from them. The binders this replaces conflated the two — they computed values inline
/// and pushed them — so there was nothing to name and nothing to reorder.
pub struct ForwardInputs<'a> {
    pub hidden: usize,
    /// The whole embedding table.
    pub embed_tokens: &'a [f32],
    /// Token id per row.
    pub tokens: &'a [usize],
    /// ABSOLUTE rotary position per row — a row's real position in ITS sequence, which for a
    /// decode batch is not its row index.
    pub positions: &'a [u32],
    /// ⭐ THE PAGED STEP'S TABLES ON THE CARD — [`crate::paged_step::card_step_sources`] /
    /// [`crate::paged_step::card_batch_sources`] of the request's KV blocks at the body that runs.
    /// `Some` exactly when the bundle states a paged step ([`ForwardShape::paged`]).
    ///
    /// ⛔ NOT DERIVABLE HERE. Which physical page holds a request's block is the scheduler's grant,
    /// and it CHURNS as requests are admitted, evicted and share a cached prefix — an input, read by
    /// the caller from the live block table, never recomputed from `positions`.
    pub paged: Option<&'a crate::paged_step::CardSources>,
}

/// Run one host kernel and return the values it fills its operand with.
///
/// ⭐ THIS IS THE WHOLE PER-MODEL FORWARD, and it does not touch a session — which is why it
/// lives in the target crate rather than the worker. The worker's launcher is left with h2d,
/// launch and readback.
pub fn kernel_values(
    step: &ForwardStep,
    shape: &ForwardShape,
    inp: &ForwardInputs<'_>,
) -> Result<Staged, String> {
    // ⭐ NO `head_dim`/`rope_theta` HERE ANY MORE. The rotary kernels read BOTH
    // off each source's own `RotaryKind` — a hybrid-attention arch (gemma-4)
    // has no single head dim or θ, and the model-wide pair this used to read
    // built every table from the WRONG class's facts.
    let (rows, h) = (shape.rows, inp.hidden);
    // ⭐⭐⭐ THE PAGED STEP, SERVED BEFORE THE f32 TAIL because all but one of its tables are not
    // floats at all.
    //
    // ⛔ A PAGED STEP WITH NO TABLES IS A REFUSAL, NOT AN EMPTY BUFFER. An unstaged page table reads
    // as ZERO, and page 0 is a real page: every block of the walk would read page 0's keys, and the
    // write would scatter this request's keys into it. Fluent, wrong and silent, so the absence is
    // named here rather than filled in.
    if let ForwardKernel::Paged = step.kernel {
        let tid = step.tensor.tid();
        let card = inp.paged.ok_or_else(|| {
            format!(
                "paged step t{tid} on a forward with no KV blocks: an unstaged page table reads as \
                 ZERO, and page 0 is a REAL page — every block would read page 0's keys"
            )
        })?;
        if let Some((_, mask)) = card.masks.iter().find(|(m, _)| *m == tid) {
            return Ok(Staged::F32(std::borrow::Cow::Owned(mask.clone())));
        }
        return card
            .words(tid)
            .map(|w| Staged::I32(w.to_vec()))
            .ok_or_else(|| {
                format!(
                    "paged step t{tid}: the step staged no such table — the forward's step and the \
                     bundle's are not one step"
                )
            });
    }
    Ok(Staged::F32(std::borrow::Cow::Owned(match step.kernel {
        ForwardKernel::EmbedRow => {
            let mut emb = vec![0.0f32; rows * h];
            for (r, &tok) in inp.tokens.iter().enumerate().take(rows) {
                let src = inp
                    .embed_tokens
                    .get(tok * h..(tok + 1) * h)
                    .ok_or_else(|| format!("token {tok} is past the embedding table"))?;
                emb[r * h..(r + 1) * h].copy_from_slice(src);
            }
            emb
        }
        ForwardKernel::Cos(i) | ForwardKernel::Sin(i) => {
            let want_cos = matches!(step.kernel, ForwardKernel::Cos(_));
            let srcs = if want_cos {
                &shape.cos_srcs
            } else {
                &shape.sin_srcs
            };
            let &(_, w, kind) = srcs.get(i).ok_or_else(|| format!("no rotary source {i}"))?;
            let (w, kind_hd) = (w as usize, kind.head_dim() as usize);
            let mut col = vec![0.0f32; rows * w];
            for r in 0..rows {
                let pos = *inp
                    .positions
                    .get(r)
                    .ok_or_else(|| format!("no position for row {r}"))?;
                // ⭐ THE SOURCE'S OWN CLASS, not the model-wide head_dim/θ: a
                // hybrid-attention arch (gemma-4) gives sliding and global
                // layers different tables, and the wrong one is silent garbage.
                let (c, sn) = crate::manifest::rope_cos_sin_kind(pos, kind);
                col[r * w..(r + 1) * w].copy_from_slice(&tile_rotary_row(
                    if want_cos { &c } else { &sn },
                    w,
                    kind_hd,
                ));
            }
            col
        }
        ForwardKernel::Paged => unreachable!("served above, where its index words stay INTEGERS"),
        ForwardKernel::Decode => {
            return Err("Decode is a device step, not a host call".to_string());
        }
    })))
}

/// Tile one rotary row across a source's full column width.
///
/// A GQA bundle has more than one rotary width (Q spans `hidden`, K spans
/// `kv_dim`), and the on-card rope consumes them full-width — so the
/// `[head_dim]` row is repeated `width / head_dim` times. This was a closure
/// defined TWICE in the worker, once per binder; it is the rotary kernel's own
/// rule and belongs with it.
pub fn tile_rotary_row(row: &[f32], width: usize, head_dim: usize) -> Vec<f32> {
    let mut buf = vec![0.0f32; width];
    for r in 0..(width / head_dim) {
        buf[r * head_dim..r * head_dim + head_dim].copy_from_slice(row);
    }
    buf
}

/// What a forward's host kernels are. Each produces exactly one bound value.
///
/// ⛔ THESE ARE KERNELS, not "the parts of the bind block". An embedding gather
/// and a rotary table are CPU compute with inputs and one output; naming them
/// so is what lets one of them later move into a device program by changing its
/// step's `site`, with nothing else changing.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ForwardKernel {
    /// Gather `embed_tokens[token]` — the row the tape's first device op reads.
    EmbedRow,
    /// Rotary `cos` / `sin` for this position, tiled to source `i`'s width.
    Cos(usize),
    Sin(usize),
    /// ⭐ ONE OF THE PAGED STEP'S TABLES ([`crate::paged_step`]), at the step's tensor: the page table
    /// the walk gathers its pages through, the mask stack it adds to each block's scores, the write
    /// rows the KV write scatters this step's keys and values to, and the query rows (a decode
    /// batch's replicated rows, chunk page ids and chunk masks instead). Present exactly when the
    /// bundle states a paged step ([`ForwardShape::paged`]).
    Paged,
    /// The device program itself.
    Decode,
}

/// One step of a forward: the kernel, and the tensor it fills.
#[derive(Clone, Copy, Debug)]
pub struct ForwardStep {
    pub kernel: ForwardKernel,
    /// ⛔ AN IDENTITY, NOT A NAME. This was an `ActName` enum carrying two SPELLINGS — `t{tid}`
    /// and `t{tid}_fp32` — because the K-split merge-init zero was bound under the second. The
    /// emitter names no such tensor (it seeds the accumulator from block 0), so that bind matched
    /// no placement and the executor's name-keyed lookup silently skipped it. The step is deleted
    /// and the spelling with it.
    pub tensor: PlaceId,
}

/// The forward's shape — WHICH kernels run, in order. Everything here is a
/// baked fact (source ids, counts, placement flags); nothing depends on the
/// token or the position, which are the kernels' INPUTS.
pub struct ForwardShape {
    /// How many token rows this forward runs. ⭐ DECODE IS `rows == 1`, NOT A SEPARATE SHAPE —
    /// the three lemmas in this module's tests are what makes that literally true rather than
    /// nearly true.
    pub rows: usize,
    pub embed_src: u32,
    /// Rotary sources as `(id, full column width, rotary class)`. GQA gives more
    /// than one width (Q spans `hidden`, K spans `kv_dim`), which is why this is
    /// a list and not a pair; the class says which table the source carries —
    /// a hybrid-attention arch (gemma-4) gives classes DIFFERENT tables.
    /// ⛔ BORROWED FROM THE EMITTED `Wiring`, NOT COPIED OUT OF IT. `to_vec()`-ing these
    /// duplicated a `&'static` slice the macro put in the binary, once per forward, per token.
    pub cos_srcs: &'static [(u32, u32, crate::wiring::RotaryKind)],
    pub sin_srcs: &'static [(u32, u32, crate::wiring::RotaryKind)],
    /// ⭐ THE BUNDLE'S PAGED STEPS — each attention class's tables' tids, so the tape carries a step
    /// per table iff the bundle placed it. A projection of the wiring's own field, never a caller's
    /// opinion: a walk with no staged table gathers through zeros, i.e. page 0.
    pub paged: crate::wiring::ClassSteps,
}

impl ForwardShape {
    /// The steps, in play order: host kernels fill their operands, then the
    /// device program runs, then the logits come back.
    pub fn steps(&self) -> Vec<ForwardStep> {
        let mut v = Vec::new();
        // The chunk's token rows, gathered on the host.
        v.push(ForwardStep {
            kernel: ForwardKernel::EmbedRow,
            tensor: PlaceId::Act(self.embed_src),
        });
        for (i, &(t, ..)) in self.cos_srcs.iter().enumerate() {
            v.push(ForwardStep {
                kernel: ForwardKernel::Cos(i),
                tensor: PlaceId::Act(t),
            });
        }
        for (i, &(t, ..)) in self.sin_srcs.iter().enumerate() {
            v.push(ForwardStep {
                kernel: ForwardKernel::Sin(i),
                tensor: PlaceId::Act(t),
            });
        }
        // ⭐ THE PAGED STEP'S TABLES — a step each, the ones its card placements stage every step.
        // See [`ForwardShape::paged`] for why the condition is the wiring and not a caller's opinion,
        // and [`kernel_values`] for why a forward with no KV blocks is a refusal rather than a buffer.
        for p in self.paged.iter() {
            for (tid, seat, _) in p.card_placements() {
                if seat == crate::wiring::CardSeat::Step {
                    v.push(ForwardStep {
                        kernel: ForwardKernel::Paged,
                        tensor: PlaceId::Act(tid),
                    });
                }
            }
        }
        v.push(ForwardStep {
            kernel: ForwardKernel::Decode,
            tensor: PlaceId::Act(u32::MAX),
        });
        v
    }
}

impl ForwardShape {
    /// Every tensor this tape would bind that the bundle did NOT place.
    ///
    /// ⭐⭐ THE PROJECTION, CHECKED AGAINST THE ARTIFACT. The shape is derived from the emitted
    /// `Wiring` and the tape binds what it names — but nothing had ever confirmed that those names
    /// are tensors this bundle actually has. They came from the same bake, so they agree by
    /// construction; "by construction" is exactly the argument that was also true of `t{tid}`
    /// before the ids and the spellings drifted, and of the K-split zero the worker bound for a
    /// tensor the emitter names nowhere.
    ///
    /// A bind with no placement is not an error the runtime notices: it was `continue` + an
    /// eprintln until this branch made it a refusal, and even now the refusal fires at STAGING
    /// time, per forward, on the card. This answers the same question once, at load.
    ///
    pub fn unplaced(&self, layout: &crate::bundle_code::BundleLayout<'_>) -> Vec<PlaceId> {
        self.steps()
            .into_iter()
            .filter(|s| !matches!(s.kernel, ForwardKernel::Decode))
            .map(|s| s.tensor)
            .filter(|id| layout.place(*id).is_none())
            .collect()
    }
}

/// The [`host_tape`] operands for `steps` — every host kernel uploads its
/// output; the device step transfers nothing (its inputs are already up, and
/// its logits come back through the launcher's own return).
///
/// [`host_tape`]: scratchy_subtile::host_tape
pub fn operands(steps: &[ForwardStep]) -> Vec<Operand> {
    steps
        .iter()
        .map(|s| match s.kernel {
            ForwardKernel::Decode => Operand::resident(TensorId(0)),
            _ => Operand {
                // Every forward operand is an `Act`, so the neutral `TensorId` handle is exactly
                // its tid — no synthetic (which would not be distinguishable by tid alone) ever
                // reaches the host.
                tensor: TensorId(s.tensor.tid()),
                transfer: Transfer::ToDevice,
            },
        })
        .collect()
}

/// The tape: a host step per producer, then the device step.
pub fn tape_steps<'a>(steps: &[ForwardStep], ops: &'a [Operand]) -> Vec<Step<'a>> {
    steps
        .iter()
        .enumerate()
        .map(|(i, s)| match s.kernel {
            ForwardKernel::Decode => Step {
                kernel: KernelId(i as u32),
                site: Site::Device,
                inputs: &[],
                outputs: &[],
            },
            _ => Step {
                kernel: KernelId(i as u32),
                site: Site::Host,
                inputs: &[],
                outputs: &ops[i..i + 1],
            },
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A single-class rotary table for the shape below — a const so the `&'static` slices can
    /// borrow it (a literal in place is a dropped temporary).
    const KIND: crate::wiring::RotaryKind = crate::wiring::RotaryKind::Default {
        head_dim: 64,
        theta_bits: 1_232_348_160, // f32::to_bits(1e6), spelled as bits: const-evaluable.
    };

    /// A granite-shaped step: 4 pages of 4 slots, 2 kv heads, a 3-row bundle walking every block.
    /// The reserved sentinels, as every real wiring's step carries them — window 0's tids in the
    /// card stagers and the placements ARE these.
    const STEP: crate::wiring::PagedStep = crate::wiring::PagedStep {
        mask: scratchy_spyre_bundle::reserved_tids::PAGED_MASK_TID,
        page_table: scratchy_spyre_bundle::reserved_tids::PAGED_PAGE_TABLE_TID,
        write_rows: scratchy_spyre_bundle::reserved_tids::PAGED_WRITE_ROWS_TID,
        query_rows: scratchy_spyre_bundle::reserved_tids::PAGED_QUERY_ROWS_TID,
        rep_rows: scratchy_spyre_bundle::reserved_tids::PAGED_REP_ROWS_TID,
        chunk_pages: scratchy_spyre_bundle::reserved_tids::PAGED_CHUNK_PAGES_TID,
        chunk_masks: scratchy_spyre_bundle::reserved_tids::PAGED_CHUNK_MASKS_TID,
        pages: 4,
        page_slots: 4,
        rows: 3,
        nkvh: 2,
        q_width: 128,
        head_size: 64,
        walk: 4,
        paged_rows: crate::paged_step::PagedRows::Sequence,
        // One request of the graph's 3 rows — the mint's own spelling, const: the pair's held
        // product IS the `rows` above, and each `None` arm is a fixture bug the const panic
        // names (const panic takes a literal, so the messages are).
        extent: match crate::wiring::QueryExtent::of_baked_rung(
            match crate::wiring::RungWidth::of_baked_rows(3) {
                ::core::option::Option::Some(w) => w,
                ::core::option::Option::None => {
                    ::core::panic!("the fixture's three rows are a baked width")
                }
            },
            crate::wiring::RungCap::ONE,
        ) {
            ::core::option::Option::Some(e) => e,
            ::core::option::Option::None => {
                ::core::panic!("the fixture's three rows by one request name a rung")
            }
        },
        attends: crate::wiring::AttnMask::Causal,
    };

    fn shape() -> ForwardShape {
        ForwardShape {
            rows: 3,
            embed_src: 0,
            cos_srcs: &[(5, 64, KIND), (7, 64, KIND)],
            sin_srcs: &[(6, 64, KIND), (8, 64, KIND)],
            paged: crate::wiring::ClassSteps {
                full: Some(STEP),
                sliding: None,
            },
        }
    }

    /// Everything a forward of `shape()` reads, with the step's tables `card`.
    fn inputs(card: Option<&crate::paged_step::CardSources>) -> ForwardInputs<'_> {
        ForwardInputs {
            hidden: 64,
            embed_tokens: &[],
            tokens: &[],
            positions: &[],
            paged: card,
        }
    }

    fn staged(kernel: ForwardKernel, tid: u32, inp: &ForwardInputs<'_>) -> Result<Staged, String> {
        let shape = shape();
        let step = shape
            .steps()
            .into_iter()
            .find(|s| s.kernel == kernel && s.tensor == PlaceId::Act(tid))
            .expect("the paged shape carries the step");
        kernel_values(&step, &shape, inp)
    }

    /// ⭐⭐⭐ THE TAPE'S TABLES ARE THE CARD STEP'S — `paged_step::card_step_sources`, the bytes the
    /// card's placements hold. The index tables arrive as INTEGER words, the mask as values.
    #[test]
    fn the_paged_steps_are_the_card_steps_tables() {
        let rows = crate::paged_step::StepRows::new(9, 3).expect("three rows is three rows");
        let card = crate::paged_step::card_step_sources(
            &STEP,
            &[crate::paged_step::RequestStep {
                blocks: &[3, 1, 0],
                rows,
            }],
        )
        .expect("staged");
        let inp = inputs(Some(&card));
        for t in [STEP.page_table, STEP.write_rows, STEP.query_rows] {
            match staged(ForwardKernel::Paged, t, &inp).expect("a table") {
                Staged::I32(w) => assert_eq!(Some(w.as_slice()), card.words(t), "t{t}"),
                _ => panic!("t{t}: an index table staged as anything but int32 words"),
            }
        }
        match staged(ForwardKernel::Paged, STEP.mask, &inp).expect("a mask stack") {
            Staged::F32(v) => assert_eq!(&v[..], &card.masks[0].1[..]),
            _ => panic!("a mask stack is a value tensor"),
        }
    }

    /// ⛔⛔⛔ A PAGED STEP WITH NO BLOCKS IS A REFUSAL — asserted, because the alternative reads
    /// page 0 for every block: an unstaged page table is zero, and page 0 is a real page.
    #[test]
    fn a_paged_step_with_no_blocks_refuses_rather_than_reading_page_zero() {
        let inp = inputs(None);
        for t in [STEP.page_table, STEP.mask, STEP.write_rows, STEP.query_rows] {
            match staged(ForwardKernel::Paged, t, &inp) {
                Err(e) => assert!(e.contains("page 0 is a REAL page"), "t{t}: {e}"),
                Ok(_) => panic!("t{t} staged with no blocks"),
            }
        }
    }

    /// A decode batch's tape stages the batch's own tables, never one sequence's.
    #[test]
    fn a_decode_batch_stages_its_own_tables() {
        let batch = ForwardShape {
            paged: crate::wiring::ClassSteps {
                full: Some(crate::wiring::PagedStep {
                    paged_rows: crate::paged_step::PagedRows::Batch,
                    ..STEP
                }),
                sliding: None,
            },
            ..shape()
        };
        let tids: Vec<PlaceId> = batch
            .steps()
            .iter()
            .filter(|s| s.kernel == ForwardKernel::Paged)
            .map(|s| s.tensor)
            .collect();
        for t in [
            STEP.rep_rows,
            STEP.chunk_pages,
            STEP.chunk_masks,
            STEP.write_rows,
        ] {
            assert!(tids.contains(&PlaceId::Act(t)), "t{t}");
        }
        for t in [STEP.page_table, STEP.mask, STEP.query_rows] {
            assert!(!tids.contains(&PlaceId::Act(t)), "t{t}");
        }
    }

    /// ⭐ THE ORDER IS THE CONTRACT. A forward binds the embedding row, then
    /// rotary, then the paged step's tables, and only then launches. Pinned as a sequence because the order is the one thing a
    /// hand-written bind block gets to be sloppy about and a tape cannot.
    #[test]
    fn a_forward_binds_then_launches_exactly_once() {
        let s = shape().steps();
        let kinds: Vec<ForwardKernel> = s.iter().map(|x| x.kernel).collect();
        assert_eq!(
            kinds,
            vec![
                ForwardKernel::EmbedRow,
                ForwardKernel::Cos(0),
                ForwardKernel::Cos(1),
                ForwardKernel::Sin(0),
                ForwardKernel::Sin(1),
                ForwardKernel::Paged,
                ForwardKernel::Paged,
                ForwardKernel::Paged,
                ForwardKernel::Paged,
                ForwardKernel::Decode,
            ]
        );
        assert_eq!(
            s.iter()
                .filter(|x| x.kernel == ForwardKernel::Decode)
                .count(),
            1,
            "exactly one device launch per forward"
        );
        assert_eq!(
            s.last().map(|x| x.kernel),
            Some(ForwardKernel::Decode),
            "the launch is LAST — every bind precedes it"
        );
    }

    /// A bundle that states no paged step reads none of its tables, so the steps do not exist.
    /// That is the shape a runtime `if let Some(..)` guard had: a bake fact deciding whether a bind
    /// happens. On a tape it decides whether a STEP EXISTS.
    #[test]
    fn a_bake_fact_removes_the_step_rather_than_branching() {
        let mut sh = shape();
        sh.paged = crate::wiring::ClassSteps::default();
        let kinds: Vec<ForwardKernel> = sh.steps().iter().map(|x| x.kernel).collect();
        assert!(!kinds.contains(&ForwardKernel::Paged));
        assert_eq!(kinds.last(), Some(&ForwardKernel::Decode));
    }

    /// Every host step uploads its own output; the device step moves nothing.
    #[test]
    fn host_steps_upload_and_the_device_step_does_not() {
        let s = shape().steps();
        let ops = operands(&s);
        let tape = tape_steps(&s, &ops);
        for (st, fs) in tape.iter().zip(&s) {
            match fs.kernel {
                ForwardKernel::Decode => {
                    assert_eq!(st.site, Site::Device);
                    assert!(st.outputs.is_empty(), "the launch uploads nothing");
                }
                _ => {
                    assert_eq!(st.site, Site::Host);
                    assert_eq!(st.outputs.len(), 1);
                    assert_eq!(st.outputs[0].transfer, Transfer::ToDevice);
                }
            }
        }
    }

    /// ⭐⭐ A TAPE THAT BINDS A TENSOR THE BAKE DID NOT PLACE IS CAUGHT AT LOAD. This is the
    /// check that would have caught the K-split merge-init zero: the worker bound
    /// `t{KSPLIT_ZERO_TID}_fp32` for years and the emitter named that tid NOWHERE, so the bind
    /// matched no placement and the executor's name-keyed lookup silently skipped it.
    #[test]
    fn a_step_whose_tensor_was_never_placed_is_reported() {
        use crate::bundle_code::{BundleLayout, Placement};
        let sh = ForwardShape { rows: 1, ..shape() };
        let bound: Vec<PlaceId> = sh
            .steps()
            .into_iter()
            .filter(|s| !matches!(s.kernel, ForwardKernel::Decode))
            .map(|s| s.tensor)
            .collect();
        let place = |id: PlaceId| Placement {
            id,
            segment: 0,
            bank: 0,
            offset: 0,
            size: 64,
            is_logits: false,
        };
        // Everything placed ⇒ nothing unplaced.
        let full = BundleLayout {
            places: std::borrow::Cow::Owned(bound.iter().map(|id| place(*id)).collect()),
            ..BundleLayout::default()
        };
        assert!(
            sh.unplaced(&full).is_empty(),
            "all bound tensors are placed"
        );

        // Drop ONE placement — the tape's bind for it now has nowhere to land.
        let missing = bound[1];
        let holey = BundleLayout {
            places: std::borrow::Cow::Owned(
                bound
                    .iter()
                    .filter(|id| **id != missing)
                    .map(|id| place(*id))
                    .collect(),
            ),
            ..BundleLayout::default()
        };
        assert_eq!(sh.unplaced(&holey), vec![missing]);
    }

    /// ⭐⭐⭐ THE TAPE STAGES THE LAUNCH'S ROWS, NOT THE BUNDLE'S CAPACITY.
    ///
    /// A decode step fills ONE row. The bundle it launches into is baked to HOLD many — granite's
    /// decode bundle holds 96. Taking `rows` from the bundle ("it already knows") staged 96 rows
    /// of embedding and 96 rotary rows on every decode step, into buffers one row is read from. The
    /// card stopped responding, and no host-side check said anything.
    #[test]
    fn a_decode_step_stages_one_row_not_the_bundles_capacity() {
        let one = ForwardShape { rows: 1, ..shape() };
        let many = ForwardShape {
            rows: 96,
            ..shape()
        };
        // The STEPS are identical — only the volume each fills differs, which is exactly why
        // taking the wrong number produced no structural complaint.
        let kinds = |s: &ForwardShape| -> Vec<ForwardKernel> {
            s.steps().into_iter().map(|x| x.kernel).collect()
        };
        assert_eq!(
            kinds(&one),
            kinds(&many),
            "the step list does not depend on rows"
        );
        // ⛔ The BYTES do: a rotary source is one row of its width per row the forward runs.
        let positions: Vec<u32> = (0..96).collect();
        let staged_len = |sh: &ForwardShape| -> usize {
            let inp = ForwardInputs {
                hidden: 64,
                embed_tokens: &[],
                tokens: &[],
                positions: &positions[..sh.rows],
                paged: None,
            };
            let cos = sh
                .steps()
                .into_iter()
                .find(|s| s.kernel == ForwardKernel::Cos(0))
                .expect("a rotary source");
            match kernel_values(&cos, sh, &inp).expect("a rotary table") {
                Staged::F32(v) => v.len(),
                _ => panic!("a rotary table is values"),
            }
        };
        assert_eq!(staged_len(&one), 64);
        assert_eq!(staged_len(&many), 96 * 64);
    }

    /// ⭐ NO TWO STEPS BIND THE SAME TENSOR — which is what makes the play ORDER a free choice.
    /// The two binders pushed their activations in different orders (decode did prefix before
    /// causal, prefill the reverse) and both were correct only because every bind targets a
    /// distinct placement. A duplicate would make last-write-wins the real semantics, and the
    /// tape would silently pick a winner by step order.
    #[test]
    fn every_step_binds_a_distinct_tensor() {
        let sh = ForwardShape { rows: 8, ..shape() };
        let mut seen = std::collections::BTreeSet::new();
        for st in sh.steps() {
            if matches!(st.kernel, ForwardKernel::Decode) {
                continue;
            }
            assert!(
                seen.insert(st.tensor),
                "{:?} re-binds {:?}",
                st.kernel,
                st.tensor
            );
        }
    }

    /// ⭐ EVERY OPERAND A FORWARD BINDS IS AN `Act`. This is what lets the neutral `TensorId`
    /// handle be the tid: a `Synth` shares its `of` tid with the tensor it derives from, so if
    /// one could reach the host the mapping would collide. Nothing here may be synthetic.
    #[test]
    fn every_bound_operand_is_a_real_tensor_never_a_synthetic() {
        for st in shape().steps() {
            assert!(
                matches!(st.tensor, PlaceId::Act(_)),
                "{:?} binds {:?}, which is device-internal",
                st.kernel,
                st.tensor
            );
        }
    }
}
