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
use crate::lower_subtile_tape_to_superdsc as sd;
use scratchy_subtile::host_tape::{KernelId, Operand, Site, Step, TensorId, Transfer};

/// What a host kernel produced, in the format it is uploaded in.
///
/// ⛔ THE DTYPE IS THE KERNEL'S, NOT THE CALLER'S. Every activation but one is built as f32 and
/// narrowed by the bind loop; the batched prefix mask is built as f16 DIRECTLY, because it is
/// quadratic in the request count and was 93% of every element that loop narrowed. That made it
/// the one bind the worker had to special-case — a second list (`acts_f16`) threaded through the
/// whole staging path. As a property of the OUTPUT it is just another kernel.
pub enum Staged {
    /// ⭐ `Cow`, BECAUSE ONE KERNEL PRODUCES NOTHING. Every other host kernel computes its rows
    /// per forward and genuinely owns them; `Const(i)` hands back a table the macro already put
    /// in the binary, and copying it to satisfy an owned type was `2·hd²` floats per token for
    /// the identity and rope-P kernels alone.
    F32(std::borrow::Cow<'static, [f32]>),
    /// IEEE-f16 bytes, already in the device's format.
    F16(Vec<u8>),
}

/// Where prefix validity comes from — the one genuinely two-regime input.
///
/// ⛔ NOT TWO PREDICATES, TWO SHAPES. A prompt chunk and a solo decode want ONE broadcast row over
/// `cap` columns; a decode BATCH wants one block per fold pass, `[nqh*mq, page_slots]`, because the
/// fold takes a pass per (request, page) and `fold_delta` steps the mask one block per pass. Same
/// question, different buffer — which is why this is an enum on the INPUT and not a second tape.
pub enum PrefixSource<'a, Valid: Fn(usize) -> bool> {
    /// `cap` additive entries; `holds(col)` carves validity out of `mask_neg`.
    Broadcast { cap: usize, holds: Valid },
    /// Fold-pass blocks, staged straight to f16 by the layout law itself.
    Blocks {
        shape: scratchy_subtile::sdsc_abstract::PrefixMaskShape<
            { scratchy_subtile::sdsc_abstract::POOL_STICK },
            { scratchy_subtile::sdsc_abstract::PagedKvPool::PAGE_SLOTS as u32 },
        >,
        fold: scratchy_subtile::sdsc_abstract::DeclaredFold,
        histories: &'a [scratchy_subtile::sdsc_abstract::KvHistory],
    },
}

/// A forward's ARGUMENTS — the data its kernels compute FROM.
///
/// ⛔ NOT WHAT THEY PRODUCE. The tokens, the positions and the validity predicates are a
/// forward's inputs; the embedding rows, the rotary tables and the staged masks are what the
/// kernels COMPUTE from them. The binders this replaces conflated the two — they computed values
/// inline and pushed them — so there was nothing to name and nothing to reorder.
///
/// `Valid` is a GENERIC BOUND, never `&dyn Fn`: monomorphised, no runtime dispatch.
pub struct ForwardInputs<'a, Valid: Fn(usize) -> bool> {
    pub hidden: usize,
    pub head_dim: usize,
    pub num_q_heads: usize,
    /// The stick-padded row count the emitter reads the causal mask at (`rows.div_ceil(64)*64`).
    pub mq_pad: usize,
    pub rope_theta: f32,
    /// The whole embedding table.
    pub embed_tokens: &'a [f32],
    /// Token id per row.
    pub tokens: &'a [usize],
    /// ABSOLUTE rotary position per row — a row's real position in ITS sequence, which for a
    /// decode batch is not its row index.
    pub positions: &'a [u32],
    /// `[rows, mq_pad]` causal validity from the regime's own `new_block_mask`.
    pub causal: &'a [f32],
    /// Prefix validity. `None` when the bundle placed no mask at all, in which case the shape
    /// carries no `PrefixMask` step either.
    pub prefix: Option<PrefixSource<'a, Valid>>,
    /// The baked constants, `(tid, values)`.
    pub consts: &'a [(u32, crate::wiring::ConstValues)],
    /// fp16-safe attention "-inf".
    pub mask_neg: f32,
    /// Stage the embedding and rotary tables stick-scattered. UNCONDITIONAL for the device, but
    /// [`stick_scatter`] is the identity at one row, so decode may pass either.
    pub stick_major: bool,
}

/// Run one host kernel and return the values it fills its operand with.
///
/// ⭐ THIS IS THE WHOLE PER-MODEL FORWARD, and it does not touch a session — which is why it
/// lives in the target crate rather than the worker. The worker's launcher is left with h2d,
/// launch and readback.
pub fn kernel_values<Valid: Fn(usize) -> bool>(
    step: &ForwardStep,
    shape: &ForwardShape,
    inp: &ForwardInputs<'_, Valid>,
) -> Result<Staged, String> {
    let (rows, h, hd) = (shape.rows, inp.hidden, inp.head_dim);
    // The one kernel whose output is already in the device's format.
    if let ForwardKernel::PrefixMask = step.kernel {
        let src = inp
            .prefix
            .as_ref()
            .ok_or_else(|| "PrefixMask step on a bundle with no mask".to_string())?;
        return Ok(match src {
            PrefixSource::Broadcast { cap, holds } => Staged::F32(std::borrow::Cow::Owned(
                broadcast_prefix_mask(*cap, inp.mask_neg, holds),
            )),
            PrefixSource::Blocks {
                shape,
                fold,
                histories,
            } => Staged::F16(
                scratchy_subtile::sdsc_abstract::decode_batch_prefix_mask_f16(
                    *shape,
                    *fold,
                    histories,
                    inp.mask_neg,
                ),
            ),
        });
    }
    // ⭐ SERVED BEFORE THE OWNED TAIL, because a constant is not COMPUTED here — it is a table
    // the macro put in the binary, and the arms below all build their rows. Routing it through
    // them cost a copy of `2·hd²` floats per token for the identity and rope-P kernels alone.
    if let ForwardKernel::Const(i) = step.kernel {
        return Ok(Staged::F32(
            inp.consts
                .get(i)
                .ok_or_else(|| format!("no constant {i}"))?
                .1
                .to_cow(),
        ));
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
            if inp.stick_major {
                stick_scatter(rows, h, &emb)
            } else {
                emb
            }
        }
        ForwardKernel::Cos(i) | ForwardKernel::Sin(i) => {
            let want_cos = matches!(step.kernel, ForwardKernel::Cos(_));
            let srcs = if want_cos {
                &shape.cos_srcs
            } else {
                &shape.sin_srcs
            };
            let w = srcs
                .get(i)
                .ok_or_else(|| format!("no rotary source {i}"))?
                .1 as usize;
            let mut col = vec![0.0f32; rows * w];
            for r in 0..rows {
                let pos = *inp
                    .positions
                    .get(r)
                    .ok_or_else(|| format!("no position for row {r}"))?;
                let (c, sn) = crate::manifest::rope_cos_sin(pos, hd, inp.rope_theta);
                col[r * w..(r + 1) * w].copy_from_slice(&tile_rotary_row(
                    if want_cos { &c } else { &sn },
                    w,
                    hd,
                ));
            }
            if inp.stick_major {
                stick_scatter(rows, w, &col)
            } else {
                col
            }
        }
        ForwardKernel::CausalMask => causal_tiled(inp.causal, rows, inp.mq_pad, inp.num_q_heads),
        ForwardKernel::PrefixMask => unreachable!("served above, where its dtype is decided"),
        ForwardKernel::Const(_) => unreachable!("served above, where it is borrowed not built"),
        ForwardKernel::Decode => {
            return Err("Decode is a device step, not a host call".to_string());
        }
    })))
}

/// The device's stick-scattered stage of a `[rows, width]` row-major activation:
/// element `(r, c)` lands at `(c/64)*(rows*64) + r*64 + (c%64)`.
///
/// ⭐ THIS IS WHY DECODE IS A CASE OF PREFILL, NOT A SECOND PATH. The prefill binder scatters its
/// embedding and its rotary tables this way (the device rope reads row `r` at `dev_off(r, 0)`);
/// the decode binder stages the same tensors ROW-MAJOR. Those look like two conventions, and the
/// prefill comment claims they agree at one row — [`stick_scatter_is_the_identity_at_one_row`]
/// PROVES it, which is what lets one tape serve both regimes.
///
/// `width` must be a whole number of 64-element sticks (the caller guards; a partial stick would
/// leave holes this cannot fill).
pub fn stick_scatter(rows: usize, width: usize, row_major: &[f32]) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * width];
    for r in 0..rows {
        for c in 0..width {
            out[(c / 64) * (rows * 64) + r * 64 + (c % 64)] = row_major[r * width + c];
        }
    }
    out
}

/// The BROADCAST prefix-validity mask: `cap` additive entries, 0 where `valid(col)` and
/// `mask_neg` elsewhere.
///
/// ⛔ EVERY SWEPT COLUMN GETS AN EXPLICIT VALUE. The ops sweep the placement's whole width, and an
/// additive-mask byte the host never wrote reads as ZERO — which means VALID. So the buffer is
/// filled with `mask_neg` first and validity is carved out of it, never the reverse.
///
/// The predicate is the REGIME: a prompt chunk asks `decode_prefix_col_valid(col, start)` (its
/// resident prefix is contiguous), a decode row asks which slots it ACTUALLY HOLDS — those stop
/// being the same question the moment a batch writes at a shared slot, because the shorter
/// requests' history is `[0, len) ∪ [shared, …)` and the hole between holds another request's
/// keys. [`a_hole_free_history_is_exactly_the_contiguous_prefix_law`] pins that they agree when
/// there is no hole, which is why one function serves both.
pub fn broadcast_prefix_mask(cap: usize, mask_neg: f32, valid: impl Fn(usize) -> bool) -> Vec<f32> {
    let mut m = vec![mask_neg; cap];
    for (col, slot) in m.iter_mut().enumerate() {
        if valid(col) {
            *slot = 0.0;
        }
    }
    m
}

/// Tile a `[mq, mq_pad]` causal mask into the BLOCK-MAJOR per-head buffer the emitter reads:
/// `nsub = mq_pad/64` blocks of `[nqh*mq, 64]`, block `j` holding causal columns `[64j, 64j+64)`.
///
/// ⛔ THE TILING IS A REAL TILE, NOT A BROADCAST. Causal validity varies per QUERY ROW (row `r`
/// attends new-block column `c` iff `c <= r`), so one mb-broadcast row would apply query row 0's
/// pattern to every row once `mq > 1`. And the emitter reads the mask through a FLAT handle whose
/// row stride IS the block width (64), so a single `[nqh*mq, mq_pad]` row-major buffer would
/// advance 64 per row where the true pitch is `mq_pad` — wrong columns for every row past the
/// first as soon as `mq_pad > 64`.
///
/// Written once because the decode binder had its own `[nqh, mq_pad]` loop over
/// `prefill_causal_col_valid(col, 0)`, which is this function at `mq == 1` —
/// [`the_decode_causal_buffer_is_the_general_tiling_at_one_row`] proves it.
pub fn causal_tiled(cmask: &[f32], mq: usize, mq_pad: usize, nqh: usize) -> Vec<f32> {
    let nsub = (mq_pad / 64).max(1);
    let mut out = Vec::with_capacity(nsub * nqh * mq * 64);
    for j in 0..nsub {
        for _h in 0..nqh {
            for r in 0..mq {
                let row = r * mq_pad + j * 64;
                out.extend_from_slice(&cmask[row..row + 64]);
            }
        }
    }
    out
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
    /// Prefix validity: 0 on slots this request holds, fp16-safe -inf elsewhere.
    PrefixMask,
    /// Causal validity for the new token's block.
    CausalMask,
    /// Baked constant `i` (see [`crate::wiring::synthetic_constants`]).
    Const(usize),
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
    /// Rotary sources as `(id, full column width)`. GQA gives more than one width (Q spans
    /// `hidden`, K spans `kv_dim`), which is why this is a list and not a pair.
    /// ⛔ BORROWED FROM THE EMITTED `Wiring`, NOT COPIED OUT OF IT. `to_vec()`-ing these
    /// duplicated a `&'static` slice the macro put in the binary, once per forward, per token.
    pub cos_srcs: &'static [(u32, u32)],
    pub sin_srcs: &'static [(u32, u32)],
    /// `false` when the bundle placed no broadcast prefix mask, so nothing reads one.
    pub prefix_mask: bool,
    pub n_consts: usize,
}

impl ForwardShape {
    /// The steps, in play order: host kernels fill their operands, then the
    /// device program runs, then the logits come back.
    pub fn steps(&self) -> Vec<ForwardStep> {
        let mut v = Vec::new();
        v.push(ForwardStep {
            kernel: ForwardKernel::EmbedRow,
            tensor: PlaceId::Act(self.embed_src),
        });
        for (i, &(t, _)) in self.cos_srcs.iter().enumerate() {
            v.push(ForwardStep {
                kernel: ForwardKernel::Cos(i),
                tensor: PlaceId::Act(t),
            });
        }
        for (i, &(t, _)) in self.sin_srcs.iter().enumerate() {
            v.push(ForwardStep {
                kernel: ForwardKernel::Sin(i),
                tensor: PlaceId::Act(t),
            });
        }
        if self.prefix_mask {
            v.push(ForwardStep {
                kernel: ForwardKernel::PrefixMask,
                tensor: PlaceId::Act(sd::ATTN_MASK_TID),
            });
        }
        v.push(ForwardStep {
            kernel: ForwardKernel::CausalMask,
            tensor: PlaceId::Act(sd::ATTN_CAUSAL_TID),
        });
        for i in 0..self.n_consts {
            // The const's own tid is resolved by the launcher from the same
            // list the kernel index addresses, so it is not repeated here.
            v.push(ForwardStep {
                kernel: ForwardKernel::Const(i),
                tensor: PlaceId::Act(u32::MAX),
            });
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
    /// `Const` steps are excluded — their tensor comes from the constant list, not the step.
    pub fn unplaced(&self, layout: &crate::bundle_code::BundleLayout<'_>) -> Vec<PlaceId> {
        self.steps()
            .into_iter()
            .filter(|s| !matches!(s.kernel, ForwardKernel::Const(_) | ForwardKernel::Decode))
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

    fn shape() -> ForwardShape {
        ForwardShape {
            rows: 1,
            embed_src: 0,
            cos_srcs: &[(5, 64), (7, 64)],
            sin_srcs: &[(6, 64), (8, 64)],
            prefix_mask: true,
            n_consts: 3,
        }
    }

    /// ⭐ THE ORDER IS THE CONTRACT. A forward binds the embedding row, then
    /// rotary, then masks, then constants, then the K-split init, and only then
    /// launches. Pinned as a sequence because the order is the one thing a
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
                ForwardKernel::PrefixMask,
                ForwardKernel::CausalMask,
                ForwardKernel::Const(0),
                ForwardKernel::Const(1),
                ForwardKernel::Const(2),
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

    /// A bundle that placed no broadcast prefix mask does not read one, so the
    /// step does not exist. That is the shape the runtime `if let Some(..)`
    /// guard had: a bake fact deciding whether a bind happens. On a tape it
    /// decides whether a STEP EXISTS.
    #[test]
    fn a_bake_fact_removes_the_step_rather_than_branching() {
        let mut sh = shape();
        sh.prefix_mask = false;
        let kinds: Vec<ForwardKernel> = sh.steps().iter().map(|x| x.kernel).collect();
        assert!(!kinds.contains(&ForwardKernel::PrefixMask));
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
            .filter(|s| !matches!(s.kernel, ForwardKernel::Const(_) | ForwardKernel::Decode))
            .map(|s| s.tensor)
            .collect();
        let place = |id: PlaceId| Placement {
            id,
            segment: 0,
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
    /// of embedding, 96 rotary rows and a 96-row causal mask on every decode step, into buffers
    /// one row is read from. The card stopped responding, and no host-side check said anything.
    ///
    /// `stick_scatter` is why this is silent rather than a length error: it is a PERMUTATION, so
    /// `rows * width` elements go in and `rows * width` come out. Ninety-six times too many of
    /// them, in a layout nothing rejects.
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
        // ⛔ The BYTES do. 96 rows of a 64-wide rotary source is 96x the buffer.
        assert_eq!(stick_scatter(1, 64, &vec![0.0; 64]).len(), 64);
        assert_eq!(stick_scatter(96, 64, &vec![0.0; 96 * 64]).len(), 96 * 64);
    }

    /// ⭐ THE PREFIX MASK IS ONE STEP WITH TWO BUFFER SHAPES, and the kernel — not the caller —
    /// decides which. A prompt chunk gets an f32 broadcast row; a decode BATCH gets f16 fold-pass
    /// blocks. Before this, only the f32 form was a tape step and the f16 form was staged by hand
    /// into a second list the whole prefill path threaded, which is why `owns_prefix` used to
    /// depend on the RUNTIME regime rather than on what the bake placed.
    #[test]
    fn the_prefix_kernel_picks_its_own_transfer_format() {
        use scratchy_subtile::sdsc_abstract as sa;
        let sh = ForwardShape { rows: 1, ..shape() };
        let step = sh
            .steps()
            .into_iter()
            .find(|s| s.kernel == ForwardKernel::PrefixMask)
            .expect("the shape carries a prefix step");
        let inp = ForwardInputs {
            hidden: 64,
            head_dim: 64,
            num_q_heads: 1,
            mq_pad: 64,
            rope_theta: 1e4,
            embed_tokens: &[0.0; 64],
            tokens: &[0],
            positions: &[0],
            causal: &[0.0; 64],
            prefix: Some(PrefixSource::Broadcast {
                cap: 8,
                holds: |c: usize| c < 3,
            }),
            consts: &[],
            mask_neg: -1.0,
            stick_major: false,
        };
        match kernel_values(&step, &sh, &inp).expect("broadcast staged") {
            Staged::F32(v) => {
                assert_eq!(v.len(), 8);
                assert_eq!(&v[..4], &[0.0, 0.0, 0.0, -1.0]);
            }
            Staged::F16(_) => panic!("a broadcast row is f32; the bind loop narrows it"),
        }
        let _ = sa::POOL_STICK; // the Blocks variant's shape params come from here
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
            if matches!(st.kernel, ForwardKernel::Const(_) | ForwardKernel::Decode) {
                continue; // a Const's tensor comes from the constant list, not the step
            }
            assert!(
                seen.insert(st.tensor),
                "{:?} re-binds {:?}",
                st.kernel,
                st.tensor
            );
        }
    }

    /// ⭐⭐ THE THIRD LEMMA. The prompt-chunk binder fills prefix validity from
    /// `decode_prefix_col_valid(col, start)`; the decode binder fills it from "which slots does
    /// this request hold". This pins that the second SUBSUMES the first — they agree exactly when
    /// the history has no hole — so the chunk form is not a separate convention.
    #[test]
    fn a_hole_free_history_is_exactly_the_contiguous_prefix_law() {
        use scratchy_subtile::sdsc_abstract::decode_prefix_col_valid;
        let neg = -7.5f32;
        for (cap, start) in [(64usize, 0usize), (64, 1), (128, 37), (256, 256)] {
            let by_law = broadcast_prefix_mask(cap, neg, |c| decode_prefix_col_valid(c, start));
            let held: std::collections::BTreeSet<usize> = (0..start).collect();
            let by_history = broadcast_prefix_mask(cap, neg, |c| held.contains(&c));
            assert_eq!(by_law, by_history, "cap={cap} start={start}");
        }
    }

    /// And a history WITH a hole differs — otherwise the batched case would be pointless. The hole
    /// holds another request's keys, so calling it valid feeds one request another's history.
    #[test]
    fn a_history_with_a_hole_is_not_the_contiguous_prefix() {
        use scratchy_subtile::sdsc_abstract::decode_prefix_col_valid;
        let neg = -7.5f32;
        // Holds [0,3) and the shared slot 9 — the hole [3,9) is another request's.
        let held: std::collections::BTreeSet<usize> = [0, 1, 2, 9].into_iter().collect();
        let by_history = broadcast_prefix_mask(16, neg, |c| held.contains(&c));
        let by_len = broadcast_prefix_mask(16, neg, |c| decode_prefix_col_valid(c, 4));
        assert_ne!(by_history, by_len, "a hole must not read as valid");
        assert_eq!(by_history[9], 0.0, "the shared slot IS held");
        assert_eq!(by_history[5], neg, "the hole is NOT");
    }

    /// ⭐⭐ THE SECOND LEMMA. The decode binder built its causal mask as its own `[nqh, mq_pad]`
    /// loop over `prefill_causal_col_valid(col, 0)`; prefill builds a `[mq, mq_pad]` block via
    /// `ChunkRows::new_block_mask` and tiles it block-major. This pins that the first is the
    /// second at one row — so the decode loop is deletable rather than a convention to maintain.
    #[test]
    fn the_decode_causal_buffer_is_the_general_tiling_at_one_row() {
        use scratchy_subtile::sdsc_abstract::prefill_causal_col_valid;
        let neg = -1234.5f32;
        for nqh in [1usize, 8, 32] {
            // What the decode binder wrote, inline.
            let mut want = vec![neg; nqh * 64];
            for head in 0..nqh {
                for col in 0..64 {
                    if prefill_causal_col_valid(col, 0) {
                        want[head * 64 + col] = 0.0;
                    }
                }
            }
            // The general path: one row of the same law, then the shared tiling.
            let mut row = vec![neg; 64];
            for (col, v) in row.iter_mut().enumerate() {
                if prefill_causal_col_valid(col, 0) {
                    *v = 0.0;
                }
            }
            assert_eq!(causal_tiled(&row, 1, 64, nqh), want, "nqh={nqh}");
        }
    }

    /// Past one row the tiling is block-major and genuinely per-row — so the lemma is not vacuous.
    #[test]
    fn the_causal_tiling_is_block_major_and_per_row() {
        let (mq, mq_pad, nqh) = (2usize, 128usize, 3usize);
        let cmask: Vec<f32> = (0..mq * mq_pad).map(|i| i as f32).collect();
        let got = causal_tiled(&cmask, mq, mq_pad, nqh);
        assert_eq!(got.len(), (mq_pad / 64) * nqh * mq * 64);
        // Block 0, head 0, row 1 starts at cmask[1*128 + 0] = 128.
        assert_eq!(got[64], 128.0);
        // Block 1 (columns 64..128), head 0, row 0 starts at cmask[0*128 + 64] = 64.
        assert_eq!(got[nqh * mq * 64], 64.0);
    }

    /// ⭐⭐ THE LEMMA THE TWO BINDERS REST ON. Decode stages row-major, prefill stick-scattered;
    /// they are the same bytes at one row, so a single row-parameterised tape can serve both and
    /// the decode path is not a separately-maintained convention. Fail-first: change the `rows*64`
    /// term to a constant `64` and this still passes (rows==1), which is why the multi-row case
    /// below is asserted too — the identity must hold at 1 and must NOT be assumed past it.
    #[test]
    fn stick_scatter_is_the_identity_at_one_row() {
        let src: Vec<f32> = (0..256).map(|i| i as f32).collect();
        assert_eq!(stick_scatter(1, 256, &src), src, "one row: no scatter");
        for w in [64usize, 128, 512, 2048] {
            let x: Vec<f32> = (0..w).map(|i| (i * 7 % 13) as f32).collect();
            assert_eq!(stick_scatter(1, w, &x), x, "width {w}");
        }
    }

    /// And past one row it is genuinely a scatter — otherwise the lemma above would be vacuous.
    #[test]
    fn stick_scatter_moves_elements_once_there_is_more_than_one_row() {
        let (rows, width) = (2usize, 128usize);
        let src: Vec<f32> = (0..rows * width).map(|i| i as f32).collect();
        let got = stick_scatter(rows, width, &src);
        assert_ne!(got, src, "two rows must interleave");
        // Row 1, column 64 → stick 1, so (1)*(2*64) + 1*64 + 0 = 192.
        assert_eq!(got[192], src[1 * width + 64]);
        // Every element still appears exactly once.
        let mut a = got.clone();
        let mut b = src.clone();
        a.sort_by(f32::total_cmp);
        b.sort_by(f32::total_cmp);
        assert_eq!(a, b, "a scatter is a permutation");
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
