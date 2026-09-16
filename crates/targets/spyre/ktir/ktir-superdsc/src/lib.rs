// SPDX-License-Identifier: Apache-2.0
//! `ktir-superdsc` — KTIR → SuperDSC (SDSC), as a leaf crate.
//!
//! ⭐ ONE `ktir → superdsc` LOWERING, FOR ANY KTIR PRODUCER. Hand it a KTIR function and it answers
//! with the SuperDSC descriptors a `dxp` bake compiles. It exists so a producer does not have to bring
//! its own: `triton-spyre`'s `triton-superdsc-lower` was an independent implementation that had never
//! run on a card, while these bodies are the ones proven on silicon, ported with only their door
//! changed.
//!
//! ⚠️ AND "PROVEN" MEANS THAT AND NOT MORE. Nothing checked in here discharges it: this crate has no
//! `tests/` directory and no golden JSON, and every fixture its comments cite (`sdsc_silu.json`,
//! `sdsc_interslicetranspose.json`, `test_gather_1core/sdsc_1.json`) lives in a separate `deeptools`
//! checkout, not in this repo. The in-crate tests are layout laws, refusals with negative controls, and
//! declared-set sweeps — not descriptor byte-identity. A `dip_standalone -s` comparison against the
//! emitter this was ported from is the check that would close that gap, and it has not been run for
//! this crate.
//!
//! ⛔ IT DEPENDS ON NO PRODUCER, AND THAT IS CHECKABLE, NOT ASPIRATIONAL:
//! `cargo tree -p ktir-superdsc -e normal` is `ktir-core`, `half`, `serde`, `serde_json`. Nothing
//! here reaches for a caller's IR — see [`ktir_node`] for the whole input contract.
//!
//! ⭐⭐⭐ WHAT THE CALLER SUPPLIES IS BINDINGS, NOT FACTS ABOUT THE PROGRAM. Everything the lowering
//! needs about a buffer, it reads from the KTIR itself: extents and strides from
//! `ktdp.construct_memory_view`, the element format from that view's attributes, a node's row count
//! from its own store windows, the attention multiplier / rmsnorm epsilon / scalarmul scale from the
//! splats the program states them as, the swept KV extent from the access tile taken over the
//! resident cache. What no program can state is which allocation each `index` parameter addresses —
//! a launch binds an address per parameter — so the caller numbers its buffers
//! ([`ktir_node::BufferId`]) and this crate treats the numbers as keys. It never asks what a buffer
//! is, which is why the contract needs no trait over the producer's tensor type.
//!
//! ⛔ THE ONE DOOR THE CALLER CROSSES IS MONOMORPHISATION. Head counts and head dim parameterise the
//! device layout (GQA grouping, slabs, head strides), so they must be `const`, not values:
//! [`emit::lower_ktir_to_superdsc::attn_at`]`::<NQH, NKVH, HD>` and
//! [`emit::lower_ktir_to_superdsc::rope_at`]`::<HD>` are generic entry points and the CALLER
//! instantiates them from its own model configs. That arm set is the consumer's model inventory, so a
//! leaf crate must not own it, and it cannot be received as data either.
//!
//! ⚠️ NOT YET LEAF-CLEAN: [`emit::EmittedOp`], this crate's own output type, still carries a
//! consumer's bake plan (`kv_page_fold`, `kv_batched_requests`, `kv_fold_rows`, `kv_page_slots`,
//! `kv_request`, `host_kv_write`, `kv_n_skip`, `slot_no_fuse`) alongside the descriptors it emits.
//! Separating what the lowering EMITS from what the caller SCHEDULES is outstanding work.
//!
//! Citations to `ibm/main` line numbers throughout are the port's provenance — the body each
//! lowering was ported from — and are deliberately kept.

/// Build-time interpreter that PROVES the emitted SDSC attention computes attention
/// (`softmax(q·kᵀ·scale+mask)·v`) over the real device addresses — a translation bug is a
/// `cargo build` panic, locking the math at compile time instead of discovering it on-card.
pub mod addr;
/// ⭐⭐⭐ THE LOWERING ITSELF — the SuperDSC emitter (`emit_sdsc`, the `assemble_*` family, `pw1`/`pw2`)
/// and `ibm/main`'s eight `lower_*_node` bodies re-driven by the KTIR. This is what the whole
/// extraction was for: any KTIR producer can reach the proven lowering, bringing nothing but its
/// program and its buffer numbers.
///
/// ⛔ A DIRECTORY MODULE BECAUSE OF ONE SEAL. `emit::bmm_site::TapeLoweringSite::witness` is
/// `pub(in crate::emit)`, so exactly the two files under here can mint the batch-inner-walk witness
/// and `ir::bridge::tiled_op_sdsc_op::attn` — which its doc always said could not — now really
/// cannot.
pub mod emit;
/// The per-head-axis COUNTS of a model as distinct types (`QueryHeads`/`KvHeads`/`HeadDim`/`Gqa`).
/// Here only because [`addr`] and [`sdsc_abstract`] name `Gqa`; the doors that turn a model config's
/// counts into these consts belong to the CALLER, for the reason the crate header gives — the arm set
/// is the consumer's model inventory.
pub mod head_counts;
/// THE TileIR PIPELINE — the assemblers' INPUT type (`TileOp`/`TileOpKind`/`TiledOp`) and the tiler
/// between them, plus the span-overflow refusal. Came in as an audit correction: the extraction filed
/// the lowering's input as `KtirNode`, but every assembler under it takes a `&TileOp`.
pub mod ir;
/// THE REQUEST TYPE — `KtirNode` (one node's program, plus the BINDINGS no program can state) and
/// the decode-ladder rung `ActiveCap`. What a KTIR producer hands in, as opposed to the device facts
/// or the caller's own plan.
pub mod ktir_node;
/// OPERAND IDENTITY — [`place::PlaceId`] and the [`place::SynthRole`] vocabulary of every
/// intermediate this lowering invents, plus the ONE site that spells an operand.
pub mod place;
/// THE WHOLE-BUNDLE MEMORY PLAN — `BundleLayout` and its family: every tensor's
/// `(segment, bank, offset, size, role)`, the synthetic allocator, and the arrangement authority.
/// ⛔ A STRUCT, NOT A TRAIT — it is a DEVICE memory plan, not scratchy's; the module header carries
/// the evidence and what a trait would have cost.
pub mod placement;
/// THE RESERVED TID SPACE — the tensor ids no model produces (RoPE `P`, the attention
/// scale/mask/causal constants, the rmsnorm Newton constants, the fp8 clamp bounds, the head-major
/// selectors), with the region table and both of its compile-time disjointness proofs. The
/// producer↔lowering ABI: a caller has to agree with these numbers to bind anything.
pub mod reserved_tids;
/// Geometry LAW types (`AttnGeometry` & friends): the typed device facts every address in a
/// descriptor is derived from. Mutually recursive with [`addr`] — that pair is one unit.
pub mod sdsc_abstract;
/// The shared build-time error for the SuperDSC emitter. A leaf below [`superdsc_opspec`] so the
/// typed witnesses and the wire/lowering layer raise the SAME error without a module cycle.
pub mod superdsc_error;
/// Typed FRONTEND core for the SuperDSC emitter — `OpSpec`/`TensorArg`/`WorkPlan` and the sealed
/// witnesses that make the historical SDSC bug classes unrepresentable rather than runtime-checked.
pub mod superdsc_opspec;
/// THE WIRE LAYER — `Dsc`/`SdscOp` and the whole `#[derive(Serialize)]` family, the three fp8
/// nested-fold generators, and the HBM segment constants. The SuperDSC TYPE itself: this is the
/// JSON dxp reads, and the typed core in [`superdsc_opspec`] lowers into it.
pub mod wire;
/// THE WORK DIVISION — the card constants plus every function that decides how one op's iteration
/// space is cut across cores (`core_split`, `CoreSplit`, `DeviceWidth`, `distribute_cores`,
/// `core_to_wk_slice`, the matmul cost search).
pub mod work;
