// SPDX-License-Identifier: Apache-2.0

//! The ported passes. Value in, value out; no pass may touch [`crate::text`].
//!
//! Each module names the C++ file it is ported from and reproduces its
//! DIAGNOSTICS as well as its rewrites -- a refusal is part of the behaviour, and
//! the `vector_add` / `mul` fixtures exist to prove it.

pub mod canonicalize;
pub mod convert_ttir_to_ktdp;
pub mod decompose_dense_constants;
pub mod distribute_work;
pub mod dot_to_linalg;
pub mod legalize_types;
pub mod plan_corelets;
pub mod to_ktir;
/// The second half of the KTDP -> KTIR stage: `to_ktir`'s rewrites, EMITTED as
/// `ktir_core::ir::IRFunction`.
///
/// This was a top-level `handoff` module, which made the pipeline look like five stages with a
/// bolt-on converter at the end. It is not a stage: it is how the KTDP -> KTIR stage states its
/// output. `to_ktir::lower` is the door; nothing else should call into here.
pub mod to_ktir_emit;
pub mod walk;
