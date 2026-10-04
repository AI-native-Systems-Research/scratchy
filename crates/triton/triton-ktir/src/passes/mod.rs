// Copyright (c) 2026 IBM Corporation. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining
// a copy of this software and associated documentation files
// (the "Software"), to deal in the Software without restriction,
// including without limitation the rights to use, copy, modify, merge,
// publish, distribute, sublicense, and/or sell copies of the Software,
// and to permit persons to whom the Software is furnished to do so,
// subject to the following conditions:
//
// The above copyright notice and this permission notice shall be
// included in all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
// EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
// MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
// IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
// CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
// TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
// SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

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
