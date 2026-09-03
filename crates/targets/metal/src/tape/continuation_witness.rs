// SPDX-License-Identifier: Apache-2.0
//! Compile-time witness that every paged-attention dispatch derives its KV
//! geometry from ONE resolver.
//!
//! ## The bug class this closes
//! Chunked-prefill *continuation* (`num_computed_tokens > 0`, so this chunk's
//! `lq` query tokens attend the whole `[0, kv_len)` cached sequence) and
//! relocatable-span *rope-on-read* both require every attention kernel to agree
//! on (a) the K-axis basis and (b) the **unit** it indexes the per-token
//! `span_ids` label vector in. Historically each kernel re-derived these
//! independently, so a migration (e.g. per-block → per-token `span_ids`) could
//! update some kernels and silently miss others (the steel `.h` prefill kernel
//! was missed exactly this way) — invisible on a single chunk, garbage past it.
//!
//! ## The tooth
//! [`KvGeometry`] has *private* fields and is constructible **only** through
//! [`KvGeometry::resolve`]. The paged-attention constants
//! ([`crate::tape::kernel_constants::AttentionPrefillPagedConstants`]) carry it as a
//! **non-optional** field. Rust requires every field to be named in a struct
//! literal, and a private-field type cannot be literalled outside this module —
//! so a lowering arm that hand-builds an attention dispatch without threading a
//! resolved `KvGeometry` fails to compile ("missing field `geom`"). There is no
//! `Default`, no public constructor, no `_` fallback: the ONLY way to obtain the
//! value is `resolve`, the single site that decides the geometry. A NEW
//! attention arm therefore cannot be added without going through it.
//!
//! ## Honest boundary
//! Rust types cannot type-check arithmetic *inside* an MSL shader. This witness
//! guarantees the Rust dispatch layer (where the missed migration actually
//! lived — a kernel wired but not updated) and **emits** the resolved
//! `span_ids` unit as a function-constant so a shader can read it from one typed
//! source rather than hardcoding its own divisor.

/// How a paged-attention kernel bounds its K axis. Exhaustive by construction:
/// [`KvGeometry::resolve`] matches this with no `_` arm, so a new axis variant
/// fails the build until its geometry is defined here rather than defaulting to
/// the `num_computed_tokens == 0` (single-chunk) assumption.
#[derive(Copy, Clone, PartialEq, Eq, Debug)]
pub enum KvAxis {
    /// `K = [0, seq_used_k)`: prefix + this chunk's new tokens. The only
    /// continuation-capable axis.
    FullSeqUsed,
    /// `K =` this chunk's new tokens only; the prefix is structurally absent.
    NewTokensOnly,
}

/// Resolved KV geometry for one paged-attention dispatch. Private fields +
/// no public constructor + no `Default` ⇒ **unforgeable**: the only value of
/// this type in the program comes from [`KvGeometry::resolve`]. Carrying it as a
/// required field on an attention constants struct forces every dispatch arm
/// through that one resolver.
#[derive(Copy, Clone, Debug)]
pub struct KvGeometry {
    axis: KvAxis,
    /// Divisor applied to a kernel's absolute token position when indexing the
    /// per-token `span_ids` label vector. The host builds one label **per
    /// token** ([`scratchy_core_common::request::span_ids_per_token`]), so this
    /// is `1`. Emitted as a function-constant so a kernel can derive its
    /// span-index unit from here rather than a hardcoded `BLOCK_SIZE_`.
    span_index_unit: u32,
}

impl KvGeometry {
    /// The single site that resolves paged-attention KV geometry. Every
    /// `AttentionPrefillPagedConstants` construction must call this.
    pub fn resolve(axis: KvAxis) -> KvGeometry {
        let span_index_unit = match axis {
            KvAxis::FullSeqUsed | KvAxis::NewTokensOnly => 1,
        };
        KvGeometry {
            axis,
            span_index_unit,
        }
    }

    /// The per-token `span_ids` index divisor to emit as a function-constant.
    pub fn span_index_unit(&self) -> u32 {
        self.span_index_unit
    }

    /// The resolved K-axis (documents/enforces the continuation basis).
    pub fn axis(&self) -> KvAxis {
        self.axis
    }
}
