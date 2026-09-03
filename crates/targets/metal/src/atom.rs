// SPDX-License-Identifier: Apache-2.0
//
// Backend-neutral atom abstraction for compiler-driven megakernel
// synthesis.
//
// An `Atom` is an `Implementation`'s view of itself as a composable
// primitive: it declares its dispatch shape, its I/O channels, and the
// per-backend code fragments that implement its body. The fuse pass
// (`fuse_pass.rs`, future) walks the solver's claim assignments,
// groups adjacent atoms whose dispatch shape and data flow are
// compatible, and emits one synthesized kernel per group by stitching
// the atom fragments together.
//
// Naming discipline: this module is backend-neutral on purpose. Atom
// kinds, signatures, channel types, and the trait surface do NOT
// reference Metal or CUDA. Only the per-backend `emit_*_body` methods
// produce backend-specific shader source. CUDA's `emit_cuda_body`
// lands in a later phase without touching this file.

use std::fmt;

/// Closed set of atom kinds the synthesis pass knows how to compose.
/// Extending the synthesis surface starts here.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AtomKind {
    /// Residual-add + RMS-scale + per-element normalize. Consumes
    /// `residual_in` + `delta` device buffers, produces a TG-memory
    /// normed activation buffer.
    AddRmsNorm,
    /// Standalone RMSNorm (no residual add). Layer-0 pre-attn shape.
    RmsNorm,
    /// Affine-int4 cooperative GEMV producing `tile_n` consecutive
    /// output rows. Consumes a TG-memory activation, produces TG-memory
    /// dot-product results.
    AffineQmv,
    /// NeoX-style RoPE pair rotation on Q.
    RopeRotateQ,
    /// NeoX-style RoPE pair rotation on K.
    RopeRotateK,
    /// Paged KV-cache element write (rotated K or pass-through V).
    KvPagedWrite,
    /// SiLU activation in registers.
    Silu,
    /// Per-element multiply (silu_mul second half).
    Mul,
    /// Fused `silu(gate) * up` over two TG-memory float channels,
    /// writing the result as `T_act` to a device buffer. Used in the
    /// MLP pre-down synth chunk.
    SiluMul,
    /// Residual write-back to the layer's residual buffer (one TG
    /// owns a disjoint slice; producer of the input to the next
    /// layer's AddRmsNorm).
    ResidualWriteBack,
}

/// Per-atom fuseability rule. The fuse pass consults this to decide
/// whether an atom can be merged with adjacent atoms into a single
/// synthesized kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fuseability {
    /// Fuse with adjacent atoms that share the same dispatch shape AND
    /// whose data-flow channels are register- or TG-memory-only.
    WithSameDispatch,
    /// Never fuse — emit this atom as its own kernel. Used for ops
    /// with incompatible dispatch shape (different threadgroup grid
    /// dims), for ops that take device-memory side effects with
    /// downstream global synchronization (attention, etc.), and for
    /// opaque library calls.
    Standalone,
}

/// Where an atom's I/O channel lives in the synthesized kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ChannelKind {
    /// Per-thread register / SSA value.
    Register,
    /// Threadgroup memory (shared across the TG's simdgroups).
    Threadgroup,
    /// Device memory: the channel is bound to one of the kernel's
    /// `[[buffer(N)]]` arguments. The fuse pass adds the binding to
    /// the synthesized kernel's signature; the atom emit references
    /// the buffer by the channel name.
    Device,
}

/// One named I/O channel of an atom. The fuse pass threads producer
/// `outputs[i].name` → consumer `inputs[i].name` when assigning kernel-
/// scope variable names.
#[derive(Clone, Debug)]
pub struct AtomChannel {
    /// Channel name. The fuse pass uses this as the substitution key
    /// in `AtomCtx::bound_*` — e.g. `"x"` for an atom's input slot is
    /// rewritten to the actual variable name in the synthesized
    /// kernel scope.
    pub name: String,
    /// Storage class for this channel.
    pub kind: ChannelKind,
    /// Element type as a Metal / CUDA source-level name (e.g.
    /// `"T_act"`, `"T_scale"`, `"uint"`, `"uint32_t"`, `"float"`). The
    /// fuse pass templates the synthesized kernel signature on the
    /// activation/scale dtype; concrete `half` / `bfloat` substitution
    /// happens at emit time.
    pub ty: String,
}

#[derive(Clone, Debug, Default)]
pub struct AtomSignature {
    pub inputs: Vec<AtomChannel>,
    pub outputs: Vec<AtomChannel>,
}

/// Per-atom dispatch shape. All atoms in a fuse group must share
/// dispatch shape — the fuse pass enforces this.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AtomDispatchShape {
    pub threadgroups: (u32, u32, u32),
    pub threads_per_threadgroup: (u32, u32, u32),
}

/// Substitutions + context the synthesizer hands to each atom's emit
/// method. The atom returns a code fragment that references the bound
/// channel names plus the constants. The fuse pass concatenates these
/// fragments to form the synthesized kernel body.
#[derive(Clone, Debug)]
pub struct AtomCtx<'a> {
    /// Concrete name in the synthesized kernel scope for each input
    /// channel. Same index as `AtomSignature::inputs`.
    pub bound_inputs: &'a [String],
    /// Same shape for outputs.
    pub bound_outputs: &'a [String],
    /// Per-shape constants the atom should reference by name (HIDDEN,
    /// NUM_Q_HEADS, HEAD_DIM, ROT_DIM, BLOCK_SIZE, M, GROUP_SIZE, EPS).
    /// The fuse pass declares these as `[[function_constant(N)]]` in
    /// the kernel signature once across all atoms in the group.
    pub constants: &'a [(&'static str, AtomConstantValue)],
    /// Activation dtype symbol (`"half"` / `"bfloat"`). Atom emits
    /// substitute this into their template fragments.
    pub t_act: &'static str,
    /// Scale dtype symbol (`"half"` for every affine-int4 model
    /// shipping today).
    pub t_scale: &'static str,
}

/// Atom constant values. Same set the lowering pass's
/// `ConstantValue` carries, but backend-neutral so atom emit can
/// reference constants without depending on a backend's specific
/// constant representation.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AtomConstantValue {
    Uint(u32),
    Int(i32),
    Float(f32),
}

/// An atom is a composable primitive of the synthesis pass. Each
/// `Implementation` that participates in fusion exposes itself as one.
///
/// Backend-neutral surface: `kind`, `signature`, `dispatch_shape`,
/// `fuseability` work for any backend. The per-backend emit hooks
/// (`emit_metal_body` / `emit_cuda_body`) return shader-source
/// fragments in their respective languages. Default-`None` so a
/// backend that doesn't support a particular atom yet falls back to
/// the existing per-Instruction emission path.
pub trait Atom: fmt::Debug {
    fn kind(&self) -> AtomKind;
    fn signature(&self) -> AtomSignature;
    fn dispatch_shape(&self, ctx: &AtomCtx) -> AtomDispatchShape;

    fn fuseability(&self) -> Fuseability {
        Fuseability::WithSameDispatch
    }

    /// Metal-source body fragment for this atom. Returns `None` if
    /// this atom doesn't (yet) support Metal.
    fn emit_metal_body(&self, _ctx: &AtomCtx) -> Option<String> {
        None
    }

    /// CUDA-source body fragment for this atom. Returns `None` if
    /// this atom doesn't (yet) support CUDA.
    fn emit_cuda_body(&self, _ctx: &AtomCtx) -> Option<String> {
        None
    }
}
