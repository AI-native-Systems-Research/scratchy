// SPDX-License-Identifier: Apache-2.0
//
// Backend-neutral atom impls for the decoder body. Each `Atom` here
// captures the per-claim parameters (group_size, bits, head dims, ...)
// and emits a per-backend code fragment from `emit_metal_body` /
// `emit_cuda_body` (CUDA bodies land in a later phase).
//
// The fuse pass consumes these atoms (one per solver claim, when the
// Impl's `as_atom` returns `Some`), groups them by dispatch shape +
// data-flow compatibility, and stitches the emitted fragments into a
// single synthesized kernel.
//
// Channel naming convention (used in the emit fragments via
// `AtomCtx::bound_inputs` / `bound_outputs`):
//
//   "x_norm"   — TG-memory activation of length HIDDEN, post-RMSNorm.
//                Produced by AddRmsNormAtom / RmsNormAtom; consumed
//                by AffineQmvAtom.
//   "qmv_smem" — TG-memory dot-product result of length HEAD_DIM (for
//                the QKV variant) or TILE_N (for gate/up/down variants).
//                Produced by AffineQmvAtom; consumed by RopeAppend /
//                ResidualWriteBack / SiluMul.
//
// All atoms in a fuse group share the dispatch shape
// (M, NUM_HEADS_TOTAL, 1) × (MK_SIMD_SIZE * HEAD_DIM / MK_ROWS_PER_SIMDGROUP).
//
// Fragments use placeholder substitutions:
//   {T_act}   → AtomCtx::t_act         (e.g. "bfloat" / "half")
//   {T_scale} → AtomCtx::t_scale       ("half" universally for affine-int4)
//
// The fuse pass performs simple string substitution before writing
// the synthesized kernel to disk.

#![allow(dead_code)] // Atoms have no callers until the fuse pass lands in Phase 3.

use crate::atom::{
    Atom, AtomChannel, AtomConstantValue, AtomCtx, AtomDispatchShape, AtomKind, AtomSignature,
    ChannelKind, Fuseability,
};

/// AddRmsNorm atom: computes `residual_new = residual + delta`,
/// then `scale = rsqrt(mean((residual_new)²) + eps)`, then writes
/// `x_norm[i] = residual_new[i] * scale * rms_weight[i]` into TG
/// memory. TGs whose `head < NUM_Q_HEADS` additionally write
/// their `[head*HEAD_DIM, (head+1)*HEAD_DIM)` slice of
/// `residual_new` back to the residual buffer for the NEXT layer's
/// consumer. K/V-head TGs skip the device write.
///
/// Output channel `x_norm`: TG-mem `{T_act} x_norm[HIDDEN_MAX]`. The
/// fuse pass binds this channel to the kernel's actual TG-mem array
/// name and threads it to the AffineQmv consumer.
#[derive(Clone, Debug, Default)]
pub struct AddRmsNormAtom {
    /// Layer-0 mode: skip the `residual += delta` step and the
    /// per-Q-head residual writeback. The kernel reads `residual_io`
    /// as the already-correct input (i.e. the embedding output) and
    /// just normalizes it. `delta` is still bound (the lowering arm
    /// keeps the binding count stable) but the kernel never reads it.
    pub init: bool,
}

impl Atom for AddRmsNormAtom {
    fn kind(&self) -> AtomKind {
        AtomKind::AddRmsNorm
    }

    fn signature(&self) -> AtomSignature {
        AtomSignature {
            inputs: vec![
                AtomChannel {
                    name: "residual_io".into(),
                    kind: ChannelKind::Device,
                    ty: "{T_act}".into(),
                },
                AtomChannel {
                    name: "delta".into(),
                    kind: ChannelKind::Device,
                    ty: "const {T_act}".into(),
                },
                AtomChannel {
                    name: "rms_weight".into(),
                    kind: ChannelKind::Device,
                    ty: "const {T_scale}".into(),
                },
            ],
            outputs: vec![
                AtomChannel {
                    name: "x_norm".into(),
                    kind: ChannelKind::Threadgroup,
                    ty: "{T_act}".into(),
                },
                // The updated residual (`residual_in + delta`), written to
                // a DISTINCT device buffer from `residual_io` so the
                // per-Q-head writeback never races the cross-threadgroup
                // rmsnorm reads. Unused by the `init` variant (layer 0 has
                // no residual add); the lowering still binds it.
                AtomChannel {
                    name: "residual_out".into(),
                    kind: ChannelKind::Device,
                    ty: "{T_act}".into(),
                },
            ],
        }
    }

    fn dispatch_shape(&self, _ctx: &AtomCtx) -> AtomDispatchShape {
        // Per (token, output_head). Compatible with AffineQmvAtom,
        // RopeAppendAtom, etc.
        AtomDispatchShape {
            threadgroups: (0, 0, 1), // M and num_heads_total filled by fuse pass
            threads_per_threadgroup: (0, 1, 1), // 32 * HEAD_DIM / 4, filled by fuse pass
        }
    }

    fn fuseability(&self) -> Fuseability {
        Fuseability::WithSameDispatch
    }

    fn emit_metal_body(&self, ctx: &AtomCtx) -> Option<String> {
        let res = &ctx.bound_inputs[0]; // residual_io
        let del = &ctx.bound_inputs[1]; // delta
        let rw = &ctx.bound_inputs[2]; // rms_weight
        let out = &ctx.bound_outputs[0]; // x_norm (TG memory)

        let t_act = ctx.t_act;
        if self.init {
            // Layer-0 mode: `residual_io` is the (already-final) input
            // (the embedding output). No delta to add, no residual
            // writeback (the caller's downstream consumer reads the
            // same `residual_io` unchanged).
            return Some(format!(
                r#"
    // --- atom: AddRmsNorm (init: layer-0 / no residual add) ---
    {{
        device const {t_act}* __res_row = (device const {t_act}*){res} + (size_t)__t * (size_t)__hidden;
        float __local_sumsq = 0.0f;
        for (uint __i = __tid; __i < __hidden; __i += __threads_per_tg) {{
            const float __v = float(__res_row[__i]);
            __local_sumsq += __v * __v;
            {out}[__i] = {t_act}(__v);
        }}
        const float __scale = mk_tg_rmsnorm_scale(__local_sumsq, __hidden, __eps,
                                                  __scratch, __num_simdgroups,
                                                  __simd_gid, __simd_lid);
        for (uint __i = __tid; __i < __hidden; __i += __threads_per_tg) {{
            const float __v_pre  = float({out}[__i]);
            const float __w      = float({rw}[__i]);
            {out}[__i] = {t_act}(__v_pre * __scale * __w);
        }}
        mk_sync();
        (void){del};
    }}
"#,
                t_act = t_act,
                res = res,
                del = del,
                rw = rw,
                out = out,
            ));
        }
        // residual_out: the DISTINCT device buffer that receives the
        // updated residual (`residual_in + delta`). The coloring assigns
        // it a slot separate from `residual_io`, so the kernel reads the
        // input and writes here — never in place. Writing in place would
        // race cross-threadgroup: every threadgroup reads the whole
        // `residual_in` row for the rmsnorm sum, and Metal has no
        // cross-threadgroup barrier to order that against an in-place
        // write of `residual_in + delta` by a peer threadgroup.
        let res_out = &ctx.bound_outputs[1]; // residual_out (device)

        // Body sourced from fused_add_rmsnorm_affine_qkv_rope_cache.metal:
        // each thread reads its strided slice of (residual_in, delta) over
        // HIDDEN, accumulates sumsq, mk_tg_rmsnorm_scale, then writes
        // x_norm[i] = (residual_in + delta) * scale * rms_weight[i]. Q-head
        // TGs additionally write residual_new to residual_out (device).
        Some(format!(
            r#"
    // --- atom: AddRmsNorm ---
    {{
        device const {t_act}* __res_in_row  = {res} + (size_t)__t * (size_t)__hidden;
        device {t_act}*       __res_out_row = {res_out} + (size_t)__t * (size_t)__hidden;
        device const {t_act}* __del_row = {del} + (size_t)__t * (size_t)__hidden;
        float __local_sumsq = 0.0f;
        for (uint __i = __tid; __i < __hidden; __i += __threads_per_tg) {{
            const float __r = float(__res_in_row[__i]);
            const float __d = float(__del_row[__i]);
            const float __v = __r + __d;
            __local_sumsq += __v * __v;
            {out}[__i] = {t_act}(__v);
        }}
        const float __scale = mk_tg_rmsnorm_scale(__local_sumsq, __hidden, __eps,
                                                  __scratch, __num_simdgroups,
                                                  __simd_gid, __simd_lid);
        // Each Q-head writes its own disjoint [__res_slice_lo,
        // __res_slice_hi) slice of __res_out_row (a buffer no peer
        // threadgroup reads this dispatch), so the residual write never
        // races the cross-threadgroup rmsnorm reads of __res_in_row.
        const bool __writes_residual = (__head < __num_q);
        const uint __res_slice_lo = __head * __head_dim;
        const uint __res_slice_hi = __res_slice_lo + __head_dim;
        for (uint __i = __tid; __i < __hidden; __i += __threads_per_tg) {{
            const float __v_pre = float({out}[__i]);
            const float __w     = float({rw}[__i]);
            const float __normed = __v_pre * __scale * __w;
            if (__writes_residual && __i >= __res_slice_lo && __i < __res_slice_hi) {{
                __res_out_row[__i] = {t_act}(__v_pre);
            }}
            {out}[__i] = {t_act}(__normed);
        }}
        mk_sync();
    }}
"#,
            t_act = t_act,
            res = res,
            res_out = res_out,
            del = del,
            rw = rw,
            out = out,
        ))
    }
}

/// Affine-int4 cooperative GEMV. Reads a TG-memory activation
/// (`x_norm`), produces TG-memory dot-products (`qmv_smem`) of length
/// HEAD_DIM. Uses `mk_qmv_fast` to compute 4 outputs per simdgroup;
/// `simd_gid` selects the output row within the head's HEAD_DIM-sized
/// slice.
///
/// The `weight_packed` / `scales` / `biases` buffers index globally as
/// `(NUM_Q + 2 * NUM_KV) * HEAD_DIM` rows × HIDDEN columns — the
/// caller's lowering arm provides the concat'd `AffineQuantLinear`.
#[derive(Clone, Debug)]
pub struct AffineQmvAtom {
    pub group_size: u32,
    /// Weight quantization width. The `mk_*` templates in
    /// `metal_kittens.h` are bits-generic (4 and 8 both implemented);
    /// this picks the instantiation. Gemma4's MLP projections are
    /// 8-bit (`mlx-affine-b4-g64-mlp8`); everything else is 4.
    pub bits: u32,
    /// Expression (in synthesized-kernel scope) for the local-head
    /// index — i.e. the row index within this atom's weight band.
    /// Defaults to `"__head"` when the atom consumes a globally
    /// concatenated QKV weight buffer. Pre-attn synthesis with three
    /// separate Q/K/V triples sets this per band to `"__head"`,
    /// `"(__head - __num_q)"`, `"(__head - __num_q - __num_kv)"` so
    /// each band's qmv addresses rows starting at 0 within its own
    /// weight buffer.
    pub local_head_expr: &'static str,
    /// When `true`, the atom signature gains a 5th input channel
    /// `linear_bias: const {T_act}*` and the emitted body adds
    /// `linear_bias[band_row]` to each simd-sum result before it
    /// lands in `qmv_smem`. Used for Qwen2/Qwen2.5 QKV projections
    /// which carry a per-row linear bias on each `LinearLayer`. The
    /// matcher in `metal/synth_pre_attn.rs` walks through optional
    /// `BiasAdd` tiles between the Gemm and RopeAppend; when present,
    /// `fuse_pass` instantiates the atom with this flag set.
    pub has_linear_bias: bool,
}

impl Atom for AffineQmvAtom {
    fn kind(&self) -> AtomKind {
        AtomKind::AffineQmv
    }

    fn signature(&self) -> AtomSignature {
        let mut inputs = vec![
            AtomChannel {
                name: "x_norm".into(),
                kind: ChannelKind::Threadgroup,
                ty: "{T_act}".into(),
            },
            AtomChannel {
                name: "weight_packed".into(),
                kind: ChannelKind::Device,
                ty: "const uint32_t".into(),
            },
            AtomChannel {
                name: "scales".into(),
                kind: ChannelKind::Device,
                ty: "const {T_scale}".into(),
            },
            AtomChannel {
                name: "biases".into(),
                kind: ChannelKind::Device,
                ty: "const {T_scale}".into(),
            },
        ];
        if self.has_linear_bias {
            // Per-output-row linear bias (Qwen2 QKV). Stored in the
            // `LinearLayer::AffineQuant.linear_bias` field; bound by the
            // fuse_pass as a kernel-scope device pointer.
            inputs.push(AtomChannel {
                name: "linear_bias".into(),
                kind: ChannelKind::Device,
                ty: "const {T_act}".into(),
            });
        }
        AtomSignature {
            inputs,
            outputs: vec![AtomChannel {
                name: "qmv_smem".into(),
                kind: ChannelKind::Threadgroup,
                ty: "float".into(),
            }],
        }
    }

    fn dispatch_shape(&self, _ctx: &AtomCtx) -> AtomDispatchShape {
        AtomDispatchShape {
            threadgroups: (0, 0, 1),
            threads_per_threadgroup: (0, 1, 1),
        }
    }

    fn emit_metal_body(&self, ctx: &AtomCtx) -> Option<String> {
        let x = &ctx.bound_inputs[0]; // x_norm
        let w = &ctx.bound_inputs[1]; // weight_packed
        let s = &ctx.bound_inputs[2]; // scales
        let b = &ctx.bound_inputs[3]; // biases
        let out = &ctx.bound_outputs[0]; // qmv_smem

        let t_act = ctx.t_act;
        let t_scale = ctx.t_scale;
        let gs = self.group_size;
        let bits = self.bits;
        let local_head_expr = self.local_head_expr;

        // Bias epilogue: `__result[__row] += linear_bias[band_row]`
        // before the simd-sum reduction (so simd_sum still folds the
        // dot-product partial AND the bias into one scalar per row).
        // Bias index is local-band-relative — each AffineQmvAtom in the
        // pre-attn synth has its own band buffer (Q / K / V each have
        // a separate `linear_bias` channel).
        let bias_decl = if self.has_linear_bias {
            let lb = &ctx.bound_inputs[4];
            format!(
                "const device {t_act}* __lb = {lb} \
                 + (size_t)__local_head * __head_dim \
                 + __simd_gid * MK_ROWS_PER_SIMDGROUP;",
                t_act = t_act,
                lb = lb,
            )
        } else {
            String::new()
        };
        let bias_apply = if self.has_linear_bias {
            // Apply on simd-lane 0 only — the only lane that writes
            // qmv_smem. Adding on every lane (then storing on lane 0)
            // would still be correct since simd_sum is already
            // computed, but the lane-0 gate makes intent clear and
            // keeps the bias DRAM read off the other 31 lanes.
            "__result[__row] += float(__lb[__row]);".to_string()
        } else {
            String::new()
        };

        Some(format!(
            r#"
    // --- atom: AffineQmv (gs={gs}, local_head={local_head_expr}, has_linear_bias={has_lb}) ---
    {{
        constexpr int __bits              = {bits};
        constexpr int __pack_factor       = mk_get_pack_factor<__bits, 32>();
        constexpr int __bytes_per_pack    = mk_get_bytes_per_pack<__bits, 32>();
        constexpr int __values_per_thread = __pack_factor * MK_PACKS_PER_THREAD;
        constexpr int __scale_step        = {gs} / __values_per_thread;
        const uint __local_head           = ({local_head_expr});
        const uint __global_out_row_base = __local_head * __head_dim + __simd_gid * MK_ROWS_PER_SIMDGROUP;
        const int  __in_vec_size_w       = (int)__hidden * __bytes_per_pack / __pack_factor;
        const int  __in_vec_size_g       = (int)__hidden / {gs};
        const device uint8_t*  __ws = (const device uint8_t*){w}
            + (size_t)__global_out_row_base * (size_t)__in_vec_size_w
            + (size_t)__simd_lid * MK_PACKS_PER_THREAD * __bytes_per_pack;
        const device {t_scale}* __sc = {s}
            + (size_t)__global_out_row_base * (size_t)__in_vec_size_g
            + __simd_lid / __scale_step;
        const device {t_scale}* __bi = {b}
            + (size_t)__global_out_row_base * (size_t)__in_vec_size_g
            + __simd_lid / __scale_step;
        {bias_decl}
        thread float __x_thread[__values_per_thread];
        thread float __result[MK_ROWS_PER_SIMDGROUP] = {{ 0 }};
        const int __block_size = __values_per_thread * MK_SIMD_SIZE;
        threadgroup {t_act}* __x_tg = {x} + __simd_lid * __values_per_thread;
        const device uint8_t*  __ws_iter = __ws;
        const device {t_scale}* __sc_iter = __sc;
        const device {t_scale}* __bi_iter = __bi;
        // Full blocks only. When __hidden is NOT a multiple of
        // __block_size (e.g. Qwen2.5-0.5B hidden=896, block=512) the
        // unconditional `__k < __hidden` loop over-reads the final
        // partial block: for the last output row that read runs past the
        // weight/scale buffers entirely (undefined memory → NaN, and
        // 0*NaN = NaN poisons the row). Mirror MLX's qmv: iterate full
        // blocks, then a bounded safe tail. `__k_full == __hidden` when
        // aligned, so the tail below is a no-op for 3B / Qwen3.5 / Llama.
        const int __k_full = (int)__hidden - ((int)__hidden % __block_size);
        for (int __k = 0; __k < __k_full; __k += __block_size) {{
            float __sum = mk_load_vector<{t_act}, float, __values_per_thread, __bits>(__x_tg, __x_thread);
            for (int __row = 0; __row < MK_ROWS_PER_SIMDGROUP; __row++) {{
                const device uint8_t*  __wl = __ws_iter + __row * __in_vec_size_w;
                float __s = float(__sc_iter[__row * __in_vec_size_g]);
                float __b = float(__bi_iter[__row * __in_vec_size_g]);
                __result[__row] += mk_qdot<float, __values_per_thread, __bits>(__wl, __x_thread, __s, __b, __sum);
            }}
            __ws_iter += __block_size * __bytes_per_pack / __pack_factor;
            __sc_iter += __block_size / {gs};
            __bi_iter += __block_size / {gs};
            __x_tg    += __block_size;
        }}
        // Bounded remainder tail (each lane dots only its valid K range,
        // so no read runs past the weight / scale / x buffers).
        {{
            const int __remaining = clamp(
                (int)__hidden - __k_full - (int)__simd_lid * __values_per_thread,
                0, __values_per_thread);
            if (__remaining > 0) {{
                float __sum = mk_load_vector_safe<{t_act}, float, __values_per_thread, __bits>(__x_tg, __x_thread, __remaining);
                for (int __row = 0; __row < MK_ROWS_PER_SIMDGROUP; __row++) {{
                    const device uint8_t*  __wl = __ws_iter + __row * __in_vec_size_w;
                    float __s = float(__sc_iter[__row * __in_vec_size_g]);
                    float __b = float(__bi_iter[__row * __in_vec_size_g]);
                    __result[__row] += mk_qdot_safe<float, __values_per_thread, __bits>(__wl, __x_thread, __s, __b, __sum, __remaining);
                }}
            }}
        }}
        for (int __row = 0; __row < MK_ROWS_PER_SIMDGROUP; __row++) {{
            __result[__row] = simd_sum(__result[__row]);
            if (__simd_lid == 0) {{
                {bias_apply}
                {out}[__simd_gid * MK_ROWS_PER_SIMDGROUP + __row] = __result[__row];
            }}
        }}
        mk_sync();
    }}
"#,
            t_act = t_act,
            t_scale = t_scale,
            gs = gs,
            bits = bits,
            local_head_expr = local_head_expr,
            has_lb = self.has_linear_bias,
            x = x,
            w = w,
            s = s,
            b = b,
            bias_decl = bias_decl,
            bias_apply = bias_apply,
            out = out,
        ))
    }
}

/// RopeAppend atom: reads `qmv_smem[HEAD_DIM]`, applies NeoX RoPE on
/// the rotational dims, writes:
///   - For `head < NUM_Q`: rotated values to `q_out[t, head, :]`.
///   - For `head < NUM_Q + NUM_KV` (K head): rotated values to
///     `kv_cache_k[block_id, kv_head, block_offset, :]`.
///   - Else (V head): pass-through values to `kv_cache_v[...]`.
///
/// Skips the cache write when `slot_mapping[t] == 0xFFFFFFFF` (padding
/// lane sentinel) to match the standalone `rope_append_*_specialized`
/// kernel's behavior.
#[derive(Clone, Debug)]
pub struct RopeAppendAtom;

impl Atom for RopeAppendAtom {
    fn kind(&self) -> AtomKind {
        // The kind discriminator is shared for the Q / K / V branches
        // since one atom emits the whole epilogue (per-band switch
        // happens inside the emitted body).
        AtomKind::KvPagedWrite
    }

    fn signature(&self) -> AtomSignature {
        AtomSignature {
            inputs: vec![
                AtomChannel {
                    name: "qmv_smem".into(),
                    kind: ChannelKind::Threadgroup,
                    ty: "float".into(),
                },
                AtomChannel {
                    name: "cos_sin".into(),
                    kind: ChannelKind::Device,
                    ty: "const {T_act}".into(),
                },
                AtomChannel {
                    name: "positions".into(),
                    kind: ChannelKind::Device,
                    ty: "const uint".into(),
                },
                AtomChannel {
                    name: "slot_mapping".into(),
                    kind: ChannelKind::Device,
                    ty: "const uint".into(),
                },
            ],
            outputs: vec![
                AtomChannel {
                    name: "q_out".into(),
                    kind: ChannelKind::Device,
                    ty: "{T_act}".into(),
                },
                AtomChannel {
                    name: "kv_cache_k".into(),
                    kind: ChannelKind::Device,
                    ty: "{T_act}".into(),
                },
                AtomChannel {
                    name: "kv_cache_v".into(),
                    kind: ChannelKind::Device,
                    ty: "{T_act}".into(),
                },
            ],
        }
    }

    fn dispatch_shape(&self, _ctx: &AtomCtx) -> AtomDispatchShape {
        AtomDispatchShape {
            threadgroups: (0, 0, 1),
            threads_per_threadgroup: (0, 1, 1),
        }
    }

    fn emit_metal_body(&self, ctx: &AtomCtx) -> Option<String> {
        let qmv = &ctx.bound_inputs[0];
        let cs = &ctx.bound_inputs[1];
        let pos = &ctx.bound_inputs[2];
        let slm = &ctx.bound_inputs[3];
        let qo = &ctx.bound_outputs[0];
        let kc = &ctx.bound_outputs[1];
        let vc = &ctx.bound_outputs[2];

        let t_act = ctx.t_act;

        Some(format!(
            r#"
    // --- atom: RopeAppend + KvPagedWrite ---
    {{
        if (__simd_lid == 0) {{
            const uint __base_d = __simd_gid * MK_ROWS_PER_SIMDGROUP;
            const uint __kQ_END = __num_q;
            const uint __kK_END = __num_q + __num_kv;
            if (__head < __kQ_END) {{
                device {t_act}* __q_row = {qo}
                    + (size_t)__t    * (size_t)(__num_q * __head_dim)
                    + (size_t)__head * (size_t)__head_dim;
                if (__base_d < __half_dim) {{
                    const uint __p = {pos}[__t];
                    device const {t_act}* __cos_row = {cs} + (size_t)__p * (size_t)__rot_dim;
                    device const {t_act}* __sin_row = __cos_row + __half_dim;
                    for (int __r = 0; __r < MK_ROWS_PER_SIMDGROUP; __r++) {{
                        const uint __d = __base_d + __r;
                        float __x0 = {qmv}[__d];
                        float __x1 = {qmv}[__half_dim + __d];
                        mk_rope_pair(__x0, __x1, float(__cos_row[__d]), float(__sin_row[__d]));
                        __q_row[__d]            = {t_act}(__x0);
                        __q_row[__half_dim + __d] = {t_act}(__x1);
                    }}
                }} else if (__base_d >= __rot_dim) {{
                    for (int __r = 0; __r < MK_ROWS_PER_SIMDGROUP; __r++)
                        __q_row[__base_d + __r] = {t_act}({qmv}[__base_d + __r]);
                }}
            }} else if (__head < __kK_END) {{
                const uint __kv_head = __head - __num_q;
                const uint __slot    = {slm}[__t];
                if (__slot != 0xFFFFFFFFu) {{
                    const uint __block_id     = __slot / __block_sz;
                    const uint __block_offset = __slot % __block_sz;
                    // Chunked KV: {kc} is the per-layer chunk-address
                    // table (device uint64 gpuAddresses), not the cache
                    // buffer. Deref the chunk that backs this block,
                    // then address with the block index WITHIN the chunk.
                    const uint __chunk        = __block_id / BLOCKS_PER_CHUNK;
                    const uint __blk_in_chunk = __block_id % BLOCKS_PER_CHUNK;
                    device {t_act}* __k_base = (device {t_act}*){kc}[__chunk];
                    device {t_act}* __k_dst = __k_base
                        + (size_t)__blk_in_chunk * (size_t)(__num_kv * __block_sz * __head_dim)
                        + (size_t)__kv_head      * (size_t)(__block_sz * __head_dim)
                        + (size_t)__block_offset * (size_t)__head_dim;
                    if (__base_d < __half_dim) {{
                        const uint __p = {pos}[__t];
                        device const {t_act}* __cos_row = {cs} + (size_t)__p * (size_t)__rot_dim;
                        device const {t_act}* __sin_row = __cos_row + __half_dim;
                        for (int __r = 0; __r < MK_ROWS_PER_SIMDGROUP; __r++) {{
                            const uint __d = __base_d + __r;
                            float __x0 = {qmv}[__d];
                            float __x1 = {qmv}[__half_dim + __d];
                            mk_rope_pair(__x0, __x1, float(__cos_row[__d]), float(__sin_row[__d]));
                            __k_dst[__d]            = {t_act}(__x0);
                            __k_dst[__half_dim + __d] = {t_act}(__x1);
                        }}
                    }} else if (__base_d >= __rot_dim) {{
                        for (int __r = 0; __r < MK_ROWS_PER_SIMDGROUP; __r++)
                            __k_dst[__base_d + __r] = {t_act}({qmv}[__base_d + __r]);
                    }}
                }}
            }} else {{
                const uint __kv_head = __head - __kK_END;
                const uint __slot    = {slm}[__t];
                if (__slot != 0xFFFFFFFFu) {{
                    const uint __block_id     = __slot / __block_sz;
                    const uint __block_offset = __slot % __block_sz;
                    // Chunked KV: {vc} is the per-layer chunk-address table.
                    const uint __chunk        = __block_id / BLOCKS_PER_CHUNK;
                    const uint __blk_in_chunk = __block_id % BLOCKS_PER_CHUNK;
                    device {t_act}* __v_base = (device {t_act}*){vc}[__chunk];
                    device {t_act}* __v_dst = __v_base
                        + (size_t)__blk_in_chunk * (size_t)(__num_kv * __block_sz * __head_dim)
                        + (size_t)__kv_head      * (size_t)(__block_sz * __head_dim)
                        + (size_t)__block_offset * (size_t)__head_dim;
                    for (int __r = 0; __r < MK_ROWS_PER_SIMDGROUP; __r++)
                        __v_dst[__base_d + __r] = {t_act}({qmv}[__base_d + __r]);
                }}
            }}
        }}
    }}
"#,
            t_act = t_act,
            qmv = qmv,
            cs = cs,
            pos = pos,
            slm = slm,
            qo = qo,
            kc = kc,
            vc = vc,
        ))
    }
}

/// Gate activation for the fused act(gate)*up epilogue.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum MlpAct {
    Silu,
    /// `gelu_pytorch_tanh` with the fast-math clamp (tanh argument
    /// clamped to ±15 — Metal fast tanh = (exp(2x)-1)/(exp(2x)+1)
    /// → NaN past |x|≈44; tanh(15) rounds to exactly 1.0f, so the
    /// clamp is bit-exact vs a saturating tanh). Mirrors
    /// `silu_mul.metal::gelu_mul`.
    Gelu,
}

/// SiluMul atom: reads two TG-memory float vectors (`gate_smem`,
/// `up_smem`), each of length `__head_dim` (= TILE_N in the MLP
/// pre-down synth kernel scope), computes `act(g) * u` per element
/// (`act` = silu `g / (1 + exp(-g))` or clamped tanh-GELU, per
/// [`MlpAct`]), and writes the result as `T_act` to the device
/// buffer `silu_mul_out` at row `__t`, tile-base
/// `__head * __head_dim`. The full output row width is
/// `__intermediate` columns — kernel scope must declare it.
///
/// Same per-simdgroup row-of-MK_ROWS_PER_SIMDGROUP layout as
/// `RopeAppendAtom`'s epilogue — only lane 0 of each simdgroup
/// stores; the floats it reads from TG memory were produced by
/// `AffineQmvAtom` (which has `qmv_smem[__simd_gid*4 + r]` as its
/// per-simdgroup write target, aliased here to `gate_smem` and
/// `up_smem`).
#[derive(Clone, Debug)]
pub struct SiluMulAtom {
    pub act: MlpAct,
}

impl Default for SiluMulAtom {
    fn default() -> Self {
        Self { act: MlpAct::Silu }
    }
}

impl Atom for SiluMulAtom {
    fn kind(&self) -> AtomKind {
        // The kind discriminator is shared with the standalone
        // SiluMul Instruction. The atom emits the per-tile epilogue
        // that the standalone kernel would otherwise emit as its
        // own dispatch.
        AtomKind::SiluMul
    }

    fn signature(&self) -> AtomSignature {
        AtomSignature {
            inputs: vec![
                AtomChannel {
                    name: "gate_smem".into(),
                    kind: ChannelKind::Threadgroup,
                    ty: "float".into(),
                },
                AtomChannel {
                    name: "up_smem".into(),
                    kind: ChannelKind::Threadgroup,
                    ty: "float".into(),
                },
            ],
            outputs: vec![AtomChannel {
                name: "silu_mul_out".into(),
                kind: ChannelKind::Device,
                ty: "{T_act}".into(),
            }],
        }
    }

    fn dispatch_shape(&self, _ctx: &AtomCtx) -> AtomDispatchShape {
        AtomDispatchShape {
            threadgroups: (0, 0, 1),
            threads_per_threadgroup: (0, 1, 1),
        }
    }

    fn fuseability(&self) -> Fuseability {
        Fuseability::WithSameDispatch
    }

    fn emit_metal_body(&self, ctx: &AtomCtx) -> Option<String> {
        let g = &ctx.bound_inputs[0]; // gate_smem
        let u = &ctx.bound_inputs[1]; // up_smem
        let out = &ctx.bound_outputs[0]; // silu_mul_out (device)

        let t_act = ctx.t_act;
        let act_expr = match self.act {
            MlpAct::Silu => "const float __sg = __gv / (1.0f + exp(-__gv));",
            MlpAct::Gelu => {
                "const float __inner = clamp(0.7978845608f * (__gv + 0.044715f * __gv * __gv * __gv), -15.0f, 15.0f);\n                const float __sg = 0.5f * __gv * (1.0f + tanh(__inner));"
            }
        };

        Some(format!(
            r#"
    // --- atom: SiluMul ---
    {{
        if (__simd_lid == 0) {{
            const uint __base_d = __simd_gid * MK_ROWS_PER_SIMDGROUP;
            device {t_act}* __out_row = {out}
                + (size_t)__t    * (size_t)__intermediate
                + (size_t)__head * (size_t)__head_dim;
            for (int __r = 0; __r < MK_ROWS_PER_SIMDGROUP; __r++) {{
                const uint __d  = __base_d + (uint)__r;
                const float __gv = {g}[__d];
                const float __uv = {u}[__d];
                {act_expr}
                __out_row[__d] = {t_act}(__sg * __uv);
            }}
        }}
    }}
"#,
            t_act = t_act,
            g = g,
            u = u,
            out = out,
            act_expr = act_expr,
        ))
    }
}

// Silence unused-import warnings in this scaffolding module.
#[allow(dead_code)]
fn _atom_lib_uses() {
    let _: AtomConstantValue = AtomConstantValue::Uint(0);
}
