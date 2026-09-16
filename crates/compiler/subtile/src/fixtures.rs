// SPDX-License-Identifier: Apache-2.0
//! Shared `LoweringInput` fixtures used by orchestrator tests, the
//! `dump_tk` example, and the `tk_emit_decode` binary.
//!
//! These are test/dev fixtures only — production wiring derives
//! `LoweringInput` directly from a solved decode FUF via the proc-macro.

use crate::lower::{InputRef, LoweredOp, LoweringInput, OpDesc};
use crate::subtile_ir::SourceShape;
use ktir_superdsc::head_counts::{HeadDim, KvHeads, ModelAttnGeometry, QueryHeads};

/// Minimal one-layer Llama-3.2-1B-style decode forward.
///
/// Source IDs:
///   0: x         [1, 2048]
///   1: rms_w0    [1, 2048]
///   2: q_w       [2048, 2048]   ([K, N] = [hidden, q_dim])
///   3: k_w       [2048, 512]    ([K, N] = [hidden, kv_dim])
///   4: v_w       [2048, 512]    ([K, N] = [hidden, kv_dim])
///   5: cos       [1, 64]
///   6: sin       [1, 64]
///   7: prefix_k  [1, 512]       (read-only prefix K cache, kv_dim wide)
///   8: prefix_v  [1, 512]       (read-only prefix V cache, kv_dim wide)
///   9: o_w       [2048, 2048]   ([K, N] = [q_dim, hidden])
///  10: rms_w1    [1, 2048]
///  11: gate_w    [2048, 8192]   ([K, N] = [hidden, intermediate])
///  12: up_w      [2048, 8192]   ([K, N] = [hidden, intermediate])
///  13: down_w    [8192, 2048]   ([K, N] = [intermediate, hidden])
///
/// Weight orientation is the SubtileIR `MatmulTile` convention `[K, N]`
/// (`eval_node`/`lower_region`: weight rows = K = activation cols,
/// weight cols = N = output cols), NOT the HF `[out, in]` storage order.
/// The square q/o weights are `[K, N] = [N, K] = [2048, 2048]` either
/// way; the non-square MLP weights below must be `[K, N]` or N-tiled
/// `lower_region` slices past the cols extent (caught by
/// `subtile_ir::tests::subtile_tape_nb128_builds_from_fixture`).
pub fn one_layer_input() -> LoweringInput {
    // Llama-3.2-1B decode layer: hidden 2048, kv_dim 512, intermediate 8192,
    // head_dim 64 (GQA 32 q-heads / 8 kv-heads). scale = 1/sqrt(64) = 0.125.
    one_layer_input_shaped(2048, 512, 8192, 64)
}

/// Parameterized single decode layer (the body of [`one_layer_input`]).
/// `h` = hidden, `kv` = kv_dim (= num_kv_heads * head_dim), `i` =
/// intermediate, `hd` = head_dim. Derives `num_q_heads = h/hd`,
/// `num_kv_heads = kv/hd`, and attention `scale = 1/sqrt(head_dim)`. Lets the
/// SAME op chain (rmsnorm -> q/k/v proj -> rope -> append -> attn -> o-proj ->
/// mlp) be exercised at a second head_dim (e.g. hd=128) with no bespoke
/// fixture — the per-layer composition test for model-generality.
pub fn one_layer_input_shaped(h: u32, kv: u32, i: u32, hd: u32) -> LoweringInput {
    let scale = 1.0f32 / (hd as f32).sqrt();
    let head_dim = HeadDim::new(hd);
    // The fixture's geometry, through the same mint the real config path uses: a `(h, kv, hd)` trio
    // whose kv-head count does not divide its query-head count is not a model, and the fixture
    // cannot name one.
    let geom = ModelAttnGeometry::mint(QueryHeads::new(h / hd), KvHeads::new(kv / hd), head_dim)
        .expect("one_layer_input_shaped: h/hd must be a multiple of kv/hd");
    LoweringInput {
        sources: vec![
            SourceShape { rows: 1, cols: h },  // 0  x
            SourceShape { rows: 1, cols: h },  // 1  rms_w0
            SourceShape { rows: h, cols: h },  // 2  q_w   [K=h, N=q_dim=h]
            SourceShape { rows: h, cols: kv }, // 3  k_w   [K=h, N=kv_dim]
            SourceShape { rows: h, cols: kv }, // 4  v_w   [K=h, N=kv_dim]
            SourceShape { rows: 1, cols: hd }, // 5  cos
            SourceShape { rows: 1, cols: hd }, // 6  sin
            // UN-RIGGED KV cache: CAPACITY 8 rows, but only valid_len=2
            // positions are valid (row 0 = prefix, row dp=1 = new RopeAppend
            // write); rows 2..8 are uninitialized and MUST be length-masked
            // out of the decode softmax. Capacity (8) is deliberately NOT
            // co-sized to valid_len (2) — that co-sizing was the rigging that
            // hid the missing AttnDecode length mask (the cache TENSOR row
            // count silently bounded the softmax).
            SourceShape { rows: 8, cols: kv }, // 7  prefix_k cache [cap=8, kv]
            SourceShape { rows: 8, cols: kv }, // 8  prefix_v cache [cap=8, kv]
            SourceShape { rows: h, cols: h },  // 9  o_w   [K=q_dim=h, N=h]
            SourceShape { rows: 1, cols: h },  // 10 rms_w1
            SourceShape { rows: h, cols: i },  // 11 gate_w [K=h, N=i]
            SourceShape { rows: h, cols: i },  // 12 up_w   [K=h, N=i]
            SourceShape { rows: i, cols: h },  // 13 down_w [K=i, N=h]
        ],
        // Production-faithful arity-5 decode layer (matches
        // to_wavefront.rs OpKind::{RopeAppend,Attention}): q/k/v all
        // project from the rmsnorm output; the rope_append FUF op splits
        // into a Q-side RopeRotate + a K-side RopeAppend(k,cos,sin,v,
        // prefix_k,prefix_v); AttnDecode reads [q', prefix_k, prefix_v,
        // new_k, v]. lower_region links RopeAppend's K-cache write
        // (prefix_k tensor) to AttnDecode's KvCacheProducer witness.
        ops: vec![
            // 0: x_norm = rmsnorm(x, rms_w0)
            OpDesc {
                op: LoweredOp::RmsNorm {
                    eps: 1e-5,
                    gain_offset: 0.0,
                },
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
            },
            // 1: q = x_norm @ q_w   [1, q_dim=h]
            OpDesc {
                op: LoweredOp::Gemm {
                    n: h,
                    weight: crate::lower::GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Op(0), InputRef::Ext(2)],
            },
            // 2: k = x_norm @ k_w   [1, kv_dim]
            OpDesc {
                op: LoweredOp::Gemm {
                    n: kv,
                    weight: crate::lower::GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Op(0), InputRef::Ext(3)],
            },
            // 3: v = x_norm @ v_w   [1, kv_dim]
            OpDesc {
                op: LoweredOp::Gemm {
                    n: kv,
                    weight: crate::lower::GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Op(0), InputRef::Ext(4)],
            },
            // 4: q' = rope(q)
            OpDesc {
                op: LoweredOp::RopeRotate { head_dim },
                m: 1,
                inputs: vec![InputRef::Op(1), InputRef::Ext(5), InputRef::Ext(6)],
            },
            // 5: k' = rope_append(k, cos, sin, v, prefix_k, prefix_v)
            //    (rotate K + write rotated K / un-roped V to layer 0 cache)
            OpDesc {
                op: LoweredOp::RopeAppend {
                    head_dim,
                    layer: 0,
                    is_global: true,
                    interleaved: false,
                },
                m: 1,
                inputs: vec![
                    InputRef::Op(2),
                    InputRef::Ext(5),
                    InputRef::Ext(6),
                    InputRef::Op(3),
                    InputRef::Ext(7),
                    InputRef::Ext(8),
                ],
            },
            // 6: attn = AttnDecode(q', prefix_k, prefix_v, new_k, v)
            //    Llama-3.2-1B GQA: 32 q-heads, 8 kv-heads, head_dim=64.
            OpDesc {
                op: LoweredOp::AttnDecode {
                    geom, // Llama-3.2-1B GQA: 32 q-heads, 8 kv-heads
                    scale,
                    // decode_position=1 → 1 prefix row + 1 new = 2 valid
                    // positions. Decoupled from the cache TENSOR capacity
                    // (sources 7,8 may be sized larger).
                    valid_len: 2,
                    sliding: false,
                },
                m: 1,
                inputs: vec![
                    InputRef::Op(4),
                    InputRef::Ext(7),
                    InputRef::Ext(8),
                    InputRef::Op(5),
                    InputRef::Op(3),
                ],
            },
            // 7: o = attn @ o_w
            OpDesc {
                op: LoweredOp::Gemm {
                    n: h,
                    weight: crate::lower::GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Op(6), InputRef::Ext(9)],
            },
            // 8: res1 = x + o
            OpDesc {
                op: LoweredOp::Add,
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Op(7)],
            },
            // 9: x_norm2 = rmsnorm(res1, rms_w1)
            OpDesc {
                op: LoweredOp::RmsNorm {
                    eps: 1e-5,
                    gain_offset: 0.0,
                },
                m: 1,
                inputs: vec![InputRef::Op(8), InputRef::Ext(10)],
            },
            // 10: gate = x_norm2 @ gate_w
            OpDesc {
                op: LoweredOp::Gemm {
                    n: i,
                    weight: crate::lower::GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Op(9), InputRef::Ext(11)],
            },
            // 11: up = x_norm2 @ up_w
            OpDesc {
                op: LoweredOp::Gemm {
                    n: i,
                    weight: crate::lower::GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Op(9), InputRef::Ext(12)],
            },
            // 12: mlp = silu(gate) * up
            OpDesc {
                op: LoweredOp::SiluMul,
                m: 1,
                inputs: vec![InputRef::Op(10), InputRef::Op(11)],
            },
            // 13: down = mlp @ down_w
            OpDesc {
                op: LoweredOp::Gemm {
                    n: h,
                    weight: crate::lower::GemmWeight::Dense,
                },
                m: 1,
                inputs: vec![InputRef::Op(12), InputRef::Ext(13)],
            },
            // 14: res2 = x + down
            OpDesc {
                op: LoweredOp::Add,
                m: 1,
                inputs: vec![InputRef::Ext(0), InputRef::Op(13)],
            },
        ],
        result: 14,
    }
}

/// A minimal NON-ATTENTION chain: `n` layers of `RmsNorm -> Gemm ->
/// Add(residual)` with DISTINCT per-layer weights, parameterized by the row
/// count `m`. Isolates the m-AGNOSTIC re-roll + strided-store path (RmsNorm /
/// Gemm / Add are all row-parallel) from the still-m=1 AttnDecode/RopeAppend
/// arm, so the re-roll numeric gate can be exercised at m>1 (PREFILL) without
/// forking attention. Distinct per-layer sources mean the re-roll's per-layer
/// source/dest table is actually exercised (a baked-layer-0 regression would
/// diverge), mirroring `n_layer_input_distinct`.
///
/// Sources: `0` = x `[m, h]`; then `n` blocks of `[rms_w [1,h], w [h,h]]` at
/// `1 + L*2`. Ops: layer `L`'s 3 ops at `L*3`; `x_in` is the global x for
/// `L=0`, else the previous layer's residual add (`base_op - 1`).
pub fn mlp_chain_distinct_m(n: u32, h: u32, m: u32) -> LoweringInput {
    assert!(n >= 1, "mlp_chain_distinct_m needs at least one layer");
    let mut sources = vec![SourceShape { rows: m, cols: h }]; // 0: x [m, h]
    for _ in 0..n {
        sources.push(SourceShape { rows: 1, cols: h }); // rms_w_L [1, h]
        sources.push(SourceShape { rows: h, cols: h }); // w_L     [K=h, N=h]
    }
    let ops_per_layer = 3usize;
    let mut ops: Vec<OpDesc> = Vec::with_capacity(n as usize * ops_per_layer);
    for layer in 0..n as usize {
        let base_op = layer * ops_per_layer;
        let rms_w = 1 + layer * 2; // source id of this layer's rms gamma
        let w = 2 + layer * 2; // source id of this layer's matmul weight
        // layer 0's x is the global input; L>0 chains the previous residual.
        let x_in = || {
            if layer == 0 {
                InputRef::Ext(0)
            } else {
                InputRef::Op(base_op - 1)
            }
        };
        // norm = rmsnorm(x_in, rms_w_L)
        ops.push(OpDesc {
            op: LoweredOp::RmsNorm {
                eps: 1e-5,
                gain_offset: 0.0,
            },
            m,
            inputs: vec![x_in(), InputRef::Ext(rms_w)],
        });
        // y = norm @ w_L   [m, h]
        ops.push(OpDesc {
            op: LoweredOp::Gemm {
                n: h,
                weight: crate::lower::GemmWeight::Dense,
            },
            m,
            inputs: vec![InputRef::Op(base_op), InputRef::Ext(w)],
        });
        // res = x_in + y
        ops.push(OpDesc {
            op: LoweredOp::Add,
            m,
            inputs: vec![x_in(), InputRef::Op(base_op + 1)],
        });
    }
    LoweringInput {
        sources,
        ops,
        result: n as usize * ops_per_layer - 1,
    }
}

/// `n` chained copies of the [`one_layer_input`] decode-layer body — a
/// host stand-in for the full Llama-3.2-1B stack (16 layers). Each
/// layer's input `x` is the previous layer's residual output (the 15th
/// op of the previous layer); the weight sources are SHARED across
/// layers (this is a STRUCTURAL fixture — it exercises the lowering +
/// §6.5 passes at full multi-layer SCALE: the `PageId` count crosses
/// well past the old u8 limit, page_coalesce / sv_coalesce run over a
/// 16×-larger tape — not numerics, so shared weights are fine).
///
/// Op `L*15 + j` is op `j` of layer `L`. Within a layer, `InputRef::Op`
/// is offset by `L*15`; the layer-input `Ext(0)` (the residual + rmsnorm
/// `x`) is remapped to the previous layer's last op for `L > 0`.
pub fn n_layer_input(n: u32) -> LoweringInput {
    assert!(n >= 1, "n_layer_input needs at least one layer");
    let base = one_layer_input();
    let body = base.ops.clone();
    let ops_per_layer = body.len(); // 15
    let mut ops: Vec<OpDesc> = Vec::with_capacity(n as usize * ops_per_layer);
    for layer in 0..n as usize {
        let base_op = layer * ops_per_layer;
        for op in body.iter() {
            let inputs = op
                .inputs
                .iter()
                .map(|inp| match inp {
                    InputRef::Op(j) => InputRef::Op(base_op + *j),
                    // Ext(0) is the layer-input x (rmsnorm + both
                    // residual adds). Layer 0 uses the global source;
                    // layer L>0 uses the previous layer's residual output
                    // (its last op = base_op - 1).
                    InputRef::Ext(0) if layer > 0 => InputRef::Op(base_op - 1),
                    InputRef::Ext(e) => InputRef::Ext(*e),
                })
                .collect();
            ops.push(OpDesc {
                op: op.op,
                m: op.m,
                inputs,
            });
        }
    }
    LoweringInput {
        sources: base.sources,
        ops,
        result: n as usize * ops_per_layer - 1,
    }
}

/// Like [`n_layer_input`] but every layer gets DISTINCT weight, KV-cache,
/// and rmsnorm-gamma sources — the production-faithful per-layer structure
/// (the real llama-3.2-1b stack has one weight set + one KV cache + two
/// rmsnorm gammas per layer, NOT shared). The rotary cos/sin tables ARE
/// shared (layer-invariant in the real model). This is the fixture that
/// exercises the re-roll's per-layer source/dest TABLE machinery: with
/// shared sources `n_layer_input` has no per-layer variation, so the
/// re-roll is trivially numerics-preserving and cannot catch a
/// baked-layer-0 regression. Here each layer's gather yields `[T^L_0,
/// T^L_1, ..]` distinct ids, so the gate fires unless EVERY per-layer load
/// AND store routes through the layer table.
///
/// Source layout: source 0 = global x; sources 1,2 = SHARED cos, sin; then
/// `n` blocks of 11 per-layer sources (the one_layer_input sources EXCEPT
/// x/cos/sin). Layer L's block starts at `3 + L*11`. The KV cache sources
/// (prefix_k, prefix_v) and both gammas are therefore per-layer; only the
/// op chain wiring differs from `n_layer_input` — by varying source ids.
pub fn n_layer_input_distinct(n: u32) -> LoweringInput {
    assert!(n >= 1, "n_layer_input_distinct needs at least one layer");
    let base = one_layer_input();
    let body = base.ops.clone();
    let ops_per_layer = body.len(); // 15
    // one_layer_input source ids: 0=x, 1=rms_w0, 2=q_w, 3=k_w, 4=v_w,
    // 5=cos, 6=sin, 7=prefix_k, 8=prefix_v, 9=o_w, 10=rms_w1, 11=gate_w,
    // 12=up_w, 13=down_w. SHARED across layers: 0 (x — global input), 5,6
    // (cos/sin — rotary tables, layer-invariant). PER-LAYER: the other 11.
    let per_layer_src_ids: Vec<usize> = (1..=13).filter(|e| *e != 5 && *e != 6).collect(); // 11 ids
    // New source list: 0=x, 1=cos, 2=sin, then n*11 per-layer.
    let mut sources = vec![base.sources[0], base.sources[5], base.sources[6]];
    for _ in 0..n {
        for &e in &per_layer_src_ids {
            sources.push(base.sources[e]);
        }
    }
    let n_per = per_layer_src_ids.len(); // 11
    // Map an original one_layer_input Ext id -> new source id for `layer`.
    let map_ext = |e: usize, layer: usize| -> usize {
        match e {
            5 => 1, // cos (shared)
            6 => 2, // sin (shared)
            _ => {
                let slot = per_layer_src_ids
                    .iter()
                    .position(|&x| x == e)
                    .expect("per-layer src");
                3 + layer * n_per + slot
            }
        }
    };
    let mut ops: Vec<OpDesc> = Vec::with_capacity(n as usize * ops_per_layer);
    for layer in 0..n as usize {
        let base_op = layer * ops_per_layer;
        for op in body.iter() {
            let inputs = op
                .inputs
                .iter()
                .map(|inp| match inp {
                    InputRef::Op(j) => InputRef::Op(base_op + *j),
                    // Ext(0) = the layer-input x: global source for layer 0,
                    // the previous layer's residual output for L>0.
                    InputRef::Ext(0) if layer > 0 => InputRef::Op(base_op - 1),
                    InputRef::Ext(0) => InputRef::Ext(0),
                    InputRef::Ext(e) => InputRef::Ext(map_ext(*e, layer)),
                })
                .collect();
            ops.push(OpDesc {
                op: op.op,
                m: op.m,
                inputs,
            });
        }
    }
    LoweringInput {
        sources,
        ops,
        result: n as usize * ops_per_layer - 1,
    }
}

/// Single-op RmsNorm reproducer for the orchestrator deadlock audit.
///
/// Layout:
///   Sources: 0 = x [1, hidden], 1 = rms_w [1, hidden]
///   Ops:     0 = RmsNorm(x, rms_w)
///   Result:  op 0
///
/// Buffer count: 2 sources + 1 op output = 3.
///
/// Used by `bin/tk_emit_rmsnorm` and the
/// `launcher_runs_on_zeros_rmsnorm_only` smoke test in
/// [`crate::launcher`]. The fixture isolates the round protocol of one
/// op so a deadlock can be pinned to RmsNorm's specific
/// loader/consumer/storer handshake — versus the 12-op forward, where a
/// bug anywhere in the orchestrator hangs the whole tape.
pub fn rmsnorm_only_input() -> LoweringInput {
    let h = 2048u32;
    LoweringInput {
        sources: vec![
            SourceShape { rows: 1, cols: h }, // 0  x
            SourceShape { rows: 1, cols: h }, // 1  rms_w
        ],
        ops: vec![OpDesc {
            op: LoweredOp::RmsNorm {
                eps: 1e-5,
                gain_offset: 0.0,
            },
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    }
}

/// Phase 6 single-op fixture: `out = a + b` element-wise. Buffer
/// count: 2 sources + 1 op output = 3. Used by `bin/tk_emit_add` and
/// the `add_kernel_matches_cpu_golden` test.
pub fn add_only_input() -> LoweringInput {
    let h = 2048u32;
    LoweringInput {
        sources: vec![
            SourceShape { rows: 1, cols: h }, // 0  a
            SourceShape { rows: 1, cols: h }, // 1  b
        ],
        ops: vec![OpDesc {
            op: LoweredOp::Add,
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    }
}

/// Phase 6 single-op fixture: `out = silu(gate) * up`. Buffer count:
/// 2 sources + 1 op output = 3. Used by `bin/tk_emit_silu_mul` and
/// the `silu_mul_kernel_matches_cpu_golden` test. Llama-1B
/// intermediate dim = 8192.
pub fn silu_mul_only_input() -> LoweringInput {
    let intermediate = 8192u32;
    LoweringInput {
        sources: vec![
            SourceShape {
                rows: 1,
                cols: intermediate,
            }, // 0  gate
            SourceShape {
                rows: 1,
                cols: intermediate,
            }, // 1  up
        ],
        ops: vec![OpDesc {
            op: LoweredOp::SiluMul,
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    }
}

/// Phase 6 single-op fixture: NeoX RoPE rotate on a single Q row of
/// `[1, num_heads * head_dim]`. Llama-1B Q-side: 32 heads × 64 head_dim
/// = 2048 cols. Buffer count: 3 sources (x, cos, sin) + 1 op output = 4.
pub fn rope_rotate_only_input() -> LoweringInput {
    let head_dim = 64u32;
    let num_heads = 32u32;
    let cols = num_heads * head_dim;
    LoweringInput {
        sources: vec![
            SourceShape { rows: 1, cols }, // 0  x
            SourceShape {
                rows: 1,
                cols: head_dim,
            }, // 1  cos
            SourceShape {
                rows: 1,
                cols: head_dim,
            }, // 2  sin
        ],
        ops: vec![OpDesc {
            op: LoweredOp::RopeRotate {
                head_dim: HeadDim::new(head_dim),
            },
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1), InputRef::Ext(2)],
        }],
        result: 0,
    }
}

/// Phase 6 single-op fixture: M=1 GEMM `y = x @ w^T`, x is `[1, k]`,
/// w is `[n, k]`, y is `[1, n]`. Llama-1B q_proj-shape: k=2048,
/// n=2048 (32 q-heads × 64 head_dim). Buffer count: 2 sources + 1 op
/// output = 3.
pub fn gemm_m1_only_input() -> LoweringInput {
    let k = 2048u32;
    let n = 2048u32;
    LoweringInput {
        sources: vec![
            SourceShape { rows: 1, cols: k }, // 0  x
            SourceShape { rows: n, cols: k }, // 1  w
        ],
        ops: vec![OpDesc {
            op: LoweredOp::Gemm {
                n,
                weight: crate::lower::GemmWeight::Dense,
            },
            m: 1,
            inputs: vec![InputRef::Ext(0), InputRef::Ext(1)],
        }],
        result: 0,
    }
}

/// Per-buffer byte sizes (bf16 = 2 bytes / element) for a
/// [`LoweringInput`], in `BufId` order: sources first, then per-op
/// output staging buffers. Mirrors the shape inference in
/// `tk_orchestrate::lower_to_tk` so the smoke-test harness can
/// allocate the exact set of device buffers the orchestrator's
/// emitted kernel expects.
pub fn buf_byte_sizes(input: &LoweringInput) -> Vec<usize> {
    let n_sources = input.sources.len();
    let mut out = Vec::with_capacity(n_sources + input.ops.len());

    for s in &input.sources {
        out.push((s.rows as usize) * (s.cols as usize) * 2);
    }

    let shape_for = |r: InputRef,
                     op_shapes: &[(u32, u32)],
                     srcs: &[crate::subtile_ir::SourceShape]|
     -> (u32, u32) {
        match r {
            InputRef::Ext(e) => (srcs[e].rows, srcs[e].cols),
            InputRef::Op(j) => op_shapes[j],
        }
    };

    let mut op_shapes: Vec<(u32, u32)> = Vec::with_capacity(input.ops.len());
    for desc in &input.ops {
        let m = desc.m;
        // Output width comes from THE op registry (`crate::ops`), not
        // from a copy of its rules kept here: this match used to
        // enumerate every op a second time, and the two sites could
        // disagree silently.
        let cols = crate::ops::lowered_op_out_cols(&desc.op, |k| {
            shape_for(desc.inputs[k], &op_shapes, &input.sources).1
        });
        op_shapes.push((m, cols));
        out.push((m as usize) * (cols as usize) * 2);
    }

    out
}

// `orchestrator_kernel_args` was removed alongside the OLD
// substrate; the new tape-build path constructs kernel args directly
// via `tk_tape::KernelArg` / `KernelArgRef` / `KernelArgTy` /
// `U32Source`. The walker rewrite owns this surface now.
