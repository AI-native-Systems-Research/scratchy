// SPDX-License-Identifier: Apache-2.0
//! Piecewise CUDA graph support for dynamic attention optimization.
//!
//! This module implements Python vLLM's `FULL_AND_PIECEWISE` CUDA graph architecture,
//! where attention operations are excluded from graphs and called dynamically between
//! graph pieces. This eliminates baked-in transpose/untranspose overhead and enables
//! dynamic split-K optimization based on actual sequence lengths.

use std::collections::HashMap;

use anyhow::Result;
use cudarc::driver::sys::CUgraphExec;

use crate::alloc::RawGpuMem;
use crate::device::GpuDevice;
use crate::driver;
use crate::dtype::DType;
use crate::tensor::GpuTensor;

/// Maximum number of blocks per sequence in the block table.
const MAX_BLOCKS_PER_SEQ: usize = 2048;

/// Type of graph piece in the piecewise execution flow.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GraphPieceType {
    /// Embedding layer (input_ids -> hidden_states).
    Embedding,
    /// Pre-attention processing for a layer (RMSNorm before attention).
    LayerPreAttn(usize),
    /// Post-attention processing for a layer (MLP + residual after attention).
    LayerPostAttn(usize),
    /// LM head (final projection to vocabulary).
    LmHead,
    /// Sampling (argmax or other sampling strategy).
    Sampling,
}

impl std::fmt::Display for GraphPieceType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphPieceType::Embedding => write!(f, "Embedding"),
            GraphPieceType::LayerPreAttn(layer) => write!(f, "LayerPreAttn({})", layer),
            GraphPieceType::LayerPostAttn(layer) => write!(f, "LayerPostAttn({})", layer),
            GraphPieceType::LmHead => write!(f, "LmHead"),
            GraphPieceType::Sampling => write!(f, "Sampling"),
        }
    }
}

/// A single captured CUDA graph piece.
pub struct GraphPiece {
    /// Instantiated CUDA graph executable.
    pub exec: CUgraphExec,
    /// Type of this piece.
    pub piece_type: GraphPieceType,
    /// Batch size this piece was captured for.
    pub batch_size: usize,
}

/// Persistent buffers for piecewise graph execution.
///
/// These buffers are allocated once at max capacity and reused across all graph pieces.
/// Pointers are baked into captured graphs, so they must remain stable.
/// All allocations use `RawGpuMem` for RAII — memory is freed on drop automatically.
pub struct PersistentBuffers {
    // ---- Metadata (shared across all pieces) ----
    /// Input token IDs: [max_batch] in U32.
    pub input_ids: RawGpuMem,
    /// Token positions: [max_batch] in U32.
    pub positions: RawGpuMem,
    /// Slot mapping for KV cache: [max_batch] in I64.
    pub slot_mapping: RawGpuMem,
    /// Cumulative sequence lengths for queries: [max_batch + 1] in I32.
    pub cu_seqlens_q: RawGpuMem,
    /// Sequence lengths used in KV cache: [max_batch] in I32.
    pub seqused_k: RawGpuMem,
    /// Block table for paged attention: [max_batch, MAX_BLOCKS_PER_SEQ] in I32.
    pub block_table: RawGpuMem,

    // ---- Hidden states (ping-pong between pieces) ----
    /// Hidden state buffer A: [max_batch, hidden_size] in model dtype.
    pub hidden_a: RawGpuMem,
    /// Hidden state buffer B: [max_batch, hidden_size] in model dtype.
    pub hidden_b: RawGpuMem,

    // ---- Residual stream (persists across layers) ----
    /// Residual stream: [max_batch, hidden_size] in model dtype.
    /// Updated in-place by fused_add_rms_norm. For layer 0, initialized
    /// from hidden_a (the embedding output).
    pub residual: RawGpuMem,

    // ---- Attention I/O (NOT in graphs, used for dynamic calls) ----
    /// Attention input buffer: [max_batch, hidden_size] in model dtype.
    pub attn_input: RawGpuMem,
    /// Attention output buffer: [max_batch, hidden_size] in model dtype.
    pub attn_output: RawGpuMem,

    // ---- Output ----
    /// Logits: [max_batch, vocab_size] in model dtype.
    pub logits: RawGpuMem,
    /// Sampled token IDs: [max_batch] in U32.
    pub token_ids: RawGpuMem,

    // ---- Metadata ----
    pub max_batch: usize,
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub dtype: DType,
    /// Number of transformer layers (used by LM head to determine final buffer).
    pub num_layers: usize,
    /// Maximum number of KV cache blocks per sequence in the block table.
    pub max_blocks_per_seq: usize,
}

unsafe impl Send for PersistentBuffers {}

impl PersistentBuffers {
    /// Allocate persistent buffers for piecewise graph execution.
    pub unsafe fn new(
        max_batch: usize,
        hidden_size: usize,
        vocab_size: usize,
        dtype: DType,
        num_layers: usize,
    ) -> Result<Self> {
        let hidden_bytes = max_batch * hidden_size * dtype.size_bytes();
        let logits_bytes = max_batch * vocab_size * dtype.size_bytes();

        Ok(Self {
            input_ids: crate::raw_cuda(driver::mem_alloc(max_batch * 4)?, max_batch * 4),
            positions: crate::raw_cuda(driver::mem_alloc(max_batch * 4)?, max_batch * 4),
            slot_mapping: crate::raw_cuda(driver::mem_alloc(max_batch * 8)?, max_batch * 8),
            cu_seqlens_q: crate::raw_cuda(
                driver::mem_alloc((max_batch + 1) * 4)?,
                (max_batch + 1) * 4,
            ),
            seqused_k: crate::raw_cuda(driver::mem_alloc(max_batch * 4)?, max_batch * 4),
            block_table: crate::raw_cuda(
                driver::mem_alloc(max_batch * MAX_BLOCKS_PER_SEQ * 4)?,
                max_batch * MAX_BLOCKS_PER_SEQ * 4,
            ),
            hidden_a: crate::raw_cuda(driver::mem_alloc(hidden_bytes)?, hidden_bytes),
            hidden_b: crate::raw_cuda(driver::mem_alloc(hidden_bytes)?, hidden_bytes),
            residual: crate::raw_cuda(driver::mem_alloc(hidden_bytes)?, hidden_bytes),
            attn_input: crate::raw_cuda(driver::mem_alloc(hidden_bytes)?, hidden_bytes),
            attn_output: crate::raw_cuda(driver::mem_alloc(hidden_bytes)?, hidden_bytes),
            logits: crate::raw_cuda(driver::mem_alloc(logits_bytes)?, logits_bytes),
            token_ids: crate::raw_cuda(driver::mem_alloc(max_batch * 4)?, max_batch * 4),
            max_batch,
            hidden_size,
            vocab_size,
            dtype,
            num_layers,
            max_blocks_per_seq: MAX_BLOCKS_PER_SEQ,
        })
    }

    /// Get input tensor views for graph capture/replay.
    pub unsafe fn input_tensors(&self, batch_size: usize) -> InputTensors {
        InputTensors {
            input_ids: GpuTensor::new(self.input_ids.ptr(), &[batch_size], DType::U32),
            positions: GpuTensor::new(self.positions.ptr(), &[batch_size], DType::U32),
            slot_mapping: GpuTensor::new(self.slot_mapping.ptr(), &[batch_size], DType::I64),
            cu_seqlens_q: GpuTensor::new(self.cu_seqlens_q.ptr(), &[batch_size + 1], DType::I32),
            seqused_k: GpuTensor::new(self.seqused_k.ptr(), &[batch_size], DType::I32),
            block_table: GpuTensor::new(
                self.block_table.ptr(),
                &[batch_size, MAX_BLOCKS_PER_SEQ],
                DType::I32,
            ),
        }
    }

    /// Get hidden state tensor for a given buffer (A or B).
    pub unsafe fn hidden_tensor(&self, batch_size: usize, use_buffer_a: bool) -> GpuTensor {
        let ptr = if use_buffer_a {
            self.hidden_a.ptr()
        } else {
            self.hidden_b.ptr()
        };
        GpuTensor::new(ptr, &[batch_size, self.hidden_size], self.dtype)
    }

    /// Get residual stream tensor.
    pub unsafe fn residual_tensor(&self, batch_size: usize) -> GpuTensor {
        GpuTensor::new(
            self.residual.ptr(),
            &[batch_size, self.hidden_size],
            self.dtype,
        )
    }

    /// Get attention input tensor.
    pub unsafe fn attn_input_tensor(&self, batch_size: usize) -> GpuTensor {
        GpuTensor::new(
            self.attn_input.ptr(),
            &[batch_size, self.hidden_size],
            self.dtype,
        )
    }

    /// Get attention output tensor.
    pub unsafe fn attn_output_tensor(&self, batch_size: usize) -> GpuTensor {
        GpuTensor::new(
            self.attn_output.ptr(),
            &[batch_size, self.hidden_size],
            self.dtype,
        )
    }

    /// Get logits tensor.
    pub unsafe fn logits_tensor(&self, batch_size: usize) -> GpuTensor {
        GpuTensor::new(
            self.logits.ptr(),
            &[batch_size, self.vocab_size],
            self.dtype,
        )
    }

    /// Get token IDs tensor.
    pub unsafe fn token_ids_tensor(&self, batch_size: usize) -> GpuTensor {
        GpuTensor::new(self.token_ids.ptr(), &[batch_size], DType::U32)
    }

    /// Get positions tensor (U32).
    pub unsafe fn positions_tensor(&self, batch_size: usize) -> GpuTensor {
        GpuTensor::new(self.positions.ptr(), &[batch_size], DType::U32)
    }

    /// Get slot_mapping tensor (I64).
    pub unsafe fn slot_mapping_tensor(&self, batch_size: usize) -> GpuTensor {
        GpuTensor::new(self.slot_mapping.ptr(), &[batch_size], DType::I64)
    }

    /// Get cu_seqlens_q tensor (I32, length batch_size+1).
    pub unsafe fn cu_seqlens_q_tensor(&self, batch_size: usize) -> GpuTensor {
        GpuTensor::new(self.cu_seqlens_q.ptr(), &[batch_size + 1], DType::I32)
    }

    /// Get seqused_k tensor (I32).
    pub unsafe fn seqused_k_tensor(&self, batch_size: usize) -> GpuTensor {
        GpuTensor::new(self.seqused_k.ptr(), &[batch_size], DType::I32)
    }

    /// Get block_table tensor (I32, shape [batch_size, max_blocks_per_seq]).
    pub unsafe fn block_table_tensor(&self, batch_size: usize) -> GpuTensor {
        GpuTensor::new(
            self.block_table.ptr(),
            &[batch_size, self.max_blocks_per_seq],
            DType::I32,
        )
    }
}

// No manual Drop needed — RawGpuMem handles cleanup via RAII.

/// Input tensor views into persistent buffers.
#[derive(Clone, Copy)]
pub struct InputTensors {
    pub input_ids: GpuTensor,
    pub positions: GpuTensor,
    pub slot_mapping: GpuTensor,
    pub cu_seqlens_q: GpuTensor,
    pub seqused_k: GpuTensor,
    pub block_table: GpuTensor,
}

/// Piecewise CUDA graph runner.
///
/// Captures and replays graph pieces with attention excluded, enabling dynamic
/// split-K optimization and eliminating baked-in transpose overhead.
pub struct PiecewiseGraphRunner {
    /// Captured graph pieces, indexed by batch_size -> list of pieces.
    pub graphs: HashMap<usize, Vec<GraphPiece>>,
    /// Persistent buffers for all graph pieces.
    pub buffers: PersistentBuffers,
    /// Number of transformer layers.
    pub num_layers: usize,
}

unsafe impl Send for PiecewiseGraphRunner {}

impl PiecewiseGraphRunner {
    /// Create a new piecewise graph runner.
    pub unsafe fn new(
        max_batch: usize,
        hidden_size: usize,
        vocab_size: usize,
        num_layers: usize,
        dtype: DType,
    ) -> Result<Self> {
        let buffers =
            PersistentBuffers::new(max_batch, hidden_size, vocab_size, dtype, num_layers)?;
        Ok(Self {
            graphs: HashMap::new(),
            buffers,
            num_layers,
        })
    }

    /// Check if graphs are captured for a given batch size.
    pub fn has_graph(&self, batch_size: usize) -> bool {
        self.graphs.contains_key(&batch_size)
    }

    /// Get the nearest captured graph size >= batch_size.
    pub fn nearest_graph_size(&self, batch_size: usize) -> Option<usize> {
        self.graphs
            .keys()
            .filter(|&&s| s >= batch_size)
            .min()
            .copied()
    }

    /// Get all captured batch sizes.
    pub fn captured_sizes(&self) -> Vec<usize> {
        let mut sizes: Vec<usize> = self.graphs.keys().copied().collect();
        sizes.sort();
        sizes
    }

    /// Replay a specific graph piece.
    pub unsafe fn replay_piece(
        &self,
        batch_size: usize,
        piece_type: GraphPieceType,
        device: &mut GpuDevice,
    ) -> Result<()> {
        let pieces = self
            .graphs
            .get(&batch_size)
            .ok_or_else(|| anyhow::anyhow!("no captured graphs for batch_size={}", batch_size))?;

        let piece = pieces
            .iter()
            .find(|p| p.piece_type == piece_type)
            .ok_or_else(|| anyhow::anyhow!("piece not found: {:?}", piece_type))?;

        driver::graph_launch(piece.exec, device.compute_stream)?;
        Ok(())
    }
}

impl PiecewiseGraphRunner {
    /// Capture all graph pieces for a given batch size.
    ///
    /// The caller must have already called `device.caching.begin_allocate_to_pool()`
    /// before calling this method, and should call `end_allocate_to_pool()` after all
    /// batch sizes have been captured. This matches the monolithic graph runner pattern.
    ///
    /// The forward_fn callback is called for each piece type during warmup and capture.
    /// It should execute the corresponding model operations using the persistent buffers.
    pub unsafe fn capture_all_pieces<F>(
        &mut self,
        batch_size: usize,
        device: &mut GpuDevice,
        mut forward_fn: F,
    ) -> Result<()>
    where
        F: FnMut(GraphPieceType, &PersistentBuffers, &mut GpuDevice) -> Result<()>,
    {
        let mut pieces = Vec::new();

        // 1. Capture embedding
        pieces.push(self.capture_piece(
            batch_size,
            GraphPieceType::Embedding,
            device,
            &mut forward_fn,
        )?);

        // 2. Capture each layer: pre-attn + post-attn
        for layer_idx in 0..self.num_layers {
            pieces.push(self.capture_piece(
                batch_size,
                GraphPieceType::LayerPreAttn(layer_idx),
                device,
                &mut forward_fn,
            )?);
            // Attention NOT captured - called dynamically between pieces
            pieces.push(self.capture_piece(
                batch_size,
                GraphPieceType::LayerPostAttn(layer_idx),
                device,
                &mut forward_fn,
            )?);
        }

        // 3. Capture LM head + sampling
        pieces.push(self.capture_piece(
            batch_size,
            GraphPieceType::LmHead,
            device,
            &mut forward_fn,
        )?);
        pieces.push(self.capture_piece(
            batch_size,
            GraphPieceType::Sampling,
            device,
            &mut forward_fn,
        )?);

        let num_pieces = pieces.len();
        self.graphs.insert(batch_size, pieces);

        tracing::info!(
            "Piecewise CUDA graphs captured for batch_size={} ({} pieces, {} layers)",
            batch_size,
            num_pieces,
            self.num_layers
        );

        Ok(())
    }

    /// Capture a single graph piece.
    ///
    /// The caching allocator's private pool (begin/end_allocate_to_pool) handles
    /// memory lifecycle — blocks allocated during capture stay permanently allocated,
    /// matching PyTorch's private graph pool behavior.
    unsafe fn capture_piece<F>(
        &self,
        batch_size: usize,
        piece_type: GraphPieceType,
        device: &mut GpuDevice,
        forward_fn: &mut F,
    ) -> Result<GraphPiece>
    where
        F: FnMut(GraphPieceType, &PersistentBuffers, &mut GpuDevice) -> Result<()>,
    {
        // Warmup run to populate cuBLAS plans and fill the pool.
        forward_fn(piece_type, &self.buffers, device)?;
        driver::stream_synchronize(device.compute_stream)?;

        // Capture the graph
        driver::stream_begin_capture(device.compute_stream)?;
        forward_fn(piece_type, &self.buffers, device)?;
        let graph = driver::stream_end_capture(device.compute_stream)?;

        let exec = driver::graph_instantiate(graph)?;
        driver::graph_destroy(graph)?;

        tracing::debug!(
            "Captured graph piece: {:?} for batch_size={}",
            piece_type,
            batch_size
        );

        Ok(GraphPiece {
            exec,
            piece_type,
            batch_size,
        })
    }

    /// Fill persistent buffers with dummy data for graph capture.
    pub unsafe fn fill_dummy_decode(
        &self,
        batch_size: usize,
        device: &mut GpuDevice,
    ) -> Result<()> {
        let stream = device.compute_stream;

        // Fill input_ids with zeros
        driver::memset_d8(self.buffers.input_ids.ptr(), 0, batch_size * 4, stream)?;

        // Fill positions with sequential values
        let positions: Vec<u32> = (0..batch_size as u32).collect();
        driver::memcpy_htod_async(
            self.buffers.positions.ptr(),
            positions.as_ptr() as *const u8,
            batch_size * 4,
            stream,
        )?;

        // Fill slot_mapping with sequential values
        let slots: Vec<i64> = (0..batch_size as i64).collect();
        driver::memcpy_htod_async(
            self.buffers.slot_mapping.ptr(),
            slots.as_ptr() as *const u8,
            batch_size * 8,
            stream,
        )?;

        // Fill cu_seqlens_q with cumulative lengths
        let cu_q: Vec<i32> = (0..=batch_size as i32).collect();
        driver::memcpy_htod_async(
            self.buffers.cu_seqlens_q.ptr(),
            cu_q.as_ptr() as *const u8,
            (batch_size + 1) * 4,
            stream,
        )?;

        // Fill seqused_k with ones (each sequence has length 1)
        let seqused_k: Vec<i32> = vec![1; batch_size];
        driver::memcpy_htod_async(
            self.buffers.seqused_k.ptr(),
            seqused_k.as_ptr() as *const u8,
            batch_size * 4,
            stream,
        )?;

        // Fill block_table with zeros
        driver::memset_d8(
            self.buffers.block_table.ptr(),
            0,
            batch_size * MAX_BLOCKS_PER_SEQ * 4,
            stream,
        )?;

        // Fill residual buffer with zeros
        let hidden_bytes = batch_size * self.buffers.hidden_size * self.buffers.dtype.size_bytes();
        driver::memset_d8(self.buffers.residual.ptr(), 0, hidden_bytes, stream)?;

        Ok(())
    }
}

impl Drop for PiecewiseGraphRunner {
    fn drop(&mut self) {
        unsafe {
            for (_, pieces) in self.graphs.drain() {
                for piece in pieces {
                    let _ = driver::graph_exec_destroy(piece.exec);
                }
            }
        }
        // PersistentBuffers dropped automatically via RawGpuMem RAII.
    }
}

/// Output from piecewise graph replay.
pub struct PiecewiseReplayOutput {
    pub logits: GpuTensor,
    pub token_ids: GpuTensor,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_piece_type_display() {
        assert_eq!(format!("{}", GraphPieceType::Embedding), "Embedding");
        assert_eq!(
            format!("{}", GraphPieceType::LayerPreAttn(0)),
            "LayerPreAttn(0)"
        );
        assert_eq!(
            format!("{}", GraphPieceType::LayerPostAttn(5)),
            "LayerPostAttn(5)"
        );
        assert_eq!(format!("{}", GraphPieceType::LmHead), "LmHead");
        assert_eq!(format!("{}", GraphPieceType::Sampling), "Sampling");
    }

    #[test]
    fn test_graph_piece_type_ordering() {
        let pre_0 = GraphPieceType::LayerPreAttn(0);
        let pre_1 = GraphPieceType::LayerPreAttn(1);
        let post_0 = GraphPieceType::LayerPostAttn(0);
        let post_1 = GraphPieceType::LayerPostAttn(1);

        assert!(matches!(pre_0, GraphPieceType::LayerPreAttn(0)));
        assert!(matches!(pre_1, GraphPieceType::LayerPreAttn(1)));
        assert!(matches!(post_0, GraphPieceType::LayerPostAttn(0)));
        assert!(matches!(post_1, GraphPieceType::LayerPostAttn(1)));
    }

    #[test]
    fn test_graph_piece_type_variants() {
        let embedding = GraphPieceType::Embedding;
        let pre_attn = GraphPieceType::LayerPreAttn(0);
        let post_attn = GraphPieceType::LayerPostAttn(0);
        let lm_head = GraphPieceType::LmHead;
        let sampling = GraphPieceType::Sampling;

        assert!(matches!(embedding, GraphPieceType::Embedding));
        assert!(matches!(pre_attn, GraphPieceType::LayerPreAttn(_)));
        assert!(matches!(post_attn, GraphPieceType::LayerPostAttn(_)));
        assert!(matches!(lm_head, GraphPieceType::LmHead));
        assert!(matches!(sampling, GraphPieceType::Sampling));
    }

    #[test]
    fn test_persistent_buffers_structure() {
        let _check_fields = |buffers: &PersistentBuffers| {
            let _ = buffers.input_ids.ptr();
            let _ = buffers.positions.ptr();
            let _ = buffers.hidden_a.ptr();
            let _ = buffers.hidden_b.ptr();
            let _ = buffers.residual.ptr();
            let _ = buffers.attn_input.ptr();
            let _ = buffers.attn_output.ptr();
            let _ = buffers.logits.ptr();
            let _ = buffers.token_ids.ptr();
        };
    }

    #[test]
    fn test_piecewise_graph_runner_structure() {
        let _check_fields = |runner: &PiecewiseGraphRunner| {
            let _ = &runner.buffers;
            let _ = &runner.graphs;
            let _ = &runner.num_layers;
        };
    }

    #[test]
    fn test_graph_piece_structure() {
        let _check_fields = |piece: &GraphPiece| {
            let _ = &piece.exec;
            let _ = &piece.piece_type;
            let _ = &piece.batch_size;
        };
    }

    #[test]
    fn test_piecewise_replay_output_structure() {
        let _check_fields = |output: &PiecewiseReplayOutput| {
            let _ = &output.logits;
            let _ = &output.token_ids;
        };
    }

    #[test]
    fn test_graph_piece_type_layer_indices() {
        for i in 0..32 {
            let pre = GraphPieceType::LayerPreAttn(i);
            let post = GraphPieceType::LayerPostAttn(i);

            if let GraphPieceType::LayerPreAttn(idx) = pre {
                assert_eq!(idx, i);
            } else {
                panic!("Expected LayerPreAttn variant");
            }

            if let GraphPieceType::LayerPostAttn(idx) = post {
                assert_eq!(idx, i);
            } else {
                panic!("Expected LayerPostAttn variant");
            }
        }
    }

    #[test]
    fn test_graph_piece_type_clone() {
        let original = GraphPieceType::LayerPreAttn(5);
        let cloned = original;

        assert!(matches!(cloned, GraphPieceType::LayerPreAttn(5)));
    }

    #[test]
    fn test_graph_piece_type_debug() {
        let piece = GraphPieceType::LayerPreAttn(3);
        let debug_str = format!("{:?}", piece);
        assert!(debug_str.contains("LayerPreAttn"));
        assert!(debug_str.contains("3"));
    }
}
