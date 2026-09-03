// SPDX-License-Identifier: Apache-2.0
//! Backend-agnostic trait abstractions for weight types.
//!
//! These traits define the interface that both CUDA and Metal weight types
//! must implement. This allows `Instruction<W>` to work with any backend.

use std::fmt::Debug;

/// Trait for embedding layers (token → hidden state lookup).
pub trait EmbeddingLayer: Debug + Send + Sync {
    /// Get the embedding dimension (hidden_size).
    fn embedding_dim(&self) -> usize;
    
    /// Get the vocabulary size.
    fn vocab_size(&self) -> usize;
}

/// Trait for RMS normalization layers.
pub trait RmsNormLayer: Debug + Send + Sync {
    /// Get the normalization dimension.
    fn dim(&self) -> usize;
    
    /// Get the epsilon value for numerical stability.
    fn eps(&self) -> f32;
}

/// Trait for layer normalization with bias.
pub trait LayerNormLayer: Debug + Send + Sync {
    /// Get the normalization dimension.
    fn dim(&self) -> usize;
    
    /// Get the epsilon value for numerical stability.
    fn eps(&self) -> f32;
    
    /// Whether this layer has a bias term.
    fn has_bias(&self) -> bool;
}

/// Trait for linear (dense) layers: y = xW^T + b.
pub trait LinearLayerTrait: Debug + Send + Sync {
    /// Get the input dimension (in_features).
    fn in_features(&self) -> usize;
    
    /// Get the output dimension (out_features).
    fn out_features(&self) -> usize;
    
    /// Whether this layer has a bias term.
    fn has_bias(&self) -> bool;
}

/// Trait for quantized linear layers (Marlin, BNB4, FP8, etc.).
pub trait QuantizedLinearLayer: Debug + Send + Sync {
    /// Get the input dimension (in_features).
    fn in_features(&self) -> usize;
    
    /// Get the output dimension (out_features).
    fn out_features(&self) -> usize;
    
    /// Get the quantization format name (e.g., "marlin_awq", "bnb4", "fp8").
    fn format(&self) -> &str;
}

/// Trait for MoE (Mixture of Experts) layers.
pub trait MoELayerTrait: Debug + Send + Sync {
    /// Get the number of experts.
    fn num_experts(&self) -> usize;
    
    /// Get the top-k value (how many experts to route to).
    fn top_k(&self) -> usize;
    
    /// Get the hidden dimension.
    fn hidden_size(&self) -> usize;
    
    /// Get the intermediate dimension per expert.
    fn intermediate_size(&self) -> usize;
}

// CUDA implementations - forward declare the concrete types
#[cfg(feature = "cuda")]
mod cuda_impls {
    use super::*;
    use scratchy_target_cuda::layers::{Embedding, RmsNorm, LayerNorm, LinearLayer};
    
    impl EmbeddingLayer for Embedding {
        fn embedding_dim(&self) -> usize {
            self.embedding_dim
        }
        
        fn vocab_size(&self) -> usize {
            self.num_embeddings
        }
    }
    
    impl RmsNormLayer for RmsNorm {
        fn dim(&self) -> usize {
            self.normalized_shape
        }
        
        fn eps(&self) -> f32 {
            self.eps
        }
    }
    
    impl LayerNormLayer for LayerNorm {
        fn dim(&self) -> usize {
            self.normalized_shape
        }
        
        fn eps(&self) -> f32 {
            self.eps
        }
        
        fn has_bias(&self) -> bool {
            self.bias.is_some()
        }
    }
    
    impl LinearLayerTrait for LinearLayer {
        fn in_features(&self) -> usize {
            self.in_features
        }
        
        fn out_features(&self) -> usize {
            self.out_features
        }
        
        fn has_bias(&self) -> bool {
            self.bias.is_some()
        }
    }
}

// Metal implementations - will be filled in Phase 5.6.2
#[cfg(feature = "metal")]
mod metal_impls {
    // TODO: Implement trait impls for Metal weight types
    // These will be created in scratchy-target-metal/src/layers.rs
}
