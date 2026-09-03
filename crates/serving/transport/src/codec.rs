// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Msgpack encoder/decoder for vLLM engine-core IPC.
//!
//! Wraps `rmp-serde` to provide a convenient interface for serializing and
//! deserializing engine-core types over ZMQ. Mirrors the Python
//! `MsgpackEncoder`/`MsgpackDecoder` from `vllm/v1/serial_utils.py`.
//!
//! # Wire format
//!
//! For Rust-to-Rust communication, types are serialized as msgpack maps
//! (named fields via serde). For interop with the Python `msgspec` encoder
//! which uses `array_like=True`, a compatibility layer will be added in a
//! future iteration.
//!
//! # Zero-copy
//!
//! The Python encoder supports zero-copy tensor serialization via auxiliary
//! buffers. The Rust encoder does not yet need this since tensor data is
//! handled by the executor/worker layer (which remains in Python during
//! the transition). When the executor is ported (Phase 4+), zero-copy
//! buffer support will be added here.

use bytes::{BufMut, Bytes, BytesMut};
use serde::{Serialize, de::DeserializeOwned};

/// Errors that can occur during encoding or decoding.
#[derive(Debug, thiserror::Error)]
pub enum CodecError {
    /// Msgpack encoding failed.
    #[error("msgpack encode error: {0}")]
    Encode(#[from] rmp_serde::encode::Error),

    /// Msgpack decoding failed.
    #[error("msgpack decode error: {0}")]
    Decode(#[from] rmp_serde::decode::Error),
}

pub type CodecResult<T> = Result<T, CodecError>;

/// Msgpack encoder for vLLM engine-core types.
///
/// Encodes any `Serialize` type into msgpack bytes. Thread-safe (stateless).
///
/// # Example
/// ```
/// use scratchy_serving_transport::codec::MsgpackEncoder;
/// use scratchy_core_common::EngineCoreOutputs;
///
/// let encoder = MsgpackEncoder::new();
/// let outputs = EngineCoreOutputs::default();
/// let bytes = encoder.encode(&outputs).unwrap();
/// assert!(!bytes.is_empty());
/// ```
#[derive(Debug, Clone, Default)]
pub struct MsgpackEncoder;

impl MsgpackEncoder {
    /// Create a new encoder.
    pub fn new() -> Self {
        Self
    }

    /// Encode a value into msgpack bytes.
    pub fn encode<T: Serialize>(&self, value: &T) -> CodecResult<Bytes> {
        let data = rmp_serde::to_vec_named(value)?;
        Ok(Bytes::from(data))
    }

    /// Encode a value into an existing buffer, returning the number of
    /// bytes written.
    pub fn encode_into<T: Serialize>(&self, value: &T, buf: &mut BytesMut) -> CodecResult<usize> {
        let before = buf.len();
        {
            let mut writer = buf.writer();
            rmp_serde::encode::write_named(&mut writer, value)?;
        }
        Ok(buf.len() - before)
    }

    /// Encode a value into a Vec<u8>.
    pub fn encode_to_vec<T: Serialize>(&self, value: &T) -> CodecResult<Vec<u8>> {
        Ok(rmp_serde::to_vec_named(value)?)
    }
}

/// Msgpack decoder for vLLM engine-core types.
///
/// Decodes msgpack bytes into any `DeserializeOwned` type. Thread-safe
/// (stateless).
///
/// # Example
/// ```
/// use scratchy_serving_transport::codec::{MsgpackEncoder, MsgpackDecoder};
/// use scratchy_core_common::EngineCoreOutputs;
///
/// let encoder = MsgpackEncoder::new();
/// let decoder = MsgpackDecoder::new();
///
/// let outputs = EngineCoreOutputs::default();
/// let bytes = encoder.encode(&outputs).unwrap();
/// let decoded: EngineCoreOutputs = decoder.decode(&bytes).unwrap();
/// assert_eq!(decoded.engine_index, 0);
/// ```
#[derive(Debug, Clone, Default)]
pub struct MsgpackDecoder;

impl MsgpackDecoder {
    /// Create a new decoder.
    pub fn new() -> Self {
        Self
    }

    /// Decode msgpack bytes into a value.
    pub fn decode<T: DeserializeOwned>(&self, data: &[u8]) -> CodecResult<T> {
        Ok(rmp_serde::from_slice(data)?)
    }
}

// ---------------------------------------------------------------------------
// Convenience functions
// ---------------------------------------------------------------------------

/// Encode a value into msgpack bytes (convenience wrapper).
pub fn encode<T: Serialize>(value: &T) -> CodecResult<Bytes> {
    MsgpackEncoder::new().encode(value)
}

/// Decode msgpack bytes into a value (convenience wrapper).
pub fn decode<T: DeserializeOwned>(data: &[u8]) -> CodecResult<T> {
    MsgpackDecoder::new().decode(data)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use scratchy_core_common::{
        EngineCoreEvent, EngineCoreEventType, EngineCoreOutput, EngineCoreOutputs,
        EngineCoreRequest, FinishReason, SamplingParams, StopReason,
    };

    #[test]
    fn test_roundtrip_engine_core_request() {
        let req = EngineCoreRequest {
            request_id: "req-42".into(),
            prompt_token_ids: Some(vec![1, 2, 3, 4, 5]),
            sampling_params: Some(SamplingParams {
                temperature: 0.7,
                top_p: 0.9,
                max_tokens: Some(256),
                ..Default::default()
            }),
            arrival_time: 1700000000.0,
            client_index: 0,
            priority: 0,
            cache_salt: None,
            data_parallel_rank: None,
            is_pooling: false,
            mm_data: None,
            block_annotations: None,
            seal: false,
            volatile: false,
        };

        let encoded = encode(&req).unwrap();
        let decoded: EngineCoreRequest = decode(&encoded).unwrap();

        assert_eq!(decoded.request_id, "req-42");
        assert_eq!(decoded.prompt_token_ids, Some(vec![1, 2, 3, 4, 5]));
        assert_eq!(decoded.sampling_params.as_ref().unwrap().temperature, 0.7);
        assert_eq!(
            decoded.sampling_params.as_ref().unwrap().max_tokens,
            Some(256)
        );
    }

    #[test]
    fn test_roundtrip_engine_core_output() {
        let out = EngineCoreOutput {
            request_id: "req-1".into(),
            new_token_ids: vec![100, 200, 300],
            finish_reason: Some(FinishReason::Stop),
            stop_reason: Some(StopReason::Token(50256)),
            num_cached_tokens: 42,
            events: Some(vec![
                EngineCoreEvent::new(EngineCoreEventType::Queued, 1.0),
                EngineCoreEvent::new(EngineCoreEventType::Scheduled, 1.5),
            ]),
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };

        let encoded = encode(&out).unwrap();
        let decoded: EngineCoreOutput = decode(&encoded).unwrap();

        assert_eq!(decoded.request_id, "req-1");
        assert_eq!(decoded.new_token_ids, vec![100, 200, 300]);
        assert_eq!(decoded.finish_reason, Some(FinishReason::Stop));
        assert_eq!(decoded.stop_reason, Some(StopReason::Token(50256)));
        assert_eq!(decoded.num_cached_tokens, 42);
        assert_eq!(decoded.events.as_ref().unwrap().len(), 2);
    }

    #[test]
    fn test_roundtrip_engine_core_outputs() {
        let outputs = EngineCoreOutputs {
            engine_index: 3,
            outputs: vec![
                EngineCoreOutput {
                    request_id: "a".into(),
                    new_token_ids: vec![1],
                    finish_reason: None,
                    stop_reason: None,
                    num_cached_tokens: 0,
                    events: None,
                    new_logprobs: None,
                    new_prompt_logprobs: None,
                    pooler_output: None,
                },
                EngineCoreOutput {
                    request_id: "b".into(),
                    new_token_ids: vec![2, 3],
                    finish_reason: Some(FinishReason::Length),
                    stop_reason: None,
                    num_cached_tokens: 10,
                    events: None,
                    new_logprobs: None,
                    new_prompt_logprobs: None,
                    pooler_output: None,
                },
            ],
            timestamp: 1234.5,
            scheduler_stats: None,
        };

        let encoded = encode(&outputs).unwrap();
        let decoded: EngineCoreOutputs = decode(&encoded).unwrap();

        assert_eq!(decoded.engine_index, 3);
        assert_eq!(decoded.outputs.len(), 2);
        assert_eq!(decoded.outputs[0].request_id, "a");
        assert_eq!(decoded.outputs[1].finish_reason, Some(FinishReason::Length));
        assert_eq!(decoded.timestamp, 1234.5);
    }

    #[test]
    fn test_roundtrip_stop_reason_string() {
        let out = EngineCoreOutput {
            request_id: "r".into(),
            new_token_ids: vec![],
            finish_reason: Some(FinishReason::Stop),
            stop_reason: Some(StopReason::String("</s>".into())),
            num_cached_tokens: 0,
            events: None,
            new_logprobs: None,
            new_prompt_logprobs: None,
            pooler_output: None,
        };

        let encoded = encode(&out).unwrap();
        let decoded: EngineCoreOutput = decode(&encoded).unwrap();
        assert_eq!(decoded.stop_reason, Some(StopReason::String("</s>".into())));
    }

    #[test]
    fn test_encode_into_buffer() {
        let encoder = MsgpackEncoder::new();
        let outputs = EngineCoreOutputs::default();

        let mut buf = BytesMut::new();
        let len = encoder.encode_into(&outputs, &mut buf).unwrap();
        assert_eq!(buf.len(), len);
        assert!(len > 0);

        let decoded: EngineCoreOutputs = MsgpackDecoder::new().decode(&buf).unwrap();
        assert_eq!(decoded.engine_index, 0);
    }

    #[test]
    fn test_empty_request_roundtrip() {
        let req = EngineCoreRequest {
            request_id: "".into(),
            prompt_token_ids: None,
            sampling_params: None,
            arrival_time: 0.0,
            client_index: 0,
            priority: 0,
            cache_salt: None,
            data_parallel_rank: None,
            is_pooling: false,
            mm_data: None,
            block_annotations: None,
            seal: false,
            volatile: false,
        };

        let encoded = encode(&req).unwrap();
        let decoded: EngineCoreRequest = decode(&encoded).unwrap();
        assert_eq!(decoded.request_id, "");
        assert!(decoded.prompt_token_ids.is_none());
        assert!(decoded.sampling_params.is_none());
    }

    #[test]
    fn test_codec_error_on_invalid_data() {
        let result: CodecResult<EngineCoreOutput> = decode(b"not valid msgpack!!!");
        assert!(result.is_err());
    }

    #[test]
    fn test_large_token_ids() {
        let req = EngineCoreRequest {
            request_id: "big".into(),
            prompt_token_ids: Some((0..10_000).collect()),
            sampling_params: None,
            arrival_time: 0.0,
            client_index: 0,
            priority: 0,
            cache_salt: None,
            data_parallel_rank: None,
            is_pooling: false,
            mm_data: None,
            block_annotations: None,
            seal: false,
            volatile: false,
        };

        let encoded = encode(&req).unwrap();
        let decoded: EngineCoreRequest = decode(&encoded).unwrap();
        assert_eq!(decoded.prompt_token_ids.as_ref().unwrap().len(), 10_000);
    }

    #[test]
    fn test_sampling_params_roundtrip() {
        let params = SamplingParams {
            temperature: 0.0, // greedy
            top_p: 1.0,
            top_k: -1,
            min_p: 0.0,
            max_tokens: Some(1024),
            presence_penalty: 0.5,
            frequency_penalty: -0.5,
            repetition_penalty: 1.2,
            ..Default::default()
        };

        let encoded = encode(&params).unwrap();
        let decoded: SamplingParams = decode(&encoded).unwrap();

        assert_eq!(decoded.temperature, 0.0);
        assert_eq!(decoded.max_tokens, Some(1024));
        assert_eq!(decoded.presence_penalty, 0.5);
        assert_eq!(decoded.frequency_penalty, -0.5);
        assert_eq!(decoded.repetition_penalty, 1.2);
    }
}
