// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Protocol-level message types for engine-core IPC.
//!
//! These types define the framing and request/response envelope used over
//! ZMQ between the API front-end and engine-core process.
//!
//! The ZMQ multipart message format is:
//! ```text
//! Frame 0: EngineCoreRequestType (1 byte)
//! Frame 1+: Msgpack-encoded payload (request data)
//! ```
//!
//! This mirrors the Python `EngineCoreRequestType` enum and the multipart
//! send/recv patterns in `vllm/v1/engine/core.py`.

use std::borrow::Cow;

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// EngineCoreRequestType
// ---------------------------------------------------------------------------

/// Request type tag sent as the first ZMQ frame.
///
/// Encoded as a single byte matching the Python `EngineCoreRequestType`:
/// - `0x00` = ADD (new request)
/// - `0x01` = ABORT
/// - `0x02` = START_DP_WAVE
/// - `0x03` = UTILITY
/// - `0x04` = EXECUTOR_FAILED (internal sentinel)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[repr(u8)]
pub enum EngineCoreRequestType {
    /// Add a new inference request.
    Add = 0x00,
    /// Abort one or more requests by ID.
    Abort = 0x01,
    /// Start a new data-parallel wave.
    StartDpWave = 0x02,
    /// Invoke a utility method on the engine core.
    Utility = 0x03,
    /// Internal sentinel: executor process failed.
    ExecutorFailed = 0x04,
}

impl EngineCoreRequestType {
    /// Convert from a raw byte.
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x00 => Some(Self::Add),
            0x01 => Some(Self::Abort),
            0x02 => Some(Self::StartDpWave),
            0x03 => Some(Self::Utility),
            0x04 => Some(Self::ExecutorFailed),
            _ => None,
        }
    }

    /// Convert to a single-byte representation.
    pub fn as_byte(self) -> u8 {
        self as u8
    }

    /// Convert to a single-byte slice (for ZMQ send).
    pub fn as_bytes(self) -> [u8; 1] {
        [self as u8]
    }

    /// Return a static byte slice for the request type tag.
    ///
    /// Each variant maps to a distinct 1-byte static slice so callers can use
    /// `Bytes::from_static` and avoid a heap allocation.
    pub fn as_static_bytes(self) -> &'static [u8] {
        match self {
            Self::Add => &[0x00],
            Self::Abort => &[0x01],
            Self::StartDpWave => &[0x02],
            Self::Utility => &[0x03],
            Self::ExecutorFailed => &[0x04],
        }
    }
}

// ---------------------------------------------------------------------------
// Handshake types
// ---------------------------------------------------------------------------

/// Status sent during the engine-core startup handshake.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeHello {
    pub status: Cow<'static, str>,
    pub local: bool,
    pub headless: bool,
}

impl HandshakeHello {
    pub fn new(local: bool, headless: bool) -> Self {
        Self {
            status: Cow::Borrowed("HELLO"),
            local,
            headless,
        }
    }
}

/// Ready message sent after engine-core initialization completes.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HandshakeReady {
    pub status: Cow<'static, str>,
    pub local: bool,
    pub headless: bool,
    pub num_gpu_blocks: usize,
    pub dp_stats_address: Option<String>,
    pub parallel_config_hash: Option<String>,
}

impl HandshakeReady {
    pub fn new(local: bool, headless: bool, num_gpu_blocks: usize) -> Self {
        Self {
            status: Cow::Borrowed("READY"),
            local,
            headless,
            num_gpu_blocks,
            dp_stats_address: None,
            parallel_config_hash: None,
        }
    }
}

/// ZMQ addresses exchanged during the engine-core handshake.
///
/// Mirrors `EngineZmqAddresses` from `vllm/v1/engine/utils.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineZmqAddresses {
    /// Input socket addresses (front-end → engine).
    pub inputs: Vec<String>,
    /// Output socket addresses (engine → front-end).
    pub outputs: Vec<String>,
    /// Coordinator input address (for DP mode).
    pub coordinator_input: Option<String>,
    /// Coordinator output address (for DP mode).
    pub coordinator_output: Option<String>,
    /// Frontend stats publish address (for DP LB).
    pub frontend_stats_publish_address: Option<String>,
}

/// Engine handshake metadata sent from front-end to engine-core.
///
/// Mirrors `EngineHandshakeMetadata` from `vllm/v1/engine/utils.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EngineHandshakeMetadata {
    pub addresses: EngineZmqAddresses,
    pub parallel_config: std::collections::HashMap<String, serde_json::Value>,
}

// ---------------------------------------------------------------------------
// Utility request
// ---------------------------------------------------------------------------

/// A utility method invocation request.
///
/// Sent with `EngineCoreRequestType::Utility`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityRequest {
    /// Index of the client to route the response back to.
    pub client_index: u32,
    /// Unique call ID for matching response to request.
    pub call_id: u64,
    /// Name of the method to invoke on EngineCore.
    pub method_name: String,
    /// Arguments to pass to the method (msgpack-encoded).
    pub args: Vec<serde_json::Value>,
}

/// Response from a utility method invocation.
///
/// Mirrors `UtilityOutput` from `vllm/v1/engine/__init__.py`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UtilityOutput {
    /// Call ID matching the request.
    pub call_id: u64,
    /// Non-None implies the call failed.
    pub failure_message: Option<String>,
    /// The result value (if successful).
    pub result: Option<serde_json::Value>,
}

impl UtilityOutput {
    pub fn new(call_id: u64) -> Self {
        Self {
            call_id,
            failure_message: None,
            result: None,
        }
    }

    pub fn success(call_id: u64, result: serde_json::Value) -> Self {
        Self {
            call_id,
            failure_message: None,
            result: Some(result),
        }
    }

    pub fn failure(call_id: u64, message: String) -> Self {
        Self {
            call_id,
            failure_message: Some(message),
            result: None,
        }
    }
}

// ---------------------------------------------------------------------------
// Pause mode
// ---------------------------------------------------------------------------

/// How to handle existing requests when pausing the scheduler.
///
/// Mirrors the Python `PauseMode` type alias.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum PauseMode {
    /// Abort all in-flight requests immediately.
    Abort,
    /// Wait for in-flight requests to complete before pausing.
    Wait,
    /// Freeze requests in queue; they resume on resume.
    Keep,
}

// ---------------------------------------------------------------------------
// Engine dead sentinel
// ---------------------------------------------------------------------------

/// Sentinel bytes sent when the engine core process dies.
pub const ENGINE_CORE_DEAD: &[u8] = b"ENGINE_CORE_DEAD";

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::codec::{decode, encode};

    #[test]
    fn test_request_type_from_byte() {
        assert_eq!(
            EngineCoreRequestType::from_byte(0x00),
            Some(EngineCoreRequestType::Add)
        );
        assert_eq!(
            EngineCoreRequestType::from_byte(0x01),
            Some(EngineCoreRequestType::Abort)
        );
        assert_eq!(
            EngineCoreRequestType::from_byte(0x02),
            Some(EngineCoreRequestType::StartDpWave)
        );
        assert_eq!(
            EngineCoreRequestType::from_byte(0x03),
            Some(EngineCoreRequestType::Utility)
        );
        assert_eq!(
            EngineCoreRequestType::from_byte(0x04),
            Some(EngineCoreRequestType::ExecutorFailed)
        );
        assert_eq!(EngineCoreRequestType::from_byte(0xFF), None);
    }

    #[test]
    fn test_request_type_as_byte() {
        assert_eq!(EngineCoreRequestType::Add.as_byte(), 0x00);
        assert_eq!(EngineCoreRequestType::Abort.as_byte(), 0x01);
        assert_eq!(EngineCoreRequestType::StartDpWave.as_byte(), 0x02);
        assert_eq!(EngineCoreRequestType::Utility.as_byte(), 0x03);
        assert_eq!(EngineCoreRequestType::ExecutorFailed.as_byte(), 0x04);
    }

    #[test]
    fn test_request_type_as_bytes() {
        assert_eq!(EngineCoreRequestType::Add.as_bytes(), [0x00]);
        assert_eq!(EngineCoreRequestType::Abort.as_bytes(), [0x01]);
    }

    #[test]
    fn test_handshake_hello_roundtrip() {
        let hello = HandshakeHello::new(true, false);
        let encoded = encode(&hello).unwrap();
        let decoded: HandshakeHello = decode(&encoded).unwrap();
        assert_eq!(decoded.status, "HELLO");
        assert!(decoded.local);
        assert!(!decoded.headless);
    }

    #[test]
    fn test_handshake_ready_roundtrip() {
        let ready = HandshakeReady::new(true, false, 1024);
        let encoded = encode(&ready).unwrap();
        let decoded: HandshakeReady = decode(&encoded).unwrap();
        assert_eq!(decoded.status, "READY");
        assert_eq!(decoded.num_gpu_blocks, 1024);
        assert!(decoded.dp_stats_address.is_none());
    }

    #[test]
    fn test_engine_zmq_addresses_roundtrip() {
        let addrs = EngineZmqAddresses {
            inputs: vec!["tcp://127.0.0.1:5555".into()],
            outputs: vec!["tcp://127.0.0.1:5556".into()],
            coordinator_input: None,
            coordinator_output: None,
            frontend_stats_publish_address: None,
        };

        let encoded = encode(&addrs).unwrap();
        let decoded: EngineZmqAddresses = decode(&encoded).unwrap();
        assert_eq!(decoded.inputs, vec!["tcp://127.0.0.1:5555"]);
        assert_eq!(decoded.outputs, vec!["tcp://127.0.0.1:5556"]);
    }

    #[test]
    fn test_utility_output_success() {
        let out = UtilityOutput::success(42, serde_json::json!(true));
        assert_eq!(out.call_id, 42);
        assert!(out.failure_message.is_none());
        assert_eq!(out.result, Some(serde_json::json!(true)));
    }

    #[test]
    fn test_utility_output_failure() {
        let out = UtilityOutput::failure(99, "method not found".into());
        assert_eq!(out.call_id, 99);
        assert_eq!(out.failure_message.as_deref(), Some("method not found"));
        assert!(out.result.is_none());
    }

    #[test]
    fn test_utility_output_roundtrip() {
        let out = UtilityOutput::success(7, serde_json::json!({"key": "value"}));
        let encoded = encode(&out).unwrap();
        let decoded: UtilityOutput = decode(&encoded).unwrap();
        assert_eq!(decoded.call_id, 7);
        assert_eq!(decoded.result, Some(serde_json::json!({"key": "value"})));
    }

    #[test]
    fn test_pause_mode_roundtrip() {
        for mode in [PauseMode::Abort, PauseMode::Wait, PauseMode::Keep] {
            let encoded = encode(&mode).unwrap();
            let decoded: PauseMode = decode(&encoded).unwrap();
            assert_eq!(decoded, mode);
        }
    }

    #[test]
    fn test_engine_core_dead_sentinel() {
        assert_eq!(ENGINE_CORE_DEAD, b"ENGINE_CORE_DEAD");
    }
}
