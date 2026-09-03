// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Async ZMQ transport layer for engine-core IPC.
//!
//! Provides socket wrappers matching the patterns used by vLLM's
//! engine-core process:
//!
//! - **DEALER** sockets for engine-core input (bidirectional async)
//! - **PUSH** sockets for engine-core output (unidirectional)
//! - **ROUTER** sockets for front-end request dispatching
//! - **PULL** sockets for front-end output receiving
//! - **XSUB/XPUB** sockets for coordinator pub/sub
//!
//! Uses the `zeromq` crate (pure-Rust async ZMQ) which integrates
//! with `tokio`.
//!
//! # Architecture
//!
//! The engine-core IPC uses this topology:
//!
//! ```text
//!   Front-end (API Server)          Engine Core Process
//!   ┌─────────────────────┐         ┌───────────────────┐
//!   │  ROUTER (input) ────┼────────►│ DEALER (input)    │
//!   │                     │         │                   │
//!   │  PULL (output) ◄────┼─────────│ PUSH (output)     │
//!   └─────────────────────┘         └───────────────────┘
//! ```
//!
//! Multipart messages carry a type tag in frame 0 and msgpack payload
//! in subsequent frames.

use std::time::Duration;

use bytes::Bytes;
use zeromq::{
    DealerSocket, PullSocket, PushSocket, RouterSocket, Socket, SocketRecv, SocketSend, ZmqMessage,
};

use crate::codec::{self, CodecError, MsgpackDecoder, MsgpackEncoder};
use crate::messages::EngineCoreRequestType;

/// Errors from the transport layer.
#[derive(Debug, thiserror::Error)]
pub enum TransportError {
    /// ZMQ error.
    #[error("zmq error: {0}")]
    Zmq(#[from] zeromq::ZmqError),

    /// Codec error (serialization/deserialization).
    #[error("codec error: {0}")]
    Codec(#[from] CodecError),

    /// Invalid message format.
    #[error("invalid message: {0}")]
    InvalidMessage(std::borrow::Cow<'static, str>),

    /// Connection timeout.
    #[error("connection timeout after {0:?}")]
    Timeout(Duration),
}

pub type TransportResult<T> = Result<T, TransportError>;

// ---------------------------------------------------------------------------
// EngineInputSocket -- DEALER socket for receiving requests
// ---------------------------------------------------------------------------

/// DEALER socket used by the engine core to receive requests from the
/// front-end ROUTER.
///
/// Wraps a ZMQ DEALER socket with typed send/recv for engine protocol
/// messages.
pub struct EngineInputSocket {
    socket: DealerSocket,
    decoder: MsgpackDecoder,
}

impl EngineInputSocket {
    /// Create a new input socket and connect to the given address.
    pub async fn connect(address: &str) -> TransportResult<Self> {
        let mut socket = DealerSocket::new();
        socket.connect(address).await?;
        Ok(Self {
            socket,
            decoder: MsgpackDecoder::new(),
        })
    }

    /// Send an initial empty frame to register with the ROUTER.
    ///
    /// Required before the front-end can route messages to this engine.
    pub async fn send_registration(&mut self) -> TransportResult<()> {
        let msg = ZmqMessage::from(Bytes::new());
        self.socket.send(msg).await?;
        Ok(())
    }

    /// Receive a typed request from the front-end.
    ///
    /// Returns the request type and the raw payload frames (not yet decoded).
    pub async fn recv_typed(&mut self) -> TransportResult<(EngineCoreRequestType, Vec<Bytes>)> {
        let msg = self.socket.recv().await?;
        let mut frames: Vec<Bytes> = msg.into_vec();

        if frames.is_empty() {
            return Err(TransportError::InvalidMessage(
                "empty multipart message".into(),
            ));
        }

        let type_frame = &frames[0];
        if type_frame.is_empty() {
            return Err(TransportError::InvalidMessage("empty type frame".into()));
        }

        let request_type = EngineCoreRequestType::from_byte(type_frame[0]).ok_or_else(|| {
            TransportError::InvalidMessage(
                format!("unknown request type: 0x{:02x}", type_frame[0]).into(),
            )
        })?;

        let data_frames = frames.split_off(1);
        Ok((request_type, data_frames))
    }

    /// Receive and decode a typed request.
    ///
    /// Deserializes the payload frames into the given type.
    pub async fn recv_decoded<T: serde::de::DeserializeOwned>(
        &mut self,
    ) -> TransportResult<(EngineCoreRequestType, T)> {
        let (req_type, data_frames) = self.recv_typed().await?;
        if data_frames.is_empty() {
            return Err(TransportError::InvalidMessage(
                "no data frames in message".into(),
            ));
        }

        let value: T = self.decoder.decode(&data_frames[0])?;
        Ok((req_type, value))
    }
}

// ---------------------------------------------------------------------------
// EngineOutputSocket -- PUSH socket for sending outputs
// ---------------------------------------------------------------------------

/// PUSH socket used by the engine core to send outputs to the front-end.
pub struct EngineOutputSocket {
    socket: PushSocket,
    encoder: MsgpackEncoder,
}

impl EngineOutputSocket {
    /// Create a new output socket and connect to the given address.
    pub async fn connect(address: &str) -> TransportResult<Self> {
        let mut socket = PushSocket::new();
        socket.connect(address).await?;
        Ok(Self {
            socket,
            encoder: MsgpackEncoder::new(),
        })
    }

    /// Send a serialized value over the socket.
    pub async fn send<T: serde::Serialize>(&mut self, value: &T) -> TransportResult<()> {
        let encoded = self.encoder.encode(value)?;
        let msg = ZmqMessage::from(encoded);
        self.socket.send(msg).await?;
        Ok(())
    }

    /// Send raw bytes over the socket.
    pub async fn send_raw(&mut self, data: Bytes) -> TransportResult<()> {
        let msg = ZmqMessage::from(data);
        self.socket.send(msg).await?;
        Ok(())
    }

    /// Send the engine-dead sentinel.
    pub async fn send_engine_dead(&mut self) -> TransportResult<()> {
        self.send_raw(Bytes::from_static(crate::messages::ENGINE_CORE_DEAD))
            .await
    }
}

// ---------------------------------------------------------------------------
// FrontendInputSocket -- ROUTER socket for dispatching requests
// ---------------------------------------------------------------------------

/// ROUTER socket used by the front-end to dispatch requests to engine
/// cores.
pub struct FrontendInputSocket {
    socket: RouterSocket,
    encoder: MsgpackEncoder,
}

impl FrontendInputSocket {
    /// Create a new frontend input socket and bind to the given address.
    pub async fn bind(address: &str) -> TransportResult<Self> {
        let mut socket = RouterSocket::new();
        socket.bind(address).await?;
        Ok(Self {
            socket,
            encoder: MsgpackEncoder::new(),
        })
    }

    /// Send a typed request to a specific engine (identified by identity).
    pub async fn send_to<T: serde::Serialize>(
        &mut self,
        identity: &[u8],
        request_type: EngineCoreRequestType,
        value: &T,
    ) -> TransportResult<()> {
        let payload = self.encoder.encode(value)?;
        let mut msg = ZmqMessage::from(Bytes::copy_from_slice(identity));
        msg.push_back(Bytes::from_static(request_type.as_static_bytes()));
        msg.push_back(payload);
        self.socket.send(msg).await?;
        Ok(())
    }

    /// Receive a registration message from an engine.
    pub async fn recv_registration(&mut self) -> TransportResult<Bytes> {
        let msg = self.socket.recv().await?;
        let mut frames: Vec<Bytes> = msg.into_vec();
        if frames.is_empty() {
            return Err(TransportError::InvalidMessage(
                "empty registration message".into(),
            ));
        }
        Ok(frames.remove(0))
    }
}

// ---------------------------------------------------------------------------
// FrontendOutputSocket -- PULL socket for receiving outputs
// ---------------------------------------------------------------------------

/// PULL socket used by the front-end to receive outputs from engine cores.
pub struct FrontendOutputSocket {
    socket: PullSocket,
    decoder: MsgpackDecoder,
}

impl FrontendOutputSocket {
    /// Create a new frontend output socket and bind to the given address.
    pub async fn bind(address: &str) -> TransportResult<Self> {
        let mut socket = PullSocket::new();
        socket.bind(address).await?;
        Ok(Self {
            socket,
            decoder: MsgpackDecoder::new(),
        })
    }

    /// Receive and decode an output message.
    ///
    /// Returns `None` if the engine-dead sentinel is received.
    pub async fn recv<T: serde::de::DeserializeOwned>(&mut self) -> TransportResult<Option<T>> {
        let msg = self.socket.recv().await?;
        let frames: Vec<Bytes> = msg.into_vec();
        if frames.is_empty() {
            return Err(TransportError::InvalidMessage(
                "empty output message".into(),
            ));
        }

        // Check for engine-dead sentinel.
        if frames[0].as_ref() == crate::messages::ENGINE_CORE_DEAD {
            return Ok(None);
        }

        let value: T = self.decoder.decode(&frames[0])?;
        Ok(Some(value))
    }

    /// Receive raw bytes from the socket.
    pub async fn recv_raw(&mut self) -> TransportResult<Vec<Bytes>> {
        let msg = self.socket.recv().await?;
        Ok(msg.into_vec())
    }
}

// ---------------------------------------------------------------------------
// Helper: multipart message construction
// ---------------------------------------------------------------------------

/// Build a multipart ZMQ message from a request type and serialized payload.
pub fn build_request_message<T: serde::Serialize>(
    request_type: EngineCoreRequestType,
    value: &T,
) -> TransportResult<ZmqMessage> {
    let payload = codec::encode(value)?;
    let mut msg = ZmqMessage::from(Bytes::from_static(request_type.as_static_bytes()));
    msg.push_back(payload);
    Ok(msg)
}

/// Parse the request type from the first frame of a multipart message.
pub fn parse_request_type(frames: &[Bytes]) -> TransportResult<EngineCoreRequestType> {
    if frames.is_empty() {
        return Err(TransportError::InvalidMessage("empty message".into()));
    }
    let type_frame = &frames[0];
    if type_frame.is_empty() {
        return Err(TransportError::InvalidMessage("empty type frame".into()));
    }
    EngineCoreRequestType::from_byte(type_frame[0]).ok_or_else(|| {
        TransportError::InvalidMessage(
            format!("unknown request type: 0x{:02x}", type_frame[0]).into(),
        )
    })
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_request_message() {
        let msg =
            build_request_message(EngineCoreRequestType::Abort, &vec!["req-1", "req-2"]).unwrap();

        let frames: Vec<Bytes> = msg.into_vec();
        assert_eq!(frames.len(), 2);
        assert_eq!(frames[0].as_ref(), &[0x01]); // Abort type byte
    }

    #[test]
    fn test_parse_request_type() {
        let frames = vec![Bytes::from_static(&[0x00]), Bytes::from_static(b"payload")];
        let req_type = parse_request_type(&frames).unwrap();
        assert_eq!(req_type, EngineCoreRequestType::Add);
    }

    #[test]
    fn test_parse_request_type_empty() {
        let result = parse_request_type(&[]);
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_request_type_unknown() {
        let frames = vec![Bytes::from_static(&[0xFF])];
        let result = parse_request_type(&frames);
        assert!(result.is_err());
    }
}
