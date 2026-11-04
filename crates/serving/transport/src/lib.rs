// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Wire formats: msgpack serialization and the TCP control channel between ranks.
//!
//! This crate provides:
//!
//! * [`codec`] -- Msgpack encoder/decoder wrapping `rmp-serde`, designed for
//!   efficient serialization of engine-core request/response types.
//! * [`messages`] -- Protocol-level message types: request type tags,
//!   framing envelopes, and handshake metadata.
//! * [`tcp_store`] -- Blocking TCP rendezvous and control channel between
//!   the ranks of a multi-node deployment.

pub mod codec;
pub mod messages;
pub mod tcp_store;

pub use tcp_store::TcpControlChannel;
