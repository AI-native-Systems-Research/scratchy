// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Wire formats: msgpack serialization and ZMQ transport for vLLM.
//!
//! This crate provides:
//!
//! * [`codec`] -- Msgpack encoder/decoder wrapping `rmp-serde`, designed for
//!   efficient serialization of engine-core request/response types.
//! * [`messages`] -- Protocol-level message types: request type tags,
//!   framing envelopes, and handshake metadata.
//! * [`transport`] -- Async ZMQ socket management using `zeromq` crate,
//!   with patterns matching the Python engine-core IPC.

pub mod codec;
pub mod messages;
#[cfg(feature = "multiproc")]
pub mod transport;
