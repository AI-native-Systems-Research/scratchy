// SPDX-License-Identifier: Apache-2.0
//! THE CARD: everything the worker does only on a Spyre card — the resident session and its
//! rungs, the launch, the batched decode's submit/drain queue. The emulator's counterpart is
//! `crate::emu`; what both run is the rest of the crate.

pub(crate) mod exec;
pub(crate) mod forward;
pub(crate) mod pool;
pub(crate) mod types;
