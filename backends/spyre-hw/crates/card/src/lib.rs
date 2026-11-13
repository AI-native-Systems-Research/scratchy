// SPDX-License-Identifier: Apache-2.0
//! THE CARD: what runs a baked bundle on a Spyre card — the typed card surface over flex-rs, the
//! resident SuperDSC executor, and the session the worker drives. Everything here needs the card's
//! runtime; nothing in `scratchy-target-spyre` depends on it, so the shared target (lowering, bake,
//! emulator) builds without it.

/// Run a bundle this binary did not bake — see the module docs.
pub mod bundle_run;
/// The bake's packed device images, decoded — the metal AOT precedent's load half. Compression runs
/// in the compiler at expansion; this is the door that turns a frame back into the bytes the card
/// receives.
pub mod baked_image;
/// #603 PR2 — one launch's compute CBs as data: the submit walk's arithmetic
/// single-sourced (`launch_site`), the plan/render twin, and the prepare-time
/// byte-identical parity gate.
pub(crate) mod cb_plan;
/// The typed card surface over flex-rs: streams, device addresses, launches, and the card's
/// memory capacity — `Arc<RuntimeContext>`-backed handles, no C ABI. It links the SDK via
/// flex-rs's cxx-shim.
pub mod runtime;
/// The worker-facing session wrapper over [`superdsc_exec`] (f32 in, f32 logits out).
pub mod sdsc_runner;
/// The resident SuperDSC executor: allocate the bundle's program + weights + KV pool once, then
/// launch each per-op kernel against the shared resident segments. The `--target sendnn` run path.
pub mod superdsc_exec;
