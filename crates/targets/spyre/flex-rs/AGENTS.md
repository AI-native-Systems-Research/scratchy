# crates/targets/spyre/flex-rs — `flex-rs` (lib `flex_rs`)

## OVERVIEW
Native Rust port of the flex C++ allocator/runtime-stream/scheduler slice (134 types / 320 fns / 17 ABI entrypoints) that scratchy's card path needs; ~31k LOC. Earned its file: standalone crate (own version, edition 2024, own `Cargo.lock`), the spyre tree's only working build script (builder/'s is a no-op), and the only FFI boundary to hardware.

## WHERE TO LOOK
| Task | Location |
|------|----------|
| Every `extern "C"` against senlib + safe handle wrappers | `src/senlib_ffi.rs` |
| C++ side of that boundary (compiled by build.rs, C++20) | `cxx-shim/senlib_ffi.cpp` |
| Boundary rationale / provenance ledger | `SENLIB_BOUNDARY.md`, `SENLIB_BOUNDARY_telemetry.md` |
| Scheduler port (Pf/Vf/Mock, `ResponseWorker`, backpressure, `HardwareSubmit` seam) | `src/scheduler.rs` (8.1k LOC) |
| Cross-stream RAW/WAR hazard DAG | `src/scheduler/hazards.rs` (`HazardTracker`) |
| CB completion verdict + capability tokens | `src/scheduler/completion.rs` (`CbRetired`, `QuarantinedBuf`, `cb_verdict`) |
| Wire format / SBF encoding | `src/control_block_wire.rs`, `src/control_blocks.rs` |
| Env config (`FLEX_*`) | `src/flex_config.rs` |
| Allocator / regions / addresses | `src/allocator.rs`, `memory_region.rs`, `address.rs` |
| Pinned shared host slots, one IOMMU mapping | `src/shared_host_pool.rs` |
| RDMA/HDMA P2P | `src/multi_device.rs` |
| Card filtering (`SPYRE_DEVICES`, `AIU_WORLD_SIZE`) | `src/runtime.rs` (`parse_spyre_devices`) |

## CONVENTIONS
- Each module doc cites the C++ files it ports and a "Recorded divergences" list; keep both current when the port changes behavior.
- Links ONLY senlib + flightlog + a fixed boost set. Never `libflex.so`, never `sendnn_*`. Link flags are emitted even when the compile is skipped, so a real link fails loudly.
- Env: `FLEX_RS_SPYRE_ROOT` (default `/opt/ibm/spyre`), `BOOST_LIB` (default `/usr/lib64`), `SCRATCHY_SKIP_SENDNN_CXX=1` (skip shim compile; check/clippy only; CI linux-cuda sets it).
- Capability tokens with private fields make misuse unwritable: `CbRetired` has exactly two constructors (`from_completion`, `never_submitted`, named so grep finds each bypass); `QuarantinedBuf` has no accessor or `Deref`.
- Pure decision split from diagnostics (`cb_verdict` vs `eval_cb_complete_status`) so the decision is Kani-provable. Proofs sit in a `#[cfg(kani)]` child module (`src/scheduler/completion/kani_proofs.rs`). They run only under the kani toolchain.
- Tests touching env go through `config_test_lock`/`EnvGuard`.
- Timeouts are bounded and fail with an observation, never hang (RB wait 10 s).

## ANTI-PATTERNS
- Any `extern "C"` outside `src/senlib_ffi.rs`, or a non-senlib `extern "C"` block.
- Ignoring a `Result`/`#[must_use]`: `#![deny(unused_must_use)]` is deliberate. The long-standing hardware hang was an unreturned `FreeResponses` credit and a discarded submit failure (`ResponseCredit`/`PipelineCredit`).
- Collapsing native orchestration into one opaque FFI wrapper around the whole C++ class.
- Holding hazard/IOMMU bookkeeping locks while parked on an event. Splitting registration from the conflict scan across lock holds (breaks DAG acyclicity). Over-edging only serializes; under-edging corrupts.
- Treating `Option` as "skip the check" in completion: absent response = zero block (`EffectiveResponse`).
- Recycling a `DmaBuf` under a live CB (was `RAS::PCI::BusFence`).
- Control-block wire: no clear operand against a word; transfer byte count is never an XLAT bound.
- Cardless tests that silently pass. On-card legs are `#[ignore]`d and refuse loudly.
- rb telemetry is always on by design; don't gate it.

## COMMANDS
```bash
SCRATCHY_SKIP_SENDNN_CXX=1 cargo check -p flex-rs     # laptop; a link fails
cargo test -p flex-rs                                  # Spyre host (links senlib)
FLEX_RS_SPYRE_ROOT=<root> BOOST_LIB=<dir> \
  cargo test -p flex-rs --test on_card_cancelled_marker -- --ignored --nocapture  # real card, no SKIP var
```
