# senlib FFI boundary — telemetry subsystem

Scope: `src/telemetry.rs` (flex's own `telemetry/` subsystem: `FlexLogExtra`,
`Histogram`, `LatencyBreakdown`, `HistTimer`, `Profiler`/`GlobalProfiler`,
`telemetry::UnifiedProfiler`, the pure-math half of `TimestampCalibrator`) and
`src/aiupti.rs` (`flex::aiupti::Profiler`, the real libaiupti source).

## Correction to the prior attempt

The previous pass at this subsystem stubbed `flex::aiupti::Profiler` wholesale,
reasoning that `aiupti` is "IBM's external, closed-source instrumentation
library... no flex algorithm to port." That was wrong. The real source is
available at `/Users/nickm/git/libaiupti-cxx/src/aiupti/` (its own LICENSE,
README, git history) and its `Profiler` class (`profiler.h`, `aiupti.cpp`,
`aiupti_api.cpp`) is ~1200 lines of real, flex-owned bookkeeping:

- Activity-buffer allocation, rotation, and drop accounting
  (`buffersCallbackRequest`, `tryToAllocateBuffer`, `MAX_BUFFER_COUNT`/
  `K_BUF_SIZE`) — ported as `Profiler::buffers_callback_request`/
  `rotate_buffer`/`release_buffer` in `aiupti.rs`.
- Per-activity-kind enable/disable tracking (`enabled_activities_`,
  `flexRegisterActivity`/`flexActivityDisable`) — ported as
  `Profiler::flex_register_activity`/`flex_activity_disable`.
- Record-kind conversion from flex's internal CB/Memory/Memcpy/Sync/API
  activity structs to the AIUPTI wire structs, including the
  prep-vs-exec / htod-vs-dtoh timestamp-window selection logic
  (`createActivity` overloads) — ported as `Profiler::create_*_activity`.
- A power-delta calculation with 32-bit charge-counter overflow handling and
  the driver's documented physical constants (12 V rail, 5.37/512 LSB-per-
  ADC-count) — ported as `get_power_value_and_timestamp`.
- Kind/name lookup tables (`getActivityKindName`, `getCBIDName`,
  `getActivityKind`, `convertFlexEventKind`/`getFlexEventKind`) — ported as
  the `*_name`/`convert_*`/`flex_event_kind_of` functions.

None of this reads a hardware register, a PCI BAR, or calls into senlib. It
allocates plain heap buffers, copies/reinterprets already-populated struct
fields, and does arithmetic. **This file's `Profiler` therefore crosses zero
senlib boundary.**

## Where the actual hardware read lives (and why it's out of this port's scope)

Grepping the full libaiupti tree for `senlib`/hardware-register access turns
up exactly one hit outside `Profiler`: `aiupti_metric.cpp` (a *separate*
class, `AIUptiMetric`, that periodically samples HPM/power counters for the
metrics-exporter path) includes `senlib/1p0/senpci.hpp` and
`senlib/shared/senlib_config.hpp` and calls `senlib::v2::SenPci` to read
PCI-config-space performance-counter/DCR values on a timer. That is the one
place in this whole subsystem where a literal hardware register read occurs.

`AIUptiMetric` is not part of the `flex::aiupti::Profiler` class this task
asks to port, is not reachable from the stated 19-type/50-function
`flex/telemetry/` closure (which lists `aiupti_activity.hpp`,
`timestamp_calibrator.hpp`, `hist_timer.hpp`, `telemetry_utils.hpp`,
`profiler.cpp`, `histogram.hpp`, `latency_breakdown.hpp` — no metric sampler),
and is not exercised by anything currently ported elsewhere in this crate. It
is intentionally **not ported** in this pass (porting a ~1000-line sampler
class that nothing in the current closure calls would be scope creep, not a
gap-fill). If a future pass needs it, the boundary should be a single
`senlib_ffi_telemetry.rs` extern wrapping the specific `SenPci` register-read
call(s) in `aiupti_metric.cpp` (functions around
`getPerfCounterSectionId`/the `metricValues_` sampling loop) — everything else
in that file (interval parsing from senlib config, JSON export, aggregation)
is software bookkeeping like the rest of this subsystem.

## `TimestampCalibrator`

`flex::TimestampCalibrator` (`timestamp_calibrator.hpp`) also touches
hardware, but indirectly: `InitialCalibration`/`Measure`/
`ParseResponseBlocks` submit control blocks through `ControlBlockStream`/
`FlexStreamer` and decode `senlib::ResponseBlockSBF` responses to get paired
host/device timestamps. That control-block submission and response-decoding
machinery is not this subsystem's own — it's the same machinery already
ported (and already carrying its own senlib boundary) in
`crate::control_blocks` / `crate::scheduler` / `senlib_ffi_scheduler.rs` /
`senlib_ffi_controlblocks.rs`. Reimplementing it inside `telemetry.rs` would
duplicate an existing boundary rather than document a new one.

This port therefore ports `TimestampCalibrator`'s own algorithm — linear
regression (`CalibrateMeasurements`), Kalman smoothing (`Kalman`), 32-bit
device-counter wraparound tracking, and host<->device time conversion
(`ToHostTime`/`ToDevTime`/`GetCurrentPhase`) — as pure Rust in `telemetry.rs`,
and leaves the device round-trip as an external input (callers feed it
`CalibrationRoundResult`s obtained via the existing control-block path).
**No new senlib boundary is introduced by this piece either.**

## Net result

`src/telemetry.rs` and `src/aiupti.rs` add **zero new `extern "C"` /
`senlib_ffi_*` declarations** to this crate. No `unsafe` block is required in
either file. The one real hardware-register-read call site that exists
anywhere in this subsystem's source tree (`aiupti_metric.cpp`'s `SenPci`
sampling) is out of the ported closure's scope, as detailed above, and is
flagged here rather than silently dropped.
