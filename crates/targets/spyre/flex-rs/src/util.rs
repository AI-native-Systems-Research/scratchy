//! Port of `flex/src/util/util.hpp` / `util.cpp` and `flex/include/flex/util/status.hpp` /
//! `flex/src/util/status.cpp`.
//!
//! Everything here is pure value/string logic — env-var parsing, percentage math,
//! rounding, alignment checks, and the `sendnn::Status` result type. No senlib
//! involvement anywhere in this file.
//!
//! Note on `FlexLogExtra`: this phase's own port of `flex::FlexLogExtra`
//! (previously duplicated here as `UtilFlexLogExtra`, per the parallel-agent
//! disclaimer) was a byte-for-byte behavioral duplicate of
//! `crate::telemetry::FlexLogExtra`, ported earlier from the same C++ source
//! (`util.hpp`/`util.cpp`). The Assembly phase consolidated on the
//! `telemetry` copy (it has broader existing test coverage and is already
//! the one referenced by `crate::telemetry`'s other config/profiler types)
//! and removed this file's copy — see `SENLIB_BOUNDARY.md` for the
//! consolidation note. Use `crate::telemetry::FlexLogExtra` for this
//! functionality.

// ---------------------------------------------------------------------------
// sendnn::Status — port of flex/include/flex/util/status.hpp + status.cpp
// ---------------------------------------------------------------------------

/// Port of `sendnn::StatusCode` (a plain `int64_t` in C++, newtyped here).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct StatusCode(pub i64);

const STATUS_CODE_UNDEFINED: StatusCode = StatusCode(0x0);
const STATUS_CODE_OK: StatusCode = StatusCode(0x1);

/// A short, human-readable status message. Newtype so `Status` never exposes
/// a bare `String` at its public boundary.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StatusMessage(pub String);

impl StatusMessage {
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for StatusMessage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Port of `sendnn::Status`: a `[[nodiscard]]` result/error-code pair used
/// pervasively across flex for fallible operations.
#[derive(Debug, Clone)]
#[must_use]
pub struct Status {
    code: StatusCode,
    message: StatusMessage,
}

impl PartialEq for Status {
    /// Port of `operator==(const Status&, const Status&)`
    /// (`status.hpp:149`): `lhs.Code() == rhs.Code()` — the message is
    /// intentionally NOT compared, unlike a derived `PartialEq` would.
    fn eq(&self, other: &Self) -> bool {
        self.code == other.code
    }
}

impl Eq for Status {}

impl Default for Status {
    /// Port of `Status::Status()`: the "undefined" default state.
    fn default() -> Self {
        Self {
            code: STATUS_CODE_UNDEFINED,
            message: StatusMessage("Status Undefined".to_string()),
        }
    }
}

impl Status {
    /// Port of `Status::Status(StatusCode, const std::string&)`. Logs a warning
    /// via `on_status_warning` if the constructed status is not OK, matching the
    /// C++ constructor's side effect.
    pub fn new(code: StatusCode, message: StatusMessage) -> Self {
        let s = Self { code, message };
        if !s.is_ok() {
            on_status_warning(&s);
        }
        s
    }

    pub fn has_been_set(&self) -> bool {
        self.code != STATUS_CODE_UNDEFINED
    }

    pub fn is_ok(&self) -> bool {
        self.code == STATUS_CODE_OK
    }

    pub fn code(&self) -> StatusCode {
        self.code
    }

    pub fn message(&self) -> &StatusMessage {
        &self.message
    }

    /// Port of `Status::ChainUpdate`: keeps the first error seen.
    pub fn chain_update(&mut self, s: &Status) {
        if !self.has_been_set() || (self.is_ok() && !s.is_ok()) {
            *self = s.clone();
        }
    }

    pub fn ok() -> Self {
        Self::new(STATUS_CODE_OK, StatusMessage("Status OK".to_string()))
    }

    pub fn invalid_argument(msg: impl Into<String>) -> Self {
        Self::new(StatusCode(400), StatusMessage(msg.into()))
    }

    /// Port of `Status::INVALID_ARGUMENT()` (`flex/include/flex/util/status.hpp:83`):
    /// `Status(400, "Invalid argument")` — the zero-arg convenience overload
    /// with the real default message, alongside `invalid_argument(msg)` above.
    pub fn invalid_argument_default() -> Self {
        Self::new(
            StatusCode(400),
            StatusMessage("Invalid argument".to_string()),
        )
    }

    pub fn key_not_found(key: &str) -> Self {
        Self::new(
            StatusCode(401),
            StatusMessage(format!("Key not found: {key}")),
        )
    }

    pub fn runtime_error(msg: impl Into<String>) -> Self {
        Self::new(StatusCode(403), StatusMessage(msg.into()))
    }

    pub fn unknown_error() -> Self {
        Self::new(StatusCode(404), StatusMessage("Unknown error".to_string()))
    }

    pub fn unsupported_operation() -> Self {
        Self::new(
            StatusCode(405),
            StatusMessage("This operation is not supported".to_string()),
        )
    }

    pub fn unsupported_scenario(msg: impl Into<String>) -> Self {
        Self::new(StatusCode(406), StatusMessage(msg.into()))
    }

    /// Port of the zero-arg `UNSUPPORTED_SCENARIO()` convenience overload
    /// (`flex/include/flex/util/status.hpp:122`): `Status(406, "The
    /// attributes of this operation are not supported")`.
    pub fn unsupported_scenario_default() -> Self {
        Self::new(
            StatusCode(406),
            StatusMessage("The attributes of this operation are not supported".to_string()),
        )
    }
}

impl std::fmt::Display for Status {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

/// Port of `sendnn::OnStatusWarning(const Status&)`. The C++ version routes
/// through `common::logging`; this crate routes through `tracing` instead
/// (RUST_LOG=warn to see these).
pub fn on_status_warning(s: &Status) {
    tracing::warn!("{s}");
}

// ---------------------------------------------------------------------------
// Small free functions — port of the generic helpers in util.hpp
// ---------------------------------------------------------------------------

/// Port of `flex::AsPercentOf`. Returns `a / b * 100`, or `0.0` if `b == 0`.
pub fn as_percent_of(a: u64, b: u64) -> f64 {
    if b == 0 {
        0.0
    } else {
        (a as f64) / (b as f64) * 100.0
    }
}

/// Port of `sendnn::ceil_to`: rounds `val` up to the nearest multiple of `ceil`.
pub fn ceil_to(val: u64, ceil: u64) -> u64 {
    let rem = val % ceil;
    if rem == 0 { val } else { val + ceil - rem }
}

/// Port of `sendnn::DEVICE_ALIGNMENT` (`sendnn/runtime/segment_table.hpp:324`):
/// `constexpr std::size_t DEVICE_ALIGNMENT = 128ULL;` — the default alignment
/// `CheckIfNotAligned`'s C++ default parameter falls back to.
pub const DEVICE_ALIGNMENT: u64 = 128;

/// Port of `flex::CheckIfNotAligned` (`util/defines.hpp`): true if `size_bytes`
/// is not a multiple of `alignment_bytes`. `alignment_bytes` defaults to
/// [`DEVICE_ALIGNMENT`] when `None`, matching the C++ default parameter
/// `CheckIfNotAligned(size_t size_bytes, size_t alignment_bytes =
/// DEVICE_ALIGNMENT)`.
pub fn check_if_not_aligned(size_bytes: u64, alignment_bytes: impl Into<Option<u64>>) -> bool {
    let alignment_bytes = alignment_bytes.into().unwrap_or(DEVICE_ALIGNMENT);
    !size_bytes.is_multiple_of(alignment_bytes)
}

// ---------------------------------------------------------------------------
// FlexLogExtra (flex::FlexLogExtra, util.hpp / util.cpp) — consolidated onto
// `crate::telemetry::FlexLogExtra`; see module doc above. Nothing else in
// this file depended on the removed `UtilFlexLogExtra`/`FoutType`/`LogLine`
// types, so no re-export shim is needed.
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ceil_to_rounds_up() {
        assert_eq!(ceil_to(10, 8), 16);
        assert_eq!(ceil_to(16, 8), 16);
        assert_eq!(ceil_to(0, 8), 0);
    }

    #[test]
    fn as_percent_of_handles_zero_denominator() {
        assert_eq!(as_percent_of(5, 0), 0.0);
        assert_eq!(as_percent_of(50, 200), 25.0);
    }

    #[test]
    fn check_if_not_aligned_basic() {
        assert!(!check_if_not_aligned(4096, 4096));
        assert!(check_if_not_aligned(4097, 4096));
    }

    #[test]
    fn check_if_not_aligned_defaults_to_device_alignment() {
        assert!(!check_if_not_aligned(DEVICE_ALIGNMENT, None));
        assert!(check_if_not_aligned(DEVICE_ALIGNMENT + 1, None));
    }

    #[test]
    fn status_chain_update_keeps_first_error() {
        let mut s = Status::ok();
        let e1 = Status::runtime_error("first");
        let e2 = Status::runtime_error("second");
        s.chain_update(&e1);
        s.chain_update(&e2);
        assert_eq!(s.message().as_str(), "first");
    }
}
