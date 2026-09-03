// SPDX-License-Identifier: Apache-2.0
//! Errors, kept concrete so callers can tell "the repo does not have this
//! file" apart from "the download failed" — the distinction whose loss made
//! a cluster failure report `no safetensors weights found` for a file that
//! existed.

use std::path::PathBuf;

/// Result alias for this crate.
pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The repo does not contain this file (Hub returned 404). Distinct from
    /// every other variant: this one, and only this one, means "not there".
    #[error("{repo}: no such file `{filename}`")]
    NotFound { repo: String, filename: String },

    /// Hub returned a status we do not know how to act on.
    #[error("{url}: unexpected HTTP status {status}")]
    UnexpectedStatus { url: String, status: u16 },

    /// A response was missing a header the download depends on.
    #[error("{url}: response missing `{header}`")]
    MissingHeader { url: String, header: &'static str },

    /// Transport failure (DNS, TLS, connection reset, timeout), after every
    /// retry was spent.
    #[error("{url}: {source} (after {attempts} attempts)")]
    Transport {
        url: String,
        attempts: usize,
        #[source]
        source: Box<ureq::Error>,
    },

    /// The caller's [`Progress`](crate::Progress) asked to stop. Partial
    /// state is left in place on purpose so the next run resumes it.
    #[error("{filename}: cancelled by caller")]
    Cancelled { filename: String },

    /// Content did not match the digest the Hub advertised. The partial file
    /// is removed before this is returned.
    #[error("{filename}: content hash mismatch (expected {expected}, got {actual})")]
    HashMismatch {
        filename: String,
        expected: String,
        actual: String,
    },

    // No free-space preflight here on purpose: the accurate check needs the
    // macOS purgeable-capacity query that `scratchy-core-model` already owns
    // (plain `statvfs` under-reports on APFS and rejected downloads that
    // would have succeeded). Callers run it; a mid-download ENOSPC arrives
    // as `Io`, whose message already says "No space left on device".
    #[error("{path}: {source}")]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },

    #[error("parsing {what}: {source}")]
    Json {
        what: String,
        #[source]
        source: serde_json::Error,
    },
}

impl Error {
    /// True when the Hub said the file does not exist, as opposed to any
    /// kind of failure to fetch it. Callers probing for an optional file
    /// (`model.safetensors` vs a shard index) branch on this instead of on
    /// `is_err()`.
    pub fn is_not_found(&self) -> bool {
        matches!(self, Error::NotFound { .. })
    }

    /// True when the caller stopped the download itself.
    pub fn is_cancelled(&self) -> bool {
        matches!(self, Error::Cancelled { .. })
    }

    /// Whether retrying could plausibly succeed. Transport failures and the
    /// Hub's transient statuses qualify; a 404, a hash mismatch or a
    /// cancellation never will, and retrying them just multiplies the wait.
    pub(crate) fn is_transient(&self) -> bool {
        match self {
            Error::Transport { .. } => true,
            Error::UnexpectedStatus { status, .. } => *status == 429 || (500..600).contains(status),
            _ => false,
        }
    }

    pub(crate) fn io(path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        Error::Io {
            path: path.into(),
            source,
        }
    }
}
