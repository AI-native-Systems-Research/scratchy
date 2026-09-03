// SPDX-License-Identifier: Apache-2.0
// Copyright contributors to the vLLM project

//! Progress bar utilities for startup initialization.
//!
//! Provides visual feedback during model loading when not in INFO or higher log level.

use std::io::IsTerminal;
use std::sync::Arc;

use indicatif::{ProgressBar, ProgressStyle};

/// Progress tracker for vLLM initialization stages.
///
/// Conditionally displays progress bars based on log level and terminal detection.
pub struct StartupProgress {
    main_bar: Option<ProgressBar>,
}

impl StartupProgress {
    /// Create a new progress tracker.
    ///
    /// Progress bars are only shown when:
    /// - `show_progress` is true (typically when log level < INFO)
    /// - stderr is a terminal (not redirected to a file)
    pub fn new(show_progress: bool) -> Self {
        if !show_progress || !std::io::stderr().is_terminal() {
            return Self { main_bar: None };
        }

        // Main progress bar: 5 major stages, full width
        let main_bar = ProgressBar::new(5);
        main_bar.set_style(
            ProgressStyle::default_bar()
                .template("[{elapsed_precise}] {wide_bar:.cyan/blue} {pos}/{len} {msg}")
                .unwrap()
                .progress_chars("=>-"),
        );

        Self {
            main_bar: Some(main_bar),
        }
    }

    /// Update main progress bar message and increment.
    pub fn set_stage(&self, message: &str) {
        if let Some(ref bar) = self.main_bar {
            bar.set_message(message.to_string());
            bar.inc(1);
        }
    }

    /// Finish and clear all progress bars.
    pub fn finish(&self) {
        if let Some(ref bar) = self.main_bar {
            bar.finish_and_clear();
        }
    }

    /// Check if progress bars are enabled.
    pub fn is_enabled(&self) -> bool {
        self.main_bar.is_some()
    }
}

/// Thread-safe progress callback for passing to worker initialization.
pub type ProgressCallback = Arc<dyn Fn(&str) + Send + Sync>;

/// Create a progress callback from a StartupProgress instance.
pub fn create_callback(progress: Arc<StartupProgress>) -> ProgressCallback {
    Arc::new(move |msg: &str| {
        // Log worker progress messages when progress tracking is enabled
        if progress.is_enabled() {
            tracing::debug!("Worker progress: {}", msg);
        }
    })
}

// Made with Bob
