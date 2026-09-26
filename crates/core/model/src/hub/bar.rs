// SPDX-License-Identifier: Apache-2.0
//! [`Progress`] drawn as terminal bars, one line per file: the file's name
//! on the left, then the bar, then how much of how much in human units, the
//! rate, and the time left — or, once done, how long it took.

use std::time::Duration;

use hf_hub_downloader::Progress;
use indicatif::{BinaryBytes, MultiProgress, ProgressBar, ProgressDrawTarget, ProgressStyle};

/// A stack of bars on stderr, one per file; hidden when stderr is not a
/// terminal.
#[derive(Default)]
pub(super) struct Bars(MultiProgress);

impl Bars {
    /// Progress for one more file.
    pub(super) fn bar(&self) -> Bar {
        Bar {
            bars: self.0.clone(),
            line: Line::Unsized,
            resumed: 0,
            moving: Duration::ZERO,
        }
    }
}

/// One file's line in a [`Bars`].
pub(super) struct Bar {
    bars: MultiProgress,
    line: Line,
    /// Credited by [`resumed`](Progress::resumed): drawn as progress, never
    /// as speed.
    resumed: u64,
    /// How long after `init` the last byte landed.
    moving: Duration,
}

enum Line {
    /// No `init` yet, so no size to draw against.
    Unsized,
    /// Sized, but off screen until its first frame can be true: a resumed
    /// download's starting point, drawn before the rate is told it is not
    /// transfer, reads as terabytes a second.
    Pending(ProgressBar),
    Shown(ProgressBar),
}

impl Bar {
    fn show(&mut self) -> Option<&ProgressBar> {
        if let Line::Pending(bar) = &self.line {
            self.line = Line::Shown(self.bars.add(bar.clone()));
        }
        match &self.line {
            Line::Shown(bar) => Some(bar),
            Line::Unsized | Line::Pending(_) => None,
        }
    }
}

impl Progress for Bar {
    fn init(&mut self, total: u64, filename: &str) {
        let bar = ProgressBar::with_draw_target(Some(total), ProgressDrawTarget::hidden())
            .with_style(style(MOVING))
            .with_message(filename.to_string());
        self.line = Line::Pending(bar);
    }

    fn resumed(&mut self, bytes: u64) {
        if let Line::Pending(bar) = &self.line {
            bar.set_position(bytes);
            // Found on disk, not transferred: the rate, and the time left it
            // predicts, start from here.
            bar.reset_eta();
        }
        self.resumed = bytes;
        self.show();
    }

    fn update(&mut self, delta: u64) {
        if let Some(bar) = self.show() {
            bar.inc(delta);
            let moving = bar.elapsed();
            self.moving = moving;
        }
    }

    fn finish(&mut self) {
        let (resumed, moving) = (self.resumed, self.moving.as_secs_f64());
        if let Some(bar) = self.show() {
            // indicatif's own rate for a finished bar is its position over
            // its life, which counts what was found on disk as moved, and the
            // hashing after the last byte as time spent moving it.
            let moved = bar.position().saturating_sub(resumed);
            let rate = if moving > 0.0 {
                moved as f64 / moving
            } else {
                0.0
            };
            bar.set_prefix(format!("{}/s", BinaryBytes(rate as u64)));
            bar.set_style(style(DONE));
            bar.finish();
        }
    }
}

/// The one layout, with the two fields that differ between a download in
/// flight and a finished one.
///
/// Padded, so the bar does not jitter as the numbers change length — but
/// only to their usual widths (`965.13 MiB`, `104.65 MiB/s`, `16s`). A shard's
/// name is 32 columns, and on an 80-column terminal whatever it and these
/// leave is all the bar gets; pad for the rare `1023.99 MiB` and the bar
/// vanishes.
macro_rules! layout {
    ($rate:literal, $time:literal) => {
        concat!(
            "{msg} {wide_bar:.cyan/blue} {binary_bytes:>10}/{binary_total_bytes:<9} {",
            $rate,
            ":>12} {",
            $time,
            ":>3}"
        )
    };
}

/// In flight: the rate now, and the time left.
const MOVING: &str = layout!("binary_bytes_per_sec", "eta");
/// Done: the rate this run moved bytes at (set as the prefix), and how long
/// the file took.
const DONE: &str = layout!("prefix", "elapsed");

fn style(template: &str) -> ProgressStyle {
    ProgressStyle::with_template(template).expect("both templates are literals, and parse")
}

#[cfg(test)]
mod tests {
    use std::io;
    use std::sync::{Arc, Mutex};
    use std::time::Duration;

    use indicatif::{ProgressDrawTarget, TermLike};

    use super::*;

    /// A terminal `COLUMNS` wide that keeps every line drawn to it.
    #[derive(Debug)]
    struct Terminal<const COLUMNS: u16>(Arc<Mutex<Vec<String>>>);

    impl<const COLUMNS: u16> Terminal<COLUMNS> {
        fn target() -> (ProgressDrawTarget, Arc<Mutex<Vec<String>>>) {
            let drawn = Arc::default();
            let target = ProgressDrawTarget::term_like(Box::new(Self(Arc::clone(&drawn))));
            (target, drawn)
        }
    }

    impl<const COLUMNS: u16> TermLike for Terminal<COLUMNS> {
        fn width(&self) -> u16 {
            COLUMNS
        }
        fn move_cursor_up(&self, _: usize) -> io::Result<()> {
            Ok(())
        }
        fn move_cursor_down(&self, _: usize) -> io::Result<()> {
            Ok(())
        }
        fn move_cursor_right(&self, _: usize) -> io::Result<()> {
            Ok(())
        }
        fn move_cursor_left(&self, _: usize) -> io::Result<()> {
            Ok(())
        }
        fn write_line(&self, line: &str) -> io::Result<()> {
            self.write_str(line)
        }
        fn write_str(&self, line: &str) -> io::Result<()> {
            if !line.trim().is_empty() {
                self.0.lock().unwrap().push(line.to_string());
            }
            Ok(())
        }
        fn clear_line(&self) -> io::Result<()> {
            Ok(())
        }
        fn flush(&self) -> io::Result<()> {
            Ok(())
        }
    }

    /// The failure this guards was seen, not imagined: the first layout, on
    /// an 80-column terminal, drew a shard's name and its numbers and no bar
    /// at all.
    #[test]
    fn a_shard_still_has_a_bar_on_an_80_column_terminal() {
        let (target, drawn) = Terminal::<80>::target();
        let mut bar = Bars(MultiProgress::with_draw_target(target)).bar();
        bar.init(3_962_000_000, "model-00001-of-00003.safetensors");
        bar.update(3_962_000_000);
        bar.finish();

        let line = drawn.lock().unwrap().last().cloned().expect("finish draws");
        assert!(
            line.starts_with("model-00001-of-00003.safetensors "),
            "{line:?}"
        );
        assert!(
            line.contains("GiB/"),
            "sizes must be in human units: {line:?}"
        );
        let width = line.chars().filter(|&c| c == '█').count();
        assert!(width >= 8, "a {width}-column bar at 80 columns: {line:?}");
    }

    /// Seen on a real resumed pull, twice over: the first frame read
    /// `26.84 TiB/s`, and the finished line an average several times the
    /// real one — both counting the gigabytes found on disk as transferred.
    #[test]
    fn bytes_found_on_disk_are_never_shown_as_speed() {
        const ON_DISK: u64 = 4 << 30;
        let (target, drawn) = Terminal::<120>::target();
        let mut bar = Bars(MultiProgress::with_draw_target(target)).bar();

        bar.init(2 * ON_DISK, "model-00001-of-00002.safetensors");
        bar.resumed(ON_DISK);
        for _ in 0..4 {
            std::thread::sleep(Duration::from_millis(60));
            bar.update(1 << 20);
        }
        bar.finish();

        // Four MiB in a quarter second is 16 MiB/s; the 4 GiB on disk, read
        // as transfer, is tens of GiB/s. Anything past MiB/s is the credit.
        let drawn = drawn.lock().unwrap();
        assert!(
            drawn.len() >= 2,
            "expected progress and a finished line: {drawn:?}"
        );
        for line in drawn.iter() {
            assert!(
                !line.contains("GiB/s") && !line.contains("TiB/s"),
                "bytes found on disk drawn as speed: {line:?}"
            );
        }
    }
}
