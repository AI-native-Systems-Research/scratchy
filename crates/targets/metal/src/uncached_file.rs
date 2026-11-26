// SPDX-License-Identifier: Apache-2.0

//! ⭐ A FILE WRITTEN AROUND THE PAGE CACHE (`UncachedFile`).
//!
//! A file scratchy writes in bulk and reads back at most once — the weight cache's blob, the
//! prefix cache's snapshot disk tier — goes to the disk without passing through the OS page
//! cache (`F_NOCACHE`): through it, every write takes pages of the memory the GPU's own buffers
//! hold most of, for bytes nothing reads from the cache. The only way to have such a file is
//! through [`UncachedFile::new`], so a cached one is not a thing either writer can hold.

use std::fs::File;
use std::os::fd::AsRawFd;

/// A file whose reads and writes bypass the OS page cache.
pub struct UncachedFile(File);

impl UncachedFile {
    /// `file`, its reads and writes from now on around the page cache.
    pub fn new(file: File) -> std::io::Result<Self> {
        // SAFETY: `file` is an open descriptor this call borrows; F_NOCACHE takes an int and
        // touches nothing but the descriptor's caching.
        if unsafe { libc::fcntl(file.as_raw_fd(), libc::F_NOCACHE, 1) } == -1 {
            return Err(std::io::Error::last_os_error());
        }
        Ok(Self(file))
    }

    /// The file, to read and write.
    pub fn file(&self) -> &File {
        &self.0
    }
}
