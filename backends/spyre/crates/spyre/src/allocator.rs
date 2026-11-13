// SPDX-License-Identifier: Apache-2.0
//! Host-memory allocator for the Spyre/KTIR backend.

use anyhow::Result;
use scratchy_tensors::{DeviceAllocator, PoolMemory};
use std::sync::Arc;

/// Host-resident bytes by PROVENANCE: owned outright, or a view into a
/// weight store's safetensors mmap.
///
/// ⭐ THE LOAD'S ONLY HOST COPY LIVES OR DIES HERE. The weight chain is
/// store mmap → (this type) → staging walk → THP shadow → card: staging reads
/// its source with plain byte slices (the retile walks index bytes, the SEN
/// converters go through `from_le_bytes`), so the bytes staging reads can be
/// the mmap's own — no private copy needs to exist between the two. A weight
/// that needs its bytes REWRITTEN (a cast at load, a pad, a fold) materializes
/// via [`HostBytes::make_owned`] — the transform's own law, not a fallback.
///
/// A `View` keeps its shard's `Arc<Mmap>` alive for as long as the value
/// lives, so handing one out of the allocator (staging's take) leaves every
/// `GpuTensor` viewing the same bytes VALID — the opposite of the `Owned`
/// handover, which orphans its viewers.
///
/// `Send + Sync`: both backings are (`Arc<Mmap>`, `Vec<u8>`); staging's
/// scoped threads share `&bound` across them.
pub enum HostBytes {
    /// Bytes this value owns outright: a load-time cast's output, a pad or
    /// fold product, a built table, a per-step activation bind.
    Owned(Vec<u8>),
    /// The store's own bytes, viewed. `offset`/`len` index `backing`; the
    /// `Arc` is the keepalive — it moves with the value, so the mmap outlives
    /// the allocator and the weight store that first held it.
    View {
        backing: Arc<memmap2::Mmap>,
        offset: usize,
        len: usize,
    },
}

impl HostBytes {
    /// The bytes themselves, either way. Every read of staged weight bytes
    /// goes through here — byte slices only; no reader may cast these to a
    /// wider type (the mmap's tensor offsets carry no alignment guarantee,
    /// unlike the GPU target's alias, whose GPU reads SIMD-wide and which therefore
    /// gates on alignment — this backend's readers are all host byte walks).
    pub fn as_slice(&self) -> &[u8] {
        match self {
            HostBytes::Owned(v) => v,
            HostBytes::View {
                backing,
                offset,
                len,
            } => &backing[*offset..*offset + *len],
        }
    }

    /// The byte count, without materializing.
    pub fn len(&self) -> usize {
        match self {
            HostBytes::Owned(v) => v.len(),
            HostBytes::View { len, .. } => *len,
        }
    }

    /// Whether this handle holds no bytes (a zero-extent tensor).
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Materialize in place: a `View` becomes an `Owned` copy of itself, an
    /// `Owned` stays. Returns the owned buffer for the rewrite the caller is
    /// about to perform — the ONE way a `View` becomes writable (the mmap is
    /// a read-only mapping; writing through it is not expressible here).
    pub fn make_owned(&mut self) -> &mut Vec<u8> {
        let copy = match self {
            HostBytes::Owned(v) => return v,
            HostBytes::View {
                backing,
                offset,
                len,
            } => backing[*offset..*offset + *len].to_vec(),
        };
        *self = HostBytes::Owned(copy);
        let HostBytes::Owned(v) = self else {
            unreachable!()
        };
        v
    }

    /// Take the bytes as an owned buffer, consuming the handle. The one-copy
    /// price of a `View` a caller must own (a TP rank's slice, a pad).
    pub fn into_vec(self) -> Vec<u8> {
        match self {
            HostBytes::Owned(v) => v,
            HostBytes::View {
                backing,
                offset,
                len,
            } => backing[offset..offset + len].to_vec(),
        }
    }
}

/// Host-memory "device" allocator for Spyre/KTIR.
///
/// Spyre/KTIR weights live in plain host RAM (the `ktir-emulator` emulator and
/// the AIU both consume host-resident bundles), so the [`DeviceAllocator`]
/// contract — "allocate `bytes` and copy `src_host` onto it" — is a host
/// allocation + `memcpy`. The allocator owns every buffer it hands out and frees
/// them on drop; the returned pointers stay valid for the allocator's lifetime
/// (the same ownership model as the GPU target's allocator).
///
/// ⭐ THE COPY IS CONDITIONAL ON WHERE THE BYTES ALREADY LIVE: a source that
/// points into a registered safetensors mmap is ALIASED, not copied — spyre's
/// "device memory" is host memory, and staging (the bytes' only reader) walks
/// byte slices, so the mmap's own pages serve directly (MEASURED 2b fp8,
/// nickm3, 2026-11-10: the unconditional copy was the whole wiring walk —
/// 1.27s for 2.65 GB at 2.1 GB/s; f16 2.46s). This is the GPU target's zero-copy mmap
/// alias with the alignment gate dropped, because its gate exists for
/// SIMD-wide GPU reads of the mapping and this backend's readers are all host
/// byte walks (see [`HostBytes::as_slice`]).
#[derive(Default)]
pub struct SpyreAllocator {
    /// Owns the host buffers backing the pointers returned below — owned
    /// copies AND aliased views, one entry per handed-out pointer.
    allocs: Vec<HostBytes>,
    /// The safetensors mmaps the weight store registered at open. The
    /// alias path classifies sources against these; each `View` it hands out
    /// clones its `Arc`, so a taken view outlives this list and the allocator.
    /// `Mutex` because [`DeviceAllocator::register_mmap`] takes `&self`.
    backings: std::sync::Mutex<Vec<Arc<memmap2::Mmap>>>,
}

impl SpyreAllocator {
    pub fn new() -> Self {
        Self::default()
    }

    /// The registered backing containing `src_host..src_host+bytes`, with the
    /// range's offset into it. `None` when the bytes are not aliasable (they
    /// live in a cast scratch, a synthesized buffer, an unregistered file).
    fn alias_of(&self, src_host: *const u8, bytes: usize) -> Option<(Arc<memmap2::Mmap>, usize)> {
        let backings = self.backings.lock().expect("SpyreAllocator backings Mutex");
        backings
            .iter()
            .filter_map(|m| {
                let base = m.as_ptr() as usize;
                let src = src_host as usize;
                // Containment with no wraparound: the range must sit whole
                // inside the mapping.
                if base <= src && src.checked_add(bytes)? <= base + m.len() {
                    Some((Arc::clone(m), src - base))
                } else {
                    None
                }
            })
            .next()
    }

    /// Hand the buffer at `ptr` (exactly `len` bytes) to the caller, WITHOUT
    /// copying — an `Owned` entry gives up its `Vec`; a `View` entry moves its
    /// keepalive `Arc`, so the backing stays alive as long as the returned
    /// value does. `None` when no buffer this allocator owns starts at `ptr`
    /// with that length.
    ///
    /// ⭐ WHY: staging a checkpoint used to COPY every tensor out of these
    /// buffers while they stayed alive, so for the length of the load the
    /// weights existed twice.
    /// MEASURED gemma-4-12b fp8 emu: a 36 GB load peak for a 12 GB checkpoint.
    ///
    /// # Safety
    /// An `Owned` handover leaves every `GpuTensor` viewing the buffer
    /// dangling: the caller must not read through any of them after the take
    /// (the allocator no longer owns the memory). A `View` handover keeps the
    /// backing alive — its viewers stay valid as long as the returned value.
    pub unsafe fn take(&mut self, ptr: *const u8, len: usize) -> Option<HostBytes> {
        let i = self
            .allocs
            .iter()
            .position(|b| std::ptr::eq(b.as_slice().as_ptr(), ptr) && b.len() == len)?;
        Some(self.allocs.swap_remove(i))
    }

    /// Clone the HANDLE of the buffer at `ptr` (exactly `len` bytes) without
    /// removing it: a `View` clones its `Arc` (free — two bindings of one
    /// shared buffer, a tied head, both view the mmap); an `Owned` buffer has
    /// no handle to clone, so `None` — the caller copies, as shared owned
    /// buffers always have.
    ///
    /// The non-last bindings of a shared buffer stage through this; the last
    /// one [`take`](Self::take)s.
    pub fn share(&mut self, ptr: *const u8, len: usize) -> Option<HostBytes> {
        match self
            .allocs
            .iter()
            .find(|b| std::ptr::eq(b.as_slice().as_ptr(), ptr) && b.len() == len)?
        {
            HostBytes::View {
                backing,
                offset,
                len,
            } => Some(HostBytes::View {
                backing: Arc::clone(backing),
                offset: *offset,
                len: *len,
            }),
            HostBytes::Owned(_) => None,
        }
    }
}

/// Device-memory witness for the host-backed Spyre allocator. The buffers are
/// owned by [`SpyreAllocator::allocs`] (the same model the GPU target uses for its
/// arenas), so the `take_allocations` / `push_alloc` tracking hooks stay at
/// their no-op defaults and no `SpyreMem` is ever constructed — it exists only
/// to satisfy the [`DeviceAllocator::Mem`] associated type (`Mem: PoolMemory`)
/// that generic `GpuWeights<A>` / pool code names.
pub struct SpyreMem(*mut u8);
// SAFETY: the pointer addresses a host buffer owned for the allocator's
// lifetime; `SpyreMem` is never actually instantiated (see above).
unsafe impl Send for SpyreMem {}
unsafe impl Sync for SpyreMem {}
impl PoolMemory for SpyreMem {
    fn ptr(&self) -> *mut u8 {
        self.0
    }
}

impl DeviceAllocator for SpyreAllocator {
    type Stream = ();
    type Mem = SpyreMem;

    fn stream(&self) {}

    /// Hold the shard for the alias path. The store also keeps its own
    /// `Arc`s; this one is what makes an alias valid after `take` removes the
    /// store's last reference to a tensor's `CpuTensorRef`.
    ///
    /// ⭐ AND DISPATCH THE PREFETCH THAT RIDES THE BRING-UP (the GPU target's own
    /// `register_mmap` shape: background `pread`s). The view load's first
    /// reader of the shard pages is staging, after the card join — on a
    /// page-cache-cold boot that puts the whole checkpoint read on the boot
    /// critical path. A sequential reader started here fills the page cache
    /// while the bring-up runs, which is the overlap the load's old host copy
    /// provided for free; see [`prefetch_shard`] for why the `madvise` hint
    /// alone does not carry it on the card's network filesystem.
    fn register_mmap(&self, path: &std::path::Path, mmap: Arc<memmap2::Mmap>) -> Result<()> {
        self.backings
            .lock()
            .expect("SpyreAllocator backings Mutex")
            .push(mmap);
        prefetch_shard(path);
        Ok(())
    }

    /// The `madvise(MADV_WILLNEED)` half of the readahead — the hint every
    /// kernel that honors it turns into readahead over the mapping. On the
    /// card's network filesystem it is inert (measured: the mapping's pages
    /// stay cold), which is why [`register_mmap`] dispatches the real reader.
    fn prefault_mmap(&self, mmap: &memmap2::Mmap) {
        will_need(mmap.as_ptr() as *mut std::ffi::c_void, mmap.len());
    }

    unsafe fn alloc_and_copy_host(&mut self, src_host: *const u8, bytes: usize) -> Result<*mut u8> {
        // ⭐ ALIAS FIRST: a source inside a registered mmap needs no copy —
        // staging, its only reader, walks byte slices (the law stated on
        // [`HostBytes::as_slice`]), so the mapping's own pages are the
        // "device memory". The `Arc` moves into the entry, keeping the shard
        // alive for as long as the allocator (or whoever takes the entry).
        if bytes > 0
            && let Some((backing, offset)) = self.alias_of(src_host, bytes)
        {
            scratchy_layers::load_time::record_alloc_view(bytes);
            self.allocs.push(HostBytes::View {
                backing,
                offset,
                len: bytes,
            });
            return Ok(src_host as *mut u8);
        }
        // `SCRATCHY_LOAD_TIME`: the zero-fill and the copy are counted apart —
        // the copy is the pass's contract, the zero is `vec!`'s price for a
        // buffer the very next statement overwrites whole.
        let zero_timer = std::time::Instant::now();
        let mut buf = vec![0u8; bytes];
        scratchy_layers::load_time::record_alloc_zero(bytes, zero_timer);
        if bytes > 0 {
            let copy_timer = std::time::Instant::now();
            // SAFETY: the caller guarantees `src_host` is valid for `bytes`
            // bytes; `buf` is exactly `bytes` long and freshly allocated, so the
            // ranges cannot overlap.
            unsafe { std::ptr::copy_nonoverlapping(src_host, buf.as_mut_ptr(), bytes) };
            scratchy_layers::load_time::record_alloc_copy(bytes, copy_timer);
        }
        let ptr = buf.as_mut_ptr();
        self.allocs.push(HostBytes::Owned(buf));
        Ok(ptr)
    }
}

/// `MADV_WILLNEED` over a mapping the store just opened — best-effort BY
/// NATURE (it is an advice): a kernel that refuses it leaves the pages to
/// fault on first read, exactly the behavior without the hint, which the
/// refusal line names when it happens.
#[cfg(target_os = "linux")]
fn will_need(ptr: *mut std::ffi::c_void, mapped: usize) {
    // SAFETY: `ptr`/`mapped` are the caller's live mapping (every caller
    // passes a shard the store holds), and `madvise` reads only the VMA
    // metadata.
    unsafe {
        if libc::madvise(ptr, mapped, libc::MADV_WILLNEED) != 0 {
            let errno = std::io::Error::last_os_error();
            tracing::debug!(
                "MADV_WILLNEED refused ({errno}) — the shard's pages fault on first read"
            );
        }
    }
}

/// Only Linux drives a card — elsewhere the store's pages fault on first read.
#[cfg(not(target_os = "linux"))]
fn will_need(_ptr: *mut std::ffi::c_void, _mapped: usize) {}

/// Read a registered shard sequentially, once, on a detached thread — the
/// readahead the card's NFS owes the view load. `madvise`/`fadvise(WILLNEED)`
/// are inert there (measured: pages stay cold behind the hint), and the view
/// load's own first reader is staging, after the card join — so without this
/// reader a client-cache-cold boot pays the checkpoint's network reads on the
/// critical path, where the load's old host copy paid them under the bring-up.
/// The thread is bounded by construction (it reads the file's own length once,
/// then exits) and best-effort by nature: a failed or refused read leaves the
/// pages to fault on demand in staging, exactly the behavior without it.
fn prefetch_shard(path: &std::path::Path) {
    let path = path.to_path_buf();
    // A spawn refusal (thread exhaustion) is the same non-event as a refused
    // read: the pages fault on demand.
    let _ = std::thread::Builder::new()
        .name("shard-prefetch".to_string())
        .spawn(move || {
            let Ok(mut file) = std::fs::File::open(&path) else {
                return;
            };
            use std::io::Read;
            // 8 MiB reads: whole RPCs on a network filesystem, not per-page
            // round trips.
            let mut buf = vec![0u8; 8 << 20];
            loop {
                match file.read(&mut buf) {
                    Ok(0) | Err(_) => return,
                    Ok(_) => {}
                }
            }
        });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mapped(bytes: &[u8]) -> Arc<memmap2::Mmap> {
        // A read-only file mapping — the store's own backing kind. The
        // mapping stays valid after the file (and its temp path) go away.
        let mut f = tempfile::NamedTempFile::new().unwrap();
        std::io::Write::write_all(&mut f, bytes).unwrap();
        Arc::new(unsafe { memmap2::MmapOptions::new().map(&f).unwrap() })
    }

    /// The alias hands out the mapping's own pointer, and the take moves the
    /// keepalive: the backing outlives the allocator that registered it.
    #[test]
    fn a_registered_mmap_source_is_aliased_and_its_view_survives_the_allocator() {
        let mut alloc = SpyreAllocator::new();
        let backing = mapped(&[1u8; 64]);
        alloc
            .register_mmap(std::path::Path::new("/dev/null"), Arc::clone(&backing))
            .unwrap();
        let src = unsafe { backing.as_ptr().add(8) };
        // SAFETY: 16 valid bytes at `src` inside the mapping.
        let ptr = unsafe { alloc.alloc_and_copy_host(src, 16) }.unwrap();
        assert_eq!(ptr as *const u8, src, "the alias IS the mapping's pointer");

        // The take hands the view out; dropping the allocator must not
        // invalidate it.
        let view = unsafe { alloc.take(src, 16) }.expect("the alias is takeable");
        drop(alloc);
        drop(backing);
        assert_eq!(view.as_slice(), &[1u8; 16]);
    }

    /// A source outside every registered backing copies, and two takes of one
    /// owned buffer are impossible: the first moves it, the second refuses.
    #[test]
    fn an_unregistered_source_copies_and_an_owned_buffer_takes_once() {
        let mut alloc = SpyreAllocator::new();
        let src = [7u8; 32];
        // SAFETY: 32 valid bytes of `src`.
        let ptr = unsafe { alloc.alloc_and_copy_host(src.as_ptr(), 32) }.unwrap();
        assert_ne!(
            ptr as *const u8,
            src.as_ptr(),
            "an outside source is copied"
        );
        // SAFETY: the returned pointer is the taken buffer's own.
        let first = unsafe { alloc.take(ptr, 32) }.expect("the copy is takeable");
        assert_eq!(first.as_slice(), &[7u8; 32]);
        // SAFETY: a second take of the moved buffer addresses freed memory —
        // asserted to refuse, not read.
        assert!(unsafe { alloc.take(ptr, 32) }.is_none());
    }

    /// `share` clones a view's keepalive (two bindings, one mapping) and
    /// refuses an owned buffer (no handle exists — the caller copies).
    #[test]
    fn share_clones_a_view_and_refuses_an_owned_buffer() {
        let mut alloc = SpyreAllocator::new();
        let backing = mapped(&[3u8; 128]);
        alloc
            .register_mmap(std::path::Path::new("/dev/null"), Arc::clone(&backing))
            .unwrap();
        let src = unsafe { backing.as_ptr().add(16) };
        // SAFETY: 32 valid bytes at `src` inside the mapping.
        unsafe { alloc.alloc_and_copy_host(src, 32) }.unwrap();
        let shared = alloc.share(src, 32).expect("a view shares its keepalive");
        assert_eq!(shared.as_slice(), &[3u8; 32]);
        // The shared handle and the entry view the same bytes.
        // SAFETY: the alias pointer is still owned by the allocator's entry.
        unsafe { alloc.take(src, 32) }.expect("the entry survives the share");
        drop(alloc);
        drop(backing);
        assert_eq!(shared.as_slice(), &[3u8; 32]);

        let mut alloc = SpyreAllocator::new();
        let src = [5u8; 16];
        // SAFETY: 16 valid bytes of `src`.
        let ptr = unsafe { alloc.alloc_and_copy_host(src.as_ptr(), 16) }.unwrap();
        assert!(
            alloc.share(ptr, 16).is_none(),
            "an owned buffer has no handle"
        );
    }

    /// `make_owned` is the one way a view becomes writable, and `into_vec`
    /// owns what a caller must own.
    #[test]
    fn a_view_materializes_only_through_make_owned() {
        let mut alloc = SpyreAllocator::new();
        let backing = mapped(&[9u8; 48]);
        alloc
            .register_mmap(std::path::Path::new("/dev/null"), Arc::clone(&backing))
            .unwrap();
        let src = unsafe { backing.as_ptr().add(8) };
        // SAFETY: 24 valid bytes at `src` inside the mapping.
        unsafe { alloc.alloc_and_copy_host(src, 24) }.unwrap();
        // SAFETY: the alias pointer is the entry's own.
        let mut view = unsafe { alloc.take(src, 24) }.unwrap();
        drop(alloc);
        drop(backing);
        let v = view.make_owned();
        v.iter_mut().for_each(|b| *b += 1);
        assert_eq!(view.as_slice(), &[10u8; 24]);
        assert_eq!(view.into_vec(), vec![10u8; 24]);
    }
}
